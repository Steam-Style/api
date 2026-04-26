import logging
from typing import Any, Dict, List, Optional, cast
from urllib.parse import unquote_plus

from litestar.openapi import OpenAPIConfig
from litestar.openapi.plugins import SwaggerRenderPlugin
from litestar.response import Redirect
from steam_style_embeddings import ColorEmbedder, SiglipEmbedder, Embedding
from config import settings

import uvicorn
from litestar import Litestar, get
from litestar.config.cors import CORSConfig
from litestar.exceptions import HTTPException
from litestar.params import Parameter
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

BOOLEAN_FILTER_FIELDS = ["animated", "tiled", "transparent"]

logger = logging.getLogger(__name__)
color_embedder = ColorEmbedder(
    hue_bins=settings.COLOR_HUE_BINS,
    sat_bins=settings.COLOR_SAT_BINS,
    val_bins=settings.COLOR_VAL_BINS,
    sigma_h=settings.COLOR_SIGMA_H,
    sigma_s=settings.COLOR_SIGMA_S,
    sigma_v=settings.COLOR_SIGMA_V,
    power=settings.COLOR_POWER,
)
siglip_embedder = SiglipEmbedder(
    model_name=settings.MODEL_NAME, device=settings.DEVICE)
qdrant_client = QdrantClient(url=settings.DATABASE_URL)


def get_text_embedding(text: str) -> Optional[Embedding]:
    if not siglip_embedder.is_ready():
        return None

    try:
        return siglip_embedder.get_text_embedding(text)
    except Exception as e:
        logger.error("Error getting text embedding: %s", e)
        return None


class SearchRequest(BaseModel):
    query: Optional[str] = None
    similar_to: Optional[int] = None
    colors: Optional[List[str]] = None
    category: List[str] = Field(
        default_factory=list,
        description="Filter by category. 'all'=all categories, empty=no items.",
    )
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    sort: Optional[str] = Field(
        default=None, description="Sort by: 'newest', 'oldest', 'updated', 'random'")
    animated: Optional[bool] = Field(
        default=None, description="True=only animated, False=exclude animated, None=all")
    tiled: Optional[bool] = Field(
        default=None, description="True=only tiled, False=exclude tiled, None=all")
    transparent: Optional[bool] = Field(
        default=None, description="True=only transparent, False=exclude transparent, None=all")


def _build_query_filter(data: SearchRequest) -> models.Filter:
    must_conditions: List[models.Condition] = []
    must_not_conditions: List[models.Condition] = []

    decoded_categories = [
        unquote_plus(category).lower().strip()
        for category in data.category
        if category and category.lower().strip() != "all"
    ]
    if decoded_categories:
        category_conditions: List[models.Condition] = [
            models.FieldCondition(
                key="item.category",
                match=models.MatchValue(value=decoded_category),
            )
            for decoded_category in decoded_categories
        ]
        if len(category_conditions) == 1:
            must_conditions.extend(category_conditions)
        else:
            must_conditions.append(models.Filter(should=category_conditions))

    for prop in BOOLEAN_FILTER_FIELDS:
        value = getattr(data, prop)
        if value is True:
            must_conditions.append(
                models.FieldCondition(
                    key=f"item.{prop}",
                    match=models.MatchValue(value=True),
                )
            )
        elif value is False:
            must_not_conditions.append(
                models.FieldCondition(
                    key=f"item.{prop}",
                    match=models.MatchValue(value=True),
                )
            )

    return models.Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )


def _get_sort_order(sort: Optional[str]) -> Optional[models.OrderBy]:
    if sort == "newest":
        return models.OrderBy(
            key="timestamps.created_at", direction=models.Direction.DESC)
    if sort == "oldest":
        return models.OrderBy(
            key="timestamps.created_at", direction=models.Direction.ASC)
    if sort == "updated":
        return models.OrderBy(
            key="timestamps.updated_at", direction=models.Direction.DESC)
    return None


def _scroll_items(
    query_filter: models.Filter,
    limit: int,
    offset: int,
    sort: Optional[str],
) -> List[Dict[str, Any]]:
    if sort == "random":
        try:
            results = qdrant_client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=models.SampleQuery(sample=models.Sample.RANDOM),
                query_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
            )
        except Exception as e:
            logger.exception("Random query points error")
            raise HTTPException(
                status_code=500, detail=f"Random query failed: {str(e)}") from e

        return [cast(Dict[str, Any], p.payload) for p in results.points]

    try:
        results = qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=limit + offset,
            with_payload=True,
            with_vectors=False,
            order_by=_get_sort_order(sort),
        )
    except Exception as e:
        logger.exception("Scroll error")
        raise HTTPException(
            status_code=500, detail=f"Scroll failed: {str(e)}") from e

    return [cast(Dict[str, Any], p.payload) for p in results[0][offset:]]


def _build_prefetch(
    data: SearchRequest,
    query_filter: models.Filter,
) -> List[models.Prefetch]:
    prefetch: List[models.Prefetch] = []
    prefetch_limit = max(200, data.limit + data.offset)

    has_colors = data.colors is not None and len(data.colors) > 0
    has_similar = data.similar_to is not None
    has_query = data.query is not None and len(data.query.strip()) > 0

    if has_colors:
        assert data.colors is not None
        try:
            color_emb = color_embedder.query_to_embedding(data.colors)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        prefetch.append(
            models.Prefetch(
                query=color_emb,
                using="color",
                filter=query_filter,
                limit=prefetch_limit,
            )
        )

    if has_similar:
        assert data.similar_to is not None
        try:
            similar_points, _ = qdrant_client.scroll(
                collection_name=settings.COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="item.id",
                            match=models.MatchValue(value=data.similar_to),
                        )
                    ]
                ),
                limit=1,
                with_vectors=True,
                with_payload=False,
            )

            if not similar_points:
                raise HTTPException(
                    status_code=404,
                    detail=f"Item not found for similar_to={data.similar_to}",
                )

            similar_point = similar_points[0]
            vector_data = similar_point.vector

            if not isinstance(vector_data, dict) or "image" not in vector_data:
                raise ValueError("Similar item is missing image vector")

            similar_image_emb = vector_data["image"]
            prefetch.append(
                models.Prefetch(
                    query=similar_image_emb,
                    using="image",
                    filter=query_filter,
                    limit=prefetch_limit,
                )
            )
            if "color" in vector_data:
                prefetch.append(
                    models.Prefetch(
                        query=vector_data["color"],
                        using="color",
                        filter=query_filter,
                        limit=prefetch_limit,
                    )
                )
        except Exception as e:
            logger.exception("Error retrieving similar item embedding")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve similar item: {str(e)}",
            ) from e

    if has_query:
        assert data.query is not None
        text_emb = get_text_embedding(data.query)
        if text_emb:
            prefetch.append(
                models.Prefetch(
                    query=text_emb,
                    using="image",
                    filter=query_filter,
                    limit=prefetch_limit,
                )
            )

    return prefetch


def _query_items(
    data: SearchRequest,
    query_filter: models.Filter,
    prefetch: List[models.Prefetch],
) -> List[Dict[str, Any]]:
    if len(prefetch) > 1:
        try:
            results = qdrant_client.query_points(
                collection_name=settings.COLLECTION_NAME,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=data.limit,
                offset=data.offset,
                with_payload=True,
            )
        except Exception as e:
            logger.exception("Query points error (fusion)")
            raise HTTPException(
                status_code=500, detail=f"Query points failed: {str(e)}") from e
    else:
        try:
            results = qdrant_client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=prefetch[0].query,
                using=prefetch[0].using,
                query_filter=query_filter,
                limit=data.limit,
                offset=data.offset,
                with_payload=True,
            )
        except Exception as e:
            logger.exception("Query points error")
            raise HTTPException(
                status_code=500, detail=f"Query points failed: {str(e)}") from e

    return [cast(Dict[str, Any], p.payload) for p in results.points]


@get("/", include_in_schema=False)
async def index() -> dict:
    return {"status": "ok", "message": "Steam Style Query API"}


@get("/search", tags=["Items"])
async def search_items(
    search_query: Optional[str] = Parameter(
        query="query", default=None, description="Search query text"),
    similar_to: Optional[int] = Parameter(
        default=None, description="Item ID to find similar items for"),
    color: Optional[List[str]] = Parameter(
        default=None, description="Colors to filter by"),
    category: Optional[List[str]] = Parameter(
        default=None,
        description="Filter by category. 'all'=all categories, empty=no items.",
    ),
    limit: int = Parameter(default=10, ge=1, le=100),
    offset: int = Parameter(
        default=0, ge=0, description="Offset for pagination"),
    sort: Optional[str] = Parameter(
        default="newest", description="Sort by: 'newest', 'oldest', 'updated', 'random'"),
    animated: Optional[bool] = Parameter(
        default=None, description="True=only animated, False=exclude animated, None=all"),
    tiled: Optional[bool] = Parameter(
        default=None, description="True=only tiled, False=exclude tiled, None=all"),
    transparent: Optional[bool] = Parameter(
        default=None, description="True=only transparent, False=exclude transparent, None=all"),
) -> dict:
    category_values = category if category is not None else ["all"]
    data = SearchRequest(
        query=search_query,
        similar_to=similar_to,
        colors=color,
        category=category_values,
        limit=limit,
        offset=offset,
        sort=sort,
        animated=animated,
        tiled=tiled,
        transparent=transparent
    )

    normalized_categories = [
        unquote_plus(category_value).lower().strip()
        for category_value in data.category
        if category_value and category_value.strip()
    ]

    if category is not None and not normalized_categories:
        return {"results": []}

    has_query = data.query is not None and len(data.query.strip()) > 0
    has_similar = data.similar_to is not None
    has_colors = data.colors is not None and len(data.colors) > 0

    query_filter = _build_query_filter(data)

    if not has_query and not has_colors and not has_similar:
        return {
            "results": _scroll_items(
                query_filter=query_filter,
                limit=data.limit,
                offset=data.offset,
                sort=data.sort,
            )
        }

    prefetch = _build_prefetch(data, query_filter)

    if not prefetch:
        raise ValueError("Failed to generate embeddings")

    return {"results": _query_items(data, query_filter, prefetch)}


@get("/item/{item_id:int}", tags=["Items"])
async def get_item(item_id: int) -> dict:
    try:
        results, _ = qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="item.id",
                        match=models.MatchValue(value=item_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.exception("Scroll error")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}") from e

    if not results:
        raise HTTPException(status_code=404, detail="Item not found")

    return cast(Dict[str, Any], results[0].payload)

cors_config = CORSConfig(
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app = Litestar(
    route_handlers=[index, search_items, get_item],
    openapi_config=OpenAPIConfig(
        title="Steam Style",
        version="1.0.0",
        path="/docs",
        render_plugins=[SwaggerRenderPlugin()],
        root_schema_site="swagger"
    ),
    cors_config=cors_config
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
