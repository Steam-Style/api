"""
Runs searches against Qdrant: browsing in order, and ranking by text, colors and similar items, with the words of a
query raising matching items.
"""

import logging
from typing import Any, cast

from litestar.exceptions import HTTPException
from qdrant_client import models

from clients import (
    collection_features,
    color_embedder,
    get_text_embedding,
    qdrant_client,
)
from config import settings
from search.filters import SearchRequest, sort_query
from search.parsing import ParsedQuery

BOOST_RANK_CONSTANT = 10
KIND_BOOST = 2.0
APP_BOOST = 3.0
NAME_PHRASE_BOOST = 3.0
NAME_WORD_BOOST = 1.5
EMOTICON_CANDIDATES = 200

logger = logging.getLogger(__name__)


def scroll_items(
    query_filter: models.Filter,
    limit: int,
    offset: int,
    sort: str | None,
) -> list[dict[str, Any]]:
    try:
        results = qdrant_client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=sort_query(sort),
            query_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.exception("Scroll error")
        raise HTTPException(
            status_code=500, detail=f"Scroll failed: {e!s}") from e

    return [cast(dict[str, Any], p.payload) for p in results.points]


def find_item_by_name(app_id: int, name: str) -> dict[str, Any] | None:
    """
    Finds an item by its game and exact name, the way the Community Market names items.

    Args:
        app_id (int): The game the item comes from.
        name (str): The item's exact name.

    Returns:
        dict[str, Any] | None: The item, None when there's no such item.
    """
    try:
        points, _ = qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="app.id", match=models.MatchValue(value=app_id)),
                    models.FieldCondition(key="item.name", match=models.MatchValue(value=name)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.exception("Scroll error")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}") from e

    return cast(dict[str, Any], points[0].payload) if points else None


def find_emoticon(code: str, query_filter: models.Filter) -> int | None:
    """
    Finds an emoticon by its code, like :steamhappy:, among the items that pass the filters of a search. Steam reads
    codes in any case, so names are matched in any case too: first as typed, in lowercase and in uppercase, then among
    the emoticons with the code's words in their title. A name in the case typed wins.

    Args:
        code (str): The emoticon's code, with its colons.
        query_filter (models.Filter): The search's filters.

    Returns:
        int | None: The emoticon's ID, None when there's no such emoticon.
    """
    emoticons = models.FieldCondition(key="item.category", match=models.MatchValue(value="emoticons"))
    lookups = [models.FieldCondition(key="item.name", match=models.MatchAny(any=list({code, code.lower(), code.upper()})))]
    if collection_features().titles:
        lookups.append(models.FieldCondition(key="item.title", match=models.MatchText(text=code.strip(":"))))

    for lookup in lookups:
        try:
            points, _ = qdrant_client.scroll(
                collection_name=settings.COLLECTION_NAME,
                scroll_filter=models.Filter(must=[query_filter, emoticons, lookup]),
                limit=EMOTICON_CANDIDATES,
                with_payload=["item.name"],
                with_vectors=False,
            )
        except Exception as e:
            logger.exception("Scroll error")
            raise HTTPException(status_code=500, detail=f"Database error: {e!s}") from e

        names = {int(point.id): str((point.payload or {}).get("item", {}).get("name", "")) for point in points}
        same = [item_id for item_id, name in names.items() if name.lower() == code.lower()]
        if same:
            return next((item_id for item_id in same if names[item_id] == code), same[0])

    return None


def matching_items(item_ids: list[int], query_filter: models.Filter) -> list[dict[str, Any]]:
    """
    The items with these IDs that pass the filters of a search, in the order given.

    Args:
        item_ids (list[int]): The items to look for.
        query_filter (models.Filter): The search's filters.

    Returns:
        list[dict[str, Any]]: The items that exist and pass.
    """
    try:
        points, _ = qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(must=[query_filter, models.HasIdCondition(has_id=list(item_ids))]),
            limit=len(item_ids),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.exception("Scroll error")
        raise HTTPException(status_code=500, detail=f"Database error: {e!s}") from e

    found = {int(point.id): cast(dict[str, Any], point.payload) for point in points}
    return [found[item_id] for item_id in item_ids if item_id in found]


def build_prefetch(
    data: SearchRequest,
    query_filter: models.Filter,
) -> list[models.Prefetch]:
    prefetch: list[models.Prefetch] = []
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
            similar_points = qdrant_client.retrieve(
                collection_name=settings.COLLECTION_NAME,
                ids=[data.similar_to],
                with_vectors=["image", "color"],
                with_payload=False,
            )
        except Exception as e:
            logger.exception("Error retrieving similar item embedding")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve similar item: {e!s}",
            ) from e

        if not similar_points:
            raise HTTPException(
                status_code=404,
                detail=f"Item not found for similar_to={data.similar_to}",
            )

        vector_data = similar_points[0].vector

        if not isinstance(vector_data, dict) or "image" not in vector_data:
            raise HTTPException(
                status_code=500, detail="Similar item is missing image vector")

        prefetch.append(
            models.Prefetch(
                query=vector_data["image"],
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

    if has_query:
        assert data.query is not None
        text_emb = get_text_embedding(data.query)
        if text_emb:
            frames = collection_features().frames
            prefetch.append(
                models.Prefetch(
                    query=[text_emb] if frames else text_emb,
                    using="frames" if frames else "image",
                    filter=query_filter,
                    limit=prefetch_limit,
                )
            )

    return prefetch


def boost_conditions(
    parsed: ParsedQuery,
    data: SearchRequest,
    requested_categories: list[str],
) -> list[tuple[models.Condition, float]]:
    """
    Turns what a query said about the kind of item into conditions for items to rank higher, with how much higher.
    Anything the filters already decide is left out. Items named like what's left of the query rank higher too, most
    when the name has the words in the same order, which is how well-known items are found by name.
    """
    is_animated = models.FieldCondition(key="item.animated", match=models.MatchValue(value=True))
    boosts: list[tuple[models.Condition, float]] = []

    if parsed.animated is True and data.animated is None:
        boosts.append((is_animated, KIND_BOOST))
    if parsed.animated is False and data.animated is None:
        boosts.append((models.Filter(must_not=[is_animated]), KIND_BOOST))

    for prop in ("tiled", "transparent"):
        if getattr(parsed, prop) and getattr(data, prop) is None:
            boosts.append((models.FieldCondition(key=f"item.{prop}", match=models.MatchValue(value=True)), KIND_BOOST))

    if parsed.categories and ("all" in requested_categories or not set(requested_categories) <= set(parsed.categories)):
        boosts.append((
            models.FieldCondition(key="item.category", match=models.MatchAny(any=parsed.categories)),
            KIND_BOOST,
        ))

    if parsed.apps:
        boosts.append((
            models.FieldCondition(key="app.id", match=models.MatchAny(any=[app_id for app_id, _ in parsed.apps])),
            APP_BOOST,
        ))

    words = (data.query or "").split()
    if words and collection_features().titles:
        title = "item.title"
        boosts.append((
            models.FieldCondition(key=title, match=models.MatchPhrase(phrase=data.query or "")),
            NAME_PHRASE_BOOST if len(words) > 1 else NAME_WORD_BOOST,
        ))
        if len(words) > 1:
            boosts.append((models.FieldCondition(key=title, match=models.MatchText(text=data.query or "")), NAME_WORD_BOOST))

    return boosts


def query_items(
    data: SearchRequest,
    query_filter: models.Filter,
    prefetch: list[models.Prefetch],
    boosts: list[tuple[models.Condition, float]],
) -> list[dict[str, Any]]:
    """
    Ranks items by the prefetched searches. Each boost repeats the main search among the items that match it, so
    matching items rank in two lists and come out higher without the others being left out.
    """
    if boosts:
        main = next((p for p in reversed(prefetch) if p.using != "color"), prefetch[0])
        boosted = [
            models.Prefetch(
                query=main.query,
                using=main.using,
                filter=models.Filter(must=[query_filter, condition]),
                limit=main.limit,
            )
            for condition, _ in boosts
        ]
        fusion: models.RrfQuery | models.FusionQuery = models.RrfQuery(rrf=models.Rrf(
            k=BOOST_RANK_CONSTANT,
            weights=[1.0] * len(prefetch) + [weight for _, weight in boosts],
        ))
        prefetch = prefetch + boosted
    else:
        fusion = models.FusionQuery(fusion=models.Fusion.RRF)

    if len(prefetch) > 1:
        try:
            results = qdrant_client.query_points(
                collection_name=settings.COLLECTION_NAME,
                prefetch=prefetch,
                query=fusion,
                limit=data.limit,
                offset=data.offset,
                with_payload=True,
            )
        except Exception as e:
            logger.exception("Query points error (fusion)")
            raise HTTPException(
                status_code=500, detail=f"Query points failed: {e!s}") from e
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
                status_code=500, detail=f"Query points failed: {e!s}") from e

    return [cast(dict[str, Any], p.payload) for p in results.points]
