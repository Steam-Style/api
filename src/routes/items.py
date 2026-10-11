"""
The routes for single items, batches of items and the whole catalogue.
"""

import logging
from typing import Any, cast

from litestar import get
from litestar.exceptions import HTTPException
from litestar.openapi import ResponseSpec
from litestar.params import Parameter
from litestar.response import Redirect
from qdrant_client import models

from clients import qdrant_client
from config import settings
from schemas import Error, Item, ItemBatch, ItemList

logger = logging.getLogger(__name__)


@get(
    "/item/{app_id:int}/{item_name:str}",
    tags=["Items"],
    sync_to_thread=True,
    summary="Find an item by name",
    response_description="Redirects to the item",
    description="Looks up an item by its app and exact name, then redirects to `/item/{item_id}`.",
    responses={404: ResponseSpec(Error, description="No item with that name in the app")},
)
def get_item_by_name(
    app_id: int = Parameter(description="Steam app ID the item comes from"),
    item_name: str = Parameter(description="Exact item name, like `Dying Light 2 Stay Human Profile`"),
) -> Item | Redirect:
    try:
        results, _ = qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="app.id",
                        match=models.MatchValue(value=app_id)
                    ),
                    models.FieldCondition(
                        key="item.name",
                        match=models.MatchValue(value=item_name)
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
            status_code=500, detail=f"Database error: {e!s}") from e

    if not results:
        raise HTTPException(status_code=404, detail="Item not found")

    result_item = cast(dict[str, Any], results[0].payload)
    item_id = result_item.get("item", {}).get("id")

    if item_id is None:
        return cast(dict[str, Any], results[0].payload)
    else:
        return Redirect(f"/item/{item_id}")


@get(
    "/item/{item_id:int}",
    tags=["Items"],
    sync_to_thread=True,
    summary="Get an item",
    response_description="The item",
    description="Everything about one item, including its images, videos and Steam links.",
    responses={404: ResponseSpec(Error, description="No item with that ID")},
)
def get_item_by_id(
    item_id: int = Parameter(description="Item ID, the same as its ID in the Points Shop"),
) -> Item:
    if item_id < 0:
        raise HTTPException(status_code=404, detail="Item not found")

    try:
        results = qdrant_client.retrieve(
            collection_name=settings.COLLECTION_NAME,
            ids=[item_id],
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.exception("Retrieve error")
        raise HTTPException(
            status_code=500, detail=f"Database error: {e!s}") from e

    if not results:
        raise HTTPException(status_code=404, detail="Item not found")

    return cast(dict[str, Any], results[0].payload)


@get(
    "/items/batch",
    tags=["Items"],
    sync_to_thread=True,
    summary="Get several items",
    response_description="The items that exist, in the order they were asked for",
    description=(
        "Up to 50 items in one request, like everything used in a shared profile design. "
        "IDs that don't exist are left out."
    ),
)
def get_items_batch(
    item_ids: list[int] = Parameter(
        query="id", min_items=1, max_items=50, description="Item ID. Repeat for several, up to 50"),
) -> ItemBatch:
    wanted = list(dict.fromkeys(item_id for item_id in item_ids if item_id >= 0))

    if not wanted:
        return {"items": []}

    try:
        results = qdrant_client.retrieve(
            collection_name=settings.COLLECTION_NAME,
            ids=wanted,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.exception("Retrieve error")
        raise HTTPException(
            status_code=500, detail=f"Database error: {e!s}") from e

    found = {int(point.id): point.payload for point in results}
    return {"items": [found[item_id] for item_id in wanted if item_id in found]}


@get(
    "/items",
    tags=["Items"],
    sync_to_thread=True,
    summary="List every item",
    response_description="A page of item IDs",
    description=(
        "Every item ID with when it last changed, in ID order. Made for going through the whole catalogue, "
        "like building a sitemap. Start without a cursor, then pass the `next_cursor` of each page to get the next "
        "one, until it's null. Use `/item/{item_id}` for the details."
    ),
)
def list_items(
    cursor: int | None = Parameter(
        default=None, ge=0, description="Where to continue from, the `next_cursor` of the previous page"),
    limit: int = Parameter(default=1000, ge=1, le=50000, description="Items per page"),
) -> ItemList:
    try:
        total = qdrant_client.count(
            collection_name=settings.COLLECTION_NAME, exact=True).count
        points, next_cursor = qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            offset=cursor,
            limit=limit,
            with_payload=["timestamps.updated_at"],
            with_vectors=False,
        )
    except Exception as e:
        logger.exception("Scroll error")
        raise HTTPException(
            status_code=500, detail=f"Database error: {e!s}") from e

    return {
        "items": [
            {
                "id": point.id,
                "updated_at": (point.payload or {}).get("timestamps", {}).get("updated_at"),
            }
            for point in points
        ],
        "total": total,
        "next_cursor": next_cursor,
    }
