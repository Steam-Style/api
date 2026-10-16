"""
Turns the filters of a search into a Qdrant filter, and the sort order into a query.
"""

from urllib.parse import unquote_plus

from pydantic import BaseModel, Field
from qdrant_client import models

from search.owners import unowned

BOOLEAN_FILTER_FIELDS = ["animated", "tiled", "transparent", "sold_separately"]


class SearchRequest(BaseModel):
    query: str | None = None
    similar_to: int | None = None
    app_id: int | None = None
    colors: list[str] | None = None
    category: list[str] = Field(
        default_factory=list,
        description="Filter by category. 'all'=all categories, empty=no items.",
    )
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    sort: str | None = Field(
        default=None, description="Sort by: 'popular', 'least_popular', 'newest', 'oldest', 'updated', 'random'")
    animated: bool | None = Field(
        default=None, description="True=only animated, False=exclude animated, None=all")
    tiled: bool | None = Field(
        default=None, description="True=only tiled, False=exclude tiled, None=all")
    transparent: bool | None = Field(
        default=None, description="True=only transparent, False=exclude transparent, None=all")
    sold_separately: bool | None = Field(
        default=None, description="True=only items sold on their own, False=only items that aren't, None=all")
    include_unavailable: bool = Field(
        default=False, description="True=also items the Points Shop no longer sells")
    owned: list[int] | None = Field(
        default=None, description="The games an account owns, for only items it can buy. None=everything")
    max_points: int | None = Field(default=None, description="Only items that cost at most this many points")


def build_query_filter(data: SearchRequest) -> models.Filter:
    must_conditions: list[models.Condition] = []
    must_not_conditions: list[models.Condition] = []

    if data.similar_to is not None:
        must_not_conditions.append(
            models.FieldCondition(
                key="item.id",
                match=models.MatchValue(value=data.similar_to),
            )
        )

    if data.app_id is not None:
        must_conditions.append(
            models.FieldCondition(
                key="app.id",
                match=models.MatchValue(value=data.app_id),
            )
        )

    decoded_categories = [
        unquote_plus(category).lower().strip()
        for category in data.category
        if category and category.lower().strip() != "all"
    ]
    if decoded_categories:
        category_conditions: list[models.Condition] = [
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

    if not data.include_unavailable:
        must_not_conditions.append(
            models.FieldCondition(
                key="item.available",
                match=models.MatchValue(value=False),
            )
        )

    if data.owned is not None:
        must_not_conditions.append(unowned(frozenset(data.owned)))
        must_not_conditions.append(
            models.FieldCondition(key="item.sold_separately", match=models.MatchValue(value=False))
        )

    if data.max_points is not None:
        must_conditions.append(
            models.FieldCondition(key="item.point_cost", range=models.Range(lte=data.max_points))
        )

    return models.Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )


def sort_query(sort: str | None) -> models.OrderByQuery | models.SampleQuery | None:
    if sort == "popular":
        return models.OrderByQuery(order_by=models.OrderBy(
            key="item.popularity", direction=models.Direction.ASC))
    if sort == "least_popular":
        return models.OrderByQuery(order_by=models.OrderBy(
            key="item.popularity", direction=models.Direction.DESC))
    if sort == "newest":
        return models.OrderByQuery(order_by=models.OrderBy(
            key="timestamps.created_at", direction=models.Direction.DESC))
    if sort == "oldest":
        return models.OrderByQuery(order_by=models.OrderBy(
            key="timestamps.created_at", direction=models.Direction.ASC))
    if sort == "updated":
        return models.OrderByQuery(order_by=models.OrderBy(
            key="timestamps.updated_at", direction=models.Direction.DESC))
    if sort == "random":
        return models.SampleQuery(sample=models.Sample.RANDOM)
    return None
