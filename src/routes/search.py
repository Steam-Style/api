"""
The /search route.
"""

from typing import Any
from urllib.parse import unquote_plus

from litestar import get
from litestar.exceptions import HTTPException
from litestar.openapi import ResponseSpec
from litestar.params import Parameter
from qdrant_client import models

from clients import app_index, collection_features, owned_apps, popular_order
from schemas import Error, SearchResults
from search.cache import SearchCache
from search.filters import SearchRequest, build_query_filter
from search.links import parse_reference
from search.owners import STEAMID_MAX, STEAMID_MIN
from search.parsing import CATEGORIES, parse_query
from search.ranking import (
    boost_conditions,
    build_prefetch,
    find_emoticon,
    find_item_by_name,
    matching_items,
    popular_items,
    query_items,
    scroll_items,
)

SEARCH_CACHE_SECONDS = 300
SEARCH_CACHE_SIZE = 500
MAX_QUERY_LENGTH = 200
MAX_SEARCH_COLORS = 10
MAX_OFFSET = 10_000
POPULAR_SORTS = ("popular", "least_popular")

search_cache = SearchCache(SEARCH_CACHE_SECONDS, SEARCH_CACHE_SIZE)


def _search(
    search_query: str | None,
    similar_to: int | None,
    app_id: int | None,
    color: list[str] | None,
    category: list[str] | None,
    limit: int,
    offset: int,
    sort: str | None,
    animated: bool | None,
    tiled: bool | None,
    transparent: bool | None,
    sold_separately: bool | None,
    include_unavailable: bool,
    max_points: int | None,
    owner: int | None,
    parse: bool,
) -> dict[str, Any]:
    """Searches the catalogue, see search_items"""
    understood = None
    parsed = None
    category_values = category if category is not None else ["all"]
    if max_points is not None and not collection_features().prices:
        raise HTTPException(status_code=503, detail="Filtering by price isn't available yet")
    owned = owned_apps(owner) if owner is not None else None
    owned_games_count = len(owned) if owned is not None else None
    requested_categories = [
        unquote_plus(value).lower().strip() for value in category_values if value and value.strip()
    ]

    reference = parse_reference(search_query) if parse and search_query else None
    item_link = reference is not None and not reference.bare and (
        reference.item_id is not None or reference.item_name is not None
    )
    linked_item: int | None = None
    pinned_item: int | None = None
    emoticon_code: str | None = None

    if reference is not None and reference.item_name is not None and reference.app_id is not None:
        found = find_item_by_name(reference.app_id, reference.item_name)
        linked_item = (found or {}).get("item", {}).get("id")
        search_query = None
    elif reference is not None and reference.item_name is not None:
        emoticon_code = reference.item_name
    elif reference is not None and reference.item_id is not None:
        if reference.bare:
            pinned_item = reference.item_id
        else:
            linked_item = reference.item_id
            search_query = None
    elif reference is not None and reference.app_id is not None:
        app_name = app_index.apps.get(reference.app_id)
        app_id = reference.app_id if app_id is None else app_id
        search_query = None
        understood = {
            "query": None,
            "animated": None,
            "tiled": None,
            "transparent": None,
            "categories": [],
            "apps": [{"id": reference.app_id, "name": app_name}] if app_name else [],
        }

    if parse and search_query and search_query.strip():
        allowed = CATEGORIES if "all" in requested_categories else requested_categories
        parsed = parse_query(search_query, allowed, app_index if app_id is None else None)

        if parsed.found:
            search_query = parsed.text or None
            understood = {
                "query": parsed.text or None,
                "animated": parsed.animated,
                "tiled": parsed.tiled,
                "transparent": parsed.transparent,
                "categories": parsed.categories,
                "apps": [{"id": found_id, "name": name} for found_id, name in parsed.apps],
            }

            if not search_query and not color and similar_to is None:
                animated = parsed.animated if animated is None else animated
                tiled = parsed.tiled if tiled is None else tiled
                transparent = parsed.transparent if transparent is None else transparent
                if parsed.categories:
                    category_values = parsed.categories
                parsed = None

    data = SearchRequest(
        query=search_query,
        similar_to=similar_to,
        app_id=app_id,
        colors=color,
        category=category_values,
        limit=limit,
        offset=offset,
        sort=sort,
        animated=animated,
        tiled=tiled,
        transparent=transparent,
        sold_separately=sold_separately,
        include_unavailable=include_unavailable,
        owned=sorted(owned or ()) if owner is not None else None,
        max_points=max_points,
    )

    normalized_categories = [
        unquote_plus(category_value).lower().strip()
        for category_value in data.category
        if category_value and category_value.strip()
    ]

    if category is not None and not normalized_categories:
        return {"results": [], "understood": understood, "owned_games": owned_games_count}

    has_query = data.query is not None and len(data.query.strip()) > 0
    has_similar = data.similar_to is not None
    has_colors = data.colors is not None and len(data.colors) > 0

    query_filter = build_query_filter(data)
    if emoticon_code is not None:
        pinned_item = find_emoticon(emoticon_code, query_filter)

    if item_link:
        linked = matching_items([linked_item], query_filter) if linked_item is not None and data.offset == 0 else []
        return {"results": linked, "understood": None, "owned_games": owned_games_count}

    pinned = matching_items([pinned_item], query_filter) if pinned_item is not None else []
    if pinned:
        query_filter = models.Filter(must=[query_filter], must_not=[models.HasIdCondition(has_id=[pinned_item])])
        if data.offset == 0:
            data = data.model_copy(update={"limit": data.limit - 1})
        else:
            pinned = []
            data = data.model_copy(update={"offset": data.offset - 1})
        if data.limit == 0:
            return {"results": pinned, "understood": understood, "owned_games": owned_games_count}

    if not has_query and not has_colors and not has_similar:
        if data.sort in POPULAR_SORTS and popular_order.ready:
            browsed = popular_items(query_filter, data.limit, data.offset, data.sort == "least_popular")
        else:
            popular_order.refresh_in_background()
            sort = "newest" if data.sort in POPULAR_SORTS else data.sort
            browsed = scroll_items(query_filter=query_filter, limit=data.limit, offset=data.offset, sort=sort)
        return {"results": pinned + browsed, "understood": understood, "owned_games": owned_games_count}

    prefetch = build_prefetch(data, query_filter)

    if not prefetch:
        raise HTTPException(
            status_code=503, detail="Failed to generate embeddings")

    boosts = boost_conditions(parsed, data, requested_categories) if parsed else []

    return {
        "results": pinned + query_items(data, query_filter, prefetch, boosts),
        "understood": understood,
        "owned_games": owned_games_count,
    }



@get(
    "/search",
    tags=["Search"],
    sync_to_thread=True,
    summary="Search items",
    response_description="Matching items, best match first when searching",
    description=(
        "Browse the catalogue, or search it by text, colors or another item.\n\n"
        "- Without `query`, `color` or `similar_to` you're browsing, and results come in the order set by `sort`.\n"
        "- With any of them, results are ranked by how well they match. Using several at once, like a text search "
        "with colors, combines the rankings.\n\n"
        "The other filters work the same either way. Page through results with `limit` and `offset`."
    ),
    responses={
        404: ResponseSpec(Error, description="The `similar_to` item doesn't exist"),
        503: ResponseSpec(Error, description="Text search isn't available right now"),
    },
)
def search_items(
    search_query: str | None = Parameter(
        query="query", default=None, max_length=MAX_QUERY_LENGTH,
        description=(
            "What you're looking for, like `rainy city at night`. Matches what items look like, not only their names. "
            "A link to an item, like `https://store.steampowered.com/points/shop/reward/75843` or a Community Market "
            "listing, returns only that item. A link to a game's Points Shop, store or Market page returns its items. "
            "An item ID like `75843` or an emoticon's code like `:steamhappy:` puts that item first, then searches as "
            "usual. Links, IDs and codes follow the other filters, and `parse=false` searches them as words"
        )),
    similar_to: int | None = Parameter(
        default=None, ge=0,
        description="ID of an item to find lookalikes of. The item itself is left out"),
    app_id: int | None = Parameter(
        default=None, ge=0, description="Only items from this Steam app"),
    color: list[str] | None = Parameter(
        default=None, max_items=MAX_SEARCH_COLORS,
        description="Hex color to match, like `#1a9fff` or `1a9fff`. Repeat to match several colors, up to 10"),
    category: list[str] | None = Parameter(
        default=None,
        description=(
            "Only items from this category. Repeat for several. One of `avatars`, `avatar frames`, "
            "`profile backgrounds`, `mini-profile backgrounds`, `game profiles`, `emoticons`, `animated stickers`, "
            "`chat effects`, `steam deck keyboards` or `steam startup movies`. Leave it out or use `all` for "
            "every category, an empty value returns nothing"
        ),
    ),
    limit: int = Parameter(default=10, ge=1, le=100, description="How many results to return"),
    offset: int = Parameter(
        default=0, ge=0, le=MAX_OFFSET, description="How many results to skip, for paging, up to 10,000"),
    sort: str | None = Parameter(
        default="popular",
        description=(
            "Order when browsing: `popular` and `least_popular` follow the Points Shop's own popular order, or "
            "`newest`, `oldest`, `updated` or `random`. Searches are always ordered by best match"
        ),
    ),
    animated: bool | None = Parameter(
        default=None, description="`true` for only animated items, `false` to leave them out"),
    tiled: bool | None = Parameter(
        default=None, description="`true` for only tiled backgrounds, `false` to leave them out"),
    transparent: bool | None = Parameter(
        default=None, description="`true` for only items with transparent parts, `false` to leave them out"),
    sold_separately: bool | None = Parameter(
        default=None,
        description=(
            "`true` for only items you can buy on their own, `false` for only items that come with a game profile "
            "or aren't sold to everyone"
        ),
    ),
    include_unavailable: bool = Parameter(
        default=False,
        description="`true` to also return items the Points Shop no longer sells, which are left out by default"),
    max_points: int | None = Parameter(
        default=None, ge=0, description="Only items that cost at most this many points, `0` for free items"),
    owner: int | None = Parameter(
        default=None, ge=STEAMID_MIN, le=STEAMID_MAX,
        description=(
            "SteamID64 of an account, for only items it can buy on their own. Steam sells a game's static "
            "backgrounds and emoticons only to people who own the game. The account's game details have to be "
            "public, when they're private every such item is left out"
        ),
    ),
    parse: bool = Parameter(
        default=True,
        description=(
            "Words in `query` like `animated`, `static`, `background` or a game's name rank matching items higher, "
            "and the response says what was understood. When the query is only such words, like `animated "
            "background`, they filter instead. `false` searches for the exact words"
        ),
    ),
) -> SearchResults:
    key = (
        search_query, similar_to, app_id, tuple(color or ()), tuple(category) if category is not None else None,
        limit, offset, sort, animated, tiled, transparent, sold_separately, include_unavailable, max_points, owner, parse,
    )
    cacheable = sort != "random"
    cached = search_cache.get(key) if cacheable else None
    if cached is not None:
        return cached

    response = _search(
        search_query, similar_to, app_id, color, category, limit, offset, sort, animated, tiled, transparent,
        sold_separately, include_unavailable, max_points, owner, parse,
    )
    if cacheable:
        search_cache.put(key, response)
    return response
