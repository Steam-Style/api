import logging
import threading
import time
from collections.abc import Hashable
from functools import lru_cache
from typing import Any, NamedTuple, cast
from urllib.parse import unquote_plus

import uvicorn
from litestar import Litestar, get, post
from litestar.config.cors import CORSConfig
from litestar.exceptions import HTTPException
from litestar.openapi import OpenAPIConfig, ResponseSpec
from litestar.openapi.spec import Contact, Tag
from litestar.params import Parameter
from litestar.response import Redirect
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from steam_style_embeddings import ColorEmbedder, Embedding, SiglipEmbedder

from config import settings
from docs import SiteSwaggerRenderPlugin
from generator import CATEGORIES as PART_CATEGORIES
from generator import PARTS, GenerationError, Generator, neutral_colors, tone_words
from generator import Request as GenerationRequest
from owners import OwnedGames, OwnedGamesError, unowned
from query_parsing import CATEGORIES, AppIndex, ParsedQuery, colors_named, parse_query
from schemas import (
    Error,
    GenerateRequest,
    GenerateResults,
    Item,
    ItemBatch,
    ItemList,
    SearchResults,
)

BOOLEAN_FILTER_FIELDS = ["animated", "tiled", "transparent", "sold_separately"]
BOOST_RANK_CONSTANT = 10
KIND_BOOST = 2.0
APP_BOOST = 3.0
NAME_PHRASE_BOOST = 3.0
NAME_WORD_BOOST = 1.5
COLLECTION_CHECK_SECONDS = 600
GENERATION_SLOTS = 2
GENERATION_WAIT_SECONDS = 15
STEAMID_MIN = 76561197960265729
STEAMID_MAX = 76561202255233023
TEXT_CACHE_SIZE = 1024
SEARCH_CACHE_SECONDS = 300
SEARCH_CACHE_SIZE = 500
MAX_QUERY_LENGTH = 200
MAX_SEARCH_COLORS = 10
MAX_OFFSET = 10_000

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
    model_name=settings.MODEL_NAME, device=settings.DEVICE, load_vision=False)
qdrant_client = QdrantClient(url=settings.DATABASE_URL, timeout=10)
app_index = AppIndex(qdrant_client, settings.COLLECTION_NAME)
owned_games = OwnedGames(settings.STEAM_API_KEY)
generator = Generator(
    qdrant_client,
    settings.COLLECTION_NAME,
    reviews=lambda: app_index.reviews,
    titles=lambda: collection_features().titles,
    frames=lambda: collection_features().frames,
)
generation_slots = threading.BoundedSemaphore(GENERATION_SLOTS)


class CollectionFeatures(NamedTuple):
    frames: bool
    titles: bool
    prices: bool


_collection_features: tuple[float, CollectionFeatures] | None = None


def collection_features() -> CollectionFeatures:
    """
    Whether the collection stores the frames of animated items, so text searches can match any frame, has a word index
    on item titles, so items named like the search rank higher, and has prices as numbers to filter by. Collections
    made before those were added have none of them. It's read again every few minutes, so they're picked up without a
    restart.
    """
    global _collection_features
    now = time.monotonic()
    if _collection_features and now - _collection_features[0] < COLLECTION_CHECK_SECONDS:
        return _collection_features[1]

    try:
        info = qdrant_client.get_collection(collection_name=settings.COLLECTION_NAME)
    except Exception:
        logger.exception("Could not read the collection config")
        return CollectionFeatures(frames=False, titles=False, prices=False)

    vectors = info.config.params.vectors
    schema = info.payload_schema or {}
    title = schema.get("item.title")
    features = CollectionFeatures(
        frames=isinstance(vectors, dict) and "frames" in vectors,
        titles=title is not None and title.data_type == models.PayloadSchemaType.TEXT,
        prices=getattr(schema.get("item.point_cost"), "data_type", None) == models.PayloadSchemaType.INTEGER,
    )
    _collection_features = (now, features)
    return features


@lru_cache(maxsize=TEXT_CACHE_SIZE)
def _text_embedding(text: str) -> tuple[float, ...]:
    """Embeds a search, kept for the next page of results or the next person searching the same. Failures aren't kept"""
    embedding = siglip_embedder.get_text_embedding(text)
    if embedding is None:
        raise ValueError("The model returned no embedding")
    return tuple(embedding)


def get_text_embedding(text: str) -> Embedding | None:
    """
    The embedding of a search. SigLIP 2 was trained on lowercase text, and capitals make it far worse at finding
    things by name, so the text is lowercased first.
    """
    if not siglip_embedder.is_ready():
        return None

    try:
        return list(_text_embedding(text.lower()))
    except Exception as e:
        logger.error("Error getting text embedding: %s", e)
        return None


class SearchCache:
    """
    Search responses from the last few minutes. Paging back and forth, the same search again and crawlers opening item
    pages, which each ask for similar items, then don't search again. The catalogue only changes once an hour anyway.
    """

    def __init__(self, seconds: float, size: int) -> None:
        self.seconds = seconds
        self.size = size
        self.found: dict[Hashable, tuple[float, dict[str, Any]]] = {}
        self.lock = threading.Lock()

    def get(self, key: Hashable) -> dict[str, Any] | None:
        with self.lock:
            kept = self.found.get(key)
            if kept is None or time.monotonic() - kept[0] >= self.seconds:
                self.found.pop(key, None)
                return None
            return kept[1]

    def put(self, key: Hashable, response: dict[str, Any]) -> None:
        with self.lock:
            self.found.pop(key, None)
            self.found[key] = (time.monotonic(), response)
            while len(self.found) > self.size:
                self.found.pop(next(iter(self.found)))


search_cache = SearchCache(SEARCH_CACHE_SECONDS, SEARCH_CACHE_SIZE)


def owned_apps(owner: int) -> frozenset[int] | None:
    """The games a Steam account owns, None when its game details are private"""
    if not owned_games.ready:
        raise HTTPException(status_code=503, detail="Checking which games an account owns isn't set up on this server")
    try:
        return owned_games.get(owner)
    except OwnedGamesError as e:
        logger.warning("Could not get owned games: %s", e)
        raise HTTPException(status_code=502, detail="Couldn't get the account's games from Steam, try again in a bit") from e


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
        default=None, description="Sort by: 'newest', 'oldest', 'updated', 'random'")
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


def _build_query_filter(data: SearchRequest) -> models.Filter:
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

    if data.max_points is not None:
        must_conditions.append(
            models.FieldCondition(key="item.point_cost", range=models.Range(lte=data.max_points))
        )

    return models.Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )


def _get_sort_query(sort: str | None) -> models.OrderByQuery | models.SampleQuery | None:
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


def _scroll_items(
    query_filter: models.Filter,
    limit: int,
    offset: int,
    sort: str | None,
) -> list[dict[str, Any]]:
    try:
        results = qdrant_client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=_get_sort_query(sort),
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


def _build_prefetch(
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


def _boost_conditions(
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


def _query_items(
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


@get("/", include_in_schema=False)
async def index() -> dict:
    return {"status": "ok", "message": "Steam Style Query API"}


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

    query_filter = _build_query_filter(data)

    if not has_query and not has_colors and not has_similar:
        return {
            "results": _scroll_items(
                query_filter=query_filter,
                limit=data.limit,
                offset=data.offset,
                sort=data.sort,
            ),
            "understood": understood,
            "owned_games": owned_games_count,
        }

    prefetch = _build_prefetch(data, query_filter)

    if not prefetch:
        raise HTTPException(
            status_code=503, detail="Failed to generate embeddings")

    boosts = _boost_conditions(parsed, data, requested_categories) if parsed else []

    return {
        "results": _query_items(data, query_filter, prefetch, boosts),
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
        description="What you're looking for, like `rainy city at night`. Matches what items look like, not only their names"),
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
        default="newest",
        description=(
            "Order when browsing: `newest`, `oldest`, `updated` or `random`. "
            "Searches are always ordered by best match"
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
            "SteamID64 of an account, for only items it can buy. Steam sells a game's static backgrounds and "
            "emoticons only to people who own the game. The account's game details have to be public, when they're "
            "private every such item is left out"
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


@post(
    "/generate",
    tags=["Generate"],
    sync_to_thread=True,
    status_code=200,
    summary="Generate profiles",
    response_description="Profiles that fit the request, best first",
    description=(
        "Puts whole profiles together: a background, mini-profile, animated avatar and avatar frame that fit what you "
        "ask for and each other, plus a theme that suits the background, either a free one or a game profile's.\n\n"
        "Start from a prompt, colors, items to keep or a game, or several of them at once. The options come back with "
        "different backgrounds. Pass the items you've already seen as `exclude` to get new ones.\n\n"
        "When no frame fits well enough, the option has none. When no avatar does, `avatar_from_background` is true: "
        "the avatar is cut from the background behind it and uploaded as a custom one.\n\n"
        "To keep a game profile, pass it as `game_profile` and leave the parts it brings out of `parts`. Its theme is "
        "used, and when its background stays, the options have no background and match the game profile's instead."
    ),
    responses={
        400: ResponseSpec(Error, description="Nothing to start from, or an item to keep isn't the right kind"),
        404: ResponseSpec(Error, description="Nothing fits what was asked"),
        503: ResponseSpec(Error, description="Generating isn't available right now"),
    },
)
def generate_profiles(data: GenerateRequest) -> GenerateResults:
    parts = tuple(data.parts) if data.parts is not None else PARTS
    keep = data.keep or {}
    unknown = sorted({part for part in [*parts, *keep] if part not in PARTS})
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown part: {', '.join(unknown)}")
    if not [part for part in parts if part not in keep]:
        raise HTTPException(status_code=400, detail="Nothing left to fill, every part is kept")

    prompt = data.prompt.strip() if data.prompt and data.prompt.strip() else None
    animated = data.animated
    owned = owned_apps(data.owner) if data.owner is not None else None
    named_apps: tuple[int, ...] = ()
    understood = None

    if prompt:
        parsed = parse_query(prompt, list(PART_CATEGORIES.values()), app_index if data.app_id is None else None)
        if parsed.found:
            understood = {
                "query": parsed.text or None,
                "animated": parsed.animated,
                "tiled": parsed.tiled,
                "transparent": parsed.transparent,
                "categories": parsed.categories,
                "apps": [{"id": found_id, "name": name} for found_id, name in parsed.apps],
            }
            prompt = parsed.text or None
            animated = parsed.animated if animated is None else animated
            named_apps = tuple(found_id for found_id, _ in parsed.apps)

    prompt_vector = get_text_embedding(prompt) if prompt else None
    if prompt and prompt_vector is None:
        raise HTTPException(status_code=503, detail="Text search isn't available right now")

    colors = list(data.colors or []) or (colors_named(prompt) if prompt else [])
    color_vector = None
    tone_vector = None
    if colors:
        try:
            neutral = neutral_colors(colors)
            hued = [color for color in colors if color not in neutral]
            color_vector = color_embedder.query_to_embedding(hued) if hued else None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        tone = tone_words(colors)
        tone_vector = get_text_embedding(tone) if tone else None
        if tone and tone_vector is None:
            raise HTTPException(status_code=503, detail="Text search isn't available right now")

    request = GenerationRequest(
        prompt_vector=prompt_vector,
        prompt_text=prompt,
        colors=tuple(colors),
        color_vector=color_vector,
        tone_vector=tone_vector,
        keep=keep or None,
        game_profile=data.game_profile,
        parts=parts,
        app_id=data.app_id,
        named_apps=named_apps,
        animated=animated,
        budget=data.budget,
        exclude=tuple(data.exclude),
        owned=(owned or frozenset()) if data.owner is not None else None,
        count=data.count,
    )

    if not generation_slots.acquire(timeout=GENERATION_WAIT_SECONDS):
        raise HTTPException(status_code=503, detail="Too many profiles are being generated, try again in a moment")
    try:
        options = generator.generate(request)
    except GenerationError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except Exception as e:
        logger.exception("Generate error")
        raise HTTPException(status_code=500, detail=f"Generating failed: {e!s}") from e
    finally:
        generation_slots.release()

    return {"options": options, "understood": understood, "owned_games": len(owned) if owned is not None else None}


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


cors_config = CORSConfig(
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app = Litestar(
    route_handlers=[index, search_items, generate_profiles, list_items, get_items_batch, get_item_by_id, get_item_by_name],
    openapi_config=OpenAPIConfig(
        title="Steam Style API",
        version="1.0.0",
        description=(
            "Search every item in the Steam Points Shop by what it looks like: avatars, frames, backgrounds, "
            "emoticons, stickers and more. This is the API behind [steam.style](https://steam.style), and it's "
            "free to use without a key.\n\n"
            "Item IDs are the same as in the Points Shop, so an item's page on Steam is "
            "`https://store.steampowered.com/points/shop/reward/{item_id}`.\n\n"
            "Found a bug or missing something? Open an issue on [GitHub](https://github.com/Steam-Style/api)."
        ),
        contact=Contact(name="Steam Style", url="https://steam.style", email="info@steam.style"),
        tags=[
            Tag(name="Search", description="Find items by text, colors or looks, or browse the whole catalogue"),
            Tag(name="Items", description="Get single items, or list every item there is"),
            Tag(name="Generate", description="Put whole profiles together from a prompt, colors or items to keep"),
        ],
        path="/docs",
        render_plugins=[SiteSwaggerRenderPlugin()],
        root_schema_site="swagger"
    ),
    cors_config=cors_config,
    on_startup=[app_index.refresh_in_background, generator.catalogue.load_in_background],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)