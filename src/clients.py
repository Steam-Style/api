"""
The connections and models every route shares: Qdrant, the embedders, the index of game names, owned games and the
generator, plus what the collection supports and the embedding of search text.
"""

import logging
import time
from functools import lru_cache
from typing import NamedTuple

from litestar.exceptions import HTTPException
from qdrant_client import QdrantClient, models
from steam_style_embeddings import ColorEmbedder, Embedding, SiglipEmbedder

from config import settings
from generator import Generator
from search.owners import OwnedGames, OwnedGamesError
from search.parsing import AppIndex
from search.popularity import PopularOrder

COLLECTION_CHECK_SECONDS = 600
TEXT_CACHE_SIZE = 1024

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
popular_order = PopularOrder(qdrant_client, settings.COLLECTION_NAME)
owned_games = OwnedGames(settings.STEAM_API_KEY)
generator = Generator(
    qdrant_client,
    settings.COLLECTION_NAME,
    reviews=lambda: app_index.reviews,
    titles=lambda: collection_features().titles,
    frames=lambda: collection_features().frames,
)


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


def owned_apps(owner: int) -> frozenset[int] | None:
    """The games a Steam account owns, None when its game details are private"""
    if not owned_games.ready:
        raise HTTPException(status_code=503, detail="Checking which games an account owns isn't set up on this server")
    try:
        return owned_games.get(owner)
    except OwnedGamesError as e:
        logger.warning("Could not get owned games: %s", e)
        raise HTTPException(status_code=502, detail="Couldn't get the account's games from Steam, try again in a bit") from e
