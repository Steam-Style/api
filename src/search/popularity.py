"""
The Points Shop's popular order, kept in memory and fetched from Steam again once a day. It isn't stored with the
items, because Qdrant rewrites an item whole, vectors and all, whenever its payload changes, and most items move in
the order every day.
"""

import logging
import threading
import time
from array import array

import httpx
from qdrant_client import QdrantClient, models

QUERY_REWARD_ITEMS_URL = "https://api.steampowered.com/ILoyaltyRewardsService/QueryRewardItems/v1"
PAGE_SIZE = 1000
TIMEOUT_SECONDS = 30
REFRESH_SECONDS = 86400
RETRY_SECONDS = 900
MIN_COVERAGE = 0.99
IDS_PAGE_SIZE = 10_000
SORTED_CACHE_SECONDS = 600
SORTED_CACHE_SIZE = 32

logger = logging.getLogger(__name__)


def fetch_popular_order(client: httpx.Client) -> dict[int, int] | None:
    """
    Walks every page of the Points Shop in the order of its own "Popular" rows, most popular first. Leaving out
    sort_descending gives the opposite order. The order is the same for every category, so one place per item works
    with any filter.

    Returns:
        dict[int, int] | None: Each item's place, 1 being the most popular, None when the listing was cut short.
    """
    ranks: dict[int, int] = {}
    cursor: str | None = None
    total_count: int | None = None

    while True:
        params: dict[str, str | int] = {
            "count": PAGE_SIZE,
            "include_direct_purchase_disabled": "true",
            "sort": 2,
            "sort_descending": "false",
        }
        if cursor:
            params["cursor"] = cursor
        response = client.get(QUERY_REWARD_ITEMS_URL, params=params)
        response.raise_for_status()
        data = response.json().get("response", {})
        definitions = data.get("definitions", [])
        total_count = data.get("total_count", total_count)

        for definition in definitions:
            if definition.get("defid") is not None:
                ranks.setdefault(int(definition["defid"]), len(ranks) + 1)

        if not definitions or data.get("next_cursor") in (None, cursor):
            break
        cursor = data["next_cursor"]

    if not total_count or len(ranks) < total_count * MIN_COVERAGE:
        logger.warning("The popular order had %d of %s items", len(ranks), total_count)
        return None
    return ranks


class PopularOrder:
    """
    Where each item stands in the Points Shop's popular order, and the items of a search sorted by it. Items Steam
    doesn't list, which are no longer sold, come after every listed item, by ID. The sorted items of a search are kept
    for a few minutes, so paging through them doesn't read them again.
    """

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name
        self.http = httpx.Client(timeout=TIMEOUT_SECONDS)
        self.ranks: dict[int, int] = {}
        self.fetched_at: float | None = None
        self.tried_at: float | None = None
        self.refreshing = threading.Lock()
        self.sorted: dict[tuple[str, bool], tuple[float, array]] = {}
        self.sorted_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return bool(self.ranks)

    def refresh(self) -> None:
        """
        Fetches the popular order again, keeping the old one when it can't be fetched.
        """
        try:
            ranks = fetch_popular_order(self.http)
        except (httpx.HTTPError, ValueError):
            logger.exception("Could not fetch the popular order")
            return

        if ranks is None:
            return

        self.ranks = ranks
        self.fetched_at = time.monotonic()
        with self.sorted_lock:
            self.sorted.clear()
        logger.info("Loaded the popular order of %d items", len(ranks))

    def refresh_in_background(self) -> None:
        """
        Starts fetching the order when it's older than a day, without waiting for it. After a failed try it waits a
        while before trying again.
        """
        now = time.monotonic()
        if self.fetched_at is not None and now - self.fetched_at < REFRESH_SECONDS:
            return
        if self.tried_at is not None and now - self.tried_at < RETRY_SECONDS:
            return
        if not self.refreshing.acquire(blocking=False):
            return
        self.tried_at = now

        def run() -> None:
            try:
                self.refresh()
            finally:
                self.refreshing.release()

        threading.Thread(target=run, daemon=True).start()

    def page(self, query_filter: models.Filter, offset: int, limit: int, least_popular: bool) -> list[int]:
        """
        The IDs of a page of the items that pass the filters, most popular first, or least popular first.
        """
        self.refresh_in_background()
        return list(self._sorted_ids(query_filter, least_popular)[offset:offset + limit])

    def _sorted_ids(self, query_filter: models.Filter, least_popular: bool) -> array:
        key = (query_filter.model_dump_json(), least_popular)
        now = time.monotonic()
        with self.sorted_lock:
            kept = self.sorted.get(key)
            if kept is not None and now - kept[0] < SORTED_CACHE_SECONDS:
                return kept[1]

        ranks = self.ranks
        after_listed = len(ranks) + 1
        item_ids = sorted(
            self._matching_ids(query_filter),
            key=lambda item_id: (ranks.get(item_id, after_listed), item_id),
            reverse=least_popular,
        )
        found = array("q", item_ids)

        with self.sorted_lock:
            self.sorted.pop(key, None)
            self.sorted[key] = (now, found)
            while len(self.sorted) > SORTED_CACHE_SIZE:
                self.sorted.pop(next(iter(self.sorted)))
        return found

    def _matching_ids(self, query_filter: models.Filter) -> list[int]:
        item_ids: list[int] = []
        offset: models.ExtendedPointId | None = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=IDS_PAGE_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            item_ids.extend(int(point.id) for point in points)
            if offset is None:
                return item_ids
