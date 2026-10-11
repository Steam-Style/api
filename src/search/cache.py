"""
Search responses from the last few minutes.
"""

import threading
import time
from collections.abc import Hashable
from typing import Any


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
