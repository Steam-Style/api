"""
What a Steam account can buy in the Points Shop. Steam only sells a game's static backgrounds and emoticons, the ones
that come with its trading cards, to people who own the game. Everything else is sold to everyone. The games come
from the Steam Web API and are kept for a few minutes, so paging through results doesn't ask Steam again.
"""

import threading
import time

import httpx
from qdrant_client import models

OWNED_GAMES_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
OWNERS_ONLY = ["profile backgrounds", "emoticons"]
KEEP_SECONDS = 600
MAX_ACCOUNTS = 5000
TIMEOUT_SECONDS = 10


class OwnedGamesError(Exception):
    pass


class OwnedGames:
    def __init__(self, key: str | None) -> None:
        self.key = key
        self.client = httpx.Client(timeout=TIMEOUT_SECONDS)
        self.found: dict[int, tuple[float, frozenset[int] | None]] = {}
        self.lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return bool(self.key)

    def get(self, steamid: int) -> frozenset[int] | None:
        """The app IDs of the games the account owns, None when its game details are private"""
        now = time.monotonic()
        with self.lock:
            while self.found and now - next(iter(self.found.values()))[0] >= KEEP_SECONDS:
                self.found.pop(next(iter(self.found)))
            kept = self.found.get(steamid)
        if kept:
            return kept[1]

        try:
            response = self.client.get(OWNED_GAMES_URL, params={
                "key": self.key,
                "steamid": steamid,
                "include_played_free_games": "true",
                "include_free_sub": "true",
                "skip_unvetted_apps": "false",
            })
            response.raise_for_status()
            data = response.json().get("response") or {}
        except httpx.HTTPStatusError as error:
            raise OwnedGamesError(f"Steam answered {error.response.status_code}") from error
        except (httpx.HTTPError, ValueError) as error:
            raise OwnedGamesError(type(error).__name__) from error

        apps = frozenset(int(game["appid"]) for game in data.get("games", [])) if "game_count" in data else None
        with self.lock:
            self.found.pop(steamid, None)
            self.found[steamid] = (now, apps)
            while len(self.found) > MAX_ACCOUNTS:
                self.found.pop(next(iter(self.found)))
        return apps


def unowned(apps: frozenset[int] | None) -> models.Filter:
    """The items Steam won't sell to an account with these games: static backgrounds and emoticons of other games"""
    must_not: list[models.Condition] = [models.FieldCondition(key="item.animated", match=models.MatchValue(value=True))]
    if apps:
        must_not.append(models.FieldCondition(key="app.id", match=models.MatchAny(any=sorted(apps))))
    return models.Filter(
        must=[models.FieldCondition(key="item.category", match=models.MatchAny(any=OWNERS_ONLY))],
        must_not=must_not,
    )
