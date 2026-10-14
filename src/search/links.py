"""
Recognises item IDs and links to items or games in a search, in the forms Steam, the Community Market and steam.style
use, so pasting where an item lives finds it.
"""

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, unquote, urlsplit

BARE_ID = re.compile(r"#?(\d{1,10})")
APP_TAG = re.compile(r"tag_app_(\d+)")
STEAM_COMMUNITY_APP = "753"


@dataclass(frozen=True)
class Reference:
    """
    What a search pointed at: an item by its ID, an item by its game and exact name, or a whole game.

    Attributes:
        item_id (int | None): The item's ID, which is its ID in the Points Shop.
        app_id (int | None): The game, alone for a link to a game, or with item_name for a Market listing.
        item_name (str | None): The item's exact name, from a Market listing.
        bare (bool): The search was only a number, which may also be words to search for, like 2077.
    """

    item_id: int | None = None
    app_id: int | None = None
    item_name: str | None = None
    bare: bool = False


def _number(value: str | None) -> int | None:
    """The value as an ID, None when it isn't a whole number"""
    return int(value) if value and value.isdigit() and len(value) <= 10 else None


def _points_shop(path: list[str]) -> Reference | None:
    """
    Reads a Points Shop address. Steam opens an item by adding /reward/<id> to any shop page, like the home page, a
    game or a category, shows game profiles and other bundles at /bundle/<id>, and a game's items at /app/<id>.
    """
    if path[:2] != ["points", "shop"]:
        return None

    if "reward" in path:
        after = path.index("reward") + 1
        item_id = _number(path[after]) if after < len(path) else None
        return Reference(item_id=item_id) if item_id is not None else None

    if len(path) >= 4 and path[2] == "bundle" and (item_id := _number(path[3])) is not None:
        return Reference(item_id=item_id)

    if len(path) >= 4 and path[2] == "app" and (app_id := _number(path[3])) is not None:
        return Reference(app_id=app_id)

    return None


def _community(path: list[str], query: dict[str, str]) -> Reference | None:
    """
    Reads a Steam Community address: a Market listing named "<app id>-<item name>", a Market search for a game's
    items, or a game's community hub.
    """
    if len(path) >= 4 and path[:3] == ["market", "listings", STEAM_COMMUNITY_APP]:
        app_part, _, name = path[3].partition("-")
        app_id = _number(app_part)
        return Reference(app_id=app_id, item_name=name) if app_id is not None and name else None

    if path[:2] == ["market", "search"]:
        for key, value in query.items():
            if key.startswith("category_753_Game") and (match := APP_TAG.fullmatch(value)):
                return Reference(app_id=int(match.group(1)))
        return None

    if len(path) >= 2 and path[0] == "app" and (app_id := _number(path[1])) is not None:
        return Reference(app_id=app_id)

    return None


def parse_reference(text: str) -> Reference | None:
    """
    Works out whether a search is an item ID or a link to an item or a game.

    Args:
        text (str): What was searched for.

    Returns:
        Reference | None: What it points at, None when it's an ordinary search.
    """
    text = text.strip()
    if match := BARE_ID.fullmatch(text):
        return Reference(item_id=int(match.group(1)), bare=True)

    if not text or any(character.isspace() for character in text):
        return None

    parts = urlsplit(text if "://" in text else f"https://{text}")
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = [unquote(segment) for segment in parts.path.split("/") if segment]
    query = dict(parse_qsl(parts.query))

    if host == "store.steampowered.com":
        if reference := _points_shop(path):
            return reference
        if len(path) >= 2 and path[0] == "app" and (app_id := _number(path[1])) is not None:
            return Reference(app_id=app_id)
        return None

    if host == "steamcommunity.com":
        return _community(path, query)

    if host == "steam.style":
        if len(path) >= 2 and path[0] == "item":
            if len(path) == 2 and (item_id := _number(path[1])) is not None:
                return Reference(item_id=item_id)
            if len(path) == 3 and (app_id := _number(path[1])) is not None:
                return Reference(app_id=app_id, item_name=path[2])
        if not path and (app_id := _number(query.get("app"))) is not None:
            return Reference(app_id=app_id)
        return _points_shop(path) or _community(path, query)

    if host == "stea.ms" and len(path) >= 2 and path[0] == "i" and (item_id := _number(path[1])) is not None:
        return Reference(item_id=item_id)

    return None
