"""
Finds words in a search query that say what kind of item someone wants, like "animated", "background" or the name of
a game, so items that match them can be ranked higher. The rest of the query is what the items should look like.
"""
import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

CATEGORIES = [
    "avatars",
    "avatar frames",
    "profile backgrounds",
    "mini-profile backgrounds",
    "game profiles",
    "emoticons",
    "animated stickers",
    "chat effects",
    "steam deck keyboards",
    "steam startup movies",
]

PROPERTY_WORDS: dict[str, tuple[str, bool]] = {
    "animated": ("animated", True),
    "animation": ("animated", True),
    "animations": ("animated", True),
    "gif": ("animated", True),
    "gifs": ("animated", True),
    "static": ("animated", False),
    "tiled": ("tiled", True),
    "tiling": ("tiled", True),
    "seamless": ("tiled", True),
    "transparent": ("transparent", True),
}

CATEGORY_PHRASES: list[tuple[tuple[str, ...], str]] = sorted(
    [
        (("profile", "background"), "profile backgrounds"),
        (("profile", "backgrounds"), "profile backgrounds"),
        (("background",), "profile backgrounds"),
        (("backgrounds",), "profile backgrounds"),
        (("wallpaper",), "profile backgrounds"),
        (("wallpapers",), "profile backgrounds"),
        (("mini", "profile"), "mini-profile backgrounds"),
        (("mini", "profiles"), "mini-profile backgrounds"),
        (("mini", "profile", "background"), "mini-profile backgrounds"),
        (("mini", "profile", "backgrounds"), "mini-profile backgrounds"),
        (("miniprofile",), "mini-profile backgrounds"),
        (("miniprofiles",), "mini-profile backgrounds"),
        (("avatar", "frame"), "avatar frames"),
        (("avatar", "frames"), "avatar frames"),
        (("frame",), "avatar frames"),
        (("frames",), "avatar frames"),
        (("avatar",), "avatars"),
        (("avatars",), "avatars"),
        (("pfp",), "avatars"),
        (("pfps",), "avatars"),
        (("game", "profile"), "game profiles"),
        (("game", "profiles"), "game profiles"),
        (("emoticon",), "emoticons"),
        (("emoticons",), "emoticons"),
        (("emote",), "emoticons"),
        (("emotes",), "emoticons"),
        (("sticker",), "animated stickers"),
        (("stickers",), "animated stickers"),
        (("animated", "sticker"), "animated stickers"),
        (("animated", "stickers"), "animated stickers"),
        (("chat", "effect"), "chat effects"),
        (("chat", "effects"), "chat effects"),
        (("keyboard",), "steam deck keyboards"),
        (("keyboards",), "steam deck keyboards"),
        (("deck", "keyboard"), "steam deck keyboards"),
        (("deck", "keyboards"), "steam deck keyboards"),
        (("startup", "movie"), "steam startup movies"),
        (("startup", "movies"), "steam startup movies"),
        (("startup", "video"), "steam startup movies"),
        (("startup", "videos"), "steam startup movies"),
    ],
    key=lambda entry: len(entry[0]),
    reverse=True,
)

ALIASES = {
    "cs": "counter strike 2",
    "csgo": "counter strike 2",
    "cs go": "counter strike 2",
    "gmod": "garrys mod",
    "dbd": "dead by daylight",
    "apex": "apex legends",
    "r6": "tom clancys rainbow six siege",
    "r6s": "tom clancys rainbow six siege",
    "rainbow six": "tom clancys rainbow six siege",
    "rainbow six siege": "tom clancys rainbow six siege",
    "tboi": "binding of isaac",
    "isaac": "binding of isaac",
    "silksong": "hollow knight silksong",
    "dst": "dont starve together",
    "drg": "deep rock galactic",
    "sdv": "stardew valley",
    "skyrim": "elder scrolls v skyrim",
    "civ": "sid meiers civilization",
    "civilization": "sid meiers civilization",
    "civ5": "sid meiers civilization v",
    "civ 5": "sid meiers civilization v",
    "civ6": "sid meiers civilization vi",
    "civ 6": "sid meiers civilization vi",
    "civ7": "sid meiers civilization vii",
    "civ 7": "sid meiers civilization vii",
    "nms": "no mans sky",
    "mgsv": "metal gear solid v",
    "mgs5": "metal gear solid v",
    "tlou": "last of us part i",
    "gow": "god of war",
    "sts": "slay the spire",
    "poe": "path of exile",
    "poe2": "path of exile 2",
    "poe 2": "path of exile 2",
    "mhw": "monster hunter world",
    "mhr": "monster hunter rise",
    "mh wilds": "monster hunter wilds",
    "tarkov": "escape from tarkov",
    "eft": "escape from tarkov",
    "cod": "call of duty",
    "ksp": "kerbal space program",
    "fo76": "fallout 76",
    "ff7r": "final fantasy vii remake intergrade",
    "cp2077": "cyberpunk 2077",
    "p3": "persona 3 reload",
    "p4": "persona 4 golden",
    "p5": "persona 5 royal",
    "ror": "risk of rain",
}
MAX_ALIAS_WORDS = max(len(alias.split()) for alias in ALIASES)
STYLE_WORDS = {"cyberpunk", "steampunk", "synthwave", "retrowave", "vaporwave", "halloween", "christmas"}
PREFIX_STOP_WORDS = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "the", "to", "with"}
ROMAN_NUMERALS = {"ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9"}
SUBTITLE = re.compile(r":|\s[-–—]\s")
MIN_APP_NAME_LENGTH = 4
MIN_ABBREVIATION_LENGTH = 3
MAX_APP_NAME_WORDS = 8
COMMON_NAME_APPS = 5
KNOWN_REVIEWS = 1_000
ABBREVIATION_REVIEWS = 10_000
POPULAR_REVIEWS = 50_000
STORE_ITEMS_URL = "https://api.steampowered.com/IStoreBrowseService/GetItems/v1/"
REVIEWS_BATCH_SIZE = 100
REVIEWS_REFRESH_SECONDS = 86_400

App = tuple[int, str]


def words_of(text: str) -> list[str]:
    """
    Splits text into lowercase words without punctuation, so "Baldur's Gate 3: Deluxe" becomes baldurs, gate, 3 and
    deluxe.
    """
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"['’]", "", text.lower())).split()


def name_variants(name: str, prefixes: bool = False) -> set[str]:
    """
    The ways someone might write a game's name. Besides the full name that's the name without its subtitle, a leading
    "the" or its sequel number, so "The Witcher 3: Wild Hunt" is also "witcher 3" and "witcher", and "Hades II" is
    also "hades 2" and "hades". With `prefixes` the first words of the name count too, like "persona 5" for Persona 5
    Royal, which is only safe for well-known games.
    """
    variants: list[list[str]] = []

    if prefixes:
        words = words_of(name)
        words = words[1:] if words[:1] == ["the"] else words
        variants.extend(words[:length] for length in range(2, len(words)) if words[length - 1] not in PREFIX_STOP_WORDS)

    for words in (words_of(name), words_of(SUBTITLE.split(name, maxsplit=1)[0])):
        for trimmed in (words, words[1:] if words[:1] == ["the"] else []):
            if not trimmed:
                continue

            variants.append(trimmed)
            last = trimmed[-1]

            if len(trimmed) > 1 and (last.isdigit() or last in ROMAN_NUMERALS):
                variants.append(trimmed[:-1])
            if len(trimmed) > 1 and last in ROMAN_NUMERALS:
                variants.append([*trimmed[:-1], ROMAN_NUMERALS[last]])

    return {key for key in (" ".join(words) for words in variants) if len(key) >= MIN_APP_NAME_LENGTH}


def abbreviations_of(name: str) -> set[str]:
    """
    Short forms of a game's name made of the first letter of each word with its numbers kept, like "cs2" for
    Counter-Strike 2 or "l4d2" for Left 4 Dead 2. Only short forms with a number count, since short forms of letters
    alone are too often real words.
    """
    found: set[str] = set()

    for words in (words_of(name), words_of(SUBTITLE.split(name, maxsplit=1)[0])):
        short = "".join(
            word if word.isdigit() else ROMAN_NUMERALS[word] if index and word in ROMAN_NUMERALS else word[0]
            for index, word in enumerate(words)
        )

        if len(short) >= MIN_ABBREVIATION_LENGTH and re.search(r"\d", short) and re.search(r"[a-z]", short):
            found.add(short)

    return found


def expand_aliases(words: list[str]) -> tuple[list[str], bool]:
    """
    Writes out nicknames of games that can't be worked out from their names, like "csgo" or "gmod", and says whether
    there were any.
    """
    expanded: list[str] = []
    position = 0

    while position < len(words):
        for length in range(min(MAX_ALIAS_WORDS, len(words) - position), 0, -1):
            alias = " ".join(words[position:position + length])
            if alias in ALIASES:
                expanded.extend(ALIASES[alias].split())
                position += length
                break
        else:
            expanded.append(words[position])
            position += 1

    return expanded, expanded != words


@dataclass
class ParsedQuery:
    """
    What a query asked for. `text` is what the items should look like, without the words that describe the kind of
    item, and with short forms of game names written out.
    """

    text: str
    animated: bool | None = None
    tiled: bool | None = None
    transparent: bool | None = None
    categories: list[str] = field(default_factory=list)
    apps: list[App] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """
        Whether any part of the query describes the kind of item.
        """
        return (
            self.animated is not None
            or self.tiled is not None
            or self.transparent is not None
            or bool(self.categories)
            or bool(self.apps)
        )


class AppIndex:
    """
    The names of every game with items in the catalogue, kept in memory and refreshed in the background.
    """

    def __init__(self, client: QdrantClient, collection_name: str, refresh_seconds: int = 3600) -> None:
        self.client = client
        self.collection_name = collection_name
        self.refresh_seconds = refresh_seconds
        self.names: dict[str, list[App]] = {}
        self.abbreviations: dict[str, list[App]] = {}
        self.reviews: dict[int, int] = {}
        self.refreshed_at = 0.0
        self.reviews_fetched_at = -float(REVIEWS_REFRESH_SECONDS)
        self.refreshing = threading.Lock()

    def refresh(self) -> None:
        """
        Reads the app and name of every item and rebuilds the lookups, then again once the review counts of new games
        are known.
        """
        apps: dict[int, str] = {}
        items: list[tuple[int, str]] = []
        offset = None

        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=5000,
                    offset=offset,
                    with_payload=["app.id", "app.name", "item.name"],
                    with_vectors=False,
                )

                for point in points:
                    payload = point.payload or {}
                    app = payload.get("app") or {}
                    app_id, app_name = app.get("id"), app.get("name")
                    item_name = (payload.get("item") or {}).get("name")

                    if isinstance(app_id, int) and app_name:
                        apps.setdefault(app_id, app_name)
                        if item_name:
                            items.append((app_id, item_name))

                if offset is None:
                    break
        except Exception:
            logger.exception("Could not read the app names")
            return

        self.names, self.abbreviations = build_lookups(apps, items, self.reviews)
        self.refreshed_at = time.monotonic()

        stale = time.monotonic() - self.reviews_fetched_at >= REVIEWS_REFRESH_SECONDS
        missing = [app_id for app_id in apps if stale or app_id not in self.reviews]

        if missing:
            self.reviews = {**self.reviews, **fetch_review_counts(missing)}
            if stale:
                self.reviews_fetched_at = time.monotonic()
            self.names, self.abbreviations = build_lookups(apps, items, self.reviews)

        logger.info("Loaded %d names and %d short forms of %d apps", len(self.names), len(self.abbreviations), len(apps))

    def refresh_in_background(self) -> None:
        """
        Starts a refresh when the names are older than the refresh interval, without waiting for it.
        """
        if time.monotonic() - self.refreshed_at < self.refresh_seconds and self.names:
            return

        if not self.refreshing.acquire(blocking=False):
            return

        def run() -> None:
            try:
                self.refresh()
            finally:
                self.refreshing.release()

        threading.Thread(target=run, daemon=True).start()

    def find(self, words: list[str], taken: set[int]) -> tuple[int, int, list[App], bool] | None:
        """
        Finds the longest game name among the words that aren't taken yet. Returns where it is, the games it names and
        whether it was a short form.
        """
        self.refresh_in_background()

        for length in range(min(MAX_APP_NAME_WORDS, len(words)), 0, -1):
            for start in range(len(words) - length + 1):
                if any(position in taken for position in range(start, start + length)):
                    continue

                key = " ".join(words[start:start + length])
                if key in self.names:
                    return start, start + length, self.names[key], False
                if length == 1 and key in self.abbreviations:
                    return start, start + 1, self.abbreviations[key], True

        return None


def fetch_review_counts(app_ids: list[int]) -> dict[int, int]:
    """
    Asks the Steam store how many reviews each game has, which tells well-known games from obscure ones. Games the
    store doesn't know count as having none.
    """
    counts: dict[int, int] = {}

    with httpx.Client(timeout=30) as client:
        for start in range(0, len(app_ids), REVIEWS_BATCH_SIZE):
            batch = app_ids[start:start + REVIEWS_BATCH_SIZE]
            request = {
                "ids": [{"appid": app_id} for app_id in batch],
                "context": {"language": "english", "country_code": "US"},
                "data_request": {"include_reviews": True},
            }

            try:
                response = client.get(STORE_ITEMS_URL, params={"input_json": json.dumps(request, separators=(",", ":"))})
                response.raise_for_status()
                store_items = response.json().get("response", {}).get("store_items", [])
            except (httpx.HTTPError, ValueError) as e:
                logger.warning("Could not get review counts: %s", e)
                continue

            for store_item in store_items:
                summary = (store_item.get("reviews") or {}).get("summary_filtered") or {}
                if isinstance(store_item.get("appid"), int):
                    counts[store_item["appid"]] = summary.get("review_count") or 0

            for app_id in batch:
                counts.setdefault(app_id, 0)

    return counts


def build_lookups(
    apps: dict[int, str],
    items: list[tuple[int, str]],
    reviews: dict[int, int],
) -> tuple[dict[str, list[App]], dict[str, list[App]]]:
    """
    Maps every way of writing each game's name to the games it can mean.

    A name that items of several other games are also called, like "room" for The Room, only counts for games with
    lots of reviews, since a search for it is otherwise more likely about the word. One-word names of games few
    people know count only when no other item uses the word. Short forms only count for games with plenty of reviews,
    and mean the one with the most.
    """
    names: dict[str, list[int]] = defaultdict(list)
    abbreviations: dict[str, list[int]] = defaultdict(list)

    for app_id, app_name in apps.items():
        for key in name_variants(app_name, prefixes=reviews.get(app_id, 0) >= POPULAR_REVIEWS):
            names[key].append(app_id)
        if reviews.get(app_id, 0) >= ABBREVIATION_REVIEWS:
            for key in abbreviations_of(app_name):
                abbreviations[key].append(app_id)

    used_by: dict[str, set[int]] = defaultdict(set)

    for app_id, item_name in items:
        words = words_of(item_name)
        for length in range(1, min(MAX_APP_NAME_WORDS, len(words)) + 1):
            for start in range(len(words) - length + 1):
                key = " ".join(words[start:start + length])
                if key in names and app_id not in names[key]:
                    used_by[key].add(app_id)

    def means(key: str, app_id: int) -> bool:
        if key in STYLE_WORDS:
            return False
        if len(used_by[key]) >= COMMON_NAME_APPS:
            return reviews.get(app_id, 0) >= POPULAR_REVIEWS
        return " " in key or reviews.get(app_id, 0) >= KNOWN_REVIEWS or not used_by[key]

    name_lookup = {
        key: [(app_id, apps[app_id]) for app_id in app_ids if means(key, app_id)] for key, app_ids in names.items()
    }
    abbreviation_lookup = {
        key: [(best, apps[best])]
        for key, app_ids in abbreviations.items()
        if key not in names
        for best in [max(app_ids, key=lambda app_id: reviews.get(app_id, 0))]
    }

    return {key: found for key, found in name_lookup.items() if found}, abbreviation_lookup


def parse_query(query: str, allowed_categories: list[str], apps: AppIndex | None) -> ParsedQuery:
    """
    Finds the words in a query that describe the kind of item. Category words only count for categories the search
    includes anyway, so searching "avatar" among backgrounds still looks for avatars in the pictures. Game names stay
    in the text, since what a game's items look like is part of what someone searching for it wants, and short forms
    and nicknames of games are written out.
    """
    words, aliased = expand_aliases(words_of(query)) if apps is not None else (words_of(query), False)
    taken: set[int] = set()
    dropped: set[int] = set()
    written_out: dict[int, str] = {}
    parsed = ParsedQuery(text=query)

    if apps is not None:
        while match := apps.find(words, taken):
            start, end, found, short = match
            taken.update(range(start, end))
            parsed.apps.extend(app for app in found if app not in parsed.apps)
            if short and len(found) == 1:
                written_out[start] = found[0][1]

    for phrase, category in CATEGORY_PHRASES:
        if category not in allowed_categories:
            continue

        for start in range(len(words) - len(phrase) + 1):
            span = range(start, start + len(phrase))
            if tuple(words[start:start + len(phrase)]) == phrase and not any(position in taken for position in span):
                taken.update(span)
                dropped.update(span)
                if category not in parsed.categories:
                    parsed.categories.append(category)

    for position, word in enumerate(words):
        if position in taken or word not in PROPERTY_WORDS:
            continue

        name, value = PROPERTY_WORDS[word]
        if getattr(parsed, name) in (None, value):
            setattr(parsed, name, value)
            taken.add(position)
            dropped.add(position)

    if dropped or written_out or aliased:
        parsed.text = " ".join(
            written_out.get(position, word) for position, word in enumerate(words) if position not in dropped
        )

    return parsed
