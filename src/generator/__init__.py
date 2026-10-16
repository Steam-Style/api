"""
Puts whole profiles together from the catalogue: a background, mini-profile, avatar and frame that fit what was asked
for and each other, as a handful of options that look different from each other, each with a theme that suits it.

Every measure is turned into how far it is from what two random items score, measured across the catalogue and kept
in stats.json, so the weights below mean the same thing for every part and pair.

Mini-profiles, avatars, frames and game profiles are few enough to keep in memory and search with numpy, so only
backgrounds are looked up in the database.

An avatar or frame has to add BLANK_SCORES to a profile to be worth having, first set to about what a quarter of the
avatars (not counting their frame) and half the frames in rated options added, then nudged by the next ratings. Below
that the frame is left out, and the avatar is cut from the background behind it, to upload as a custom one.

Backgrounds rank higher the better their best theme suits them (THEME_FIT): in ratings, the picked options had
themes that fit noticeably better than the rest. They also rank higher the better they look (QUALITY_WEIGHT), going by
how much more their picture is like "high quality art" than "low quality image" (four wordings averaged, the vector is
in stats.json). Picked options looked better that way in all three rounds of ratings.

Item names matching the prompt add a little (NAME_MATCH_BONUS, scaled like the prompt for each part): ratings showed
no preference for them, and a strong bonus let sets with the prompt in their names, like "Kasumi_Anime", take over.

Greys, near-black and near-white have no hue, and stored colour vectors leave them out: a black picture is stored as
dark red, and one with a small logo as the logo's colour. So ingestion stores how much of each item has no colour
(item.neutral), measured from its pixels. Most backgrounds are largely dark, so only shares past the first number of
NEUTRAL_MEASURED count, fully from the second. Colour comparisons treat that part as its own colour, so black goes with
black and white and not with dark red, and a theme for a neutral background is matched against grey.
Neutral colours asked for are also described in words (tone_vector) to find pictures, since colour search can't.

Themes are Steam's free ones, with the colours of the body.*Theme rules in profilev2.css, and every game profile's.
THEME_COLOR_WEIGHTS is how much each theme colour shows on a profile, the gradients and showcase headers the most.
"""

import colorsys
import json
import logging
import math
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from search.owners import unowned

logger = logging.getLogger(__name__)

PARTS = ("background", "mini_profile", "avatar", "frame")
CACHED_PARTS = ("mini_profile", "avatar", "frame")
CATEGORIES = {
    "background": "profile backgrounds",
    "mini_profile": "mini-profile backgrounds",
    "avatar": "avatars",
    "frame": "avatar frames",
}
GAME_PROFILES = "game profiles"
PART_NAMES = {
    "background": "a profile background",
    "mini_profile": "a mini-profile background",
    "avatar": "an avatar",
    "frame": "an avatar frame",
}
PAIRS = [(a, b) for index, a in enumerate(PARTS) for b in PARTS[index + 1:]]
CANDIDATE_FIELDS = ["item.id", "item.category", "item.animated", "item.point_cost", "item.title", "item.neutral", "app.id"]
THEME_FIELDS = ["item.id", "item.point_cost", "app.id", "profile.theme.colors"]

PROMPT_WEIGHTS = {"background": 1.0, "mini_profile": 0.5, "avatar": 0.7, "frame": 0.35}
COLOR_REQUEST_WEIGHTS = {"background": 0.9, "mini_profile": 0.5, "avatar": 0.6, "frame": 0.4}
PICTURE_WEIGHTS = {
    ("background", "mini_profile"): 0.6,
    ("background", "avatar"): 0.45,
    ("background", "frame"): 0.25,
    ("mini_profile", "avatar"): 0.3,
    ("mini_profile", "frame"): 0.15,
    ("avatar", "frame"): 0.45,
}
COLOR_WEIGHTS = {pair: 0.375 if pair[0] == "background" else 0.18 for pair in PAIRS}
COLOR_REQUEST_SCALE = (0.45, 0.24)
SET_BONUS = 0.35
NAMED_GAME_BONUS = 1.0
NAME_MATCH_BONUS = 0.4
QUALITY_WEIGHT = 1.0
ANIMATION_MATCH = 1.0
POPULARITY_WEIGHT = 0.2
POPULAR_REVIEWS = 1000

BLANK_SCORES = {"avatar": 4.8, "frame": 5.0}
BLANK_SHARE = 0.5

THEME_MATCH = 1.0
THEME_FIT = 1.0
THEME_SET_BONUS = 1.5
THEME_PART_BONUS = 0.3
GAME_THEME_MARGIN = 0.5
GAME_THEME_SHARE = 0.5

BACKGROUND_LIMIT = 60
NAMED_BACKGROUND_LIMIT = 30
PART_LIMIT = 36
SET_MATES_LIMIT = 60
ANCHOR_BACKGROUNDS = 6
ANCHOR_MATCHES = 12
POOL_SIZE = 36
TOP_COMBINATIONS = 4000
TOP_PER_BACKGROUND = 120
MAX_PART_USES = 2
SIMILAR_BACKGROUND = 0.6
CATALOGUE_REFRESH_SECONDS = 6 * 3600
CATALOGUE_PAGE = 256

COLOR_BINS = (24, 3, 3)
COLOR_SIGMAS = (0.05, 0.25, 0.25)
GREY_SATURATION = 0.12
NEUTRAL_SATURATION = 0.15
NEUTRAL_MEASURED = (0.7, 0.95)

THEME_COLOR_WEIGHTS = {
    "gradient-left": 1.0,
    "gradient-right": 1.0,
    "color-showcase-header": 1.0,
    "gradient-showcase-header-left": 1.0,
    "gradient-background": 0.5,
    "btn-background": 0.5,
    "gradient-background-left": 0.5,
    "gradient-background-right": 0.5,
}

FREE_THEMES = {
    "default": {
        "gradient-right": "rgba(109, 38, 44, 0.301)", "gradient-left": "rgba(50, 255, 193, 0.103)",
        "gradient-background": "rgba(34, 35, 48, 0.93)", "gradient-background-right": "rgba(109, 38, 44, 0)",
        "gradient-background-left": "rgba(50, 255, 193, 0.103)", "color-showcase-header": "rgba(43, 45, 68, 0.93)",
        "gradient-showcase-header-left": "rgba(115, 173, 184, 0.247)", "btn-background": "rgba(43, 52, 68)",
    },
    "summer": {
        "gradient-right": "rgba(252, 197, 16, 0.301)", "gradient-left": "rgba(9, 243, 153, 0.247)",
        "gradient-background": "rgba(51, 27, 16, 0.93)", "gradient-background-right": "rgba(252, 197, 16, 0)",
        "gradient-background-left": "rgba(48, 243, 9, 0.13)", "color-showcase-header": "rgba(70, 53, 31, 0.93)",
        "gradient-showcase-header-left": "rgba(33, 78, 76, 0.93)", "btn-background": "rgb(95, 76, 39)",
    },
    "midnight": {
        "gradient-right": "rgba(51, 54, 253, 0.233)", "gradient-left": "rgba(12, 85, 241, 0.37)",
        "gradient-background": "rgba(10, 14, 32, 0.93)", "gradient-background-right": "rgba(51, 54, 253, 0)",
        "gradient-background-left": "rgba(12, 85, 241, 0.13)", "color-showcase-header": "rgba(34, 32, 61, 0.93)",
        "gradient-showcase-header-left": "rgba(20, 33, 68, 0.93)", "btn-background": "rgb(38, 36, 68)",
    },
    "steel": {
        "gradient-right": "rgba(70, 106, 128, 0.233)", "gradient-left": "rgba(86, 120, 134, 0.37)",
        "gradient-background": "rgba(41, 46, 51, 0.93)", "gradient-background-right": "rgba(17, 24, 29, 0)",
        "gradient-background-left": "rgba(19, 27, 31, 0)", "color-showcase-header": "rgba(55, 62, 76, 0.93)",
        "gradient-showcase-header-left": "rgba(68, 83, 93, 0.93)", "btn-background": "rgb(66, 76, 92)",
    },
    "cosmic": {
        "gradient-right": "rgba(248, 70, 180, 0.301)", "gradient-left": "rgba(9, 243, 99, 0.247)",
        "gradient-background": "rgba(46, 13, 36, 0.93)", "gradient-background-right": "rgba(70, 227, 248, 0)",
        "gradient-background-left": "rgba(239, 243, 9, 0.13)", "color-showcase-header": "rgba(57, 24, 61, 0.93)",
        "gradient-showcase-header-left": "rgba(20, 60, 68, 0.93)", "btn-background": "rgb(90, 40, 92)",
    },
    "darkMode": {
        "gradient-right": "rgba(49, 49, 49, 0.233)", "gradient-left": "rgba(51, 51, 51, 0.37)",
        "gradient-background": "rgba(24, 24, 24, 0.93)", "gradient-background-right": "rgba(34, 34, 34, 0)",
        "gradient-background-left": "rgba(32, 32, 32, 0.13)", "color-showcase-header": "rgba(0, 0, 0, 0.5)",
        "gradient-showcase-header-left": "rgba(0, 0, 0, 0.5)", "btn-background": "rgba(0, 0, 0, 0.5)",
    },
}
GAME_PROFILE_THEME = "gameProfile"

NAME_STOP_WORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "into", "is", "it", "my", "of", "on", "or", "the", "to", "with",
    "background", "backgrounds", "profile", "profiles", "steam", "theme", "style", "aesthetic", "vibe", "vibes",
}

_stats = json.loads((Path(__file__).parent / "stats.json").read_text())
MEANS = {part: np.array(values, dtype=np.float32) for part, values in _stats["means"].items()}
PICTURE_SCALE = {tuple(pair.split(":")): values for pair, values in _stats["picture"].items()}
COLOR_SCALE = {tuple(pair.split(":")): values for pair, values in _stats["color"].items()}
PROMPT_SCALE = _stats["prompt"]
THEME_SCALE = _stats["theme"]
QUALITY_VECTOR = np.array(_stats["quality"]["vector"], dtype=np.float32)
QUALITY_SCALE = _stats["quality"]["scale"]


class GenerationError(Exception):
    """Raised when there's nothing to build a profile around (400), nothing fits (404) or it isn't ready yet (503)"""

    def __init__(self, message: str, status_code: int = 404) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Request:
    """
    What a profile is generated from. At least one of prompt, colors, keep, game_profile or app_id has to be given.
    The game in app_id is where the background comes from, the other parts lean towards it. Items named like the words
    of prompt_text rank higher. color_vector holds the colours with a hue and tone_vector the words for the neutral
    ones, both from colors (see neutral_colors). game_profile is a game profile that stays: its theme is every option's
    theme, and when the background isn't one of the parts it's the game profile's, which the rest is matched to. theme is
    one of Steam's free themes that stays the same way. reviews is how many store reviews each game has, so well-known
    games rank a little higher
    """

    prompt_vector: list[float] | None = None
    prompt_text: str | None = None
    colors: tuple[str, ...] = ()
    color_vector: list[float] | None = None
    tone_vector: list[float] | None = None
    keep: dict[str, int] | None = None
    game_profile: int | None = None
    theme: str | None = None
    parts: tuple[str, ...] = PARTS
    app_id: int | None = None
    named_apps: tuple[int, ...] = ()
    animated: bool | None = None
    budget: int | None = None
    exclude: tuple[int, ...] = ()
    owned: frozenset[int] | None = None
    count: int = 6
    reviews: dict[int, int] = field(default_factory=dict)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-9)


def _centered(part: str, picture: np.ndarray) -> np.ndarray:
    """
    A picture vector with what every item of its category has in common taken out. Searching Qdrant's plain vectors
    with it ranks items close to how the centred vectors would
    """
    return _normalize(picture - MEANS[part])


def _scale(pair: tuple[str, str], table: dict) -> tuple[float, float]:
    return table.get(pair) or table[(pair[1], pair[0])]


def _stem(word: str) -> str:
    """A word without its plural ending, so "flowers" and "flower" match"""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _name_words(text: str) -> frozenset[str]:
    """The words of a name or prompt that say what it's about, lowercased and without plural endings"""
    words = re.sub(r"[^a-z0-9]+", " ", re.sub(r"['’]", "", text.lower())).split()
    return frozenset(_stem(word) for word in words if word not in NAME_STOP_WORDS and len(word) > 1)


def _css_color(value: str) -> tuple[tuple[float, float, float], float] | None:
    """A CSS colour as hex or rgb()/rgba(), with commas or spaces, as RGB from 0 to 1 and its alpha"""
    value = value.strip().lower()
    try:
        if value.startswith("#"):
            digits = value[1:]
            if len(digits) in (3, 4):
                digits = "".join(digit * 2 for digit in digits)
            red, green, blue = (int(digits[index:index + 2], 16) / 255 for index in (0, 2, 4))
            return (red, green, blue), int(digits[6:8], 16) / 255 if len(digits) == 8 else 1.0
        numbers = re.findall(r"[\d.]+%?", value)
        red, green, blue = (float(number.rstrip("%")) / 255 for number in numbers[:3])
        alpha = 1.0
        if len(numbers) > 3:
            alpha = float(numbers[3][:-1]) / 100 if numbers[3].endswith("%") else float(numbers[3])
        return (red, green, blue), alpha
    except (ValueError, IndexError):
        return None


_hues = np.linspace(0, 1, COLOR_BINS[0], endpoint=False)
_saturations = np.linspace(1 / COLOR_BINS[1], 1, COLOR_BINS[1])
_values = np.linspace(1 / COLOR_BINS[2], 1, COLOR_BINS[2])
_palette = np.stack(np.meshgrid(_hues, _saturations, _values, indexing="ij"), axis=-1).reshape(-1, 3)


def _soft_histogram(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Colours given as hue, saturation and value spread over the palette, like the colour embedding does"""
    hue_gap = np.abs(points[:, None, 0] - _palette[None, :, 0])
    hue_gap = np.minimum(hue_gap, 1 - hue_gap)
    saturation_gap = points[:, None, 1] - _palette[None, :, 1]
    value_gap = points[:, None, 2] - _palette[None, :, 2]
    distance = sum((gap / sigma) ** 2 for gap, sigma in zip((hue_gap, saturation_gap, value_gap), COLOR_SIGMAS, strict=True))
    return (np.exp(-distance / 2) * weights[:, None]).sum(axis=0)



def _hues_and_saturations(histogram: np.ndarray) -> np.ndarray:
    """
    Which hues and how saturated, from colour histograms, leaving out how light or dark, as shares that add up to one.
    Theme colours are dark versions of a background's colours, so lightness would only get in the way
    """
    cube = np.maximum(histogram, 0).reshape(-1, *COLOR_BINS).sum(axis=-1)
    cube = cube.reshape(len(cube), -1)
    return cube / np.maximum(cube.sum(axis=1, keepdims=True), 1e-12)


def _color_profile(histogram: np.ndarray) -> np.ndarray:
    """The colour profile of colour histograms, compared by cosine"""
    return _normalize(np.sqrt(_hues_and_saturations(np.atleast_2d(histogram))))


_GREY = _hues_and_saturations(_soft_histogram(np.array([(hue, 0.05, 0.5) for hue in _hues]), np.ones(len(_hues))))[0]


def _stored_profile(vectors: np.ndarray, neutral: np.ndarray | None = None) -> np.ndarray:
    """
    The colour profile of stored colour vectors, which keep the square root of their histogram. The part of each
    picture without colour (`neutral`) counts as grey
    """
    shares = _hues_and_saturations(np.atleast_2d(vectors) ** 2)
    if neutral is not None:
        shares = (1 - neutral)[:, None] * shares + neutral[:, None] * _GREY[None, :]
    return _normalize(np.sqrt(shares))


def _neutral(measured: np.ndarray) -> np.ndarray:
    """
    How much of each picture has no colour, from 0 to 1, from the share measured from its pixels at ingestion,
    counting only shares past the first number of NEUTRAL_MEASURED and fully from the second
    """
    floor, full = NEUTRAL_MEASURED
    return np.clip((np.nan_to_num(measured, nan=0.0) - floor) / (full - floor), 0, 1).astype(np.float32)


def _color_similarity(first_color: np.ndarray, first_neutral: np.ndarray, second_color: np.ndarray,
                      second_neutral: np.ndarray) -> np.ndarray:
    """
    How alike the colours of every first item are to every second one, with the part of each without colour as a
    colour of its own: two black pictures are alike, a black one and a colourful one aren't
    """
    hued = np.sqrt(np.outer(1 - first_neutral, 1 - second_neutral)) * (first_color @ second_color.T)
    return hued + np.sqrt(np.outer(first_neutral, second_neutral))


def _theme_profile(colors: dict[str, str]) -> np.ndarray | None:
    """The colour profile of a theme, from its colours weighted by how much they show"""
    weighted = []
    for key, weight in THEME_COLOR_WEIGHTS.items():
        parsed = _css_color(colors.get(key) or "")
        if parsed is not None:
            weighted.append((parsed[0], weight * parsed[1]))
    return _palette_profile(weighted)


def _colors_profile(colors: tuple[str, ...]) -> np.ndarray:
    """The colour profile of the colours asked for, each counting the same"""
    profile = _palette_profile([(_hex_rgb(color), 1.0) for color in colors])
    return profile if profile is not None else np.zeros(COLOR_BINS[0] * COLOR_BINS[1], dtype=np.float32)


def _hex_rgb(color: str) -> tuple[float, float, float]:
    digits = color.lstrip("#")
    parsed = _css_color(f"#{digits}") if len(digits) in (3, 6) else None
    if parsed is None:
        raise ValueError(f"Invalid hex color: {color}")
    return parsed[0]


def neutral_colors(colors: list[str] | tuple[str, ...]) -> list[str]:
    """The colours without a hue to speak of: greys, near-black and near-white"""
    return [color for color in colors if colorsys.rgb_to_hsv(*_hex_rgb(color))[1] < NEUTRAL_SATURATION]


def tone_words(colors: list[str] | tuple[str, ...]) -> str | None:
    """
    The neutral colours asked for in words, to match against pictures: "a black and white monochrome picture" when
    they're all neutral, "a picture with black" next to colours with a hue
    """
    neutral = neutral_colors(colors)
    if not neutral:
        return None
    values = [colorsys.rgb_to_hsv(*_hex_rgb(color))[2] for color in neutral]
    words = [word for word, found in (
        ("black", any(value < 0.25 for value in values)),
        ("grey", any(0.25 <= value <= 0.8 for value in values)),
        ("white", any(value > 0.8 for value in values)),
    ) if found]
    if len(neutral) == len(colors):
        return f"a {' and '.join(words)} monochrome picture"
    return f"a picture with {' and '.join(words)}"


def _palette_profile(weighted: list[tuple[tuple[float, float, float], float]]) -> np.ndarray | None:
    """
    The colour profile of a few colours, each with how much it counts. Greys have no hue, so they count towards every
    hue a little
    """
    hsv, weights = [], []
    for rgb, weight in weighted:
        hue, saturation, value = colorsys.rgb_to_hsv(*rgb)
        if saturation < GREY_SATURATION:
            hsv.extend((any_hue, saturation, value) for any_hue in _hues)
            weights.extend([weight / len(_hues)] * len(_hues))
        else:
            hsv.append((hue, saturation, value))
            weights.append(weight)
    if not hsv or sum(weights) <= 0:
        return None

    return _color_profile(_soft_histogram(np.array(hsv), np.array(weights)))[0]


@dataclass
class Pool:
    """
    Candidates for one part, with everything the scoring needs about them, row by row. A blank pool has one more row
    at the end that stands for no item: no frame, or an avatar cut from the background
    """

    part: str
    ids: np.ndarray
    apps: np.ndarray
    animated: np.ndarray
    costs: np.ndarray
    names: list[frozenset[str]]
    picture: np.ndarray
    color: np.ndarray
    relevance: np.ndarray
    measured: np.ndarray
    blank: bool = False
    centered: np.ndarray = field(init=False)
    neutral: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.centered = _centered(self.part, self.picture) if len(self.ids) else self.picture
        self.neutral = _neutral(self.measured)
        if self.blank:
            self.centered[-1] = 0
            self.neutral[-1] = 0

    @classmethod
    def from_points(cls, part: str, points: list[Any]) -> Pool:
        payloads = [point.payload for point in points]
        return cls(
            part=part,
            ids=np.array([int(point.id) for point in points], dtype=np.int64),
            apps=np.array([int(payload["app"]["id"]) for payload in payloads], dtype=np.int64),
            animated=np.array([bool(payload["item"].get("animated")) for payload in payloads], dtype=bool),
            costs=np.array([int(payload["item"].get("point_cost") or 0) for payload in payloads], dtype=np.int64),
            names=[_name_words(payload["item"].get("title") or "") for payload in payloads],
            picture=_normalize(np.array([point.vector["image"] for point in points], dtype=np.float32).reshape(-1, 768)),
            color=_normalize(np.array([point.vector["color"] for point in points], dtype=np.float32).reshape(-1, 216)),
            relevance=np.zeros(len(points), dtype=np.float32),
            measured=np.array([payload["item"].get("neutral", np.nan) for payload in payloads], dtype=np.float32),
        )

    def take(self, rows: np.ndarray | list[int]) -> Pool:
        rows = np.asarray(rows, dtype=np.int64)
        return Pool(
            part=self.part,
            ids=self.ids[rows],
            apps=self.apps[rows],
            animated=self.animated[rows],
            costs=self.costs[rows],
            names=[self.names[row] for row in rows],
            picture=self.picture[rows],
            color=self.color[rows],
            relevance=self.relevance[rows].copy(),
            measured=self.measured[rows],
        )

    def with_blank(self, score: float) -> Pool:
        """The pool with the row for no item added, worth `score` and going with everything equally"""
        return Pool(
            part=self.part,
            ids=np.r_[self.ids, 0],
            apps=np.r_[self.apps, -1],
            animated=np.r_[self.animated, False],
            costs=np.r_[self.costs, 0],
            names=[*self.names, frozenset()],
            picture=np.vstack([self.picture, np.zeros((1, self.picture.shape[1]), dtype=np.float32)]),
            color=np.vstack([self.color, np.zeros((1, self.color.shape[1]), dtype=np.float32)]),
            relevance=np.r_[self.relevance, np.float32(score)],
            measured=np.r_[self.measured, np.float32(0)],
            blank=True,
        )

    def is_blank(self, row: int) -> bool:
        return self.blank and row == len(self.ids) - 1

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Themes:
    """The themes a profile can have: Steam's free ones first, then every game profile's"""

    names: list[str]
    ids: np.ndarray
    apps: np.ndarray
    costs: np.ndarray
    profiles: np.ndarray

    @classmethod
    def build(cls, game_profiles: list[Any]) -> Themes:
        names, ids, apps, costs, profiles = [], [], [], [], []
        for name, colors in FREE_THEMES.items():
            profile = _theme_profile(colors)
            if profile is not None:
                names.append(name)
                ids.append(0)
                apps.append(-1)
                costs.append(0)
                profiles.append(profile)
        for point in game_profiles:
            colors = ((point.payload.get("profile") or {}).get("theme") or {}).get("colors") or {}
            profile = _theme_profile(colors)
            if profile is None:
                continue
            names.append(GAME_PROFILE_THEME)
            ids.append(int(point.id))
            apps.append(int(point.payload["app"]["id"]))
            costs.append(int(point.payload["item"].get("point_cost") or 0))
            profiles.append(profile)
        return cls(
            names=names,
            ids=np.array(ids, dtype=np.int64),
            apps=np.array(apps, dtype=np.int64),
            costs=np.array(costs, dtype=np.int64),
            profiles=np.array(profiles, dtype=np.float32),
        )


FREE_ONLY = Themes.build([])


def _relevance(pool: Pool, request: Request) -> np.ndarray:
    """How well each candidate fits what was asked for, before looking at the other parts"""
    relevance = np.zeros(len(pool), dtype=np.float32)

    if request.prompt_vector is not None:
        mean, spread = PROMPT_SCALE[pool.part]
        prompt = pool.picture @ _normalize(np.array(request.prompt_vector, dtype=np.float32))
        relevance += PROMPT_WEIGHTS[pool.part] * (prompt - mean) / spread
    if request.colors:
        neutral = np.array([len(neutral_colors(request.colors)) / len(request.colors)], dtype=np.float32)
        hued = np.zeros((1, pool.color.shape[1]), dtype=np.float32)
        if request.color_vector is not None:
            hued = _normalize(np.array([request.color_vector], dtype=np.float32))
        wanted = _color_similarity(pool.color, pool.neutral, hued, neutral)[:, 0]
        relevance += COLOR_REQUEST_WEIGHTS[pool.part] * (wanted - COLOR_REQUEST_SCALE[0]) / COLOR_REQUEST_SCALE[1]
    if pool.part == "background":
        mean, spread = QUALITY_SCALE
        relevance += QUALITY_WEIGHT * ((pool.picture @ QUALITY_VECTOR) - mean) / spread

    words = _name_words(request.prompt_text or "")
    if words:
        named = np.array([len(words & name) / len(words) for name in pool.names], dtype=np.float32)
        relevance += NAME_MATCH_BONUS * PROMPT_WEIGHTS[pool.part] * named

    named = [*request.named_apps, *([request.app_id] if request.app_id is not None else [])]
    if named:
        relevance += NAMED_GAME_BONUS * np.isin(pool.apps, named)
    if request.reviews:
        reviews = request.reviews
        popularity = [max(0.0, math.log10(1 + reviews.get(int(app), 0) / POPULAR_REVIEWS)) for app in pool.apps]
        relevance += POPULARITY_WEIGHT * np.array(popularity, dtype=np.float32)

    return relevance


def _pair_scores(first: Pool, second: Pool, weighed_as: tuple[str, str] | None = None) -> np.ndarray:
    """
    How well every candidate of one part goes with every candidate of another. Blank rows go with everything.
    `weighed_as` counts the pair as another one, for a part that stands in for another
    """
    pair = (first.part, second.part)
    picture_mean, picture_spread = _scale(pair, PICTURE_SCALE)
    color_mean, color_spread = _scale(pair, COLOR_SCALE)
    weighed_as = weighed_as or pair
    weights_key = weighed_as if weighed_as in PICTURE_WEIGHTS else (weighed_as[1], weighed_as[0])

    scores = PICTURE_WEIGHTS[weights_key] * ((first.centered @ second.centered.T) - picture_mean) / picture_spread
    colors = _color_similarity(first.color, first.neutral, second.color, second.neutral)
    scores += COLOR_WEIGHTS[weights_key] * (colors - color_mean) / color_spread
    scores += SET_BONUS * (first.apps[:, None] == second.apps[None, :])
    if set(pair) == {"background", "mini_profile"}:
        scores += ANIMATION_MATCH * (first.animated[:, None] == second.animated[None, :])
    if first.blank:
        scores[-1, :] = 0
    if second.blank:
        scores[:, -1] = 0
    return scores.astype(np.float32)


def _not_for_sale() -> list[models.Condition]:
    """
    Items that can't be bought on their own: ones the Points Shop stopped selling, game profile parts, which come with
    their game profile, and event rewards. A generated profile is bought item by item, so they're never part of one
    """
    return [
        models.FieldCondition(key="item.available", match=models.MatchValue(value=False)),
        models.FieldCondition(key="item.sold_separately", match=models.MatchValue(value=False)),
    ]


def _filter(part: str, request: Request, extra: list[models.Condition] | None = None) -> models.Filter:
    """Only items of the part's category that are for sale, within the game and animation asked for and that the owner can buy"""
    must: list[models.Condition] = [models.FieldCondition(key="item.category", match=models.MatchValue(value=CATEGORIES[part]))]
    must_not = _not_for_sale()

    if part == "background" and request.app_id is not None:
        must.append(models.FieldCondition(key="app.id", match=models.MatchValue(value=request.app_id)))
    if part == "background" and request.animated is not None:
        must.append(models.FieldCondition(key="item.animated", match=models.MatchValue(value=request.animated)))
    if part == "background" and request.owned is not None:
        must_not.append(unowned(request.owned))
    if request.exclude:
        must_not.append(models.HasIdCondition(has_id=list(request.exclude)))
    if extra:
        must.extend(extra)

    return models.Filter(must=must, must_not=must_not)


def _search(part: str, query: Any, using: str | None, request: Request, limit: int,
            extra: list[models.Condition] | None = None) -> models.QueryRequest:
    if isinstance(query, np.ndarray):
        query = [float(value) for value in query]
    return models.QueryRequest(
        query=query,
        using=using,
        filter=_filter(part, request, extra),
        limit=limit,
        with_payload=CANDIDATE_FIELDS,
        with_vector=["image", "color"],
    )


def _shuffled(part: str, request: Request, limit: int, extra: list[models.Condition]) -> models.QueryRequest:
    """Items in no particular order, for a game's items when there's nothing to rank them by"""
    return _search(part, models.SampleQuery(sample=models.Sample.RANDOM), None, request, limit, extra)


def _scroll(client: QdrantClient, collection: str, category: str, fields: list[str], vectors: list[str] | bool) -> list[Any]:
    """Every item of a category that's for sale"""
    points: list[Any] = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="item.category", match=models.MatchValue(value=category))],
                must_not=_not_for_sale(),
            ),
            limit=CATALOGUE_PAGE,
            offset=offset,
            with_payload=fields,
            with_vectors=vectors,
        )
        points.extend(page)
        if offset is None:
            return points


class Catalogue:
    """
    The mini-profiles, avatars, frames and game profile themes, kept in memory. There are only a few thousand of
    each, and every profile searches all of them, so this saves a database query per search. Reloaded every few hours
    for new items
    """

    def __init__(self, client: QdrantClient, collection: str, refresh_seconds: int = CATALOGUE_REFRESH_SECONDS) -> None:
        self.client = client
        self.collection = collection
        self.refresh_seconds = refresh_seconds
        self.pools: dict[str, Pool] = {}
        self.rows: dict[str, dict[int, int]] = {}
        self.themes = FREE_ONLY
        self.loaded_at = -float(refresh_seconds)
        self.loading = threading.Lock()

    @property
    def ready(self) -> bool:
        return len(self.pools) == len(CACHED_PARTS)

    def load(self) -> None:
        if not self.loading.acquire(blocking=False):
            return
        try:
            started = time.monotonic()
            pools = {
                part: Pool.from_points(part, _scroll(self.client, self.collection, CATEGORIES[part], CANDIDATE_FIELDS,
                                                     ["image", "color"]))
                for part in CACHED_PARTS
            }
            themes = Themes.build(_scroll(self.client, self.collection, GAME_PROFILES, THEME_FIELDS, False))
            self.pools = pools
            self.rows = {part: {int(item_id): row for row, item_id in enumerate(pool.ids)} for part, pool in pools.items()}
            self.themes = themes
            self.loaded_at = time.monotonic()
            logger.info("Loaded %s and %d themes for generating in %.0fs", {part: len(pool) for part, pool in pools.items()},
                        len(themes.names), time.monotonic() - started)
        except Exception:
            logger.exception("Could not load the parts for generating")
        finally:
            self.loading.release()

    def load_in_background(self) -> None:
        threading.Thread(target=self.load, daemon=True).start()

    def refresh_if_stale(self) -> None:
        if time.monotonic() - self.loaded_at >= self.refresh_seconds:
            self.load_in_background()


class Generator:
    """Generates profiles against one collection"""

    def __init__(self, client: QdrantClient, collection: str,
                 reviews: Callable[[], dict[int, int]] | None = None,
                 titles: Callable[[], bool] | None = None,
                 frames: Callable[[], bool] | None = None) -> None:
        self.client = client
        self.collection = collection
        self.reviews = reviews
        self.titles = titles
        self.frames = frames
        self.catalogue = Catalogue(client, collection)

    def generate(self, request: Request) -> list[dict[str, Any]]:
        profile_start = request.game_profile is not None and "background" not in request.parts
        if (request.prompt_vector is None and request.color_vector is None and request.tone_vector is None
                and not request.keep and not profile_start and request.app_id is None):
            raise GenerationError("Give a prompt, colors, an item to keep or a game to start from", 400)

        self.catalogue.refresh_if_stale()
        parts = [part for part in PARTS if part in request.parts]
        if any(part in CACHED_PARTS for part in parts) and not self.catalogue.ready:
            raise GenerationError("Still getting ready, try again in a minute", 503)

        request = replace(request, reviews=self.reviews() if self.reviews else {})
        kept = self._kept(request)
        profile_background = request.game_profile is not None and "background" not in parts and "background" not in kept
        if profile_background:
            kept["background"] = self._game_profile_background(request.game_profile)
            parts = ["background", *parts]
        blanks = {
            part: score for part, score in BLANK_SCORES.items()
            if part in parts and part not in kept and (part != "avatar" or "background" in parts)
        }
        pools: dict[str, Pool] = {}

        if "background" in parts:
            pools["background"] = kept.get("background") or self._backgrounds(request, kept)
            if not len(pools["background"]):
                raise GenerationError("No background fits that")
            backgrounds = pools["background"]
            fits = self._theme_scores(backgrounds, np.arange(len(backgrounds)), request)
            fixed = self._fixed_theme(request)
            backgrounds.relevance = _relevance(backgrounds, request) + THEME_FIT * (
                fits[fixed] if fixed is not None else fits.max(axis=0))

        anchors = self._anchors(pools.get("background"), kept)
        for part in parts:
            if part == "background":
                continue
            pools[part] = kept.get(part) or self._from_catalogue(part, request, kept, pools.get("background"), anchors)
            if not len(pools[part]) and part not in blanks:
                raise GenerationError(f"Nothing fits for the {part.replace('_', ' ')}")
            pools[part].relevance = _relevance(pools[part], request)

        pools = self._narrow(pools, parts, anchors)
        for part, score in blanks.items():
            pools[part] = pools[part].with_blank(score)
        options = self._options(pools, parts, request)
        if profile_background:
            for option in options:
                option["background"] = None
        return options

    def _game_profile_background(self, item_id: int) -> Pool:
        """
        The background of a game profile that stays, as a background that can't change. A game profile's picture is
        its background. It costs nothing here, since the game profile's points count with its theme
        """
        points = self.client.retrieve(self.collection, ids=[item_id], with_payload=CANDIDATE_FIELDS,
                                      with_vectors=["image", "color"])
        if not points or points[0].payload["item"]["category"] != GAME_PROFILES:
            raise GenerationError(f"Item {item_id} isn't a game profile", 400)
        pool = Pool.from_points("background", points)
        pool.costs[:] = 0
        return pool

    def _kept(self, request: Request) -> dict[str, Pool]:
        """The items that stay put, checked to be the right kind of item for their part"""
        if not request.keep:
            return {}

        kept: dict[str, Pool] = {}
        missing: dict[str, int] = {}
        for part, item_id in request.keep.items():
            row = self.catalogue.rows.get(part, {}).get(item_id)
            if row is not None:
                kept[part] = self.catalogue.pools[part].take([row])
            else:
                missing[part] = item_id

        if missing:
            points = self.client.retrieve(
                self.collection, ids=list(missing.values()), with_payload=CANDIDATE_FIELDS, with_vectors=["image", "color"])
            found = {int(point.id): point for point in points}
            for part, item_id in missing.items():
                point = found.get(item_id)
                if point is None or point.payload["item"]["category"] != CATEGORIES[part]:
                    raise GenerationError(f"Item {item_id} isn't {PART_NAMES[part]}", 400)
                kept[part] = Pool.from_points(part, [point])

        return kept

    def _backgrounds(self, request: Request, kept: dict[str, Pool]) -> Pool:
        """
        Backgrounds that fit the prompt and colours, are named like the prompt, look like the kept items or come from
        the same set. Collections that keep the frames of animated items are searched by every frame, like search does
        """
        frames = bool(self.frames and self.frames())

        def by_text(vector: list[float], limit: int, extra: list[models.Condition] | None = None) -> models.QueryRequest:
            if frames:
                return _search("background", [vector], "frames", request, limit, extra)
            return _search("background", vector, "image", request, limit, extra)

        searches = []
        if request.prompt_vector is not None:
            searches.append(by_text(request.prompt_vector, BACKGROUND_LIMIT))
            words = _name_words(request.prompt_text or "")
            if words and self.titles and self.titles():
                searches.append(by_text(request.prompt_vector, NAMED_BACKGROUND_LIMIT, [
                    models.FieldCondition(key="item.title", match=models.MatchTextAny(text_any=" ".join(sorted(words))))]))
        if request.color_vector is not None:
            searches.append(_search("background", request.color_vector, "color", request, BACKGROUND_LIMIT // 2))
        if request.tone_vector is not None:
            searches.append(by_text(request.tone_vector, BACKGROUND_LIMIT // 2 if request.color_vector else BACKGROUND_LIMIT))
        for pool in kept.values():
            searches.append(_search("background", pool.centered[0], "image", request, BACKGROUND_LIMIT // 2))
        if kept:
            apps = sorted({int(pool.apps[0]) for pool in kept.values()})
            searches.append(_shuffled("background", request, SET_MATES_LIMIT, [
                models.FieldCondition(key="app.id", match=models.MatchAny(any=apps))]))
        if (request.app_id is not None and request.prompt_vector is None and request.color_vector is None
                and request.tone_vector is None):
            searches.append(_shuffled("background", request, BACKGROUND_LIMIT, []))

        if not searches:
            return Pool.from_points("background", [])

        points: dict[int, Any] = {}
        for response in self.client.query_batch_points(self.collection, requests=searches):
            for point in response.points:
                points.setdefault(int(point.id), point)
        return Pool.from_points("background", list(points.values()))

    @staticmethod
    def _anchors(backgrounds: Pool | None, kept: dict[str, Pool]) -> np.ndarray:
        """The backgrounds the other parts are matched against first: the kept one, or the ones that fit best"""
        if backgrounds is None:
            return np.array([], dtype=np.int64)
        if "background" in kept:
            return np.arange(len(backgrounds))
        return np.argsort(-backgrounds.relevance)[:ANCHOR_BACKGROUNDS]

    def _from_catalogue(self, part: str, request: Request, kept: dict[str, Pool], backgrounds: Pool | None,
                        anchors: np.ndarray) -> Pool:
        """
        Candidates for a part from memory: the ones that fit the prompt and colours best, are named like the prompt,
        look most like the kept items and the best backgrounds or share their colours, and everything from the same
        sets as those
        """
        catalogue = self.catalogue.pools[part]
        allowed = ~np.isin(catalogue.ids, list(request.exclude))
        chosen: set[int] = set()

        def best(scores: np.ndarray, limit: int) -> None:
            scores = np.where(allowed, scores, -np.inf)
            count = min(limit, int(np.isfinite(scores).sum()))
            if count:
                chosen.update(np.argpartition(-scores, count - 1)[:count].tolist())

        if request.prompt_vector is not None:
            prompt = catalogue.picture @ _normalize(np.array(request.prompt_vector, dtype=np.float32))
            best(prompt, PART_LIMIT)
            words = _name_words(request.prompt_text or "")
            if words:
                named = np.array([bool(words & name) for name in catalogue.names], dtype=bool)
                best(np.where(named, prompt, -np.inf), PART_LIMIT)
        if request.color_vector is not None:
            best(catalogue.color @ _normalize(np.array(request.color_vector, dtype=np.float32)), PART_LIMIT // 2)
        if request.tone_vector is not None:
            best(catalogue.picture @ _normalize(np.array(request.tone_vector, dtype=np.float32)), PART_LIMIT // 2)
        for pool in kept.values():
            best(catalogue.centered @ pool.centered[0], PART_LIMIT // 2)
            best(_color_similarity(catalogue.color, catalogue.neutral, pool.color[:1], pool.neutral[:1])[:, 0], PART_LIMIT // 2)
        if backgrounds is not None:
            for anchor in anchors[:3]:
                best(catalogue.centered @ backgrounds.centered[anchor], ANCHOR_MATCHES)
                best(_color_similarity(catalogue.color, catalogue.neutral, backgrounds.color[[anchor]],
                                       backgrounds.neutral[[anchor]])[:, 0], ANCHOR_MATCHES)

        set_apps = {*request.named_apps, *([request.app_id] if request.app_id is not None else [])}
        set_apps |= {int(pool.apps[0]) for pool in kept.values()}
        if backgrounds is not None:
            set_apps |= {int(app) for app in backgrounds.apps[anchors]}
        if set_apps:
            mates = np.flatnonzero(np.isin(catalogue.apps, list(set_apps)) & allowed)
            chosen.update(mates[:SET_MATES_LIMIT].tolist())

        return catalogue.take(sorted(chosen))

    @staticmethod
    def _narrow(pools: dict[str, Pool], parts: list[str], anchors: np.ndarray) -> dict[str, Pool]:
        """
        Keeps the most promising candidates of each part, judged by how well they fit the request and the best
        backgrounds, so every combination of what's left can be scored
        """
        narrowed = {}
        for part in parts:
            pool = pools[part]
            score = pool.relevance.copy()
            if part != "background" and "background" in pools and len(anchors) and len(pool):
                score += _pair_scores(pools["background"], pool)[anchors].max(axis=0)
            narrowed[part] = pool.take(np.argsort(-score)[:POOL_SIZE])
        return narrowed

    def _options(self, pools: dict[str, Pool], parts: list[str], request: Request) -> list[dict[str, Any]]:
        """
        Scores every combination, picks the best ones that don't look alike, and fills in the items and themes. A frame
        around an avatar cut from the background is judged against the background, as if it were the avatar. The best
        combinations of every background are kept, not just the best overall, since a few backgrounds that score well
        would otherwise fill the whole list and leave too few different ones to choose from
        """
        shape = [len(pools[part]) for part in parts]
        axes = {part: index for index, part in enumerate(parts)}

        def spread(values: np.ndarray, *along: str) -> np.ndarray:
            view = [1] * len(parts)
            for part, size in zip(along, values.shape, strict=True):
                view[axes[part]] = size
            return values.reshape(view)

        total = np.zeros(shape, dtype=np.float32)
        for part in parts:
            total = total + spread(pools[part].relevance, part)
        for a, b in PAIRS:
            if a in axes and b in axes:
                total = total + spread(_pair_scores(pools[a], pools[b]), a, b)
        if {"background", "avatar", "frame"} <= axes.keys() and pools["avatar"].blank:
            cut = np.zeros(len(pools["avatar"]), dtype=np.float32)
            cut[-1] = 1
            stand_in = _pair_scores(pools["background"], pools["frame"], weighed_as=("avatar", "frame"))
            total = total + spread(stand_in, "background", "frame") * spread(cut, "avatar")

        if request.budget is not None:
            cost = np.zeros(shape, dtype=np.int64)
            for part in parts:
                cost = cost + spread(pools[part].costs, part)
            total = np.where(cost <= request.budget, total, -np.inf)

        flat = total.ravel()
        if not np.isfinite(flat).any():
            raise GenerationError("Nothing fits within that budget")
        if "background" in axes:
            by_background = total.reshape(shape[0], -1)
            count = min(max(TOP_PER_BACKGROUND, TOP_COMBINATIONS // shape[0]), by_background.shape[1])
            best = np.argpartition(-by_background, count - 1, axis=1)[:, :count]
            best = (np.arange(shape[0])[:, None] * by_background.shape[1] + best).ravel()
        else:
            count = min(TOP_COMBINATIONS, len(flat))
            best = np.argpartition(-flat, count - 1)[:count]
        best = best[np.isfinite(flat[best])]
        best = best[np.argsort(-flat[best])]
        combinations = np.array(np.unravel_index(best, total.shape)).T

        chosen = self._diverse(combinations, pools, parts, request.count, uses=1, blanks=BLANK_SHARE)
        if len(chosen) < request.count:
            chosen += self._diverse(combinations, pools, parts, request.count, uses=MAX_PART_USES, taken=chosen)

        themes: list[int] = []
        for row in chosen:
            themes.append(self._theme(row, pools, axes, request, taken=themes))
        ids = {int(pools[part].ids[row[axes[part]]]) for row in chosen for part in parts}
        ids |= {int(self.catalogue.themes.ids[theme]) for theme in themes}
        ids.discard(0)
        items = {int(point.id): point.payload for point in self.client.retrieve(self.collection, ids=list(ids), with_payload=True)}

        options = []
        for row, theme in zip(chosen, themes, strict=True):
            filled = [part for part in parts if not pools[part].is_blank(int(row[axes[part]]))]
            picked = {part: items.get(int(pools[part].ids[row[axes[part]]])) for part in filled}
            game_profile = items.get(int(self.catalogue.themes.ids[theme])) if self.catalogue.themes.ids[theme] else None
            if any(payload is None for payload in picked.values()) or (self.catalogue.themes.ids[theme] and not game_profile):
                continue
            apps = {int(pools[part].apps[row[axes[part]]]) for part in filled}
            options.append({
                **dict.fromkeys(PARTS),
                **picked,
                "avatar_from_background": "avatar" in axes and pools["avatar"].is_blank(int(row[axes["avatar"]])),
                "theme": self.catalogue.themes.names[theme],
                "game_profile": game_profile,
                "set": next(iter(apps)) if len(apps) == 1 and len(filled) > 1 else None,
                "points": int(sum(int(pools[part].costs[row[axes[part]]]) for part in parts)
                              + int(self.catalogue.themes.costs[theme])),
                "score": round(float(total[tuple(row)]), 3),
            })
        return options

    def _fixed_theme(self, request: Request) -> int | None:
        """The theme every option gets, when the request keeps one: a game profile's or one of Steam's free ones"""
        themes = self.catalogue.themes
        if request.game_profile is not None:
            own = np.flatnonzero(themes.ids == request.game_profile)
            return int(own[0]) if len(own) else None
        if request.theme is not None:
            free = [index for index, name in enumerate(themes.names) if name == request.theme and themes.ids[index] == 0]
            return free[0] if free else None
        return None

    def _theme_scores(self, backgrounds: Pool | None, rows: np.ndarray, request: Request) -> np.ndarray:
        """
        How well every theme goes with each of the background `rows`, as themes by backgrounds: their colours matched,
        the part of a background without colour as grey, averaged with any colours asked for. A game profile's theme
        costs points, so it has to fit clearly better than the free ones, unless it's from the same game as the
        background or a game named in the request
        """
        themes = self.catalogue.themes
        matches = []
        apps = backgrounds.apps[rows] if backgrounds is not None else np.full(len(rows), -1)
        if backgrounds is not None:
            matches.append(themes.profiles @ _stored_profile(backgrounds.color[rows], backgrounds.neutral[rows]).T)
        if request.colors:
            matches.append((themes.profiles @ _colors_profile(request.colors))[:, None])
        match = sum(matches) / len(matches)
        mean, spread = THEME_SCALE
        score = THEME_MATCH * (match - mean) / spread

        game = (themes.ids != 0)[:, None]
        score -= GAME_THEME_MARGIN * game
        score += THEME_SET_BONUS * (game & (themes.apps[:, None] == apps[None, :]))
        named = [*request.named_apps, *([request.app_id] if request.app_id is not None else [])]
        if named:
            score += NAMED_GAME_BONUS * np.isin(themes.apps, named)[:, None]
        return score

    def _theme(self, row: np.ndarray, pools: dict[str, Pool], axes: dict[str, int], request: Request,
               taken: list[int]) -> int:
        """
        The theme for a profile, the one that goes best with its background, and a little with its other parts' games.
        Each game profile's theme goes in one option at most, and game profiles in half the options at most, counting
        the `taken` ones
        """
        themes = self.catalogue.themes
        fixed = self._fixed_theme(request)
        if fixed is not None:
            return fixed
        if "background" in axes:
            background = row[axes["background"]]
            score = self._theme_scores(pools["background"], np.array([background]), request)[:, 0]
        elif request.colors:
            score = self._theme_scores(None, np.array([0]), request)[:, 0]
        else:
            return 0

        game = themes.ids != 0
        for part, axis in axes.items():
            if part != "background":
                score += THEME_PART_BONUS * (game & (themes.apps == pools[part].apps[row[axis]]))
        used = [theme for theme in taken if themes.ids[theme]]
        score[used] = -np.inf
        if len(used) >= GAME_THEME_SHARE * request.count or request.game_profile is not None:
            score[game] = -np.inf
        if request.budget is not None:
            spent = sum(int(pools[part].costs[row[axis]]) for part, axis in axes.items())
            score[themes.costs > request.budget - spent] = -np.inf
        return int(np.argmax(score))

    @staticmethod
    def _diverse(combinations: np.ndarray, pools: dict[str, Pool], parts: list[str], count: int,
                 uses: int, taken: list[np.ndarray] | None = None, blanks: float | None = None) -> list[np.ndarray]:
        """
        Walks the combinations from best to worst and keeps the ones that look different from those already kept:
        always a new background, one that doesn't look like the kept ones, and no other part in more than `uses`
        options, unless there are too few of it to go round. With `blanks`, a part is left out in at most that share
        of the options, so a background that stays doesn't get the same empty avatar and frame in every option.
        Without it, leaving out a part can happen in every option
        """
        chosen: list[np.ndarray] = list(taken or [])
        start = len(chosen)
        background_axis = parts.index("background") if "background" in parts else None
        used = {part: Counter(int(row[index]) for row in chosen) for index, part in enumerate(parts)}
        most_blanks = math.ceil(count * blanks) if blanks is not None else count

        for row in combinations:
            if len(chosen) >= count:
                break
            if background_axis is not None:
                backgrounds = pools["background"]
                background = row[background_axis]
                if used["background"][int(background)] and len(backgrounds) > 1:
                    continue
                if chosen and len(backgrounds) > count:
                    kept = backgrounds.centered[[done[background_axis] for done in chosen]]
                    if (kept @ backgrounds.centered[background]).max() > SIMILAR_BACKGROUND:
                        continue
            if any(
                used[part][int(row[index])] >= uses and len(pools[part]) > count and not pools[part].is_blank(int(row[index]))
                for index, part in enumerate(parts) if part != "background"
            ):
                continue
            if any(
                pools[part].is_blank(int(row[index])) and used[part][int(row[index])] >= most_blanks and len(pools[part]) > 1
                for index, part in enumerate(parts)
            ):
                continue
            chosen.append(row)
            for index, part in enumerate(parts):
                used[part][int(row[index])] += 1

        return chosen[start:]
