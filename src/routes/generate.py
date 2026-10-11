"""
The /generate route.
"""

import logging
import threading

from litestar import post
from litestar.exceptions import HTTPException
from litestar.openapi import ResponseSpec

from clients import app_index, color_embedder, generator, get_text_embedding, owned_apps
from generator import CATEGORIES as PART_CATEGORIES
from generator import PARTS, GenerationError, neutral_colors, tone_words
from generator import Request as GenerationRequest
from schemas import Error, GenerateRequest, GenerateResults
from search.parsing import colors_named, parse_query

GENERATION_SLOTS = 2
GENERATION_WAIT_SECONDS = 15

logger = logging.getLogger(__name__)
generation_slots = threading.BoundedSemaphore(GENERATION_SLOTS)


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
