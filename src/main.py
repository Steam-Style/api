"""
The Steam Style API: its routes, docs and CORS settings.
"""

import uvicorn
from litestar import Litestar, get
from litestar.config.cors import CORSConfig
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Contact, Tag

from clients import app_index, generator
from docs import SiteSwaggerRenderPlugin
from routes.generate import generate_profiles
from routes.items import get_item_by_id, get_item_by_name, get_items_batch, list_items
from routes.search import search_items


@get("/", include_in_schema=False)
async def index() -> dict:
    return {"status": "ok", "message": "Steam Style Query API"}


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
