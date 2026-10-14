from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from litestar import Request
from litestar.openapi.plugins import SwaggerRenderPlugin

SITE_URL = "https://steam.style"
STYLESHEET = (Path(__file__).parent / "docs.css").read_text(encoding="utf-8")

LOGO = f"""
<a class="site-logo" href="{SITE_URL}" aria-label="Steam Style home">
  <svg viewBox="0 0 32 32" aria-hidden="true">
    <path d="M0 12H20V32H4C1.79086 32 0 30.2091 0 28V12Z" />
    <path d="M0 4C0 1.79086 1.79086 0 4 0H20V8H0V4Z" />
    <path d="M24 0H28C30.2091 0 32 1.79086 32 4V28C32 30.2091 30.2091 32 28 32H24V0Z" />
  </svg>
  <span>Steam<span class="dot">.</span><span class="rest">Style</span></span>
</a>
"""

HEADER = f"""
<header class="site-header">
  <div class="site-header__inner">
    {LOGO}
    <span class="site-badge">API</span>
  </div>
</header>
"""

ITEM_LINKS = [
    ("Backgrounds", "backgrounds"),
    ("Animated backgrounds", "animated-backgrounds"),
    ("Avatars", "avatars"),
    ("Avatar frames", "avatar-frames"),
    ("Mini-profiles", "mini-profiles"),
    ("Game profiles", "game-profiles"),
    ("Emoticons", "emoticons"),
    ("Stickers", "stickers"),
    ("Chat effects", "chat-effects"),
    ("Deck keyboards", "deck-keyboards"),
    ("Startup movies", "startup-movies"),
    ("Free items", "free"),
]
SITE_LINKS = [
    ("Explore", f"{SITE_URL}/"),
    ("Studio", f"{SITE_URL}/studio"),
    ("About", f"{SITE_URL}/about"),
    ("Privacy", f"{SITE_URL}/privacy"),
]
MORE_LINKS = [
    ("API docs", "/docs"),
    ("GitHub", "https://github.com/Steam-Style"),
    ("Sponsor", "https://github.com/sponsors/Steam-Style"),
    ("Contact", "mailto:info@steam.style"),
]


def link_column(title: str, links: list[tuple[str, str]], wide: bool = False) -> str:
    """
    One titled group of footer links, laid out like the site's footer.

    Args:
        title (str): The group's title.
        links (list[tuple[str, str]]): Each link's text and address.
        wide (bool): Lays the links out in two columns.

    Returns:
        str: The group's HTML.
    """
    items = []
    for label, href in links:
        target = "" if href.startswith(("/", SITE_URL)) else ' target="_blank"'
        items.append(f'<li><a href="{href}"{target}>{label}</a></li>')

    modifier = " site-footer__group--wide" if wide else ""
    return (
        f'<nav class="site-footer__group{modifier}" aria-label="{title}">'
        f'<p class="site-footer__title">{title}</p><ul class="site-footer__list">{"".join(items)}</ul></nav>'
    )


FOOTER = f"""
<footer class="site-footer">
  <div class="site-footer__inner">
    <div class="site-footer__top">
      <div class="site-footer__about">
        {LOGO}
        <p>A better way to find profile items from the Steam Points Shop.</p>
      </div>
      {link_column("Items", [(label, f"{SITE_URL}/{slug}") for label, slug in ITEM_LINKS], wide=True)}
      {link_column("Site", SITE_LINKS)}
      {link_column("More", MORE_LINKS)}
    </div>

    <div class="site-footer__bottom">
      <p>
        Steam Style is a fan project and isn't affiliated with Valve. Steam and the Steam logo are trademarks of
        Valve Corporation, and all item artwork belongs to its creators.
      </p>
      <p class="site-footer__copyright">&copy; {{year}} Steam Style</p>
    </div>
  </div>
</footer>
"""

SCROLL_TO_TOP = """
<button type="button" class="scroll-to-top" aria-label="Scroll to the top" title="Back to top">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true"><path d="m18 15-6-6-6 6" /></svg>
</button>
<script>
  const scrollToTop = document.querySelector(".scroll-to-top");
  const updateScrollToTop = () => scrollToTop.classList.toggle("visible", window.scrollY > 300);
  window.addEventListener("scroll", updateScrollToTop, { passive: true });
  scrollToTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  updateScrollToTop();
</script>
"""


class SiteSwaggerRenderPlugin(SwaggerRenderPlugin):
    """
    Swagger UI with steam.style's header, footer and colors, so the docs feel like part of the site.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            favicon=f"<link rel='icon' href='{SITE_URL}/favicon.ico'>",
            style=f"<style>{STYLESHEET}</style>",
            **kwargs,
        )

    def render(self, request: Request, openapi_schema: dict[str, Any]) -> bytes:
        """
        Renders the Swagger UI page with the site's header and footer around it.

        Args:
            request (Request): The request for the docs page.
            openapi_schema (dict[str, Any]): The OpenAPI schema to show.

        Returns:
            bytes: The HTML page.
        """
        page = super().render(request, openapi_schema)
        footer = FOOTER.replace("{year}", str(datetime.now(UTC).year))

        return page.replace(
            b"<div id='swagger-container'/>",
            HEADER.encode() + b"<div id='swagger-container'></div>",
            1,
        ).replace(b"</body>", footer.encode() + SCROLL_TO_TOP.encode() + b"</body>", 1)
