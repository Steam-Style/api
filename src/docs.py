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

GITHUB_ICON = (
    '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" clip-rule="evenodd" '
    'd="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013'
    "-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 "
    "1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 "
    "0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 "
    "1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 "
    "0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 "
    '10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/></svg>'
)

HEADER = f"""
<header class="site-header">
  <div class="site-header__inner">
    {LOGO}
    <span class="site-badge">API</span>
  </div>
</header>
"""

FOOTER = f"""
<footer class="site-footer">
  <div class="site-footer__inner">
    <div class="site-footer__top">
      <div class="site-footer__about">
        {LOGO}
        <p>A better way to find profile items from the Steam Points Shop.</p>
      </div>

      <div class="site-footer__links">
        <a href="{SITE_URL}">Explore</a>
        <a href="{SITE_URL}/about">About</a>
        <a href="{SITE_URL}/privacy">Privacy</a>
        <a href="/docs">API docs</a>
        <a href="https://github.com/sponsors/Steam-Style" target="_blank">Sponsor</a>
        <a href="mailto:info@steam.style" target="_blank">Contact</a>
        <a href="https://github.com/Steam-Style" target="_blank" aria-label="Steam Style on GitHub">{GITHUB_ICON}</a>
      </div>
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
