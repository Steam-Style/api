from pydantic import BaseModel, Field

EXAMPLE_ASSETS = "https://shared.fastly.steamstatic.com/community_assets/images/items/534380"


class ImageSizes(BaseModel):
    """
    An image in the sizes Steam provides. Animated avatars, frames and stickers only move in the small version.
    """

    large: str | None = Field(
        description="URL of the large version", examples=[f"{EXAMPLE_ASSETS}/4a85f77cbdb6b67057858d93a61e157ffc374b8c.jpg"])
    small: str | None = Field(
        description="URL of the small version", examples=[f"{EXAMPLE_ASSETS}/deb70e9ed015c524581f96beb9a477146936ceda.png"])


class WebmVideos(BaseModel):
    """
    WebM versions of a video.
    """

    large: str | None = Field(
        description="URL of the full size video",
        examples=[f"{EXAMPLE_ASSETS}/5adfa32bf0f5f7414a2d2546904160e416ae8517.webm"],
    )
    small: str | None = Field(
        description="URL of the preview sized video, sometimes a zoomed in crop",
        examples=[f"{EXAMPLE_ASSETS}/bb76f65654b8b2db59b3611810014202ecd8bfb3.webm"],
    )


class Mp4Videos(BaseModel):
    """
    MP4 versions of a video.
    """

    large: str | None = Field(
        description="URL of the full size video",
        examples=[f"{EXAMPLE_ASSETS}/36d92e465af833290ff1def5b2de70c8c534a81d.mp4"],
    )
    small: str | None = Field(
        description="URL of the preview sized video, sometimes a zoomed in crop",
        examples=[f"{EXAMPLE_ASSETS}/3e61025052ae091bdfa6fb9776f23dfdac738dda.mp4"],
    )


class Videos(BaseModel):
    """
    Videos of animated items, in both formats Steam provides.
    """

    webm: WebmVideos
    mp4: Mp4Videos


class Assets(BaseModel):
    """
    Images and videos of an item.
    """

    images: ImageSizes
    videos: Videos


class ItemDetails(BaseModel):
    """
    What the item is and what it looks like.
    """

    id: int = Field(description="Item ID, the same as its ID in the Points Shop", examples=[198096])
    name: str | None = Field(description="Item name", examples=["Dying Light 2 Stay Human Profile"])
    title: str | None = Field(description="Title shown in the Points Shop", examples=["Dying Light 2 Stay Human Profile"])
    description: str | None = Field(description="Description shown in the Points Shop", examples=[None])
    internal_description: str | None = Field(
        description="Steam's own description of the item", examples=["Dying Light 2 Stay Human Profile"])
    category: str = Field(
        description="One of `avatars`, `avatar frames`, `profile backgrounds`, `mini-profile backgrounds`, "
        "`game profiles`, `emoticons`, `animated stickers`, `chat effects`, `steam deck keyboards` "
        "or `steam startup movies`",
        examples=["game profiles"],
    )
    point_cost: str = Field(description="Price in Steam points, as Steam returns it", examples=["10000"])
    animated: bool = Field(description="Whether the item moves", examples=[False])
    transparent: bool = Field(description="Whether the image has transparent parts", examples=[False])
    tiled: bool = Field(description="Whether the background repeats as a pattern", examples=[False])
    available: bool = Field(
        default=True,
        description="Whether the Points Shop still sells the item. Items it stopped selling stay in the catalogue",
        examples=[True],
    )
    sold_separately: bool = Field(
        default=True,
        description=(
            "Whether the item can be bought on its own. False for the parts of a game profile and for items the "
            "Points Shop only sells to some people, like the Steam Deck profile"
        ),
        examples=[True],
    )
    assets: Assets


class App(BaseModel):
    """
    The game or event the item comes from.
    """

    id: int = Field(description="Steam app ID", examples=[534380])
    name: str | None = Field(description="App name", examples=["Dying Light 2: Reloaded Edition"])
    icon: str | None = Field(
        description="URL of the app icon",
        examples=["https://shared.fastly.steamstatic.com/community_assets/images/apps/534380/4036cc941abaeeed5ee8bd433ca69388723f7f19.jpg"],
    )


class MarketLinks(BaseModel):
    """
    Community Market links. Most Points Shop items can't be sold, so these may lead to an empty page.
    """

    item: str | None = Field(
        description="Link to the item's market page",
        examples=["https://steamcommunity.com/market/listings/753/534380-Dying%20Light%202%20Stay%20Human%20Profile"],
    )
    app: str | None = Field(
        description="Link to every market item from the same app",
        examples=["https://steamcommunity.com/market/search?appid=753&category_753_Game%5B%5D=tag_app_534380"],
    )


class PointsShopLinks(BaseModel):
    """
    Points Shop links.
    """

    item: str | None = Field(
        description="Link to the item in the Points Shop",
        examples=["https://store.steampowered.com/points/shop/reward/198096"],
    )
    app: str | None = Field(
        description="Link to every Points Shop item from the same app",
        examples=["https://store.steampowered.com/points/shop/app/534380"],
    )


class Urls(BaseModel):
    """
    Links to the item on Steam.
    """

    market: MarketLinks
    points_shop: PointsShopLinks


class Timestamps(BaseModel):
    """
    When the item was added and changed, in UTC.
    """

    created_at: str | None = Field(description="When the item was added to the Points Shop", examples=["2022-12-16T12:20:07Z"])
    updated_at: str | None = Field(description="When Steam last changed the item", examples=["2026-04-02T20:43:19Z"])
    available_at: str | None = Field(description="When the item goes on sale, if it's limited", examples=[None])
    unavailable_at: str | None = Field(description="When the item stops being sold, if it's limited", examples=[None])
    usable_duration_seconds: int = Field(description="How long the item lasts after buying it, 0 for forever", examples=[0])
    removed_at: str | None = Field(
        default=None, description="When the item was found to be gone from the Points Shop", examples=[None])


class ProfilePart(Assets):
    """
    One of the items a game profile applies.
    """

    animated: bool = Field(description="Whether this part moves", examples=[True])


class ProfileTheme(BaseModel):
    """
    The colors a game profile gives a Steam profile.
    """

    name: str | None = Field(
        description="`GameProfile` for a theme made for the game, otherwise one of Steam's shared themes",
        examples=["GameProfile"],
    )
    colors: dict[str, str] | None = Field(
        description="The theme's CSS variables without the leading dashes, like `gradient-background`",
        examples=[{"gradient-background": "rgba(255, 255, 255, 0)", "btn-background": "rgba(0, 0, 0, .2)"}],
    )


class GameProfile(BaseModel):
    """
    What a game profile applies to a Steam profile. Parts the game profile doesn't include are null.
    """

    theme: ProfileTheme
    background: ProfilePart | None
    mini_profile: ProfilePart | None
    avatar: ProfilePart | None
    frame: ProfilePart | None


class Item(BaseModel):
    """
    An item from the Points Shop.
    """

    item: ItemDetails
    app: App
    urls: Urls
    timestamps: Timestamps
    profile: GameProfile | None = Field(
        default=None,
        description="Only on game profiles. Null while the profile hasn't been fetched from Steam yet",
    )


class SearchResults(BaseModel):
    """
    A page of search results.
    """

    results: list[Item]


class ItemBatch(BaseModel):
    """
    Several items, fetched by ID.
    """

    items: list[Item]


class ItemSummary(BaseModel):
    """
    An item ID with when it last changed.
    """

    id: int = Field(description="Item ID", examples=[198096])
    updated_at: str | None = Field(description="When Steam last changed the item", examples=["2026-04-02T20:43:19Z"])


class ItemList(BaseModel):
    """
    A page of item IDs.
    """

    items: list[ItemSummary]
    total: int = Field(description="How many items there are in total", examples=[157000])
    next_cursor: int | None = Field(
        description="Pass as `cursor` to get the next page, null on the last page", examples=[198097])


class Error(BaseModel):
    """
    An error response.
    """

    status_code: int = Field(examples=[404])
    detail: str = Field(examples=["Item not found"])
