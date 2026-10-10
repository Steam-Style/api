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
    point_cost: int = Field(description="Price in Steam points", examples=[10000])
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


class UnderstoodApp(BaseModel):
    """
    A game named in the query.
    """

    id: int = Field(description="Steam app ID", examples=[1145360])
    name: str = Field(description="App name", examples=["Hades"])


class Understood(BaseModel):
    """
    What the words of a query said about the kind of item. Items that match it rank higher, or only they are returned
    when the query had nothing else to search for.
    """

    query: str | None = Field(
        description="What the items should look like, with game names written out, null if nothing",
        examples=["hades cat"],
    )
    animated: bool | None = Field(description="`animated` or `static` in the query", examples=[True])
    tiled: bool | None = Field(description="`tiled` in the query", examples=[None])
    transparent: bool | None = Field(description="`transparent` in the query", examples=[None])
    categories: list[str] = Field(description="Categories named in the query", examples=[["profile backgrounds"]])
    apps: list[UnderstoodApp] = Field(description="Games named in the query")


class SearchResults(BaseModel):
    """
    A page of search results.
    """

    results: list[Item]
    understood: Understood | None = Field(
        default=None,
        description="What the words of the query said about the kind of item, null when the query was searched as it is",
    )
    owned_games: int | None = Field(
        default=None,
        description="How many games `owner` owns, null without `owner` or when the account's game details are private",
    )


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


class GenerateRequest(BaseModel):
    """
    What to build profiles around. Give at least one of `prompt`, `colors`, `keep`, `game_profile` or `app_id`.
    """

    prompt: str | None = Field(
        default=None, max_length=200, description="What the profile should look like", examples=["neon city at night"])
    colors: list[str] | None = Field(
        default=None,
        max_length=5,
        description="Hex colors the profile should have. When left out, colors named in the prompt, like black, are used",
        examples=[["#1a9fff"]],
    )
    keep: dict[str, int] | None = Field(
        default=None,
        description="Items to keep, by part: `background`, `mini_profile`, `avatar` or `frame`. The rest is found around them",
        examples=[{"avatar": 430398}],
    )
    game_profile: int | None = Field(
        default=None,
        description=(
            "A game profile to keep. Its theme is used in every option, and when `background` isn't one of the parts, "
            "the rest is matched to the game profile's background"
        ),
        examples=[None],
    )
    parts: list[str] | None = Field(
        default=None,
        description="The parts to fill, all four when left out",
        examples=[["background", "avatar", "frame"]],
    )
    app_id: int | None = Field(
        default=None, description="Game the background comes from. The other parts lean towards it", examples=[1145360])
    animated: bool | None = Field(
        default=None, description="True for an animated background, false for a still one", examples=[None])
    budget: int | None = Field(default=None, ge=0, description="Most points the profile may cost", examples=[5000])
    exclude: list[int] = Field(
        default_factory=list,
        max_length=300,
        description="Items not to use, like the ones already shown, for a new round of options",
        examples=[[]],
    )
    owner: int | None = Field(
        default=None,
        ge=76561197960265729,
        le=76561202255233023,
        description=(
            "SteamID64 of an account, for only items it can buy. Steam sells a game's static backgrounds only to "
            "people who own the game. The account's game details have to be public, when they're private every such "
            "background is left out"
        ),
        examples=[None],
    )
    count: int = Field(default=6, ge=1, le=12, description="How many options to return", examples=[6])


class GeneratedProfile(BaseModel):
    """
    One generated profile.
    """

    background: Item | None = Field(default=None, description="Profile background")
    mini_profile: Item | None = Field(default=None, description="Mini-profile background")
    avatar: Item | None = Field(
        default=None, description="Animated avatar, null when none fits well enough or it wasn't asked for")
    frame: Item | None = Field(
        default=None, description="Avatar frame, null when none fits well enough or it wasn't asked for")
    avatar_from_background: bool = Field(
        default=False,
        description="True when the avatar is best cut from the background behind it, to upload as a custom avatar",
        examples=[False],
    )
    theme: str = Field(
        description=(
            "Theme that suits the background: one of Steam's free themes (default, summer, midnight, steel, cosmic or "
            "darkMode), or gameProfile for the colors of `game_profile`"
        ),
        examples=["midnight"],
    )
    game_profile: Item | None = Field(
        default=None, description="Game profile whose theme the profile uses, when `theme` is gameProfile")
    set: int | None = Field(
        description="App ID when every part comes from the same game or event, null for a mix", examples=[3331000])
    points: int = Field(description="What the parts and theme cost together in Steam points", examples=[9000])
    score: float = Field(description="How well the parts fit the request and each other, higher is better", examples=[19.4])


class GenerateResults(BaseModel):
    """
    Generated profiles, best first, each with a different background.
    """

    options: list[GeneratedProfile]
    understood: Understood | None = Field(
        default=None,
        description="What the words of the prompt said about the kind of profile, null when it was used as it is",
    )
    owned_games: int | None = Field(
        default=None,
        description="How many games `owner` owns, null without `owner` or when the account's game details are private",
    )


class Error(BaseModel):
    """
    An error response.
    """

    status_code: int = Field(examples=[404])
    detail: str = Field(examples=["Item not found"])
