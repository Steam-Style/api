# Steam Style API

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/license/gpl-3-0)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Steam-Style)](https://github.com/sponsors/Steam-Style)
[![Docs](https://img.shields.io/badge/Docs-api.steam.style-green)](https://api.steam.style/docs)

This is the API behind [Steam Style](https://steam.style). It searches every item in the Steam Points Shop by what it looks like, and puts whole profiles together from them. The items come from the [ingestion](https://github.com/Steam-Style/ingestion) service, which keeps them in a [Qdrant](https://qdrant.tech) vector database.

It's public at [api.steam.style](https://api.steam.style) and free to use without a key. The [docs](https://api.steam.style/docs) list every endpoint and parameter, and you can try them there.

## Endpoints

| Endpoint                         | What it does                                                     |
| -------------------------------- | ---------------------------------------------------------------- |
| `GET /search`                    | Search by text, colors or a similar item, or browse with filters |
| `POST /generate`                 | Put whole profiles together                                      |
| `GET /item/{item_id}`            | One item                                                         |
| `GET /item/{app_id}/{item_name}` | One item by its game and exact name                              |
| `GET /items/batch?id=1&id=2`     | Up to 50 items at once                                           |
| `GET /items`                     | Every item, page by page                                         |

Item IDs are the same as in the Points Shop, so an item's page on Steam is `https://store.steampowered.com/points/shop/reward/{item_id}`.

### Search

`/search?query=rainy city at night` matches what items look like, not only their names. The text and the stored pictures are embedded with the same [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) model. Searches can also match colors (`color=1a9fff`), look for items like another one (`similar_to=123`), or do several at once.

The query is read for words that describe the kind of item, so "animated cat avatar" looks for animated avatars of cats, and names of games are recognised, so items from that game rank higher. Items named like the query rank higher too. Turn this off with `parse=false`.

Without a query you're browsing, sorted by `sort`, which is the Points Shop's own popular order unless you pick another. Filters like `category`, `app_id`, `animated` and `tiled` work either way.

### Generate

`/generate` puts together a background, mini-profile, avatar and avatar frame that fit what you ask for and each other, plus a theme. Start from a prompt, colors, items to keep or a game, and you get several options with different backgrounds:

```json
{ "prompt": "neon city at night", "budget": 10000 }
```

Parts are matched by how alike their pictures and colors are, and complete sets from one game score higher. The theme is one of Steam's free ones or a game profile's, whichever suits the background's colors. When no frame fits well enough it's left out, and when no avatar does, the option says to cut one from the background behind it. `game_profile` keeps a game profile and its theme, `exclude` gets new options.

## Running

You need Docker with Docker Compose, and a Qdrant filled by the [ingestion](https://github.com/Steam-Style/ingestion) service.

```bash
git clone https://github.com/Steam-Style/api.git
cd api
echo DATABASE_URL=http://qdrant:6333 > .env
docker compose up -d
```

The API listens on port 8000 inside its network, so put it on the same network as Qdrant or behind a proxy. On start it downloads the embedding model, kept in a volume, and loads game names and the parts for generating in the background. Search works right away, generating after a minute or two.

To run it without Docker, set `DATABASE_URL=http://localhost:6333` in `.env` and run:

```bash
uv sync
uv run --env-file .env uvicorn main:app --app-dir src --port 8000
```

### Settings

Set these as environment variables.

| Variable          | Default                           | What it's for                                                             |
| ----------------- | --------------------------------- | ------------------------------------------------------------------------- |
| `DATABASE_URL`    | required                          | Where Qdrant is                                                           |
| `COLLECTION_NAME` | `Steam-Style-Items`               | The collection to read, the one ingestion fills                           |
| `DEVICE`          | `cpu`                             | Where the model runs, like `cuda`                                         |
| `MODEL_NAME`      | `google/siglip2-base-patch16-224` | The embedding model, must match ingestion's                               |
| `STEAM_API_KEY`   | none                              | A [Steam Web API key](https://steamcommunity.com/dev/apikey), for `owner` |

---

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/Steam-Style/api?style=social)](https://github.com/Steam-Style/api/stargazers)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Steam-Style?style=social)](https://github.com/sponsors/Steam-Style)

Made with ❤️ by the Steam Style team

[Report an Issue](https://github.com/Steam-Style/api/issues) • [Visit Steam Style](https://steam.style)

</div>
