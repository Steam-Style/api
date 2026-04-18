# Steam Style API

[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/license/gpl-3-0)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Steam-Style)](https://github.com/sponsors/Steam-Style)
[![Demo](https://img.shields.io/badge/Demo-green)](https://api.steam.style)

This repository contains the ingestion portion of the [Steam Style project](https://www.steam.style), which functions as the portion of the project responsible for handling querying of the vector database through the API. The contents of the database is collected and processed from the Steam web API through a [a separate repository](https://github.com/Steam-Style/ingestion)

## Running

### Prerequisites

- Git
- Docker + Docker Compose

### 1. Clone the repository

```bash
git clone https://github.com/Steam-Style/api.git
```

### 2. Navigate to the project directory

```bash
 cd api
```

### 3. Start the services using Docker Compose

```bash
docker compose up -d
```

---

<div align="center">

[![GitHub Stars](https://img.shields.io/github/stars/Steam-Style/api?style=social)](https://github.com/Steam-Style/api/stargazers)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/Steam-Style?style=social)](https://github.com/sponsors/Steam-Style)

Made with ❤️ by the Steam Style team

[Report an Issue](https://github.com/Steam-Style/api/issues) • [Visit Steam Style](https://steam.style)

</div>
