<div align="center">
    <span style="background-color: #2ea44f; color: #fff; padding: 8px 16px; border-radius: 20px 0 0 20px; font-weight: bold; font-family: sans-serif;">English</span><a href="README.zh-TW.md" style="background-color: #ffffff; color: #24292e; padding: 8px 16px; border-radius: 0 20px 20px 0; text-decoration: none; font-family: sans-serif; border: 1px solid #d0d7de; border-left: none;">繁體中文</a>
</div>

# PTT Trending Post Predictor & Adaptive System

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Google%20Gemini-AI-orange?logo=google&logoColor=white" alt="Gemini">
    <img src="https://img.shields.io/badge/Obsidian-Plugin-purple?logo=obsidian&logoColor=white" alt="Obsidian">
</div>

This project is a PTT (Bulletin Board System) data analysis and adaptive prediction system. Its core engine, `src/main.py`, continuously crawls data, predicts trending posts in real-time, and uses machine learning models to adaptively learn and optimize its predictions.

## Features

- **Real-time Crawling**: Continuously monitors PTT boards (e.g., Gossiping) for new posts and updates.
- **Adaptive Prediction**: Uses LightGBM to predict future popularity (push count) of posts.
- **Dynamic Learning**: The system retrains itself based on new data to adapt to changing trends.
- **Dashboard**: Visualizes data and predictions using Streamlit.

## Data Structure

The CSV snapshots (e.g., `data/YYYYMMDD/...csv`) contain the following key fields:

<details>
<summary>Click to expand detailed data fields</summary>

- `Post_ID`: Unique identifier for the post.
- `source_board`: Board name of the post source (e.g., Gossiping).
- `title`: Post title.
- `url`: Full URL of the post.
- `author`: Post author.
- `crawl_time`: Time when the data was crawled.
- `post_time`: Time when the post was published.
- `nrec_tag`: Recommendation Tag: Push/Boo tag indicating popularity (e.g., 爆, X1, X2).
- `category`: Post category (if any).
- `title_char_count`: Number of characters in the title.
- `post_hour`: Hour of publication (0-23).
- `real_push_score`: Net score (Push count - Boo count).
- `push_count`: Number of pushes.
- `boo_count`: Number of boos.
- `life_minutes`: Minutes elapsed from publication to crawl time.
- `push_acceleration`: Rate of push accumulation.
- `push_boo_ratio`: Ratio of pushes to boos.
- `author_avg_push`: Average push count for the author.
- `content_word_count`: Word count of the content.
- `content_url_ratio`: Ratio of URLs in the content.
- `q_mark_density`: Density of question marks in the content.
- `e_mark_density`: Density of exclamation marks in the content.
- `key_phrases`: Keywords or phrases extracted from content.
- `hour_sin`: Sine transformation of the hour (for cyclical features).
- `hour_cos`: Cosine transformation of the hour (for cyclical features).
- `is_weekend`: Boolean indicating if it creates on a weekend (0=No, 1=Yes).

</details>

## Installation & Setup

<details>
<summary>Click to expand detailed installation steps</summary>

This project uses `uv` for fast and reliable dependency management.

### 1. Install `uv`

If you haven't installed `uv` yet:

```bash
pip install uv
```

### 2. Initialize Environment

Create a virtual environment:

```bash
uv venv
```

### 3. Install Dependencies

Sync dependencies from `pyproject.toml`:

```bash
uv sync
```

</details>

## Usage

### Run the Main System

The main entry point is `src/main.py`. This script starts the full cycle of crawling, prediction, and adaptive learning ("Agentic" behavior).

```bash
uv run python src/main.py
```

### Run the Dashboard

To visualize the trending data:

```bash
uv run streamlit run src/dashboard/dashboard.py
```

## Project Structure

```text
Social-Media-Trending-Post-Predictor/
├── src/
│   ├── analysis/          # Analysis scripts
│   ├── dashboard/         # Dashboard application
│   │   └── dashboard.py
│   ├── data/              # Data collection and storage
│   │   └── ptt_monitor.py
│   ├── features/          # Feature engineering
│   │   └── feature_utils.py
│   ├── model/             # Model training logic
│   └── main.py            # Main entry point
├── data/                  # Data storage (CSV snapshots)
├── ptt_data.db            # SQLite database
├── pyproject.toml         # Project configuration
└── README.md
```

- `src/main.py`: Entry point.
- `src/data/`: Crawlers and Database management.
- `src/features/`: Feature engineering utilities.
- `src/model/`: Model training and evaluation logic.
- `src/dashboard/`: Streamlit dashboard app.
