# PTT 資料分析與自適應預測系統

這個專案是一個 PTT（批踢踢實業坊）資料分析與自適應預測系統，其核心 `adaptive_system.py` 腳本負責持續進行資料爬取、實時預測熱門文章，並透過機器學習模型進行自適應學習與優化。

## 資料說明

專案中的 CSV 檔案（例如 `data/20251212/ptt_snapshot_v2_20251212_2321.csv`）包含了 PTT 文章的快照資料，其欄位含義如下：

- `Post_ID`: 文章的唯一識別碼。
- `source_board`: 文章來源的看板名稱 (例如：Gossiping)。
- `title`: 文章標題。
- `url`: 文章的完整 URL。
- `author`: 文章作者。
- `crawl_time`: 資料被爬取的時間。
- `post_time`: 文章發布的時間。
- `nrec_tag`: 文章的推文/噓文標籤，表示人氣概況（例如：爆, X1, X2 等）。
- `category`: 文章分類（如果有的話）。
- `title_char_count`: 文章標題的字元數。
- `post_hour`: 文章發布的小時數 (0-23)。
- `real_push_score`: 實際推文分數（推文數 - 噓文數）。
- `push_count`: 推文數量。
- `boo_count`: 噓文數量。
- `life_minutes`: 文章從發布到被抓取或特定時間點經過的分鐘數。
- `push_acceleration`: 推文加速率。
- `push_boo_ratio`: 推噓文比例。
- `author_avg_push`: 作者平均推文數。
- `content_word_count`: 文章內容的字數。
- `content_url_ratio`: 文章內容中 URL 佔比。
- `q_mark_density`: 文章內容中問號的密度。
- `e_mark_density`: 文章內容中驚嘆號的密度。
- `key_phrases`: 文章內容中的關鍵詞或關鍵短語。
- `hour_sin`: 小時數的正弦轉換（用於週期性特徵）。
- `hour_cos`: 小時數的餘弦轉換（用於週期性特徵）。
- `is_weekend`: 判斷是否為週末 (0=否, 1=是)。

## 環境建立與啟動

本專案建議使用 `uv` 進行環境管理和依賴安裝。

### 1. 安裝 `uv`

如果您尚未安裝 `uv`，可以使用 `pip` 進行安裝：

```bash
pip install uv
```

### 2. 建立虛擬環境

進入專案根目錄，建立一個虛擬環境：

```bash
uv venv
```

### 3. 啟動虛擬環境

```bash
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 4. 安裝依賴套件

使用 `uv` 安裝 `pyproject.toml` 中定義的所有依賴：

```bash
uv pip sync
```

### 5. 專案啟動

本專案的主要執行入口是 `adaptive_system.py`，它負責啟動完整的自適應預測系統，包含資料爬取、實時預測與模型自適應學習。

```bash
python adaptive_system.py
```

或透過 `uv` 直接執行：

```bash
uv run python adaptive_system.py
```

專案中還有其他輔助腳本（例如 `train_model_lifecycle.py`, `ptt_post_time_analysis.py` 等），它們作為模組被 `adaptive_system.py` 調用，或者可以單獨執行進行特定的開發或分析工作。

12/15 新增dashboard.py 展示數據:

需先安裝pip install streamlit-autorefresh

執行: uv run streamlit run dashboard.py ，會產生網頁視窗
