import pandas as pd
import numpy as np
import re
import jieba.analyse
from collections import Counter

# --- 共用參數 ---
NUMERIC_COLS = [
    'real_push_score', 'push_count', 'boo_count', 'life_minutes', 
    'push_acceleration', 'author_avg_push', 'content_word_count',
    'q_mark_density', 'e_mark_density', 'content_url_ratio', 
    'hour_sin', 'hour_cos', 'is_weekend',
    'is_morning', 'is_noon', 'is_afternoon', 'is_evening', 'is_night',
    'title_hot_score', 
    'push_velocity' # 新增: 10分鐘瞬時速度
]
CAT_COLS = ['category', 'source_board']

STOPWORDS = set([
    'http', 'https', 'com', 'tw', 'imgur', 'jpg', 'jpeg', 'png', 'gif', 
    'youtu', 'be', 'link', 'url', '新聞', '圖片', '連結', '記者', '報導', 
    '問題', '大家', '一個', '什麼', '這樣', '出來', '沒有', '可以', '怎麼', 
    '問卦', '爆卦', 'Re' 
])

def extract_tokens(text):
    if not isinstance(text, str): return []
    words = jieba.cut(text)
    return [w for w in words if len(w) > 1 and w.lower() not in STOPWORDS]

def calculate_hot_score_group(group_df):
    word_heat_map = Counter()
    tokens_list = group_df['title'].apply(extract_tokens).tolist()
    push_counts = group_df['push_count'].tolist()
    
    for tokens, push in zip(tokens_list, push_counts):
        weight = max(1, push) 
        for w in tokens:
            word_heat_map[w] += weight
            
    scores = []
    for tokens in tokens_list:
        if not tokens:
            scores.append(0)
            continue
        article_hot_score = sum(word_heat_map[w] for w in tokens) / len(tokens)
        scores.append(np.log1p(article_hot_score))
    return scores

def prepare_features_for_model(df, df_prev=None):
    """
    將原始 DataFrame 轉換為模型可吃的 X (特徵矩陣)
    參數:
      df: 當前的快照資料
      df_prev: (選用) 10分鐘前的快照資料，用於計算瞬時速度
    """
    df_processed = df.copy()
    
    # --- 1. 時間特徵處理 ---
    if 'post_hour' in df_processed.columns:
        h = df_processed['post_hour']
    elif 'post_time' in df_processed.columns:
        h = pd.to_datetime(df_processed['post_time']).dt.hour
    else:
        h = 0 

    df_processed['is_morning']   = ((h >= 6) & (h < 12)).astype(int)
    df_processed['is_noon']      = ((h >= 12) & (h < 14)).astype(int)
    df_processed['is_afternoon'] = ((h >= 14) & (h < 18)).astype(int)
    df_processed['is_evening']   = ((h >= 18) | (h == 0)).astype(int)
    df_processed['is_night']     = ((h >= 1) & (h < 6)).astype(int)

    # --- 2. 關鍵字熱度計算 ---
    if 'crawl_time' in df_processed.columns:
        df_processed['title_hot_score'] = 0.0
        for _, group in df_processed.groupby('crawl_time'):
            if group.empty: continue
            scores = calculate_hot_score_group(group)
            df_processed.loc[group.index, 'title_hot_score'] = scores
    else:
        df_processed['title_hot_score'] = calculate_hot_score_group(df_processed)

    # 假設 df 已經包含了該時間點的所有文章
    # 先計算當下的排名
    df_processed['current_rank'] = df_processed['push_count'].rank(ascending=False)
    
    # 計算與第一名的推文差距
    max_push = df_processed['push_count'].max()
    df_processed['gap_to_leader'] = max_push - df_processed['push_count']
    
    # 計算與上一名的差距 (Gap to Next) - 競爭激烈度
    # 這需要先排序
    sorted_push = df_processed['push_count'].sort_values(ascending=False)

    # --- 3. 10分鐘瞬時速度計算 (修正版) ---
    # 優先檢查是否已經有 push_count_prev 欄位 (來自 train_model_lifecycle 的 merge)
    if 'push_count_prev' in df_processed.columns and 'crawl_time_prev' in df_processed.columns:
        # 【情境 A】訓練模式：資料已經 merge 好了
        t_now = pd.to_datetime(df_processed['crawl_time'])
        t_prev = pd.to_datetime(df_processed['crawl_time_prev'])
        time_diff = (t_now - t_prev).dt.total_seconds() / 60
        time_diff = time_diff.replace(0, 10).fillna(10)
        
        push_diff = df_processed['push_count'] - df_processed['push_count_prev']
        df_processed['push_velocity'] = push_diff / time_diff
        
    elif df_prev is not None:
        # 【情境 B】預測模式：傳入獨立的 df_prev
        prev_data = df_prev[['Post_ID', 'push_count', 'crawl_time']].copy()
        prev_data.columns = ['Post_ID', 'push_count_prev', 'crawl_time_prev']
        
        df_processed = pd.merge(df_processed, prev_data, on='Post_ID', how='left')
        
        t_now = pd.to_datetime(df_processed['crawl_time'])
        t_prev = pd.to_datetime(df_processed['crawl_time_prev'])
        time_diff = (t_now - t_prev).dt.total_seconds() / 60
        time_diff = time_diff.replace(0, 10).fillna(10)
        
        push_diff = df_processed['push_count'] - df_processed['push_count_prev']
        df_processed['push_velocity'] = push_diff / time_diff
        
    else:
        # 【情境 C】無資料
        pass

    # 填補缺失值 (統一處理)
    if 'push_velocity' not in df_processed.columns:
        df_processed['push_velocity'] = df_processed['push_acceleration'] # Fallback
    else:
        df_processed['push_velocity'] = df_processed['push_velocity'].fillna(df_processed['push_acceleration'])

    # --- 4. 填補缺失值 ---
    for col in NUMERIC_COLS:
        if col not in df_processed.columns: df_processed[col] = 0
        df_processed[col] = df_processed[col].fillna(0)
    
    # --- 5. 類別特徵 Hash ---
    for col in CAT_COLS:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str).apply(lambda x: hash(x) % 1000)
        else:
            df_processed[col] = 0
            
    return df_processed[NUMERIC_COLS + CAT_COLS]