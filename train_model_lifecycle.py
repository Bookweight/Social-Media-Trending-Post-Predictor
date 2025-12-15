import pandas as pd
import numpy as np
import glob
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import math
import gc
import re
from datetime import datetime, timedelta
import feature_utils

import db_manager

# --- 設定與參數 ---
DATA_DIR = 'data'
MODEL_OUTPUT = 'ptt_lifecycle_model.txt'
LOOK_AHEAD_MINUTES = 120
TOLERANCE_MINUTES = 30
VELOCITY_DELTA_MINUTES = 10 

NUMERIC_COLS = feature_utils.NUMERIC_COLS
CAT_COLS = feature_utils.CAT_COLS

def parse_file_time(filepath):
    match = re.search(r'(\d{8}_\d{4})', filepath)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M')
    return None

def load_recent_data(days_back=7):
    print(f"📂 正在從資料庫讀取最近 {days_back} 天的資料...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    
    # 直接用 SQL 篩選，速度快非常多
    df = db_manager.query_snapshots_by_time_range(start_time, end_time)
    
    print(f"✅ 載入完成！總資料列數: {len(df)}")
    return df

def create_lifecycle_dataset(df):
    print(f"🔄 執行生命週期配對 (T+{LOOK_AHEAD_MINUTES}min)...")
    
    # 1. 建立「未來」標籤 (T+60)
    df['target_lookup_time'] = df['crawl_time'] + pd.Timedelta(minutes=LOOK_AHEAD_MINUTES)
    df_future = df[['Post_ID', 'crawl_time', 'push_count', 'boo_count']].copy()
    df_future = df_future.sort_values('crawl_time')
    
    merged = pd.merge_asof(
        df,
        df_future,
        left_on='target_lookup_time',
        right_on='crawl_time',
        by='Post_ID',
        tolerance=pd.Timedelta(minutes=TOLERANCE_MINUTES),
        direction='nearest',
        suffixes=('', '_future')
    )
    
    # 2. 建立「過去」特徵 (T-10)
    print(f"🔄 執行瞬時動能配對 (T-{VELOCITY_DELTA_MINUTES}min)...")
    merged['velocity_lookup_time'] = merged['crawl_time'] - pd.Timedelta(minutes=VELOCITY_DELTA_MINUTES)
    
    df_past = df[['Post_ID', 'crawl_time', 'push_count']].copy()
    df_past = df_past.sort_values('crawl_time')
    
    merged_final = pd.merge_asof(
        merged,
        df_past,
        left_on='velocity_lookup_time',
        right_on='crawl_time',
        by='Post_ID',
        tolerance=pd.Timedelta(minutes=TOLERANCE_MINUTES),
        direction='nearest',
        suffixes=('', '_prev') 
    )
    
    # 移除配對失敗的樣本 (只保留有未來的資料)
    valid_data = merged_final.dropna(subset=['push_count_future'])
    
    # 重置索引，避免後續處理出錯
    valid_data = valid_data.reset_index(drop=True)
    
    del df, df_future, df_past, merged, merged_final
    gc.collect()
    
    return valid_data

def prepare_data_for_train(df):
    print("🛠️ 正在生成訓練特徵...")
    
    # 🚨 [修正] 直接將包含 _prev 欄位的 df 傳入
    # feature_utils 會自動偵測並使用這些欄位，不會觸發 merge，避免爆炸
    X = feature_utils.prepare_features_for_model(df, df_prev=None)
    
    # 標籤
    raw_score = df['push_count_future'] + df['boo_count_future']
    y = np.floor(5 * np.log1p(raw_score)).astype(int).clip(0, 30)
    
    # Group
    group = df.groupby('crawl_time', sort=False).size().to_list()
    
    return X, y, group

def run_training_pipeline(days_back=7):
    print("\n" + "="*50)
    print(f"🏋️‍♂️ 啟動模型重訓流程 (資料範圍: 近 {days_back} 天)")
    print("="*50)
    
    full_df = load_recent_data(days_back)
    if full_df.empty: return False
    
    dataset = create_lifecycle_dataset(full_df)
    if dataset.empty: return False

    n = len(dataset)
    train_end = int(n * 0.8)
    
    df_train = dataset.iloc[:train_end].copy()
    df_val = dataset.iloc[train_end:].copy()
    
    print(f"📊 樣本數: Train={len(df_train)}, Val={len(df_val)}")
    
    X_train, y_train, g_train = prepare_data_for_train(df_train)
    X_val, y_val, g_val = prepare_data_for_train(df_val)
    print("🧠 開始訓練 LightGBM (Full Retrain)...")
    
    # 🆕 定義權重階梯 (0~30級)
    custom_label_gain = [2**i - 1 for i in range(31)]

    gbm = lgb.LGBMRanker(
        objective='lambdarank',
        metric=['ndcg', 'map', 'rmse'],
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        importance_type='gain',
        lambdarank_truncation_level=10, # 只專注優化前 10 名
        label_gain=custom_label_gain    # 給予爆文極高的權重
    )
    
    gbm.fit(
        X_train, y_train,
        group=g_train,
        eval_set=[(X_val, y_val)],
        eval_group=[g_val],
        eval_at=[10],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True),
            lgb.log_evaluation(period=50)
        ]
    )
    
    gbm.booster_.save_model(MODEL_OUTPUT)
    print(f"💾 重訓完成！模型已儲存至 {MODEL_OUTPUT}")
    
    # 顯示新特徵的重要性
    imp = pd.DataFrame({
        'feature': X_train.columns,
        'gain': gbm.feature_importances_
    }).sort_values('gain', ascending=False)
    print("\n🏆 新模型特徵重要性 (Top 10):")
    print(imp.head(10))
    
    del X_train, y_train, X_val, y_val
    gc.collect()
    
    return True

if __name__ == "__main__":
    run_training_pipeline(days_back=7)