import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import ndcg_score
import os
import gc

# 引入專案模組
import db_manager
import feature_utils
import adaptive_system

# --- 設定 ---
MODEL_FILE = 'ptt_lifecycle_model.txt'
HORIZONS = range(3, 13) # 3 到 12 小時
EVAL_DAYS = 3           # 使用最近 3 天的資料來評估
TOLERANCE_MINUTES = 30  # 配對容許誤差

def compute_ndcg(y_true, y_score, k=10):
    y_true = np.asarray([y_true])
    y_score = np.asarray([y_score])
    if k > y_true.shape[1]: k = y_true.shape[1]
    return ndcg_score(y_true, y_score, k=k) if k > 0 else 0.0

def load_data_pool(days=3):
    print(f"📂 載入最近 {days} 天的資料作為評估池...")
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    df = db_manager.query_snapshots_by_time_range(start_time, end_time)
    print(f"   -> 取得 {len(df)} 筆資料")
    return df

def evaluate_horizon(model, df_pool, hours):
    print(f"   Testing Horizon: {hours} hours...")
    
    # 建立配對
    df_pool = df_pool.sort_values('crawl_time')
    df_pool['target_time'] = df_pool['crawl_time'] + pd.Timedelta(hours=hours)
    
    df_future = df_pool[['Post_ID', 'crawl_time', 'push_count', 'boo_count']].copy()
    
    # 1. 執行未來配對 (T + H)
    merged = pd.merge_asof(
        df_pool,
        df_future,
        left_on='target_time',
        right_on='crawl_time',
        by='Post_ID',
        tolerance=pd.Timedelta(minutes=TOLERANCE_MINUTES),
        direction='nearest',
        suffixes=('', '_future')
    )
    
    # 移除無效資料
    valid_data = merged.dropna(subset=['push_count_future']).copy()
    if len(valid_data) < 100:
        print(f"      ⚠️ 樣本不足 ({len(valid_data)}), 跳過")
        return None

    # 2. 執行過去配對 (T - 10min) 用於計算速度
    valid_data['prev_time'] = valid_data['crawl_time'] - pd.Timedelta(minutes=10)
    df_prev_lookup = df_pool[['Post_ID', 'crawl_time', 'push_count']].copy()
    
    valid_with_prev = pd.merge_asof(
        valid_data.sort_values('crawl_time'),
        df_prev_lookup.sort_values('crawl_time'),
        left_on='prev_time',
        right_on='crawl_time',
        by='Post_ID',
        tolerance=pd.Timedelta(minutes=10),
        suffixes=('', '_prev')
    )
    
    # 🚨 [關鍵修正] 重置索引，確保資料列順序是 0, 1, 2...
    # 這樣後續用 groupby 取得的索引就能直接對應到 numpy array
    valid_with_prev = valid_with_prev.reset_index(drop=True)
    
    # 計算特徵 (Velocity)
    t_now = valid_with_prev['crawl_time']
    t_prev = valid_with_prev['crawl_time_prev']
    valid_with_prev['time_diff'] = (t_now - t_prev).dt.total_seconds() / 60
    valid_with_prev['push_diff'] = valid_with_prev['push_count'] - valid_with_prev['push_count_prev']
    valid_with_prev['push_velocity'] = valid_with_prev['push_diff'] / valid_with_prev['time_diff']
    valid_with_prev['push_velocity'] = valid_with_prev['push_velocity'].fillna(0)
    
    # 準備模型特徵
    X = feature_utils.prepare_features_for_model(valid_with_prev)
    X['push_velocity'] = valid_with_prev['push_velocity'] # 覆蓋確保正確
    
    # 預測
    preds = model.predict(X)
    
    # 計算 Ground Truth
    raw_score = valid_with_prev['push_count_future'] + valid_with_prev['boo_count_future']
    y_true = np.floor(5 * np.log1p(raw_score)).astype(int).clip(0, 30)
    
    # 計算 NDCG
    ndcg_list = []
    
    # Group by crawl_time (這時 group_idx 是整數索引)
    grouped = valid_with_prev.groupby('crawl_time')
    
    for name, group_idx in grouped.groups.items():
        if len(group_idx) < 5: continue
        
        # 因為前面做了 reset_index，這裡可以直接用整數索引取值
        g_y_true = y_true.iloc[group_idx].values
        g_preds = preds[group_idx]
        
        s = compute_ndcg(g_y_true, g_preds, k=10)
        ndcg_list.append(s)
        
    if not ndcg_list:
        return 0.0
        
    avg_ndcg = np.mean(ndcg_list)
    print(f"      -> Avg NDCG@10: {avg_ndcg:.4f}")
    
    return avg_ndcg

def main():
    print("🚀 啟動長程預測評估系統 (3hr - 12hr)...")
    
    if not os.path.exists(MODEL_FILE):
        print("❌ 找不到模型檔案")
        return

    model = lgb.Booster(model_file=MODEL_FILE)
    df_pool = load_data_pool(days=EVAL_DAYS)
    
    if df_pool.empty:
        print("❌ 無資料可供評估")
        return

    results = []
    
    print("\n📊 開始評估各時段準確度...")
    for h in HORIZONS:
        score = evaluate_horizon(model, df_pool, h)
        if score is not None:
            results.append({'Horizon (Hours)': h, 'NDCG@10': score})
            
    # 繪圖
    if results:
        res_df = pd.DataFrame(results)
        print("\n📈 評估結果:")
        print(res_df)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=res_df, x='Horizon (Hours)', y='NDCG@10', marker='o')
        plt.title('Model Accuracy Decay over Time (3-12 Hours)')
        plt.ylim(0.5, 1.0)
        plt.grid(True)
        plt.savefig('long_term_accuracy.png')
        print("✅ 圖表已儲存至 long_term_accuracy.png")
        
        # 預測未來
        print("\n🔮 [即時預測] 基於最新資料預測未來...")
        latest_df, latest_time = adaptive_system.load_latest_snapshot_from_db()
        if latest_df is not None:
            # 取得 T-10 計算速度
            t_prev = latest_time - timedelta(minutes=10)
            df_prev, _ = db_manager.query_nearest_snapshot(t_prev)
            
            X_latest = feature_utils.prepare_features_for_model(latest_df, df_prev)
            scores = model.predict(X_latest)
            
            latest_df['pred_score'] = scores
            top10 = latest_df.sort_values('pred_score', ascending=False).head(10)
            
            print(f"資料時間: {latest_time.strftime('%Y-%m-%d %H:%M')}")
            print("-" * 60)
            print(f"{'預測排名':<5} | {'目前推數':<8} | {'標題'}")
            print("-" * 60)
            for i, row in enumerate(top10.itertuples()):
                print(f"#{i+1:<4} | {row.push_count:<8} | {row.title}")
            print("-" * 60)
            print("(註：這是模型認為在未來 3~12 小時內最具競爭力的文章)")

if __name__ == "__main__":
    main()