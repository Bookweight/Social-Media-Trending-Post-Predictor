import time
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, ndcg_score
import math
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# 引入模組
import ptt_moniter 
import feature_utils 
import train_model_lifecycle
import db_manager  # 引入資料庫模組

# --- 設定 ---
MODEL_FILE = 'ptt_lifecycle_model.txt'
VERSION_FILE = 'model_version.txt'
PRED_LOG_FILE = 'pred.csv'
PLOT_DIR = 'results'
LOOK_AHEAD_MINUTES = 120  # 預測 2 小時後
VELOCITY_DELTA_MINUTES = 10 

def get_current_version():
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 1
    return 1

def increment_version():
    v = get_current_version() + 1
    with open(VERSION_FILE, 'w') as f:
        f.write(str(v))
    print(f"🆙 模型版本已升級為 v{v}")
    return v

# 🆕 [DB] 從資料庫讀取最新快照
def load_latest_snapshot_from_db():
    conn = db_manager.get_conn()
    cursor = conn.cursor()
    # 找最新的爬蟲時間
    cursor.execute("SELECT MAX(crawl_time) FROM snapshots")
    result = cursor.fetchone()
    
    if not result or not result[0]:
        conn.close()
        return None, None
        
    latest_time_str = result[0]
    # 轉換為 datetime 物件
    latest_time = pd.to_datetime(latest_time_str)
    
    # 讀取該時間點的所有文章
    # 注意：這裡直接讀取該時間點的所有資料
    query = "SELECT * FROM snapshots WHERE crawl_time = ?"
    df = pd.read_sql(query, conn, params=(latest_time_str,))
    conn.close()
    
    # 確保時間格式正確
    if not df.empty:
        df['crawl_time'] = pd.to_datetime(df['crawl_time'])
        if 'post_time' in df.columns:
            df['post_time'] = pd.to_datetime(df['post_time'])
            
    return df, latest_time

def compute_ranking_metrics(y_true, y_score, k=10):
    y_true = np.asarray([y_true])
    y_score = np.asarray([y_score])
    k = min(k, y_true.shape[1])
    if k <= 0: return 0.0, 0.0, 0.0
    try:
        from scipy.stats import kendalltau
        ndcg_10 = ndcg_score(y_true, y_score, k=10)
        ndcg_3 = ndcg_score(y_true, y_score, k=3)
        tau, _ = kendalltau(y_true[0], y_score[0])
        return ndcg_10, ndcg_3, tau
    except:
        return 0.0, 0.0, 0.0

def log_prediction_performance(timestamp, model_metrics, base_metrics, stage="adaptive_verify"):
    file_exists = os.path.isfile(PRED_LOG_FILE)
    
    lift = 0.0
    if base_metrics['ndcg'] > 0:
        lift = (model_metrics['ndcg'] - base_metrics['ndcg']) / base_metrics['ndcg'] * 100

    with open(PRED_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'stage', 'model_rmse', 'model_ndcg', 'base_rmse', 'base_ndcg', 'lift_percent'])
            
        writer.writerow([
            timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            stage,
            f"{model_metrics['rmse']:.4f}",
            f"{model_metrics['ndcg']:.4f}",
            f"{base_metrics['rmse']:.4f}",
            f"{base_metrics['ndcg']:.4f}",
            f"{lift:+.2f}%"
        ])
    print(f"📝 績效已記錄至 {PRED_LOG_FILE} (Lift: {lift:+.2f}%)")

def print_side_by_side(list_a, list_b, title_a, title_b):
    print("-" * 95)
    print(f"{title_a:<45} | {title_b:<45}")
    print("-" * 95)
    
    for i in range(10):
        str_a, str_b = "", ""
        if i < len(list_a):
            row = list_a.iloc[i]
            title = str(row.title)[:18] + "..." if len(str(row.title)) > 18 else str(row.title)
            score_info = f"[{row.score_val:.1f}]" if 'score_val' in row else f"(推:{row.push_count})"
            str_a = f"#{i+1} {score_info} {title}"

        if i < len(list_b):
            row = list_b.iloc[i]
            title = str(row.title)[:18] + "..." if len(str(row.title)) > 18 else str(row.title)
            score_info = f"[{row.score_val:.1f}]" if 'score_val' in row else f"(推:{row.push_count})"
            str_b = f"#{i+1} {score_info} {title}"
            
        print(f"{str_a:<45} | {str_b:<45}")
    print("-" * 95)

def save_feature_importance_plot(model, timestamp):
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        
    try:
        importance = model.feature_importance(importance_type='gain')
        feature_name = model.feature_name()
        
        df_importance = pd.DataFrame({
            'feature': feature_name,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        
        # 🚨 [修正] 加入 hue 與 legend=False 以消除警告
        sns.barplot(
            x='importance', 
            y='feature', 
            hue='feature', 
            data=df_importance, 
            palette='viridis', 
            legend=False
        )
        
        plt.title(f'Feature Importance (Gain) - {timestamp.strftime("%Y-%m-%d %H:%M")}', fontsize=16)
        plt.xlabel('Gain Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(df_importance['importance']):
            plt.text(v, i, f' {v:.0f}', va='center', fontsize=9)

        plt.tight_layout()
        filename = f"feature_importance_{timestamp.strftime('%Y%m%d_%H%M')}.png"
        save_path = os.path.join(PLOT_DIR, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ 繪圖失敗: {e}")

def calculate_dynamic_weight(df):
    """根據全站推文總量 (流量) 決定 AI 的權重"""
    total_push = df['push_count'].sum()
    article_count = len(df)
    
    if article_count == 0: return 0.5
    
    avg_push = total_push / article_count
    
    # Sigmoid function centered at 10
    sigmoid = 1 / (1 + np.exp(-(avg_push - 10) / 3))
    weight = 0.3 + (0.6 * sigmoid)
    
    print(f"⚖️ [動態權重] 全站平均推文: {avg_push:.1f} -> AI 權重: {weight:.2f}")
    return weight

# 🆕 [DB] 預測邏輯 (接收 DataFrame 和 時間)
def predict_future_rank(df, current_time, model):
    if df.empty: return

    print(f"\n🔮 [預測模式] 資料時間: {current_time.strftime('%Y-%m-%d %H:%M')}")
    
    # 1. 從 DB 找 T-10 資料 (計算動能)
    target_prev_time = current_time - timedelta(minutes=VELOCITY_DELTA_MINUTES)
    df_prev, _ = db_manager.query_nearest_snapshot(target_prev_time)
    
    if df_prev is not None and not df_prev.empty:
        print("   -> 成功載入 T-10min 資料以計算瞬時動能")
    else:
        print("   -> ⚠️ 無法載入 T-10min 資料 (可能剛啟動)，動能特徵將使用預設值")

    # 2. 準備特徵
    X = feature_utils.prepare_features_for_model(df, df_prev)
    
    # 3. 預測
    ai_score = model.predict(X)
    base_score = np.floor(5 * np.log1p(df['push_count'])).clip(0, 30)
    w = calculate_dynamic_weight(df)
    
    df['pred_score'] = (w * ai_score) + ((1-w) * base_score)
    
    # 4. 顯示結果
    top_pred = df.sort_values('pred_score', ascending=False).head(10).copy()
    top_pred['score_val'] = top_pred['pred_score']
    top_curr = df.sort_values('push_count', ascending=False).head(10).copy()
    
    print(f"🚀 預測 {LOOK_AHEAD_MINUTES} 分鐘後的趨勢分析")
    print_side_by_side(
        top_pred, top_curr, 
        f"🤖 AI 預測排名 (權重 {w:.2f})", 
        f"🔥 目前實際排名 (當下熱度)"
    )

# 🆕 [DB] 學習邏輯
def adaptive_learning(df_now, current_time, model, stage_label):
    # 1. 從 DB 找 T-120 (過去的預測當下)
    target_time_verify = current_time - timedelta(minutes=LOOK_AHEAD_MINUTES)
    df_past, real_past_time = db_manager.query_nearest_snapshot(target_time_verify, tolerance_seconds=1800)
    
    if df_past is None or df_past.empty:
        print(f"⚠️ 資料庫中找不到 T-{LOOK_AHEAD_MINUTES} 分鐘前的資料，跳過驗證")
        return model

    print(f"\n🧠 [學習模式] 回溯驗證: {real_past_time.strftime('%H:%M')} -> {current_time.strftime('%H:%M')}")

    # 2. 從 DB 找 T-130 (過去的過去，為了算當時的速度)
    target_time_velocity = real_past_time - timedelta(minutes=VELOCITY_DELTA_MINUTES)
    df_prev_past, _ = db_manager.query_nearest_snapshot(target_time_velocity)

    # 3. 合併驗證
    merged = pd.merge(df_past, df_now[['Post_ID', 'push_count', 'boo_count']], 
                      on='Post_ID', suffixes=('', '_future'))
    
    if len(merged) < 5: return model

    # 1. 準備真實標籤 (Ground Truth)
    merged['real_future_score'] = merged['push_count_future'] + merged['boo_count_future']
    y_true_grade = np.floor(5 * np.log1p(merged['real_future_score'])).astype(int).clip(0, 30)
    
    # 2. 準備特徵與預測
    X_train = feature_utils.prepare_features_for_model(merged, df_prev_past)
    
    # --- A. AI 模型預測 ---
    ai_preds = model.predict(X_train)
    
    # --- B. 笨蛋基準 (Stupid Baseline: Rank Freeze) ---
    base_preds = np.floor(5 * np.log1p(merged['push_count'])).clip(0, 30)

    # --- C. [新增] 動能基準 (Velocity Baseline) ---
    # 邏輯: 預測推數 = 目前推數 + (目前速度 * 預測時間長度)
    # 從 X_train 取出計算好的速度 (fillna防呆)
    current_velocity = X_train['push_velocity'].fillna(0)
    
    # 線性推演未來推文數
    projected_push = merged['push_count'] + (current_velocity * LOOK_AHEAD_MINUTES)
    # 防呆: 推文數不應減少
    projected_push = np.maximum(projected_push, merged['push_count'])
    
    # 轉成等級分 (0-30) 以便比較
    vel_preds = np.floor(5 * np.log1p(projected_push)).astype(int).clip(0, 30)

    # 3. 混合權重計算 (AI + 笨蛋)
    w = calculate_dynamic_weight(merged)
    mixed_preds = (w * ai_preds) + ((1-w) * base_preds)
    merged['old_pred_score'] = mixed_preds
    
    # 4. 指標計算 (Metrics)
    # (1) AI 混合模型
    model_rmse = math.sqrt(mean_squared_error(y_true_grade, mixed_preds))
    model_ndcg_10, model_ndcg_3, _ = compute_ranking_metrics(y_true_grade, mixed_preds, k=10)   
    
    # (2) 笨蛋基準
    base_eval = np.floor(5 * np.log1p(merged['push_count'])).astype(int).clip(0, 30)
    base_rmse = math.sqrt(mean_squared_error(y_true_grade, base_eval))
    base_ndcg_10, base_ndcg_3, _ = compute_ranking_metrics(y_true_grade, merged['push_count'], k=10)

    # (3) [新增] 動能基準
    vel_ndcg_10, vel_ndcg_3, _ = compute_ranking_metrics(y_true_grade, vel_preds, k=10)

    # 5. Lift 計算
    lift_base = 0.0
    if base_ndcg_10 > 0:
        lift_base = (model_ndcg_10 - base_ndcg_10) / base_ndcg_10 * 100
        
    lift_vel = 0.0
    if vel_ndcg_10 > 0:
        lift_vel = (model_ndcg_10 - vel_ndcg_10) / vel_ndcg_10 * 100

    # 6. 輸出結果
    print(f"📊 驗證成效比較 (Hybrid):")
    print(f"   - 混合模型 : NDCG@10={model_ndcg_10:.4f}, Lift(v.s.笨蛋)={lift_base:+.2f}%")
    print(f"   - 笨蛋基準 : NDCG@10={base_ndcg_10:.4f}")
    print(f"   - 動能基準 : NDCG@10={vel_ndcg_10:.4f} | AI v.s. 動能: {lift_vel:+.2f}%")
    
    # 寫入 Log (維持原格式，以免破壞 CSV 結構，但您可以在這裡考慮是否要加欄位)
    log_prediction_performance(
        current_time, 
        {'rmse': model_rmse, 'ndcg': model_ndcg_10}, 
        {'rmse': base_rmse, 'ndcg': base_ndcg_10},   
        stage=stage_label 
    )

    top_past_pred = merged.sort_values('old_pred_score', ascending=False).head(10).copy()
    top_past_pred['score_val'] = top_past_pred['old_pred_score']
    top_now_real = merged.sort_values('real_future_score', ascending=False).head(10).copy()
    top_now_real['push_count'] = top_now_real['real_future_score']
    
    print_side_by_side(
        top_past_pred, top_now_real,
        f"🤖 {LOOK_AHEAD_MINUTES}分前 混合預測 (w={w:.2f})",
        f"✅ {LOOK_AHEAD_MINUTES}分後 真實結果"
    )

    # 4. 增量訓練
    custom_label_gain = [2**i - 1 for i in range(31)]
    group = [len(X_train)]
    lgb_train = lgb.Dataset(X_train, y_true_grade, group=group)
    
    params = {
        'objective': 'lambdarank', 'metric': ['ndcg', 'map'], 'learning_rate': 0.01,
        'num_leaves': 31, 'verbosity': -1, 'lambdarank_truncation_level': 10,
        'label_gain': custom_label_gain
    }
    
    new_model = lgb.train(params, lgb_train, num_boost_round=10, init_model=model, keep_training_booster=True)
    new_model.save_model(MODEL_FILE)
    print("💾 模型已微調並存檔")
    
    return new_model

def smart_start_wait():
    print("🕵️‍♂️ [系統檢查] 偵測資料庫新鮮度...")
    df_last, last_time = load_latest_snapshot_from_db()
    
    if last_time is None:
        print("   -> 資料庫無資料，準備立即啟動。")
        return

    current_time = datetime.now()
    elapsed_seconds = (current_time - last_time).total_seconds()
    interval = ptt_moniter.INTERVAL_SECONDS
    
    if 0 <= elapsed_seconds < interval:
        wait_seconds = interval - elapsed_seconds + 5 
        print(f"✅ 最新資料 ({last_time.strftime('%H:%M')}) 僅在 {elapsed_seconds/60:.1f} 分鐘前產生。")
        print(f"⏳ 為避免重複爬取，系統將休眠 {wait_seconds:.0f} 秒...")
        try:
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            exit()
    else:
        print(f"⚡ 最新資料已是 {elapsed_seconds/60:.1f} 分鐘前，立即啟動爬蟲！")

def main_loop():
    print("🚀 PTT 自適應預測系統啟動 (資料庫版 + 動態防禦)")
    
    # 確保資料庫初始化
    if not os.path.exists(db_manager.DB_NAME):
        db_manager.init_db()

    if os.path.exists(MODEL_FILE):
        print("📂 載入現有模型...")
        model = lgb.Booster(model_file=MODEL_FILE)
    else:
        print("❌ 找不到模型，執行初始化訓練...")
        # 注意: 這裡 train_model_lifecycle 也需要更新為支援 DB 的版本
        train_model_lifecycle.run_training_pipeline(days_back=3)
        model = lgb.Booster(model_file=MODEL_FILE)

    author_cache = ptt_moniter.load_author_history()
    last_plot_hour = -1
    SCHEDULED_HOURS = [0, 6, 12, 18]
    
    last_retrain_time = datetime.now()
    RETRAIN_INTERVAL = timedelta(hours=24) 
    model_version = get_current_version()
    print(f"🔢 目前模型版本: v{model_version}")

    smart_start_wait()

    while True:
        cycle_start_time = datetime.now()
        try:
            print("\n" + "="*95)
            
            # --- 重訓檢查 ---
            time_since_retrain = datetime.now() - last_retrain_time
            if time_since_retrain > RETRAIN_INTERVAL:
                print(f"⏰ 已距離上次重訓 {time_since_retrain}，開始執行每日重訓...")
                success = train_model_lifecycle.run_training_pipeline(days_back=7)
                if success:
                    model = lgb.Booster(model_file=MODEL_FILE)
                    last_retrain_time = datetime.now()
                    model_version = increment_version()
            
            # 1. 爬蟲 (會自動寫入 DB)
            has_data, _ = ptt_moniter.run_snapshot(author_cache)
            
            # 2. 從 DB 讀取
            df_now, current_time = load_latest_snapshot_from_db()
            
            if df_now is not None:
                # ... (預測、學習、繪圖邏輯保持不變) ...
                predict_future_rank(df_now, current_time, model)
                if has_data:
                    model = adaptive_learning(df_now, current_time, model, stage_label=f"adaptive_v{model_version}")
                # ...
            else:
                print("❌ 資料庫讀取失敗或無資料")
            
            # 3. 計算下一輪的目標時間 (Fixed Rate Scheduling)
            target_next_time = cycle_start_time + timedelta(seconds=ptt_moniter.INTERVAL_SECONDS)
            now = datetime.now()
            sleep_seconds = (target_next_time - now).total_seconds()
            
            if sleep_seconds > 0:
                print(f"✅ 本輪耗時: {(now - cycle_start_time).total_seconds():.1f} 秒")
                print(f"😴 等待中... 下次執行: {target_next_time.strftime('%H:%M:%S')}")
                time.sleep(sleep_seconds)
            else:
                print(f"⚠️ 警告: 本輪耗時過長 ({(now - cycle_start_time).total_seconds():.1f} 秒)，立即啟動下一輪！")
                # 不睡覺，直接趕進度

        except KeyboardInterrupt:
            print("🛑 停止")
            break
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60) # 出錯時休息一下

if __name__ == "__main__":
    main_loop()