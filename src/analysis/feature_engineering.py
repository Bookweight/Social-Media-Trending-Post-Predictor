import pandas as pd
import numpy as np
import os
from collections import Counter
from datetime import datetime, timedelta
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# --- 1. 設定與常量 ---
DATA_ROOT = 'data'
# 假設我們從發文後 60 分鐘的 real_push_score 變化來定義爆紅程度
TARGET_DELTA_MINUTES = 60
# 設定 One-Hot Encoding 的 Top N 關鍵詞數量
TOP_N_KEYWORDS = 300 
# Target Encoding 用的平滑係數 (用於防止低頻詞過度編碼)
SMOOTHING_ALPHA = 100 
# 優化: push_boo_ratio 的截斷上限 (取代除以零的 1000.0 佔位符)
MAX_PUSH_BOO_RATIO_CLIP = 500.0 
# 新增: push_acceleration 的截斷上限 (防止極端加速值)
MAX_PUSH_ACCELERATION_CLIP = 5.0 

# --- 2. 數據加載與預處理核心功能 ---

def load_and_prepare_data(data_root):
    """
    從資料夾結構中載入所有快照數據，並將它們組合、清洗。
    此版本確保載入所有必需的欄位。
    """
    all_data = []
    # 遍歷 'data' 資料夾下的所有日期子資料夾
    for date_folder in os.listdir(data_root):
        date_path = os.path.join(data_root, date_folder)
        if os.path.isdir(date_path):
            # 遍歷日期資料夾內的所有 CSV 快照
            for filename in os.listdir(date_path):
                if filename.endswith('.csv'):
                    filepath = os.path.join(date_path, filename)
                    try:
                        df = pd.read_csv(filepath)
                        # 更新: 確保所有用於清洗和特徵工程的欄位都被載入
                        required_cols = ['Post_ID', 'crawl_time', 'post_time', 'real_push_score', 'key_phrases', 
                                         'push_count', 'boo_count', 'push_boo_ratio', 'push_acceleration', 'author_avg_push']
                        if not all(col in df.columns for col in required_cols):
                            continue

                        # 轉換時間格式
                        df['crawl_time'] = pd.to_datetime(df['crawl_time'])
                        df['post_time'] = pd.to_datetime(df['post_time'])
                        df['snapshot_id'] = df['crawl_time'].astype(str)
                        all_data.append(df[required_cols + ['snapshot_id']])
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        continue
                        
    if not all_data:
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)
    # 確保按 Post_ID 和時間排序，對 Target 計算至關重要
    full_df.sort_values(by=['Post_ID', 'crawl_time'], inplace=True, ignore_index=True)
    return full_df

def clean_data(df, max_ratio_clip, max_acceleration_clip):
    """
    清洗數據中的異常值：push_boo_ratio 的除以零錯誤、push_acceleration 的極端值，
    以及作者平均推文數和關鍵詞的缺失值。
    """
    print(f"  [清洗中] 正在處理數值與關鍵詞異常點...")
    
    # --- 1. 清洗 push_boo_ratio (除以零錯誤處理) ---
    mask_zero_boo = (df['boo_count'] == 0)
    
    # Case A: push_count > 0 且 boo_count = 0 (真正的高比率) -> 截斷
    mask_high_ratio = mask_zero_boo & (df['push_count'] > 0)
    df.loc[mask_high_ratio, 'push_boo_ratio'] = max_ratio_clip
    
    # Case B: push_count = 0 且 boo_count = 0 (零分母、零分子) -> 設為中性值 1.0
    mask_neutral_ratio = mask_zero_boo & (df['push_count'] == 0)
    df.loc[mask_neutral_ratio, 'push_boo_ratio'] = 1.0

    print(f"  [清洗] push_boo_ratio 已處理 {mask_zero_boo.sum()} 筆 boo_count=0 的記錄。")
    
    # --- 2. 清洗 push_acceleration (極端值截斷) ---
    print(f"  [清洗中] 正在處理 push_acceleration 的異常值 (截斷上限: {max_acceleration_clip})...")
    # 將所有高於上限的數值設為上限值
    df['push_acceleration'] = df['push_acceleration'].clip(upper=max_acceleration_clip)
    
    # --- 3. 處理 author_avg_push (缺失值填補) ---
    global_author_avg_push_mean = df['author_avg_push'].mean()
    # 使用全局均值填補 NaN (新作者或爬取失敗)
    df['author_avg_push'] = df['author_avg_push'].fillna(global_author_avg_push_mean)
    print(f"  [清洗] author_avg_push 的 NaN 已用全局均值 {global_author_avg_push_mean:.2f} 填補。")

    # --- 4. 處理 key_phrases (缺失值標記) ---
    # 將 NaN 關鍵詞替換為單一標籤，避免訊息丟失
    df['key_phrases'] = df['key_phrases'].fillna('NO_KEYWORDS')
    print("  [清洗] key_phrases 的 NaN 已替換為 'NO_KEYWORDS' 標籤。")
    
    return df

def calculate_target_delta(df, delta_minutes):
    """
    優化: 使用 groupby().apply() 在組內進行高效篩選，取代在整個 DF 上重複查詢。
    計算文章在 'delta_minutes' 後的推文分數增量 (Delta Push Score)。
    """
    print("  [優化中] 正在計算目標變量 (Delta Push Score)...")
    df['future_crawl_time'] = df['crawl_time'] + timedelta(minutes=delta_minutes)
    
    def compute_delta_for_post(group):
        """對單篇文章 (Post_ID group) 應用此邏輯"""
        target_scores = []
        
        # 遍歷該文章的所有快照
        for _, row in group.iterrows():
            current_score = row['real_push_score']
            target_time = row['future_crawl_time']
            
            # 在該文章組 (group) 內，尋找未來所有時間 >= target_time 的快照
            future_snapshots = group[group['crawl_time'] >= target_time]
            
            if not future_snapshots.empty:
                # 採用 T+delta_minutes 後所有快照中的最高分數
                max_future_score = future_snapshots['real_push_score'].max()
                
                # 計算 Delta Push Score，結果至少為 0
                delta_score = max(0, max_future_score - current_score)
                target_scores.append(delta_score)
            else:
                # 無法計算 Target (太新的快照)
                target_scores.append(np.nan) 
        
        # 返回一個與 group 索引對齊的 Series
        return pd.Series(target_scores, index=group.index)

    # 應用計算函式到每個 Post_ID 群組
    tqdm.pandas(desc="Calculating Target Delta")
    df['target'] = df.groupby('Post_ID', group_keys=False).progress_apply(compute_delta_for_post)

    # 刪除無法計算 Target 的記錄
    df.dropna(subset=['target'], inplace=True)
    
    # 設置最終的 Target Score
    df['target_score'] = df['target'] 
    
    return df

# --- 3. 關鍵詞特徵轉換 ---

def get_global_vocabulary(df):
    """提取並計算所有關鍵詞的總頻率，用於稀疏化處理。"""
    all_keywords = []
    # 確保 'key_phrases' 不是 NaN，並用分號分隔
    # 統一將關鍵詞轉為列表，方便後續處理
    # 注意: clean_data 已經將 NaN 替換為 'NO_KEYWORDS'
    keyword_lists = df['key_phrases'].apply(lambda x: x.split(' ; '))
    for kw_list in keyword_lists:
        all_keywords.extend(kw_list)
    
    keyword_counts = Counter(all_keywords)
    # 篩選出出現次數最多的 TOP_N_KEYWORDS
    most_common_keywords = {word for word, count in keyword_counts.most_common(TOP_N_KEYWORDS)}
    
    print(f"總詞彙量: {len(keyword_counts)}，選取 Top {TOP_N_KEYWORDS} 進行 One-Hot 編碼。")
    return most_common_keywords

def create_keyword_features(df, most_common_keywords):
    """
    創建兩種關鍵詞特徵：One-Hot 稀疏特徵 和 Target Encoding 特徵。
    優化: 使用 MultiLabelBinarizer 進行 One-Hot Encoding。
    """
    # -----------------------------------------------------------
    print("\n--- 3.1 關鍵詞 One-Hot Encoding (Top N, 向量化) ---")
    
    # 1. 準備數據: 將 key_phrases 轉換為列表，並只保留 Top N 關鍵詞
    keyword_list_filtered = df['key_phrases'].apply(
        lambda x: [kw for kw in x.split(' ; ') if kw in most_common_keywords]
    )
    
    # 2. 使用 MultiLabelBinarizer 進行向量化 One-Hot
    mlb = MultiLabelBinarizer()
    # 必須先 fit 再 transform，並確保索引是對齊的
    keyword_matrix = mlb.fit_transform(keyword_list_filtered)
    
    # 創建新的 DataFrame，並確保欄位名清晰
    keyword_df = pd.DataFrame(
        keyword_matrix, 
        columns=[f'KW_OH_{kw}' for kw in mlb.classes_], 
        index=df.index
    )
    
    # 3. 合併回主 DataFrame
    df = pd.concat([df, keyword_df], axis=1)

    # -----------------------------------------------------------
    print("\n--- 3.2 關鍵詞 Target Encoding (歷史熱度, K-Fold 平滑) ---")
    
    # Target Encoding 欄位名稱
    df['KW_Target_Encoded'] = np.nan
    # 計算全局平均 Target (先驗知識)
    global_mean_target = df['target_score'].mean()
    
    print(f"全局平均 Target Score: {global_mean_target:.4f}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 輔助函式: 將 key_phrases 轉換為關鍵詞列表 (在迴圈外一次處理)
    df['keywords_list'] = df['key_phrases'].apply(lambda x: x.split(' ; '))


    for fold, (train_index, val_index) in enumerate(kf.split(df)):
        df_train = df.iloc[train_index]
        
        # 1. 在訓練集上建立 Keyword -> Target 映射
        
        # 優化: 使用 explode/groupby/agg 進行 Target 統計
        temp_train = df_train[['keywords_list', 'target_score']].explode('keywords_list')
        kw_stats = temp_train.groupby('keywords_list')['target_score'].agg(['count', 'mean']).reset_index()
        
        encoded_dict = {}
        for _, row in kw_stats.iterrows():
            kw = row['keywords_list']
            count = row['count']
            mean_target = row['mean']
            
            # Target Encoding 公式: (Count * Mean + Alpha * Global_Mean) / (Count + Alpha)
            smoothed_mean = (count * mean_target + SMOOTHING_ALPHA * global_mean_target) / (count + SMOOTHING_ALPHA)
            encoded_dict[kw] = smoothed_mean

        # 2. 應用映射到驗證集 (避免數據洩露)
        def apply_encoding_optimized(keywords_list):
            if not keywords_list: return global_mean_target
            # 文章的 Target Encoding 取其所有關鍵詞的平均 Target Score
            scores = [encoded_dict.get(kw, global_mean_target) for kw in keywords_list]
            return np.mean(scores)

        # 應用到驗證集
        df.loc[val_index, 'KW_Target_Encoded'] = df.iloc[val_index]['keywords_list'].apply(apply_encoding_optimized)
        
    # 清理臨時欄位
    df.drop(columns=['keywords_list'], inplace=True)
    
    return df

# --- 4. 執行流程 ---
if __name__ == '__main__':
    # 模擬數據加載
    print(f"--- 1. 載入原始快照數據 (從 {DATA_ROOT} 資料夾)... ---")
    
    # 為了讓腳本能運行，我們在沒有完整資料夾結構時，先使用提供的單一CSV作為模擬數據
    if not os.path.exists(DATA_ROOT) or not any(os.path.isdir(os.path.join(DATA_ROOT, d)) for d in os.listdir(DATA_ROOT)):
        print("警告: 缺少 data/日期/ 結構。使用單一上傳檔案進行模擬。")
        try:
            # 使用用戶提供的 ptt_snapshot_v2_20251127_0042.csv 作為測試數據
            df = pd.read_csv('ptt_snapshot_v2_20251127_0042.csv')
            df['crawl_time'] = pd.to_datetime(df['crawl_time'])
            df['snapshot_id'] = df['crawl_time'].astype(str) # 模擬 Group ID
        except FileNotFoundError:
            print("錯誤: 找不到提供的 CSV 檔案。無法繼續。")
            df = pd.DataFrame()
    else:
        df = load_and_prepare_data(DATA_ROOT)
        
    if df.empty:
        print("數據加載失敗，終止執行。")
    else:
        print(f"總共載入 {len(df)} 筆快照記錄。")
        
        # 執行數據清洗
        df_cleaned = clean_data(df, MAX_PUSH_BOO_RATIO_CLIP, MAX_PUSH_ACCELERATION_CLIP)

        # Step 2: 計算 Target (Delta Push Score)
        print(f"\n--- 2. 計算目標變量 (Delta Push Score @ T+{TARGET_DELTA_MINUTES}分鐘) ---")
        df_with_target = calculate_target_delta(df_cleaned, TARGET_DELTA_MINUTES)
        print(f"成功計算 Target 的記錄數: {len(df_with_target)}")

        # Step 3: 提取詞彙表
        most_common_keywords = get_global_vocabulary(df_with_target)

        # Step 4: 創建關鍵詞特徵
        final_df = create_keyword_features(df_with_target, most_common_keywords)

        # 顯示最終特徵的形狀和部分欄位
        print("\n--- 4. 關鍵詞特徵化結果 ---")
        print(f"最終 DataFrame 形狀: {final_df.shape}")
        
        feature_cols = [col for col in final_df.columns if col.startswith('KW_OH_') or col.startswith('KW_Target_')]
        print(f"生成的關鍵詞特徵欄位數量: {len(feature_cols)}")
        print("部分欄位名稱 (前3個):")
        print(feature_cols[:3])
        print("\n部分結果展示 (包含清洗後的 push_boo_ratio, push_acceleration 和 Target Encoding 特徵):")
        display_cols = ['Post_ID', 'crawl_time', 'key_phrases', 'push_boo_ratio', 'push_acceleration', 'author_avg_push', 'target_score', 'KW_Target_Encoded'] + [c for c in feature_cols if c.startswith('KW_OH_')][:3]
        print(final_df[display_cols].head())
        
        # --- 5. LGBM 排名模型準備 ---
        
        # Group ID: 排名模型的關鍵。在同一個 'snapshot_id' 下的文章進行排名。
        final_df['group_id'] = final_df['snapshot_id']
        group_sizes = final_df.groupby('group_id').size().tolist()

        print("\n--- 5. LGBM 訓練準備 ---")
        print("LGBM 訓練時所需的核心參數:")
        print(f"- 特徵 (X): 包含 {len(feature_cols)} 個關鍵詞特徵 + 其他數值特徵。")
        print(f"- 分組 (group): {group_sizes[:5]}... (前 5 組的大小)")