import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager

# --- 1. 設定與常量 ---

# 🚨 請指定您的數據根目錄
DATA_ROOT = 'data'
# 🚨 請將此處替換為您要分析的實際「日期資料夾名稱」
# 程式將會讀取 DATA_ROOT/TARGET_DATE_FOLDER/ 下的所有 CSV
TARGET_DATE_FOLDER = '20251210' 
# 範例：如果您的資料路徑是 data/20251127/ptt_snapshot_v2_....csv，則保留 '20251127'

def analyze_post_time(data_root, target_folder):
    """
    讀取目標資料夾內所有 CSV 檔案，進行貼文去重，並繪製發文時段長條圖。
    """
    target_path = os.path.join(data_root, target_folder)
    all_data = []
    
    # --- 2. 數據載入 ---
    if not os.path.isdir(target_path):
        print(f"❌ 錯誤: 找不到目標資料夾 {target_path}。請檢查 TARGET_DATE_FOLDER 設定。")
        return

    print(f"--- 載入 {target_folder} 資料夾中所有快照數據 ---")
    
    for filename in os.listdir(target_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(target_path, filename)
            try:
                df = pd.read_csv(filepath)
                if 'post_time' in df.columns and 'Post_ID' in df.columns:
                    # 只保留 Post_ID, post_time 兩個關鍵欄位
                    all_data.append(df[['Post_ID', 'post_time']])
                else:
                    print(f"⚠️ 警告: 檔案 {filename} 缺少 'Post_ID' 或 'post_time' 欄位。")
            except Exception as e:
                print(f"❌ 讀取檔案 {filename} 時發生錯誤: {e}")
                
    if not all_data:
        print("沒有可用的數據進行分析。")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ 總共載入 {len(full_df)} 筆快照記錄。")
    
    # --- 3. 數據處理與分析 (去重是關鍵) ---
    
    # 針對 Post_ID 進行去重：只保留第一次出現的記錄作為該文章的代表
    # 這確保了我們只計算每個唯一的貼文一次
    unique_posts_df = full_df.drop_duplicates(subset=['Post_ID'], keep='first').copy()
    print(f"✅ 去除重複貼文後，剩下 {len(unique_posts_df)} 篇獨立文章。")

    # 修正：由於數據格式不確定性，重新強制轉換為 datetime 類型
    # 這是為了確保後續的 .dt 存取器可以使用
    unique_posts_df['post_time'] = pd.to_datetime(unique_posts_df['post_time'], errors='coerce')
    
    # 提取發文的小時 (0-23)。
    # 如果 post_time 轉換失敗（NaT），.dt.hour 會返回 NaN。
    unique_posts_df.loc[:, 'post_hour'] = unique_posts_df['post_time'].dt.hour
    
    # 清除任何 post_hour 為 NaN 的記錄（即 post_time 無效的記錄）
    unique_posts_df.dropna(subset=['post_hour'], inplace=True)
    # 將 post_hour 轉換為整數類型 (小時數)
    unique_posts_df.loc[:, 'post_hour'] = unique_posts_df['post_hour'].astype(int)

    
    # 計算每個小時的獨立發文數量
    hourly_counts = unique_posts_df['post_hour'].value_counts().sort_index()

    # 確保 0 到 23 小時都有計數，沒有的補 0
    full_index = pd.Index(range(24), name='post_hour')
    hourly_counts = hourly_counts.reindex(full_index, fill_value=0)
    
    # --- 4. 繪製長條圖 ---
    print("--- 正在繪製圖表 ---")
    
    plt.figure(figsize=(14, 7))
    
    # 繪製長條圖
    bars = plt.bar(hourly_counts.index, hourly_counts.values, color='#8a2be2', alpha=0.9, zorder=3)
    
    # 標題與軸標籤
    plt.title(f'PTT post time distribution in {target_folder} ', fontsize=18, pad=20)
    plt.xlabel('post time(Hour)', fontsize=14, labelpad=15)
    plt.ylabel('Number of different posts', fontsize=14, labelpad=15)
    
    # X 軸設定 (顯示 00:00, 01:00, ...)
    plt.xticks(hourly_counts.index, [f'{h:02d}:00' for h in hourly_counts.index], rotation=45, ha='right')
    
    # 網格線
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    
    # 在每個長條上方顯示數值
    max_count = max(hourly_counts.values) if not hourly_counts.empty else 0
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval + (max_count * 0.01), 
                     f'{yval}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    print("✅ 圖表已生成。")

if __name__ == '__main__':
    # 確保 DATA_ROOT 存在，如果沒有則嘗試使用當前目錄
    if not os.path.exists(DATA_ROOT):
        print(f"⚠️ 警告: 找不到 '{DATA_ROOT}' 資料夾。請在程式碼頂部修改 DATA_ROOT 設定。")
        # 即使警告，仍嘗試運行，因為用戶可能已將資料夾結構放在當前目錄下
        # 這裡假設 TARGET_DATE_FOLDER 可以直接被找到
        analyze_post_time('.', TARGET_DATE_FOLDER) 
    else:
        analyze_post_time(DATA_ROOT, TARGET_DATE_FOLDER)