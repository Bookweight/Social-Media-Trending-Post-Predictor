import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_NAME = 'ptt_data.db'

def get_conn():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    
    # 建立包含完整欄位的 snapshots 表格
    c.execute('''
    CREATE TABLE IF NOT EXISTS snapshots (
        -- 核心識別
        Post_ID TEXT,
        crawl_time TIMESTAMP,
        
        -- 文章基本資訊
        source_board TEXT,    -- 🚨 之前缺少的欄位
        title TEXT,
        author TEXT,
        category TEXT,
        url TEXT,
        
        -- 數值指標
        push_count INTEGER,
        boo_count INTEGER,
        real_push_score INTEGER,
        arrow_count INTEGER,
        
        -- 時間與週期
        post_time TIMESTAMP,
        post_hour INTEGER,
        life_minutes REAL,
        is_weekend INTEGER,
        hour_sin REAL,
        hour_cos REAL,
        
        -- 內容特徵
        content_word_count INTEGER,
        content_url_ratio REAL,
        title_char_count INTEGER,
        nrec_tag TEXT,
        key_phrases TEXT,
        q_mark_density REAL,
        e_mark_density REAL,
        
        -- 計算特徵
        push_acceleration REAL,
        push_boo_ratio REAL,
        author_avg_push REAL,
        push_velocity REAL, -- 預留欄位 (若未來爬蟲直接計算)
        
        PRIMARY KEY (Post_ID, crawl_time)
    )
    ''')
    
    # 建立索引以加速查詢
    c.execute('CREATE INDEX IF NOT EXISTS idx_crawl_time ON snapshots (crawl_time)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_post_id ON snapshots (Post_ID)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_post_time ON snapshots (post_time)')
    
    conn.commit()
    conn.close()
    print("✅ 資料庫重新初始化完成 (含完整 Schema)")

def insert_snapshot_df(df):
    if df.empty: return
    
    # 1. 確保時間格式正確
    for col in ['crawl_time', 'post_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # 2. 處理缺少欄位的防呆機制
    # 如果 DataFrame 裡有些欄位是 DB 沒有的，to_sql 預設會報錯或忽略
    # 如果 DataFrame 缺欄位，資料庫會填 NULL，這沒問題
    
    conn = get_conn()
    try:
        # 使用 append 模式
        df.to_sql('snapshots', conn, if_exists='append', index=False, chunksize=1000)
    except sqlite3.IntegrityError:
        # 忽略主鍵重複 (PK collision)
        pass
    except Exception as e:
        # 印出具體錯誤以便除錯，但不中斷程式
        print(f"❌ DB Write Error: {e}")
    finally:
        conn.close()

def query_snapshots_by_time_range(start_time, end_time):
    conn = get_conn()
    query = "SELECT * FROM snapshots WHERE crawl_time BETWEEN ? AND ?"
    df = pd.read_sql(query, conn, params=(start_time, end_time))
    conn.close()
    
    if not df.empty:
        df['crawl_time'] = pd.to_datetime(df['crawl_time'])
        if 'post_time' in df.columns:
            df['post_time'] = pd.to_datetime(df['post_time'])
    return df

def query_nearest_snapshot(target_time, tolerance_seconds=900):
    conn = get_conn()
    
    # 🚨 [修正] 將 Timestamp 轉為字串格式 (YYYY-MM-DD HH:MM:SS)
    if isinstance(target_time, pd.Timestamp):
        target_time_str = target_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        target_time_str = str(target_time)

    query = f'''
    SELECT * FROM snapshots 
    WHERE crawl_time BETWEEN datetime(?, '-{tolerance_seconds} seconds') 
                         AND datetime(?, '+{tolerance_seconds} seconds')
    '''
    
    # 使用轉換後的字串作為參數
    df = pd.read_sql(query, conn, params=(target_time_str, target_time_str))
    conn.close()
    
    if df.empty: return None, None

    df['crawl_time'] = pd.to_datetime(df['crawl_time'])
    
    # Python 端進行精確比對
    unique_times = df['crawl_time'].unique()
    best_time = min(unique_times, key=lambda x: abs((x - target_time).total_seconds()))
    
    df_result = df[df['crawl_time'] == best_time].copy()
    return df_result, best_time

# 初始化檢查
if not os.path.exists(DB_NAME):
    init_db()