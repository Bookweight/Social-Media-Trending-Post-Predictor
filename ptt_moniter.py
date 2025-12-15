import requests
import cloudscraper
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import pandas as pd
import time
import random
import os
import glob
import jieba.analyse
import numpy as np

import db_manager

# --- 1. 配置與設定 ---
BOARD = 'Gossiping'
INITIAL_LOOKBACK_HOURS = 24  
REGULAR_LOOKBACK_HOURS = 12
INTERVAL_SECONDS = 600       
DATA_DIR = 'data'
AUTHOR_HISTORY_FILE = 'data/author_history_recalc.csv' 
DEBUG_MODE = False          
CLIPPING_THRESHOLD = 100     

# PTT 網址與 Headers
PTT_BASE_URL = 'https://www.ptt.cc'
PTT_URL = f'{PTT_BASE_URL}/bbs/{BOARD}/index.html'
HEADERS = {
    'User-Agent': 'Mozilla/50 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': f'{PTT_BASE_URL}/bbs/{BOARD}/index.html'
}
COOKIES = {'over18': '1'}

# 建立 CloudScraper
scraper = cloudscraper.create_scraper()

STOPWORDS = set([
    'http', 'https', 'com', 'tw', 'imgur', 'jpg', 'jpeg', 'png', 'gif', 
    'youtu', 'be', 'link', 'url', '新聞', '圖片', '連結', '記者', '報導', 
    '問題', '大家', '一個', '什麼', '這樣', '出來', '沒有', '可以', '怎麼'
])

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def log(msg):
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")

# --- 2. 數據讀取與輔助函式 ---

def update_author_history_index():
    print("🔄 正在更新作者歷史數據索引...")
    
    all_csv_files = glob.glob(os.path.join(DATA_DIR, '**', '*.csv'), recursive=True)
    history_file_abs = os.path.abspath(AUTHOR_HISTORY_FILE)
    
    target_files = []
    for f in all_csv_files:
        if os.path.abspath(f) != history_file_abs:
            target_files.append(f)
            
    if not target_files:
        print("⚠️ 無原始歷史資料可更新。")
        return {}

    df_list = []
    for f in all_csv_files:
        try:
            df = pd.read_csv(f, usecols=['Post_ID', 'author', 'real_push_score'])
            df_list.append(df)
        except:
            continue
    
    if not df_list:
        return {}

    full_df = pd.concat(df_list, ignore_index=True)
    
    unique_posts = full_df.sort_values('real_push_score', ascending=False).drop_duplicates(subset=['Post_ID'], keep='first')
    
    author_stats = unique_posts.groupby('author').agg(
        raw_avg=('real_push_score', 'mean'),
        count=('Post_ID', 'count')
    ).reset_index()
    
    C = 3
    global_mean = unique_posts['real_push_score'].mean() 
    
    author_stats['author_avg_push'] = (
        (C * global_mean) + (author_stats['count'] * author_stats['raw_avg'])
    ) / (C + author_stats['count'])
    
    final_df = author_stats[['author', 'author_avg_push']]
    final_df.to_csv(AUTHOR_HISTORY_FILE, index=False, encoding='utf-8-sig')
    
    print(f"✅ 作者歷史索引已更新 (含貝式平滑)，全站平均: {global_mean:.2f}")
    
    return final_df.set_index('author')['author_avg_push'].to_dict()

def load_author_history():
    if os.path.exists(AUTHOR_HISTORY_FILE):
        try:
            df = pd.read_csv(AUTHOR_HISTORY_FILE)
            return df.set_index('author')['author_avg_push'].to_dict()
        except Exception as e:
            print(f"⚠️ 讀取歷史索引檔失敗: {e}，嘗試重新計算...")
            return update_author_history_index()
    else:
        return update_author_history_index()

def get_soup(url):
    time.sleep(random.uniform(2.0, 5.0)) 
    try:
        resp = scraper.get(url, headers=HEADERS, cookies=COOKIES, timeout=30)
        if resp.status_code == 200:
            return BeautifulSoup(resp.text, 'html.parser')
        elif resp.status_code == 403:
            log("⚠️ Cloudflare 403 Forbidden")
        return None
    except Exception as e:
        log(f"❌ 連線錯誤: {e}")
        return None

def extract_key_phrases(text, topK=5):
    if not text or len(text) < 10: return ""
    keywords = jieba.analyse.textrank(text, topK=topK*2, withWeight=False, allowPOS=('n', 'ns', 'nt', 'nz', 'vn', 'v', 'eng', 'a', 'vg'))
    filtered = [k for k in keywords if k.lower() not in STOPWORDS and len(k) > 1]
    return " ; ".join(filtered[:topK])

def get_article_category(title):
    match = re.search(r'\[(.*?)\]', title) # 修正: 移除 ^ 以支援 Re:
    return match.group(1).strip() if match else 'General'

def clean_article_content(soup):
    main_content = soup.find(id='main-content')
    if not main_content: return "", 0, 0, 0, 0, 0
    
    for cls in ['article-metaline', 'article-metaline-right', 'push']:
        for div in main_content.find_all('div', class_=cls): div.extract()
    for span in main_content.find_all('span', class_='f2'): span.extract()

    text = main_content.text.strip()
    sp_count = len(re.findall(r'[^\w\s\u4E00-\u9FFF]', text))
    links = len(main_content.find_all('a', href=True))
    
    return text, len(list(jieba.cut(text))), text.count('?'), text.count('!'), links, sp_count

def get_article_content(url):
    try:
        resp = scraper.get(url, headers=HEADERS, cookies=COOKIES, timeout=30)
        if resp.status_code != 200: return None
    except: return None

    soup = BeautifulSoup(resp.text, 'html.parser')
    
    push_score, push_c, boo_c, arrow_c = 0, 0, 0, 0
    for p in soup.find_all('div', class_='push'):
        tag = p.find('span', class_='push-tag')
        if tag:
            t = tag.text.strip()
            if t == '推': push_score += 1; push_c += 1
            elif t == '噓': push_score -= 1; boo_c += 1
            elif t == '→': arrow_c += 1
    
    meta = soup.find_all('span', class_='article-meta-value')
    if len(meta) < 4: return None
    
    try:
        post_time = datetime.strptime(meta[3].text.strip(), '%a %b %d %H:%M:%S %Y')
    except: post_time = datetime.now()

    clean_text, wc, qc, ec, lc, spc = clean_article_content(soup)

    return {
        'author': meta[0].text.split('(')[0].strip(),
        'title': meta[2].text.strip(),
        'post_time': post_time,
        'real_push_score': push_score,
        'push_count': push_c, 'boo_count': boo_c, 'arrow_count': arrow_c,
        'clean_text': clean_text,
        'content_word_count': wc, 'question_mark_count': qc, 'exclamation_mark_count': ec,
        'link_count': lc, 'special_char_count': spc
    }

# --- 4. 主要爬蟲邏輯 ---

def run_snapshot(author_history_cache):
    current_time = datetime.now()
    crawl_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{crawl_time_str}] 啟動 V2 快照爬蟲任務 (含重試機制)...")

    lookback_hours = REGULAR_LOOKBACK_HOURS
    time_threshold = current_time - timedelta(hours=lookback_hours)
    
    # 🆕 新增: 重試設定
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        articles_data = []
        url = PTT_URL
        keep_scraping = True
        
        # --- 開始爬取迴圈 ---
        while keep_scraping:
            soup = get_soup(url)
            if not soup: 
                # 抓不到網頁，視為異常，跳出讓外層重試
                break 
                
            divs = soup.find_all('div', class_='r-ent')
            if not divs: break
            
            sep = soup.find('div', class_='r-list-sep')
            if sep:
                divs = sep.find_all_previous('div', class_='r-ent')
            
            for div in divs[::-1]:
                try:
                    link = div.find('a')
                    if not link: continue
                    
                    href = link['href']
                    article_url = PTT_BASE_URL + href
                    
                    nrec_node = div.find('div', class_='nrec')
                    nrec_tag = nrec_node.get_text().strip() if nrec_node else ""
                    
                    details = get_article_content(article_url)
                    if not details: continue

                    if details['title']:
                        details['title'] = details['title'].replace(',', '_')
                    
                    if details['post_time'] < time_threshold:
                        keep_scraping = False
                        break 
                    
                    post_time = details['post_time']
                    life_mins = (current_time - post_time).total_seconds() / 60
                    
                    accel = details['real_push_score'] / life_mins if life_mins > 1 else details['real_push_score']
                    accel = min(accel, CLIPPING_THRESHOLD)
                    
                    author_avg = author_history_cache.get(details['author'], 0.0)
                    h = post_time.hour
                    pb_ratio = details['push_count'] / details['boo_count'] if details['boo_count'] > 0 else 1000.0
                    word_count = details['content_word_count']
                    url_ratio = details['link_count'] / word_count if word_count > 0 else 0.0

                    articles_data.append({
                        'Post_ID': href.split('/')[-1].replace('.html', ''),
                        'source_board': BOARD,
                        'title': details['title'],
                        'url': article_url,
                        'author': details['author'],
                        'crawl_time': crawl_time_str,
                        'post_time': post_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'nrec_tag': nrec_tag,
                        'category': get_article_category(details['title']),
                        'title_char_count': len(details['title']),
                        'post_hour': h,
                        'real_push_score': details['real_push_score'],
                        'push_count': details['push_count'],
                        'boo_count': details['boo_count'],
                        'life_minutes': round(life_mins, 2),
                        'push_acceleration': round(accel, 4),
                        'push_boo_ratio': round(pb_ratio, 4),
                        'author_avg_push': round(author_avg, 2),
                        'content_word_count': word_count,
                        'content_url_ratio': round(url_ratio, 4),
                        'q_mark_density': round(details['question_mark_count']/(word_count or 1), 4),
                        'e_mark_density': round(details['exclamation_mark_count']/(word_count or 1), 4),
                        'key_phrases': extract_key_phrases(details['clean_text']),
                        'hour_sin': round(np.sin(2 * np.pi * h / 24), 4),
                        'hour_cos': round(np.cos(2 * np.pi * h / 24), 4),
                        'is_weekend': 1 if post_time.weekday() >= 5 else 0
                    })
                    
                except Exception as e:
                    continue
            
            if not keep_scraping: break
            
            btn = soup.find('div', class_='btn-group btn-group-paging')
            prev = btn.find('a', string='‹ 上頁') if btn else None
            if prev and 'href' in prev.attrs:
                url = PTT_BASE_URL + prev['href']
            else:
                break

        # --- 檢查結果並決定是否重試 ---
        if len(articles_data) > 0:
            df = pd.DataFrame(articles_data)
            # 1. (選用) 繼續存 CSV 當備份
            date_str = current_time.strftime('%Y%m%d')
            target_dir = os.path.join(DATA_DIR, date_str)
            if not os.path.exists(target_dir): os.makedirs(target_dir)
            fname = os.path.join(target_dir, f"ptt_snapshot_v2_{current_time.strftime('%Y%m%d_%H%M')}.csv")
            df.to_csv(fname, index=False, encoding='utf-8-sig')
            # 2. (核心) 存入資料庫
            print("正在寫入資料庫...")
            db_manager.insert_snapshot_df(df)
            print(f"✅ 成功儲存 {len(df)} 筆資料至 DB 與 CSV")
            return True, fname # fname 可以留著印 log，但後續邏輯改用 DB
        else:
            # 沒抓到資料，準備重試
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠️ 警告: 本次爬取未獲得任何資料 (嘗試 {retry_count}/{max_retries})")
                print("   -> 可能是連線不穩或被擋，休息 15 秒後重試...")
                time.sleep(15)
                # 重新初始化 scraper 以更換 session
                global scraper
                scraper = cloudscraper.create_scraper()
            else:
                print("❌ 錯誤: 重試多次仍無法獲取資料，跳過本次更新。")
                return False, None

if __name__ == '__main__':
    print(f"🚀 PTT 爆紅預測爬蟲 V2 (優化版) 已啟動")
    print(f"頻率: {INTERVAL_SECONDS/60} 分鐘 | 回溯: {REGULAR_LOOKBACK_HOURS} 小時")
    
    print("⏳ 初始化：正在建立作者歷史數據庫...")
    author_history_cache = update_author_history_index()
    
    loop_count = 0
    UPDATE_HISTORY_EVERY_N_LOOPS = 6 

    while True:
        try:
            # 🚨 修改: 接收新的回傳格式
            has_data, new_file = run_snapshot(author_history_cache)
            
            loop_count += 1
            if loop_count >= UPDATE_HISTORY_EVERY_N_LOOPS:
                print("🔄 定期更新作者歷史數據...")
                author_history_cache = update_author_history_index()
                loop_count = 0
            
            next_run = datetime.now() + timedelta(seconds=INTERVAL_SECONDS)
            print(f"😴 休眠中... 下次執行: {next_run.strftime('%H:%M:%S')}\n")
            time.sleep(INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            print("\n🛑 停止。")
            break
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")
            time.sleep(60)