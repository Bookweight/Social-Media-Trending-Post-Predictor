import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from streamlit_autorefresh import st_autorefresh

# Import your database manager
# Import your database manager
from src.data import db_manager

# --- Page Config ---
st.set_page_config(
    page_title="PTT Prediction Dashboard",
    page_icon="icon",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    /* Global Font Settings */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    }
    
    /* Card Styling */
    .metric-card {
        border: 1px solid #dee2e6;
        padding: 0 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        height: 90px;
        display: flex;
        flex-direction: row; 
        align-items: center; 
        justify-content: space-between;
        /* box-shadow removed for cleaner look, or keep light shadow */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
        transition: all 0.2s;
        background-color: white; /* Default bg */
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.1);
    }
    
    /* Left Side */
    .card-left {
        display: flex;
        align-items: center;
        overflow: hidden; 
        flex-grow: 1;
        margin-right: 15px;
    }
    
    .rank-badge {
        font-size: 32px;
        font-weight: 800;
        color: #343a40;
        margin-right: 18px;
        min-width: 45px;
        text-align: center;
        line-height: 1;
    }
    
    .article-info {
        display: flex;
        flex-direction: column;
        overflow: hidden;
        justify-content: center;
    }

    .article-title {
        font-size: 18px;
        font-weight: 700;
        color: #0d6efd;
        text-decoration: none;
        white-space: nowrap;      
        overflow: hidden;         
        text-overflow: ellipsis;  
        margin-bottom: 4px;
        display: block;
    }

    .rank-movement {
        font-size: 14px;
        font-weight: bold;
    }
    
    /* Right Side */
    .card-right {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        min-width: 100px;
        justify-content: center;
    }
    
    .metric-row {
        display: flex;
        align-items: baseline;
        margin-bottom: 2px;
    }
    
    .metric-val {
        font-size: 18px;
        font-weight: bold;
        color: #212529;
        margin-right: 4px;
    }
    
    .metric-label {
        font-size: 11px;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    .section-header {
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 8px;
        margin-bottom: 20px;
        margin-top: 15px;
    }
    
    /* Remove Plotly Border/Shadow */
    .js-plotly-plot .plotly .main-svg {
        box-shadow: none !important;
    }
    iframe {
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data(ttl=60)
def load_pred_log():
    if not os.path.exists('pred.csv'):
        return pd.DataFrame()
    df = pd.read_csv('pred.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if df['lift_percent'].dtype == object:
        df['lift_percent'] = df['lift_percent'].str.rstrip('%').astype(float)
        
    return df.sort_values('timestamp')

def load_comparison_data(pred_time):
    df_pred, _ = db_manager.query_nearest_snapshot(pred_time, tolerance_seconds=300)
    future_time = pred_time + timedelta(minutes=120)
    df_real, real_timestamp = db_manager.query_nearest_snapshot(future_time, tolerance_seconds=600)
    return df_pred, df_real, real_timestamp

# --- Sidebar ---
st.sidebar.title("Control Panel")
now = datetime.now()
st.sidebar.markdown(f"### Time\n## {now.strftime('%H:%M')}")
st.sidebar.caption(f"{now.strftime('%Y-%m-%d')}")

df_log = load_pred_log()

if not df_log.empty:
    st.sidebar.divider()
    st.sidebar.subheader("History")
    available_times = df_log['timestamp'].sort_values(ascending=False).tolist()
    selected_time = st.sidebar.selectbox("Select Time", available_times, format_func=lambda x: x.strftime('%m/%d %H:%M'))
    
    if selected_time:
        rec = df_log[df_log['timestamp'] == selected_time].iloc[0]
        st.sidebar.markdown("### Metrics")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("NDCG", f"{rec['model_ndcg']:.3f}")
        c2.metric("Lift", f"{rec['lift_percent']:+.1f}%", delta_color="normal" if rec['lift_percent']>0 else "inverse")
        st.sidebar.info(f"Ver: {rec['stage']}")
else:
    st.error("No Data")
    st.stop()

# --- Main Page ---
st.title("PTT Popularity Prediction System")

# 1. Chart (Optimized) - Full History & Vivid Colors
st.markdown("<div class='section-header'><h3>Performance Trend</h3></div>", unsafe_allow_html=True)

if not df_log.empty:
    # 定義鮮豔的色票給各個模型版本 (避開綠色，因為綠色留給 Baseline)
    # 包含：洋紅、青色、亮橘、黃色、紫色
    vivid_colors = ["#FF00FF", "#00FFFF", "#FFA500", "#FFD700", "#9D00FF"]

    # --- Step 1: 使用 Plotly Express 繪製模型線條 (根據 stage 分色) ---
    # 這會自動把 v1, v2, v3 畫成不同顏色的線
    fig = px.line(
        df_log, 
        x='timestamp', 
        y='model_ndcg', 
        color='stage', # 關鍵：根據版本自動分色
        color_discrete_sequence=vivid_colors, # 套用鮮豔色系
        title='NDCG Score Evolution'
    )

    # --- Step 2: 手動加入 Baseline (綠色虛線) ---
    fig.add_trace(go.Scatter(
        x=df_log['timestamp'], 
        y=df_log['base_ndcg'],
        mode='lines',
        name='Baseline',
        line=dict(
            color='#00FF00', # ★ 指定綠色 (Neon Green)
            width=2, 
            dash='dash'      # 虛線
        ),
        opacity=0.8,
        hovertemplate='Baseline: %{y:.4f}<extra></extra>'
    ))

    # --- Step 3: 版面與互動設定 ---
    fig.update_layout(
        template='plotly_dark', # 深色主題
        height=500,
        
        # 背景色整合
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        
        # X 軸：顯示全歷史 + 卷軸
        xaxis=dict(
            title='',
            rangeslider=dict(
                visible=False,        # 開啟卷軸
                bgcolor="#1E1E1E",   # 卷軸背景微調
                thickness=0.1        # 卷軸高度
            ),
            type="date",
            # 注意：這裡不再設定 range=...，預設即顯示全部歷史
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        
        # Y 軸：自動縮放 (解決直線問題)
        yaxis=dict(
            title='NDCG Score',
            autorange=True,      # ★ 自動抓最大最小值，突顯波動
            fixedrange=False,    # 允許使用者縱向縮放
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        
        # 圖例與邊距
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.02, 
            xanchor="right", x=1,
            title_text='' # 隱藏圖例標題 (stage)
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    # 修正線條寬度 (px.line 預設較細，加粗一點比較好看)
    fig.update_traces(patch={"line": {"width": 3}}, selector={"mode": "lines"})
    # 但把 Baseline (剛才加的最後一條) 改回細一點，避免喧賓奪主
    fig.data[-1].line.width = 2

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No historical data available to plot.")

# 2. Prediction vs Reality
st.markdown("<div class='section-header'><h3>Forecast vs Reality (T+120min)</h3></div>", unsafe_allow_html=True)

if selected_time:
    df_pred, df_real, real_ts = load_comparison_data(selected_time)
    
    if df_pred is not None and not df_pred.empty:
        accel = df_pred['push_acceleration'] if 'push_acceleration' in df_pred.columns else 0
        df_pred['score'] = df_pred['push_count'] * (1 + accel)
        top_pred = df_pred.sort_values('score', ascending=False).head(10).to_dict('records')
        
        if df_real is not None and not df_real.empty:
            top_real = df_real.sort_values('push_count', ascending=False).head(10).to_dict('records')
            real_txt = f"Actual Data at {real_ts.strftime('%H:%M')}"
            has_future = True
        else:
            top_real = []
            real_txt = "Data not available yet"
            has_future = False

        real_rank_map = {row['url']: idx+1 for idx, row in enumerate(top_real)}
        pred_rank_map = {row['url']: idx+1 for idx, row in enumerate(top_pred)}

        col_l, col_r = st.columns(2)
        
        with col_l:
            st.subheader("AI Prediction (T)")
            st.caption(f"Forecast at {selected_time.strftime('%H:%M')}")
            for i, row in enumerate(top_pred):
                url = row['url']
                is_hit = url in real_rank_map
                real_rank = real_rank_map.get(url, None)
                
                # Default Miss Style
                bg = "#f8f9fa" # Grey
                border = "#dee2e6" # Grey
                bw = "2px"
                move_html = "<span style='color:#999; font-size:12px;'>Prediction Failed</span>"
                
                # Hit Style
                if is_hit:
                    bg = "#ffffff" # White
                    border = "#198754" # Green
                    bw = "8px"
                    if (i+1) == real_rank: 
                        move_html = "<span class='rank-movement' style='color:#198754;'>Exact Match</span>"
                    else: 
                        move_html = f"<span class='rank-movement' style='color:#198754;'>Actual #{real_rank}</span>"
                
                acc = row.get('push_acceleration', 0.0)
                
                st.markdown(f"""
                <div class="metric-card" style="background-color: {bg}; border-left: {bw} solid {border};">
                    <div class="card-left">
                        <span class="rank-badge">#{i+1}</span>
                        <div class="article-info">
                            <a href="{url}" target="_blank" class="article-title" title="{row['title']}">{row['title']}</a>
                            {move_html}
                        </div>
                    </div>
                    <div class="card-right">
                        <div class="metric-row"><span class="metric-val">{row['push_count']}</span> <span class="metric-label">PUSH</span></div>
                        <div class="metric-row"><span class="metric-val">{acc:.2f}</span> <span class="metric-label">ACCEL</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_r:
            st.subheader("Reality (T+120m)")
            st.caption(real_txt)
            if has_future:
                for i, row in enumerate(top_real):
                    url = row['url']
                    is_hit = url in pred_rank_map
                    pred_rank = pred_rank_map.get(url, None)
                    
                    # Default Miss Style (Reality side)
                    bg = "#f8f9fa"
                    border = "#dee2e6"
                    bw = "2px"
                    move_html = "<span style='color:#999; font-size:12px;'>Unpredicted</span>"
                    
                    # Hit Style
                    if is_hit:
                        bg = "#ffffff"
                        border = "#198754"
                        bw = "8px"
                        move_html = f"<span class='rank-movement' style='color:#198754;'>AI Predicted #{pred_rank}</span>"
                        
                    st.markdown(f"""
                    <div class="metric-card" style="background-color: {bg}; border-left: {bw} solid {border};">
                        <div class="card-left">
                            <span class="rank-badge">#{i+1}</span>
                            <div class="article-info">
                                <a href="{url}" target="_blank" class="article-title" title="{row['title']}">{row['title']}</a>
                                {move_html}
                            </div>
                        </div>
                        <div class="card-right">
                            <div class="metric-row"><span class="metric-val">{row['push_count']}</span> <span class="metric-label">FINAL</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Waiting for data...")

st_autorefresh(interval=60 * 1000, key="clock_refresh")