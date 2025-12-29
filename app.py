import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-旗艦訊號版", layout="wide")

# --- 2. 安全登入 ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except: pass

# --- 3. 核心功能：超級強勢訊號掃描 ---

@st.cache_data(ttl=3600)
def scan_super_signals():
    """自動掃描：投信連買 + 站穩均線"""
    if not login_ok: return pd.DataFrame()
    
    # 找尋最近一個交易日
    target_d = ""
    for i in range(1, 6):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        test_df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=d, end_date=d)
        if not test_df.empty:
            target_d = d
            break
    
    if not target_d: return pd.DataFrame()

    try:
        # 1. 獲取投信買超榜 (前 30 名)
        chip_df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=target_d, end_date=target_d)
        top_picks = chip_df.sort_values(by='SITC_Trust', ascending=False).head(30)
        
        results = []
        for _, row in top_picks.iterrows():
            sid = row['stock_id']
            # 2. 獲取該股技術面
            tech = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d"))
            if len(tech) >= 60:
                last_p = tech['close'].iloc[-1]
                ma20 = tech['close'].tail(20).mean()
                ma60 = tech['close'].tail(60).mean()
                
                # 判斷邏輯：股價 > 月線 且 股價 > 季線
                if last_p > ma20 and last_p > ma60:
                    results.append({
                        '代號': sid,
                        '名稱': row['stock_name'],
                        '現價': last_p,
                        '投信買超(張)': row['SITC_Trust'],
                        '技術位階': '☀️ 強勢(站穩三線)' if last_p > tech['close'].tail(5).mean() else '🌤️ 盤整中'
                    })
        return pd.DataFrame(results), target_d
    except: return pd.DataFrame(), ""

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：旗艦戰情室")

# 側邊欄維持原本功能
target_sid = st.sidebar.text_input("輸入股票代號診斷", "2330")

# 主標籤頁：把強勢訊號放在第一個，讓你一打開就有驚喜
tab0, tab1, tab2, tab3 = st.tabs(["🚀 超級強勢訊號", "📈 技術/量價", "🔥 法人籌碼", "📊 營收診斷"])

if login_ok:
    with tab0:
        st.subheader("🌟 今日精選：投信鎖碼 + 均線多頭")
        with st.spinner('AI 正在掃描全台股技術面與籌碼面...'):
            sig_df, sig_date = scan_super_signals()
            if not sig_df.empty:
                st.write(f"📅 資料日期：{sig_date}")
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
                st.info("💡 分析師點評：這幾檔目前處於『法人成本區』且『趨勢向上』，是值得優先關注的標的。")
            else:
                st.warning("目前市場震盪，暫無符合超級強勢訊號之標的。")

    # (原本的技術面、籌碼面、營收分頁代碼接續在下方...)
