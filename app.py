import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-盤中爆量版", layout="wide")

# --- 2. 安全登入 ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except:
        if "FINMIND_TOKEN" in st.secrets:
            try:
                dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
                login_ok = True
            except: pass

# --- 3. 核心功能：盤中爆量掃描 ---

@st.cache_data(ttl=60) # 盤中每分鐘更新一次
def scan_intraday_breakout():
    """盤中掃描：今日量能異常 + 站上雙均線"""
    if not login_ok: return pd.DataFrame(), ""
    results = []
    
    # 取得今天日期
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # 1. 抓取今日目前成交量排行 (台股即時行情)
        # 註：此處以投信近日關注股為掃描池，確保 API 穩定不崩潰
        chip_df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=(datetime.now()-timedelta(days=3)).strftime("%Y-%m-%d"))
        if chip_df is not None and not chip_df.empty:
            top_list = chip_df.sort_values(by='SITC_Trust', ascending=False).head(30)
            
            for _, row in top_list.iterrows():
                sid = row['stock_id']
                # 抓取技術面
                t = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=100)).strftime("%Y-%m-%d"))
                if len(t) >= 60:
                    last = t.iloc[-1]
                    prev_5_avg_vol = t['Trading_Volume'].tail(6).head(5).mean()
                    curr_vol = last['Trading_Volume']
                    
                    # 計算爆量比例 (今日成交量 / 5日均量)
                    vol_ratio = round(curr_vol / prev_5_avg_vol, 2)
                    
                    # 均線判定
                    ma20 = t['close'].tail(20).mean()
                    ma60 = t['close'].tail(60).mean()
                    
                    # 條件：量增 1.5 倍以上 且 站穩 20MA & 60MA
                    if vol_ratio >= 1.5 and last['close'] > ma20 and last['close'] > ma60:
                        results.append({
                            '代號': sid,
                            '名稱': row['stock_name'],
                            '目前成交量': f"{int(curr_vol/1000)}k",
                            '量能倍數': f"🔥 {vol_ratio}x",
                            '現價': last['close'],
                            '狀態': "🚀 爆量突破" if last['close'] > t['close'].iloc[-2] else "⚖️ 高檔震盪"
                        })
            return pd.DataFrame(results), today
    except: return pd.DataFrame(), ""
    return pd.DataFrame(), ""

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：盤中爆量追蹤儀")

target_sid = st.sidebar.text_input("個股深度診斷 (代碼)", "2330")

# 盤中刷新按鈕
if st.sidebar.button('🔄 手動刷新盤中數據'):
    st.cache_data.clear()

tab0, tab1, tab2 = st.tabs(["⚡ 盤中爆量追蹤", "📉 技術扣抵解析", "🔥 籌碼/營收"])

if login_ok:
    with tab0:
        st.subheader("⚠️ 盤中即時警示：量能異常且站穩雙線")
        st.caption("自動監控投信關注股中，今日成交量已達 5 日均量 1.5 倍以上之標的")
        
        sig_df, sig_date = scan_intraday_breakout()
        if not sig_df.empty:
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
            st.success("💡 專業分析：盤中爆量通常代表大戶正在強力吃貨或換手，若股價維持在黃色月線之上，極具攻擊力。")
        else:
            st.info("目前盤中暫無符合『爆量且站上雙線』之標的。")

    # --- 個股深度資料 (Tab 1-2 維持之前最強大的扣抵與籌碼邏輯) ---
    # ... (此處接續之前的 get_all_data, MA20_Ref 等邏輯)
