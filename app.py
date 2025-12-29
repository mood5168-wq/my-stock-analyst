import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-即時現價版", layout="wide")

# --- 2. 安全登入 ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except: pass

# --- 3. 核心功能：抓取資料 ---

@st.cache_data(ttl=300) # 現價每 5 分鐘更新一次
def get_current_price(sid):
    """抓取最新一筆收盤價"""
    try:
        # 抓取最近 3 天的資料確保一定有最新價格
        start_dt = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        df = dl.taiwan_stock_daily(stock_id=sid, start_date=start_dt)
        if not df.empty:
            return df['close'].iloc[-1] # 取得最後一筆收盤價
    except: pass
    return 0.0

@st.cache_data(ttl=600)
def get_stock_all_info(sid):
    start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    try:
        rev = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        chip = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        if not chip.empty:
            chip['net_buy'] = chip['buy'] - chip['sell']
        return rev, chip
    except: return pd.DataFrame(), pd.DataFrame()

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：台股全方位戰情室")

# 側邊欄
st.sidebar.header("🎯 診斷與風控")
target_sid = st.sidebar.text_input("輸入股票代號", "2330")

# --- 自動更新現價邏輯 ---
real_time_price = get_current_price(target_sid) if login_ok else 0.0

st.sidebar.markdown("---")
# 成本價讓你手動輸入
my_buy = st.sidebar.number_input("您的買入成本", value=real_time_price if real_time_price != 0 else 1000.0)
# 現價改為自動帶入，但保留手動微調空間
curr_p = st.sidebar.number_input("目前市價 (自動偵測)", value=real_time_price)

sl_price = round(my_buy * 0.93, 2)
st.sidebar.metric("系統偵測現價", f"${real_time_price}", delta=f"{round(real_time_price-my_buy, 2)} (盈虧)")
st.sidebar.write(f"🛑 停損參考價 (-7%): **{sl_price}**")

# 主畫面標籤頁
tab1, tab2, tab3 = st.tabs(["📈 大盤/個股籌碼", "📊 營收診斷", "🛡️ 風控雷達"])

if login_ok:
    rev_df, chip_df = get_stock_all_info(target_sid)
    
    with tab1:
        # 大盤走勢 (省略代碼以節省篇幅，邏輯同前)
        st.subheader(f"🔥 {target_sid} 法人淨買賣超")
        if not chip_df.empty:
            plot_df = chip_df[chip_df['name'].isin(['Foreign_Investor', 'Investment_Trust'])]
            fig_chip = px.bar(plot_df, x='date', y='net_buy', color='name', barmode='group')
            fig_chip.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig_chip, use_container_width=True)

    with tab2:
        st.subheader(f"📊 {target_sid} 營收趨勢")
        if not rev_df.empty:
            st.plotly_chart(px.bar(rev_df, x='revenue_month', y='revenue'), use_container_width=True)

    with tab3:
        st.subheader("🛡️ 風控位階")
        if curr_p <= sl_price:
            st.error(f"🚨 警報：現價 {curr_p} 已低於停損點 {sl_price}！")
        else:
            st.success(f"✅ 安全：現價離停損點還有 {round(curr_p - sl_price, 2)} 元")
