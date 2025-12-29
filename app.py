import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-個股全能版", layout="wide")

# --- 2. 安全登入 (相容模式) ---
dl = DataLoader()
login_ok = False
try:
    if "FINMIND_USER_ID" in st.secrets:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    elif "FINMIND_TOKEN" in st.secrets:
        dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
        login_ok = True
except: st.error("API 登入失敗")

# --- 3. 核心功能：個股深度診斷 ---

@st.cache_data(ttl=600)
def get_stock_details(sid):
    """一次抓取營收與法人買賣超"""
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    try:
        # 抓營收
        rev = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        # 抓三大法人買賣超 (這比掃描全台股快非常多)
        chip = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        return rev, chip
    except: return pd.DataFrame(), pd.DataFrame()

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：個股深度戰情室")

# 側邊欄：快速選單
st.sidebar.header("🎯 診斷目標")
target_sid = st.sidebar.text_input("輸入股票代號", "2330")

if login_ok:
    with st.spinner('正在分析該股籌碼與基本面...'):
        rev_df, chip_df = get_stock_details(target_sid)
        
        # A. 籌碼面：法人買賣超 (最有意思的地方！)
        st.subheader(f"🔥 {target_sid} 法人買賣超監控 (近半年)")
        if not chip_df.empty:
            # 整理資料，只看外資與投信
            chip_plot = chip_df[chip_df['name'].isin(['Foreign_Investor', 'Investment_Trust'])]
            fig_chip = px.bar(chip_plot, x='date', y='buy', color='name', 
                              title="外資與投信買賣力道", barmode='group')
            st.plotly_chart(fig_chip, use_container_width=True)
            
            # 計算最近三天的合計
            latest_chip = chip_df.tail(6) # 兩類法人 x 3天
            st.info(f"💡 筆記：觀察最近法人是否有「連買」現象，通常是起漲訊號！")
        else:
            st.warning("暫時無法取得該股籌碼資料")

        # B. 基本面：營收趨勢
        st.markdown("---")
        st.subheader(f"📊 {target_sid} 營收成長追蹤")
        if not rev_df.empty:
            fig_rev = px.line(rev_df, x='revenue_month', y='revenue', markers=True, title="月營收走勢")
            st.plotly_chart(fig_rev, use_container_width=True)
        
else:
    st.warning("請先設定 API 登入資訊")

# 風控提示維持
st.sidebar.markdown("---")
cost = st.sidebar.number_input("持股成本", value=100.0)
st.sidebar.metric("停損線 (-7%)", f"{round(cost*0.93, 2)}")
