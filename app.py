import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-全能終極版", layout="wide")

# --- 2. 安全登入 ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except: pass

# --- 3. 資料抓取函數 (含快取機制) ---

@st.cache_data(ttl=3600)
def get_market_data():
    """抓取大盤資料"""
    try:
        url = f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={datetime.now().strftime('%Y%m%d')}"
        res = requests.get(url, timeout=5)
        df = pd.DataFrame(res.json()['data'], columns=res.json()['fields'])
        df['收盤指數'] = df['收盤指數'].str.replace(',', '').astype(float)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def get_stock_all_info(sid):
    """一次抓取營收與法人淨買賣超"""
    start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    try:
        rev = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        chip = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        if not chip.empty:
            chip['net_buy'] = chip['buy'] - chip['sell']
        return rev, chip
    except: return pd.DataFrame(), pd.DataFrame()

# --- 4. 網頁介面佈局 ---
st.title("🏹 超級分析師：台股全方位戰情室")

# 側邊欄：功能選單與風控
st.sidebar.header("🎯 診斷與風控")
target_sid = st.sidebar.text_input("輸入股票代號", "2330")
st.sidebar.markdown("---")
my_buy = st.sidebar.number_input("您的買入成本", value=600.0)
curr_p = st.sidebar.number_input("當前市價", value=620.0)
sl_price = round(my_buy * 0.93, 2)
st.sidebar.metric("停損參考價 (-7%)", sl_price, delta=round(curr_p - sl_price, 2))

# 主畫面標籤頁
tab1, tab2, tab3 = st.tabs(["📈 大盤/個股籌碼", "📊 營收診斷", "🛡️ 風控雷達"])

if login_ok:
    with tab1:
        # A. 大盤走勢
        m_df = get_market_data()
        if not m_df.empty:
            st.plotly_chart(px.line(m_df, x='日期', y='收盤指數', title="加權指數走勢"), use_container_width=True)
        
        # B. 個股籌碼力道
        st.markdown(f"### 🔥 {target_sid} 法人淨買賣超 (紅進綠出)")
        rev_df, chip_df = get_stock_all_info(target_sid)
        if not chip_df.empty:
            plot_df = chip_df[chip_df['name'].isin(['Foreign_Investor', 'Investment_Trust'])]
            fig_chip = px.bar(plot_df, x='date', y='net_buy', color='name',
                              barmode='group', color_discrete_map={'Foreign_Investor': '#EF553B', 'Investment_Trust': '#00CC96'})
            fig_chip.add_hline(y=0, line_dash="dash", line_color="white")
            st.plotly_chart(fig_chip, use_container_width=True)
        else:
            st.warning("暫時抓不到個股籌碼...")

    with tab2:
        st.subheader(f"📊 {target_sid} 營收成長趨勢")
        if not rev_df.empty:
            st.plotly_chart(px.bar(rev_df, x='revenue_month', y='revenue', title="月營收走勢"), use_container_width=True)
        else:
            st.info("請確認代號後查看營收數據。")

    with tab3:
        st.subheader("🛡️ 持股風險位階")
        risk_data = pd.DataFrame({
            '項目': ['成本', '現價', '停損線'],
            '價格': [my_buy, curr_p, sl_price]
        })
        st.plotly_chart(px.bar(risk_data, x='項目', y='價格', color='項目', text='價格'), use_container_width=True)
        
        if curr_p <= sl_price:
            st.error(f"🚨 警告：目前股價已低於停損線 {sl_price}，請嚴格執行紀律！")
        else:
            st.success("✅ 目前股價仍位處安全區間。")
else:
    st.error("API 登入失敗，請確認 Secrets。")
