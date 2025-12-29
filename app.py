import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="超級分析師-FinMind實戰版", layout="wide")

# --- 2. 初始化 FinMind (請填入你的 Token) ---
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0xMi0zMCAwMjoxMDoxOSIsInVzZXJfaWQiOiJtb29kNTE2OCIsImlwIjoiMjIwLjEzMi4xNzAuOTgifQ.RrqPHdFnPEFM_jHWfkvcSt4OjGEFsoTjoHcjJHot1xg" 
dl = DataLoader()
try:
    dl.login(api_variant="token", token=FINMIND_TOKEN)
except:
    st.error("FinMind Token 登入失敗，請檢查 Token 是否正確。")

# --- 3. 核心資料抓取函式 ---

@st.cache_data(ttl=3600)
def get_market_data():
    """抓取大盤走勢 (證交所 API)"""
    date_str = datetime.now().strftime("%Y%m%d")
    url = f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={date_str}"
    try:
        res = requests.get(url)
        data = res.json()
        df = pd.DataFrame(data['data'], columns=data['fields'])
        df['收盤指數'] = df['收盤指數'].str.replace(',', '').astype(float)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_finmind_chip_data():
    """使用 FinMind 抓取全台股投信鎖碼榜"""
    # 抓取昨日日期 (API 通常盤後更新)
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        # 抓取三大法人買賣超
        df_chip = dl.taiwan_stock_holding_shares_per(
            stock_id="ALL", 
            start_date=yesterday
        )
        if not df_chip.empty:
            # 篩選投信 (SITC) 買超前 15 名
            top_sitc = df_chip.sort_values(by='SITC_Trust', ascending=False).head(15)
            return top_sitc[['stock_id', 'stock_name', 'SITC_Trust']]
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- 4. 網頁介面佈局 ---
st.title("🏹 超級分析師：FinMind 全台股自動戰情室")
st.markdown(f"系統檢查時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- 側邊欄：個人持股風控診斷 ---
st.sidebar.header("🛡️ 個人持股風控")
my_stock = st.sidebar.text_input("股票名稱/代號", "2330 台積電")
my_buy_price = st.sidebar.number_input("買入成本", value=1000.0)
my_high_price = st.sidebar.number_input("買入後最高價", value=1050.0)
my_curr_price = st.sidebar.number_input("當前股價", value=1030.0)

# 風控計算
stop_loss = round(my_buy_price * 0.93, 2)
trailing_stop = round(my_high_price * 0.90, 2)

st.sidebar.markdown("---")
if my_curr_price <= stop_loss:
    st.sidebar.error(f"🚨 停損警報！建議價：{stop_loss}")
elif my_curr_price <= trailing_stop:
    st.sidebar.warning(f"⚠️ 獲利回落！移動停利點：{trailing_stop}")
else:
    st.sidebar.success("✅ 持股狀態正常")

# --- 主畫面區塊 ---

# A. 大盤診斷
st.subheader("📊 大盤趨勢 (證交所即時數據)")
m_df = get_market_data()
if not m_df.empty:
    fig_m = px.line(m_df, x='日期', y='收盤指數', title="加權指數走勢圖")
    st.plotly_chart(fig_m, use_container_width=True)

# B. FinMind 全台股掃描
st.markdown("---")
st.subheader("🔥 投信鎖碼榜 (FinMind 籌碼大數據)")
chip_df = get_finmind_chip_data()
if not chip_df.empty:
    st.write("以下為昨日投信買超張數前 15 名，代表法人資金流向：")
    st.dataframe(chip_df, use_container_width=True)
else:
    st.info("尚未抓取到今日籌碼資料，請確認 API Token 或收盤時間。")

# C. 風控視覺化
st.markdown("---")
st.subheader(f"📈 {my_stock} 持股位階圖")
risk_df = pd.DataFrame({
    '項目': ['成本', '現價', '停損線', '停利線'],
    '價格': [my_buy_price, my_curr_price, stop_loss, trailing_stop]
})
fig_risk = px.bar(risk_df, x='項目', y='價格', color='項目', text='價格')
st.plotly_chart(fig_risk, use_container_width=True)

st.caption("數據來源：FinMind API & TWSE 官網。請遵守操作紀律，投資盈虧自負。")
