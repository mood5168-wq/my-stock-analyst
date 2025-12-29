import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-除錯版", layout="wide")

# --- 2. 除錯與 Token 讀取 ---
# 這是為了幫你檢查到底是哪個環節出錯
st.sidebar.subheader("🛠️ 系統狀態檢查")
if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 錯誤：Streamlit Secrets 找不到 'FINMIND_TOKEN' 這個鍵值。")
    st.info(f"目前系統偵測到的 Secrets 鍵值有：{list(st.secrets.keys())}")
    st.warning("請確保在 Settings > Secrets 裡寫的是 FINMIND_TOKEN = '你的Token'")
    st.stop()
else:
    st.sidebar.success("✅ 成功讀取 Secrets 設定")

# 登入 FinMind
try:
    token = st.secrets["FINMIND_TOKEN"]
    dl = DataLoader()
    dl.login(api_variant="token", token=token)
    st.sidebar.success("✅ FinMind API 登入成功")
except Exception as e:
    st.error(f"❌ FinMind 登入失敗：{e}")
    st.stop()

# --- 3. 資料抓取函式 ---
@st.cache_data(ttl=3600)
def get_market_data():
    date_str = datetime.now().strftime("%Y%m%d")
    url = f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={date_str}"
    try:
        res = requests.get(url)
        df = pd.DataFrame(res.json()['data'], columns=res.json()['fields'])
        df['收盤指數'] = df['收盤指數'].str.replace(',', '').astype(float)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_chip_data():
    # 抓取昨日籌碼
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=yesterday)
        if not df.empty:
            return df.sort_values(by='SITC_Trust', ascending=False).head(15)
        return pd.DataFrame()
    except: return pd.DataFrame()

# --- 4. 網頁呈現 ---
st.title("🏹 超級分析師：台股戰情室")

# 大盤診斷
st.subheader("📊 大盤走勢")
m_df = get_market_data()
if not m_df.empty:
    st.plotly_chart(px.line(m_df, x='日期', y='收盤指數'), use_container_width=True)

# 籌碼篩選
st.subheader("🔥 投信鎖碼榜 (全台股掃描)")
c_df = get_chip_data()
if not c_df.empty:
    st.dataframe(c_df[['stock_id', 'stock_name', 'SITC_Trust']], use_container_width=True)
else:
    st.info("尚未抓取到最新籌碼數據，請確認是否為開盤日。")
