import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-全能登入版", layout="wide")

# --- 2. 安全登入邏輯 (Token + 帳密 雙相容) ---
st.sidebar.title("🛡️ 系統狀態")

login_success = False
dl = DataLoader()

try:
    # 嘗試方式 A: 使用 Token 登入
    if "FINMIND_TOKEN" in st.secrets:
        token = st.secrets["FINMIND_TOKEN"].strip().strip('"')
        try:
            dl.login(token=token)
            login_success = True
        except:
            pass # 失敗則嘗試下一種
            
    # 嘗試方式 B: 如果方式 A 失敗，使用帳號密碼登入
    if not login_success and "FINMIND_USER_ID" in st.secrets:
        user_id = st.secrets["FINMIND_USER_ID"]
        password = st.secrets["FINMIND_PASSWORD"]
        dl.login(user_id=user_id, password=password)
        login_success = True
        
    if login_success:
        st.sidebar.success("✅ FinMind API 登入成功")
    else:
        st.error("❌ 登入失敗：請檢查 Secrets 中的帳號密碼或 Token。")
        st.stop()
except Exception as e:
    st.sidebar.error(f"❌ 系統錯誤：{e}")
    st.stop()

# --- 3. 資料抓取與顯示 (其餘邏輯保持不變) ---

@st.cache_data(ttl=3600)
def get_market_data():
    try:
        url = f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={datetime.now().strftime('%Y%m%d')}"
        res = requests.get(url, timeout=10)
        df = pd.DataFrame(res.json()['data'], columns=res.json()['fields'])
        df['收盤指數'] = df['收盤指數'].str.replace(',', '').astype(float)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_chip_data():
    for i in range(1, 6):
        target_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=target_date)
            if not df.empty:
                top_sitc = df.sort_values(by='SITC_Trust', ascending=False).head(15)
                return top_sitc[['stock_id', 'stock_name', 'SITC_Trust']], target_date
        except: continue
    return pd.DataFrame(), None

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：台股戰情室")
m_df = get_market_data()
if not m_df.empty:
    st.plotly_chart(px.line(m_df, x='日期', y='收盤指數', title="大盤即時走勢"), use_container_width=True)

st.markdown("---")
chip_df, d_date = get_chip_data()
if not chip_df.empty:
    st.subheader(f"🔥 投信鎖碼榜 ({d_date})")
    st.dataframe(chip_df, use_container_width=True)
else:
    st.info("💡 正在同步最新籌碼資料...")
