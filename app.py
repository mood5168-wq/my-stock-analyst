import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-最終修復版", layout="wide")

# --- 2. 安全讀取 Token 並登入 ---
st.sidebar.subheader("🛠️ 系統狀態檢查")

# 檢查 Secrets 是否存在
if "FINMIND_TOKEN" not in st.secrets:
    st.error("❌ 錯誤：Streamlit Secrets 找不到 'FINMIND_TOKEN'。")
    st.info(f"目前偵測到的鍵值：{list(st.secrets.keys())}")
    st.stop()

try:
    # 修正後的登入語法：直接傳入 token
    token = st.secrets["FINMIND_TOKEN"]
    dl = DataLoader()
    dl.login(token=token) # 移除 api_variant 參數
    st.sidebar.success("✅ FinMind API 登入成功")
except Exception as e:
    st.error(f"❌ 登入失敗。錯誤原因：{e}")
    st.info("提示：請確認您的 Token 格式是否正確（包含雙引號）。")
    st.stop()

# --- 3. 資料抓取函式 ---
@st.cache_data(ttl=3600)
def get_market_data():
    """抓取大盤走勢"""
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
    """抓取全台股投信鎖碼榜"""
    # 抓取最近一個交易日資料
    search_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=search_date)
        if not df.empty:
            # 依投信買超張數排序
            top_sitc = df.sort_values(by='SITC_Trust', ascending=False).head(15)
            top_sitc = top_sitc.rename(columns={'stock_id':'代號', 'stock_name':'名稱', 'SITC_Trust':'投信買超(張)'})
            return top_sitc[['代號', '名稱', '投信買超(張)']]
        return pd.DataFrame()
    except: return pd.DataFrame()

# --- 4. 網頁呈現 ---
st.title("🏹 超級分析師：台股全方位戰情室")

# 第一區塊：大盤診斷
st.subheader("📊 大盤趨勢診斷")
m_df = get_market_data()
if not m_df.empty:
    fig_m = px.line(m_df, x='日期', y='收盤指數', title="加權指數走勢")
    st.plotly_chart(fig_m, use_container_width=True)
else:
    st.warning("⚠️ 暫時無法獲取大盤數據。")

# 第二區塊：籌碼掃描
st.markdown("---")
st.subheader("🔥 投信鎖碼榜 (昨日法人買超前 15 名)")
c_df = get_chip_data()
if not c_df.empty:
    st.dataframe(c_df, use_container_width=True, hide_index=True)
else:
    st.info("💡 尚未獲取最新籌碼，通常盤後 16:00 更新。")

# 第三區塊：個人持股風控 (簡易版)
st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 持股監控")
buy_p = st.sidebar.number_input("成本價", value=600.0)
curr_p = st.sidebar.number_input("現價", value=610.0)
st.sidebar.write(f"停損價參考 (-7%): {round(buy_p * 0.93, 2)}")
