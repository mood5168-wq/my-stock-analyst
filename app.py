import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="超級分析師-FinMind實戰版", layout="wide")

# --- 2. 安全讀取 Token 並初始化 FinMind ---
# 這裡會從 Streamlit Cloud 的 Secrets 自動抓取，不會洩漏在程式碼中
try:
    FINMIND_TOKEN = st.secrets["eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0xMi0zMCAwMjoxMDoxOSIsInVzZXJfaWQiOiJtb29kNTE2OCIsImlwIjoiMjIwLjEzMi4xNzAuOTgifQ.RrqPHdFnPEFM_jHWfkvcSt4OjGEFsoTjoHcjJHot1xg"]
    dl = DataLoader()
    dl.login(api_variant="token", token=FINMIND_TOKEN)
except Exception as e:
    st.error("❌ 無法讀取 Secrets 中的 Token。請前往 Settings -> Secrets 設定 FINMIND_TOKEN。")
    st.stop() # 停止執行後續程式碼，避免報錯

# --- 3. 核心資料抓取函式 ---

@st.cache_data(ttl=3600)
def get_market_data():
    """抓取加權指數近期資料 (從證交所 API)"""
    # 取得今天日期
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
    """使用 FinMind 抓取昨日全台股投信鎖碼榜"""
    # 考慮盤後資料更新，抓取最近一個交易日
    search_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df_chip = dl.taiwan_stock_holding_shares_per(
            stock_id="ALL", 
            start_date=search_date
        )
        if not df_chip.empty:
            # 依投信買超張數(SITC_Trust)排序取前 15 名
            top_sitc = df_chip.sort_values(by='SITC_Trust', ascending=False).head(15)
            # 重新命名欄位讓表格更美觀
            top_sitc = top_sitc.rename(columns={
                'stock_id': '股票代號',
                'stock_name': '股票名稱',
                'SITC_Trust': '投信買超(張)'
            })
            return top_sitc[['股票代號', '股票名稱', '投信買超(張)']]
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- 4. 網頁介面佈局 ---
st.title("🏹 超級分析師：台股戰情室 (API 即時版)")
st.info(f"📅 目前系統時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 側邊欄：個人持股風控系統 ---
st.sidebar.header("🛡️ 個人持股風控診斷")
my_stock = st.sidebar.text_input("輸入監控代號 (例: 2330)", "2330")
my_buy_price = st.sidebar.number_input("買入成本價", value=1000.0)
my_high_price = st.sidebar.number_input("買入後最高價", value=1050.0)
my_curr_price = st.sidebar.number_input("當前市價", value=1030.0)

# 計算風控價位
stop_loss = round(my_buy_price * 0.93, 2) # -7% 停損
trailing_stop = round(my_high_price * 0.90, 2) # 高點回檔 10% 停利

st.sidebar.markdown("---")
st.sidebar.subheader("📢 紀律指令")
if my_curr_price <= stop_loss:
    st.sidebar.error(f"🚨 觸發停損！出場價位：{stop_loss}")
elif my_curr_price <= trailing_stop:
    st.sidebar.warning(f"⚠️ 觸發移動停利！出場價位：{trailing_stop}")
else:
    st.sidebar.success("✅ 目前位階安全，請續抱。")

# --- 主畫面區塊 ---

# A. 大盤走勢
st.subheader("📊 加權指數趨勢 (證交所來源)")
m_df = get_market_data()
if not m_df.empty:
    fig_m = px.line(m_df, x='日期', y='收盤指數', title="加權指數近日走勢圖")
    st.plotly_chart(fig_m, use_container_width=True)
else:
    st.warning("⚠️ 無法獲取大盤即時數據，可能為非交易時段。")

# B. 籌碼選股 (FinMind)
st.markdown("---")
st.subheader("🔥 投信鎖碼榜 (昨日法人買超前 15 名)")
chip_df = get_finmind_chip_data()
if not chip_df.empty:
    st.dataframe(chip_df, use_container_width=True, hide_index=True)
else:
    st.info("💡 尚未獲取最新籌碼數據，通常於盤後 16:00-18:00 更新。")

# C. 風控視覺化圖表
st.markdown("---")
st.subheader(f"📈 {my_stock} 風控位階分析")
risk_df = pd.DataFrame({
    '項目': ['買入成本', '當前市價', '停損底線', '移動停利點'],
    '價格': [my_buy_price, my_curr_price, stop_loss, trailing_stop]
})
fig_risk = px.bar(risk_df, x='項目', y='價格', color='項目', text='價格',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig_risk, use_container_width=True)

st.caption("🚨 免責聲明：本程式數據僅供參考，不構成任何投資建議。投資人需自負盈虧。")
