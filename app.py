import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="超級分析師-台股戰情室", layout="wide")

# --- 2. 安全讀取 Token 並登入 ---
st.sidebar.subheader("🛠️ 系統狀態檢查")

try:
    # 從 Secrets 讀取並自動清理空白
    raw_token = st.secrets["FINMIND_TOKEN"]
    clean_token = raw_token.strip().strip('"').strip("'")
    
    dl = DataLoader()
    # 針對 FinMind 最新版本 1.x 的登入語法
    dl.login(token=clean_token)
    st.sidebar.success("✅ FinMind API 登入成功")
except Exception as e:
    st.sidebar.error("❌ 登入失敗")
    st.error(f"無法讀取 Secrets 中的 Token。請前往 Settings → Secrets 設定 FINMIND_TOKEN。")
    st.info("提示：格式應為 FINMIND_TOKEN = \"您的代碼\"")
    st.stop()

# --- 3. 核心資料抓取函式 ---

@st.cache_data(ttl=3600)
def get_market_data():
    """抓取加權指數近期資料 (證交所 API)"""
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
    # 考量假日與盤後更新，抓取最近 1-3 天的資料
    search_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        df_chip = dl.taiwan_stock_holding_shares_per(
            stock_id="ALL", 
            start_date=search_date
        )
        if not df_chip.empty:
            # 依投信買超張數排序
            top_sitc = df_chip.sort_values(by='SITC_Trust', ascending=False).head(15)
            top_sitc = top_sitc.rename(columns={
                'stock_id': '代號',
                'stock_name': '名稱',
                'SITC_Trust': '投信買超(張)'
            })
            return top_sitc[['代號', '名稱', '投信買超(張)']]
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- 4. 網頁介面佈局 ---
st.title("🏹 超級分析師：台股戰情室 (FinMind 實戰版)")
st.markdown(f"📅 系統時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- 側邊欄：風控診斷器 ---
st.sidebar.markdown("---")
st.sidebar.header("🛡️ 個人持股風控")
my_stock = st.sidebar.text_input("監控代號", "2330")
buy_p = st.sidebar.number_input("買入成本", value=1000.0)
high_p = st.sidebar.number_input("買入後最高價", value=1050.0)
curr_p = st.sidebar.number_input("目前市價", value=1030.0)

# 風控計算
sl = round(buy_p * 0.93, 2)
ts = round(high_p * 0.90, 2)

if curr_p <= sl:
    st.sidebar.error(f"🚨 停損出場：{sl}")
elif curr_p <= ts:
    st.sidebar.warning(f"⚠️ 移動停利：{ts}")
else:
    st.sidebar.success("✅ 目前安全")

# --- 主畫面顯示 ---

# A. 大盤走勢
st.subheader("📊 大盤趨勢 (證交所數據)")
m_df = get_market_data()
if not m_df.empty:
    fig_m = px.line(m_df, x='日期', y='收盤指數', title="加權指數近日走勢")
    st.plotly_chart(fig_m, use_container_width=True)

# B. 籌碼掃描
st.markdown("---")
st.subheader("🔥 投信鎖碼榜 (昨日法人買超前 15 名)")
chip_df = get_finmind_chip_data()
if not chip_df.empty:
    st.dataframe(chip_df, use_container_width=True, hide_index=True)
else:
    st.info("💡 尚未獲取最新籌碼數據，可能非交易日或資料處理中。")

# C. 風控視覺化
st.markdown("---")
st.subheader(f"📈 {my_stock} 風控位階圖")
risk_data = pd.DataFrame({
    '項目': ['成本', '現價', '停損線', '停利線'],
    '價格': [buy_p, curr_p, sl, ts]
})
fig_risk = px.bar(risk_data, x='項目', y='價格', color='項目', text='價格')
st.plotly_chart(fig_risk, use_container_width=True)

st.caption("免責聲明：本程式數據僅供參考，投資請務必遵守個人紀律。")
