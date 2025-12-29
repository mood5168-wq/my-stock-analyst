import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-最終相容版", layout="wide")

# --- 2. 安全讀取 Token 並登入 (相容性優化) ---
st.sidebar.title("🛡️ 系統狀態")

login_success = False
dl = DataLoader()

if "FINMIND_TOKEN" in st.secrets:
    try:
        # 自動清理 Token 格式
        raw_token = st.secrets["FINMIND_TOKEN"]
        clean_token = str(raw_token).strip().strip('"').strip("'")
        
        # 嘗試第一種登入語法 (api_token)
        try:
            dl.login(api_token=clean_token)
        except TypeError:
            # 如果失敗，嘗試第二種語法 (token)
            dl.login(token=clean_token)
            
        login_success = True
        st.sidebar.success("✅ FinMind API 登入成功")
    except Exception as e:
        st.sidebar.error(f"❌ 登入失敗：{e}")
        st.stop()
else:
    st.error("❌ 無法讀取 Secrets 中的 Token。")
    st.info("請前往 Settings -> Secrets 設定 FINMIND_TOKEN = '您的代碼'")
    st.stop()

# --- 3. 資料抓取邏輯 ---

@st.cache_data(ttl=3600)
def get_market_data():
    """抓取加權指數近期資料"""
    try:
        date_str = datetime.now().strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={date_str}"
        res = requests.get(url, timeout=10)
        data = res.json()
        if 'data' in data:
            df = pd.DataFrame(data['data'], columns=data['fields'])
            df['收盤指數'] = df['收盤指數'].str.replace(',', '').astype(float)
            return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_chip_data():
    """抓取全台股投信鎖碼榜"""
    if not login_success: return pd.DataFrame(), None
    
    # 搜尋最近 5 天內有開盤的日期
    for i in range(1, 6):
        target_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=target_date)
            if not df.empty and 'SITC_Trust' in df.columns:
                top_sitc = df.sort_values(by='SITC_Trust', ascending=False).head(15)
                top_sitc = top_sitc.rename(columns={'stock_id':'代號','stock_name':'名稱','SITC_Trust':'投信買超(張)'})
                return top_sitc[['代號', '名稱', '投信買超(張)']], target_date
        except:
            continue
    return pd.DataFrame(), None

# --- 4. 網頁呈現 ---
st.title("🏹 超級分析師：台股戰情室")

# 大盤走勢
st.subheader("📊 大盤趨勢 (證交所數據)")
m_df = get_market_data()
if not m_df.empty:
    st.plotly_chart(px.line(m_df, x='日期', y='收盤指數'), use_container_width=True)

# 籌碼鎖碼榜
st.markdown("---")
chip_df, data_date = get_chip_data()
st.subheader(f"🔥 投信鎖碼榜 (資料日期：{data_date if data_date else '搜尋中'})")
if not chip_df.empty:
    st.dataframe(chip_df, use_container_width=True, hide_index=True)
else:
    st.info("💡 正在從 FinMind 伺服器同步籌碼數據...")

# 個人持股診斷
st.sidebar.markdown("---")
st.sidebar.header("🛡️ 持股監控")
my_buy = st.sidebar.number_input("成本價", value=600.0)
my_curr = st.sidebar.number_input("目前價", value=630.0)
sl = round(my_buy * 0.93, 2)
st.sidebar.write(f"建議停損點 (-7%): {sl}")
if my_curr <= sl:
    st.sidebar.error("🚨 建議出場")
