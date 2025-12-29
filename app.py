import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-全版本相容版", layout="wide")

# --- 2. 安全讀取 Token 並登入 (全版本相容語法) ---
st.sidebar.title("🛡️ 系統狀態")

login_success = False
dl = DataLoader()

if "FINMIND_TOKEN" in st.secrets:
    try:
        # 自動清理 Token 格式
        raw_token = st.secrets["FINMIND_TOKEN"]
        clean_token = str(raw_token).strip().strip('"').strip("'")
        
        # --- 全版本相容登入邏輯 ---
        try:
            # 嘗試 1: 最新的 api_token 參數
            dl.login(api_token=clean_token)
        except TypeError:
            try:
                # 嘗試 2: 部分版本的 token 參數
                dl.login(token=clean_token)
            except TypeError:
                # 嘗試 3: 舊版的直接傳入 (無參數名稱)
                dl.login(clean_token)
            
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
    """抓取加權指數近期資料 (證交所 API)"""
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
    
    # 搜尋最近 5 天內有開盤的日期 (解決週末/連假問題)
    for i in range(1, 6):
        target_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            # 獲取三大法人買賣超資料
            df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=target_date)
            if not df.empty and 'SITC_Trust' in df.columns:
                # 篩選投信買超張數前 15 名
                top_sitc = df.sort_values(by='SITC_Trust', ascending=False).head(15)
                top_sitc = top_sitc.rename(columns={'stock_id':'代號','stock_name':'名稱','SITC_Trust':'投信買超(張)'})
                return top_sitc[['代號', '名稱', '投信買超(張)']], target_date
        except:
            continue
    return pd.DataFrame(), None

# --- 4. 網頁呈現 ---
st.title("🏹 超級分析師：台股戰情室")
st.caption(f"最後檢查時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 分欄
col_main, col_side = st.columns([3, 1])

with col_main:
    # 大盤走勢
    st.subheader("📊 大盤趨勢 (加權指數)")
    m_df = get_market_data()
    if not m_df.empty:
        fig_m = px.line(m_df, x='日期', y='收盤指數', template="plotly_dark")
        st.plotly_chart(fig_m, use_container_width=True)

    # 籌碼鎖碼榜
    st.markdown("---")
    chip_df, data_date = get_chip_data()
    st.subheader(f"🔥 投信鎖碼榜 (日期：{data_date if data_date else '搜尋中'})")
    if not chip_df.empty:
        st.dataframe(chip_df, use_container_width=True, hide_index=True)
    else:
        st.info("💡 正在從 FinMind 同步籌碼大數據...")

with col_side:
    st.subheader("🛡️ 持股風險監控")
    my_buy = st.number_input("您的買入成本", value=600.0)
    my_curr = st.number_input("當前市價", value=615.0)
    
    sl_price = round(my_buy * 0.93, 2)
    tp_price = round(my_buy * 1.10, 2)
    
    st.metric("當前盈虧", f"{round((my_curr-my_buy)/my_buy*100, 2)}%")
    st.write(f"🛑 停損參考價 (-7%): **{sl_price}**")
    st.write(f"🎯 目標挑戰 (+10%): **{tp_price}**")
    
    if my_curr <= sl_price:
        st.error("🚨 已觸發停損，請執行紀律！")
    elif my_curr >= tp_price:
        st.success("💰 已達到初始目標，考慮分批停利。")
    else:
        st.info("✅ 股價尚在安全區間。")

st.markdown("---")
st.caption("警語：數據僅供參考，不構成投資建議。")
