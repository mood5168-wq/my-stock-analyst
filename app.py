import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-台股戰情室", layout="wide")

# --- 2. 核心診斷與 Token 處理 ---
st.sidebar.title("🛡️ 系統狀態")

# 初始化 Token 與 登入狀態
login_success = False
dl = None

# 自動偵測 Secrets 內容
if "FINMIND_TOKEN" in st.secrets:
    try:
        # 自動清理 Token (去除可能誤加入的引號或空白)
        raw_token = st.secrets["FINMIND_TOKEN"]
        clean_token = str(raw_token).strip().strip('"').strip("'")
        
        # 初始化 FinMind
        dl = DataLoader()
        dl.login(token=clean_token)
        login_success = True
        st.sidebar.success("✅ FinMind API 登入成功")
    except Exception as e:
        st.sidebar.error(f"❌ 登入失敗：{e}")
else:
    st.error("❌ 無法讀取 Secrets 中的 Token。")
    st.info("請檢查 Streamlit Cloud Settings -> Secrets，確保格式為：FINMIND_TOKEN = \"你的代碼\"")
    st.sidebar.warning("⚠️ 等待 Secrets 設定...")
    st.stop()

# --- 3. 資料抓取邏輯 (含防卡死機制) ---

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
        pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_chip_data():
    """抓取全台股投信鎖碼榜 (自動搜尋最近交易日)"""
    if not login_success:
        return pd.DataFrame()
    
    # 嘗試往回找 5 天，確保週末也能看到最後一個交易日的資料
    for i in range(1, 6):
        target_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=target_date)
            if not df.empty:
                # 篩選投信買超前 15 名
                top_sitc = df.sort_values(by='SITC_Trust', ascending=False).head(15)
                top_sitc = top_sitc.rename(columns={
                    'stock_id': '代號',
                    'stock_name': '名稱',
                    'SITC_Trust': '投信買超(張)'
                })
                return top_sitc[['代號', '名稱', '投信買超(張)']], target_date
        except:
            continue
    return pd.DataFrame(), None

# --- 4. 網頁介面開發 ---

st.title("🏹 超級分析師：台股戰情室")
st.caption(f"系統檢查時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 分欄佈局
tab1, tab2 = st.tabs(["📈 市場掃描", "🛡️ 持股診斷"])

with tab1:
    # A. 大盤走勢
    st.subheader("📊 大盤趨勢 (證交所即時數據)")
    m_df = get_market_data()
    if not m_df.empty:
        fig_m = px.line(m_df, x='日期', y='收盤指數', title="加權指數近日走勢")
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.warning("目前無法獲取大盤數據，請確認網路連線。")

    # B. 籌碼鎖碼榜
    st.markdown("---")
    chip_df, data_date = get_chip_data()
    st.subheader(f"🔥 投信鎖碼榜 (資料日期：{data_date if data_date else '搜尋中'})")
    if not chip_df.empty:
        st.dataframe(chip_df, use_container_width=True, hide_index=True)
        st.success(f"已成功載入全台股籌碼數據。")
    else:
        st.info("💡 正在搜尋最近的籌碼資料，請稍候...")

with tab2:
    st.subheader("🛡️ 個人持股風控分析")
    c1, c2, c3 = st.columns(3)
    with c1:
        my_buy = st.number_input("買入成本價", value=600.0)
    with c2:
        my_high = st.number_input("買入後最高價", value=650.0)
    with c3:
        my_curr = st.number_input("目前股價", value=630.0)
    
    # 計算風控價位
    sl = round(my_buy * 0.93, 2)
    ts = round(my_high * 0.90, 2)
    
    # 視覺化
    risk_df = pd.DataFrame({
        '項目': ['成本', '現價', '停損線(-7%)', '移動停利(-10%)'],
        '價格': [my_buy, my_curr, sl, ts]
    })
    fig_risk = px.bar(risk_df, x='項目', y='價格', color='項目', text='價格')
    st.plotly_chart(fig_risk, use_container_width=True)
    
    if my_curr <= sl:
        st.error(f"🚨 觸發停損！建議出場位：{sl}")
    elif my_curr <= ts:
        st.warning(f"⚠️ 觸發移動停利！建議出場位：{ts}")
    else:
        st.success("✅ 目前安全，請遵守紀律續抱。")

st.markdown("---")
st.caption("數據來源：台灣證券交易所、FinMind API。本程式僅供參考。")
