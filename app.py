import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-進階戰情室", layout="wide")

# --- 2. 安全登入 ---
st.sidebar.title("🛡️ 系統設定與診斷")
login_success = False
dl = DataLoader()

try:
    if "FINMIND_USER_ID" in st.secrets and "FINMIND_PASSWORD" in st.secrets:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_success = True
    elif "FINMIND_TOKEN" in st.secrets:
        dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
        login_success = True
except:
    st.sidebar.error("❌ 登入失敗，請檢查 Secrets")

# --- 3. 功能開發：資料抓取 ---

@st.cache_data(ttl=3600)
def get_revenue_data(stock_id):
    """抓取特定個股營收趨勢"""
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    try:
        df = dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_advanced_chip_data(min_buy, filter_ma):
    """進階選股：投信買超張數 + 月線過濾"""
    if not login_success: return pd.DataFrame(), None
    
    for i in range(1, 6):
        target_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            # 1. 抓取投信資料
            df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=target_date, end_date=target_date)
            if df.empty: continue
            
            # 2. 基礎篩選：買超張數
            df = df[df['SITC_Trust'] >= min_buy]
            
            if filter_ma and not df.empty:
                # 這裡為了效能，我們只針對買超前 30 名進行股價過濾
                top_30 = df.sort_values(by='SITC_Trust', ascending=False).head(30)
                passed_list = []
                for _, row in top_30.iterrows():
                    # 抓取近一個月收盤價計算 MA20
                    price_df = dl.taiwan_stock_daily(
                        stock_id=row['stock_id'], 
                        start_date=(datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d")
                    )
                    if len(price_df) >= 20:
                        ma20 = price_df['close'].tail(20).mean()
                        curr_price = price_df['close'].iloc[-1]
                        if curr_price > ma20: # 股價在月線之上
                            passed_list.append(row)
                df = pd.DataFrame(passed_list)
            
            if not df.empty:
                df = df.sort_values(by='SITC_Trust', ascending=False)
                return df[['stock_id', 'stock_name', 'SITC_Trust']], target_date
        except: continue
    return pd.DataFrame(), None

# --- 4. 介面呈現 ---

# A. 側邊欄濾網設定
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 選股濾網設定")
min_buy_vol = st.sidebar.number_input("投信最少買超(張)", value=500, step=100)
ma_filter = st.sidebar.checkbox("僅顯示「站上月線(20MA)」個股", value=True)

# B. 主頁面：個股診斷區
st.title("🏹 超級分析師：進階戰情室")
with st.expander("🔍 特定股票：營收趨勢診斷", expanded=False):
    target_stock = st.text_input("輸入股票代號 (例: 2330)", "2330")
    rev_df = get_revenue_data(target_stock)
    if not rev_df.empty:
        fig_rev = px.bar(rev_df, x='revenue_month', y='revenue', 
                         title=f"{target_stock} 近兩年營收走勢",
                         labels={'revenue':'月營收(元)', 'revenue_month':'月份'})
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info("請輸入代號以查詢營收...")

# C. 主頁面：自訂選股區
st.markdown("---")
st.subheader(f"🔥 專業篩選：投信買超 > {min_buy_vol} 張 " + ("(已過濾月線以下)" if ma_filter else ""))

with st.spinner('🚀 正在依您的濾網條件掃描全台股...'):
    chip_df, d_date = get_advanced_chip_data(min_buy_vol, ma_filter)
    if not chip_df.empty:
        chip_df.columns = ['代號', '名稱', '投信買超(張)']
        st.dataframe(chip_df, use_container_width=True, hide_index=True)
        st.success(f"✅ 找到 {len(chip_df)} 檔符合條件的標的 (資料日期：{d_date})")
    else:
        st.warning("⚠️ 目前條件下無符合股票，建議調降買超張數門檻。")

# D. 側邊欄風控維持
st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 個人持股風控")
my_buy = st.sidebar.number_input("成本價", value=600.0)
st.sidebar.write(f"🛑 建議停損線: {round(my_buy * 0.93, 2)}")
