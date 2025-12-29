import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-穩健版", layout="wide")

# --- 2. 安全登入 ---
st.sidebar.title("🛡️ 系統狀態")
login_success = False
dl = DataLoader()

if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_success = True
        st.sidebar.success("✅ 帳密登入成功")
    except:
        if "FINMIND_TOKEN" in st.secrets:
            try:
                dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
                login_success = True
                st.sidebar.success("✅ Token 登入成功")
            except: st.sidebar.error("❌ 登入失敗")

# --- 3. 核心功能：個股營收與籌碼 ---

@st.cache_data(ttl=3600)
def get_revenue_data(stock_id):
    if not login_success: return pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    try:
        return dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_guaranteed_chip_data(min_buy):
    """保證有資料的抓取邏輯"""
    if not login_success: return pd.DataFrame(), None
    
    # 往回找最近的交易日
    for i in range(1, 7):
        target_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            # 關鍵修改：先抓取當天所有籌碼資料，不做 stock_id 篩選以加快速度
            df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=target_date, end_date=target_date)
            
            if not df.empty and 'SITC_Trust' in df.columns:
                # 排除買超為 0 的股票
                df = df[df['SITC_Trust'] > 0]
                
                # 套用使用者設定的濾網
                filtered = df[df['SITC_Trust'] >= min_buy]
                
                # 如果濾完是空的，就直接給前 15 名 (保底)
                if filtered.empty:
                    st.sidebar.warning(f"{target_date} 無達標股票，已顯示當日買超榜")
                    return df.sort_values(by='SITC_Trust', ascending=False).head(15), target_date
                
                return filtered.sort_values(by='SITC_Trust', ascending=False), target_date
        except:
            continue
    return pd.DataFrame(), None

# --- 4. 介面呈現 ---
st.title("🏹 超級分析師：台股戰情室")

# 第一區塊：個股診斷
with st.expander("🔍 特定股票營收診斷", expanded=True):
    tid = st.text_input("輸入股票代號", "2330")
    r_df = get_revenue_data(tid)
    if not r_df.empty:
        st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title=f"{tid} 營收走勢"), use_container_width=True)

# 第二區塊：籌碼選股
st.markdown("---")
buy_threshold = st.sidebar.slider("投信買超門檻 (張)", 0, 1000, 100)

with st.spinner('正在分析大數據...'):
    c_df, d_date = get_guaranteed_chip_data(buy_threshold)
    if not c_df.empty:
        st.subheader(f"🔥 投信鎖碼名單 ({d_date})")
        display_df = c_df[['stock_id', 'stock_name', 'SITC_Trust']].copy()
        display_df.columns = ['代號', '名稱', '買超(張)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.error("暫時抓不到籌碼資料，請稍後再試。")

# 側邊欄風控
st.sidebar.markdown("---")
cost = st.sidebar.number_input("持股成本", value=100.0)
st.sidebar.write(f"🛑 停損點 (-7%): {round(cost*0.93, 2)}")
