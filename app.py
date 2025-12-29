import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-強勢族群回歸版", layout="wide")

# --- 2. 安全登入 ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except: pass

# --- 3. 核心功能：資料抓取 ---

@st.cache_data(ttl=600)
def get_current_price(sid):
    """抓取最新一筆收盤價"""
    try:
        df = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"))
        if not df.empty: return df['close'].iloc[-1]
    except: pass
    return 0.0

@st.cache_data(ttl=3600)
def get_hot_groups():
    """雷達：掃描昨日投信買超最強勢的前 10 名"""
    # 往回找最近一個交易日
    for i in range(1, 6):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=d, end_date=d)
            if not df.empty:
                # 依投信買超排序並取前 10
                hot = df.sort_values(by='SITC_Trust', ascending=False).head(10)
                hot = hot[hot['SITC_Trust'] > 0] # 只要有買超的
                return hot[['stock_id', 'stock_name', 'SITC_Trust']], d
        except: continue
    return pd.DataFrame(), ""

@st.cache_data(ttl=600)
def get_stock_all_info(sid):
    start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    try:
        rev = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        chip = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        if not chip.empty:
            chip['net_buy'] = chip['buy'] - chip['sell']
        return rev, chip
    except: return pd.DataFrame(), pd.DataFrame()

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：台股全方位戰情室")

# 側邊欄
st.sidebar.header("🎯 診斷與風控")
target_sid = st.sidebar.text_input("輸入股票代號", "2330")
real_price = get_current_price(target_sid) if login_ok else 0.0

st.sidebar.markdown("---")
my_buy = st.sidebar.number_input("您的買入成本", value=real_price if real_price != 0 else 1000.0)
curr_p = st.sidebar.number_input("目前市價 (自動帶入)", value=real_price)
sl_p = round(my_buy * 0.93, 2)
st.sidebar.metric("系統偵測現價", f"${real_price}", delta=f"{round(real_price-my_buy, 2)}")
st.sidebar.write(f"🛑 停損參考價: **{sl_p}**")

# 主畫面標籤頁
tab1, tab2, tab3, tab4 = st.tabs(["📈 個股籌碼", "📊 營收診斷", "🛡️ 風控雷達", "🔥 強勢族群雷達"])

if login_ok:
    rev_df, chip_df = get_stock_all_info(target_sid)
    
    with tab1:
        st.subheader(f"🔥 {target_sid} 法人買賣力道 (紅進綠出)")
        if not chip_df.empty:
            plot_df = chip_df[chip_df['name'].isin(['Foreign_Investor', 'Investment_Trust'])]
            fig_chip = px.bar(plot_df, x='date', y='net_buy', color='name', barmode='group',
                              color_discrete_map={'Foreign_Investor': '#EF553B', 'Investment_Trust': '#00CC96'})
            fig_chip.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig_chip, use_container_width=True)

    with tab2:
        st.subheader(f"📊 {target_sid} 營收趨勢")
        if not rev_df.empty:
            st.plotly_chart(px.bar(rev_df, x='revenue_month', y='revenue'), use_container_width=True)

    with tab3:
        st.subheader("🛡️ 持股風險分析")
        risk_df = pd.DataFrame({'項目':['成本','現價','停損線'], '價格':[my_buy, curr_p, sl_p]})
        st.plotly_chart(px.bar(risk_df, x='項目', y='價格', color='項目', text='價格'), use_container_width=True)

    with tab4:
        st.subheader("🔥 投信最新鎖碼強勢股 (Top 10)")
        with st.spinner('掃描全台股籌碼中...'):
            hot_df, hot_date = get_hot_groups()
            if not hot_df.empty:
                st.write(f"📅 資料日期：{hot_date}")
                hot_df.columns = ['代號', '名稱', '投信買超(張)']
                st.table(hot_df) # 用 Table 在手機上閱讀更直觀
                st.success("這 10 檔是目前投信最看好的標的！")
            else:
                st.warning("暫時無法抓取族群資料。")

else:
    st.error("API 尚未登入")
