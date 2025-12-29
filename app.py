import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-籌碼力道版", layout="wide")

# --- 2. 安全登入 ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except: pass

# --- 3. 核心功能：抓取淨買賣超 ---

@st.cache_data(ttl=600)
def get_stock_chip_trend(sid):
    """抓取法人買賣超，並計算淨額"""
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    try:
        df = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        if not df.empty:
            # 計算淨買賣超：買進張數 - 賣出張數
            df['net_buy'] = df['buy'] - df['sell']
            return df
    except: pass
    return pd.DataFrame()

# --- 4. 介面呈現 ---
st.title("🏹 超級分析師：法人力道診斷")

target_sid = st.sidebar.text_input("輸入股票代號", "2330")

if login_ok:
    with st.spinner('正在分析買賣力道...'):
        chip_df = get_stock_chip_trend(target_sid)
        
        if not chip_df.empty:
            st.subheader(f"🔥 {target_sid} 法人淨買賣超 (紅進綠出)")
            
            # 過濾外資與投信
            plot_df = chip_df[chip_df['name'].isin(['Foreign_Investor', 'Investment_Trust'])]
            
            # 建立圖表：y 軸改用 net_buy
            fig = px.bar(plot_df, x='date', y='net_buy', color='name',
                         title="向上代表法人買超，向下代表法人賣超",
                         barmode='group',
                         color_discrete_map={'Foreign_Investor': '#EF553B', 'Investment_Trust': '#00CC96'})
            
            # 加入一條零軸橫線，方便看正負
            fig.add_hline(y=0, line_dash="dash", line_color="white")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示最近五天的詳細數據表格
            st.markdown("### 📋 最近 5 日數據清單")
            recent_df = plot_df.tail(10).sort_values(by='date', ascending=False)
            recent_df = recent_df[['date', 'name', 'buy', 'sell', 'net_buy']]
            recent_df.columns = ['日期', '法人名稱', '買進', '賣出', '淨買賣超']
            st.table(recent_df)
        else:
            st.warning("查無此標的籌碼資料，請檢查代號是否正確。")
else:
    st.error("API 尚未連線成功")
