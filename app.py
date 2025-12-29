import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-旗艦戰情室", layout="wide")

# --- 2. 安全登入 (支援帳密與 Token) ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except:
        if "FINMIND_TOKEN" in st.secrets:
            try:
                dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
                login_ok = True
            except: pass

# --- 3. 核心功能：數據抓取與扣抵計算 ---

@st.cache_data(ttl=600)
def get_full_analysis_data(sid):
    """一鍵抓取技術、籌碼、融資、營收全資料"""
    start_date = (datetime.now() - timedelta(days=250)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        
        if not t.empty:
            # 技術指標：5/20/60MA
            t['MA5'] = t['close'].rolling(5).mean()
            t['MA20'] = t['close'].rolling(20).mean()
            t['MA60'] = t['close'].rolling(60).mean()
            # 扣抵值計算
            t['MA20_Ref'] = t['close'].shift(20)
            t['MA60_Ref'] = t['close'].shift(60)
            # 趨勢斜率
            t['MA20_Slope'] = t['MA20'].diff()
            t['MA60_Slope'] = t['MA60'].diff()
            
        if not c.empty: 
            c['net_buy'] = c['buy'] - c['sell']
            
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：旗艦整合戰情室")

# 側邊欄診斷
st.sidebar.header("🎯 核心診斷")
target_sid = st.sidebar.text_input("輸入股票代號", "2330")
my_cost = st.sidebar.number_input("您的買入成本", value=0.0)

if login_ok:
    t_df, c_df, m_df, r_df = get_full_analysis_data(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1]
        
        # --- A. 均線扣抵儀表板 ---
        st.markdown("### 📋 趨勢與扣抵解析")
        col_ma20, col_ma60, col_price = st.columns(3)
        
        with col_ma20:
            s20 = "🟢 上揚 (助漲)" if last['MA20_Slope'] > 0 else "🔴 下彎 (助跌)"
            st.metric("20MA 月線", s20)
            st.caption(f"今日收盤 {last['close']} vs 扣抵 {last['MA20_Ref']}")

        with col_ma60:
            s60 = "🟢 上揚 (助漲)" if last['MA60_Slope'] > 0 else "🔴 下彎 (助跌)"
            st.metric("60MA 季線", s60)
            st.caption(f"今日收盤 {last['close']} vs 扣抵 {last['MA60_Ref']}")
            
        with col_price:
            st.metric("目前現價", f"${last['close']}")
            if my_cost > 0:
                sl = round(my_cost * 0.93, 2)
                st.write(f"🛑 停損點: **{sl}**")

        # --- B. 功能分頁 ---
        tab1, tab2, tab3, tab4 = st.tabs(["📉 技術扣抵圖", "🔥 籌碼對決", "📊 營收診斷", "📅 扣抵預測"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='現價', line=dict(color='white', width=1.5)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            
            # 標註扣抵位置
            ref_20_date = t_df['date'].iloc[-21] if len(t_df) > 21 else t_df['date'].iloc[0]
            fig.add_trace(go.Scatter(x=[ref_20_date], y=[last['MA20_Ref']], mode='markers', name='月扣抵點', marker=dict(size=12, color='yellow', symbol='x')))
            
            fig.update_layout(template="plotly_dark", height=450, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            # 成交量
            st.plotly_chart(px.bar(t_df, x='date', y='Trading_Volume', title="成交量", color_discrete_sequence=['#555555']), use_container_width=True, height=150)

        with tab2:
            st.subheader("🔥 大戶(法人) vs 散戶(融資)")
            if not c_df.empty:
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣差額"), use_container_width=True)
            if not m_df.empty and 'MarginPurchaseStock' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="散戶融資餘額"), use_container_width=True)

        with tab3:
            if not r_df.empty:
                st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收趨勢"), use_container_width=True)

        with tab4:
            st.subheader("未來 5 日扣抵預估")
            f_20 = t_df['close'].iloc[-25:-20].values[::-1]
            f_60 = t_df['close'].iloc[-65:-60].values[::-1]
            st.table(pd.DataFrame({'時間':['D+1','D+2','D+3','D+4','D+5'], '月扣抵價':f_20, '季扣抵價':f_60}))
            st.info("💡 只要現價大於扣抵價，均線就會上揚。")

else:
    st.error("API 登入失敗，請檢查 Secrets 設定。")
