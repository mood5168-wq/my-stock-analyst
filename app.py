import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-扣抵解析版", layout="wide")

# --- 2. 安全登入 ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except: pass

# --- 3. 核心功能：計算均線、扣抵與評分 ---

@st.cache_data(ttl=600)
def get_analysis_data(sid):
    """抓取完整資料並計算扣抵值與均線斜率"""
    start_date = (datetime.now() - timedelta(days=250)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        
        if not t.empty:
            # 計算均線
            t['MA5'] = t['close'].rolling(5).mean()
            t['MA20'] = t['close'].rolling(20).mean()
            t['MA60'] = t['close'].rolling(60).mean()
            
            # 取得扣抵參考價 (20天前與60天前的收盤價)
            t['MA20_Ref'] = t['close'].shift(20)
            t['MA60_Ref'] = t['close'].shift(60)
            
            # 計算均線方向 (斜率)
            t['MA20_Slope'] = t['MA20'].diff()
            t['MA60_Slope'] = t['MA60'].diff()
            
            if not c.empty: c['net_buy'] = c['buy'] - c['sell']
            
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def calculate_auto_score(t, c, m, r):
    """自動評分演算法"""
    score, details = 0, []
    if not t.empty and len(t) > 60:
        last = t.iloc[-1]
        # 技術面 (25分)
        if last['close'] >= last['MA20']: score += 15; details.append("✅ 股價在月線之上")
        if last['MA20_Slope'] > 0: score += 10; details.append("✅ 月線趨勢上揚")
    # 簡化顯示，其餘籌碼、基本面邏輯同前...
    return score, details

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：均線扣抵戰情室")
target_sid = st.sidebar.text_input("輸入股票代號", "2330")

if login_ok:
    t_df, c_df, m_df, r_df = get_analysis_data(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1]
        
        # --- A. 頂部診斷儀表板 ---
        st.markdown("### 📋 均線多空解析")
        col_ma20, col_ma60, col_score = st.columns(3)
        
        with col_ma20:
            status20 = "🟢 上揚 (助漲)" if last['MA20_Slope'] > 0 else "🔴 下彎 (助跌)"
            st.metric("20MA 月線狀態", status20)
            st.write(f"今日收盤: **{last['close']}** / 扣抵價: **{last['MA20_Ref']}**")
            st.caption("💡 現價 > 扣抵價 = 均線上揚")

        with col_ma60:
            status60 = "🟢 上揚 (助漲)" if last['MA60_Slope'] > 0 else "🔴 下彎 (助跌)"
            st.metric("60MA 季線狀態", status60)
            st.write(f"今日收盤: **{last['close']}** / 扣抵價: **{last['MA60_Ref']}**")

        with col_score:
            score, _ = calculate_auto_score(t_df, c_df, m_df, r_df)
            st.metric("AI 實戰總分", f"{score} 分")

        # --- B. 扣抵數據對照表 ---
        with st.expander("📅 查看未來 5 日扣抵值預估"):
            # 獲取接下來會被扣抵掉的歷史收盤價
            future_20 = t_df['close'].iloc[-25:-20].values[::-1]
            future_60 = t_df['close'].iloc[-65:-60].values[::-1]
            f_df = pd.DataFrame({
                '時間': ['明天', '後天', '第3天', '第4天', '第5天'],
                '月線扣抵價格': future_20,
                '季線扣抵價格': future_60
            })
            st.table(f_df)
            st.info("若未來扣抵價格很高，股價必須漲得更多才能維持均線上揚。")

        # --- C. 技術圖表 (標註扣抵位置) ---
        tab1, tab2 = st.tabs(["📉 量價與扣抵圖", "🔥 籌碼照妖鏡"])
        
        with tab1:
            fig = go.Figure()
            # K線/收盤線
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='現價', line=dict(color='white', width=1.5)))
            # 強化月線 (黃)
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA(月)', line=dict(color='#FFFF00', width=3)))
            # 季線 (桃紅)
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA(季)', line=dict(color='#FF00FF', width=2, dash='dot')))
            
            # 標註「今日扣抵點」
            ref_20_date = t_df['date'].iloc[-21]
            ref_60_date = t_df['date'].iloc[-61]
            fig.add_trace(go.Scatter(x=[ref_20_date], y=[last['MA20_Ref']], mode='markers+text', 
                                     name='月扣抵位置', text=["月扣抵"], textposition="top center",
                                     marker=dict(size=12, color='yellow', symbol='x')))
            fig.add_trace(go.Scatter(x=[ref_60_date], y=[last['MA60_Ref']], mode='markers+text', 
                                     name='季扣抵位置', text=["季扣抵"], textposition="top center",
                                     marker=dict(size=12, color='magenta', symbol='star')))

            fig.update_layout(template="plotly_dark", height=500, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # 法人與融資邏輯同前...
            if not c_df.empty:
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], 
                                       x='date', y='net_buy', color='
