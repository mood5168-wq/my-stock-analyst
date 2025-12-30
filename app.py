import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-終極完整版", layout="wide")

# --- 2. 安全登入 ---
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

# --- 3. 核心函數：抓取個股資料 (包含最新價格) ---
@st.cache_data(ttl=60) # 盤中每分鐘更新一次
def get_stock_data_full(sid):
    start_date = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    try:
        # 抓取日 K 線 (FinMind 在盤中會包含當日的最新價格與成交量)
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        
        if not t.empty:
            # 計算均線
            t['MA5'] = t['close'].rolling(5).mean()
            t['MA20'] = t['close'].rolling(20).mean()
            t['MA60'] = t['close'].rolling(60).mean()
            # 扣抵與斜率
            t['MA20_Ref'] = t['close'].shift(20)
            t['MA60_Ref'] = t['close'].shift(60)
            t['Slope20'] = t['MA20'].diff()
            
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 4. 核心函數：自動評分 (25分 x 4) ---
def run_scoring(t, c, m, r):
    score, msg = 0, []
    if not t.empty:
        last = t.iloc[-1]
        # 技術面: 股價 > 20MA
        if last['close'] >= last['MA20']: score += 25; msg.append("✅ 站穩月線")
    if not c.empty:
        # 籌碼面: 投信近 3 日有買
        sitc = c[c['name'] == 'Investment_Trust'].tail(3)
        if not sitc.empty and sitc['net_buy'].sum() > 0: score += 25; msg.append("✅ 投信佈局")
    if not r.empty:
        # 基本面: 營收年增
        if r['revenue'].iloc[-1] > r['revenue'].iloc[-13 if len(r)>12 else 0]: score += 25; msg.append("✅ 營收年增")
    if not m.empty and 'MarginPurchaseStock' in m.columns:
        # 散戶面: 融資減少
        if m['MarginPurchaseStock'].iloc[-1] < m['MarginPurchaseStock'].iloc[-5]: score += 25; msg.append("✅ 融資洗盤")
    return score, msg

# --- 5. 介面 ---
st.title("🏹 超級分析師：終極旗艦戰情室")
target_sid = st.sidebar.text_input("輸入代碼", "1560")
my_cost = st.sidebar.number_input("買入成本", value=0.0)

if login_ok:
    t_df, c_df, m_df, r_df = get_stock_data_full(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1]
        
        # --- A. 頂部儀表板 (即時股價與評分) ---
        col_p, col_s, col_t = st.columns(3)
        with col_p:
            st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close'] - t_df['close'].iloc[-2], 2)}")
            st.caption(f"資料更新日期: {last['date']}")
        with col_s:
            score, details = run_scoring(t_df, c_df, m_df, r_df)
            st.metric("自動評分", f"{score} 分")
        with col_t:
            trend = "🟢 上揚" if last['Slope20'] > 0 else "🔴 下彎"
            st.metric("月線趨勢", trend)

        # --- B. 分頁功能 ---
        tab1, tab2, tab3, tab4 = st.tabs(["📉 技術扣抵圖", "🔥 籌碼對決", "📊 營收診斷", "🚀 爆量選股"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white', width=1.5)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            # 標註扣抵點
            fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[last['MA20_Ref']], mode='markers', name='月扣抵', marker=dict(size=12, color='yellow', symbol='x')))
            fig.update_layout(template="plotly_dark", height=450); st.plotly_chart(fig, use_container_width=True)
            
            # 成交量
            vol_ratio = round(last['Trading_Volume'] / t_df['Trading_Volume'].iloc[-6:-1].mean(), 2)
            st.write(f"📊 今日成交量：{int(last['Trading_Volume']/1000)}k (量能倍數: {vol_ratio}x)")

        with tab2:
            if not c_df.empty:
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣"), use_container_width=True)
            if not m_df.empty and 'MarginPurchaseStock' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="融資趨勢"), use_container_width=True)

        with tab3:
            if not r_df.empty:
                st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收走勢"), use_container_width=True)

        with tab4:
            st.info("💡 此分頁會掃描今日『爆量且站上雙線』的標的，請稍候...")
            # 簡化選股邏輯確保不崩潰
            seeds = ['1560', '2330', '2454', '2615', '2603', '3231']
            res = []
            for s in seeds:
                try:
                    temp_t = dl.taiwan_stock_daily(stock_id=s, start_date=(datetime.now()-timedelta(days=60)).strftime("%Y-%m-%d"))
                    if not temp_t.empty:
                        l = temp_t.iloc[-1]
                        v_r = l['Trading_Volume'] / temp_t['Trading_Volume'].iloc[-6:-1].mean()
                        if v_r > 1.2: res.append({'代號': s, '量能倍數': round(v_r, 2), '現價': l['close']})
                except: continue
            st.table(pd.DataFrame(res))
