import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-終極旗艦版", layout="wide")

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

# --- 3. 核心函數：自動評分系統 ---
def calculate_comprehensive_score(t, c, m, r):
    score = 0
    details = []
    if not t.empty and 'MA20' in t.columns:
        last = t.iloc[-1]
        # 技術面 (25分)
        if last['close'] >= last['MA20']:
            score += 15; details.append("✅ 站穩螢光黃月線 (+15)")
        if t['MA20'].diff().iloc[-1] > 0:
            score += 10; details.append("✅ 月線斜率向上 (+10)")
    # 籌碼面 (25分)
    if not c.empty and 'net_buy' in c.columns:
        sitc = c[c['name'] == 'Investment_Trust'].tail(3)
        if not sitc.empty and (sitc['net_buy'] > 0).all():
            score += 25; details.append("✅ 投信連 3 買鎖碼 (+25)")
    # 基本面 (25分)
    if not r.empty:
        if r['revenue'].iloc[-1] > r['revenue'].iloc[-13 if len(r)>12 else 0]:
            score += 25; details.append("✅ 營收年增成長 (+25)")
    # 散戶面 (25分)
    if not m.empty and 'MarginPurchaseStock' in m.columns:
        m_diff = m['MarginPurchaseStock'].iloc[-1] - m['MarginPurchaseStock'].iloc[-5]
        if m_diff < 0:
            score += 25; details.append("✅ 融資減少籌碼乾淨 (+25)")
    return score, details

# --- 4. 核心函數：盤中爆量 + 強勢選股 ---
@st.cache_data(ttl=60)
def scan_all_signals():
    if not login_ok: return pd.DataFrame()
    results = []
    # 擴大掃描池 (投信榜 + 盤中熱門種子)
    seeds = ['1560', '2330', '2454', '2615', '2317', '3231', '2382', '2603', '3037']
    try:
        chip = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=(datetime.now()-timedelta(days=3)).strftime("%Y-%m-%d"))
        top_list = list(set(chip.sort_values(by='SITC_Trust', ascending=False).head(30)['stock_id'].tolist() + seeds))
        for sid in top_list:
            try:
                t = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=100)).strftime("%Y-%m-%d"))
                if len(t) >= 60:
                    last = t.iloc[-1]
                    avg_v = t['Trading_Volume'].iloc[-6:-1].mean()
                    v_ratio = round(last['Trading_Volume'] / avg_v, 2)
                    ma20 = t['close'].tail(20).mean()
                    ma60 = t['close'].tail(60).mean()
                    # 選股條件：爆量 1.2x 且 站上雙線
                    if v_ratio >= 1.2 and last['close'] >= ma20 and last['close'] >= ma60:
                        results.append({'代號': sid, '量能倍數': v_ratio, '現價': last['close'], '雙線狀態': '☀️ 站穩'})
            except: continue
        return pd.DataFrame(results).sort_values(by='量能倍數', ascending=False)
    except: return pd.DataFrame()

# --- 5. 核心函數：個股全資料抓取 ---
@st.cache_data(ttl=300)
def get_stock_data(sid):
    start = (datetime.now() - timedelta(days=250)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start)
        if not t.empty:
            t['MA5'] = t['close'].rolling(5).mean()
            t['MA20'] = t['close'].rolling(20).mean()
            t['MA60'] = t['close'].rolling(60).mean()
            t['MA20_Ref'] = t['close'].shift(20)
            t['MA60_Ref'] = t['close'].shift(60)
            t['Slope'] = t['MA20'].diff()
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 6. UI 介面 ---
st.title("🏹 超級分析師：終極旗艦戰情室")
target_sid = st.sidebar.text_input("輸入股票代號", "1560")
my_cost = st.sidebar.number_input("買入成本", value=0.0)

tab0, tab1, tab2, tab3, tab4 = st.tabs(["⚡ 盤中爆量選股", "📈 技術扣抵圖", "🔥 籌碼照妖鏡", "📊 營收診斷", "📅 扣抵預測"])

if login_ok:
    t_df, c_df, m_df, r_df = get_stock_data(target_sid)
    
    with tab0:
        st.subheader("🚀 今日盤中爆量 + 站穩雙線名單")
        df_breakout = scan_all_signals()
        st.dataframe(df_breakout, use_container_width=True)

    with tab1:
        if not t_df.empty:
            # 自動評分顯示
            score, s_details = calculate_comprehensive_score(t_df, c_df, m_df, r_df)
            st.metric("🔥 實戰綜合評分", f"{score} 分")
            st.caption(" | ".join(s_details))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='現價', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA(月)', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA(季)', line=dict(color='#FF00FF', width=2, dash='dot')))
            
            # 扣抵點標註
            last = t_df.iloc[-1]
            fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[last['MA20_Ref']], mode='markers', name='月扣抵', marker=dict(size=12, color='yellow', symbol='x')))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not c_df.empty:
            st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人動向"), use_container_width=True)
        if not m_df.empty and 'MarginPurchaseStock' in m_df.columns:
            st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="散戶融資 (照妖鏡)"), use_container_width=True)

    with tab3:
        if not r_df.empty:
            st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收趨勢"), use_container_width=True)
            
    with tab4:
        st.subheader("📅 未來 5 日扣抵值預估")
        f_df = pd.DataFrame({'天數':['D+1','D+2','D+3','D+4','D+5'], '月扣抵價格':t_df['close'].iloc[-25:-20].values[::-1]})
        st.table(f_df)
