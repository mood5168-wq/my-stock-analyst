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

# --- 3. 核心運算：盤中量能估算因子 ---
def get_v_factor():
    """計算盤中時間權重，用於預估全天量能"""
    now = datetime.now()
    if now.hour < 9: return 0.1
    if now.hour >= 14: return 1.0
    total_min = (now.hour - 9) * 60 + now.minute
    return max(total_min / 270, 0.1)

# --- 4. 數據抓取與邏輯計算 ---
@st.cache_data(ttl=60) # 盤中每分鐘更新一次
def get_stock_data_full(sid):
    start_date = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        
        if not t.empty:
            t['MA5'] = t['close'].rolling(5).mean()
            t['MA20'] = t['close'].rolling(20).mean()
            t['MA60'] = t['close'].rolling(60).mean()
            t['MA20_Ref'] = t['close'].shift(20)
            t['MA60_Ref'] = t['close'].shift(60)
            t['Slope20'] = t['MA20'].diff()
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def run_scoring(t, c, m, r):
    """自動評分系統 (25/25/25/25)"""
    score, msg = 0, []
    if not t.empty:
        last = t.iloc[-1]
        if last['close'] >= last['MA20']: score += 25; msg.append("✅ 站穩月線")
    if not c.empty:
        sitc = c[c['name'] == 'Investment_Trust'].tail(3)
        if not sitc.empty and sitc['net_buy'].sum() > 0: score += 25; msg.append("✅ 投信佈局")
    if not r.empty:
        if r['revenue'].iloc[-1] > r['revenue'].iloc[-13 if len(r)>12 else 0]: score += 25; msg.append("✅ 營收年增")
    if not m.empty and 'MarginPurchaseStock' in m.columns and len(m) >= 5:
        if m['MarginPurchaseStock'].iloc[-1] <= m['MarginPurchaseStock'].iloc[-5]: score += 25; msg.append("✅ 融資洗盤")
    return score, msg

# --- 5. UI 介面 ---
st.title("🏹 超級分析師：終極旗艦版")
target_sid = st.sidebar.text_input("輸入股票代碼", "1560")
my_cost = st.sidebar.number_input("您的買入成本", value=0.0)

if login_ok:
    t_df, c_df, m_df, r_df = get_stock_data_full(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1]
        # A. 頂部即時儀表板
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close'] - t_df['close'].iloc[-2], 2)}")
            st.caption(f"數據日期: {last['date']}")
        with c2:
            score, details = run_scoring(t_df, c_df, m_df, r_df)
            st.metric("自動診斷評分", f"{score} 分")
            st.write(" | ".join(details))
        with c3:
            trend = "🟢 上揚 (助漲)" if last['Slope20'] > 0 else "🔴 下彎 (助跌)"
            st.metric("月線趨勢", trend)
        st.markdown("---")

        # B. 功能分頁
        tab0, tab1, tab2, tab3 = st.tabs(["🚀 盤中爆量/選股", "📉 量價扣抵圖", "🔥 籌碼照妖鏡", "📊 營收診斷"])
        
        with tab0:
            st.subheader("🔥 盤中爆量偵測 (站穩雙線標的)")
            # 掃描邏輯：投信買超前 20 + 權值種子
            seeds = ['1560', '2330', '2454', '2615', '2603', '3231', '2317']
            res = []
            v_f = get_v_factor()
            for s in seeds:
                try:
                    temp_t = dl.taiwan_stock_daily(stock_id=s, start_date=(datetime.now()-timedelta(days=10)).strftime("%Y-%m-%d"))
                    if not temp_t.empty:
                        l = temp_t.iloc[-1]
                        avg_v = temp_t['Trading_Volume'].iloc[-6:-1].mean()
                        v_r = round(l['Trading_Volume'] / avg_v, 2)
                        # 爆量判定：考慮盤中時間因子
                        if v_r > (v_f * 1.5):
                            res.append({'代號': s, '目前量能倍數': f"{v_r}x", '現價': l['close'], '量能狀態': '🔥 爆量'})
                except: continue
            st.table(pd.DataFrame(res))

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white', width=1.5)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            # 扣抵標註
            if len(t_df) > 21:
                fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[last['MA20_Ref']], mode='markers', name='月扣抵', marker=dict(size=12, color='yellow', symbol='x')))
            fig.update_layout(template="plotly_dark", height=500); st.plotly_chart(fig, use_container_width=True)
            
            # 風控
            if my_cost > 0:
                sl = round(my_cost * 0.93, 2)
                st.sidebar.markdown(f"🛑 **停損參考價(-7%): {sl}**")
                if last['close'] <= sl: st.sidebar.error("🚨 警告：已觸發停損！")

        with tab2:
            if not c_df.empty:
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣超"), use_container_width=True)
            if not m_df.empty and 'MarginPurchaseStock' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="散戶融資餘額 (照妖鏡)"), use_container_width=True)

        with tab3:
            if not r_df.empty:
                st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收走勢"), use_container_width=True)
else:
    st.error("登入失敗，請檢查 Secrets")
    
