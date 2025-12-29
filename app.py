import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-旗艦版", layout="wide")

# --- 2. 安全登入 (支援 Secrets 帳密與 Token) ---
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

# --- 3. 核心功能：自動評分演算法 ---

def calculate_score(t_df, c_df, m_df, r_df):
    """自動評分系統 (各項 25 分，總分 100)"""
    score = 0
    details = []
    
    if not t_df.empty and len(t_df) > 20:
        last = t_df.iloc[-1]
        # A. 技術面 (25分)
        if last['close'] >= last['MA20']:
            score += 15
            details.append("✅ 站穩螢光黃月線 (+15)")
        if last['MA5'] > t_df['MA5'].iloc[-2]:
            score += 10
            details.append("✅ 短線 5MA 動能向上 (+10)")
            
    # B. 籌碼面 (25分)
    if not c_df.empty:
        sitc = c_df[c_df['name'] == 'Investment_Trust'].tail(3)
        if (sitc['net_buy'] > 0).all():
            score += 25
            details.append("✅ 投信連買 3 日鎖碼 (+25)")
        elif (sitc['net_buy'] > 0).any():
            score += 10
            details.append("✅ 投信近期有買盤 (+10)")

    # C. 基本面 (25分)
    if not r_df.empty:
        if r_df['revenue'].iloc[-1] > r_df['revenue'].iloc[-13 if len(r_df)>12 else 0]:
            score += 25
            details.append("✅ 月營收呈現年增 (+25)")

    # D. 散戶面 (25分)
    if not m_df.empty and len(m_df) > 5:
        m_diff = m_df['MarginPurchaseStock'].iloc[-1] - m_df['MarginPurchaseStock'].iloc[-5]
        if m_diff < 0:
            score += 25
            details.append("✅ 散戶退場/融資減少 (+25)")
            
    return score, details

@st.cache_data(ttl=3600)
def scan_super_signals():
    """自動掃描全台股強勢標的"""
    if not login_ok: return pd.DataFrame(), ""
    results = []
    target_d = ""
    for i in range(1, 6):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            chip_df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=d, end_date=d)
            if chip_df is not None and not chip_df.empty:
                target_d = d
                top_picks = chip_df.sort_values(by='SITC_Trust', ascending=False).head(15)
                for _, row in top_picks.iterrows():
                    sid = row['stock_id']
                    try:
                        tech = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=100)).strftime("%Y-%m-%d"))
                        if len(tech) >= 60:
                            if tech['close'].iloc[-1] > tech['close'].tail(20).mean():
                                results.append({'代號': sid, '名稱': row['stock_name'], '投信買超': row['SITC_Trust']})
                    except: continue
                if results: break
        except: continue
    return pd.DataFrame(results), target_d

@st.cache_data(ttl=600)
def get_all_stock_data(sid):
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        if not t.empty:
            t['MA5'] = t['close'].rolling(5).mean(); t['MA20'] = t['close'].rolling(20).mean(); t['MA60'] = t['close'].rolling(60).mean()
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 4. UI 介面佈局 ---
st.title("🏹 超級分析師：旗艦診斷儀表板")

# 側邊欄診斷區
st.sidebar.header("🎯 診斷核心")
target_sid = st.sidebar.text_input("輸入股票代號", "2330")
my_cost = st.sidebar.number_input("您的買入成本", value=0.0)

tab0, tab1, tab2, tab3 = st.tabs(["🚀 超級強勢訊號", "📈 量價技術面", "🔥 籌碼照妖鏡", "📊 營收診斷"])

if login_ok:
    t_df, c_df, m_df, r_df = get_all_stock_data(target_sid)
    
    # 計算並顯示分數
    f_score, f_details = calculate_score(t_df, c_df, m_df, r_df)
    
    with st.container():
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("🔥 實戰評分", f"{f_score} 分")
            if f_score >= 75: st.success("🎯 買進訊號：強烈建議關注")
            elif f_score >= 50: st.warning("⚖️ 觀望訊號：中性待變")
            else: st.error("🚨 避開訊號：籌碼或技術走弱")
        with c2:
            st.write("💡 **評分明細**")
            st.caption(" | ".join(f_details))

    with tab0:
        st.subheader("🌟 全台股強勢標的掃描 (投信鎖碼榜)")
        sig_df, sig_date = scan_super_signals()
        if not sig_df.empty:
            st.write(f"📅 資料日期：{sig_date}"); st.table(sig_df)
        else: st.info("暫無符合強勢條件標的。")

    with tab1:
        st.subheader(f"📊 {target_sid} 量價三線圖 (🟡 螢光黃=月線)")
        if not t_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='現價', line=dict(color='white', width=1)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA(月)', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA(季)', line=dict(color='#FF00FF', width=2, dash='dot')))
            fig.update_layout(template="plotly_dark", height=450); st.plotly_chart(fig, use_container_width=True)
            
            # 風控顯示
            last_p = t_df['close'].iloc[-1]
            st.sidebar.metric("目前價格", f"${last_p}", delta=f"{round(last_p-my_cost, 2)}" if my_cost > 0 else None)
            if my_cost > 0:
                sl = round(my_cost * 0.93, 2); st.sidebar.write(f"🛑 停損線: **{sl}**")
                if last_p <= sl: st.sidebar.error("🚨 警告：已破停損線！")

    with tab2:
        st.subheader("🔥 籌碼對決 (法人 vs 散戶)")
        if not c_df.empty:
            p_df = c_df[c_df['name'].isin(['Foreign_Investor', 'Investment_Trust'])]
            st.plotly_chart(px.bar(p_df, x='date', y='net_buy', color='name', barmode='group', title="法人買賣超"), use_container_width=True)
        if not m_df.empty:
            st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="散戶融資餘額"), use_container_width=True)

    with tab3:
        if not r_df.empty:
            st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收走勢"), use_container_width=True)

else:
    st.error("API 登入失敗")
