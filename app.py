import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-帶量突破版", layout="wide")

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

# --- 3. 核心功能：帶量突破掃描儀 ---

@st.cache_data(ttl=3600)
def scan_volume_breakout():
    """自動掃描：當日量大 + 站上 20/60MA + 投信買超"""
    if not login_ok: return pd.DataFrame(), ""
    results = []
    target_d = ""
    
    # 找尋最近交易日
    for i in range(1, 6):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            chip_df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=d, end_date=d)
            if chip_df is not None and not chip_df.empty:
                target_d = d
                # 篩選投信有買的前 30 名進行深度技術分析
                top_sitc = chip_df[chip_df['SITC_Trust'] > 0].sort_values(by='SITC_Trust', ascending=False).head(30)
                
                for _, row in top_picks.iterrows():
                    sid = row['stock_id']
                    try:
                        tech = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=100)).strftime("%Y-%m-%d"))
                        if len(tech) >= 60:
                            last = tech.iloc[-1]
                            avg_vol = tech['Trading_Volume'].tail(6).head(5).mean() # 前 5 日均量
                            curr_vol = last['Trading_Volume']
                            ma20 = tech['close'].tail(20).mean()
                            ma60 = tech['close'].tail(60).mean()
                            
                            # 判定條件：量增 1.5 倍 + 站在雙線之上
                            if curr_vol > (avg_vol * 1.5) and last['close'] > ma20 and last['close'] > ma60:
                                results.append({
                                    '代號': sid,
                                    '名稱': row['stock_name'],
                                    '成交量(張)': int(curr_vol/1000),
                                    '量增倍數': round(curr_vol/avg_vol, 2),
                                    '收盤價': last['close'],
                                    '投信買超': row['SITC_Trust']
                                })
                    except: continue
                if results: break
        except: continue
    return pd.DataFrame(results), target_d

@st.cache_data(ttl=600)
def get_all_data(sid):
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
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
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：帶量突破戰情室")

target_sid = st.sidebar.text_input("輸入代碼診斷", "2330")

tab0, tab1, tab2, tab3 = st.tabs(["🚀 帶量突破強勢股", "📈 技術扣抵圖", "🔥 籌碼照妖鏡", "📊 營收診斷"])

if login_ok:
    with tab0:
        st.subheader("🔥 今日精選：帶量突破 + 站穩雙均線")
        st.caption("條件：成交量 > 5日均量 1.5 倍，且股價 > 20MA & 60MA")
        sig_df, sig_date = scan_volume_breakout()
        if not sig_df.empty:
            st.write(f"📅 資料日期：{sig_date}")
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
            st.success("💡 這些股票剛發動攻擊，且上方無短期均線壓力，值得重點關注！")
        else:
            st.info("目前尚無符合『帶量突破』條件之標的。")

    # (Tab 1-3 保持原本的強化版技術圖表、籌碼與營收邏輯...)
    t_df, c_df, m_df, r_df = get_all_data(target_sid)
    with tab1:
        if not t_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='現價', line=dict(color='white', width=1.5)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA(月)', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA(季)', line=dict(color='#FF00FF', width=2, dash='dot')))
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # 成交量
            st.plotly_chart(px.bar(t_df, x='date', y='Trading_Volume', title="成交量 (觀察今日是否爆量)", color_discrete_sequence=['#555555']), use_container_width=True, height=150)
