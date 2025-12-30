import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-Pro全配版", layout="wide")

# --- 2. Pro 版安全登入 ---
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

# --- 3. 核心功能 A：十大族群資金流向 (防錯版) ---
@st.cache_data(ttl=300)
def get_all_sector_flows():
    # 預設空表防止崩潰
    default_df = pd.DataFrame(columns=["族群", "平均漲跌%", "資金熱度(張)"])
    if not login_ok: return default_df
    
    sectors = {
        "半導體設備": ["2330", "1560", "3131", "3583", "6139", "8028"],
        "AI伺服器": ["2382", "3231", "2376", "6669", "2356", "3017"],
        "散熱/機殼": ["3324", "3653", "3013", "8210", "2421"],
        "光通訊/矽光": ["4979", "3363", "6451", "3081", "3450", "3163"],
        "貨櫃/航運": ["2603", "2609", "2615", "2605", "2637", "5608"],
        "重電/能源": ["1513", "1519", "1503", "6806", "1514", "1609"],
        "記憶體": ["2408", "3260", "2344", "2337", "8299", "3006"],
        "面板/驅動": ["2409", "3481", "3034", "4961", "3545", "6116"],
        "PCB/載板": ["3037", "8046", "2367", "2313", "6213", "3044"],
        "金融/權值": ["2881", "2882", "2891", "2884", "2886", "5880"]
    }
    
    try:
        snap_df = dl.taiwan_stock_daily_snapshot()
        if snap_df.empty: return default_df
        
        flow_results = []
        for name, sids in sectors.items():
            targets = snap_df[snap_df['stock_id'].isin(sids)]
            if not targets.empty:
                avg_chg = targets['tv_change_rate'].mean()
                total_vol = targets['volume'].sum()
                flow_results.append({
                    "族群": name,
                    "平均漲跌%": round(avg_chg, 2) if not pd.isna(avg_chg) else 0.0,
                    "資金熱度(張)": int(total_vol/1000)
                })
        
        if not flow_results: return default_df
        return pd.DataFrame(flow_results).sort_values(by="平均漲跌%", ascending=False)
    except: return default_df

# --- 4. 核心功能 B：全台股相對大量 (Snapshot) ---
@st.cache_data(ttl=300)
def get_taiwan_relative_volume():
    if not login_ok: return pd.DataFrame()
    try:
        df = dl.taiwan_stock_daily_snapshot()
        if df.empty: return pd.DataFrame()
        # 過濾成交量 > 1000 張
        df = df[df['volume'] > 1000000]
        df['相對量能'] = round(df['volume'] / (df['last_close_volume'] + 1), 2)
        return df.sort_values(by='相對量能', ascending=False).head(15)[['stock_id','stock_name','last_close','相對量能']]
    except: return pd.DataFrame()

# --- 5. 核心功能 C：個股全方位診斷 (含營收與自動評分) ---
@st.cache_data(ttl=60)
def get_stock_data_pro(sid):
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start)
        
        # 強制 12/30 補丁
        snap_df = dl.taiwan_stock_daily_snapshot()
        snap = snap_df[snap_df['stock_id'] == sid]
        if not t.empty and not snap.empty and t['date'].iloc[-1] != today:
            new_row = t.iloc[-1].copy()
            new_row['date'], new_row['close'] = today, snap['last_close'].iloc[0]
            new_row['Trading_Volume'] = snap['volume'].iloc[0]
            t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
        
        if not t.empty:
            t['MA20'] = t['close'].rolling(20).mean()
            t['MA60'] = t['close'].rolling(60).mean()
            t['MA20_Ref'] = t['close'].shift(20)
            t['Slope20'] = t['MA20'].diff()
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def run_scoring(t, c, m, r):
    """恢復 0-100 分自動評分系統"""
    score, msgs = 0, []
    if not t.empty:
        last = t.iloc[-1]
        if not pd.isna(last['MA20']):
            if last['close'] > last['MA20']: score += 25; msgs.append("✅ 站穩月線")
            if t['Slope20'].iloc[-1] > 0: score += 10; msgs.append("✅ 月線上揚")
    if not c.empty:
        sitc = c[c['name'] == 'Investment_Trust'].tail(3)
        if not sitc.empty and sitc['net_buy'].sum() > 0: score += 20; msgs.append("✅ 投信佈局")
    if not r.empty:
        if r['revenue'].iloc[-1] > r['revenue'].iloc[-13 if len(r)>12 else 0]: score += 20; msgs.append("✅ 營收成長")
    if not m.empty and 'MarginPurchaseStock' in m.columns:
        if m['MarginPurchaseStock'].iloc[-1] < m['MarginPurchaseStock'].iloc[-5]: score += 25; msgs.append("✅ 融資減少")
    return score, msgs

# --- 6. UI 介面佈局 ---
st.title("🏹 超級分析師：Sponsor Pro 全功能終極版")
target_sid = st.sidebar.text_input("輸入個股代碼", "1560")

if login_ok:
    # A. 頂部看板：資金流向 + 相對大量
    st.subheader("🌊 Pro 級全市場監控")
    tab_m1, tab_m2 = st.tabs(["💰 十大族群資金流向", "🔥 全台股量能增溫榜"])
    
    with tab_m1:
        flow_df = get_all_sector_flows()
        if not flow_df.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_flow = px.bar(flow_df, x="族群", y="平均漲跌%", color="平均漲跌%",
                                   color_continuous_scale='RdYlGn', text="平均漲跌%")
                st.plotly_chart(fig_flow, use_container_width=True)
            with c2: st.dataframe(flow_df, hide_index=True, use_container_width=True)
        else: st.info("數據讀取中...")

    with tab_m2:
        vol_df = get_taiwan_relative_volume()
        if not vol_df.empty:
            st.dataframe(vol_df, hide_index=True, use_container_width=True)
        else: st.info("量能數據讀取中...")

    st.markdown("---")
    
    # B. 個股深度診斷
    t_df, c_df, m_df, r_df = get_stock_data_pro(target_sid)
    if not t_df.empty and 'MA20' in t_df.columns:
        last = t_df.iloc[-1]
        st.markdown(f"### 🎯 {target_sid} 深度即時分析")
        
        # 儀表板
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            st.caption(f"數據日期: {last['date']}")
        with col2:
            score, details = run_scoring(t_df, c_df, m_df, r_df)
            st.metric("自動評分", f"{score} 分")
        with col3:
            trend = "🟢 上揚" if last['Slope20'] > 0 else "🔴 下彎"
            st.metric("月線趨勢", trend)
        with col4:
            avg_v = t_df['Trading_Volume'].iloc[-6:-1].mean()
            st.metric("今日相對量", f"{round(last['Trading_Volume']/(avg_v+1), 2)}x")
            
        st.write(" | ".join(details))

        # 功能分頁 (保證全齊)
        tabs = st.tabs(["📉 技術三線扣抵", "🔥 籌碼照妖鏡", "📊 營收診斷"])
        
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white', width=1.5)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA(月)', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA(季)', line=dict(color='#FF00FF', width=2, dash='dot')))
            if len(t_df) > 21:
                fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[last['MA20_Ref']], mode='markers', name='月扣抵', marker=dict(size=12, color='yellow', symbol='x')))
            fig.update_layout(template="plotly_dark", height=450); st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            if not c_df.empty:
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣超"), use_container_width=True)
            if not m_df.empty and 'MarginPurchaseStock' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="散戶融資照妖鏡"), use_container_width=True)

        with tabs[2]:
            if not r_df.empty:
                st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收趨勢圖"), use_container_width=True)
            else: st.info("暫無營收資料")
else:
    st.error("登入失敗，請確認 Secrets 設定。")
