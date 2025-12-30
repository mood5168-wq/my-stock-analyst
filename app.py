import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader
import time

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-Pro診斷版", layout="wide")

# --- 2. 登入診斷系統 ---
dl = DataLoader()
login_status = "未登入"
try:
    if "FINMIND_USER_ID" in st.secrets and "FINMIND_PASSWORD" in st.secrets:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_status = "✅ 帳密登入成功 (Sponsor Pro)"
    elif "FINMIND_TOKEN" in st.secrets:
        dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
        login_status = "✅ Token 登入成功"
    else:
        login_status = "⚠️ 未偵測到 Secrets (請檢查設定)"
except Exception as e:
    login_status = f"❌ 登入失敗: {str(e)}"

# --- 3. 核心功能：十大族群資金流向 (雙軌制：快照失敗轉逐檔) ---
@st.cache_data(ttl=300)
def get_sector_flows_safe():
    # 定義族群與代表股
    sectors = {
        "半導體": ["2330", "2454", "1560", "3131"],
        "AI伺服器": ["2382", "3231", "2376", "6669"],
        "航運": ["2603", "2609", "2615"],
        "重電": ["1513", "1519", "1503"],
        "光通訊": ["4979", "3363", "6451"],
        "金融": ["2881", "2882", "2891"]
    }
    
    # 方法 A: 嘗試全市場快照 (最快)
    try:
        snap_df = dl.taiwan_stock_daily_snapshot()
        if not snap_df.empty:
            results = []
            for name, sids in sectors.items():
                targets = snap_df[snap_df['stock_id'].isin(sids)]
                if not targets.empty:
                    # 排除異常值
                    targets = targets[targets['volume'] > 0]
                    avg_chg = targets['tv_change_rate'].mean()
                    total_vol = targets['volume'].sum()
                    results.append({"族群": name, "漲跌幅%": round(avg_chg, 2), "熱度": int(total_vol/1000)})
            if results:
                return pd.DataFrame(results).sort_values("漲跌幅%", ascending=False)
    except: pass

    # 方法 B: 備案 - 逐檔抓取 (較慢但穩)
    try:
        results = []
        for name, sids in sectors.items():
            vals = []
            vols = 0
            for sid in sids:
                df = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=5)).strftime("%Y-%m-%d"))
                # 嘗試補抓即時價
                try:
                    tick = dl.taiwan_stock_tick(stock_id=sid, date=datetime.now().strftime("%Y-%m-%d"))
                    if not tick.empty:
                        curr = tick['deal_price'].iloc[-1]
                        vol = tick['volume'].sum()
                        prev = df['close'].iloc[-2] if len(df) > 1 else curr
                        chg = (curr - prev) / prev * 100
                        vals.append(chg)
                        vols += vol
                except: continue
            
            if vals:
                results.append({"族群": name, "漲跌幅%": round(sum(vals)/len(vals), 2), "熱度": int(vols/1000)})
        return pd.DataFrame(results).sort_values("漲跌幅%", ascending=False)
    except: return pd.DataFrame()

# --- 4. 核心功能：個股全方位數據 ---
@st.cache_data(ttl=60)
def get_stock_data_full(sid):
    start = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    
    t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
    c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
    m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start)
    r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start)

    # 即時補丁
    try:
        snap_df = dl.taiwan_stock_daily_snapshot()
        snap = snap_df[snap_df['stock_id'] == sid]
        if not t.empty and not snap.empty and t['date'].iloc[-1] != today:
            new_row = t.iloc[-1].copy()
            new_row['date'], new_row['close'] = today, snap['last_close'].iloc[0]
            new_row['Trading_Volume'] = snap['volume'].iloc[0]
            t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
    except: pass

    if not t.empty:
        t['MA20'] = t['close'].rolling(20).mean()
        t['MA60'] = t['close'].rolling(60).mean()
        t['MA20_Ref'] = t['close'].shift(20)
        t['Slope20'] = t['MA20'].diff()
    if not c.empty: c['net_buy'] = c['buy'] - c['sell']
    return t, c, m, r

# --- 5. UI 介面 ---
st.title("🏹 超級分析師：戰情室診斷版")

# 側邊欄：系統狀態與控制
st.sidebar.subheader("🔧 系統狀態")
st.sidebar.info(login_status)
target_sid = st.sidebar.text_input("輸入代碼", "1560")
if st.sidebar.button("🔄 強制刷新數據"):
    st.cache_data.clear()

# 主畫面
if "成功" in login_status:
    # A. 資金流向
    st.subheader("🌊 十大族群資金流向")
    with st.spinner("正在掃描全市場..."):
        flow_df = get_sector_flows_safe()
    
    if not flow_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(flow_df, x="族群", y="漲跌幅%", color="漲跌幅%", color_continuous_scale='RdYlGn', text="漲跌幅%")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(flow_df, hide_index=True, use_container_width=True)
    else:
        st.warning("⚠️ 暫無資金流向數據。可能原因：1. 開盤前 2. API 忙碌。請點擊左側『強制刷新』再試一次。")

    st.markdown("---")

    # B. 個股診斷
    t_df, c_df, m_df, r_df = get_stock_data_full(target_sid)
    if not t_df.empty:
        last = t_df.iloc[-1]
        st.markdown(f"### 🎯 {target_sid} 個股診斷")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最新價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            st.caption(f"日期: {last['date']}")
        with col2:
            trend = "🟢 上揚" if last.get('Slope20', 0) > 0 else "🔴 下彎"
            st.metric("月線趨勢", trend)
        with col3:
            # 自動評分簡化版
            score = 0
            if last['close'] > last.get('MA20', 0): score += 40
            if not c_df.empty and c_df['net_buy'].tail(3).sum() > 0: score += 30
            if not r_df.empty and r_df['revenue'].iloc[-1] > r_df['revenue'].iloc[-13]: score += 30
            st.metric("綜合評分", f"{score} 分")

        tabs = st.tabs(["📉 技術三線", "🔥 籌碼/融資", "📊 營收"])
        
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white')))
            if 'MA20' in t_df.columns:
                fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            if 'MA60' in t_df.columns:
                fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            # 扣抵
            if len(t_df) > 21:
                fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[t_df['close'].shift(20).iloc[-1]], mode='markers', name='扣抵', marker=dict(size=10, color='yellow', symbol='x')))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            if not c_df.empty: st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group'), use_container_width=True)
            if not m_df.empty: st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="融資"), use_container_width=True)

        with tabs[2]:
            if not r_df.empty: st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="營收"), use_container_width=True)
else:
    st.error(f"系統無法啟動：{login_status}")
