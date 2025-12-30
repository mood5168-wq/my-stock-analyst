import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-Pro資金流向版", layout="wide")

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

# --- 3. 核心功能：十大族群資金流向監控 ---
@st.cache_data(ttl=300)
def get_all_sector_flows():
    """Pro 級數據：掃描台股十大主流族群當日強弱"""
    if not login_ok: return pd.DataFrame()
    
    sectors = {
        "半導體設備": ["2330", "1560", "3131", "3583", "6139"],
        "AI伺服器": ["2382", "3231", "2376", "6669", "2356"],
        "散熱/機殼": ["3017", "3324", "3653", "3013", "8210"],
        "光通訊/矽光": ["4979", "3363", "6451", "3081", "3450"],
        "貨櫃/航運": ["2603", "2609", "2615", "2605", "2637"],
        "重電/能源": ["1513", "1519", "1503", "6806", "1514"],
        "記憶體": ["2408", "3260", "2344", "2337", "8299"],
        "面板/驅動": ["2409", "3481", "3034", "4961", "3545"],
        "PCB/載板": ["3037", "8046", "2367", "2313", "6213"],
        "金融/權值": ["2881", "2882", "2891", "2884", "2886"]
    }
    
    flow_results = []
    # 使用 Snapshot 獲取族群內即時數據
    try:
        snap_df = dl.taiwan_stock_daily_snapshot()
        for name, sids in sectors.items():
            targets = snap_df[snap_df['stock_id'].isin(sids)]
            if not targets.empty:
                avg_chg = targets['tv_change_rate'].mean() # 漲跌幅百分比
                total_vol = targets['volume'].sum() # 總成交量
                flow_results.append({
                    "族群": name,
                    "平均漲跌%": round(avg_chg, 2),
                    "資金熱度(張)": int(total_vol/1000)
                })
    except: pass
    return pd.DataFrame(flow_results).sort_values(by="平均漲跌%", ascending=False)

# --- 4. 個股深度診斷與 12/30 補丁 ---
@st.cache_data(ttl=60)
def get_stock_data_pro(sid):
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
        # 強制快照補丁
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
        return t
    except: return pd.DataFrame()

# --- 5. UI 介面 ---
st.title("🏹 超級分析師：Pro 資金流向戰情室")
target_sid = st.sidebar.text_input("輸入個股代碼", "1560")

if login_ok:
    # --- Tab 0: 十大族群資金流向 ---
    st.subheader("🌊 今日全市場十大族群資金流向")
    flow_df = get_all_sector_flows()
    if not flow_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_flow = px.bar(flow_df, x="族群", y="平均漲跌%", color="平均漲跌%",
                               color_continuous_scale='RdYlGn', text="平均漲跌%")
            st.plotly_chart(fig_flow, use_container_width=True)
        with c2:
            st.write("📋 族群強度排行")
            st.dataframe(flow_df, hide_index=True)
    
    st.markdown("---")
    
    # --- 個股深度區 ---
    t_df = get_stock_data_pro(target_sid)
    if not t_df.empty:
        last = t_df.iloc[-1]
        st.markdown(f"### 🎯 {target_sid} 深度診斷")
        st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
        
        tab1, tab2 = st.tabs(["📉 三線扣抵圖", "📊 數據詳情"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white', width=1.5)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            if len(t_df) > 21:
                fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[last['MA20_Ref']], mode='markers', name='月扣抵', marker=dict(size=12, color='yellow', symbol='x')))
            fig.update_layout(template="plotly_dark", height=450); st.plotly_chart(fig, use_container_width=True)
else:
    st.error("登入失敗，請確認 Secrets。")
