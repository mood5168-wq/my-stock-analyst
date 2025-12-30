import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader
import time

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-Pro診斷版", layout="wide")

# --- 2. Pro 權限診斷系統 (關鍵步驟) ---
dl = DataLoader()
login_msg = "初始化中..."
is_pro = False

try:
    # 檢查 Secrets 是否存在
    if "FINMIND_USER_ID" in st.secrets and "FINMIND_PASSWORD" in st.secrets:
        user_id = st.secrets["FINMIND_USER_ID"]
        password = st.secrets["FINMIND_PASSWORD"]
        
        # 嘗試登入
        dl.login(user_id=user_id, password=password)
        login_msg = f"✅ Sponsor Pro 登入成功！\n(帳號: {user_id[:3]}***)"
        is_pro = True
    else:
        login_msg = "❌ 失敗：未在 Secrets 設定帳號密碼，目前為 Guest 限制模式。"
except Exception as e:
    login_msg = f"❌ 登入發生錯誤：{str(e)}"

# --- 3. 核心功能：全台股相對大量 (Snapshot) ---
@st.cache_data(ttl=60) # Pro 版設為 60秒刷新
def get_snapshot_data():
    if not is_pro: return pd.DataFrame()
    try:
        # Sponsor Pro 專屬接口
        df = dl.taiwan_stock_daily_snapshot()
        if df.empty: return pd.DataFrame()
        
        # 資料清洗與計算
        df = df[df['volume'] > 500000] # 過濾成交量太小的
        df['相對量能'] = round(df['volume'] / (df['last_close_volume'] + 1), 2)
        return df
    except Exception as e:
        print(e)
        return pd.DataFrame()

# --- 4. 核心功能：個股深度資料 (含 12/30 補丁) ---
@st.cache_data(ttl=60)
def get_stock_data(sid):
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start)
        
        # Pro 即時補丁：嘗試抓取 Snapshot 來補今天 (12/30) 的資料
        if is_pro:
            snap_df = dl.taiwan_stock_daily_snapshot()
            if not snap_df.empty:
                snap = snap_df[snap_df['stock_id'] == sid]
                # 如果日線最後一筆不是今天，但快照有今天，就補上去
                if not t.empty and not snap.empty and t['date'].iloc[-1] != today:
                    new_row = t.iloc[-1].copy()
                    new_row['date'] = today
                    new_row['close'] = snap['last_close'].iloc[0]
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

# --- 5. UI 介面 ---
st.title("🏹 超級分析師：Sponsor Pro 戰情室")

# --- 側邊欄診斷區 ---
st.sidebar.header("🔧 連線狀態診斷")
if is_pro:
    st.sidebar.success(login_msg)
    st.sidebar.caption("🚀 已啟用 20,000次/小時 極速模式")
else:
    st.sidebar.error(login_msg)
    st.sidebar.warning("⚠️ 請檢查 Streamlit Secrets 設定，否則無法抓取即時大數據。")

target_sid = st.sidebar.text_input("輸入個股代碼", "1560")
if st.sidebar.button("🔄 強制刷新"):
    st.cache_data.clear()

# --- 主畫面 ---
if is_pro:
    # A. 資金流向與排行
    st.subheader("🔥 全市場即時掃描 (Pro Exclusive)")
    snap_df = get_snapshot_data()
    
    if not snap_df.empty:
        tab1, tab2 = st.tabs(["💰 十大族群資金流向", "🚀 全台相對大量榜"])
        
        with tab1:
            # 現場計算族群流向，不依賴緩存函數以防出錯
            sectors = {"半導體": ["2330","2454","1560"], "AI伺服器": ["2382","3231","6669"], "航運": ["2603","2609","2615"], "重電": ["1513","1519"], "光通訊": ["4979","3363"]}
            res = []
            for k, v in sectors.items():
                sub = snap_df[snap_df['stock_id'].isin(v)]
                if not sub.empty:
                    res.append({"族群": k, "平均漲跌%": round(sub['tv_change_rate'].mean(), 2), "熱度": int(sub['volume'].sum()/1000)})
            if res:
                df_sec = pd.DataFrame(res).sort_values("平均漲跌%", ascending=False)
                c1, c2 = st.columns([2,1])
                with c1: st.plotly_chart(px.bar(df_sec, x="族群", y="平均漲跌%", color="平均漲跌%", color_continuous_scale='RdYlGn'), use_container_width=True)
                with c2: st.dataframe(df_sec, hide_index=True)
        
        with tab2:
            # 取相對量前 15 名
            top15 = snap_df.sort_values('相對量能', ascending=False).head(15)[['stock_id','stock_name','last_close','相對量能']]
            st.dataframe(top15, use_container_width=True, hide_index=True)
    else:
        st.info("⏳ 正在連線 FinMind Pro 伺服器獲取即時快照，請稍候...")

    st.markdown("---")

    # B. 個股診斷
    t_df, c_df, m_df, r_df = get_stock_data(target_sid)
    if not t_df.empty:
        last = t_df.iloc[-1]
        st.markdown(f"### 🎯 {target_sid} 個股診斷")
        
        # 1. 儀表板
        c1, c2, c3 = st.columns(3)
        with c1: 
            st.metric("最新價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            st.caption(f"資料日期: {last['date']}")
        with c2:
            st.metric("月線趨勢", "🟢 上揚" if last.get('Slope20', 0) > 0 else "🔴 下彎")
        with c3:
            # 2. 自動評分
            score = 0
            if last['close'] > last.get('MA20', 0): score += 30
            if not c_df.empty and c_df['net_buy'].tail(3).sum() > 0: score += 30
            if not r_df.empty and r_df['revenue'].iloc[-1] > r_df['revenue'].iloc[-13]: score += 40
            st.metric("綜合評分", f"{score} 分")

        # 3. 三線扣抵 + 4. 籌碼 + 5. 營收
        tabs = st.tabs(["📉 技術三線", "🔥 籌碼/融資", "📊 營收"])
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white')))
            if 'MA20' in t_df.columns: fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            if 'MA60' in t_df.columns: fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            if len(t_df) > 21: fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[t_df['close'].shift(20).iloc[-1]], mode='markers', name='扣抵', marker=dict(size=10, color='yellow', symbol='x')))
            st.plotly_chart(fig, use_container_width=True)
        with tabs[1]:
            if not c_df.empty: st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group'), use_container_width=True)
            if not m_df.empty: st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="融資"), use_container_width=True)
        with tabs[2]:
            if not r_df.empty: st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="營收"), use_container_width=True)
    else:
        st.error(f"⚠️ 無法獲取 {target_sid} 資料。若上方診斷顯示登入成功，可能是該股代號錯誤或今日暫無交易。")
else:
    st.info("👋 請先在左側 Secrets 設定您的 Sponsor Pro 帳號，解鎖全功能。")
