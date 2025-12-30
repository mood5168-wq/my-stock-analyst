import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader
import time

# --- 1. 基礎設定 ---
st.set_page_config(page_title="超級分析師-Pro真即時版", layout="wide")

# --- 2. 登入 (Sponsor Pro 權限核心) ---
dl = DataLoader()
login_ok = False
user_id = st.secrets.get("FINMIND_USER_ID", "未設定")

try:
    if "FINMIND_USER_ID" in st.secrets:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    elif "FINMIND_TOKEN" in st.secrets:
        dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
        login_ok = True
except: pass

# --- 3. 核心引擎：真·即時數據拼接 (Real-time Hybrid) ---
@st.cache_data(ttl=30) # Pro 用戶設定 30秒更新一次，非常即時
def get_stock_data_realtime(sid):
    # A. 獲取歷史日線 (只到昨天)
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    try:
        t_df = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
        c_df = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        m_df = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
        r_df = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # B. 獲取盤中即時快照 (Pro 專屬火力)
    try:
        # 這裡直接指定 stock_id，速度最快
        snap = dl.taiwan_stock_daily_snapshot(stock_id=sid)
        
        # C. 數據拼接 (關鍵步驟)
        if not snap.empty and not t_df.empty:
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            # 檢查日線最後一筆是不是今天
            # 如果不是今天，代表日線還沒更新，我們必須手動把 Snapshot 接上去
            if t_df['date'].iloc[-1] != today_str:
                new_row = {
                    'date': today_str,
                    'stock_id': sid,
                    'close': snap['last_close'].iloc[0], # 最新成交價
                    'open': snap['open'].iloc[0],
                    'max': snap['high'].iloc[0],
                    'min': snap['low'].iloc[0],
                    'Trading_Volume': snap['volume'].iloc[0] # 即時量
                }
                # 使用 concat 拼接
                t_df = pd.concat([t_df, pd.DataFrame([new_row])], ignore_index=True)
    except Exception as e:
        print(f"Snapshot Error: {e}")

    # D. 計算技術指標 (包含剛補上去的即時數據)
    if not t_df.empty and len(t_df) > 60:
        t_df['MA20'] = t_df['close'].rolling(20).mean()
        t_df['MA60'] = t_df['close'].rolling(60).mean()
        t_df['Slope20'] = t_df['MA20'].diff()
        t_df['MA20_Ref'] = t_df['close'].shift(20)

    # E. 籌碼整理
    if not c_df.empty: c_df['net_buy'] = c_df['buy'] - c_df['sell']
    
    return t_df, c_df, m_df, r_df

# --- 4. 資金流向：全市場快照 (Market Snapshot) ---
@st.cache_data(ttl=60)
def get_market_flow_pro():
    if not login_ok: return pd.DataFrame()
    
    # 因為你是 Pro，我們直接抓全市場快照，這才是正確用法
    try:
        # 這行指令會回傳台股所有股票的即時狀態
        snap_all = dl.taiwan_stock_daily_snapshot()
        if snap_all.empty: return pd.DataFrame()
        
        sectors = {
            "半導體": ["2330", "2454", "1560", "3131", "3583"],
            "AI伺服器": ["2382", "3231", "2376", "6669", "2356", "3017"],
            "航運": ["2603", "2609", "2615", "2605", "2637"],
            "重電": ["1513", "1519", "1503", "1514", "1609"],
            "光通訊": ["4979", "3363", "6451", "3081", "3450"],
            "金融": ["2881", "2882", "2891", "2886", "5880"]
        }
        
        results = []
        for name, sids in sectors.items():
            sub = snap_all[snap_all['stock_id'].isin(sids)]
            if not sub.empty:
                # 排除無量
                sub = sub[sub['volume'] > 0]
                if not sub.empty:
                    # tv_change_rate 是快照裡的即時漲跌幅
                    results.append({
                        "族群": name,
                        "平均漲跌%": round(sub['tv_change_rate'].mean(), 2),
                        "熱度(張)": int(sub['volume'].sum()/1000)
                    })
        
        if results:
            return pd.DataFrame(results).sort_values("平均漲跌%", ascending=False)
    except: pass
    return pd.DataFrame()

# --- 5. UI 介面 ---
st.title("🏹 超級分析師：Sponsor Pro 真・即時戰情室")

if login_ok:
    st.sidebar.success(f"✅ Pro 連線運作中 ({str(user_id)[:3]}***)")
    target_sid = st.sidebar.text_input("輸入代碼", "1560")
    if st.sidebar.button("🔄 立即刷新"): st.cache_data.clear()

    # A. 資金流向
    st.subheader("🌊 十大族群資金流向 (Real-time Snapshot)")
    with st.spinner("🚀 正在調用 Pro 專屬快照接口..."):
        flow_df = get_market_flow_pro()
        
    if not flow_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1: st.plotly_chart(px.bar(flow_df, x="族群", y="平均漲跌%", color="平均漲跌%", color_continuous_scale='RdYlGn', text="平均漲跌%"), use_container_width=True)
        with c2: st.dataframe(flow_df, hide_index=True, use_container_width=True)
    else:
        st.info("⌛ 盤中資料讀取中... (若目前非交易時間則無變動)")

    st.markdown("---")

    # B. 個股診斷
    t_df, c_df, m_df, r_df = get_stock_data_realtime(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1] # 這筆就是「拼接」上去的即時資料
        
        st.markdown(f"### 🎯 {target_sid} 深度即時分析")
        
        # 儀表板
        col1, col2, col3, col4 = st.columns(4)
        with col1: 
            st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            # 這裡應該會顯示今天的日期
            st.caption(f"數據日期: {last['date']}") 
        with col2:
            # 綜合評分 (修復 Index Error)
            score = 0
            if last['close'] > last.get('MA20', 0): score += 30
            if not c_df.empty and len(c_df)>=3 and c_df['net_buy'].tail(3).sum() > 0: score += 30
            if not r_df.empty:
                if len(r_df) >= 13 and r_df['revenue'].iloc[-1] > r_df['revenue'].iloc[-13]: score += 40
                elif len(r_df) < 13: score += 10 # 新股補償
            st.metric("綜合評分", f"{score} 分")
        with col3:
            trend = "🟢 上揚" if last.get('Slope20', 0) > 0 else "🔴 下彎"
            st.metric("月線趨勢", trend)
        with col4:
            # 相對量
            avg_v = t_df['Trading_Volume'].iloc[-6:-1].mean()
            curr_v = last['Trading_Volume']
            st.metric("相對量能", f"{round(curr_v/(avg_v+1), 2)}x")

        # 功能分頁
        tabs = st.tabs(["📉 技術三線(含即時)", "🔥 籌碼/融資", "📊 營收"])
        
        with tabs[0]:
            fig = go.Figure()
            # 畫 K 線或收盤連線
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white')))
            # 畫均線
            if 'MA20' in t_df.columns: fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            if 'MA60' in t_df.columns: fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            # 畫扣抵
            if len(t_df) > 21: fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[t_df['close'].shift(20).iloc[-1]], mode='markers', name='扣抵', marker=dict(size=10, color='yellow', symbol='x')))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            # 籌碼
            if not c_df.empty: 
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣超"), use_container_width=True)
            # 融資 (修復 ValueError: 欄位名稱)
            if not m_df.empty and 'MarginPurchaseTodayBalance' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseTodayBalance', title="融資今日餘額"), use_container_width=True)

        with tabs[2]:
            if not r_df.empty: st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收"), use_container_width=True)
    
    else:
        st.error(f"⚠️ 無法讀取 {target_sid}。請確認代號或 API 連線。")

else:
    st.error("❌ 請檢查 Secrets 設定 Sponsor Pro 帳號。")
