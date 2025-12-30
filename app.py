import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="超級分析師-Pro終極旗艦版", layout="wide")

# --- 2. Sponsor Pro 登入 ---
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

# --- 3. 核心運算：技術指標與 AI 邏輯 ---
def calculate_technicals(df):
    """計算 RSI, MACD, 布林通道"""
    if df.empty or len(df) < 30: return df
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['Upper'] = df['MA20'] + (std * 2)
    df['Lower'] = df['MA20'] - (std * 2)
    
    # MA60
    df['MA60'] = df['close'].rolling(60).mean()
    return df

def get_ai_advice(df):
    """AI 分析師判斷邏輯"""
    if df.empty or len(df) < 30: return "資料不足", [], 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 50
    reasons = []

    # 趨勢
    if last['close'] > last['MA20'] > last['MA60']:
        score += 20; reasons.append("✅ 多頭排列 (價>月>季)")
    elif last['close'] < last['MA20'] < last['MA60']:
        score -= 20; reasons.append("❌ 空頭排列 (價<月<季)")
    
    # 動能 RSI
    if last['RSI'] > 80: score -= 10; reasons.append("⚠️ RSI 過熱 (>80)")
    elif last['RSI'] < 20: score += 15; reasons.append("⭕ RSI 超賣 (<20)")
    
    # 訊號 MACD
    if last['MACD'] > last['Signal'] and prev['MACD'] <= prev['Signal']:
        score += 15; reasons.append("⭐ MACD 黃金交叉")
        
    # 布林
    if last['close'] > last['Upper']: reasons.append("⚠️ 觸及布林上軌(壓力)")
    if last['close'] < last['Lower']: reasons.append("⭕ 觸及布林下軌(支撐)")

    # 結論
    if score >= 80: advice = "🔥 強力買進"
    elif score >= 60: advice = "✅ 偏多操作"
    elif score <= 35: advice = "❌ 建議賣出"
    elif score <= 50: advice = "🔻 偏空/觀望"
    else: advice = "👀 中立震盪"
    
    return advice, reasons, score

# --- 4. 數據抓取：市場全景 (Snapshot) ---
@st.cache_data(ttl=60)
def get_market_data_pro():
    """一次抓取全市場快照，產出資金流向與相對大量榜"""
    if not login_ok: return pd.DataFrame(), pd.DataFrame()
    try:
        # Sponsor Pro 極速快照
        snap_all = dl.taiwan_stock_daily_snapshot()
        if snap_all.empty: return pd.DataFrame(), pd.DataFrame()
        
        # A. 處理細分族群資金流向
        sectors = {
            "晶圓代工": ["2330", "2303", "5347", "6770"], 
            "IC設計": ["2454", "3034", "3035", "3529"], 
            "CoWoS設備": ["1560", "3131", "3583", "6187", "6640"], 
            "矽光子CPO": ["3363", "4979", "6451", "3081", "3450"], 
            "AI組裝": ["2382", "3231", "2376", "6669", "2356"], 
            "散熱": ["3017", "3324", "3653", "2421"], 
            "航運": ["2603", "2609", "2615", "2637"], 
            "重電": ["1513", "1519", "1503", "1514"]
        }
        flow_res = []
        for name, sids in sectors.items():
            sub = snap_all[snap_all['stock_id'].isin(sids)]
            if not sub.empty:
                sub = sub[sub['volume'] > 0]
                if not sub.empty:
                    flow_res.append({
                        "族群": name,
                        "平均漲跌%": round(sub['tv_change_rate'].mean(), 2),
                        "熱度(張)": int(sub['volume'].sum()/1000)
                    })
        flow_df = pd.DataFrame(flow_res).sort_values("平均漲跌%", ascending=False) if flow_res else pd.DataFrame()

        # B. 處理全台相對大量榜
        # 簡單計算：今日量 / 昨日量 (snapshot 內建 last_close_volume)
        snap_all['相對量'] = round(snap_all['volume'] / (snap_all['last_close_volume'] + 1), 2)
        # 取量大於 1000 張且相對量大的前 15 名
        rank_df = snap_all[snap_all['volume'] > 1000000].sort_values('相對量', ascending=False).head(15)
        rank_df = rank_df[['stock_id', 'stock_name', 'last_close', '相對量']]
        
        return flow_df, rank_df
    except: return pd.DataFrame(), pd.DataFrame()

# --- 5. 數據抓取：個股深度 (Hybrid Patch) ---
@st.cache_data(ttl=30)
def get_stock_data_full(sid):
    today_str = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    
    # 抓取四大報表
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start)
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # 即時補丁
    try:
        snap = dl.taiwan_stock_daily_snapshot(stock_id=sid)
        if not snap.empty and not t.empty:
            if t['date'].iloc[-1] != today_str:
                new_row = {
                    'date': today_str, 'close': snap['last_close'].iloc[0],
                    'open': snap['open'].iloc[0], 'high': snap['high'].iloc[0], 'low': snap['low'].iloc[0],
                    'Trading_Volume': snap['volume'].iloc[0]
                }
                t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
    except: pass
    
    # 計算指標
    if not t.empty:
        t = calculate_technicals(t)
        t['Slope20'] = t['MA20'].diff()
        t['MA20_Ref'] = t['close'].shift(20)
        
    if not c.empty: c['net_buy'] = c['buy'] - c['sell']
    
    return t, c, m, r

# --- 6. UI 顯示層 ---
st.title("🏹 超級分析師：Sponsor Pro 終極旗艦版")

if login_ok:
    st.sidebar.success(f"✅ Pro 連線中 ({user_id[:3]}***)")
    target_sid = st.sidebar.text_input("輸入代碼", "1560")
    if st.sidebar.button("🔄 刷新"): st.cache_data.clear()

    # --- 上半部：市場全景 ---
    st.subheader("🌊 Pro 級全市場戰情室")
    flow_df, rank_df = get_market_data_pro()
    
    tab_m1, tab_m2 = st.tabs(["💰 細分族群資金流向", "🔥 全台股相對大量榜"])
    with tab_m1:
        if not flow_df.empty:
            c1, c2 = st.columns([2, 1])
            with c1: st.plotly_chart(px.bar(flow_df, x="族群", y="平均漲跌%", color="平均漲跌%", color_continuous_scale='RdYlGn', text="平均漲跌%"), use_container_width=True)
            with c2: st.dataframe(flow_df, hide_index=True, use_container_width=True)
        else: st.info("數據讀取中...")
    with tab_m2:
        if not rank_df.empty: st.dataframe(rank_df, hide_index=True, use_container_width=True)
        else: st.info("數據讀取中...")

    st.markdown("---")

    # --- 下半部：個股智能診斷 ---
    t_df, c_df, m_df, r_df = get_stock_data_full(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1]
        advice, reasons, score = get_ai_advice(t_df)
        
        st.markdown(f"### 🎯 {target_sid} 智能診斷報告")
        
        # 1. 建議卡片
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("最新價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            color = "green" if score >= 60 else "red" if score <= 40 else "orange"
            st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 10px; border-radius: 10px; text-align: center;">
                <h2 style="color: {color}; margin:0;">{advice}</h2>
                <p style="margin:0;">信心分數: {score}/100</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("#### 🕵️ AI 分析師觀點：")
            st.write("  \n".join(reasons))
            
        # 2. 深度圖表
        tabs = st.tabs(["📉 主圖(布林/扣抵)", "🔥 籌碼/融資", "📊 營收/副圖"])
        
        with tabs[0]:
            fig = go.Figure()
            # 價格與布林
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white', width=2)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['Upper'], name='上軌', line=dict(color='rgba(0,255,0,0.3)', width=1), showlegend=False))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['Lower'], name='下軌', line=dict(color='rgba(0,255,0,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,255,0,0.05)', showlegend=False))
            # 均線
            if 'MA20' in t_df.columns: fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=2)))
            if 'MA60' in t_df.columns: fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            # 扣抵
            if len(t_df) > 21: fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[t_df['close'].shift(20).iloc[-1]], mode='markers', name='扣抵值', marker=dict(size=12, color='yellow', symbol='x')))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            if not c_df.empty: 
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣超"), use_container_width=True)
            # 修正後的融資欄位
            if not m_df.empty and 'MarginPurchaseTodayBalance' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseTodayBalance', title="融資今日餘額"), use_container_width=True)
        
        with tabs[2]:
            c1, c2 = st.columns(2)
            with c1:
                # 營收 (防 IndexError)
                if not r_df.empty: st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收"), use_container_width=True)
            with c2:
                # MACD
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=t_df['date'], y=t_df['MACD']-t_df['Signal'], name='MACD柱狀', marker_color=np.where((t_df['MACD']-t_df['Signal'])>0, 'red', 'green')))
                fig2.add_trace(go.Scatter(x=t_df['date'], y=t_df['MACD'], name='DIF', line=dict(color='yellow')))
                fig2.add_trace(go.Scatter(x=t_df['date'], y=t_df['Signal'], name='MACD', line=dict(color='cyan')))
                fig2.update_layout(title="MACD 指標")
                st.plotly_chart(fig2, use_container_width=True)

    else:
        st.error(f"無法獲取 {target_sid} 數據。")
else:
    st.error("❌ 請檢查 Secrets 設定 Sponsor Pro 帳號。")
