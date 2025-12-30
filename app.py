import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="超級分析師-Pro嚴格風控版", layout="wide")

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

# --- 3. 核心運算：技術指標 ---
def calculate_technicals(df):
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
    df['MA20'] = df['close'].rolling(20).mean() # 月線
    std = df['close'].rolling(20).std()
    df['Upper'] = df['MA20'] + (std * 2)
    df['Lower'] = df['MA20'] - (std * 2)
    
    # MA60
    df['MA60'] = df['close'].rolling(60).mean() # 季線
    return df

# --- [關鍵修改] AI 分析師邏輯：加入嚴格乖離率評分 ---
def get_ai_advice(df):
    if df.empty or len(df) < 30: return "資料不足", [], 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 50 # 初始分
    reasons = []

    # 1. 乖離率診斷 (Bias Ratio Check) - 權重最重
    # 公式：(股價 - 月線) / 月線
    if last['MA20'] > 0:
        bias = (last['close'] - last['MA20']) / last['MA20'] * 100
        
        if bias > 18:
            score -= 40 # 重扣
            reasons.append(f"❌ 乖離過大 (+{round(bias,1)}%)：危險！股價離月線太遠，隨時修正。")
        elif bias > 12:
            score -= 15
            reasons.append(f"⚠️ 乖離偏高 (+{round(bias,1)}%)：追高風險增，建議等拉回。")
        elif 0 < bias <= 8:
            score += 20 # 最佳買點
            reasons.append(f"✅ 乖離適中 (+{round(bias,1)}%)：趨勢健康，適合佈局。")
        elif bias < -10:
            score += 15
            reasons.append(f"⭕ 負乖離大 ({round(bias,1)}%)：超賣區，醞釀反彈。")

    # 2. 趨勢診斷
    if last['close'] > last['MA20'] > last['MA60']:
        score += 15; reasons.append("✅ 多頭排列 (價>月>季)")
    elif last['close'] < last['MA20'] < last['MA60']:
        score -= 20; reasons.append("❌ 空頭排列 (價<月<季)")
    
    # 3. 動能 RSI
    if last['RSI'] > 80: score -= 10; reasons.append("⚠️ RSI 過熱 (>80)")
    elif last['RSI'] < 20: score += 10; reasons.append("⭕ RSI 超賣 (<20)")
    
    # 4. 訊號 MACD
    if last['MACD'] > last['Signal'] and prev['MACD'] <= prev['Signal']:
        score += 15; reasons.append("⭐ MACD 黃金交叉")
        
    # 5. 布林通道
    if last['close'] > last['Upper']: 
        score -= 5
        reasons.append("⚠️ 觸及布林上軌 (短線壓力)")

    # 結論總結
    if score >= 80: advice = "🔥 強力買進"
    elif score >= 60: advice = "✅ 偏多操作"
    elif score <= 30: advice = "❌ 建議賣出/避開"
    elif score <= 50: advice = "🔻 偏空/觀望"
    else: advice = "👀 中立/區間震盪"
    
    # 分數校正 (0-100)
    score = max(0, min(100, score))
    
    return advice, reasons, score

# --- 4. 數據抓取：市場全景 ---
@st.cache_data(ttl=60)
def get_market_data_pro():
    if not login_ok: return pd.DataFrame(), pd.DataFrame()
    try:
        snap_all = dl.taiwan_stock_daily_snapshot()
        if snap_all.empty: return pd.DataFrame(), pd.DataFrame()
        
        # 細分族群
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

        # 相對大量榜
        snap_all['相對量'] = round(snap_all['volume'] / (snap_all['last_close_volume'] + 1), 2)
        rank_df = snap_all[snap_all['volume'] > 1000000].sort_values('相對量', ascending=False).head(15)
        rank_df = rank_df[['stock_id', 'stock_name', 'last_close', '相對量']]
        
        return flow_df, rank_df
    except: return pd.DataFrame(), pd.DataFrame()

# --- 5. 數據抓取：個股深度 (Hybrid Patch) ---
@st.cache_data(ttl=30)
def get_stock_data_full(sid):
    today_str = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    
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
    
    if not t.empty:
        t = calculate_technicals(t)
        t['Slope20'] = t['MA20'].diff()
        t['MA20_Ref'] = t['close'].shift(20)
    if not c.empty: c['net_buy'] = c['buy'] - c['sell']
    
    return t, c, m, r

# --- 6. UI 顯示層 ---
st.title("🏹 超級分析師：Sponsor Pro 嚴格風控版")

if login_ok:
    st.sidebar.success(f"✅ Pro 連線中 ({user_id[:3]}***)")
    target_sid = st.sidebar.text_input("輸入代碼", "1560")
    if st.sidebar.button("🔄 刷新"): st.cache_data.clear()

    # 上半部：全景
    st.subheader("🌊 全市場戰情")
    flow_df, rank_df = get_market_data_pro()
    t1, t2 = st.tabs(["💰 族群資金流向", "🔥 相對大量榜"])
    with t1:
        if not flow_df.empty: 
            c1, c2 = st.columns([2,1])
            with c1: st.plotly_chart(px.bar(flow_df, x="族群", y="平均漲跌%", color="平均漲跌%", color_continuous_scale='RdYlGn'), use_container_width=True)
            with c2: st.dataframe(flow_df, hide_index=True, use_container_width=True)
        else: st.info("讀取中...")
    with t2:
        if not rank_df.empty: st.dataframe(rank_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # 下半部：個股
    t_df, c_df, m_df, r_df = get_stock_data_full(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1]
        advice, reasons, score = get_ai_advice(t_df)
        
        st.markdown(f"### 🎯 {target_sid} 智能診斷 (含乖離率風控)")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("最新價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            # 根據分數變色
            color = "green" if score >= 60 else "red" if score <= 30 else "orange"
            st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 10px; border-radius: 10px; text-align: center;">
                <h2 style="color: {color}; margin:0;">{advice}</h2>
                <p style="margin:0;">信心分數: {score}/100</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("#### 🕵️ 評分理由：")
            for r in reasons:
                st.write(r)
            
        tabs = st.tabs(["📉 主圖(乖離/布林)", "🔥 籌碼/融資", "📊 營收/副圖"])
        
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white', width=2)))
            # 布林
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
            if not m_df.empty and 'MarginPurchaseTodayBalance' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseTodayBalance', title="融資今日餘額"), use_container_width=True)
        
        with tabs[2]:
            c1, c2 = st.columns(2)
            with c1:
                if not r_df.empty: st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收"), use_container_width=True)
            with c2:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=t_df['date'], y=t_df['MACD']-t_df['Signal'], name='MACD柱狀', marker_color=np.where((t_df['MACD']-t_df['Signal'])>0, 'red', 'green')))
                fig2.add_trace(go.Scatter(x=t_df['date'], y=t_df['MACD'], name='DIF', line=dict(color='yellow')))
                fig2.add_trace(go.Scatter(x=t_df['date'], y=t_df['Signal'], name='MACD', line=dict(color='cyan')))
                st.plotly_chart(fig2, use_container_width=True)

    else: st.error("查無資料")
else: st.error("❌ Secrets 未設定")
