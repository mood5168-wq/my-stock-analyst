import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader
import time

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-Pro穩定版", layout="wide")

# --- 2. Pro 帳號登入 ---
dl = DataLoader()
login_ok = False
user_id = st.secrets.get("FINMIND_USER_ID", "")

# 嘗試登入
try:
    if "FINMIND_USER_ID" in st.secrets:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    elif "FINMIND_TOKEN" in st.secrets:
        dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
        login_ok = True
except: pass

# --- 3. 核心引擎：安全抓取即時 Tick ---
def get_realtime_tick_safe(sid):
    """利用 Pro 權限抓取最新一筆成交，不依賴 snapshot"""
    try:
        # 抓取今日逐筆成交
        df = dl.taiwan_stock_tick(stock_id=sid, date=datetime.now().strftime("%Y-%m-%d"))
        if not df.empty:
            return df['deal_price'].iloc[-1], df['volume'].sum()
    except: pass
    return None, None

# --- 4. 資金流向：手動遍歷 (避開 AttributeError) ---
@st.cache_data(ttl=60)
def get_sector_flow_manual():
    if not login_ok: return pd.DataFrame()
    
    sectors = {
        "半導體": ["2330", "2454", "1560", "3131"],
        "AI伺服器": ["2382", "3231", "2376", "6669"],
        "航運": ["2603", "2609", "2615"],
        "重電": ["1513", "1519", "1503"],
        "光通訊": ["4979", "3363", "6451"],
        "金融": ["2881", "2882", "2891"]
    }
    
    results = []
    for name, sids in sectors.items():
        chg_list = []
        vol_total = 0
        for sid in sids:
            rt_price, rt_vol = get_realtime_tick_safe(sid)
            if rt_price:
                try:
                    # 抓前一日收盤計算漲跌
                    hist = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=5)).strftime("%Y-%m-%d"))
                    if not hist.empty:
                        prev_close = hist['close'].iloc[-2] if len(hist) > 1 else hist['close'].iloc[-1]
                        chg = (rt_price - prev_close) / prev_close * 100
                        chg_list.append(chg)
                        vol_total += rt_vol
                except: pass
        
        if chg_list:
            results.append({
                "族群": name,
                "平均漲跌%": round(sum(chg_list) / len(chg_list), 2),
                "熱度(張)": int(vol_total/1000)
            })
            
    if results:
        return pd.DataFrame(results).sort_values("平均漲跌%", ascending=False)
    return pd.DataFrame()

# --- 5. 個股深度數據 (含 12/30 補丁) ---
@st.cache_data(ttl=30)
def get_stock_data(sid):
    today = datetime.now().strftime("%Y-%m-%d")
    # 抓取範圍加大到 400 天，確保有足夠的營收數據計算 YoY
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    
    t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
    c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
    m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start)
    r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start)
    
    # 手動補丁：抓 Tick 補日線
    rt_price, rt_vol = get_realtime_tick_safe(sid)
    if rt_price and not t.empty and t['date'].iloc[-1] != today:
        new_row = t.iloc[-1].copy()
        new_row['date'] = today
        new_row['close'] = rt_price
        new_row['Trading_Volume'] = rt_vol
        t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)

    if not t.empty:
        t['MA20'] = t['close'].rolling(20).mean()
        t['MA60'] = t['close'].rolling(60).mean()
        t['Slope20'] = t['MA20'].diff()
        t['MA20_Ref'] = t['close'].shift(20)

    if not c.empty: c['net_buy'] = c['buy'] - c['sell']
    return t, c, m, r

# --- 6. UI 介面 ---
st.title("🏹 超級分析師：Sponsor Pro 穩定防護版")

if login_ok:
    st.sidebar.success(f"✅ Pro 登入成功 ({user_id[:3]}***)")
    target_sid = st.sidebar.text_input("輸入代碼", "1560")
    if st.sidebar.button("🔄 刷新數據"):
        st.cache_data.clear()

    # A. 資金流向
    st.subheader("🌊 十大族群資金流向 (Pro 即時)")
    flow_df = get_sector_flow_manual()
    
    if not flow_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(px.bar(flow_df, x="族群", y="平均漲跌%", color="平均漲跌%", color_continuous_scale='RdYlGn', text="平均漲跌%"), use_container_width=True)
        with c2:
            st.dataframe(flow_df, hide_index=True, use_container_width=True)
    else:
        st.warning("⚠️ 盤中暫無數據或今日未開盤 (API 正常)。")

    st.markdown("---")

    # B. 個股診斷
    t_df, c_df, m_df, r_df = get_stock_data(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1]
        st.markdown(f"### 🎯 {target_sid} 深度即時分析")
        
        # 1. 儀表板
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最新價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            st.caption(f"資料日期: {last['date']}")
        with col2:
            trend = "🟢 上揚" if last.get('Slope20', 0) > 0 else "🔴 下彎"
            st.metric("月線趨勢", trend)
        with col3:
            # --- 自動評分 (安全版) ---
            score = 0
            # 技術面 check
            if not pd.isna(last.get('MA20')) and last['close'] > last['MA20']: 
                score += 30
            # 籌碼面 check
            if not c_df.empty and len(c_df) >= 3 and c_df['
