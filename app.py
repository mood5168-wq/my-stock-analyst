import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-資金流向版", layout="wide")

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

# --- 3. 核心運算：資金流向分析 ---
@st.cache_data(ttl=600)
def get_sector_flow():
    """模擬資金流向：計算關鍵族群的當日表現"""
    sectors = {
        "半導體/權值": ["2330", "2303", "2454", "2317"],
        "AI/伺服器": ["2382", "2376", "3231", "6669"],
        "航運/貨櫃": ["2603", "2609", "2615"],
        "記憶體": ["2408", "3260", "2344", "2337", "1560"], # 中砂近期與記憶體/設備連動
        "散裝/重電": ["2605", "1513", "1503", "1519"]
    }
    
    flow_data = []
    for name, sids in sectors.items():
        total_change = 0
        total_vol = 0
        for sid in sids:
            try:
                df = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=5)).strftime("%Y-%m-%d"))
                if not df.empty:
                    change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
                    total_change += change
                    total_vol += df['Trading_Volume'].iloc[-1]
            except: continue
        flow_data.append({"族群": name, "平均漲跌": round(total_change/len(sids), 2), "總成交量": total_vol})
    return pd.DataFrame(flow_data)

# --- 4. 原有功能函數 (維持不變) ---
@st.cache_data(ttl=60)
def get_stock_data_full(sid):
    start_date = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        if not t.empty:
            t['MA20'] = t['close'].rolling(20).mean(); t['MA60'] = t['close'].rolling(60).mean()
            t['MA20_Ref'] = t['close'].shift(20); t['Slope20'] = t['MA20'].diff()
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 5. UI 介面佈局 ---
st.title("🏹 超級分析師：資金流向旗艦版")
target_sid = st.sidebar.text_input("輸入股票代碼", "1560")

if login_ok:
    t_df, c_df, m_df, r_df = get_stock_data_full(target_sid)
    
    # 頂部即時摘要 (功能整合)
    if not t_df.empty:
        last = t_df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
        with c2: st.metric("20MA 月線趨勢", "🟢 上揚" if last['Slope20'] > 0 else "🔴 下彎")
        with c3: st.write(f"📊 今日成交量：{int(last['Trading_Volume']/1000)}k")

    # 分頁整合：爆量、技術、籌碼、營收、資金流向
    tabs = st.tabs(["🚀 資金流向/族群", "📉 技術扣抵圖", "🔥 籌碼照妖鏡", "📊 營收診斷"])
    
    with tabs[0]:
        st.subheader("🌊 今日資金流向 (族群強度分析)")
        sector_df = get_sector_flow()
        if not sector_df.empty:
            fig_flow = px.bar(sector_df, x="族群", y="平均漲跌", color="平均漲跌",
                               color_continuous_scale='RdYlGn', title="族群資金熱度 (綠色代表資金湧入)")
            st.plotly_chart(fig_flow, use_container_width=True)
            st.table(sector_df.sort_values(by="平均漲跌", ascending=False))
        
        st.markdown("---")
        st.subheader("🔥 盤中爆量提醒")
        # 這裡放入原本的爆量掃描邏輯... (略，保持與上一版一致)

    with tabs[1]:
        # 原本的螢光黃月線、桃紅季線、扣抵 X 標註功能
        if not t_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            if len(t_df) > 21:
                fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[t_df['close'].shift(20).iloc[-1]], mode='markers', name='月扣抵', marker=dict(size=12, color='yellow', symbol='x')))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        # 法人/融資照妖鏡邏輯
        if not c_df.empty:
            st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣"), use_container_width=True)
        if not m_df.empty and 'MarginPurchaseStock' in m_df.columns:
            st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="融資趨勢"), use_container_width=True)

    with tabs[3]:
        # 月營收邏輯
        if not r_df.empty:
            st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收走勢"), use_container_width=True)

else:
    st.error("登入失敗，請檢查 Secrets")
