import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-Pro終極旗艦版", layout="wide")

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

# --- 3. 核心功能：全台股相對大量掃描 (Pro 專屬) ---
@st.cache_data(ttl=300)
def scan_all_taiwan_relative_volume():
    """利用 Snapshot 接口快速掃描全台股量能增溫標的"""
    if not login_ok: return pd.DataFrame()
    try:
        df_all = dl.taiwan_stock_daily_snapshot()
        if df_all.empty: return pd.DataFrame()
        # 過濾流動性過低股 (成交量需大於 500 張)
        df_all = df_all[df_all['volume'] > 500000] 
        # 計算相對量能 (目前量 / 昨量)
        df_all['相對量能'] = round(df_all['volume'] / df_all['last_close_volume'], 2)
        top_vol = df_all.sort_values(by='相對量能', ascending=False).head(15)
        output = top_vol[['stock_id', 'stock_name', 'last_close', '相對量能']]
        output.columns = ['代號', '名稱', '現價', '量能增溫倍數']
        return output
    except: return pd.DataFrame()

# --- 4. 核心功能：個股深度資料與即時補丁 ---
@st.cache_data(ttl=60)
def get_stock_details_pro(sid):
    start = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start)
        
        # Pro 即時價格補丁：抓取今日 Tick 數據
        try:
            rt_df = dl.taiwan_stock_tick(stock_id=sid, date=today)
            if not rt_df.empty:
                rt_p = rt_df['deal_price'].iloc[-1]
                rt_v = rt_df['volume'].sum()
                rt_t = rt_df['time'].iloc[-1]
                if t['date'].iloc[-1] != today:
                    new_row = t.iloc[-1].copy()
                    new_row['date'], new_row['close'], new_row['Trading_Volume'] = today, rt_p, rt_v
                    t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
            else: rt_t = None
        except: rt_t = None

        if not t.empty:
            t['MA20'] = t['close'].rolling(20).mean()
            t['MA60'] = t['close'].rolling(60).mean()
            t['MA20_Ref'] = t['close'].shift(20)
            t['Slope20'] = t['MA20'].diff()
            
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r, rt_t
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

# --- 5. UI 介面 ---
st.title("🏹 超級分析師：Pro 終極全功能戰情室")
target_sid = st.sidebar.text_input("輸入股票代碼", "1560")
my_cost = st.sidebar.number_input("買入成本", value=0.0)

if login_ok:
    # A. 頂部：全台股相對大量監測
    st.subheader("🔥 今日全台股量能增溫排行榜 (相對大量)")
    top_df = scan_all_taiwan_relative_volume()
    if not top_df.empty:
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        st.caption("💡 倍數 > 1 代表今日成交量已超越昨天全天總量。")
    
    # B. 個股診斷區
    t_df, c_df, m_df, r_df, rt_time = get_stock_details_pro(target_sid)
    if not t_df.empty:
        last = t_df.iloc[-1]
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            st.write(f"數據日期: **{last['date']}**")
            if rt_time: st.caption(f"⏱️ 即時更新: {rt_time}")
        with col2:
            trend = "🟢 月線上揚 (助漲)" if last['Slope20'] > 0 else "🔴 月線下彎 (助跌)"
            st.metric("趨勢位階", trend)
        with col3:
            st.metric("今日成交張數", f"{int(last['Trading_Volume']/1000)}k")

        # C. 整合功能分頁
        tabs = st.tabs(["📉 量價扣抵圖", "🔥 籌碼照妖鏡", "🌊 資金流向", "📊 營收診斷"])
        
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white', width=1.5)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            # 扣抵點標註
            if len(t_df) > 21:
                fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[last['MA20_Ref']], mode='markers', name='月扣抵', marker=dict(size=12, color='yellow', symbol='x')))
            fig.update_layout(template="plotly_dark", height=450); st.plotly_chart(fig, use_container_width=True)
            
            if my_cost > 0:
                sl = round(my_cost * 0.93, 2)
                st.sidebar.error(f"🛑 停損參考價(-7%): {sl}")

        with tabs[1]:
            st.subheader("🔥 主力法人買賣超 vs 散戶融資餘額")
            if not c_df.empty:
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group'), use_container_width=True)
            if not m_df.empty and 'MarginPurchaseStock' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="散戶融資照妖鏡"), use_container_width=True)

        with tabs[2]:
            st.subheader("🌊 十大族群資金流向監控")
            # 族群掃描邏輯 (Pro 級數據)
            sectors = {"半導體": ["2330","2454","1560"], "AI伺服器": ["2382","3231","3017"], "航運": ["2603","2609","2615"]}
            flow_res = []
            for name, sids in sectors.items():
                try:
                    chg_sum = 0
                    for s in sids:
                        d = dl.taiwan_stock_daily(stock_id=s, start_date=(datetime.now()-timedelta(days=5)).strftime("%Y-%m-%d"))
                        chg_sum += (d['close'].iloc[-1]-d['close'].iloc[-2])/d['close'].iloc[-2]*100
                    flow_res.append({"族群": name, "漲跌幅": round(chg_sum/len(sids), 2)})
                except: continue
            st.table(pd.DataFrame(flow_res).sort_values("漲跌幅", ascending=False))

        with tabs[3]:
            if not r_df.empty:
                st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收趨勢圖"), use_container_width=True)

else:
    st.error("登入失敗，請檢查 Streamlit Secrets 設定是否正確。")
