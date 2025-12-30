import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-Pro終極版", layout="wide")

# --- 2. Pro 版安全登入 (建議使用 ID/Password 以獲取完整 Pro 權限) ---
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

# --- 3. Pro 級核心計算：全方位資金流向 (十大主流族群) ---
@st.cache_data(ttl=300)
def get_pro_sector_flow():
    """Pro 級數據：掃描十大主流族群資金流向"""
    sectors = {
        "半導體/設備": ["2330", "2303", "2454", "1560", "3131", "3583"],
        "AI伺服器/散熱": ["2382", "2376", "3231", "3017", "3324", "6669"],
        "光通訊/矽光子": ["4979", "3363", "3450", "6451", "3081"],
        "航運/貨櫃": ["2603", "2609", "2615", "2605", "2637"],
        "記憶體": ["2408", "3260", "2344", "2337", "8299"],
        "重電/能源": ["1513", "1519", "1503", "6806", "1514"],
        "面板/驅動IC": ["2409", "3481", "3034", "4961", "3545"],
        "PCB/載板": ["3037", "8046", "2367", "2313", "6213"],
        "金融/權值": ["2881", "2882", "2891", "2886", "2884"],
        "生技/醫療": ["1760", "4147", "6472", "1795", "6446"]
    }
    flow_data = []
    for name, sids in sectors.items():
        try:
            total_chg, total_vol = 0, 0
            count = 0
            for sid in sids:
                d = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=5)).strftime("%Y-%m-%d"))
                if not d.empty:
                    chg = (d['close'].iloc[-1] - d['close'].iloc[-2]) / d['close'].iloc[-2] * 100
                    total_chg += chg
                    total_vol += d['Trading_Volume'].iloc[-1]
                    count += 1
            if count > 0:
                flow_data.append({"族群": name, "平均漲跌": round(total_chg/count, 2), "資金熱度": total_vol})
        except: continue
    return pd.DataFrame(flow_data).sort_values(by="平均漲跌", ascending=False)

# --- 4. 核心數據抓取 (Pro 版穩定性優化) ---
@st.cache_data(ttl=60)
def get_stock_data_full(sid):
    start_date = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start_date)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        
        if not t.empty:
            t['MA20'] = t['close'].rolling(20).mean()
            t['MA60'] = t['close'].rolling(60).mean()
            t['MA20_Ref'] = t['close'].shift(20)
            t['MA60_Ref'] = t['close'].shift(60)
            t['Slope20'] = t['MA20'].diff()
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def run_scoring(t, c, m, r):
    """Pro 級自動評分系統 (25/25/25/25)"""
    score, msg = 0, []
    if not t.empty:
        last = t.iloc[-1]
        if last['close'] >= last['MA20']: score += 25; msg.append("✅ 站穩月線")
    if not c.empty:
        sitc = c[c['name'] == 'Investment_Trust'].tail(3)
        if not sitc.empty and sitc['net_buy'].sum() > 0: score += 25; msg.append("✅ 投信佈局")
    if not r.empty:
        if r['revenue'].iloc[-1] > r['revenue'].iloc[-13 if len(r)>12 else 0]: score += 25; msg.append("✅ 營收年增")
    if not m.empty and 'MarginPurchaseStock' in m.columns:
        if m['MarginPurchaseStock'].iloc[-1] <= m['MarginPurchaseStock'].iloc[-5]: score += 25; msg.append("✅ 融資洗盤")
    return score, msg

# --- 5. UI 介面佈局 ---
st.title("🏹 超級分析師：Sponsor Pro 終極戰情室")
target_sid = st.sidebar.text_input("輸入股票代碼", "1560")
my_cost = st.sidebar.number_input("您的買入成本", value=0.0)

if login_ok:
    t_df, c_df, m_df, r_df = get_stock_data_full(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1]
        # A. 頂部即時摘要
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            st.caption(f"數據日期: {last['date']}")
        with c2:
            score, details = run_scoring(t_df, c_df, m_df, r_df)
            st.metric("自動診斷評分", f"{score} 分")
            st.write(" | ".join(details))
        with c3:
            trend = "🟢 上揚 (助漲)" if last['Slope20'] > 0 else "🔴 下彎 (助跌)"
            st.metric("月線趨勢", trend)
        with c4:
            st.metric("今日成交張數", f"{int(last['Trading_Volume']/1000)}k")
        st.markdown("---")

        # B. 五大功能分頁
        tab0, tab1, tab2, tab3, tab4 = st.tabs(["🌊 資金流向", "🚀 盤中爆量/選股", "📉 量價扣抵圖", "🔥 籌碼照妖鏡", "📊 營收診斷"])
        
        with tab0:
            st.subheader("🌊 Pro 級資金流向掃描 (十大主流族群)")
            sector_df = get_pro_sector_flow()
            if not sector_df.empty:
                fig_flow = px.bar(sector_df, x="族群", y="平均漲跌", color="平均漲跌", color_continuous_scale='RdYlGn', title="族群強度 (越綠代表資金流入越強)")
                st.plotly_chart(fig_flow, use_container_width=True)
                st.table(sector_df.sort_values(by="平均漲跌", ascending=False))

        with tab1:
            st.subheader("🚀 盤中爆量突破偵測 (精選種子股)")
            seeds = ['1560', '2330', '2454', '2615', '2603', '3037', '2317', '3231', '2382', '2303', '3017', '4979']
            res = []
            for s in seeds:
                try:
                    temp_t = dl.taiwan_stock_daily(stock_id=s, start_date=(datetime.now()-timedelta(days=10)).strftime("%Y-%m-%d"))
                    if len(temp_t) > 5:
                        l = temp_t.iloc[-1]
                        avg_v = temp_t['Trading_Volume'].iloc[-6:-1].mean()
                        vr = round(l['Trading_Volume'] / avg_v, 2)
                        # 爆量條件：當前量 > 5日均量
                        if vr >= 1.2:
                            res.append({'代號': s, '量能倍數': f"{vr}x", '現價': l['close'], '狀態': '🔥 爆量發動中'})
                except: continue
            st.table(pd.DataFrame(res))

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white', width=1.5)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            if len(t_df) > 21:
                fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[last['MA20_Ref']], mode='markers', name='月扣抵', marker=dict(size=12, color='yellow', symbol='x')))
            fig.update_layout(template="plotly_dark", height=500); st.plotly_chart(fig, use_container_width=True)
            
            if my_cost > 0:
                sl = round(my_cost * 0.93, 2)
                st.sidebar.error(f"🚨 停損線參考(-7%): {sl}")

        with tab3:
            if not c_df.empty:
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣超"), use_container_width=True)
            if not m_df.empty and 'MarginPurchaseStock' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="融資趨勢 (照妖鏡)"), use_container_width=True)

        with tab4:
            if not r_df.empty:
                st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收走勢"), use_container_width=True)
else:
    st.error("登入失敗，請確認是否為 Sponsor Pro 權限之帳密。")
