import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-Pro專業旗艦版", layout="wide")

# --- 2. Pro 版安全登入 ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        # Sponsor Pro 建議使用帳密登入以解鎖所有權限
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except:
        if "FINMIND_TOKEN" in st.secrets:
            dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
            login_ok = True

# --- 3. Pro 級核心計算：全方位資金流向 ---
@st.cache_data(ttl=300) # Pro 版資料更新快，縮短緩存
def get_pro_sector_flow():
    """Pro 級資金流向：掃描十大主流族群"""
    sectors = {
        "半導體/設備": ["2330", "2303", "2454", "1560", "3131"],
        "AI/伺服器": ["2382", "2376", "3231", "6669", "2356"],
        "光通訊/矽光子": ["4979", "3363", "3450", "6451"],
        "航運/貨櫃": ["2603", "2609", "2615", "2605"],
        "記憶體": ["2408", "3260", "2344", "2337"],
        "重電/能源": ["1513", "1519", "1503", "6806"],
        "面板/驅動IC": ["2409", "3481", "3034", "4961"],
        "PCB/載板": ["3037", "8046", "2367", "2313"],
        "金融/權值": ["2881", "2882", "2891", "2886"],
        "生技/防疫": ["1760", "4147", "6472", "1795"]
    }
    flow_data = []
    for name, sids in sectors.items():
        try:
            # 獲取族群內所有個股最新狀態
            total_chg, total_vol = 0, 0
            for sid in sids:
                d = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=5)).strftime("%Y-%m-%d"))
                if not d.empty:
                    chg = (d['close'].iloc[-1] - d['close'].iloc[-2]) / d['close'].iloc[-2] * 100
                    total_chg += chg
                    total_vol += d['Trading_Volume'].iloc[-1]
            flow_data.append({"族群": name, "平均漲跌": round(total_chg/len(sids), 2), "資金熱度": total_vol})
        except: continue
    return pd.DataFrame(flow_data).sort_values(by="平均漲跌", ascending=False)

# --- 4. 核心數據引擎 ---
@st.cache_data(ttl=60)
def get_stock_data_pro(sid):
    start = (datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start)
        if not t.empty:
            t['MA20'] = t['close'].rolling(20).mean(); t['MA60'] = t['close'].rolling(60).mean()
            t['MA20_Ref'] = t['close'].shift(20); t['Slope20'] = t['MA20'].diff()
        if not c.empty: c['net_buy'] = c['buy'] - c['sell']
        return t, c, m, r
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 5. UI 介面佈局 ---
st.title("🏹 超級分析師：Sponsor Pro 專業旗艦版")
target_sid = st.sidebar.text_input("輸入股票代碼", "1560")
my_cost = st.sidebar.number_input("您的買入成本", value=0.0)

if login_ok:
    t_df, c_df, m_df, r_df = get_stock_data_pro(target_sid)
    
    # 頂部即時指標 (Pro 快速反應)
    if not t_df.empty:
        last = t_df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
        with c2: st.metric("月線趨勢", "🟢 上揚" if last['Slope20'] > 0 else "🔴 下彎")
        with c3: 
            # 自動評分邏輯
            score = 0
            if last['close'] >= last['MA20']: score += 50
            if not c_df.empty and c_df['net_buy'].tail(3).sum() > 0: score += 50
            st.metric("核心診斷評分", f"{score} 分")
        with c4: st.write(f"📊 今日量：{int(last['Trading_Volume']/1000)}k")

    # 全功能 Tabs
    tabs = st.tabs(["🌊 資金流向/族群", "🚀 盤中爆量選股", "📉 量價扣抵圖", "🔥 籌碼照妖鏡", "📊 營收診斷"])
    
    with tabs[0]:
        st.subheader("🌊 Pro 級資金流向掃描 (十大族群)")
        sector_df = get_pro_sector_flow()
        if not sector_df.empty:
            fig_flow = px.bar(sector_df, x="族群", y="平均漲跌", color="平均漲跌", color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_flow, use_container_width=True)
            st.table(sector_df)

    with tabs[1]:
        st.subheader("🚀 盤中爆量突破偵測")
        # 利用 Pro 權限掃描更多種子
        seeds = ['1560', '2330', '2454', '2615', '2603', '3037', '2317', '3231', '2382']
        res = []
        for s in seeds:
            try:
                temp_t = dl.taiwan_stock_daily(stock_id=s, start_date=(datetime.now()-timedelta(days=10)).strftime("%Y-%m-%d"))
                if len(temp_t) > 5:
                    l = temp_t.iloc[-1]
                    vr = round(l['Trading_Volume'] / temp_t['Trading_Volume'].iloc[-6:-1].mean(), 2)
                    if vr > 1.2 and l['close'] > temp_t['close'].tail(20).mean():
                        res.append({'代號': s, '量能倍數': f"{vr}x", '現價': l['close'], '狀態': '🔥 爆量突破'})
            except: continue
        st.table(pd.DataFrame(res))

    with tabs[2]:
        if not t_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA(月)', line=dict(color='#FFFF00', width=3)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA(季)', line=dict(color='#FF00FF', width=2, dash='dot')))
            # 扣抵點
            fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[t_df['close'].shift(20).iloc[-1]], mode='markers', name='月扣抵', marker=dict(size=12, color='yellow', symbol='x')))
            st.plotly_chart(fig, use_container_width=True)
            
            # 風控
            if my_cost > 0:
                sl = round(my_cost * 0.93, 2)
                st.sidebar.error(f"🚨 停損線(-7%): {sl}")

    with tabs[3]:
        if not c_df.empty:
            st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣"), use_container_width=True)
        if not m_df.empty and 'MarginPurchaseStock' in m_df.columns:
            st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseStock', title="融資照妖鏡"), use_container_width=True)

    with tabs[4]:
        if not r_df.empty:
            st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收趨勢"), use_container_width=True)

else:
    st.error("登入失敗，請確認是否為 Sponsor Pro 權限之帳密。")
