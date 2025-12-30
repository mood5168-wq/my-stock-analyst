import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader
import time

# --- 1. 基礎設定 ---
st.set_page_config(page_title="超級分析師-Pro終極版", layout="wide")

# --- 2. 登入與 API 初始化 ---
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

# --- 3. 核心工具：安全抓取即時 Tick ---
def get_tick_safe(sid):
    """最底層的安全抓取函數，保證不崩潰"""
    try:
        # 抓取今日即時 Tick
        df = dl.taiwan_stock_tick(stock_id=sid, date=datetime.now().strftime("%Y-%m-%d"))
        if not df.empty:
            return df['deal_price'].iloc[-1], df['volume'].sum()
    except: pass
    return None, None

# --- 4. 功能模組 A：十大族群資金流向 (手動遍歷模式) ---
@st.cache_data(ttl=60)
def get_sector_flow_manual():
    # 預設空表，防止 UI 渲染時報錯
    empty_df = pd.DataFrame(columns=["族群", "平均漲跌%", "熱度(張)"])
    if not login_ok: return empty_df

    sectors = {
        "半導體": ["2330", "2454", "1560", "3131"],
        "AI伺服器": ["2382", "3231", "2376", "6669"],
        "航運": ["2603", "2609", "2615", "2605"],
        "重電": ["1513", "1519", "1503", "1514"],
        "光通訊": ["4979", "3363", "6451", "3081"],
        "金融": ["2881", "2882", "2891", "2886"]
    }
    
    results = []
    # 使用 Pro 權限逐檔掃描
    for name, sids in sectors.items():
        chg_list = []
        vol_total = 0
        for sid in sids:
            # 1. 抓即時
            p, v = get_tick_safe(sid)
            if p:
                # 2. 抓昨日收盤算漲跌
                try:
                    h = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=5)).strftime("%Y-%m-%d"))
                    if not h.empty:
                        # 確保有足夠歷史資料
                        prev = h['close'].iloc[-2] if len(h) > 1 else h['close'].iloc[-1]
                        if prev > 0:
                            chg = (p - prev) / prev * 100
                            chg_list.append(chg)
                            vol_total += v
                except: pass
        
        if chg_list:
            results.append({
                "族群": name,
                "平均漲跌%": round(sum(chg_list)/len(chg_list), 2),
                "熱度(張)": int(vol_total/1000)
            })
    
    if not results: return empty_df
    return pd.DataFrame(results).sort_values("平均漲跌%", ascending=False)

# --- 5. 功能模組 B：個股深度全資料 ---
@st.cache_data(ttl=30)
def get_stock_data_safe(sid):
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d") # 抓長一點避免營收 IndexError
    
    # 1. 基礎資料抓取 (即時容錯)
    try:
        t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
        c = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start)
        m = dl.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date=start)
        r = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start)
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # 2. 即時補丁 (Tick Patch)
    rt_p, rt_v = get_tick_safe(sid)
    if rt_p and not t.empty:
        # 如果最後一筆不是今天，補上去
        if t['date'].iloc[-1] != today:
            new_row = t.iloc[-1].copy()
            new_row['date'] = today
            new_row['close'] = rt_p
            new_row['Trading_Volume'] = rt_v
            t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
    
    # 3. 技術指標計算 (安全檢查)
    if not t.empty and len(t) > 20:
        t['MA20'] = t['close'].rolling(20).mean()
        t['MA60'] = t['close'].rolling(60).mean()
        t['Slope20'] = t['MA20'].diff()
        t['MA20_Ref'] = t['close'].shift(20)
    
    # 4. 籌碼整理
    if not c.empty: c['net_buy'] = c['buy'] - c['sell']

    return t, c, m, r

def run_safe_score(t, c, m, r):
    """絕對安全的評分系統"""
    score = 0
    msgs = []
    
    # 技術面
    if not t.empty and 'MA20' in t.columns:
        last = t.iloc[-1]
        if not pd.isna(last['MA20']) and last['close'] > last['MA20']:
            score += 30; msgs.append("✅ 站上月線")
            
    # 籌碼面
    if not c.empty and len(c) >= 3:
        if c['net_buy'].tail(3).sum() > 0:
            score += 30; msgs.append("✅ 投信買超")
            
    # 營收面 (IndexError 修復點)
    if not r.empty:
        if len(r) >= 13: # 只有資料夠長才比
            if r['revenue'].iloc[-1] > r['revenue'].iloc[-13]:
                score += 40; msgs.append("✅ 營收年增")
        else:
            score += 10; msgs.append("⚠️ 資料不足(新股)")
            
    return score, msgs

# --- 6. UI 介面 ---
st.title("🏹 超級分析師：Sponsor Pro 終極穩定版")

if login_ok:
    st.sidebar.success(f"✅ Pro 連線成功 ({str(user_id)[:3]}***)")
    target_sid = st.sidebar.text_input("輸入代碼", "1560")
    if st.sidebar.button("🔄 重整數據"): st.cache_data.clear()

    # A. 資金流向
    st.subheader("🌊 十大族群資金流向 (Pro 即時)")
    flow_df = get_sector_flow_manual()
    if not flow_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1: st.plotly_chart(px.bar(flow_df, x="族群", y="平均漲跌%", color="平均漲跌%", color_continuous_scale='RdYlGn', text="平均漲跌%"), use_container_width=True)
        with c2: st.dataframe(flow_df, hide_index=True, use_container_width=True)
    else:
        st.info("⌛ 盤中資料讀取中... (若今日未開盤則無數據)")

    st.markdown("---")

    # B. 個股診斷
    t_df, c_df, m_df, r_df = get_stock_data_safe(target_sid)
    
    if not t_df.empty:
        last = t_df.iloc[-1]
        st.markdown(f"### 🎯 {target_sid} 深度分析")
        
        # 儀表板
        col1, col2, col3, col4 = st.columns(4)
        with col1: 
            st.metric("最新價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
            st.caption(f"日期: {last['date']}")
        with col2:
            score, details = run_safe_score(t_df, c_df, m_df, r_df)
            st.metric("綜合評分", f"{score} 分")
        with col3:
            trend = "🟢 上揚" if last.get('Slope20', 0) > 0 else "🔴 下彎"
            st.metric("月線趨勢", trend)
        with col4:
            # 相對量能 (安全計算)
            avg_v = t_df['Trading_Volume'].iloc[-6:-1].mean()
            curr_v = last['Trading_Volume']
            st.metric("相對量能", f"{round(curr_v/(avg_v+1), 2)}x")
            
        st.write(" | ".join(details))

        # 功能頁籤
        tabs = st.tabs(["📉 技術三線", "🔥 籌碼/融資", "📊 營收"])
        
        with tabs[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white')))
            if 'MA20' in t_df.columns: fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
            if 'MA60' in t_df.columns: fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
            # 扣抵值
            if len(t_df) > 21: fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[t_df['close'].shift(20).iloc[-1]], mode='markers', name='扣抵', marker=dict(size=10, color='yellow', symbol='x')))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            # 籌碼圖
            if not c_df.empty: 
                st.plotly_chart(px.bar(c_df[c_df['name'].isin(['Foreign_Investor','Investment_Trust'])], x='date', y='net_buy', color='name', barmode='group', title="法人買賣超"), use_container_width=True)
            
            # 融資圖 (已修正欄位名稱: MarginPurchaseTodayBalance)
            if not m_df.empty and 'MarginPurchaseTodayBalance' in m_df.columns:
                st.plotly_chart(px.line(m_df, x='date', y='MarginPurchaseTodayBalance', title="融資餘額"), use_container_width=True)
            elif not m_df.empty:
                st.warning("查無融資餘額欄位，請確認資料來源。")

        with tabs[2]:
            if not r_df.empty: st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="營收"), use_container_width=True)
            else: st.info("暫無營收資料")
            
    else:
        st.error(f"⚠️ 無法讀取 {target_sid}。可能原因：代號錯誤、今日未開盤、或 API 暫時無回應。")
else:
    st.error("❌ 請檢查 Secrets 設定 Sponsor Pro 帳號。")
