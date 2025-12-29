import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-旗艦戰情室", layout="wide")

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

# --- 3. 核心功能：自動掃描強勢訊號 ---

@st.cache_data(ttl=3600)
def scan_super_signals():
    """AI 掃描儀：投信連買 + 站穩 20/60 均線"""
    if not login_ok: return pd.DataFrame(), ""
    results = []
    target_d = ""
    # 往回找 5 天內有資料的交易日
    for i in range(1, 6):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            # 抓取投信買超榜
            chip_df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", start_date=d, end_date=d)
            # 安全檢查：確保 API 有回傳資料欄位
            if chip_df is not None and not chip_df.empty and 'SITC_Trust' in chip_df.columns:
                target_d = d
                # 為了 API 穩定性，掃描前 15 名即可
                top_picks = chip_df.sort_values(by='SITC_Trust', ascending=False).head(15)
                for _, row in top_picks.iterrows():
                    sid = row['stock_id']
                    try:
                        # 檢查技術面位階
                        tech = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d"))
                        if tech is not None and len(tech) >= 60:
                            last_p = tech['close'].iloc[-1]
                            ma20 = tech['close'].tail(20).mean()
                            ma60 = tech['close'].tail(60).mean()
                            # 篩選：股價站上月線與季線
                            if last_p > ma20 and last_p > ma60:
                                results.append({
                                    '代號': sid, '名稱': row['stock_name'], '現價': last_p,
                                    '投信買超(張)': row['SITC_Trust'],
                                    '趨勢': '🔥 強勢多頭' if last_p > tech['close'].tail(5).mean() else '⚖️ 橫盤整理'
                                })
                    except: continue
                if results: break
        except: continue
    return pd.DataFrame(results), target_d

@st.cache_data(ttl=600)
def get_stock_details(sid):
    """個股詳細診斷資料"""
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    try:
        tech = dl.taiwan_stock_daily(stock_id=sid, start_date=start_date)
        rev = dl.taiwan_stock_month_revenue(stock_id=sid, start_date=start_date)
        chip = dl.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
        if not tech.empty:
            tech['MA5'] = tech['close'].rolling(5).mean()
            tech['MA20'] = tech['close'].rolling(20).mean()
            tech['MA60'] = tech['close'].rolling(60).mean()
        if not chip.empty:
            chip['net_buy'] = chip['buy'] - chip['sell']
        return tech, rev, chip
    except: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 4. UI 介面 ---
st.title("🏹 超級分析師：旗艦整合戰情室")

# 側邊欄診斷區
st.sidebar.header("🎯 持股診斷")
target_sid = st.sidebar.text_input("輸入股票代號", "2330")
my_buy_price = st.sidebar.number_input("您的買入成本", value=0.0)

# 分頁佈局
tab0, tab1, tab2, tab3 = st.tabs(["🚀 超級強勢訊號", "📈 量價技術面", "🔥 法人籌碼", "📊 營收診斷"])

if login_ok:
    # 執行全台股掃描 (Tab 0)
    with tab0:
        st.subheader("🌟 專家精選：投信鎖碼 + 多頭排列")
        with st.spinner('AI 正在分析市場數據...'):
            sig_df, sig_date = scan_super_signals()
            if not sig_df.empty:
                st.write(f"📅 資料日期：{sig_date}")
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
                st.success("以上標的符合：投信大買、股價在月線與季線之上。")
            else:
                st.warning("目前市場環境較弱，暫無符合強勢訊號之標的。")

    # 執行個股深度診斷 (Tab 1-3)
    t_df, r_df, c_df = get_stock_details(target_sid)
    
    with tab1:
        if not t_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='收盤價', line=dict(color='white', width=2)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA(月線)', line=dict(color='cyan', width=1)))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA(季線)', line=dict(color='orange', width=2, dash='dot')))
            fig.update_layout(template="plotly_dark", height=400, title=f"{target_sid} 三線走勢圖")
            st.plotly_chart(fig, use_container_width=True)
            
            # 風控提示
            last_p = t_df['close'].iloc[-1]
            st.sidebar.metric("當前現價", f"${last_p}", delta=f"{round(last_p-my_buy_price, 2)}" if my_buy_price > 0 else None)
            if my_buy_price > 0:
                sl = round(my_buy_price * 0.93, 2)
                st.sidebar.write(f"🛑 停損參考價(-7%): **{sl}**")
                if last_p <= sl: st.sidebar.error("🚨 已觸發停損！")
        else: st.info("請輸入代號查看技術圖表")

    with tab2:
        if not c_df.empty:
            p_df = c_df[c_df['name'].isin(['Foreign_Investor', 'Investment_Trust'])]
            fig_c = px.bar(p_df, x='date', y='net_buy', color='name', barmode='group', title="法人淨買賣超(紅進綠出)")
            fig_c.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig_c, use_container_width=True)

    with tab3:
        if not r_df.empty:
            st.plotly_chart(px.bar(r_df, x='revenue_month', y='revenue', title="月營收趨勢"), use_container_width=True)

else:
    st.error("API 登入失敗，請確認 Secrets 設定。")
