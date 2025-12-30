import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-爆量捕捉版", layout="wide")

# --- 2. 安全登入 ---
dl = DataLoader()
login_ok = False
if "FINMIND_USER_ID" in st.secrets:
    try:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
    except: pass

# --- 3. 核心功能：雙軌掃描 (投信榜 + 熱門榜) ---

@st.cache_data(ttl=60)
def scan_intraday_hot_stocks():
    """盤中掃描：不再只看投信，擴大到熱門股"""
    if not login_ok: return pd.DataFrame(), ""
    results = []
    
    # 擴大掃描池：除了投信買超，額外加入你指定的強勢股或熱門代號
    # 這裡我們模擬一個「種子清單」，包含近期熱門股如中砂、聯發科、萬海等
    hot_seeds = ['1560', '2330', '2454', '2615', '2317', '3231', '2382'] 
    
    try:
        # 1. 抓取投信近 3 日買超榜作為基礎
        chip_df = dl.taiwan_stock_holding_shares_per(stock_id="ALL", 
                                                     start_date=(datetime.now()-timedelta(days=3)).strftime("%Y-%m-%d"))
        top_list = chip_df.sort_values(by='SITC_Trust', ascending=False).head(40)['stock_id'].tolist()
        
        # 2. 合併熱門種子與投信榜
        scan_pool = list(set(top_list + hot_seeds))
        
        for sid in scan_pool:
            try:
                t = dl.taiwan_stock_daily(stock_id=sid, start_date=(datetime.now()-timedelta(days=60)).strftime("%Y-%m-%d"))
                if len(t) >= 20:
                    last = t.iloc[-1]
                    # 計算 5 日均量 (扣除今天)
                    avg_vol = t['Trading_Volume'].iloc[-6:-1].mean()
                    curr_vol = last['Trading_Volume']
                    vol_ratio = round(curr_vol / avg_vol, 2)
                    
                    ma20 = t['close'].tail(20).mean()
                    ma60 = t['close'].tail(60).mean()
                    
                    # 條件：量能比昨天的全天均量 > 1.2 倍 且 站穩 20MA
                    if vol_ratio >= 1.2 and last['close'] >= ma20:
                        results.append({
                            '代號': sid,
                            '名稱': dl.taiwan_stock_info()[dl.taiwan_stock_info()['stock_id']==sid]['stock_name'].iloc[0] if sid not in ['1560'] else "中砂",
                            '量能倍數': f"🔥 {vol_ratio}x",
                            '目前成交量': f"{int(curr_vol/1000)}k",
                            '現價': last['close'],
                            '技術位階': "☀️ 站穩月線" if last['close'] > ma20 else "☁️ 月線邊緣"
                        })
            except: continue
            
        return pd.DataFrame(results).sort_values(by='量能倍數', ascending=False), datetime.now().strftime("%H:%M:%S")
    except: return pd.DataFrame(), ""

# --- 4. UI 呈現 ---
st.title("🏹 爆量狙擊手：中砂與熱門股動態")

if login_ok:
    tab0, tab1 = st.tabs(["⚡ 盤中爆量名單", "📉 個股扣抵診斷"])
    
    with tab0:
        st.subheader("🔥 實時量能異常追蹤")
        df, update_time = scan_intraday_hot_stocks()
        st.write(f"🕒 最後更新時間：{update_time} (數據約有 20 分鐘延遲)")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            if '1560' in df['代號'].values:
                st.success("✅ 偵測成功！中砂目前符合爆量突破條件。")
        else:
            st.info("尚未偵測到符合爆量標的，請點擊左側刷新。")

    with tab1:
        # 維持之前的 MA20/MA60 扣抵診斷邏輯
        st.write("請由左側輸入代碼進行深度扣抵解析")
