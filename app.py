import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader
import time

# --- 1. 頁面設定 (強制寬版) ---
st.set_page_config(page_title="超級分析師-急速救援", layout="wide")

# --- 2. 初始化與登入 (不使用 Cache，直接執行) ---
st.title("⚡ Sponsor Pro 急速救援戰情室")
status_text = st.empty() # 建立一個狀態顯示區
status_text.info("🚀 系統啟動中...正在連線 FinMind Pro...")

dl = DataLoader()
login_ok = False

# 嘗試登入
try:
    if "FINMIND_USER_ID" in st.secrets:
        dl.login(user_id=st.secrets["FINMIND_USER_ID"], password=st.secrets["FINMIND_PASSWORD"])
        login_ok = True
        # st.toast("✅ Sponsor Pro 登入成功！") # 輕量提示
    elif "FINMIND_TOKEN" in st.secrets:
        dl.login(token=st.secrets["FINMIND_TOKEN"].strip().strip('"'))
        login_ok = True
except Exception as e:
    st.error(f"登入失敗: {e}")

# --- 3. 核心函數：優先抓個股 (輕量級，保證秒開) ---
def get_stock_fast(sid):
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    
    # 1. 抓歷史日線
    t = dl.taiwan_stock_daily(stock_id=sid, start_date=start)
    
    # 2. 抓即時快照 (只抓這一檔，速度極快)
    try:
        snap = dl.taiwan_stock_daily_snapshot(stock_id=sid) # Pro 支援指定 stock_id 抓快照
        if not snap.empty:
            # 強制補丁
            if not t.empty and t['date'].iloc[-1] != today:
                new_row = t.iloc[-1].copy()
                new_row['date'] = today
                new_row['close'] = snap['last_close'].iloc[0]
                new_row['Trading_Volume'] = snap['volume'].iloc[0]
                t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
            return t, snap # 回傳日線與快照
    except: pass
    
    return t, pd.DataFrame()

# --- 4. 介面渲染 (分段執行) ---
target_sid = st.sidebar.text_input("輸入代碼", "1560")

if login_ok:
    status_text.info(f"🔍 正在獲取 {target_sid} 數據...")
    
    # [第一階段]：先顯示個股，確保畫面不轉圈
    t_df, snap_df = get_stock_fast(target_sid)
    
    if not t_df.empty:
        status_text.empty() # 清除讀取訊息
        last = t_df.iloc[-1]
        
        # 建立即時看板
        st.subheader(f"🎯 {target_sid} 個股診斷")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("最新成交價", f"${last['close']}", delta=f"{round(last['close']-t_df['close'].iloc[-2], 2)}")
        with c2:
            t_df['MA20'] = t_df['close'].rolling(20).mean()
            trend = "🟢 上揚" if t_df['MA20'].iloc[-1] > t_df['MA20'].iloc[-2] else "🔴 下彎"
            st.metric("月線趨勢", trend)
        with c3:
            # 如果有快照，計算相對量
            if not snap_df.empty:
                avg_v = t_df['Trading_Volume'].iloc[-6:-1].mean()
                rel_v = round(snap_df['volume'].iloc[0] / (avg_v+1), 2)
                st.metric("今日相對量", f"{rel_v}x")
            else:
                st.metric("今日相對量", "計算中...")

        # 繪圖
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='價格', line=dict(color='white')))
        fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='20MA', line=dict(color='#FFFF00', width=3)))
        t_df['MA60'] = t_df['close'].rolling(60).mean()
        fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA60'], name='60MA', line=dict(color='#FF00FF', width=2, dash='dot')))
        if len(t_df) > 21:
             fig.add_trace(go.Scatter(x=[t_df['date'].iloc[-21]], y=[t_df['close'].shift(20).iloc[-1]], mode='markers', name='扣抵', marker=dict(size=10, color='yellow', symbol='x')))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f"❌ 無法獲取 {target_sid} 資料。請檢查代碼或稍後再試。")

    st.markdown("---")
    
    # [第二階段]：最後才載入全市場資金流向 (避免卡住主畫面)
    if st.checkbox("顯示十大族群資金流向 (可能需載入 3-5 秒)", value=True):
        st.write("🌊 正在掃描全市場資金...")
        try:
            # 這裡我們只抓一次全市場快照
            all_snap = dl.taiwan_stock_daily_snapshot()
            
            if not all_snap.empty:
                sectors = {
                    "半導體": ["2330", "2454", "1560"], "AI伺服器": ["2382", "3231", "6669"],
                    "航運": ["2603", "2609", "2615"], "重電": ["1513", "1519"], 
                    "光通訊": ["4979", "3363"], "金融": ["2881", "2891"]
                }
                res = []
                for k, v in sectors.items():
                    sub = all_snap[all_snap['stock_id'].isin(v)]
                    if not sub.empty:
                        res.append({"族群": k, "漲跌幅%": round(sub['tv_change_rate'].mean(), 2), "熱度": int(sub['volume'].sum()/1000)})
                
                if res:
                    df_sec = pd.DataFrame(res).sort_values("漲跌幅%", ascending=False)
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.plotly_chart(px.bar(df_sec, x="族群", y="漲跌幅%", color="漲跌幅%", color_continuous_scale='RdYlGn'), use_container_width=True)
                    with col_b:
                        st.dataframe(df_sec, hide_index=True)
                    
                    # 順便顯示全台相對大量榜
                    st.subheader("🔥 全台相對大量榜")
                    all_snap['相對量'] = all_snap['volume'] / (all_snap['last_close_volume'] + 1)
                    st.dataframe(all_snap.sort_values('相對量', ascending=False).head(10)[['stock_id','stock_name','last_close','相對量']], use_container_width=True)
                else:
                    st.warning("查無族群資料。")
            else:
                st.warning("⚠️ 全市場快照暫無回應 (API 繁忙)。但上方個股功能不受影響。")
        except Exception as e:
            st.error(f"資金流向載入失敗: {e}")

else:
    st.error("⚠️ 請先設定 Secrets 進行登入。")
