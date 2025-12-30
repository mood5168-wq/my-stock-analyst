import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from FinMind.data import DataLoader
import time

# --- 1. 頁面設定 ---
st.set_page_config(page_title="超級分析師-Pro除錯版", layout="wide")

st.title("🔧 Sponsor Pro 深度除錯模式")

# --- 2. 登入檢查 (白盒模式) ---
dl = DataLoader()
user_id = st.secrets.get("FINMIND_USER_ID", None)
password = st.secrets.get("FINMIND_PASSWORD", None)

st.sidebar.header("1️⃣ 帳號檢測")
if user_id and password:
    try:
        dl.login(user_id=user_id, password=password)
        st.sidebar.success(f"✅ 登入 API 成功\nID: {str(user_id)[:3]}***")
    except Exception as e:
        st.sidebar.error(f"❌ 登入 API 失敗: {e}")
else:
    st.sidebar.error("❌ Secrets 未設定帳號密碼")

# --- 3. 數據抓取測試 (顯示詳細流程) ---
target_sid = st.sidebar.text_input("輸入測試代碼", "1560")
st.sidebar.markdown("---")

if st.button("🚀 開始診斷抓取"):
    st.subheader(f"正在診斷 {target_sid} 的數據鏈路...")
    
    # A. 測試抓取歷史日線
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    st.write(f"🔹 嘗試抓取日線數據 (Start: {start_date})...")
    
    try:
        t_df = dl.taiwan_stock_daily(stock_id=target_sid, start_date=start_date)
        if not t_df.empty:
            st.success(f"✅ 日線數據獲取成功！共 {len(t_df)} 筆。最後日期: {t_df['date'].iloc[-1]}")
            st.dataframe(t_df.tail(3))
        else:
            st.error("❌ 日線數據回傳為空 (Empty DataFrame)。可能原因：API 額度耗盡或該股無資料。")
    except Exception as e:
        st.error(f"❌ 抓取日線時發生崩潰錯誤: {e}")

    # B. 測試抓取即時快照 (Snapshot)
    st.write("🔹 嘗試抓取 Pro 即時快照 (Snapshot)...")
    try:
        snap_df = dl.taiwan_stock_daily_snapshot()
        if not snap_df.empty:
            target_snap = snap_df[snap_df['stock_id'] == target_sid]
            if not target_snap.empty:
                st.success(f"✅ 即時快照獲取成功！最新價: {target_snap['last_close'].iloc[0]}")
                st.dataframe(target_snap)
                
                # 嘗試整合
                if not t_df.empty:
                    st.info("💡 正在嘗試將快照合併入日線...")
                    today = datetime.now().strftime("%Y-%m-%d")
                    if t_df['date'].iloc[-1] != today:
                        new_row = t_df.iloc[-1].copy()
                        new_row['date'] = today
                        new_row['close'] = target_snap['last_close'].iloc[0]
                        new_row['Trading_Volume'] = target_snap['volume'].iloc[0]
                        t_df = pd.concat([t_df, pd.DataFrame([new_row])], ignore_index=True)
                        st.success("✅ 合併成功！日線圖已包含今日數據。")
            else:
                st.warning(f"⚠️ 快照 API 有回應，但找不到 {target_sid} 的資料 (可能今日未交易或代號錯誤)。")
        else:
            st.error("❌ 快照 API 回傳全空。FinMind 伺服器可能繁忙或權限不足。")
    except Exception as e:
        st.error(f"❌ 抓取快照時發生崩潰錯誤: {e}")

    # C. 繪圖測試
    if 't_df' in locals() and not t_df.empty:
        try:
            t_df['MA20'] = t_df['close'].rolling(20).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['close'], name='Price'))
            fig.add_trace(go.Scatter(x=t_df['date'], y=t_df['MA20'], name='MA20', line=dict(color='yellow')))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ 繪圖時發生錯誤: {e}")

# --- 4. 恢復全功能介面 (若上述測試通過) ---
st.markdown("---")
st.write("🔍 若上方診斷全綠，以下為完整功能區：")

# 這裡放入最穩定的全功能代碼，但加上了保護
if user_id and password:
    try:
        # 簡單載入完整功能，不隱藏錯誤
        t = dl.taiwan_stock_daily(stock_id=target_sid, start_date=(datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d"))
        if not t.empty:
             # 計算指標
            t['MA20'] = t['close'].rolling(20).mean()
            t['MA60'] = t['close'].rolling(60).mean()
            
            # 嘗試補快照
            try:
                snap = dl.taiwan_stock_daily_snapshot()
                tgt = snap[snap['stock_id'] == target_sid]
                if not tgt.empty and t['date'].iloc[-1] != datetime.now().strftime("%Y-%m-%d"):
                     new_row = t.iloc[-1].copy()
                     new_row['date'] = datetime.now().strftime("%Y-%m-%d")
                     new_row['close'] = tgt['last_close'].iloc[0]
                     new_row['Trading_Volume'] = tgt['volume'].iloc[0]
                     t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
            except: pass # 快照失敗不影響歷史圖

            # 顯示圖表
            st.subheader(f"📈 {target_sid} 最終圖表")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最新價", t['close'].iloc[-1])
            with col2:
                st.metric("相對量", round(t['Trading_Volume'].iloc[-1]/t['Trading_Volume'].iloc[-6:-1].mean(), 2))
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t['date'], y=t['close'], name='Close', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=t['date'], y=t['MA20'], name='20MA', line=dict(color='yellow')))
            fig.add_trace(go.Scatter(x=t['date'], y=t['MA60'], name='60MA', line=dict(color='magenta')))
            st.plotly_chart(fig, use_container_width=True)
            
            # 資金流向
            st.subheader("🌊 資金流向")
            sectors = {"半導體": ["2330","2454"], "AI": ["2382","3231"], "航運": ["2603","2615"]}
            res = []
            try:
                snap_all = dl.taiwan_stock_daily_snapshot()
                for k,v in sectors.items
