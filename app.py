import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# ---------- 工具函式 ----------
def normalize_date(df):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")
    return df

def get_snapshot_price_volume(df):
    price_col = next((c for c in ['last_close', 'close'] if c in df.columns), None)
    vol_col = next((c for c in ['Trading_Volume', 'volume', 'trade_volume'] if c in df.columns), None)
    return price_col, vol_col

# ---------- 頁面 ----------
st.set_page_config(page_title="超級分析師-Pro 除錯穩定版", layout="wide")
st.title("🔧 Sponsor Pro 深度除錯模式（修正版）")

# ---------- 登入 ----------
dl = DataLoader()
user_id = st.secrets.get("FINMIND_USER_ID")
password = st.secrets.get("FINMIND_PASSWORD")

st.sidebar.header("1️⃣ 帳號與 API 健康檢查")
api_ok = False

if user_id and password:
    try:
        dl.login(user_id=user_id, password=password)
        test = dl.taiwan_stock_daily(stock_id="2330", start_date="2024-01-01")
        if not test.empty:
            api_ok = True
            st.sidebar.success("✅ 登入完成，API 可正常回傳資料")
        else:
            st.sidebar.warning("⚠️ 登入成功，但 API 回傳為空（可能是額度 / 伺服器狀態）")
    except Exception as e:
        st.sidebar.error(f"❌ API 登入或測試失敗：{e}")
else:
    st.sidebar.error("❌ 未設定 FinMind Secrets")

# ---------- 代碼 ----------
target_sid = st.sidebar.text_input("輸入測試代碼", "1560")

# ---------- 診斷 ----------
diagnostic_pass = False

if st.button("🚀 開始診斷抓取") and api_ok:
    st.subheader(f"📡 診斷 {target_sid} 數據鏈路")

    # A. 日線
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    try:
        t_df = dl.taiwan_stock_daily(stock_id=target_sid, start_date=start_date)
        if t_df.empty:
            st.error("❌ 日線資料為空（可能為非交易日 / 下市 / API 異常）")
        else:
            t_df = normalize_date(t_df)
            st.success(f"✅ 日線成功：{len(t_df)} 筆，最後日期 {t_df['date'].iloc[-1]}")
            st.dataframe(t_df.tail(3))
            diagnostic_pass = True
    except Exception as e:
        st.error(f"❌ 日線抓取失敗：{e}")

    # B. Snapshot
    if diagnostic_pass:
        try:
            snap = dl.taiwan_stock_daily_snapshot()
            tgt = snap[snap['stock_id'] == target_sid]

            if tgt.empty:
                st.warning("⚠️ Snapshot 有回傳，但此股票目前無即時資料")
            else:
                price_col, vol_col = get_snapshot_price_volume(tgt)
                if not price_col:
                    st.error("❌ Snapshot 找不到價格欄位")
                else:
                    st.success(f"✅ Snapshot 成功：最新價 {tgt[price_col].iloc[0]}")
        except Exception as e:
            st.error(f"❌ Snapshot 抓取失敗：{e}")

# ---------- 完整功能 ----------
st.markdown("---")
st.subheader("📈 完整功能區（僅在診斷通過後啟用）")

if diagnostic_pass:
    t = t_df.copy()

    # 補 snapshot
    try:
        snap = dl.taiwan_stock_daily_snapshot()
        tgt = snap[snap['stock_id'] == target_sid]
        price_col, vol_col = get_snapshot_price_volume(tgt)

        today = datetime.now().strftime("%Y-%m-%d")
        if not tgt.empty and price_col and t['date'].iloc[-1] != today:
            new_row = t.iloc[-1].copy()
            new_row['date'] = today
            new_row['close'] = tgt[price_col].iloc[0]
            if vol_col:
                new_row['Trading_Volume'] = tgt[vol_col].iloc[0]
            t = pd.concat([t, pd.DataFrame([new_row])], ignore_index=True)
    except:
        pass

    # 指標
    t['MA20'] = t['close'].rolling(20).mean()
    t['MA60'] = t['close'].rolling(60).mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("最新價", round(t['close'].iloc[-1], 2))

    with col2:
        vol_base = t['Trading_Volume'].iloc[-6:-1].mean()
        if pd.notna(vol_base) and vol_base > 0:
            st.metric("相對量", round(t['Trading_Volume'].iloc[-1] / vol_base, 2))
        else:
            st.metric("相對量", "N/A")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t['date'], y=t['close'], name="Close"))
    fig.add_trace(go.Scatter(x=t['date'], y=t['MA20'], name="MA20"))
    fig.add_trace(go.Scatter(x=t['date'], y=t['MA60'], name="MA60"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("請先完成診斷流程，確認資料鏈路正常。")
