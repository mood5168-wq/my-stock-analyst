import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# -------------------------
# 工具函式
# -------------------------
def normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and not df.empty and "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df

def pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# -------------------------
# Streamlit 基本設定（必須放最前面）
# -------------------------
st.set_page_config(page_title="超級分析師-Pro 除錯穩定版", layout="wide")
st.title("Sponsor Pro 深度除錯模式（穩定版）")

# -------------------------
# Session State 初始化（避免 rerun 洗掉狀態）
# -------------------------
if "api_ok" not in st.session_state:
    st.session_state.api_ok = False
if "diagnostic_pass" not in st.session_state:
    st.session_state.diagnostic_pass = False
if "t_df" not in st.session_state:
    st.session_state.t_df = pd.DataFrame()
if "last_sid" not in st.session_state:
    st.session_state.last_sid = ""

# -------------------------
# 登入 / API 健康檢查
# -------------------------
st.sidebar.header("1) 帳號與 API 健康檢查")

dl = DataLoader()

finmind_token = st.secrets.get("FINMIND_TOKEN", "")
user_id = st.secrets.get("FINMIND_USER_ID", "")
password = st.secrets.get("FINMIND_PASSWORD", "")

def do_login_and_test() -> tuple[bool, str]:
    """
    回傳 (api_ok, message)
    """
    try:
        # 優先用 token（v4 建議用法）:contentReference[oaicite:5]{index=5}
        if finmind_token:
            dl.login_by_token(api_token=finmind_token)
        elif user_id and password:
            dl.login(user_id=user_id, password=password)
        else:
            return False, "未設定 FINMIND_TOKEN 或 FINMIND_USER_ID/FINMIND_PASSWORD（請在 .streamlit/secrets.toml 設定）"

        # 測試日線：抓 2330 近一段資料
        test = dl.taiwan_stock_daily(stock_id="2330", start_date="2024-01-01")
        if test is None or test.empty:
            return False, "登入成功，但日線測試回傳為空（可能是 API 異常、網路、或 FinMind 服務端狀態）"
        return True, f"登入成功，日線測試 OK（筆數 {len(test)}）"
    except Exception as e:
        return False, f"登入/測試失敗：{type(e).__name__}: {e}"

if st.sidebar.button("重新檢查登入 / API"):
    ok, msg = do_login_and_test()
    st.session_state.api_ok = ok
    st.sidebar.success(msg) if ok else st.sidebar.error(msg)

# 首次載入也跑一次（避免使用者不知道要按）
if st.session_state.api_ok is False and (finmind_token or (user_id and password)):
    ok, msg = do_login_and_test()
    st.session_state.api_ok = ok
    st.sidebar.success(msg) if ok else st.sidebar.error(msg)
elif not (finmind_token or (user_id and password)):
    st.sidebar.error("請先設定 Secrets：FINMIND_TOKEN（建議）或 FINMIND_USER_ID / FINMIND_PASSWORD")

st.sidebar.divider()

# -------------------------
# 參數輸入
# -------------------------
target_sid = st.sidebar.text_input("輸入測試代碼", value=(st.session_state.last_sid or "1560")).strip()
st.session_state.last_sid = target_sid

# -------------------------
# 診斷流程（按鈕觸發）
# -------------------------
st.sidebar.header("2) 診斷")
run_diag = st.sidebar.button("開始診斷抓取")

if run_diag:
    st.session_state.diagnostic_pass = False
    st.session_state.t_df = pd.DataFrame()

    if not st.session_state.api_ok:
        st.error("API 尚未通過健康檢查，請先在左側完成登入/測試。")
    else:
        st.subheader(f"診斷：{target_sid} 數據鏈路")

        # A) 日線
        start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
        try:
            t_df = dl.taiwan_stock_daily(stock_id=target_sid, start_date=start_date)
            if t_df is None or t_df.empty:
                st.error("日線資料為空（可能：代碼錯誤、下市、或該區間無資料）")
            else:
                t_df = normalize_date(t_df)
                st.success(f"日線成功：{len(t_df)} 筆，最後日期 {t_df['date'].iloc[-1]}")
                st.dataframe(t_df.tail(5), use_container_width=True)
                st.session_state.t_df = t_df
                st.session_state.diagnostic_pass = True
        except Exception as e:
            st.error(f"日線抓取失敗：{type(e).__name__}: {e}")
            st.exception(e)

        # B) 即時 Snapshot（改用官方即時資料：taiwan_stock_tick_snapshot）:contentReference[oaicite:6]{index=6}
        if st.session_state.diagnostic_pass:
            try:
                snap = dl.taiwan_stock_tick_snapshot(stock_id=target_sid)
                if snap is None or snap.empty:
                    st.warning("即時 Snapshot 回傳為空（可能：該商品暫無即時、非交易時段、或權限未開）")
                else:
                    # 不假設欄位名稱，盡量自動辨識
                    price_col = pick_first_col(snap, ["last_price", "lastPrice", "price", "close", "last_close"])
                    vol_col = pick_first_col(snap, ["volume", "trade_volume", "Trading_Volume", "totalVolume", "accVolume"])

                    st.write("Snapshot 欄位預覽：", list(snap.columns))
                    if price_col:
                        st.success(f"Snapshot 成功：{price_col} = {snap[price_col].iloc[0]}")
                    else:
                        st.warning("Snapshot 有資料，但找不到可辨識的價格欄位（請看上方欄位預覽，指定欄位名即可）")

                    if vol_col:
                        st.info(f"Volume：{vol_col} = {snap[vol_col].iloc[0]}")
            except Exception as e:
                st.error(f"Snapshot 抓取失敗：{type(e).__name__}: {e}")
                st.exception(e)

st.markdown("---")
st.subheader("完整功能區（診斷通過後啟用）")

if not st.session_state.diagnostic_pass:
    st.info("請先在左側按「開始診斷抓取」，並確認日線可正常回傳。")
else:
    t = st.session_state.t_df.copy()

    # 指標（欄位容錯）
    close_col = pick_first_col(t, ["close", "Close", "adj_close", "Adj Close"])
    vol_col = pick_first_col(t, ["Trading_Volume", "volume", "trade_volume"])

    if not close_col:
        st.error(f"日線資料找不到 close 欄位。現有欄位：{list(t.columns)}")
        st.stop()

    t["MA20"] = pd.to_numeric(t[close_col], errors="coerce").rolling(20).mean()
    t["MA60"] = pd.to_numeric(t[close_col], errors="coerce").rolling(60).mean()

    c1, c2, c3 = st.columns(3)
    last_price = safe_float(t[close_col].iloc[-1])
    c1.metric("最新收盤價", "N/A" if last_price is None else round(last_price, 2))

    if vol_col:
        vol_base = pd.to_numeric(t[vol_col], errors="coerce").iloc[-6:-1].mean()
        vol_now = pd.to_numeric(t[vol_col], errors="coerce").iloc[-1]
        if pd.notna(vol_base) and vol_base > 0 and pd.notna(vol_now):
            c2.metric("相對量", round(float(vol_now) / float(vol_base), 2))
        else:
            c2.metric("相對量", "N/A")
    else:
        c2.metric("相對量", "N/A（無成交量欄位）")

    c3.write(f"資料區間：{t['date'].iloc[0]} ~ {t['date'].iloc[-1]}") if "date" in t.columns else c3.write("資料區間：N/A")

    fig = go.Figure()
    x = t["date"] if "date" in t.columns else list(range(len(t)))
    fig.add_trace(go.Scatter(x=x, y=t[close_col], name="Close"))
    fig.add_trace(go.Scatter(x=x, y=t["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=x, y=t["MA60"], name="MA60"))
    st.plotly_chart(fig, use_container_width=True)
