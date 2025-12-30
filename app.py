import os
from datetime import datetime, timedelta
from typing import Optional, Union, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# Streamlit: set_page_config must be the first Streamlit command
st.set_page_config(page_title="超級分析師-Pro（七大功能完整版）", layout="wide")

# -----------------------------
# Timezone
# -----------------------------
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ = ZoneInfo("Asia/Taipei")
except Exception:
    TZ = None  # fallback


# -----------------------------
# FinMind REST helpers (stable + timeout)
# -----------------------------
FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TICK_SNAPSHOT_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_tick_snapshot"


def _now_date_str() -> str:
    if TZ:
        return datetime.now(tz=TZ).strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _as_data_id_param(data_id: Optional[Union[str, Iterable[str]]]):
    if data_id is None:
        return None
    if isinstance(data_id, str):
        return data_id
    return list(data_id)


def _raise_if_api_error(payload: dict, context: str):
    status = payload.get("status", None)
    # Some responses may not include status; treat as OK
    if status is not None and int(status) != 200:
        raise RuntimeError(f"{context} API error: status={status}, msg={payload.get('msg')}")


def finmind_get_data(
    token: str,
    dataset: str,
    data_id: Optional[Union[str, Iterable[str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    params = {"dataset": dataset}
    did = _as_data_id_param(data_id)
    if did is not None and did != "":
        params["data_id"] = did
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    resp = requests.get(FINMIND_DATA_URL, headers=_headers(token), params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    _raise_if_api_error(payload, f"dataset={dataset}")
    df = pd.DataFrame(payload.get("data", []))
    return df


def finmind_tick_snapshot(
    token: str,
    data_id: Optional[Union[str, Iterable[str]]] = None,
    timeout: int = 15,
) -> pd.DataFrame:
    # FinMind docs: data_id can be single, list, or "" for all.
    params = {"data_id": ""} if data_id is None else {"data_id": _as_data_id_param(data_id)}
    resp = requests.get(FINMIND_TICK_SNAPSHOT_URL, headers=_headers(token), params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    _raise_if_api_error(payload, "tick_snapshot")
    df = pd.DataFrame(payload.get("data", []))
    return df


# -----------------------------
# Small utilities
# -----------------------------
def normalize_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    df = df.copy()
    df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
    return df


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_div(a, b, default=np.nan):
    try:
        if b == 0 or pd.isna(b):
            return default
        return a / b
    except Exception:
        return default


# -----------------------------
# Cached downloads
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def get_stock_info_cached(token: str) -> pd.DataFrame:
    # TaiwanStockInfo: stock_id / stock_name / industry_category / type / date
    df = finmind_get_data(token, dataset="TaiwanStockInfo", timeout=30)
    return df


@st.cache_data(ttl=10)
def get_snapshot_all_cached(token: str) -> pd.DataFrame:
    # data_id="" => all
    return finmind_tick_snapshot(token, data_id="", timeout=15)


@st.cache_data(ttl=10)
def get_snapshot_one_cached(token: str, stock_id: str) -> pd.DataFrame:
    return finmind_tick_snapshot(token, data_id=stock_id, timeout=15)


@st.cache_data(ttl=600)
def get_daily_one_cached(token: str, stock_id: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    df = finmind_get_data(token, dataset="TaiwanStockPrice", data_id=stock_id, start_date=start_date, end_date=end_date, timeout=40)
    return normalize_date_col(df, "date")


@st.cache_data(ttl=3600)
def get_trading_dates_cached(token: str) -> pd.DataFrame:
    df = finmind_get_data(token, dataset="TaiwanStockTradingDate", timeout=40)
    return normalize_date_col(df, "date")


@st.cache_data(ttl=3600)
def get_institutional_cached(token: str, stock_id: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    df = finmind_get_data(
        token,
        dataset="TaiwanStockInstitutionalInvestorsBuySell",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        timeout=40,
    )
    return normalize_date_col(df, "date")


@st.cache_data(ttl=3600)
def get_margin_cached(token: str, stock_id: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    df = finmind_get_data(
        token,
        dataset="TaiwanStockMarginPurchaseShortSale",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        timeout=40,
    )
    return normalize_date_col(df, "date")


@st.cache_data(ttl=12 * 3600)
def get_month_revenue_cached(token: str, stock_id: str, start_date: str) -> pd.DataFrame:
    df = finmind_get_data(token, dataset="TaiwanStockMonthRevenue", data_id=stock_id, start_date=start_date, timeout=40)
    return normalize_date_col(df, "date")


@st.cache_data(ttl=6 * 3600)
def get_daily_all_cached(token: str, date_str: str) -> pd.DataFrame:
    # TaiwanStockPrice with start_date only => all stocks for that date (backer/sponsor)
    df = finmind_get_data(token, dataset="TaiwanStockPrice", start_date=date_str, timeout=60)
    return normalize_date_col(df, "date")


# -----------------------------
# 12/30 realtime patch (Pro snapshot forced)
# -----------------------------
def patch_daily_with_snapshot(daily_df: pd.DataFrame, snap_row: pd.Series) -> tuple[pd.DataFrame, bool, str]:
    """
    Patch TaiwanStockPrice daily DF with tick snapshot (intraday) to force today's row.
    Returns (patched_df, patched_flag, patched_date)
    """
    if daily_df is None or daily_df.empty:
        return daily_df, False, ""

    daily_df = normalize_date_col(daily_df.copy(), "date")

    snap_dt = pd.to_datetime(snap_row.get("date"))
    snap_date = snap_dt.strftime("%Y-%m-%d") if not pd.isna(snap_dt) else _now_date_str()

    last_date = str(daily_df["date"].iloc[-1])
    if last_date == snap_date:
        return daily_df, False, snap_date

    new_row = {c: np.nan for c in daily_df.columns}
    new_row["date"] = snap_date

    # Snapshot -> TaiwanStockPrice mapping
    for k_from, k_to in [
        ("open", "open"),
        ("high", "max"),
        ("low", "min"),
        ("close", "close"),
        ("total_volume", "Trading_Volume"),
        ("total_amount", "Trading_money"),
        ("change_price", "spread"),
    ]:
        if k_to in new_row and k_from in snap_row.index:
            new_row[k_to] = snap_row.get(k_from)

    daily_df = pd.concat([daily_df, pd.DataFrame([new_row])], ignore_index=True)
    return daily_df, True, snap_date


# -----------------------------
# Score system (0-100): 4 dimensions
# -----------------------------
def compute_score(t: pd.DataFrame) -> dict:
    """
    Returns dict:
      total, trend, momentum, volume, chip (0-25 each), plus notes.
    """
    out = {"trend": 0.0, "momentum": 0.0, "volume": 0.0, "chip": 0.0, "total": 0.0, "notes": []}
    if t is None or t.empty or "close" not in t.columns:
        out["notes"].append("缺少股價資料")
        return out

    df = t.copy()
    df["close"] = to_numeric_series(df["close"])
    if "Trading_Volume" in df.columns:
        df["Trading_Volume"] = to_numeric_series(df["Trading_Volume"])
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA60"] = df["close"].rolling(60).mean()

    last = df.iloc[-1]
    c = float(last["close"]) if pd.notna(last["close"]) else np.nan
    ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan
    ma60 = float(last["MA60"]) if pd.notna(last["MA60"]) else np.nan

    # Trend (0-25)
    trend = 0
    if pd.notna(c) and pd.notna(ma20) and pd.notna(ma60):
        if c > ma20 > ma60:
            trend = 25
        elif c > ma20 and ma20 >= ma60:
            trend = 20
        elif c > ma20:
            trend = 15
        elif c > ma60:
            trend = 10
        else:
            trend = 5
        if len(df) >= 25:
            ma20_slope = df["MA20"].iloc[-1] - df["MA20"].iloc[-6]
            if pd.notna(ma20_slope) and ma20_slope > 0:
                trend = min(25, trend + 2)
    out["trend"] = float(trend)

    # Momentum (0-25)
    momentum = 0
    if len(df) >= 11 and pd.notna(df["close"].iloc[-11]) and pd.notna(c):
        r10 = c / float(df["close"].iloc[-11]) - 1.0
        if r10 >= 0.10:
            momentum = 25
        elif r10 >= 0.05:
            momentum = 20
        elif r10 >= 0.02:
            momentum = 15
        elif r10 >= 0.0:
            momentum = 10
        elif r10 >= -0.02:
            momentum = 5
        else:
            momentum = 0
        if pd.notna(ma20) and ma20 > 0:
            dist = abs(c / ma20 - 1.0)
            if dist > 0.12:
                momentum = max(0, momentum - 5)
    out["momentum"] = float(momentum)

    # Volume (0-25): today vs last 5 days
    vol_score = 0
    if "Trading_Volume" in df.columns and len(df) >= 6:
        v_now = df["Trading_Volume"].iloc[-1]
        v_base = df["Trading_Volume"].iloc[-6:-1].mean()
        vr = safe_div(v_now, v_base, default=np.nan)
        if pd.notna(vr):
            if vr >= 2.0:
                vol_score = 25
            elif vr >= 1.5:
                vol_score = 20
            elif vr >= 1.2:
                vol_score = 15
            elif vr >= 1.0:
                vol_score = 10
            else:
                vol_score = 5
    out["volume"] = float(vol_score)

    out["total"] = float(out["trend"] + out["momentum"] + out["volume"] + out["chip"])
    return out


def compute_chip_score(inst_df: pd.DataFrame, margin_df: pd.DataFrame) -> tuple[float, list[str]]:
    """
    Chip score 0-25 based on:
      - last 5 days institutional net buy
      - margin balance trend
    """
    notes: list[str] = []
    score = 0.0

    if inst_df is not None and not inst_df.empty:
        tmp = inst_df.copy()
        for c in ["buy", "sell"]:
            if c in tmp.columns:
                tmp[c] = to_numeric_series(tmp[c]).fillna(0.0)
        if {"date", "buy", "sell"}.issubset(tmp.columns):
            tmp["net"] = tmp["buy"] - tmp["sell"]
            daily_net = tmp.groupby("date", as_index=False)["net"].sum().sort_values("date")
            inst_net_5 = float(daily_net.tail(5)["net"].sum()) if len(daily_net) else 0.0
            if inst_net_5 > 0:
                score += 15
                notes.append(f"法人近5日偏買超（合計 {inst_net_5:,.0f}）")
            elif inst_net_5 < 0:
                score += 5
                notes.append(f"法人近5日偏賣超（合計 {inst_net_5:,.0f}）")
            else:
                score += 8
                notes.append("法人近5日買賣超接近平衡")

    if margin_df is not None and not margin_df.empty and "date" in margin_df.columns:
        m = margin_df.copy().sort_values("date")
        bal_col = None
        for cand in ["MarginPurchaseTodayBalance", "TodayBalance", "margin_purchase_today_balance"]:
            if cand in m.columns:
                bal_col = cand
                break
        if bal_col:
            m[bal_col] = to_numeric_series(m[bal_col])
            last6 = m.tail(6)
            if len(last6) >= 2 and pd.notna(last6[bal_col].iloc[-1]) and pd.notna(last6[bal_col].iloc[0]):
                delta = float(last6[bal_col].iloc[-1] - last6[bal_col].iloc[0])
                if delta < 0:
                    score += 10
                    notes.append("融資餘額下降（散戶槓桿降溫）")
                elif abs(delta) < 0.01 * max(1.0, float(last6[bal_col].iloc[-1])):
                    score += 8
                    notes.append("融資餘額持平")
                else:
                    score += 4
                    notes.append("融資餘額上升（散戶槓桿升溫）")

    score = float(max(0.0, min(25.0, score)))
    return score, notes


# -----------------------------
# Market scan: sector flow + relative volume ranking
# -----------------------------
def compute_sector_flow_from_snapshot(snapshot_all: pd.DataFrame, stock_info: pd.DataFrame) -> pd.DataFrame:
    df = snapshot_all.copy()
    df["stock_id"] = df["stock_id"].astype(str)

    info = stock_info[["stock_id", "stock_name", "industry_category"]].drop_duplicates()
    info["stock_id"] = info["stock_id"].astype(str)
    df = df.merge(info, on="stock_id", how="left")

    df["industry_category"] = df["industry_category"].fillna("其他")

    amount_col = "total_amount" if "total_amount" in df.columns else ("amount" if "amount" in df.columns else None)
    if amount_col is None:
        raise ValueError("Snapshot 缺少 total_amount/amount 欄位")
    df[amount_col] = to_numeric_series(df[amount_col]).fillna(0.0)

    if "change_rate" in df.columns:
        df["change_rate"] = to_numeric_series(df["change_rate"]).fillna(0.0)
        sign = np.sign(df["change_rate"])
    elif "change_price" in df.columns:
        df["change_price"] = to_numeric_series(df["change_price"]).fillna(0.0)
        sign = np.sign(df["change_price"])
    else:
        sign = 0.0

    df["signed_money"] = df[amount_col] * sign

    g = df.groupby("industry_category", as_index=False)["signed_money"].sum()
    g = g.sort_values("signed_money", ascending=False)
    return g


def compute_volume_ranking_from_snapshot(snapshot_all: pd.DataFrame, stock_info: pd.DataFrame) -> pd.DataFrame:
    df = snapshot_all.copy()
    df["stock_id"] = df["stock_id"].astype(str)

    info = stock_info[["stock_id", "stock_name", "industry_category"]].drop_duplicates()
    info["stock_id"] = info["stock_id"].astype(str)
    df = df.merge(info, on="stock_id", how="left")
    df["industry_category"] = df["industry_category"].fillna("其他")

    if "volume_ratio" in df.columns:
        df["volume_ratio"] = to_numeric_series(df["volume_ratio"])
    else:
        tv = to_numeric_series(df.get("total_volume", pd.Series(dtype=float)))
        yv = to_numeric_series(df.get("yesterday_volume", pd.Series(dtype=float)))
        df["volume_ratio"] = tv / yv.replace(0, np.nan)

    df["volume_ratio"] = df["volume_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in ["close", "change_rate", "total_amount", "total_volume"]:
        if c in df.columns:
            df[c] = to_numeric_series(df[c])

    df = df.sort_values("volume_ratio", ascending=False)
    return df


def compute_last_n_trading_dates(token: str, n: int = 6) -> list[str]:
    try:
        tdf = get_trading_dates_cached(token)
        if tdf is not None and not tdf.empty and "date" in tdf.columns:
            tdf = tdf.sort_values("date")
            today = _now_date_str()
            dates = [d for d in tdf["date"].astype(str).tolist() if d <= today]
            return dates[-n:]
    except Exception:
        pass

    # brute-force fallback
    out: list[str] = []
    d0 = datetime.now(tz=TZ) if TZ else datetime.now()
    for i in range(0, 35):
        ds = (d0 - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            df = get_daily_all_cached(token, ds)
            if df is not None and not df.empty:
                out.append(ds)
                if len(out) >= n:
                    break
        except Exception:
            continue
    return sorted(out)[-n:]


def compute_sector_flow_from_daily(token: str, stock_info: pd.DataFrame, date_str: str, prev_date_str: Optional[str]) -> pd.DataFrame:
    today_df = get_daily_all_cached(token, date_str)
    if today_df is None or today_df.empty:
        raise ValueError(f"無法取得 {date_str} 全市場日線資料")

    df = today_df.copy()
    df["stock_id"] = df["stock_id"].astype(str)

    info = stock_info[["stock_id", "stock_name", "industry_category"]].drop_duplicates()
    info["stock_id"] = info["stock_id"].astype(str)
    df = df.merge(info, on="stock_id", how="left")
    df["industry_category"] = df["industry_category"].fillna("其他")

    money_col = "Trading_money" if "Trading_money" in df.columns else None
    if money_col is None:
        raise ValueError("日線資料缺少 Trading_money 欄位")
    df[money_col] = to_numeric_series(df[money_col]).fillna(0.0)

    if "spread" in df.columns:
        df["spread"] = to_numeric_series(df["spread"]).fillna(0.0)
        sign = np.sign(df["spread"])
    else:
        sign = 0.0

    df["signed_money"] = df[money_col] * sign

    g = df.groupby("industry_category", as_index=False)["signed_money"].sum()
    g = g.sort_values("signed_money", ascending=False)
    return g


def compute_volume_ranking_from_daily(token: str, stock_info: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    if len(dates) < 2:
        raise ValueError("交易日不足，無法計算相對大量")

    today = dates[-1]
    prevs = dates[:-1]

    today_df = get_daily_all_cached(token, today)[["stock_id", "Trading_Volume", "Trading_money", "close", "spread"]].copy()
    today_df["stock_id"] = today_df["stock_id"].astype(str)
    today_df["Trading_Volume"] = to_numeric_series(today_df["Trading_Volume"]).fillna(0.0)
    today_df["Trading_money"] = to_numeric_series(today_df["Trading_money"]).fillna(0.0)
    today_df["close"] = to_numeric_series(today_df["close"])
    today_df["spread"] = to_numeric_series(today_df["spread"])

    vol_frames = []
    for d in prevs[-5:]:
        df = get_daily_all_cached(token, d)[["stock_id", "Trading_Volume"]].copy()
        df["stock_id"] = df["stock_id"].astype(str)
        df["Trading_Volume"] = to_numeric_series(df["Trading_Volume"]).fillna(0.0)
        df = df.rename(columns={"Trading_Volume": f"vol_{d}"})
        vol_frames.append(df)

    base = today_df.copy()
    for vf in vol_frames:
        base = base.merge(vf, on="stock_id", how="left")

    vol_cols = [c for c in base.columns if c.startswith("vol_")]
    base["vol_avg_5"] = base[vol_cols].mean(axis=1, skipna=True).fillna(0.0)
    base["volume_ratio"] = base.apply(lambda r: safe_div(r["Trading_Volume"], r["vol_avg_5"], default=0.0), axis=1)
    base["volume_ratio"] = base["volume_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    info = stock_info[["stock_id", "stock_name", "industry_category"]].drop_duplicates()
    info["stock_id"] = info["stock_id"].astype(str)
    base = base.merge(info, on="stock_id", how="left")
    base["industry_category"] = base["industry_category"].fillna("其他")

    base = base.sort_values("volume_ratio", ascending=False)
    base["scan_date"] = today
    return base


def run_seven_features(token: str, stock_id: str, stock_info: pd.DataFrame, scan_mode: str) -> dict:
    """
    One-shot compute and store results to session_state.
    """
    res: dict = {"stock_id": stock_id, "scan_mode": scan_mode, "ts": datetime.utcnow().isoformat()}

    # 1) daily + realtime patch
    start_date = (datetime.now(tz=TZ) - timedelta(days=260) if TZ else datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
    daily = get_daily_one_cached(token, stock_id, start_date, None)
    snap = get_snapshot_one_cached(token, stock_id)

    patched_flag = False
    patch_date = ""
    if daily is not None and not daily.empty:
        if snap is not None and not snap.empty:
            patched, patched_flag, patch_date = patch_daily_with_snapshot(daily, snap.iloc[0])
        else:
            patched = daily.copy()
            patch_date = str(patched["date"].iloc[-1])

        patched = patched.copy()
        patched["close"] = to_numeric_series(patched["close"])
        if "Trading_Volume" in patched.columns:
            patched["Trading_Volume"] = to_numeric_series(patched["Trading_Volume"])
        patched["MA20"] = patched["close"].rolling(20).mean()
        patched["MA60"] = patched["close"].rolling(60).mean()

        res["price_df"] = patched
        res["patch_meta"] = {"patched": patched_flag, "patch_date": patch_date}
        res["snap_row"] = snap.iloc[0].to_dict() if (snap is not None and not snap.empty) else {}
    else:
        res["price_df"] = pd.DataFrame()
        res["patch_meta"] = {"patched": False, "patch_date": ""}
        res["snap_row"] = {}

    # 4) scoring base + chip score (institutional + margin)
    score = compute_score(res["price_df"])
    chip_start = (datetime.now(tz=TZ) - timedelta(days=180) if TZ else datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    inst_df = get_institutional_cached(token, stock_id, chip_start, None)
    margin_df = get_margin_cached(token, stock_id, chip_start, None)

    chip_score, chip_notes = compute_chip_score(inst_df, margin_df)
    score["chip"] = chip_score
    score["total"] = score["trend"] + score["momentum"] + score["volume"] + score["chip"]
    score["notes"].extend(chip_notes)
    res["score"] = score

    # 2 & 3) market scan tables
    meta = {"source": "", "scan_date": ""}
    sector_flow = pd.DataFrame()
    vol_rank = pd.DataFrame()

    if scan_mode.startswith("即時"):
        try:
            snap_all = get_snapshot_all_cached(token)
            if snap_all is not None and not snap_all.empty:
                meta["source"] = "realtime_snapshot"
                meta["scan_date"] = pd.to_datetime(snap_all["date"].iloc[0]).strftime("%Y-%m-%d") if "date" in snap_all.columns else _now_date_str()
                sector_flow = compute_sector_flow_from_snapshot(snap_all, stock_info)
                vol_rank = compute_volume_ranking_from_snapshot(snap_all, stock_info)
        except Exception:
            pass

    if sector_flow.empty or vol_rank.empty:
        dates = compute_last_n_trading_dates(token, n=6)
        if dates:
            meta["source"] = "daily"
            meta["scan_date"] = dates[-1]
            prev = dates[-2] if len(dates) >= 2 else None
            sector_flow = compute_sector_flow_from_daily(token, stock_info, dates[-1], prev)
            vol_rank = compute_volume_ranking_from_daily(token, stock_info, dates)

    res["market_meta"] = meta
    res["sector_flow"] = sector_flow
    res["volume_rank"] = vol_rank

    # 6) chip mirror raw
    res["inst_raw"] = inst_df
    res["margin_raw"] = margin_df

    # 7) revenue
    rev_start = (datetime.now(tz=TZ) - timedelta(days=365 * 6) if TZ else datetime.now() - timedelta(days=365 * 6)).strftime("%Y-%m-%d")
    res["revenue_raw"] = get_month_revenue_cached(token, stock_id, rev_start)

    return res


# -----------------------------
# UI
# -----------------------------
st.title("超級分析師-Pro（七大功能完整版）")

token = st.secrets.get("FINMIND_TOKEN") or os.environ.get("FINMIND_API_TOKEN") or ""
if not token:
    st.error("未偵測到 FINMIND_TOKEN。請在 .streamlit/secrets.toml 設定 FINMIND_TOKEN 或設環境變數 FINMIND_API_TOKEN。")
    st.stop()

st.sidebar.header("參數")
target_sid = st.sidebar.text_input("個股代碼", value="2330").strip()
scan_mode = st.sidebar.selectbox("全市場掃描來源", options=["即時快照（推薦）", "日線（收盤資料）"], index=0)
top_n = st.sidebar.slider("排行榜 Top N", min_value=20, max_value=200, value=80, step=10)

if "result" not in st.session_state:
    st.session_state["result"] = None

with st.spinner("載入股票清單與產業分類..."):
    stock_info = get_stock_info_cached(token)

if st.sidebar.button("一鍵更新七大功能"):
    with st.spinner("更新中：即時補丁 / 全市場掃描 / 籌碼 / 營收..."):
        st.session_state["result"] = run_seven_features(token, target_sid, stock_info, scan_mode)
    st.sidebar.success("更新完成")

if st.sidebar.button("清除結果"):
    st.session_state["result"] = None
    st.sidebar.info("已清除")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. 即時補丁 + 技術圖/評分", "2. 十大族群資金流向", "3. 全台股相對大量榜", "4. 籌碼照妖鏡", "5. 營收診斷"]
)

res = st.session_state.get("result")

# ---------------------------------
# 1) 12/30 即時補丁 + 三線技術圖 + 扣抵值 + 評分
# ---------------------------------
with tab1:
    st.subheader("即時股價補丁 + 三線技術圖 + 扣抵值 + 自動評分（0-100）")

    if res is None or res.get("stock_id") != target_sid:
        st.info("請先按左側「一鍵更新七大功能」。")
    else:
        t = res.get("price_df", pd.DataFrame())
        meta = res.get("patch_meta", {})
        score = res.get("score", {})

        if t is None or t.empty:
            st.error("日線資料為空。")
        else:
            patch_date = meta.get("patch_date", "")
            patched_flag = bool(meta.get("patched", False))

            # 扣抵值
            kdr20 = float(t["close"].iloc[-20]) if len(t) >= 20 and pd.notna(t["close"].iloc[-20]) else np.nan
            kdr60 = float(t["close"].iloc[-60]) if len(t) >= 60 and pd.notna(t["close"].iloc[-60]) else np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最新價（含補丁）", f"{float(t['close'].iloc[-1]):.2f}")
            c2.metric("資料日期", patch_date)
            c3.metric("MA20 扣抵值", "-" if pd.isna(kdr20) else f"{kdr20:.2f}")
            c4.metric("MA60 扣抵值", "-" if pd.isna(kdr60) else f"{kdr60:.2f}")

            if patched_flag:
                st.success(f"已套用 Pro 即時快照補丁：新增 {patch_date} 盤中資料列（解決日線日期卡住問題）。")
            else:
                st.caption("本次未新增補丁（日線已是最新日期或快照無資料）。")

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("總分", f"{score.get('total', 0):.0f} / 100")
            s2.metric("趨勢", f"{score.get('trend', 0):.0f} / 25")
            s3.metric("動能", f"{score.get('momentum', 0):.0f} / 25")
            s4.metric("量能", f"{score.get('volume', 0):.0f} / 25")
            s5.metric("籌碼", f"{score.get('chip', 0):.0f} / 25")

            notes = score.get("notes", [])
            if notes:
                st.caption("診斷摘要：" + "；".join(notes[:6]))

            # 三線技術圖：Close + MA20(螢光黃) + MA60(桃紅) + 黃色 X
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t["date"], y=t["close"], name="Close", mode="lines"))
            fig.add_trace(go.Scatter(x=t["date"], y=t["MA20"], name="MA20", mode="lines", line=dict(color="#F7FF00", width=2)))
            fig.add_trace(go.Scatter(x=t["date"], y=t["MA60"], name="MA60", mode="lines", line=dict(color="#FF2DA4", width=2)))
            fig.add_trace(
                go.Scatter(
                    x=[t["date"].iloc[-1]],
                    y=[t["close"].iloc[-1]],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color="#FFD400", line=dict(width=2, color="#FFD400")),
                    showlegend=False,
                )
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# 2) 十大族群資金流向
# ---------------------------------
with tab2:
    st.subheader("十大族群資金流向（全市場掃描，已修復空值報錯）")

    if res is None:
        st.info("請先按左側「一鍵更新七大功能」。")
    else:
        meta = res.get("market_meta", {})
        sector_flow = res.get("sector_flow", pd.DataFrame())
        if sector_flow is None or sector_flow.empty:
            st.error("族群資金流向資料為空（可能：快照/日線掃描失敗）。")
        else:
            st.caption(f"資料來源：{meta.get('source','')}；掃描日期：{meta.get('scan_date','')}")
            show = sector_flow.head(10).copy()
            show["資金流向(億)"] = show["signed_money"] / 1e8
            show = show.rename(columns={"industry_category": "族群"})
            st.dataframe(show[["族群", "signed_money", "資金流向(億)"]], use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=show["族群"], y=show["資金流向(億)"], name="資金流向(億)"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="族群", yaxis_title="資金流向(億)")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------
# 3) 全台股相對大量榜
# ---------------------------------
with tab3:
    st.subheader("全台股相對大量榜（全市場量能增溫排行）")

    if res is None:
        st.info("請先按左側「一鍵更新七大功能」。")
    else:
        meta = res.get("market_meta", {})
        vol_rank = res.get("volume_rank", pd.DataFrame())
        if vol_rank is None or vol_rank.empty:
            st.error("相對大量榜資料為空（可能：快照/日線掃描失敗）。")
        else:
            st.caption(f"資料來源：{meta.get('source','')}；掃描日期：{meta.get('scan_date','')}")
            cols = ["stock_id", "stock_name", "industry_category", "volume_ratio"]
            for extra in ["close", "change_rate", "total_amount", "Trading_money", "total_volume", "Trading_Volume", "spread", "scan_date"]:
                if extra in vol_rank.columns and extra not in cols:
                    cols.append(extra)
            top_df = vol_rank[cols].head(top_n).copy()
            top_df.insert(0, "rank", range(1, len(top_df) + 1))
            st.dataframe(top_df, use_container_width=True)

# ---------------------------------
# 4) 籌碼照妖鏡：法人買賣超 + 散戶融資
# ---------------------------------
with tab4:
    st.subheader("籌碼照妖鏡（法人買賣超 + 散戶融資對照圖）")

    if res is None:
        st.info("請先按左側「一鍵更新七大功能」。")
    else:
        inst_df = res.get("inst_raw", pd.DataFrame())
        margin_df = res.get("margin_raw", pd.DataFrame())

        inst_plot = None
        if inst_df is not None and not inst_df.empty and {"date", "buy", "sell"}.issubset(inst_df.columns):
            tmp = inst_df.copy()
            tmp["buy"] = to_numeric_series(tmp["buy"]).fillna(0.0)
            tmp["sell"] = to_numeric_series(tmp["sell"]).fillna(0.0)
            tmp["net"] = tmp["buy"] - tmp["sell"]
            inst_plot = tmp.groupby("date", as_index=False)["net"].sum().sort_values("date")

        margin_plot = None
        if margin_df is not None and not margin_df.empty and "date" in margin_df.columns:
            m = margin_df.copy().sort_values("date")
            bal_col = None
            for cand in ["MarginPurchaseTodayBalance", "TodayBalance", "margin_purchase_today_balance"]:
                if cand in m.columns:
                    bal_col = cand
                    break
            if bal_col:
                m[bal_col] = to_numeric_series(m[bal_col])
                margin_plot = m[["date", bal_col]].rename(columns={bal_col: "margin_balance"})

        if inst_plot is None and margin_plot is None:
            st.error("法人/融資資料不足，無法繪圖。")
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if inst_plot is not None:
                fig.add_trace(go.Bar(x=inst_plot["date"], y=inst_plot["net"], name="法人買賣超(股)"), secondary_y=False)
            if margin_plot is not None:
                fig.add_trace(go.Scatter(x=margin_plot["date"], y=margin_plot["margin_balance"], mode="lines", name="融資餘額"), secondary_y=True)
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
            fig.update_yaxes(title_text="法人買賣超(股)", secondary_y=False)
            fig.update_yaxes(title_text="融資餘額", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if inst_plot is not None and not inst_plot.empty:
                st.metric("法人近5日合計(股)", f"{inst_plot.tail(5)['net'].sum():,.0f}")
                st.dataframe(inst_plot.tail(10), use_container_width=True)
        with c2:
            if margin_plot is not None and not margin_plot.empty:
                delta = margin_plot.tail(5)["margin_balance"].iloc[-1] - margin_plot.tail(5)["margin_balance"].iloc[0]
                st.metric("近5日融資餘額變化", f"{delta:,.0f}")
                st.dataframe(margin_plot.tail(10), use_container_width=True)

# ---------------------------------
# 5) 營收診斷：月營收年增率
# ---------------------------------
with tab5:
    st.subheader("營收診斷（月營收年增率圖表）")

    if res is None:
        st.info("請先按左側「一鍵更新七大功能」。")
    else:
        rev = res.get("revenue_raw", pd.DataFrame())
        if rev is None or rev.empty:
            st.warning("月營收資料為空。")
        else:
            rev = rev.copy()
            rev["date"] = pd.to_datetime(rev["date"])
            rev = rev.sort_values("date")
            rev["revenue"] = to_numeric_series(rev["revenue"])
            rev["yoy"] = (rev["revenue"] / rev["revenue"].shift(12) - 1.0) * 100.0
            rev["yoy"] = rev["yoy"].replace([np.inf, -np.inf], np.nan)

            last = rev.dropna(subset=["yoy"]).tail(1)
            c1, c2 = st.columns(2)
            if not last.empty:
                c1.metric("最新月營收年增率(%)", f"{last['yoy'].iloc[0]:.2f}")
            c2.metric("最新月營收(元)", f"{rev['revenue'].iloc[-1]:,.0f}")

            fig = go.Figure()
            fig.add_trace(go.Bar(x=rev["date"], y=rev["yoy"], name="YoY(%)"))
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="年增率(%)")
            st.plotly_chart(fig, use_container_width=True)

            show = rev[["date", "revenue", "yoy"]].tail(24).copy()
            show["date"] = show["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(show, use_container_width=True)
