import os
from datetime import datetime, timedelta
from typing import Optional, Union, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="超級分析師-Pro（七大功能 + 第八分類 + 交易計畫引擎）", layout="wide")
st.title("超級分析師-Pro（七大功能 + 第八分類 + 交易計畫引擎）")

# -----------------------------
# Timezone
# -----------------------------
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    TZ = ZoneInfo("Asia/Taipei")
except Exception:
    TZ = None  # fallback

# -----------------------------
# Constants
# -----------------------------
LARGE_CAP_IDS = {
    "2330", "2454", "2317", "2308", "2412", "3711", "2382", "2357", "3008",
    "2002", "1301", "1303",
    "2881", "2882", "2891", "2886", "2884", "2885", "2880", "2887", "2892",
}

THEME_GROUPS = {
    "半導體族群": {"industry": ["半導體業"], "stocks": []},
    "記憶體族群": {"industry": [], "stocks": ["2408","2344","2337","3260","8299","3006","4967"]},
    "PCB族群": {"industry": [], "stocks": ["3037","8046","2313","2368","4958","2383","6213","3189","6274"]},
    "CPO族群": {"industry": [], "stocks": ["4979","3081","3163","3363","4909","3450","2345"]},
    "航運族群": {"industry": [], "stocks": ["2603","2609","2615","2606","2605","2637","2617","5608","2641"]},
    "伺服器族群": {"industry": [], "stocks": ["2382","3231","6669","2356","2317","2324","3706","2376","4938"]},
    "散熱族群": {"industry": [], "stocks": ["3017","3324","3653","2421","3338","6230"]},
    "面板族群": {"industry": [], "stocks": ["3481","2409","6116","3673"]},
    "金融族群": {"industry": ["金融保險業"], "stocks": ["2881","2882","2891","2886","2884","2885","2880","2887","2892"]},
    "電力重電": {"industry": [], "stocks": ["1519","1513","1503","1504","1514","1609","1617"]},
}

# -----------------------------
# FinMind REST endpoints
# -----------------------------
FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TICK_SNAPSHOT_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_tick_snapshot"

# -----------------------------
# Helpers
# -----------------------------
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
    return pd.DataFrame(payload.get("data", []))


def finmind_tick_snapshot(
    token: str,
    data_id: Optional[Union[str, Iterable[str]]] = None,
    timeout: int = 15,
) -> pd.DataFrame:
    params = {"data_id": ""} if data_id is None else {"data_id": _as_data_id_param(data_id)}
    resp = requests.get(FINMIND_TICK_SNAPSHOT_URL, headers=_headers(token), params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    _raise_if_api_error(payload, "tick_snapshot")
    return pd.DataFrame(payload.get("data", []))


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


def ensure_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure change_rate exists (%).
    - snapshot typically has change_rate
    - daily: compute via spread / prev_close
    """
    if df is None or df.empty:
        return df
    df = df.copy()

    if "change_rate" in df.columns and not df["change_rate"].isna().all():
        df["change_rate"] = pd.to_numeric(df["change_rate"], errors="coerce").fillna(0.0)
        return df

    if "spread" in df.columns and "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        spread = pd.to_numeric(df["spread"], errors="coerce").fillna(0.0)
        prev_close = (close - spread).replace(0, np.nan)
        df["change_rate"] = (spread / prev_close) * 100.0
        df["change_rate"] = df["change_rate"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    df["change_rate"] = 0.0
    return df


def pick_money_col(df: pd.DataFrame) -> str:
    for c in ["total_amount", "Trading_money", "amount", "Trading_amount"]:
        if c in df.columns:
            return c
    df["money"] = 0.0
    return "money"


def pick_volume_col(df: pd.DataFrame) -> str:
    for c in ["total_volume", "Trading_Volume", "volume"]:
        if c in df.columns:
            return c
    df["vol"] = 0.0
    return "vol"


def _get_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Cached downloads
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def get_stock_info_cached(token: str) -> pd.DataFrame:
    return finmind_get_data(token, dataset="TaiwanStockInfo", timeout=30)


@st.cache_data(ttl=10)
def get_snapshot_all_cached(token: str) -> pd.DataFrame:
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


@st.cache_data(ttl=6 * 3600)
def get_daily_all_cached(token: str, date_str: str) -> pd.DataFrame:
    df = finmind_get_data(token, dataset="TaiwanStockPrice", start_date=date_str, timeout=60)
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


# -----------------------------
# 1) Pro snapshot patch to force today's row
# -----------------------------
def patch_daily_with_snapshot(daily_df: pd.DataFrame, snap_row: pd.Series) -> tuple[pd.DataFrame, bool, str]:
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
    mapping = [
        ("open", "open"),
        ("high", "max"),
        ("low", "min"),
        ("close", "close"),
        ("total_volume", "Trading_Volume"),
        ("total_amount", "Trading_money"),
        ("change_price", "spread"),
    ]
    for k_from, k_to in mapping:
        if k_to in new_row and k_from in snap_row.index:
            new_row[k_to] = snap_row.get(k_from)

    daily_df = pd.concat([daily_df, pd.DataFrame([new_row])], ignore_index=True)
    return daily_df, True, snap_date


# -----------------------------
# 4) Score system (0-100): 4 dimensions
# -----------------------------
def compute_score(t: pd.DataFrame) -> dict:
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

    # Momentum (0-25): 10-day return
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

        # over-extended vs MA20 penalty
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
    notes: list[str] = []
    score = 0.0

    # Institutional net (all institutions aggregated)
    if inst_df is not None and not inst_df.empty and {"date", "buy", "sell"}.issubset(inst_df.columns):
        tmp = inst_df.copy()
        tmp["buy"] = to_numeric_series(tmp["buy"]).fillna(0.0)
        tmp["sell"] = to_numeric_series(tmp["sell"]).fillna(0.0)
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

    # Margin balance trend
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
# 2 & 3) Market scan: sector flow + relative volume ranking
# -----------------------------

def compute_chip_summary(inst_df: pd.DataFrame, margin_df: pd.DataFrame, lookback_days: int = 20) -> dict:
    """
    籌碼摘要（給健診/持股判斷用）：
      - 法人近1日/5日合計買賣超（並嘗試拆外資/投信/自營商）
      - 融資餘額近5日變化
      - 給出 signal：偏多/中性/偏空
    注意：FinMind 欄位/名稱可能隨來源略有差異，因此做了容錯。
    """
    out = {
        "signal": "中性",
        "net_1d_total": None,
        "net_5d_total": None,
        "net_5d_foreign": None,
        "net_5d_trust": None,
        "net_5d_dealer": None,
        "trust_streak": None,
        "foreign_streak": None,
        "margin_balance_col": None,
        "margin_last": None,
        "margin_delta_5d": None,
        "inst_daily": pd.DataFrame(),    # columns: date, net_total, net_foreign, net_trust, net_dealer
        "margin_daily": pd.DataFrame(),  # columns: date, margin_balance
        "notes": [],
    }

    # -------- 法人 --------
    if inst_df is not None and not inst_df.empty and {"date", "buy", "sell"}.issubset(inst_df.columns):
        tmp = inst_df.copy()
        tmp["buy"] = pd.to_numeric(tmp["buy"], errors="coerce").fillna(0.0)
        tmp["sell"] = pd.to_numeric(tmp["sell"], errors="coerce").fillna(0.0)
        tmp["net"] = tmp["buy"] - tmp["sell"]

        # normalize date
        try:
            tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            tmp["date"] = tmp["date"].astype(str)

        tmp = tmp.sort_values("date")

        def map_inst_type(name: str) -> str:
            n = str(name or "")
            if "Investment_Trust" in n or "投信" in n:
                return "投信"
            if "Foreign" in n or "外資" in n:
                return "外資"
            if "Dealer" in n or "自營商" in n:
                return "自營商"
            return n

        if "name" in tmp.columns:
            tmp["inst_type"] = tmp["name"].map(map_inst_type)
        else:
            tmp["inst_type"] = "總計"

        # daily totals (all types)
        daily_total = tmp.groupby("date", as_index=False)["net"].sum().sort_values("date")
        if not daily_total.empty:
            out["net_1d_total"] = float(daily_total["net"].iloc[-1])
            last5_dates = daily_total["date"].tail(5).tolist()
            out["net_5d_total"] = float(daily_total[daily_total["date"].isin(last5_dates)]["net"].sum())

            # build inst_daily for plotting (lookback_days)
            inst_daily = daily_total.tail(lookback_days).rename(columns={"net": "net_total"}).copy()
            out["inst_daily"] = inst_daily

            # by type for last 5 days
            last5 = tmp[tmp["date"].isin(last5_dates)].copy()
            by_type = last5.groupby("inst_type")["net"].sum()

            def _get_type_value(key: str):
                v = by_type.get(key, np.nan)
                return None if pd.isna(v) else float(v)

            out["net_5d_foreign"] = _get_type_value("外資")
            out["net_5d_trust"] = _get_type_value("投信")
            out["net_5d_dealer"] = _get_type_value("自營商")

            # trust/foreign streak (consecutive net>0 or <0) in last 10 days
            def _streak(series: pd.Series) -> Optional[int]:
                if series is None or series.empty:
                    return None
                s = series.dropna().astype(float)
                if s.empty:
                    return None
                # streak of last sign
                sign = 1 if s.iloc[-1] > 0 else (-1 if s.iloc[-1] < 0 else 0)
                if sign == 0:
                    return 0
                cnt = 0
                for x in reversed(s.tolist()):
                    if (x > 0 and sign == 1) or (x < 0 and sign == -1):
                        cnt += 1
                    else:
                        break
                return cnt * sign  # positive means consecutive buy, negative means consecutive sell

            # daily by type for streak calc
            by_date_type = tmp.groupby(["date", "inst_type"], as_index=False)["net"].sum()
            pivot = by_date_type.pivot(index="date", columns="inst_type", values="net").sort_index()
            if "投信" in pivot.columns:
                out["trust_streak"] = _streak(pivot["投信"].tail(10))
            if "外資" in pivot.columns:
                out["foreign_streak"] = _streak(pivot["外資"].tail(10))

            # enrich inst_daily with type nets (optional)
            if not out["inst_daily"].empty:
                inst_daily = out["inst_daily"].set_index("date")
                if "外資" in pivot.columns:
                    inst_daily["net_foreign"] = pivot["外資"].reindex(inst_daily.index).fillna(0.0)
                else:
                    inst_daily["net_foreign"] = 0.0
                if "投信" in pivot.columns:
                    inst_daily["net_trust"] = pivot["投信"].reindex(inst_daily.index).fillna(0.0)
                else:
                    inst_daily["net_trust"] = 0.0
                if "自營商" in pivot.columns:
                    inst_daily["net_dealer"] = pivot["自營商"].reindex(inst_daily.index).fillna(0.0)
                else:
                    inst_daily["net_dealer"] = 0.0
                out["inst_daily"] = inst_daily.reset_index()

    # -------- 融資 --------
    if margin_df is not None and not margin_df.empty and "date" in margin_df.columns:
        m = margin_df.copy()
        try:
            m["date"] = pd.to_datetime(m["date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            m["date"] = m["date"].astype(str)
        m = m.sort_values("date")

        bal_col = None
        for cand in ["MarginPurchaseTodayBalance", "TodayBalance", "margin_purchase_today_balance"]:
            if cand in m.columns:
                bal_col = cand
                break
        if bal_col:
            m[bal_col] = pd.to_numeric(m[bal_col], errors="coerce")
            out["margin_balance_col"] = bal_col
            margin_series = m[["date", bal_col]].dropna().rename(columns={bal_col: "margin_balance"}).copy()
            out["margin_daily"] = margin_series.tail(lookback_days)

            if not margin_series.empty:
                out["margin_last"] = float(margin_series["margin_balance"].iloc[-1])
                if len(margin_series) >= 6:
                    out["margin_delta_5d"] = float(margin_series["margin_balance"].iloc[-1] - margin_series["margin_balance"].iloc[-6])
                elif len(margin_series) >= 2:
                    out["margin_delta_5d"] = float(margin_series["margin_balance"].iloc[-1] - margin_series["margin_balance"].iloc[0])

    # -------- signal 判讀 --------
    net5 = out.get("net_5d_total")
    f5 = out.get("net_5d_foreign")
    t5 = out.get("net_5d_trust")
    m5 = out.get("margin_delta_5d")

    bullish = (net5 is not None and net5 > 0) and ((f5 is not None and f5 > 0) or (t5 is not None and t5 > 0))
    bearish = (net5 is not None and net5 < 0) and ((f5 is not None and f5 < 0) or (t5 is not None and t5 < 0))

    # margin as confirmation: 融資增加偏風險、融資下降偏健康
    if bullish and (m5 is None or m5 <= 0):
        out["signal"] = "偏多"
        out["notes"].append("籌碼偏多：法人近5日偏買超，且融資未明顯增加")
    elif bearish and (m5 is None or m5 > 0):
        out["signal"] = "偏空"
        out["notes"].append("籌碼偏空：法人近5日偏賣超，且融資偏升溫")
    else:
        out["signal"] = "中性"
        # keep note light to avoid noise
        if net5 is not None:
            out["notes"].append("籌碼中性：法人與融資未形成一致方向")

    return out


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

    df = ensure_change_rate(df)
    sign = np.sign(df["change_rate"])
    df["signed_money"] = df[amount_col] * sign

    g = df.groupby("industry_category", as_index=False)["signed_money"].sum()
    return g.sort_values("signed_money", ascending=False)


def compute_volume_ranking_from_snapshot(snapshot_all: pd.DataFrame, stock_info: pd.DataFrame) -> pd.DataFrame:
    df = snapshot_all.copy()
    df["stock_id"] = df["stock_id"].astype(str)

    info = stock_info[["stock_id", "stock_name", "industry_category"]].drop_duplicates()
    info["stock_id"] = info["stock_id"].astype(str)
    df = df.merge(info, on="stock_id", how="left")
    df["industry_category"] = df["industry_category"].fillna("其他")

    # ensure numeric columns commonly used downstream
    for c in ["close", "change_rate", "total_amount", "total_volume", "volume_ratio", "open", "high", "low", "change_price"]:
        if c in df.columns:
            df[c] = to_numeric_series(df[c])

    if "volume_ratio" in df.columns:
        df["volume_ratio"] = pd.to_numeric(df["volume_ratio"], errors="coerce")
    else:
        tv = to_numeric_series(df.get("total_volume", pd.Series(dtype=float)))
        yv = to_numeric_series(df.get("yesterday_volume", pd.Series(dtype=float)))
        df["volume_ratio"] = tv / yv.replace(0, np.nan)

    df["volume_ratio"] = df["volume_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.sort_values("volume_ratio", ascending=False)


def compute_sector_flow_from_daily(token: str, stock_info: pd.DataFrame, date_str: str) -> pd.DataFrame:
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

    df = ensure_change_rate(df)
    sign = np.sign(df["change_rate"])
    df["signed_money"] = df[money_col] * sign

    g = df.groupby("industry_category", as_index=False)["signed_money"].sum()
    return g.sort_values("signed_money", ascending=False)


def compute_volume_ranking_from_daily(token: str, stock_info: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """
    Relative volume = today's volume / avg(prev 5 days volume)
    Returns a DF that contains today's OHLC + volume_ratio + industry info
    """
    if len(dates) < 2:
        raise ValueError("交易日不足，無法計算相對大量")

    today = dates[-1]
    prevs = dates[:-1]

    today_df = get_daily_all_cached(token, today).copy()
    need_cols = ["stock_id", "Trading_Volume", "Trading_money", "close", "spread", "open", "max", "min"]
    keep = [c for c in need_cols if c in today_df.columns]
    today_df = today_df[keep].copy()

    today_df["stock_id"] = today_df["stock_id"].astype(str)
    for c in keep:
        if c != "stock_id":
            today_df[c] = to_numeric_series(today_df[c])

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

    if "Trading_Volume" in base.columns:
        base["volume_ratio"] = base.apply(lambda r: safe_div(r["Trading_Volume"], r["vol_avg_5"], default=0.0), axis=1)
    else:
        base["volume_ratio"] = 0.0

    base["volume_ratio"] = base["volume_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    info = stock_info[["stock_id", "stock_name", "industry_category"]].drop_duplicates()
    info["stock_id"] = info["stock_id"].astype(str)
    base = base.merge(info, on="stock_id", how="left")
    base["industry_category"] = base["industry_category"].fillna("其他")
    base["scan_date"] = today

    # compute change_rate (for downstream tables)
    base = ensure_change_rate(base)
    return base.sort_values("volume_ratio", ascending=False)


# -----------------------------
# NEW (Upgrade 1): Pattern classification & Trade plan engine
# -----------------------------
def compute_volume_ratio_from_df(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "Trading_Volume" not in df.columns:
        return None
    v = pd.to_numeric(df["Trading_Volume"], errors="coerce")
    if len(v) < 6 or pd.isna(v.iloc[-1]):
        return None
    base = v.iloc[-6:-1].mean()
    if pd.isna(base) or base <= 0:
        return None
    return float(v.iloc[-1] / base)


def compute_structure(df: pd.DataFrame) -> dict:
    """
    Compute key levels:
      prev20_high/low (exclude today), prev60_high/low, MA20/MA60, MA20 slope, trend label
    """
    out = {
        "close": np.nan, "ma20": np.nan, "ma60": np.nan, "ma20_slope": np.nan,
        "prev20_high": np.nan, "prev20_low": np.nan, "prev60_high": np.nan, "prev60_low": np.nan,
        "range20": np.nan, "trend": "unknown"
    }
    if df is None or df.empty or "close" not in df.columns:
        return out

    t = df.copy()
    t["close"] = pd.to_numeric(t["close"], errors="coerce")

    if "MA20" not in t.columns:
        t["MA20"] = t["close"].rolling(20).mean()
    if "MA60" not in t.columns:
        t["MA60"] = t["close"].rolling(60).mean()

    high_col = _get_col(t, ["max", "high", "High"]) or "close"
    low_col = _get_col(t, ["min", "low", "Low"]) or "close"
    t[high_col] = pd.to_numeric(t[high_col], errors="coerce")
    t[low_col] = pd.to_numeric(t[low_col], errors="coerce")

    last = t.iloc[-1]
    out["close"] = float(last["close"]) if pd.notna(last["close"]) else np.nan
    out["ma20"] = float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan
    out["ma60"] = float(last["MA60"]) if pd.notna(last["MA60"]) else np.nan

    if len(t) >= 25 and pd.notna(t["MA20"].iloc[-1]) and pd.notna(t["MA20"].iloc[-6]):
        out["ma20_slope"] = float(t["MA20"].iloc[-1] - t["MA20"].iloc[-6])

    if len(t) >= 21:
        out["prev20_high"] = float(t[high_col].rolling(20).max().shift(1).iloc[-1])
        out["prev20_low"] = float(t[low_col].rolling(20).min().shift(1).iloc[-1])
    if len(t) >= 61:
        out["prev60_high"] = float(t[high_col].rolling(60).max().shift(1).iloc[-1])
        out["prev60_low"] = float(t[low_col].rolling(60).min().shift(1).iloc[-1])

    if pd.notna(out["prev20_high"]) and pd.notna(out["prev20_low"]):
        out["range20"] = float(out["prev20_high"] - out["prev20_low"])

    c = out["close"]; ma20 = out["ma20"]; ma60 = out["ma60"]; slope = out["ma20_slope"]
    if pd.notna(c) and pd.notna(ma20) and pd.notna(ma60):
        if c > ma20 > ma60 and (pd.isna(slope) or slope > 0):
            out["trend"] = "bull"
        elif c < ma20 < ma60 and (pd.isna(slope) or slope < 0):
            out["trend"] = "bear"
        elif c > ma20:
            out["trend"] = "mild_bull"
        elif c < ma20:
            out["trend"] = "mild_bear"
        else:
            out["trend"] = "neutral"
    return out


def classify_pattern(struct: dict, vol_ratio: Optional[float], profile: str) -> dict:
    c = struct.get("close", np.nan)
    ma20 = struct.get("ma20", np.nan)
    prev20_high = struct.get("prev20_high", np.nan)
    prev20_low = struct.get("prev20_low", np.nan)
    trend = struct.get("trend", "unknown")

    vol_confirm = 1.2 if profile == "大型權值股" else 1.5

    if pd.notna(prev20_high) and pd.notna(c) and c > prev20_high:
        if vol_ratio is not None and vol_ratio >= vol_confirm:
            return {"pattern": "突破盤", "entry_type": "突破", "reason": f"突破20日區間前高 {prev20_high:.2f} 且放量（相對量 {vol_ratio:.2f}）。", "vol_confirm": vol_confirm}
        return {"pattern": "突破盤", "entry_type": "突破（待量能確認）", "reason": f"突破20日區間前高 {prev20_high:.2f}，但量能未達確認（相對量 {vol_ratio if vol_ratio is not None else 'N/A'}）。", "vol_confirm": vol_confirm}

    if pd.notna(ma20) and pd.notna(c):
        dist = abs(c / ma20 - 1.0)
        if dist <= 0.015 and trend not in ["bear", "mild_bear"]:
            return {"pattern": "拉回盤", "entry_type": "拉回", "reason": "回測20MA附近且趨勢未轉空，屬回測承接型態。", "vol_confirm": vol_confirm}

    if pd.notna(prev20_low) and pd.notna(c) and c < prev20_low:
        return {"pattern": "轉弱破底", "entry_type": "避開", "reason": f"跌破20日區間低點 {prev20_low:.2f}，短線結構轉弱。", "vol_confirm": vol_confirm}

    return {"pattern": "盤整盤", "entry_type": "等待", "reason": "區間整理，等待突破或回測承接。", "vol_confirm": vol_confirm}


def classify_volume_price(price_df: pd.DataFrame, vol_ratio: Optional[float]) -> dict:
    out = {"label": "資料不足", "detail": ""}
    if price_df is None or price_df.empty:
        return out

    df = ensure_change_rate(price_df)
    last = df.iloc[-1]

    close = float(pd.to_numeric(last.get("close", np.nan), errors="coerce"))
    open_ = float(pd.to_numeric(last.get("open", np.nan), errors="coerce")) if "open" in df.columns else close
    high = float(pd.to_numeric(last.get("max", np.nan), errors="coerce")) if "max" in df.columns else close
    low = float(pd.to_numeric(last.get("min", np.nan), errors="coerce")) if "min" in df.columns else close
    chg = float(pd.to_numeric(last.get("change_rate", np.nan), errors="coerce")) if "change_rate" in df.columns else np.nan

    if open_ == 0 or pd.isna(open_):
        open_ = close

    body_pct = abs(close - open_) / open_ * 100 if open_ else np.nan
    range_pct = (high - low) / open_ * 100 if open_ else np.nan

    vr = vol_ratio

    if vr is not None and vr >= 1.8 and pd.notna(chg) and chg > 1:
        if close >= open_:
            return {"label": "放量上漲（健康）", "detail": f"相對量 {vr:.2f}，上漲 {chg:.2f}%：偏『起漲/續漲量』。"}
        return {"label": "放量轉弱（風險）", "detail": f"相對量 {vr:.2f} 但收黑：留意高檔震盪/洗盤或轉弱。"}

    if vr is not None and vr >= 1.5 and pd.notna(chg) and abs(chg) < 0.5:
        return {"label": "放量不漲（出貨疑慮）", "detail": f"相對量 {vr:.2f} 但漲跌幅僅 {chg:.2f}%：常見於高檔換手/出貨。"}

    if vr is not None and vr >= 1.5 and pd.notna(chg) and chg < -1:
        return {"label": "放量下跌（風險）", "detail": f"相對量 {vr:.2f} 且下跌 {chg:.2f}%：偏『壓力出現』，不利追多。"}

    if vr is not None and vr <= 0.8 and pd.notna(range_pct) and range_pct <= 2.0:
        return {"label": "量縮整理（等待）", "detail": f"相對量 {vr:.2f} 且波動不大：偏『以盤代跌/等突破』。"}

    if pd.notna(body_pct) and body_pct >= 3 and pd.notna(chg) and chg < 0:
        return {"label": "長黑K（風險）", "detail": f"單日跌幅 {chg:.2f}%、實體約 {body_pct:.2f}%：偏弱勢K棒。"}

    return {"label": "一般波動", "detail": "量價未出現典型突破/出貨訊號，建議搭配趨勢與關鍵價位判讀。"}



# -----------------------------
# NEW: 「老王」策略（持股判斷依據）
# -----------------------------
def compute_ma(df: pd.DataFrame, n: int) -> pd.Series:
    s = pd.to_numeric(df["close"], errors="coerce")
    return s.rolling(n).mean()


def compute_volume_ratio_series(df: pd.DataFrame, window: int = 5) -> Optional[pd.Series]:
    if df is None or df.empty or "Trading_Volume" not in df.columns:
        return None
    v = pd.to_numeric(df["Trading_Volume"], errors="coerce")
    base = v.rolling(window).mean()
    vr = v / base.replace(0, np.nan)
    return vr.replace([np.inf, -np.inf], np.nan)


def hit_theme(stock_id: str, theme_name: str) -> bool:
    try:
        return str(stock_id) in set(str(x) for x in THEME_GROUPS.get(theme_name, {}).get("stocks", []))
    except Exception:
        return False


def compute_leader_status(market_df: pd.DataFrame, stock_id: str) -> dict:
    """
    買強不買弱：判斷該股是否為其族群的領導股（同族群Top3）。
    使用同族群內：漲跌幅、成交金額、量比的綜合排序。
    """
    out = {
        "industry": "",
        "is_leader": False,
        "rank": None,
        "top_leaders": [],
    }
    if market_df is None or market_df.empty:
        return out

    df = market_df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df["industry_category"] = df.get("industry_category", "其他").fillna("其他")
    df["stock_name"] = df.get("stock_name", "").fillna("")
    df = ensure_change_rate(df)

    row = df[df["stock_id"] == str(stock_id)]
    if row.empty:
        return out

    industry = str(row.iloc[0]["industry_category"])
    out["industry"] = industry

    sub = df[df["industry_category"] == industry].copy()
    if sub.empty:
        return out

    money_col = pick_money_col(sub)
    sub[money_col] = pd.to_numeric(sub[money_col], errors="coerce").fillna(0.0)

    if "volume_ratio" not in sub.columns:
        sub["volume_ratio"] = 0.0
    sub["volume_ratio"] = pd.to_numeric(sub["volume_ratio"], errors="coerce").fillna(0.0)

    # 綜合領導分數（可依你的偏好調權重）
    sub["leader_score"] = (
        sub["change_rate"].rank(pct=True) * 0.45
        + sub[money_col].rank(pct=True) * 0.35
        + sub["volume_ratio"].rank(pct=True) * 0.20
    )

    sub = sub.sort_values("leader_score", ascending=False).reset_index(drop=True)
    top3 = sub.head(3)[["stock_id", "stock_name", "change_rate", money_col, "volume_ratio"]].copy()
    out["top_leaders"] = top3.to_dict("records")

    idx = sub.index[sub["stock_id"] == str(stock_id)]
    if len(idx):
        rank = int(idx[0]) + 1
        out["rank"] = rank
        out["is_leader"] = rank <= 3

    return out


def compute_oldwang_signals(stock_id: str, price_df: pd.DataFrame, profile: str) -> dict:
    """
    老王策略檢核：
      - 均線策略：5/10判斷極短線強弱；20判斷多頭趨勢
      - 形態：三陽開泰（站上5/10/20）、四海遊龍（站上所有短中期均線）
      - 量價：突破前高需帶量；大量後不應快速縮量
      - 一條線策略：大型主流/特定題材（記憶體/面板等）用10MA守線，不破續抱
      - 無視單一K棒：若未破關鍵支撐，不因單日長黑就判死刑
    """
    out = {
        "ma5": np.nan, "ma10": np.nan, "ma20": np.nan, "ma60": np.nan,
        "above_ma5": None, "above_ma10": None, "above_ma20": None, "above_ma60": None,
        "tri": False, "tri_strong": False, "four": False,
        "key_ma": 10, "key_hold": False, "key_break_2d": False,
        "breakout_need_volume": False, "post_surge_volume_ok": None,
        "washout_ignore": False,
        "notes": [],
    }
    if price_df is None or price_df.empty or "close" not in price_df.columns:
        out["notes"].append("缺少日線資料")
        return out

    df = price_df.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "Trading_Volume" in df.columns:
        df["Trading_Volume"] = pd.to_numeric(df["Trading_Volume"], errors="coerce")
    df = ensure_change_rate(df)

    # ensure chronological order for rolling calculations
    if "date" in df.columns:
        try:
            df = df.sort_values("date")
        except Exception:
            pass

    # MA
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    if "MA20" not in df.columns:
        df["MA20"] = df["close"].rolling(20).mean()
    if "MA60" not in df.columns:
        df["MA60"] = df["close"].rolling(60).mean()

    last = df.iloc[-1]
    out["ma5"] = float(last["MA5"]) if pd.notna(last["MA5"]) else np.nan
    out["ma10"] = float(last["MA10"]) if pd.notna(last["MA10"]) else np.nan
    out["ma20"] = float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan
    out["ma60"] = float(last["MA60"]) if pd.notna(last["MA60"]) else np.nan

    close = float(last["close"]) if pd.notna(last["close"]) else np.nan
    ma5, ma10, ma20, ma60 = out["ma5"], out["ma10"], out["ma20"], out["ma60"]

    # explicit "above MA" flags for transparency/debug
    if pd.notna(close) and pd.notna(ma5):
        out["above_ma5"] = bool(close >= ma5)
    if pd.notna(close) and pd.notna(ma10):
        out["above_ma10"] = bool(close >= ma10)
    if pd.notna(close) and pd.notna(ma20):
        out["above_ma20"] = bool(close >= ma20)
    if pd.notna(close) and pd.notna(ma60):
        out["above_ma60"] = bool(close >= ma60)

    # 三陽開泰 / 四海遊龍
    if pd.notna(close) and all(pd.notna(x) for x in [ma5, ma10, ma20]):
        out["tri"] = (close >= ma5) and (close >= ma10) and (close >= ma20)
        out["tri_strong"] = out["tri"] and (ma5 > ma10 > ma20)

    if pd.notna(close) and all(pd.notna(x) for x in [ma5, ma10, ma20, ma60]):
        out["four"] = out["tri"] and (close >= ma60)

    # If user feels "above all MAs" but signals are off, this note explains the short MA gate
    if (not out["tri"] or not out["four"]) and pd.notna(close) and pd.notna(ma20) and pd.notna(ma60):
        if (close >= ma20) and (close >= ma60) and ((pd.notna(ma5) and close < ma5) or (pd.notna(ma10) and close < ma10)):
            out["notes"].append("價格在20/60MA之上，但尚未重新站回5/10MA，因此三陽開泰/四海遊龍未成立")

    # 一條線策略：預設大型權值股與記憶體/面板主題用 10MA
    if profile == "大型權值股" or hit_theme(stock_id, "記憶體族群") or hit_theme(stock_id, "面板族群"):
        out["key_ma"] = 10
    else:
        out["key_ma"] = 10

    # keyline hold/break（避免被洗：用「連兩日」才算有效跌破）
    if len(df) >= 2 and pd.notna(df["MA10"].iloc[-1]) and pd.notna(df["MA10"].iloc[-2]):
        c1 = float(df["close"].iloc[-1])
        c2 = float(df["close"].iloc[-2])
        m1 = float(df["MA10"].iloc[-1])
        m2 = float(df["MA10"].iloc[-2])

        out["key_hold"] = (c1 >= m1)
        out["key_break_2d"] = (c1 < m1) and (c2 < m2)

    # 量價：突破前高必須帶量（用20日區間前高 + 相對量）
    high_col = _get_col(df, ["max", "high"]) or "close"
    df[high_col] = pd.to_numeric(df[high_col], errors="coerce")
    prev20_high = df[high_col].rolling(20).max().shift(1).iloc[-1] if len(df) >= 21 else np.nan

    vr = compute_volume_ratio_series(df, window=5)
    vr_last = float(vr.iloc[-1]) if vr is not None and len(vr) else None
    vol_confirm = 1.2 if profile == "大型權值股" else 1.5

    if pd.notna(prev20_high) and pd.notna(close) and close > float(prev20_high):
        out["breakout_need_volume"] = True
        if vr_last is None or vr_last < vol_confirm:
            out["notes"].append(f"突破前高但量能未達確認（相對量 {vr_last if vr_last is not None else 'N/A'} < {vol_confirm}）")

    # 大量後不能快速縮量：找近5日是否出現「大量日」，若有，後兩日量比不應掉到過低
    if vr is not None and len(df) >= 5:
        recent = df.tail(5).copy()
        recent_vr = vr.tail(5).reset_index(drop=True)
        surge_idx = None
        for i in range(len(recent)-1, -1, -1):
            chg = float(recent["change_rate"].iloc[i]) if pd.notna(recent["change_rate"].iloc[i]) else 0.0
            if pd.notna(recent_vr.iloc[i]) and float(recent_vr.iloc[i]) >= 1.8 and chg > 0:
                surge_idx = i
                break

        if surge_idx is not None and surge_idx < len(recent)-1:
            after = recent_vr.iloc[surge_idx+1:]
            # 若後續量比立刻掉到 <0.7 視為縮量過快（反轉風險）
            out["post_surge_volume_ok"] = bool((after.dropna() >= 0.7).all()) if len(after.dropna()) else None
            if out["post_surge_volume_ok"] is False:
                out["notes"].append("大量後快速縮量（反轉風險提高）")

    # 無視單一K棒：若單日長黑但未破10MA/20MA關鍵支撐，視為洗盤可能
    if pd.notna(df["change_rate"].iloc[-1]) and float(df["change_rate"].iloc[-1]) <= -3.0:
        if pd.notna(ma10) and close >= ma10:
            out["washout_ignore"] = True
            out["notes"].append("出現長黑但仍守住10MA：偏洗盤，勿因單一K棒翻空")

    # 訊號整理
    if out["tri_strong"]:
        out["notes"].append("形態：三陽開泰（強勢版：5>10>20 且站上三均）")
    elif out["tri"]:
        out["notes"].append("形態：三陽開泰（站上5/10/20）")
    if out["four"]:
        out["notes"].append("形態：四海遊龍（站上短中期均線）")
    if out["key_break_2d"]:
        out["notes"].append("一條線策略：連兩日跌破10MA（續抱條件失效）")
    elif out["key_hold"]:
        out["notes"].append("一條線策略：守住10MA（可續抱/偏多）")

    return out


def compose_oldwang_decision(
    is_holding: bool,
    bias_level: str,
    oldwang: dict,
    leader: dict,
    chip_summary: dict,
    contrarian_flag: bool,
) -> dict:
    """
    產出「持股判斷依據」與建議動作（老王策略 + 籌碼分析）。
    bias_level: ok/warn/buy/danger/na
    """
    action = "等待"
    reasons: list[str] = []

    # 反向警訊（手動）
    if contrarian_flag:
        reasons.append("反向警訊：市場/外資過度樂觀（手動標記），短線需防反轉")

    # 籌碼摘要（寫入持股判斷依據）
    chip = chip_summary or {}
    chip_sig = chip.get("signal", "中性")

    net5 = chip.get("net_5d_total")
    f5 = chip.get("net_5d_foreign")
    t5 = chip.get("net_5d_trust")
    d5 = chip.get("net_5d_dealer")
    m5 = chip.get("margin_delta_5d")

    def fmt(v):
        return "-" if v is None or pd.isna(v) else f"{float(v):,.0f}"

    if chip_sig == "偏多":
        reasons.append(f"籌碼偏多：法人5日 {fmt(net5)}；外資5日 {fmt(f5)}；投信5日 {fmt(t5)}；融資5日 {fmt(m5)}")
    elif chip_sig == "偏空":
        reasons.append(f"籌碼偏空：法人5日 {fmt(net5)}；外資5日 {fmt(f5)}；投信5日 {fmt(t5)}；融資5日 {fmt(m5)}")
    else:
        if any(v is not None for v in [net5, f5, t5, m5]):
            reasons.append(f"籌碼中性：法人5日 {fmt(net5)}；外資5日 {fmt(f5)}；投信5日 {fmt(t5)}；融資5日 {fmt(m5)}")

    # 買強不買弱（領導股）
    if not leader.get("is_leader", False):
        rank = leader.get("rank")
        reasons.append(f"買強不買弱：此股非族群Top3領導股（族群排名 {rank if rank else '-'}），優先關注領導股")

    # -------- 持股模式：續抱/加碼/減碼 --------
    if is_holding:
        # 一條線策略：連兩日破10MA -> 續抱條件失效
        if oldwang.get("key_break_2d"):
            action = "減碼/出場"
            reasons.append("一條線策略失守：連兩日跌破10MA，續抱條件失效")
        else:
            # 乖離過大：禁止追高（持股以不加碼為主）
            if bias_level == "danger":
                action = "續抱但不加碼"
                reasons.append("乖離過大：禁止追高；持股以移動停利/不加碼為主")
            else:
                # 籌碼偏空且跌破20MA：偏轉弱
                if chip_sig == "偏空" and (oldwang.get("above_ma20") is False):
                    action = "減碼/出場"
                    reasons.append("籌碼轉弱且跌破20MA：偏結構轉空，建議控風險")
                # 型態成立 + 籌碼偏多：可續抱甚至加碼
                elif (oldwang.get("four") or oldwang.get("tri_strong") or oldwang.get("tri")) and chip_sig == "偏多":
                    action = "續抱/可擇機加碼"
                    reasons.append("型態成立且籌碼偏多：符合買強續抱邏輯")
                # 型態成立但籌碼未同步：續抱不加碼
                elif (oldwang.get("four") or oldwang.get("tri_strong") or oldwang.get("tri")) and chip_sig != "偏多":
                    action = "續抱但不加碼"
                    reasons.append("型態成立但籌碼未同步：續抱觀察，不加碼")
                else:
                    action = "續抱觀察"
                    reasons.append("尚未明確轉空：先守10MA/20MA，等待型態與籌碼明朗")

    # -------- 非持股模式：是否進場 --------
    else:
        if bias_level == "danger":
            action = "等待"
            reasons.append("乖離過大：禁止追高，等回測/整理後再評估")
        elif chip_sig == "偏空":
            action = "等待"
            reasons.append("籌碼偏空：不急著進，等法人回補/量價轉強")
        else:
            # 最佳：領導股 + 四海/三陽強勢 + 籌碼偏多
            if leader.get("is_leader", False) and (oldwang.get("four") or oldwang.get("tri_strong")) and chip_sig == "偏多":
                action = "可分批進場"
                reasons.append("領導股 + 四海/三陽強勢 + 籌碼偏多：符合買強不買弱")
            # 次佳：三陽/四海成立但條件未滿（等待拉回）
            elif oldwang.get("tri") or oldwang.get("four"):
                action = "等待拉回"
                reasons.append("型態成立但條件未滿：偏等回測10MA/20MA或帶量突破再進")
            else:
                action = "等待"
                reasons.append("型態未成形：等待三陽開泰/四海遊龍或回測承接點")

    # -------- 量價風險提醒（來自老王檢核）--------
    if oldwang.get("breakout_need_volume"):
        reasons.append("量價規則：突破前高需帶量；若無量不追")
    if oldwang.get("post_surge_volume_ok") is False:
        reasons.append("量價風險：大量後快速縮量，留意反轉")
    if oldwang.get("washout_ignore"):
        reasons.append("無視單一K棒：長黑但守住10MA，偏洗盤，勿因單日翻空")

    return {"action": action, "reasons": reasons}

def build_trade_plan(struct: dict, pattern_info: dict, vol_quality: dict, bias_alert: dict, score: dict) -> dict:
    close = struct.get("close", np.nan)
    ma20 = struct.get("ma20", np.nan)
    ma60 = struct.get("ma60", np.nan)
    prev20_high = struct.get("prev20_high", np.nan)
    prev20_low = struct.get("prev20_low", np.nan)
    prev60_high = struct.get("prev60_high", np.nan)
    prev60_low = struct.get("prev60_low", np.nan)
    range20 = struct.get("range20", np.nan)

    pattern = pattern_info.get("pattern", "盤整盤")
    entry_type = pattern_info.get("entry_type", "等待")
    vol_confirm = pattern_info.get("vol_confirm", None)

    total = float(score.get("total", 0)) if isinstance(score, dict) else 0.0

    supports = [x for x in [ma20, prev20_low, ma60, prev60_low] if pd.notna(x)]
    resistances = [x for x in [prev20_high, prev60_high] if pd.notna(x)]
    supports = sorted({round(float(x), 2) for x in supports})
    resistances = sorted({round(float(x), 2) for x in resistances})

    buffer = 0.985
    stop = None
    t1 = None
    t2 = None

    lines = []
    lines.append(f"型態判定：**{pattern}**（{entry_type}）")
    lines.append(f"量價品質：**{vol_quality.get('label','-')}** — {vol_quality.get('detail','')}")
    lines.append("")

    # If bias danger => forbid chase
    if (bias_alert or {}).get("level") == "danger":
        lines.append("結論：**❌ 乖離過大，禁止追高。**")
        lines.append("策略：只接受『回測承接』或『整理後再突破』，不在高乖離區追價。")
        if pd.notna(ma20):
            stop = float(ma20) * buffer
            lines.append(f"若已持有：可用 20MA 下方緩衝作為風險點（約 {stop:.2f}）。")
        return {"summary": "\n".join(lines), "supports": supports, "resistances": resistances, "stop": stop, "t1": t1, "t2": t2, "stance": "禁止追高"}

    if pattern == "突破盤" and pd.notna(prev20_high):
        lines.append("交易計畫（突破型）：")
        if vol_confirm is not None:
            lines.append(f"- 進場條件：站上 **{prev20_high:.2f}**，且相對量 ≥ **{vol_confirm:.2f}**（越接近越好）。")
        else:
            lines.append(f"- 進場條件：站上 **{prev20_high:.2f}** 並有量能配合。")
        lines.append("- 操作原則：突破後不追高，優先等回測不破前高再分批。")

        base_stop = prev20_high
        if pd.notna(ma20):
            base_stop = min(base_stop, ma20)
        stop = float(base_stop) * buffer if pd.notna(base_stop) else None
        lines.append(f"- 停損：跌破 **{stop:.2f}**（前高/20MA 下方緩衝）視為假突破。")

        if pd.notna(prev60_high):
            t1 = float(prev60_high)
        if pd.notna(range20) and pd.notna(prev20_high):
            measured = float(prev20_high + range20)
            if t1 is None:
                t1 = measured
            t2 = float(prev20_high + 2 * range20)

        if t1 is not None:
            lines.append(f"- 目標1：**{t1:.2f}**，到位可先落袋。")
        if t2 is not None:
            lines.append(f"- 目標2：**{t2:.2f}**，以移動停利追蹤。")

    elif pattern == "拉回盤" and pd.notna(ma20):
        lines.append("交易計畫（拉回型）：")
        lines.append(f"- 進場區：**20MA 附近（約 {ma20:.2f}）** 分批試單，觀察是否守住。")

        base_stop = ma60 if pd.notna(ma60) else prev20_low
        if pd.notna(prev20_low) and pd.notna(base_stop):
            base_stop = min(base_stop, prev20_low)
        stop = float(base_stop) * buffer if pd.notna(base_stop) else None
        lines.append(f"- 停損：跌破 **{stop:.2f}**（60MA/區間低點下方緩衝）視為承接失敗。")

        if pd.notna(prev20_high):
            t1 = float(prev20_high)
            lines.append(f"- 目標1：回到 **{t1:.2f}**（20日區間前高）。")
        if pd.notna(prev60_high):
            t2 = float(prev60_high)
            lines.append(f"- 目標2：上看 **{t2:.2f}**（60日壓力/前高）。")

    elif pattern == "轉弱破底":
        lines.append("交易計畫（轉弱破底）：")
        lines.append("- 結構已破，不建議追多。")
        if pd.notna(prev20_low):
            lines.append(f"- 觀察：能否站回 **{prev20_low:.2f}** 並縮量止跌。")
        if pd.notna(ma20):
            lines.append(f"- 反轉條件：至少站回 20MA（約 {ma20:.2f}）才重新評估。")
        return {"summary": "\n".join(lines), "supports": supports, "resistances": resistances, "stop": None, "t1": None, "t2": None, "stance": "偏空/避開"}

    else:
        lines.append("交易計畫（盤整/等待型）：")
        if pd.notna(prev20_high):
            lines.append(f"- 突破條件：站上 **{prev20_high:.2f}** 再轉積極。")
        if pd.notna(ma20):
            lines.append(f"- 拉回承接：回測 **20MA {ma20:.2f}** 不破才有較佳RR。")
        if pd.notna(prev20_low):
            lines.append(f"- 風險點：跌破 **{prev20_low:.2f}** 代表區間下緣失守。")

    # position suggestion
    if total >= 80:
        pos = "可採『試單 1/3 → 確認再加碼』"
    elif total >= 60:
        pos = "建議『小部位試單』，以確認型態為主"
    else:
        pos = "偏向觀望，等待更明確型態/量能"
    lines.append("")
    lines.append(f"部位建議：{pos}（總分 {total:.0f}/100）")
    lines.append("備註：此為技術面規則化交易計畫，不構成投資建議。")

    stance = "可留意" if total >= 60 else "觀望"
    return {"summary": "\n".join(lines), "supports": supports, "resistances": resistances, "stop": stop, "t1": t1, "t2": t2, "stance": stance}


# -----------------------------
# NEW (Upgrade 2): Volume-price quality integrated above
# -----------------------------

# -----------------------------
# NEW (Upgrade 3): Sector breadth & leadership concentration
# -----------------------------
def compute_sector_strength(market_df: pd.DataFrame, sector_flow: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if market_df is None or market_df.empty or sector_flow is None or sector_flow.empty:
        return pd.DataFrame()

    df = market_df.copy()
    df["industry_category"] = df.get("industry_category", "其他").fillna("其他")
    df = ensure_change_rate(df)

    money_col = pick_money_col(df)
    df[money_col] = pd.to_numeric(df[money_col], errors="coerce").fillna(0.0)

    top_sectors = sector_flow.sort_values("signed_money", ascending=False)["industry_category"].head(top_n).tolist()

    rows = []
    for sec in top_sectors:
        sub = df[df["industry_category"] == sec].copy()
        if sub.empty:
            continue

        total_cnt = int(len(sub))
        up_cnt = int((sub["change_rate"] > 0).sum())
        dn_cnt = int((sub["change_rate"] < 0).sum())
        eq_cnt = total_cnt - up_cnt - dn_cnt

        total_money = float(sub[money_col].sum())
        signed_money = float((sub[money_col] * np.sign(sub["change_rate"])).sum())
        top3_money = float(sub.sort_values(money_col, ascending=False).head(3)[money_col].sum())
        conc = safe_div(top3_money, total_money, default=np.nan)

        rows.append({
            "族群": sec,
            "資金偏多(億)": round(signed_money / 1e8, 2),
            "總成交金額(億)": round(total_money / 1e8, 2),
            "上漲家數": up_cnt,
            "下跌家數": dn_cnt,
            "平盤家數": eq_cnt,
            "上漲比例(%)": round(safe_div(up_cnt, total_cnt, 0.0) * 100, 1),
            "領導集中度Top3(%)": round(conc * 100, 1) if pd.notna(conc) else np.nan,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Ten-sector leaders (from prior upgrade)
# -----------------------------
def build_sector_leaders(
    vol_rank: pd.DataFrame,
    sector_flow: pd.DataFrame,
    top_sectors: int = 10,
    k: int = 5,
    only_up: bool = True,
) -> dict:
    out = {}
    if vol_rank is None or vol_rank.empty or sector_flow is None or sector_flow.empty:
        return out

    df = vol_rank.copy()
    df["industry_category"] = df.get("industry_category", "其他").fillna("其他")
    df["stock_id"] = df["stock_id"].astype(str)
    df["stock_name"] = df.get("stock_name", "").fillna("")

    if "volume_ratio" not in df.columns:
        df["volume_ratio"] = 0.0
    df["volume_ratio"] = pd.to_numeric(df["volume_ratio"], errors="coerce").fillna(0.0)

    df = ensure_change_rate(df)

    money_col = pick_money_col(df)
    df[money_col] = pd.to_numeric(df[money_col], errors="coerce").fillna(0.0)

    vol_col = pick_volume_col(df)
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)

    # liquidity gate
    q = df[money_col].quantile(0.30)
    df = df[df[money_col] >= q].copy()

    top_sector_names = sector_flow.sort_values("signed_money", ascending=False)["industry_category"].head(top_sectors).tolist()

    for sec in top_sector_names:
        sub = df[df["industry_category"] == sec].copy()
        if sub.empty:
            continue
        if only_up:
            sub = sub[sub["change_rate"] > 0].copy()
            if sub.empty:
                continue

        sub["score"] = sub["change_rate"].rank(pct=True) * 0.6 + sub["volume_ratio"].rank(pct=True) * 0.4
        sub = sub.sort_values(["score", money_col], ascending=False).head(k).copy()

        show_cols = ["stock_id", "stock_name", "change_rate", "volume_ratio", money_col, vol_col]
        if "close" in sub.columns:
            show_cols.append("close")

        out[sec] = sub[show_cols].rename(columns={
            "change_rate": "漲跌幅(%)",
            "volume_ratio": "量比",
            money_col: "成交金額",
            vol_col: "成交量",
            "close": "價格",
        })
    return out


# -----------------------------
# Theme radar (with breadth & concentration)
# -----------------------------
def compute_theme_radar(
    market_df: pd.DataFrame,
    stock_info: pd.DataFrame,
    theme_groups: dict,
    top_k: int = 10,
    money_threshold_yi: float = 1.0,
) -> dict:
    if market_df is None or market_df.empty:
        return {"summary": pd.DataFrame(), "leaders": {}}

    df = market_df.copy()
    df["stock_id"] = df["stock_id"].astype(str)

    if "stock_name" not in df.columns or "industry_category" not in df.columns:
        info = stock_info[["stock_id", "stock_name", "industry_category"]].drop_duplicates()
        info["stock_id"] = info["stock_id"].astype(str)
        df = df.merge(info, on="stock_id", how="left")

    df["stock_name"] = df.get("stock_name", "").fillna("")
    df["industry_category"] = df.get("industry_category", "其他").fillna("其他")

    df = ensure_change_rate(df)
    money_col = pick_money_col(df)
    vol_col = pick_volume_col(df)

    df[money_col] = pd.to_numeric(df[money_col], errors="coerce").fillna(0.0)
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)

    df["signed_money"] = df[money_col] * np.sign(df["change_rate"].fillna(0.0))

    summary_rows = []
    leaders: dict = {}
    money_threshold = float(money_threshold_yi) * 1e8

    for theme, rule in theme_groups.items():
        industry_list = rule.get("industry", []) or []
        stock_list = [str(x) for x in (rule.get("stocks", []) or [])]

        mask = pd.Series(False, index=df.index)
        if industry_list:
            mask = mask | df["industry_category"].isin(industry_list)
        if stock_list:
            mask = mask | df["stock_id"].isin(stock_list)

        sub = df[mask].copy()
        if sub.empty:
            summary_rows.append({
                "主題": theme,
                "總成交金額(億)": 0.0,
                "資金偏多(億)": 0.0,
                "平均漲跌幅(%)": 0.0,
                "中位數漲跌幅(%)": 0.0,
                "上漲家數": 0,
                "下跌家數": 0,
                "平盤家數": 0,
                "上漲比例(%)": 0.0,
                "領導集中度Top3(%)": np.nan,
            })
            leaders[theme] = pd.DataFrame()
            continue

        total_cnt = int(len(sub))
        up_cnt = int((sub["change_rate"] > 0).sum())
        dn_cnt = int((sub["change_rate"] < 0).sum())
        eq_cnt = total_cnt - up_cnt - dn_cnt

        total_money = float(sub[money_col].sum())
        signed_money = float(sub["signed_money"].sum())
        avg_chg = float(sub["change_rate"].mean())
        med_chg = float(sub["change_rate"].median())

        top3_money = float(sub.sort_values(money_col, ascending=False).head(3)[money_col].sum())
        conc = safe_div(top3_money, total_money, default=np.nan)

        summary_rows.append({
            "主題": theme,
            "總成交金額(億)": round(total_money / 1e8, 2),
            "資金偏多(億)": round(signed_money / 1e8, 2),
            "平均漲跌幅(%)": round(avg_chg, 2),
            "中位數漲跌幅(%)": round(med_chg, 2),
            "上漲家數": up_cnt,
            "下跌家數": dn_cnt,
            "平盤家數": eq_cnt,
            "上漲比例(%)": round(safe_div(up_cnt, total_cnt, 0.0) * 100, 1),
            "領導集中度Top3(%)": round(conc * 100, 1) if pd.notna(conc) else np.nan,
        })

        pick = sub[sub[money_col] >= money_threshold].copy()
        if pick.empty:
            pick = sub.copy()
        pick = pick.sort_values(["change_rate", money_col], ascending=[False, False]).head(top_k).copy()

        show_cols = ["stock_id", "stock_name", "industry_category", "change_rate", money_col, vol_col]
        if "volume_ratio" in pick.columns:
            show_cols.append("volume_ratio")
        if "close" in pick.columns:
            show_cols.append("close")

        pick = pick[show_cols].rename(columns={
            "industry_category": "產業",
            "change_rate": "漲跌幅(%)",
            money_col: "成交金額",
            vol_col: "成交量",
            "volume_ratio": "量比",
            "close": "價格",
        })
        pick["成交金額(億)"] = pick["成交金額"] / 1e8
        leaders[theme] = pick

    summary = pd.DataFrame(summary_rows).sort_values("資金偏多(億)", ascending=False).reset_index(drop=True)
    return {"summary": summary, "leaders": leaders}


# -----------------------------
# Bias alert system
# -----------------------------
def classify_stock_profile(stock_id: str, stock_info: pd.DataFrame) -> str:
    sid = str(stock_id).strip()
    if sid in LARGE_CAP_IDS:
        return "大型權值股"
    try:
        row = stock_info[stock_info["stock_id"].astype(str) == sid]
        if not row.empty:
            ind = str(row.iloc[-1].get("industry_category", "") or "")
            if ind == "金融保險業":
                return "大型權值股"
    except Exception:
        pass
    return "中小型飆股"


def get_bias_alert(bias20: float, bias60: float, profile: str) -> dict:
    out = {"level": "na", "headline": "乖離率資料不足", "detail": "", "thresholds": {}}
    if bias20 is None or pd.isna(bias20):
        return out

    b20 = float(bias20)
    b60 = float(bias60) if (bias60 is not None and not pd.isna(bias60)) else np.nan

    endings = (
        "乖離過大常見三種結局：\n"
        "1) 直接崩跌：股價快速下跌去找均線。\n"
        "2) 橫盤整理：以盤代跌，等均線慢慢上來。\n"
        "3) 假突破真拉回：再漲一點點，然後瞬間急殺。"
    )

    if profile == "大型權值股":
        out["thresholds"] = {"warn_hi": 10.0, "warn_lo": -10.0}
        if b20 >= 10.0:
            out["level"] = "danger"
            out["headline"] = f"❌ 乖離過大，禁止追高（20MA乖離 {b20:.2f}% ≥ +10%）"
            out["detail"] = "大型權值股推動不易，+10% 以上常見波段高檔。\n操作建議：不追價；偏向分批調節。\n\n" + endings
        elif b20 <= -10.0:
            out["level"] = "buy"
            out["headline"] = f"✅ 超賣區（20MA乖離 {b20:.2f}% ≤ -10%）"
            out["detail"] = "大型權值股 -10% 常見恐慌超賣。\n操作建議：可留意分批布局，仍以止跌/轉強確認。"
        elif b20 >= 6.0:
            out["level"] = "warn"
            out["headline"] = f"⚠️ 乖離偏高（20MA乖離 {b20:.2f}%）"
            out["detail"] = "距離 +10% 警戒線不遠，RR開始下降。\n操作建議：避免追高，等回測20MA/整理。"
        else:
            out["level"] = "ok"
            out["headline"] = f"乖離合理（20MA乖離 {b20:.2f}%）"
            out["detail"] = "乖離未達警戒，仍需搭配趨勢/量能/籌碼。"
    else:
        out["thresholds"] = {"warn_hi": 15.0, "danger_hi": 20.0, "mania": 30.0, "warn_lo": -12.0}
        if b20 >= 30.0:
            out["level"] = "danger"
            out["headline"] = f"❌ 瘋狂區：禁止追高（20MA乖離 {b20:.2f}% ≥ +30%）"
            out["detail"] = "末升段噴出，RR極差。\n操作建議：不追價；偏向落袋/移動停利。\n\n" + endings
        elif b20 >= 20.0:
            out["level"] = "danger"
            out["headline"] = f"❌ 高風險：禁止追高（20MA乖離 {b20:.2f}% ≥ +20%）"
            out["detail"] = "+20% 隨時可能長黑修正。\n操作建議：不追價；偏向移動停利。\n\n" + endings
        elif b20 >= 15.0:
            out["level"] = "warn"
            out["headline"] = f"⚠️ 警戒區（20MA乖離 {b20:.2f}% ≥ +15%）"
            out["detail"] = "+15% 後勝率與RR開始變差。\n操作建議：寧可錯過，不可做錯；等回測/整理。"
        elif b20 <= -12.0:
            out["level"] = "buy"
            out["headline"] = f"✅ 偏超賣（20MA乖離 {b20:.2f}% ≤ -12%）"
            out["detail"] = "中小型股超賣彈性大但也可能續跌。\n操作建議：分批試單 + 嚴格停損。"
        else:
            out["level"] = "ok"
            out["headline"] = f"乖離合理（20MA乖離 {b20:.2f}%）"
            out["detail"] = "可搭配趨勢/量能/題材強度綜合判讀。"

    if pd.notna(b60) and abs(b60) >= 25 and out["level"] in ["ok", "warn", "buy"]:
        out["detail"] += f"\n\n補充：60MA 乖離 {b60:.2f}% 已偏極端（中期過熱/過冷），請提高風險控管。"
    return out


def get_ai_advice(stock_id: str, stock_info: pd.DataFrame, price_df: pd.DataFrame, score: dict, bias_profile_mode: str) -> dict:
    if bias_profile_mode == "大型權值股":
        profile = "大型權值股"
    elif bias_profile_mode == "中小型飆股":
        profile = "中小型飆股"
    else:
        profile = classify_stock_profile(stock_id, stock_info)

    b20 = np.nan
    b60 = np.nan
    if price_df is not None and not price_df.empty:
        if "BIAS20" in price_df.columns and pd.notna(price_df["BIAS20"].iloc[-1]):
            b20 = float(price_df["BIAS20"].iloc[-1])
        if "BIAS60" in price_df.columns and pd.notna(price_df["BIAS60"].iloc[-1]):
            b60 = float(price_df["BIAS60"].iloc[-1])

    bias_alert = get_bias_alert(b20, b60, profile)

    total = float(score.get("total", 0)) if isinstance(score, dict) else 0.0
    stance = "中性"
    if total >= 75:
        stance = "偏多"
    elif total <= 45:
        stance = "偏空"

    return {"profile": profile, "bias_alert": bias_alert, "stance": stance}



# -----------------------------
# 10) 老王選股器（基於 5/10/20/60 + 三陽開泰/四海遊龍 + 量價/領導股/守10MA）
# -----------------------------
def _is_four_digit_stock_id(s: str) -> bool:
    s = str(s).strip()
    return len(s) == 4 and s.isdigit()

def _profile_for_sid(stock_id: str, stock_info: pd.DataFrame) -> str:
    sid = str(stock_id).strip()
    if sid in LARGE_CAP_IDS:
        return "大型權值股"
    try:
        row = stock_info[stock_info["stock_id"].astype(str) == sid]
        if not row.empty:
            ind = str(row.iloc[-1].get("industry_category", "") or "")
            if ind == "金融保險業":
                return "大型權值股"
    except Exception:
        pass
    return "中小型飆股"

def _bias_danger_threshold(profile: str) -> float:
    return 10.0 if profile == "大型權值股" else 20.0

def _bias_warn_threshold(profile: str) -> float:
    return 6.0 if profile == "大型權值股" else 15.0

def _vol_confirm_threshold(profile: str) -> float:
    # 老王：突破要帶量；權值股門檻可低一些
    return 1.2 if profile == "大型權值股" else 1.5

def _simple_vol_quality(chg_pct: float, vol_ratio: float) -> str:
    if pd.isna(chg_pct) or pd.isna(vol_ratio):
        return "N/A"
    if vol_ratio >= 1.8 and chg_pct > 1:
        return "放量上漲"
    if vol_ratio >= 1.5 and abs(chg_pct) < 0.5:
        return "放量不漲"
    if vol_ratio >= 1.5 and chg_pct < -1:
        return "放量下跌"
    if vol_ratio <= 0.8 and abs(chg_pct) < 1:
        return "量縮整理"
    return "一般"

@st.cache_data(ttl=1800)
def get_multi_stock_price_cached(token: str, stock_ids: list[str], start_date: str) -> pd.DataFrame:
    """
    優先嘗試一次抓多檔（若 API 支援 data_id=list），不支援則降級逐檔抓取（僅限候選池）。
    """
    if not stock_ids:
        return pd.DataFrame()
    # Try multi-id in one call
    try:
        df = finmind_get_data(token, dataset="TaiwanStockPrice", data_id=stock_ids, start_date=start_date, timeout=120)
        df = normalize_date_col(df, "date")
        if not df.empty and "stock_id" in df.columns:
            return df
    except Exception:
        pass

    # Fallback: per-stock loop
    frames = []
    for sid in stock_ids:
        try:
            dfi = finmind_get_data(token, dataset="TaiwanStockPrice", data_id=sid, start_date=start_date, timeout=40)
            if dfi is not None and not dfi.empty:
                frames.append(dfi)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return normalize_date_col(df, "date")


@st.cache_data(ttl=1800)
def get_candidate_history_dailyall_cached(token: str, stock_ids: list[str], end_date: str, lookback: int = 80) -> pd.DataFrame:
    """
    以「每日全市場日線」拼出候選股歷史（避免逐檔抓取過慢/不穩）。
    - 會抓 end_date 往前 lookback 個交易日（以 TaiwanStockTradingDate 為準）
    - 每個交易日打一支 API（TaiwanStockPrice + start_date=該日），再過濾出 stock_ids
    """
    if not stock_ids:
        return pd.DataFrame()

    try:
        tdf = get_trading_dates_cached(token)
        if tdf is None or tdf.empty or "date" not in tdf.columns:
            return pd.DataFrame()
        tdf = tdf.sort_values("date")
        dates = [d for d in tdf["date"].astype(str).tolist() if d <= str(end_date)]
        dates = dates[-int(lookback):]
    except Exception:
        return pd.DataFrame()

    stock_set = set([str(x) for x in stock_ids])

    frames = []
    for d in dates:
        try:
            day = get_daily_all_cached(token, d)
            if day is None or day.empty or "stock_id" not in day.columns:
                continue
            day = day.copy()
            day["stock_id"] = day["stock_id"].astype(str)
            day = day[day["stock_id"].isin(stock_set)].copy()
            if day.empty:
                continue

            keep = [c for c in ["stock_id", "date", "open", "max", "min", "close", "Trading_Volume", "Trading_money", "spread"] if c in day.columns]
            day = day[keep].copy()
            frames.append(day)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, ignore_index=True)
    hist = normalize_date_col(hist, "date")
    hist["stock_id"] = hist["stock_id"].astype(str)

    for c in ["close", "open", "max", "min", "Trading_Volume", "Trading_money", "spread"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")

    return hist

def oldwang_screener(
    token: str,
    stock_info: pd.DataFrame,
    vol_rank_today: pd.DataFrame,
    universe_top_n: int = 300,
    output_top_k: int = 80,
    require_leader: bool = True,
    require_pattern: str = "不限",
    require_breakout: bool = False,
    min_money_yi: float = 1.0,
) -> pd.DataFrame:
    """
    依據老王策略做選股（全市場掃描）：
      - 均線：MA5/MA10 判斷極短線；MA20（月線）判斷多頭；MA60 中期
      - 形態：三陽開泰（站上 MA5/10/20）、四海遊龍（站上 MA5/10/20/60）
      - 量價：突破前高壓力必須帶量（相對量門檻依股性）
      - 量的延續：大量後不能快速縮量（縮量警訊）
      - 一條線：守 10MA，不跌破就抱（連兩日跌破才視為失守，避免單一長黑洗盤）
      - 買強不買弱：只做族群領導股（以當日成交金額 Top3 代理）
      - 乖離率：過熱禁止追高（依大型/中小股容忍度）
    """
    if vol_rank_today is None or vol_rank_today.empty:
        return pd.DataFrame()

    df0 = vol_rank_today.copy()
    if "stock_id" not in df0.columns:
        return pd.DataFrame()

    df0["stock_id"] = df0["stock_id"].astype(str)
    df0["stock_name"] = df0.get("stock_name", "").fillna("")
    df0["industry_category"] = df0.get("industry_category", "其他").fillna("其他")

    # As-of date
    as_of_date = _now_date_str()
    if "scan_date" in df0.columns and len(df0) and pd.notna(df0["scan_date"].iloc[0]):
        as_of_date = str(df0["scan_date"].iloc[0])
    elif "date" in df0.columns and len(df0) and pd.notna(df0["date"].iloc[0]):
        try:
            as_of_date = pd.to_datetime(df0["date"].iloc[0]).strftime("%Y-%m-%d")
        except Exception:
            as_of_date = _now_date_str()

    # Ensure change_rate & volume_ratio exist
    df0 = ensure_change_rate(df0)
    if "volume_ratio" not in df0.columns:
        # fallback: approximate from volume / yesterday volume if exists, else 0
        tv = pd.to_numeric(df0.get("total_volume", df0.get("Trading_Volume", 0)), errors="coerce")
        yv = pd.to_numeric(df0.get("yesterday_volume", np.nan), errors="coerce")
        df0["volume_ratio"] = tv / yv.replace(0, np.nan)
    df0["volume_ratio"] = pd.to_numeric(df0["volume_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    money_col = pick_money_col(df0)
    df0[money_col] = pd.to_numeric(df0[money_col], errors="coerce").fillna(0.0)

    # liquidity gate
    min_money = float(min_money_yi) * 1e8
    df0 = df0[df0[money_col] >= min_money].copy()
    if df0.empty:
        return pd.DataFrame()

    # Universe by money
    df0 = df0.sort_values(money_col, ascending=False).head(int(universe_top_n)).copy()
    candidate_ids = df0["stock_id"].tolist()
    if not candidate_ids:
        return pd.DataFrame()

    # Leader flag (industry Top3 by money)
    df0["industry_rank_money"] = df0.groupby("industry_category")[money_col].rank(method="first", ascending=False)
    df0["is_leader"] = df0["industry_rank_money"] <= 3

    # History for candidates (lookback trading days)
    hist = get_candidate_history_dailyall_cached(token, candidate_ids, as_of_date, lookback=80)
    if hist is None or hist.empty:
        # last-resort fallback: per-stock (slower)
        start_date = (datetime.now(tz=TZ) - timedelta(days=150) if TZ else datetime.now() - timedelta(days=150)).strftime("%Y-%m-%d")
        hist = get_multi_stock_price_cached(token, candidate_ids, start_date)
        if hist is None or hist.empty:
            return pd.DataFrame()

    hist["stock_id"] = hist["stock_id"].astype(str)
    hist = normalize_date_col(hist, "date")

    # Ensure numeric columns
    for c in ["close", "open", "max", "min", "Trading_Volume", "Trading_money", "spread"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")

    def _profile_for_sid(sid: str) -> str:
        return classify_stock_profile(sid, stock_info)

    def _vol_confirm_threshold(profile: str) -> float:
        return 1.2 if profile == "大型權值股" else 1.5

    def _bias_warn_threshold(profile: str) -> float:
        return 6.0 if profile == "大型權值股" else 15.0

    def _bias_danger_threshold(profile: str) -> float:
        return 10.0 if profile == "大型權值股" else 20.0

    rows = []
    for sid in candidate_ids:
        row0 = df0[df0["stock_id"] == sid]
        if row0.empty:
            continue
        row0 = row0.iloc[0]

        g = hist[hist["stock_id"] == sid].sort_values("date").copy()
        if g.empty or "close" not in g.columns:
            continue

        # Patch to as_of_date with today's snapshot close if history not updated to as_of_date yet
        last_hist_date = str(g["date"].iloc[-1]) if "date" in g.columns else ""
        patched_intraday = False
        if last_hist_date and last_hist_date < str(as_of_date):
            today_close = pd.to_numeric(row0.get("close", np.nan), errors="coerce")
            if pd.notna(today_close):
                patched_intraday = True
                today_open = pd.to_numeric(row0.get("open", today_close), errors="coerce")
                today_high = pd.to_numeric(row0.get("high", row0.get("max", today_close)), errors="coerce")
                today_low = pd.to_numeric(row0.get("low", row0.get("min", today_close)), errors="coerce")

                today_vol = pd.to_numeric(row0.get("total_volume", row0.get("Trading_Volume", np.nan)), errors="coerce")
                today_money = pd.to_numeric(row0.get("total_amount", row0.get("Trading_money", np.nan)), errors="coerce")

                add = {"stock_id": sid, "date": str(as_of_date), "close": float(today_close)}
                add["open"] = float(today_open) if pd.notna(today_open) else float(today_close)
                add["max"] = float(today_high) if pd.notna(today_high) else float(today_close)
                add["min"] = float(today_low) if pd.notna(today_low) else float(today_close)
                if "Trading_Volume" in g.columns and pd.notna(today_vol):
                    add["Trading_Volume"] = float(today_vol)
                if "Trading_money" in g.columns and pd.notna(today_money):
                    add["Trading_money"] = float(today_money)

                g = pd.concat([g, pd.DataFrame([add])], ignore_index=True).sort_values("date")

        # Need enough history
        if len(g) < 60:
            continue

        close = float(g["close"].iloc[-1])
        g["MA5"] = g["close"].rolling(5).mean()
        g["MA10"] = g["close"].rolling(10).mean()
        g["MA20"] = g["close"].rolling(20).mean()
        g["MA60"] = g["close"].rolling(60).mean()

        ma5 = float(g["MA5"].iloc[-1]) if pd.notna(g["MA5"].iloc[-1]) else np.nan
        ma10 = float(g["MA10"].iloc[-1]) if pd.notna(g["MA10"].iloc[-1]) else np.nan
        ma20 = float(g["MA20"].iloc[-1]) if pd.notna(g["MA20"].iloc[-1]) else np.nan
        ma60 = float(g["MA60"].iloc[-1]) if pd.notna(g["MA60"].iloc[-1]) else np.nan

        if pd.isna(ma20) or ma20 == 0:
            continue

        bias20 = (close / ma20 - 1) * 100

        # 三陽開泰 / 四海遊龍
        three = pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20) and close > ma5 and close > ma10 and close > ma20
        four = three and pd.notna(ma60) and close > ma60

        # Breakout: close > prev 20-day high (exclude today)
        high_col = "max" if "max" in g.columns else "close"
        g[high_col] = pd.to_numeric(g[high_col], errors="coerce")
        prev20_high = g[high_col].rolling(20).max().shift(1).iloc[-1]
        breakout = pd.notna(prev20_high) and close > float(prev20_high)

        # Volume conditions
        vol_ratio = float(row0.get("volume_ratio", 0.0))
        profile = _profile_for_sid(sid)
        vol_need = _vol_confirm_threshold(profile)
        bias_danger = _bias_danger_threshold(profile)
        bias_warn = _bias_warn_threshold(profile)

        forbid_chase = pd.notna(bias20) and bias20 >= bias_danger
        bias_state = "OK"
        if pd.notna(bias20) and bias20 >= bias_danger:
            bias_state = "DANGER"
        elif pd.notna(bias20) and bias20 >= bias_warn:
            bias_state = "WARN"

        vol_ok = (not pd.isna(vol_ratio)) and (vol_ratio >= vol_need)
        breakout_ok = breakout and vol_ok and (not forbid_chase)

        # 大量後不能快速縮量（用「完成日」避免盤中誤判）
        shrink_warn = False
        if "Trading_Volume" in g.columns:
            v = pd.to_numeric(g["Trading_Volume"], errors="coerce").fillna(0.0)
            # 若有盤中補丁，避免拿盤中量去判斷縮量：用「倒數第3天（大量）→倒數第2天（隔日）」檢查
            if patched_intraday and len(v) >= 8:
                v_surge = float(v.iloc[-3])
                v_after = float(v.iloc[-2])
                base = float(v.iloc[-8:-3].mean())
                if base > 0 and (v_surge / base) >= 1.5 and v_after < 0.6 * v_surge:
                    shrink_warn = True
            elif (not patched_intraday) and len(v) >= 7:
                v_surge = float(v.iloc[-2])
                v_after = float(v.iloc[-1])
                base = float(v.iloc[-7:-2].mean())
                if base > 0 and (v_surge / base) >= 1.5 and v_after < 0.6 * v_surge:
                    shrink_warn = True

        # 守10MA（避免被洗）：連兩日跌破才算失守
        hold10 = False
        break10_two_days = False
        if pd.notna(ma10) and ma10 != 0:
            hold10 = close >= ma10
            if len(g) >= 11 and pd.notna(g["MA10"].iloc[-2]):
                below_today = float(g["close"].iloc[-1]) < float(g["MA10"].iloc[-1])
                below_y = float(g["close"].iloc[-2]) < float(g["MA10"].iloc[-2])
                break10_two_days = below_today and below_y

        # Leader / industry / today stats
        leader = bool(row0.get("is_leader", False))
        industry = str(row0.get("industry_category", "其他") or "其他")
        name = str(row0.get("stock_name", "") or "")
        chg = float(row0.get("change_rate", 0.0))
        money = float(row0.get(money_col, 0.0))

        # Filters
        pass_filter_leader = (not require_leader) or leader

        if require_pattern == "三陽開泰":
            pass_filter_pattern = bool(three)
        elif require_pattern == "四海遊龍":
            pass_filter_pattern = bool(four)
        else:
            pass_filter_pattern = True

        pass_filter_breakout = (not require_breakout) or breakout_ok

        if not (pass_filter_leader and pass_filter_pattern and pass_filter_breakout):
            continue

        # 量價型態（快速標籤）
        try:
            vol_style = classify_volume_price(ensure_change_rate(g), vol_ratio).get("label", "一般")
        except Exception:
            vol_style = "一般"

        # 老王分數（規則化）
        s = 0
        reasons = []

        if leader:
            s += 20
            reasons.append("族群領導股")
        else:
            reasons.append("非領導股")

        if four:
            s += 25
            reasons.append("四海遊龍")
        elif three:
            s += 18
            reasons.append("三陽開泰")

        if breakout_ok:
            s += 22
            reasons.append("突破前高且帶量")
        elif breakout and vol_ok and forbid_chase:
            s -= 15
            reasons.append("突破但乖離過熱")
        elif breakout and not vol_ok:
            s += 8
            reasons.append("突破但量能未確認")

        if hold10:
            s += 10
            reasons.append("守10MA")
        if break10_two_days:
            s -= 10
            reasons.append("連兩日破10MA")

        if shrink_warn:
            s -= 12
            reasons.append("大量後快速縮量（反轉風險）")

        if bias_state == "WARN":
            s -= 8
            reasons.append("乖離偏高")
        elif bias_state == "DANGER":
            s -= 30
            reasons.append("乖離過熱（禁止追高）")

        if chg > 0:
            s += 3
        if vol_ratio >= vol_need:
            s += 3

        # Clamp score
        s = max(0, min(100, s))

        rows.append({
            "stock_id": sid,
            "stock_name": name,
            "industry_category": industry,
            "漲跌幅(%)": round(chg, 2),
            "成交金額(億)": round(money / 1e8, 2),
            "量比": round(vol_ratio, 2),
            "BIAS20(%)": round(bias20, 2) if pd.notna(bias20) else np.nan,
            "股性": profile,
            "三陽開泰": bool(three),
            "四海遊龍": bool(four),
            "突破": bool(breakout),
            "突破帶量": bool(breakout_ok),
            "守10MA": bool(hold10),
            "連兩日破10MA": bool(break10_two_days),
            "縮量警訊": bool(shrink_warn),
            "量價型態": vol_style,
            "老王分數": int(s),
            "理由": " / ".join(reasons),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["老王分數", "成交金額(億)"], ascending=[False, False]).head(int(output_top_k)).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out

# -----------------------------
# Orchestrator
# -----------------------------
def run_all_features(
    token: str,
    stock_id: str,
    stock_info: pd.DataFrame,
    scan_mode: str,
    sector_pick_k: int,
    theme_top_k: int,
    theme_money_threshold_yi: float,
    bias_profile_mode: str,
    is_holding: bool,
    contrarian_flag: bool,
) -> dict:
    res: dict = {"stock_id": stock_id, "scan_mode": scan_mode, "ts": datetime.utcnow().isoformat()}

    # 1) Daily + patch
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
        for c in ["close", "open", "max", "min", "Trading_Volume", "Trading_money", "spread"]:
            if c in patched.columns:
                patched[c] = to_numeric_series(patched[c])

        patched["MA5"] = patched["close"].rolling(5).mean()
        patched["MA10"] = patched["close"].rolling(10).mean()
        patched["MA20"] = patched["close"].rolling(20).mean()
        patched["MA60"] = patched["close"].rolling(60).mean()

        # 乖離率（%）
        patched["BIAS20"] = (patched["close"] / patched["MA20"] - 1) * 100
        patched["BIAS60"] = (patched["close"] / patched["MA60"] - 1) * 100
        patched["BIAS20"] = patched["BIAS20"].replace([np.inf, -np.inf], np.nan)
        patched["BIAS60"] = patched["BIAS60"].replace([np.inf, -np.inf], np.nan)

        res["price_df"] = patched
        res["patch_meta"] = {"patched": patched_flag, "patch_date": patch_date}
    else:
        res["price_df"] = pd.DataFrame()
        res["patch_meta"] = {"patched": False, "patch_date": ""}

    # 4) Score + chip
    score = compute_score(res["price_df"])

    chip_start = (datetime.now(tz=TZ) - timedelta(days=180) if TZ else datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    inst_df = get_institutional_cached(token, stock_id, chip_start, None)
    margin_df = get_margin_cached(token, stock_id, chip_start, None)

    chip_score, chip_notes = compute_chip_score(inst_df, margin_df)
    score["chip"] = chip_score
    score["total"] = score["trend"] + score["momentum"] + score["volume"] + score["chip"]
    score["notes"].extend(chip_notes)

    chip_summary = compute_chip_summary(inst_df, margin_df)
    res["chip_summary"] = chip_summary
    # keep notes light in score to avoid flooding
    if (chip_summary or {}).get("notes"):
        score["notes"].extend((chip_summary["notes"][:1]))

    ai = get_ai_advice(stock_id, stock_info, res["price_df"], score, bias_profile_mode)
    res["ai_advice"] = ai
    if ai.get("bias_alert", {}).get("level") in ["danger", "warn"]:
        score["notes"].append(ai["bias_alert"]["headline"])
    res["score"] = score

    # 2 & 3) Market scan
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
            sector_flow = compute_sector_flow_from_daily(token, stock_info, dates[-1])
            vol_rank = compute_volume_ranking_from_daily(token, stock_info, dates)

    res["market_meta"] = meta
    res["sector_flow"] = sector_flow
    res["volume_rank"] = vol_rank

    # sector strength & leaders
    res["sector_strength"] = compute_sector_strength(vol_rank, sector_flow, top_n=10)
    res["sector_leaders"] = build_sector_leaders(vol_rank, sector_flow, top_sectors=10, k=sector_pick_k, only_up=True)

    # chip raw
    res["inst_raw"] = inst_df
    res["margin_raw"] = margin_df

    # 7) revenue raw
    rev_start = (datetime.now(tz=TZ) - timedelta(days=365 * 6) if TZ else datetime.now() - timedelta(days=365 * 6)).strftime("%Y-%m-%d")
    res["revenue_raw"] = get_month_revenue_cached(token, stock_id, rev_start)

    # 8) theme radar
    res["theme_radar"] = compute_theme_radar(vol_rank, stock_info, THEME_GROUPS, top_k=theme_top_k, money_threshold_yi=theme_money_threshold_yi)

    # 9) trade plan engine
    profile = ai.get("profile", "中小型飆股")
    vol_ratio = compute_volume_ratio_from_df(res["price_df"])
    struct = compute_structure(res["price_df"])
    pattern_info = classify_pattern(struct, vol_ratio, profile)
    vol_quality = classify_volume_price(res["price_df"], vol_ratio)
    plan = build_trade_plan(struct, pattern_info, vol_quality, ai.get("bias_alert", {}), score)

    # 老王策略：領導股判斷 + 形態/一條線/量價檢核（寫入持股判斷依據）
    leader = compute_leader_status(vol_rank, stock_id)
    oldwang = compute_oldwang_signals(stock_id, res["price_df"], profile)
    decision = compose_oldwang_decision(
        is_holding=is_holding,
        bias_level=(ai.get("bias_alert", {}) or {}).get("level", "na"),
        oldwang=oldwang,
        leader=leader,
        chip_summary=chip_summary,
        contrarian_flag=contrarian_flag,
    )

    # 把老王檢核寫入交易計畫文字（附加段落）
    add_lines = []
    add_lines.append("")
    add_lines.append("——")
    add_lines.append("老王策略檢核（持股判斷依據）")
    add_lines.append(f"- 建議動作：**{decision['action']}**")
    cs = chip_summary or {}
    def _fmt(x):
        return "-" if x is None or pd.isna(x) else f"{float(x):,.0f}"
    add_lines.append(f"- 籌碼摘要：{cs.get('signal','-')}｜法人5日 {_fmt(cs.get('net_5d_total'))}｜外資5日 {_fmt(cs.get('net_5d_foreign'))}｜投信5日 {_fmt(cs.get('net_5d_trust'))}｜融資5日 {_fmt(cs.get('margin_delta_5d'))}")
    for r in decision["reasons"][:6]:
        add_lines.append(f"- {r}")
    # 顯示族群領導股清單（前3）
    if leader.get("top_leaders"):
        add_lines.append("- 族群Top3領導股：")
        for it in leader["top_leaders"]:
            sid = it.get("stock_id", "")
            nm = it.get("stock_name", "")
            chg = it.get("change_rate", np.nan)
            add_lines.append(f"  - {sid} {nm}（{chg:.2f}%）")
    # 形態訊號
    if oldwang.get("notes"):
        add_lines.append("- 訊號：")
        for n in oldwang["notes"][:8]:
            add_lines.append(f"  - {n}")

    plan["summary"] = plan.get("summary", "") + "\n" + "\n".join(add_lines)
    plan["oldwang_action"] = decision["action"]
    plan["oldwang_reasons"] = decision["reasons"]
    

    res["trade_engine"] = {

        "leader": leader,
        "oldwang": oldwang,
        "decision": decision,

        "profile": profile,
        "vol_ratio": vol_ratio,
        "structure": struct,
        "pattern": pattern_info,
        "vol_quality": vol_quality,
        "plan": plan,
    }

    return res


# -----------------------------
# UI
# -----------------------------
token = st.secrets.get("FINMIND_TOKEN") or os.environ.get("FINMIND_API_TOKEN") or ""
if not token:
    st.error("未偵測到 FINMIND_TOKEN。請到 Streamlit Cloud → App → Settings → Secrets 設定 FINMIND_TOKEN。")
    st.stop()

st.sidebar.header("參數")
target_sid = st.sidebar.text_input("個股代碼", value="2330").strip()
scan_mode = st.sidebar.selectbox("全市場掃描來源", options=["即時快照（推薦）", "日線（收盤資料）"], index=0)
top_n = st.sidebar.slider("相對大量榜 Top N", min_value=20, max_value=200, value=80, step=10)

st.sidebar.divider()
st.sidebar.subheader("乖離率警報")
bias_profile_mode = st.sidebar.selectbox("股票屬性（影響乖離警戒線）", options=["自動判斷", "大型權值股", "中小型飆股"], index=0)

st.sidebar.checkbox("我已持有此股（持股模式）", value=False, key="is_holding")
st.sidebar.checkbox("反向警訊：外資/市場過度樂觀（手動）", value=False, key="contrarian_flag")


st.sidebar.divider()
st.sidebar.subheader("十大族群代表股")
sector_pick_k = st.sidebar.slider("每族群挑幾檔", 3, 12, 5, 1)

st.sidebar.divider()
st.sidebar.subheader("第八分類：主題族群資金雷達")
theme_top_k = st.sidebar.slider("每主題顯示個股數", 5, 30, 10, 1)
theme_money_threshold_yi = st.sidebar.number_input("個股成交金額門檻（億）", min_value=0.0, value=1.0, step=0.5)

if "result" not in st.session_state:
    st.session_state["result"] = None

with st.spinner("載入股票清單與產業分類..."):
    stock_info = get_stock_info_cached(token)

if st.sidebar.button("一鍵更新（含交易計畫引擎）"):
    with st.spinner("更新中：即時補丁 / 全市場掃描 / 籌碼 / 營收 / 主題雷達 / 交易計畫..."):
        st.session_state["result"] = run_all_features(
            token=token,
            stock_id=target_sid,
            stock_info=stock_info,
            scan_mode=scan_mode,
            sector_pick_k=sector_pick_k,
            theme_top_k=theme_top_k,
            theme_money_threshold_yi=theme_money_threshold_yi,
            bias_profile_mode=bias_profile_mode,
            is_holding=st.session_state.get("is_holding", False),
            contrarian_flag=st.session_state.get("contrarian_flag", False),
        )
    st.sidebar.success("更新完成")

if st.sidebar.button("清除結果"):
    st.session_state["result"] = None
    st.sidebar.info("已清除")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["1. 健診/技術圖", "2. 十大族群資金流向", "3. 全台股相對大量榜", "4. 籌碼照妖鏡", "5. 營收診斷", "8. 主題族群雷達", "9. 交易計畫引擎", "10. 老王選股器"]
)

res = st.session_state.get("result")

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.subheader("健診：老王策略 + 籌碼分析（5/10/20/60 + 三陽開泰/四海遊龍 + 法人/融資）")
    if res is None or res.get("stock_id") != target_sid:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」取得資料。")
    else:
        t = res.get("price_df", pd.DataFrame())
        meta = res.get("patch_meta", {})
        score = res.get("score", {})
        ai = res.get("ai_advice", {})
        chip = res.get("chip_summary", {})
        te = res.get("trade_engine", {})

        ow = te.get("oldwang", {}) if isinstance(te, dict) else {}
        decision = te.get("decision", {}) if isinstance(te, dict) else {}
        leader = te.get("leader", {}) if isinstance(te, dict) else {}

        if t is None or t.empty:
            st.error("日線資料為空。")
        else:
            patch_date = meta.get("patch_date", "")
            patched_flag = bool(meta.get("patched", False))
            holding_mode = bool(st.session_state.get("is_holding", False))

            # Ensure short MAs exist (for display/chart)
            t = t.copy()
            if "MA5" not in t.columns and "close" in t.columns:
                t["MA5"] = pd.to_numeric(t["close"], errors="coerce").rolling(5).mean()
            if "MA10" not in t.columns and "close" in t.columns:
                t["MA10"] = pd.to_numeric(t["close"], errors="coerce").rolling(10).mean()
            if "MA20" not in t.columns and "close" in t.columns:
                t["MA20"] = pd.to_numeric(t["close"], errors="coerce").rolling(20).mean()
            if "MA60" not in t.columns and "close" in t.columns:
                t["MA60"] = pd.to_numeric(t["close"], errors="coerce").rolling(60).mean()

            close = float(pd.to_numeric(t["close"].iloc[-1], errors="coerce"))
            ma5 = ow.get("ma5", np.nan)
            ma10 = ow.get("ma10", np.nan)
            ma20 = ow.get("ma20", np.nan)
            ma60 = ow.get("ma60", np.nan)

            # fallback to df values if needed
            if pd.isna(ma5) and pd.notna(t["MA5"].iloc[-1]): ma5 = float(t["MA5"].iloc[-1])
            if pd.isna(ma10) and pd.notna(t["MA10"].iloc[-1]): ma10 = float(t["MA10"].iloc[-1])
            if pd.isna(ma20) and pd.notna(t["MA20"].iloc[-1]): ma20 = float(t["MA20"].iloc[-1])
            if pd.isna(ma60) and pd.notna(t["MA60"].iloc[-1]): ma60 = float(t["MA60"].iloc[-1])

            bias20 = float(t["BIAS20"].iloc[-1]) if "BIAS20" in t.columns and pd.notna(t["BIAS20"].iloc[-1]) else np.nan
            bias60 = float(t["BIAS60"].iloc[-1]) if "BIAS60" in t.columns and pd.notna(t["BIAS60"].iloc[-1]) else np.nan
            vol_ratio = te.get("vol_ratio", None)

            # --- 結論（以老王決策 + 籌碼為主） ---
            st.markdown("### 健診結論")
            action = decision.get("action", "等待")
            headline = f"模式：{'持股' if holding_mode else '觀察/找進場'}｜建議動作：{action}"

            if ("出場" in action) or ("減碼" in action):
                st.error(headline)
            elif ("不加碼" in action) or ("觀察" in action) or ("等待" in action):
                st.warning(headline)
            else:
                st.success(headline)

            # 領導股提示
            if leader.get("is_leader", False):
                st.caption("買強不買弱：此股為族群Top3領導股（符合老王偏好）。")
            else:
                if leader.get("rank"):
                    st.caption(f"買強不買弱：此股族群排名 {leader.get('rank')}（非Top3），優先關注領導股。")

            # 建議依據（理由）
            rs = decision.get("reasons", []) if isinstance(decision, dict) else []
            if rs:
                st.markdown("**持股判斷依據（老王策略 + 籌碼）**")
                for r in rs[:10]:
                    st.write(f"- {r}")

            # --- 快速指標列（老王 + 籌碼重點） ---
            st.markdown("### 快速指標")
            n1, n2, n3, n4, n5, n6, n7, n8 = st.columns(8)
            n1.metric("最新價", f"{close:.2f}")
            n2.metric("資料日期", patch_date)
            n3.metric("MA10（守線）", "-" if pd.isna(ma10) else f"{ma10:.2f}")
            n4.metric("MA20", "-" if pd.isna(ma20) else f"{ma20:.2f}")
            n5.metric("MA60", "-" if pd.isna(ma60) else f"{ma60:.2f}")
            n6.metric("20MA乖離(%)", "-" if pd.isna(bias20) else f"{bias20:.2f}%")
            n7.metric("法人5日合計", "-" if chip.get("net_5d_total") is None else f"{float(chip.get('net_5d_total')):,.0f}")
            n8.metric("融資5日變化", "-" if chip.get("margin_delta_5d") is None else f"{float(chip.get('margin_delta_5d')):,.0f}")

            if vol_ratio is not None and not pd.isna(vol_ratio):
                st.caption(f"相對量（今/近5日均量）：{float(vol_ratio):.2f}")

            if patched_flag:
                st.success(f"已套用 Pro 即時快照補丁：新增 {patch_date} 盤中資料列。")
            else:
                st.caption("本次未新增補丁（日線已是最新日期或快照無資料）。")

            # --- 老王策略檢核（結構化） ---
            st.markdown("### 老王策略檢核（結構化）")
            sig_rows = [{
                "三陽開泰": bool(ow.get("tri")),
                "三陽開泰(強)": bool(ow.get("tri_strong")),
                "四海遊龍": bool(ow.get("four")),
                "守10MA": bool(ow.get("key_hold")),
                "連兩日破10MA": bool(ow.get("key_break_2d")),
                "突破需帶量": bool(ow.get("breakout_need_volume")),
                "大量後不縮量": ow.get("post_surge_volume_ok"),
                "長黑洗盤提示": bool(ow.get("washout_ignore")),
            }]
            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True)

            ma_rows = [{
                "收盤": close,
                "MA5": ma5, "MA10": ma10, "MA20": ma20, "MA60": ma60,
                "收盤>=MA5": ow.get("above_ma5"),
                "收盤>=MA10": ow.get("above_ma10"),
                "收盤>=MA20": ow.get("above_ma20"),
                "收盤>=MA60": ow.get("above_ma60"),
            }]
            st.dataframe(pd.DataFrame(ma_rows), use_container_width=True)

            # --- 籌碼分析（摘要 + 圖） ---
            st.markdown("### 籌碼分析（法人 + 融資）")
            csig = chip.get("signal", "中性")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("籌碼判讀", csig)
            c2.metric("外資5日", "-" if chip.get("net_5d_foreign") is None else f"{float(chip.get('net_5d_foreign')):,.0f}")
            c3.metric("投信5日", "-" if chip.get("net_5d_trust") is None else f"{float(chip.get('net_5d_trust')):,.0f}")
            c4.metric("自營5日", "-" if chip.get("net_5d_dealer") is None else f"{float(chip.get('net_5d_dealer')):,.0f}")
            c5.metric("法人5日合計", "-" if chip.get("net_5d_total") is None else f"{float(chip.get('net_5d_total')):,.0f}")
            c6.metric("融資餘額", "-" if chip.get("margin_last") is None else f"{float(chip.get('margin_last')):,.0f}")

            # 乖離警報 headline（簡化顯示）
            alert = (ai.get("bias_alert", {}) or {})
            if alert.get("level") in ["danger", "warn", "buy", "ok"]:
                st.caption(f"乖離率提醒：{alert.get('headline','')}")

            # Plot chip chart if available
            inst_daily = chip.get("inst_daily", pd.DataFrame())
            margin_daily = chip.get("margin_daily", pd.DataFrame())
            if isinstance(inst_daily, pd.DataFrame) and not inst_daily.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=inst_daily["date"], y=inst_daily.get("net_total", inst_daily.get("net", 0)), name="法人買賣超(合計)"), secondary_y=False)
                if isinstance(margin_daily, pd.DataFrame) and (not margin_daily.empty) and "margin_balance" in margin_daily.columns:
                    fig.add_trace(go.Scatter(x=margin_daily["date"], y=margin_daily["margin_balance"], mode="lines", name="融資餘額"), secondary_y=True)
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                fig.update_yaxes(title_text="法人買賣超", secondary_y=False)
                fig.update_yaxes(title_text="融資餘額", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("籌碼圖表資料不足（仍可到『籌碼照妖鏡』頁查看）。")

            # --- 技術圖（加上 5/10/20/60） ---
            st.markdown("### 技術圖（含 5/10/20/60）")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t["date"], y=t["close"], name="Close", mode="lines"))
            if "MA5" in t.columns:
                fig.add_trace(go.Scatter(x=t["date"], y=t["MA5"], name="MA5", mode="lines"))
            if "MA10" in t.columns:
                fig.add_trace(go.Scatter(x=t["date"], y=t["MA10"], name="MA10", mode="lines"))
            fig.add_trace(go.Scatter(x=t["date"], y=t["MA20"], name="MA20", mode="lines", line=dict(color="#F7FF00", width=2)))
            fig.add_trace(go.Scatter(x=t["date"], y=t["MA60"], name="MA60", mode="lines", line=dict(color="#FF2DA4", width=2)))
            fig.add_trace(go.Scatter(
                x=[t["date"].iloc[-1]],
                y=[t["close"].iloc[-1]],
                mode="markers",
                marker=dict(symbol="x", size=12, color="#FFD400", line=dict(width=2, color="#FFD400")),
                showlegend=False
            ))
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.subheader("十大族群資金流向 + 廣度/集中度 + 代表股")
    if res is None:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」。")
    else:
        meta = res.get("market_meta", {})
        sector_flow = res.get("sector_flow", pd.DataFrame())
        strength = res.get("sector_strength", pd.DataFrame())

        if sector_flow is None or sector_flow.empty:
            st.error("族群資金流向資料為空。")
        else:
            st.caption(f"資料來源：{meta.get('source','')}；掃描日期：{meta.get('scan_date','')}")
            show = sector_flow.head(10).copy()
            show["資金流向(億)"] = show["signed_money"] / 1e8
            show = show.rename(columns={"industry_category": "族群"})
            st.dataframe(show[["族群", "signed_money", "資金流向(億)"]], use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=show["族群"], y=show["資金流向(億)"], name="資金流向(億)"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 族群強弱（Breadth + Leadership）")
        if strength is None or strength.empty:
            st.info("本次無法計算族群廣度/集中度（資料不足）。")
        else:
            st.dataframe(strength, use_container_width=True)

        leaders = res.get("sector_leaders", {})
        if leaders:
            st.markdown(f"### 各族群：領漲放量 Top {sector_pick_k}")
            for sec, sdf in leaders.items():
                with st.expander(f"{sec}｜Top {len(sdf)}", expanded=False):
                    df_show = sdf.copy()
                    if "成交金額" in df_show.columns:
                        df_show["成交金額(億)"] = df_show["成交金額"] / 1e8
                    st.dataframe(df_show, use_container_width=True)

# -----------------------------
# Tab 3
# -----------------------------
with tab3:
    st.subheader("全台股相對大量榜（全市場量能增溫排行）")
    if res is None:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」。")
    else:
        meta = res.get("market_meta", {})
        vol_rank = res.get("volume_rank", pd.DataFrame())
        if vol_rank is None or vol_rank.empty:
            st.error("相對大量榜資料為空。")
        else:
            st.caption(f"資料來源：{meta.get('source','')}；掃描日期：{meta.get('scan_date','')}")
            cols = ["stock_id", "stock_name", "industry_category", "volume_ratio"]
            for extra in ["close", "change_rate", "total_amount", "Trading_money", "Trading_Volume", "spread", "scan_date"]:
                if extra in vol_rank.columns and extra not in cols:
                    cols.append(extra)
            top_df = vol_rank[cols].head(top_n).copy()
            top_df.insert(0, "rank", range(1, len(top_df) + 1))
            st.dataframe(top_df, use_container_width=True)

# -----------------------------
# Tab 4
# -----------------------------
with tab4:
    st.subheader("籌碼照妖鏡（法人買賣超 + 散戶融資對照圖）")
    if res is None:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」。")
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

# -----------------------------
# Tab 5
# -----------------------------
with tab5:
    st.subheader("營收診斷（月營收年增率圖表）")
    if res is None:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」。")
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

# -----------------------------
# Tab 6
# -----------------------------
with tab6:
    st.subheader("第八分類：主題族群資金雷達（含廣度/集中度）")
    if res is None:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」。")
    else:
        meta = res.get("market_meta", {})
        radar = res.get("theme_radar", {"summary": pd.DataFrame(), "leaders": {}})
        summary = radar.get("summary", pd.DataFrame())
        leaders = radar.get("leaders", {})

        st.caption(f"資料來源：{meta.get('source','')}；掃描日期：{meta.get('scan_date','')}")
        if summary is None or summary.empty:
            st.error("主題族群雷達資料為空（可能非交易時段或掃描失敗）。")
        else:
            st.dataframe(summary, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=summary["主題"], y=summary["資金偏多(億)"], name="資金偏多(億)"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"### 各主題：當日個股漲幅（Top {theme_top_k}；成交金額門檻 {theme_money_threshold_yi} 億）")
            for theme in THEME_GROUPS.keys():
                df_pick = leaders.get(theme, pd.DataFrame())
                with st.expander(f"{theme}", expanded=False):
                    if df_pick is None or df_pick.empty:
                        st.info("本主題本次未抓到資料（可能名單未命中或資料源未含該股）。")
                    else:
                        st.dataframe(df_pick, use_container_width=True)

# -----------------------------
# Tab 7
# -----------------------------
with tab7:
    st.subheader("交易計畫引擎（型態分類 + 量價品質 + 可執行計畫）")
    if res is None:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」。")
    else:
        te = res.get("trade_engine", {})
        plan = (te.get("plan") or {})
        if not te:
            st.error("交易計畫引擎資料不足。")
        else:
            vol_ratio = te.get("vol_ratio")
            st.caption(f"股票屬性：{te.get('profile','-')}；相對量（今/近5日均量）：{vol_ratio:.2f}" if vol_ratio is not None else f"股票屬性：{te.get('profile','-')}；相對量：N/A")

            # Plan text
            st.markdown(plan.get("summary", ""))

            st.markdown("### 老王策略檢核（結構化）")
            ow = te.get("oldwang", {})
            ld = te.get("leader", {})
            dc = te.get("decision", {})
            cA, cB = st.columns(2)
            cA.metric("建議動作", dc.get("action", "-"))
            cB.metric("是否族群領導股", "是" if ld.get("is_leader") else "否")

            if ld.get("industry"):
                st.caption(f"族群：{ld.get('industry')}；族群排名：{ld.get('rank') if ld.get('rank') else '-'}")
            if ld.get("top_leaders"):
                st.write("族群Top3領導股：")
                st.dataframe(pd.DataFrame(ld["top_leaders"]), use_container_width=True)

            # 形態/守線訊號
            sig_rows = [{
                "三陽開泰": ow.get("tri"),
                "三陽開泰(強)": ow.get("tri_strong"),
                "四海遊龍": ow.get("four"),
                "守住10MA": ow.get("key_hold"),
                "連兩日破10MA": ow.get("key_break_2d"),
                "突破需帶量": ow.get("breakout_need_volume"),
                "大量後不縮量": ow.get("post_surge_volume_ok"),
                "長黑洗盤提示": ow.get("washout_ignore"),
            }]
            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True)

            # 均線明細（避免只看20/60就誤以為站上所有均線）
            ma_rows = [{
                "收盤": te.get("structure", {}).get("close"),
                "MA5": ow.get("ma5"),
                "MA10": ow.get("ma10"),
                "MA20": ow.get("ma20"),
                "MA60": ow.get("ma60"),
                "收盤>=MA5": ow.get("above_ma5"),
                "收盤>=MA10": ow.get("above_ma10"),
                "收盤>=MA20": ow.get("above_ma20"),
                "收盤>=MA60": ow.get("above_ma60"),
            }]
            st.dataframe(pd.DataFrame(ma_rows), use_container_width=True)

            if dc.get("reasons"):
                st.write("依據：")
                for r in dc["reasons"]:
                    st.write(f"- {r}")


            # Key levels
            st.markdown("### 關鍵價位")
            sup = plan.get("supports", [])
            resis = plan.get("resistances", [])
            c1, c2 = st.columns(2)
            c1.write("支撐帶：" + ("、".join([f"{x:.2f}" for x in sup]) if sup else "-"))
            c2.write("壓力帶：" + ("、".join([f"{x:.2f}" for x in resis]) if resis else "-"))

            st.markdown("### 計畫數值（若有）")
            n1, n2, n3 = st.columns(3)
            stop = plan.get("stop")
            t1 = plan.get("t1")
            t2 = plan.get("t2")
            n1.metric("停損參考", "-" if stop is None else f"{stop:.2f}")
            n2.metric("目標1", "-" if t1 is None else f"{t1:.2f}")
            n3.metric("目標2", "-" if t2 is None else f"{t2:.2f}")

# -----------------------------
# Tab 8
# -----------------------------
with tab8:
    st.subheader("老王選股器（5/10/20/60 + 三陽開泰/四海遊龍 + 帶量突破 + 守10MA + 領導股）")

    if res is None:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」，取得全市場掃描資料後再跑選股器。")
    else:
        vol_rank = res.get("volume_rank", pd.DataFrame())
        sector_flow = res.get("sector_flow", pd.DataFrame())
        meta = res.get("market_meta", {})

        st.caption(f"資料來源：{meta.get('source','')}；掃描日期：{meta.get('scan_date','')}")

        c1, c2, c3, c4 = st.columns(4)
        universe_top_n = c1.number_input("候選池（依成交金額前 N）", min_value=50, max_value=800, value=300, step=50)
        output_top_k = c2.number_input("輸出 Top K", min_value=20, max_value=200, value=80, step=10)
        min_money_yi = c3.number_input("成交金額門檻（億）", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
        require_leader = c4.checkbox("只挑族群領導股（Top3）", value=True)

        c5, c6 = st.columns(2)
        require_pattern = c5.selectbox("型態過濾", ["不限", "三陽開泰", "四海遊龍"], index=0)
        require_breakout = c6.checkbox("只挑『突破前高且帶量』", value=False)

        run_btn = st.button("🚀 執行老王選股器", type="primary")

        if "oldwang_screener_df" not in st.session_state:
            st.session_state["oldwang_screener_df"] = pd.DataFrame()

        if run_btn:
            if vol_rank is None or vol_rank.empty:
                st.error("全市場資料為空，無法選股。請確認掃描來源是否可回傳資料。")
            else:
                with st.spinner("選股器運算中（抓取候選股近 60+ 日資料並計算訊號）..."):
                    df_pick = oldwang_screener(
                        token=token,
                        stock_info=stock_info,
                        vol_rank_today=vol_rank,
                        universe_top_n=int(universe_top_n),
                        output_top_k=int(output_top_k),
                        require_leader=bool(require_leader),
                        require_pattern=str(require_pattern),
                        require_breakout=bool(require_breakout),
                        min_money_yi=float(min_money_yi),
                    )
                st.session_state["oldwang_screener_df"] = df_pick

        df_pick = st.session_state.get("oldwang_screener_df", pd.DataFrame())
        if df_pick is None or df_pick.empty:
            st.info("尚未執行選股器，或本次條件下沒有符合的股票。")
        else:
            st.markdown("### 選股結果（老王分數越高越符合策略）")
            st.dataframe(df_pick, use_container_width=True)

            # quick summary counts
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("結果數量", len(df_pick))
            k2.metric("四海遊龍", int(df_pick["四海遊龍"].sum()) if "四海遊龍" in df_pick.columns else 0)
            k3.metric("突破帶量", int(df_pick["突破帶量"].sum()) if "突破帶量" in df_pick.columns else 0)
            k4.metric("縮量警訊", int(df_pick["縮量警訊"].sum()) if "縮量警訊" in df_pick.columns else 0)

            csv = df_pick.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下載 CSV", data=csv, file_name="oldwang_screener.csv", mime="text/csv")
