import os
from datetime import datetime, timedelta
from typing import Optional, Union, Iterable

import numpy as np
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import xml.etree.ElementTree as ET
import streamlit as st

from pathlib import Path
import io
import os

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
# Market framework: TAIEX vs OTC (divergence)
# -----------------------------
def _detect_market_col(info: pd.DataFrame) -> Optional[str]:
    # Choose the most plausible market/exchange column from TaiwanStockInfo.
    # Score columns by how well they map to BOTH TSE and OTC.
    candidates = ["market", "exchange", "market_type", "stock_market", "type", "stock_type"]
    cols = [c for c in candidates if c in info.columns]
    if not cols:
        return None

    best_col = None
    best_score = -1
    sample = info[cols].drop_duplicates().head(5000).copy()

    for c in cols:
        try:
            mapped = sample[c].apply(_normalize_market_value)
            ct_tse = int((mapped == "TSE").sum())
            ct_otc = int((mapped == "OTC").sum())
            score = min(ct_tse, ct_otc) * 100 + (ct_tse + ct_otc)
            if score > best_score:
                best_score = score
                best_col = c
        except Exception:
            continue

    if best_score <= 0:
        return None
    return best_col

    best_col = None
    best_score = -1
    sample = info[cols].drop_duplicates().head(5000).copy()

    for c in cols:
        try:
            mapped = sample[c].apply(_normalize_market_value)
            ct_tse = int((mapped == "TSE").sum())
            ct_otc = int((mapped == "OTC").sum())
            score = min(ct_tse, ct_otc) * 100 + (ct_tse + ct_otc)
            if score > best_score:
                best_score = score
                best_col = c
        except Exception:
            continue

    if best_score <= 0:
        return None
    return best_col


def _normalize_market_value(v: str) -> Optional[str]:
    if v is None:
        return None


# -----------------------------
# Local-first market sets: robust TSE/OTC classification (Streamlit Cloud friendly)
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def get_official_market_sets() -> dict:
    """Read local CSV lists from repo: data/market_tse.csv, data/market_otc.csv."""
    import pandas as pd
    import re as _re

    def _read(path: str) -> set:
        p = Path(path)
        if not p.exists():
            return set()
        df = pd.read_csv(p)
        if df is None or df.empty:
            return set()
        col = None
        for c in df.columns:
            if "stock" in str(c).lower() or "代號" in str(c) or "code" in str(c).lower():
                col = c
                break
        if col is None:
            col = df.columns[0]
        codes = set()
        for v in df[col].astype(str).tolist():
            m = _re.match(r"^\s*(\d{4})\s*$", v)
            if m:
                codes.add(m.group(1))
        return codes

    return {"TSE": _read("data/market_tse.csv"), "OTC": _read("data/market_otc.csv")}



    s = str(v).strip().lower()

    # English exchange codes
    if "tpex" in s or "otc" in s:
        return "OTC"
    if "twse" in s or "tse" in s:
        return "TSE"

    # Chinese labels
    if "上櫃" in s or "上柜" in s or "櫃買" in s or "柜买" in s:
        return "OTC"
    if "上市" in s:
        return "TSE"

    # Short / numeric codes seen in some datasets
    if s in {"2", "otc"}:
        return "OTC"
    if s in {"1", "tse"}:
        return "TSE"

    return None
    s = str(v).strip().lower()
    if "otc" in s or "上櫃" in s or "柜" in s:
        return "OTC"
    if "tse" in s or "twse" in s or "上市" in s:
        return "TSE"
    return None


@st.cache_data(ttl=3600)
def get_index_proxy_cached(token: str, proxy_id: str, start_date: str) -> pd.DataFrame:
    df = finmind_get_data(token, dataset="TaiwanStockPrice", data_id=proxy_id, start_date=start_date, timeout=40)
    df = normalize_date_col(df, "date")
    return df


def compute_index_state(token: str, proxy_id: str, lookback_days: int = 260) -> dict:
    start_date = (datetime.now(tz=TZ) - timedelta(days=lookback_days) if TZ else datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    df = get_index_proxy_cached(token, proxy_id, start_date)
    if df is None or df.empty or "close" not in df.columns:
        return {"proxy": proxy_id, "trend": "N/A", "close": np.nan, "ma20": np.nan, "ma60": np.nan, "ret20": np.nan, "ret60": np.nan, "last_date": ""}

    t = df.copy()
    t["close"] = pd.to_numeric(t["close"], errors="coerce")
    t = t.dropna(subset=["close"])
    if t.empty:
        return {"proxy": proxy_id, "trend": "N/A", "close": np.nan, "ma20": np.nan, "ma60": np.nan, "ret20": np.nan, "ret60": np.nan, "last_date": ""}

    t["MA20"] = t["close"].rolling(20).mean()
    t["MA60"] = t["close"].rolling(60).mean()
    last = t.iloc[-1]
    close = float(last["close"])
    ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan
    ma60 = float(last["MA60"]) if pd.notna(last["MA60"]) else np.nan
    last_date = str(last["date"]) if "date" in t.columns else ""

    trend = "neutral"
    if pd.notna(ma20) and pd.notna(ma60):
        if close > ma20 > ma60:
            trend = "bull"
        elif close < ma20 < ma60:
            trend = "bear"
        elif close > ma20:
            trend = "mild_bull"
        elif close < ma20:
            trend = "mild_bear"

    def _ret(n: int) -> float:
        if len(t) <= n:
            return np.nan
        p0 = t["close"].iloc[-(n+1)]
        if pd.isna(p0) or p0 == 0:
            return np.nan
        return float((close / float(p0) - 1.0) * 100.0)

    return {"proxy": proxy_id, "trend": trend, "close": close, "ma20": ma20, "ma60": ma60, "ret20": _ret(20), "ret60": _ret(60), "last_date": last_date}


def compute_breadth_by_market(vol_rank_all: pd.DataFrame, stock_info: pd.DataFrame) -> dict:
    if vol_rank_all is None or vol_rank_all.empty:
        return {"breadth_tse": np.nan, "breadth_otc": np.nan, "coverage_note": "無全市場資料"}
    if stock_info is None or stock_info.empty:
        return {"breadth_tse": np.nan, "breadth_otc": np.nan, "coverage_note": "無股票清單"}

    mcol = _detect_market_col(stock_info)
    if mcol is None:
        return {"breadth_tse": np.nan, "breadth_otc": np.nan, "coverage_note": "TaiwanStockInfo 無市場欄位（僅顯示整體 breadth）"}

    info = stock_info[["stock_id", mcol]].drop_duplicates().copy()
    info["stock_id"] = info["stock_id"].astype(str)
    info["_m"] = info[mcol].apply(_normalize_market_value)

    df = vol_rank_all.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df = ensure_change_rate(df)
    df = df.merge(info[["stock_id", "_m"]], on="stock_id", how="left")

    tse = df[df["_m"] == "TSE"]
    otc = df[df["_m"] == "OTC"]

    breadth_tse = float((tse["change_rate"] > 0).mean() * 100) if len(tse) else np.nan
    breadth_otc = float((otc["change_rate"] > 0).mean() * 100) if len(otc) else np.nan
    note = f"覆蓋：TSE {len(tse)} 檔 / OTC {len(otc)} 檔"
    return {"breadth_tse": breadth_tse, "breadth_otc": breadth_otc, "coverage_note": note}


def build_divergence_hint(tw: dict, oc: dict, breadth: dict) -> dict:
    tw_tr = tw.get("trend", "N/A")
    oc_tr = oc.get("trend", "N/A")
    bt = breadth.get("breadth_tse", np.nan)
    bo = breadth.get("breadth_otc", np.nan)

    if tw_tr == "N/A" or oc_tr == "N/A":
        return {"level": "info", "title": "分歧提示：資料不足", "detail": "無法完整判讀加權 vs 櫃買趨勢。"}

    if tw_tr in ["bull", "mild_bull"] and oc_tr in ["bear", "mild_bear"]:
        return {"level": "warning", "title": "大小盤分歧：加權偏強、櫃買偏弱", "detail": "常見於權值拉盤、中小退潮。策略偏向主流/權值/族群領導股，避免弱勢補漲。"}
    if tw_tr in ["bear", "mild_bear"] and oc_tr in ["bull", "mild_bull"]:
        return {"level": "warning", "title": "大小盤分歧：加權偏弱、櫃買偏強", "detail": "資金可能轉向中小題材。策略偏向強勢中小領導，避免死守大型權值。"}

    if pd.notna(bt) and pd.notna(bo) and abs(bt - bo) >= 15:
        if bt > bo:
            return {"level": "info", "title": "廣度分歧：上市較熱、上櫃較冷", "detail": "上漲家數比例差異明顯，盤面可能偏權值或大型股。"}
        else:
            return {"level": "info", "title": "廣度分歧：上櫃較熱、上市較冷", "detail": "中小活躍度較高，留意題材輪動。"}

    return {"level": "success", "title": "大小盤同步", "detail": "加權與櫃買趨勢方向一致，盤勢一致性較佳。"}


# -----------------------------
# TDCC 股權分散：散戶(1-10張) vs 大戶(>=100/500/1000張)（每週）
# -----------------------------
LOT = 1000
RETAIL_LOW = 1 * LOT
RETAIL_HIGH = 10 * LOT
BIG_100 = 100 * LOT
BIG_500 = 500 * LOT
BIG_1000 = 1000 * LOT

def _parse_level_range(level: str):
    """
    Parse HoldingSharesLevel to (lower, upper) in shares.
    Examples:
      '1-999'
      '1,000-5,000'
      '1,000,001-999,999,999'
      '1,000,001以上'
    """
    if level is None:
        return (np.nan, np.nan)
    s = str(level).strip()

    # '以上'
    if "以上" in s:
        m = re.search(r"[\d,]+", s)
        if not m:
            return (np.nan, np.nan)
        lo = float(m.group(0).replace(",", ""))
        return (lo, float("inf"))

    nums = re.findall(r"[\d,]+", s)
    if len(nums) >= 2:
        lo = float(nums[0].replace(",", ""))
        hi = float(nums[1].replace(",", ""))
        return (lo, hi)
    if len(nums) == 1:
        lo = float(nums[0].replace(",", ""))
        return (lo, lo)
    return (np.nan, np.nan)

@st.cache_data(ttl=24 * 3600)
def get_holding_shares_per_cached(token: str, stock_id: str, start_date: str) -> pd.DataFrame:
    df = finmind_get_data(
        token,
        dataset="TaiwanStockHoldingSharesPer",
        data_id=stock_id,
        start_date=start_date,
        timeout=40,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df

def build_retail_big_weekly(token: str, stock_id: str, start_date: str = "2019-01-01") -> pd.DataFrame:
    """
    Weekly holding structure:
      retail: 1-10 lots (1,000-10,000 shares)
      big: >=100/500/1000 lots (>=100k/500k/1M shares)
    Returns percent/people and WoW diffs.
    """
    df = get_holding_shares_per_cached(token, stock_id, start_date)
    if df is None or df.empty or "HoldingSharesLevel" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d[["lo", "hi"]] = d["HoldingSharesLevel"].apply(lambda x: pd.Series(_parse_level_range(x)))
    for c in ["people", "percent", "shares"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    retail = d[(d["lo"] >= RETAIL_LOW) & (d["hi"] <= RETAIL_HIGH)].copy()
    big100 = d[d["lo"] >= BIG_100].copy()
    big500 = d[d["lo"] >= BIG_500].copy()
    big1000 = d[d["lo"] >= BIG_1000].copy()

    def _agg(x: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if x.empty:
            return pd.DataFrame(columns=["date", f"{prefix}_people", f"{prefix}_percent"])
        g = x.groupby("date", as_index=False).agg(
            people=("people", "sum"),
            percent=("percent", "sum"),
        )
        return g.rename(columns={"people": f"{prefix}_people", "percent": f"{prefix}_percent"})

    out = _agg(retail, "retail_1_10")
    for pref, chunk in [("big_100", big100), ("big_500", big500), ("big_1000", big1000)]:
        out = out.merge(_agg(chunk, pref), on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)

    # WoW changes
    for pref in ["retail_1_10", "big_100", "big_500", "big_1000"]:
        if f"{pref}_people" in out.columns:
            out[f"{pref}_people_wow"] = out[f"{pref}_people"].diff()
        if f"{pref}_percent" in out.columns:
            out[f"{pref}_percent_wow"] = out[f"{pref}_percent"].diff()
    return out

def _streak(series: pd.Series, direction: str = "up") -> int:
    """
    Count consecutive weeks at the end where diff is >0 (up) or <0 (down).
    """
    if series is None or series.empty:
        return 0
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return 0
    d = s.diff().dropna()
    cnt = 0
    for v in reversed(d.tolist()):
        if direction == "up" and v > 0:
            cnt += 1
        elif direction == "down" and v < 0:
            cnt += 1
        else:
            break
    return cnt

def compute_holding_signal(w: pd.DataFrame) -> dict:
    """
    Return a compact signal dict for health/decision:
      - latest percents
      - WoW deltas
      - streaks (big up / retail down)
      - light: GREEN/YELLOW/RED
    """
    if w is None or w.empty:
        return {"ok": False, "light": "N/A", "msg": "無股權分散資料"}
    last = w.iloc[-1]
    def _g(name, default=np.nan):
        return float(last[name]) if name in w.columns and pd.notna(last[name]) else default

    big1000_wow = _g("big_1000_percent_wow", 0.0)
    big500_wow = _g("big_500_percent_wow", 0.0)
    retail_wow = _g("retail_1_10_percent_wow", 0.0)

    big1000_streak = _streak(w.get("big_1000_percent", pd.Series(dtype=float)), "up")
    big500_streak = _streak(w.get("big_500_percent", pd.Series(dtype=float)), "up")
    retail_down_streak = _streak(w.get("retail_1_10_percent", pd.Series(dtype=float)), "down")

    # Light rules (simple & interpretable)
    green = ((big500_wow > 0) or (big1000_wow > 0)) and (retail_wow < 0)
    red = ((big500_wow < 0) and (big1000_wow < 0)) and (retail_wow > 0)

    if green:
        light = "GREEN"
        msg = "大戶比例上升、散戶比例下降（結構偏健康）"
    elif red:
        light = "RED"
        msg = "大戶比例下降、散戶比例上升（結構偏弱/分散）"
    else:
        light = "YELLOW"
        msg = "結構中性（需搭配法人/融資與型態）"

    return {
        "ok": True,
        "light": light,
        "msg": msg,
        "date": str(last.get("date", "")),
        "retail_pct": _g("retail_1_10_percent"),
        "retail_wow": retail_wow,
        "big100_pct": _g("big_100_percent"),
        "big100_wow": _g("big_100_percent_wow", 0.0),
        "big500_pct": _g("big_500_percent"),
        "big500_wow": big500_wow,
        "big1000_pct": _g("big_1000_percent"),
        "big1000_wow": big1000_wow,
        "big500_up_weeks": big500_streak,
        "big1000_up_weeks": big1000_streak,
        "retail_down_weeks": retail_down_streak,
    }


# -----------------------------
# Macro tab helpers
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_forexfactory_calendar_thisweek() -> pd.DataFrame:
    """
    Economic calendar (weekly) from ForexFactory public feed (XML).
    Source: https://nfs.faireconomy.media/ff_calendar_thisweek.xml citeturn0search5
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        xml_text = r.text
    except Exception:
        return pd.DataFrame()

    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return pd.DataFrame()

    rows = []
    for ev in root.findall(".//event"):
        row = {}
        for child in list(ev):
            tag = child.tag
            val = child.text.strip() if child.text else ""
            row[tag] = val
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


def _ff_to_datetime_taipei(date_str: str, time_str: str) -> str:
    """
    Convert FF date+time to Asia/Taipei string.
    FF feed does not explicitly specify timezone; we assume US/Eastern for conversion.
    If time is 'All Day' or 'Tentative', return date only.
    """
    if not date_str:
        return ""
    date_str = date_str.strip()
    time_str = (time_str or "").strip()
    if time_str.lower() in ["all day", "tentative", ""]:
        return date_str
    # Try parse
    try:
        # FF often uses 'YYYY.MM.DD' or 'YYYY-MM-DD' or 'MMM DD, YYYY' depending on feed; handle a few
        dt = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            return f"{date_str} {time_str}".strip()
        # Assume US/Eastern
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            tw = ZoneInfo("Asia/Taipei")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=et)
            dt_tw = dt.astimezone(tw)
            return dt_tw.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return f"{date_str} {time_str}".strip()


@st.cache_data(ttl=3600)
def get_us_etf_close_from_stooq(symbol_us: str) -> dict:
    """
    Fetch last close and 1D change from Stooq CSV.
    Example: soxx.us, qqq.us citeturn0search3
    """
    url = f"https://stooq.com/q/d/l/?s={symbol_us}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return {"ok": False}
    if df is None or df.empty or "Close" not in df.columns:
        return {"ok": False}
    df = df.dropna(subset=["Close"]).copy()
    if len(df) < 2:
        last = float(df["Close"].iloc[-1])
        return {"ok": True, "date": str(df["Date"].iloc[-1]), "close": last, "chg": np.nan, "chg_pct": np.nan}
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2])
    chg = last - prev
    chg_pct = (last / prev - 1.0) * 100.0 if prev != 0 else np.nan
    return {"ok": True, "date": str(df["Date"].iloc[-1]), "close": last, "chg": chg, "chg_pct": chg_pct}


@st.cache_data(ttl=24*3600)
def get_moneydj_etf_holdings_codes(etf_code: str) -> list[str]:
    """
    Fetch Taiwan ETF holdings from MoneyDJ holdings page (HTML table).
    Works for: 0050 / 006208 / 00878 / 0056 (and more).
    Example pages:
      https://www.moneydj.com/etf/x/basic/basic0007.xdjhtm?etfid=0050.tw citeturn7search9
      https://www.moneydj.com/etf/x/basic/basic0007.xdjhtm?etfid=006208.tw citeturn7search10
      https://www.moneydj.com/etf/x/basic/basic0007.xdjhtm?etfid=00878.tw citeturn5search9
      https://www.moneydj.com/etf/x/basic/basic0007.xdjhtm?etfid=0056.tw citeturn5search0
    """
    url = f"https://www.moneydj.com/etf/x/basic/basic0007.xdjhtm?etfid={etf_code}.tw"
    try:
        tables = pd.read_html(url)
    except Exception:
        return []
    if not tables:
        return []
    # find table containing '個股名稱' and '投資比例'
    target = None
    for tb in tables:
        cols = [str(c) for c in tb.columns]
        if any("個股名稱" in c for c in cols) and any("投資比例" in c for c in cols):
            target = tb
            break
    if target is None:
        # fallback: first table with at least one column containing (.TW)
        for tb in tables:
            s = tb.to_string()
            if ".TW" in s:
                target = tb
                break
    if target is None:
        return []
    # extract 4-digit codes from strings like '台積電(2330.TW)'
    codes = set()
    for col in target.columns:
        series = target[col].astype(str)
        for v in series.tolist():
            m = re.findall(r"\((\d{4})\.TW\)", v)
            for c in m:
                codes.add(c)
    # sometimes code may be in separate column
    for col in target.columns:
        series = target[col].astype(str)
        for v in series.tolist():
            if re.fullmatch(r"\d{4}", v.strip()):
                codes.add(v.strip())
    return sorted(list(codes))


def compute_tw_etf_breadth(vol_rank_all: pd.DataFrame, etf_code: str) -> dict:
    """
    Breadth of ETF constituents using today's market change_rate.
    """
    codes = get_moneydj_etf_holdings_codes(etf_code)
    if not codes:
        return {"ok": False, "etf": etf_code, "msg": "無法取得成分股清單"}
    if vol_rank_all is None or vol_rank_all.empty:
        return {"ok": False, "etf": etf_code, "msg": "無全市場日線/快照資料"}

    df = vol_rank_all.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df = ensure_change_rate(df)

    sub = df[df["stock_id"].isin(codes)].copy()
    if sub.empty:
        return {"ok": False, "etf": etf_code, "msg": "成分股與全市場資料無法對齊"}

    up = int((sub["change_rate"] > 0).sum())
    dn = int((sub["change_rate"] < 0).sum())
    eq = int((sub["change_rate"] == 0).sum())
    total = int(len(sub))
    up_ratio = (up / total * 100.0) if total else np.nan
    return {"ok": True, "etf": etf_code, "total": total, "up": up, "dn": dn, "eq": eq, "up_ratio": up_ratio}

# -----------------------------
# Cached downloads
# -----------------------------
@st.cache_data(ttl=24 * 3600)

def compute_etf_breadth_regime(vol_rank_all: pd.DataFrame) -> dict:
    """
    Build a compact breadth regime from Taiwan ETF constituent breadth.
    Uses: 0050, 006208 (large-cap) and 00878, 0056 (defensive).
    Returns:
      {
        "rows": DataFrame (ETF, up_ratio, up, dn, eq, total),
        "large_avg": float,
        "def_avg": float,
        "state": "STRONG"/"WEAK"/"DEFENSIVE_ROTATION"/"NEUTRAL",
        "msg": str
      }
    """
    etfs = ["0050", "006208", "00878", "0056"]
    rows = []
    for etf in etfs:
        b = compute_tw_etf_breadth(vol_rank_all, etf)
        if b.get("ok"):
            rows.append({
                "ETF": etf,
                "成分股數": b["total"],
                "上漲": b["up"],
                "下跌": b["dn"],
                "平盤": b["eq"],
                "上漲比例(%)": float(b["up_ratio"]) if pd.notna(b["up_ratio"]) else np.nan,
            })
        else:
            rows.append({"ETF": etf, "成分股數": np.nan, "上漲": np.nan, "下跌": np.nan, "平盤": np.nan, "上漲比例(%)": np.nan})

    dfb = pd.DataFrame(rows)
    def _avg(codes):
        s = dfb[dfb["ETF"].isin(codes)]["上漲比例(%)"]
        s = pd.to_numeric(s, errors="coerce")
        return float(s.mean()) if s.notna().any() else np.nan

    large_avg = _avg(["0050", "006208"])
    def_avg = _avg(["00878", "0056"])

    state = "NEUTRAL"
    msg = "市場寬度中性：以個股與族群主流判斷為主。"

    if pd.notna(large_avg) and large_avg <= 45:
        state = "WEAK"
        msg = "市場寬度偏弱：盤面可能『權值撐盤或退潮』，策略宜保守、避免追突破。"
    elif pd.notna(large_avg) and pd.notna(def_avg) and (large_avg >= 60 and def_avg >= 60):
        state = "STRONG"
        msg = "市場寬度強：多數成分股齊漲，趨勢延續機率較高，可較積極。"
    elif pd.notna(large_avg) and pd.notna(def_avg) and (def_avg - large_avg >= 15):
        state = "DEFENSIVE_ROTATION"
        msg = "防禦寬度較強：高股息/防禦股較活躍，風險偏好下降，突破追價需更謹慎。"

    return {"rows": dfb, "large_avg": large_avg, "def_avg": def_avg, "state": state, "msg": msg}



# -----------------------------
# Breadth dataset (CSV) helpers for Streamlit Cloud (reduce load for multi-user)
# -----------------------------
BREADTH_CSV_PATH = Path("data") / "breadth_2y.csv"

def load_breadth_csv() -> pd.DataFrame:
    if not BREADTH_CSV_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(BREADTH_CSV_PATH)
        # expected columns: date, adv, dec, eq, ratio, proxy_close, ret_1d, fwd_1d, fwd_5d, fwd_20d, idx_up_breadth_bad, idx_dn_breadth_good
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        for c in ["ratio", "proxy_close", "ret_1d", "fwd_1d", "fwd_5d", "fwd_20d"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["idx_up_breadth_bad", "idx_dn_breadth_good"]:
            if c in df.columns:
                df[c] = df[c].astype(bool)
        return df.dropna(subset=["ratio", "proxy_close"]).sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def save_breadth_csv(df: pd.DataFrame) -> None:
    BREADTH_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BREADTH_CSV_PATH, index=False)

def breadth_csv_meta() -> str:
    if not BREADTH_CSV_PATH.exists():
        return "尚未生成 breadth_2y.csv"
    try:
        ts = BREADTH_CSV_PATH.stat().st_mtime
        return f"breadth_2y.csv 已存在（更新時間：{datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}）"
    except Exception:
        return "breadth_2y.csv 已存在"

# -----------------------------
# Breadth backtest (2 years): Full-market breadth vs index proxy (0050)
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def get_trading_dates_range_cached(token: str, start_date: str, end_date: str) -> list[str]:
    df = finmind_get_data(token, dataset="TaiwanStockTradingDate", timeout=40)
    if df is None or df.empty or "date" not in df.columns:
        return []
    d = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    out = [x for x in d.tolist() if (x >= start_date and x <= end_date)]
    return out

@st.cache_data(ttl=24 * 3600)
def get_breadth_for_date_cached(token: str, date_str: str) -> dict:
    df = finmind_get_data(token, dataset="TaiwanStockPrice", start_date=date_str, timeout=60)
    if df is None or df.empty:
        return {"date": date_str, "adv": np.nan, "dec": np.nan, "eq": np.nan, "ratio": np.nan}
    df = df.copy()
    df = ensure_change_rate(df)
    adv = int((df["change_rate"] > 0).sum())
    dec = int((df["change_rate"] < 0).sum())
    eq = int((df["change_rate"] == 0).sum())
    denom = adv + dec
    ratio = (adv / denom) if denom > 0 else np.nan
    return {"date": date_str, "adv": adv, "dec": dec, "eq": eq, "ratio": ratio}

@st.cache_data(ttl=24 * 3600)
def build_breadth_series_2y(token: str, proxy_id: str = "0050", years: int = 2) -> pd.DataFrame:
    # Date range
    end_date = _now_date_str()
    start_date = (datetime.now(tz=TZ) - timedelta(days=365 * years + 30) if TZ else datetime.now() - timedelta(days=365 * years + 30)).strftime("%Y-%m-%d")

    dates = get_trading_dates_range_cached(token, start_date, end_date)
    if not dates:
        return pd.DataFrame()

    # Keep last ~2 years trading days
    # (trading dates list may include extra buffer)
    dates = dates[-(years * 260 + 30):]

    # Breadth per day
    rows = []
    for d in dates:
        rows.append(get_breadth_for_date_cached(token, d))
    b = pd.DataFrame(rows).dropna(subset=["ratio"])
    if b.empty:
        return pd.DataFrame()

    # Index proxy series
    px = finmind_get_data(token, dataset="TaiwanStockPrice", data_id=proxy_id, start_date=dates[0], timeout=40)
    if px is None or px.empty or "date" not in px.columns or "close" not in px.columns:
        return pd.DataFrame()
    px = normalize_date_col(px, "date")
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.dropna(subset=["close"])

    df = b.merge(px[["date", "close"]].rename(columns={"close": "proxy_close"}), on="date", how="inner").sort_values("date")
    if df.empty:
        return pd.DataFrame()

    # Returns
    df["ret_1d"] = df["proxy_close"].pct_change(1) * 100.0
    df["fwd_1d"] = df["proxy_close"].pct_change(-1) * -100.0
    df["fwd_5d"] = df["proxy_close"].pct_change(-5) * -100.0
    df["fwd_20d"] = df["proxy_close"].pct_change(-20) * -100.0

    # Divergence flags
    df["idx_up_breadth_bad"] = (df["ret_1d"] > 0) & (df["ratio"] < 0.50)
    df["idx_dn_breadth_good"] = (df["ret_1d"] < 0) & (df["ratio"] > 0.55)

    return df.reset_index(drop=True)

def backtest_breadth_vs_index(df: pd.DataFrame, horizon: int = 5) -> dict:
    if df is None or df.empty:
        return {"ok": False, "msg": "無資料"}

    fwd_col = {1: "fwd_1d", 5: "fwd_5d", 20: "fwd_20d"}.get(int(horizon), "fwd_5d")
    d = df.copy()
    d = d.dropna(subset=["ratio", fwd_col])

    if d.empty:
        return {"ok": False, "msg": "資料不足（無法計算）"}

    # Correlation
    corr = float(d["ratio"].corr(d[fwd_col])) if d["ratio"].notna().any() else np.nan

    # Quintiles
    d["q"] = pd.qcut(d["ratio"], 5, labels=False, duplicates="drop")
    qtbl = d.groupby("q")[fwd_col].agg(["mean", "count"]).reset_index()
    qtbl["win_rate"] = d.groupby("q")[fwd_col].apply(lambda s: float((s > 0).mean()) if len(s) else np.nan).values

    # Divergence stats
    def _evt(mask):
        x = d[mask]
        if x.empty:
            return {"n": 0, "avg": np.nan, "win": np.nan}
        return {"n": int(len(x)), "avg": float(x[fwd_col].mean()), "win": float((x[fwd_col] > 0).mean())}

    evt1 = _evt(d["idx_up_breadth_bad"])
    evt2 = _evt(d["idx_dn_breadth_good"])

    return {"ok": True, "fwd_col": fwd_col, "corr": corr, "qtbl": qtbl, "evt_up_bad": evt1, "evt_dn_good": evt2}

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
        "ma20_slope": np.nan, "ma20_turn_up": None, "hold_ma10_2d": None,
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
        # 強勢版額外條件：
        # 1) MA20 走平翻揚（以近5日 MA20 斜率判斷）
        if len(df) >= 25 and pd.notna(df["MA20"].iloc[-1]) and pd.notna(df["MA20"].iloc[-6]):
            out["ma20_slope"] = float(df["MA20"].iloc[-1] - df["MA20"].iloc[-6])
            out["ma20_turn_up"] = bool(out["ma20_slope"] >= 0)
        # 2) 10MA 不破守線（連兩日收盤都在 10MA 之上）
        if len(df) >= 2 and pd.notna(df["MA10"].iloc[-1]) and pd.notna(df["MA10"].iloc[-2]):
            c1_ = float(df["close"].iloc[-1])
            c2_ = float(df["close"].iloc[-2])
            m1_ = float(df["MA10"].iloc[-1])
            m2_ = float(df["MA10"].iloc[-2])
            out["hold_ma10_2d"] = bool((c1_ >= m1_) and (c2_ >= m2_))
        # 三陽開泰（強）：站上5/10/20 + 5>10>20，且（MA20翻揚 或 10MA連續守住）
        out["tri_strong"] = bool(out["tri"] and (ma5 > ma10 > ma20) and ((out["ma20_turn_up"] is True) or (out["hold_ma10_2d"] is True)))

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
        out["notes"].append("形態：三陽開泰（強勢版：站上5/10/20 + 5>10>20，且MA20翻揚/10MA連續守住）")
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
    startup_mode: str = "自訂",  # 自訂 / 起漲-拉回承接 / 起漲-突破發動 / 趨勢-四海遊龍續漲
    require_pattern: str = "不限",
    require_breakout: bool = False,
    min_money_yi: float = 1.0,
    rs_proxy_id: str = "0050",
    rs_window: int = 20,
    rs_bonus_weight: int = 6,
    market_filter: str = "ALL",  # ALL / TSE / OTC
    strict_market: bool = True,
    new_complete_filter: str = "不限",  # 不限 / 今日新三陽 / 今日新三陽(強) / 今日新四海
    new_complete_days: int = 1,
    universe_all: bool = False,
    exclude_etf_index: bool = True,
    require_any_pattern: bool = False,
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
    # Defensive: keep mkt_close defined for compatibility with older screener variants
    mkt_close = pd.DataFrame()

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
    if universe_all:
        df0 = df0.copy()
    else:
        df0 = df0.sort_values(money_col, ascending=False).head(int(universe_top_n)).copy()

        # --- 市場篩選（上市/上櫃）---


    # 以本機 data/market_tse.csv、data/market_otc.csv 為準（Streamlit Cloud 不依賴外網抓名單）


    if market_filter in ["TSE", "OTC"]:


        ms = get_official_market_sets()


        target_set = ms.get(market_filter, set())


        if target_set:


            df0["stock_id"] = df0["stock_id"].astype(str)


            df0 = df0[df0["stock_id"].isin(target_set)].copy()


    candidate_ids = df0["stock_id"].drop_duplicates().tolist()
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


    # --- RS bonus (加分項)：相對強度 vs 大盤代理 ---
    rs_out_map = {}
    rs_rank_map = {}
    try:
        # 拉大盤代理的日線（預設 0050），用來計算 RS 超越大盤
        start_proxy = (datetime.now(tz=TZ) - timedelta(days=260) if TZ else datetime.now() - timedelta(days=260)).strftime("%Y-%m-%d")
        proxy_df = get_index_proxy_cached(token, rs_proxy_id, start_proxy)
        if proxy_df is not None and (not proxy_df.empty) and "date" in proxy_df.columns and "close" in proxy_df.columns:
            proxy_df = proxy_df.copy()
            proxy_df["date"] = pd.to_datetime(proxy_df["date"]).dt.strftime("%Y-%m-%d")
            proxy_df["proxy_close"] = pd.to_numeric(proxy_df["close"], errors="coerce")
            proxy_df = proxy_df[["date", "proxy_close"]].dropna()

            def _rs_outperformance(g: pd.DataFrame) -> float:
                gg = g[["date", "close"]].copy()
                gg["date"] = pd.to_datetime(gg["date"]).dt.strftime("%Y-%m-%d")
                gg["close"] = pd.to_numeric(gg["close"], errors="coerce")
                gg = gg.dropna()
                mm = gg.merge(proxy_df, on="date", how="inner").sort_values("date")
                if len(mm) <= rs_window:
                    return np.nan
                last_s = float(mm["close"].iloc[-1])
                prev_s = float(mm["close"].iloc[-(rs_window + 1)])
                last_p = float(mm["proxy_close"].iloc[-1])
                prev_p = float(mm["proxy_close"].iloc[-(rs_window + 1)])
                if prev_s == 0 or prev_p == 0:
                    return np.nan
                ret_s = (last_s / prev_s - 1.0) * 100.0
                ret_p = (last_p / prev_p - 1.0) * 100.0
                return float(ret_s - ret_p)

            for sid in candidate_ids:
                g0 = hist[hist["stock_id"] == sid].sort_values("date")
                if g0 is None or g0.empty or "close" not in g0.columns:
                    rs_out_map[sid] = np.nan
                    continue
                rs_out_map[sid] = _rs_outperformance(g0)

            ser = pd.Series(rs_out_map, dtype="float64")
            if ser.notna().any():
                rs_rank = ser.rank(pct=True)
                rs_rank_map = rs_rank.to_dict()
    except Exception:
        # RS 不影響主流程，失敗就略過
        rs_out_map = {}
        rs_rank_map = {}


    # --- 主題標記（散熱/記憶體/CPO...）---
    # 以 THEME_GROUPS 的 stocks 名單做標記；同一檔可能屬於多個主題
    theme_tag_map = {}
    try:
        for _theme, _rule in THEME_GROUPS.items():
            for _sid in (_rule.get("stocks", []) or []):
                _sid = str(_sid)
                theme_tag_map.setdefault(_sid, set()).add(_theme)
    except Exception:
        theme_tag_map = {}

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
        three = pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20) and close >= ma5 and close >= ma10 and close >= ma20
        four = three and pd.notna(ma60) and close >= ma60

        # 至少要有三陽/四海（可選過濾）
        pass_filter_anypattern = True
        if require_any_pattern:
            pass_filter_anypattern = bool(three or four)


        # 三陽開泰（強）：站上 MA5/10/20 + 均線多頭排列 +（MA20翻揚 或 10MA連兩日守住）
        ma20_slope = np.nan
        try:
            if len(g) >= 25:
                ma20_slope = float(g["close"].rolling(20).mean().iloc[-1] - g["close"].rolling(20).mean().iloc[-6])
        except Exception:
            ma20_slope = np.nan
        hold10_two = False
        try:
            if pd.notna(ma10) and len(g) >= 2:
                hold10_two = bool(g["close"].iloc[-1] >= ma10 and g["close"].iloc[-2] >= ma10)
        except Exception:
            hold10_two = False
        three_strong = bool(three and pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20) and (ma5 > ma10 > ma20) and (pd.isna(ma20_slope) or ma20_slope >= 0 or hold10_two))
        # 今日新成立（多日視窗）：最近 N 日內出現「由 False -> True」的成立事件（抓起漲股）
        new_three = False
        new_three_strong = False
        new_four = False
        confirm_three_2d = False
        confirm_three_strong_2d = False
        confirm_four_2d = False
        ma20_up = False
        kdr20_good = False
        kdr20 = np.nan
        ma20_up = False
        kdr20_good = False
        kdr20 = np.nan
        try:
            # 計算 MA20 斜率（近5日）與扣抵（20交易日前收盤）
            if len(g) >= 25:
                ma20_series = g["close"].rolling(20).mean()
                ma20_slope = float(ma20_series.iloc[-1] - ma20_series.iloc[-6]) if pd.notna(ma20_series.iloc[-1]) and pd.notna(ma20_series.iloc[-6]) else np.nan
                ma20_up = bool(pd.notna(ma20_slope) and ma20_slope >= 0)
            if len(g) >= 21:
                kdr20 = float(g["close"].iloc[-21]) if pd.notna(g["close"].iloc[-21]) else np.nan
                kdr20_good = bool(pd.notna(kdr20) and close >= kdr20)

            n = int(max(1, min(10, new_complete_days)))

            ma5_s = g["close"].rolling(5).mean()
            ma10_s = g["close"].rolling(10).mean()
            ma20_s = g["close"].rolling(20).mean()
            ma60_s = g["close"].rolling(60).mean()

            three_s = (g["close"] >= ma5_s) & (g["close"] >= ma10_s) & (g["close"] >= ma20_s)
            four_s = three_s & (g["close"] >= ma60_s)

            ma20_slope_s = ma20_s - ma20_s.shift(5)
            hold10_two_s = (g["close"] >= ma10_s) & (g["close"].shift(1) >= ma10_s.shift(1))
            three_strong_s = three_s & (ma5_s > ma10_s) & (ma10_s > ma20_s) & ((ma20_slope_s >= 0) | hold10_two_s)

            new_three = bool((three_s & (~three_s.shift(1).fillna(False))).tail(n).any())
            new_four = bool((four_s & (~four_s.shift(1).fillna(False))).tail(n).any())
            new_three_strong = bool((three_strong_s & (~three_strong_s.shift(1).fillna(False))).tail(n).any())

            # 連續2日確認（避免盤整反覆穿越造成假訊號）
            confirm_three_2d = bool(three_s.iloc[-1] and three_s.iloc[-2]) if len(three_s) >= 2 else False
            confirm_four_2d = bool(four_s.iloc[-1] and four_s.iloc[-2]) if len(four_s) >= 2 else False
            confirm_three_strong_2d = bool(three_strong_s.iloc[-1] and three_strong_s.iloc[-2]) if len(three_strong_s) >= 2 else False
        except Exception:
            pass



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


        # --- 起漲模式（預設為加分/篩選）---
        if startup_mode != "自訂":
            if startup_mode == "起漲-拉回承接":
                pass_filter_pattern = bool(three_strong and (bias_state != "DANGER"))
                startup_tag = "起漲-拉回承接"
            elif startup_mode == "起漲-突破發動":
                pass_filter_pattern = bool(breakout_ok and (bias_state != "DANGER"))
                startup_tag = "起漲-突破發動"
            elif startup_mode == "趨勢-四海遊龍續漲":
                pass_filter_pattern = bool(four and (bias_state != "DANGER"))
                startup_tag = "趨勢-四海遊龍續漲"
            else:
                startup_tag = "自訂"


        # 今日新成立過濾（兩段式）：第一段只抓事件
        two_stage_new = bool(new_complete_filter != "不限")

        # 今日新成立過濾（抓新成立事件）
        pass_filter_new = True
        if new_complete_filter == "今日新三陽":
            pass_filter_new = bool(new_three)
        elif new_complete_filter == "今日新三陽(強)":
            pass_filter_new = bool(new_three_strong)
        elif new_complete_filter == "今日新四海":
            pass_filter_new = bool(new_four)

        if two_stage_new:
            pass_filter_leader = True
            pass_filter_breakout = True
            pass_filter_pattern = True
        if require_pattern == "三陽開泰":
            pass_filter_pattern = bool(three)
        elif require_pattern == "四海遊龍":
            pass_filter_pattern = bool(four)
        else:
            startup_tag = "自訂"
        pass_filter_pattern = True

        pass_filter_breakout = (not require_breakout) or breakout_ok

        if not (pass_filter_leader and pass_filter_pattern and pass_filter_breakout and pass_filter_anypattern and pass_filter_new):
            continue

        # 新成立確認狀態（僅做標記，不硬過濾）
        confirm_status = ""
        if new_complete_filter != "不限":
            if new_complete_filter == "今日新三陽":
                confirm_status = "已確認" if confirm_three_2d else "待確認"
            elif new_complete_filter == "今日新三陽(強)":
                confirm_status = "已確認" if confirm_three_strong_2d else "待確認"
            elif new_complete_filter == "今日新四海":
                confirm_status = "已確認" if confirm_four_2d else "待確認"

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

        # --- RS bonus apply（加分項）---
        rs_out = rs_out_map.get(sid, np.nan) if isinstance(rs_out_map, dict) else np.nan
        rs_rank = rs_rank_map.get(sid, np.nan) if isinstance(rs_rank_map, dict) else np.nan

        # 加分：領先大盤越多，加分越多（不做硬過濾）
        if pd.notna(rs_out):
            if rs_out >= 5:
                s += int(rs_bonus_weight)
                reasons.append(f"RS{rs_window}領先大盤 +{rs_out:.1f}%")
            elif rs_out >= 2:
                s += max(1, int(rs_bonus_weight * 0.66))
                reasons.append(f"RS{rs_window}領先大盤 +{rs_out:.1f}%")
            elif rs_out >= 0:
                s += max(1, int(rs_bonus_weight * 0.33))
            elif rs_out <= -3:
                s -= max(1, int(rs_bonus_weight * 0.33))
                reasons.append(f"RS{rs_window}落後大盤 {rs_out:.1f}%")

        # 額外加分：RS 排名在前段班（候選池內相對強）
        if pd.notna(rs_rank):
            if rs_rank >= 0.8:
                s += 3
            elif rs_rank >= 0.6:
                s += 1

        # --- startup_mode bonus（加分項）---
        if startup_tag == "起漲-拉回承接" and three_strong:
            s += 4
        elif startup_tag == "起漲-突破發動" and breakout_ok:
            s += 5
        elif startup_tag == "趨勢-四海遊龍續漲" and four and hold10:
            s += 4

        # 市場寬度加權（盤後）：寬度偏弱時偏保守；寬度強時略加分
        try:
            breg = st.session_state.get('breadth_regime', {})
            bstate = str(breg.get('state','NEUTRAL'))
            if bstate == 'WEAK':
                s -= 3
                reasons.append('市場寬度偏弱')
            elif bstate == 'STRONG':
                s += 1
            elif bstate == 'DEFENSIVE_ROTATION':
                s -= 1
                reasons.append('防禦寬度較強')
        except Exception:
            pass

        # Clamp score
        s = max(0, min(100, s))

        rows.append({
            "stock_id": sid,
            "stock_name": name,
            "industry_category": industry,
            "漲跌幅(%)": round(chg, 2),
            "成交金額(億)": round(money / 1e8, 2),
            "量比": round(vol_ratio, 2),
            "確認狀態": confirm_status,
            "連2日三陽": bool(confirm_three_2d),
            "連2日三陽(強)": bool(confirm_three_strong_2d),
            "連2日四海": bool(confirm_four_2d),
            "MA20翻揚": bool(ma20_up),
            "扣抵值(20)": round(kdr20, 2) if pd.notna(kdr20) else np.nan,
            "扣抵有利": bool(kdr20_good),
            "RS{}超越大盤(%)".format(rs_window): round(rs_out, 2) if pd.notna(rs_out) else np.nan,
            "RS Rank(%)": round(rs_rank * 100, 1) if pd.notna(rs_rank) else np.nan,
            "BIAS20(%)": round(bias20, 2) if pd.notna(bias20) else np.nan,
            "股性": profile,
            "起漲型態": startup_tag,
            "三陽開泰(強)": bool(three_strong),
            "主題標記": " / ".join(sorted(list(theme_tag_map.get(sid, set())))) if isinstance(theme_tag_map, dict) else "",
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
    out = out.drop_duplicates(subset=["stock_id"], keep="first").reset_index(drop=True)
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

    # ETF market breadth regime (0050/006208 vs 00878/0056)
    try:
        etf_b = compute_etf_breadth_regime(vol_rank)
    except Exception:
        etf_b = {"rows": pd.DataFrame(), "large_avg": np.nan, "def_avg": np.nan, "state": "NEUTRAL", "msg": "市場寬度資料不足"}
    res["etf_breadth"] = etf_b
    # also expose to session_state for screener / UI
    st.session_state["breadth_regime"] = etf_b


    # sector strength & leaders
    res["sector_strength"] = compute_sector_strength(vol_rank, sector_flow, top_n=60)
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 , tab8 = st.tabs([
        "1. 健診/技術圖", "2. 資金流向儀錶板", "3. 營收診斷", "4. 主題族群雷達", "5. 交易計畫引擎", "6. 老王選股器", "7. 宏觀追蹤", "8. 回測研究"
    ])
res = st.session_state.get("result")

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.subheader("健診：老王策略 + 籌碼分析（5/10/20/60 + 三陽開泰/四海遊龍 + 法人/融資）")
    st.caption("提示：健診結論會參考 ETF 市場寬度（0050/006208 vs 00878/0056）。")
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
                "MA20翻揚": ow.get("ma20_turn_up"),
                "10MA連續守住": ow.get("hold_ma10_2d"),
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
                st.caption("籌碼圖表資料不足（請稍後重試或確認資料區間）。")

            # --- 技術圖（加上 5/10/20/60） ---
            
            # --- 大戶 vs 散戶（每週股權分散）---
            st.markdown("### 籌碼結構：大戶 vs 散戶（每週）")
            w_holding = build_retail_big_weekly(token, target_sid, start_date="2019-01-01")
            sig_h = compute_holding_signal(w_holding)

            if sig_h.get("ok"):
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("籌碼結構燈號", sig_h["light"])
                k2.metric("大戶>=500張 WoW(%)", f"{sig_h['big500_wow']:+.2f}")
                k3.metric("大戶>=1000張 WoW(%)", f"{sig_h['big1000_wow']:+.2f}")
                k4.metric("散戶1-10張 WoW(%)", f"{sig_h['retail_wow']:+.2f}")
                st.caption(f"{sig_h['date']}｜{sig_h['msg']}｜連續週數：大戶>=500 ↑{sig_h['big500_up_weeks']}、大戶>=1000 ↑{sig_h['big1000_up_weeks']}、散戶(1-10) ↓{sig_h['retail_down_weeks']}")
            else:
                st.info("無法取得股權分散資料（可能該股資料不足或資料源暫時不可用）。")

            with st.expander("查看：大戶/散戶每週明細與圖表", expanded=False):
                if w_holding is None or w_holding.empty:
                    st.info("本股無股權分散資料。")
                else:
                    show_cols = ["date",
                                 "retail_1_10_percent","retail_1_10_percent_wow",
                                 "big_100_percent","big_100_percent_wow",
                                 "big_500_percent","big_500_percent_wow",
                                 "big_1000_percent","big_1000_percent_wow"]
                    show_cols = [c for c in show_cols if c in w_holding.columns]
                    st.dataframe(w_holding[show_cols].tail(26), use_container_width=True)

                    figh = go.Figure()
                    if "retail_1_10_percent" in w_holding.columns:
                        figh.add_trace(go.Scatter(x=w_holding["date"], y=w_holding["retail_1_10_percent"], name="散戶(1-10張)%"))
                    if "big_100_percent" in w_holding.columns:
                        figh.add_trace(go.Scatter(x=w_holding["date"], y=w_holding["big_100_percent"], name="大戶(>=100張)%"))
                    if "big_500_percent" in w_holding.columns:
                        figh.add_trace(go.Scatter(x=w_holding["date"], y=w_holding["big_500_percent"], name="大戶(>=500張)%"))
                    if "big_1000_percent" in w_holding.columns:
                        figh.add_trace(go.Scatter(x=w_holding["date"], y=w_holding["big_1000_percent"], name="大戶(>=1000張)%"))
                    figh.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(figh, use_container_width=True)

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
    st.subheader("資金流向儀錶板（熱力圖 + 族群強弱 + 代表股）")
    if res is None:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」。")
    else:
        meta = res.get("market_meta", {})
        sector_flow = res.get("sector_flow", pd.DataFrame())
        strength = res.get("sector_strength", pd.DataFrame())

        vol_rank_all = res.get("volume_rank", pd.DataFrame())

        if sector_flow is None or sector_flow.empty:
            st.error("族群資金流向資料為空。")
        else:
            st.caption(f"資料來源：{meta.get('source','')}；掃描日期：{meta.get('scan_date','')}")

            # --- 盤勢框架：加權 vs 櫃買（分歧提示）---
            st.markdown("### 盤勢框架：加權 vs 櫃買（分歧提示）")
            tw = compute_index_state(token, "0050")     # 加權代理
            oc = compute_index_state(token, "006201")   # 櫃買代理
            breadth = compute_breadth_by_market(vol_rank_all, stock_info)
            hint = build_divergence_hint(tw, oc, breadth)

            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("加權代理 0050", f"{tw.get('close', np.nan):.2f}" if pd.notna(tw.get("close", np.nan)) else "N/A")
                st.caption(f"趨勢：{tw.get('trend','-')}｜20D:{tw.get('ret20', np.nan):+.2f}%｜60D:{tw.get('ret60', np.nan):+.2f}%")
            with a2:
                st.metric("櫃買代理 006201", f"{oc.get('close', np.nan):.2f}" if pd.notna(oc.get("close", np.nan)) else "N/A")
                st.caption(f"趨勢：{oc.get('trend','-')}｜20D:{oc.get('ret20', np.nan):+.2f}%｜60D:{oc.get('ret60', np.nan):+.2f}%")
            with a3:
                bt = breadth.get("breadth_tse", np.nan)
                bo = breadth.get("breadth_otc", np.nan)
                st.metric("上市/上櫃廣度", f"{bt:.1f}% / {bo:.1f}%" if (pd.notna(bt) and pd.notna(bo)) else "N/A")
                st.caption(breadth.get("coverage_note",""))

            if hint["level"] == "warning":
                st.warning(f"{hint['title']}：{hint['detail']}")
            elif hint["level"] == "success":
                st.success(f"{hint['title']}：{hint['detail']}")
            else:
                st.info(f"{hint['title']}：{hint['detail']}")

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

            # --- 市場總覽 KPI ---
            vol_rank_all = res.get("volume_rank", pd.DataFrame())
            if isinstance(vol_rank_all, pd.DataFrame) and not vol_rank_all.empty:
                mkt = vol_rank_all.copy()
                mkt["industry_category"] = mkt.get("industry_category", "其他").fillna("其他")
                mkt = ensure_change_rate(mkt)
                m_money_col = pick_money_col(mkt)
                mkt[m_money_col] = pd.to_numeric(mkt[m_money_col], errors="coerce").fillna(0.0)

                total_money_yi = float(mkt[m_money_col].sum() / 1e8)
                signed_money_yi = float((mkt[m_money_col] * np.sign(mkt["change_rate"])).sum() / 1e8)
                up_ratio = float((mkt["change_rate"] > 0).mean() * 100)

                top10_share = np.nan
                if "總成交金額(億)" in strength.columns and total_money_yi > 0:
                    top10_share = float(strength.head(10)["總成交金額(億)"].sum() / total_money_yi * 100)

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("全市場成交金額(億)", f"{total_money_yi:,.0f}")
                k2.metric("全市場資金偏多(億)", f"{signed_money_yi:,.0f}")
                k3.metric("上漲比例(%)", f"{up_ratio:.1f}%")
                k4.metric("Top10 佔比(%)", "-" if pd.isna(top10_share) else f"{top10_share:.1f}%")

            
            # --- 方塊熱力圖（Treemap）---
            st.markdown("### 方塊熱力圖（Treemap：大小=成交金額、顏色=族群漲跌）")
            st.caption("更接近交易App的呈現方式：區塊大小代表成交金額，顏色代表族群當日強弱（加權平均漲跌幅）。")
            tm_n = st.slider("Treemap 顯示族群數（依成交金額排序）", min_value=10, max_value=80, value=40, step=5)

            if isinstance(vol_rank_all, pd.DataFrame) and not vol_rank_all.empty:
                tm = vol_rank_all.copy()
                tm["industry_category"] = tm.get("industry_category", "其他").fillna("其他")
                tm = ensure_change_rate(tm)
                tm_money_col = pick_money_col(tm)
                tm[tm_money_col] = pd.to_numeric(tm[tm_money_col], errors="coerce").fillna(0.0)

                # 加權平均漲跌幅（以成交金額加權）
                def _wavg(g):
                    money = g[tm_money_col].sum()
                    if money <= 0:
                        return 0.0
                    return float((g["change_rate"] * g[tm_money_col]).sum() / money)

                grp = tm.groupby("industry_category", as_index=False).apply(
                    lambda g: pd.Series({
                        "成交金額(億)": float(g[tm_money_col].sum() / 1e8),
                        "加權漲跌幅(%)": _wavg(g),
                        "上漲比例(%)": float((g["change_rate"] > 0).mean() * 100),
                    })
                ).reset_index(drop=True)

                grp = grp.sort_values("成交金額(億)", ascending=False).head(int(tm_n)).copy()
                grp["label"] = grp.apply(lambda r: f"{r['industry_category']}<br>{r['加權漲跌幅(%)']:+.2f}%", axis=1)

                fig_tm = go.Figure(
                    go.Treemap(
                        labels=grp["label"],
                        parents=[""] * len(grp),
                        values=grp["成交金額(億)"],
                        customdata=np.stack([grp["industry_category"], grp["成交金額(億)"], grp["加權漲跌幅(%)"], grp["上漲比例(%)"]], axis=1),
                        marker=dict(
                            colors=grp["加權漲跌幅(%)"],
                            colorscale="RdYlGn",
                            cmid=0,
                            line=dict(width=1),
                        ),
                        hovertemplate=(
                            "族群: %{customdata[0]}<br>"
                            "成交金額(億): %{customdata[1]:.2f}<br>"
                            "加權漲跌幅(%): %{customdata[2]:+.2f}<br>"
                            "上漲比例(%): %{customdata[3]:.1f}<extra></extra>"
                        ),
                        textinfo="label",
                    )
                )
                fig_tm.update_layout(
                    height=620,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_tm, use_container_width=True)
            else:
                st.info("Treemap 需要全市場資料（volume_rank），目前資料不足。")


            st.markdown("### 矩陣熱力圖（Heatmap：相對強弱矩陣）")
            heat_n = st.slider("熱力圖顯示族群數（依資金偏多排序）", min_value=10, max_value=60, value=30, step=5)
            heat_mode = st.radio("熱力圖模式", ["標準化（相對強弱）", "原始值（絕對數字）"], horizontal=True)

            heat_src = strength.head(int(heat_n)).copy()
            cols = [c for c in ["資金偏多(億)", "總成交金額(億)", "上漲比例(%)", "領導集中度Top3(%)"] if c in heat_src.columns]
            if cols:
                heat_base = heat_src.set_index("族群")[cols].copy()
                heat_base = heat_base.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                if heat_mode.startswith("標準化"):
                    # z-score by column
                    z = (heat_base - heat_base.mean(axis=0)) / heat_base.std(axis=0, ddof=0).replace(0, np.nan)
                    heat_z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    z_vals = heat_z.values
                else:
                    z_vals = heat_base.values

                fig_h = go.Figure(
                    data=go.Heatmap(
                        z=z_vals,
                        x=list(heat_base.columns),
                        y=list(heat_base.index),
                        customdata=heat_base.values,
                        hovertemplate="族群: %{y}<br>%{x}: %{customdata:.2f}<extra></extra>",
                    )
                )
                fig_h.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.info("熱力圖欄位不足（strength 表格缺少必要欄位）。")

            st.markdown("### 族群雙榜（流入 / 流出）")
            if isinstance(sector_flow, pd.DataFrame) and not sector_flow.empty:
                flow = sector_flow.copy()
                flow["資金流向(億)"] = flow["signed_money"] / 1e8
                flow = flow.rename(columns={"industry_category": "族群"})

                colA, colB = st.columns(2)
                with colA:
                    st.caption("強勢流入 Top 10")
                    st.dataframe(flow.sort_values("signed_money", ascending=False).head(10)[["族群", "資金流向(億)"]], use_container_width=True)
                with colB:
                    st.caption("強勢流出 Top 10")
                    st.dataframe(flow.sort_values("signed_money", ascending=True).head(10)[["族群", "資金流向(億)"]], use_container_width=True)

                st.caption("輪動觀察（第 11–30 名）")
                rot = flow.sort_values("signed_money", ascending=False).iloc[10:30][["族群", "資金流向(億)"]].copy()
                st.dataframe(rot, use_container_width=True)


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
# Tab 4
# -----------------------------
with tab4:
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
# Tab 5
# -----------------------------
with tab5:
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
                "MA20翻揚": ow.get("ma20_turn_up"),
                "10MA連續守住": ow.get("hold_ma10_2d"),
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
# Tab 6
# -----------------------------
with tab6:
    st.subheader("老王選股器（5/10/20/60 + 三陽開泰/四海遊龍 + 帶量突破 + 守10MA + 領導股）")

    # 盤後 SOP 使用說明（內嵌備註）
    with st.expander("盤後SOP：怎麼用選股器（超清楚版本）", expanded=False):
        st.markdown("""
**你要抓「今天才剛起漲」——用今日新成立（平常建議用這個）**
1. 今日新成立過濾：**今日新三陽(強)**
2. 新成立視窗：**1**
3. 起漲模式：**自訂**（讓它只是標籤，不要硬篩）
4. 用表格排序挑：
   - 先看 **確認狀態 = 已確認**
   - 再看 **RS、成交金額、MA20翻揚、扣抵有利**

**你要抓「突破型」但不在乎是不是今天剛發生——用起漲模式**
1. 今日新成立過濾：**不限**
2. 起漲模式：**起漲-突破發動**
3. 其他加分：**RS、成交金額、量比**

**你要抓「可以抱的趨勢股」——用起漲模式**
1. 今日新成立過濾：**不限**
2. 起漲模式：**趨勢-四海遊龍續漲**
""")


    if res is None:
        st.info("請先按左側「一鍵更新（含交易計畫引擎）」，取得全市場掃描資料後再跑選股器。")
    else:
        vol_rank = res.get("volume_rank", pd.DataFrame())
        sector_flow = res.get("sector_flow", pd.DataFrame())
        meta = res.get("market_meta", {})

        st.caption(f"資料來源：{meta.get('source','')}；掃描日期：{meta.get('scan_date','')}")

        # 上市/上櫃本機名單診斷（Local-first）
        ms_local = get_official_market_sets()
        st.caption(f"本機名單：上市(TSE) {len(ms_local.get('TSE', set()))} 檔｜上櫃(OTC) {len(ms_local.get('OTC', set()))} 檔（若為0，請確認 data/market_tse.csv、data/market_otc.csv 已上傳）")


        screener_mode = st.selectbox(
            "選股器模式",
            options=["起漲雷達（新成立/兩段式）", "噴出雷達（大漲/漲停警示）"],
            index=0,
            key="ow_mode"
        )
        if screener_mode.startswith("噴出"):
            st.info("噴出雷達用途：抓『今天爆拉』的股票並提示過熱風險（通常適合持有者續抱/移動停利，不適合新追價）。")

        st.markdown("### 盤後一鍵模式")
        if st.button("📌 套用盤後雷達模式（推薦）", key="ow_preset_postclose"):
            # 盤後建議：全市場掃描 + 今日新三陽(強) + 視窗=1天，其他條件改用表格排序挑選
            st.session_state["ow_universe_mode"] = "全市場（較慢）"
            st.session_state["ow_new_complete"] = "今日新三陽(強)"
            st.session_state["ow_new_days"] = 1
            st.session_state["ow_startup_mode"] = "自訂"
            st.session_state["ow_market_filter"] = "全部"
            st.session_state["ow_strict_market"] = True
            st.session_state["ow_debug_market"] = False
            st.session_state["ow_min_money_yi"] = 0.0
            st.session_state["ow_require_leader"] = False
            st.session_state["ow_require_breakout"] = False
            st.session_state["ow_require_pattern"] = "不限"
            st.session_state["ow_universe_top_n"] = 500
            st.session_state["ow_output_top_k"] = 80
            st.session_state["ow_rs_bonus_weight"] = 6

        # -----------------------------
        # 噴出雷達：大漲/漲停（盤後警示）
        # -----------------------------
        if screener_mode.startswith("噴出"):
            v_all = res.get("volume_rank", pd.DataFrame())
            if v_all is None or v_all.empty:
                st.error("全市場資料為空，無法產生噴出雷達。請先一鍵更新。")
            else:
                v = ensure_change_rate(v_all.copy())
                money_col = None
                for c in ["Trading_money", "total_amount", "amount"]:
                    if c in v.columns:
                        money_col = c
                        break
                if money_col is None:
                    v["money"] = 0.0
                    money_col = "money"
                v[money_col] = pd.to_numeric(v[money_col], errors="coerce").fillna(0.0)

                pct_th = st.slider("漲幅門檻(%)", min_value=5.0, max_value=10.0, value=9.0, step=0.5, key="pop_pct")
                min_money = st.number_input("成交金額門檻（億）", min_value=0.0, max_value=50.0, value=1.0, step=0.5, key="pop_money")
                pop_top = st.slider("輸出筆數", min_value=20, max_value=200, value=80, step=10, key="pop_top")

                pick = v[(v["change_rate"] >= pct_th) & (v[money_col] >= float(min_money) * 1e8)].copy()
                if pick.empty:
                    st.info("本次條件下沒有符合的噴出股票。")
                else:
                    # Basic output; avoid heavy history fetch. Flag overheat by BIAS20 if available in today data, else use change_rate proxy.
                    pick["成交金額(億)"] = pick[money_col] / 1e8
                    cols = [c for c in ["stock_id", "stock_name", "industry_category", "change_rate", "成交金額(億)", "volume_ratio"] if c in pick.columns]
                    show = pick.sort_values(["change_rate", "成交金額(億)"], ascending=[False, False]).head(int(pop_top)).copy()
                    show = show.rename(columns={"change_rate": "漲跌幅(%)", "industry_category": "族群"})
                    # Overheat hint
                    show["風險提示"] = show["漲跌幅(%)"].apply(lambda x: "過熱（避免追價）" if pd.notna(x) and x >= 9 else "留意回測")
                    st.dataframe(show[["stock_id","stock_name","族群","漲跌幅(%)","成交金額(億)","風險提示"] + ([ "volume_ratio"] if "volume_ratio" in show.columns else [])], use_container_width=True)
                    st.caption("提示：噴出雷達是盤後警示清單，建議持有者用守10MA/移動停利管理；不建議新追價。")
            st.stop()


        c1, c2, c3, c4 = st.columns(4)
        universe_top_n = c1.number_input("候選池（依成交金額前 N）", min_value=50, max_value=800, value=300, step=50, key="ow_universe_top_n")
        universe_mode_ui = st.selectbox("候選池模式", options=["成交金額TopN（較快）", "全市場（較慢）"], index=0, key="ow_universe_mode")
        output_top_k = c2.number_input("輸出 Top K", min_value=20, max_value=200, value=80, step=10, key="ow_output_top_k")
        rs_bonus_weight = st.slider("RS 加分權重（加分項）", min_value=0, max_value=10, value=6, step=1, key="ow_rs_bonus_weight")
        exclude_etf_index_ui = st.checkbox("排除 ETF / 指數", value=True, key="ow_exclude_etf")
        market_filter_ui = st.selectbox("市場篩選", options=["全部", "上市(TSE)", "上櫃(OTC)"], index=0, key="ow_market_filter")
        strict_market_ui = st.checkbox("嚴格市場篩選（不回退）", value=True, key="ow_strict_market")
        debug_market_ui = st.checkbox("顯示市場篩選診斷", value=False, key="ow_debug_market")
        startup_mode_ui = st.selectbox("起漲模式", options=["自訂", "起漲-拉回承接", "起漲-突破發動", "趨勢-四海遊龍續漲"], index=0, key="ow_startup_mode")
        new_complete_ui = st.selectbox("今日新成立過濾", options=["不限", "今日新三陽", "今日新三陽(強)", "今日新四海"], index=0, key="ow_new_complete")
        # 今日新成立視窗（交易日）：1=今天剛成立；3/5=近幾天剛成立
        new_complete_days_ui = 1
        if str(new_complete_ui) != "不限":
            new_complete_days_ui = st.slider("新成立視窗（交易日）", min_value=1, max_value=10, value=1, step=1, key="ow_new_days")
            st.info("兩段式：只抓『新成立事件』；其他嚴格條件將自動忽略。請用下方表格欄位（RS/成交金額/是否領導股等）排序人工挑。")
        st.caption("市場篩選：若資料源無法辨識上市/上櫃欄位，系統會自動回退為不篩選（避免結果為空）。")
        min_money_yi = c3.number_input("成交金額門檻（億）", min_value=0.0, max_value=50.0, value=1.0, step=0.5, key="ow_min_money_yi")
        require_leader = c4.checkbox("只挑族群領導股（Top3）", value=True, key="ow_require_leader")

        c5, c6 = st.columns(2)
        require_pattern = c5.selectbox("型態過濾", ["不限", "三陽開泰", "四海遊龍"], index=0, key="ow_require_pattern")
        require_breakout = c6.checkbox("只挑『突破前高且帶量』", value=False, key="ow_require_breakout")
        require_any_pattern_ui = st.checkbox("至少要有三陽/四海（避免出現沒有型態的股票）", value=True, key="ow_any_pattern")

        run_btn = st.button("🚀 執行老王選股器", type="primary")

        if "oldwang_screener_df" not in st.session_state:
            st.session_state["oldwang_screener_df"] = pd.DataFrame()

        if run_btn:
            if vol_rank is None or vol_rank.empty:
                st.error("全市場資料為空，無法選股。請確認掃描來源是否可回傳資料。")
            else:
                with st.spinner("選股器運算中（抓取候選股近 60+ 日資料並計算訊號）..."):
                    market_filter = "ALL"
                    if 'market_filter_ui' in locals():
                        if market_filter_ui.startswith('上市'):
                            market_filter = 'TSE'
                        elif market_filter_ui.startswith('上櫃'):
                            market_filter = 'OTC'
                    
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
                        rs_bonus_weight=int(rs_bonus_weight) if 'rs_bonus_weight' in locals() else 6,
                        rs_window=20,
                        rs_proxy_id='0050',
                        market_filter=market_filter,
                        exclude_etf_index=bool(exclude_etf_index_ui) if 'exclude_etf_index_ui' in locals() else True,
                        strict_market=bool(strict_market_ui) if 'strict_market_ui' in locals() else True,
                        startup_mode=str(startup_mode_ui) if 'startup_mode_ui' in locals() else '自訂',
                        new_complete_filter=str(new_complete_ui) if 'new_complete_ui' in locals() else '不限',
                        new_complete_days=int(new_complete_days_ui) if 'new_complete_days_ui' in locals() else 1,
                        universe_all=bool(universe_mode_ui.startswith('全市場')) if 'universe_mode_ui' in locals() else False,
                    )

                    if 'debug_market_ui' in locals() and debug_market_ui:
                        with st.expander("市場篩選診斷（上市/上櫃辨識）", expanded=True):
                            mcol = _detect_market_col(stock_info)
                            st.write("偵測到的市場欄位：", mcol if mcol else "（未偵測到）")
                            if mcol:
                                info_m = stock_info[["stock_id", mcol]].drop_duplicates().copy()
                                info_m["_m"] = info_m[mcol].apply(_normalize_market_value)
                                st.write("映射統計：TSE", int((info_m["_m"]=="TSE").sum()), "OTC", int((info_m["_m"]=="OTC").sum()), "Unknown", int(info_m["_m"].isna().sum()))
                                st.write("原始值樣本（去重前20筆）：")
                                st.dataframe(info_m[[mcol]].drop_duplicates().head(20), use_container_width=True)
                            st.write("若 Unknown 很高，代表 TaiwanStockInfo 的欄位內容不是上市/上櫃代碼，需調整映射規則。")
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

# -----------------------------
# Tab 7: 宏觀追蹤
# -----------------------------
with tab7:
    st.subheader("宏觀追蹤（美國經濟數據日曆 + 美股ETF收盤 + 台灣ETF市場寬度）")

    # 1) US economic calendar
    st.markdown("### 1) 美國經濟數據日曆（本週）")
    cal = fetch_forexfactory_calendar_thisweek()
    if cal is None or cal.empty:
        st.info("無法取得經濟日曆資料（來源可能暫時不可用）。")
    else:
        # normalize columns if present
        show = cal.copy()
        # filter USD only (common column 'country')
        if "country" in show.columns:
            show = show[show["country"].astype(str).str.upper() == "USD"].copy()
        # impact filter
        impact_levels = []
        if "impact" in show.columns:
            impact_levels = sorted(show["impact"].dropna().unique().tolist())
        impact_sel = st.multiselect("重要性（impact）", options=impact_levels if impact_levels else ["High","Medium","Low"], default=[x for x in ["High","Medium"] if x in (impact_levels or ["High","Medium"])])
        if "impact" in show.columns and impact_sel:
            show = show[show["impact"].isin(impact_sel)].copy()

        # build Taipei time column
        if "date" in show.columns:
            if "time" in show.columns:
                show["台北時間"] = show.apply(lambda r: _ff_to_datetime_taipei(str(r.get("date","")), str(r.get("time",""))), axis=1)
            else:
                show["台北時間"] = show["date"].astype(str)

        cols = []
        for c in ["台北時間", "title", "impact", "previous", "forecast", "actual"]:
            if c in show.columns:
                cols.append(c)
        # fallback columns
        if not cols:
            cols = list(show.columns)[:8]
        show = show.sort_values(cols[0]) if cols else show
        st.dataframe(show[cols].head(80), use_container_width=True)

    st.markdown("---")

    # 2) US ETFs after close
    st.markdown("### 2) 美股收盤追蹤：SOXX 與 QQQ（收盤價與漲跌幅）")
    c1, c2 = st.columns(2)
    with c1:
        soxx = get_us_etf_close_from_stooq("soxx.us")
        if soxx.get("ok"):
            st.metric("SOXX 收盤", f"{soxx['close']:.2f}", f"{soxx['chg']:+.2f} ({soxx['chg_pct']:+.2f}%)")
            st.caption(f"日期：{soxx.get('date','')}")
        else:
            st.info("SOXX 資料取得失敗")
    with c2:
        qqq = get_us_etf_close_from_stooq("qqq.us")
        if qqq.get("ok"):
            st.metric("QQQ 收盤", f"{qqq['close']:.2f}", f"{qqq['chg']:+.2f} ({qqq['chg_pct']:+.2f}%)")
            st.caption(f"日期：{qqq.get('date','')}")
        else:
            st.info("QQQ 資料取得失敗")

    st.markdown("---")

    # 3) Taiwan ETF breadth
    st.markdown("### 3) 台灣ETF市場寬度（成分股：上漲/下跌/平盤）")
    if res is None:
        st.info("請先在主流程更新全市場資料（左側一鍵更新）後再查看寬度。")
    else:
        vol_rank_all = res.get("volume_rank", pd.DataFrame())
        etfs = ["0050", "006208", "00878", "0056"]
        rows = []
        for etf in etfs:
            b = compute_tw_etf_breadth(vol_rank_all, etf)
            if b.get("ok"):
                rows.append({
                    "ETF": etf,
                    "成分股數": b["total"],
                    "上漲": b["up"],
                    "下跌": b["dn"],
                    "平盤": b["eq"],
                    "上漲比例(%)": round(b["up_ratio"], 1) if pd.notna(b["up_ratio"]) else np.nan,
                })
            else:
                rows.append({"ETF": etf, "成分股數": np.nan, "上漲": np.nan, "下跌": np.nan, "平盤": np.nan, "上漲比例(%)": np.nan})
        dfb = pd.DataFrame(rows)
        st.dataframe(dfb, use_container_width=True)

        # breadth regime summary
        try:
            etf_b = compute_etf_breadth_regime(vol_rank_all)
            st.markdown(f"**寬度狀態：{etf_b.get('state','NEUTRAL')}**｜{etf_b.get('msg','')}")
        except Exception:
            pass

        st.caption("說明：成分股清單來源為 MoneyDJ ETF 持股表，並用全市場日線/快照的 change_rate 計算當日上漲/下跌家數。")



# -----------------------------
# Tab 8: 回測研究（2年）
# -----------------------------

# -----------------------------
# Tab 8: 回測研究（2年）
# -----------------------------
with tab8:

    # --- 回測研究上鎖（Streamlit Cloud 建議）---
    backtest_key = st.secrets.get("BACKTEST_KEY", "")
    if not backtest_key:
        st.warning("回測研究已設定為上鎖模式，但尚未在 Secrets 設定 BACKTEST_KEY。請到 Streamlit Cloud → Settings → Secrets 加入 BACKTEST_KEY。")
        st.stop()

    if "backtest_authed" not in st.session_state:
        st.session_state["backtest_authed"] = False

    if not st.session_state["backtest_authed"]:
        st.info("回測研究已上鎖。")
        entered = st.text_input("輸入 BACKTEST_KEY 以解鎖回測研究", type="password", key="bt_pw")
        if st.button("解鎖回測研究", key="unlock_bt"):
            if entered == backtest_key:
                st.session_state["backtest_authed"] = True
                st.success("已解鎖回測研究。")
            else:
                st.error("密碼錯誤。")
        st.stop()

    st.subheader("回測研究：市場寬度 vs 台股（近2年）")

    st.caption("建議做法（多人使用更穩）：回測預設讀取 data/breadth_2y.csv（預先計算好的寬度序列）。只有管理者才需要更新該檔案。")

    st.info(breadth_csv_meta())

    horizon = st.selectbox("預測視窗（交易日）", options=[1, 5, 20], index=1)

    df_csv = load_breadth_csv()
    if df_csv.empty:
        st.warning("尚未偵測到 data/breadth_2y.csv。請由管理者在下方生成（或把檔案放進 repo 的 data/ 目錄）。")
    else:
        rb = backtest_breadth_vs_index(df_csv, horizon=int(horizon))
        if isinstance(rb, dict) and rb.get("ok"):
            st.markdown("### 統計摘要")
            c1, c2, c3 = st.columns(3)
            c1.metric("相關係數 Corr(Breadth, FutureReturn)", "-" if pd.isna(rb.get("corr")) else f"{rb.get('corr'):.3f}")
            c2.metric("事件：指數漲但寬度差", f"{rb['evt_up_bad']['n']} 次")
            c3.metric("事件：指數跌但寬度好", f"{rb['evt_dn_good']['n']} 次")

            st.markdown("### 分位數回測（Breadth 五分位）")
            qtbl = rb.get("qtbl", pd.DataFrame())
            if isinstance(qtbl, pd.DataFrame) and not qtbl.empty:
                qtbl2 = qtbl.copy()
                qtbl2["q"] = qtbl2["q"].astype(int) + 1
                qtbl2 = qtbl2.rename(columns={"q": "五分位(1低→5高)", "mean": "未來報酬均值(%)", "count": "樣本數", "win_rate": "勝率"})
                st.dataframe(qtbl2, use_container_width=True)

            st.markdown("### 分歧事件統計")
            e1 = rb["evt_up_bad"]; e2 = rb["evt_dn_good"]
            st.write(f"- 指數漲但寬度<50%：樣本 {e1['n']}，未來平均報酬 {e1['avg'] if pd.notna(e1['avg']) else '-'}，勝率 {e1['win'] if pd.notna(e1['win']) else '-'}")
            st.write(f"- 指數跌但寬度>55%：樣本 {e2['n']}，未來平均報酬 {e2['avg'] if pd.notna(e2['avg']) else '-'}，勝率 {e2['win'] if pd.notna(e2['win']) else '-'}")

            st.markdown("### 序列檢視（最近 120 筆）")
            show_cols = ["date", "ratio", "adv", "dec", "proxy_close", rb["fwd_col"], "idx_up_breadth_bad", "idx_dn_good"] if "idx_dn_good" in df_csv.columns else []
            # be defensive
            show_cols = [c for c in ["date", "ratio", "adv", "dec", "proxy_close", rb["fwd_col"], "idx_up_breadth_bad", "idx_dn_breadth_good"] if c in df_csv.columns]
            st.dataframe(df_csv[show_cols].tail(120), use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_csv["date"], y=df_csv["ratio"], name="Breadth(上漲家數比例)"))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Breadth")
            st.plotly_chart(fig, use_container_width=True)

        # download csv
        st.download_button(
            "下載 breadth_2y.csv",
            data=df_csv.to_csv(index=False).encode("utf-8"),
            file_name="breadth_2y.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("---")
    st.markdown("### 生成/更新 breadth_2y.csv（重運算）")
    st.warning("此操作會重算近2年全市場寬度序列，耗時且會消耗 API 額度；建議僅在盤後或必要時執行。")
    if st.button("生成/更新 breadth_2y.csv（近2年）", type="primary"):
        with st.spinner("生成中：逐日計算全市場寬度（近2年）..."):
            df = build_breadth_series_2y(token, proxy_id="0050", years=2)
            if df is None or df.empty:
                st.error("生成失敗：未能取得寬度序列。")
            else:
                save_breadth_csv(df)
                st.success("已更新 data/breadth_2y.csv（若要多人穩定使用，請把 data/breadth_2y.csv commit 回 repo）。")
