import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


def fetch_codes(url: str) -> list[str]:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))

    # 找「公司代號」欄位
    col = None
    for c in df.columns:
        if "公司代號" in str(c):
            col = c
            break
    if col is None:
        raise ValueError(f"找不到公司代號欄位，實際欄位={list(df.columns)}")

    codes = []
    for v in df[col].astype(str).tolist():
        m = re.match(r"^\s*(\d{4})\s*$", v)
        if m:
            codes.append(m.group(1))

    return sorted(set(codes))


def main():
    tse_url = "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"  # 上市
    otc_url = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"  # 上櫃

    tse = fetch_codes(tse_url)
    otc = fetch_codes(otc_url)

    Path("data").mkdir(exist_ok=True)

    pd.DataFrame({"stock_id": tse}).to_csv("data/market_tse.csv", index=False, encoding="utf-8")
    pd.DataFrame({"stock_id": otc}).to_csv("data/market_otc.csv", index=False, encoding="utf-8")

    print("done")
    print("TSE:", len(tse), "rows -> data/market_tse.csv")
    print("OTC:", len(otc), "rows -> data/market_otc.csv")


if __name__ == "__main__":
    main()
