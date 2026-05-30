"""
news.py — Fetch and store historical headlines via Alpha Vantage News Sentiment API
LNA-Agent: Lobster News Alpha

Alpha Vantage free tier: 25 requests/day, news back to ~2020
Timestamps: minute-level precision

Usage:
  # Fetch headlines for one ticker
  python src/news.py --ticker AAPL --date-from 2020-01-01 --date-to 2020-12-31

  # Fetch for multiple tickers
  python src/news.py --tickers AAPL MSFT JPM --date-from 2020-01-01 --date-to 2020-12-31
"""

from dotenv import load_dotenv
import os
load_dotenv()

import requests
import pandas as pd
from pathlib import Path
import argparse
import time
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY  = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
NEWS_DIR = Path("data/news/raw")
LIMIT    = 200  # max per request
# ─────────────────────────────────────────────────────────────────────────────


def fetch_batch(ticker: str, time_from: str, time_to: str) -> list[dict]:
    """
    Fetch one batch of headlines.
    time_from/time_to format: "20200101T0000"
    """
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   ticker,
        "time_from": time_from,
        "time_to":   time_to,
        "limit":     LIMIT,
        "apikey":    API_KEY,
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        data = r.json()
    except Exception as e:
        print(f"  [ERROR] request failed: {e}")
        return []

    if "feed" not in data:
        msg = data.get("Note") or data.get("Information") or data.get("message") or str(data)
        print(f"  [WARN] {msg[:120]}")
        return []

    return data["feed"]


def parse_articles(articles: list[dict], ticker: str) -> pd.DataFrame:
    """Parse raw Alpha Vantage articles into clean DataFrame."""
    rows = []
    for a in articles:
        # Find ticker-specific sentiment score
        ticker_sentiment = 0.0
        for ts in a.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                try:
                    ticker_sentiment = float(ts.get("ticker_sentiment_score", 0))
                except Exception:
                    pass

        rows.append({
            "timestamp":        a.get("time_published", ""),
            "headline":         a.get("title", ""),
            "source":           a.get("source", ""),
            "summary":          a.get("summary", ""),
            "overall_sentiment": a.get("overall_sentiment_score", None),
            "ticker_sentiment":  ticker_sentiment,
            "ticker":           ticker,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Parse AV timestamp format: "20200115T143000"
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%S", errors="coerce")
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def fetch_headlines(ticker: str, date_from: str, date_to: str) -> pd.DataFrame:
    """
    Fetch all headlines for a ticker between two dates.
    Loops month by month — each month = 1 API call (free tier: 25/day).
    """
    start   = pd.to_datetime(date_from)
    end     = pd.to_datetime(date_to)
    current = start
    all_frames = []

    print(f"\n  Fetching {ticker} headlines {date_from} → {date_to}...")
    print(f"  (Alpha Vantage free tier: 25 req/day — fetching month by month)")

    while current <= end:
        year, month = current.year, current.month

        # Build AV timestamp strings
        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(seconds=1)

        tf = month_start.strftime("%Y%m%dT%H%M")
        tt = month_end.strftime("%Y%m%dT%H%M")

        articles = fetch_batch(ticker, tf, tt)
        df = parse_articles(articles, ticker)

        if not df.empty:
            all_frames.append(df)
            print(f"  {year}-{month:02d}: {len(df)} headlines")
        else:
            print(f"  {year}-{month:02d}: 0 headlines")

        # Advance to next month
        if month == 12:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, month + 1, 1)

        time.sleep(12)  # ~5 req/min to stay safe on free tier

    if not all_frames:
        print(f"  No headlines found for {ticker}")
        return pd.DataFrame()

    combined = pd.concat(all_frames).drop_duplicates(subset=["timestamp", "headline"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def save_headlines(ticker: str, df: pd.DataFrame, suffix: str = "") -> Path:
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{ticker}_headlines{'_' + suffix if suffix else ''}.csv"
    path  = NEWS_DIR / fname
    df.to_csv(path, index=False)
    print(f"\n  Saved {len(df)} headlines → {path}")
    return path


def load_headlines(ticker: str, suffix: str = "") -> pd.DataFrame:
    """Load saved headlines for a ticker."""
    fname = f"{ticker}_headlines{'_' + suffix if suffix else ''}.csv"
    path  = NEWS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"No headlines file at {path}. Run news.py first.")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical headlines via Alpha Vantage")
    parser.add_argument("--ticker",    type=str,  default=None,  help="Single ticker")
    parser.add_argument("--tickers",   nargs="+", default=None,  help="Multiple tickers")
    parser.add_argument("--date-from", type=str,  required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--date-to",   type=str,  required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    tickers = args.tickers or ([args.ticker] if args.ticker else [])
    if not tickers:
        print("Error: provide --ticker or --tickers")
        exit(1)

    suffix = f"{args.date_from}_{args.date_to}"

    for ticker in tickers:
        df = fetch_headlines(ticker, args.date_from, args.date_to)
        if not df.empty:
            save_headlines(ticker, df, suffix=suffix)
            print(f"\n  {ticker} summary:")
            print(f"  Total headlines:  {len(df)}")
            print(f"  Date range:       {df['date'].min()} → {df['date'].max()}")
            print(f"  Avg daily:        {len(df) / df['date'].nunique():.1f}")
            print(f"  Sources (top 3):  {df['source'].value_counts().head(3).to_dict()}")
            print(f"\n  Sample headlines:")
            for _, row in df.head(3).iterrows():
                print(f"  [{row['date']} {row['time']}] {row['headline'][:80]}")