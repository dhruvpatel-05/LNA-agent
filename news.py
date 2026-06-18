import os
import requests
import pandas as pd
import time

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

TICKERS = ["AMD", "AAPL", "NVDA", "META", "TSLA"]

headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
}

rows = []
page_token = None
page = 0

while True:
    page += 1

    params = {
        "symbols": ",".join(TICKERS),
        "start": "2024-01-01T00:00:00Z",
        "end": "2025-12-31T23:59:59Z",
        "limit": 50,
        "sort": "asc",
    }

    if page_token:
        params["page_token"] = page_token

    r = requests.get(
        "https://data.alpaca.markets/v1beta1/news",
        headers=headers,
        params=params,
        timeout=30,
    )

    if r.status_code == 429:
        print("Rate limited. Sleeping...")
        time.sleep(15)
        continue

    r.raise_for_status()

    data = r.json()
    articles = data.get("news", [])

    print(f"Page {page} | Articles: {len(articles)}")

    for article in articles:

        timestamp = article.get("created_at")

        for ticker in article.get("symbols", []):

            if ticker not in TICKERS:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "headline": article.get("headline"),
                }
            )

    page_token = data.get("next_page_token")

    if not page_token:
        break

print(f"\nDownloaded {len(rows):,} raw rows")

df = pd.DataFrame(rows)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="coerce"
)

# Hard filter to exactly 2024-2025
df = df[
    (df["timestamp"] >= "2024-01-01")
    & (df["timestamp"] < "2026-01-01")
]

# Remove duplicates
df = df.drop_duplicates(
    subset=[
        "ticker",
        "timestamp",
        "headline"
    ]
)

# Sort
df = df.sort_values(
    ["timestamp", "ticker"]
)

# Convert timestamp to second precision
df["timestamp"] = (
    df["timestamp"]
    .dt.floor("s")
    .astype(str)
)

output_file = "stock_news_2024_2025.csv"

df.to_csv(output_file, index=False)

print("\nDone!")
print(f"Rows: {len(df):,}")
print(f"Saved: {output_file}")
print("\nColumns:")
print(df.columns.tolist())