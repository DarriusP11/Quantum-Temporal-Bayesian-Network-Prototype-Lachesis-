"""
Builds lachasis_benchmark_prices.csv from Yahoo Finance data.

Tickers: AAPL, MSFT, QQQ, TLT, GLD
Frequency: daily
Fields: date + Adjusted Close
"""

import datetime as dt
import pandas as pd

# You may need to: pip install yfinance
import yfinance as yf

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "QQQ", "TLT", "GLD"]

# How far back you want data (change if you like)
START = "2018-01-01"          # start date
END = dt.date.today().isoformat()   # today

OUT_CSV = "lachesis_benchmark_prices.csv"


def download_prices(tickers, start, end):
    """
    Download adjusted-close daily prices for the given tickers.
    Returns a DataFrame with columns = tickers and index = date.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,   # we will use 'Adj Close'
        progress=True,
    )

    # yfinance returns a multi-index column: (field, ticker)
    # We want just the 'Adj Close' slice.
    adj_close = data["Adj Close"].copy()

    # Ensure the columns are in the same order as TICKERS
    adj_close = adj_close[tickers]

    # Make sure index is a normal Date index (no timezone stuff)
    adj_close.index = pd.to_datetime(adj_close.index).date

    return adj_close


def main():
    print(f"Downloading prices for {TICKERS} from {START} to {END} ...")
    df = download_prices(TICKERS, START, END)

    # Reset index so 'date' becomes a column
    df_reset = df.reset_index()
    df_reset.rename(columns={"index": "date"}, inplace=True)
    df_reset.rename(columns={df_reset.columns[0]: "date"}, inplace=True)

    # Save CSV in the exact format Lachesis expects
    df_reset.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV} with shape {df_reset.shape}")


if __name__ == "__main__":
    main()
