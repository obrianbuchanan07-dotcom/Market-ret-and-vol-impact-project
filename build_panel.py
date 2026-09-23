# panel.py turns my 11 separate CSVs into ONE aligned table keyed on date
# This does NOT pull data (IB Data script already did that)
# It only reads the CSVs off disk and lines them up on a common set of
# trading days, so that every later regression sees the same rows.

import pandas as pd  # library for working with tables of data
import numpy as np   # library for math stuff

# list of stock tickers we want in the panel
STOCKS = [
    'AAPL', 'MSFT', 'JNJ', 'GOOGL', 'WMT', 'JPM', 'KO', 'NVDA',
]
YEARS = 0.5  # how many years of data the CSV files cover


def load_series(symbol, col_name, kind='log_return'):
    # build the filename, e.g. "AAPL_0.5yr_data.csv"
    fname = f"{symbol}_{YEARS}yr_data.csv"

    # read the CSV into a table, treating the 'datetime' column as real dates
    df = pd.read_csv(fname, parse_dates=['datetime'])

    # use the date column as the row labels (so we can line up by date later)
    df = df.set_index('datetime')

    if kind == 'log_return':
        # grab the 'return' column (daily log returns)
        s = df['return']
    elif kind == 'dvix':
        # grab the 'close' column and take day-to-day changes (today minus yesterday)
        s = df['close'].diff()
    else:
        # if someone passes a kind we don't recognize, stop with an error
        raise ValueError("Unknown kind: choose 'log_return' or 'dvix'")

    # give the column a clear name and hand it back
    return s.rename(col_name)


# load the S&P 500 returns
spx = load_series('SPX', 'spx_ret', kind='log_return')

# load the VIX and turn it into daily changes
dvix = load_series('VIX', 'dvix', kind='dvix')

# load returns for every stock in the STOCKS list, one column each
stock_series = [load_series(
    sym, f"{sym}_ret", kind='log_return') for sym in STOCKS]

# glue all columns side by side, keeping only dates that exist in EVERY file
panel = pd.concat([spx, dvix] + stock_series, axis=1, join='inner')

# drop any rows that still have missing values
panel = panel.dropna()


# quick sanity checks printed to the screen
print("Panel shape (rows, cols):", panel.shape)  # how many rows and columns
print("Date range:", panel.index.min().date(), "to", panel.index.max().date())  # first and last date
print("SPX rows pulled originally:", len(
    pd.read_csv(f'SPX_{YEARS}yr_data.csv')))  # row count before aligning, to see how many got dropped
print("Any NaNs left?:", panel.isna().any().any())  # should print False
print(panel.head())  # show the first 5 rows

# save the finished panel to a new CSV file
panel.to_csv('panel.csv')
print("Saved panel.csv")