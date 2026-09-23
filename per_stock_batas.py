# per_stock_betas.py
# -----------------------------------------------------------------------------
# STEP THREE of the analysis pipeline (this is H1): for EACH stock, run one
# time-series regression of its daily return on a constant, the SPX return, and
# the orthogonalized vol shock (dvix_orth). Each regression hands back three
# numbers -- alpha, market beta, vol-shock beta -- which we collect into a table.
# H1 asks one thing: is the vol-shock beta negative (and significant) for most
# stocks? It almost certainly is. This is a sanity check that the pipeline works,
# NOT a headline result, so we don't oversell it.
# -----------------------------------------------------------------------------

import pandas as pd # for working with tables of data
import statsmodels.api as sm # for running the regressions

# --- load the panel (now carrying dvix_orth from step two)
panel = pd.read_csv('panel.csv', parse_dates=['datetime']) # run the table, treat ' datetime ' as real dates
panel = panel.set_index('datetime') # use the date colum as the row labels (so we can line up by date later)

# --- figure out which columns are the stock return columns ------------------
# every stock column was named like 'AAPL_ret', so we keep all '_ret' columns
# except 'spx_ret' (that's a factor, not a stock we're explaining)

stock_cols = [c for c in panel.columns if c.endswith('ret') and c != 'spx_ret']

# --- the two factors are the SAME for every regression ----------------------
# right-hand side: a constant (gives us alpha) + market return + pure vol shock
X = sm.add_constant(panel[['spx_ret', 'dvix_orth']])

# ---- run one regression per stock, collect the results into a table ----------------
rows = [] # empty list, well add one row of results per stock
for col in stock_cols: # loop through every stock return column, one at a time
    y = panel[col] # this stock's daily returns are the thing we're trying to explain

 # fit the regression; the cov_type stuff just makes the t-stats more
    # trustworthy for daily data (fancy standard errors, doesn't change the betas)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kws={'maxlags': 5})
    
    rows.append({                                      # save the numbers we care about for this stock
        'stock':      col.replace('_ret', ''),         # clean name, e.g. 'AAPL_ret' -> 'AAPL'
        'alpha':      model.params['const'],           # average daily return the two factors can't explain
        'mkt_beta':   model.params['spx_ret'],         # sensitivity to the market
        'vol_beta':   model.params['dvix_orth'],       # THE number for H1: sensitivity to a pure vol shock
        'vol_tstat':  model.tvalues['dvix_orth'],      # is vol_beta real or just noise? past ~2 in size = real
        'r2':         model.rsquared,                  # how much of this stock's movement the factors explain
    })
    
# --- assemble the results table and sort by vol sensitivity ----
results = pd.DataFrame(rows) # turns the list for results into a table
results = results.sort_values('vol_beta') # most negative vol beta as the top


# --- show, rounded so its readable ---
print(results.round(4).to_string(index=False)) # print the table, no row numbers

# --- a one-line summary that IS the H1 answer ----
