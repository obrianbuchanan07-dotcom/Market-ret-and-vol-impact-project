# STEP TWO of the analysis pipeline: strip the market move out of dVIX.
# We regress dVIX on the SPX return and KEEP THE RESIDUALS. The residual is the
# part of each day's volatility move that is NOT just the mechanical mirror of
# the market move (the leverage effect). That residual, "dvix_orth", becomes the
# second factor in every per-stock regression from here on -- never raw dvix.


import pandas as pd               # for working with tables of data
import statsmodels.api as sm      # for running the regression
import matplotlib.pyplot as plt   # for making the charts


# read the panel we built in step one, treating 'datetime' as real dates
panel = pd.read_csv("panel.csv", parse_dates=['datetime'])

# use the date column as the row labels
panel = panel.set_index('datetime')


# y is the thing we're explaining: daily VIX changes
y = panel['dvix']

# X is the thing doing the explaining: SPX returns, plus a constant (the intercept)
X = sm.add_constant(panel['spx_ret'])

#  fit it and pull out the residuals 
# run the regression: dvix = a + b * spx_ret + leftover
model = sm.OLS(y, X).fit()

# save the leftover part (what SPX could NOT explain) as a new column
panel['dvix_orth'] = model.resid


# should be basically zero, since we just removed the SPX part
print("corr(dvix_orth, spx_ret):", panel['dvix_orth'].corr(panel['spx_ret']))

# should be negative: market down = VIX up (the leverage effect)
print("slope on spx_ret:", model.params['spx_ret'])

# how much of dvix's movement SPX explains (0 to 1)
print("R-squared:", model.rsquared)

# save the panel back to disk, now with the new dvix_orth column
panel.to_csv('panel.csv')
print("Saved panel.csv with dvix_orth")


# set up two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 1. The regression itself: dvix vs spx_ret with the fitted line
axes[0].scatter(panel['spx_ret'], panel['dvix'], s=8, alpha=0.4)  # each dot = one day

# sort the x values so the fitted line draws left to right cleanly
x_line = pd.Series(sorted(panel['spx_ret']))

# draw the red regression line using the fitted intercept and slope
axes[0].plot(x_line, model.params['const'] + model.params['spx_ret'] * x_line,
             color='red', lw=1.5)
axes[0].set_xlabel('SPX return')                # label the x axis
axes[0].set_ylabel('dVIX')                      # label the y axis
axes[0].set_title(f"Leverage effect (slope = {model.params['spx_ret']:.2f})")  # title with the slope number

# 2. The residual over time: should look like noise around zero
axes[1].plot(panel.index, panel['dvix_orth'], lw=0.6)  # thin line of residuals by date
axes[1].axhline(0, color='red', lw=0.8)                # red reference line at zero
axes[1].set_title('dvix_orth over time')

plt.tight_layout()  # stop the two charts from overlapping
plt.show()          # display the window with the charts