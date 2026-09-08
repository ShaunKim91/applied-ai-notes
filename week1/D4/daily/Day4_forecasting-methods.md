# Day 4 — Forecasting Methods and Evaluation

Yesterday we decomposed a series into trend, seasonality, and residual. Today: use that structure to predict what happens next, and measure how wrong we are.

## From naive to seasonal: a ladder of methods

- **Moving-average forecast** — predict tomorrow as the average of the last N days. No concept of trend or seasonality; it lags behind both.
- **Simple Exponential Smoothing (SES)** — a weighted average where recent points count more than old ones, controlled by a smoothing parameter `alpha`. Still flat — it can't extrapolate a trend or repeat a seasonal pattern.
- **Holt's method** — adds a second smoothed component for trend, so the forecast can slope upward or downward instead of staying flat.
- **Holt-Winters** — adds a third smoothed component for seasonality (additive or multiplicative), so the forecast repeats a weekly/monthly/yearly pattern on top of the trend.

Each method is a strict superset of the one before it — Holt-Winters with seasonality "turned off" collapses back to Holt, and Holt with trend "turned off" collapses back to SES.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

train, test = foot_traffic[:-60], foot_traffic[-60:]  # last 60 days held out

model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=7,
).fit()

forecast = model.forecast(len(test))
```

## Splitting time-respecting data

Never use a random train/test split on a time series — shuffling lets the model "see the future" through leaked nearby points and wildly overstates accuracy. Always split by cutting a chronological point: everything before goes to train, everything after to test, exactly as in the snippet above.

## A range, not a point

A single forecasted number implies false precision. Because Holt-Winters is a statistical model with an estimated error distribution, you can **simulate** many plausible future paths (by repeatedly sampling noise consistent with the fitted residuals) and report a percentile range instead of one line:

```python
simulations = model.simulate(len(test), repetitions=200, error="add")
lower = simulations.quantile(0.05, axis=1)
upper = simulations.quantile(0.95, axis=1)
```

That band communicates "here's the plausible range," which is usually more honest than a single confident-looking line.

## Stationarity and differencing

Many classical forecasting techniques (ARIMA in particular) assume the series is **stationary** — its mean and variance don't drift over time. The **Augmented Dickey-Fuller (ADF) test** checks this: a low p-value (conventionally < 0.05) rejects the "non-stationary" null hypothesis.

```python
from statsmodels.tsa.stattools import adfuller

stat, p_value, *_ = adfuller(foot_traffic)
```

If the series isn't stationary, **differencing** — replacing each value with the difference from the previous one (`foot_traffic.diff()`) — often removes a trend and can make the series stationary enough to model.

## Scoring the methods

Compare forecasts against the held-out test set with a few standard metrics:
- **MAE** (mean absolute error) — average magnitude of error, in the original units.
- **MAPE** (mean absolute percentage error) — same idea scaled to a percentage, useful for comparing across series with different scales (but unstable near zero).
- **RMSE** (root mean squared error) — like MAE but penalizes large errors more heavily.

```python
import numpy as np

def mae(y, yhat): return np.mean(np.abs(y - yhat))
def mape(y, yhat): return np.mean(np.abs((y - yhat) / y)) * 100
def rmse(y, yhat): return np.sqrt(np.mean((y - yhat) ** 2))
```

Run all three across moving-average, SES, Holt, and Holt-Winters forecasts on the same held-out window, and — for any series with real seasonality — Holt-Winters should come out ahead, because it's the only method actually modeling the pattern the others treat as noise.

**Takeaway:** pick the simplest method that captures the structure actually present in your data — but always evaluate on a chronological holdout, and prefer a prediction interval over a single "best guess" number.
