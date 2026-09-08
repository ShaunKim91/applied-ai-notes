# Day 3 — Time Series Fundamentals

A time series isn't just a list of numbers — it's a list of numbers where *order carries information*. Shuffle a customer-churn table and nothing breaks; shuffle a sales-by-day series and you've destroyed the thing you were trying to model.

## Order matters: autocorrelation

**Autocorrelation** measures how correlated a series is with a lagged copy of itself — e.g. today's sales vs. sales 7 days ago. A strong autocorrelation at lag 7 in daily retail data is a dead giveaway of weekly seasonality (weekend spikes, say). This is worth checking before modeling any series at all: is there structure to exploit, and at what lag?

## Frequency and resampling

Every series has an implicit sampling frequency — hourly sensor reads, daily sales, monthly revenue. **Resampling** changes that frequency after the fact: downsampling (hourly → daily, usually via `.mean()` or `.sum()`) smooths out noise and cuts data volume; upsampling (daily → hourly) requires inventing values via interpolation, which manufactures information that wasn't actually measured. `pandas`' `.resample()` handles both directions.

## The four components

Most series can be thought of as a combination of:
- **Trend** — the long-run direction (a coffee shop's slow year-over-year growth).
- **Seasonality** — a fixed, calendar-driven repeating pattern (weekday vs. weekend traffic).
- **Cycle** — a repeating-but-not-fixed-period pattern, often tied to external conditions — easy to confuse with seasonality, but without a fixed calendar period.
- **Residual (noise)** — whatever's left after removing the above.

Whether components combine **additively** (`observed = trend + seasonal + residual`, appropriate when seasonal swings stay roughly constant in absolute size) or **multiplicatively** (`observed = trend * seasonal * residual`, appropriate when swings grow proportionally with the trend) changes how you decompose the series.

## Decomposing with statsmodels

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

rng = pd.date_range("2023-01-01", periods=730, freq="D")
trend = np.linspace(80, 160, len(rng))                     # slow growth
weekly = 15 * np.sin(2 * np.pi * rng.dayofweek / 7)         # weekly seasonality
noise = np.random.normal(0, 4, len(rng))
foot_traffic = pd.Series(trend + weekly + noise, index=rng)

result = seasonal_decompose(foot_traffic, model="additive", period=7)
# result.trend, result.seasonal, result.resid are each their own Series
```

## Smoothing and anomaly detection

A **moving average** replaces each point with the mean of a surrounding window, damping short-term noise so the trend is easier to see (`foot_traffic.rolling(7).mean()`).

A common cheap anomaly detector flags any point more than `k` standard deviations from a rolling mean:

```python
roll_mean = foot_traffic.rolling(14).mean()
roll_std = foot_traffic.rolling(14).std()
anomaly = (foot_traffic - roll_mean).abs() > 3 * roll_std
```

The catch: run this on the **raw** series and every normal Saturday spike looks like an anomaly, because the rolling window mixes weekday and weekend values into one baseline. Run it on `result.resid` (the deseasonalized residual) instead, and only genuinely unusual days — a surprise closure, a viral social post — get flagged. Anomaly detection without removing seasonality first isn't detecting anomalies; it's just rediscovering the seasonal pattern you already knew about.

**Takeaway:** before forecasting or flagging anything unusual in a time series, decompose it into trend, seasonality, and residual — most "anomalies" caught on raw data are actually just seasonality nobody accounted for yet.
