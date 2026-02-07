# Exploratory Data Analysis Report

**Generated:** 2025-12-02
**Analysis Period:** 2025-11-19 to 2025-12-02 (13 days)

## Executive Summary

This report presents the exploratory data analysis (EDA) of telemetry data collected from demo services running in Kubernetes. The dataset contains **17,915 samples** spanning **312 hours** (13 days) with metrics collected every minute.

### Key Findings

1. **Data Quality:** No missing values, clean dataset ready for ML training
2. **Temporal Patterns:** Clear hourly and weekly seasonality detected
3. **Correlations:** Negative correlation between request rate and latency (expected behavior)
4. **Issue Identified:** `active_jobs` metric is consistently 0 (requires investigation)

---

## Dataset Overview

### Dataset Composition

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 12,540 | 70% |
| Validation | 2,687 | 15% |
| Test | 2,688 | 15% |
| **Total** | **17,915** | **100%** |

### Time Range

- **Start:** 2025-11-19 01:39:00 UTC
- **End:** 2025-12-02 02:01:00 UTC
- **Duration:** 312.4 hours (13 days)
- **Sampling Frequency:** 1 minute

### Feature Engineering

The preprocessor created **48 features** including:
- **Raw metrics:** request_rate, latency_p50/p95/p99, active_jobs
- **Temporal features:** hour, day_of_week, is_weekend, minute_of_day
- **Lag features:** lag_1, lag_5, lag_15, lag_30 for all metrics
- **Rolling statistics:** rolling_mean_5/15/30 for all metrics
- **Target variables:** target_request_rate_t+5/15/30 (forecast horizons)

---

## Statistical Analysis

### Metric Statistics (Normalized Data)

**Note:** Data has been standardized (mean≈0, std≈1) during preprocessing.

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| request_rate | 0.00 | 1.00 | -3.62 | 3.13 | 0.00 |
| latency_p50 | 0.00 | 1.00 | -4.79 | 5.08 | 0.17 |
| latency_p95 | 0.00 | 1.00 | -6.00 | 5.51 | 0.28 |
| latency_p99 | 0.00 | 1.00 | -5.01 | 1.92 | 0.32 |
| active_jobs | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Missing Values

**Zero missing values** across all features - excellent data quality!

---

## Temporal Patterns

### Hourly Patterns

Request rate shows clear **daily seasonality**:

- **Peak hours:** 5-7 AM UTC (morning traffic)
- **Low hours:** 11 PM - 2 AM UTC (night-time)
- **Standard deviation:** Higher variance during peak hours

### Weekly Patterns

Day-of-week analysis reveals:

| Day | Avg Request Rate (normalized) |
|-----|-------------------------------|
| Monday | 0.05 |
| Tuesday | -0.32 |
| Wednesday | 0.44 |
| Thursday | 0.17 |
| Friday | -0.13 |
| Saturday | -0.44 |
| Sunday | 0.12 |

**Observations:**
- **Wednesday** has highest traffic
- **Saturday** has lowest traffic
- Clear weekly seasonality pattern

### Weekend vs Weekday

| Period | Request Rate | Latency P95 |
|--------|--------------|-------------|
| Weekday | 0.077 | 0.060 |
| Weekend | -0.178 | -0.138 |

**Lower traffic on weekends** with correspondingly lower latency.

---

## Correlation Analysis

### Key Correlations

| Metric Pair | Correlation | Interpretation |
|-------------|-------------|----------------|
| request_rate ↔ latency_p50 | **-0.26** | Moderate negative correlation |
| request_rate ↔ latency_p95 | **-0.17** | Weak negative correlation |
| latency_p50 ↔ latency_p95 | **+0.80** | Strong positive correlation |
| latency_p95 ↔ latency_p99 | **+0.94** | Very strong correlation |

---

## Visualizations

Generated visualizations available in `docs/eda_figures/`:

1. **timeseries.png** - Time series plots of key metrics
2. **hourly_patterns.png** - Hourly aggregations and distributions
3. **weekly_patterns.png** - Day-of-week comparisons
4. **correlation_heatmap.png** - Correlation matrix heatmap

---

## Issues & Recommendations

### Critical Issue: Active Jobs

**Problem:** The `active_jobs` metric is consistently 0 across all samples.

**Recommendation:**
- May need to exclude this feature from ML models
- Consider implementing actual background job processing

### ML Model Readiness

**Ready for training:**
- Train/val/test splits properly created (70/15/15)
- Features engineered (lag, rolling stats, temporal)
- Data normalized/scaled
- Target variables defined (t+5, t+15, t+30 forecasts)

**Recommendations for modeling:**
1. **Prophet:** Good candidate due to strong seasonality
2. **LSTM/GRU:** Leverage sequence data
3. **Feature selection:** Consider dropping `active_jobs` due to zero variance
4. **Forecast horizons:** Prioritize t+5 and t+15 (shorter = better accuracy)

---

## Next Steps

Based on this EDA, proceed with **Phase 2**:

1. Research best practices for time series forecasting
2. Implement Prophet model - Leverage seasonality patterns
3. Implement LSTM model - Use sequence data
4. Model comparison - Evaluate RMSE, MAE, MAPE metrics
5. Hyperparameter tuning - Optimize best performing model
