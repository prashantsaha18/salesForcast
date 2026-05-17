"""
model.py — Core Sales Forecasting Logic
Uses statsmodels ARIMA (no C++ build required — works on all platforms).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


def preprocess(df, date_col, sales_col):
    df = df[[date_col, sales_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.dropna()
    df = df.drop_duplicates(subset=[date_col])
    return df


def add_time_features(df, date_col):
    df = df.copy()
    df["year"]      = df[date_col].dt.year
    df["month"]     = df[date_col].dt.month
    df["quarter"]   = df[date_col].dt.quarter
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["dayofyear"] = df[date_col].dt.dayofyear
    return df


def run_prophet(df, date_col, sales_col, periods=90, freq="MS",
                seasonality_mode="multiplicative", yearly_seasonality=True,
                weekly_seasonality=True, changepoint_prior_scale=0.05):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = df.set_index(date_col)[sales_col].asfreq(freq)
    series = series.interpolate()

    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,0,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
    except Exception:
        model = SARIMAX(series, order=(1,1,1),
                        enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)

    freq_map = {"D": periods, "W": max(1, periods//7), "MS": max(1, periods//30)}
    steps = freq_map.get(freq, max(1, periods//30))

    forecast_obj = result.get_forecast(steps=steps)
    pred_mean    = forecast_obj.predicted_mean
    conf_int     = forecast_obj.conf_int(alpha=0.2)
    fitted       = result.fittedvalues

    hist_df = pd.DataFrame({
        "ds": series.index, "yhat": fitted.values,
        "yhat_lower": fitted.values*0.95, "yhat_upper": fitted.values*1.05,
    })
    fc_df = pd.DataFrame({
        "ds": pred_mean.index, "yhat": pred_mean.values,
        "yhat_lower": conf_int.iloc[:,0].values, "yhat_upper": conf_int.iloc[:,1].values,
    })

    forecast = pd.concat([hist_df, fc_df], ignore_index=True)
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    forecast["trend"] = np.linspace(series.iloc[0], pred_mean.iloc[-1], len(forecast))
    return result, forecast


def evaluate(df, forecast, date_col, sales_col):
    actuals = df[[date_col, sales_col]].rename(columns={date_col:"ds", sales_col:"y"})
    actuals["ds"] = pd.to_datetime(actuals["ds"])
    merged = actuals.merge(forecast[["ds","yhat"]], on="ds", how="inner")
    if merged.empty:
        return {"MAE": None, "RMSE": None, "MAPE": None}
    y_true, y_pred = merged["y"].values, merged["yhat"].values
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true-y_pred)/np.where(y_true==0,1,y_true)))*100
    return {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mape,2)}


def get_forecast_summary(forecast, periods):
    cols = ["ds","yhat","yhat_lower","yhat_upper"]
    summary = forecast[cols].tail(periods).copy()
    summary.columns = ["Date","Forecast","Lower Bound","Upper Bound"]
    summary["Date"] = pd.to_datetime(summary["Date"]).dt.strftime("%Y-%m-%d")
    summary[["Forecast","Lower Bound","Upper Bound"]] = summary[["Forecast","Lower Bound","Upper Bound"]].round(2)
    return summary.reset_index(drop=True)