import pandas as pd
import numpy as np

def riskless_profit(cur_back, cur_lay, future_back, future_lay, forecast_back, forecast_lay, commisions = 0.05):
    p = np.nan

    forecast_p1 = (forecast_back - cur_lay + commisions * (1 - forecast_back)) / \
                  (forecast_back * cur_lay - forecast_back + cur_lay - commisions)
    forecast_p2 = (cur_back - forecast_lay + commisions * (1 - cur_back)) / \
                  (cur_back * forecast_lay - cur_back + forecast_lay - commisions)

    if forecast_p1 > forecast_p2 and forecast_p1 > 0:
        p = (future_back - cur_lay + commisions * (1 - future_back)) / \
            (future_back * cur_lay - future_back + cur_lay - commisions)
    elif forecast_p2 > forecast_p1 and forecast_p2 > 0:
        p = (cur_back - future_lay + commisions * (1 - cur_back)) / \
            (cur_back * future_lay - cur_back + future_lay - commisions)

    return p

def profits_summary(x, perfect_riskless_profit):
    x = np.array(x)
    s = {}

    x_valid = x[~np.isnan(x)]
    prp_valid = perfect_riskless_profit[~np.isnan(perfect_riskless_profit)]

    s["Min"] = np.min(x_valid) if len(x_valid) > 0 else np.nan
    s["1st Qu."] = np.percentile(x_valid, 25) if len(x_valid) > 0 else np.nan
    s["Median"] = np.median(x_valid) if len(x_valid) > 0 else np.nan
    s["Mean"] = np.mean(x_valid) if len(x_valid) > 0 else np.nan
    s["3rd Qu."] = np.percentile(x_valid, 75) if len(x_valid) > 0 else np.nan
    s["Max"] = np.max(x_valid) if len(x_valid) > 0 else np.nan
    s["Betted Number"] = len(x_valid)
    s["Betting Rate in Total Bets"] = len(x_valid) / len(x)
    s["Betting Rate in Profitable Bets"] = len(x_valid) / len(prp_valid) if len(prp_valid) > 0 else np.nan
    s["Profit Sum"] = np.sum(x_valid)
    s["Profit Sum/Total Possible Profit"] = np.sum(x_valid) / np.sum(prp_valid) if np.sum(prp_valid) != 0 else np.nan
    s["Positive Rate in Betted Race"] = np.sum(x_valid > 0) / len(x_valid) if len(x_valid) > 0 else np.nan
    s["Lower Bound Confidence Interval (ROI)"] = s["Mean"] - 1.96 * np.std(x_valid) / np.sqrt(len(x_valid))

    return pd.Series(s)