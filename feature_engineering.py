import pandas as pd


def create_features(df):

    df["power_rolling_mean"] = (
        df["power"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    df["temp_rolling_mean"] = (
        df["temp"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    df["power_diff"] = df["power"].diff().fillna(0)

    df["temp_diff"] = df["temp"].diff().fillna(0)

    return df