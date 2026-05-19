import numpy as np
import pandas as pd
from sklearn.utils import shuffle

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_sensor_data(n_samples=20000):

    t = np.arange(n_samples)

    power = (
        50
        + 5 * np.sin(0.02 * t)
        + np.random.normal(0, 0.5, size=n_samples)
    )

    temp = (
        30
        + 2 * np.cos(0.015 * t)
        + np.random.normal(0, 0.3, size=n_samples)
    )

    voltage = (
        220
        + np.random.normal(0, 1.5, size=n_samples)
    )

    df = pd.DataFrame({
        "power": power,
        "temp": temp,
        "voltage": voltage
    })

    # Inject anomalies
    n_anomalies = int(0.03 * n_samples)

    anomaly_idx = np.random.choice(
        n_samples,
        n_anomalies,
        replace=False
    )

    df.loc[anomaly_idx, "power"] += np.random.normal(
        20,
        5,
        size=n_anomalies
    )

    df.loc[anomaly_idx, "temp"] += np.random.normal(
        10,
        3,
        size=n_anomalies
    )

    df.loc[anomaly_idx, "voltage"] += np.random.normal(
        15,
        4,
        size=n_anomalies
    )

    df["label"] = 0
    df.loc[anomaly_idx, "label"] = 1

    return shuffle(df, random_state=RANDOM_STATE)


if __name__ == "__main__":

    df = generate_sensor_data()

    df.to_csv("../data/sensor_data.csv", index=False)

    print("✅ sensor_data.csv created")
