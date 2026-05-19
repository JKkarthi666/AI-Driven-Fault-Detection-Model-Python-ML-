import time
import pandas as pd

from model import FaultDetector


def monitor_stream(
    filepath="../data/sensor_data.csv",
    batch_size=100
):

    detector = FaultDetector()

    df = pd.read_csv(filepath)

    for start in range(0, len(df), batch_size):

        batch = df.iloc[start:start + batch_size]

        X = batch[
            ["power", "temp", "voltage"]
        ].values

        preds = detector.predict(X)

        probs = detector.predict_proba(X)

        anomaly_rows = batch.index[preds == 1].tolist()

        if anomaly_rows:

            print("\n🚨 ANOMALIES DETECTED")

            for row, prob in zip(
                anomaly_rows,
                probs[preds == 1]
            ):

                print(
                    f"Row: {row} | "
                    f"Confidence: {prob:.2f}"
                )

        time.sleep(0.2)


if __name__ == "__main__":

    monitor_stream()