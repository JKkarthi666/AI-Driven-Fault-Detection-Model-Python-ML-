import joblib
import pandas as pd

from feature_engineering import create_features


class FaultDetector:

    def __init__(self, model_path="../models/model.pkl"):

        self.pipeline = joblib.load(model_path)

    def preprocess(self, samples):

        df = pd.DataFrame(
            samples,
            columns=["power", "temp", "voltage"]
        )

        df = create_features(df)

        return df

    def predict(self, samples):

        df = self.preprocess(samples)

        return self.pipeline.predict(df)

    def predict_proba(self, samples):

        df = self.preprocess(samples)

        return self.pipeline.predict_proba(df)[:, 1]
