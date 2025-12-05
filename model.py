"""Helpers for loading model and performing prediction."""
import joblib
import numpy as np




class FaultDetector:
def __init__(self, model_path='model.pkl'):
self.pipeline = joblib.load(model_path)


def predict(self, samples):
# samples: array-like shape (n, 2) -> [power, temp]
return self.pipeline.predict(samples)


def predict_proba(self, samples):
if hasattr(self.pipeline, 'predict_proba'):
return self.pipeline.predict_proba(samples)[:, 1]
else:
return None