import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from feature_engineering import create_features


MODEL_PATH = "../models/model.pkl"


def load_data():

    df = pd.read_csv("../data/sensor_data.csv")

    df = create_features(df)

    X = df.drop(columns=["label"])

    y = df["label"]

    return X, y


if __name__ == "__main__":

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)

    print(f"\n✅ Model saved to {MODEL_PATH}")
