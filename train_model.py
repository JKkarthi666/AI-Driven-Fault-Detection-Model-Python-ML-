"""Train a model for anomaly detection. Produces model.pkl and a metrics report."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




def load_data(path='sensor_data.csv'):
df = pd.read_csv(path)
X = df[['power', 'temp']].values
y = df['label'].values
return X, y




if __name__ == '__main__':
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


pipe = Pipeline([
('scaler', StandardScaler()),
('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])


pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)


print('Accuracy:', accuracy_score(y_test, preds))
print(classification_report(y_test, preds))


joblib.dump(pipe, 'model.pkl')
print('Saved model to model.pkl')