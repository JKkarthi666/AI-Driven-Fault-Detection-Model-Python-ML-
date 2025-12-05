"""Simple streaming inference example that watches a CSV and flags anomalies."""
import time
import pandas as pd
from model import FaultDetector




def stream_check(filepath='sensor_data.csv', batch_size=100):
fd = FaultDetector()
df = pd.read_csv(filepath)
for start in range(0, len(df), batch_size):
batch = df.iloc[start:start+batch_size]
X = batch[['power', 'temp']].values
preds = fd.predict(X)
where = batch.index[preds == 1].tolist()
if where:
print('Anomalies at rows:', where)
time.sleep(0.2)




if __name__ == '__main__':
stream_check()