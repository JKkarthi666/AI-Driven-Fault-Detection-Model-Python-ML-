"""Generate synthetic power/thermal sensor data with anomalies."""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


RND = 42
np.random.seed(RND)




def generate_windowed_features(n_samples=10000, seq_len=1):
# Basic synthetic features for power and thermal
t = np.arange(n_samples)
power = 50 + 5*np.sin(0.02*t) + np.random.normal(0, 0.5, size=n_samples)
temp = 30 + 2*np.cos(0.015*t) + np.random.normal(0, 0.3, size=n_samples)


df = pd.DataFrame({
'power': power,
'temp': temp,
})


# Insert anomalies
n_anom = int(0.03 * n_samples)
idx = np.random.choice(n_samples, n_anom, replace=False)
df.loc[idx, 'power'] += np.random.normal(20, 5, size=n_anom) # spike
df.loc[idx, 'temp'] += np.random.normal(10, 3, size=n_anom) # heating
df['label'] = 0
df.loc[idx, 'label'] = 1


return shuffle(df)




if __name__ == '__main__':
df = generate_windowed_features(20000)
df.to_csv('sensor_data.csv', index=False)
print('Wrote sensor_data.csv with', len(df), 'rows')