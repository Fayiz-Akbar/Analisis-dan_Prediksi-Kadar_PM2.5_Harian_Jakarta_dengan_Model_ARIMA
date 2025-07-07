import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('ispu_dki1.csv')

df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')

print("Tanggal paling awal:", df['tanggal'].min())
print("Tanggal paling akhir:", df['tanggal'].max())

df_pm25 = df[['tanggal', 'pm25']].copy()
df_pm25.dropna(inplace=True)

df_pm25.set_index('tanggal', inplace=True)
df_pm25 = df_pm25.resample('D').mean()

data_pm25 = df_pm25.loc['2023-01-01':'2025-02-28']
print("Jumlah data:", len(data_pm25))

data_fe = data_pm25.copy()
data_fe['dayofweek'] = data_fe.index.dayofweek     
data_fe['month'] = data_fe.index.month
data_fe['is_weekend'] = data_fe['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

print(data_fe.head())

plt.figure(figsize=(12,4))
data_fe.groupby('dayofweek')['pm25'].mean().plot(kind='bar', color='skyblue')
plt.title('Rata-rata PM2.5 per Hari dalam Seminggu')
plt.xlabel('0 = Senin ... 6 = Minggu')
plt.ylabel('PM2.5 (µg/m³)')
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
data_fe.groupby('month')['pm25'].mean().plot(kind='bar', color='orange')
plt.title('Rata-rata PM2.5 per Bulan')
plt.xlabel('Bulan')
plt.ylabel('PM2.5 (µg/m³)')
plt.grid(True)
plt.show()

pm25_series = data_fe['pm25']

train_size = int(len(pm25_series) * 0.8)
train = pm25_series[:train_size]
test = pm25_series[train_size:]

print("Train:", len(train), "Test:", len(test))

model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index

df_eval = pd.DataFrame({
    'actual': test,
    'predicted': forecast
}).dropna()

mae = mean_absolute_error(df_eval['actual'], df_eval['predicted'])
rmse = np.sqrt(mean_squared_error(df_eval['actual'], df_eval['predicted']))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

plt.figure(figsize=(14,5))
plt.plot(test, label='Aktual')
plt.plot(forecast, label='Prediksi', color='red')
plt.title('Hasil Prediksi vs Aktual PM2.5')
plt.xlabel('Tanggal')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

future_forecast = model_fit.forecast(steps=30)
start_date = data_pm25.index[-1] + pd.Timedelta(days=1)
future_dates = pd.date_range(start=start_date, periods=30)

# Membat DataFrame prediksi
future_df = pd.DataFrame({
    'Tanggal': future_dates,
    'Prediksi_PM25': future_forecast
})

print("Prediksi PM2.5 30 Hari ke Depan:")
print(future_df)
