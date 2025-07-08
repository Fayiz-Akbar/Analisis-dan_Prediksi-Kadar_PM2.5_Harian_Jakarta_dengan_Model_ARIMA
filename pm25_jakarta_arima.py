# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


# Load dataset dari file CSV
df = pd.read_csv('ispu_dki1.csv')

# Tampilkan informasi awal mengenai dataset
print("Tanggal paling awal:", df['tanggal'].min())
print("Tanggal paling akhir:", df['tanggal'].max())

# Menampilkan jumlah baris dan kolom
print("Jumlah baris dan kolom:", df.shape)

# Informasi struktur dataset
print("\nInformasi dataset:")
print(df.info())

# Statistik deskriptif dari dataset
print("\nStatistik deskriptif:")
print(df.describe())

# Mengecek jumlah nilai kosong per kolom
print("\nJumlah nilai kosong per kolom:")
print(df.isnull().sum())

# Mengecek apakah ada data duplikat
print("\nJumlah duplikasi:")
print(df.duplicated().sum())


# Preprocessing data tanggal dan PM2.5
# Mengonversi kolom tanggal menjadi format datetime
df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')

# Menyimpan hanya kolom tanggal dan pm25 untuk analisis time series
df_pm25 = df[['tanggal', 'pm25']].copy()

# Menghapus data yang memiliki nilai kosong
df_pm25.dropna(inplace=True)

# Menjadikan tanggal sebagai index dan resample menjadi data harian
df_pm25.set_index('tanggal', inplace=True)
df_pm25 = df_pm25.resample('D').mean()

# Mengambil subset data dari tahun 2023 hingga awal 2025
data_pm25 = df_pm25.loc['2023-01-01':'2025-02-28']
print("Jumlah data:", len(data_pm25))


# Feature Engineering - Tambahan fitur musiman
data_fe = data_pm25.copy()
data_fe['dayofweek'] = data_fe.index.dayofweek     # 0 = Senin, 6 = Minggu
data_fe['month'] = data_fe.index.month             # Bulan 1-12
data_fe['is_weekend'] = data_fe['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# Tampilkan lima baris awal data hasil feature engineering
print(data_fe.head())


# Visualisasi pola musiman harian dan bulanan
# Rata-rata PM2.5 per hari dalam seminggu
plt.figure(figsize=(12,4))
data_fe.groupby('dayofweek')['pm25'].mean().plot(kind='bar', color='skyblue')
plt.title('Rata-rata PM2.5 per Hari dalam Seminggu')
plt.xlabel('0 = Senin ... 6 = Minggu')
plt.ylabel('PM2.5 (µg/m³)')
plt.grid(True)
plt.show()

# Rata-rata PM2.5 per bulan
plt.figure(figsize=(12,4))
data_fe.groupby('month')['pm25'].mean().plot(kind='bar', color='orange')
plt.title('Rata-rata PM2.5 per Bulan')
plt.xlabel('Bulan')
plt.ylabel('PM2.5 (µg/m³)')
plt.grid(True)
plt.show()

# Persiapan data untuk modeling
# Ambil hanya data PM2.5 sebagai time series
pm25_series = data_fe['pm25']

# Split data menjadi train dan test (80% - 20%)
train_size = int(len(pm25_series) * 0.8)
train = pm25_series[:train_size]
test = pm25_series[train_size:]

print("Train:", len(train), "Test:", len(test))


# Modeling menggunakan ARIMA
# Membuat model ARIMA dengan parameter (5,1,0)
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# Ringkasan model
print(model_fit.summary())



# Evaluasi Model dengan Test Set
# Melakukan forecasting sebanyak panjang data test
forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index

# Membuat DataFrame evaluasi
df_eval = pd.DataFrame({
    'actual': test,
    'predicted': forecast
}).dropna()

# Menghitung metrik MAE dan RMSE
mae = mean_absolute_error(df_eval['actual'], df_eval['predicted'])
rmse = np.sqrt(mean_squared_error(df_eval['actual'], df_eval['predicted']))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# Visualisasi hasil prediksi vs data aktual
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


# Prediksi PM2.5 30 Hari ke Depan
# Forecast 30 hari ke depan dari data terakhir
future_forecast = model_fit.forecast(steps=30)

# Menyusun tanggal prediksi masa depan
start_date = data_pm25.index[-1] + pd.Timedelta(days=1)
future_dates = pd.date_range(start=start_date, periods=30)

# Membuat DataFrame hasil prediksi masa depan
future_df = pd.DataFrame({
    'Tanggal': future_dates,
    'Prediksi_PM25': future_forecast
})

print("Prediksi PM2.5 30 Hari ke Depan:")
print(future_df)
