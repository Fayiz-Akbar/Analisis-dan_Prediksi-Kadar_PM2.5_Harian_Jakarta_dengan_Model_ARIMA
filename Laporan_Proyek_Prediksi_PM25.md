
# Laporan Proyek Machine Learning Prediksi Kadar PM2.5 Harian Jakarta dengan Model ARIMA - Fayiz Akbar Daifullah

## Domain Proyek

Kualitas udara menjadi masalah serius di kota metropolitan seperti Jakarta. Salah satu polutan yang berbahaya adalah PM2.5 (partikulat dengan diameter ≤2.5 mikron) yang dapat menembus sistem pernapasan dan menyebabkan penyakit seperti asma, penyakit paru obstruktif kronis, bahkan kanker paru-paru.

Dengan prediksi PM2.5, kita dapat:
- Membantu masyarakat merencanakan aktivitas luar ruangan
- Memberi sinyal dini kepada pemerintah untuk mengambil kebijakan pembatasan
- Menyediakan informasi berbasis data bagi sistem monitoring kualitas udara

> Referensi:
> - IQAir, “Jakarta Air Quality Index,” 2024. [Online]. Available: https://www.iqair.com/indonesia/jakarta  
> - WHO, “Ambient (outdoor) air pollution,” [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi konsentrasi PM2.5 harian di Jakarta secara akurat?
2. Apakah model berbasis data historis dapat digunakan untuk membantu masyarakat menghindari polusi berlebih?
3. Sejauh apa akurasi model ARIMA untuk prediksi PM2.5?

### Goals

1. Membangun model forecasting untuk memprediksi nilai PM2.5 harian berdasarkan data sebelumnya.
2. Mengevaluasi kinerja model menggunakan metrik MAE dan RMSE.
3. Menyediakan hasil prediksi dalam bentuk grafik dan tabel untuk pemantauan.

### Solution Statements

Kami menggunakan pendekatan time series dengan model ARIMA karena:
- Efisien untuk data univariat (PM2.5)
- Tidak memerlukan banyak fitur eksternal
- Cocok untuk dataset terbatas

Alternatif solusi:
- SARIMA → jika ada seasonality kuat
- LSTM (Deep Learning) → untuk skala besar dan variabel banyak

Namun, dalam proyek ini difokuskan pada ARIMA sebagai baseline untuk efisiensi dan interpretabilitas.

## Data Understanding

Dataset diambil dari Open Data Jakarta: https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021?select=ispu_dki1.csv, khususnya data ISPU dari stasiun DKI1 (Bundaran HI).

### Fitur utama:
- `tanggal` : waktu pengambilan data
- `pm25` : konsentrasi partikel PM2.5
- `pm10`, `so2`, `co`, `o3`, `no2` : data polutan lainnya
- `kategori` : klasifikasi kualitas udara
- `stasiun` : nama lokasi pengukuran

Dataset mencakup data dari tahun 2010–2025, namun karena keterbatasan kelengkapan, data yang digunakan untuk model adalah dari 2023–2025.

### Pemeriksaan Kondisi Data

Untuk memahami kondisi data, dilakukan pemeriksaan struktur dan kualitas data sebagai berikut:

```python
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

Hasil pemeriksaan:
- Jumlah total baris: 4672
- Jumlah kolom: 8
- Nilai kosong (`NaN`) pada kolom `pm25`: 1416 baris
- Duplikasi data: 0 baris
- Outlier: Terdapat nilai PM2.5 tinggi (>150) yang mencerminkan polusi ekstrem. Tidak dihapus karena representatif kondisi nyata Jakarta.

### Data yang digunakan:
- Periode: 1 Januari 2023 – 28 Februari 2025
- Setelah pembersihan data: 790 baris data PM2.5 harian

### Visualisasi dan EDA

Kami melakukan:
- Plot tren waktu PM2.5 dari 2023–2025
- Rata-rata PM2.5 per hari dalam seminggu (dayofweek)
- Rata-rata PM2.5 per bulan (seasonality)
- Statistik deskriptif PM2.5

## Data Preparation

Langkah-langkah utama:
- Mengonversi kolom `tanggal` ke format datetime
- Menjadikan `tanggal` sebagai indeks time-series
- Menghapus data kosong pada kolom `pm25`
- Melakukan resampling ke rata-rata harian
- Menambahkan fitur musiman:
  - `dayofweek`: 0 = Senin, ..., 6 = Minggu
  - `month`: bulan 1–12
  - `is_weekend`: 1 jika Sabtu/Minggu

### Contoh kode:

```python
df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
df_pm25 = df[['tanggal', 'pm25']].dropna()
df_pm25.set_index('tanggal', inplace=True)
df_pm25 = df_pm25.resample('D').mean()

data_fe = df_pm25.copy()
data_fe['dayofweek'] = data_fe.index.dayofweek
data_fe['month'] = data_fe.index.month
data_fe['is_weekend'] = data_fe['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
```

### Data Splitting:

```python
pm25_series = data_fe['pm25']
train_size = int(len(pm25_series) * 0.8)
train = pm25_series[:train_size]
test = pm25_series[train_size:]
```

- Jumlah data latih: 632
- Jumlah data uji: 158

## Modeling

Model yang digunakan adalah **ARIMA(5,1,0)**  
Parameter dipilih berdasarkan pengamatan terhadap grafik ACF dan PACF serta eksperimen.

### Penjelasan ARIMA:

- **AR (AutoRegressive)**: Menggunakan hubungan antara observasi saat ini dengan beberapa observasi sebelumnya (lag).
- **I (Integrated)**: Melibatkan differencing untuk membuat data menjadi stasioner.
- **MA (Moving Average)**: Menggunakan ketergantungan antara observasi dan error residual dari model sebelumnya.

ARIMA cocok untuk time-series univariat seperti PM2.5 karena:
- Tidak membutuhkan variabel eksternal
- Dapat menangkap pola musiman dengan penyesuaian

### Implementasi Model:

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))
```

### Prediksi 30 Hari ke Depan:

```python
future_forecast = model_fit.forecast(steps=30)
```

## Evaluation

Model dievaluasi menggunakan dua metrik:

- **MAE (Mean Absolute Error)**: Mengukur rata-rata kesalahan absolut
- **RMSE (Root Mean Squared Error)**: Mengukur akar rata-rata kesalahan kuadrat

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
```

### Hasil:
- MAE: 28.31
- RMSE: 33.36

### Interpretasi:
- Model memiliki kesalahan rata-rata sebesar ~9.24 µg/m³
- Cukup baik mengingat fluktuasi PM2.5 harian di Jakarta bisa sangat tinggi

### Visualisasi:

Grafik hasil prediksi terhadap data aktual menunjukkan bahwa model mengikuti pola data dengan cukup baik, terutama tren dan lonjakan polusi.

## Kesimpulan

Model ARIMA (5,1,0) berhasil digunakan untuk memprediksi kadar PM2.5 harian di Jakarta dengan akurasi yang cukup baik. Nilai MAE dan RMSE menunjukkan kesalahan yang dapat diterima untuk skenario baseline dan dapat dijadikan dasar dalam sistem peringatan dini kualitas udara.

### Rencana Pengembangan:
- Uji coba model SARIMA jika pola musiman tahunan ditemukan
- Tambahkan variabel eksternal (suhu, kelembaban, volume lalu lintas)
- Uji model deep learning (LSTM/GRU) untuk dataset lebih besar

Model ini cocok digunakan dalam sistem pemantauan kualitas udara yang ringan dan mudah diimplementasikan.
