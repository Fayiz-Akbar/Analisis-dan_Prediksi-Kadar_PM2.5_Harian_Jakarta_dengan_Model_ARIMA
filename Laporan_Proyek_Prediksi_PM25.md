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

Dataset mencakup data dari tahun 2010–2025, namun karena keterbatasan kelengkapan, data yang digunakan untuk model adalah dari 2023–2025 dengan total 790 data.

### Visualisasi dan EDA

Kami melakukan:
- Plot tren waktu PM2.5 dari 2023–2025
- Visualisasi musiman (pola per bulan dan hari)
- Statistik deskriptif PM2.5

## Data Preparation

Langkah yang dilakukan:
- Parsing kolom `tanggal` menjadi `datetime`
- Set `tanggal` sebagai indeks time-series
- Hapus baris dengan `pm25` kosong (NaN)
- Resample data ke format harian (`daily mean`)
- Tambahkan fitur musiman:
  - `dayofweek` (Senin–Minggu)
  - `month`
  - `is_weekend`

Tujuannya untuk analisis pola musiman dan validasi karakteristik data terhadap waktu.

## Modeling

Model yang digunakan: ARIMA(5,1,0)  
Parameter (p=5, d=1, q=0) dipilih berdasarkan grafik ACF dan PACF.

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))
