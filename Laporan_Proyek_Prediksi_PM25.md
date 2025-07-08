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

---

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi konsentrasi PM2.5 harian di Jakarta secara akurat?  
2. Apakah model berbasis data historis dapat digunakan untuk membantu masyarakat menghindari polusi berlebih?  
3. Sejauh apa akurasi model ARIMA untuk prediksi PM2.5?

### Goals

1. Membangun model forecasting untuk memprediksi nilai PM2.5 harian berdasarkan data sebelumnya.  
2. Mengevaluasi kinerja model menggunakan metrik MAE dan RMSE.  
3. Menyediakan hasil prediksi dalam bentuk grafik dan tabel untuk pemantauan.  
4. Memberikan peringatan dini dalam bentuk visualisasi grafik jika kadar PM2.5 melebihi ambang batas aman agar masyarakat dapat merencanakan aktivitas luar ruangannya.

### Solution Statements

Kami menggunakan pendekatan time series dengan model ARIMA karena:
- Efisien untuk data univariat (PM2.5)
- Tidak memerlukan banyak fitur eksternal
- Cocok untuk dataset terbatas

Alternatif solusi:
- SARIMA → jika ada seasonality kuat
- LSTM (Deep Learning) → untuk skala besar dan variabel banyak

Namun, dalam proyek ini difokuskan pada ARIMA sebagai baseline untuk efisiensi dan interpretabilitas.

---

## Data Understanding

### Sumber Dataset

Dataset digunakan dari Kaggle:  
[https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021?select=ispu_dki1.csv](https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021?select=ispu_dki1.csv)

### Pemeriksaan Struktur Data

```python
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### Hasil Pemeriksaan:

- Jumlah total baris: 5173  
- Jumlah kolom: 11  
- Nilai kosong (`NaN`) pada kolom `pm25`: 4027  
- Data duplikat: 0  

### Uraian Fitur Dataset

| Fitur     | Deskripsi |
|-----------|-----------|
| `tanggal` | Tanggal pengukuran kualitas udara |
| `pm25`    | Kadar partikel PM2.5 |
| `pm10`    | Kadar partikel PM10 |
| `so2`     | Kandungan sulfur dioksida |
| `co`      | Kandungan karbon monoksida |
| `o3`      | Kandungan ozon |
| `no2`     | Kandungan nitrogen dioksida |
| `kategori` | Kategori kualitas udara berdasarkan ISPU |
| `stasiun` | Nama stasiun pengukuran (misal: DKI1 - Bundaran HI) |

---

## Data Preparation

Langkah-langkah preprocessing:

1. Parsing kolom `tanggal` ke format `datetime`
2. Set `tanggal` sebagai index untuk analisis time-series
3. Hapus nilai `pm25` yang kosong (NaN)
4. Resampling data ke format harian menggunakan rata-rata harian
5. Tambahkan fitur musiman:
   - `dayofweek`: hari dalam minggu
   - `month`: bulan
   - `is_weekend`: akhir pekan atau bukan

### Filter Data Periode

Data difokuskan pada rentang 1 Januari 2023 hingga 28 Februari 2025.  
Alasannya:
- Data sebelum 2023 mengandung banyak nilai kosong, terutama pada tahun 2022
- Data terbaru cenderung lebih relevan terhadap kondisi kualitas udara pasca pandemi dan regulasi emisi baru

### Splitting Data

```python
train_size = int(len(pm25_series) * 0.8)
train = pm25_series[:train_size]
test = pm25_series[train_size:]
```

- Train: 80%  
- Test: 20%  
- Tidak dilakukan pengacakan agar urutan time series tetap terjaga.

---

## Modeling

### Model: ARIMA (AutoRegressive Integrated Moving Average)

Model ARIMA terdiri dari tiga komponen utama:
- **AR (AutoRegressive)**: menggunakan data masa lalu (lag) sebagai input untuk prediksi masa depan
- **I (Integrated)**: melakukan differencing agar data menjadi stasioner
- **MA (Moving Average)**: memodelkan error dari model sebelumnya

### Parameter ARIMA(5,1,0)

- `p = 5`: Menggunakan 5 nilai lag sebelumnya
- `d = 1`: Differencing satu kali agar data stasioner
- `q = 0`: Tidak menggunakan rata-rata kesalahan sebelumnya

Parameter ini dipilih karena hasil eksperimen menunjukkan kombinasi ini memberikan hasil prediksi yang paling stabil.

```python
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
```

---

## Evaluation

### Evaluasi Model

Evaluasi menggunakan dua metrik:
- **MAE (Mean Absolute Error)**: rata-rata kesalahan absolut
- **RMSE (Root Mean Squared Error)**: penalti lebih tinggi untuk error besar

### Hasil Evaluasi

- **MAE**: 28.31  
- **RMSE**: 33.36

### Formula RMSE:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

Keterangan:
- \( y_i \): nilai aktual  
- \( \hat{y}_i \): nilai prediksi  
- \( n \): jumlah data

### Visualisasi Hasil

Grafik prediksi vs aktual menunjukkan bahwa model mengikuti tren secara cukup akurat:

```python
plt.plot(test, label='Aktual')
plt.plot(forecast, label='Prediksi')
```

---

## Penutup

Model ARIMA(5,1,0) memberikan performa yang cukup baik untuk memprediksi kadar PM2.5 harian di Jakarta. Hasil prediksi ini divisualisasikan dalam bentuk grafik yang bisa digunakan untuk sistem peringatan dini jika kadar PM2.5 mendekati atau melebihi ambang batas aman.

Prediksi ini diharapkan dapat:
- Membantu masyarakat menghindari aktivitas luar saat polusi tinggi
- Menjadi dasar pertimbangan pemerintah untuk membuat kebijakan lingkungan

Dengan hasil evaluasi yang cukup baik dan proses yang transparan, model ini dapat digunakan sebagai baseline yang kuat dalam pemantauan kualitas udara.
