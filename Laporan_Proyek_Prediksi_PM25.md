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

### Sumber Dataset
Dataset digunakan dari Kaggle:
[https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021?select=ispu_dki1.csv](https://www.kaggle.com/datasets/senadu34/air-quality-index-in-jakarta-2010-2021?select=ispu_dki1.csv)

### Struktur Dataset

```python
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### Hasil Pemeriksaan:

- **Jumlah total baris:** 4672
- **Jumlah kolom:** 8
- **Nilai kosong (NaN)** pada kolom `pm25`: 1416 baris
- **Data duplikat:** 0
- **Outlier:** Terdapat nilai PM2.5 tinggi (>150 µg/m³), namun tidak dihapus karena representatif terhadap polusi ekstrem di Jakarta.

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

## Data Preparation

### Langkah yang dilakukan:
- Parsing kolom `tanggal` menjadi `datetime`
- Set `tanggal` sebagai indeks time-series
- Hapus baris dengan `pm25` kosong (NaN)
- Resample data ke format harian (`daily mean`)
- Tambahkan fitur musiman:
  - `dayofweek` (Senin–Minggu)
  - `month`
  - `is_weekend`

### Splitting Data:
```python
train_size = int(len(pm25_series) * 0.8)
train = pm25_series[:train_size]
test = pm25_series[train_size:]
```

Data dibagi menjadi:
- **Train**: 80% data
- **Test**: 20% data
- Metode ini menjaga urutan waktu (tidak diacak) agar model ARIMA dapat bekerja optimal.

## Modeling

### Model: ARIMA (AutoRegressive Integrated Moving Average)

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
```

### Penjelasan ARIMA(5,1,0):
- `p = 5`: Menggunakan 5 nilai lag sebelumnya
- `d = 1`: Melakukan differencing 1x untuk membuat data stasioner
- `q = 0`: Tidak menggunakan moving average

### Kelebihan:
- Simpel, cocok untuk dataset kecil
- Bisa diinterpretasi
- Tidak membutuhkan banyak fitur

### Kekurangan:
- Tidak cocok untuk banyak variabel
- Tidak menangkap seasonality (tidak seperti SARIMA)

## Evaluation

### Evaluasi dengan MAE dan RMSE
```python
mae = mean_absolute_error(df_eval['actual'], df_eval['predicted'])
rmse = np.sqrt(mean_squared_error(df_eval['actual'], df_eval['predicted']))
```

### Hasil:
- **MAE**: Mean Absolute Error — rata-rata kesalahan absolut
- **RMSE**: Root Mean Squared Error — penalti lebih besar untuk kesalahan besar
- Nilai yang kecil menunjukkan model prediktif baik

### Visualisasi
Grafik perbandingan antara hasil prediksi vs aktual ditampilkan untuk validasi visual performa model.

## Penutup

Model ARIMA berhasil membentuk baseline yang cukup baik untuk prediksi kadar PM2.5 harian di Jakarta. Dengan akurasi yang memadai, hasil ini bisa digunakan untuk mendukung pengambilan keputusan terkait kesehatan publik dan kebijakan lingkungan.

