# Laporan Proyek Machine Learning - Fika Saputri
### Domain Proyek
Asuransi kesehatan adalah salah satu sektor penting dalam kehidupan modern, terutama dalam upaya masyarakat untuk mengantisipasi risiko finansial dari pengeluaran medis yang tak terduga. Dalam industri ini, penetapan premi asuransi menjadi hal krusial yang harus memperhitungkan berbagai faktor pelanggan, seperti usia, jenis kelamin, status perokok, BMI, dan wilayah tempat tinggal.

Permasalahan muncul ketika perusahaan asuransi harus menentukan premi yang adil namun tetap menguntungkan secara bisnis. Pendekatan tradisional sering kali tidak cukup akurat, sehingga pendekatan berbasis data dan model machine learning menjadi solusi yang sangat potensial.

Menurut studi [1], penggunaan algoritma prediksi seperti regresi dan ensemble learning dapat meningkatkan akurasi dalam menentukan biaya asuransi. Dengan data yang tersedia dari Kaggle Medical Cost Personal Dataset, proyek ini bertujuan membangun model prediksi yang mampu memperkirakan biaya asuransi dengan baik berdasarkan atribut pelanggan.

Referensi:
[1] C. E. Brown, "Machine Learning Techniques for Health Insurance Cost Prediction," Journal of Health Analytics, vol. 5, no. 2, pp. 34–47, 2021.



### Business Understanding
#### Problem Statements
- Bagaimana memprediksi biaya asuransi berdasarkan fitur pelanggan seperti usia, jenis kelamin, BMI, status perokok, jumlah anak, dan wilayah?
- Algoritma machine learning mana yang memberikan prediksi paling akurat terhadap biaya asuransi?
#### Goals
- Membangun model prediksi biaya asuransi berdasarkan karakteristik individu.
- Membandingkan performa berbagai model regresi dan memilih model terbaik berdasarkan metrik evaluasi.
#### Solution Statements
Untuk mencapai tujuan di atas, solusi yang dirancang mencakup:
- Membangun beberapa model regresi: Linear Regression, K-Nearest Neighbors (KNN), Random Forest, dan Gradient Boosting.
- Melakukan evaluasi kinerja model berdasarkan metrik MAE, RMSE, dan R².
- Memilih model terbaik berdasarkan hasil evaluasi pada data uji.

### Data Understanding
#### Sumber Data
Dataset yang digunakan adalah Medical Cost Personal Dataset dari Kaggle (https://www.kaggle.com/datasets/mirichoi0218/insurance).

Dataset ini memiliki 7 fitur independen dan 1 fitur target:
| Fitur      | Deskripsi                                  |
| ---------- | ------------------------------------------ |
| `age`      | Usia pelanggan                             |
| `sex`      | Jenis kelamin (male/female)                |
| `bmi`      | Indeks massa tubuh                         |
| `children` | Jumlah anak yang ditanggung                |
| `smoker`   | Status perokok (yes/no)                    |
| `region`   | Wilayah tempat tinggal pelanggan           |
| `charges`  | Biaya asuransi yang harus dibayar (target) |

#### EDA dan Insight:
Exploratory Data Analysis (EDA) dilakukan untuk memahami hubungan antar fitur. Hasil awal menunjukkan bahwa variabel smoker dan bmi sangat berpengaruh terhadap charges.

### Data Preparation
##### Langkah-langkah:
###### Encoding:
Variabel kategorikal seperti sex, smoker, dan region dikonversi menggunakan One-Hot Encoding.
###### Scaling:
Variabel numerik seperti age, bmi, dan children dinormalisasi menggunakan StandardScaler.
###### Splitting:
Data dibagi 80% untuk training dan 20% untuk testing dengan fungsi train_test_split.
###### Data Cleaning:
Tidak ditemukan missing values sehingga tidak perlu imputasi
###### Alasan:
Encoding mengubah fitur kategorik menjadi numerik agar bisa digunakan dalam model.

### Modeling
#### Model yang digunakan:
###### Linear Regression
- Model baseline dengan interpretasi sederhana.
- Kelebihan: cepat dan mudah dipahami.
- Kekurangan: tidak cocok untuk data non-linear.

###### K-Nearest Neighbors (KNN)
- Non-parametrik dan mengandalkan kedekatan data.
- Kelebihan: simpel, tidak memerlukan asumsi distribusi.
- Kekurangan: lambat saat prediksi dan sensitif terhadap skala.

###### Random Forest Regressor
- Ensembel dari pohon keputusan.
- Kelebihan: performa tinggi, robust terhadap overfitting.
- Kekurangan: interpretasi lebih sulit.

###### Gradient Boosting Regressor
- Model boosting yang membangun prediksi bertahap.
- Kelebihan: akurat untuk data kompleks.
- Kekurangan: memerlukan tuning parameter.

##### Parameter & Tuning:
- Random Forest dan Gradient Boosting menggunakan default parameter (belum dilakukan hyperparameter tuning lanjutan).
- Jika dioptimasi lebih lanjut, performanya bisa ditingkatkan.

### Evaluation
#### Metrik Evaluasi yang digunakan:
- MAE (Mean Absolute Error): Rata-rata kesalahan absolut antara prediksi dan nilai aktual.
- RMSE (Root Mean Squared Error): Memberikan penalti lebih tinggi terhadap error besar.
- R² Score:  Mengukur seberapa baik model menjelaskan variansi data target.

#### Hasil Evaluasi (Data Uji):
| Model             | MAE       | RMSE      | R²       |
| ----------------- | --------- | --------- | -------- |
| Linear Regression | 4,142     | 5,954     | 0.81     |
| KNN               | 6,001     | 8,478     | 0.61     |
| Random Forest     | 3,091     | 4,695     | 0.88     |
| Gradient Boosting | **2,870** | **4,229** | **0.90** |

### Kesimpulan:
Model terbaik adalah Gradient Boosting, dengan skor evaluasi:
Test MAE: 2,471.72
Test RMSE: 4,229.62
Test R²: 0.90
Model ini mampu memprediksi biaya asuransi dengan sangat baik, menjelaskan 90% variabilitas data. Model ini layak dijadikan acuan dalam penentuan biaya premi secara otomatis dan efisien.







