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

#### Kualitas Data
- Missing Values:Tidak ditemukan nilai yang hilang di dataset (df.isnull().sum() = 0 untuk semua kolom).
- Terdapat 1 baris duplikat (df.duplicated().sum() = 1) dan telah dihapus.
- Outlier:
    - bmi Terdeteksi 9 outlier menggunakan metode IQR. Diatasi dengan winsorization.
     - charges: Terdeteksi 139 outlier, yang dikelompokkan berdasarkan status smoker dan dibersihkan sebagian untuk menghindari bias model.
#### EDA dan Insight:
- Pelanggan yang merokok (smoker = yes) memiliki biaya charges yang sangat tinggi dibanding non-perokok.
- Terdapat hubungan positif antara age, bmi, dan charges.
- region tidak memiliki pengaruh signifikan terhadap biaya charges.

### Data Preparation
##### Langkah-langkah:
###### Data Cleaning:
- Tidak ditemukan missing values sehingga tidak perlu imputasi
- Duplikat: Data tidak memiliki baris duplikat (df.duplicated().sum() = 0).
- Outlier pada charges:
    - Deteksi menggunakan metode IQR (Interquartile Range).
    - Outlier dihapus berdasarkan kelompok smoker agar distribusi tetap adil untuk perokok dan non-perokok.
    - Nilai charges yang berada di luar batas [Q1 - 1.5IQR, Q3 + 1.5IQR] dihapus.
- Winsorization pada bmi:
    - Untuk meredam pengaruh nilai ekstrem tanpa menghapus data.
    - Nilai bmi di bawah dan di atas ambang batas disesuaikan ke batas bawah dan atas berdasarkan perhitungan IQR.
###### Encoding:
- sex dan smoker diubah menjadi nilai numerik menggunakan Label Encoding (male/yes = 1, female/no = 0).
- region diubah menjadi fitur dummy menggunakan One-Hot Encoding (dengan drop_first=True untuk menghindari multikolinearitas).
###### Splitting:
- Fitur (X) dan target (y = charges) dipisahkan.
- Data dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan train_test_split dengan random_state = 42.
###### Feature Scaling
- Fitur numerik seperti age, bmi, dan children distandardisasi menggunakan StandardScaler.
- Proses fit dilakukan pada X_train, lalu X_test hanya ditransformasikan menggunakan parameter dari data latih untuk menghindari data leakage.
###### Alasan:
Encoding mengubah fitur kategorik menjadi numerik agar bisa digunakan dalam model.

### Modeling
#### Model yang digunakan:
###### Linear Regression
Linear Regression adalah model dasar yang mengasumsikan hubungan linier antara variabel input dan target. Model ini meminimalkan selisih kuadrat antara nilai prediksi dan aktual (metode Ordinary Least Squares).
- Cara Kerja:
    - Linear Regression mencoba menemukan garis lurus terbaik yang meminimalkan selisih kuadrat antara nilai aktual dan prediksi (metode OLS - Ordinary Least Squares).
    - Parameter Utama: Tidak banyak parameter. Dalam proyek ini digunakan versi default dari LinearRegression() dari scikit-learn.
- Model baseline dengan interpretasi sederhana.
- Kelebihan: cepat dan mudah dipahami.
- Kekurangan: tidak cocok untuk data non-linear.

###### K-Nearest Neighbors (KNN)
KNN merupakan algoritma non-parametrik berbasis instance. Prediksi dilakukan dengan menghitung rata-rata dari k tetangga terdekat dalam ruang fitur.
- Cara Kerja:
    - KNN memprediksi nilai target suatu titik data baru dengan menghitung rata-rata target dari k tetangga terdekat di data latih.
-  Parameter Utama:
    - Parameter: Pada proyek ini digunakan k = 10, artinya prediksi didasarkan pada rata-rata 10 tetangga terdekat berdasarkan Euclidean distance.
- Pertimbangan: KNN dipilih karena kesederhanaannya, namun sensitif terhadap skala fitur (karena itu dilakukan scaling sebelumnya).
- Non-parametrik dan mengandalkan kedekatan data.
- Kelebihan: simpel, tidak memerlukan asumsi distribusi.
- Kekurangan: lambat saat prediksi dan sensitif terhadap skala.

###### Random Forest Regressor
Random Forest merupakan metode ensemble berbasis bagging yang terdiri dari banyak pohon keputusan. Setiap pohon dilatih pada subset acak data, lalu hasil prediksi dirata-ratakan.
- Cara Kerja: 
    - Merupakan ensemble dari banyak decision tree yang dilatih secara acak (bootstrapping dan pemilihan fitur acak). Hasil akhir adalah rata-rata dari semua prediksi pohon.
- Parameter Utama:
   - Parameter: Digunakan parameter default dari scikit-learn, yaitu n_estimators=100.
- Pertimbangan: Dipilih karena kemampuannya menangani data non-linear dan mengurangi overfitting dibanding satu decision tree.
- Ensembel dari pohon keputusan.
- Kelebihan: performa tinggi, robust terhadap overfitting.
- Kekurangan: interpretasi lebih sulit.

###### Gradient Boosting Regressor
Gradient Boosting adalah teknik ensemble boosting yang membangun model secara bertahap, di mana tiap pohon bertujuan memperbaiki kesalahan dari pohon sebelumnya.
- Cara Kerja:
     - Model boosting yang membangun model secara bertahap. Setiap model baru berusaha memperbaiki kesalahan dari model sebelumnya menggunakan metode gradient descent.
- Parameter Utama:
    - learning_rate = 0.1 (default): menentukan seberapa besar kontribusi tiap model baru.
    - n_estimators = 100 (default): jumlah boosting stages.
- Pertimbangan: Model ini kuat dalam mempelajari pola kompleks dan memberikan performa prediksi yang sangat baik.
- Model boosting yang membangun prediksi bertahap.
- Kelebihan: akurat untuk data kompleks.
- Kekurangan: memerlukan tuning parameter.

##### Catatan:
- Random Forest dan Gradient Boosting menggunakan default parameter (belum dilakukan hyperparameter tuning lanjutan).
- Jika dioptimasi lebih lanjut, performanya bisa ditingkatkan.

### Evaluation
#### Metrik Evaluasi yang digunakan:
- MAE (Mean Absolute Error): Rata-rata kesalahan absolut antara prediksi dan nilai aktual.
- RMSE (Root Mean Squared Error): Memberikan penalti lebih tinggi terhadap error besar.
- R² Score:  Mengukur seberapa baik model menjelaskan variansi data target.

#### Hasil Evaluasi :
| Model             | Train MAE | Test MAE  | Train RMSE | Test RMSE | Train R² | Test R² |
| ----------------- | --------- | --------- | -----------| ----------| -------- | --------|
| Linear Regression | 4,186.07  | 4,181.90  | 6,080.12   | 5,954.23  | 0.73     | 0.81    |
| KNN               | 7,147.71  | 8,840.99  | 9,791.52   | 12,524.93 | 0.30     | 0.15    |
| Random Forest     | 1,063.71  | 2,636.78  | 1,899.09   | 4,701.34  | 0.97     | 0.88    |
| Gradient Boosting | 2,107.04  | 2,471.72  | 3,846.43   | 4,229.62  | 0.89     | 0.90    |


### Kesimpulan:
Berdasarkan hasil evaluasi terhadap empat model regresi untuk memprediksi biaya asuransi (charges), diperoleh hasil sebagai berikut:
- Linear Regression menunjukkan performa cukup baik dengan hasil yang stabil antara data latih dan uji, serta R² test sebesar 0.81. Model ini cocok jika dibutuhkan interpretasi yang jelas dan sederhana.
- KNN Regressor menunjukkan performa paling buruk, dengan Test MAE tertinggi (8,840.99) dan R² test paling rendah (0.15). Ini menandakan model gagal menangkap pola data dengan baik dan tidak cocok untuk digunakan.
- Random Forest memiliki Train R² sangat tinggi (0.97) namun terjadi sedikit overfitting karena Test R²-nya menurun ke 0.88. Meski begitu, model ini tetap memberikan prediksi yang sangat baik.
- Gradient Boosting menjadi model terbaik secara keseluruhan, dengan Test MAE terendah kedua (2,471.72) dan Test R² tertinggi (0.90). Artinya, model ini mampu menjelaskan 90% variabilitas dalam data uji dan memberikan prediksi yang akurat serta seimbang.

Gradient Boosting adalah model paling optimal dan direkomendasikan untuk digunakan dalam sistem prediksi biaya premi asuransi secara otomatis dan efisien.







