# Telco Customer Churn Prediction 📉
End-to-End Data Analytics & Machine Learning Model untuk memprediksi probabilitas Telco Customer Churn guna mendukung optimalisasi strategi pemasaran dan retensi pelanggan.

## 📌 Latar Belakang & Tujuan Bisnis
Di industri telekomunikasi, mempertahankan pelanggan lama jauh lebih murah daripada mengakuisisi pelanggan baru. Proyek ini bertujuan untuk membangun model Machine Learning yang mampu mendeteksi pelanggan mana yang berpotensi untuk berhenti berlangganan (*churn*), sehingga tim pemasaran dapat melakukan intervensi preventif berupa penawaran promo atau diskon loyalitas secara tepat sasaran.

## 📊 Eksplorasi Data (EDA)
Dataset ini memiliki tantangan berupa kelas target yang tidak seimbang (*Imbalanced Data*), di mana jumlah pelanggan yang *churn* jauh lebih sedikit.
![Proporsi Churn](nama_file_gambar_proporsi_churn.png)

Selain itu, analisis menunjukkan bahwa pelanggan dengan kontrak **Month-to-month** memiliki tingkat *churn* yang jauh lebih masif dibandingkan kontrak tahunan.
![Churn Berdasarkan Kontrak](nama_file_gambar_kontrak.png)

## 🤖 Evaluasi Model Machine Learning
Proyek ini menguji 6 algoritma Machine Learning. Karena data tidak seimbang, evaluasi difokuskan pada nilai **Recall** dan **F1-Score** untuk meminimalkan *False Negative* (mesin gagal mendeteksi pelanggan yang kabur).
*   **Logistic Regression** dan **CatBoost** memberikan keseimbangan metrik terbaik di kisaran F1-Score ~60-61%.
*   Algoritma *tree-based* (seperti LightGBM dan CatBoost) menunjukkan ketangguhan luar biasa dalam menekan *False Negative* pada *Confusion Matrix*.

![Confusion Matrix](nama_file_gambar_confusion_matrix.png)

## 💡 Wawasan Bisnis (Business Insights) & Rekomendasi
Melalui ekstraksi **Feature Importance**, model secara konsisten mengungkap 3 pemicu utama *churn*:
1. **Total Charges & Monthly Charges:** Tagihan bulanan yang dirasa terlalu tinggi/membengkak.
2. **Tenure Months:** Pelanggan baru (umur langganan pendek) sangat rentan untuk pindah operator.
3. **Internet Service (Fiber Optic) & Contract:** Pengguna Fiber Optic tanpa ikatan kontrak tahunan adalah kelompok paling berisiko.

![Feature Importance](nama_file_gambar_feature_importance_lightgbm.png)

**Rekomendasi Aksi Pemasaran:**
Perusahaan harus memfokuskan anggaran retensi untuk memberikan promo khusus atau diskon peningkatan layanan pada pelanggan baru di bulan ke-2 hingga ke-6 berlangganan, terutama membujuk mereka agar beralih dari kontrak bulanan menjadi kontrak 1 tahun.
