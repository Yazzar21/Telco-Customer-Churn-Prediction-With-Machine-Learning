import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==========================================
# --- FASE 1: PEMAHAMAN & PEMBERSIHAN DATA ---
# ==========================================
print("--- FASE 1: MEMUAT DAN MEMBERSIHKAN DATA ---")
file_path = r'D:\Data Analytics\Telco Customer Churn Project Portofolio\Telco_customer_churn.xlsx'
df = pd.read_excel(file_path)

print(df.head())

print(df.info())

print(f"Dimensi data awal: {df.shape[0]} baris dan {df.shape[1]} kolom.")

df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

jumlah_kosong = df['Total Charges'].isnull().sum()
if jumlah_kosong > 0:
    print(f"Ditemukan {jumlah_kosong} pelanggan baru tanpa tagihan awal. Mengubah nilainya menjadi 0")
    df['Total Charges'] = df['Total Charges'].fillna(0)

print("Pembersihan data dasar selesai!\n")

# ==========================================
# --- FASE 2: EKSPLORASI DATA VISUAL (EDA) ---
# ==========================================
print("--- FASE 2: MENYIAPKAN GRAFIK EDA ---")
print("Menyiapkan grafik Distribusi Churn")

plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x='Churn Label', palette='Set2')
plt.title('Proporsi Pelanggan Berhenti (Churn) vs Bertahan', fontsize=14, fontweight='bold')
plt.xlabel('Status Churn', fontsize=12)
plt.ylabel('Jumlah Pelanggan', fontsize=12)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.show()

print("Menyiapkan grafik Churn berdasarkan Kontrak")

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x='Contract', hue='Churn Label', palette='viridis')
plt.title('Tingkat Churn Berdasarkan Jenis Kontrak Pelanggan', fontsize=14, fontweight='bold')
plt.xlabel('Jenis Kontrak', fontsize=12)
plt.ylabel('Jumlah Pelanggan', fontsize=12)
plt.legend(title='Status Churn')
plt.tight_layout()
plt.show()

print("--- FASE 3: PERSIAPAN DATA (PREPROCESSING) ---")
file_path = r'D:\Data Analytics\Telco Customer Churn Project Portofolio\Telco_customer_churn.xlsx'
df = pd.read_excel(file_path)

df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

kolom_dihapus = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 
                 'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 
                 'Churn Score', 'CLTV', 'Churn Reason']

df_bersih = df.drop(columns=kolom_dihapus)
print(f"Sisa kolom setelah fitur sampah dibuang: {df_bersih.shape[1]} kolom.")

# 4. MEMISAHKAN SOAL (X) DAN JAWABAN (y)
X = df_bersih.drop('Churn Value', axis=1) # Soal (Fitur)
y = df_bersih['Churn Value']              # Kunci Jawaban (Target)

# 5. MENGUBAH TEKS MENJADI ANGKA (ONE-HOT ENCODING)
X_encoded = pd.get_dummies(X, drop_first=True)

print(f"Dimensi data X sebelum encoding: {X.shape}")
print(f"Dimensi data X setelah encoding: {X_encoded.shape}")

print("\nSukses! Semua data teks kini telah menjadi angka matematis (True/False atau 1/0).")
print("Contoh nama kolom baru:")
print(X_encoded.columns[:10].tolist())

# ==========================================
# --- FASE 4: PELATIHAN & BENCHMARKING MULTI-MODEL ---
# ==========================================
print("\n--- FASE 4: MEMBAGI DATA DAN MENGADU 6 MODEL ---")

# 1. Membagi Data Latih (80%) dan Data Uji (20%)
# stratify=y memastikan proporsi churn/bertahan tetap seimbang di kedua potongan data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

print(f"Jumlah data untuk mesin belajar (Train): {X_train.shape[0]} baris")
print(f"Jumlah data untuk mesin ujian (Test): {X_test.shape[0]} baris\n")

# 2. Mendefinisikan 6 Algoritma Machine Learning
daftar_model = {
    "Logistic Regression": LogisticRegression(max_iter=5000, solver='liblinear', random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

hasil_evaluasi = []

print("Mulai melatih dan menguji 6 model secara berurutan...")
# 3. Looping untuk melatih, menguji, dan mencatat nilai metrik setiap model
for nama_model, model in daftar_model.items():
    print(f"-> Sedang memproses: {nama_model}...")
    
    # Mesin Belajar
    model.fit(X_train, y_train)
    
    # Mesin Menebak
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Menghitung Nilai Ujian
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    
    # Menyimpan nilai
    hasil_evaluasi.append({
        "Model": nama_model,
        "Akurasi": round(acc * 100, 2),
        "Presisi": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1-Score": round(f1 * 100, 2),
        "ROC-AUC": round(roc * 100, 2)
    })

# 4. Menampilkan Hasil ke dalam Tabel
import pandas as pd # Memastikan pandas terpanggil untuk DataFrame
df_hasil = pd.DataFrame(hasil_evaluasi)
df_hasil = df_hasil.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

print("\n========================================================")
print("TABEL PERBANDINGAN PERFORMA MODEL CHURN (BENCHMARKING)")
print("========================================================")
print(df_hasil.to_string(index=False))
print("========================================================")

# 5. Visualisasi Hasil Benchmarking
print("\nMenyiapkan grafik perbandingan performa model... (Silakan cek popup)")
plt.figure(figsize=(12, 6))

# Mengubah bentuk tabel untuk visualisasi
df_melted = df_hasil.melt(id_vars="Model", value_vars=["F1-Score", "ROC-AUC", "Recall"], 
                          var_name="Metrik", value_name="Skor (%)")

# Menggambar diagram batang
sns.barplot(data=df_melted, x="Model", y="Skor (%)", hue="Metrik", palette="viridis")

plt.title("Perbandingan Performa 6 Model ML (Telco Customer Churn)", fontsize=14, fontweight='bold')
plt.xlabel("Algoritma Model", fontsize=12)
plt.ylabel("Skor Persentase (%)", fontsize=12)
plt.ylim(0, 110)
plt.xticks(rotation=15, fontsize=10)
plt.legend(title="Metrik Evaluasi", loc='lower right')

plt.tight_layout()
plt.show()

# ==========================================
# --- FASE 5: CONFUSION MATRIX & FEATURE IMPORTANCE ---
# ==========================================
print("\n--- FASE 5: MEMBEDAH KEPUTUSAN MODEL ---")

# ---------------------------------------------------------
# 1. VISUALISASI CONFUSION MATRIX (Grid)
# ---------------------------------------------------------
print("Menyiapkan Confusion Matrix... (Silakan cek popup pertama)")
fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 10))
axes_cm = axes_cm.flatten()

for i, (nama_model, model) in enumerate(daftar_model.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i], cbar=False,
                xticklabels=['Bertahan (0)', 'Churn (1)'], 
                yticklabels=['Bertahan (0)', 'Churn (1)'])
    axes_cm[i].set_title(f'{nama_model}', fontweight='bold')
    axes_cm[i].set_xlabel('') 
    axes_cm[i].set_ylabel('Kenyataan (Aktual)')

plt.suptitle("Confusion Matrix: False Positive vs False Negative", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. VISUALISASI FEATURE IMPORTANCE (SATU PER SATU)
# ---------------------------------------------------------
print("\nMenyiapkan grafik Feature Importance satu per satu...")

for nama_model, model in daftar_model.items():
    importances = None
    
    # Ekstraksi bobot berdasarkan jenis algoritma
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        
    if importances is not None:
        # Menyusun data ke dalam tabel dan mengambil 10 teratas
        df_importance = pd.DataFrame({'Fitur': X_encoded.columns, 'Skor Kepentingan': importances})
        df_importance = df_importance.sort_values(by='Skor Kepentingan', ascending=False).head(10)
        
        # Membuat dan menampilkan grafik secara terpisah (Satu per satu)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_importance, x='Skor Kepentingan', y='Fitur', palette='viridis')
        plt.title(f"10 Faktor Penentu - {nama_model}", fontsize=16, fontweight='bold')
        plt.xlabel("Tingkat Kepentingan (Bobot)", fontsize=12)
        plt.ylabel("Faktor / Variabel", fontsize=12)
        
        plt.tight_layout()
        plt.show() # Grafik akan muncul, dan program akan berhenti sejenak sampai Anda menutupnya
    else:
        # Melewati model yang tidak memiliki Feature Importance (seperti Naive Bayes)
        print(f"-> Melewati {nama_model}: Tidak memiliki atribut Feature Importance secara matematis.")

print("\nSeluruh proses visualisasi Fase 5 telah selesai!")