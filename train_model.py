import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

df = pd.read_csv('datasets/processed_afet_verisi.csv')
# Yapay zekanın bakacağı 6 sütun:
X = df[['Nufus_Yogunlugu', 'Bina_Yas_Ortalamasi', 'Hastane_Yatak_Kapasitesi', 'Toplanma_Alani', 'Itfaiye_Gucu', 'Zemin_Riski']]
y = df['Kategori_No']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

if not os.path.exists('models'): os.makedirs('models')
joblib.dump(model, 'models/afet_model.pkl')
print("2. ADIM TAMAM: Yapay zeka yeni verileri öğrendi.")