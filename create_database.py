import sqlite3
import os

base = os.path.dirname(os.path.abspath(__file__))

datasets = os.path.join(base, "datasets")

if not os.path.exists(datasets):
    os.makedirs(datasets)

db_yolu = os.path.join(datasets, "afet_veritabani.db")

conn = sqlite3.connect(db_yolu)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS analiz_kayitlari (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sehir TEXT,
    ilce TEXT,
    mahalle TEXT,
    risk_sonucu TEXT,
    risk_skoru INTEGER,
    zemin_riski REAL,
    tarih TEXT
)
""")

conn.commit()
conn.close()

print("Veritabanı oluşturuldu.")
print("Dosya:", db_yolu)