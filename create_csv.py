import os
import csv
import requests

base = os.path.dirname(os.path.abspath(__file__))
datasets = os.path.join(base, "datasets")
os.makedirs(datasets, exist_ok=True)

url = "https://raw.githubusercontent.com/ferhat-mousavi/turkiye-il-ilce-mahalle-koy/main/turkiye-il-ilce-mahalle.json"

district_csv = os.path.join(datasets, "turkey_districts.csv")
zemin_csv = os.path.join(datasets, "zemin_verileri.csv")

response = requests.get(url, timeout=20)
response.raise_for_status()

data = response.json()

district_rows = []
zemin_rows = []

for sehir, ilceler in data.items():
    for ilce, mahalleler in ilceler.items():
        district_rows.append([sehir, ilce])

        for mahalle in mahalleler:
            zemin_rows.append([
                sehir,
                ilce,
                mahalle,
                "Varsayılan zemin",
                5,
                "Bu bölge için detaylı zemin verisi bulunmadığından varsayılan orta düzey zemin riski kullanılmıştır."
            ])

with open(district_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Sehir", "Ilce"])
    writer.writerows(sorted(set(tuple(row) for row in district_rows)))

with open(zemin_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Sehir",
        "Ilce",
        "Mahalle",
        "Zemin_Tipi",
        "Zemin_Riski",
        "Zemin_Aciklama"
    ])
    writer.writerows(zemin_rows)

print("CSV dosyaları oluşturuldu.")
print("turkey_districts.csv:", len(district_rows), "ilçe kaydı")
print("zemin_verileri.csv:", len(zemin_rows), "mahalle kaydı")