from flask import Flask, request, make_response
import pandas as pd
import joblib
import os
import folium
import requests
import sqlite3
from datetime import datetime

app = Flask(__name__)

base = os.path.dirname(os.path.abspath(__file__))
data_yolu = os.path.join(base, 'datasets', 'processed_afet_verisi.csv')
model_yolu = os.path.join(base, 'models', 'afet_model.pkl')
db_yolu = os.path.join(base, 'analizler.db')


def veritabani_olustur():
    conn = sqlite3.connect(db_yolu)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analizler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sehir TEXT,
            nufus_yogunlugu REAL,
            bina_yasi REAL,
            yatak_kapasitesi REAL,
            toplanma_alani REAL,
            itfaiye_gucu REAL,
            zemin_riski REAL,
            tahmin_sonucu TEXT,
            tarih TEXT
        )
    """)
    conn.commit()
    conn.close()


def analiz_kaydet(sehir, inputs, sonuc):
    conn = sqlite3.connect(db_yolu)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analizler (
            sehir, nufus_yogunlugu, bina_yasi, yatak_kapasitesi,
            toplanma_alani, itfaiye_gucu, zemin_riski, tahmin_sonucu, tarih
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sehir,
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        sonuc,
        datetime.now().strftime("%d.%m.%Y %H:%M")
    ))
    conn.commit()
    conn.close()


def son_analizleri_getir():
    conn = sqlite3.connect(db_yolu)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sehir, tahmin_sonucu, tarih
        FROM analizler
        ORDER BY id DESC
        LIMIT 5
    """)
    veriler = cursor.fetchall()
    conn.close()
    return veriler


def canlı_depremleri_getir():
    try:
        url = "https://api.orhanaydogdu.com.tr/deprem/kandilli/live"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("result", [])[:15]
    except:
        pass
    return []


veritabani_olustur()


@app.route("/", methods=["GET", "POST"])
def index():
    tahmin_sonucu = ""
    risk_rengi = "#2ecc71"
    aciklama = ""
    secilen_sehir = ""
    sehirler = []

    if os.path.exists(data_yolu):
        df = pd.read_csv(data_yolu)
        sehirler = sorted(df["Sehir"].dropna().unique())

    son_depremler = canlı_depremleri_getir()
    deprem_ozeti = " | ".join(
        [f"{d.get('title','?')} ({d.get('mag','?')})" for d in son_depremler[:5]]
    )

    m = folium.Map(location=[39, 35], zoom_start=6, tiles="cartodbpositron")

    for d in son_depremler:
        try:
            lon, lat = d["geojson"]["coordinates"]
            mag = float(d["mag"])

            folium.Circle(
                location=[lat, lon],
                radius=mag * 5000,
                color="darkred",
                fill=True,
                fill_color="red",
                fill_opacity=0.4,
                popup=f"{d.get('title','?')} - {mag}"
            ).add_to(m)
        except:
            continue

    map_html = m._repr_html_()

    if request.method == "POST":
        try:
            secilen_sehir = request.form.get("sehir", "")

            inputs = [float(request.form.get(x, 0)) for x in ['n', 'b', 'y', 't', 'i', 'z']]

            if os.path.exists(model_yolu):
                model = joblib.load(model_yolu)

                df_test = pd.DataFrame([inputs], columns=[
                    'Nufus_Yogunlugu',
                    'Bina_Yas_Ortalamasi',
                    'Hastane_Yatak_Kapasitesi',
                    'Toplanma_Alani',
                    'Itfaiye_Gucu',
                    'Zemin_Riski'
                ])

                res = model.predict(df_test)[0]

                sade_sonuc, risk_rengi = {
                    0: ["Güvenli Bölge", "#2ecc71"],
                    1: ["Orta Riskli", "#f1c40f"],
                    2: ["Kritik / Riskli", "#e74c3c"]
                }[res]

                if secilen_sehir:
                    tahmin_sonucu = f"{secilen_sehir} için sonuç: {sade_sonuc}"
                else:
                    tahmin_sonucu = sade_sonuc

                analiz_kaydet(secilen_sehir, inputs, sade_sonuc)

                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    feats = ['Nüfus', 'Bina Yaşı', 'Yatak', 'Toplanma', 'İtfaiye', 'Zemin']

                    pairs = sorted(zip(feats, imp), key=lambda x: x[1], reverse=True)
                    aciklama = "<br>".join(
                        [f"{f}: %{round(i * 100, 1)} etkili" for f, i in pairs]
                    )

        except Exception as e:
            tahmin_sonucu = "Veri hatası!"
            print("Hata:", e)

    sehir_options = ""
    for sehir in sehirler:
        selected = "selected" if sehir == secilen_sehir else ""
        sehir_options += f'<option value="{sehir}" {selected}>{sehir}</option>'

    son_analizler = son_analizleri_getir()
    son_analiz_html = ""

    for sehir, sonuc, tarih in son_analizler:
        son_analiz_html += f"""
        <tr>
            <td>{sehir}</td>
            <td>{sonuc}</td>
            <td>{tarih}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="60">
        <style>
            body {{ background:#8e0000; color:white; font-family:sans-serif; }}
            .box {{ background:white; color:black; padding:20px; border-radius:10px; margin:10px auto; max-width:1200px; }}
            input, select {{ padding:10px; margin:5px; border-radius:8px; border:1px solid #ccc; }}
            button {{ padding:12px; background:red; color:white; border:none; border-radius:8px; cursor:pointer; }}
            .legend {{ margin-top:15px; background:white; padding:12px; border-radius:10px; color:black; }}
            .warning {{ margin-top:20px; padding:15px; background:#fff3cd; color:#856404; border-radius:10px; }}
            .example-box {{
                margin-top:15px;
                padding:15px;
                background:#e9ecef;
                color:#333;
                border-radius:10px;
                font-size:0.92em;
                line-height:1.6;
            }}
            table {{
                width:100%;
                border-collapse:collapse;
                margin-top:15px;
            }}
            th, td {{
                border:1px solid #ddd;
                padding:10px;
                text-align:left;
            }}
            th {{
                background:#f2f2f2;
            }}
        </style>
    </head>
    <body>

    <h2>🔴 Canlı Depremler: {deprem_ozeti}</h2>

    <div class="box">
        {map_html}

        <div class="legend">
            <b>Risk Seviyeleri:</b><br>
            🟢 Düşük Risk<br>
            🟡 Orta Risk<br>
            🔴 Yüksek Risk
        </div>
    </div>

    <div class="box">
        <form method="POST">
            <select name="sehir" required>
                <option value="">Şehir seçiniz</option>
                {sehir_options}
            </select>

            <br>

            <input type="number" step="any" name="n" placeholder="Nüfus Yoğunluğu örn: 5000" required>
            <input type="number" step="any" name="b" placeholder="Bina Yaşı örn: 20" required>
            <input type="number" step="any" name="y" placeholder="Yatak Kapasitesi örn: 1000" required>
            <input type="number" step="any" name="t" placeholder="Toplanma Alanı örn: 50000" required>
            <input type="number" step="any" name="i" placeholder="İtfaiye Gücü örn: 50" required>
            <input type="number" step="any" name="z" placeholder="Zemin Riski 1-10 örn: 7" required>

            <br>
            <button type="submit">Analiz Et</button>
        </form>

        <div class="example-box">
            <b>📌 Örnek Değer Rehberi:</b><br><br>
            • <b>Nüfus Yoğunluğu:</b> 5000 → yoğun şehirler için yüksek değer kabul edilir.<br>
            • <b>Bina Yaşı:</b> 20 → bölgedeki ortalama bina yaşı gibi düşünülmelidir.<br>
            • <b>Yatak Kapasitesi:</b> 1000 → hastane/acil durum kapasitesini temsil eder.<br>
            • <b>Toplanma Alanı:</b> 50000 → m² cinsinden düşünülebilir; yüksek değer daha avantajlıdır.<br>
            • <b>İtfaiye Gücü:</b> 50 → ekip, araç veya müdahale kapasitesi gibi düşünülebilir.<br>
            • <b>Zemin Riski:</b> 1-10 arası girilir. 1 düşük risk, 10 çok yüksek risk anlamına gelir.
        </div>

        <h2 style="color:{risk_rengi};">{tahmin_sonucu}</h2>
        <p>{aciklama}</p>

        <div class="warning">
            <strong>⚠️ Zemin Uyarısı:</strong><br>
            Yumuşak zeminler deprem etkisini büyütür.
        </div>
    </div>

    <div class="box">
        <h2>📁 Son Yapılan Analizler</h2>
        <table>
            <tr>
                <th>Şehir</th>
                <th>Tahmin Sonucu</th>
                <th>Tarih</th>
            </tr>
            {son_analiz_html}
        </table>
    </div>

    </body>
    </html>
    """

    return make_response(html)


if __name__ == "__main__":
    app.run(debug=True)
