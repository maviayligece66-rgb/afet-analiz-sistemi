from flask import Flask, request, make_response
import pandas as pd
import joblib
import os
import folium
import requests

app = Flask(__name__)

base = os.path.dirname(os.path.abspath(__file__))
data_yolu = os.path.join(base, 'datasets', 'processed_afet_verisi.csv')
model_yolu = os.path.join(base, 'models', 'afet_model.pkl')


def canlı_depremleri_getir():
    try:
        url = "https://api.orhanaydogdu.com.tr/deprem/kandilli/live"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("result", [])[:15]
    except Exception as e:
        print("Deprem API hatası:", e)
    return []


def acil_oneriler_uret(risk_durumu, inputs):
    oneriler = []

    if not inputs:
        return oneriler

    nufus, bina_yasi, yatak, toplanma, itfaiye, zemin = inputs

    if risk_durumu == "Güvenli Bölge":
        oneriler.extend([
            "Mevcut afet hazırlık planları düzenli olarak güncellenmelidir.",
            "Acil durum çantası ve aile iletişim planı hazır tutulmalıdır.",
            "Düzenli afet farkındalık tatbikatları yapılmalıdır."
        ])

    elif risk_durumu == "Orta Riskli":
        oneriler.extend([
            "Tahliye yolları ve toplanma alanları yeniden kontrol edilmelidir.",
            "Riskli yapıların ön incelemesi yapılmalıdır.",
            "Acil iletişim ve yerel müdahale planı oluşturulmalıdır."
        ])

    elif risk_durumu == "Kritik / Riskli":
        oneriler.extend([
            "Bu bölgede acil tahliye planı oluşturulmalıdır.",
            "Toplanma alanı kapasitesi artırılmalıdır.",
            "Eski yapılar için bina dayanıklılık analizi ve güçlendirme önerilir.",
            "Hastane, itfaiye ve ana ulaşım yolları önceliklendirilmelidir."
        ])

    if bina_yasi >= 25:
        oneriler.append("Bina yaşı yüksek olduğu için yapı güvenliği analizi yapılmalıdır.")

    if nufus >= 5000:
        oneriler.append("Nüfus yoğunluğu yüksek olduğu için tahliye süresi uzayabilir.")

    if toplanma <= 3:
        oneriler.append("Toplanma alanı yetersiz görünüyor; alternatif güvenli alanlar belirlenmelidir.")

    if itfaiye <= 3:
        oneriler.append("İtfaiye müdahale kapasitesi artırılmalıdır.")

    if yatak <= 3:
        oneriler.append("Sağlık kapasitesi düşük görünüyor; geçici sağlık noktaları planlanmalıdır.")

    if zemin >= 7:
        oneriler.append("Zemin riski yüksek olduğu için detaylı zemin etüdü yapılmalıdır.")

    return list(dict.fromkeys(oneriler))


@app.route("/", methods=["GET", "POST"])
def index():
    tahmin_sonucu = ""
    risk_durumu = ""
    risk_rengi = "#2ecc71"
    aciklama = ""
    secilen_sehir = ""
    sehirler = []
    oneriler = []
    acil_mod_aktif = False

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
        except Exception:
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

                risk_durumu, risk_rengi = {
                    0: ["Güvenli Bölge", "#2ecc71"],
                    1: ["Orta Riskli", "#f1c40f"],
                    2: ["Kritik / Riskli", "#e74c3c"]
                }[res]

                tahmin_sonucu = risk_durumu

                if secilen_sehir:
                    tahmin_sonucu = f"{secilen_sehir} için sonuç: {risk_durumu}"

                oneriler = acil_oneriler_uret(risk_durumu, inputs)
                acil_mod_aktif = risk_durumu == "Kritik / Riskli"

                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    feats = ['Nüfus', 'Bina Yaşı', 'Yatak', 'Toplanma', 'İtfaiye', 'Zemin']

                    pairs = sorted(zip(feats, imp), key=lambda x: x[1], reverse=True)
                    aciklama = "<br>".join(
                        [f"{f}: %{round(i * 100, 1)} etkili" for f, i in pairs]
                    )

            else:
                tahmin_sonucu = "Model dosyası bulunamadı."

        except Exception as e:
            tahmin_sonucu = "Veri hatası!"
            print("Model hata:", e)

    sehir_options = ""
    for sehir in sehirler:
        selected = "selected" if sehir == secilen_sehir else ""
        sehir_options += f'<option value="{sehir}" {selected}>{sehir}</option>'

    oneriler_html = "".join([f"<li>{o}</li>" for o in oneriler])

    html = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="60">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="theme-color" content="#d90429">
        <link rel="manifest" href="/static/manifest.json">

        <style>
            body {{
                background:#8e0000;
                color:white;
                font-family:sans-serif;
                margin:0;
                padding:15px;
            }}

            h1, h2 {{
                text-align:center;
            }}

            .box {{
                background:white;
                color:black;
                padding:20px;
                border-radius:10px;
                margin:10px auto;
                max-width:1200px;
                box-shadow:0 8px 25px rgba(0,0,0,0.25);
            }}

            input, select {{
                padding:10px;
                margin:5px;
                border-radius:8px;
                border:1px solid #ccc;
            }}

            button {{
                padding:12px;
                background:#c31432;
                color:white;
                border:none;
                border-radius:8px;
                cursor:pointer;
                font-weight:bold;
            }}

            .legend {{
                margin-top:15px;
                background:white;
                padding:12px;
                border-radius:10px;
                color:black;
            }}

            .warning {{
                margin-top:20px;
                padding:15px;
                background:#fff3cd;
                color:#856404;
                border-radius:10px;
            }}

            .example-box {{
                margin-top:15px;
                padding:15px;
                background:#e9ecef;
                color:#333;
                border-radius:10px;
                font-size:0.92em;
                line-height:1.6;
            }}

            .suggestion-box {{
                margin-top:20px;
                padding:18px;
                background:#eafaf1;
                color:#145a32;
                border-left:6px solid #27ae60;
                border-radius:10px;
                line-height:1.5;
            }}

            .accessibility-note {{
                margin-top:20px;
                padding:15px;
                background:#e8f4ff;
                color:#0b3954;
                border-radius:10px;
                line-height:1.5;
            }}

            .emergency-alert {{
                display:none;
                position:fixed;
                z-index:99999;
                top:0;
                left:0;
                width:100%;
                height:100vh;
                background:red;
                color:white;
                text-align:center;
                padding:30vh 20px 0 20px;
                box-sizing:border-box;
                animation: flash 0.7s infinite;
            }}

            .emergency-alert h1 {{
                font-size:44px;
                margin-bottom:15px;
            }}

            .emergency-alert p {{
                font-size:24px;
                font-weight:bold;
            }}

            .close-alert {{
                margin-top:20px;
                background:white;
                color:#b00000;
                font-size:18px;
            }}

            @keyframes flash {{
                0% {{ background-color:#ff0000; }}
                50% {{ background-color:#6b0000; }}
                100% {{ background-color:#ff0000; }}
            }}

            @media (max-width:700px) {{
                body {{
                    padding:8px;
                }}

                .box {{
                    padding:12px;
                }}

                input, select, button {{
                    width:100%;
                    box-sizing:border-box;
                    margin:6px 0;
                }}

                h1 {{
                    font-size:24px;
                }}

                h2 {{
                    font-size:18px;
                }}

                .emergency-alert h1 {{
                    font-size:34px;
                }}

                .emergency-alert p {{
                    font-size:20px;
                }}
            }}
        </style>
    </head>

    <body>

    <div id="emergencyAlert" class="emergency-alert">
        <h1>🚨 ACİL DURUM</h1>
        <p>Risk seviyesi kritik görünüyor.</p>
        <p>Güvenli alana geçin. Asansör kullanmayın. Toplanma alanına yönelin.</p>
        <button class="close-alert" onclick="acilDurumKapat()">Uyarıyı Kapat</button>
    </div>

    <h1>ResiliCity: Afet Direnç Analiz Sistemi</h1>
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

        {'''
        <button type="button" onclick="acilDurumGoster()">
            🚨 Erişilebilir Acil Durum Alarmını Test Et
        </button>
        ''' if acil_mod_aktif else ""}

        {f'''
        <div class="suggestion-box">
            <h3>🧭 Acil Durum Öneri Sistemi</h3>
            <ul>{oneriler_html}</ul>
        </div>
        ''' if oneriler else ""}

        <div class="accessibility-note">
            <strong>♿ Erişilebilir Afet Modu:</strong><br><br>
            ✅ İşitme engelli bireyler için kırmızı yanıp sönen tam ekran görsel alarm<br>
            ✅ Mobil cihazlarda titreşim desteği<br>
            ✅ Görme engelli bireyler için Türkçe sesli yönlendirme<br>
            ✅ Büyük yazı ve yüksek kontrastlı acil durum ekranı
        </div>

        <div class="warning">
            <strong>⚠️ Zemin Uyarısı:</strong><br>
            Yumuşak zeminler deprem etkisini büyütür.
        </div>
    </div>

    <script>
        if ("serviceWorker" in navigator) {{
            navigator.serviceWorker.register("/static/service-worker.js")
            .then(() => console.log("Service Worker kayıt edildi."))
            .catch(error => console.log("Service Worker hatası:", error));
        }}

        function sesliUyariVer() {{
            if ("speechSynthesis" in window) {{
                const mesaj = new SpeechSynthesisUtterance(
                    "Dikkat. Kritik risk tespit edildi. Güvenli alana geçin. Asansör kullanmayın. Toplanma alanına yönelin."
                );

                mesaj.lang = "tr-TR";
                mesaj.rate = 0.9;

                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(mesaj);
            }}
        }}

        function acilDurumGoster() {{
            const alertBox = document.getElementById("emergencyAlert");
            alertBox.style.display = "block";

            if (navigator.vibrate) {{
                navigator.vibrate([500, 300, 500, 300, 1000]);
            }}

            sesliUyariVer();
        }}

        function acilDurumKapat() {{
            document.getElementById("emergencyAlert").style.display = "none";

            if (navigator.vibrate) {{
                navigator.vibrate(0);
            }}

            if ("speechSynthesis" in window) {{
                window.speechSynthesis.cancel();
            }}
        }}
    </script>

    </body>
    </html>
    """

    return make_response(html)


if __name__ == "__main__":
    app.run(debug=True)
