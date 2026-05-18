from flask import Flask, request, make_response
import pandas as pd
import joblib
import os
import folium
import requests
import json
import unicodedata
import sqlite3

app = Flask(__name__)

base = os.path.dirname(os.path.abspath(__file__))

data_yolu = os.path.join(base, 'datasets', 'processed_afet_verisi.csv')
db_yolu = os.path.join(base, 'datasets', 'afet_veritabani.db')
model_yolu = os.path.join(base, 'models', 'afet_model.pkl')
geojson_yolu = os.path.join(base, 'datasets', 'turkey_provinces.geojson')
ilce_yolu = os.path.join(base, 'datasets', 'turkey_districts.csv')
zemin_yolu = os.path.join(base, 'datasets', 'zemin_verileri.csv')


def normalize_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if not unicodedata.combining(c))


def turkce_sirala(liste):
    """Türkçe karakterleri dikkate alarak alfabetik sıralama yapar."""
    return sorted(liste, key=lambda x: normalize_text(x))


def afad_depremleri_getir():
    try:
        url = "https://deprem.afad.gov.tr/apiv2/event/latest"
        r = requests.get(url, timeout=8)

        if r.status_code == 200:
            veriler = r.json()
            depremler = []

            for d in veriler[:30]:
                try:
                    mag = float(d.get("magnitude", 0))
                    lat = float(d.get("latitude", 0))
                    lon = float(d.get("longitude", 0))

                    depremler.append({
                        "kaynak": "AFAD",
                        "title": d.get("location", "Bilinmeyen Konum"),
                        "mag": mag,
                        "date": d.get("date", ""),
                        "geojson": {
                            "coordinates": [lon, lat]
                        }
                    })

                except Exception:
                    continue

            return depremler

    except Exception as e:
        print("AFAD API hatası:", e)

    return []


def kandilli_depremleri_getir():
    try:
        url = "https://api.orhanaydogdu.com.tr/deprem/kandilli/live"
        r = requests.get(url, timeout=8)

        if r.status_code == 200:
            depremler = r.json().get("result", [])

            for d in depremler:
                d["kaynak"] = "Kandilli"

            return depremler[:30]

    except Exception as e:
        print("Kandilli API hatası:", e)

    return []


def canlı_depremleri_getir():
    # Önce AFAD verisi alınır. AFAD çalışmazsa Kandilli yedek kaynak olarak kullanılır.
    afad = afad_depremleri_getir()

    if afad:
        return afad

    return kandilli_depremleri_getir()


def zemin_bilgisi_getir(zemin_df, sehir, ilce="", mahalle=""):
    """
    Zemin verisi CSV dosyasından şehir/ilçe/mahalle bilgisine göre zemin bilgisini getirir.
    CSV yoksa veya eşleşme bulunamazsa varsayılan orta risk değeri kullanılır.
    Beklenen CSV kolonları:
    Sehir, Ilce, Mahalle, Zemin_Tipi, Zemin_Riski, Zemin_Aciklama
    """
    varsayilan = {
        "tip": "Zemin verisi bulunamadı",
        "risk": 5,
        "aciklama": "Bu bölge için kayıtlı zemin verisi bulunamadığı için analizde varsayılan orta düzey zemin riski kullanılmıştır."
    }

    if zemin_df is None or zemin_df.empty or not sehir:
        return varsayilan

    try:
        df = zemin_df.copy()

        if "Sehir" in df.columns:
            df = df[df["Sehir"].apply(normalize_text) == normalize_text(sehir)]

        if ilce and "Ilce" in df.columns:
            ilce_eslesme = df[df["Ilce"].apply(normalize_text) == normalize_text(ilce)]
            if not ilce_eslesme.empty:
                df = ilce_eslesme

        if mahalle and "Mahalle" in df.columns:
            mahalle_eslesme = df[df["Mahalle"].apply(normalize_text) == normalize_text(mahalle)]
            if not mahalle_eslesme.empty:
                df = mahalle_eslesme

        if df.empty:
            return varsayilan

        satir = df.iloc[0]

        return {
            "tip": satir.get("Zemin_Tipi", "Belirtilmemiş"),
            "risk": float(satir.get("Zemin_Riski", 5)),
            "aciklama": satir.get("Zemin_Aciklama", "Bu bölgenin zemin bilgisi veri setinden alınmıştır.")
        }

    except Exception as e:
        print("Zemin bilgisi okuma hatası:", e)
        return varsayilan


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


def risk_skoru_getir(risk_durumu):
    if risk_durumu == "Güvenli Bölge":
        return 1
    elif risk_durumu == "Orta Riskli":
        return 3
    elif risk_durumu == "Kritik / Riskli":
        return 5
    return 0


def risk_rengi_getir(risk_skoru):
    renkler = {
        0: "#d9d9d9",
        1: "#2ecc71",
        2: "#f1c40f",
        3: "#f39c12",
        4: "#e74c3c",
        5: "#8b0000"
    }

    return renkler.get(risk_skoru, "#d9d9d9")


def geojson_sehir_adi_bul(feature):
    props = feature.get("properties", {})

    olasi_alanlar = [
        "name",
        "NAME_1",
        "Name",
        "il",
        "Il",
        "IL",
        "province",
        "Province",
        "sehir",
        "Sehir"
    ]

    for alan in olasi_alanlar:
        if alan in props:
            return props[alan]

    return ""


def sehirleri_renklendir(m, secilen_sehir, risk_skoru, risk_durumu):
    if not os.path.exists(geojson_yolu):
        print("GeoJSON dosyası bulunamadı:", geojson_yolu)
        return

    try:
        with open(geojson_yolu, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        secilen_norm = normalize_text(secilen_sehir)

        def style_function(feature):
            sehir_adi = geojson_sehir_adi_bul(feature)
            sehir_norm = normalize_text(sehir_adi)

            if secilen_norm and secilen_norm == sehir_norm:
                return {
                    "fillColor": risk_rengi_getir(risk_skoru),
                    "color": "#111111",
                    "weight": 2,
                    "fillOpacity": 0.75
                }

            return {
                "fillColor": "#f7f7f7",
                "color": "#666666",
                "weight": 1,
                "fillOpacity": 0.25
            }

        def highlight_function(feature):
            return {
                "fillColor": "#ffff99",
                "color": "#000000",
                "weight": 3,
                "fillOpacity": 0.7
            }

        folium.GeoJson(
            geojson_data,
            name="Şehir Risk Haritası",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=[],
                aliases=[],
                sticky=True,
                labels=False
            )
        ).add_to(m)

        if secilen_sehir and risk_skoru > 0:
            folium.Marker(
                location=[39, 35],
                popup=f"{secilen_sehir} - {risk_durumu} - Risk Skoru: {risk_skoru}/5",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)

    except Exception as e:
        print("GeoJSON harita hatası:", e)


@app.route("/", methods=["GET", "POST"])
def index():
    tahmin_sonucu = ""
    risk_durumu = ""
    risk_rengi = "#2ecc71"
    risk_skoru = 0
    aciklama = ""
    secilen_sehir = ""
    secilen_ilce = ""
    secilen_mahalle = ""
    sehirler = []
    ilce_verileri = {}
    mahalle_verileri = {}
    zemin_df = None
    zemin_bilgisi = None
    oneriler = []
    deprem_alarm_var = False
    alarm_mesaji = ""

    # Şehir listesi önce SQLite veritabanından alınır.
    # Veritabanı yoksa eski CSV dosyası yedek olarak kullanılır.
    if os.path.exists(db_yolu):
        try:
            conn = sqlite3.connect(db_yolu)
            df = pd.read_sql_query("SELECT * FROM afet_verileri", conn)
            conn.close()

            if "Sehir" in df.columns:
                sehirler = turkce_sirala(df["Sehir"].dropna().unique().tolist())

        except Exception as e:
            print("Veritabanı okuma hatası:", e)

    elif os.path.exists(data_yolu):
        try:
            df = pd.read_csv(data_yolu, encoding="utf-8-sig")

            if "Sehir" in df.columns:
                sehirler = turkce_sirala(df["Sehir"].dropna().unique().tolist())

        except Exception as e:
            print("CSV veri okuma hatası:", e)

    # İlçe CSV dosyası sonra eklenecek. Dosya yoksa sistem bozulmadan çalışır.
    if os.path.exists(ilce_yolu):
        try:
            ilce_df = pd.read_csv(ilce_yolu, encoding="utf-8-sig")

            if "Sehir" in ilce_df.columns and "Ilce" in ilce_df.columns:
                for sehir, grup in ilce_df.groupby("Sehir"):
                    ilce_verileri[sehir] = turkce_sirala(grup["Ilce"].dropna().unique().tolist())

        except Exception as e:
            print("İlçe CSV okuma hatası:", e)

    # Zemin CSV dosyası sonra eklenecek. Dosya yoksa sistem varsayılan zemin riskiyle çalışır.
    if os.path.exists(zemin_yolu):
        try:
            zemin_df = pd.read_csv(zemin_yolu, encoding="utf-8-sig")

            if "Sehir" in zemin_df.columns and "Ilce" in zemin_df.columns:
                for sehir, grup in zemin_df.groupby("Sehir"):
                    mevcut_ilceler = set(ilce_verileri.get(sehir, []))
                    yeni_ilceler = set(grup["Ilce"].dropna().unique().tolist())
                    ilce_verileri[sehir] = turkce_sirala(list(mevcut_ilceler.union(yeni_ilceler)))

            if all(kolon in zemin_df.columns for kolon in ["Sehir", "Ilce", "Mahalle"]):
                for (sehir, ilce), grup in zemin_df.groupby(["Sehir", "Ilce"]):
                    anahtar = f"{sehir}|||{ilce}"
                    mahalle_verileri[anahtar] = turkce_sirala(grup["Mahalle"].dropna().unique().tolist())

        except Exception as e:
            print("Zemin CSV okuma hatası:", e)

    # Şehir listesi yalnızca model/veritabanı verisinden gelirse 81 ilin tamamı görünmeyebilir.
    # Bu nedenle ilçe ve zemin CSV dosyalarındaki şehirler de listeye eklenir.
    tum_sehirler = set(sehirler)
    tum_sehirler.update(ilce_verileri.keys())

    if zemin_df is not None and not zemin_df.empty and "Sehir" in zemin_df.columns:
        tum_sehirler.update(zemin_df["Sehir"].dropna().unique().tolist())

    sehirler = turkce_sirala(list(tum_sehirler))

    tum_depremler = canlı_depremleri_getir()

    # Üst canlı deprem alanında sadece 4.0 ve üzeri depremler gösterilir.
    ust_depremler = [
        d for d in tum_depremler
        if float(d.get("mag", 0)) >= 4
    ]

    # Alarm, analiz sonucuna göre değil; canlı deprem verisinde kritik eşik aşılırsa çalışır.
    kritik_depremler = [
        d for d in tum_depremler
        if float(d.get("mag", 0)) >= 4.5
    ]

    if kritik_depremler:
        deprem_alarm_var = True
        en_kritik = kritik_depremler[0]
        alarm_mesaji = (
            f"{en_kritik.get('title', 'Bilinmeyen Konum')} bölgesinde "
            f"{en_kritik.get('mag', '?')} büyüklüğünde deprem tespit edildi."
        )

    if ust_depremler:
        deprem_ozeti = " | ".join(
            [f"{d.get('title', '?')} ({d.get('mag', '?')})" for d in ust_depremler[:5]]
        )
    else:
        deprem_ozeti = "Son 4+ büyüklüğünde deprem bulunamadı."

    if request.method == "POST":
        try:
            secilen_sehir = request.form.get("sehir", "")
            secilen_ilce = request.form.get("ilce", "")
            secilen_mahalle = request.form.get("mahalle", "")

            zemin_bilgisi = zemin_bilgisi_getir(
                zemin_df,
                secilen_sehir,
                secilen_ilce,
                secilen_mahalle
            )

            zemin_riski = float(zemin_bilgisi.get("risk", 5))

            inputs = [
                float(request.form.get(x, 0))
                for x in ['n', 'b', 'y', 't', 'i']
            ]

            inputs.append(zemin_riski)

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
                    1: ["Orta Riskli", "#f39c12"],
                    2: ["Kritik / Riskli", "#e74c3c"]
                }[res]

                risk_skoru = risk_skoru_getir(risk_durumu)

                tahmin_sonucu = risk_durumu

                if secilen_sehir:
                    konum_metni = secilen_sehir

                    if secilen_ilce:
                        konum_metni = f"{secilen_sehir} / {secilen_ilce}"

                    if secilen_mahalle:
                        konum_metni = f"{konum_metni} / {secilen_mahalle}"

                    tahmin_sonucu = f"{konum_metni} için sonuç: {risk_durumu}"

                oneriler = acil_oneriler_uret(risk_durumu, inputs)

                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_

                    feats = [
                        'Nüfus',
                        'Bina Yaşı',
                        'Yatak',
                        'Toplanma',
                        'İtfaiye',
                        'Zemin'
                    ]

                    pairs = sorted(
                        zip(feats, imp),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    aciklama = "<br>".join(
                        [f"{f}: %{round(i * 100, 1)} etkili" for f, i in pairs]
                    )

            else:
                tahmin_sonucu = "Model dosyası bulunamadı."

        except Exception as e:
            tahmin_sonucu = "Veri hatası!"
            print("Model hata:", e)

    m = folium.Map(
        location=[39, 35],
        zoom_start=6,
        tiles="cartodbpositron"
    )

    sehirleri_renklendir(
        m,
        secilen_sehir,
        risk_skoru,
        risk_durumu
    )

    for d in tum_depremler:
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
                popup=f"{d.get('title', '?')} - {mag}"
            ).add_to(m)

        except Exception:
            continue

    folium.LayerControl().add_to(m)

    map_html = m._repr_html_()

    sehir_options = ""

    for sehir in sehirler:
        selected = "selected" if sehir == secilen_sehir else ""
        sehir_options += f'<option value="{sehir}" {selected}>{sehir}</option>'

    ilce_options = '<option value="">Önce şehir seçiniz</option>'

    if secilen_sehir and secilen_sehir in ilce_verileri:
        ilce_options = '<option value="">İlçe seçiniz</option>'

        for ilce in ilce_verileri[secilen_sehir]:
            selected = "selected" if ilce == secilen_ilce else ""
            ilce_options += f'<option value="{ilce}" {selected}>{ilce}</option>'

    mahalle_options = '<option value="">Önce ilçe seçiniz</option>'

    if secilen_sehir and secilen_ilce:
        mahalle_anahtar = f"{secilen_sehir}|||{secilen_ilce}"

        if mahalle_anahtar in mahalle_verileri:
            mahalle_options = '<option value="">Mahalle seçiniz</option>'

            for mahalle in mahalle_verileri[mahalle_anahtar]:
                selected = "selected" if mahalle == secilen_mahalle else ""
                mahalle_options += f'<option value="{mahalle}" {selected}>{mahalle}</option>'

    ilce_verileri_json = json.dumps(ilce_verileri, ensure_ascii=False)
    mahalle_verileri_json = json.dumps(mahalle_verileri, ensure_ascii=False)

    if zemin_bilgisi:
        zemin_bilgisi_html = f"""
            <div class="zemin-info-box">
                <h3>🌍 Zemin Bilgisi</h3>
                <p><b>Zemin Türü:</b> {zemin_bilgisi.get('tip', 'Belirtilmemiş')}</p>
                <p><b>Tahmini Zemin Riski:</b> {zemin_bilgisi.get('risk', '?')}/10</p>
                <p>{zemin_bilgisi.get('aciklama', '')}</p>
            </div>
        """
    else:
        zemin_bilgisi_html = ""

    oneriler_html = "".join([f"<li>{o}</li>" for o in oneriler])

    if tum_depremler:
        deprem_listesi_html = "".join([
            f"<li>{d.get('title', 'Bilinmeyen Konum')} - Büyüklük: {d.get('mag', '?')} - Kaynak: {d.get('kaynak', 'Bilinmiyor')}</li>"
            for d in tum_depremler[:15]
        ])
    else:
        deprem_listesi_html = "<li>Güncel deprem verisi alınamadı.</li>"

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

            input,
            select {{
                padding:10px;
                margin:5px;
                border-radius:8px;
                border:1px solid #ccc;
            }}

            label {{
                display:block;
                margin-top:10px;
                font-weight:bold;
            }}

            small {{
                display:block;
                color:#555;
                margin:0 5px 8px 5px;
                line-height:1.4;
            }}

            .zemin-info-box {{
                margin-top:15px;
                padding:15px;
                background:#eef7ff;
                color:#0b3954;
                border-left:6px solid #3498db;
                border-radius:10px;
                line-height:1.5;
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

            .risk-score-box {{
                margin-top:15px;
                padding:15px;
                border-radius:10px;
                background:#f8f9fa;
                color:#222;
                text-align:center;
                font-size:20px;
                font-weight:bold;
                border:3px solid {risk_rengi};
            }}

            .earthquake-list {{
                margin-top:15px;
                padding:15px;
                background:#f8f9fa;
                color:#222;
                border-radius:10px;
                line-height:1.6;
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
                animation:flash 0.7s infinite;
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

                input,
                select,
                button {{
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

        <div
            id="emergencyAlert"
            class="emergency-alert"
            role="alertdialog"
            aria-live="assertive"
        >
            <h1>🚨 ACİL DURUM</h1>

            <p>{alarm_mesaji if alarm_mesaji else "Canlı deprem verisi kritik seviyeye ulaştı."}</p>

            <p>
                Güvenli alana geçin.
                Asansör kullanmayın.
                Toplanma alanına yönelin.
            </p>

            <button
                class="close-alert"
                onclick="acilDurumKapat()"
            >
                Uyarıyı Kapat
            </button>
        </div>

        <h1>RiskAtlas: AI Destekli Afet Risk Analiz Platformu</h1>

        <h2>
            🔴 4.0+ Canlı Deprem Uyarıları:
            {deprem_ozeti}
        </h2>

        <div class="box">
            {map_html}

            <div class="earthquake-list" aria-label="Canlı deprem listesi">
                <h3>📋 Tüm Güncel Deprem Listesi</h3>
                <ul>
                    {deprem_listesi_html}
                </ul>
            </div>
        </div>

        <div class="box">

            <form method="POST">

                <label for="sehir">Şehir Seçiniz</label>
                <select id="sehir" name="sehir" required onchange="ilceleriGuncelle()">
                    <option value="">Şehir seçiniz</option>
                    {sehir_options}
                </select>

                <label for="ilce">İlçe Seçiniz</label>
                <select id="ilce" name="ilce" onchange="mahalleleriGuncelle()">
                    {ilce_options}
                </select>

                <label for="mahalle">Mahalle Seçiniz</label>
                <select id="mahalle" name="mahalle">
                    {mahalle_options}
                </select>

                <label for="n">Yaşadığınız Bölgedeki Tahmini Nüfus Yoğunluğu</label>
                <input
                    id="n"
                    type="number"
                    step="any"
                    name="n"
                    placeholder="Örn: 5000 kişi/km²"
                    required
                >
                <small>
                    Bu değer binada yaşayan kişi sayısını değil, bulunduğunuz mahalle veya ilçedeki genel nüfus yoğunluğunu temsil eder.
                </small>

                <label for="b">Bina Yaşı</label>
                <input
                    id="b"
                    type="number"
                    step="any"
                    name="b"
                    placeholder="Örn: 20"
                    required
                >

                <label for="y">Yatak Kapasitesi</label>
                <input
                    id="y"
                    type="number"
                    step="any"
                    name="y"
                    placeholder="Örn: 1000"
                    required
                >

                <label for="t">Toplanma Alanı</label>
                <input
                    id="t"
                    type="number"
                    step="any"
                    name="t"
                    placeholder="Örn: 50000"
                    required
                >

                <label for="i">İtfaiye Gücü</label>
                <input
                    id="i"
                    type="number"
                    step="any"
                    name="i"
                    placeholder="Örn: 50"
                    required
                >

                <div class="zemin-info-box">
                    <b>🌍 Zemin Riski:</b><br>
                    Zemin riski kullanıcıdan istenmez. Seçilen şehir, ilçe ve mahalle bilgisine göre sistem tarafından otomatik değerlendirilir.
                </div>

                <br><br>

                <button type="submit">
                    Analiz Et
                </button>

            </form>

            <div class="example-box">
                <b>📌 Örnek Değer Rehberi:</b><br><br>
                • <b>Yaşadığınız Bölgedeki Tahmini Nüfus Yoğunluğu:</b> 5000 kişi/km² → bulunduğunuz mahalle veya ilçedeki genel yoğunluğu temsil eder.<br>
                • <b>Bina Yaşı:</b> 20 → bölgedeki ortalama bina yaşı gibi düşünülmelidir.<br>
                • <b>Yatak Kapasitesi:</b> 1000 → hastane/acil durum kapasitesini temsil eder.<br>
                • <b>Toplanma Alanı:</b> 50000 → m² cinsinden düşünülebilir; yüksek değer daha avantajlıdır.<br>
                • <b>İtfaiye Gücü:</b> 50 → ekip, araç veya müdahale kapasitesi gibi düşünülebilir.<br>
                • <b>Zemin Riski:</b> kullanıcı tarafından girilmez; seçilen bölgeye göre sistem tarafından otomatik kullanılır.
            </div>

            <h2
                style="color:{risk_rengi};"
                aria-live="assertive"
                role="alert"
            >
                {tahmin_sonucu}
            </h2>

            {f'''
            <div class="risk-score-box">
                Risk Skoru: {risk_skoru}/5
            </div>
            ''' if risk_skoru > 0 else ""}

            {zemin_bilgisi_html}

            <p>{aciklama}</p>

            <button
                type="button"
                onclick="acilDurumGoster()"
            >
                🚨 Erişilebilir Acil Durum Alarmını Test Et
            </button>

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
                ✅ Harita altında ekran okuyucu uyumlu deprem listesi<br>
                ✅ Risk sonucuna göre renklendirilen şehir haritası<br>
                ✅ Büyük yazı ve yüksek kontrastlı acil durum ekranı
            </div>
</div>

        <script>
            if ("serviceWorker" in navigator) {{
                navigator.serviceWorker.register("/static/service-worker.js")
                .then(() => console.log("Service Worker kayıt edildi."))
                .catch(error => console.log("Service Worker hatası:", error));
            }}

            const ilceVerileri = {ilce_verileri_json};
            const mahalleVerileri = {mahalle_verileri_json};

            function ilceleriGuncelle() {{
                const sehirSelect = document.getElementById("sehir");
                const ilceSelect = document.getElementById("ilce");
                const mahalleSelect = document.getElementById("mahalle");

                if (!sehirSelect || !ilceSelect) {{
                    return;
                }}

                const secilenSehir = sehirSelect.value;
                const ilceler = (ilceVerileri[secilenSehir] || []).slice().sort((a, b) => a.localeCompare(b, "tr"));

                ilceSelect.innerHTML = "";

                if (mahalleSelect) {{
                    mahalleSelect.innerHTML = "";
                    const mahalleOption = document.createElement("option");
                    mahalleOption.value = "";
                    mahalleOption.textContent = "Önce ilçe seçiniz";
                    mahalleSelect.appendChild(mahalleOption);
                }}

                if (!secilenSehir) {{
                    const option = document.createElement("option");
                    option.value = "";
                    option.textContent = "Önce şehir seçiniz";
                    ilceSelect.appendChild(option);
                    return;
                }}

                if (ilceler.length === 0) {{
                    const option = document.createElement("option");
                    option.value = "";
                    option.textContent = "İlçe verisi bulunamadı";
                    ilceSelect.appendChild(option);
                    return;
                }}

                const ilkOption = document.createElement("option");
                ilkOption.value = "";
                ilkOption.textContent = "İlçe seçiniz";
                ilceSelect.appendChild(ilkOption);

                ilceler.forEach(function(ilce) {{
                    const option = document.createElement("option");
                    option.value = ilce;
                    option.textContent = ilce;
                    ilceSelect.appendChild(option);
                }});
            }}

            function mahalleleriGuncelle() {{
                const sehirSelect = document.getElementById("sehir");
                const ilceSelect = document.getElementById("ilce");
                const mahalleSelect = document.getElementById("mahalle");

                if (!sehirSelect || !ilceSelect || !mahalleSelect) {{
                    return;
                }}

                const anahtar = sehirSelect.value + "|||" + ilceSelect.value;
                const mahalleler = (mahalleVerileri[anahtar] || []).slice().sort((a, b) => a.localeCompare(b, "tr"));

                mahalleSelect.innerHTML = "";

                if (mahalleler.length === 0) {{
                    const option = document.createElement("option");
                    option.value = "";
                    option.textContent = "Mahalle verisi bulunamadı";
                    mahalleSelect.appendChild(option);
                    return;
                }}

                const ilkOption = document.createElement("option");
                ilkOption.value = "";
                ilkOption.textContent = "Mahalle seçiniz";
                mahalleSelect.appendChild(ilkOption);

                mahalleler.forEach(function(mahalle) {{
                    const option = document.createElement("option");
                    option.value = mahalle;
                    option.textContent = mahalle;
                    mahalleSelect.appendChild(option);
                }});
            }}

            function sesliUyariVer() {{
                if ("speechSynthesis" in window) {{
                    const mesaj = new SpeechSynthesisUtterance(
                        "Dikkat. Canlı deprem verisinde kritik seviyede deprem tespit edildi. Güvenli alana geçin. Asansör kullanmayın. Toplanma alanına yönelin."
                    );

                    mesaj.lang = "tr-TR";
                    mesaj.rate = 0.9;
                    mesaj.pitch = 1;

                    window.speechSynthesis.cancel();

                    setTimeout(() => {{
                        window.speechSynthesis.speak(mesaj);
                    }}, 200);
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

            window.onload = function () {{
                const depremAlarmVar = "{deprem_alarm_var}" === "True";

                if (depremAlarmVar) {{
                    setTimeout(() => {{
                        acilDurumGoster();
                    }}, 1000);
                }}
            }};
        </script>

    </body>
    </html>
    """

    return make_response(html)


if __name__ == "__main__":
    app.run(debug=True)
