from flask import Flask, request, make_response
import pandas as pd
import joblib
import os
import folium
import requests
import json
import unicodedata
import sqlite3
from datetime import datetime
import time

app = Flask(__name__)

base = os.path.dirname(os.path.abspath(__file__))

data_yolu = os.path.join(base, 'datasets', 'processed_afet_verisi.csv')
db_yolu = os.path.join(base, 'datasets', 'afet_veritabani.db')
model_yolu = os.path.join(base, 'models', 'afet_model.pkl')
geojson_yolu = os.path.join(base, 'datasets', 'turkey_provinces.geojson')
ilce_yolu = os.path.join(base, 'datasets', 'turkey_districts.csv')
zemin_yolu = os.path.join(base, 'datasets', 'zemin_verileri.csv')


# SQLite veritabanı ve analiz kayıt tablosu otomatik oluşturulur.
def veritabani_olustur():
    try:
        os.makedirs(os.path.dirname(db_yolu), exist_ok=True)

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
        print("Veritabanı hazır: analiz_kayitlari tablosu kontrol edildi.")

    except Exception as e:
        print("Veritabanı oluşturma hatası:", e)


veritabani_olustur()


def normalize_text(text):
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if not unicodedata.combining(c))


def turkce_sirala(liste):
    """Türkçe karakterleri dikkate alarak alfabetik sıralama yapar."""
    return sorted(liste, key=lambda x: normalize_text(x))


def gorunum_duzelt(text):
    """KONYA / konya gibi değerleri kullanıcıya Konya biçiminde gösterir."""
    text = str(text).strip()
    if not text:
        return ""
    return text.title()


def tekil_ve_sirali(liste):
    """Büyük/küçük harf farkından doğan tekrarları temizler ve Türkçe uyumlu sıralar."""
    temiz = {}

    for item in liste:
        item = str(item).strip()
        if not item or item.lower() == "nan":
            continue

        anahtar = normalize_text(item)

        if anahtar not in temiz:
            temiz[anahtar] = gorunum_duzelt(item)

    return turkce_sirala(list(temiz.values()))


def afad_depremleri_getir():
    try:
        url = "https://deprem.afad.gov.tr/apiv2/event/latest"
        r = requests.get(url, timeout=3)

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
        r = requests.get(url, timeout=3)

        if r.status_code == 200:
            depremler = r.json().get("result", [])

            for d in depremler:
                d["kaynak"] = "Kandilli"

            return depremler[:30]

    except Exception as e:
        print("Kandilli API hatası:", e)

    return []


DEPREM_CACHE = {
    "zaman": 0,
    "veri": []
}

def canlı_depremleri_getir():
    """
    Canlı deprem verisini her sayfa açılışında tekrar tekrar çekmek siteyi yavaşlatıyordu.
    Bu yüzden veriler 5 dakika boyunca cache içinde tutulur.
    """
    simdi = time.time()

    if DEPREM_CACHE["veri"] and simdi - DEPREM_CACHE["zaman"] < 300:
        return DEPREM_CACHE["veri"]

    # Önce AFAD verisi alınır. AFAD çalışmazsa Kandilli yedek kaynak olarak kullanılır.
    afad = afad_depremleri_getir()

    if afad:
        DEPREM_CACHE["veri"] = afad
        DEPREM_CACHE["zaman"] = simdi
        return afad

    kandilli = kandilli_depremleri_getir()
    DEPREM_CACHE["veri"] = kandilli
    DEPREM_CACHE["zaman"] = simdi
    return kandilli


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


@app.route("/healthz")
def healthz():
    return "OK", 200


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
    analiz_yapildi = False

    # Şehir listesi önce SQLite veritabanından alınır.
    # Veritabanı yoksa eski CSV dosyası yedek olarak kullanılır.
    if os.path.exists(db_yolu):
        try:
            conn = sqlite3.connect(db_yolu)
            df = pd.read_sql_query("SELECT * FROM afet_verileri", conn)
            conn.close()

            if "Sehir" in df.columns:
                sehirler = tekil_ve_sirali(df["Sehir"].dropna().unique().tolist())

        except Exception as e:
            print("Veritabanı okuma hatası:", e)

    # Veritabanında afet_verileri tablosu yoksa veya şehir listesi boşsa CSV yedek olarak kullanılır.
    if not sehirler and os.path.exists(data_yolu):
        try:
            df = pd.read_csv(data_yolu, encoding="utf-8-sig")

            if "Sehir" in df.columns:
                sehirler = tekil_ve_sirali(df["Sehir"].dropna().unique().tolist())

        except Exception as e:
            print("CSV veri okuma hatası:", e)

    # İlçe CSV dosyası sonra eklenecek. Dosya yoksa sistem bozulmadan çalışır.
    if os.path.exists(ilce_yolu):
        try:
            ilce_df = pd.read_csv(ilce_yolu, encoding="utf-8-sig")

            if "Sehir" in ilce_df.columns and "Ilce" in ilce_df.columns:
                for sehir, grup in ilce_df.groupby("Sehir"):
                    sehir_temiz = gorunum_duzelt(sehir)
                    ilce_verileri[sehir_temiz] = tekil_ve_sirali(grup["Ilce"].dropna().unique().tolist())

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
                    sehir_temiz = gorunum_duzelt(sehir)
                    ilce_verileri[sehir_temiz] = tekil_ve_sirali(list(mevcut_ilceler.union(yeni_ilceler)))

            if all(kolon in zemin_df.columns for kolon in ["Sehir", "Ilce", "Mahalle"]):
                for (sehir, ilce), grup in zemin_df.groupby(["Sehir", "Ilce"]):
                    anahtar = f"{sehir}|||{ilce}"
                    sehir_temiz = gorunum_duzelt(sehir)
                    ilce_temiz = gorunum_duzelt(ilce)
                    anahtar = f"{sehir_temiz}|||{ilce_temiz}"
                    mahalle_verileri[anahtar] = tekil_ve_sirali(grup["Mahalle"].dropna().unique().tolist())

        except Exception as e:
            print("Zemin CSV okuma hatası:", e)

    # Şehir listesi yalnızca model/veritabanı verisinden gelirse 81 ilin tamamı görünmeyebilir.
    # Bu nedenle ilçe ve zemin CSV dosyalarındaki şehirler de listeye eklenir.
    tum_sehirler = set(sehirler)
    tum_sehirler.update(ilce_verileri.keys())

    if zemin_df is not None and not zemin_df.empty and "Sehir" in zemin_df.columns:
        tum_sehirler.update(zemin_df["Sehir"].dropna().unique().tolist())

    sehirler = tekil_ve_sirali(list(tum_sehirler))

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
        analiz_yapildi = True
        try:
            secilen_sehir = gorunum_duzelt(request.form.get("sehir", ""))
            secilen_ilce = gorunum_duzelt(request.form.get("ilce", ""))
            secilen_mahalle = gorunum_duzelt(request.form.get("mahalle", ""))

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

                # Analiz sonucu SQLite veritabanına kaydedilir
                try:
                    conn = sqlite3.connect(db_yolu)
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT INTO analiz_kayitlari (
                            sehir,
                            ilce,
                            mahalle,
                            risk_sonucu,
                            risk_skoru,
                            zemin_riski,
                            tarih
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        secilen_sehir,
                        secilen_ilce,
                        secilen_mahalle,
                        risk_durumu,
                        risk_skoru,
                        zemin_riski,
                        datetime.now().strftime("%d.%m.%Y %H:%M")
                    ))

                    conn.commit()
                    conn.close()

                    print("Analiz veritabanına kaydedildi.")

                except Exception as db_hata:
                    print("Veritabanı kayıt hatası:", db_hata)

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

    deprem_verileri_json = json.dumps(tum_depremler[:30], ensure_ascii=False)

    html = f"""
    <!DOCTYPE html>
    <html lang="tr">

    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="theme-color" content="#081120">

        <script>
            // PWA/uygulama icon ayarları sadece telefonda aktif olur.
            // Böylece bilgisayarda site normal web sitesi gibi açılır.
            if (/Android|iPhone|iPad|iPod/i.test(navigator.userAgent)) {{

                const manifest = document.createElement("link");
                manifest.rel = "manifest";
                manifest.href = "/static/manifest.json";
                document.head.appendChild(manifest);

                const appleIcon = document.createElement("link");
                appleIcon.rel = "apple-touch-icon";
                appleIcon.href = "/static/icons/icon-192.png";
                document.head.appendChild(appleIcon);

                const appleCapable = document.createElement("meta");
                appleCapable.name = "apple-mobile-web-app-capable";
                appleCapable.content = "yes";
                document.head.appendChild(appleCapable);

                const appleStatus = document.createElement("meta");
                appleStatus.name = "apple-mobile-web-app-status-bar-style";
                appleStatus.content = "black-translucent";
                document.head.appendChild(appleStatus);
            }}
        </script>

        <style>
            :root {{
                --bg1:#06111f;
                --bg2:#0b1e35;
                --card:rgba(12, 29, 52, 0.88);
                --card2:rgba(17, 41, 72, 0.88);
                --text:#eaf4ff;
                --muted:#a8c5df;
                --blue:#2f89ff;
                --blue2:#00c2ff;
                --danger:#ff3b3b;
                --border:rgba(95, 177, 255, 0.25);
            }}



            /* Splash ekranı masaüstünde görünmez; sadece telefon ekranında açılır. */
            #splash-screen {{
                display:none;
            }}

            #splash-screen.fade-out {{
                opacity:0;
                pointer-events:none;
            }}

            .splash-image {{
                width:100%;
                height:100%;
                object-fit:cover;
            }}

            body {{
                background:
                    radial-gradient(circle at top left, rgba(0,194,255,0.18), transparent 28%),
                    radial-gradient(circle at bottom right, rgba(47,137,255,0.20), transparent 30%),
                    linear-gradient(135deg, var(--bg1), var(--bg2));
                color:var(--text);
                font-family:Arial, sans-serif;
                margin:0;
                padding:15px;
            }}

            h1, h2 {{
                text-align:center;
            }}

            .box {{
                background:var(--card);
                color:var(--text);
                padding:22px;
                border-radius:18px;
                margin:14px auto;
                max-width:1200px;
                box-shadow:0 18px 45px rgba(0,0,0,0.35);
                border:1px solid var(--border);
                backdrop-filter: blur(10px);
            }}

            input,
            select {{
                padding:12px;
                margin:5px;
                border-radius:10px;
                border:1px solid var(--border);
                background:#0b1b31;
                color:var(--text);
            }}

            label {{
                display:block;
                margin-top:10px;
                font-weight:bold;
                color:#dceeff;
            }}

            small {{
                display:block;
                color:var(--muted);
                margin:0 5px 8px 5px;
                line-height:1.4;
            }}

            .zemin-info-box {{
                margin-top:15px;
                padding:15px;
                background:rgba(0,194,255,0.10);
                color:#dff6ff;
                border-left:6px solid var(--blue2);
                border-radius:12px;
                line-height:1.5;
            }}

            button {{
                padding:12px 16px;
                background:linear-gradient(135deg, var(--blue), var(--blue2));
                color:white;
                border:none;
                border-radius:10px;
                cursor:pointer;
                font-weight:bold;
                box-shadow:0 10px 25px rgba(0, 194, 255, 0.18);
            }}

            button:hover {{
                filter:brightness(1.08);
            }}

            .risk-score-box {{
                margin-top:15px;
                padding:15px;
                border-radius:12px;
                background:rgba(255,255,255,0.08);
                color:var(--text);
                text-align:center;
                font-size:20px;
                font-weight:bold;
                border:3px solid {risk_rengi};
            }}

            .earthquake-list {{
                margin-top:15px;
                padding:15px;
                background:var(--card2);
                color:var(--text);
                border-radius:14px;
                line-height:1.6;
                border:1px solid var(--border);
            }}

            .example-box {{
                margin-top:15px;
                padding:15px;
                background:rgba(255,255,255,0.08);
                color:#dceeff;
                border-radius:12px;
                font-size:0.92em;
                line-height:1.6;
            }}

            .suggestion-box {{
                margin-top:20px;
                padding:18px;
                background:rgba(39,174,96,0.14);
                color:#d8ffe8;
                border-left:6px solid #27ae60;
                border-radius:12px;
                line-height:1.5;
            }}

            .accessibility-note {{
                margin-top:20px;
                padding:15px;
                background:rgba(0,194,255,0.10);
                color:#dff6ff;
                border-radius:12px;
                line-height:1.5;
            }}

            .landing-screen {{
                min-height:100vh;
                position:relative;
                overflow:hidden;
                border-radius:24px;
                margin:0 auto 20px auto;
                max-width:1250px;
                background:
                    linear-gradient(180deg, rgba(3,10,22,0.18), rgba(3,10,22,0.78)),
                    url('/static/riskatlas-bg.png');
                background-size:cover;
                background-position:center;
                border:1px solid var(--border);
                box-shadow:0 25px 60px rgba(0,0,0,0.45);
                padding:28px;
                box-sizing:border-box;
            }}

            .landing-topbar {{
                display:flex;
                justify-content:space-between;
                align-items:flex-start;
                gap:18px;
                position:relative;
                z-index:2;
            }}

            .brand {{
                display:flex;
                align-items:center;
                gap:12px;
            }}

            .brand-icon {{
                width:54px;
                height:54px;
                border-radius:16px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:linear-gradient(135deg, #ef233c, #9d0208);
                box-shadow:0 10px 25px rgba(255,59,59,0.28);
                font-size:28px;
            }}

            .brand-title {{
                font-size:30px;
                font-weight:800;
                letter-spacing:-0.5px;
            }}

            .brand-title span {{
                color:#ff3b3b;
            }}

            .brand-subtitle {{
                color:#d7eaff;
                font-size:14px;
                margin-top:2px;
            }}

            .top-actions {{
                display:flex;
                flex-wrap:wrap;
                gap:10px;
                justify-content:flex-end;
            }}

            .voice-toggle-btn {{
                background:rgba(39,174,96,0.22) !important;
                border:1px solid rgba(39,174,96,0.55) !important;
            }}

            .voice-toggle-btn.off {{
                background:rgba(255,255,255,0.10) !important;
                border:1px solid rgba(255,255,255,0.22) !important;
                color:#cfd8e3;
            }}

            .top-actions button {{
                background:rgba(8,24,43,0.72);
                border:1px solid var(--border);
                box-shadow:none;
                padding:10px 14px;
            }}

            .landing-center {{
                min-height:62vh;
                display:flex;
                align-items:center;
                justify-content:center;
                position:relative;
                z-index:2;
            }}

            .landing-panel {{
                width:min(560px, 92%);
                padding:32px;
                background:rgba(7,17,33,0.86);
                border:1px solid var(--border);
                border-radius:24px;
                backdrop-filter:blur(14px);
                text-align:center;
                box-shadow:0 20px 60px rgba(0,0,0,0.42);
            }}

            .landing-panel h1 {{
                text-align:center;
                font-size:36px;
                margin:0 0 8px 0;
                letter-spacing:-0.5px;
            }}

            .landing-panel h1 span {{
                color:#ff3b3b;
            }}

            .landing-panel h2 {{
                text-align:center;
                margin:18px 0 8px 0;
                font-size:22px;
            }}

            .landing-panel p {{
                color:#d7eaff;
                font-size:16px;
                line-height:1.55;
                margin:8px 0;
            }}

            .location-symbol {{
                margin:22px auto 14px auto;
                width:92px;
                height:92px;
                border-radius:50%;
                display:flex;
                align-items:center;
                justify-content:center;
                background:radial-gradient(circle, rgba(47,137,255,0.28), rgba(0,194,255,0.06));
                border:1px solid rgba(95,177,255,0.35);
                font-size:54px;
                box-shadow:0 0 45px rgba(47,137,255,0.28);
            }}

            .landing-actions {{
                display:flex;
                flex-wrap:wrap;
                justify-content:center;
                gap:12px;
                margin-top:22px;
            }}

            .landing-actions button {{
                min-width:170px;
                font-size:16px;
            }}

            .secondary-btn {{
                background:rgba(255,255,255,0.10);
                border:1px solid var(--border);
            }}

            .status-box {{
                margin-top:16px;
                padding:14px;
                border-radius:12px;
                background:rgba(255,255,255,0.08);
                color:#dff6ff;
                line-height:1.5;
            }}

            .side-card {{
                position:absolute;
                z-index:2;
                width:250px;
                padding:18px;
                border-radius:16px;
                background:rgba(7,17,33,0.72);
                border:1px solid var(--border);
                backdrop-filter:blur(10px);
                line-height:1.5;
                box-shadow:0 18px 45px rgba(0,0,0,0.28);
            }}

            .side-card h3 {{
                margin:0 0 8px 0;
            }}

            .side-card p,
            .side-card li {{
                color:#d7eaff;
                font-size:14px;
            }}

            .side-card.left {{
                left:28px;
                bottom:145px;
                border-color:rgba(255,59,59,0.42);
            }}

            .side-card.right {{
                right:28px;
                bottom:145px;
                border-color:rgba(39,174,96,0.46);
            }}

            .side-card ul {{
                margin:8px 0 0 0;
                padding-left:22px;
            }}

            .bottom-features {{
                position:absolute;
                z-index:2;
                left:28px;
                right:28px;
                bottom:28px;
                display:grid;
                grid-template-columns:repeat(4, 1fr);
                gap:12px;
                padding:14px;
                background:rgba(7,17,33,0.64);
                border:1px solid var(--border);
                border-radius:18px;
                backdrop-filter:blur(10px);
            }}

            .feature-item {{
                display:flex;
                gap:12px;
                align-items:flex-start;
                padding:10px;
                border-right:1px solid rgba(95,177,255,0.16);
            }}

            .feature-item:last-child {{
                border-right:none;
            }}

            .feature-icon {{
                width:42px;
                height:42px;
                border-radius:50%;
                display:flex;
                align-items:center;
                justify-content:center;
                background:rgba(47,137,255,0.20);
                font-size:22px;
                flex-shrink:0;
            }}

            .feature-item b {{
                display:block;
                margin-bottom:4px;
            }}

            .feature-item span {{
                color:#c8dff2;
                font-size:14px;
                line-height:1.4;
            }}

            .main-content {{
                display:none;
            }}

            .main-content.active {{
                display:block;
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
                #splash-screen {{
                    position:fixed;
                    inset:0;
                    background:#081120;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    z-index:999999;
                    transition:opacity .8s ease;
                }}

                body {{
                    padding:8px;
                    background:
                        linear-gradient(180deg, rgba(3,10,22,0.35), rgba(3,10,22,0.88)),
                        url('/static/mobile-bg.png');
                    background-size: cover;
                    background-position: center top;
                    background-repeat: no-repeat;
                    background-attachment: scroll;
                }}

                .landing-screen {{
                    background:
                        linear-gradient(180deg, rgba(3,10,22,0.25), rgba(3,10,22,0.82)),
                        url('/static/mobile-bg.png');
                    background-size: cover;
                    background-position: center top;
                    background-repeat: no-repeat;
                    border-radius: 22px;
                    overflow: hidden;
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

                .landing-screen {{
                    min-height:auto;
                    padding:14px;
                }}

                .landing-topbar {{
                    flex-direction:column;
                }}

                .top-actions {{
                    justify-content:flex-start;
                }}

                .top-actions button {{
                    width: 100%;
                    font-size: 16px;
                    padding: 14px;
                }}

                .landing-center {{
                    min-height:auto;
                    padding:28px 0;
                }}

                .landing-panel {{
                    padding:26px 20px;
                    width:100%;
                    border-radius:28px;
                }}

                .landing-panel h1 {{
                    font-size:42px;
                    line-height:1.15;
                }}

                .side-card {{
                    position:static;
                    width:auto;
                    margin:12px 0;
                }}

                .bottom-features {{
                    position:static;
                    grid-template-columns:1fr;
                    margin-top:12px;
                }}

                .feature-item {{
                    border-right:none;
                    border-bottom:1px solid rgba(95,177,255,0.16);
                }}

                .feature-item:last-child {{
                    border-bottom:none;
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

        <div id="splash-screen">
            <img src="/static/splash/splash.png" alt="RiskAtlas Açılış Ekranı" class="splash-image">
        </div>

        <section class="landing-screen" id="landingScreen">
            <div class="landing-topbar">
                <div class="brand" aria-label="RiskAtlas logo ve başlık">
                    <div class="brand-icon">〽️</div>
                    <div>
                        <div class="brand-title">Risk<span>Atlas</span></div>
                        <div class="brand-subtitle">Deprem Risk Analiz ve Uyarı Sistemi</div>
                    </div>
                </div>

                <div class="top-actions">
                    <button type="button" onclick="girisSesliAciklama('manual')" aria-label="Erişilebilir sesli rehberi başlat">
                        ♿ Erişilebilir Sesli Rehber
                    </button>
                    <button type="button" id="voiceToggleButton" class="voice-toggle-btn" onclick="sesliYonlendirmeAyariniDegistir()" aria-label="Sesli yönlendirme ayarını aç veya kapat">
                        🔊 Sesli Yönlendirme: Açık
                    </button>
                    <button type="button" onclick="detayliAnalizeGec()" aria-label="Detaylı analiz ekranına geç">
                        ⚙️ Analiz Ekranı
                    </button>
                </div>
            </div>

            <div class="landing-center">
                <div class="landing-panel" role="region" aria-label="RiskAtlas giriş ve konum modu">
                    <h1>Risk<span>Atlas</span>'a Hoş Geldiniz</h1>
                    <p>
                        Konumunuza göre deprem risklerini analiz eder, size özel uyarılar ve öneriler sunar.
                    </p>

                    <div class="location-symbol" aria-hidden="true">📍</div>

                    <h2>Güvenliğiniz için konumunuzu kullanıyoruz.</h2>
                    <p>
                        4.5 ve üzeri depremlerde sadece bulunduğunuz bölge etkileniyorsa sizi uyarır,
                        uzak depremler için gereksiz alarm vermez.
                    </p>

                    <div class="landing-actions">
                        <button
                            type="button"
                            onclick="konumModunuBaslat()"
                            aria-label="Konumumu kullan ve yakın deprem uyarılarını başlat"
                        >
                            📍 Konumumu Kullan
                        </button>

                        <button
                            type="button"
                            class="secondary-btn"
                            onclick="detayliAnalizeGec()"
                            aria-label="Konum kullanmadan şehir seçerek detaylı analiz ekranına geç"
                        >
                            Şehir Seçerek Devam Et
                        </button>

                        <button
                            type="button"
                            class="secondary-btn"
                            onclick="girisSesliAciklama('manual')"
                            aria-label="Giriş ekranındaki erişilebilir sesli rehberi başlat"
                        >
                            ♿ Sesli Rehberi Başlat
                        </button>
                    </div>

                    <div class="status-box" id="konumDurumu" aria-live="polite">
                        Konum modu henüz başlatılmadı.
                    </div>
                </div>
            </div>

            <div class="side-card left">
                <h3>🚨 Neden Konum İzni?</h3>
                <p>
                    Size en doğru deprem uyarılarını sunabilmek için bulunduğunuz konuma ihtiyaç duyarız.
                    Sadece yakınınızdaki risklerde sizi uyarırız.
                </p>
            </div>

            <div class="side-card right">
                <h3>♿ Erişilebilir Özellikler</h3>
                <ul>
                    <li>Sesli yönlendirme</li>
                    <li>Ekran okuyucu uyumu</li>
                    <li>Büyük yazı ve yüksek kontrast</li>
                    <li>Titreşimli uyarılar</li>
                </ul>
            </div>

            <div class="bottom-features">
                <div class="feature-item">
                    <div class="feature-icon">🎯</div>
                    <div>
                        <b>Konuma Dayalı Uyarı</b>
                        <span>Sadece size yakın depremlerde uyarı alın.</span>
                    </div>
                </div>

                <div class="feature-item">
                    <div class="feature-icon">🔔</div>
                    <div>
                        <b>Gerçek Zamanlı Bildirim</b>
                        <span>4.5+ depremlerde sesli, görsel ve titreşimli uyarı.</span>
                    </div>
                </div>

                <div class="feature-item">
                    <div class="feature-icon">🛡️</div>
                    <div>
                        <b>Güvenilir Kaynaklar</b>
                        <span>AFAD ve Kandilli verileri kullanılır.</span>
                    </div>
                </div>

                <div class="feature-item">
                    <div class="feature-icon">👥</div>
                    <div>
                        <b>Herkes İçin Erişilebilir</b>
                        <span>Engelli bireyler düşünülerek tasarlandı.</span>
                    </div>
                </div>
            </div>
        </section>

        <main id="mainContent" class="main-content">
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

            <section id="analizSonucAlani">
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
            </section>

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
                ✅ Görme engelli bireyler için varsayılan açık gelen, ilk etkileşimde çalışan ve ayarlardan kapatılabilen Türkçe sesli yönlendirme<br>
                ✅ Harita altında ekran okuyucu uyumlu deprem listesi<br>
                ✅ Risk sonucuna göre renklendirilen şehir haritası<br>
                ✅ Büyük yazı ve yüksek kontrastlı acil durum ekranı
            </div>
</div>
        </main>

        <script>
            // Service Worker sadece telefon/PWA kullanımı için kaydedilir.
            // Masaüstünde eski icon veya PWA davranışı oluşmasını engeller.
            if (/Android|iPhone|iPad|iPod/i.test(navigator.userAgent) && "serviceWorker" in navigator) {{
                navigator.serviceWorker.register("/static/service-worker.js")
                .then(() => console.log("Service Worker kayıt edildi."))
                .catch(error => console.log("Service Worker hatası:", error));
            }}

            // Bilgisayarda daha önce kaydedilmiş Service Worker varsa temizlenir.
            if (!/Android|iPhone|iPad|iPod/i.test(navigator.userAgent) && "serviceWorker" in navigator) {{
                navigator.serviceWorker.getRegistrations().then(function(registrations) {{
                    for (let registration of registrations) {{
                        registration.unregister();
                    }}
                }});
            }}

            const analizYapildi = "{analiz_yapildi}" === "True";
            const depremVerileri = {deprem_verileri_json};
            const ilceVerileri = {ilce_verileri_json};
            const mahalleVerileri = {mahalle_verileri_json};

            let girisRehberiEtkilesimleBasladi = false;

            function sesliYonlendirmeAcikMi() {{
                return localStorage.getItem("riskatlasSesliYonlendirme") !== "kapali";
            }}

            function sesliYonlendirmeButonunuGuncelle() {{
                const btn = document.getElementById("voiceToggleButton");

                if (!btn) {{
                    return;
                }}

                if (sesliYonlendirmeAcikMi()) {{
                    btn.textContent = "🔊 Sesli Yönlendirme: Açık";
                    btn.classList.remove("off");
                    btn.setAttribute("aria-label", "Sesli yönlendirme açık. Kapatmak için dokunun.");
                }} else {{
                    btn.textContent = "🔇 Sesli Yönlendirme: Kapalı";
                    btn.classList.add("off");
                    btn.setAttribute("aria-label", "Sesli yönlendirme kapalı. Açmak için dokunun.");
                }}
            }}

            function sesliYonlendirmeAyariniDegistir() {{
                if (sesliYonlendirmeAcikMi()) {{
                    localStorage.setItem("riskatlasSesliYonlendirme", "kapali");

                    if ("speechSynthesis" in window) {{
                        window.speechSynthesis.cancel();
                    }}
                }} else {{
                    localStorage.setItem("riskatlasSesliYonlendirme", "acik");
                    setTimeout(() => {{
                        girisSesliAciklama('manual');
                    }}, 300);
                }}

                sesliYonlendirmeButonunuGuncelle();
            }}

            function sesliBilgi(metin) {{
                if (!sesliYonlendirmeAcikMi()) {{
                    return;
                }}

                if ("speechSynthesis" in window) {{
                    const mesaj = new SpeechSynthesisUtterance(metin);
                    mesaj.lang = "tr-TR";
                    mesaj.rate = 0.9;
                    mesaj.pitch = 1;
                    window.speechSynthesis.cancel();
                    setTimeout(() => {{
                        window.speechSynthesis.speak(mesaj);
                    }}, 200);
                }}
            }}

            function girisSesliAciklama(kaynak) {{

                if (!sesliYonlendirmeAcikMi()) {{
                    return;
                }}

                if (kaynak === 'manual' || kaynak === 'firstInteraction') {{
                    girisRehberiEtkilesimleBasladi = true;
                }}

                const metin =
                    "RiskAtlas erişilebilir afet bilgilendirme sistemine hoş geldiniz. " +
                    "Sesli yönlendirme varsayılan olarak açıktır. İsterseniz ayarlar kısmındaki sesli yönlendirme düğmesinden bu özelliği kapatabilirsiniz. " +
                    "Bu rehber, görme engelli kullanıcıların uygulamayı daha rahat kullanabilmesi için hazırlanmıştır. " +
                    "Konumumu kullan butonuna bastığınızda sistem sizden konum izni isteyecektir. " +
                    "İzin verirseniz yakınınızdaki kritik depremler kontrol edilir. " +
                    "İsterseniz şehir seçerek detaylı analiz ekranına da geçebilirsiniz. " +
                    "Açıklama şimdi ikinci kez tekrar edilecektir.";

                if ("speechSynthesis" in window) {{

                    window.speechSynthesis.cancel();

                    const mesaj1 = new SpeechSynthesisUtterance(metin);
                    mesaj1.lang = "tr-TR";
                    mesaj1.rate = 0.9;
                    mesaj1.pitch = 1;

                    const mesaj2 = new SpeechSynthesisUtterance(metin);
                    mesaj2.lang = "tr-TR";
                    mesaj2.rate = 0.9;
                    mesaj2.pitch = 1;

                    mesaj1.onend = function () {{
                        setTimeout(() => {{
                            if (sesliYonlendirmeAcikMi()) {{
                                window.speechSynthesis.speak(mesaj2);
                            }}
                        }}, 700);
                    }};

                    window.speechSynthesis.speak(mesaj1);
                }}
            }}

            function ilkEtkilesimdeSesliRehberiBaslat() {{
                if (!sesliYonlendirmeAcikMi()) {{
                    return;
                }}

                if (girisRehberiEtkilesimleBasladi) {{
                    return;
                }}

                girisSesliAciklama('firstInteraction');
            }}

            function detayliAnalizeGec() {{
                document.getElementById("landingScreen").style.display = "none";
                document.getElementById("mainContent").classList.add("active");
                sesliBilgi("Detaylı analiz ekranına geçildi.");
            }}

            function mesafeKm(lat1, lon1, lat2, lon2) {{
                const R = 6371;
                const dLat = (lat2 - lat1) * Math.PI / 180;
                const dLon = (lon2 - lon1) * Math.PI / 180;
                const a =
                    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                    Math.cos(lat1 * Math.PI / 180) *
                    Math.cos(lat2 * Math.PI / 180) *
                    Math.sin(dLon / 2) * Math.sin(dLon / 2);

                const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
                return R * c;
            }}

            function yakinDepremKontrolEt(kullaniciLat, kullaniciLon) {{
                let yakinKritik = null;

                depremVerileri.forEach(function(d) {{
                    try {{
                        const mag = parseFloat(d.mag || 0);
                        const coords = d.geojson && d.geojson.coordinates ? d.geojson.coordinates : null;

                        if (!coords || mag < 4.5) {{
                            return;
                        }}

                        const depremLon = parseFloat(coords[0]);
                        const depremLat = parseFloat(coords[1]);
                        const uzaklik = mesafeKm(kullaniciLat, kullaniciLon, depremLat, depremLon);

                        if (
                            (mag >= 4.5 && uzaklik <= 100) ||
                            (mag >= 5.5 && uzaklik <= 250)
                        ) {{
                            if (!yakinKritik || uzaklik < yakinKritik.uzaklik) {{
                                yakinKritik = {{
                                    title: d.title || "Bilinmeyen Konum",
                                    mag: mag,
                                    uzaklik: Math.round(uzaklik)
                                }};
                            }}
                        }}
                    }} catch (e) {{
                        console.log("Yakın deprem kontrol hatası:", e);
                    }}
                }});

                if (yakinKritik) {{
                    document.getElementById("konumDurumu").innerHTML =
                        "Yakınınızda kritik deprem algılandı: " +
                        yakinKritik.title +
                        " - Büyüklük: " +
                        yakinKritik.mag +
                        " - Yaklaşık uzaklık: " +
                        yakinKritik.uzaklik +
                        " km.";

                    sesliBilgi(
                        "Dikkat. Yakınınızda kritik seviyede deprem algılandı. Güvenli alana geçin. Asansör kullanmayın."
                    );

                    acilDurumGoster();
                }} else {{
                    document.getElementById("konumDurumu").innerHTML =
                        "Konum alındı. Yakınınızda kritik seviyede deprem uyarısı bulunmuyor.";

                    sesliBilgi("Konum alındı. Yakınınızda kritik seviyede deprem uyarısı bulunmuyor.");
                }}
            }}

            function konumModunuBaslat() {{
                sesliBilgi(
                    "Şimdi konum izni istenecek. Açılan pencerede izin ver seçeneğini seçerseniz yakın deprem uyarıları başlatılacaktır."
                );

                const durum = document.getElementById("konumDurumu");

                if (!navigator.geolocation) {{
                    durum.innerHTML = "Bu cihazda konum özelliği desteklenmiyor.";
                    sesliBilgi("Bu cihazda konum özelliği desteklenmiyor.");
                    return;
                }}

                durum.innerHTML = "Konum izni bekleniyor...";

                navigator.geolocation.getCurrentPosition(
                    function(position) {{
                        const lat = position.coords.latitude;
                        const lon = position.coords.longitude;

                        durum.innerHTML = "Konum alındı. Yakın deprem verileri kontrol ediliyor.";
                        yakinDepremKontrolEt(lat, lon);
                    }},
                    function(error) {{
                        durum.innerHTML =
                            "Konum izni alınamadı. İsterseniz şehir seçerek detaylı analiz ekranına geçebilirsiniz.";

                        sesliBilgi(
                            "Konum izni alınamadı. İsterseniz şehir seçerek detaylı analiz ekranına geçebilirsiniz."
                        );
                    }},
                    {{
                        enableHighAccuracy: true,
                        timeout: 10000,
                        maximumAge: 60000
                    }}
                );
            }}

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

                // Splash ekranı yalnızca telefon ekranında çalışır.
                if (window.innerWidth <= 700) {{
                    setTimeout(function () {{
                        const splash = document.getElementById("splash-screen");
                        if (splash) {{
                            splash.classList.add("fade-out");
                            setTimeout(function () {{
                                splash.remove();
                            }}, 800);
                        }}
                    }}, 1800);
                }} else {{
                    const splash = document.getElementById("splash-screen");
                    if (splash) {{
                        splash.remove();
                    }}
                }}

                sesliYonlendirmeButonunuGuncelle();

                // Sesli yönlendirme varsayılan olarak açıktır.
                // Tarayıcı izin verirse girişte otomatik başlar.
                // Telefon otomatik sesi engellerse kullanıcının ekrana ilk dokunuşunda başlar.
                setTimeout(() => {{
                    if (sesliYonlendirmeAcikMi() && !girisRehberiEtkilesimleBasladi) {{
                        girisSesliAciklama('auto');
                    }}
                }}, 900);

                document.addEventListener('click', ilkEtkilesimdeSesliRehberiBaslat, {{ once: true }});
                document.addEventListener('touchstart', ilkEtkilesimdeSesliRehberiBaslat, {{ once: true }});
                document.addEventListener('keydown', ilkEtkilesimdeSesliRehberiBaslat, {{ once: true }});

                const depremAlarmVar = "{deprem_alarm_var}" === "True";

                if (depremAlarmVar) {{
                    console.log("Genel canlı deprem alarmı mevcut. Konum modu açılırsa yakınlık kontrolü yapılır.");
                }}

                if (analizYapildi) {{
                    document.getElementById("landingScreen").style.display = "none";
                    document.getElementById("mainContent").classList.add("active");

                    setTimeout(() => {{
                        const sonucAlani = document.getElementById("analizSonucAlani");

                        if (sonucAlani) {{
                            sonucAlani.scrollIntoView({{
                                behavior: "smooth",
                                block: "start"
                            }});

                            const sonucMetni = sonucAlani.innerText.trim();
                            if (sonucMetni && sesliYonlendirmeAcikMi()) {{
                                sesliBilgi("Analiz sonucu hazır. " + sonucMetni);
                            }}
                        }}
                    }}, 450);
                }}
            }};
        </script>

    </body>
    </html>
    """

    return make_response(html)


if __name__ == "__main__":
    app.run(debug=True)
