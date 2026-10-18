#!/usr/bin/env python3
"""
RiskAtlas - Yerleşim verisi çekici (ülke / eyalet / şehir-kasaba + Türkiye ilçe / mahalle)

Ne yapar?
  1) Dünya verisi  -> datasets/dunya_yerlesimleri.json
       Kaynak: dr5hn/countries-states-cities-database (ODbL-1.0 lisanslı)
       Tüm ülkeler, eyalet/bölgeler ve şehir/kasabalar. Ülke adları mümkünse Türkçe yazılır.
  2) Türkiye adresi -> datasets/turkiye_adres.csv   (Sehir, Ilce, Mahalle)
       Kaynak: hsndmr/turkiye-city-county-district-neighborhood (data.json)
       81 il, 973 ilçe, ~73 bin mahalle/köy. Uygulama bu dosyayı varsa mevcut CSV'lerle birleştirir.

Kullanım (proje klasöründe, internet bağlantısı gerekir):
    python veri_cek.py                 # ikisini de indirir
    python veri_cek.py --sadece-dunya
    python veri_cek.py --sadece-turkiye
    python veri_cek.py --ulkeler US,DE,JP   # dünya verisini yalnızca bu ülkelerle sınırla (test için)

Not: Çıktı dosyaları önce geçici dosyaya yazılır, doğrulama geçerse yerine konur.
Gerekli paket: requests (projede zaten var).
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(BASE_DIR, "datasets")

DR5HN_DOSYALAR = {
    ad: [
        f"https://raw.githubusercontent.com/dr5hn/countries-states-cities-database/{dal}/json/{ad}.json"
        for dal in ("master", "main")
    ]
    for ad in ("countries", "states", "cities")
}
TR_ADRES_URL = [
    "https://raw.githubusercontent.com/hsndmr/turkiye-city-county-district-neighborhood/main/data.json",
    "https://raw.githubusercontent.com/hsndmr/turkiye-city-county-district-neighborhood/master/data.json",
]

# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
_KUCUK = str.maketrans({"I": "ı", "İ": "i"})
_BUYUK = str.maketrans({"i": "İ", "ı": "I"})


def tr_kucuk(s: str) -> str:
    return s.translate(_KUCUK).lower()


def tr_buyuk(s: str) -> str:
    return s.translate(_BUYUK).upper()


def tr_baslik(s: str) -> str:
    """TAMAMI BÜYÜK yazılmış adı Türkçe kurallarıyla 'Başlık Biçimi'ne çevirir; karışık harfli adlara dokunmaz."""
    s = (s or "").strip()
    if not s or s != tr_buyuk(s):
        return s
    return " ".join((w[:1].translate(_BUYUK).upper() + tr_kucuk(w[1:])) if w else w for w in s.split(" "))


def indir_json(urller, ad):
    son_hata = None
    for url in urller:
        try:
            print(f"  ↓ {ad}: {url}")
            r = requests.get(url, timeout=(10, 300), headers={"User-Agent": "RiskAtlas-veri-cek/1.0"})
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            son_hata = e
            print(f"    başarısız: {e}")
    sys.exit(f"HATA: {ad} indirilemedi ({son_hata}). İnternet bağlantınızı/güvenlik duvarınızı kontrol edin.")


def atomik_yaz(yol, yaz_fonksiyonu, **acma):
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    fd, gecici = tempfile.mkstemp(dir=os.path.dirname(yol), suffix=".tmp")
    os.close(fd)
    try:
        with open(gecici, "w", encoding="utf-8", **acma) as f:
            yaz_fonksiyonu(f)
        os.replace(gecici, yol)
    finally:
        if os.path.exists(gecici):
            os.remove(gecici)


# ---------------------------------------------------------------------------
# 1) Dünya verisi
# ---------------------------------------------------------------------------
def dunya_olustur(ulkeler, eyaletler, sehirler, sadece=None):
    """dr5hn ham listelerinden {ISO2: {ad, ad_en, bolgeler: {eyalet: [şehirler]}}} üretir."""
    ulke_bilgi = {}
    for u in ulkeler:
        kod = (u.get("iso2") or "").upper()
        if not kod or (sadece and kod not in sadece):
            continue
        tr_ad = (u.get("translations") or {}).get("tr") or u.get("name") or kod
        ulke_bilgi[kod] = {"id": u.get("id"), "ad": tr_ad, "ad_en": u.get("name") or kod}

    id_to_kod = {v["id"]: k for k, v in ulke_bilgi.items()}
    eyalet_adi = {}                     # state_id -> ad
    eyalet_ulke = {}                    # state_id -> ISO2
    bolge_sozlugu = defaultdict(dict)   # ISO2 -> {eyalet adı: set(şehirler)}

    for e in eyaletler:
        kod = id_to_kod.get(e.get("country_id"))
        if not kod:
            continue
        ad = (e.get("name") or "").strip()
        if not ad:
            continue
        eyalet_adi[e["id"]] = ad
        eyalet_ulke[e["id"]] = kod
        bolge_sozlugu[kod].setdefault(ad, set())

    for s in sehirler:
        kod = id_to_kod.get(s.get("country_id"))
        ad = (s.get("name") or "").strip()
        if not kod or not ad:
            continue
        eyalet = eyalet_adi.get(s.get("state_id"))
        if not eyalet:  # eyaleti olmayan ülkeler / eşleşmeyen kayıtlar
            eyalet = ulke_bilgi[kod]["ad"] if not bolge_sozlugu[kod] else "Diğer"
        bolge_sozlugu[kod].setdefault(eyalet, set()).add(ad)

    sonuc = {}
    for kod, bilgi in ulke_bilgi.items():
        bolgeler = {b: sorted(c) for b, c in sorted(bolge_sozlugu.get(kod, {}).items())}
        sonuc[kod] = {"ad": bilgi["ad"], "ad_en": bilgi["ad_en"], "bolgeler": bolgeler}
    return sonuc


def dunya_calistir(sadece=None):
    print("\n[1/2] Dünya verisi indiriliyor (dr5hn)…")
    ulkeler = indir_json(DR5HN_DOSYALAR["countries"], "countries.json")
    eyaletler = indir_json(DR5HN_DOSYALAR["states"], "states.json")
    sehirler = indir_json(DR5HN_DOSYALAR["cities"], "cities.json")
    sonuc = dunya_olustur(ulkeler, eyaletler, sehirler, sadece)

    n_ulke = len(sonuc)
    n_bolge = sum(len(u["bolgeler"]) for u in sonuc.values())
    n_sehir = sum(len(c) for u in sonuc.values() for c in u["bolgeler"].values())
    if not sadece and (n_ulke < 200 or n_sehir < 50000):
        sys.exit(f"HATA: Beklenenden az veri geldi ({n_ulke} ülke, {n_sehir} şehir). Dosya yazılmadı.")

    yol = os.path.join(DATASETS, "dunya_yerlesimleri.json")
    veri = {
        "surum": 1,
        "kaynak": "dr5hn/countries-states-cities-database (ODbL-1.0)",
        "ulkeler": sonuc,
    }
    atomik_yaz(yol, lambda f: json.dump(veri, f, ensure_ascii=False, separators=(",", ":")))
    print(f"  ✓ {yol}\n    {n_ulke} ülke, {n_bolge} eyalet/bölge, {n_sehir} şehir/kasaba")


# ---------------------------------------------------------------------------
# 2) Türkiye il / ilçe / mahalle
# ---------------------------------------------------------------------------
def turkiye_satirlari(veri):
    """hsndmr data.json (il > ilçe > semt > mahalle) -> sıralı benzersiz (Sehir, Ilce, Mahalle) satırları."""
    satirlar = set()
    for il in veri:
        sehir = tr_baslik(il.get("name", ""))
        for ilce in il.get("counties", []) or []:
            ilce_adi = tr_baslik(ilce.get("name", ""))
            mahalle_var = False
            for semt in ilce.get("districts", []) or []:
                for mah in semt.get("neighborhoods", []) or []:
                    ad = tr_baslik(mah.get("name", ""))
                    if ad:
                        satirlar.add((sehir, ilce_adi, ad))
                        mahalle_var = True
            if not mahalle_var:
                satirlar.add((sehir, ilce_adi, ""))  # mahallesi olmayan ilçe de listede kalsın
    return sorted(satirlar, key=lambda x: (tr_kucuk(x[0]), tr_kucuk(x[1]), tr_kucuk(x[2])))


def turkiye_calistir():
    print("\n[2/2] Türkiye il/ilçe/mahalle verisi indiriliyor (hsndmr)…")
    veri = indir_json(TR_ADRES_URL, "data.json")
    satirlar = turkiye_satirlari(veri)

    n_il = len({s[0] for s in satirlar})
    n_ilce = len({(s[0], s[1]) for s in satirlar})
    n_mah = sum(1 for s in satirlar if s[2])
    if n_il < 80 or n_ilce < 900:
        sys.exit(f"HATA: Beklenenden az veri geldi ({n_il} il, {n_ilce} ilçe). Dosya yazılmadı.")

    yol = os.path.join(DATASETS, "turkiye_adres.csv")

    def yaz(f):
        w = csv.writer(f)
        w.writerow(["Sehir", "Ilce", "Mahalle"])
        w.writerows(satirlar)

    atomik_yaz(yol, yaz, newline="")
    # Excel/pandas uyumu için BOM'lu okuma: uygulama utf-8-sig ile okur, BOM'suz dosya da sorunsuzdur.
    print(f"  ✓ {yol}\n    {n_il} il, {n_ilce} ilçe, {n_mah} mahalle/köy")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RiskAtlas yerleşim verisi çekici")
    ap.add_argument("--sadece-dunya", action="store_true")
    ap.add_argument("--sadece-turkiye", action="store_true")
    ap.add_argument("--ulkeler", help="Virgülle ISO2 kodları (ör. US,DE,JP). Yalnızca test içindir.")
    a = ap.parse_args()

    sadece = {k.strip().upper() for k in a.ulkeler.split(",")} if a.ulkeler else None
    if not a.sadece_turkiye:
        dunya_calistir(sadece)
    if not a.sadece_dunya:
        turkiye_calistir()

    os.makedirs(DATASETS, exist_ok=True)
    with open(os.path.join(DATASETS, "VERI_KAYNAKLARI.txt"), "w", encoding="utf-8") as f:
        f.write(
            "dunya_yerlesimleri.json: dr5hn/countries-states-cities-database, Open Database License (ODbL-1.0).\n"
            "  https://github.com/dr5hn/countries-states-cities-database\n"
            "turkiye_adres.csv: hsndmr/turkiye-city-county-district-neighborhood.\n"
            "  https://github.com/hsndmr/turkiye-city-county-district-neighborhood\n"
        )
    print("\nBitti. Uygulamayı yeniden başlatın; yeni veriler otomatik yüklenir.")


if __name__ == "__main__":
    main()
