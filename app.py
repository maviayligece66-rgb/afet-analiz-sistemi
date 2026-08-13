


import json
import logging
import os
import sqlite3
import threading
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import folium
import joblib
import pandas as pd
import requests
from flask import Flask, flash, make_response, redirect, render_template, request, session, url_for


# -----------------------------------------------------------------------------
# 1. FLASK / LOGGING / CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENVIRONMENT = os.environ.get("FLASK_ENV", "development").strip().lower()
IS_PRODUCTION = ENVIRONMENT == "production"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RiskAtlas")

app = Flask(__name__)

_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    # Existing local development installations must continue to start even if
    # an environment variable has not yet been configured. Production should
    # always provide a persistent, private SECRET_KEY.
    _secret_key = "riskatlas-development-key-change-me"
    logger.warning("SECRET_KEY ortam değişkeni tanımlı değil; geliştirme anahtarı kullanılıyor.")

app.config.update(
    SECRET_KEY=_secret_key,
    PERMANENT_SESSION_LIFETIME=int(os.environ.get("SESSION_LIFETIME", "1800")),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=(
        os.environ.get("SESSION_COOKIE_SECURE", "1" if IS_PRODUCTION else "0") == "1"
    ),
    MAX_CONTENT_LENGTH=int(os.environ.get("MAX_CONTENT_LENGTH", str(2 * 1024 * 1024))),
)

DATA_PATHS = {
    "processed_afet": os.path.join(BASE_DIR, "datasets", "processed_afet_verisi.csv"),
    "db": os.path.join(BASE_DIR, "datasets", "afet_veritabani.db"),
    "model": os.path.join(BASE_DIR, "models", "afet_model.pkl"),
    "geojson": os.path.join(BASE_DIR, "datasets", "turkey_provinces.geojson"),
    "ilce": os.path.join(BASE_DIR, "datasets", "turkey_districts.csv"),
    "zemin": os.path.join(BASE_DIR, "datasets", "zemin_verileri.csv"),
}

# Eski değişken adları, dosyanın ilerleyen bölümlerindeki mevcut kodu bozmamak
# için korunuyor.
base = BASE_DIR
data_yolu = DATA_PATHS["processed_afet"]
db_yolu = DATA_PATHS["db"]
model_yolu = DATA_PATHS["model"]
geojson_yolu = DATA_PATHS["geojson"]
ilce_yolu = DATA_PATHS["ilce"]
zemin_yolu = DATA_PATHS["zemin"]


# -----------------------------------------------------------------------------
# 2. HTTP / DATABASE HELPERS
# -----------------------------------------------------------------------------
HTTP_TIMEOUT = (
    float(os.environ.get("HTTP_CONNECT_TIMEOUT", "3.05")),
    float(os.environ.get("HTTP_READ_TIMEOUT", "5.0")),
)
HTTP_RETRIES = max(0, int(os.environ.get("HTTP_RETRIES", "1")))


@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    """SQLite bağlantısını güvenli şekilde açar ve işlem sonunda kapatır."""
    os.makedirs(os.path.dirname(db_yolu), exist_ok=True)
    conn = sqlite3.connect(db_yolu, timeout=10.0)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA journal_mode = WAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _http_get_json(url: str, *, timeout: Tuple[float, float] = HTTP_TIMEOUT) -> Any:
    """JSON döndüren dış API çağrılarını timeout ve sınırlı retry ile yürütür."""
    last_error: Optional[Exception] = None

    for attempt in range(HTTP_RETRIES + 1):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"Accept": "application/json", "User-Agent": "RiskAtlas/2.0"},
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < HTTP_RETRIES:
                time.sleep(0.25 * (attempt + 1))

    if last_error:
        raise last_error
    return None


# -----------------------------------------------------------------------------
# 3. DATABASE INITIALIZATION
# -----------------------------------------------------------------------------
def veritabani_olustur() -> None:
    """Gerekli SQLite tablolarını ve indekslerini oluşturur."""
    try:
        with get_db_connection() as conn:
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

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS takip_edilen_sehirler (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sehir TEXT UNIQUE
                )
            """)

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_analiz_sehir "
                "ON analiz_kayitlari(sehir)"
            )

            # Kullanıcı hesapları. Eski veritabanları bozulmadan tablo yoksa eklenir.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kullanicilar (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    fullname TEXT NOT NULL,
                    phone TEXT,
                    profile_photo TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Her kullanıcının kendi ana/ek konumları. Maksimum 3 kayıt uygulama
            # katmanında ve veritabanı transaction'ında ayrıca korunur.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kullanici_konumlari (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    sehir TEXT NOT NULL,
                    konum_tipi TEXT NOT NULL DEFAULT 'ek',
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id, sehir),
                    FOREIGN KEY(user_id) REFERENCES kullanicilar(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kullanici_konumlari_user ON kullanici_konumlari(user_id)")

            analysis_columns = [row[1] for row in cursor.execute("PRAGMA table_info(analiz_kayitlari)").fetchall()]
            if "user_id" not in analysis_columns:
                cursor.execute("ALTER TABLE analiz_kayitlari ADD COLUMN user_id INTEGER")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analiz_user ON analiz_kayitlari(user_id)")

        logger.info("Veritabanı hazır: gerekli tablolar ve indeksler kontrol edildi.")
    except sqlite3.Error:
        logger.exception("Veritabanı oluşturma hatası")
        raise


veritabani_olustur()


# -----------------------------------------------------------------------------
# 3A. KULLANICI / MİSAFİR / KONUM HESABI
# -----------------------------------------------------------------------------
def aktif_kullanici_id() -> Optional[int]:
    value = session.get("user_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def misafir_mi() -> bool:
    return bool(session.get("guest_mode")) and aktif_kullanici_id() is None


def erisim_gerekli(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if aktif_kullanici_id() is None and not misafir_mi():
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def kullanici_konumlarini_getir() -> List[Dict[str, Any]]:
    user_id = aktif_kullanici_id()
    if user_id is not None:
        try:
            with get_db_connection() as conn:
                rows = conn.execute(
                    "SELECT id, sehir, konum_tipi FROM kullanici_konumlari WHERE user_id = ? ORDER BY CASE WHEN konum_tipi='ana' THEN 0 ELSE 1 END, id",
                    (user_id,),
                ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error:
            logger.exception("Kullanıcı konumları okunamadı")
            return []

    guest = session.get("guest_locations", [])
    if not isinstance(guest, list):
        return []
    return [dict(x) for x in guest if isinstance(x, dict) and x.get("sehir")]


def konum_ekle(sehir: str, konum_tipi: str = "ek") -> Tuple[bool, str]:
    sehir = gorunum_duzelt(sehir)
    if not sehir:
        return False, "Geçerli bir şehir seçmelisiniz."

    mevcut = kullanici_konumlarini_getir()
    if any(normalize_text(x.get("sehir", "")) == normalize_text(sehir) for x in mevcut):
        return False, "Bu konum zaten kayıtlı."
    if len(mevcut) >= MAX_USER_LOCATIONS:
        return False, "En fazla 3 konum kaydedebilirsiniz: 1 ana konum ve 2 ek konum."

    if not mevcut:
        konum_tipi = "ana"
    elif konum_tipi not in {"ana", "ek"}:
        konum_tipi = "ek"

    user_id = aktif_kullanici_id()
    if user_id is not None:
        try:
            with get_db_connection() as conn:
                if konum_tipi == "ana":
                    conn.execute("UPDATE kullanici_konumlari SET konum_tipi='ek' WHERE user_id=?", (user_id,))
                conn.execute(
                    "INSERT INTO kullanici_konumlari(user_id, sehir, konum_tipi, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, sehir, konum_tipi, datetime.now().isoformat(timespec="seconds")),
                )
            return True, "Konum kaydedildi."
        except sqlite3.IntegrityError:
            return False, "Bu konum zaten kayıtlı veya konum sınırına ulaştınız."
        except sqlite3.Error:
            logger.exception("Kullanıcı konumu eklenemedi")
            return False, "Konum kaydedilirken bir veritabanı hatası oluştu."

    guest = mevcut
    guest.append({"sehir": sehir, "konum_tipi": konum_tipi})
    session["guest_locations"] = guest[:MAX_USER_LOCATIONS]
    session.modified = True
    return True, "Konum misafir oturumuna kaydedildi."


def ana_konum_getir() -> str:
    locations = kullanici_konumlarini_getir()
    for item in locations:
        if item.get("konum_tipi") == "ana":
            return str(item.get("sehir", ""))
    return str(locations[0].get("sehir", "")) if locations else ""


def konum_sil(sehir: str) -> Tuple[bool, str]:
    sehir = gorunum_duzelt(sehir)
    user_id = aktif_kullanici_id()
    if user_id is not None:
        try:
            with get_db_connection() as conn:
                row = conn.execute(
                    "SELECT id, konum_tipi FROM kullanici_konumlari WHERE user_id=? AND sehir=?",
                    (user_id, sehir),
                ).fetchone()
                if not row:
                    return False, "Konum bulunamadı."
                conn.execute("DELETE FROM kullanici_konumlari WHERE id=?", (row["id"],))
                if row["konum_tipi"] == "ana":
                    first = conn.execute(
                        "SELECT id FROM kullanici_konumlari WHERE user_id=? ORDER BY id LIMIT 1", (user_id,)
                    ).fetchone()
                    if first:
                        conn.execute("UPDATE kullanici_konumlari SET konum_tipi='ana' WHERE id=?", (first["id"],))
            return True, "Konum silindi."
        except sqlite3.Error:
            logger.exception("Konum silinemedi")
            return False, "Konum silinirken hata oluştu."

    guest = [x for x in session.get("guest_locations", []) if normalize_text(x.get("sehir", "")) != normalize_text(sehir)]
    if guest and not any(x.get("konum_tipi") == "ana" for x in guest):
        guest[0]["konum_tipi"] = "ana"
    session["guest_locations"] = guest
    session.modified = True
    return True, "Konum silindi."


def misafir_konumlarini_hesaba_aktar(user_id: int) -> None:
    guest_locations = session.get("guest_locations", [])
    if not isinstance(guest_locations, list):
        guest_locations = []
    try:
        with get_db_connection() as conn:
            count = conn.execute("SELECT COUNT(*) AS c FROM kullanici_konumlari WHERE user_id=?", (user_id,)).fetchone()["c"]
            for item in guest_locations:
                if count >= MAX_USER_LOCATIONS:
                    break
                sehir = gorunum_duzelt(item.get("sehir", ""))
                if not sehir:
                    continue
                exists = conn.execute("SELECT 1 FROM kullanici_konumlari WHERE user_id=? AND sehir=?", (user_id, sehir)).fetchone()
                if exists:
                    continue
                tip = "ana" if count == 0 else ("ana" if item.get("konum_tipi") == "ana" else "ek")
                if tip == "ana":
                    conn.execute("UPDATE kullanici_konumlari SET konum_tipi='ek' WHERE user_id=?", (user_id,))
                conn.execute(
                    "INSERT INTO kullanici_konumlari(user_id,sehir,konum_tipi,created_at) VALUES(?,?,?,?)",
                    (user_id, sehir, tip, datetime.now().isoformat(timespec="seconds")),
                )
                count += 1
    except sqlite3.Error:
        logger.exception("Misafir konumları hesaba aktarılırken hata oluştu")


# -----------------------------------------------------------------------------
# 4. TEXT / DATA HELPERS
# -----------------------------------------------------------------------------
def normalize_text(text: Any) -> str:
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def turkce_sirala(liste: List[str]) -> List[str]:
    """Türkçe karakterleri dikkate alarak alfabetik sıralama yapar."""
    return sorted(liste, key=normalize_text)


def gorunum_duzelt(text: Any) -> str:
    """KONYA / konya gibi değerleri Türkçe karakterleri bozmadan düzgün gösterir."""
    text = str(text).strip()

    if not text or text.lower() == "nan":
        return ""

    text = text.replace("I", "ı").replace("İ", "i").lower()
    return " ".join(
        kelime[0].upper() + kelime[1:]
        for kelime in text.split()
        if kelime
    )


def tekil_ve_sirali(liste: List[Any]) -> List[str]:
    """Tekrarları temizler ve Türkçe uyumlu sıralama yapar."""
    temiz: Dict[str, str] = {}

    for item in liste:
        item = str(item).strip()
        if not item or item.lower() == "nan":
            continue

        anahtar = normalize_text(item)
        if anahtar not in temiz:
            temiz[anahtar] = gorunum_duzelt(item)

    return turkce_sirala(list(temiz.values()))


# -----------------------------------------------------------------------------
# 5. EARTHQUAKE API / CACHE
# -----------------------------------------------------------------------------
DEPREM_CACHE: Dict[str, Any] = {"zaman": 0.0, "veri": []}
DEPREM_CACHE_LOCK = threading.Lock()
MAX_USER_LOCATIONS = 3
PROFILE_UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads", "profiles")
ALLOWED_PROFILE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}

os.makedirs(PROFILE_UPLOAD_DIR, exist_ok=True)

DEPREM_CACHE_TTL = max(30, int(os.environ.get("DEPREM_CACHE_TTL", "300")))


def afad_depremleri_getir() -> List[Dict[str, Any]]:
    url = "https://deprem.afad.gov.tr/apiv2/event/latest"

    try:
        veriler = _http_get_json(url)
        if not isinstance(veriler, list):
            logger.warning("AFAD API beklenmeyen veri döndürdü.")
            return []

        depremler: List[Dict[str, Any]] = []
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
                    "geojson": {"coordinates": [lon, lat]},
                })
            except (TypeError, ValueError):
                continue

        return depremler
    except Exception:
        logger.exception("AFAD API hatası")
        return []


def kandilli_depremleri_getir() -> List[Dict[str, Any]]:
    url = "https://api.orhanaydogdu.com.tr/deprem/kandilli/live"

    try:
        payload = _http_get_json(url)
        if not isinstance(payload, dict):
            logger.warning("Kandilli API beklenmeyen veri döndürdü.")
            return []

        depremler = payload.get("result", [])
        if not isinstance(depremler, list):
            return []

        temiz_depremler: List[Dict[str, Any]] = []
        for deprem in depremler[:30]:
            if isinstance(deprem, dict):
                item = dict(deprem)
                item["kaynak"] = "Kandilli"
                temiz_depremler.append(item)

        return temiz_depremler
    except Exception:
        logger.exception("Kandilli API hatası")
        return []


def usgs_depremleri_getir() -> List[Dict[str, Any]]:
    """Küresel deprem akışı için USGS GeoJSON kaynağı."""
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
    try:
        payload = _http_get_json(url)
        features = payload.get("features", []) if isinstance(payload, dict) else []
        result: List[Dict[str, Any]] = []
        for feature in features[:50]:
            props = feature.get("properties", {}) or {}
            geometry = feature.get("geometry", {}) or {}
            coords = geometry.get("coordinates", []) or []
            if len(coords) < 2:
                continue
            result.append({
                "kaynak": "USGS",
                "title": props.get("place", "Bilinmeyen Konum"),
                "mag": float(props.get("mag") or 0),
                "date": datetime.fromtimestamp((props.get("time") or 0) / 1000).isoformat() if props.get("time") else "",
                "geojson": {"coordinates": [coords[0], coords[1]]},
            })
        return result
    except Exception:
        logger.exception("USGS API hatası")
        return []


def canlı_depremleri_getir() -> List[Dict[str, Any]]:
    simdi = time.time()

    with DEPREM_CACHE_LOCK:
        if (
            DEPREM_CACHE["veri"]
            and simdi - float(DEPREM_CACHE["zaman"]) < DEPREM_CACHE_TTL
        ):
            return list(DEPREM_CACHE["veri"])

    afad = afad_depremleri_getir()
    kandilli = kandilli_depremleri_getir()
    usgs = usgs_depremleri_getir()

    # Türkiye kaynakları önceliklidir; USGS bunlara ek küresel akış sağlar.
    veri = afad + kandilli + usgs
    benzersiz = []
    gorulen = set()
    for item in veri:
        anahtar = (normalize_text(item.get("title", "")), round(float(item.get("mag", 0)), 1), item.get("date", ""))
        if anahtar in gorulen:
            continue
        gorulen.add(anahtar)
        benzersiz.append(item)

    with DEPREM_CACHE_LOCK:
        DEPREM_CACHE["veri"] = list(benzersiz[:100])
        DEPREM_CACHE["zaman"] = simdi

    return veri


# -----------------------------------------------------------------------------
# 6. RISK / SOIL / MAP HELPERS
# -----------------------------------------------------------------------------
def zemin_bilgisi_getir(zemin_df: Optional[pd.DataFrame], sehir: str, ilce: str = "", mahalle: str = "") -> Dict[str, Any]:
    varsayilan = {
        "tip": "Zemin verisi bulunamadı",
        "risk": 5,
        "aciklama": "Bu bölge için kayıtlı zemin verisi bulunamadığı için analizde varsayılan orta düzey zemin riski kullanılmıştır.",
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
            "aciklama": satir.get(
                "Zemin_Aciklama",
                "Bu bölgenin zemin bilgisi veri setinden alınmıştır.",
            ),
        }
    except Exception:
        logger.exception("Zemin bilgisi okuma hatası")
        return varsayilan


def otomatik_model_degerleri(sehir: str, ilce: str = "") -> Tuple[float, float, float, float]:
    """Kullanıcıdan istenmeyen altyapı değişkenlerini veri setinden otomatik doldurur."""
    varsayilan = (5000.0, 1000.0, 50000.0, 50.0)
    if not os.path.exists(data_yolu):
        return varsayilan
    try:
        df = pd.read_csv(data_yolu, encoding="utf-8-sig")
        if "Sehir" in df.columns and sehir:
            df = df[df["Sehir"].apply(normalize_text) == normalize_text(sehir)]
        if ilce and "Ilce" in df.columns:
            eslesen = df[df["Ilce"].apply(normalize_text) == normalize_text(ilce)]
            if not eslesen.empty:
                df = eslesen

        def median_for(names: List[str], fallback: float) -> float:
            for name in names:
                if name in df.columns:
                    series = pd.to_numeric(df[name], errors="coerce").dropna()
                    if not series.empty:
                        return float(series.median())
            return fallback

        return (
            median_for(["Nufus_Yogunlugu", "Nüfus_Yogunlugu", "Nufus"], varsayilan[0]),
            median_for(["Hastane_Yatak_Kapasitesi", "Yatak_Kapasitesi", "Yatak"], varsayilan[1]),
            median_for(["Toplanma_Alani", "Toplanma_Alanı", "Toplanma"], varsayilan[2]),
            median_for(["Itfaiye_Gucu", "Itfaiye_Gücü", "Itfaiye"], varsayilan[3]),
        )
    except Exception:
        logger.exception("Otomatik model değişkenleri alınamadı")
        return varsayilan


def kullaniciye_ozel_oneriler_uret(bina_yasi: float, kat_sayisi: int, bulundugu_kat: int, kisi_sayisi: int, evcil_hayvan: bool, erisim_ihtiyaci: str) -> List[str]:
    oneriler: List[str] = []
    if bina_yasi >= 25:
        oneriler.append("Bina yaşı yüksek olduğu için yetkili bir yapı güvenliği incelemesi düşünülmelidir.")
    if kat_sayisi >= 8:
        oneriler.append("Yüksek katlı bir binada yaşıyorsanız deprem sırasında merdivenleri kullanmaya yönelik tahliye planınızı önceden belirleyin.")
    if bulundugu_kat > 0 and kat_sayisi > 0 and bulundugu_kat == kat_sayisi:
        oneriler.append("En üst katta olduğunuz için deprem sırasında asansör kullanmayın ve önceden belirlediğiniz güvenli alana yönelin.")
    if kisi_sayisi >= 5:
        oneriler.append("Evdeki kişi sayısı fazla olduğu için aile iletişim ve buluşma planınızı önceden belirleyin.")
    if evcil_hayvan:
        oneriler.append("Evcil hayvanınız için su, mama, tasma/taşıma çantası ve temel ihtiyaçların bulunduğu küçük bir afet çantası hazırlayın.")
    if erisim_ihtiyaci == "gorme":
        oneriler.append("Görme desteği ihtiyacınız varsa sesli yönlendirme ve sesli deprem alarmı ayarlarını ayrıca kontrol edin.")
    elif erisim_ihtiyaci == "isitme":
        oneriler.append("İşitme desteği ihtiyacınız varsa görsel kırmızı alarm ve titreşimli uyarıların açık olduğundan emin olun.")
    elif erisim_ihtiyaci == "hareket":
        oneriler.append("Hareket desteği ihtiyacınız varsa binadan erişilebilir çıkış rotasını ve yardım edecek kişiyi önceden belirleyin.")
    return oneriler


def acil_oneriler_uret(risk_durumu: str, inputs: Optional[Tuple[float, float, float, float, float, float]]) -> List[str]:
    oneriler: List[str] = []

    if not inputs:
        return oneriler

    nufus, bina_yasi, yatak, toplanma, itfaiye, zemin = inputs

    if risk_durumu == "Güvenli Bölge":
        oneriler.extend([
            "Mevcut afet hazırlık planları düzenli olarak güncellenmelidir.",
            "Acil durum çantası ve aile iletişim planı hazır tutulmalıdır.",
            "Düzenli afet farkındalık tatbikatları yapılmalıdır.",
        ])
    elif risk_durumu == "Orta Riskli":
        oneriler.extend([
            "Tahliye yolları ve toplanma alanları yeniden kontrol edilmelidir.",
            "Riskli yapıların ön incelemesi yapılmalıdır.",
            "Acil iletişim ve yerel müdahale planı oluşturulmalıdır.",
        ])
    elif risk_durumu == "Kritik / Riskli":
        oneriler.extend([
            "Bu bölgede acil tahliye planı oluşturulmalıdır.",
            "Toplanma alanı kapasitesi artırılmalıdır.",
            "Eski yapılar için bina dayanıklılık analizi ve güçlendirme önerilir.",
            "Hastane, itfaiye ve ana ulaşım yolları önceliklendirilmelidir.",
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


def risk_skoru_getir(risk_durumu: str) -> int:
    return {
        "Güvenli Bölge": 1,
        "Orta Riskli": 3,
        "Kritik / Riskli": 5,
    }.get(risk_durumu, 0)


def risk_rengi_getir(risk_skoru: int) -> str:
    return {
        0: "#d9d9d9",
        1: "#2ecc71",
        2: "#f1c40f",
        3: "#f39c12",
        4: "#e74c3c",
        5: "#8b0000",
    }.get(risk_skoru, "#d9d9d9")


def geojson_sehir_adi_bul(feature: Dict[str, Any]) -> Any:
    props = feature.get("properties", {}) or {}
    olasi_alanlar = [
        "name", "NAME_1", "Name", "il", "Il", "IL",
        "province", "Province", "sehir", "Sehir",
    ]

    for alan in olasi_alanlar:
        if alan in props:
            return props[alan]
    return ""


def sehirleri_renklendir(m: folium.Map, secilen_sehir: str, risk_skoru: int, risk_durumu: str) -> None:
    if not os.path.exists(geojson_yolu):
        logger.warning("GeoJSON dosyası bulunamadı: %s", geojson_yolu)
        return

    try:
        with open(geojson_yolu, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        secilen_norm = normalize_text(secilen_sehir)

        def style_function(feature: Dict[str, Any]) -> Dict[str, Any]:
            sehir_adi = geojson_sehir_adi_bul(feature)
            sehir_norm = normalize_text(sehir_adi)

            if secilen_norm and secilen_norm == sehir_norm:
                return {
                    "fillColor": risk_rengi_getir(risk_skoru),
                    "color": "#111111",
                    "weight": 2,
                    "fillOpacity": 0.75,
                }

            return {
                "fillColor": "#f7f7f7",
                "color": "#666666",
                "weight": 1,
                "fillOpacity": 0.25,
            }

        def highlight_function(feature: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "fillColor": "#ffff99",
                "color": "#000000",
                "weight": 3,
                "fillOpacity": 0.7,
            }

        folium.GeoJson(
            geojson_data,
            name="Şehir Risk Haritası",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=[], aliases=[], sticky=True, labels=False
            ),
        ).add_to(m)

        if secilen_sehir and risk_skoru > 0:
            folium.Marker(
                location=[39, 35],
                popup=f"{secilen_sehir} - {risk_durumu} - Risk Skoru: {risk_skoru}/5",
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(m)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.exception("GeoJSON harita işleme hatası")


# -----------------------------------------------------------------------------
# 7. AUTH / ACCOUNT / LOCATION ROUTES
# -----------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    return "OK", 200

@app.route("/login", methods=["GET", "POST"])
def login():
    if aktif_kullanici_id() is not None or misafir_mi():
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("E-posta ve şifre alanları zorunludur.", "error")
            return render_template("login.html")
        try:
            with get_db_connection() as conn:
                user = conn.execute("SELECT * FROM kullanicilar WHERE email=?", (email,)).fetchone()
            if not user or not check_password_hash(user["password_hash"], password):
                flash("E-posta veya şifre hatalı.", "error")
                return render_template("login.html")
            session.clear()
            session.permanent = bool(request.form.get("remember"))
            session["user_id"] = int(user["id"])
            session["email"] = user["email"]
            session["fullname"] = user["fullname"]
            next_url = request.args.get("next") or url_for("index")
            if not next_url.startswith("/") or next_url.startswith("//"):
                next_url = url_for("index")
            return redirect(next_url)
        except sqlite3.Error:
            logger.exception("Giriş sırasında veritabanı hatası")
            flash("Giriş sırasında bir hata oluştu. Lütfen tekrar deneyin.", "error")
    return render_template("login.html")

@app.route("/guest-login")
def guest_login():
    session.clear()
    session.permanent = True
    session["guest_mode"] = True
    session["guest_locations"] = []
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if aktif_kullanici_id() is not None:
        return redirect(url_for("index"))
    guest_locations = session.get("guest_locations", []) if misafir_mi() else []
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        home_city = gorunum_duzelt(request.form.get("home_city", ""))
        if not fullname or not email or not password:
            flash("Ad soyad, e-posta ve şifre alanları zorunludur.", "error")
            return render_template("register.html")
        if len(password) < 6:
            flash("Şifreniz en az 6 karakter olmalıdır.", "error")
            return render_template("register.html")
        try:
            with get_db_connection() as conn:
                existing = conn.execute("SELECT id FROM kullanicilar WHERE email=?", (email,)).fetchone()
                if existing:
                    flash("Bu e-posta adresiyle zaten bir hesap bulunuyor. Giriş yapabilirsiniz.", "error")
                    return render_template("register.html")
                cursor = conn.execute(
                    "INSERT INTO kullanicilar(email,password_hash,fullname,created_at) VALUES(?,?,?,?)",
                    (email, generate_password_hash(password), fullname, datetime.now().isoformat(timespec="seconds")),
                )
                user_id = cursor.lastrowid
                aktarilacak = list(guest_locations) if isinstance(guest_locations, list) else []
                if home_city and not any(normalize_text(x.get("sehir", "")) == normalize_text(home_city) for x in aktarilacak):
                    if len(aktarilacak) < MAX_USER_LOCATIONS:
                        aktarilacak.insert(0, {"sehir": home_city, "konum_tipi": "ana"})
                elif home_city:
                    for item in aktarilacak:
                        if normalize_text(item.get("sehir", "")) == normalize_text(home_city):
                            item["konum_tipi"] = "ana"
                # Üç misafir konumu zaten doluysa hiçbirini silmeyiz; mevcut ana konum korunur.
                for index, item in enumerate(aktarilacak[:MAX_USER_LOCATIONS]):
                    city = gorunum_duzelt(item.get("sehir", ""))
                    if not city:
                        continue
                    tip = "ana" if index == 0 else "ek"
                    conn.execute(
                        "INSERT OR IGNORE INTO kullanici_konumlari(user_id,sehir,konum_tipi,created_at) VALUES(?,?,?,?)",
                        (user_id, city, tip, datetime.now().isoformat(timespec="seconds")),
                    )
            session.clear()
            session.permanent = True
            session["user_id"] = int(user_id)
            session["email"] = email
            session["fullname"] = fullname
            flash("Hesabınız oluşturuldu. Misafirken kaydettiğiniz konumlar hesabınıza aktarıldı.", "success")
            return redirect(url_for("index"))
        except sqlite3.Error:
            logger.exception("Kayıt sırasında veritabanı hatası")
            flash("Hesap oluşturulurken bir hata oluştu. Lütfen tekrar deneyin.", "error")
    return render_template("register.html")

@app.route("/takip-ekle", methods=["POST"])
@erisim_gerekli
def takip_ekle():
    sehir = gorunum_duzelt(request.form.get("takipSehirInput", ""))
    konum_tipi = request.form.get("konumTipi", "ek")
    ok, message = konum_ekle(sehir, konum_tipi)
    if ok and konum_tipi == "ana":
        user_id = aktif_kullanici_id()
        if user_id is None:
            locations = session.get("guest_locations", [])
            for item in locations:
                item["konum_tipi"] = "ana" if normalize_text(item.get("sehir", "")) == normalize_text(sehir) else "ek"
            session["guest_locations"] = locations
            session.modified = True
        else:
            try:
                with get_db_connection() as conn:
                    conn.execute("UPDATE kullanici_konumlari SET konum_tipi='ek' WHERE user_id=?", (user_id,))
                    conn.execute("UPDATE kullanici_konumlari SET konum_tipi='ana' WHERE user_id=? AND sehir=?", (user_id, sehir))
            except sqlite3.Error:
                logger.exception("Ana konum işaretlenemedi")
    flash(message, "success" if ok else "error")
    return redirect(url_for("index"))

@app.route("/takip-sil/<sehir>", methods=["GET", "POST"])
@erisim_gerekli
def takip_sil(sehir):
    ok, message = konum_sil(sehir)
    flash(message, "success" if ok else "error")
    return redirect(url_for("index"))

@app.route("/ana-konum-yap", methods=["POST"])
@erisim_gerekli
def ana_konum_yap():
    sehir = gorunum_duzelt(request.form.get("sehir", ""))
    user_id = aktif_kullanici_id()
    if user_id is None:
        locations = session.get("guest_locations", [])
        for item in locations:
            item["konum_tipi"] = "ana" if normalize_text(item.get("sehir", "")) == normalize_text(sehir) else "ek"
        session["guest_locations"] = locations
        session.modified = True
        flash("Ana konum güncellendi.", "success")
        return redirect(url_for("index"))
    try:
        with get_db_connection() as conn:
            exists = conn.execute("SELECT id FROM kullanici_konumlari WHERE user_id=? AND sehir=?", (user_id, sehir)).fetchone()
            if not exists:
                flash("Bu şehir kayıtlı konumlarınız arasında bulunmuyor.", "error")
            else:
                conn.execute("UPDATE kullanici_konumlari SET konum_tipi='ek' WHERE user_id=?", (user_id,))
                conn.execute("UPDATE kullanici_konumlari SET konum_tipi='ana' WHERE user_id=? AND sehir=?", (user_id, sehir))
                flash("Ana konum güncellendi.", "success")
    except sqlite3.Error:
        logger.exception("Ana konum güncellenemedi")
        flash("Ana konum güncellenirken hata oluştu.", "error")
    return redirect(url_for("index"))

@app.route("/hesap", methods=["GET", "POST"])
@erisim_gerekli
def hesap():
    user_id = aktif_kullanici_id()
    if user_id is None:
        return redirect(url_for("register"))
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        profile_photo = request.files.get("profile_photo")
        try:
            with get_db_connection() as conn:
                if fullname:
                    conn.execute("UPDATE kullanicilar SET fullname=? WHERE id=?", (fullname, user_id))
                conn.execute("UPDATE kullanicilar SET phone=? WHERE id=?", (phone, user_id))
                if password:
                    if len(password) < 6:
                        flash("Yeni şifre en az 6 karakter olmalıdır.", "error")
                        return redirect(url_for("hesap"))
                    conn.execute("UPDATE kullanicilar SET password_hash=? WHERE id=?", (generate_password_hash(password), user_id))
                if profile_photo and profile_photo.filename:
                    ext = profile_photo.filename.rsplit(".", 1)[-1].lower() if "." in profile_photo.filename else ""
                    if ext not in ALLOWED_PROFILE_EXTENSIONS:
                        flash("Profil fotoğrafı PNG, JPG, JPEG, WEBP veya GIF olmalıdır.", "error")
                        return redirect(url_for("hesap"))
                    filename = secure_filename(f"user_{user_id}_{int(time.time())}.{ext}")
                    profile_photo.save(os.path.join(PROFILE_UPLOAD_DIR, filename))
                    conn.execute("UPDATE kullanicilar SET profile_photo=? WHERE id=?", (filename, user_id))
            session["fullname"] = fullname or session.get("fullname", "")
            flash("Hesap bilgileriniz güncellendi.", "success")
        except sqlite3.Error:
            logger.exception("Hesap güncelleme hatası")
            flash("Hesap bilgileriniz güncellenirken hata oluştu.", "error")
    try:
        with get_db_connection() as conn:
            user = conn.execute("SELECT id,email,fullname,phone,profile_photo FROM kullanicilar WHERE id=?", (user_id,)).fetchone()
        if not user:
            session.clear()
            return redirect(url_for("login"))
    except sqlite3.Error:
        logger.exception("Profil okunamadı")
        flash("Profil bilgileri okunamadı.", "error")
        return redirect(url_for("index"))
    locations = kullanici_konumlarini_getir()
    location_items = []
    for item in locations:
        label = "🏠 Ana Konum" if item.get("konum_tipi") == "ana" else "📍 Ek Konum"
        city = item.get("sehir", "")
        location_items.append(
            f'<div style="padding:12px;margin:8px 0;border:1px solid rgba(95,177,255,.25);border-radius:12px;">'
            f'<strong>{label}</strong> — {city} '
            f'<a href="/takip-sil/{city}" style="float:right;color:#ff6b6b;">Sil</a></div>'
        )
    location_html = "".join(location_items) or "<p>Henüz kayıtlı konumunuz yok.</p>"
    photo = f'/static/uploads/profiles/{user["profile_photo"]}' if user["profile_photo"] else ''
    photo_html = f'<img src="{photo}" alt="Profil fotoğrafı" style="width:90px;height:90px;border-radius:50%;object-fit:cover;border:2px solid #28a9ff;">' if photo else '<div style="font-size:70px">👤</div>'
    fullname_value = user["fullname"] or ""
    email_value = user["email"] or ""
    phone_value = user["phone"] or ""
    html = f'''<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RiskAtlas Hesap Ayarları</title>
<style>body{{margin:0;background:#06111f;color:#eef6ff;font-family:Arial,sans-serif;padding:24px}}.card{{max-width:760px;margin:auto;background:#0b1b30;border:1px solid rgba(95,177,255,.25);border-radius:20px;padding:28px}}input{{width:100%;box-sizing:border-box;padding:12px;margin:7px 0 16px;border-radius:10px;border:1px solid #294663;background:#10243d;color:#fff}}button,.btn{{display:inline-block;padding:12px 18px;border-radius:10px;border:0;background:#159eea;color:white;font-weight:bold;text-decoration:none;cursor:pointer;margin:4px}}.muted{{color:#b8cbe0}}</style></head><body><div class="card">
<h1>👤 Hesap Ayarları</h1><p class="muted">Profilinizi ve RiskAtlas'taki kişisel konumlarınızı buradan yönetebilirsiniz.</p>{photo_html}
<form method="POST" enctype="multipart/form-data"><label>Ad Soyad</label><input name="fullname" value="{fullname_value}" required><label>E-Posta</label><input value="{email_value}" disabled><label>Telefon (isteğe bağlı)</label><input name="phone" value="{phone_value}" placeholder="05xx xxx xx xx"><label>Yeni Şifre (isteğe bağlı)</label><input type="password" name="password" placeholder="Değiştirmek istemiyorsanız boş bırakın"><label>Profil Fotoğrafı (isteğe bağlı)</label><input type="file" name="profile_photo" accept="image/png,image/jpeg,image/webp,image/gif"><button type="submit">Bilgilerimi Kaydet</button></form>
<h2>📍 Konumlarım ({len(locations)}/{MAX_USER_LOCATIONS})</h2>{location_html}<p><a class="btn" href="/">← RiskAtlas'a Dön</a><a class="btn" href="/logout">Çıkış Yap</a></p>
</div></body></html>'''
    return make_response(html)

# -----------------------------------------------------------------------------
# 8. BASIC ROUTES
# -----------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
@erisim_gerekli
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
    takip_listesi_json = "[]"
    son_analizler_listesi = []

    if os.path.exists(db_yolu):
        try:
            conn = sqlite3.connect(db_yolu)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='afet_verileri'
            """)
            tablo_var = cursor.fetchone() is not None

            if tablo_var:
                df = pd.read_sql_query("SELECT * FROM afet_verileri", conn)

                if "Sehir" in df.columns:
                    sehirler = tekil_ve_sirali(df["Sehir"].dropna().unique().tolist())

            conn.close()

        except Exception as e:
            logger.exception("Veritabanı okuma hatası")

    if not sehirler and os.path.exists(data_yolu):
        try:
            df = pd.read_csv(data_yolu, encoding="utf-8-sig")

            if "Sehir" in df.columns:
                sehirler = tekil_ve_sirali(df["Sehir"].dropna().unique().tolist())

        except Exception as e:
            logger.exception("CSV veri okuma hatası")

    if os.path.exists(ilce_yolu):
        try:
            ilce_df = pd.read_csv(ilce_yolu, encoding="utf-8-sig")

            if "Sehir" in ilce_df.columns and "Ilce" in ilce_df.columns:
                for sehir, grup in ilce_df.groupby("Sehir"):
                    sehir_temiz = gorunum_duzelt(sehir)
                    ilce_verileri[sehir_temiz] = tekil_ve_sirali(grup["Ilce"].dropna().unique().tolist())

        except Exception as e:
            logger.exception("İlçe CSV okuma hatası")

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
                    sehir_temiz = gorunum_duzelt(sehir)
                    ilce_temiz = gorunum_duzelt(ilce)
                    anahtar = f"{sehir_temiz}|||{ilce_temiz}"
                    mahalle_verileri[anahtar] = tekil_ve_sirali(grup["Mahalle"].dropna().unique().tolist())

        except Exception as e:
            logger.exception("Zemin CSV okuma hatası")

    tum_sehirler = set(sehirler)
    tum_sehirler.update(ilce_verileri.keys())

    if zemin_df is not None and not zemin_df.empty and "Sehir" in zemin_df.columns:
        tum_sehirler.update(zemin_df["Sehir"].dropna().unique().tolist())

    sehirler = tekil_ve_sirali(list(tum_sehirler))

    tum_depremler = canlı_depremleri_getir()

    ust_depremler = [
        d for d in tum_depremler
        if float(d.get("mag", 0)) >= 4
    ]

    critik_limit = 4.5
    kritik_depremler = [
        d for d in tum_depremler
        if float(d.get("mag", 0)) >= critik_limit
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

            bina_yasi = float(request.form.get("bina_yasi", 0) or 0)
            kat_sayisi = max(0, int(float(request.form.get("kat_sayisi", 0) or 0)))
            bulundugu_kat = max(0, int(float(request.form.get("bulundugu_kat", 0) or 0)))
            kisi_sayisi = max(1, int(float(request.form.get("kisi_sayisi", 1) or 1)))
            evcil_hayvan = request.form.get("evcil_hayvan") == "evet"
            erisim_ihtiyaci = request.form.get("erisim_ihtiyaci", "yok")

            otomatik_nufus, otomatik_yatak, otomatik_toplanma, otomatik_itfaiye = otomatik_model_degerleri(secilen_sehir, secilen_ilce)
            inputs = [otomatik_nufus, bina_yasi, otomatik_yatak, otomatik_toplanma, otomatik_itfaiye, zemin_riski]

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
                oneriler.extend(kullaniciye_ozel_oneriler_uret(
                    bina_yasi, kat_sayisi, bulundugu_kat, kisi_sayisi, evcil_hayvan, erisim_ihtiyaci
                ))
                oneriler = list(dict.fromkeys(oneriler))

                try:
                    conn = sqlite3.connect(db_yolu)
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT INTO analiz_kayitlari (
                            sehir, ilce, mahalle, risk_sonucu, risk_skoru, zemin_riski, tarih, user_id
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        secilen_sehir, secilen_ilce, secilen_mahalle, risk_durumu, risk_skoru, zemin_riski,
                        datetime.now().strftime("%d.%m.%Y %H:%M"), aktif_kullanici_id()
                    ))

                    conn.commit()
                    conn.close()

                    logger.info("Analiz veritabanına kaydedildi.")

                except Exception as db_hata:
                    logger.exception("Veritabanı kayıt hatası")

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
            logger.exception("Model yükleme/tahmin hatası")

    if os.path.exists(db_yolu):
        try:
            conn = sqlite3.connect(db_yolu)
            cursor = conn.cursor()
            user_id = aktif_kullanici_id()
            if user_id is not None:
                cursor.execute("SELECT sehir, ilce, risk_sonucu, tarih FROM analiz_kayitlari WHERE user_id=? ORDER BY id DESC LIMIT 5", (user_id,))
            else:
                cursor.execute("SELECT sehir, ilce, risk_sonucu, tarih FROM analiz_kayitlari WHERE 1=0")
            son_analizler_listesi = cursor.fetchall()
            conn.close()

            konumlar = kullanici_konumlarini_getir()
            takip_listesi_json = json.dumps([x.get("sehir") for x in konumlar], ensure_ascii=False)
        except Exception as e:
            logger.exception("Veritabanı panel verisi çekme hatası")

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
    landing_sehir_options = '<option value="">Hızlı Şehir Seçiniz</option>'

    for sehir in sehirler:
        selected = "selected" if sehir == secilen_sehir else ""
        sehir_options += f'<option value="{sehir}" {selected}>{sehir}</option>'
        landing_sehir_options += f'<option value="{sehir}">{sehir}</option>'

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

    if son_analizler_listesi:
        gecmis_panel_html = '<div class="earthquake-list" style="margin-top:20px;"><h3>📋 Son Veritabanı Geçmişi</h3><table style="width:100%; border-collapse:collapse;"><thead><tr style="text-align:left; background:rgba(0,0,0,0.2);"><th style="padding:10px;">Şehir</th><th style="padding:10px;">İlçe</th><th style="padding:10px;">Risk Durumu</th><th style="padding:10px;">Tarih</th></tr></thead><tbody>'
        for r in son_analizler_listesi:
            gecmis_panel_html += f'<tr style="border-bottom:1px solid var(--border);"><td style="padding:10px; font-weight:bold;">{r[0]}</td><td style="padding:10px;">{r[1]}</td><td style="padding:10px; color:var(--muted);">{r[2]}</td><td style="padding:10px; font-size:12px;">{r[3]}</td></tr>'
        gecmis_panel_html += '</tbody></table></div>'
    else:
        gecmis_panel_html = ""

    if tum_depremler:
        deprem_listesi_html = "".join([
            f"<li>{d.get('title', 'Bilinmeyen Konum')} - Büyüklük: {d.get('mag', '?')} - Kaynak: {d.get('kaynak', 'Bilinmiyor')}</li>"
            for d in tum_depremler[:15]
        ])
    else:
        deprem_listesi_html = "<li>Güncel deprem verisi alınamadı.</li>"

    deprem_verileri_json = json.dumps(tum_depremler[:30], ensure_ascii=False)

    takip_listesi_python = json.loads(takip_listesi_json)
    takip_badgeleri_html = ""
    for item in kullanici_konumlarini_getir():
        sehir_adi = item.get("sehir", "")
        tip = item.get("konum_tipi", "ek")
        etiket = "🏠 Ana" if tip == "ana" else "📍 Ek"
        ana_link = "" if tip == "ana" else f'<form method="POST" action="/ana-konum-yap" style="display:inline;"><input type="hidden" name="sehir" value="{sehir_adi}"><button type="submit" style="background:none;border:0;color:#8bd5ff;padding:0;cursor:pointer;font-size:12px;">Ana yap</button></form>'
        takip_badgeleri_html += f"""
            <span style="background:rgba(0,194,255,0.1); border:1px solid var(--blue2); color:#eaf6ff; padding:7px 10px; border-radius:8px; font-size:13px; font-weight:bold; display:inline-flex; align-items:center; gap:7px; margin:4px;">
                {etiket} {sehir_adi} {ana_link}
                <a href="/takip-sil/{sehir_adi}" style="color:var(--danger); text-decoration:none; font-weight:extrabold;">×</a>
            </span>"""

    aktif_ana_konum = ana_konum_getir()
    konum_sayisi = len(kullanici_konumlarini_getir())
    if misafir_mi():
        hesap_url = "/register"
        oturum_badge = "👤 Misafir Modu — keşif için giriş yaptınız"
        hesap_banner_html = '<div style="margin:14px auto;padding:14px 18px;max-width:760px;border-radius:14px;background:rgba(20,210,150,.10);border:1px solid rgba(20,210,150,.35);text-align:left;"><strong>✨ Misafir olarak devam ediyorsunuz.</strong><br><span style="color:#c8d8e8;">Siteyi hesap oluşturmadan keşfedebilirsiniz. Kaydettiğiniz konumlarınız hesap oluşturduğunuzda korunur.</span><br><a href="/register" style="display:inline-block;margin-top:9px;padding:9px 14px;border-radius:9px;background:#16b981;color:#fff;text-decoration:none;font-weight:bold;">Hesap Oluştur ve Konumlarımı Koru</a></div>'
    else:
        hesap_url = "/hesap"
        oturum_badge = f"👤 {session.get('fullname', 'Kullanıcı')}"
        hesap_banner_html = '<div style="margin:14px auto;padding:10px 14px;max-width:760px;border-radius:12px;background:rgba(95,177,255,.08);border:1px solid rgba(95,177,255,.22);text-align:center;"><a href="/hesap" style="color:#eaf4ff;text-decoration:none;font-weight:bold;">👤 Hesap Ayarları</a> · <a href="/logout" style="color:#ffb3b3;text-decoration:none;font-weight:bold;">Çıkış Yap</a></div>'

    html = f"""
    <!DOCTYPE html>
    <html lang="tr">

    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="theme-color" content="#081120">

        <script>
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

    <body id="mainBody">

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
                    <span style="display:inline-flex;align-items:center;padding:9px 12px;border-radius:10px;border:1px solid rgba(95,177,255,.22);background:rgba(95,177,255,.06);font-weight:bold;">{oturum_badge}</span>
                    <button type="button" onclick="location.href='{hesap_url}'" aria-label="Hesap ayarlarını aç">
                        👤 Hesap
                    </button>
                    <button type="button" onclick="location.href='/gecmis'" aria-label="Tüm veritabanı analiz geçmişini gör">
                        📋 Geçmiş Analizler
                    </button>
                    <button type="button" id="aiRobotToggleBtn" onclick="aiRobotAyariniDegistir()" style="background:rgba(128,0,128,0.22); border:1px solid purple;">
                        🤖 Yapay Zekâ Robotu: Açık
                    </button>
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

                    {hesap_banner_html}

                    <div class="location-symbol" aria-hidden="true">📍</div>

                    <h2 id="anaEvGosterge">{("Mevcut Ana Konumunuz: " + aktif_ana_konum) if aktif_ana_konum else "Henüz ana konum seçilmedi"}</h2>
                    <p>
                        4.5 ve üzeri depremlerde sadece bulunduğunuz bölge etkilenebiliyorsa sizi uyarır,
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
                        
                        <form method="POST" action="/takip-ekle" id="konumKayitFormu" style="width: 100%; display: flex; flex-direction: column; align-items: center; gap: 7px; margin: 6px 0;">
                            <select id="takipSehirSecimAlani" style="width: 100%; max-width: 320px;" aria-label="Türkiye şehirlerinden konum seçimi">
                                {landing_sehir_options}
                            </select>
                            <input id="takipSehirMetin" type="text" placeholder="Yurt dışı için şehir / ülke yazabilirsiniz" style="width:100%;max-width:320px;box-sizing:border-box;" aria-label="Yurt dışı şehir veya ülke">
                            <input type="hidden" id="takipSehirInput" name="takipSehirInput">
                            <input type="hidden" id="konumTipiInput" name="konumTipi" value="ek">
                            <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:7px;width:100%;max-width:500px;">
                                <button type="button" onclick="konumTipiniSecVeKaydet('ana')" style="background:linear-gradient(135deg,#159eea,#0077b6);">🏠 Ana Konum Olarak Kaydet</button>
                                <button type="button" onclick="konumTipiniSecVeKaydet('ek')" style="background:linear-gradient(135deg,#0ea5a8,#087f8c);">📍 Ek Konum Kaydet</button>
                            </div>
                            <small style="color:#b8cbe0;">En fazla 3 konum: 1 ana + 2 ek. Yurt dışındaki şehirleri de yazabilirsiniz.</small>
                        </form>

                        <div style="width:100%; text-align:center; max-width:500px; margin-bottom:10px;">
                            <strong>📍 Kayıtlı Konumlar: {konum_sayisi}/{MAX_USER_LOCATIONS}</strong><br>
                            {takip_badgeleri_html}
                        </div>

                        <button
                            type="button"
                            class="secondary-btn"
                            onclick="sehirSecerekDevamEt()"
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

            <p id="alertMainParagraph">{alarm_mesaji if alarm_mesaji else "Canlı deprem verisi kritik seviyeye ulaştı."}</p>

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

            <form method="POST" id="analizFormuElementi">

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

                <div style="padding:16px;border-radius:14px;background:rgba(21,158,234,.08);border:1px solid rgba(95,177,255,.20);margin-bottom:16px;">
                    <strong>ℹ️ Bu analizde sizden yalnızca bilebileceğiniz bilgiler istenir.</strong><br>
                    Nüfus, hastane kapasitesi, toplanma alanı ve itfaiye gibi bölgesel veriler kullanıcıdan istenmez; sistem bunları mevcut veri kaynaklarından otomatik olarak kullanır.
                </div>

                <label for="bina_yasi">🏠 Bina Yaşı</label>
                <input id="bina_yasi" type="number" min="0" max="200" step="1" name="bina_yasi" placeholder="Örn: 15" required>

                <label for="kat_sayisi">🏢 Binanız Kaç Katlı?</label>
                <input id="kat_sayisi" type="number" min="1" max="100" step="1" name="kat_sayisi" placeholder="Örn: 5" required>

                <label for="bulundugu_kat">📍 Hangi Katta Bulunuyorsunuz?</label>
                <input id="bulundugu_kat" type="number" min="0" max="100" step="1" name="bulundugu_kat" placeholder="Örn: 3" required>

                <label for="kisi_sayisi">👥 Evde Kaç Kişi Yaşıyor?</label>
                <input id="kisi_sayisi" type="number" min="1" max="100" step="1" name="kisi_sayisi" placeholder="Örn: 4" required>

                <label for="erisim_ihtiyaci">♿ Erişilebilirlik İhtiyacı</label>
                <select id="erisim_ihtiyaci" name="erisim_ihtiyaci">
                    <option value="yok">Belirtmek istemiyorum / Yok</option>
                    <option value="gorme">Görme desteği</option>
                    <option value="isitme">İşitme desteği</option>
                    <option value="hareket">Hareket desteği</option>
                </select>

                <label for="evcil_hayvan">🐾 Evcil Hayvanınız Var mı?</label>
                <select id="evcil_hayvan" name="evcil_hayvan">
                    <option value="hayir">Hayır</option>
                    <option value="evet">Evet</option>
                </select>

                <div class="zemin-info-box">
                    <b>🌍 Bölgesel Veriler:</b><br>
                    Zemin bilgisi ve diğer teknik altyapı verileri kullanıcıdan istenmez; sistem mevcut veri kaynaklarından otomatik olarak değerlendirir.
                </div>

                <br>
                <button type="submit">Analiz Et</button>

            </form>

            <div class="example-box">
                <b>📌 Neleri bilmeniz yeterli?</b><br><br>
                • Bina yaşınızı yaklaşık olarak bilmeniz yeterlidir.<br>
                • Binanızın toplam kat sayısını ve bulunduğunuz katı girin.<br>
                • Evde yaşayan kişi sayısını belirtin.<br>
                • İsterseniz erişilebilirlik ihtiyacınızı ve evcil hayvan bilgisini ekleyin.<br>
                • Nüfus, hastane, itfaiye, toplanma alanı ve zemin gibi teknik değerleri sizin araştırmanız gerekmez.
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

                {f'''
                <div style="margin-top:14px; padding:15px; border-radius:12px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); display:flex; align-items:center; justify-content:center; gap:12px;">
                    <span style="font-size:36px;">{"🟢" if risk_skoru==1 else "🟡" if risk_skoru==3 else "🔴"}</span>
                    <span style="font-weight:bold; font-size:16px;">Durum Durumu: {risk_durumu if risk_durumu else "Sistem Analiz Sonucu"}</span>
                </div>
                ''' if risk_skoru > 0 else ""}

                {zemin_bilgisi_html}

                <p>{aciklama}</p>
            </section>

            {gecmis_panel_html}

            <br>
            <button
                type="button"
                onclick="acilDurumGoster()"
            >
                🚨 Erişilebilir Acil Durum Alarmını Test Et
            </button>

            {f'''
            <div class="suggestion-box">
                <h3>🧭 Acil Durum Öneri Sistemi ve Tahmini Sesli Yön Bulucu</h3>
                <ul>{oneriler_html}</ul>
            </div>
            ''' if oneriler else ""}

            <div class="accessibility-note">
                <strong>♿ Erişilebilir Afet Modu:</strong><br><br>
                ✅ İşitme engelli bireyler için kırmızı yanıp sönen tam ekran görsel alarm ve mobil ritmik güçlü titreşim desenleri (Vibration API)<br>
                ✅ Mobil cihazlarda titreşim desteği<br>
                ✅ Görme engelli bireyler için varsayılan açık gelen, ilk etkileşimde veya sayfa yüklendiği an selamlama bittiğinde otomatik çalışan akıllı eller serbest asistan dinleme yapısı<br>
                ✅ Harita altında ekran okuyucu uyumlu deprem listesi ve veritabanı tabloları<br>
                ✅ Risk sonucuna göre renklendirilen şehir haritası<br>
                ✅ Büyük yazı ve yüksek kontrastlı acil durum ekranı ve zihinsel engelli bireyler için evrensel sembol tasarımı
            </div>
        </div>
        </main>

        <script>
            if (/Android|iPhone|iPad|iPod/i.test(navigator.userAgent) && "serviceWorker" in navigator) {{
                navigator.serviceWorker.register("/static/service-worker.js")
                .then(() => console.log("Service Worker kayıt edildi."))
                .catch(error => console.log("Service Worker hatası:", error));
            }}

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
            const pythonTakipListesi = {takip_listesi_json};

            let girisRehberiEtkilesimleBasladi = false;

            // KALICI HAFIZA VE ROBOT AYARLARI ALTYAPISI
            let aktifEvKonumu = {json.dumps(aktif_ana_konum, ensure_ascii=False)};
            let aiRobotAktifMi = localStorage.getItem("riskAtlasAiRobotAyar") === "acik" || localStorage.getItem("riskAtlasMikrofonIzni") === "acik";

            function aiRobotAyariniGuncelle() {{
                const btn = document.getElementById("aiRobotToggleBtn");
                if(!btn) return;
                if(aiRobotAktifMi) {{
                    btn.textContent = "🤖 Yapay Zekâ Robotu: Açık";
                    btn.style.background = "rgba(128,0,128,0.22)";
                }} else {{
                    btn.textContent = "🤖 Yapay Zekâ Robotu: Kapalı";
                    btn.style.background = "rgba(255,255,255,0.10)";
                    window.speechSynthesis.cancel();
                }}
            }}

            function aiRobotAyariniDegistir() {{
                aiRobotAktifMi = !aiRobotAktifMi;
                localStorage.setItem("riskAtlasAiRobotAyar", aiRobotAktifMi ? "acik" : "kapali");
                aiRobotAyariniGuncelle();
                if(aiRobotAktifMi) {{
                    robotKonusveSoruSor("Yapay zekâ robotu yeniden aktif hale getirildi. Size yardımcı olmak için dinliyorum.");
                }}
            }}

            function evKonumunuGuncelle(sehir) {{
                if(!sehir) return;
                aktifEvKonumu = sehir;
                localStorage.setItem("riskAtlasAnaEvKonumu", sehir);
                const gosterge = document.getElementById("anaEvGosterge");
                if(gosterge) gosterge.textContent = "Mevcut Ev Konumunuz: " + sehir + " (Hafızada Kayıtlı)";
                
                const sehirSelect = document.getElementById("sehir");
                if(sehirSelect) {{
                    sehirSelect.value = sehir;
                    ilceleriGuncelle();
                }}
                robotKonusveSoruSor("Ana ev konumunuz başarıyla " + sehir + " olarak güncellendi ve kalıcı hafızaya alındı.");
            }}

            function konumTipiniSecVeKaydet(tip) {{
                const select = document.getElementById("takipSehirSecimAlani");
                const text = document.getElementById("takipSehirMetin");
                const hidden = document.getElementById("takipSehirInput");
                const tipInput = document.getElementById("konumTipiInput");
                const city = (text && text.value.trim()) || (select && select.value) || "";
                if (!city) {{
                    alert("Lütfen önce bir şehir veya ülke seçin/yazın.");
                    return;
                }}
                if (hidden) hidden.value = city;
                if (tipInput) tipInput.value = tip;
                if (tip === "ana") {{
                    const form = document.getElementById("konumKayitFormu");
                    if (form) form.submit();
                }} else {{
                    const form = document.getElementById("konumKayitFormu");
                    if (form) form.submit();
                }}
            }}

            function buSehriTakibeEkle() {{
                konumTipiniSecVeKaydet('ek');
            }}

            function robotKonus(metin) {{
                if (!aiRobotAktifMi) return;
                if ("speechSynthesis" in window) {{
                    const mesaj = new SpeechSynthesisUtterance(metin);
                    mesaj.lang = "tr-TR";
                    mesaj.rate = 0.92;
                    window.speechSynthesis.cancel();
                    setTimeout(() => {{ window.speechSynthesis.speak(mesaj); }}, 150);
                }}
            }}

            function robotKonusveSoruSor(metin, soruMu = false) {{
                if (!aiRobotAktifMi) return;
                if ("speechSynthesis" in window) {{
                    const mesaj = new SpeechSynthesisUtterance(metin);
                    mesaj.lang = "tr-TR";
                    mesaj.rate = 0.92;
                    window.speechSynthesis.cancel();
                    mesaj.onend = function() {{
                        if(soruMu) {{
                            console.log("Robot soru sordu, dinleme modu tetikleniyor...");
                        }}
                    }};
                    setTimeout(() => {{ window.speechSynthesis.speak(mesaj); }}, 150);
                }}
            }}

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

            // INTERAKTİF SESLİ ASİSTAN VE YAPAY ZEKÂ ROBOTU (MİKROFON MOTORU)
            function otomatikSesliAsistanBaslat() {{
                if (!aiRobotAktifMi) return;
                if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
                    console.log("Tarayıcınız ses tanıma desteği sunmuyor.");
                    return;
                }}

                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                const recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = false;
                recognition.lang = 'tr-TR';

                recognition.onresult = function(event) {{
                    for (let i = event.resultIndex; i < event.results.length; ++i) {{
                        if (event.results[i].isFinal) {{
                            const komut = event.results[i][0].transcript.trim().toLowerCase();
                            console.log("Yapay Zekâ Algıladı:", komut);

                            if (komut.includes("analiz ekranı") || komut.includes("formu aç")) {{
                                sehirSecerekDevamEt();
                            }} else if (komut.includes("uyarıyı kapat") || komut.includes("alarmı kapat") || komut.includes("sustur")) {{
                                acilDurumKapat();
                            }} else if (komut.includes("analiz et") || komut.includes("hesapla")) {{
                                document.getElementById("analizFormuElementi").submit();
                            }} else if (komut.includes("evimi") || komut.includes("ana konumumu")) {{
                                let sehirBul = komut.replace("evimi", "").replace("ana konumumu", "").replace("yap", "").trim();
                                if(sehirBul) {{
                                    let sehirDüzgün = sehirBul.charAt(0).toUpperCase() + sehirBul.slice(1);
                                    evKonumunuGuncelle(sehirDüzgün);
                                }}
                            }} else if (komut.includes("yardım et") || komut.includes("neredeyim")) {{
                                robotKonusveSoruSor("Şu anda giriş ekranındasınız. Hafızadaki ev konumunuz " + aktifEvKonumu + " olarak ayarlanmıştır. Başka bir işlem yapmak ister misiniz?", true);
                            }}
                        }}
                    }}
                }};

                recognition.onend = function() {{
                    if (aiRobotAktifMi) {{
                        try {{ recognition.start(); }} catch(e) {{}}
                    }}
                }};

                try {{
                    recognition.start();
                }} catch(e) {{}}
            }}

            function seyahatListesiDepremDenetle() {{
                if (!pythonTakipListesi || pythonTakipListesi.length === 0) return;
                
                depremVerileri.forEach(function(d) {{
                    const mag = parseFloat(d.mag || 0);
                    const titleFix = d.title ? d.title.toLowerCase() : "";
                    
                    if (mag >= 4.5) {{
                        pythonTakipListesi.forEach(function(takipSehir) {{
                            const normTakip = takipSehir.toLowerCase();
                            if (titleFix.includes(normTakip)) {{
                                const uyariMetni = "Dikkat! Seyahat listenizdeki " + takipSehir + " bölgesinde " + mag + " büyüklüğünde kritik deprem tespit edildi! Güvenli yerlere geçin.";
                                document.getElementById("alertMainParagraph").innerText = uyariMetni;
                                
                                if (navigator.vibrate) {{
                                    navigator.vibrate([400, 200, 400, 200, 800, 200, 400]);
                                }}
                                
                                document.getElementById("emergencyAlert").style.display = "block";
                                robotKonus(uyariMetni);
                            }}
                        }});
                    }}
                }});
            }}

            function girisSesliAciklama(kaynak) {{
                if (!aiRobotAktifMi) return;
                if (kaynak === 'manual' || kaynak === 'firstInteraction') {{
                    girisRehberiEtkilesimleBasladi = true;
                }}

                const metin =
                    "RiskAtlas interaktif yapay zekâ sesli asistan sistemine hoş geldiniz. " +
                    "Uygulama hafızası etkindir. Şu anda ev konumunuz kalıcı olarak " + aktifEvKonumu + " şeklinde ayarlanmıştır. " +
                    "Her girişte form doldurmak zorunda kalmazsınız. Seyahat listenizdeki ek şehirlerde risk algılandığında sistem sizi sesle uyaracaktır. " +
                    "Sesli robotumuz sizi dinlemektedir, bana komut verebilir veya soru sorabilirsiniz.";

                if ("speechSynthesis" in window) {{
                    window.speechSynthesis.cancel();
                    const mesaj1 = new SpeechSynthesisUtterance(metin);
                    mesaj1.lang = "tr-TR";
                    mesaj1.rate = 0.92;
                    mesaj1.onend = function () {{
                        otomatikSesliAsistanBaslat();
                    }};
                    window.speechSynthesis.speak(mesaj1);
                }}
            }}

            function ilkEtkilesimdeSesliRehberiBaslat() {{
                if (!aiRobotAktifMi) return;
                if (girisRehberiEtkilesimleBasladi) return;
                girisSesliAciklama('firstInteraction');
            }}

            function sehirSecerekDevamEt() {{
                detayliAnalizeGec();
            }}

            function detayliAnalizeGec() {{
                document.getElementById("landingScreen").style.display = "none";
                document.getElementById("mainContent").classList.add("active");
                robotKonus("Detaylı analiz ekranına geçildi. Harita ve parametre formları yüklendi.");
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
                    }} catch (e) {{}}
                }});

                if (yakinKritik) {{
                    document.getElementById("konumDurumu").innerHTML =
                        "Yakınınızda kritik deprem algılandı: " + yakinKritik.title + " - Büyüklük: " + yakinKritik.mag;
                    robotKonus("Dikkat. Yakınınızda kritik seviyede deprem uyarısı var. Güvenli alanlara geçiş yapın.");
                    acilDurumGoster();
                }} else {{
                    document.getElementById("konumDurumu").innerHTML = "Konum tarandı. Yakınınızda aktif kritik alarm bulunmuyor.";
                    robotKonus("Konum tarandı. Yakınınızda aktif kritik alarm bulunmuyor.");
                }}
            }}

            function konumModunuBaslat() {{
                robotKonus("Konum analizi başlatılıyor. Lütfen gelen tarayıcı penceresinde izin ver seçeneğini onaylayın.");
                const durum = document.getElementById("konumDurumu");
                if (!navigator.geolocation) {{
                    durum.innerHTML = "Bu cihaz konum özelliğini desteklemiyor.";
                    return;
                }}
                durum.innerHTML = "Konum verisi alınıyor...";
                navigator.geolocation.getCurrentPosition(
                    function(position) {{
                        yakinDepremKontrolEt(position.coords.latitude, position.coords.longitude);
                    }},
                    function(error) {{
                        durum.innerHTML = "Konum izni reddedildi.";
                        robotKonus("Konum izni alınamadı. İşleme şehir seçerek devam edebilirsiniz.");
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

            function acilDurumGoster() {{
                document.getElementById("emergencyAlert").style.display = "block";
                if (navigator.vibrate) navigator.vibrate([500, 300, 500, 300, 1000]);
            }}

            function acilDurumKapat() {{
                document.getElementById("emergencyAlert").style.display = "none";
                if (navigator.vibrate) navigator.vibrate(0);
                window.speechSynthesis.cancel();
            }}

            window.onload = function () {{
                // Hafızadaki ev konumunu başlangıçta yükle ve göstergeyi ayarla
                const gosterge = document.getElementById("anaEvGosterge");
                if(gosterge) gosterge.textContent = aktifEvKonumu ? "Mevcut Ana Konumunuz: " + aktifEvKonumu : "Henüz ana konum seçilmedi";
                
                const sehirSelect = document.getElementById("sehir");
                if(sehirSelect && aktifEvKonumu) {{
                    sehirSelect.value = aktifEvKonumu;
                    setTimeout(() => {{ ilceleriGuncelle(); }}, 300);
                }}

                seyahatListesiDepremDenetle();
                sesliYonlendirmeButonunuGuncelle();
                aiRobotAyariniGuncelle();

                setTimeout(function () {{
                    const splash = document.getElementById("splash-screen");
                    if (splash) {{
                        splash.classList.add("fade-out");
                        setTimeout(function () {{ splash.remove(); }}, 800);
                    }}
                }}, 1800);

                // Sesli yardım giriş ekranında açıkça etkinleştirildiyse devam eder.
                // Ana sayfa kendi kendine mikrofon açmaz.
                if (aiRobotAktifMi && localStorage.getItem("riskAtlasMikrofonIzni") === "acik") {{
                    setTimeout(() => {{ girisSesliAciklama('auto'); }}, 900);
                }}
            }};
        </script>

    </body>
    </html>
    """

    return make_response(html)


@app.route("/gecmis")
@erisim_gerekli
def gecmis():
    try:
        conn = sqlite3.connect(db_yolu)
        df = pd.read_sql_query("""
            SELECT
                id,
                sehir,
                ilce,
                mahalle,
                risk_sonucu,
                risk_skoru,
                zemin_riski,
                tarih
            FROM analiz_kayitlari
            WHERE user_id = ?
            ORDER BY id DESC
        """, conn, params=(aktif_kullanici_id(),))
        conn.close()

        if df.empty:
            tablo_html = "<p>Henüz kayıtlı analiz sonucu bulunmuyor.</p>"
        else:
            tablo_html = df.to_html(index=False, classes="history-table", border=0)

    except Exception as e:
        tablo_html = f"<p>Veritabanı okunurken hata oluştu: {e}</p>"

    html = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RiskAtlas Analiz Geçmişi</title>
        <style>
            body {{
                background:#06111f;
                color:#eaf4ff;
                font-family:Arial, sans-serif;
                padding:20px;
            }}
            .box {{
                max-width:1200px;
                margin:auto;
                background:rgba(12,29,52,0.92);
                border:1px solid rgba(95,177,255,0.25);
                border-radius:18px;
                padding:22px;
                overflow-x:auto;
            }}
            a {{
                color:#00c2ff;
                text-decoration:none;
                font-weight:bold;
            }}
            table {{
                width:100%;
                border-collapse:collapse;
                margin-top:18px;
            }}
            th, td {{
                border:1px solid rgba(95,177,255,0.25);
                padding:10px;
                text-align:left;
            }}
            th {{
                background:#0b1e35;
            }}
            tr:nth-child(even) {{
                background:rgba(255,255,255,0.04);
            }}
        </style>
    </head>
    <body>
        <div class="box">
            <h1>📋 RiskAtlas Analiz Geçmişi</h1>
            <p><a href="/">← Ana sayfaya dön</a></p>
            {tablo_html}
        </div>
    </body>
    </html>
    """

    return make_response(html)


if __name__ == "__main__":
    # Flask development server is retained for local development only.
    # Production should be served by a WSGI server such as Gunicorn/Waitress.
    is_debug = ENVIRONMENT == "development"
    port = int(os.environ.get("PORT", "5000"))
    logger.info(
        "RiskAtlas V2 başlatılıyor | ortam=%s | port=%s | debug=%s",
        ENVIRONMENT,
        port,
        is_debug,
    )
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=port,
        debug=is_debug,
    )
