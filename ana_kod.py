import pandas as pd
import numpy as np

def veri_yukle():
    # Gerçek projede buraya TÜİK'ten indirdiğin 81 ilin dosyasını koyacağız
    # Şimdilik dosya yoksa hata vermemesi için örnek bir yapı kuralım:
    try:
        # Eğer 'data/tuik_afet_verisi.csv' dosyan hazırsa bunu kullanır
        df_81 = pd.read_csv('data/tuik_afet_verisi.csv')
    except FileNotFoundError:
        # Dosya henüz yoksa 81 il için rastgele ama mantıklı veriler üretelim (Test için)
        iller = ["Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Amasya", "Ankara", "Antalya", "Artvin", "Aydın", "Balıkesir", "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır", "Edirne", "Elazığ", "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", "Hakkari", "Hatay", "Isparta", "Mersin", "İstanbul", "İzmir", "Kars", "Kastamonu", "Kayseri", "Kırklareli", "Kırşehir", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa", "Kahramanmaraş", "Mardin", "Muğla", "Muş", "Nevşehir", "Niğde", "Ordu", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Tekirdağ", "Tokat", "Trabzon", "Tunceli", "Şanlıurfa", "Uşak", "Van", "Yozgat", "Zonguldak", "Aksaray", "Bayburt", "Karaman", "Kırıkkale", "Batman", "Şırnak", "Bartın", "Ardahan", "Iğdır", "Yalova", "Karabük", "Kilis", "Osmaniye", "Düzce"]
        
        data = {
            'Sehir': iller,
            'Bina_Yas_Ortalamasi': np.random.randint(15, 35, size=81),
            'Hastane_Yatak_Kapasitesi': np.random.uniform(1.5, 5.0, size=81),
            'Itfaiye_Ekip_Sayisi': np.random.randint(10, 150, size=81),
            'Toplanma_Alani_Sayisi': np.random.randint(100, 4000, size=81),
            'Nufus_Yogunlugu': np.random.randint(50, 3000, size=81)
        }
        df_81 = pd.DataFrame(data)
    
    return df_81

# 1. Fonksiyonu çalıştır ve veriyi al
df_sonuc = veri_yukle()

# 2. Üstüne o hesaplamaları yap (Eski koddaki hesaplama satırlarını buraya ekle)
df_sonuc['Direnc_Skoru'] = (df_sonuc['Hastane_Yatak_Kapasitesi'] * df_sonuc['Itfaiye_Ekip_Sayisi']) / (df_sonuc['Nufus_Yogunlugu'] / 100)
df_sonuc['Risk_Skoru'] = df_sonuc['Bina_Yas_Ortalamasi'] * (df_sonuc['Nufus_Yogunlugu'] / 1000)

# 3. VE EN ÖNEMLİSİ: Terminale yazdır
print("\n--- 81 İL AFET ANALİZ SONUÇLARI ---")
print(df_sonuc[['Sehir', 'Direnc_Skoru', 'Risk_Skoru']].head(20)) # İlk 20 ili gösterir

import joblib # Eğer yüklü değilse: pip install joblib

# Sonuçları 'cikti_verisi.csv' olarak kaydet (Web sitesi buradan okuyacak)
df_sonuc.to_csv('datasets/processed_afet_verisi.csv', index=False)

# K-Means modelini kaydet (Gelecekte yeni veri tahmini için)
joblib.dump(kmeans, 'models/afet_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\n--- Model ve Veriler Başarıyla Kaydedildi! ---")