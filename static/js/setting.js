/**
 * RiskAtlas - Ayarlar sayfası (templates/setting.html)
 * Anahtarları okur, anlık önizleme yapar ve POST /setting ile sunucuya kaydeder.
 */
(() => {
    'use strict';

    // setting.html'deki checkbox id'leri -> sunucudaki ayar anahtarları
    const ANAHTARLAR = {
        voiceAssistant: 'voice_assistant',
        voiceLogin: 'voice_login',
        voiceAnalysis: 'voice_analysis',
        blindMode: 'blind_mode',
        hearingMode: 'hearing_mode',
        largeText: 'large_text',
        contrastMode: 'high_contrast',
        soundAlarm: 'sound_alarm',
        vibration: 'vibration',
        flashAlert: 'flash_alert'
    };

    const kaydetBtn = document.getElementById('saveSettings');
    if (!kaydetBtn) return;

    const durum = document.createElement('p');
    durum.id = 'settingsStatus';
    durum.className = 'settings-status';
    durum.setAttribute('role', 'status');
    durum.setAttribute('aria-live', 'polite');
    kaydetBtn.insertAdjacentElement('afterend', durum);

    const oku = () => {
        const ayarlar = {};
        Object.entries(ANAHTARLAR).forEach(([id, anahtar]) => {
            const el = document.getElementById(id);
            ayarlar[anahtar] = Boolean(el && el.checked);
        });
        return ayarlar;
    };

    // Büyük yazı ve yüksek kontrast, kaydetmeden önce de anında görülsün
    const onizle = () => {
        const a = oku();
        document.documentElement.classList.toggle('ayar-buyuk-yazi', a.large_text);
        document.documentElement.classList.toggle('ayar-yuksek-kontrast', a.high_contrast);
    };

    Object.keys(ANAHTARLAR).forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', onizle);
    });
    onizle();

    const mesaj = (metin, hata) => {
        durum.textContent = metin;
        durum.classList.toggle('hata', Boolean(hata));
    };

    kaydetBtn.addEventListener('click', async () => {
        kaydetBtn.disabled = true;
        mesaj('Kaydediliyor...');
        try {
            const yanit = await fetch('/setting', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(oku())
            });
            const veri = await yanit.json();
            if (!yanit.ok || !veri.ok) throw new Error('kayit-basarisiz');
            mesaj('Ayarlarınız kaydedildi.');
        } catch (e) {
            mesaj('Ayarlar kaydedilemedi. Lütfen tekrar deneyin.', true);
        } finally {
            kaydetBtn.disabled = false;
        }
    });
})();
