/**
 * @fileoverview RiskAtlas V2 - Gelişmiş Giriş İş Mantığı ve Form Yönetimi
 * @version 2.0.0
 * @author RiskAtlas Dev Team
 * @description SOLID, Clean Code ve ES2022+ standartlarına uygun Vanilla JS motoru.
 */

(() => {
    'use strict';

    /**
     * @constant {Object} CONFIG Uygulama genelinde kullanılan Magic String ve yapılandırma sabitleri.
     */
    const CONFIG = {
        MIN_PASSWORD_LENGTH: 6,
        CLASSES: {
            VISIBLE: 'visible',
            INVALID: 'is-invalid',
            VALID: 'is-valid',
            SPINNER: 'spinner-inline',
            ICON_EYE: 'fa-eye',
            ICON_EYE_SLASH: 'fa-eye-slash'
        },
        ATTRIBUTES: {
            TYPE: 'type',
            ARIA_LABEL: 'aria-label',
            ARIA_BUSY: 'aria-busy',
            TEXT: 'text',
            PASSWORD: 'password'
        },
        LABELS: {
            SHOW_PASSWORD: 'Şifreyi göster',
            HIDE_PASSWORD: 'Şifreyi gizle',
            LOADING: 'Giriş yapılıyor...'
        },
        REGEXP: {
            EMAIL: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
        },
        KEYS: {
            CAPS_LOCK: 'CapsLock',
            ESCAPE: 'Escape',
            ENTER: 'Enter'
        },
        ERRORS: {
            EMAIL_EMPTY: 'E-posta alanı boş bırakılamaz.',
            EMAIL_INVALID: 'Lütfen geçerli bir e-posta adresi giriniz.',
            PASSWORD_EMPTY: 'Şifre alanı boş bırakılamaz.',
            PASSWORD_SHORT: 'Şifreniz en az 6 karakter olmalıdır.',
            GENERAL_VALIDATION: 'Lütfen formdaki eksik veya hatalı alanları düzeltiniz.'
        }
    };

    /**
     * @type {Object} DOM Tek seferlik sorgulanan DOM element referansları deposu (DOM Cache).
     */
    const DOM = {
        form: null,
        email: null,
        password: null,
        togglePasswordBtn: null,
        togglePasswordIcon: null,
        capsLockWarning: null,
        submitBtn: null,
        submitBtnText: null,
        submitIcon: null,
        jsErrorSummary: null,
        voiceLoginBtn: null,
        emailError: null,
        passwordError: null
    };

    /**
     * DOM Elementlerini tek seferde önbelleğe alan ilklendirici.
     * @function cacheElements
     */
    const cacheElements = () => {
        DOM.form = document.getElementById('loginForm');
        DOM.email = document.getElementById('email');
        DOM.password = document.getElementById('password');
        DOM.togglePasswordBtn = document.getElementById('togglePassword');
        DOM.togglePasswordIcon = document.getElementById('togglePasswordIcon');
        DOM.capsLockWarning = document.getElementById('capsLockWarning');
        DOM.submitBtn = document.getElementById('submitBtn');
        DOM.submitBtnText = document.getElementById('submitBtnText');
        DOM.submitIcon = document.getElementById('submitIcon');
        DOM.jsErrorSummary = document.getElementById('jsErrorSummary');
        DOM.voiceLoginBtn = document.getElementById('voiceLogin');

        // Dinamik alan seviyeli hata elementleri (Defensive fallback dahil)
        DOM.emailError = document.getElementById('email-error');
        DOM.passwordError = document.getElementById('password-error');
    };

    /**
     * Şifre göster/gizle butonunun iş mantığı.
     * @function initializePasswordToggle
     */
    const initializePasswordToggle = () => {
        if (!DOM.togglePasswordBtn || !DOM.password || !DOM.togglePasswordIcon) return;

        DOM.togglePasswordBtn.addEventListener('click', () => {
            const isPassword = DOM.password.getAttribute(CONFIG.ATTRIBUTES.TYPE) === CONFIG.ATTRIBUTES.PASSWORD;

            DOM.password.setAttribute(CONFIG.ATTRIBUTES.TYPE, isPassword ? CONFIG.ATTRIBUTES.TEXT : CONFIG.ATTRIBUTES.PASSWORD);

            // Icon Class Senkronizasyonu
            DOM.togglePasswordIcon.classList.toggle(CONFIG.CLASSES.ICON_EYE, isPassword);
            DOM.togglePasswordIcon.classList.toggle(CONFIG.CLASSES.ICON_EYE_SLASH, !isPassword);

            // ARIA & Title Senkronizasyonu
            const labelText = isPassword ? CONFIG.LABELS.HIDE_PASSWORD : CONFIG.LABELS.SHOW_PASSWORD;
            DOM.togglePasswordBtn.setAttribute(CONFIG.ATTRIBUTES.ARIA_LABEL, labelText);
            DOM.togglePasswordBtn.title = labelText;
        });
    };

    /**
     * Caps Lock durumunu algılayan ve arayüzü tetikleyen mekanizma.
     * @function initializeCapsLock
     */
    const initializeCapsLock = () => {
        if (!DOM.password || !DOM.capsLockWarning) return;

        /**
         * @param {KeyboardEvent} event 
         */
        const handleCapsLockState = (event) => {
            if (event.getModifierState && event.getModifierState(CONFIG.KEYS.CAPS_LOCK)) {
                DOM.capsLockWarning.classList.add(CONFIG.CLASSES.VISIBLE);
            } else {
                DOM.capsLockWarning.classList.remove(CONFIG.CLASSES.VISIBLE);
            }
        };

        DOM.password.addEventListener('keydown', handleCapsLockState);
        DOM.password.addEventListener('keyup', handleCapsLockState);
        DOM.password.addEventListener('blur', () => DOM.capsLockWarning.classList.remove(CONFIG.CLASSES.VISIBLE));
    };

    /**
     * Form elemanlarındaki hata ve geçerlilik sınıflarını yöneten servis.
     * @class FieldUIController
     */
    const FieldUIController = {
        /**
         * Elemanı hatalı (geçersiz) olarak işaretler.
         * @param {HTMLInputElement} inputElement 
         * @param {HTMLElement} errorElement 
         * @param {string} errorMessage 
         */
        setError(inputElement, errorElement, errorMessage) {
            if (!inputElement) return;
            inputElement.classList.add(CONFIG.CLASSES.INVALID);
            inputElement.classList.remove(CONFIG.CLASSES.VALID);

            if (errorElement) {
                errorElement.textContent = errorMessage;
                errorElement.classList.remove('visually-hidden');
            }
        },

        /**
         * Elemanı başarılı (geçerli) olarak işaretler.
         * @param {HTMLInputElement} inputElement 
         * @param {HTMLElement} errorElement 
         */
        setValid(inputElement, errorElement) {
            if (!inputElement) return;
            inputElement.classList.remove(CONFIG.CLASSES.INVALID);
            inputElement.classList.add(CONFIG.CLASSES.VALID);

            if (errorElement) {
                errorElement.textContent = '';
                errorElement.classList.add('visually-hidden');
            }
        },

        /**
         * Elemanın tüm durum sınıflarını temizler.
         * @param {HTMLInputElement} inputElement 
         * @param {HTMLElement} errorElement 
         */
        clearState(inputElement, errorElement) {
            if (!inputElement) return;
            inputElement.classList.remove(CONFIG.CLASSES.INVALID, CONFIG.CLASSES.VALID);

            if (errorElement) {
                errorElement.textContent = '';
                errorElement.classList.add('visually-hidden');
            }
        }
    };

    /**
     * Gelişmiş girdi doğrulama mimarisi.
     * @function initializeValidation
     * @returns {Object} validate fonksiyonunu barındıran kontrol nesnesi.
     */
    const initializeValidation = () => {
        if (!DOM.email || !DOM.password) return { validate: () => false };

        // Dinamik girdi takibi (Kullanıcı yazarken hatayı anlık temizleme)
        DOM.email.addEventListener('input', () => FieldUIController.clearState(DOM.email, DOM.emailError));
        DOM.password.addEventListener('input', () => FieldUIController.clearState(DOM.password, DOM.passwordError));

        const validate = () => {
            let errorStack = [];
            const emailValue = DOM.email.value.trim();
            const passwordValue = DOM.password.value;

            // 1. E-Posta Validasyon Kuralları
            if (!emailValue) {
                errorStack.push(CONFIG.ERRORS.EMAIL_EMPTY);
                FieldUIController.setError(DOM.email, DOM.emailError, CONFIG.ERRORS.EMAIL_EMPTY);
            } else if (!CONFIG.REGEXP.EMAIL.test(emailValue.toLowerCase())) {
                errorStack.push(CONFIG.ERRORS.EMAIL_INVALID);
                FieldUIController.setError(DOM.email, DOM.emailError, CONFIG.ERRORS.EMAIL_INVALID);
            } else {
                FieldUIController.setValid(DOM.email, DOM.emailError);
            }

            // 2. Şifre Validasyon Kuralları (Backend / WTForms senkronize)
            if (!passwordValue) {
                errorStack.push(CONFIG.ERRORS.PASSWORD_EMPTY);
                FieldUIController.setError(DOM.password, DOM.passwordError, CONFIG.ERRORS.PASSWORD_EMPTY);
            } else if (passwordValue.length < CONFIG.MIN_PASSWORD_LENGTH) {
                errorStack.push(CONFIG.ERRORS.PASSWORD_SHORT);
                FieldUIController.setError(DOM.password, DOM.passwordError, CONFIG.ERRORS.PASSWORD_SHORT);
            } else {
                FieldUIController.setValid(DOM.password, DOM.passwordError);
            }

            return errorStack;
        };

        return { validate };
    };

    /**
     * Küresel Form Gönderim ve Yüklenme (Loading) Yönetimi.
     * @class SubmitController
     */
    const SubmitController = {
        /**
         * Arayüzü yükleniyor moduna sokar veya normal moda geri döndürür.
         * @param {boolean} isLoading 
         */
        setLoading(isLoading) {
            if (!DOM.submitBtn || !DOM.submitBtnText) return;

            DOM.submitBtn.disabled = isLoading;
            DOM.submitBtn.setAttribute(CONFIG.ATTRIBUTES.ARIA_BUSY, isLoading ? 'true' : 'false');

            if (isLoading) {
                // Orijinal ikonu yedekle ve spinner ata
                if (DOM.submitIcon) {
                    DOM.submitIcon.dataset.originalClass = DOM.submitIcon.className;
                    DOM.submitIcon.className = CONFIG.CLASSES.SPINNER;
                }
                DOM.submitBtnText.textContent = CONFIG.LABELS.LOADING;
            } else {
                // İkonu ve metni eski haline getir (Başarısız denemeler için recovery)
                if (DOM.submitIcon && DOM.submitIcon.dataset.originalClass) {
                    DOM.submitIcon.className = DOM.submitIcon.dataset.originalClass;
                }
                DOM.submitBtnText.textContent = 'Giriş Yap';
            }
        }
    };

    /**
     * Erişilebilirlik (ARIA) ve Klavye Navigasyon Geliştirmeleri.
     * @function initializeAccessibility
     */
    const initializeAccessibility = () => {
        // Global Esc Tuşu Takibi (Hata panelini kapatmak için)
        document.addEventListener('keydown', (event) => {
            if (event.key === CONFIG.KEYS.ESCAPE && DOM.jsErrorSummary) {
                DOM.jsErrorSummary.classList.remove(CONFIG.CLASSES.VISIBLE);
            }
        });
    };

    /**
     * Sesli Asistan Buton Entegrasyonu (assistant.js Korumalı Güvenli Katman).
     * @function initializeVoiceButton
     */
    const initializeVoiceButton = () => {
        if (!DOM.voiceLoginBtn) return;

        // assistant.js tarafından atanmış bir click event listener'ı veya küresel durum varsa çakışma önlenir.
        if (DOM.voiceLoginBtn.dataset.listenerInitialized) return;

        DOM.voiceLoginBtn.dataset.listenerInitialized = 'true';

        // Ekstra yerel arayüz iş mantığı gereksinimleri buraya modüler eklenebilir.
    };

    /**
     * Tüm alt modülleri organize eden ve form submit olayını yöneten ana başlatıcı.
     * @function initializeLogin
     */
    const initializeLogin = () => {
        cacheElements();

        if (!DOM.form) return;

        // Modüllerin Ayağa Kaldırılması (SRP Prensibi)
        initializePasswordToggle();
        initializeCapsLock();
        initializeAccessibility();
        initializeVoiceButton();

        const validator = initializeValidation();

        // Form Submit Dinleyicisi
        DOM.form.addEventListener('submit', (event) => {
            // Önceki üst hata özetini temizle
            if (DOM.jsErrorSummary) {
                DOM.jsErrorSummary.classList.remove(CONFIG.CLASSES.VISIBLE);
                DOM.jsErrorSummary.innerHTML = '';
            }

            const errors = validator.validate();

            if (errors.length > 0) {
                event.preventDefault();

                // JS Üst Hata Paneli Güncellemesi (Aria-live duyurusu için)
                if (DOM.jsErrorSummary) {
                    DOM.jsErrorSummary.innerHTML = errors.join('<br>');
                    DOM.jsErrorSummary.classList.add(CONFIG.CLASSES.VISIBLE);
                }

                // Başarısızlık durumunda buton durumunu güvene al (Aktif bırak)
                SubmitController.setLoading(false);

                window.scrollTo({ top: 0, behavior: 'smooth' });
                return false;
            }

            // Doğrulama başarılı ise Loading animasyonunu tetikle
            SubmitController.setLoading(true);
        });
    };

    // DOM Hazır olduğunda ana motoru tetikle
    document.addEventListener('DOMContentLoaded', initializeLogin);

})();