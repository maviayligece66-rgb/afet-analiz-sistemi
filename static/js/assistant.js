const synth = window.speechSynthesis;

function atlasKonus(metin) {

    if (!("speechSynthesis" in window))
        return;

    synth.cancel();

    const ses = new SpeechSynthesisUtterance(metin);

    ses.lang = "tr-TR";

    ses.rate = 1;

    ses.pitch = 1;

    synth.speak(ses);

}

window.addEventListener("load", () => {

    setTimeout(() => {

        atlasKonus(

            "Merhaba. RiskAtlas'a hoş geldiniz. Giriş yapmak için e posta adresinizi ve şifrenizi giriniz. Sesli yardım için mor butona basabilirsiniz."

        );

    }, 1200);

});

document.getElementById("voiceLogin")?.addEventListener("click", () => {

    atlasKonus(

        "Sesli yardım aktif edildi. Önce e posta adresinizi giriniz. Daha sonra şifrenizi yazınız ve giriş yap butonuna basınız."

    );

});