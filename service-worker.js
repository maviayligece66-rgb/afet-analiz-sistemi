self.addEventListener("install", function (event) {
    console.log("Service Worker kuruldu.");
});

self.addEventListener("activate", function (event) {
    console.log("Service Worker aktif.");
});

self.addEventListener("fetch", function (event) {
    event.respondWith(fetch(event.request));
});