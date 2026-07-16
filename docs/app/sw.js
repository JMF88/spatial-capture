// Minimal offline shell for Trove (installable PWA). Cache-first for the app files.
const CACHE = "trove-v3";   // v3: no picker accept filter (iOS greys unknown extensions)
const ASSETS = ["./", "index.html", "styles.css", "app.js", "manifest.webmanifest",
                "icon.svg", "assets/collection.json"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});
