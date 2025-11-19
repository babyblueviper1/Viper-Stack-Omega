self.addEventListener('install', e => {
  e.waitUntil(
    caches.open('omega-cache').then(cache => {
      return cache.addAll([
        '/',
        '/icon-192.png',
        '/icon-512.png'
      ]);
    })
  );
});

self.addEventListener('fetch', e => {
  e.respondWith(
    caches.match(e.request).then(response => {
      return response || fetch(e.request);
    })
  );
});
