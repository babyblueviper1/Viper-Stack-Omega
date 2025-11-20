const CACHE_NAME = 'omega-pruner-v8.4';

// Files to cache for offline use
const FILES_TO_CACHE = [
  '/',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  // Add your main CSS/JS if you want faster repeat visits (optional but recommended)
  // '/static/css/style.css',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(FILES_TO_CACHE);
    })
  );
  // Force the waiting service worker to become the active one
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keyList => {
      return Promise.all(
        keyList.map(key => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    })
  );
  // Take control of all clients immediately
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  // Only cache GET requests
  if (e.request.method !== 'GET') return;

  e.respondWith(
    caches.match(e.request).then(cachedResponse => {
      // Return cached version OR fetch from network
      return cachedResponse || fetch(e.request).then(networkResponse => {
        // Optional: cache new responses dynamically (except for API calls)
        if (networkResponse && networkResponse.status === 200 && !e.request.url.includes('/api/')) {
          return caches.open(CACHE_NAME).then(cache => {
            cache.put(e.request, networkResponse.clone());
            return networkResponse;
          });
        }
        return networkResponse;
      });
    }).catch(() => {
      // Fully offline fallback (optional — you can serve a custom offline page)
      if (e.request.destination === 'document') {
        return caches.match('/');
      }
    })
  );
});
