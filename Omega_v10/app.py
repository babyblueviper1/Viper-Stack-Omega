from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
import os
from gradio import mount_gradio_app

app = FastAPI()

# Import your demos (no changes here)
from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es

# Mount English at root
app = mount_gradio_app(
    app,
    demo_en,
    path="/",
    root_path=""  # Explicit for root; prevents double-prefixing on proxies
)

# Mount Spanish at /es
app = mount_gradio_app(
    app,
    demo_es,
    path="/es",
    root_path="/es"  # Prefixes all endpoints/assets (e.g., /es/info, /es.html)
)

# Catchall redirect for Gradio's .html shares (covers /es.html and variants)
@app.get("/{path:path}.html")  # Matches any /es.html, /foo.html, etc.
async def catch_html_redirect(path: str):
    # Redirect to the base path (e.g., /es.html → /es)
    if path == "es":
        return RedirectResponse("/es", status_code=307)
    return RedirectResponse("/", status_code=307)  # Fallback to root

# Health check
@app.get("/_health")
async def health():
    return {"status": "ok", "languages": {"en": "/", "es": "/es"}}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
