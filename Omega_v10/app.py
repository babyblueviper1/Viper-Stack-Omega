from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
import os
from gradio import mount_gradio_app

app = FastAPI()

# Import your demos
from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es

# Mount English at root (no subpath issues here)
app = mount_gradio_app(
    app,
    demo_en,
    path="/",
    root_path=""  # Empty for root; tells Gradio assets are at base /
)

# Mount Spanish at /es (fixes 404s on /es, /es/info, /es/theme.css, etc.)
app = mount_gradio_app(
    app,
    demo_es,
    path="/es",
    root_path="/es"  # Exact subpath; no trailing slash—Gradio 6 enforces strict matching
)

# Handle Gradio's .html share links (redirects preserve query params)
@app.get("/es.html")
async def fix_es_html():
    return RedirectResponse("/es", status_code=307)

# Optional health check (hit /_health to verify both mounts)
@app.get("/_health")
async def health():
    return {"status": "ok", "languages": {"en": "/", "es": "/es"}, "gradio": "6.0"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
