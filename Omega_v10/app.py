from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
import os
from gradio import mount_gradio_app

app = FastAPI()

# Import your demos
from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es

# Mount English at root
app = mount_gradio_app(
    app,
    demo_en,
    path="/",
    app_kwargs={"root_path": ""}  # Gradio 6 needs this explicit for root on proxies
)

# Mount Spanish at /es
app = mount_gradio_app(
    app,
    demo_es,
    path="/es",
    app_kwargs={"root_path": "/es"}  # No trailing slash; critical for 6.x asset loading
)

# Fix Gradio 6's .html share links (still generates them, but 404s on subpaths)
@app.get("/es.html")
async def fix_es_html():
    return RedirectResponse("/es", status_code=307)  # Use 307 for query param preservation

# Quick health check (optional, but useful for Render logs)
@app.get("/_health")
async def health():
    return {"status": "ok", "gradio_version": "6.0", "languages": ["en", "es"]}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
