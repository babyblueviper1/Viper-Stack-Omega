from fastapi import FastAPI, Request, Response
import uvicorn
import os

app = FastAPI()

# Import your demos
from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es

# Mount with the new 2025 helper that fixes the subpath bug
from gradio.routes import mount_gradio_app   # ← this is the correct import now

app = mount_gradio_app(app, demo_en, path="/")
app = mount_gradio_app(app, demo_es, path="/es")   # ← NO trailing slash on /es

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
