from fastapi import FastAPI
import uvicorn
import os
from gradio import mount_gradio_app  # Standard import (no sub-module)

app = FastAPI()

# Import language versions
from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es

# Mount: No trailing slashes to avoid // redirects
app = mount_gradio_app(app, demo_en, path="/")     # English at root
app = mount_gradio_app(app, demo_es, path="/es")   # Spanish subpath

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
