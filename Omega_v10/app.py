from fastapi import FastAPI
import uvicorn
import os
from gradio import mount_gradio_app  # Correct import — from the main gradio module

app = FastAPI()

from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es

# These two lines — now with the fixed import
app = mount_gradio_app(app, demo_en, path="/")
app = mount_gradio_app(app, demo_es, path="/es")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
