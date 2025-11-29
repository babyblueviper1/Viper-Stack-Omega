# app.py
from fastapi import FastAPI
import uvicorn
import os
from gradio import mount_gradio_app

app = FastAPI()

# Import language versions
from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es
# from omega_v10_pt import demo as demo_pt
# from omega_v10_fr import demo as demo_fr

# Mount them
app = mount_gradio_app(app, demo_en, path="/")
app = mount_gradio_app(app, demo_es, path="/es/")
# app = mount_gradio_app(app, demo_pt, path="/pt")
# app = mount_gradio_app(app, demo_fr, path="/fr")

# This is the crucial part Render needs
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses 10000 internally
    uvicorn.run(app, host="0.0.0.0", port=port)
