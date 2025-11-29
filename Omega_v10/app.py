# app.py — single entry point for Render
from fastapi import FastAPI
from gradio import mount_gradio_app

app = FastAPI()

# Import all language versions
from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es
# from omega_v10_pt import demo as demo_pt   # uncomment when ready
# from omega_v10_fr import demo as demo_fr   # uncomment when ready

# Mount them
app = mount_gradio_app(app, demo_en, path="/")
app = mount_gradio_app(app, demo_es, path="/es")
# app = mount_gradio_app(app, demo_pt, path="/pt")
# app = mount_gradio_app(app, demo_fr, path="/fr")
