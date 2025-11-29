from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
import os
from gradio import mount_gradio_app

app = FastAPI()

from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es

app = mount_gradio_app(app, demo_en, path="/")
app = mount_gradio_app(app, demo_es, path="/es")

# This one line fixes the /es.html 404 forever
@app.get("/es.html")
async def fix_es_share_link():
    return RedirectResponse("/es")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
