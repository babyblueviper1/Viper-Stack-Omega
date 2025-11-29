from fastapi import FastAPI
import uvicorn
import os
from gradio import mount_gradio_app  # Standard import for 6.0.1

app = FastAPI()

# Import language versions
from Omega_v10_en import demo as demo_en
from Omega_v10_es import demo as demo_es

# Mount English at root (empty root_path)
app = mount_gradio_app(
    app, 
    demo_en, 
    path="/",
    app_kwargs={"root_path": ""}  # Empty for root (prevents //)
)

# Mount Spanish at subpath
app = mount_gradio_app(
    app, 
    demo_es, 
    path="/es",
    app_kwargs={"root_path": "/es"}  # Matches path exactly (fixes frontend fetches)
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
