import os
import sys

# Ensure backend directory is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

# Reconfigure stdout for logging UTF-8 compatibility
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import uvicorn
import gradio as gr
from backend.main import app

# Create a Gradio interface so HF Space initializes cleanly on free tier
with gr.Blocks(title="LPR-Vision Lab") as demo:
    gr.Markdown("# 🚗 LPR-Vision: License Plate Recognition Lab")
    gr.HTML('''
    <div style="width: 100%; height: 850px; overflow: hidden; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.3);">
        <iframe src="/index.html" style="width: 100%; height: 100%; border: none;"></iframe>
    </div>
    ''')

# Mount Gradio demo onto FastAPI app at /demo
app = gr.mount_gradio_app(app, demo, path="/demo")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
