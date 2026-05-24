"""Modal deployment of the OSS assistant on a T4 GPU.

Usage:
    pip install modal
    modal token new
    modal deploy deployment/modal_app.py

The endpoint exposes a Gradio UI at the URL Modal prints on deploy.
"""
from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .env({"HF_HOME": "/cache/hf"})
)

app = modal.App("ollive-oss-assistant")
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/cache/hf": volume},
    # Keep the model warm for 10 minutes after the last request
    container_idle_timeout=600,
)
@modal.asgi_app()
def gradio_ui():
    import gradio as gr
    from app.gradio_app import build_ui

    demo = build_ui()
    # Modal serves the underlying FastAPI app
    return gr.routes.App.create_app(demo)
