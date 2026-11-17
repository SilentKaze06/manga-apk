import os
import gradio as gr
from main import app as fastapi_app

# Default to unauthenticated for native mobile clients
os.environ.setdefault("MANGA_ALLOW_UNAUTHENTICATED", "true")

# Create a clean Gradio dashboard UI
with gr.Blocks(title="AI Manga Tarjimon Server") as demo:
    gr.Markdown("# ⚡ AI Manga & Manhwa OCR Translation Server")
    gr.Markdown("""
    ### ✅ Server ZeroGPU yordamida muvaffaqiyatli ishlamoqda (24/7)!
    
    Android ilovangizni ulash uchun quyidagi doimiy havolani ishlating:
    
    **Manga OCR Server URL:**
    ```text
    https://dosjannazirbayev-dosjanmanga.hf.space
    ```
    
    ---
    
    #### 🚀 Imkoniyatlar:
    - **Yaponcha (Manga):** `manga-ocr` (Vision Transformer)
    - **Koreyscha (Manhwa):** `PaddleOCR` (Korean)
    - **Xitoycha (Manhua):** `PaddleOCR` (Chinese)
    - **Ingliz, Rus, Fransuz, Nemis va Latin:** `PaddleOCR`
    - **PDF kitob va komikslar:** Ko'p sahifali avtomatik tarjima va eksport
    """)

# Mount Gradio onto the existing FastAPI app
# All FastAPI endpoints (/health, /translate-manga, /ocr-manga, /detect-and-translate, /translate-pdf)
# remain fully functional and unhindered.
@fastapi_app.get("/api/info")
def api_info():
    return {
        "name": "Dosjan Manga & Manhwa OCR Translation Server",
        "status": "running",
        "service": "huggingface-space",
        "endpoints": ["/health", "/translate-manga", "/ocr-manga", "/detect-and-translate", "/translate-pdf", "/pdf-jobs/{job_id}"]
    }

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
