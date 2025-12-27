from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import io
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from transformers import ViTForImageClassification

from predict import load_model, predict, preprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Detection API", version="0.0.1")

model: Optional[ViTForImageClassification] = None
device: Optional[torch.device] = None

@app.on_event("startup")
async def setup():
    """Load model once at startup"""
    global model, device

    logger.info("Starting up Deepfake Detection API...")

    model_path = "./models/deepfake-detector-v2"

    try:
        model, _, device = load_model(model_path)
        logger.info(f"✓ Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {"message": "Deepfake Detection API", "status": "running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    is_healthy = model is not None
    logger.debug(f"Health check: model_loaded={is_healthy}")
    return {"status": "healthy" if is_healthy else "unhealthy", "model_loaded": is_healthy}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict if an image is deepfake or real
    """
    logger.info(f"Prediction request received: filename={file.filename}, content_type={file.content_type}")

    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")

    if model is None or device is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = datetime.now()

        image_bytes = await file.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info(f"Image dimensions: {image.size}")

        model_input = preprocess(image, device)

        logger.info("Running inference...")
        result = predict(model_input, model)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Prediction complete: label={result['label']}, score={result['score']:.4f}, time={processing_time:.2f}ms")

        result['processing_time_ms'] = processing_time

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
