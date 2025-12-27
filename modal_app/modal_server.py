import modal
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import io
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from transformers import ViTForImageClassification
import torchvision.transforms as transforms  # type: ignore

# Create Modal app
app = modal.App("deepfake-detection-server")

# Define Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi==0.127.1",
        "transformers==4.57.3",
        "torch==2.9.1",
        "torchvision==0.24.1",
        "Pillow==11.0.0",
        "python-multipart==0.0.21",
    )
)

# Create persistent volume for model
model_volume = modal.Volume.from_name("deepfake-model", create_if_missing=True)
MODEL_DIR = "/model"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
web_app = FastAPI(title="Deepfake Detection API", version="0.0.1")

# Global model and device variables
model: Optional[ViTForImageClassification] = None
device: Optional[torch.device] = None


def load_model_on_startup():
    """Load model once at container startup"""
    global model, device

    from pathlib import Path

    logger.info("Starting up Deepfake Detection API...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_path = Path(f"{MODEL_DIR}/deepfake-detector-v2")
    logger.info(f"Loading model from: {model_path}")

    try:
        model = ViTForImageClassification.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16
        )
        model.to(device)
        if device.type == "cuda":
            model.half()
            logger.info("Model converted to half precision")
        model.eval()

        # Compile model on GPU for better performance
        if device.type == "cuda":
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compilation complete")

        logger.info(f"✓ Model loaded successfully on {device}")
        logger.info(f"✓ Labels: {model.config.id2label}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL Image for model inference"""
    global device

    preprocess_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = preprocess_transform(image)
    model_input = input_tensor.unsqueeze(0).to(device)
    if device.type == "cuda":
        model_input = model_input.half()

    return model_input


def predict_image(model_input: torch.Tensor) -> Dict[str, Any]:
    """Perform deepfake detection inference"""
    global model

    from uuid import uuid4

    prediction_id = str(uuid4())

    try:
        with torch.no_grad():
            outputs = model(model_input)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_idx = int(torch.argmax(logits, dim=1).item())
            confidence = float(probs[0][predicted_class_idx].item())

        id2label = model.config.id2label
        if id2label is None:
            raise ValueError("Model config missing id2label mapping")

        label = id2label[predicted_class_idx]
        swapped_label = 'Deepfake' if label == 'Realism' else 'Realism'

        logger.info(f"Prediction {prediction_id}: label={swapped_label}, confidence={confidence:.4f}")

        return {
            'id': prediction_id,
            'label': swapped_label,
            'score': confidence,
            'error': "",
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction {prediction_id} failed: {e}", exc_info=True)
        return {
            'id': prediction_id,
            'label': 'Error',
            'score': 0.0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@web_app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {"message": "Deepfake Detection API", "status": "running"}


@web_app.get("/health")
async def health():
    """Health check endpoint"""
    is_healthy = model is not None
    return {"status": "healthy" if is_healthy else "unhealthy", "model_loaded": is_healthy}


@web_app.post("/predict")
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

        model_input = preprocess_image(image)

        logger.info("Running inference...")
        result = predict_image(model_input)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Prediction complete: label={result['label']}, score={result['score']:.4f}, time={processing_time:.2f}ms")

        result['processing_time_ms'] = processing_time

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.function(
    image=image,
    gpu="B200",
    volumes={MODEL_DIR: model_volume},
    container_idle_timeout=300,
    timeout=3600,
)
@modal.asgi_app()
def serve():
    """Serve the FastAPI app on Modal"""
    # Load model when container starts
    load_model_on_startup()
    return web_app


@app.local_entrypoint()
def main():
    """Deploy the web server"""
    print("\n" + "="*60)
    print("Deploying Deepfake Detection API to Modal")
    print("="*60)
    print("\nUse 'modal serve modal_server.py' for development")
    print("Use 'modal deploy modal_server.py' for production")
