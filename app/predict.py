import torch
import torchvision.transforms as transforms  # type: ignore
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
import logging
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

model_path = "./models/deepfake-detector-v2"

def load_model(model_path: str) -> tuple[ViTForImageClassification, ViTImageProcessor, torch.device]:
    """
    Download, load and prepare Vision Transformer model for inference.

    Args:
        model_path: Directory path where model will be saved/loaded

    Returns:
        Tuple containing the model, image processor, and device used
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    logger.info(f"Downloading model: {model_name}")

    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    processor = ViTImageProcessor.from_pretrained(model_name, torch_dtype=torch.float16)

    os.makedirs(model_path, exist_ok=True)

    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)

    logger.info(f"Model downloaded and saved to {model_path}")

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model size: {param_count:,} parameters")

    model.to(device)
    if device.type == "cuda":
        model.half()
        logger.info("Model converted to half precision")
    model.eval()

    logger.info(f"Model loaded on {device}")

    return model, processor, device
 
def preprocess(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference.

    Args:
        image: PIL Image object
        device: PyTorch device (CPU or CUDA) for tensor placement

    Returns:
        Preprocessed image tensor ready for model input
    """
    logger.debug(f"Preprocessing image of size {image.size}")

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
        logger.debug("Converted tensor to half precision")

    logger.debug(f"Preprocessed tensor shape: {model_input.shape}")
    return model_input

def predict(model_input: torch.Tensor, model: ViTForImageClassification) -> Dict[str, Any]:
    """
    Perform deepfake detection inference on preprocessed image tensor.

    Args:
        model_input: Preprocessed image tensor
        model: Vision Transformer model for classification

    Returns:
        Dictionary containing prediction results with id, label, score, error, and timestamp
    """
    prediction_id = str(uuid4())
    logger.debug(f"Starting prediction {prediction_id}")

    try:
        with torch.no_grad():
            outputs = model(model_input)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_idx: int = int(torch.argmax(logits, dim=1).item())
            confidence: float = float(probs[0][predicted_class_idx].item())

        id2label = model.config.id2label
        if id2label is None:
            logger.error("Model config missing id2label mapping")
            raise ValueError("Model config missing id2label mapping")

        label = id2label[predicted_class_idx]
        swapped_label = 'Deepfake' if label == 'Realism' else 'Realism'

        logger.info(f"Prediction {prediction_id}: original_label={label}, swapped_label={swapped_label}, confidence={confidence:.4f}")

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

