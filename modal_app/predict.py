import torch
import torchvision.transforms as transforms  # type: ignore
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
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


def load_model(model_path: str) -> tuple[ViTForImageClassification, ViTImageProcessor, torch.device]:
    """
    Load and prepare Vision Transformer model for inference.

    Args:
        model_path: Directory path where model is stored

    Returns:
        Tuple containing the model, image processor, and device used
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_path_obj = Path(model_path)
    logger.info(f"Loading model from: {model_path_obj}")

    model = ViTForImageClassification.from_pretrained(
        str(model_path_obj),
        torch_dtype=torch.float16
    )
    processor = ViTImageProcessor.from_pretrained(str(model_path_obj))

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model size: {param_count:,} parameters")

    model.to(device)
    if device.type == "cuda":
        model.half()
        logger.info("Model converted to half precision")
    model.eval()

    if device.type == "cuda":
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Model compilation complete")

    logger.info(f"✓ Model loaded successfully on {device}")
    logger.info(f"✓ Labels: {model.config.id2label}")

    return model, processor, device


def preprocess_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Preprocess a PIL Image for model inference"""
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


def predict_image(model_input: torch.Tensor, model: ViTForImageClassification) -> Dict[str, Any]:
    """Perform deepfake detection inference"""
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
