"""
Upload the model to Modal volume.
Run this once to store your model in Modal's persistent storage.
"""
import modal
from pathlib import Path

app = modal.App("upload-deepfake-model")

# Create persistent volume for model
model_volume = modal.Volume.from_name("deepfake-model", create_if_missing=True)
MODEL_DIR = "/model"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "transformers==4.57.3",
    "torch==2.9.1",
    "Pillow==11.0.0",
)


@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=3600,
)
def upload_model():
    """Download and save model to Modal volume"""
    from transformers import ViTForImageClassification, ViTImageProcessor
    from pathlib import Path

    model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    model_path = Path(f"{MODEL_DIR}/deepfake-detector-v2")

    print(f"Downloading model: {model_name}")
    print(f"Saving to: {model_path}")

    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=None  # Download in full precision, will convert later
    )
    processor = ViTImageProcessor.from_pretrained(model_name)

    model_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(model_path))
    processor.save_pretrained(str(model_path))

    # Commit the volume to persist changes
    model_volume.commit()

    print(f"✓ Model saved to Modal volume at {model_path}")
    print(f"✓ Model size: {sum(p.numel() for p in model.parameters()):,} parameters")

    return {"status": "success", "model_path": str(model_path)}


@app.local_entrypoint()
def main():
    """Upload model to Modal"""
    print("\n" + "="*60)
    print("Uploading Deepfake Detection Model to Modal")
    print("="*60)
    result = upload_model.remote()
    print(f"\nUpload result: {result}")
