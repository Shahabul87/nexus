"""
Text Embedding Export Script

Exports pre-computed text embeddings for zero-shot classification on mobile.
These embeddings are used with the TFLite vision model for offline inference.

HAI-DEF Model: MedSigLIP text encoder
Output: Binary files containing Float32 embeddings (768-dim)
"""

import argparse
import struct
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

try:
    from transformers import AutoProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# Medical prompts for zero-shot classification (same as detector modules)
PROMPTS = {
    "anemia_positive": [
        "anemic pale conjunctiva indicating low hemoglobin",
        "pale conjunctiva with signs of anemia",
        "conjunctival pallor consistent with iron deficiency anemia",
    ],
    "anemia_negative": [
        "healthy pink conjunctiva with normal blood supply",
        "normal conjunctiva with adequate hemoglobin levels",
        "well-perfused pink inner eyelid without pallor",
    ],
    "jaundice_positive": [
        "neonatal jaundice with yellow skin discoloration",
        "yellow sclera indicating elevated bilirubin",
        "newborn with jaundiced appearance and icterus",
    ],
    "jaundice_negative": [
        "healthy newborn skin with normal pigmentation",
        "normal infant without signs of jaundice",
        "pink newborn skin with no yellow discoloration",
    ],
}


def export_embeddings(
    model_name: str = "google/siglip-base-patch16-224",
    output_dir: Path = Path("models/tflite/embeddings"),
) -> Dict[str, np.ndarray]:
    """
    Export text embeddings for all prompt categories.

    Args:
        model_name: HuggingFace model name
        output_dir: Output directory for embedding files

    Returns:
        Dictionary of embedding arrays
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required: pip install transformers")

    print(f"Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = {}

    for category, prompts in PROMPTS.items():
        print(f"\nProcessing {category}...")

        with torch.no_grad():
            # Tokenize prompts
            inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )

            # Get text embeddings
            if hasattr(model, 'get_text_features'):
                text_embeddings = model.get_text_features(**inputs)
            else:
                text_outputs = model.text_model(**inputs)
                text_embeddings = text_outputs.pooler_output

            # Normalize
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

            # Average across prompts
            avg_embedding = text_embeddings.mean(dim=0)
            avg_embedding = avg_embedding / avg_embedding.norm()

            # Convert to numpy
            embedding_array = avg_embedding.numpy().astype(np.float32)
            embeddings[category] = embedding_array

            # Save as binary file
            output_path = output_dir / f"{category}.bin"
            save_embedding_binary(embedding_array, output_path)

            print(f"  Shape: {embedding_array.shape}")
            print(f"  Saved to: {output_path}")

    # Also save as combined JSON for debugging
    metadata = {
        "model": model_name,
        "embedding_dim": embeddings[list(embeddings.keys())[0]].shape[0],
        "categories": list(embeddings.keys()),
    }

    import json
    metadata_path = output_dir / "embeddings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {metadata_path}")

    return embeddings


def save_embedding_binary(embedding: np.ndarray, path: Path) -> None:
    """
    Save embedding as binary file (Float32).

    Format: Raw float32 values (4 bytes per value)
    """
    with open(path, "wb") as f:
        embedding.tofile(f)


def load_embedding_binary(path: Path) -> np.ndarray:
    """Load embedding from binary file."""
    return np.fromfile(path, dtype=np.float32)


def verify_embeddings(embedding_dir: Path) -> None:
    """Verify exported embeddings are correct."""
    print("\nVerifying embeddings...")

    for category in PROMPTS.keys():
        path = embedding_dir / f"{category}.bin"
        if path.exists():
            embedding = load_embedding_binary(path)
            print(f"  {category}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
        else:
            print(f"  {category}: NOT FOUND")


def main():
    """Main export script."""
    parser = argparse.ArgumentParser(description="Export text embeddings for Edge AI")
    parser.add_argument(
        "--model",
        type=str,
        default="google/siglip-base-patch16-224",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/tflite/embeddings",
        help="Output directory",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported embeddings",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Export embeddings
    embeddings = export_embeddings(args.model, output_dir)

    # Verify if requested
    if args.verify:
        verify_embeddings(output_dir)

    print("\n" + "=" * 60)
    print("Embedding Export Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.iterdir():
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
