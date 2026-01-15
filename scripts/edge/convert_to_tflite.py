"""
TensorFlow Lite Conversion Script

Converts ONNX models to TFLite format for mobile deployment.
Part of NEXUS Edge AI implementation for Week 3 deliverables.

Target: React Native + TensorFlow Lite for on-device inference
Quantization: INT8 post-training quantization for 4x size reduction
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")

try:
    import onnx
    from onnx_tf.backend import prepare
    HAS_ONNX_TF = True
except ImportError:
    HAS_ONNX_TF = False
    print("Warning: onnx-tf not installed. Install with: pip install onnx-tf")


class TFLiteConverter:
    """
    Converts models to TensorFlow Lite format with INT8 quantization.

    Supports:
    - ONNX to TFLite conversion
    - Post-training INT8 quantization
    - Model validation and benchmarking
    """

    def __init__(self, model_path: Path):
        """
        Initialize the converter.

        Args:
            model_path: Path to ONNX model file
        """
        if not HAS_TF:
            raise ImportError("TensorFlow required: pip install tensorflow")

        self.model_path = Path(model_path)
        self.saved_model_dir: Optional[Path] = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"TFLite Converter initialized with {model_path}")

    def onnx_to_saved_model(self, output_dir: Path) -> Path:
        """
        Convert ONNX model to TensorFlow SavedModel format.

        Args:
            output_dir: Directory for SavedModel

        Returns:
            Path to SavedModel directory
        """
        if not HAS_ONNX_TF:
            raise ImportError("onnx-tf required: pip install onnx-tf")

        print(f"\nConverting ONNX to SavedModel...")

        output_dir = Path(output_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        # Load ONNX model
        onnx_model = onnx.load(str(self.model_path))

        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)

        # Export as SavedModel
        tf_rep.export_graph(str(output_dir))

        self.saved_model_dir = output_dir
        print(f"SavedModel exported to {output_dir}")

        return output_dir

    def convert_to_tflite(
        self,
        output_path: Path,
        quantize: bool = True,
        representative_data: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Convert SavedModel to TFLite format.

        Args:
            output_path: Path for TFLite model
            quantize: Apply INT8 quantization
            representative_data: Calibration data for quantization

        Returns:
            Path to TFLite model
        """
        if self.saved_model_dir is None:
            raise ValueError("Run onnx_to_saved_model() first")

        print(f"\nConverting to TFLite (quantize={quantize})...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(str(self.saved_model_dir))

        if quantize:
            # Enable INT8 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if representative_data is not None:
                # Full integer quantization with calibration data
                def representative_dataset():
                    for sample in representative_data:
                        yield [sample.astype(np.float32)]

                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                print("Applied full INT8 quantization with calibration")
            else:
                # Dynamic range quantization
                print("Applied dynamic range quantization")

        # Convert
        tflite_model = converter.convert()

        # Save
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_mb = output_path.stat().st_size / 1e6
        print(f"TFLite model saved to {output_path} ({size_mb:.1f} MB)")

        return output_path

    def validate_tflite(self, tflite_path: Path, input_shape: Tuple) -> Dict:
        """
        Validate TFLite model with test inference.

        Args:
            tflite_path: Path to TFLite model
            input_shape: Input tensor shape

        Returns:
            Validation results dictionary
        """
        print(f"\nValidating TFLite model...")

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("Input details:", input_details[0])
        print("Output details:", output_details[0])

        # Run test inference
        test_input = np.random.randn(*input_shape).astype(
            input_details[0]["dtype"]
        )
        interpreter.set_tensor(input_details[0]["index"], test_input)

        # Time inference
        import time
        start = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start) * 1000

        output = interpreter.get_tensor(output_details[0]["index"])

        results = {
            "input_shape": list(input_details[0]["shape"]),
            "output_shape": list(output_details[0]["shape"]),
            "input_dtype": str(input_details[0]["dtype"]),
            "output_dtype": str(output_details[0]["dtype"]),
            "inference_time_ms": round(inference_time, 2),
            "output_sample": output.flatten()[:5].tolist(),
        }

        print("Validation passed!")
        return results


def convert_medsiglip_vision(
    onnx_path: Path,
    output_dir: Path,
    quantize: bool = True,
) -> Dict:
    """
    Convert MedSigLIP vision encoder to TFLite.

    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory
        quantize: Apply INT8 quantization

    Returns:
        Conversion results
    """
    print("\n" + "=" * 60)
    print("Converting MedSigLIP Vision Encoder to TFLite")
    print("=" * 60)

    converter = TFLiteConverter(onnx_path)

    # Convert to SavedModel
    saved_model_dir = converter.onnx_to_saved_model(output_dir / "medsiglip_savedmodel")

    # Generate representative data for calibration
    print("\nGenerating calibration data...")
    calibration_data = np.random.randn(100, 1, 3, 224, 224).astype(np.float32)

    # Convert to TFLite
    tflite_path = output_dir / "medsiglip_vision.tflite"
    converter.convert_to_tflite(
        tflite_path,
        quantize=quantize,
        representative_data=calibration_data if quantize else None,
    )

    # Validate
    results = converter.validate_tflite(tflite_path, (1, 3, 224, 224))

    results["model"] = "medsiglip_vision"
    results["tflite_path"] = str(tflite_path)
    results["tflite_size_mb"] = round(tflite_path.stat().st_size / 1e6, 1)

    return results


def convert_acoustic_features(
    onnx_path: Path,
    output_dir: Path,
    quantize: bool = True,
) -> Dict:
    """
    Convert acoustic feature model to TFLite.

    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory
        quantize: Apply INT8 quantization

    Returns:
        Conversion results
    """
    print("\n" + "=" * 60)
    print("Converting Acoustic Feature Model to TFLite")
    print("=" * 60)

    converter = TFLiteConverter(onnx_path)

    # Convert to SavedModel
    saved_model_dir = converter.onnx_to_saved_model(output_dir / "acoustic_savedmodel")

    # Generate representative data
    print("\nGenerating calibration data...")
    calibration_data = np.random.randn(100, 1, 128, 100).astype(np.float32)

    # Convert to TFLite
    tflite_path = output_dir / "acoustic_features.tflite"
    converter.convert_to_tflite(
        tflite_path,
        quantize=quantize,
        representative_data=calibration_data if quantize else None,
    )

    # Validate
    results = converter.validate_tflite(tflite_path, (1, 128, 100))

    results["model"] = "acoustic_features"
    results["tflite_path"] = str(tflite_path)
    results["tflite_size_mb"] = round(tflite_path.stat().st_size / 1e6, 1)

    return results


def create_model_metadata(output_dir: Path, results: Dict) -> Path:
    """
    Create metadata file for TFLite models.

    This metadata is used by the mobile app to load and configure models.
    """
    metadata = {
        "version": "1.0.0",
        "framework": "tensorflow_lite",
        "quantization": "int8",
        "models": {
            "medsiglip_vision": {
                "file": "medsiglip_vision.tflite",
                "input_shape": [1, 3, 224, 224],
                "output_shape": [1, 768],
                "input_type": "float32",
                "output_type": "float32",
                "description": "MedSigLIP vision encoder for anemia/jaundice detection",
            },
            "acoustic_features": {
                "file": "acoustic_features.tflite",
                "input_shape": [1, 128, 100],
                "output_shape": [1, 64],
                "input_type": "float32",
                "output_type": "float32",
                "description": "Acoustic feature extractor for cry analysis",
            },
        },
        "text_embeddings": {
            "anemia_positive": "embeddings/anemia_positive.bin",
            "anemia_negative": "embeddings/anemia_negative.bin",
            "jaundice_positive": "embeddings/jaundice_positive.bin",
            "jaundice_negative": "embeddings/jaundice_negative.bin",
        },
        "classifiers": {
            "cry_classifier": "classifiers/cry_classifier.bin",
        },
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel metadata saved to {metadata_path}")
    return metadata_path


def main():
    """Main TFLite conversion script."""
    parser = argparse.ArgumentParser(description="Convert models to TFLite")
    parser.add_argument(
        "--model",
        choices=["medsiglip", "acoustic", "all"],
        default="all",
        help="Model to convert",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="models/edge",
        help="Directory with ONNX models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/tflite",
        help="Output directory for TFLite models",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable INT8 quantization",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantize = not args.no_quantize
    results = {}

    # Convert MedSigLIP
    if args.model in ["medsiglip", "all"]:
        onnx_path = input_dir / "medsiglip_vision.onnx"
        if onnx_path.exists():
            results["medsiglip"] = convert_medsiglip_vision(
                onnx_path, output_dir, quantize
            )
        else:
            print(f"Warning: {onnx_path} not found. Run quantize_models.py --export-onnx first.")

    # Convert Acoustic model
    if args.model in ["acoustic", "all"]:
        onnx_path = input_dir / "acoustic_features.onnx"
        if onnx_path.exists():
            results["acoustic"] = convert_acoustic_features(
                onnx_path, output_dir, quantize
            )
        else:
            print(f"Warning: {onnx_path} not found. Run quantize_models.py --export-onnx first.")

    # Create model metadata
    if results:
        create_model_metadata(output_dir, results)

        # Save conversion results
        results_path = output_dir / "conversion_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print("TFLite Conversion Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size / 1e6
            print(f"  - {f.name} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
