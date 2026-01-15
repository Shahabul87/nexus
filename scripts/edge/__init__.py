"""
NEXUS Edge AI Scripts

Model quantization and TFLite conversion for mobile deployment.

Usage:
    # Step 1: Quantize models and export to ONNX
    python scripts/edge/quantize_models.py --export-onnx --export-torchscript --benchmark

    # Step 2: Convert ONNX to TFLite
    python scripts/edge/convert_to_tflite.py

    # Output: models/tflite/
    # - medsiglip_vision.tflite (~25MB INT8)
    # - acoustic_features.tflite (~2MB INT8)
    # - model_metadata.json
"""
