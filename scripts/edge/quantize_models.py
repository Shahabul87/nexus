"""
Edge AI Model Quantization Script

Quantizes HAI-DEF models (MedSigLIP, HeAR) to INT8 for mobile deployment.
Part of NEXUS Edge AI implementation for Week 3 deliverables.

Target: Low-end Android devices (2GB RAM, ARM Cortex-A53)
Quantization: Dynamic INT8 (PyTorch) + Post-training quantization (TFLite)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
import json
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# Use qnnpack backend on macOS/ARM (fbgemm not available)
if not torch.backends.quantized.engine == "fbgemm":
    torch.backends.quantized.engine = "qnnpack"

try:
    from transformers import AutoProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class MedSigLIPQuantizer:
    """
    Quantizes MedSigLIP vision encoder to INT8 for edge deployment.

    Supported export formats:
    - PyTorch INT8 (dynamic quantization)
    - ONNX (for TFLite conversion)
    - TorchScript (for mobile)
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str = "cpu",
    ):
        """
        Initialize the quantizer.

        Args:
            model_name: HuggingFace model name
            device: Device for quantization (must be CPU for INT8)
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required: pip install transformers")

        self.model_name = model_name
        self.device = device

        print(f"Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # Get model info
        self.original_size = self._get_model_size(self.model)
        print(f"Original model size: {self.original_size / 1e6:.1f} MB")

    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size

    def quantize_dynamic_int8(self, output_path: Optional[Path] = None) -> nn.Module:
        """
        Apply dynamic INT8 quantization to the model.

        Dynamic quantization quantizes weights to INT8 and activations
        are quantized dynamically during inference.

        Args:
            output_path: Path to save quantized model

        Returns:
            Quantized PyTorch model
        """
        print("\nApplying dynamic INT8 quantization...")

        # Quantize linear layers to INT8
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8,
        )

        # Get quantized size
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = self.original_size / quantized_size

        print(f"Quantized model size: {quantized_size / 1e6:.1f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(quantized_model.state_dict(), output_path)
            print(f"Saved quantized model to {output_path}")

        return quantized_model

    def export_vision_encoder_onnx(
        self,
        output_path: Path,
        opset_version: int = 14,
        dynamic_batch: bool = True,
    ) -> str:
        """
        Export vision encoder to ONNX format for TFLite conversion.

        Args:
            output_path: Path for ONNX model
            opset_version: ONNX opset version
            dynamic_batch: Enable dynamic batch size

        Returns:
            Path to exported ONNX model
        """
        print(f"\nExporting vision encoder to ONNX...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create wrapper for vision encoder only
        class VisionEncoderWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.vision_model = model.vision_model

            def forward(self, pixel_values):
                outputs = self.vision_model(pixel_values=pixel_values)
                return outputs.pooler_output

        wrapper = VisionEncoderWrapper(self.model)
        wrapper.eval()

        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)

        # Dynamic axes for variable batch size
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "pixel_values": {0: "batch_size"},
                "embeddings": {0: "batch_size"},
            }

        # Export to ONNX
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["pixel_values"],
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        print(f"Exported ONNX model to {output_path}")

        # Validate ONNX model
        if HAS_ONNX:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("ONNX model validation passed")

        return str(output_path)

    def export_torchscript(
        self,
        output_path: Path,
        optimize_for_mobile: bool = True,
    ) -> str:
        """
        Export model to TorchScript for mobile deployment.

        Args:
            output_path: Path for TorchScript model
            optimize_for_mobile: Apply mobile optimizations

        Returns:
            Path to exported TorchScript model
        """
        print(f"\nExporting to TorchScript...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create wrapper for vision encoder
        class VisionEncoderForMobile(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.vision_model = model.vision_model

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                outputs = self.vision_model(pixel_values=pixel_values)
                return outputs.pooler_output

        wrapper = VisionEncoderForMobile(self.model)
        wrapper.eval()

        # Trace the model
        dummy_input = torch.randn(1, 3, 224, 224)
        traced_model = torch.jit.trace(wrapper, dummy_input)

        # Optimize for mobile if requested
        if optimize_for_mobile:
            from torch.utils.mobile_optimizer import optimize_for_mobile as mobile_opt
            traced_model = mobile_opt(traced_model)
            print("Applied mobile optimizations")

        # Save TorchScript model
        traced_model.save(str(output_path))

        file_size = output_path.stat().st_size / 1e6
        print(f"Saved TorchScript model to {output_path} ({file_size:.1f} MB)")

        return str(output_path)

    def benchmark(
        self,
        quantized_model: Optional[nn.Module] = None,
        num_iterations: int = 100,
    ) -> Dict:
        """
        Benchmark original vs quantized model performance.

        Args:
            quantized_model: Quantized model to benchmark
            num_iterations: Number of inference iterations

        Returns:
            Benchmark results dictionary
        """
        print(f"\nBenchmarking models ({num_iterations} iterations)...")

        dummy_input = torch.randn(1, 3, 224, 224)

        # Benchmark original model
        self.model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = self.model.vision_model(pixel_values=dummy_input)

            # Timed runs
            start = time.time()
            for _ in range(num_iterations):
                _ = self.model.vision_model(pixel_values=dummy_input)
            original_time = (time.time() - start) / num_iterations * 1000  # ms

        results = {
            "original_latency_ms": round(original_time, 2),
            "original_size_mb": round(self.original_size / 1e6, 1),
        }

        # Benchmark quantized model if provided
        if quantized_model is not None:
            quantized_model.eval()
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = quantized_model.vision_model(pixel_values=dummy_input)

                # Timed runs
                start = time.time()
                for _ in range(num_iterations):
                    _ = quantized_model.vision_model(pixel_values=dummy_input)
                quantized_time = (time.time() - start) / num_iterations * 1000

            quantized_size = self._get_model_size(quantized_model)

            results.update({
                "quantized_latency_ms": round(quantized_time, 2),
                "quantized_size_mb": round(quantized_size / 1e6, 1),
                "speedup": round(original_time / quantized_time, 2),
                "compression": round(self.original_size / quantized_size, 2),
            })

        print("\nBenchmark Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        return results


class AcousticFeatureQuantizer:
    """
    Creates a lightweight acoustic feature extraction model for HeAR fallback.

    Since HeAR requires specific integration, we create a quantized
    acoustic feature extractor that can be used offline.
    """

    def __init__(self):
        """Initialize the acoustic feature model."""
        self.model = self._create_acoustic_model()
        self.original_size = self._get_model_size(self.model)
        print(f"Acoustic model size: {self.original_size / 1e6:.3f} MB")

    def _create_acoustic_model(self) -> nn.Module:
        """
        Create a lightweight CNN for acoustic feature extraction.

        Architecture: Simple 1D CNN that extracts features from mel spectrograms.
        """
        class AcousticFeatureNet(nn.Module):
            def __init__(self, input_dim: int = 128, output_dim: int = 64):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    # Block 1
                    nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    # Block 2
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    # Block 3
                    nn.Conv1d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )

                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, output_dim),
                    nn.ReLU(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (batch, n_mels, time_steps)
                x = self.conv_layers(x)
                x = self.fc(x)
                return x

        return AcousticFeatureNet()

    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size

    def quantize_int8(self, output_path: Optional[Path] = None) -> nn.Module:
        """Apply INT8 quantization to the acoustic model."""
        print("\nQuantizing acoustic feature model...")

        quantized = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv1d},
            dtype=torch.qint8,
        )

        quantized_size = self._get_model_size(quantized)
        print(f"Quantized size: {quantized_size / 1e6:.3f} MB")
        print(f"Compression: {self.original_size / quantized_size:.2f}x")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(quantized.state_dict(), output_path)
            print(f"Saved to {output_path}")

        return quantized

    def export_onnx(self, output_path: Path) -> str:
        """Export acoustic model to ONNX."""
        print(f"\nExporting acoustic model to ONNX...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        # Input: mel spectrogram (batch, n_mels=128, time_steps=100)
        dummy_input = torch.randn(1, 128, 100)

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=14,
            input_names=["mel_spectrogram"],
            output_names=["features"],
            dynamic_axes={
                "mel_spectrogram": {0: "batch", 2: "time"},
                "features": {0: "batch"},
            },
        )

        print(f"Exported to {output_path}")
        return str(output_path)


def main():
    """Main quantization script."""
    parser = argparse.ArgumentParser(description="Quantize HAI-DEF models for Edge AI")
    parser.add_argument(
        "--model",
        choices=["medsiglip", "acoustic", "all"],
        default="all",
        help="Model to quantize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/edge",
        help="Output directory for quantized models",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after quantization",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export ONNX models for TFLite conversion",
    )
    parser.add_argument(
        "--export-torchscript",
        action="store_true",
        help="Export TorchScript models for mobile",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Quantize MedSigLIP
    if args.model in ["medsiglip", "all"]:
        print("\n" + "=" * 60)
        print("Quantizing MedSigLIP Vision Encoder")
        print("=" * 60)

        quantizer = MedSigLIPQuantizer()

        # Dynamic INT8 quantization
        quantized = quantizer.quantize_dynamic_int8(
            output_path=output_dir / "medsiglip_int8.pt"
        )

        # Export ONNX
        if args.export_onnx:
            quantizer.export_vision_encoder_onnx(
                output_path=output_dir / "medsiglip_vision.onnx"
            )

        # Export TorchScript
        if args.export_torchscript:
            quantizer.export_torchscript(
                output_path=output_dir / "medsiglip_vision.ptl"
            )

        # Benchmark
        if args.benchmark:
            results["medsiglip"] = quantizer.benchmark(quantized)

    # Quantize Acoustic model (HeAR fallback)
    if args.model in ["acoustic", "all"]:
        print("\n" + "=" * 60)
        print("Quantizing Acoustic Feature Model (HeAR fallback)")
        print("=" * 60)

        acoustic_quantizer = AcousticFeatureQuantizer()

        # INT8 quantization
        acoustic_quantizer.quantize_int8(
            output_path=output_dir / "acoustic_int8.pt"
        )

        # Export ONNX
        if args.export_onnx:
            acoustic_quantizer.export_onnx(
                output_path=output_dir / "acoustic_features.onnx"
            )

    # Save results
    if results:
        results_path = output_dir / "quantization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print("Quantization Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.iterdir():
        size = f.stat().st_size / 1e6
        print(f"  - {f.name} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
