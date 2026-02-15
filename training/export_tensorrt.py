#!/usr/bin/env python3
"""
Convert trained GestureNet from PyTorch → ONNX → TensorRT FP16 engine.

Usage::

    # Full pipeline: .pth → .onnx → .engine
    python -m training.export_tensorrt

    # From existing ONNX file
    python -m training.export_tensorrt --onnx models/weights/gesture_net.onnx

    # Custom paths
    python -m training.export_tensorrt \\
        --checkpoint models/weights/gesture_net.pth \\
        --output models/weights/gesture_net.engine

The TensorRT engine is platform-specific (built for the exact GPU).
Build on Jetson Nano for Jetson Nano deployment.

Requirements:
    - tensorrt >= 7.0 (JetPack 4.x)
    - pycuda
    - torch (only for .pth → .onnx step)

Compatible with Python 3.6+.
"""

import os
import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def convert_pth_to_onnx(pth_path, onnx_path):
    """Convert PyTorch checkpoint to ONNX.

    Args:
        pth_path: Path to .pth checkpoint
        onnx_path: Output .onnx path

    Returns:
        onnx_path
    """
    from models.gesture_net import GestureNet
    model = GestureNet.load_checkpoint(pth_path, device="cpu")
    model.export_onnx(onnx_path)
    return onnx_path


def validate_onnx(onnx_path):
    """Validate ONNX model structure.

    Returns:
        True if valid
    """
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info("ONNX model validated: %s", onnx_path)

        # Print model info
        graph = model.graph
        for inp in graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            logger.info("  Input: %s shape=%s", inp.name, dims)
        for out in graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            logger.info("  Output: %s shape=%s", out.name, dims)

        return True
    except ImportError:
        logger.warning("onnx package not installed — skipping validation")
        return True
    except Exception as e:
        logger.error("ONNX validation failed: %s", e)
        return False


def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16=True,
                              max_workspace_mb=256):
    """Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to .onnx file
        engine_path: Output .engine path
        fp16: Enable FP16 precision
        max_workspace_mb: GPU workspace size

    Returns:
        engine_path
    """
    from models.tensorrt_engine import TensorRTEngine

    logger.info("Converting ONNX → TensorRT (FP16=%s)...", fp16)
    result = TensorRTEngine.build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fp16=fp16,
        max_workspace_mb=max_workspace_mb,
    )

    # Verify the engine loads correctly
    logger.info("Verifying engine...")
    engine = TensorRTEngine(result)
    logger.info("Engine output classes: %d", engine.num_classes)

    # Quick inference test with random data
    import numpy as np
    dummy = np.random.randn(81).astype(np.float32)
    probs = engine.predict_proba(dummy)
    logger.info("Test inference OK: probabilities sum=%.4f (expected ~1.0)", probs.sum())

    engine.destroy()

    return result


def benchmark_engine(engine_path, num_iterations=1000):
    """Benchmark TensorRT inference speed.

    Args:
        engine_path: Path to .engine file
        num_iterations: Number of inference iterations

    Returns:
        dict with timing stats
    """
    import time
    import numpy as np
    from models.tensorrt_engine import TensorRTEngine

    engine = TensorRTEngine(engine_path)
    dummy = np.random.randn(81).astype(np.float32)

    # Warmup
    for _ in range(50):
        engine.predict(dummy)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        engine.predict(dummy)
        times.append((time.time() - start) * 1000)  # ms

    times = np.array(times)
    stats = {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "iterations": num_iterations,
    }

    logger.info("TensorRT Benchmark (%d iterations):", num_iterations)
    logger.info("  Mean:   %.3f ms", stats["mean_ms"])
    logger.info("  Median: %.3f ms", stats["median_ms"])
    logger.info("  Min:    %.3f ms", stats["min_ms"])
    logger.info("  Max:    %.3f ms", stats["max_ms"])
    logger.info("  P99:    %.3f ms", stats["p99_ms"])

    engine.destroy()
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert GestureNet to TensorRT engine"
    )
    parser.add_argument("--checkpoint", default="models/weights/gesture_net.pth",
                        help="PyTorch checkpoint path")
    parser.add_argument("--onnx", default=None,
                        help="Existing ONNX model path (skip PyTorch → ONNX)")
    parser.add_argument("--output", default="models/weights/gesture_net.engine",
                        help="Output TensorRT engine path")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 (use FP32)")
    parser.add_argument("--workspace-mb", type=int, default=256,
                        help="TensorRT workspace size in MB")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark after conversion")
    parser.add_argument("--benchmark-iters", type=int, default=1000,
                        help="Number of benchmark iterations")
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1: PyTorch → ONNX (if not provided)
    if args.onnx and os.path.isfile(args.onnx):
        onnx_path = args.onnx
        logger.info("Using existing ONNX model: %s", onnx_path)
    else:
        if not os.path.isfile(args.checkpoint):
            logger.error("Checkpoint not found: %s", args.checkpoint)
            logger.error("Train first: python -m training.train")
            sys.exit(1)

        onnx_path = args.onnx or args.checkpoint.replace(".pth", ".onnx")
        logger.info("Step 1: Converting PyTorch → ONNX...")
        convert_pth_to_onnx(args.checkpoint, onnx_path)

    # Step 2: Validate ONNX
    logger.info("Step 2: Validating ONNX model...")
    if not validate_onnx(onnx_path):
        sys.exit(1)

    # Step 3: ONNX → TensorRT
    logger.info("Step 3: Converting ONNX → TensorRT...")
    try:
        engine_path = convert_onnx_to_tensorrt(
            onnx_path=onnx_path,
            engine_path=args.output,
            fp16=not args.no_fp16,
            max_workspace_mb=args.workspace_mb,
        )
        logger.info("TensorRT engine saved: %s", engine_path)
    except RuntimeError as e:
        logger.error("TensorRT conversion failed: %s", e)
        logger.error("This must be run on the Jetson Nano (TensorRT is platform-specific)")
        sys.exit(1)

    # Step 4: Optional benchmark
    if args.benchmark:
        logger.info("Step 4: Running benchmark...")
        benchmark_engine(engine_path, args.benchmark_iters)

    logger.info("=" * 60)
    logger.info("DONE! TensorRT engine ready at: %s", args.output)
    logger.info("The system will auto-detect it on next startup.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
