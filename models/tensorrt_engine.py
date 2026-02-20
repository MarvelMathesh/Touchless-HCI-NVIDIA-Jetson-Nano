"""
TensorRT FP16 inference engine for GestureNet.

Loads a serialized TensorRT engine (.engine / .trt) built from the
ONNX export of GestureNet and runs inference on the Jetson Nano
Maxwell GPU with FP16 precision.

Requirements (pre-installed with JetPack 4.x):
    - tensorrt >= 7.0
    - pycuda

Compatible with Python 3.6+.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# TensorRT + PyCUDA — available on Jetson, optional on dev machines
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (initialises CUDA context)
    TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    TRT_AVAILABLE = False
    logger.info("TensorRT/PyCUDA not available — TensorRTEngine disabled")


class TensorRTEngine:
    """Wraps a serialized TensorRT engine for single-batch inference.

    Typical latency on Jetson Nano with GestureNet FP16: <0.3 ms.
    """

    def __init__(self, engine_path):
        """Load a serialized TensorRT engine.

        Args:
            engine_path: Path to .engine or .trt file

        Raises:
            RuntimeError: If TensorRT is not available or engine fails to load
        """
        if not TRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT and PyCUDA are required. "
                "These are pre-installed on Jetson Nano with JetPack 4.x."
            )

        if not os.path.isfile(engine_path):
            raise FileNotFoundError("TensorRT engine not found: %s" % engine_path)

        self._engine_path = engine_path
        self._engine = None
        self._context = None
        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._stream = None

        self._load_engine()
        logger.info("TensorRT engine loaded: %s", engine_path)

    def _load_engine(self):
        """Deserialize and set up I/O buffers."""
        with open(self._engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        for binding in self._engine:
            shape = self._engine.get_binding_shape(binding)
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            size = abs(int(np.prod(shape)))

            # Allocate host + device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self._bindings.append(int(device_mem))

            if self._engine.binding_is_input(binding):
                self._inputs.append({
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "name": binding,
                })
            else:
                self._outputs.append({
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "name": binding,
                })

        logger.debug("TRT bindings: %d inputs, %d outputs",
                      len(self._inputs), len(self._outputs))

    def predict(self, features):
        """Run inference on a single feature vector.

        Args:
            features: np.ndarray of shape (81,) or (1, 81), float32

        Returns:
            np.ndarray of shape (num_classes,) — raw logits
        """
        features = np.asarray(features, dtype=np.float32).ravel()

        # Copy input to pagelocked host buffer
        np.copyto(self._inputs[0]["host"][:len(features)], features)

        # H2D transfer
        cuda.memcpy_htod_async(
            self._inputs[0]["device"],
            self._inputs[0]["host"],
            self._stream,
        )

        # Execute
        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        )

        # D2H transfer
        cuda.memcpy_dtoh_async(
            self._outputs[0]["host"],
            self._outputs[0]["device"],
            self._stream,
        )

        self._stream.synchronize()

        # Return logits reshaped to (num_classes,)
        out_shape = self._outputs[0]["shape"]
        return self._outputs[0]["host"][:int(np.prod(out_shape))].copy()

    def predict_proba(self, features):
        """Softmax probabilities from inference.

        Args:
            features: np.ndarray of shape (81,)

        Returns:
            np.ndarray of shape (num_classes,) — probabilities summing to 1
        """
        logits = self.predict(features)
        # Stable softmax
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / exp_l.sum()

    @property
    def num_classes(self):
        """Number of output classes."""
        return int(np.prod(self._outputs[0]["shape"]))

    def destroy(self):
        """Release GPU resources."""
        # PyCUDA handles cleanup via garbage collection, but we can
        # explicitly delete references.
        del self._context
        del self._engine
        self._inputs.clear()
        self._outputs.clear()
        self._bindings.clear()
        logger.info("TensorRT engine destroyed")

    def __del__(self):
        try:
            # Skip logging during interpreter shutdown to avoid NameError
            del self._context
            del self._engine
            self._inputs.clear()
            self._outputs.clear()
            self._bindings.clear()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Static builder — ONNX → TensorRT engine
    # -----------------------------------------------------------------

    @staticmethod
    def build_engine(onnx_path, engine_path, fp16=True,
                     max_batch_size=1, max_workspace_mb=256):
        """Build a TensorRT engine from an ONNX model.

        Args:
            onnx_path: Path to .onnx file (exported from GestureNet)
            engine_path: Path to write serialized .engine file
            fp16: Enable FP16 precision (2× speed on Maxwell GPU)
            max_batch_size: Maximum batch size for dynamic shapes
            max_workspace_mb: GPU workspace in MB

        Returns:
            Path to the saved engine file
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("ONNX parse error: %s", parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        # Builder config
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_mb * (1 << 20)

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled (Maxwell GPU)")
        elif fp16:
            logger.warning("FP16 requested but not supported on this GPU")

        # Set optimization profile for dynamic batch
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        input_shape = network.get_input(0).shape  # e.g. (-1, 81)
        feat_dim = input_shape[1]

        profile.set_shape(
            input_name,
            min=(1, feat_dim),
            opt=(1, feat_dim),
            max=(max_batch_size, feat_dim),
        )
        config.add_optimization_profile(profile)

        # Build engine
        logger.info("Building TensorRT engine (this may take 1-2 minutes)...")
        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("TensorRT engine build failed")

        # Serialize
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

        logger.info("TensorRT engine saved to %s (%.1f KB)",
                     engine_path, os.path.getsize(engine_path) / 1024.0)
        return engine_path
