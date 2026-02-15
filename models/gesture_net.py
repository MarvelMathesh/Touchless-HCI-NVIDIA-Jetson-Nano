"""
GestureNet — lightweight MLP for gesture classification.

Architecture (optimized for Jetson Nano Maxwell GPU):
    Input  : 81 features (from GestureFeatureExtractor)
    FC1    : 128 units, BatchNorm, ReLU, Dropout(0.3)
    FC2    : 64 units, BatchNorm, ReLU, Dropout(0.2)
    FC3    : 32 units, ReLU
    Output : num_classes (softmax applied externally or via loss fn)

Total parameters: ~15 K — runs in <0.5 ms on TensorRT FP16.

Compatible with:
    - PyTorch >= 1.8 (Jetson Nano JetPack 4.x wheels)
    - ONNX opset 11 (for TensorRT 7.x conversion)
    - Python 3.6+
"""

import os
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — GestureNet disabled")

# Default classes matching core.types.GestureType (excluding NONE, SWIPE_*)
DEFAULT_GESTURE_CLASSES = [
    "thumbs_up",
    "thumbs_down",
    "peace_sign",
    "ok_sign",
    "fist",
    "open_palm",
    "finger_point",
    "i_love_you",
    "none",          # "no gesture" class for rejection
]

NUM_DEFAULT_CLASSES = len(DEFAULT_GESTURE_CLASSES)


def _check_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for GestureNet. "
            "Install from https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
        )


if TORCH_AVAILABLE:

    class GestureNet(nn.Module):
        """Lightweight MLP for hand gesture classification.

        Designed to be small enough for real-time inference on the
        Jetson Nano (Maxwell 128-core GPU) while achieving high
        accuracy on the 9-class gesture vocabulary.
        """

        def __init__(self, input_dim=81, num_classes=NUM_DEFAULT_CLASSES,
                     dropout1=0.3, dropout2=0.2):
            super(GestureNet, self).__init__()

            self.features = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout1),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout2),

                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
            )

            self.classifier = nn.Linear(32, num_classes)

            # Initialize weights for faster convergence
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            """Forward pass.

            Args:
                x: Tensor of shape (batch, 81)

            Returns:
                Tensor of shape (batch, num_classes) — raw logits
            """
            x = self.features(x)
            x = self.classifier(x)
            return x

        def predict_proba(self, x):
            """Get softmax probabilities (for inference).

            Args:
                x: Tensor of shape (batch, 81)

            Returns:
                Tensor of shape (batch, num_classes) — probabilities
            """
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                return torch.softmax(logits, dim=1)

        def export_onnx(self, output_path, opset_version=11):
            """Export model to ONNX format for TensorRT conversion.

            Args:
                output_path: Path to save .onnx file
                opset_version: ONNX opset (11 for TensorRT 7.x compat)
            """
            self.eval()
            dummy_input = torch.randn(1, 81)
            if next(self.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            torch.onnx.export(
                self,
                dummy_input,
                output_path,
                input_names=["landmarks_features"],
                output_names=["gesture_logits"],
                dynamic_axes={
                    "landmarks_features": {0: "batch_size"},
                    "gesture_logits": {0: "batch_size"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
            logger.info("ONNX model exported to %s", output_path)

        @classmethod
        def load_checkpoint(cls, path, device="cpu"):
            """Load a trained model from checkpoint.

            Args:
                path: Path to .pth checkpoint file
                device: Device to load onto ('cpu' or 'cuda')

            Returns:
                Loaded GestureNet model in eval mode
            """
            _check_torch()
            checkpoint = torch.load(path, map_location=device)

            # Support both full checkpoint dict and raw state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                num_classes = checkpoint.get("num_classes", NUM_DEFAULT_CLASSES)
                input_dim = checkpoint.get("input_dim", 81)
            else:
                state_dict = checkpoint
                # Infer from classifier weight shape
                clf_key = "classifier.weight"
                if clf_key in state_dict:
                    num_classes = state_dict[clf_key].shape[0]
                    input_dim = 81
                else:
                    num_classes = NUM_DEFAULT_CLASSES
                    input_dim = 81

            model = cls(input_dim=input_dim, num_classes=num_classes)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            logger.info("Loaded GestureNet (%d classes) from %s", num_classes, path)
            return model

else:
    # Stub when PyTorch is not installed
    class GestureNet:
        """Stub — PyTorch not available."""
        def __init__(self, *args, **kwargs):
            _check_torch()
