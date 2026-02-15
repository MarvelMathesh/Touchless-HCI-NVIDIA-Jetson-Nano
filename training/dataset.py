"""
PyTorch Dataset for gesture landmark data.

Loads (21, 3) landmark .npy files saved by DatasetCollector,
converts them to 81-dim feature vectors via GestureFeatureExtractor,
and provides standard PyTorch Dataset/DataLoader interface.

Directory structure expected::

    data/gestures/
        thumbs_up/
            thumbs_up_session_xxx_0000_landmarks.npy
            thumbs_up_session_xxx_0000_aug_flip_landmarks.npy
            ...
        fist/
            ...

Compatible with Python 3.6+, PyTorch >= 1.8.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Must match models.gesture_net.DEFAULT_GESTURE_CLASSES order
GESTURE_CLASSES = [
    "thumbs_up",
    "thumbs_down",
    "peace_sign",
    "ok_sign",
    "fist",
    "open_palm",
    "finger_point",
    "i_love_you",
    "none",
]

LABEL_MAP = {name: idx for idx, name in enumerate(GESTURE_CLASSES)}


def _check_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training")


if TORCH_AVAILABLE:

    class GestureDataset(Dataset):
        """PyTorch Dataset that loads landmark .npy files and extracts features.

        Each sample is a (features, label) pair where:
            - features: FloatTensor of shape (81,)
            - label: LongTensor scalar (class index 0-8)
        """

        def __init__(self, data_dir="data/gestures", transform=None,
                     include_augmented=True, feature_extractor=None):
            """
            Args:
                data_dir: Root directory containing per-gesture subdirectories
                transform: Optional callable(features_tensor) → features_tensor
                include_augmented: Whether to include *_aug_*_landmarks.npy files
                feature_extractor: Optional GestureFeatureExtractor instance
            """
            _check_torch()
            self._data_dir = data_dir
            self._transform = transform
            self._samples = []  # list of (npy_path, class_idx)

            # Feature extractor
            if feature_extractor is None:
                from models.feature_extractor import GestureFeatureExtractor
                self._feat_ext = GestureFeatureExtractor()
            else:
                self._feat_ext = feature_extractor

            self._load_samples(include_augmented)

        def _load_samples(self, include_augmented):
            """Scan data_dir for landmark .npy files."""
            if not os.path.isdir(self._data_dir):
                logger.warning("Data directory not found: %s", self._data_dir)
                return

            for gesture_name in os.listdir(self._data_dir):
                if gesture_name not in LABEL_MAP:
                    continue

                gesture_dir = os.path.join(self._data_dir, gesture_name)
                if not os.path.isdir(gesture_dir):
                    continue

                label = LABEL_MAP[gesture_name]

                for fname in sorted(os.listdir(gesture_dir)):
                    if not fname.endswith("_landmarks.npy"):
                        continue

                    # Skip augmented files if requested
                    if not include_augmented and "_aug_" in fname:
                        continue

                    fpath = os.path.join(gesture_dir, fname)
                    self._samples.append((fpath, label))

            logger.info("Loaded %d landmark samples from %s", len(self._samples), self._data_dir)

            # Print per-class counts
            counts = {}
            for _, label in self._samples:
                name = GESTURE_CLASSES[label]
                counts[name] = counts.get(name, 0) + 1
            for name in GESTURE_CLASSES:
                logger.info("  %-15s %d samples", name, counts.get(name, 0))

        def __len__(self):
            return len(self._samples)

        def __getitem__(self, idx):
            npy_path, label = self._samples[idx]

            # Load landmarks
            landmarks = np.load(npy_path).astype(np.float32)
            if landmarks.shape == (21, 3):
                features = self._feat_ext.extract(landmarks)
            elif landmarks.shape == (63,) or landmarks.shape == (81,):
                # Pre-extracted features (fallback)
                features = landmarks
            else:
                raise ValueError("Unexpected landmark shape %s in %s" %
                                 (str(landmarks.shape), npy_path))

            features_tensor = torch.from_numpy(features)

            if self._transform is not None:
                features_tensor = self._transform(features_tensor)

            return features_tensor, torch.tensor(label, dtype=torch.long)

        @property
        def num_classes(self):
            return len(GESTURE_CLASSES)

        @property
        def class_names(self):
            return list(GESTURE_CLASSES)

        def class_weights(self):
            """Compute inverse-frequency class weights for imbalanced data.

            Returns:
                FloatTensor of shape (num_classes,)
            """
            counts = np.zeros(len(GESTURE_CLASSES), dtype=np.float32)
            for _, label in self._samples:
                counts[label] += 1

            # Avoid division by zero
            counts = np.maximum(counts, 1.0)

            # Inverse frequency, normalized
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(GESTURE_CLASSES)

            return torch.from_numpy(weights)

else:
    class GestureDataset:
        """Stub — PyTorch not available."""
        def __init__(self, *args, **kwargs):
            _check_torch()


class NumpyDataset:
    """Pure-NumPy fallback dataset (no PyTorch dependency).

    Returns (features, label) as numpy arrays. For use with sklearn
    or manual training loops on systems without PyTorch.
    """

    def __init__(self, data_dir="data/gestures", include_augmented=True):
        from models.feature_extractor import GestureFeatureExtractor
        self._feat_ext = GestureFeatureExtractor()
        self._features = []
        self._labels = []

        if not os.path.isdir(data_dir):
            logger.warning("Data directory not found: %s", data_dir)
            return

        for gesture_name in sorted(os.listdir(data_dir)):
            if gesture_name not in LABEL_MAP:
                continue
            gesture_dir = os.path.join(data_dir, gesture_name)
            if not os.path.isdir(gesture_dir):
                continue

            label = LABEL_MAP[gesture_name]
            for fname in sorted(os.listdir(gesture_dir)):
                if not fname.endswith("_landmarks.npy"):
                    continue
                if not include_augmented and "_aug_" in fname:
                    continue
                lm = np.load(os.path.join(gesture_dir, fname)).astype(np.float32)
                if lm.shape == (21, 3):
                    feat = self._feat_ext.extract(lm)
                    self._features.append(feat)
                    self._labels.append(label)

        if self._features:
            self._features = np.stack(self._features)
            self._labels = np.array(self._labels, dtype=np.int64)
        else:
            self._features = np.zeros((0, 81), dtype=np.float32)
            self._labels = np.zeros(0, dtype=np.int64)

        logger.info("NumpyDataset: %d samples loaded", len(self._labels))

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self._labels)
