"""
ML models package for gesture classification.

Provides:
    - GestureFeatureExtractor: Landmark → normalized feature vector
    - GestureNet: Lightweight MLP for gesture classification
    - TensorRTEngine: FP16 TensorRT inference wrapper
    - HybridClassifier: ML-first classifier with rule-based fallback
"""

__all__ = [
    "GestureFeatureExtractor",
    "GestureNet",
    "TensorRTEngine",
    "HybridClassifier",
]
