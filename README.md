# Touchless Media Control - Edge AI on Jetson Nano

A high-performance, edge-optimized computer vision system for controlling media playback using hand gestures. Designed specifically for the NVIDIA Jetson Nano, this project utilizes MediaPipe for real-time hand tracking, PyTorch for model training, and TensorRT (FP16) for lightning-fast ML inference.

## 🌟 Key Features

- **Real-Time Edge AI**: Operates at ~30 FPS on a Jetson Nano (Maxwell GPU).
- **Hybrid Gesture Recognition**: Combines a sub-millisecond TensorRT Neural Network for static poses with a heuristic trajectory engine for dynamic gestures (swipes).
- **Thermal Adaptive Performance**: Intelligently monitors SoC temperatures to throttle FPS gracefully, preventing hard thermal shutdowns.
- **Robust Pipeline Architecture**: Decoupled, event-driven components communicating over an internal `EventBus`.
- **User Adaptation**: 30-second profiling sequence to learn user-specific hand mechanics and adjust confidence thresholds automatically.
- **VLC Integration**: Direct media player control via `xdotool` keyboard simulation and D-Bus MPRIS2.

---

## 🏗️ Architecture Overview

The architecture moves away from monolithic processing into a composable, modular pipeline orchestrated in `core/pipeline.py`. 

### The Recognition Pipeline
1. **Capture (`CameraManager`, `FrameProcessor`)**: Asynchronous threaded capture from V4L2/GStreamer, applying optional low-light enhancement.
2. **Detection (`HandDetector`, `HandTracker`)**: Employs MediaPipe (GPU-accelerated) to detect hands and extract 21 3D landmarks.
3. **Feature Extraction (`GestureFeatureExtractor`)**: Transforms the 21 landmarks into an 81-dimensional normalized feature vector (position/scale invariant, captures curl angles, inter-finger distances).
4. **Classification (`HybridClassifier`)**:
   - **Tier 1 (Priority)**: TensorRT FP16 Engine (`<0.3ms`)
   - **Tier 2 (Fallback)**: PyTorch Model (`~1ms`)
   - **Tier 3 (Baseline)**: Rule-based Geometric Classifier based on strict finger curl heuristics.
5. **Temporal Filtering (`TemporalFilter`)**: Hysteresis and consensus voting over rolling windows to eliminate micro-fluctuations.
6. **Execution (`ActionExecutor`)**: Fire-once debounced execution using D-Bus or `xdotool`, executing commands against VLC media player.

### Module Topology
- **`core/`**: Type definitions (`types.py`), event pub/sub (`events.py`), and the core orchestrator (`pipeline.py`).
- **`models/`**: Neural network definitions (`gesture_net.py`), feature extractors, and TensorRT runtime wrappers (`tensorrt_engine.py`).
- **`modules/capture/`**: Camera IO and lighting validation (`calibration.py`).
- **`modules/intelligence/`**: Subsystems for thermal throttling (`thermal_manager.py`), user behavior profiling (`user_profiler.py`), and anomaly detection.
- **`training/`**: Standalone pipelines to train the PyTorch model and export to ONNX/TensorRT.

---

## ✋ Gesture Vocabulary

Mapped in `config/gestures.yaml`:

### Static Gestures
- 👍 **Thumbs Up**: Volume Up
- 👎 **Thumbs Down**: Volume Down
- ✌️ **Peace Sign**: Toggle Aspect Ratio
- 👌 **OK Sign**: Toggle Fullscreen
- ✊ **Closed Fist**: Mute / Unmute
- 🖐️ **Open Palm**: Play / Pause
- ☝️ **Finger Point**: Toggle Playback Speed
- 🤟 **I Love You**: Toggle Subtitles

### Dynamic Gestures
- ➡️ **Swipe Right**: Seek Forward 10s
- ⬅️ **Swipe Left**: Seek Backward 10s

---

## ⚙️ Hardware & Software Requirements

### Hardware
- **NVIDIA Jetson Nano** (4GB recommended) or equivalent Jetson device.
- USB Webcam or CSI Camera.
- Adequate cooling (Active cooling fan highly recommended).

### Software & Dependencies
- JetPack 4.x (Includes CUDA 10.2, TensorRT 7.x, OpenCV 4.1.1)
- Python 3.6
- **MediaPipe 0.8.5** (ARM64 Wheel for Jetson)
- **PyTorch 1.10.0** (NVIDIA compiled aarch64 wheel)
- `xdotool`, `vlc`, `jetson-stats`

Install system packages:
```bash
sudo apt update
sudo apt install xdotool vlc
sudo pip3 install jetson-stats
```

Install Python requirements via the provided file:
```bash
pip3 install -r requirements.txt
```
*(Refer to inline comments in `requirements.txt` for links to explicit aarch64 wheels for MediaPipe and PyTorch).*

---

## 🚀 Usage

The main entry point handles argument parsing and initialises the `TouchlessMediaControl` application.

### Operating Modes

**1. Normal Control Mode (Default)**
Actively controls the media player using inferred gestures.
```bash
python3 main.py
```

**2. Demo Mode**
Processes camera feeds, displays bounding boxes, landmarks, and confidence bars on-screen without emitting actual system control commands.
```bash
python3 main.py --mode demo
```

**3. Benchmark Mode**
Runs a 300-frame stress test to evaluate inference latency and pipeline bottlenecks.
```bash
python3 main.py --mode benchmark
```

**4. Data Collection Mode**
Interactive tool to capture custom landmark datasets for retraining the ML classifier.
```bash
python3 main.py --mode collect
```

### User Calibration
Users have different hand mechanics. Run the calibration sequence prior to starting to tune confidence thresholds specifically to the current user:
```bash
python3 main.py --calibrate
```

---

## 🧠 Model Training & TensorRT Optimization

The project allows for end-to-end retraining directly on the Jetson Nano.

**1. Collect Data**
```bash
python3 main.py --mode collect
```
*Follow on-screen prompts to capture landmarks for different classes.*

**2. Train PyTorch Model**
Trains the lightweight MLP (`GestureNet`).
```bash
python3 -m training.train --epochs 100 --batch-size 32
```

**3. Export to TensorRT**
Exports the `.pth` model to `.onnx`, parses it via TRT Builder, and serializes a highly optimized `.engine` for Maxwell GPUs.
```bash
python3 -m training.train --export
```
The pipeline automatically attempts to load `gesture_net.engine` on the next run.

---

## 🌡️ Thermal Management

The Jetson Nano can reach thermal limits under sustained ML workloads. `ThermalManager` monitors `/sys/class/thermal/thermal_zone*`:
- **< 60°C**: Full Target FPS.
- **60°C - 70°C**: FPS slightly reduced to stabilize temps.
- **70°C - 80°C**: Aggressive FPS halving, restricts heavy pre-processing.
- **> 80°C**: Enters critical fallback state.

---

## 🔧 Configuration

All system parameters are cleanly exposed in `config/config.yaml`:
- **`camera`**: V4L2 backend settings, resolution, FPS lock.
- **`performance`**: Target max latency, thread enabling, thermal envelopes.
- **`recognition`**: Confidence thresholds, temporal hysteresis window sizes.
- **`debouncing`**: Anti-flicker guard timers, repeating action limits.
- **`ml_classifier`**: Fallback directives and model filepaths.

---
*Developed for edge computing workflows demonstrating complex ML pipelines within constrained thermal and memory envelopes.*
