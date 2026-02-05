# Touchless HCI Media Control System

A high-performance hand gesture recognition system for media control on NVIDIA Jetson Nano.

## Features

- **Real-time Hand Detection**: MediaPipe Hands for 21-point landmark detection
- **Static Gesture Recognition**: 7 predefined gestures (fist, palm, thumbs, victory, etc.)
- **Dynamic Gesture Recognition**: Swipes and circular motions
- **VLC Media Control**: Play/pause, volume, seek, mute, fullscreen
- **Performance Optimized**: <30ms latency, ≥25 FPS on Jetson Nano
- **Dataset Collection Tool**: Interactive capture for training custom gestures

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install xdotool for media control
sudo apt-get install xdotool

# Verify MediaPipe installation
python -c "import mediapipe; print(mediapipe.__version__)"
```

### Run the Application

```bash
# Full control mode
python src/main.py --mode control

# Demo mode (visualization only)
python src/main.py --mode demo

# With debug logging
python src/main.py --mode control --debug
```

### Collect Training Data

```bash
python src/data/dataset_collector.py \
    --participant your_name \
    --lighting indoor \
    --background office
```

## Gesture Vocabulary

| Gesture | Action | Description |
|---------|--------|-------------|
| ✊ Closed Fist | Play/Pause | All fingers curled |
| 🖐 Open Palm | Play/Pause | All fingers extended |
| 👍 Thumb Up | Volume Up | Thumb up, others closed |
| 👎 Thumb Down | Volume Down | Thumb down, others closed |
| ✌️ Victory | Seek Forward | Index + middle extended |
| ☝️ Pointing | Seek Backward | Index only extended |
| 🤟 I Love You | Mute | Thumb + index + pinky |
| → Swipe Right | Seek Forward | Quick rightward motion |
| ← Swipe Left | Seek Backward | Quick leftward motion |
| ⭕ Circle | Fullscreen | Circular hand motion |

## Keyboard Controls

- `q` / `ESC`: Quit application
- `m`: Toggle control/demo mode
- `p`: Print performance report

## Project Structure

```
jetson_second/
├── config/
│   ├── config.yaml        # Main configuration
│   └── gestures.yaml      # Gesture definitions
├── src/
│   ├── main.py            # Application entry point
│   ├── capture/           # Camera capture module
│   ├── detection/         # Hand detection (MediaPipe)
│   ├── recognition/       # Gesture classification
│   ├── control/           # Media control (xdotool)
│   ├── utils/             # Performance & visualization
│   └── data/              # Dataset collection
├── tests/                 # Test suite
├── data/gestures/         # Collected gesture data
└── requirements.txt
```

## Performance Targets

- **Latency**: <30ms end-to-end
- **Accuracy**: >99% gesture recognition
- **Frame Rate**: ≥25 FPS sustained

## Hardware Requirements

- NVIDIA Jetson Nano (4GB recommended)
- USB Webcam (1280x720 @ 30fps)
- JetPack 4.6+ with CUDA support

## License

MIT License
