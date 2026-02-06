#!/bin/bash
# Quick camera test for Jetson Nano

echo "Testing camera with V4L2 backend..."
python3 -c "
import cv2
import time

# Try V4L2 backend
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print('❌ Failed to open camera with V4L2')
    exit(1)

print('✓ Camera opened with V4L2')

# Try to read a frame
ret, frame = cap.read()

if  ret and frame is not None:
    h, w = frame.shape[:2]
    print('✓ Frame captured: {}x{}'.format(w, h))
    
    # Try to capture 10 frames
    success = 0
    for i in range(10):
        ret, _ = cap.read()
        if ret:
            success += 1
        time.sleep(0.033)
    
    print('✓ Captured {}/10 frames'.format(success))
    
    if success >= 8:
        print('\\n✅ Camera is working well!')
        print('   Use this in config.yaml:')
        print('   camera:')
        print('     device_id: 0')
    else:
        print('\\n⚠️  Camera is unstable ({}/10 frames)'.format(success))
else:
    print('❌ Failed to read frame')

cap.release()
"
