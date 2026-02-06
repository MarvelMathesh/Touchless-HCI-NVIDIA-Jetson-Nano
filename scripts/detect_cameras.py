#!/usr/bin/env python3
"""
Camera detection utility for Jetson Nano.
Lists available cameras and helps configure the correct device.
"""

import cv2
import subprocess
import os


def list_video_devices():
    """List all /dev/video* devices."""
    devices = []
    for i in range(10):
        device = "/dev/video{}".format(i)
        if os.path.exists(device):
            devices.append(device)
    return devices


def test_camera(device_id):
    """Test if camera at device_id works."""
    print("Testing camera {}...".format(device_id))
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print("  ✗ Failed to open")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        height, width = frame.shape[:2]
        print("  ✓ Works! Resolution: {}x{}".format(width, height))
        return True
    else:
        print("  ✗ Opened but can't read frames")
        return False


def check_gstreamer_cameras():
    """Check for CSI cameras using GStreamer."""
    print("\nChecking for CSI cameras (GStreamer)...")
    
    # Try common Jetson CSI camera pipelines
    pipelines = [
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink",
        "nvcamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=I420, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    ]
    
    for i, pipeline in enumerate(pipelines):
        print("  Testing CSI pipeline {}...".format(i))
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("    ✓ CSI camera found!")
                print("    Pipeline: {}".format(pipeline))
                return pipeline
        else:
            print("    ✗ No CSI camera")
    
    return None


def main():
    print("=" * 60)
    print("JETSON NANO CAMERA DETECTION")
    print("=" * 60)
    
    # Check for video devices
    print("\nUSB Cameras (/dev/video*):")
    devices = list_video_devices()
    
    if not devices:
        print("  No /dev/video* devices found")
    else:
        print("  Found devices: {}".format(", ".join(devices)))
    
    # Test each device
    working_cameras = []
    for device in devices:
        device_id = int(device.replace("/dev/video", ""))
        if test_camera(device_id):
            working_cameras.append(device_id)
    
    # Check for CSI cameras
    csi_pipeline = check_gstreamer_cameras()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if working_cameras:
        print("\n✓ Found {} working USB camera(s):".format(len(working_cameras)))
        for cam_id in working_cameras:
            print("  - Device ID: {}".format(cam_id))
        
        print("\nTo use in config.yaml:")
        print("  camera:")
        print("    device_id: {}".format(working_cameras[0]))
    
    if csi_pipeline:
        print("\n✓ Found CSI camera!")
        print("\nTo use CSI camera, modify camera.py to use GStreamer pipeline:")
        print("  cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)")
    
    if not working_cameras and not csi_pipeline:
        print("\n✗ No cameras detected!")
        print("\nTroubleshooting:")
        print("  1. Connect a USB webcam and run 'lsusb' to verify")
        print("  2. For CSI camera, check ribbon cable connection")
        print("  3. Try: sudo chmod 666 /dev/video*")
        print("  4. Check dmesg for camera errors: dmesg | grep -i camera")


if __name__ == "__main__":
    main()
