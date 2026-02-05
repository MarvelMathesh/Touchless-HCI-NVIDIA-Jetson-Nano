"""
Tests for Performance Module
=============================
"""

import pytest
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.performance import PerformanceMonitor, Timer


class TestTimer:
    """Test suite for Timer class."""
    
    def test_basic_timing(self):
        """Test basic timer functionality."""
        timer = Timer("test")
        timer.start()
        time.sleep(0.1)
        elapsed = timer.stop()
        
        assert elapsed >= 0.09
        assert elapsed < 0.2
    
    def test_context_manager(self):
        """Test timer as context manager."""
        with Timer("test") as t:
            time.sleep(0.05)
        
        assert t.elapsed >= 0.04
        assert t.elapsed_ms >= 40
    
    def test_elapsed_ms(self):
        """Test milliseconds conversion."""
        timer = Timer("test")
        timer.start()
        time.sleep(0.01)
        timer.stop()
        
        # Should be at least 10ms
        assert timer.elapsed_ms >= 9
    
    def test_elapsed_without_stop(self):
        """Test getting elapsed time while running."""
        timer = Timer("test")
        timer.start()
        time.sleep(0.05)
        
        # Should return running time
        assert timer.elapsed >= 0.04


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create performance monitor."""
        mon = PerformanceMonitor(window_size=5)
        mon.start()
        return mon
    
    def test_fps_calculation(self, monitor):
        """Test FPS calculation."""
        # Simulate frames at ~30 FPS
        for _ in range(10):
            monitor.frame_start()
            time.sleep(0.033)  # ~30 FPS
            monitor.frame_complete()
        
        fps = monitor.fps
        assert 25 < fps < 35
    
    def test_stage_timing(self, monitor):
        """Test per-stage timing."""
        for _ in range(5):
            monitor.frame_start()
            
            with monitor.measure("capture"):
                time.sleep(0.005)
            
            with monitor.measure("detection"):
                time.sleep(0.010)
            
            monitor.frame_complete()
        
        capture_time = monitor.stage_time_ms("capture")
        detection_time = monitor.stage_time_ms("detection")
        
        assert capture_time >= 4
        assert detection_time >= 9
        assert detection_time > capture_time
    
    def test_metrics_snapshot(self, monitor):
        """Test getting metrics snapshot."""
        monitor.frame_start()
        time.sleep(0.02)
        monitor.frame_complete()
        
        metrics = monitor.get_metrics()
        
        assert hasattr(metrics, 'fps')
        assert hasattr(metrics, 'frame_time_ms')
        assert hasattr(metrics, 'total_frames')
        assert metrics.total_frames >= 1
    
    def test_targets(self, monitor):
        """Test performance target checking."""
        monitor.target_fps = 25
        monitor.target_latency_ms = 40
        
        # Simulate meeting targets
        for _ in range(5):
            monitor.frame_start()
            time.sleep(0.03)  # 33ms < 40ms target
            monitor.frame_complete()
        
        # Should be meeting targets
        assert monitor.fps > 20
    
    def test_report_generation(self, monitor):
        """Test report string generation."""
        monitor.frame_start()
        monitor.frame_complete()
        
        report = monitor.get_report()
        
        assert "FPS" in report
        assert "Latency" in report
        assert isinstance(report, str)
    
    def test_stop_clears_state(self, monitor):
        """Test that stop logs correctly."""
        monitor.frame_start()
        monitor.frame_complete()
        
        # Should not raise
        monitor.stop()


class TestPerformanceTargets:
    """Test target-based performance validation."""
    
    def test_latency_under_30ms(self):
        """Validate that typical operations complete under 30ms."""
        monitor = PerformanceMonitor()
        monitor.target_latency_ms = 30
        monitor.start()
        
        # Simulate fast processing
        for _ in range(10):
            monitor.frame_start()
            time.sleep(0.015)  # 15ms
            monitor.frame_complete()
        
        assert monitor.total_latency_ms < 30
    
    def test_sustained_25_fps(self):
        """Validate ability to sustain 25+ FPS."""
        monitor = PerformanceMonitor()
        monitor.target_fps = 25
        monitor.start()
        
        # Simulate 30 FPS processing
        for _ in range(30):
            monitor.frame_start()
            time.sleep(0.033)
            monitor.frame_complete()
        
        assert monitor.fps >= 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
