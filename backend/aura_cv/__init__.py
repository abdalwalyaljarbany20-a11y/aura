"""AURA VISION Python computer-vision engine.

Pipeline (one active camera owner, bounded buffers):

    CameraManager (OpenCV capture thread, latest-frame slot)
      -> validate_frame
      -> HandTracker (MediaPipe Tasks HandLandmarker, VIDEO mode)
      -> landmark validation + TrackingStateMachine
      -> extract_features
      -> GestureRecognitionEngine (static rules, motion gate, stabilizer, swipes)
      -> structured result dict (protocol.py)
      -> WebSocket server (server.py) -> existing browser frontend
"""

__version__ = "0.1.0"
