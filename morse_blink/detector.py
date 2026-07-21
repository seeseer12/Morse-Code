"""Eye blink detection using MediaPipe Face Mesh and Eye Aspect Ratio (EAR)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe Face Mesh landmark indices for EAR calculation.
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)


def _ear(landmarks, indices, width: int, height: int) -> float:
    """Compute Eye Aspect Ratio from six landmark points."""
    points = np.array(
        [(landmarks[i].x * width, landmarks[i].y * height) for i in indices],
        dtype=np.float64,
    )
    vertical_a = np.linalg.norm(points[1] - points[5])
    vertical_b = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


class BlinkPhase(Enum):
    OPEN = auto()
    CLOSED = auto()


@dataclass
class BlinkEvent:
    """A completed blink with duration in seconds."""

    duration: float
    is_dot: bool


@dataclass
class FrameResult:
    """Output from processing a single camera frame."""

    ear: float
    face_detected: bool
    phase: BlinkPhase
    blink: BlinkEvent | None = None


class BlinkDetector:
    """
    Detects eye blinks from webcam frames.

    Uses average EAR across both eyes. When EAR drops below `ear_threshold`,
    eyes are considered closed. When they reopen, blink duration is measured
    and classified as dot (short) or dash (long).
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        dot_max_duration: float = 0.35,
        consecutive_frames: int = 2,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.dot_max_duration = dot_max_duration
        self.consecutive_frames = consecutive_frames

        self._phase = BlinkPhase.OPEN
        self._closed_frames = 0
        self._open_frames = 0
        self._close_start: float | None = None

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame: np.ndarray) -> FrameResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._reset_blink_state()
            return FrameResult(ear=0.0, face_detected=False, phase=BlinkPhase.OPEN)

        landmarks = results.multi_face_landmarks[0].landmark
        left = _ear(landmarks, LEFT_EYE, width, height)
        right = _ear(landmarks, RIGHT_EYE, width, height)
        ear = (left + right) / 2.0

        blink_event = self._update_state(ear)
        return FrameResult(
            ear=ear,
            face_detected=True,
            phase=self._phase,
            blink=blink_event,
        )

    def _update_state(self, ear: float) -> BlinkEvent | None:
        eyes_closed = ear < self.ear_threshold

        if self._phase == BlinkPhase.OPEN:
            if eyes_closed:
                self._closed_frames += 1
                if self._closed_frames >= self.consecutive_frames:
                    self._phase = BlinkPhase.CLOSED
                    self._close_start = time.monotonic()
                    self._open_frames = 0
            else:
                self._closed_frames = 0
            return None

        # CLOSED phase — waiting for eyes to reopen
        if eyes_closed:
            self._open_frames = 0
            return None

        self._open_frames += 1
        if self._open_frames < self.consecutive_frames:
            return None

        duration = time.monotonic() - (self._close_start or time.monotonic())
        self._phase = BlinkPhase.OPEN
        self._closed_frames = 0
        self._close_start = None

        if duration < 0.08:
            return None

        is_dot = duration <= self.dot_max_duration
        return BlinkEvent(duration=duration, is_dot=is_dot)

    def _reset_blink_state(self) -> None:
        self._phase = BlinkPhase.OPEN
        self._closed_frames = 0
        self._open_frames = 0
        self._close_start = None

    def close(self) -> None:
        self._face_mesh.close()
