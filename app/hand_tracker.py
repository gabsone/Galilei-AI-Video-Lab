"""Modulo di rilevamento mani e riconoscimento gesti basato su MediaPipe Tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

# Connessioni anatomiche standard dei 21 punti della mano MediaPipe
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pollice
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indice
    (5, 9), (9, 10), (10, 11), (11, 12),   # Medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anulare
    (13, 17), (17, 18), (18, 19), (19, 20),# Mignolo
    (0, 17),                               # Base del palmo
]

FINGER_TIPS = (4, 8, 12, 16, 20)
FINGER_PIPS = (3, 6, 10, 14, 18)


@dataclass
class HandData:
    """Rappresenta una mano rilevata con coordinate in pixel e metriche utili."""

    handedness: str                     # "Left" o "Right"
    landmarks_px: List[Tuple[int, int]] # 21 punti (x, y) in pixel
    landmarks_norm: List[Tuple[float, float, float]]
    fingers_up: List[int]               # [pollice, indice, medio, anulare, mignolo] -> 0/1
    gesture: str                        # "SASSO", "CARTA", "FORBICI", "PINCH", "POLLICE SU", "ROCK", "ALTRO"
    index_tip: Tuple[int, int]
    thumb_tip: Tuple[int, int]
    palm_center: Tuple[int, int]
    pinch_distance_px: float
    pinch_ratio: float                  # Distanza pollice-indice normalizzata sulla larghezza del palmo
    is_pinching: bool


class HandTracker:
    """Wrapper offline-ready per MediaPipe HandLandmarker (modalità VIDEO)."""

    def __init__(
        self,
        model_path: str | Path,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.55,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modello MediaPipe non trovato: {model_path}. "
                "Assicurati che models/hand_landmarker.task sia presente."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[HandData]:
        """Rileva le mani nel frame BGR (già specchiato) e restituisce HandData."""
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        hands: List[HandData] = []

        if not result.hand_landmarks:
            return hands

        for idx, lm_list in enumerate(result.hand_landmarks):
            handedness = "Right"
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                handedness = result.handedness[idx][0].category_name

            landmarks_px: List[Tuple[int, int]] = []
            landmarks_norm: List[Tuple[float, float, float]] = []
            for lm in lm_list:
                px = int(max(0, min(w - 1, lm.x * w)))
                py = int(max(0, min(h - 1, lm.y * h)))
                landmarks_px.append((px, py))
                landmarks_norm.append((lm.x, lm.y, lm.z))

            fingers_up = self._compute_fingers_up(landmarks_px)
            index_tip = landmarks_px[8]
            thumb_tip = landmarks_px[4]

            # Centro del palmo (media tra polso e base delle 4 dita principali)
            palm_pts = [landmarks_px[i] for i in (0, 5, 9, 13, 17)]
            palm_center = (
                int(sum(p[0] for p in palm_pts) / len(palm_pts)),
                int(sum(p[1] for p in palm_pts) / len(palm_pts)),
            )

            # Scala di riferimento della mano (polso -> nocca medio)
            palm_scale = max(25.0, math.hypot(
                landmarks_px[9][0] - landmarks_px[0][0],
                landmarks_px[9][1] - landmarks_px[0][1],
            ))
            pinch_dist = math.hypot(
                index_tip[0] - thumb_tip[0],
                index_tip[1] - thumb_tip[1],
            )
            pinch_ratio = pinch_dist / palm_scale
            is_pinching = pinch_ratio < 0.34

            gesture = self._classify_gesture(fingers_up, landmarks_px, is_pinching)

            hands.append(
                HandData(
                    handedness=handedness,
                    landmarks_px=landmarks_px,
                    landmarks_norm=landmarks_norm,
                    fingers_up=fingers_up,
                    gesture=gesture,
                    index_tip=index_tip,
                    thumb_tip=thumb_tip,
                    palm_center=palm_center,
                    pinch_distance_px=pinch_dist,
                    pinch_ratio=pinch_ratio,
                    is_pinching=is_pinching,
                )
            )

        return hands

    @staticmethod
    def _compute_fingers_up(pts: List[Tuple[int, int]]) -> List[int]:
        """Determina quali dita sono estese usando distanze geometriche dal polso."""
        wrist = pts[0]
        mcp_middle = pts[9]
        palm_len = max(1.0, math.hypot(mcp_middle[0] - wrist[0], mcp_middle[1] - wrist[1]))

        # Pollice: esteso se la punta (4) è lontana dalla base del mignolo (17) rispetto all'articolazione (3)
        thumb_tip_dist = math.hypot(pts[4][0] - pts[17][0], pts[4][1] - pts[17][1])
        thumb_ip_dist = math.hypot(pts[3][0] - pts[17][0], pts[3][1] - pts[17][1])
        thumb_up = 1 if thumb_tip_dist > thumb_ip_dist + 0.12 * palm_len else 0

        fingers = [thumb_up]
        # Per indice, medio, anulare, mignolo: confrontiamo la distanza dal polso tra punta e PIP
        for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            tip_d = math.hypot(pts[tip_idx][0] - wrist[0], pts[tip_idx][1] - wrist[1])
            pip_d = math.hypot(pts[pip_idx][0] - wrist[0], pts[pip_idx][1] - wrist[1])
            fingers.append(1 if tip_d > pip_d * 1.12 else 0)

        return fingers

    @staticmethod
    def _classify_gesture(
        fingers: List[int],
        pts: List[Tuple[int, int]],
        is_pinching: bool,
    ) -> str:
        """Classifica gesti intuitivi per Sasso-Carta-Forbici e controlli UI."""
        thumb, index, middle, ring, pinky = fingers
        four_fingers = index + middle + ring + pinky

        if four_fingers == 0 and thumb == 1 and pts[4][1] < pts[2][1] - 12:
            return "POLLICE SU"
        if four_fingers == 0:
            return "SASSO"
        if index == 1 and middle == 1 and ring == 0 and pinky == 0:
            return "FORBICI"
        if index == 1 and middle == 0 and ring == 0 and pinky == 1:
            return "ROCK"
        if four_fingers >= 4:
            return "CARTA"
        if is_pinching:
            return "PINCH"
        if index == 1 and middle == 0 and ring == 0 and pinky == 0:
            return "INDICE"
        return "ALTRO"

    def close(self) -> None:
        self._landmarker.close()
