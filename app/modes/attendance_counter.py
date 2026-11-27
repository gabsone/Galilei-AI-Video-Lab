"""Modalità 5: CONTA PRESENTI — Appello AI della classe in tempo reale (100% offline)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from app.hand_tracker import HandData
from app.hud import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_MAGENTA,
    COLOR_MUTED,
    COLOR_ORANGE,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_panel,
    put_text_centered,
)
from app.sound_engine import SoundEngine

BOX_PALETTE = [
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_ORANGE,
    COLOR_MAGENTA,
]


@dataclass
class DetectedStudent:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class AttendanceCounterMode:
    """Rileva e conta gli alunni presenti inquadrando la classe con EfficientDet-Lite2."""

    def __init__(self, sound: SoundEngine, model_path: str | Path) -> None:
        self.sound = sound
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modello rilevamento persone non trovato: {self.model_path}"
            )

        self.threshold = 0.32
        self._detector = self._create_detector()
        self._last_timestamp_ms = 0
        self._start_time = time.perf_counter()
        self.reset()

    def _create_detector(self) -> mp_vision.ObjectDetector:
        options = mp_vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            score_threshold=0.20,  # Filtraggio dinamico in Python cosi [+] / [-] agiscono istantaneamente
            max_results=35,
            category_allowlist=["person"],
        )
        return mp_vision.ObjectDetector.create_from_options(options)

    def reset(self) -> None:
        self._raw_detections: List[DetectedStudent] = []
        self._count_history: Deque[int] = deque(maxlen=12)
        self.stable_count = 0
        self.max_count = 0
        self.frozen = False
        self.frozen_count = 0
        self._frame_counter = 0

    def handle_key(self, key: int) -> None:
        if key == ord(" "):
            self.frozen = not self.frozen
            if self.frozen:
                self.frozen_count = self.stable_count
                self.sound.play("win")
            else:
                self.sound.play("switch")
        elif key in (ord("+"), ord("=")):
            self.threshold = min(0.75, round(self.threshold + 0.04, 2))
            self.sound.play("switch")
        elif key in (ord("-"), ord("_")):
            self.threshold = max(0.20, round(self.threshold - 0.04, 2))
            self.sound.play("switch")

    @staticmethod
    def _draw_corner_box(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        thickness: int = 3,
    ) -> None:
        """Disegna un riquadro stile visore AI con angoli rinforzati e contorno sottile."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        corner = max(14, min(32, int(min(x2 - x1, y2 - y1) * 0.22)))
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, thickness, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, thickness, cv2.LINE_AA)

    def _detect_people(self, frame_bgr: np.ndarray) -> List[DetectedStudent]:
        h, w = frame_bgr.shape[:2]
        ts = int((time.perf_counter() - self._start_time) * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect_for_video(mp_image, ts)

        students: List[DetectedStudent] = []
        for det in result.detections:
            if not det.categories:
                continue
            score = float(det.categories[0].score)
            bbox = det.bounding_box
            x1 = max(0, min(w - 1, int(bbox.origin_x)))
            y1 = max(0, min(h - 1, int(bbox.origin_y)))
            x2 = max(0, min(w - 1, int(bbox.origin_x + bbox.width)))
            y2 = max(0, min(h - 1, int(bbox.origin_y + bbox.height)))
            if x2 - x1 > 20 and y2 - y1 > 20:
                students.append(DetectedStudent(x1, y1, x2, y2, score))

        # Ordina da sinistra a destra per numerazione stabile #1, #2, #3...
        students.sort(key=lambda s: (s.x1 + s.x2) * 0.5)
        return students

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]

        if not self.frozen:
            # Esegui inferenza 1 frame su 2 per mantenere video fluidissimo
            if self._frame_counter % 2 == 0 or not self._raw_detections:
                self._raw_detections = self._detect_people(frame)
            self._frame_counter += 1

        visible = [s for s in self._raw_detections if s.score >= self.threshold]

        if not self.frozen:
            self._count_history.append(len(visible))
            # Mediana mobile per evitare sfarfallii nel contatore
            sorted_hist = sorted(self._count_history)
            self.stable_count = sorted_hist[len(sorted_hist) // 2]
            if self.stable_count > self.max_count:
                self.max_count = self.stable_count

        # Disegna riquadro + badge attorno a ogni alunno
        for idx, st in enumerate(visible, start=1):
            col = BOX_PALETTE[(idx - 1) % len(BOX_PALETTE)]
            self._draw_corner_box(frame, st.x1, st.y1, st.x2, st.y2, col, thickness=3)

            label = f"ALUNNO #{idx}  {int(st.score * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            by1 = max(68, st.y1 - th - 14)
            by2 = by1 + th + 10
            bx2 = min(w - 8, st.x1 + tw + 14)
            draw_panel(frame, st.x1, by1, bx2, by2, bg_color=(14, 18, 28), border_color=col, alpha=0.82, thickness=1)
            cv2.putText(
                frame,
                label,
                (st.x1 + 7, by2 - 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                COLOR_WHITE,
                1,
                cv2.LINE_AA,
            )

            if debug_mode:
                info = f"box={st.x2 - st.x1}x{st.y2 - st.y1}px"
                cv2.putText(frame, info, (st.x1 + 6, st.y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_YELLOW, 1, cv2.LINE_AA)

        # Pannello gigante "CONTA PRESENTI" in alto a destra
        displayed_count = self.frozen_count if self.frozen else self.stable_count
        px1, py1, px2, py2 = w - 285, 76, w - 16, 242
        draw_panel(
            frame,
            px1,
            py1,
            px2,
            py2,
            bg_color=(14, 20, 32),
            border_color=COLOR_GREEN if self.frozen else COLOR_CYAN,
            alpha=0.84,
            thickness=3 if self.frozen else 2,
        )
        put_text_centered(
            frame,
            "APPELLO BLOCCATO" if self.frozen else "APPELLO GALILEI-DI PALO",
            ((px1 + px2) // 2, py1 + 26),
            scale=0.52,
            color=COLOR_GREEN if self.frozen else COLOR_CYAN,
            thickness=2,
            shadow=False,
        )
        put_text_centered(
            frame,
            str(displayed_count),
            ((px1 + px2) // 2, py1 + 92),
            scale=2.5,
            color=COLOR_YELLOW if self.frozen else COLOR_WHITE,
            thickness=5,
        )
        cv2.line(frame, (px1 + 16, py1 + 132), (px2 - 16, py1 + 132), (55, 70, 95), 1)
        cv2.putText(
            frame,
            f"PICCO MAX : {self.max_count}   |   SOGLIA: {int(self.threshold * 100)}%",
            (px1 + 14, py1 + 154),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

        # Banner celebrativo quando l'appello viene bloccato con SPAZIO
        if self.frozen:
            cx = w // 2
            draw_panel(frame, cx - 265, h - 125, cx + 265, h - 52, border_color=COLOR_GREEN, alpha=0.88, thickness=3)
            put_text_centered(
                frame,
                f"APPELLO COMPLETATO: {self.frozen_count} ALUNNI PRESENTI!",
                (cx, h - 98),
                scale=0.72,
                color=COLOR_GREEN,
                thickness=2,
            )
            put_text_centered(
                frame,
                "Premi [SPAZIO] per riprendere la scansione live",
                (cx, h - 68),
                scale=0.48,
                color=COLOR_WHITE,
                thickness=1,
                shadow=False,
            )

        return "[SPAZIO] Blocca/Riprendi Appello  |  [+]/[-] Sensibilita AI  |  [R] Azzera Picco"

    def close(self) -> None:
        self._detector.close()
