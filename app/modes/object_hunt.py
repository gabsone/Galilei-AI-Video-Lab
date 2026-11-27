"""Modalità 10: CACCIA AGLI OGGETTI — Sfida a tempo in classe con EfficientDet (100% offline)."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from app.dwell_ui import DwellButton, DwellController
from app.hand_tracker import HandData
from app.hud import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_panel,
    put_text_centered,
)
from app.sound_engine import SoundEngine

# Oggetti COCO facilissimi da trovare in una classe di terza media!
CLASSROOM_TARGETS: List[Tuple[str, str]] = [
    ("cell phone", "UN CELLULARE"),
    ("bottle", "UNA BOTTIGLIA"),
    ("backpack", "UNO ZAINO"),
    ("book", "UN LIBRO / QUADERNO"),
    ("chair", "UNA SEDIA"),
    ("cup", "UN BICCHIERE / TAZZA"),
    ("laptop", "UN COMPUTER"),
    ("scissors", "DELLE FORBICI"),
]


class ObjectHuntMode:
    """Sfida Caccia al Tesoro AI: trova e mostra davanti alla webcam l'oggetto richiesto in 12s!"""

    ROUND_TIME = 12.0

    def __init__(self, sound: SoundEngine, model_path: str | Path) -> None:
        self.sound = sound
        self.dwell = DwellController(sound)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modello ObjectDetector non trovato: {self.model_path}")

        allowlist = [coco_name for coco_name, _ in CLASSROOM_TARGETS]
        options = mp_vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            score_threshold=0.32,
            max_results=10,
            category_allowlist=allowlist,
        )
        self._detector = mp_vision.ObjectDetector.create_from_options(options)
        self._start_time = time.perf_counter()
        self._last_ts = 0
        self.reset()

    def reset(self) -> None:
        self.score = 0
        self.streak = 0
        self.target_coco, self.target_it = random.choice(CLASSROOM_TARGETS)
        self.time_left = self.ROUND_TIME
        self.found_banner = 0.0
        self.timeout_banner = 0.0
        self._frame_counter = 0
        self._last_boxes: List[Tuple[int, int, int, int, str, float]] = []

    def _next_target(self) -> None:
        choices = [t for t in CLASSROOM_TARGETS if t[0] != self.target_coco]
        self.target_coco, self.target_it = random.choice(choices)
        self.time_left = self.ROUND_TIME

    def handle_key(self, key: int) -> None:
        if key == ord(" "):
            self._next_target()
            self.sound.play("switch")

    def _detect_objects(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        h, w = frame_bgr.shape[:2]
        ts = int((time.perf_counter() - self._start_time) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        out: List[Tuple[int, int, int, int, str, float]] = []
        for det in res.detections:
            if not det.categories:
                continue
            cat = det.categories[0]
            bb = det.bounding_box
            x1 = max(0, min(w - 1, int(bb.origin_x)))
            y1 = max(0, min(h - 1, int(bb.origin_y)))
            x2 = max(0, min(w - 1, int(bb.origin_x + bb.width)))
            y2 = max(0, min(h - 1, int(bb.origin_y + bb.height)))
            out.append((x1, y1, x2, y2, cat.category_name, float(cat.score)))
        return out

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]

        # Inferenza 1 frame su 2 per fluidita massima
        if self._frame_counter % 2 == 0:
            self._last_boxes = self._detect_objects(frame)
        self._frame_counter += 1

        # Aggiorna banner e timer
        if self.found_banner > 0:
            self.found_banner = max(0.0, self.found_banner - dt)
            if self.found_banner == 0.0:
                self._next_target()
        elif self.timeout_banner > 0:
            self.timeout_banner = max(0.0, self.timeout_banner - dt)
            if self.timeout_banner == 0.0:
                self._next_target()
        else:
            self.time_left = max(0.0, self.time_left - dt)
            if self.time_left <= 0.0:
                self.streak = 0
                self.timeout_banner = 1.4
                self.sound.play("lose")

        # Disegna gli oggetti rilevati e controlla se il bersaglio e stato trovato!
        for x1, y1, x2, y2, name, score in self._last_boxes:
            is_target = name == self.target_coco
            col = COLOR_GREEN if is_target else COLOR_CYAN
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 4 if is_target else 2, cv2.LINE_AA)
            label = f"{name.upper()} {int(score * 100)}%"
            draw_panel(frame, x1, max(70, y1 - 28), min(w - 8, x1 + 210), max(98, y1), border_color=col, alpha=0.82, thickness=1)
            cv2.putText(frame, label, (x1 + 6, max(91, y1 - 7)), cv2.FONT_HERSHEY_DUPLEX, 0.50, COLOR_WHITE, 1, cv2.LINE_AA)

            if is_target and score >= 0.42 and self.found_banner <= 0 and self.timeout_banner <= 0:
                self.streak += 1
                pts = 100 + int(self.time_left * 10)
                self.score += pts
                self.found_banner = 1.6
                self.sound.play("win")

        # Banner sfida gigante in alto al centro
        cx = w // 2
        draw_panel(frame, cx - 285, 74, cx + 285, 152, border_color=COLOR_YELLOW, alpha=0.84, thickness=2)
        put_text_centered(frame, f"MOSTRAMI: {self.target_it}!", (cx, 102), scale=0.86, color=COLOR_YELLOW, thickness=2)
        pct = max(0.0, self.time_left / self.ROUND_TIME)
        bx1, bx2 = cx - 230, cx + 230
        cv2.rectangle(frame, (bx1, 130), (bx2, 142), (40, 48, 66), -1)
        cv2.rectangle(frame, (bx1, 130), (int(bx1 + (bx2 - bx1) * pct), 142), COLOR_RED if pct < 0.3 else COLOR_GREEN, -1)

        # Segnapunti in alto a destra + pulsante touch-free SALTA OGGETTO
        draw_panel(frame, w - 255, 74, w - 16, 154, border_color=COLOR_GREEN, alpha=0.80)
        cv2.putText(frame, f"PUNTI : {self.score}", (w - 238, 106), cv2.FONT_HERSHEY_DUPLEX, 0.72, COLOR_GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, f"SERIE : x{self.streak}  |  {int(self.time_left):02d}s", (w - 238, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_WHITE, 1, cv2.LINE_AA)

        skip_btn = DwellButton("skip_obj", w - 255, 162, w - 16, 206, "SALTA OGGETTO >", COLOR_ORANGE, label_scale=0.52)
        if self.dwell.update_and_draw(frame, [skip_btn], hands, dt) == "skip_obj":
            self._next_target()

        # Feedback vittoria / tempo scaduto
        if self.found_banner > 0:
            draw_panel(frame, cx - 240, h // 2 - 55, cx + 240, h // 2 + 55, border_color=COLOR_GREEN, alpha=0.88, thickness=3)
            put_text_centered(frame, "TROVATO! +PUNTI!", (cx, h // 2), scale=1.15, color=COLOR_GREEN, thickness=3)
        elif self.timeout_banner > 0:
            draw_panel(frame, cx - 240, h // 2 - 55, cx + 240, h // 2 + 55, border_color=COLOR_RED, alpha=0.88, thickness=3)
            put_text_centered(frame, "TEMPO SCADUTO!", (cx, h // 2), scale=1.15, color=COLOR_RED, thickness=3)

        return "Porta davanti alla webcam l'oggetto richiesto prima che scada il tempo! (Punta + PINCH su SALTA per cambiare)"

    def close(self) -> None:
        self._detector.close()
