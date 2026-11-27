"""Modalità 9: AIR GRAFFITI — Lavagna Neon in aria (disegna con l'indice, cancella col palmo!)."""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
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
    draw_hand_skeleton,
    put_text_centered,
)
from app.sound_engine import SoundEngine

NEON_COLORS: List[Tuple[str, str, Tuple[int, int, int]]] = [
    ("c_cyan", "CIANO", COLOR_CYAN),
    ("c_mag", "FUCSIA", COLOR_MAGENTA),
    ("c_green", "VERDE", COLOR_GREEN),
    ("c_yel", "GIALLO", COLOR_YELLOW),
    ("c_org", "ARANCIO", COLOR_ORANGE),
    ("c_wht", "BIANCO", COLOR_WHITE),
]


class AirGraffitiMode:
    """Disegna graffiti luminosi nell'aria con l'indice (fino a 2 mani!). Mano aperta = gomma."""

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self.dwell = DwellController(sound)
        self.color_idx = 0
        self._canvas: Optional[np.ndarray] = None
        self._prev_pt: Dict[int, Optional[Tuple[int, int]]] = {0: None, 1: None}
        self._saved_banner = 0.0
        self.reset()

    def reset(self) -> None:
        self._canvas = None
        self._prev_pt = {0: None, 1: None}
        self._saved_banner = 0.0

    def handle_key(self, key: int) -> None:
        if key in (ord("c"), ord("C")):
            self._canvas = None
            self.sound.play("switch")

    def _save_drawing(self, frame: np.ndarray) -> None:
        out_dir = Path.cwd() / "foto_orientamento"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"graffiti_galilei_{stamp}.png"
        cv2.imwrite(str(out_path), frame)
        self._saved_banner = 2.0
        self.sound.play("win")

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]
        if self._canvas is None or self._canvas.shape[:2] != (h, w):
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)

        active_color = NEON_COLORS[self.color_idx][2]

        # Tavolozza colori + PULISCI + SALVA in alto (attivabili con Punta + PINCH)
        buttons: List[DwellButton] = []
        bx = 24
        by1, by2 = 74, 118
        for idx, (cid, lbl, col) in enumerate(NEON_COLORS):
            buttons.append(
                DwellButton(
                    btn_id=cid,
                    x1=bx,
                    y1=by1,
                    x2=bx + 96,
                    y2=by2,
                    label=lbl,
                    color=col,
                    active=(idx == self.color_idx),
                    label_scale=0.45,
                )
            )
            bx += 104

        buttons.append(DwellButton("clear", w - 280, by1, w - 155, by2, "PULISCI", COLOR_RED, label_scale=0.50))
        buttons.append(DwellButton("save", w - 145, by1, w - 20, by2, "SALVA", COLOR_GREEN, label_scale=0.50))

        # Disegno o gomma per ciascuna mano (solo sotto la barra strumenti y > 124)
        for h_idx in range(2):
            if h_idx >= len(hands):
                self._prev_pt[h_idx] = None
                continue

            hand = hands[h_idx]
            ix, iy = hand.index_tip

            # Mano completamente aperta (4+ dita su) -> GOMMA!
            if hand.gesture == "CARTA" and iy > 124:
                cv2.circle(self._canvas, hand.palm_center, 46, (0, 0, 0), -1)
                cv2.circle(frame, hand.palm_center, 46, COLOR_WHITE, 2, cv2.LINE_AA)
                put_text_centered(frame, "GOMMA", hand.palm_center, scale=0.48, color=COLOR_YELLOW, thickness=1)
                self._prev_pt[h_idx] = None
            # Solo indice alzato (non pinch) -> DISEGNA NEON!
            elif hand.fingers_up[1] == 1 and hand.fingers_up[2] == 0 and not hand.is_pinching and iy > 124:
                prev = self._prev_pt[h_idx]
                if prev is not None and math.hypot(ix - prev[0], iy - prev[1]) < 160:
                    cv2.line(self._canvas, prev, (ix, iy), active_color, 13, cv2.LINE_AA)
                    cv2.line(self._canvas, prev, (ix, iy), COLOR_WHITE, 4, cv2.LINE_AA)
                self._prev_pt[h_idx] = (ix, iy)
                cv2.circle(frame, (ix, iy), 11, active_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), 13, COLOR_WHITE, 2, cv2.LINE_AA)
            else:
                self._prev_pt[h_idx] = None

            if debug_mode:
                draw_hand_skeleton(frame, hand, accent=active_color, show_ids=True)

        # Sovrapponi il canvas neon al video (effetto glow) solo se c'e almeno un tratto
        gray = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        mask = gray > 5
        if np.any(mask):
            frame[mask] = cv2.addWeighted(frame[mask], 0.25, self._canvas[mask], 0.95, 0)

        # Gestisci pulsanti tavolozza
        hit = self.dwell.update_and_draw(frame, buttons, hands, dt)
        if hit == "clear":
            self._canvas[:] = 0
            self.sound.play("go")
        elif hit == "save":
            self._save_drawing(frame)
        elif hit is not None:
            for idx, (cid, _, _) in enumerate(NEON_COLORS):
                if hit == cid:
                    self.color_idx = idx
                    break

        if self._saved_banner > 0:
            self._saved_banner = max(0.0, self._saved_banner - dt)
            put_text_centered(frame, "DISEGNO SALVATO IN foto_orientamento/ !", (w // 2, h - 75), scale=0.72, color=COLOR_GREEN, thickness=2)

        return "INDICE alzato = Disegna Neon (anche in 2!)  |  MANO APERTA = Gomma  |  PINCH in alto = Colore / Pulisci"
