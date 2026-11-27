"""Modalità 4: STARK HUD — Interfaccia olografica stile Tony Stark / Iron Man.

Interazioni gestuali:
- 1 Mano + PINCH su una finestra           -> Afferra e SPOSTA la finestra
- 1 Mano + PINCH sull'angolo in basso a dx -> RIDIMENSIONA la finestra
- 2 Mani + PINCH insieme                   -> ESPANDI / RIDUCI (Pinch-to-Zoom olografico!)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
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
    draw_hand_skeleton,
    draw_panel,
    put_text_centered,
)
from app.sound_engine import SoundEngine

COLOR_STARK_BLUE = (255, 200, 30)   # Ciano-azzurro J.A.R.V.I.S.
COLOR_STARK_GOLD = (40, 185, 255)   # Oro/arancio selezione Iron Man


@dataclass
class HoloWindow:
    win_id: str
    title: str
    kind: str  # "arc", "neural", "code", "radar"
    x: float
    y: float
    w: float
    h: float
    accent: Tuple[int, int, int]
    min_w: float = 210.0
    min_h: float = 155.0
    max_w: float = 820.0
    max_h: float = 540.0

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def in_resize_handle(self, px: float, py: float, size: float = 38.0) -> bool:
        return (
            self.x + self.w - size <= px <= self.x + self.w + 14
            and self.y + self.h - size <= py <= self.y + self.h + 14
        )

    def center(self) -> Tuple[float, float]:
        return (self.x + self.w * 0.5, self.y + self.h * 0.5)


class StarkHUDMode:
    """Finestre olografiche manipolabili a mani nude stile Tony Stark."""

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self._initialized_for_size: Optional[Tuple[int, int]] = None
        self.windows: List[HoloWindow] = []
        self.reset()

    def reset(self) -> None:
        self._initialized_for_size = None
        self._active_win: Optional[HoloWindow] = None
        self._interaction: Optional[str] = None  # "drag", "resize", "two_hand_zoom"
        self._drag_offset = (0.0, 0.0)
        self._zoom_start_dist = 1.0
        self._zoom_start_size = (300.0, 220.0)
        self._zoom_center = (0.0, 0.0)
        self._t0 = time.perf_counter()

    def _init_layout(self, w: int, h: int) -> None:
        self._initialized_for_size = (w, h)
        self.windows = [
            HoloWindow("arc", "GALILEI-DI PALO // CORE", "arc", 38, 86, 310, 235, COLOR_STARK_BLUE),
            HoloWindow("neural", "AI NEURAL MATRIX", "neural", w - 358, 86, 320, 235, COLOR_MAGENTA),
            HoloWindow("code", "INFORMATICA // PYTHON", "code", 38, h - 290, 340, 225, COLOR_GREEN),
            HoloWindow("radar", "3D HAND RADAR", "radar", w - 338, h - 290, 300, 225, COLOR_STARK_GOLD),
        ]

    def handle_key(self, key: int) -> None:
        pass

    def _bring_to_front(self, win: HoloWindow) -> None:
        if win in self.windows:
            self.windows.remove(win)
            self.windows.append(win)

    @staticmethod
    def _clamp_window(win: HoloWindow, frame_w: int, frame_h: int) -> None:
        win.w = max(win.min_w, min(win.max_w, win.w))
        win.h = max(win.min_h, min(win.max_h, win.h))
        win.x = max(8.0, min(frame_w - win.w - 8.0, win.x))
        win.y = max(68.0, min(frame_h - win.h - 44.0, win.y))

    @staticmethod
    def _draw_arc_content(frame: np.ndarray, win: HoloWindow, t: float) -> None:
        cx = int(win.x + win.w * 0.34)
        cy = int(win.y + 34 + (win.h - 40) * 0.5)
        r = max(28, int(min(win.w * 0.28, (win.h - 54) * 0.44)))

        # Anelli rotanti del Reattore Arc
        ang1 = (t * 70) % 360
        ang2 = (-t * 110) % 360
        cv2.circle(frame, (cx, cy), r, COLOR_STARK_BLUE, 2, cv2.LINE_AA)
        for seg in range(0, 360, 60):
            cv2.ellipse(frame, (cx, cy), (r - 6, r - 6), 0, ang1 + seg, ang1 + seg + 36, COLOR_CYAN, 4, cv2.LINE_AA)
        for seg in range(0, 360, 90):
            cv2.ellipse(frame, (cx, cy), (int(r * 0.65), int(r * 0.65)), 0, ang2 + seg, ang2 + seg + 52, COLOR_STARK_GOLD, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(8, int(r * 0.3)), COLOR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), max(12, int(r * 0.38)), COLOR_CYAN, 2, cv2.LINE_AA)

        # Barre telemetria a destra
        bx = int(win.x + win.w * 0.62)
        bw = max(40, int(win.w * 0.32))
        metrics = [
            ("CPU", 0.55 + 0.25 * math.sin(t * 2.4), COLOR_CYAN),
            ("NPU", 0.72 + 0.18 * math.cos(t * 1.9), COLOR_GREEN),
            ("PWR", 0.88 + 0.08 * math.sin(t * 3.5), COLOR_STARK_GOLD),
        ]
        for i, (lbl, val, col) in enumerate(metrics):
            my = int(win.y + 56 + i * ((win.h - 76) / 3))
            cv2.putText(frame, f"{lbl} {int(val * 100)}%", (bx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_WHITE, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (bx, my + 6), (bx + bw, my + 16), (35, 45, 65), -1)
            cv2.rectangle(frame, (bx, my + 6), (bx + int(bw * val), my + 16), col, -1)

    @staticmethod
    def _draw_neural_content(frame: np.ndarray, win: HoloWindow, t: float) -> None:
        layers = [3, 5, 4, 2]
        x0, y0 = win.x + 28, win.y + 46
        uw, uh = win.w - 56, win.h - 64
        node_positions: List[List[Tuple[int, int]]] = []

        for l_idx, count in enumerate(layers):
            lx = int(x0 + (l_idx / (len(layers) - 1)) * uw)
            col_pts = []
            for n_idx in range(count):
                ly = int(y0 + ((n_idx + 0.5) / count) * uh)
                col_pts.append((lx, ly))
            node_positions.append(col_pts)

        # Connessioni sinaptiche
        for l_idx in range(len(layers) - 1):
            for i, p1 in enumerate(node_positions[l_idx]):
                for j, p2 in enumerate(node_positions[l_idx + 1]):
                    pulse = 0.5 + 0.5 * math.sin(t * 4.0 - l_idx * 1.1 + i * 0.7 + j * 0.5)
                    col = COLOR_MAGENTA if pulse > 0.72 else (70, 55, 95)
                    cv2.line(frame, p1, p2, col, 2 if pulse > 0.72 else 1, cv2.LINE_AA)

        # Nodi
        for l_idx, col_pts in enumerate(node_positions):
            for n_idx, pt in enumerate(col_pts):
                active = math.sin(t * 3.5 + l_idx + n_idx) > 0.2
                cv2.circle(frame, pt, 7, COLOR_CYAN if active else COLOR_MAGENTA, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, COLOR_WHITE, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_code_content(frame: np.ndarray, win: HoloWindow, t: float) -> None:
        code_lines = [
            "hands = tracker.detect(frame)",
            "for hand in hands:",
            "  d = dist(thumb, index)",
            "  if d < 0.34 * palm:",
            "    window.move(x, y)",
            "  if len(hands) == 2:",
            "    window.scale(dist_2h)",
            "render_stark_hud(frame)",
        ]
        active_line = int(t * 2.5) % len(code_lines)
        scale = max(0.38, min(0.62, win.w / 620.0))
        line_h = max(18, int((win.h - 46) / len(code_lines)))

        for i, line in enumerate(code_lines):
            ly = int(win.y + 52 + i * line_h)
            if ly > win.y + win.h - 10:
                break
            if i == active_line:
                cv2.rectangle(
                    frame,
                    (int(win.x + 10), ly - line_h + 6),
                    (int(win.x + win.w - 10), ly + 5),
                    (28, 60, 42),
                    -1,
                )
            cv2.putText(
                frame,
                line,
                (int(win.x + 16), ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                COLOR_GREEN if i == active_line else (195, 225, 205),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_radar_content(
        frame: np.ndarray,
        win: HoloWindow,
        t: float,
        hands: List[HandData],
        frame_w: int,
        frame_h: int,
    ) -> None:
        cx = int(win.x + win.w * 0.5)
        cy = int(win.y + 34 + (win.h - 40) * 0.5)
        r = max(28, int(min(win.w, win.h - 42) * 0.42))

        for mult in (1.0, 0.66, 0.33):
            cv2.circle(frame, (cx, cy), int(r * mult), (70, 95, 120), 1, cv2.LINE_AA)
        cv2.line(frame, (cx - r, cy), (cx + r, cy), (60, 80, 100), 1)
        cv2.line(frame, (cx, cy - r), (cx, cy + r), (60, 80, 100), 1)

        # Raggio scanner rotante
        ang = t * 2.6
        sx = int(cx + math.cos(ang) * r)
        sy = int(cy + math.sin(ang) * r)
        cv2.line(frame, (cx, cy), (sx, sy), COLOR_STARK_GOLD, 2, cv2.LINE_AA)

        # Blip delle mani rilevate in tempo reale
        for idx, hand in enumerate(hands):
            nx = (hand.palm_center[0] / max(1, frame_w) - 0.5) * 1.9
            ny = (hand.palm_center[1] / max(1, frame_h) - 0.5) * 1.9
            bx = int(cx + max(-0.9, min(0.9, nx)) * r)
            by = int(cy + max(-0.9, min(0.9, ny)) * r)
            col = COLOR_CYAN if idx == 0 else COLOR_MAGENTA
            cv2.circle(frame, (bx, by), 7, col, -1, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 12, col, 1, cv2.LINE_AA)

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]
        t = time.perf_counter() - self._t0

        if self._initialized_for_size != (w, h):
            self._init_layout(w, h)

        # Calcola i punti pinch delle mani attive
        pinch_points: List[Tuple[int, int]] = []
        for hand in hands:
            if hand.is_pinching:
                px = (hand.index_tip[0] + hand.thumb_tip[0]) // 2
                py = (hand.index_tip[1] + hand.thumb_tip[1]) // 2
                pinch_points.append((px, py))

        # --- GESTIONE INTERAZIONI STILE TONY STARK ---
        if len(pinch_points) >= 2:
            # 2 MANI PINCH -> Espandi / Riduci la finestra attiva (o la piu vicina)
            p1, p2 = pinch_points[0], pinch_points[1]
            dist = max(30.0, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)

            if self._interaction != "two_hand_zoom":
                # Scegli finestra sotto il punto medio oppure la piu vicina
                target = None
                for win in reversed(self.windows):
                    if win.contains(mid[0], mid[1]) or win.contains(p1[0], p1[1]) or win.contains(p2[0], p2[1]):
                        target = win
                        break
                if target is None and self.windows:
                    target = min(
                        self.windows,
                        key=lambda wn: math.hypot(wn.center()[0] - mid[0], wn.center()[1] - mid[1]),
                    )
                if target is not None:
                    self._active_win = target
                    self._bring_to_front(target)
                    self._interaction = "two_hand_zoom"
                    self._zoom_start_dist = dist
                    self._zoom_start_size = (target.w, target.h)
                    self._zoom_center = target.center()
                    self.sound.play("hud_grab")

            if self._active_win is not None and self._interaction == "two_hand_zoom":
                scale = dist / max(30.0, self._zoom_start_dist)
                new_w = self._zoom_start_size[0] * scale
                new_h = self._zoom_start_size[1] * scale
                self._active_win.w = new_w
                self._active_win.h = new_h
                self._active_win.x = self._zoom_center[0] - new_w * 0.5
                self._active_win.y = self._zoom_center[1] - new_h * 0.5
                self._clamp_window(self._active_win, w, h)

        elif len(pinch_points) == 1:
            px, py = pinch_points[0]
            if self._interaction in (None, "two_hand_zoom"):
                # Cerca finestra dall'alto verso il basso (z-order)
                self._active_win = None
                self._interaction = None
                for win in reversed(self.windows):
                    if win.in_resize_handle(px, py):
                        self._active_win = win
                        self._bring_to_front(win)
                        self._interaction = "resize"
                        self.sound.play("hud_grab")
                        break
                    elif win.contains(px, py):
                        self._active_win = win
                        self._bring_to_front(win)
                        self._interaction = "drag"
                        self._drag_offset = (px - win.x, py - win.y)
                        self.sound.play("hud_grab")
                        break
            elif self._active_win is not None:
                if self._interaction == "drag":
                    self._active_win.x = px - self._drag_offset[0]
                    self._active_win.y = py - self._drag_offset[1]
                    self._clamp_window(self._active_win, w, h)
                elif self._interaction == "resize":
                    self._active_win.w = px - self._active_win.x + 10
                    self._active_win.h = py - self._active_win.y + 10
                    self._clamp_window(self._active_win, w, h)
        else:
            if self._interaction is not None:
                self.sound.play("hud_release")
            self._interaction = None
            self._active_win = None

        # --- RENDERING FINESTRE OLOGRAFICHE ---
        for win in self.windows:
            is_selected = win is self._active_win
            border_col = COLOR_STARK_GOLD if is_selected else win.accent
            x1, y1 = int(win.x), int(win.y)
            x2, y2 = int(win.x + win.w), int(win.y + win.h)

            draw_panel(
                frame,
                x1,
                y1,
                x2,
                y2,
                bg_color=(12, 22, 34),
                border_color=border_col,
                alpha=0.68 if is_selected else 0.56,
                thickness=3 if is_selected else 2,
            )

            # Barra titolo olografica
            cv2.rectangle(frame, (x1, y1), (x2, y1 + 30), (22, 42, 62), -1)
            cv2.line(frame, (x1, y1 + 30), (x2, y1 + 30), border_col, 1)
            cv2.putText(
                frame,
                win.title,
                (x1 + 12, y1 + 21),
                cv2.FONT_HERSHEY_DUPLEX,
                0.52,
                COLOR_STARK_GOLD if is_selected else COLOR_WHITE,
                1,
                cv2.LINE_AA,
            )

            # Contenuto specifico del widget
            if win.kind == "arc":
                self._draw_arc_content(frame, win, t)
            elif win.kind == "neural":
                self._draw_neural_content(frame, win, t)
            elif win.kind == "code":
                self._draw_code_content(frame, win, t)
            elif win.kind == "radar":
                self._draw_radar_content(frame, win, t, hands, w, h)

            # Maniglia luminosa di resize nell'angolo in basso a destra
            cv2.line(frame, (x2 - 20, y2 - 4), (x2 - 4, y2 - 20), border_col, 2, cv2.LINE_AA)
            cv2.line(frame, (x2 - 12, y2 - 4), (x2 - 4, y2 - 12), border_col, 2, cv2.LINE_AA)

            # Badge dimensione durante resize/zoom
            if is_selected and self._interaction in ("resize", "two_hand_zoom"):
                put_text_centered(
                    frame,
                    f"{int(win.w)} x {int(win.h)} px",
                    ((x1 + x2) // 2, y2 - 16),
                    scale=0.52,
                    color=COLOR_YELLOW,
                    thickness=1,
                )

        # Laser di misura durante il Pinch-to-Zoom a 2 mani!
        if self._interaction == "two_hand_zoom" and len(pinch_points) >= 2:
            p1, p2 = pinch_points[0], pinch_points[1]
            cv2.line(frame, p1, p2, COLOR_STARK_GOLD, 3, cv2.LINE_AA)
            cv2.circle(frame, p1, 14, COLOR_STARK_GOLD, 3, cv2.LINE_AA)
            cv2.circle(frame, p2, 14, COLOR_STARK_GOLD, 3, cv2.LINE_AA)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            put_text_centered(frame, "ZOOM OLOGRAFICO", (mid[0], mid[1] - 20), scale=0.62, color=COLOR_YELLOW, thickness=2)

        # Scheletri mano stile J.A.R.V.I.S.
        for idx, hand in enumerate(hands):
            accent = COLOR_STARK_GOLD if hand.is_pinching else COLOR_STARK_BLUE
            draw_hand_skeleton(frame, hand, accent=accent, show_ids=debug_mode)
            px = (hand.index_tip[0] + hand.thumb_tip[0]) // 2
            py = (hand.index_tip[1] + hand.thumb_tip[1]) // 2
            cv2.circle(frame, (px, py), 13 if hand.is_pinching else 7, accent, 2, cv2.LINE_AA)

        return "1 Mano PINCH = Sposta / Ridimensiona angolo  |  2 Mani PINCH = Espandi/Riduci (Iron Man!)"
