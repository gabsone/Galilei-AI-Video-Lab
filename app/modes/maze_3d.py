"""Modalità 12: LABIRINTO 3D — Motore Raycasting 3D (stile Wolfenstein / Tron) controllato con la mano!"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import cv2
import numpy as np

from app.dwell_ui import DwellButton, DwellController
from app.hand_tracker import HandData
from app.hud import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_panel,
    put_text_centered,
)
from app.sound_engine import SoundEngine

# 0 = corridoio, 1 = muro ciano, 2 = muro magenta, 9 = portale d'uscita verde!
MAZE_MAPS: List[List[str]] = [
    [
        "1111111111",
        "1000100001",
        "1010102201",
        "1010000201",
        "1022210101",
        "1000010001",
        "1110111201",
        "1000000291",
        "1111111111",
    ],
    [
        "1111111111",
        "1000001001",
        "1022101021",
        "1000100001",
        "1110222101",
        "1000000101",
        "1011101101",
        "1000100091",
        "1111111111",
    ],
    [
        "1111111111",
        "1001000001",
        "1001022101",
        "1200000101",
        "1222110201",
        "1000010201",
        "1012010001",
        "1000000191",
        "1111111111",
    ],
]


class Maze3DMode:
    """Vero labirinto 3D Raycasting: mano aperta = avanti, Sx/Dx = ruota, pugno = stop!"""

    FOV = math.pi / 3.0   # 60 gradi
    NUM_RAYS = 160        # 160 colonne verticali: nitido e velocissimo (60 FPS!)

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self.dwell = DwellController(sound)
        self.level_idx = 0
        self.best_time: float | None = None
        self.reset()

    def reset(self) -> None:
        self.grid = [[int(c) for c in row] for row in MAZE_MAPS[self.level_idx]]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.px = 1.5
        self.py = 1.5
        self.angle = 0.0
        self.elapsed = 0.0
        self.won = False
        self.moving = False

    def _next_level(self) -> None:
        self.level_idx = (self.level_idx + 1) % len(MAZE_MAPS)
        self.reset()
        self.sound.play("go")

    def handle_key(self, key: int) -> None:
        if key == ord(" ") and self.won:
            self._next_level()

    def _is_wall(self, x: float, y: float) -> int:
        gx, gy = int(x), int(y)
        if gx < 0 or gy < 0 or gy >= self.rows or gx >= self.cols:
            return 1
        return self.grid[gy][gx]

    def _cast_ray(self, ray_angle: float) -> Tuple[float, int, bool]:
        """Algoritmo DDA: restituisce (distanza_corretta, tipo_muro, lato_verticale)."""
        sin_a = math.sin(ray_angle)
        cos_a = math.cos(ray_angle)
        if abs(cos_a) < 1e-6:
            cos_a = 1e-6
        if abs(sin_a) < 1e-6:
            sin_a = 1e-6

        map_x, map_y = int(self.px), int(self.py)
        delta_dist_x = abs(1.0 / cos_a)
        delta_dist_y = abs(1.0 / sin_a)

        if cos_a < 0:
            step_x = -1
            side_dist_x = (self.px - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - self.px) * delta_dist_x

        if sin_a < 0:
            step_y = -1
            side_dist_y = (self.py - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - self.py) * delta_dist_y

        side = False
        cell = 1
        for _ in range(32):
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = False
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = True

            if map_x < 0 or map_y < 0 or map_y >= self.rows or map_x >= self.cols:
                cell = 1
                break
            cell = self.grid[map_y][map_x]
            if cell != 0:
                break

        raw_dist = (side_dist_x - delta_dist_x) if not side else (side_dist_y - delta_dist_y)
        corrected = max(0.08, raw_dist * math.cos(ray_angle - self.angle))
        return corrected, cell, side

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]
        cam_thumb = cv2.resize(frame, (220, 124), interpolation=cv2.INTER_AREA)

        # --- CONTROLLI A MANO ---
        self.moving = False
        turn_rate = 0.0
        if hands and not self.won:
            hand = hands[0]
            # Posizione orizzontale della mano -> rotazione fluida (con deadzone al centro)
            norm_x = (hand.palm_center[0] / max(1, w)) - 0.5
            if abs(norm_x) > 0.09:
                turn_rate = (norm_x - math.copysign(0.09, norm_x)) * 3.4
                self.angle += turn_rate * dt

            # Mano aperta (CARTA o >= 3 dita su) -> avanza nei corridoi!
            if sum(hand.fingers_up) >= 3 and not hand.is_pinching:
                self.moving = True
                speed = 2.05 * dt
                nx = self.px + math.cos(self.angle) * speed
                ny = self.py + math.sin(self.angle) * speed
                margin = 0.24
                cell_x = self._is_wall(nx + math.copysign(margin, math.cos(self.angle)), self.py)
                cell_y = self._is_wall(self.px, ny + math.copysign(margin, math.sin(self.angle)))

                if cell_x == 9 or cell_y == 9:
                    self.won = True
                    if self.best_time is None or self.elapsed < self.best_time:
                        self.best_time = self.elapsed
                    self.sound.play("win")
                else:
                    if cell_x == 0:
                        self.px = nx
                    if cell_y == 0:
                        self.py = ny

        if not self.won:
            self.elapsed += dt

        # --- RENDERING 3D RAYCASTING ---
        top_y, bot_y = 66, h - 38
        view_h = bot_y - top_y
        mid_y = top_y + view_h // 2

        # Soffitto e pavimento cyber scuri
        frame[top_y:mid_y, :] = (20, 14, 30)
        frame[mid_y:bot_y, :] = (24, 28, 40)

        col_w = w / self.NUM_RAYS
        pulse = 0.78 + 0.22 * math.sin(time.perf_counter() * 6.0)

        for i in range(self.NUM_RAYS):
            ray_angle = self.angle - self.FOV * 0.5 + (i / self.NUM_RAYS) * self.FOV
            dist, cell, side = self._cast_ray(ray_angle)

            wall_h = min(view_h, int((view_h * 0.88) / dist))
            y1 = max(top_y, mid_y - wall_h // 2)
            y2 = min(bot_y, mid_y + wall_h // 2)
            x1 = int(i * col_w)
            x2 = max(x1 + 1, int((i + 1) * col_w) + 1)

            shade = max(0.18, min(1.0, 1.0 / (1.0 + dist * dist * 0.14)))
            if side:
                shade *= 0.72

            if cell == 9:
                # Portale d'uscita verde neon pulsante!
                base = (80, 255, 110)
                shade = min(1.0, shade * 1.25 * pulse)
            elif cell == 2:
                base = (220, 70, 240)  # Pareti magenta
            else:
                base = (240, 200, 40)  # Pareti ciano/blu

            col = (int(base[0] * shade), int(base[1] * shade), int(base[2] * shade))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            # Bordo neon in cima e alla base della parete
            if wall_h > 26:
                edge = (min(255, col[0] + 45), min(255, col[1] + 45), min(255, col[2] + 45))
                cv2.line(frame, (x1, y1), (x2, y1), edge, 2)
                cv2.line(frame, (x1, y2), (x2, y2), edge, 2)

        # Mirino centrale discreto
        cx, cy = w // 2, mid_y
        cv2.circle(frame, (cx, cy), 8, COLOR_GREEN if self.moving else COLOR_CYAN, 2, cv2.LINE_AA)

        # --- MINIMAPPA RADAR 2D (in alto a sinistra) ---
        cell_px = 16
        mx0, my0 = 18, 78
        mw, mh = self.cols * cell_px, self.rows * cell_px
        draw_panel(frame, mx0 - 6, my0 - 6, mx0 + mw + 6, my0 + mh + 6, border_color=COLOR_CYAN, alpha=0.82)
        for r in range(self.rows):
            for c in range(self.cols):
                v = self.grid[r][c]
                if v == 0:
                    continue
                col = COLOR_GREEN if v == 9 else (COLOR_MAGENTA if v == 2 else COLOR_CYAN)
                cv2.rectangle(
                    frame,
                    (mx0 + c * cell_px + 1, my0 + r * cell_px + 1),
                    (mx0 + (c + 1) * cell_px - 1, my0 + (r + 1) * cell_px - 1),
                    col,
                    -1,
                )
        # Posizione giocatore + direzione sguardo sulla minimappa
        pmx = int(mx0 + self.px * cell_px)
        pmy = int(my0 + self.py * cell_px)
        cv2.circle(frame, (pmx, pmy), 4, COLOR_YELLOW, -1, cv2.LINE_AA)
        cv2.line(
            frame,
            (pmx, pmy),
            (int(pmx + math.cos(self.angle) * 14), int(pmy + math.sin(self.angle) * 14)),
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        # Riquadro PiP webcam in basso a sinistra per vedere la propria mano!
        px1, py1 = 18, bot_y - 134
        frame[py1 : py1 + 124, px1 : px1 + 220] = cam_thumb
        cv2.rectangle(frame, (px1, py1), (px1 + 220, py1 + 124), COLOR_CYAN, 2)

        # HUD Cronometro e stato movimento (in alto a destra)
        draw_panel(frame, w - 255, 76, w - 16, 166, border_color=COLOR_GREEN if self.moving else COLOR_CYAN, alpha=0.80)
        cv2.putText(frame, f"LIVELLO {self.level_idx + 1}/3", (w - 238, 104), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_YELLOW, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TEMPO: {self.elapsed:05.1f}s", (w - 238, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_WHITE, 2, cv2.LINE_AA)
        st = "AVANTI!" if self.moving else "FERMO (Apri mano)"
        cv2.putText(frame, st, (w - 238, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_GREEN if self.moving else COLOR_ORANGE, 1, cv2.LINE_AA)

        # Vittoria -> pulsante touch-free NUOVO LABIRINTO!
        if self.won:
            draw_panel(frame, cx - 275, cy - 120, cx + 275, cy + 125, border_color=COLOR_GREEN, alpha=0.90, thickness=3)
            put_text_centered(frame, "LABIRINTO COMPLETATO!", (cx, cy - 68), scale=1.08, color=COLOR_GREEN, thickness=3)
            best_txt = f"  |  RECORD: {self.best_time:.1f}s" if self.best_time is not None else ""
            put_text_centered(frame, f"TEMPO: {self.elapsed:.1f}s{best_txt}", (cx, cy - 12), scale=0.72, color=COLOR_WHITE, thickness=2)
            btn = DwellButton("next_maze", cx - 175, cy + 32, cx + 175, cy + 98, "PROSSIMO LABIRINTO", COLOR_YELLOW, active=True, label_scale=0.68)
            if self.dwell.update_and_draw(frame, [btn], hands, dt) == "next_maze":
                self._next_level()

        return "MANO APERTA = Cammina avanti  |  Sposta mano Sx/Dx = Ruota  |  PUGNO = Fermo  |  Trova il PORTALE VERDE!"
