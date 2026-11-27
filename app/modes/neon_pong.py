"""Modalità 7: NEON PONG — Sfida arcade 1 vs 1 (due ragazzi affiancati) o 1 vs AI!"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from app.dwell_ui import DwellButton, DwellController
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


@dataclass
class PongSpark:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float
    max_life: float


class NeonPongMode:
    """Pong stile Tron a mani nude: 2 mani ai lati = 1 vs 1, 1 mano = 1 vs AI!"""

    WIN_SCORE = 5

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self.dwell = DwellController(sound)
        self.reset()

    def reset(self) -> None:
        self.score_l = 0
        self.score_r = 0
        self.winner: str = ""
        self.paddle_l_y = 0.5
        self.paddle_r_y = 0.5
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_vx = 0.44 * random.choice([-1, 1])
        self.ball_vy = random.uniform(-0.24, 0.24)
        self.serve_pause = 0.8
        self.trail: List[Tuple[int, int]] = []
        self.sparks: List[PongSpark] = []
        self.is_two_player = False

    def _serve(self, direction: int) -> None:
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_vx = 0.44 * direction
        self.ball_vy = random.uniform(-0.26, 0.26)
        self.serve_pause = 0.75
        self.trail.clear()

    def handle_key(self, key: int) -> None:
        if key == ord(" ") and self.winner:
            self.reset()
            self.sound.play("go")

    def _emit_sparks(self, x: float, y: float, color: Tuple[int, int, int], count: int = 16) -> None:
        for _ in range(count):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(110, 420)
            life = random.uniform(0.25, 0.55)
            self.sparks.append(PongSpark(x, y, math.cos(ang) * spd, math.sin(ang) * spd, color, life, life))

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]
        top_y, bot_y = 76, h - 46
        play_h = max(100, bot_y - top_y)
        pad_h = int(play_h * 0.24)
        pad_w = 18
        lx = 46
        rx = w - 46 - pad_w

        # Leggera patina scura per far brillare il campo Tron
        overlay = np.full_like(frame, (8, 12, 22))
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        # Assegna le mani alle racchette sinistra e destra
        left_hands = [hd for hd in hands if hd.palm_center[0] < w * 0.5]
        right_hands = [hd for hd in hands if hd.palm_center[0] >= w * 0.5]

        if left_hands:
            target = (left_hands[0].palm_center[1] - top_y) / play_h
            self.paddle_l_y += (max(0.12, min(0.88, target)) - self.paddle_l_y) * min(1.0, dt * 18.0)

        if right_hands:
            self.is_two_player = True
            target = (right_hands[0].palm_center[1] - top_y) / play_h
            self.paddle_r_y += (max(0.12, min(0.88, target)) - self.paddle_r_y) * min(1.0, dt * 18.0)
        else:
            self.is_two_player = False
            # AI fluida (battibile ma divertente!)
            ai_speed = 0.56 if self.ball_vx > 0 else 0.28
            diff = self.ball_y - self.paddle_r_y
            self.paddle_r_y += max(-ai_speed * dt, min(ai_speed * dt, diff))
            self.paddle_r_y = max(0.12, min(0.88, self.paddle_r_y))

        # Fisica pallina
        if not self.winner:
            if self.serve_pause > 0:
                self.serve_pause = max(0.0, self.serve_pause - dt)
            else:
                self.ball_x += self.ball_vx * dt
                self.ball_y += self.ball_vy * dt

                # Rimbalzo bordi sopra/sotto
                if self.ball_y <= 0.03:
                    self.ball_y = 0.03
                    self.ball_vy = abs(self.ball_vy)
                    self.sound.play("beep")
                elif self.ball_y >= 0.97:
                    self.ball_y = 0.97
                    self.ball_vy = -abs(self.ball_vy)
                    self.sound.play("beep")

                bx_px = int(self.ball_x * w)
                by_px = int(top_y + self.ball_y * play_h)
                ly_px = int(top_y + self.paddle_l_y * play_h)
                ry_px = int(top_y + self.paddle_r_y * play_h)

                # Collisione racchetta sinistra
                if self.ball_vx < 0 and lx <= bx_px <= lx + pad_w + 18 and abs(by_px - ly_px) <= pad_h * 0.5 + 18:
                    self.ball_vx = min(1.05, abs(self.ball_vx) * 1.07)
                    offset = (by_px - ly_px) / max(1.0, pad_h * 0.5)
                    self.ball_vy = max(-0.75, min(0.75, offset * 0.55))
                    self._emit_sparks(lx + pad_w, by_px, COLOR_CYAN, 16)
                    self.sound.play("slice")

                # Collisione racchetta destra
                if self.ball_vx > 0 and rx - 18 <= bx_px <= rx + pad_w and abs(by_px - ry_px) <= pad_h * 0.5 + 18:
                    self.ball_vx = -min(1.05, abs(self.ball_vx) * 1.07)
                    offset = (by_px - ry_px) / max(1.0, pad_h * 0.5)
                    self.ball_vy = max(-0.75, min(0.75, offset * 0.55))
                    self._emit_sparks(rx, by_px, COLOR_MAGENTA, 16)
                    self.sound.play("slice")

                # Punto segnato
                if self.ball_x < 0.0:
                    self.score_r += 1
                    self._emit_sparks(30, by_px, COLOR_MAGENTA, 24)
                    self.sound.play("combo")
                    if self.score_r >= self.WIN_SCORE:
                        self.winner = "VINCE DESTRA!" if self.is_two_player else "VINCE L'AI!"
                        self.sound.play("win")
                    else:
                        self._serve(-1)
                elif self.ball_x > 1.0:
                    self.score_l += 1
                    self._emit_sparks(w - 30, by_px, COLOR_CYAN, 24)
                    self.sound.play("combo")
                    if self.score_l >= self.WIN_SCORE:
                        self.winner = "VINCE SINISTRA!" if self.is_two_player else "HAI VINTO TU!"
                        self.sound.play("win")
                    else:
                        self._serve(1)

        # Linea tratteggiata centrale del campo
        cx = w // 2
        for y in range(top_y + 10, bot_y - 10, 28):
            cv2.line(frame, (cx, y), (cx, min(bot_y - 10, y + 14)), (65, 85, 115), 2, cv2.LINE_AA)

        # Segnapunti
        mode_lbl = "2 GIOCATORI (1 vs 1)" if self.is_two_player else "1 GIOCATORE vs AI"
        draw_panel(frame, cx - 180, 76, cx + 180, 144, border_color=COLOR_CYAN, alpha=0.78)
        put_text_centered(frame, f"{self.score_l}   -   {self.score_r}", (cx, 104), scale=1.2, color=COLOR_WHITE, thickness=3)
        put_text_centered(frame, mode_lbl, (cx, 132), scale=0.45, color=COLOR_YELLOW, thickness=1, shadow=False)

        # Racchette laser
        ly_px = int(top_y + self.paddle_l_y * play_h)
        ry_px = int(top_y + self.paddle_r_y * play_h)
        cv2.rectangle(frame, (lx - 3, ly_px - pad_h // 2 - 3), (lx + pad_w + 3, ly_px + pad_h // 2 + 3), COLOR_CYAN, 2)
        cv2.rectangle(frame, (lx, ly_px - pad_h // 2), (lx + pad_w, ly_px + pad_h // 2), COLOR_WHITE, -1)
        cv2.rectangle(frame, (rx - 3, ry_px - pad_h // 2 - 3), (rx + pad_w + 3, ry_px + pad_h // 2 + 3), COLOR_MAGENTA, 2)
        cv2.rectangle(frame, (rx, ry_px - pad_h // 2), (rx + pad_w, ry_px + pad_h // 2), COLOR_WHITE, -1)

        # Scia + pallina neon
        bx_px = int(self.ball_x * w)
        by_px = int(top_y + self.ball_y * play_h)
        self.trail.append((bx_px, by_px))
        if len(self.trail) > 12:
            self.trail.pop(0)
        for i in range(1, len(self.trail)):
            r = max(3, int(14 * (i / len(self.trail))))
            cv2.circle(frame, self.trail[i], r, COLOR_YELLOW, -1, cv2.LINE_AA)
        cv2.circle(frame, (bx_px, by_px), 16, COLOR_ORANGE, -1, cv2.LINE_AA)
        cv2.circle(frame, (bx_px, by_px), 10, COLOR_WHITE, -1, cv2.LINE_AA)

        # Particelle
        alive: List[PongSpark] = []
        for s in self.sparks:
            s.life -= dt
            if s.life > 0:
                s.x += s.vx * dt
                s.y += s.vy * dt
                cv2.circle(frame, (int(s.x), int(s.y)), max(2, int(6 * s.life / s.max_life)), s.color, -1, cv2.LINE_AA)
                alive.append(s)
        self.sparks = alive

        # Schermata vittoria con pulsante touch-free RIGIOCA
        if self.winner:
            cy = h // 2
            draw_panel(frame, cx - 260, cy - 115, cx + 260, cy + 125, border_color=COLOR_YELLOW, alpha=0.88, thickness=3)
            put_text_centered(frame, self.winner, (cx, cy - 55), scale=1.25, color=COLOR_YELLOW, thickness=3)
            put_text_centered(frame, f"Finale: {self.score_l} - {self.score_r}", (cx, cy - 5), scale=0.75, color=COLOR_WHITE, thickness=2)
            btn = DwellButton("pong_replay", cx - 150, cy + 36, cx + 150, cy + 102, "RIGIOCA!", COLOR_GREEN, active=True, label_scale=0.8)
            if self.dwell.update_and_draw(frame, [btn], hands, dt) == "pong_replay":
                self.reset()
                self.sound.play("go")

        return "Alza e abbassa la mano per muovere la racchetta! Con 2 ragazzi affiancati diventa 1 vs 1!"
