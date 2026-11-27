"""Modalità 2: SASSO · CARTA · FORBICI vs AI (con finestra di 1s su 'VIA!' per massima fluidità)."""

from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple

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
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_hand_skeleton,
    draw_panel,
    put_text_centered,
)
from app.sound_engine import SoundEngine

VALID_MOVES = ("SASSO", "CARTA", "FORBICI")
BEATS = {
    "SASSO": "FORBICI",
    "FORBICI": "CARTA",
    "CARTA": "SASSO",
}
LOSES_TO = {v: k for k, v in BEATS.items()}


class RPSGameMode:
    """Sfida Sasso-Carta-Forbici in tempo reale contro l'AI."""

    THROW_WINDOW_SEC = 1.0  # 1 secondo dopo il 3-2-1 per mostrare la mano comodamente

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self.dwell = DwellController(sound)
        self.unbeatable_mode = False  # Tasto [C]: sorpresa per far ridere la classe!
        self.reset()

    def reset(self) -> None:
        self.player_score = 0
        self.ai_score = 0
        self.ties = 0
        self.state = "IDLE"  # IDLE -> COUNTDOWN -> THROW -> RESULT
        self.state_start = time.perf_counter()
        self.last_beep_sec = -1
        self.player_move: Optional[str] = None
        self.ai_move: Optional[str] = None
        self._last_seen_valid_move: Optional[str] = None
        self.outcome_text = ""
        self.outcome_color = COLOR_WHITE
        self._thumb_hold = 0.0

    def start_round(self) -> None:
        self.state = "COUNTDOWN"
        self.state_start = time.perf_counter()
        self.last_beep_sec = 4
        self.player_move = None
        self.ai_move = None
        self._last_seen_valid_move = None
        self._thumb_hold = 0.0

    def handle_key(self, key: int) -> None:
        if key == ord(" "):
            self.start_round()
        elif key in (ord("c"), ord("C")):
            self.unbeatable_mode = not self.unbeatable_mode
            self.sound.play("switch")

    @staticmethod
    def _draw_move_icon(
        frame: np.ndarray,
        move: Optional[str],
        center: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Disegna un'icona vettoriale pulita per Sasso, Carta o Forbici."""
        cx, cy = center
        if move == "SASSO":
            cv2.circle(frame, (cx, cy), 46, color, 4, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 34, color, -1, cv2.LINE_AA)
            put_text_centered(frame, "SASSO", (cx, cy + 75), scale=0.85, color=color, thickness=2)
        elif move == "CARTA":
            cv2.rectangle(frame, (cx - 38, cy - 48), (cx + 38, cy + 48), color, 4, cv2.LINE_AA)
            for dy in (-22, 0, 22):
                cv2.line(frame, (cx - 22, cy + dy), (cx + 22, cy + dy), color, 3, cv2.LINE_AA)
            put_text_centered(frame, "CARTA", (cx, cy + 75), scale=0.85, color=color, thickness=2)
        elif move == "FORBICI":
            cv2.line(frame, (cx - 34, cy - 42), (cx + 28, cy + 28), color, 5, cv2.LINE_AA)
            cv2.line(frame, (cx + 34, cy - 42), (cx - 28, cy + 28), color, 5, cv2.LINE_AA)
            cv2.circle(frame, (cx - 24, cy + 34), 14, color, 4, cv2.LINE_AA)
            cv2.circle(frame, (cx + 24, cy + 34), 14, color, 4, cv2.LINE_AA)
            put_text_centered(frame, "FORBICI", (cx, cy + 75), scale=0.85, color=color, thickness=2)
        else:
            put_text_centered(frame, "?", (cx, cy), scale=2.2, color=COLOR_MUTED, thickness=4)

    def _resolve_round(self, now: float) -> None:
        self.player_move = self._last_seen_valid_move
        if self.player_move is None:
            self.ai_move = random.choice(VALID_MOVES)
            self.outcome_text = "MOSSA NON RILEVATA!"
            self.outcome_color = COLOR_ORANGE
        else:
            if self.unbeatable_mode:
                self.ai_move = LOSES_TO[self.player_move]
            else:
                self.ai_move = random.choice(VALID_MOVES)

            if self.player_move == self.ai_move:
                self.ties += 1
                self.outcome_text = "PAREGGIO!"
                self.outcome_color = COLOR_YELLOW
                self.sound.play("tie")
            elif BEATS[self.player_move] == self.ai_move:
                self.player_score += 1
                self.outcome_text = "HAI VINTO!"
                self.outcome_color = COLOR_GREEN
                self.sound.play("win")
            else:
                self.ai_score += 1
                self.outcome_text = "VINCE L'AI!"
                self.outcome_color = COLOR_RED
                self.sound.play("lose")

        self.state = "RESULT"
        self.state_start = now

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]
        now = time.perf_counter()

        live_gesture = hands[0].gesture if hands else "NESSUNA MANO"
        live_valid_move = live_gesture if live_gesture in VALID_MOVES else None

        for hand in hands:
            draw_hand_skeleton(frame, hand, accent=COLOR_MAGENTA, show_ids=debug_mode)

        # Avvio hands-free con POLLICE SU tenuto per 0.7s
        if self.state in ("IDLE", "RESULT"):
            if live_gesture == "POLLICE SU":
                self._thumb_hold += dt
                if self._thumb_hold >= 0.7:
                    self.start_round()
            else:
                self._thumb_hold = max(0.0, self._thumb_hold - dt * 2)

        # Gestione stati: COUNTDOWN (3s) -> THROW (1s "VIA!") -> RESULT
        if self.state == "COUNTDOWN":
            elapsed = now - self.state_start
            remaining = 3.0 - elapsed
            sec = int(remaining) + 1
            if 1 <= sec <= 3 and sec < self.last_beep_sec:
                self.last_beep_sec = sec
                self.sound.play("beep")

            if elapsed >= 3.0:
                self.sound.play("go")
                self.state = "THROW"
                self.state_start = now
                self._last_seen_valid_move = live_valid_move

        elif self.state == "THROW":
            # Aggiorna la mossa rilevata durante il secondo di grazia su "VIA!"
            if live_valid_move is not None:
                self._last_seen_valid_move = live_valid_move
            if now - self.state_start >= self.THROW_WINDOW_SEC:
                self._resolve_round(now)

        # Pannello segnapunti centrale in alto
        cx = w // 2
        draw_panel(frame, cx - 230, 74, cx + 230, 140, border_color=COLOR_MAGENTA, alpha=0.8)
        put_text_centered(
            frame,
            f"TU  {self.player_score}   -   {self.ai_score}  AI",
            (cx, 102),
            scale=1.0,
            color=COLOR_WHITE,
            thickness=2,
        )
        mode_tag = "AI IMBROGLIONA ON" if self.unbeatable_mode else f"Pareggi: {self.ties}"
        put_text_centered(
            frame,
            mode_tag,
            (cx, 128),
            scale=0.48,
            color=COLOR_RED if self.unbeatable_mode else COLOR_MUTED,
            thickness=1,
            shadow=False,
        )

        # Due card: sinistra (TU), destra (AI)
        card_w, card_h = 250, 260
        y1 = max(165, h // 2 - card_h // 2)
        y2 = y1 + card_h

        # Card Giocatore (sinistra)
        lx1, lx2 = 34, 34 + card_w
        draw_panel(frame, lx1, y1, lx2, y2, border_color=COLOR_CYAN, alpha=0.74)
        put_text_centered(frame, "LA TUA MANO", ((lx1 + lx2) // 2, y1 + 30), scale=0.68, color=COLOR_CYAN, thickness=2)
        shown_player = self.player_move if self.state == "RESULT" else live_valid_move
        self._draw_move_icon(frame, shown_player, ((lx1 + lx2) // 2, y1 + 135), COLOR_CYAN)

        # Card AI (destra)
        rx1, rx2 = w - 34 - card_w, w - 34
        draw_panel(frame, rx1, y1, rx2, y2, border_color=COLOR_MAGENTA, alpha=0.74)
        put_text_centered(frame, "MOSSA AI", ((rx1 + rx2) // 2, y1 + 30), scale=0.68, color=COLOR_MAGENTA, thickness=2)
        shown_ai = self.ai_move if self.state == "RESULT" else None
        self._draw_move_icon(frame, shown_ai, ((rx1 + rx2) // 2, y1 + 135), COLOR_MAGENTA)

        # Centro schermo: countdown, finestra VIA! (1s), risultato o invito
        cy = h // 2
        if self.state == "COUNTDOWN":
            sec = max(1, int(3.0 - (now - self.state_start)) + 1)
            cv2.circle(frame, (cx, cy), 82, (20, 22, 36), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 82, COLOR_YELLOW, 5, cv2.LINE_AA)
            put_text_centered(frame, str(sec), (cx, cy), scale=3.2, color=COLOR_YELLOW, thickness=6)
            put_text_centered(frame, "PREPARA LA MANO...", (cx, cy + 120), scale=0.78, color=COLOR_WHITE, thickness=2)
        elif self.state == "THROW":
            pct = max(0.0, 1.0 - (now - self.state_start) / self.THROW_WINDOW_SEC)
            cv2.circle(frame, (cx, cy), 92, (16, 36, 28), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 92, COLOR_GREEN, 6, cv2.LINE_AA)
            put_text_centered(frame, "VIA!", (cx, cy), scale=2.1, color=COLOR_GREEN, thickness=5)
            put_text_centered(frame, "MOSTRA LA MOSSA!", (cx, cy + 125), scale=0.82, color=COLOR_YELLOW, thickness=2)
            # Barra del secondo di grazia
            bx1, bx2 = cx - 120, cx + 120
            by = cy + 150
            cv2.rectangle(frame, (bx1, by), (bx2, by + 10), (45, 55, 70), -1)
            cv2.rectangle(frame, (bx1, by), (int(bx1 + (bx2 - bx1) * pct), by + 10), COLOR_GREEN, -1)
        elif self.state == "RESULT":
            draw_panel(frame, cx - 215, cy + 35, cx + 215, cy + 115, border_color=self.outcome_color, alpha=0.86, thickness=3)
            put_text_centered(frame, self.outcome_text, (cx, cy + 75), scale=1.05, color=self.outcome_color, thickness=3)
            play_btn = DwellButton(
                btn_id="play_rps",
                x1=cx - 175,
                y1=h - 122,
                x2=cx + 175,
                y2=h - 54,
                label="GIOCA ANCORA!",
                color=COLOR_GREEN,
                dwell_sec=0.58,
                active=True,
                label_scale=0.72,
            )
            if self.dwell.update_and_draw(frame, [play_btn], hands, dt) == "play_rps":
                self.start_round()
        else:
            play_btn = DwellButton(
                btn_id="play_rps",
                x1=cx - 190,
                y1=h - 126,
                x2=cx + 190,
                y2=h - 54,
                label="INIZIA SFIDA!",
                subtitle="Punta qui o fai POLLICE SU",
                color=COLOR_GREEN,
                dwell_sec=0.58,
                active=True,
                label_scale=0.72,
            )
            if self.dwell.update_and_draw(frame, [play_btn], hands, dt) == "play_rps":
                self.start_round()

        # Barra caricamento pollice su
        if self._thumb_hold > 0.05 and self.state not in ("COUNTDOWN", "THROW"):
            pct = min(1.0, self._thumb_hold / 0.7)
            bx1, bx2 = cx - 120, cx + 120
            by = h - 148
            cv2.rectangle(frame, (bx1, by), (bx2, by + 10), (50, 60, 80), -1)
            cv2.rectangle(frame, (bx1, by), (int(bx1 + (bx2 - bx1) * pct), by + 10), COLOR_GREEN, -1)

        return "Punta su INIZIA SFIDA o fai POLLICE SU per giocare contro l'AI!"
