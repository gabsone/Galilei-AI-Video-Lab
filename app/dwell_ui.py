"""Componenti UI 100% touch-free (Dwell-to-Click) per schermi grandi (60"+) senza tastiera."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


@dataclass
class DwellButton:
    btn_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    color: Tuple[int, int, int] = COLOR_CYAN
    subtitle: str = ""
    dwell_sec: float = 0.62
    active: bool = False
    label_scale: float = 0.62


# Stato globale del mouse (aggiornato da cv2.setMouseCallback in main.py)
_MOUSE_POS: Optional[Tuple[int, int]] = None
_MOUSE_CLICK_POS: Optional[Tuple[int, int]] = None


def on_mouse_event(event: int, x: int, y: int, flags: int, param: object) -> None:
    """Callback OpenCV per supportare hover e click sinistro su tutti i pulsanti."""
    global _MOUSE_POS, _MOUSE_CLICK_POS
    _MOUSE_POS = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        _MOUSE_CLICK_POS = (x, y)


def clear_unhandled_mouse_click() -> None:
    """Azzera eventuali click a vuoto a fine frame."""
    global _MOUSE_CLICK_POS
    _MOUSE_CLICK_POS = None


class DwellController:
    """Gestisce i pulsanti touch-free stile Apple Vision Pro: Punta (hover) + PINCH per cliccare."""

    PINCH_HOLD_SEC = 0.14  # Quasi istantaneo, filtra eventuali falsi positivi di 1-2 frame

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self._progress: Dict[str, float] = {}
        self._cooldown: Dict[str, float] = {}
        self._armed: Dict[str, bool] = {}  # True se l'utente ha fatto hover con dita aperte prima del pinch

    def reset(self) -> None:
        self._progress.clear()
        self._cooldown.clear()
        self._armed.clear()

    def update_and_draw(
        self,
        frame: np.ndarray,
        buttons: List[DwellButton],
        hands: List[HandData],
        dt: float,
    ) -> Optional[str]:
        """Aggiorna e disegna i pulsanti; clicca solo con Punta + PINCH intenzionale."""
        triggered: Optional[str] = None
        cursor_rings: List[Tuple[Tuple[int, int], float, Tuple[int, int, int], bool]] = []

        for btn in buttons:
            bid = btn.btn_id
            if bid in self._cooldown:
                self._cooldown[bid] -= dt
                if self._cooldown[bid] <= 0:
                    del self._cooldown[bid]

        global _MOUSE_CLICK_POS
        for btn in buttons:
            bid = btn.btn_id
            pad = 10
            hovered_tip: Optional[Tuple[int, int]] = None
            hovered_pinching = False

            # 1. Controllo click immediato con il mouse (priorita assoluta!)
            mouse_hover = (
                _MOUSE_POS is not None
                and btn.x1 <= _MOUSE_POS[0] <= btn.x2
                and btn.y1 <= _MOUSE_POS[1] <= btn.y2
            )
            if (
                _MOUSE_CLICK_POS is not None
                and triggered is None
                and btn.x1 <= _MOUSE_CLICK_POS[0] <= btn.x2
                and btn.y1 <= _MOUSE_CLICK_POS[1] <= btn.y2
            ):
                triggered = bid
                _MOUSE_CLICK_POS = None
                self._progress[bid] = 0.0
                self._armed[bid] = False
                self._cooldown[bid] = 0.25
                self.sound.play("switch")

            # 2. Controllo mano (Punta + PINCH stile Apple Vision Pro)
            for hand in hands:
                px = (hand.index_tip[0] + hand.thumb_tip[0]) // 2
                py = (hand.index_tip[1] + hand.thumb_tip[1]) // 2
                for pt in (hand.index_tip, (px, py)):
                    if (btn.x1 - pad) <= pt[0] <= (btn.x2 + pad) and (btn.y1 - pad) <= pt[1] <= (btn.y2 + pad):
                        hovered_tip = pt
                        hovered_pinching = hand.is_pinching
                        break
                if hovered_tip is not None:
                    break

            if hovered_tip is not None and bid not in self._cooldown:
                if not hovered_pinching:
                    self._armed[bid] = True
                    self._progress[bid] = max(0.0, self._progress.get(bid, 0.0) - dt * 4.0)
                    cursor_rings.append((hovered_tip, 0.0, btn.color, False))
                elif self._armed.get(bid, False):
                    prog = min(1.0, self._progress.get(bid, 0.0) + dt / self.PINCH_HOLD_SEC)
                    self._progress[bid] = prog
                    cursor_rings.append((hovered_tip, prog, btn.color, True))
                    if prog >= 1.0 and triggered is None:
                        triggered = bid
                        self._progress[bid] = 0.0
                        self._armed[bid] = False
                        self._cooldown[bid] = 0.65
                        self.sound.play("switch")
                else:
                    self._progress[bid] = 0.0
            else:
                self._armed[bid] = False
                self._progress[bid] = max(0.0, self._progress.get(bid, 0.0) - dt * 4.0)

            prog = self._progress.get(bid, 0.0)
            is_hover = (hovered_tip is not None) or mouse_hover
            border = COLOR_YELLOW if is_hover else (btn.color if btn.active else (85, 98, 125))
            bg = (30, 42, 62) if (is_hover or btn.active) else (16, 22, 34)

            draw_panel(
                frame,
                btn.x1,
                btn.y1,
                btn.x2,
                btn.y2,
                bg_color=bg,
                border_color=border,
                alpha=0.86 if (is_hover or btn.active) else 0.70,
                thickness=3 if is_hover else (2 if btn.active else 1),
            )

            cx = (btn.x1 + btn.x2) // 2
            cy = (btn.y1 + btn.y2) // 2
            if btn.subtitle:
                put_text_centered(
                    frame,
                    btn.label,
                    (cx, cy - 14),
                    scale=btn.label_scale,
                    color=COLOR_YELLOW if is_hover else btn.color,
                    thickness=2,
                )
                put_text_centered(
                    frame,
                    btn.subtitle,
                    (cx, cy + 20),
                    scale=0.48,
                    color=COLOR_WHITE if is_hover else COLOR_MUTED,
                    thickness=1,
                    shadow=False,
                )
            else:
                put_text_centered(
                    frame,
                    btn.label,
                    (cx, cy),
                    scale=btn.label_scale,
                    color=COLOR_YELLOW if is_hover else (btn.color if btn.active else COLOR_WHITE),
                    thickness=2 if (is_hover or btn.active) else 1,
                    shadow=False,
                )

            # Feedback visivo sul bordo inferiore durante il pinch
            if prog > 0.02:
                bar_w = int((btn.x2 - btn.x1 - 6) * prog)
                cv2.rectangle(
                    frame,
                    (btn.x1 + 3, btn.y2 - 7),
                    (btn.x1 + 3 + bar_w, btn.y2 - 2),
                    COLOR_YELLOW,
                    -1,
                )

        # Cursore stile Apple Vision Pro: cerchietto in hover, anello dorato al pinch!
        for tip, prog, col, pinching in cursor_rings:
            if pinching:
                cv2.circle(frame, tip, 24, (30, 35, 50), 4, cv2.LINE_AA)
                end_ang = int(-90 + 360 * prog)
                cv2.ellipse(frame, tip, (24, 24), 0, -90, end_ang, COLOR_YELLOW, 5, cv2.LINE_AA)
                cv2.circle(frame, tip, 8, COLOR_YELLOW, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, tip, 18, col, 2, cv2.LINE_AA)
                put_text_centered(frame, "PINCH", (tip[0], tip[1] - 28), scale=0.42, color=COLOR_YELLOW, thickness=1)

        return triggered


class HomeMenuScreen:
    """Schermata Home olografica a 11 card (griglia 4x3) per scegliere l'app con Punta + PINCH o Mouse."""

    CARDS = [
        (1, "CYBER SLICE", "Taglia le sfere tech!", COLOR_GREEN),
        (2, "MORRA AI", "Sasso-Carta-Forbici auto", COLOR_MAGENTA),
        (3, "MUSIC STUDIO", "DJ · Piano · Batteria", COLOR_CYAN),
        (4, "STARK HUD", "Finestre stile Iron Man", COLOR_ORANGE),
        (5, "PRESENTI", "Appello AI della classe", COLOR_YELLOW),
        (6, "FACE EMOJI", "Emoji + Photo Booth!", (120, 220, 255)),
        (7, "NEON PONG", "Sfida 1 vs 1 o vs AI", COLOR_CYAN),
        (8, "SUPERPOTERI", "Kamehameha & Repulsor", COLOR_ORANGE),
        (9, "AIR GRAFFITI", "Lavagna Neon in aria", COLOR_MAGENTA),
        (10, "CACCIA OGGETTI", "Trova oggetti in 12s!", COLOR_GREEN),
        (11, "NEON DRIVE 3D", "Guida 3D col Volante!", COLOR_YELLOW),
        (12, "LABIRINTO 3D", "Esplora in 3D Raycasting!", COLOR_MAGENTA),
    ]

    def __init__(self, sound: SoundEngine) -> None:
        self.dwell = DwellController(sound)

    def reset(self) -> None:
        self.dwell.reset()

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
    ) -> Optional[int]:
        """Disegna il menu Home 4x3; restituisce l'ID modalita (1..11) quando selezionata."""
        h, w = frame.shape[:2]

        overlay = np.full_like(frame, (10, 14, 24))
        cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

        cx = w // 2
        put_text_centered(
            frame,
            "SCEGLI UN'ESPERIENZA AI CON LA MANO",
            (cx, 102),
            scale=0.92,
            color=COLOR_WHITE,
            thickness=2,
        )
        put_text_centered(
            frame,
            "Punta su una scheda e fai PINCH (o clicca col mouse) per aprire!",
            (cx, 132),
            scale=0.52,
            color=COLOR_CYAN,
            thickness=1,
            shadow=False,
        )

        # Griglia 4 x 3 (ultima riga centrata) ben leggibile sul 60 pollici
        cols, rows = 4, 3
        margin_x = max(28, int(w * 0.03))
        top_y = 154
        bot_y = h - 50
        gap_x, gap_y = 16, 14
        card_w = (w - 2 * margin_x - (cols - 1) * gap_x) // cols
        card_h = (bot_y - top_y - (rows - 1) * gap_y) // rows

        buttons: List[DwellButton] = []
        for idx, (mode_id, title, sub, col) in enumerate(self.CARDS):
            r = idx // cols
            c = idx % cols
            # Centra l'ultima riga se ha meno di 4 card
            row_count = min(cols, len(self.CARDS) - r * cols)
            row_offset = (cols - row_count) * (card_w + gap_x) // 2
            x1 = margin_x + row_offset + c * (card_w + gap_x)
            y1 = top_y + r * (card_h + gap_y)
            x2 = x1 + card_w
            y2 = y1 + card_h
            buttons.append(
                DwellButton(
                    btn_id=str(mode_id),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=title,
                    subtitle=sub,
                    color=col,
                    active=True,
                    label_scale=0.68,
                )
            )

        # Disegna mani prima del dwell così l'anello di caricamento resta sopra il dito
        for idx, hand in enumerate(hands):
            accent = COLOR_CYAN if idx == 0 else COLOR_MAGENTA
            draw_hand_skeleton(frame, hand, accent=accent, show_ids=False)
            cv2.circle(frame, hand.index_tip, 11, COLOR_YELLOW, 3, cv2.LINE_AA)

        hit = self.dwell.update_and_draw(frame, buttons, hands, dt)
        if hit is not None:
            return int(hit)
        return None
