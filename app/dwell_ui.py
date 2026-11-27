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


class DwellController:
    """Gestisce l'hover con l'indice e il caricamento circolare per attivare pulsanti senza tastiera."""

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self._progress: Dict[str, float] = {}
        self._cooldown: Dict[str, float] = {}

    def reset(self) -> None:
        self._progress.clear()
        self._cooldown.clear()

    def update_and_draw(
        self,
        frame: np.ndarray,
        buttons: List[DwellButton],
        hands: List[HandData],
        dt: float,
    ) -> Optional[str]:
        """Aggiorna e disegna i pulsanti; restituisce btn_id se uno viene attivato."""
        triggered: Optional[str] = None
        cursor_rings: List[Tuple[Tuple[int, int], float, Tuple[int, int, int]]] = []

        # Aggiorna cooldown
        for bid in list(self._cooldown.keys()):
            self._cooldown[bid] -= dt
            if self._cooldown[bid] <= 0:
                del self._cooldown[bid]

        active_ids = {b.btn_id for b in buttons}
        for bid in list(self._progress.keys()):
            if bid not in active_ids:
                del self._progress[bid]

        for btn in buttons:
            # Tolleranza invisibile di +14px + accetta sia punta dell'indice che centro del palmo
            pad = 14
            hovered_tip: Optional[Tuple[int, int]] = None
            for hand in hands:
                for pt in (hand.index_tip, hand.palm_center):
                    if (btn.x1 - pad) <= pt[0] <= (btn.x2 + pad) and (btn.y1 - pad) <= pt[1] <= (btn.y2 + pad):
                        hovered_tip = pt
                        break
                if hovered_tip is not None:
                    break

            if hovered_tip is not None and btn.btn_id not in self._cooldown:
                prev = self._progress.get(btn.btn_id, 0.0)
                prog = min(1.0, prev + dt / max(0.15, btn.dwell_sec))
                self._progress[btn.btn_id] = prog
                cursor_rings.append((hovered_tip, prog, btn.color))
                if prog >= 1.0 and triggered is None:
                    triggered = btn.btn_id
                    self._progress[btn.btn_id] = 0.0
                    self._cooldown[btn.btn_id] = 0.70
                    self.sound.play("switch")
            else:
                # Decadimento morbido anti-tremolio della mano
                prev = self._progress.get(btn.btn_id, 0.0)
                self._progress[btn.btn_id] = max(0.0, prev - dt * 0.85)

            prog = self._progress.get(btn.btn_id, 0.0)
            is_hover = hovered_tip is not None
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

            # Barra di avanzamento sul bordo inferiore del pulsante
            if prog > 0.02:
                bar_w = int((btn.x2 - btn.x1 - 6) * prog)
                cv2.rectangle(
                    frame,
                    (btn.x1 + 3, btn.y2 - 7),
                    (btn.x1 + 3 + bar_w, btn.y2 - 2),
                    COLOR_YELLOW,
                    -1,
                )

        # Disegna anelli di caricamento circolari attorno al dito (visibilissimi sul 60"!)
        for tip, prog, col in cursor_rings:
            cv2.circle(frame, tip, 26, (30, 35, 50), 4, cv2.LINE_AA)
            end_ang = int(-90 + 360 * prog)
            cv2.ellipse(frame, tip, (26, 26), 0, -90, end_ang, COLOR_YELLOW, 5, cv2.LINE_AA)
            cv2.circle(frame, tip, 7, col, -1, cv2.LINE_AA)

        return triggered


class HomeMenuScreen:
    """Schermata Home olografica a 6 grandi card per scegliere l'app con la mano."""

    CARDS = [
        (1, "CYBER SLICE", "Taglia le sfere tech con l'indice", COLOR_GREEN),
        (2, "MORRA AI", "Sfida l'AI a Sasso-Carta-Forbici", COLOR_MAGENTA),
        (3, "MUSIC STUDIO", "DJ 4-Zone · Pianoforte · Batteria", COLOR_CYAN),
        (4, "STARK HUD", "Finestre olografiche stile Iron Man", COLOR_ORANGE),
        (5, "CONTA PRESENTI", "Appello AI della classe in diretta", COLOR_YELLOW),
        (6, "FACE EMOJI", "Emoji animate sulle espressioni!", (120, 220, 255)),
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
        """Disegna il menu Home; restituisce l'ID modalita (1..6) quando selezionata."""
        h, w = frame.shape[:2]

        # Leggera patina scura per far risaltare le 6 card sul proiettore / 60"
        overlay = np.full_like(frame, (10, 14, 24))
        cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

        # Intestazione centrale
        cx = w // 2
        put_text_centered(
            frame,
            "SCEGLI UN'ESPERIENZA AI CON LA MANO",
            (cx, 108),
            scale=0.98,
            color=COLOR_WHITE,
            thickness=2,
        )
        put_text_centered(
            frame,
            "Punta l'INDICE su una scheda per 1 secondo  (nessuna tastiera necessaria!)",
            (cx, 142),
            scale=0.56,
            color=COLOR_CYAN,
            thickness=1,
            shadow=False,
        )

        # Griglia 3 x 2 di card giganti
        cols, rows = 3, 2
        margin_x = max(36, int(w * 0.04))
        top_y = 172
        bot_y = h - 58
        gap_x, gap_y = 22, 22
        card_w = (w - 2 * margin_x - (cols - 1) * gap_x) // cols
        card_h = (bot_y - top_y - (rows - 1) * gap_y) // rows

        buttons: List[DwellButton] = []
        for idx, (mode_id, title, sub, col) in enumerate(self.CARDS):
            r = idx // cols
            c = idx % cols
            x1 = margin_x + c * (card_w + gap_x)
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
                    dwell_sec=0.52,
                    active=True,
                    label_scale=0.88,
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
