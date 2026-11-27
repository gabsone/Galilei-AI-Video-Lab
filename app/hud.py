"""Componenti grafici neon/HUD per OpenCV (stile Apple Vision Pro / Cyberpunk)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.hand_tracker import HAND_CONNECTIONS, HandData

# Cache del logo ufficiale IIS Galilei - Di Palo (44x44 BGRA)
_LOGO_BGRA_CACHE: Optional[np.ndarray] = None
_LOGO_LOADED = False


def _get_school_logo() -> Optional[np.ndarray]:
    global _LOGO_BGRA_CACHE, _LOGO_LOADED
    if _LOGO_LOADED:
        return _LOGO_BGRA_CACHE
    _LOGO_LOADED = True
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent.parent
    logo_path = base / "assets" / "galilei_logo.png"
    if logo_path.exists():
        img = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            _LOGO_BGRA_CACHE = cv2.resize(img, (44, 44), interpolation=cv2.INTER_AREA)
    return _LOGO_BGRA_CACHE


def _overlay_bgra(frame: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    oh, ow = overlay.shape[:2]
    fh, fw = frame.shape[:2]
    if x < 0 or y < 0 or x + ow > fw or y + oh > fh:
        return
    roi = frame[y : y + oh, x : x + ow]
    alpha = (overlay[:, :, 3:4].astype(np.float32)) / 255.0
    rgb = overlay[:, :, :3].astype(np.float32)
    blended = rgb * alpha + roi.astype(np.float32) * (1.0 - alpha)
    frame[y : y + oh, x : x + ow] = blended.astype(np.uint8)

# Palette BGR (OpenCV usa Blue-Green-Red)
COLOR_CYAN = (255, 225, 0)
COLOR_MAGENTA = (220, 60, 255)
COLOR_GREEN = (90, 255, 130)
COLOR_ORANGE = (40, 165, 255)
COLOR_RED = (70, 70, 255)
COLOR_YELLOW = (40, 235, 255)
COLOR_WHITE = (245, 248, 255)
COLOR_DARK = (18, 20, 30)
COLOR_MUTED = (165, 175, 195)


def draw_panel(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    bg_color: Tuple[int, int, int] = COLOR_DARK,
    border_color: Tuple[int, int, int] = COLOR_CYAN,
    alpha: float = 0.68,
    thickness: int = 2,
) -> None:
    """Disegna un pannello semitrasparente con bordo neon."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    overlay = np.full_like(roi, bg_color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    if thickness > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness, cv2.LINE_AA)


def put_text_centered(
    frame: np.ndarray,
    text: str,
    center: Tuple[int, int],
    scale: float = 1.0,
    color: Tuple[int, int, int] = COLOR_WHITE,
    thickness: int = 2,
    shadow: bool = True,
) -> None:
    """Disegna testo centrato con leggera ombra per leggibilità sul proiettore."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = int(center[0] - tw / 2)
    y = int(center[1] + th / 2)
    if shadow:
        cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hand_skeleton(
    frame: np.ndarray,
    hand: HandData,
    accent: Tuple[int, int, int] = COLOR_CYAN,
    show_ids: bool = False,
) -> None:
    """Disegna lo scheletro della mano con effetto neon luminoso."""
    pts = hand.landmarks_px

    # Connessioni con doppio tratto (glow + core)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], accent, 4, cv2.LINE_AA)
        cv2.line(frame, pts[a], pts[b], COLOR_WHITE, 1, cv2.LINE_AA)

    # Giunture
    for idx, (x, y) in enumerate(pts):
        is_tip = idx in (4, 8, 12, 16, 20)
        radius = 7 if is_tip else 4
        cv2.circle(frame, (x, y), radius + 2, accent, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius - 1, COLOR_WHITE, -1, cv2.LINE_AA)
        if show_ids:
            cv2.putText(
                frame,
                str(idx),
                (x + 7, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                COLOR_YELLOW,
                1,
                cv2.LINE_AA,
            )


MODE_TITLES = {
    0: ("MENU PRINCIPALE", COLOR_CYAN),
    1: ("CYBER SLICE", COLOR_GREEN),
    2: ("MORRA AI", COLOR_MAGENTA),
    3: ("AI MUSIC STUDIO", COLOR_CYAN),
    4: ("STARK HUD", COLOR_ORANGE),
    5: ("CONTA PRESENTI", COLOR_YELLOW),
    6: ("FACE EMOJI & PHOTO", (120, 220, 255)),
    7: ("NEON PONG", COLOR_CYAN),
    8: ("SUPERPOTERI AR", COLOR_ORANGE),
    9: ("AIR GRAFFITI", COLOR_MAGENTA),
    10: ("CACCIA OGGETTI", COLOR_GREEN),
    11: ("NEON DRIVE 3D", COLOR_YELLOW),
}


def get_top_bar_buttons(w: int, active_mode: int, debug_mode: bool) -> List[Tuple[str, int, int, int, int, str, Tuple[int, int, int], bool]]:
    """Restituisce i pulsanti touch-free della barra superiore (attivi quando active_mode != 0)."""
    if active_mode == 0:
        return []
    # 3 pulsanti grandi (54px di altezza) facilissimi da centrare su un 60 pollici!
    y1, y2 = 6, 60
    buttons = [
        ("nav_menu", w - 445, y1, w - 305, y2, "MENU", COLOR_CYAN, True),
        ("nav_reset", w - 295, y1, w - 155, y2, "RESET", COLOR_GREEN, False),
        ("nav_xray", w - 145, y1, w - 12, y2, "X-RAY", COLOR_YELLOW, debug_mode),
    ]
    return buttons


def draw_top_bar(
    frame: np.ndarray,
    active_mode: int,
    fps: float,
    muted: bool,
    debug_mode: bool,
) -> None:
    """Barra superiore brandizzata IIS Galilei - Di Palo per schermi grandi."""
    h, w = frame.shape[:2]
    draw_panel(frame, 0, 0, w, 66, bg_color=(14, 16, 26), border_color=(60, 75, 110), alpha=0.80, thickness=1)

    # Logo ufficiale + brand IIS Galilei - Di Palo a sinistra
    logo = _get_school_logo()
    text_x = 18
    if logo is not None:
        cv2.circle(frame, (34, 33), 24, (245, 248, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (34, 33), 24, COLOR_CYAN, 2, cv2.LINE_AA)
        _overlay_bgra(frame, logo, 12, 11)
        text_x = 66

    cv2.putText(
        frame,
        "GALILEI - DI PALO",
        (text_x, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "INFORMATICA · SALERNO",
        (text_x, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )

    # Badge della modalita attiva al centro
    title, accent = MODE_TITLES.get(active_mode, ("AI HAND LAB", COLOR_CYAN))
    cx = w // 2 - (70 if active_mode != 0 else 0)
    draw_panel(frame, cx - 150, 10, cx + 150, 56, bg_color=(24, 32, 48), border_color=accent, alpha=0.85, thickness=2)
    put_text_centered(frame, title, (cx, 33), scale=0.68, color=accent, thickness=2, shadow=False)

    # Nel Menu Home mostra FPS e stato audio a destra
    if active_mode == 0:
        status_parts = [f"{int(fps)} FPS"]
        if muted:
            status_parts.append("MUTE")
        status_text = " | ".join(status_parts)
        (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(
            frame,
            status_text,
            (w - tw - 18, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_MUTED,
            2,
            cv2.LINE_AA,
        )


def draw_bottom_hints(frame: np.ndarray, extra_hint: str = "") -> None:
    """Barra inferiore pulita con guida gestuale + sito istituto."""
    h, w = frame.shape[:2]
    draw_panel(frame, 0, h - 38, w, h, bg_color=(12, 14, 22), border_color=(45, 55, 80), alpha=0.76, thickness=1)
    text = extra_hint if extra_hint else "Punta e fai PINCH sui pulsanti in alto per MENU, RESET o X-RAY"
    cv2.putText(
        frame,
        text,
        (16, h - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )
    site = "galileidipalo.edu.it"
    (sw, _), _ = cv2.getTextSize(site, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(
        frame,
        site,
        (w - sw - 16, h - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )


def draw_xray_panel(frame: np.ndarray, hands: List[HandData]) -> None:
    """Modalità 'Sotto il Cofano' (tasto D): mostra la matematica dei 21 landmark."""
    h, w = frame.shape[:2]
    px1, py1, px2, py2 = 16, 76, 365, 285
    draw_panel(frame, px1, py1, px2, py2, bg_color=(10, 18, 28), border_color=COLOR_YELLOW, alpha=0.82, thickness=2)

    cv2.putText(
        frame,
        "SOTTO IL COFANO (AI MATH)",
        (px1 + 14, py1 + 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        COLOR_YELLOW,
        1,
        cv2.LINE_AA,
    )
    cv2.line(frame, (px1 + 12, py1 + 38), (px2 - 12, py1 + 38), (80, 90, 60), 1)

    if not hands:
        cv2.putText(
            frame,
            "Mostra una mano alla webcam...",
            (px1 + 14, py1 + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )
        return

    hand = hands[0]
    ix, iy = hand.index_tip
    lines = [
        f"Mani rilevate : {len(hands)}  (21 punti/mano)",
        f"Indice (id=8) : x={ix:4d}px, y={iy:4d}px",
        f"Dita [P,I,M,A,m]: {hand.fingers_up}",
        f"Distanza pinch: {hand.pinch_distance_px:5.1f} px",
        f"Pinch ratio   : {hand.pinch_ratio:4.2f} ({'ON' if hand.is_pinching else 'OFF'})",
        f"Gesto stimato : {hand.gesture}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (px1 + 14, py1 + 66 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            COLOR_WHITE if i < 5 else COLOR_GREEN,
            1,
            cv2.LINE_AA,
        )
