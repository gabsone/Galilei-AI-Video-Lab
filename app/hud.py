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


def draw_top_bar(
    frame: np.ndarray,
    active_mode: int,
    fps: float,
    muted: bool,
    debug_mode: bool,
) -> None:
    """Barra superiore con selezione modalità [1] [2] [3] e stato sistema."""
    h, w = frame.shape[:2]
    draw_panel(frame, 0, 0, w, 62, bg_color=(14, 16, 26), border_color=(60, 75, 110), alpha=0.78, thickness=1)

    # Logo ufficiale + brand IIS Galilei - Di Palo a sinistra
    logo = _get_school_logo()
    text_x = 18
    if logo is not None:
        cv2.circle(frame, (34, 31), 24, (245, 248, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (34, 31), 24, COLOR_CYAN, 2, cv2.LINE_AA)
        _overlay_bgra(frame, logo, 12, 9)
        text_x = 66

    cv2.putText(
        frame,
        "GALILEI - DI PALO",
        (text_x, 29),
        cv2.FONT_HERSHEY_DUPLEX,
        0.60,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "INFORMATICA · SALERNO",
        (text_x, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )

    # Tab delle 5 modalità al centro
    tabs = [
        (1, "1: SLICE", COLOR_GREEN),
        (2, "2: MORRA", COLOR_MAGENTA),
        (3, "3: MUSIC", COLOR_CYAN),
        (4, "4: STARK", COLOR_ORANGE),
        (5, "5: PRESENTI", COLOR_YELLOW),
    ]
    tab_w = 136
    gap = 8
    total_w = len(tabs) * tab_w + (len(tabs) - 1) * gap
    start_x = max(272, (w - total_w) // 2)

    for i, (mode_id, label, color) in enumerate(tabs):
        tx1 = start_x + i * (tab_w + gap)
        tx2 = tx1 + tab_w
        is_active = mode_id == active_mode
        draw_panel(
            frame,
            tx1,
            10,
            tx2,
            52,
            bg_color=(32, 42, 62) if is_active else (18, 22, 34),
            border_color=color if is_active else (70, 80, 100),
            alpha=0.88 if is_active else 0.55,
            thickness=2 if is_active else 1,
        )
        put_text_centered(
            frame,
            label,
            ((tx1 + tx2) // 2, 31),
            scale=0.55,
            color=color if is_active else COLOR_MUTED,
            thickness=2 if is_active else 1,
            shadow=False,
        )

    # Info FPS / Audio / Debug a destra
    status_parts = [f"{int(fps)} FPS"]
    if debug_mode:
        status_parts.append("X-RAY")
    if muted:
        status_parts.append("MUTE")
    status_text = " | ".join(status_parts)
    (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(
        frame,
        status_text,
        (w - tw - 18, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_YELLOW if debug_mode else COLOR_MUTED,
        2,
        cv2.LINE_AA,
    )


def draw_bottom_hints(frame: np.ndarray, extra_hint: str = "") -> None:
    """Barra inferiore compatta con i comandi da tastiera per il docente."""
    h, w = frame.shape[:2]
    draw_panel(frame, 0, h - 38, w, h, bg_color=(12, 14, 22), border_color=(45, 55, 80), alpha=0.72, thickness=1)
    base = "[1-5] Modalita  |  [D] Sotto il cofano  |  [R] Reset  |  [F] Fullscreen  |  [M] Audio  |  [Q] Esci"
    text = f"{extra_hint}    ||    {base}" if extra_hint else base
    cv2.putText(
        frame,
        text,
        (16, h - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COLOR_MUTED,
        1,
        cv2.LINE_AA,
    )
    site = "galileidipalo.edu.it"
    (sw, _), _ = cv2.getTextSize(site, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(
        frame,
        site,
        (w - sw - 16, h - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
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
