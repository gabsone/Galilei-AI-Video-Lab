"""IIS Galilei - Di Palo (Salerno) — AI Hand & Face Lab (Kiosk 100% Touch-Free).

Progettato per schermi grandi (60"+) con webcam sopra:
- Navigazione stile Apple Vision Pro (Punta + PINCH) + supporto Mouse + Tastiera backup.
- Display gestito via SDL2 (pygame) per un vero Fullscreen senza barra bianca su Mac e Windows!
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

from app.dwell_ui import (
    DwellButton,
    DwellController,
    HomeMenuScreen,
    clear_unhandled_mouse_click,
    on_mouse_event,
)
from app.hand_tracker import HandTracker
from app.hud import draw_bottom_hints, draw_top_bar, draw_xray_panel, get_top_bar_buttons
from app.modes.air_graffiti import AirGraffitiMode
from app.modes.attendance_counter import AttendanceCounterMode
from app.modes.face_emoji import FaceEmojiMode
from app.modes.fruit_ninja import FruitNinjaMode
from app.modes.hand_synth import HandSynthMode
from app.modes.neon_pong import NeonPongMode
from app.modes.object_hunt import ObjectHuntMode
from app.modes.rps_game import RPSGameMode
from app.modes.stark_hud import StarkHUDMode
from app.modes.superpowers import SuperpowersMode
from app.sound_engine import SoundEngine

WINDOW_NAME = "IIS Galilei - Di Palo (Salerno) | AI Hand & Face Lab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IIS Galilei - Di Palo AI Hand & Face Lab")
    parser.add_argument("--camera", type=int, default=0, help="Indice webcam (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Larghezza video (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Altezza video (default: 720)")
    parser.add_argument("--fullscreen", action="store_true", help="Avvia a schermo intero")
    return parser.parse_args()


def get_base_dir() -> Path:
    """Restituisce la cartella base del progetto (compatibile anche con PyInstaller .exe)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def open_camera(camera_idx: int, width: int, height: int) -> cv2.VideoCapture:
    if sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_idx, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(
            "Impossibile aprire la webcam. Verifica i permessi Fotocamera del sistema operativo."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def create_display(is_fullscreen: bool, width: int, height: int) -> pygame.Surface:
    """Crea la finestra SDL2 (vero fullscreen senza barra bianca su macOS e Windows)."""
    if is_fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
    else:
        screen = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.DOUBLEBUF)
    pygame.display.set_caption(WINDOW_NAME)
    return screen


def main() -> int:
    args = parse_args()
    project_root = get_base_dir()
    hand_model_path = project_root / "models" / "hand_landmarker.task"
    person_model_path = project_root / "models" / "efficientdet_lite2.tflite"
    face_model_path = project_root / "models" / "face_landmarker.task"

    sound = SoundEngine()
    pygame.display.init()

    tracker = HandTracker(model_path=hand_model_path, max_num_hands=2)
    home_menu = HomeMenuScreen(sound)
    nav_dwell = DwellController(sound)

    modes = {
        1: FruitNinjaMode(sound),
        2: RPSGameMode(sound),
        3: HandSynthMode(sound),
        4: StarkHUDMode(sound),
        5: AttendanceCounterMode(sound, model_path=person_model_path),
        6: FaceEmojiMode(sound, model_path=face_model_path),
        7: NeonPongMode(sound),
        8: SuperpowersMode(sound),
        9: AirGraffitiMode(sound),
        10: ObjectHuntMode(sound, model_path=person_model_path),
    }
    active_mode = 0  # 0 = Menu Home 5x2 touch-free
    debug_mode = False
    is_fullscreen = args.fullscreen

    def switch_mode(new_mode: int) -> None:
        nonlocal active_mode, debug_mode
        if new_mode == active_mode:
            return
        if active_mode in modes and hasattr(modes[active_mode], "on_exit"):
            modes[active_mode].on_exit()
        active_mode = new_mode
        nav_dwell.reset()
        if active_mode == 0:
            home_menu.reset()
            debug_mode = False
        elif active_mode in modes and hasattr(modes[active_mode], "on_enter"):
            modes[active_mode].on_enter()
        sound.play("switch")

    cap = open_camera(args.camera, args.width, args.height)
    screen = create_display(is_fullscreen, args.width, args.height)

    prev_time = time.perf_counter()
    fps_smooth = 30.0
    start_time = prev_time
    running = True
    frame_w, frame_h = args.width, args.height

    try:
        while running:
            # Gestione eventi SDL2 (Mouse mappato alla risoluzione del frame + Tastiera backup)
            win_w, win_h = screen.get_size()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE and not is_fullscreen:
                    screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE | pygame.DOUBLEBUF)
                elif ev.type == pygame.MOUSEMOTION:
                    fx = int(ev.pos[0] * frame_w / max(1, win_w))
                    fy = int(ev.pos[1] * frame_h / max(1, win_h))
                    on_mouse_event(cv2.EVENT_MOUSEMOVE, fx, fy, 0, None)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    fx = int(ev.pos[0] * frame_w / max(1, win_w))
                    fy = int(ev.pos[1] * frame_h / max(1, win_h))
                    on_mouse_event(cv2.EVENT_LBUTTONDOWN, fx, fy, 0, None)
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif k in (pygame.K_0, pygame.K_h):
                        switch_mode(0)
                    elif pygame.K_1 <= k <= pygame.K_9:
                        switch_mode(k - pygame.K_0)
                    elif k == pygame.K_a:
                        switch_mode(10)
                    elif k == pygame.K_d:
                        debug_mode = not debug_mode
                        sound.play("switch")
                    elif k == pygame.K_r and active_mode in modes:
                        modes[active_mode].reset()
                        sound.play("go")
                    elif k == pygame.K_m:
                        sound.toggle_mute()
                    elif k == pygame.K_f:
                        is_fullscreen = not is_fullscreen
                        screen = create_display(is_fullscreen, args.width, args.height)
                    elif active_mode in modes:
                        # Passa il carattere ASCII alla modalita attiva (es. SPAZIO, P, G, T, C, +, -)
                        ch = ev.unicode
                        if ch:
                            modes[active_mode].handle_key(ord(ch))
                        elif k == pygame.K_SPACE:
                            modes[active_mode].handle_key(ord(" "))

            if not running:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Effetto specchio naturale per chi e davanti al 60 pollici
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            now = time.perf_counter()
            dt = min(0.05, max(0.001, now - prev_time))
            prev_time = now
            fps_smooth = fps_smooth * 0.88 + (1.0 / dt) * 0.12
            timestamp_ms = int((now - start_time) * 1000)

            hands = tracker.detect(frame, timestamp_ms)

            # Aggiorna e disegna Menu Home (0) oppure Modalita attiva (1..10)
            if active_mode == 0:
                chosen = home_menu.update_and_draw(frame, hands, dt)
                extra_hint = "Punta su una scheda e fai PINCH (o clicca col mouse) per iniziare!"
                if chosen is not None and chosen in modes:
                    switch_mode(chosen)
            else:
                extra_hint = modes[active_mode].update_and_draw(frame, hands, dt, debug_mode)
                if debug_mode and active_mode not in (5, 6):
                    draw_xray_panel(frame, hands)

            # HUD globale
            draw_top_bar(frame, active_mode, fps_smooth, sound.muted, debug_mode)

            # Pulsanti barra superiore (MENU, RESET, X-RAY)
            if active_mode != 0:
                top_specs = get_top_bar_buttons(frame_w, active_mode, debug_mode)
                top_btns = [
                    DwellButton(
                        btn_id=bid,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        label=lbl,
                        color=col,
                        dwell_sec=0.48,
                        active=act,
                        label_scale=0.56,
                    )
                    for bid, x1, y1, x2, y2, lbl, col, act in top_specs
                ]
                nav_hit = nav_dwell.update_and_draw(frame, top_btns, hands, dt)
                if nav_hit == "nav_menu":
                    switch_mode(0)
                elif nav_hit == "nav_reset" and active_mode in modes:
                    modes[active_mode].reset()
                    sound.play("go")
                elif nav_hit == "nav_xray":
                    debug_mode = not debug_mode

            draw_bottom_hints(frame, extra_hint)
            clear_unhandled_mouse_click()

            # Rendering su finestra SDL2 (vero fullscreen senza barra bianca!)
            win_w, win_h = screen.get_size()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(frame_rgb.tobytes(), (frame_w, frame_h), "RGB")
            if (win_w, win_h) != (frame_w, frame_h):
                surf = pygame.transform.scale(surf, (win_w, win_h))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
    finally:
        cap.release()
        tracker.close()
        for m in modes.values():
            if hasattr(m, "close"):
                m.close()
        sound.close()
        pygame.display.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
