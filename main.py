"""IIS Galilei - Di Palo (Salerno) — AI Hand & Face Lab (Kiosk 100% Touch-Free).

Progettato per schermi grandi (60"+) con webcam sopra:
tutta la navigazione avviene con l'indice (Dwell-to-Click), senza tastiera!
I tasti rapidi (0/H, 1-6, D, R, F, M, Q/ESC) restano comunque attivi come backup.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from app.dwell_ui import DwellButton, DwellController, HomeMenuScreen
from app.hand_tracker import HandTracker
from app.hud import draw_bottom_hints, draw_top_bar, draw_xray_panel, get_top_bar_buttons
from app.modes.attendance_counter import AttendanceCounterMode
from app.modes.face_emoji import FaceEmojiMode
from app.modes.fruit_ninja import FruitNinjaMode
from app.modes.hand_synth import HandSynthMode
from app.modes.rps_game import RPSGameMode
from app.modes.stark_hud import StarkHUDMode
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


def main() -> int:
    args = parse_args()
    project_root = get_base_dir()
    hand_model_path = project_root / "models" / "hand_landmarker.task"
    person_model_path = project_root / "models" / "efficientdet_lite2.tflite"
    face_model_path = project_root / "models" / "face_landmarker.task"

    sound = SoundEngine()
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
    }
    active_mode = 0  # 0 = Menu Home touch-free (6 card giganti)
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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if is_fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    prev_time = time.perf_counter()
    fps_smooth = 30.0
    start_time = prev_time

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Effetto specchio naturale per chi e davanti al 60 pollici
            frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = min(0.05, max(0.001, now - prev_time))
            prev_time = now
            fps_smooth = fps_smooth * 0.88 + (1.0 / dt) * 0.12
            timestamp_ms = int((now - start_time) * 1000)

            hands = tracker.detect(frame, timestamp_ms)

            # Aggiorna e disegna Menu Home (0) oppure Modalita attiva (1..6)
            if active_mode == 0:
                chosen = home_menu.update_and_draw(frame, hands, dt)
                extra_hint = "Alza la mano e punta l'INDICE sulla scheda che vuoi provare!"
                if chosen is not None and chosen in modes:
                    switch_mode(chosen)
            else:
                extra_hint = modes[active_mode].update_and_draw(frame, hands, dt, debug_mode)
                if debug_mode and active_mode not in (5, 6):
                    draw_xray_panel(frame, hands)

            # HUD globale
            draw_top_bar(frame, active_mode, fps_smooth, sound.muted, debug_mode)

            # Pulsanti touch-free in barra superiore (MENU, RESET, X-RAY)
            if active_mode != 0:
                top_specs = get_top_bar_buttons(frame.shape[1], active_mode, debug_mode)
                top_btns = [
                    DwellButton(
                        btn_id=bid,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        label=lbl,
                        color=col,
                        dwell_sec=0.60,
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

            cv2.imshow(WINDOW_NAME, frame)

            # Backup tastiera (opzionale per il tecnico)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("0"), ord("h"), ord("H")):
                switch_mode(0)
            elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                switch_mode(int(chr(key)))
            elif key in (ord("d"), ord("D")):
                debug_mode = not debug_mode
                sound.play("switch")
            elif key in (ord("r"), ord("R")) and active_mode in modes:
                modes[active_mode].reset()
                sound.play("go")
            elif key in (ord("m"), ord("M")):
                sound.toggle_mute()
            elif key in (ord("f"), ord("F")):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
            elif key != 255 and active_mode in modes:
                modes[active_mode].handle_key(key)
    finally:
        cap.release()
        tracker.close()
        for m in modes.values():
            if hasattr(m, "close"):
                m.close()
        sound.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
