"""Modalità 6: FACE EMOJI AI — Copre fino a 4 volti con emoji animate che seguono le espressioni!"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from app.dwell_ui import DwellButton, DwellController
from app.hand_tracker import HandData
from app.hud import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_panel,
    put_text_centered,
)
from app.sound_engine import SoundEngine


def _rot_pt(cx: float, cy: float, dx: float, dy: float, angle: float) -> Tuple[int, int]:
    ca, sa = math.cos(angle), math.sin(angle)
    return (int(cx + dx * ca - dy * sa), int(cy + dx * sa + dy * ca))


class FaceEmojiMode:
    """Rileva fino a 4 volti + 52 espressioni (blendshapes) e disegna emoji/avatar live."""

    STYLES = ("emoji", "robot", "alien")

    def __init__(self, sound: SoundEngine, model_path: str | Path) -> None:
        self.sound = sound
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modello FaceLandmarker non trovato: {self.model_path}")

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=4,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.perf_counter()
        self._last_timestamp_ms = 0
        self.dwell = DwellController(sound)
        self.style = "emoji"
        self.reset()

    def reset(self) -> None:
        self.style = "emoji"
        self.dwell.reset()

    def handle_key(self, key: int) -> None:
        if key == ord(" "):
            idx = (self.STYLES.index(self.style) + 1) % len(self.STYLES)
            self.style = self.STYLES[idx]
            self.sound.play("switch")

    @staticmethod
    def _expression_label(bs: Dict[str, float]) -> Tuple[str, Tuple[int, int, int]]:
        smile = 0.5 * (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0))
        jaw = bs.get("jawOpen", 0.0)
        blink_l = bs.get("eyeBlinkLeft", 0.0)
        blink_r = bs.get("eyeBlinkRight", 0.0)
        brow_up = bs.get("browInnerUp", 0.0)
        pucker = bs.get("mouthPucker", 0.0)
        frown = 0.5 * (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0))

        if jaw > 0.45 and brow_up > 0.35:
            return ("SORPRESO!", COLOR_YELLOW)
        if jaw > 0.42:
            return ("WHOA!", COLOR_ORANGE)
        if (blink_l > 0.55 and blink_r < 0.28) or (blink_r > 0.55 and blink_l < 0.28):
            return ("OCCHIOLINO!", COLOR_MAGENTA)
        if smile > 0.48:
            return (f"SORRISO {int(smile * 100)}%", COLOR_GREEN)
        if pucker > 0.50:
            return ("BACIO / FISCHIO", COLOR_MAGENTA)
        if frown > 0.45:
            return ("CONCENTRATO", COLOR_RED)
        return ("NEUTRO", COLOR_CYAN)

    @staticmethod
    def _draw_classic_emoji(
        frame: np.ndarray,
        cx: float,
        cy: float,
        r: float,
        angle: float,
        bs: Dict[str, float],
    ) -> None:
        smile = 0.5 * (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0))
        jaw = bs.get("jawOpen", 0.0)
        # Nota: frame specchiato -> blinkLeft/Right seguono lo schermo
        blink_l = bs.get("eyeBlinkLeft", 0.0)
        blink_r = bs.get("eyeBlinkRight", 0.0)
        brow_up = bs.get("browInnerUp", 0.0)
        brow_down = 0.5 * (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0))
        pucker = bs.get("mouthPucker", 0.0)

        icx, icy, ir = int(cx), int(cy), max(28, int(r))
        # Cerchio base giallo emoji con bordo caldo
        cv2.circle(frame, (icx, icy), ir + 4, (20, 155, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (icx, icy), ir, (40, 215, 255), -1, cv2.LINE_AA)
        # Riflesso lucido in alto a sinistra
        shine_pt = _rot_pt(cx, cy, -r * 0.34, -r * 0.38, angle)
        cv2.circle(frame, shine_pt, max(6, int(r * 0.16)), (140, 240, 255), -1, cv2.LINE_AA)

        # Guance rosate quando sorride
        if smile > 0.35:
            for side in (-1, 1):
                cheek = _rot_pt(cx, cy, side * r * 0.52, r * 0.12, angle)
                cv2.circle(frame, cheek, max(6, int(r * 0.15)), (110, 130, 255), -1, cv2.LINE_AA)

        # Occhi (sinistro e destro indipendenti per occhiolino!)
        for side, blink in ((-1, blink_r), (1, blink_l)):
            ex, ey = _rot_pt(cx, cy, side * r * 0.34, -r * 0.16, angle)
            eye_r = max(5, int(r * 0.15))
            if blink > 0.52:
                # Occhio chiuso / occhiolino
                p1 = _rot_pt(cx, cy, side * r * 0.20, -r * 0.16, angle)
                p2 = _rot_pt(cx, cy, side * r * 0.48, -r * 0.16, angle)
                cv2.line(frame, p1, p2, (25, 25, 35), max(4, int(r * 0.08)), cv2.LINE_AA)
            else:
                open_h = max(3, int(eye_r * (1.15 - blink * 0.8 + brow_up * 0.3)))
                cv2.ellipse(
                    frame, (ex, ey), (eye_r, open_h), math.degrees(angle), 0, 360, (24, 24, 34), -1, cv2.LINE_AA
                )
                # Pupilla lucida bianca
                glint = _rot_pt(ex, ey, -eye_r * 0.25, -open_h * 0.25, angle)
                cv2.circle(frame, glint, max(2, eye_r // 3), COLOR_WHITE, -1, cv2.LINE_AA)

            # Sopracciglio espressivo
            by_off = -r * (0.40 + brow_up * 0.16 - brow_down * 0.12)
            tilt = side * brow_down * r * 0.12
            bp1 = _rot_pt(cx, cy, side * r * 0.18, by_off + tilt, angle)
            bp2 = _rot_pt(cx, cy, side * r * 0.50, by_off - tilt, angle)
            cv2.line(frame, bp1, bp2, (40, 70, 140), max(3, int(r * 0.065)), cv2.LINE_AA)

        # Bocca espressiva
        mx, my = _rot_pt(cx, cy, 0, r * 0.34, angle)
        if jaw > 0.24:
            # Bocca aperta (proporzionale a jawOpen!)
            mw = max(10, int(r * (0.28 + smile * 0.24 - pucker * 0.12)))
            mh = max(8, int(r * (0.12 + jaw * 0.38)))
            cv2.ellipse(frame, (mx, my), (mw, mh), math.degrees(angle), 0, 360, (35, 25, 75), -1, cv2.LINE_AA)
            # Lingua rossa in fondo alla bocca aperta
            tongue = _rot_pt(mx, my, 0, mh * 0.42, angle)
            cv2.ellipse(frame, tongue, (int(mw * 0.65), max(4, int(mh * 0.45))), math.degrees(angle), 0, 360, (90, 80, 245), -1, cv2.LINE_AA)
        elif pucker > 0.45:
            cv2.circle(frame, (mx, my), max(7, int(r * 0.13)), (50, 40, 120), -1, cv2.LINE_AA)
        else:
            # Sorriso o linea morbida
            curve = max(-0.1, smile - 0.12)
            pts = []
            for i in range(-5, 6):
                t = i / 5.0
                dx = t * r * (0.36 + smile * 0.14)
                dy = r * 0.34 + (1.0 - t * t) * r * (curve * 0.38)
                pts.append(_rot_pt(cx, cy, dx, dy, angle))
            cv2.polylines(frame, [np.array(pts, np.int32)], False, (35, 30, 50), max(4, int(r * 0.075)), cv2.LINE_AA)

    @staticmethod
    def _draw_robot_avatar(
        frame: np.ndarray,
        cx: float,
        cy: float,
        r: float,
        angle: float,
        bs: Dict[str, float],
    ) -> None:
        jaw = bs.get("jawOpen", 0.0)
        smile = 0.5 * (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0))
        blink_l = bs.get("eyeBlinkLeft", 0.0)
        blink_r = bs.get("eyeBlinkRight", 0.0)

        # Testa robotica ottagonale/rettangolare ruotata con la testa
        corners = [
            _rot_pt(cx, cy, -r * 0.92, -r * 0.85, angle),
            _rot_pt(cx, cy, r * 0.92, -r * 0.85, angle),
            _rot_pt(cx, cy, r * 0.92, r * 0.85, angle),
            _rot_pt(cx, cy, -r * 0.92, r * 0.85, angle),
        ]
        poly = np.array(corners, np.int32)
        cv2.fillPoly(frame, [poly], (24, 34, 52), cv2.LINE_AA)
        cv2.polylines(frame, [poly], True, COLOR_CYAN, 4, cv2.LINE_AA)

        # Antenna LED superiore
        a_base = _rot_pt(cx, cy, 0, -r * 0.85, angle)
        a_tip = _rot_pt(cx, cy, 0, -r * 1.20, angle)
        cv2.line(frame, a_base, a_tip, COLOR_CYAN, 4, cv2.LINE_AA)
        cv2.circle(frame, a_tip, max(6, int(r * 0.11)), COLOR_YELLOW if jaw > 0.25 else COLOR_MAGENTA, -1, cv2.LINE_AA)

        # Occhi LED neon
        for side, blink in ((-1, blink_r), (1, blink_l)):
            ex, ey = _rot_pt(cx, cy, side * r * 0.38, -r * 0.18, angle)
            ew = max(8, int(r * 0.22))
            eh = max(3, int(r * 0.16 * max(0.15, 1.0 - blink)))
            eye_col = COLOR_GREEN if smile > 0.45 else COLOR_CYAN
            cv2.ellipse(frame, (ex, ey), (ew, eh), math.degrees(angle), 0, 360, eye_col, -1, cv2.LINE_AA)
            cv2.ellipse(frame, (ex, ey), (ew, eh), math.degrees(angle), 0, 360, COLOR_WHITE, 2, cv2.LINE_AA)

        # Bocca equalizzatore digitale (5 barre che reagiscono a jawOpen!)
        for i in range(-2, 3):
            bar_h = max(6, int(r * (0.10 + jaw * 0.36 * (1.1 - abs(i) * 0.18))))
            p_top = _rot_pt(cx, cy, i * r * 0.20, r * 0.36 - bar_h * 0.5, angle)
            p_bot = _rot_pt(cx, cy, i * r * 0.20, r * 0.36 + bar_h * 0.5, angle)
            cv2.line(frame, p_top, p_bot, COLOR_YELLOW if jaw > 0.3 else COLOR_CYAN, max(5, int(r * 0.11)), cv2.LINE_AA)

    @staticmethod
    def _draw_alien_avatar(
        frame: np.ndarray,
        cx: float,
        cy: float,
        r: float,
        angle: float,
        bs: Dict[str, float],
    ) -> None:
        jaw = bs.get("jawOpen", 0.0)
        smile = 0.5 * (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0))
        blink_l = bs.get("eyeBlinkLeft", 0.0)
        blink_r = bs.get("eyeBlinkRight", 0.0)

        icx, icy = int(cx), int(cy)
        # Testa aliena verde fluo
        cv2.ellipse(frame, (icx, icy), (int(r * 0.92), int(r * 1.08)), math.degrees(angle), 0, 360, (55, 225, 110), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (icx, icy), (int(r * 0.92), int(r * 1.08)), math.degrees(angle), 0, 360, COLOR_GREEN, 4, cv2.LINE_AA)

        # Due antenne oscillanti
        wobble = math.sin(time.perf_counter() * 5.0) * r * 0.08
        for side in (-1, 1):
            b = _rot_pt(cx, cy, side * r * 0.36, -r * 0.92, angle)
            t = _rot_pt(cx, cy, side * r * 0.58 + wobble, -r * 1.32, angle)
            cv2.line(frame, b, t, (40, 190, 90), 4, cv2.LINE_AA)
            cv2.circle(frame, t, max(6, int(r * 0.12)), COLOR_MAGENTA, -1, cv2.LINE_AA)

        # Grandi occhi alieni obliqui neri lucidi
        for side, blink in ((-1, blink_r), (1, blink_l)):
            ex, ey = _rot_pt(cx, cy, side * r * 0.36, -r * 0.12, angle)
            ew = max(10, int(r * 0.28))
            eh = max(3, int(r * 0.20 * max(0.15, 1.0 - blink)))
            eye_ang = math.degrees(angle) + (22 * side)
            cv2.ellipse(frame, (ex, ey), (ew, eh), eye_ang, 0, 360, (15, 18, 25), -1, cv2.LINE_AA)
            if blink < 0.5:
                glint = _rot_pt(ex, ey, -ew * 0.25, -eh * 0.22, angle)
                cv2.circle(frame, glint, max(3, ew // 4), COLOR_WHITE, -1, cv2.LINE_AA)

        # Bocca aliena
        mx, my = _rot_pt(cx, cy, 0, r * 0.44, angle)
        if jaw > 0.22:
            cv2.ellipse(frame, (mx, my), (max(8, int(r * 0.22)), max(8, int(r * (0.10 + jaw * 0.34)))), math.degrees(angle), 0, 360, (20, 45, 25), -1, cv2.LINE_AA)
        else:
            pts = [
                _rot_pt(cx, cy, -r * 0.28, r * 0.42, angle),
                _rot_pt(cx, cy, 0, r * (0.45 + smile * 0.16), angle),
                _rot_pt(cx, cy, r * 0.28, r * 0.42, angle),
            ]
            cv2.polylines(frame, [np.array(pts, np.int32)], False, (18, 45, 22), max(4, int(r * 0.07)), cv2.LINE_AA)

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]
        ts = int((time.perf_counter() - self._start_time) * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, ts)

        faces_count = len(result.face_landmarks) if result.face_landmarks else 0
        for idx in range(faces_count):
            lms = result.face_landmarks[idx]
            bs: Dict[str, float] = {}
            if result.face_blendshapes and idx < len(result.face_blendshapes):
                bs = {cat.category_name: float(cat.score) for cat in result.face_blendshapes[idx]}

            # Centro volto, raggio e inclinazione testa (occhi: 33 e 263, fronte: 10, mento: 152)
            p_top = (lms[10].x * w, lms[10].y * h)
            p_bot = (lms[152].x * w, lms[152].y * h)
            p_left = (lms[234].x * w, lms[234].y * h)
            p_right = (lms[454].x * w, lms[454].y * h)
            p_eye_l = (lms[33].x * w, lms[33].y * h)
            p_eye_r = (lms[263].x * w, lms[263].y * h)

            cx = 0.25 * (p_top[0] + p_bot[0] + p_left[0] + p_right[0])
            cy = 0.25 * (p_top[1] + p_bot[1] + p_left[1] + p_right[1])
            face_h = math.hypot(p_bot[0] - p_top[0], p_bot[1] - p_top[1])
            face_w = math.hypot(p_right[0] - p_left[0], p_right[1] - p_left[1])
            radius = max(face_h, face_w) * 0.62
            angle = math.atan2(p_eye_r[1] - p_eye_l[1], p_eye_r[0] - p_eye_l[0])

            if debug_mode:
                # Modalita X-RAY: mostra i 468 punti della Face Mesh sul volto!
                for lm in lms[::2]:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, COLOR_CYAN, -1)
            else:
                if self.style == "emoji":
                    self._draw_classic_emoji(frame, cx, cy, radius, angle, bs)
                elif self.style == "robot":
                    self._draw_robot_avatar(frame, cx, cy, radius, angle, bs)
                else:
                    self._draw_alien_avatar(frame, cx, cy, radius, angle, bs)

            # Badge espressione rilevata sotto il volto
            expr_txt, expr_col = self._expression_label(bs)
            put_text_centered(
                frame,
                expr_txt,
                (int(cx), int(min(h - 52, cy + radius + 24))),
                scale=0.62,
                color=expr_col,
                thickness=2,
            )

        # 3 pulsanti touch-free in alto al centro per cambiare stile avatar col dito!
        style_specs = [
            ("emoji", "EMOJI", COLOR_YELLOW),
            ("robot", "ROBOT", COLOR_CYAN),
            ("alien", "ALIEN", COLOR_GREEN),
        ]
        tw, gap = 155, 12
        total_w = len(style_specs) * tw + (len(style_specs) - 1) * gap
        sx = (w - total_w) // 2
        buttons = [
            DwellButton(
                btn_id=sid,
                x1=sx + i * (tw + gap),
                y1=74,
                x2=sx + i * (tw + gap) + tw,
                y2=118,
                label=lbl,
                color=col,
                dwell_sec=0.55,
                active=(sid == self.style),
                label_scale=0.58,
            )
            for i, (sid, lbl, col) in enumerate(style_specs)
        ]

        hit = self.dwell.update_and_draw(frame, buttons, hands, dt)
        if hit in self.STYLES:
            self.style = hit

        return f"Volti rilevati: {faces_count}/4  |  Sorridi, fai l'occhiolino o apri la bocca! Punta in alto per cambiare avatar"

    def close(self) -> None:
        self._landmarker.close()
