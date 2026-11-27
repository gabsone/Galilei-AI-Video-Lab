"""Modalità 11: NEON DRIVE 3D — Corsa autostradale 3D Synthwave col VOLANTE IN ARIA!"""

from __future__ import annotations

import math
import random
import time
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
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_panel,
    put_text_centered,
)
from app.sound_engine import SoundEngine


@dataclass
class RoadEntity:
    lane: float       # -0.68 (sinistra), 0.0 (centro), +0.68 (destra)
    z: float          # 0.02 (orizzonte) .. 1.0 (giocatore)
    kind: str         # "car" oppure "turbo"
    color: Tuple[int, int, int]
    hit: bool = False


class NeonDrive3DMode:
    """Corsa pseudo-3D OutRun/Synthwave: guida con 2 mani a volante (o 1 mano Sx/Dx)."""

    LANES = (-0.65, 0.0, 0.65)

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self.dwell = DwellController(sound)
        self.best_score = 0
        self.reset()

    def reset(self) -> None:
        self.player_x = 0.0       # -0.95 .. +0.95
        self.steer_angle = 0.0    # radianti per visualizzare il volante
        self.speed = 0.58         # unita z/s
        self.turbo_timer = 0.0
        self.shields = 3
        self.invuln_timer = 0.0
        self.score = 0.0
        self.distance_km = 0.0
        self.road_scroll = 0.0
        self.curve = 0.0
        self.curve_target = 0.0
        self._curve_timer = 3.0
        self._spawn_timer = 0.6
        self.entities: List[RoadEntity] = []
        self.flash_red = 0.0
        self.game_over = False

    def handle_key(self, key: int) -> None:
        if key == ord(" ") and self.game_over:
            self.reset()
            self.sound.play("go")

    def _spawn_entity(self) -> None:
        lane = random.choice(self.LANES)
        if random.random() < 0.28:
            self.entities.append(RoadEntity(lane=lane, z=0.02, kind="turbo", color=COLOR_YELLOW))
        else:
            car_col = random.choice([COLOR_MAGENTA, COLOR_RED, COLOR_ORANGE, (255, 130, 80)])
            self.entities.append(RoadEntity(lane=lane, z=0.02, kind="car", color=car_col))

    @staticmethod
    def _project_road(z: float, lane_x: float, curve: float, w: int, horizon_y: int, bot_y: int) -> Tuple[int, int, float]:
        """Proietta una coordinata 3D (z in 0..1, lane_x in -1..1) in pixel (x, y, scale)."""
        # Prospettiva non lineare per vero effetto profondita 3D
        persp = z * z
        y = int(horizon_y + (bot_y - horizon_y) * persp)
        half_road_w = w * (0.045 + 0.44 * persp)
        # Curva dell'orizzonte (piu forte lontano dal giocatore)
        curve_shift = curve * ((1.0 - persp) ** 2) * w * 0.24
        x = int(w * 0.5 + curve_shift + lane_x * half_road_w)
        return x, y, persp

    @staticmethod
    def _draw_player_car(frame: np.ndarray, cx: int, cy: int, tilt: float, turbo: bool) -> None:
        """Disegna la supercar cyber vista da dietro con fari posteriori e scarichi nitro."""
        dx = int(tilt * 18)
        # Ombra sotto l'auto
        cv2.ellipse(frame, (cx, cy + 32), (82, 16), 0, 0, 360, (10, 10, 18), -1, cv2.LINE_AA)
        # Carrozzeria principale
        body = np.array([
            [cx - 72, cy + 26],
            [cx + 72, cy + 26],
            [cx + 60 + dx // 2, cy - 8],
            [cx - 60 + dx // 2, cy - 8],
        ], np.int32)
        cv2.fillPoly(frame, [body], (28, 36, 58), cv2.LINE_AA)
        cv2.polylines(frame, [body], True, COLOR_CYAN, 3, cv2.LINE_AA)

        # Abitacolo/tetto
        roof = np.array([
            [cx - 44 + dx // 2, cy - 8],
            [cx + 44 + dx // 2, cy - 8],
            [cx + 30 + dx, cy - 34],
            [cx - 30 + dx, cy - 34],
        ], np.int32)
        cv2.fillPoly(frame, [roof], (18, 24, 40), cv2.LINE_AA)
        cv2.polylines(frame, [roof], True, COLOR_CYAN, 2, cv2.LINE_AA)

        # Barra fari posteriori neon rossi/magenta
        cv2.rectangle(frame, (cx - 56, cy + 2), (cx - 24, cy + 12), COLOR_RED, -1)
        cv2.rectangle(frame, (cx + 24, cy + 2), (cx + 56, cy + 12), COLOR_RED, -1)
        cv2.line(frame, (cx - 24, cy + 7), (cx + 24, cy + 7), COLOR_MAGENTA, 2, cv2.LINE_AA)

        # Scarichi Nitro quando TURBO attivo!
        if turbo:
            flame = random.randint(18, 36)
            for ex in (cx - 34, cx + 34):
                pts = np.array([[ex - 9, cy + 26], [ex + 9, cy + 26], [ex, cy + 26 + flame]], np.int32)
                cv2.fillPoly(frame, [pts], COLOR_YELLOW, cv2.LINE_AA)
                pts2 = np.array([[ex - 5, cy + 26], [ex + 5, cy + 26], [ex, cy + 26 + flame // 2]], np.int32)
                cv2.fillPoly(frame, [pts2], COLOR_WHITE, cv2.LINE_AA)

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]
        horizon_y = int(h * 0.42)
        bot_y = h - 38

        # --- CONTROLLO STERZO (2 mani = VOLANTE, 1 mano = posizione Sx/Dx) ---
        wheel_center = None
        wheel_radius = 75
        if len(hands) >= 2:
            # Ordina mano sinistra e destra
            h_sorted = sorted(hands[:2], key=lambda hd: hd.palm_center[0])
            p_left = h_sorted[0].palm_center
            p_right = h_sorted[1].palm_center
            raw_ang = math.atan2(p_right[1] - p_left[1], max(25, p_right[0] - p_left[0]))
            self.steer_angle = max(-0.85, min(0.85, raw_ang * 1.45))
            wheel_center = ((p_left[0] + p_right[0]) // 2, (p_left[1] + p_right[1]) // 2)
            wheel_radius = max(48, min(115, int(math.hypot(p_right[0] - p_left[0], p_right[1] - p_left[1]) * 0.46)))
            if not self.game_over:
                self.player_x += self.steer_angle * dt * 2.15
        elif len(hands) == 1:
            # 1 mano: posizione orizzontale controlla direttamente la corsia
            target_x = ((hands[0].palm_center[0] / max(1, w)) - 0.5) * 2.1
            self.steer_angle = max(-0.8, min(0.8, (target_x - self.player_x) * 1.4))
            if not self.game_over:
                self.player_x += (target_x - self.player_x) * min(1.0, dt * 8.0)
        else:
            self.steer_angle *= max(0.0, 1.0 - dt * 5.0)

        self.player_x = max(-0.92, min(0.92, self.player_x))

        # --- LOGICA DI GIOCO 3D ---
        if not self.game_over:
            self.turbo_timer = max(0.0, self.turbo_timer - dt)
            self.invuln_timer = max(0.0, self.invuln_timer - dt)

            base_speed = min(0.95, 0.56 + self.distance_km * 0.025)
            cur_speed = base_speed * (1.55 if self.turbo_timer > 0 else 1.0)
            self.speed = cur_speed
            self.road_scroll = (self.road_scroll + cur_speed * dt * 2.6) % 1.0
            self.distance_km += cur_speed * dt * 0.42
            self.score += cur_speed * dt * (120 if self.turbo_timer > 0 else 65)
            self.best_score = max(self.best_score, int(self.score))

            # Curve dinamiche dell'autostrada
            self._curve_timer -= dt
            if self._curve_timer <= 0:
                self.curve_target = random.uniform(-0.85, 0.85)
                self._curve_timer = random.uniform(2.5, 4.8)
            self.curve += (self.curve_target - self.curve) * min(1.0, dt * 1.1)

            # Spawn auto e anelli turbo
            self._spawn_timer -= dt
            if self._spawn_timer <= 0:
                self._spawn_entity()
                self._spawn_timer = max(0.45, random.uniform(0.65, 1.15) - self.distance_km * 0.015)

            # Aggiorna entita e collisioni
            alive_entities: List[RoadEntity] = []
            for ent in self.entities:
                ent.z += cur_speed * dt * (0.72 if ent.kind == "car" else 0.92)
                if not ent.hit and 0.84 <= ent.z <= 0.98 and abs(ent.lane - self.player_x) < 0.34:
                    ent.hit = True
                    if ent.kind == "turbo":
                        self.turbo_timer = 3.2
                        self.score += 150
                        self.sound.play("combo")
                    elif self.invuln_timer <= 0:
                        self.shields -= 1
                        self.invuln_timer = 1.25
                        self.flash_red = 0.45
                        self.sound.play("bomb")
                        if self.shields <= 0:
                            self.game_over = True
                            self.sound.play("lose")
                if ent.z < 1.12 and not ent.hit:
                    alive_entities.append(ent)
            self.entities = alive_entities

        # --- RENDERING SCENA 3D SYNTHWAVE (mantenendo la webcam visibile in alto!) ---
        # Leggera tinta tramonto sopra l'orizzonte (si vede il ragazzo che guida!)
        sky_roi = frame[66:horizon_y, :]
        sky_tint = np.full_like(sky_roi, (65, 18, 55))
        cv2.addWeighted(sky_tint, 0.36, sky_roi, 0.64, 0, sky_roi)

        # Sole Synthwave all'orizzonte
        sun_cx = int(w * 0.5 + self.curve * w * 0.12)
        sun_r = int(min(w, h) * 0.11)
        cv2.circle(frame, (sun_cx, horizon_y - 6), sun_r, (40, 150, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (sun_cx, horizon_y - 6), int(sun_r * 0.78), (50, 220, 255), -1, cv2.LINE_AA)
        for stripe in range(0, sun_r, 12):
            sy = horizon_y - 4 - stripe // 2
            cv2.line(frame, (sun_cx - sun_r, sy), (sun_cx + sun_r, sy), (50, 18, 52), 2)

        # Terreno neon sotto l'orizzonte
        ground = frame[horizon_y:bot_y, :]
        g_tint = np.full_like(ground, (26, 12, 34))
        cv2.addWeighted(g_tint, 0.78, ground, 0.22, 0, ground)

        # Disegna l'autostrada 3D a segmenti prospettici
        num_seg = 22
        for i in range(num_seg):
            z1 = i / num_seg
            z2 = (i + 1) / num_seg
            lx1, y1, _ = self._project_road(z1, -1.0, self.curve, w, horizon_y, bot_y)
            rx1, _, _ = self._project_road(z1, 1.0, self.curve, w, horizon_y, bot_y)
            lx2, y2, _ = self._project_road(z2, -1.0, self.curve, w, horizon_y, bot_y)
            rx2, _, _ = self._project_road(z2, 1.0, self.curve, w, horizon_y, bot_y)

            band = int((1.0 - z1) * 14 + self.road_scroll * 4) % 2
            asphalt = (34, 34, 48) if band == 0 else (26, 26, 38)
            poly = np.array([[lx1, y1], [rx1, y1], [rx2, y2], [lx2, y2]], np.int32)
            cv2.fillPoly(frame, [poly], asphalt)

            # Cordoli neon laterali
            edge_col = COLOR_MAGENTA if band == 0 else COLOR_CYAN
            cv2.line(frame, (lx1, y1), (lx2, y2), edge_col, 4, cv2.LINE_AA)
            cv2.line(frame, (rx1, y1), (rx2, y2), edge_col, 4, cv2.LINE_AA)

            # Linee corsie centrali tratteggiate
            if band == 0:
                for lane_div in (-0.33, 0.33):
                    dx1, _, _ = self._project_road(z1, lane_div, self.curve, w, horizon_y, bot_y)
                    dx2, _, _ = self._project_road(z2, lane_div, self.curve, w, horizon_y, bot_y)
                    cv2.line(frame, (dx1, y1), (dx2, y2), (160, 185, 220), max(1, int(z2 * 4)), cv2.LINE_AA)

        # Disegna auto rivali e anelli turbo ordinati dal lontano al vicino (painter's algorithm!)
        for ent in sorted(self.entities, key=lambda e: e.z):
            ex, ey, sc = self._project_road(ent.z, ent.lane, self.curve, w, horizon_y, bot_y)
            if ent.kind == "turbo":
                r = max(6, int(44 * sc))
                cv2.circle(frame, (ex, ey - r), r, COLOR_YELLOW, max(2, int(6 * sc)), cv2.LINE_AA)
                cv2.circle(frame, (ex, ey - r), max(3, r // 2), COLOR_WHITE, 2, cv2.LINE_AA)
                if sc > 0.25:
                    put_text_centered(frame, "NITRO", (ex, ey - r), scale=0.45 * sc + 0.2, color=COLOR_YELLOW, thickness=1, shadow=False)
            else:
                cw = max(12, int(115 * sc))
                ch = max(8, int(64 * sc))
                cv2.rectangle(frame, (ex - cw // 2, ey - ch), (ex + cw // 2, ey), ent.color, -1)
                cv2.rectangle(frame, (ex - cw // 2, ey - ch), (ex + cw // 2, ey), COLOR_WHITE, max(1, int(2 * sc)))
                # Lunotto + fari rossi
                cv2.rectangle(frame, (ex - cw // 3, ey - int(ch * 0.85)), (ex + cw // 3, ey - int(ch * 0.45)), (20, 22, 34), -1)
                cv2.circle(frame, (ex - cw // 3, ey - ch // 4), max(2, int(7 * sc)), COLOR_RED, -1)
                cv2.circle(frame, (ex + cw // 3, ey - ch // 4), max(2, int(7 * sc)), COLOR_RED, -1)

        # Auto del giocatore (lampeggia se invulnerabile)
        px, py, _ = self._project_road(0.92, self.player_x, self.curve, w, horizon_y, bot_y)
        if self.invuln_timer <= 0 or int(time.perf_counter() * 14) % 2 == 0:
            self._draw_player_car(frame, px, py - 18, self.steer_angle, self.turbo_timer > 0)

        # Volante olografico tra le 2 mani!
        if wheel_center is not None:
            wx, wy = wheel_center
            cv2.circle(frame, (wx, wy), wheel_radius, COLOR_CYAN, 4, cv2.LINE_AA)
            cv2.circle(frame, (wx, wy), 14, COLOR_YELLOW, 2, cv2.LINE_AA)
            # Razze del volante ruotate secondo steer_angle
            for base_a in (0.0, math.pi, math.pi * 0.5):
                a = base_a + self.steer_angle
                sx = int(wx + math.cos(a) * wheel_radius)
                sy = int(wy + math.sin(a) * wheel_radius)
                cv2.line(frame, (wx, wy), (sx, sy), COLOR_CYAN, 3, cv2.LINE_AA)

        # Flash rosso urto
        if self.flash_red > 0:
            self.flash_red = max(0.0, self.flash_red - dt * 1.4)
            red = np.full_like(frame, (0, 0, 220))
            cv2.addWeighted(red, self.flash_red * 0.55, frame, 1.0 - self.flash_red * 0.55, 0, frame)

        # HUD Tachimetro & Scudi
        kmh = int(self.speed * 265)
        draw_panel(frame, 18, 76, 265, 178, border_color=COLOR_YELLOW if self.turbo_timer > 0 else COLOR_CYAN, alpha=0.80)
        cv2.putText(frame, f"{kmh:3d} KM/H", (34, 114), cv2.FONT_HERSHEY_DUPLEX, 0.92, COLOR_YELLOW if self.turbo_timer > 0 else COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"PUNTI : {int(self.score)}", (34, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_GREEN, 2, cv2.LINE_AA)
        scudi_txt = "SCUDI : " + "O " * self.shields + ". " * (3 - self.shields)
        cv2.putText(frame, scudi_txt, (34, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.54, COLOR_CYAN, 1, cv2.LINE_AA)

        # Game Over con pulsante touch-free RIGIOCA!
        if self.game_over:
            cx, cy = w // 2, h // 2
            draw_panel(frame, cx - 265, cy - 125, cx + 265, cy + 125, border_color=COLOR_RED, alpha=0.90, thickness=3)
            put_text_centered(frame, "GARA TERMINATA!", (cx, cy - 72), scale=1.18, color=COLOR_RED, thickness=3)
            put_text_centered(frame, f"PUNTI: {int(self.score)}   |   RECORD: {self.best_score}", (cx, cy - 14), scale=0.75, color=COLOR_WHITE, thickness=2)
            btn = DwellButton("drive_replay", cx - 150, cy + 30, cx + 150, cy + 98, "RIGIOCA!", COLOR_GREEN, active=True, label_scale=0.8)
            if self.dwell.update_and_draw(frame, [btn], hands, dt) == "drive_replay":
                self.reset()
                self.sound.play("go")

        return "Tieni 2 MANI COME UN VOLANTE per sterzare (o 1 mano Sx/Dx)! Prendi gli anelli NITRO e schiva le auto!"
