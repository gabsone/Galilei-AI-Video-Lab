"""Modalità 8: SUPERPOTERI AR — Kamehameha a 2 mani & Iron Man Repulsor!"""

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
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_hand_skeleton,
    put_text_centered,
)
from app.sound_engine import SoundEngine


@dataclass
class EnergySpark:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float
    max_life: float


@dataclass
class BlastWave:
    x: int
    y: int
    radius: float
    max_radius: float
    color: Tuple[int, int, int]


ELEMENTS = {
    "plasma": ("PLASMA", COLOR_CYAN, (255, 160, 40)),
    "fuoco": ("FUOCO", COLOR_ORANGE, COLOR_RED),
    "fulmine": ("FULMINE", COLOR_YELLOW, COLOR_MAGENTA),
}


class SuperpowersMode:
    """Carica una sfera di energia tra le 2 mani (Kamehameha) e scatenala aprendole di scatto!"""

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self.dwell = DwellController(sound)
        self.element = "plasma"
        self.reset()

    def reset(self) -> None:
        self.charge = 0.0          # 0.0 .. 1.0
        self.last_center = (640, 360)
        self.sparks: List[EnergySpark] = []
        self.waves: List[BlastWave] = []
        self.flash_alpha = 0.0
        self.blast_banner_timer = 0.0
        self._last_repulsor_time = 0.0
        self._last_charge_beep = 0.0

    def handle_key(self, key: int) -> None:
        if key == ord(" "):
            keys = list(ELEMENTS.keys())
            self.element = keys[(keys.index(self.element) + 1) % len(keys)]
            self.sound.play("switch")

    def _emit_sparks(self, x: int, y: int, color: Tuple[int, int, int], count: int = 10, speed_mult: float = 1.0) -> None:
        for _ in range(count):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(80, 360) * speed_mult
            life = random.uniform(0.25, 0.60)
            self.sparks.append(EnergySpark(x, y, math.cos(ang) * spd, math.sin(ang) * spd, color, life, life))

    @staticmethod
    def _draw_lightning(frame: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        pts = [p1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = max(1.0, math.hypot(dx, dy))
        nx, ny = -dy / length, dx / length
        for i in range(1, 10):
            t = i / 10.0
            jit = random.uniform(-22, 22) * math.sin(t * math.pi)
            pts.append((int(p1[0] + dx * t + nx * jit), int(p1[1] + dy * t + ny * jit)))
        pts.append(p2)
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, 5, cv2.LINE_AA)
            cv2.line(frame, pts[i], pts[i + 1], COLOR_WHITE, 2, cv2.LINE_AA)

    def _trigger_super_blast(self, cx: int, cy: int, primary: Tuple[int, int, int], secondary: Tuple[int, int, int]) -> None:
        self.waves.append(BlastWave(cx, cy, 35.0, 560.0, primary))
        self.waves.append(BlastWave(cx, cy, 15.0, 420.0, secondary))
        self._emit_sparks(cx, cy, primary, count=45, speed_mult=1.7)
        self._emit_sparks(cx, cy, COLOR_WHITE, count=20, speed_mult=1.4)
        self.flash_alpha = 0.55
        self.blast_banner_timer = 1.4
        self.sound.play("bomb")
        self.sound.play("combo")

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]
        now = time.perf_counter()
        _, col_primary, col_secondary = ELEMENTS[self.element]

        # 3 pulsanti touch-free in alto per scegliere l'elemento (PLASMA / FUOCO / FULMINE)
        elem_specs = [
            ("plasma", "PLASMA", COLOR_CYAN),
            ("fuoco", "FUOCO", COLOR_ORANGE),
            ("fulmine", "FULMINE", COLOR_YELLOW),
        ]
        tw, gap = 150, 12
        sx = (w - (3 * tw + 2 * gap)) // 2
        buttons = [
            DwellButton(eid, sx + i * (tw + gap), 74, sx + i * (tw + gap) + tw, 118, lbl, col, active=(eid == self.element), label_scale=0.56)
            for i, (eid, lbl, col) in enumerate(elem_specs)
        ]
        hit_elem = self.dwell.update_and_draw(frame, buttons, hands, dt)
        if hit_elem in ELEMENTS:
            self.element = hit_elem

        if len(hands) >= 2:
            p1 = hands[0].palm_center
            p2 = hands[1].palm_center
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            self.last_center = mid

            if dist < w * 0.28:
                # Mani vicine -> CARICA LA SFERA KAMEHAMEHA!
                self.charge = min(1.0, self.charge + dt * 0.72)
                if now - self._last_charge_beep > max(0.10, 0.34 - self.charge * 0.22):
                    self._last_charge_beep = now
                    self.sound.play("hud_grab")

                # Fulmini tra i palmi
                self._draw_lightning(frame, p1, p2, col_primary)
                if self.charge > 0.45:
                    self._draw_lightning(frame, hands[0].index_tip, hands[1].index_tip, col_secondary)

                # Sfera energetica pulsante crescente
                orb_r = int(26 + self.charge * 92 + math.sin(now * 24) * 6)
                cv2.circle(frame, mid, orb_r + 18, col_secondary, 3, cv2.LINE_AA)
                cv2.circle(frame, mid, orb_r + 8, col_primary, 5, cv2.LINE_AA)
                overlay = frame.copy()
                cv2.circle(overlay, mid, orb_r, col_primary, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
                cv2.circle(frame, mid, max(12, int(orb_r * 0.55)), COLOR_WHITE, -1, cv2.LINE_AA)
                self._emit_sparks(mid[0], mid[1], col_primary, count=4)

                put_text_centered(
                    frame,
                    f"CARICA {int(self.charge * 100)}% - APRI LE MANI PER LANCIARE!" if self.charge > 0.6 else f"CARICAMENTO {int(self.charge * 100)}%...",
                    (mid[0], max(145, mid[1] - orb_r - 24)),
                    scale=0.62,
                    color=COLOR_YELLOW,
                    thickness=2,
                )
            else:
                # Mani allontanate di scatto dopo la carica -> ESPLOSIONE!
                if self.charge >= 0.48:
                    self._trigger_super_blast(mid[0], mid[1], col_primary, col_secondary)
                self.charge = max(0.0, self.charge - dt * 2.5)
        else:
            # 1 Mano: Repulsor Iron Man sul palmo aperto!
            if self.charge >= 0.52:
                self._trigger_super_blast(self.last_center[0], self.last_center[1], col_primary, col_secondary)
            self.charge = max(0.0, self.charge - dt * 1.8)

            if len(hands) == 1:
                hand = hands[0]
                pc = hand.palm_center
                if hand.gesture == "CARTA":
                    r = int(34 + math.sin(now * 18) * 5)
                    cv2.circle(frame, pc, r + 12, col_secondary, 2, cv2.LINE_AA)
                    cv2.circle(frame, pc, r, col_primary, 4, cv2.LINE_AA)
                    cv2.circle(frame, pc, r // 2, COLOR_WHITE, -1, cv2.LINE_AA)
                    self._emit_sparks(pc[0], pc[1], col_primary, count=2)
                    if now - self._last_repulsor_time > 1.2:
                        self._last_repulsor_time = now
                        self.waves.append(BlastWave(pc[0], pc[1], 24.0, 220.0, col_primary))
                        self.sound.play("slice")

        # Scheletri mano discreti (o completi in X-Ray)
        if debug_mode:
            for idx, hand in enumerate(hands):
                draw_hand_skeleton(frame, hand, accent=col_primary if idx == 0 else col_secondary, show_ids=True)

        # Onde d'urto
        alive_waves: List[BlastWave] = []
        for wave in self.waves:
            wave.radius += 780 * dt
            if wave.radius < wave.max_radius:
                cv2.circle(frame, (wave.x, wave.y), int(wave.radius), wave.color, 6, cv2.LINE_AA)
                cv2.circle(frame, (wave.x, wave.y), max(2, int(wave.radius - 10)), COLOR_WHITE, 2, cv2.LINE_AA)
                alive_waves.append(wave)
        self.waves = alive_waves

        # Particelle
        alive_sparks: List[EnergySpark] = []
        for s in self.sparks:
            s.life -= dt
            if s.life > 0:
                s.x += s.vx * dt
                s.y += s.vy * dt
                cv2.circle(frame, (int(s.x), int(s.y)), max(2, int(7 * s.life / s.max_life)), s.color, -1, cv2.LINE_AA)
                alive_sparks.append(s)
        self.sparks = alive_sparks

        # Flash energetico
        if self.flash_alpha > 0:
            self.flash_alpha = max(0.0, self.flash_alpha - dt * 1.4)
            flash = np.full_like(frame, col_primary)
            cv2.addWeighted(flash, self.flash_alpha * 0.55, frame, 1.0 - self.flash_alpha * 0.55, 0, frame)

        if self.blast_banner_timer > 0:
            self.blast_banner_timer = max(0.0, self.blast_banner_timer - dt)
            put_text_centered(frame, "SUPER RAGGIO SCATENATO!", (w // 2, h // 2), scale=1.35, color=COLOR_YELLOW, thickness=4)

        return "Avvicina 2 mani per caricare l'energia e APRILE DI SCATTO per l'esplosione! (1 mano aperta = Repulsor)"
