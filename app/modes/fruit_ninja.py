"""Modalità 1: CYBER SLICE (Fruit Ninja con l'indice di una o due mani)."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import cv2
import numpy as np

from app.hand_tracker import HandData
from app.hud import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_hand_skeleton,
    draw_panel,
    put_text_centered,
)
from app.sound_engine import SoundEngine


@dataclass
class TargetOrb:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    color: Tuple[int, int, int]
    label: str
    is_bomb: bool = False
    is_bonus: bool = False
    sliced: bool = False


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float
    max_life: float
    radius: int


@dataclass
class FloatingText:
    x: float
    y: float
    text: str
    color: Tuple[int, int, int]
    life: float


class FruitNinjaMode:
    """Mini-gioco arcade da 45 secondi: taglia le sfere tech, evita i virus rossi!"""

    ROUND_DURATION = 45.0
    GRAVITY_FACTOR = 0.95  # Gravita proporzionale all'altezza schermo (arco ampio e fluido)

    ORB_STYLES = [
        ("AI", COLOR_CYAN),
        ("PY", COLOR_GREEN),
        ("CPU", COLOR_ORANGE),
        ("GPU", COLOR_MAGENTA),
        ("BIT", COLOR_YELLOW),
    ]

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self.best_score = 0
        self.reset()

    def reset(self) -> None:
        self.score = 0
        self.combo = 0
        self.last_slice_time = 0.0
        self.time_left = self.ROUND_DURATION
        self.game_over = False
        self.targets: List[TargetOrb] = []
        self.particles: List[Particle] = []
        self.popups: List[FloatingText] = []
        self.trails: List[Deque[Tuple[int, int]]] = [deque(maxlen=14), deque(maxlen=14)]
        self._spawn_timer = 0.4
        self._flash_red = 0.0

    def handle_key(self, key: int) -> None:
        if key == ord(" ") and self.game_over:
            self.reset()
            self.sound.play("go")

    def _spawn_wave(self, w: int, h: int) -> None:
        gravity = h * self.GRAVITY_FACTOR
        count = random.choices([1, 2, 3], weights=[0.45, 0.4, 0.15])[0]
        for _ in range(count):
            x = random.uniform(w * 0.15, w * 0.85)
            y = h + 40.0
            # Calcola vy esatto per far salire l'oggetto fino alla fascia alta (13%..28% dall'alto)
            target_apex_y = random.uniform(h * 0.13, h * 0.28)
            rise = max(120.0, y - target_apex_y)
            vy = -math.sqrt(2.0 * gravity * rise)
            # Spinta orizzontale verso il centro cosi resta ben dentro lo schermo
            center_bias = (w * 0.5 - x) * random.uniform(0.30, 0.60)
            vx = center_bias + random.uniform(-85, 85)

            roll = random.random()
            if roll < 0.18:
                self.targets.append(
                    TargetOrb(x, y, vx, vy, 36, COLOR_RED, "BUG!", is_bomb=True)
                )
            elif roll < 0.28:
                self.targets.append(
                    TargetOrb(x, y, vx, vy, 30, COLOR_YELLOW, "+25", is_bonus=True)
                )
            else:
                label, color = random.choice(self.ORB_STYLES)
                self.targets.append(
                    TargetOrb(x, y, vx, vy, random.randint(34, 42), color, label)
                )

    def _emit_particles(
        self, x: float, y: float, color: Tuple[int, int, int], count: int = 18
    ) -> None:
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(120, 480)
            life = random.uniform(0.32, 0.65)
            self.particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed - 60,
                    color=color,
                    life=life,
                    max_life=life,
                    radius=random.randint(3, 7),
                )
            )

    @staticmethod
    def _segment_circle_hit(
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        cx: float,
        cy: float,
        radius: float,
    ) -> bool:
        """Collisione segmento (lama del dito) - cerchio (bersaglio)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 25:  # Richiede un minimo movimento di taglio
            return False
        t = max(0.0, min(1.0, ((cx - p1[0]) * dx + (cy - p1[1]) * dy) / seg_len_sq))
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        return math.hypot(cx - closest_x, cy - closest_y) <= radius + 12

    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        h, w = frame.shape[:2]
        now = time.perf_counter()

        # Aggiorna scie delle dita (supporta fino a 2 mani contemporaneamente)
        active_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for idx in range(2):
            if idx < len(hands):
                tip = hands[idx].index_tip
                if self.trails[idx]:
                    prev = self.trails[idx][-1]
                    if math.hypot(tip[0] - prev[0], tip[1] - prev[1]) < 260:
                        active_segments.append((prev, tip))
                    else:
                        self.trails[idx].clear()
                self.trails[idx].append(tip)
            else:
                if self.trails[idx]:
                    self.trails[idx].popleft()

        # Disegna scheletro mano discreto (o completo in debug)
        for idx, hand in enumerate(hands):
            accent = COLOR_CYAN if idx == 0 else COLOR_MAGENTA
            draw_hand_skeleton(frame, hand, accent=accent, show_ids=debug_mode)

        # Logica di gioco
        if not self.game_over:
            self.time_left = max(0.0, self.time_left - dt)
            if self.time_left <= 0.0:
                self.game_over = True
                self.best_score = max(self.best_score, self.score)
                self.sound.play("win")

            self._spawn_timer -= dt
            if self._spawn_timer <= 0.0:
                self._spawn_wave(w, h)
                self._spawn_timer = random.uniform(0.75, 1.25)

        # Aggiorna bersagli e collisioni
        gravity = h * self.GRAVITY_FACTOR
        remaining_targets: List[TargetOrb] = []
        for orb in self.targets:
            orb.x += orb.vx * dt
            orb.y += orb.vy * dt
            orb.vy += gravity * dt

            if not self.game_over and not orb.sliced:
                for p1, p2 in active_segments:
                    if self._segment_circle_hit(p1, p2, orb.x, orb.y, orb.radius):
                        orb.sliced = True
                        if orb.is_bomb:
                            self.score = max(0, self.score - 20)
                            self.combo = 0
                            self._flash_red = 0.35
                            self._emit_particles(orb.x, orb.y, COLOR_RED, 28)
                            self.popups.append(FloatingText(orb.x, orb.y, "-20 BUG!", COLOR_RED, 0.9))
                            self.sound.play("bomb")
                        else:
                            if now - self.last_slice_time < 0.75:
                                self.combo += 1
                            else:
                                self.combo = 1
                            self.last_slice_time = now
                            base_pts = 25 if orb.is_bonus else 10
                            pts = base_pts * min(4, self.combo)
                            self.score += pts
                            self.best_score = max(self.best_score, self.score)
                            self._emit_particles(orb.x, orb.y, orb.color, 20)
                            label = f"+{pts} x{self.combo}!" if self.combo >= 2 else f"+{pts}"
                            self.popups.append(FloatingText(orb.x, orb.y, label, orb.color, 0.85))
                            self.sound.play("combo" if self.combo >= 2 else "slice")
                        break

            if not orb.sliced and orb.y < h + 110:
                remaining_targets.append(orb)
        self.targets = remaining_targets

        # Disegna bersagli neon
        for orb in self.targets:
            cx, cy = int(orb.x), int(orb.y)
            cv2.circle(frame, (cx, cy), orb.radius + 6, orb.color, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), orb.radius, (24, 26, 38), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), orb.radius, orb.color, 4, cv2.LINE_AA)
            if orb.is_bomb:
                r = orb.radius // 2
                cv2.line(frame, (cx - r, cy - r), (cx + r, cy + r), COLOR_RED, 4, cv2.LINE_AA)
                cv2.line(frame, (cx + r, cy - r), (cx - r, cy + r), COLOR_RED, 4, cv2.LINE_AA)
            else:
                put_text_centered(frame, orb.label, (cx, cy), scale=0.72, color=COLOR_WHITE, thickness=2)

        # Aggiorna e disegna particelle
        alive_particles: List[Particle] = []
        for p in self.particles:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx * dt
                p.y += p.vy * dt
                p.vy += 420 * dt
                r = max(1, int(p.radius * (p.life / p.max_life)))
                cv2.circle(frame, (int(p.x), int(p.y)), r, p.color, -1, cv2.LINE_AA)
                alive_particles.append(p)
        self.particles = alive_particles

        # Disegna scie spada laser
        for idx, trail in enumerate(self.trails):
            color = COLOR_CYAN if idx == 0 else COLOR_MAGENTA
            pts = list(trail)
            for i in range(1, len(pts)):
                thickness = max(2, int(14 * (i / len(pts))))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
                cv2.line(frame, pts[i - 1], pts[i], COLOR_WHITE, max(1, thickness // 3), cv2.LINE_AA)

        # Popup punteggio fluttuanti
        alive_popups: List[FloatingText] = []
        for pop in self.popups:
            pop.life -= dt
            if pop.life > 0:
                pop.y -= 75 * dt
                put_text_centered(frame, pop.text, (int(pop.x), int(pop.y)), scale=0.85, color=pop.color, thickness=2)
                alive_popups.append(pop)
        self.popups = alive_popups

        # Flash rosso se colpisce un BUG
        if self._flash_red > 0:
            self._flash_red = max(0.0, self._flash_red - dt)
            overlay = np.full_like(frame, (0, 0, 220))
            cv2.addWeighted(overlay, self._flash_red * 0.7, frame, 1.0 - self._flash_red * 0.7, 0, frame)

        # HUD punteggio e timer (in alto a destra)
        draw_panel(frame, w - 265, 76, w - 16, 178, border_color=COLOR_GREEN, alpha=0.76)
        cv2.putText(frame, f"PUNTI : {self.score}", (w - 248, 110), cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TEMPO : {int(math.ceil(self.time_left)):02d}s", (w - 248, 140), cv2.FONT_HERSHEY_DUPLEX, 0.72, COLOR_YELLOW if self.time_left <= 10 else COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"RECORD: {self.best_score}", (w - 248, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_CYAN, 1, cv2.LINE_AA)

        # Schermata di fine round
        if self.game_over:
            cx, cy = w // 2, h // 2
            draw_panel(frame, cx - 270, cy - 115, cx + 270, cy + 115, border_color=COLOR_YELLOW, alpha=0.88, thickness=3)
            put_text_centered(frame, "TEMPO SCADUTO!", (cx, cy - 60), scale=1.15, color=COLOR_YELLOW, thickness=3)
            put_text_centered(frame, f"PUNTEGGIO FINALE: {self.score}", (cx, cy + 2), scale=0.95, color=COLOR_WHITE, thickness=2)
            put_text_centered(frame, "Premi [SPAZIO] o [R] per un nuovo round", (cx, cy + 62), scale=0.62, color=COLOR_GREEN, thickness=1)

        return "Muovi l'INDICE per tagliare le sfere tech ed evitare i BUG rossi!"
