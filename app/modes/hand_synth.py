"""Modalità 3: AI MUSIC STUDIO — DJ 4-Zone + Pianoforte Aereo + Batteria Aerea (Air Drums)."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.dwell_ui import DwellButton, DwellController
from app.hand_tracker import HandData
from app.hud import (
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_MAGENTA,
    COLOR_MUTED,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    draw_hand_skeleton,
    draw_panel,
    put_text_centered,
)
from app.sound_engine import SoundEngine

ITALIAN_NOTE_NAMES = [
    "DO", "DO#", "RE", "RE#", "MI", "FA", "FA#", "SOL", "SOL#", "LA", "LA#", "SI", "DO5"
]

# Melodie guidate per il Pianoforte Aereo (solo tasti bianchi giganti: DO=0, RE=2, MI=4, FA=5, SOL=7, LA=9, SI=11, DO5=12)
GUIDED_SONGS: List[Tuple[str, List[int]]] = [
    ("LIBERO", []),
    (
        "INNO ALLA GIOIA",
        [4, 4, 5, 7, 7, 5, 4, 2, 0, 0, 2, 4, 4, 2, 2],
    ),
    (
        "SEVEN NATION ARMY",
        [9, 9, 12, 9, 7, 5, 4, 9, 9, 12, 9, 7, 5, 7, 5, 4],
    ),
    (
        "FRA MARTINO",
        [0, 2, 4, 0, 0, 2, 4, 0, 4, 5, 7, 4, 5, 7],
    ),
]


@dataclass
class TrackPad:
    track_id: str
    title: str
    subtitle: str
    color: Tuple[int, int, int]
    volume: float
    enabled: bool
    knob_x_norm: float = 0.5


@dataclass
class MusicSpark:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float
    max_life: float


@dataclass
class DrumWave:
    x: int
    y: int
    radius: float
    max_radius: float
    color: Tuple[int, int, int]


class HandSynthMode:
    """Studio musicale a 3 schermate: DJ 4-Zone, Pianoforte Aereo e Batteria Aerea."""

    SUBMODES = ("studio", "piano", "drums")

    def __init__(self, sound: SoundEngine) -> None:
        self.sound = sound
        self.dwell = DwellController(sound)
        self.submode = "studio"
        self.song_idx = 1  # Parte con guida "Inno alla Gioia" pronta sul piano
        self.reset()

    def reset(self) -> None:
        self.tracks: List[TrackPad] = [
            TrackPad("drums", "BATTERIA", "Kick · Snare · Hi-Hat", COLOR_CYAN, 0.72, True, 0.5),
            TrackPad("bass", "BASSO SYNTH", "Groove 116 BPM", COLOR_MAGENTA, 0.0, False, 0.5),
            TrackPad("piano", "ACCORDI PIANO", "Am · F · C · G", COLOR_GREEN, 0.60, True, 0.5),
            TrackPad("arp", "SYNTH LEAD", "Arpeggio Elettronico", COLOR_ORANGE, 0.0, False, 0.5),
        ]
        # Stato Studio DJ
        self._grabbed_track: Dict[int, Optional[int]] = {0: None, 1: None}
        self._last_pinch_state: Dict[int, bool] = {0: False, 1: False}

        # Stato Pianoforte Aereo (con rilevamento colpetto verso il basso + pinch + guida)
        self._piano_prev_y: Dict[int, Optional[int]] = {0: None, 1: None}
        self._piano_prev_pinch: Dict[int, bool] = {0: False, 1: False}
        self._piano_prev_key: Dict[int, int] = {0: -1, 1: -1}
        self._piano_cooldown: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._key_flash: Dict[int, float] = {}
        self._song_step = 0
        self._song_complete_flash = 0.0

        # Stato Batteria Aerea (Air Drumstick Strike: velocità verticale verso il basso)
        self._drum_prev_tip: Dict[int, Optional[Tuple[int, int]]] = {0: None, 1: None}
        self._drum_cooldown: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._drum_pad_flash: Dict[str, float] = {}
        self._drum_waves: List[DrumWave] = []
        self._drum_hits = 0

        self._sparks: List[MusicSpark] = []
        self._toggle_cooldown = 0.0
        self._beat_start = time.perf_counter()

        if self.submode == "studio":
            self._sync_all_volumes()

    def on_enter(self) -> None:
        if self.submode == "studio":
            self._start_loops()

    def on_exit(self) -> None:
        self.sound.stop_studio_loops()

    def _start_loops(self) -> None:
        vols = {t.track_id: (t.volume if t.enabled else 0.0) for t in self.tracks}
        self.sound.start_studio_loops(vols)
        self._sync_all_volumes()

    def _sync_all_volumes(self) -> None:
        for t in self.tracks:
            self.sound.set_track_volume(t.track_id, t.volume if t.enabled else 0.0, morph=t.knob_x_norm)

    def set_submode(self, target: str) -> None:
        if target == self.submode or target not in self.SUBMODES:
            return
        self.submode = target
        if target == "studio":
            self._start_loops()
        else:
            self.sound.stop_studio_loops()
        self._grabbed_track = {0: None, 1: None}
        self._piano_prev_y = {0: None, 1: None}
        self._drum_prev_tip = {0: None, 1: None}
        self.sound.play("switch")

    def cycle_submode(self) -> None:
        idx = (self.SUBMODES.index(self.submode) + 1) % len(self.SUBMODES)
        self.set_submode(self.SUBMODES[idx])

    def handle_key(self, key: int) -> None:
        if key in (ord("p"), ord("P")):
            self.cycle_submode()
        elif key in (ord("g"), ord("G")) and self.submode == "piano":
            self.song_idx = (self.song_idx + 1) % len(GUIDED_SONGS)
            self._song_step = 0
            self.sound.play("switch")

    def _emit_sparks(self, x: int, y: int, color: Tuple[int, int, int], count: int = 8) -> None:
        for _ in range(count):
            ang = random.uniform(-math.pi, 0)
            spd = random.uniform(70, 260)
            life = random.uniform(0.28, 0.55)
            self._sparks.append(
                MusicSpark(x, y, math.cos(ang) * spd, math.sin(ang) * spd - 40, color, life, life)
            )

    def _draw_sparks(self, frame: np.ndarray, dt: float) -> None:
        alive: List[MusicSpark] = []
        for s in self._sparks:
            s.life -= dt
            if s.life > 0:
                s.x += s.vx * dt
                s.y += s.vy * dt
                r = max(2, int(6 * (s.life / s.max_life)))
                cv2.circle(frame, (int(s.x), int(s.y)), r, s.color, -1, cv2.LINE_AA)
                alive.append(s)
        self._sparks = alive

    # =========================================================================
    # SUBMODE 1: STUDIO DJ 4-ZONE (XY: Volume + DEEP <-> HYPER)
    # =========================================================================
    def _update_and_draw_studio(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> None:
        h, w = frame.shape[:2]
        self._start_loops()

        margin_x, top_y, bot_y, gap = 24, 126, h - 54, 18
        mid_x = w // 2
        mid_y = (top_y + bot_y) // 2

        rects = [
            (margin_x, top_y, mid_x - gap // 2, mid_y - gap // 2),
            (mid_x + gap // 2, top_y, w - margin_x, mid_y - gap // 2),
            (margin_x, mid_y + gap // 2, mid_x - gap // 2, bot_y),
            (mid_x + gap // 2, mid_y + gap // 2, w - margin_x, bot_y),
        ]

        for h_idx in range(2):
            if h_idx >= len(hands):
                self._grabbed_track[h_idx] = None
                self._last_pinch_state[h_idx] = False
                continue

            hand = hands[h_idx]
            px = (hand.index_tip[0] + hand.thumb_tip[0]) // 2
            py = (hand.index_tip[1] + hand.thumb_tip[1]) // 2

            hovered_pad: Optional[int] = None
            for p_idx, (x1, y1, x2, y2) in enumerate(rects):
                if x1 <= px <= x2 and y1 <= py <= y2:
                    hovered_pad = p_idx
                    break

            if hand.is_pinching:
                if self._grabbed_track[h_idx] is None and hovered_pad is not None:
                    self._grabbed_track[h_idx] = hovered_pad
                    self.sound.play("hud_grab")

                g_idx = self._grabbed_track[h_idx]
                if g_idx is not None:
                    x1, y1, x2, y2 = rects[g_idx]
                    pad = self.tracks[g_idx]
                    slider_top = y1 + 54
                    slider_bot = y2 - 28
                    vol = 1.0 - (py - slider_top) / max(1.0, slider_bot - slider_top)
                    vol = max(0.0, min(1.0, vol))
                    pad.volume = vol
                    pad.enabled = vol > 0.05
                    pad.knob_x_norm = max(0.0, min(1.0, (px - (x1 + 26)) / max(1.0, (x2 - 68) - (x1 + 26))))
                    self.sound.set_track_volume(
                        pad.track_id,
                        pad.volume if pad.enabled else 0.0,
                        morph=pad.knob_x_norm,
                    )
                    self._emit_sparks(px, py, pad.color, count=2)
            else:
                if self._grabbed_track[h_idx] is not None:
                    self._grabbed_track[h_idx] = None
                    self.sound.play("hud_release")

            self._last_pinch_state[h_idx] = hand.is_pinching

        beat_phase = ((time.perf_counter() - self._beat_start) * (self.sound.STUDIO_BPM / 60.0)) % 1.0
        pulse = math.exp(-beat_phase * 5.0)

        for idx, (pad, (x1, y1, x2, y2)) in enumerate(zip(self.tracks, rects)):
            is_grabbed = idx in self._grabbed_track.values()
            active = pad.enabled and pad.volume > 0.05
            border_col = COLOR_YELLOW if is_grabbed else (pad.color if active else (70, 80, 100))

            draw_panel(
                frame, x1, y1, x2, y2,
                bg_color=(20, 26, 40) if active else (14, 16, 24),
                border_color=border_col,
                alpha=0.58 if active else 0.45,
                thickness=3 if is_grabbed else (2 if active else 1),
            )

            cv2.putText(frame, pad.title, (x1 + 18, y1 + 34), cv2.FONT_HERSHEY_DUPLEX, 0.78, pad.color if active else COLOR_MUTED, 2, cv2.LINE_AA)
            cv2.putText(frame, pad.subtitle, (x1 + 18, y1 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_MUTED, 1, cv2.LINE_AA)

            vol_pct = int(pad.volume * 100) if active else 0
            status_txt = f"{vol_pct}%" if active else "OFF"
            cv2.putText(frame, status_txt, (x2 - 86, y1 + 38), cv2.FONT_HERSHEY_DUPLEX, 0.82, COLOR_YELLOW if is_grabbed else (COLOR_WHITE if active else COLOR_MUTED), 2, cv2.LINE_AA)

            slider_top = y1 + 66
            slider_bot = y2 - 22
            bar_x1, bar_x2 = x2 - 42, x2 - 20
            cv2.rectangle(frame, (bar_x1, slider_top), (bar_x2, slider_bot), (36, 42, 60), -1)
            if active:
                fill_y = int(slider_bot - (slider_bot - slider_top) * pad.volume)
                cv2.rectangle(frame, (bar_x1, fill_y), (bar_x2, slider_bot), pad.color, -1)
            cv2.rectangle(frame, (bar_x1, slider_top), (bar_x2, slider_bot), border_col, 1)

            fx_x1, fx_x2 = x1 + 22, x2 - 62
            fx_y = y2 - 18
            cv2.line(frame, (fx_x1, fx_y), (fx_x2, fx_y), (55, 65, 85), 3, cv2.LINE_AA)
            fx_pos = int(fx_x1 + (fx_x2 - fx_x1) * pad.knob_x_norm)
            cv2.line(frame, (fx_x1, fx_y), (fx_pos, fx_y), pad.color if active else (70, 80, 100), 3, cv2.LINE_AA)
            cv2.circle(frame, (fx_pos, fx_y), 6, COLOR_YELLOW if is_grabbed else (COLOR_WHITE if active else COLOR_MUTED), -1, cv2.LINE_AA)
            cv2.putText(frame, "DEEP", (fx_x1, fx_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, "HYPER", (fx_x2 - 42, fx_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_YELLOW if (active and pad.knob_x_norm > 0.65) else COLOR_MUTED, 1, cv2.LINE_AA)

            knob_x = int(fx_x1 + (fx_x2 - fx_x1) * pad.knob_x_norm)
            knob_y = int(slider_bot - 16 - (slider_bot - slider_top - 22) * (pad.volume if active else 0.08))
            base_r = int(20 + 22 * pad.volume + (12 * pulse * pad.volume if active else 0))

            if active:
                cv2.circle(frame, (knob_x, knob_y), base_r + 10, pad.color, 2, cv2.LINE_AA)
            cv2.circle(frame, (knob_x, knob_y), base_r, (26, 32, 48), -1, cv2.LINE_AA)
            cv2.circle(frame, (knob_x, knob_y), base_r, COLOR_YELLOW if is_grabbed else border_col, 3, cv2.LINE_AA)
            fx_label = "DEEP" if pad.knob_x_norm < 0.33 else ("HYPER" if pad.knob_x_norm > 0.67 else "MIX")
            put_text_centered(
                frame, "PINCH" if not active else fx_label, (knob_x, knob_y),
                scale=0.48, color=COLOR_YELLOW if (active and fx_label == "HYPER") else (COLOR_WHITE if active else COLOR_MUTED),
                thickness=1, shadow=False,
            )

        for idx, hand in enumerate(hands):
            accent = COLOR_YELLOW if hand.is_pinching else (COLOR_CYAN if idx == 0 else COLOR_MAGENTA)
            draw_hand_skeleton(frame, hand, accent=accent, show_ids=debug_mode)
            px = (hand.index_tip[0] + hand.thumb_tip[0]) // 2
            py = (hand.index_tip[1] + hand.thumb_tip[1]) // 2
            cv2.circle(frame, (px, py), 12 if hand.is_pinching else 8, accent, 3, cv2.LINE_AA)

    # =========================================================================
    # SUBMODE 2: PIANOFORTE / XILOFONO NEON A 8 TASTI GIGANTI (Tocco Istantaneo!)
    # =========================================================================
    def _update_and_draw_piano(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> None:
        h, w = frame.shape[:2]

        # 8 tasti larghi a tutta larghezza (impossibili da mancare su un 60 pollici!)
        kb_x1, kb_x2 = 34, w - 34
        kb_y1, kb_y2 = int(h * 0.42), h - 58
        kb_w = kb_x2 - kb_x1

        white_indices = [0, 2, 4, 5, 7, 9, 11, 12]  # DO, RE, MI, FA, SOL, LA, SI, DO5
        key_colors = [
            (70, 80, 255),   # DO  - Rosso/Corallo
            (40, 155, 255),  # RE  - Arancio
            (40, 225, 255),  # MI  - Giallo
            (90, 255, 130),  # FA  - Verde
            (255, 220, 40),  # SOL - Ciano
            (255, 140, 60),  # LA  - Blu elettrico
            (230, 80, 255),  # SI  - Viola/Magenta
            (130, 160, 255), # DO5 - Rosa neon
        ]
        white_w = kb_w / len(white_indices)

        white_rects: Dict[int, Tuple[int, int, int, int]] = {}
        key_color_map: Dict[int, Tuple[int, int, int]] = {}
        for pos, (note_idx, col) in enumerate(zip(white_indices, key_colors)):
            x1 = int(kb_x1 + pos * white_w) + 3
            x2 = int(kb_x1 + (pos + 1) * white_w) - 3
            white_rects[note_idx] = (x1, kb_y1, x2, kb_y2)
            key_color_map[note_idx] = col

        song_name, song_notes = GUIDED_SONGS[self.song_idx]
        target_note = song_notes[self._song_step] if song_notes else -1

        for k in list(self._key_flash.keys()):
            self._key_flash[k] -= dt
            if self._key_flash[k] <= 0:
                del self._key_flash[k]
        self._song_complete_flash = max(0.0, self._song_complete_flash - dt)

        hovered_keys: set[int] = set()
        for h_idx in range(2):
            self._piano_cooldown[h_idx] = max(0.0, self._piano_cooldown[h_idx] - dt)
            if h_idx >= len(hands):
                self._piano_prev_y[h_idx] = None
                self._piano_prev_pinch[h_idx] = False
                self._piano_prev_key[h_idx] = -1
                continue

            hand = hands[h_idx]
            ix, iy = hand.index_tip

            hit_key = -1
            for note_idx, (x1, y1, x2, y2) in white_rects.items():
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    hit_key = note_idx
                    break

            prev_y = self._piano_prev_y[h_idx]
            vy = (iy - prev_y) / max(0.001, dt) if prev_y is not None else 0.0
            just_entered = hit_key != -1 and hit_key != self._piano_prev_key[h_idx]
            just_pinched = hand.is_pinching and not self._piano_prev_pinch[h_idx]
            soft_tap = hit_key != -1 and vy > 115.0

            if hit_key != -1:
                hovered_keys.add(hit_key)
                # TOCCO ISTANTANEO appena entri nel tasto, oppure colpetto leggero / pinch per ribattere!
                if self._piano_cooldown[h_idx] <= 0.0 and (just_entered or soft_tap or just_pinched):
                    self.sound.play_piano_note(hit_key, volume=0.95)
                    self._key_flash[hit_key] = 0.30
                    self._piano_cooldown[h_idx] = 0.16
                    col = COLOR_GREEN if hit_key == target_note else key_color_map[hit_key]
                    self._emit_sparks(ix, iy, col, count=14)

                    if song_notes and hit_key == target_note:
                        self._song_step += 1
                        if self._song_step >= len(song_notes):
                            self._song_step = 0
                            self._song_complete_flash = 1.6
                            self.sound.play("win")

            self._piano_prev_y[h_idx] = iy
            self._piano_prev_pinch[h_idx] = hand.is_pinching
            self._piano_prev_key[h_idx] = hit_key

        # Banner guida melodia sopra la tastiera (tocco touch-free per cambiare canzone!)
        cx = w // 2
        if self._song_complete_flash > 0:
            draw_panel(frame, cx - 270, 126, cx + 270, 180, border_color=COLOR_GREEN, alpha=0.86)
            put_text_centered(frame, "MELODIA COMPLETATA! BRAVISSIMO!", (cx, 153), scale=0.72, color=COLOR_GREEN, thickness=2)
        else:
            song_label = (
                f"MELODIA: {song_name} ({self._song_step + 1}/{len(song_notes)}) -> {ITALIAN_NOTE_NAMES[target_note]}"
                if song_notes
                else "PIANO LIBERO  |  Punta qui per Guida Melodie"
            )
            guide_btn = DwellButton(
                btn_id="next_song",
                x1=cx - 295,
                y1=124,
                x2=cx + 295,
                y2=178,
                label=song_label,
                color=COLOR_YELLOW if song_notes else COLOR_CYAN,
                dwell_sec=0.50,
                active=bool(song_notes),
                label_scale=0.54,
            )
            if self.dwell.update_and_draw(frame, [guide_btn], hands, dt) == "next_song":
                self.song_idx = (self.song_idx + 1) % len(GUIDED_SONGS)
                self._song_step = 0

        # Disegna gli 8 tasti giganti stile Xilofono/Piano Neon
        for note_idx, (x1, y1, x2, y2) in white_rects.items():
            base_col = key_color_map[note_idx]
            is_lit = note_idx in self._key_flash
            is_target = note_idx == target_note
            is_hover = note_idx in hovered_keys

            if is_lit:
                bg, border, alpha = base_col, COLOR_WHITE, 0.90
            elif is_target:
                bg, border, alpha = (35, 195, 255), COLOR_YELLOW, 0.78
            elif is_hover:
                bg, border, alpha = base_col, COLOR_WHITE, 0.52
            else:
                bg, border, alpha = (24, 30, 46), base_col, 0.62

            draw_panel(
                frame, x1, y1, x2, y2,
                bg_color=bg, border_color=border, alpha=alpha,
                thickness=4 if (is_lit or is_target) else 2,
            )
            # Fascia superiore colorata stile tasto musicale
            cv2.rectangle(frame, (x1 + 3, y1 + 3), (x2 - 3, y1 + 20), base_col, -1)
            put_text_centered(
                frame,
                ITALIAN_NOTE_NAMES[note_idx],
                ((x1 + x2) // 2, y2 - 38),
                scale=0.95,
                color=(15, 20, 30) if (is_lit or is_target) else COLOR_WHITE,
                thickness=2,
                shadow=not (is_lit or is_target),
            )

        # Freccia gialla rimbalzante sul tasto da suonare
        if target_note in white_rects:
            tx1, ty1, tx2, _ = white_rects[target_note]
            bounce = int(math.sin(time.perf_counter() * 8.0) * 9)
            tx = (tx1 + tx2) // 2
            ty = ty1 - 16 + bounce
            pts = np.array([[tx - 20, ty - 22], [tx + 20, ty - 22], [tx, ty]], np.int32)
            cv2.fillPoly(frame, [pts], COLOR_YELLOW, cv2.LINE_AA)

        for idx, hand in enumerate(hands):
            accent = COLOR_CYAN if idx == 0 else COLOR_MAGENTA
            draw_hand_skeleton(frame, hand, accent=accent, show_ids=debug_mode)
            cv2.circle(frame, hand.index_tip, 15, COLOR_YELLOW, 3, cv2.LINE_AA)

    # =========================================================================
    # SUBMODE 3: BATTERIA AEREA (Air Drumstick Strike — colpo veloce verso il basso!)
    # =========================================================================
    def _update_and_draw_drums(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> None:
        h, w = frame.shape[:2]

        # 6 pad disposti ad arco ergonomico da batterista
        pad_r = max(56, min(78, int(min(w, h) * 0.105)))
        pads = [
            ("crash", "CRASH", (int(w * 0.18), int(h * 0.30)), pad_r, COLOR_YELLOW),
            ("clap", "CLAP", (int(w * 0.82), int(h * 0.30)), pad_r, COLOR_MAGENTA),
            ("hihat", "HI-HAT", (int(w * 0.16), int(h * 0.58)), pad_r, COLOR_CYAN),
            ("tom", "TOM", (int(w * 0.84), int(h * 0.58)), pad_r, COLOR_ORANGE),
            ("snare", "SNARE", (int(w * 0.34), int(h * 0.78)), pad_r + 6, COLOR_GREEN),
            ("kick", "KICK", (int(w * 0.66), int(h * 0.78)), pad_r + 10, COLOR_RED),
        ]

        # Decadimento flash dei pad
        for pid in list(self._drum_pad_flash.keys()):
            self._drum_pad_flash[pid] -= dt
            if self._drum_pad_flash[pid] <= 0:
                del self._drum_pad_flash[pid]

        # Calcola le punte delle bacchette virtuali e rileva il COLPO VERSO IL BASSO (Strike!)
        stick_tips: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
        for h_idx in range(2):
            self._drum_cooldown[h_idx] = max(0.0, self._drum_cooldown[h_idx] - dt)
            if h_idx >= len(hands):
                self._drum_prev_tip[h_idx] = None
                continue

            hand = hands[h_idx]
            wrist = hand.landmarks_px[0]
            # La bacchetta parte dal polso e si estende oltre la nocca dell'indice/medio (funziona anche a pugno chiuso!)
            knuckle = hand.landmarks_px[9]
            dx = knuckle[0] - wrist[0]
            dy = knuckle[1] - wrist[1]
            tip_x = int(max(0, min(w - 1, wrist[0] + dx * 2.15)))
            tip_y = int(max(0, min(h - 1, wrist[1] + dy * 2.15)))
            tip = (tip_x, tip_y)

            prev = self._drum_prev_tip[h_idx]
            vy = (tip_y - prev[1]) / max(0.001, dt) if prev is not None else 0.0
            stick_tips.append((wrist, tip, vy))

            # Gesture dedicata: colpo deciso verso il basso (vy > 260 px/s) dentro un pad!
            if self._drum_cooldown[h_idx] <= 0.0 and vy > 260.0:
                for pid, _, center, radius, col in pads:
                    if math.hypot(tip_x - center[0], tip_y - center[1]) <= radius + 24:
                        vel = max(0.4, min(1.0, (vy - 220.0) / 950.0))
                        self.sound.play_drum(pid, velocity=vel)
                        self._drum_pad_flash[pid] = 0.24
                        self._drum_cooldown[h_idx] = 0.13
                        self._drum_hits += 1
                        self._drum_waves.append(DrumWave(center[0], center[1], float(radius), radius + 95 * vel, col))
                        self._emit_sparks(center[0], center[1], col, count=int(14 + 14 * vel))
                        break

            self._drum_prev_tip[h_idx] = tip

        # Disegna i 6 pad della batteria
        for pid, label, (cx, cy), radius, col in pads:
            flash = self._drum_pad_flash.get(pid, 0.0)
            r = int(radius + flash * 38)
            fill_col = col if flash > 0.08 else (22, 26, 40)
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), r, fill_col, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.62 if flash > 0 else 0.45, frame, 0.38 if flash > 0 else 0.55, 0, frame)
            cv2.circle(frame, (cx, cy), r, COLOR_WHITE if flash > 0.1 else col, 4 if flash > 0 else 3, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), int(r * 0.72), col, 1, cv2.LINE_AA)
            put_text_centered(frame, label, (cx, cy), scale=0.72, color=(15, 20, 30) if flash > 0.12 else COLOR_WHITE, thickness=2, shadow=flash <= 0.12)

        # Onde d'urto dei colpi
        alive_waves: List[DrumWave] = []
        for wave in self._drum_waves:
            wave.radius += 520 * dt
            if wave.radius < wave.max_radius:
                cv2.circle(frame, (wave.x, wave.y), int(wave.radius), wave.color, 3, cv2.LINE_AA)
                alive_waves.append(wave)
        self._drum_waves = alive_waves

        # Disegna le bacchette neon + scheletri
        for idx, hand in enumerate(hands):
            accent = COLOR_CYAN if idx == 0 else COLOR_MAGENTA
            if debug_mode:
                draw_hand_skeleton(frame, hand, accent=accent, show_ids=True)
            wrist, tip, vy = stick_tips[idx]
            glow_col = COLOR_YELLOW if vy > 260.0 else accent
            cv2.line(frame, wrist, tip, glow_col, 10, cv2.LINE_AA)
            cv2.line(frame, wrist, tip, COLOR_WHITE, 3, cv2.LINE_AA)
            cv2.circle(frame, tip, 12, glow_col, -1, cv2.LINE_AA)
            cv2.circle(frame, tip, 6, COLOR_WHITE, -1, cv2.LINE_AA)

        # Contatore colpi al centro in alto
        put_text_centered(
            frame,
            f"COLPI: {self._drum_hits}   |   Colpisci VERSO IL BASSO con le bacchette!",
            (w // 2, 146),
            scale=0.60,
            color=COLOR_YELLOW,
            thickness=2,
        )

    # =========================================================================
    # ENTRY POINT MODALITÀ 3
    # =========================================================================
    def update_and_draw(
        self,
        frame: np.ndarray,
        hands: List[HandData],
        dt: float,
        debug_mode: bool,
    ) -> str:
        w = frame.shape[1]
        self._toggle_cooldown = max(0.0, self._toggle_cooldown - dt)

        # Barra 3 sotto-modalità in alto al centro (100% touch-free: basta puntare l'indice!)
        sub_tabs = [
            ("studio", "DJ 4-ZONE", COLOR_GREEN),
            ("piano", "PIANOFORTE", COLOR_MAGENTA),
            ("drums", "BATTERIA", COLOR_ORANGE),
        ]
        tw, gap = 175, 12
        total_w = len(sub_tabs) * tw + (len(sub_tabs) - 1) * gap
        sx = (w - total_w) // 2

        tab_buttons = [
            DwellButton(
                btn_id=sid,
                x1=sx + i * (tw + gap),
                y1=72,
                x2=sx + i * (tw + gap) + tw,
                y2=114,
                label=label,
                color=col,
                dwell_sec=0.55,
                active=(sid == self.submode),
                label_scale=0.54,
            )
            for i, (sid, label, col) in enumerate(sub_tabs)
        ]

        if self.submode == "studio":
            self._update_and_draw_studio(frame, hands, dt, debug_mode)
            self._draw_sparks(frame, dt)
            hint = "PINCH nei riquadri: Su/Giu = Volume, Sx/Dx = DEEP <-> HYPER  |  Punta in alto per cambiare strumento"
        elif self.submode == "piano":
            self._update_and_draw_piano(frame, hands, dt, debug_mode)
            self._draw_sparks(frame, dt)
            hint = "Colpetto giu o PINCH sui tasti per suonare  |  Punta in alto per cambiare melodia o strumento"
        else:
            self._update_and_draw_drums(frame, hands, dt, debug_mode)
            self._draw_sparks(frame, dt)
            hint = "AIR DRUMS: colpisci VERSO IL BASSO sui pad con le bacchette!  |  Punta in alto per cambiare strumento"

        hit_tab = self.dwell.update_and_draw(frame, tab_buttons, hands, dt)
        if hit_tab in self.SUBMODES:
            self.set_submode(hit_tab)

        return hint
