"""Motore audio procedurale (100% offline, zero file esterni) con pygame + numpy.

Include:
- Effetti arcade (slice, combo, bomb, countdown, win/lose)
- Effetti sci-fi stile Iron Man HUD (grab, release, zoom)
- Pianoforte cromatico completo (C4 -> C5, 13 tasti bianchi e neri)
- 4 tracce loop sincronizzate a 116 BPM (Drums, Bass, Piano Chords, Synth Arp)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False


class SoundEngine:
    """Genera effetti sonori, pianoforte polifonico e 4 loop musicali via DSP."""

    SAMPLE_RATE = 44100
    STUDIO_BPM = 116.0

    # Ottava cromatica completa C4 -> C5 (13 tasti)
    PIANO_NOTES = [
        ("C4", 261.63, False),
        ("C#4", 277.18, True),
        ("D4", 293.66, False),
        ("D#4", 311.13, True),
        ("E4", 329.63, False),
        ("F4", 349.23, False),
        ("F#4", 369.99, True),
        ("G4", 392.00, False),
        ("G#4", 415.30, True),
        ("A4", 440.00, False),
        ("A#4", 466.16, True),
        ("B4", 493.88, False),
        ("C5", 523.25, False),
    ]

    TRACK_IDS = ("drums", "bass", "piano", "arp")
    DJ_PACKS = [
        ("tetris", "TETRIS CYBER", 128.0),
        ("stranger", "STRANGER 80s", 116.0),
        ("electro", "CRAZY ELECTRO", 122.0),
    ]

    def __init__(self) -> None:
        self.muted = False
        self.enabled = False
        self._sounds: Dict[str, "pygame.mixer.Sound"] = {}
        self._drum_sounds: Dict[str, "pygame.mixer.Sound"] = {}
        self._piano_sounds: List["pygame.mixer.Sound"] = []
        self._pack_loops: Dict[str, Dict[str, "pygame.mixer.Sound"]] = {}
        self._pack_loops_hyper: Dict[str, Dict[str, "pygame.mixer.Sound"]] = {}
        self.current_pack_idx = 0
        self._loop_channels: Dict[str, "pygame.mixer.Channel"] = {}
        self._loop_channels_hyper: Dict[str, "pygame.mixer.Channel"] = {}
        self._loops_running = False
        self._track_volumes: Dict[str, float] = {k: 0.0 for k in self.TRACK_IDS}
        self._track_morphs: Dict[str, float] = {k: 0.5 for k in self.TRACK_IDS}

        if not PYGAME_AVAILABLE:
            return

        try:
            pygame.mixer.pre_init(self.SAMPLE_RATE, -16, 2, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(24)
            self.enabled = True
            self._build_sound_bank()
            self._build_drum_kit()
            self._build_piano_bank()
            self._build_studio_loops()
        except Exception:
            self.enabled = False

    def _to_sound(self, mono_wave: np.ndarray, volume: float = 0.5) -> "pygame.mixer.Sound":
        wave = np.clip(mono_wave * volume, -1.0, 1.0)
        pcm = (wave * 32767).astype(np.int16)
        stereo = np.column_stack((pcm, pcm))
        return pygame.sndarray.make_sound(stereo)

    @staticmethod
    def _synth_piano_tone(freq: float, duration: float, sr: int) -> np.ndarray:
        """Sintesi additiva di un timbro di pianoforte caldo e brillante."""
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        attack = np.minimum(1.0, t / 0.006)
        # Armoniche con decadimento progressivamente piu rapido sulle alte frequenze
        wave = (
            1.00 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 2.6)
            + 0.45 * np.sin(2 * np.pi * (2 * freq) * t) * np.exp(-t * 4.2)
            + 0.22 * np.sin(2 * np.pi * (3 * freq) * t) * np.exp(-t * 6.5)
            + 0.10 * np.sin(2 * np.pi * (4 * freq) * t) * np.exp(-t * 9.0)
            + 0.05 * np.sin(2 * np.pi * (5 * freq) * t) * np.exp(-t * 12.0)
        )
        # Leggero "martelletto" iniziale
        hammer = np.sin(2 * np.pi * (freq * 7.2) * t) * np.exp(-t * 45.0) * 0.08
        return (wave + hammer) * attack / 1.6

    def _build_sound_bank(self) -> None:
        sr = self.SAMPLE_RATE

        # 1. Slice / taglio frutto
        t = np.linspace(0, 0.14, int(sr * 0.14), endpoint=False)
        freq = np.linspace(620, 1450, t.size)
        self._sounds["slice"] = self._to_sound(
            np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 18), 0.45
        )

        # 2. Combo bonus
        t = np.linspace(0, 0.24, int(sr * 0.24), endpoint=False)
        combo_wave = (
            np.sin(2 * np.pi * 880 * t)
            + 0.7 * np.sin(2 * np.pi * 1318.5 * t)
            + 0.5 * np.sin(2 * np.pi * 1760 * t)
        ) / 2.2 * np.exp(-t * 10)
        self._sounds["combo"] = self._to_sound(combo_wave, 0.45)

        # 3. Bomba / errore
        t = np.linspace(0, 0.42, int(sr * 0.42), endpoint=False)
        rng = np.random.default_rng(42)
        noise = rng.uniform(-1.0, 1.0, t.size)
        sub_freq = np.linspace(150, 38, t.size)
        sub = np.sin(2 * np.pi * np.cumsum(sub_freq) / sr)
        self._sounds["bomb"] = self._to_sound((0.55 * sub + 0.45 * noise) * np.exp(-t * 6.5), 0.55)

        # 4. Beep conto alla rovescia
        t = np.linspace(0, 0.11, int(sr * 0.11), endpoint=False)
        self._sounds["beep"] = self._to_sound(np.sin(2 * np.pi * 660 * t) * np.exp(-t * 20), 0.4)

        # 5. Via! / Go!
        t = np.linspace(0, 0.22, int(sr * 0.22), endpoint=False)
        go_wave = (np.sin(2 * np.pi * 987.77 * t) + 0.5 * np.sin(2 * np.pi * 1479.98 * t)) / 1.5 * np.exp(-t * 11)
        self._sounds["go"] = self._to_sound(go_wave, 0.46)

        # 6. Vittoria
        segs = []
        for f in (523.25, 659.25, 783.99, 1046.50):
            ts = np.linspace(0, 0.09, int(sr * 0.09), endpoint=False)
            segs.append(np.sin(2 * np.pi * f * ts) * np.exp(-ts * 10))
        self._sounds["win"] = self._to_sound(np.concatenate(segs), 0.45)

        # 7. Sconfitta
        segs = []
        for f in (392.0, 293.66):
            ts = np.linspace(0, 0.16, int(sr * 0.16), endpoint=False)
            segs.append(np.sin(2 * np.pi * f * ts) * np.exp(-ts * 8))
        self._sounds["lose"] = self._to_sound(np.concatenate(segs), 0.42)

        # 8. Pareggio
        t = np.linspace(0, 0.18, int(sr * 0.18), endpoint=False)
        self._sounds["tie"] = self._to_sound(np.sin(2 * np.pi * 440 * t) * np.exp(-t * 11), 0.38)

        # 9. Cambio modalita UI
        t = np.linspace(0, 0.08, int(sr * 0.08), endpoint=False)
        sw_freq = np.linspace(480, 760, t.size)
        self._sounds["switch"] = self._to_sound(
            np.sin(2 * np.pi * np.cumsum(sw_freq) / sr) * np.exp(-t * 25), 0.3
        )

        # 10. Stark HUD Grab (chirp futuristico ascendente)
        t = np.linspace(0, 0.09, int(sr * 0.09), endpoint=False)
        f_grab = np.linspace(520, 1240, t.size)
        self._sounds["hud_grab"] = self._to_sound(
            np.sin(2 * np.pi * np.cumsum(f_grab) / sr) * np.exp(-t * 22), 0.28
        )

        # 11. Stark HUD Release (chirp discendente morbido)
        t = np.linspace(0, 0.08, int(sr * 0.08), endpoint=False)
        f_rel = np.linspace(920, 420, t.size)
        self._sounds["hud_release"] = self._to_sound(
            np.sin(2 * np.pi * np.cumsum(f_rel) / sr) * np.exp(-t * 24), 0.24
        )

    def _build_drum_kit(self) -> None:
        """Genera 6 suoni di batteria acustica/elettronica per la modalità Air Drums."""
        sr = self.SAMPLE_RATE
        rng = np.random.default_rng(777)

        # 1. KICK (Grancassa profonda e potente)
        t = np.linspace(0, 0.30, int(sr * 0.30), endpoint=False)
        kf = 160 * np.exp(-t * 28) + 44
        kick = np.tanh(np.sin(2 * np.pi * np.cumsum(kf) / sr) * 1.6) * np.exp(-t * 9.5)
        self._drum_sounds["kick"] = self._to_sound(kick, 0.90)

        # 2. SNARE (Rullante croccante)
        t = np.linspace(0, 0.22, int(sr * 0.22), endpoint=False)
        body = np.sin(2 * np.pi * (185 * np.exp(-t * 8) + 120) * t) * np.exp(-t * 18)
        wires = rng.uniform(-1.0, 1.0, t.size) * np.exp(-t * 16)
        self._drum_sounds["snare"] = self._to_sound(0.45 * body + 0.70 * wires, 0.78)

        # 3. HI-HAT (Charleston metallico veloce)
        t = np.linspace(0, 0.09, int(sr * 0.09), endpoint=False)
        metal = rng.uniform(-1.0, 1.0, t.size) * np.sin(2 * np.pi * 6400 * t)
        self._drum_sounds["hihat"] = self._to_sound(metal * np.exp(-t * 48), 0.65)

        # 4. TOM (Tamburo tonale caldo)
        t = np.linspace(0, 0.28, int(sr * 0.28), endpoint=False)
        tf = 190 * np.exp(-t * 14) + 105
        tom = np.sin(2 * np.pi * np.cumsum(tf) / sr) * np.exp(-t * 10)
        self._drum_sounds["tom"] = self._to_sound(tom, 0.80)

        # 5. CRASH (Piatto esplosivo lungo)
        t = np.linspace(0, 0.65, int(sr * 0.65), endpoint=False)
        crash_noise = rng.uniform(-1.0, 1.0, t.size)
        shimmer = (
            np.sin(2 * np.pi * 3120 * t)
            + np.sin(2 * np.pi * 4730 * t)
            + np.sin(2 * np.pi * 6890 * t)
        ) / 3.0
        crash = (0.72 * crash_noise + 0.28 * shimmer) * np.exp(-t * 4.8)
        self._drum_sounds["crash"] = self._to_sound(crash, 0.62)

        # 6. CLAP (Battito mani elettronico multi-impulso)
        t = np.linspace(0, 0.22, int(sr * 0.22), endpoint=False)
        clap = np.zeros_like(t)
        for delay in (0.0, 0.012, 0.024):
            mask = t >= delay
            dt_arr = t[mask] - delay
            clap[mask] += rng.uniform(-1.0, 1.0, dt_arr.size) * np.exp(-dt_arr * 28) * 0.55
        self._drum_sounds["clap"] = self._to_sound(clap, 0.72)

    def play_drum(self, drum_id: str, velocity: float = 0.85) -> None:
        if not self.enabled or self.muted:
            return
        snd = self._drum_sounds.get(drum_id)
        if snd is not None:
            snd.set_volume(max(0.25, min(1.0, velocity)))
            snd.play()

    def _build_piano_bank(self) -> None:
        sr = self.SAMPLE_RATE
        self._piano_sounds = []
        for _, freq, _ in self.PIANO_NOTES:
            tone = self._synth_piano_tone(freq, 0.85, sr)
            self._piano_sounds.append(self._to_sound(tone, 0.55))

    def _build_studio_loops(self) -> None:
        """Costruisce 3 DJ Pack famosi (4 tracce sincronizzate Deep+Hyper ciascuno)."""
        sr = self.SAMPLE_RATE
        rng = np.random.default_rng(123)

        # Campioni batteria condivisi
        tk = np.linspace(0, 0.18, int(sr * 0.18), endpoint=False)
        kf = 145 * np.exp(-tk * 26) + 46
        kick = np.sin(2 * np.pi * np.cumsum(kf) / sr) * np.exp(-tk * 12)
        ts = np.linspace(0, 0.15, int(sr * 0.15), endpoint=False)
        snare = (rng.uniform(-1, 1, ts.size) * 0.75 + np.sin(2 * np.pi * 190 * ts) * 0.35) * np.exp(-ts * 20)
        th = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
        hat = rng.uniform(-1, 1, th.size) * np.exp(-th * 75)

        def add_at(buf: np.ndarray, start_sec: float, clip: np.ndarray, gain: float = 1.0) -> None:
            idx = int(start_sec * sr)
            if idx >= len(buf):
                return
            end = min(len(buf), idx + len(clip))
            buf[idx:end] += clip[: end - idx] * gain

        def lowpass_smooth(sig: np.ndarray, window: int = 11) -> np.ndarray:
            kernel = np.ones(window, dtype=np.float64) / window
            return np.convolve(sig, kernel, mode="same")

        def synth_lead(freq: float, dur: float, bright: bool = False) -> np.ndarray:
            n = max(1, int(sr * dur))
            t = np.linspace(0, dur, n, endpoint=False)
            att = np.minimum(1.0, t / 0.008)
            rel = np.exp(-t * (5.5 if bright else 4.2))
            if bright:
                w = np.tanh((np.sin(2 * np.pi * freq * t) + 0.55 * np.sin(2 * np.pi * 2 * freq * t) + 0.3 * np.sin(2 * np.pi * 3 * freq * t)) * 1.5)
            else:
                w = 0.72 * np.sin(2 * np.pi * freq * t) + 0.28 * np.sin(2 * np.pi * 2 * freq * t)
            return w * att * rel

        def synth_bass(freq: float, dur: float) -> np.ndarray:
            n = max(1, int(sr * dur))
            t = np.linspace(0, dur, n, endpoint=False)
            return (
                0.65 * np.sin(2 * np.pi * freq * t)
                + 0.30 * np.sin(2 * np.pi * 2 * freq * t)
                + 0.15 * np.sin(2 * np.pi * 3 * freq * t)
            ) * np.minimum(1.0, t / 0.008) * np.exp(-t * 5.0)

        def build_drums(total_beats: int, beat_sec: float, total_samples: int) -> tuple[np.ndarray, np.ndarray]:
            d = np.zeros(total_samples, dtype=np.float64)
            for b in range(total_beats):
                t0 = b * beat_sec
                if b % 4 in (0, 2):
                    add_at(d, t0, kick, 0.95)
                if b % 4 in (1, 3):
                    add_at(d, t0, snare, 0.76)
                add_at(d, t0, hat, 0.32)
                add_at(d, t0 + beat_sec * 0.5, hat, 0.45)
            dh = d.copy()
            for b in range(total_beats):
                t0 = b * beat_sec
                add_at(dh, t0, kick, 0.45)  # four-on-the-floor EDM kick
                for sub in (0.25, 0.75):
                    add_at(dh, t0 + beat_sec * sub, hat, 0.52)
            return d, dh

        def register_pack(
            pack_id: str,
            drums: np.ndarray,
            drums_h: np.ndarray,
            bass: np.ndarray,
            piano: np.ndarray,
            piano_h: np.ndarray,
            lead: np.ndarray,
            lead_h: np.ndarray,
        ) -> None:
            self._pack_loops[pack_id] = {
                "drums": self._to_sound(lowpass_smooth(drums, 13), 0.78),
                "bass": self._to_sound(lowpass_smooth(bass, 15), 0.75),
                "piano": self._to_sound(lowpass_smooth(piano, 11), 0.75),
                "arp": self._to_sound(lowpass_smooth(lead, 9), 0.68),
            }
            self._pack_loops_hyper[pack_id] = {
                "drums": self._to_sound(drums_h, 0.75),
                "bass": self._to_sound(np.tanh(bass * 3.0) * 0.85, 0.68),
                "piano": self._to_sound(piano_h, 0.72),
                "arp": self._to_sound(lead_h, 0.62),
            }

        # =====================================================================
        # PACK 1: TETRIS CYBER (Korobeiniki EDM - 128 BPM, 16 beat / 4 battute)
        # =====================================================================
        beat = 60.0 / 128.0
        n_samp = int(sr * 16 * beat)
        d, dh = build_drums(16, beat, n_samp)
        bass = np.zeros(n_samp)
        piano = np.zeros(n_samp)
        piano_h = np.zeros(n_samp)
        lead = np.zeros(n_samp)
        lead_h = np.zeros(n_samp)

        # Accordi: Em - Am - Em - Am (classico Tetris)
        t_chords = [
            (164.81, 196.00, 246.94),  # Em
            (220.00, 261.63, 329.63),  # Am
            (164.81, 207.65, 246.94),  # E
            (220.00, 261.63, 329.63),  # Am
        ]
        t_roots = [82.41, 110.00, 82.41, 110.00]
        for bar in range(4):
            for s in range(8):
                mult = 2.0 if s % 2 == 1 else 1.0
                add_at(bass, (bar * 4 + s * 0.5) * beat, synth_bass(t_roots[bar] * mult, beat * 0.45), 0.68)
            for hit in (0.0, 2.0):
                cw = sum(self._synth_piano_tone(f, beat * 1.6, sr) for f in t_chords[bar]) / 3.0
                add_at(piano, (bar * 4 + hit) * beat, cw, 0.82)
            for hit in (0.0, 0.5, 1.5, 2.0, 2.5, 3.5):
                cw = sum(self._synth_piano_tone(f * 2.0, beat * 0.42, sr) for f in t_chords[bar]) / 3.0
                add_at(piano_h, (bar * 4 + hit) * beat, cw, 0.58)

        # Melodia iconica Tetris: E5 B4 C5 D5 C5 B4 A4 A4 C5 E5 D5 C5 B4 C5 D5 E5 C5 A4 A4
        tetris_melody = [
            (0.0, 659.25, 1.0), (1.0, 493.88, 0.5), (1.5, 523.25, 0.5), (2.0, 587.33, 1.0),
            (3.0, 523.25, 0.5), (3.5, 493.88, 0.5),
            (4.0, 440.00, 1.0), (5.0, 440.00, 0.5), (5.5, 523.25, 0.5), (6.0, 659.25, 1.0),
            (7.0, 587.33, 0.5), (7.5, 523.25, 0.5),
            (8.0, 493.88, 1.5), (9.5, 523.25, 0.5), (10.0, 587.33, 1.0), (11.0, 659.25, 1.0),
            (12.0, 523.25, 1.0), (13.0, 440.00, 1.0), (14.0, 440.00, 1.5),
        ]
        for bpos, freq, dur in tetris_melody:
            add_at(lead, bpos * beat, synth_lead(freq, dur * beat * 0.92, bright=False), 0.62)
            add_at(lead_h, bpos * beat, synth_lead(freq * 2.0, dur * beat * 0.85, bright=True), 0.48)
        register_pack("tetris", d, dh, bass, piano, piano_h, lead, lead_h)

        # =====================================================================
        # PACK 2: STRANGER 80s (Arpeggio Synthwave Cmaj7 - 116 BPM, 16 beat)
        # =====================================================================
        beat = 60.0 / 116.0
        n_samp = int(sr * 16 * beat)
        d, dh = build_drums(16, beat, n_samp)
        bass = np.zeros(n_samp)
        piano = np.zeros(n_samp)
        piano_h = np.zeros(n_samp)
        lead = np.zeros(n_samp)
        lead_h = np.zeros(n_samp)

        # Basso pulsante 80s in ottavi su C2 -> C2 -> G1 -> A1
        s_roots = [65.41, 65.41, 98.00, 110.00]
        s_chords = [
            (261.63, 329.63, 392.00, 493.88),  # Cmaj7
            (261.63, 329.63, 392.00, 493.88),
            (196.00, 246.94, 293.66, 392.00),  # G
            (220.00, 261.63, 329.63, 392.00),  # Am7
        ]
        for bar in range(4):
            for eighth in range(8):
                add_at(bass, (bar * 4 + eighth * 0.5) * beat, synth_bass(s_roots[bar], beat * 0.44), 0.78)
            cw = sum(self._synth_piano_tone(f, beat * 3.6, sr) for f in s_chords[bar]) / 4.0
            add_at(piano, (bar * 4) * beat, cw, 0.80)
            for hit in (0.0, 1.5, 2.5):
                cwh = sum(self._synth_piano_tone(f * 2.0, beat * 1.1, sr) for f in s_chords[bar]) / 4.0
                add_at(piano_h, (bar * 4 + hit) * beat, cwh, 0.55)

        # Iconico arpeggio C-E-G-B-C5-B-G-E continuo in sedicesimi!
        arp_seq = [261.63, 329.63, 392.00, 493.88, 523.25, 493.88, 392.00, 329.63]
        for step in range(64):
            f = arp_seq[step % len(arp_seq)]
            add_at(lead, step * 0.25 * beat, synth_lead(f, beat * 0.23, bright=False), 0.55)
            add_at(lead_h, step * 0.25 * beat, synth_lead(f * 2.0, beat * 0.21, bright=True), 0.45)
        register_pack("stranger", d, dh, bass, piano, piano_h, lead, lead_h)

        # =====================================================================
        # PACK 3: CRAZY ELECTRO (Riff Electro-Pop stile Axel F - 122 BPM, 16 beat)
        # =====================================================================
        beat = 60.0 / 122.0
        n_samp = int(sr * 16 * beat)
        d, dh = build_drums(16, beat, n_samp)
        bass = np.zeros(n_samp)
        piano = np.zeros(n_samp)
        piano_h = np.zeros(n_samp)
        lead = np.zeros(n_samp)
        lead_h = np.zeros(n_samp)

        e_roots = [87.31, 77.78, 103.83, 130.81]  # F2, Eb2, Ab2, C3
        e_chords = [
            (174.61, 207.65, 261.63),  # Fm
            (155.56, 196.00, 233.08),  # Eb
            (207.65, 261.63, 311.13),  # Ab
            (196.00, 261.63, 329.63),  # C
        ]
        for bar in range(4):
            for step, mult in ((0.0, 1.0), (0.75, 1.0), (1.5, 2.0), (2.0, 1.0), (2.75, 1.0), (3.5, 2.0)):
                add_at(bass, (bar * 4 + step) * beat, synth_bass(e_roots[bar] * mult, beat * 0.42), 0.72)
            for hit in (0.0, 1.5, 2.5):
                cw = sum(self._synth_piano_tone(f, beat * 1.2, sr) for f in e_chords[bar]) / 3.0
                add_at(piano, (bar * 4 + hit) * beat, cw, 0.80)
                cwh = sum(self._synth_piano_tone(f * 2.0, beat * 0.6, sr) for f in e_chords[bar]) / 3.0
                add_at(piano_h, (bar * 4 + hit) * beat, cwh, 0.56)

        # Riff electro in Fa minore (F4=349.23, Ab4=415.30, Bb4=466.16, Eb4=311.13, C5=523.25, Db5=554.37)
        electro_riff = [
            (0.0, 349.23, 0.9), (1.0, 415.30, 0.7), (1.75, 349.23, 0.45), (2.25, 349.23, 0.25),
            (2.5, 466.16, 0.45), (3.0, 349.23, 0.45), (3.5, 311.13, 0.45),
            (4.0, 349.23, 0.9), (5.0, 523.25, 0.7), (5.75, 349.23, 0.45), (6.25, 349.23, 0.25),
            (6.5, 554.37, 0.45), (7.0, 523.25, 0.45), (7.5, 415.30, 0.45),
            (8.0, 349.23, 0.45), (8.5, 523.25, 0.45), (9.0, 698.46, 0.45), (9.5, 349.23, 0.25),
            (9.75, 311.13, 0.45), (10.25, 311.13, 0.25), (10.5, 261.63, 0.45), (11.0, 392.00, 0.45),
            (11.5, 349.23, 2.2),
        ]
        for bpos, freq, dur in electro_riff:
            add_at(lead, bpos * beat, synth_lead(freq, dur * beat, bright=False), 0.64)
            add_at(lead_h, bpos * beat, synth_lead(freq * 2.0, dur * beat * 0.9, bright=True), 0.48)
        register_pack("electro", d, dh, bass, piano, piano_h, lead, lead_h)

    @property
    def current_pack_name(self) -> str:
        return self.DJ_PACKS[self.current_pack_idx][1]

    @property
    def current_pack_bpm(self) -> float:
        return self.DJ_PACKS[self.current_pack_idx][2]

    def next_dj_pack(self) -> str:
        was_running = self._loops_running
        vols = dict(self._track_volumes)
        if was_running:
            self.stop_studio_loops()
        self.current_pack_idx = (self.current_pack_idx + 1) % len(self.DJ_PACKS)
        if was_running:
            self.start_studio_loops(vols)
        return self.current_pack_name

    def _apply_track_mix(self, track_id: str) -> None:
        ch_deep = self._loop_channels.get(track_id)
        ch_hyper = self._loop_channels_hyper.get(track_id)
        if ch_deep is None or ch_hyper is None:
            return
        if self.muted:
            ch_deep.set_volume(0.0, 0.0)
            ch_hyper.set_volume(0.0, 0.0)
            return

        vol = self._track_volumes.get(track_id, 0.0)
        morph = self._track_morphs.get(track_id, 0.5)  # 0.0 (DEEP/L) .. 1.0 (HYPER/R)

        # Crossfade DEEP <-> HYPER + leggero pan stereo (sempre udibile anche in mono!)
        deep_gain = vol * (1.0 - morph)
        hyper_gain = vol * morph
        left_bias = 1.0 - 0.35 * morph
        right_bias = 0.65 + 0.35 * morph

        ch_deep.set_volume(deep_gain * left_bias, deep_gain * right_bias)
        ch_hyper.set_volume(hyper_gain * left_bias, hyper_gain * right_bias)

    def start_studio_loops(self, initial_volumes: Dict[str, float]) -> None:
        """Avvia i 4 loop (Deep + Hyper) del DJ Pack corrente in perfetta sincronia."""
        if not self.enabled or self._loops_running:
            return
        pack_id = self.DJ_PACKS[self.current_pack_idx][0]
        loops_deep = self._pack_loops[pack_id]
        loops_hyper = self._pack_loops_hyper[pack_id]
        self._loop_channels = {}
        self._loop_channels_hyper = {}
        for idx, tid in enumerate(self.TRACK_IDS):
            ch_deep = pygame.mixer.Channel(idx * 2)
            ch_hyper = pygame.mixer.Channel(idx * 2 + 1)
            self._track_volumes[tid] = max(0.0, min(1.0, initial_volumes.get(tid, 0.0)))
            self._track_morphs[tid] = self._track_morphs.get(tid, 0.5)
            ch_deep.play(loops_deep[tid], loops=-1)
            ch_hyper.play(loops_hyper[tid], loops=-1)
            self._loop_channels[tid] = ch_deep
            self._loop_channels_hyper[tid] = ch_hyper
            self._apply_track_mix(tid)
        self._loops_running = True

    def set_track_volume(self, track_id: str, volume: float, morph: float | None = None) -> None:
        self._track_volumes[track_id] = max(0.0, min(1.0, volume))
        if morph is not None:
            self._track_morphs[track_id] = max(0.0, min(1.0, morph))
        self._apply_track_mix(track_id)

    def stop_studio_loops(self) -> None:
        if not self._loops_running:
            return
        for ch in self._loop_channels.values():
            ch.stop()
        for ch in self._loop_channels_hyper.values():
            ch.stop()
        self._loop_channels.clear()
        self._loop_channels_hyper.clear()
        self._loops_running = False

    def play(self, name: str) -> None:
        if not self.enabled or self.muted:
            return
        snd = self._sounds.get(name)
        if snd is not None:
            snd.play()

    def play_piano_note(self, index: int, volume: float = 0.9) -> None:
        if not self.enabled or self.muted or not self._piano_sounds:
            return
        idx = max(0, min(len(self._piano_sounds) - 1, index))
        snd = self._piano_sounds[idx]
        snd.set_volume(max(0.2, min(1.0, volume)))
        snd.play()

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        for tid in self.TRACK_IDS:
            self._apply_track_mix(tid)
        return self.muted

    def close(self) -> None:
        self.stop_studio_loops()
        if self.enabled:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
