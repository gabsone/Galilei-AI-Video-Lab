# AI Hand Lab — Suite di Hand Tracking per Orientamento

Applicazione Python **100% offline** pensata per far appassionare gli studenti di terza media all'informatica e all'Intelligenza Artificiale usando la webcam del Mac.

## Avvio rapido

Da terminale nella cartella del progetto:

```bash
./run.sh
```

Per partire subito a schermo intero (ideale sul proiettore):

```bash
./run.sh --fullscreen
```

## Le 5 Modalità (tasti `1`, `2`, `3`, `4`, `5`)

1. **`1` — CYBER SLICE (Fruit Ninja)**
   - Muovi l'indice di una o entrambe le mani (possono giocare anche 2 ragazzi insieme!) per tagliare le sfere tech (`AI`, `PY`, `CPU`, `GPU`, `BIT`) ed evitare i `BUG!` rossi.
   - Round arcade da 45 secondi con combo, esplosioni di particelle e record.
2. **`2` — SASSO · CARTA · FORBICI vs AI**
   - L'AI riconosce in tempo reale il gesto della mano (`SASSO`, `CARTA`, `FORBICI`).
   - Premi `SPAZIO` oppure fai **Pollice Su (👍)** per avviare il conto alla rovescia `3 - 2 - 1` seguito da **1 secondo su "VIA!"** per mostrare la mano in modo naturale.
   - **Tasto segreto `C`**: attiva/disattiva l'**"AI Imbrogliona"** per un round divertentissimo.
3. **`3` — AI MUSIC STUDIO (DJ 4-Zone + Pianoforte Aereo + Batteria Aerea)**
   - **DJ 4-Zone (116 BPM sincronizzati)**: **Batteria**, **Basso Synth**, **Accordi Piano**, **Synth Lead**. Fai **PINCH** dentro un riquadro: **Su/Giù = Volume**, **Sinistra/Destra = Morphing `DEEP ◄───► HYPER` + pan stereo**.
   - **Pianoforte Aereo (`P`)**: ottava completa (DO–DO5). Passa il dito per l'anteprima e suona con un **colpetto verso il basso** o un **pinch** (puoi ribattere facilmente la stessa nota). Premi **`G`** per la **Guida Melodie** (*Inno alla Gioia*, *Seven Nation Army*) con freccia luminosa sul tasto successivo!
   - **Batteria Aerea / Air Drums (`P`)**: **gesture dedicata "Air Drumstick Strike"** — tieni le mani come bacchette neon e dai un **colpo secco verso il basso** sui 6 pad (`CRASH`, `CLAP`, `HI-HAT`, `TOM`, `SNARE`, `KICK`). Più colpisci veloce, più suona forte!
4. **`4` — STARK HUD (Interfaccia Olografica stile Iron Man)**
   - 4 finestre olografiche fluttuanti (**Arc Reactor**, **AI Neural Matrix**, **Python Live Code**, **3D Hand Radar**).
   - **1 Mano + PINCH** su una finestra → **afferra e sposta** la finestra.
   - **1 Mano + PINCH** sull'angolo in basso a destra → **ridimensiona** la finestra.
   - **2 Mani + PINCH** insieme → **espandi / riduci (Pinch-to-Zoom olografico)** proprio come Tony Stark!
5. **`5` — CONTA PRESENTI (Appello AI della classe)**
   - Inquadra la classe con la webcam del Mac: l'AI disegna un riquadro neon con numero progressivo (`ALUNNO #1`, `#2`, ...) e confidenza attorno a ogni studente, contando i presenti in tempo reale (fino a 35 alunni) e memorizzando il **Picco Max**.
   - Premi **`SPAZIO`** per **bloccare/scattare l'appello** con banner celebrativo, e **`+` / `-`** per regolare al volo la sensibilità in base alla luce dell'aula.

## Comandi da tastiera

| Tasto | Funzione |
|---|---|
| `1` / `2` / `3` / `4` | Cambia modalità |
| `P` | (In Music Studio) Cicla tra DJ 4-Zone, Pianoforte Aereo e Batteria Aerea |
| `G` | (Sul Pianoforte Aereo) Cambia melodia guidata (Libero / Inno alla Gioia / Seven Nation Army) |
| `D` | **Modalità "Sotto il cofano"**: mostra i 21 punti numerati, le coordinate `(x, y)` e la matematica in tempo reale |
| `SPAZIO` | Avvia il round in Morra Cinese / riavvia Cyber Slice a fine tempo |
| `R` | Reset istantaneo della modalità corrente (perfetto al cambio studente) |
| `F` | Schermo intero ON/OFF |
| `M` | Audio ON/OFF |
| `C` | (Solo Morra AI) Attiva/disattiva modalità segreta AI Imbrogliona |
| `Q` / `ESC` | Esci |
