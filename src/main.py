import sys
import time
import socket
import csv
from datetime import datetime
from collections import deque
from typing import Annotated, Optional
import typer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, lfilter

# Typer CLI App initialisieren
app = typer.Typer(help="Uni Bremen - Breathing-Clip UDP Live Tracking & Analyse")

# ==============================================================================
# EINSTELLUNGEN FOR DIE SIGNALVERARBEITUNG
# ==============================================================================
THRESHOLD = 80.0        # Schwung-Schwelle in mG (Höher = unempfindlicher)
MIN_HOLD_TIME = 0     # Sperrzeit gegen Zappeln (in Sekunden)
HIGHCUT_FREQ = 0.4      # Filter-Obergrenze (0.4 Hz = max. 24 Atemzüge/Min)
FS = 50.0               # Abtastfrequenz M5Stick (50 Hz)
HISTORY_SECONDS = 40    # Anzeigefenster in Sekunden
HISTORY_LENGTH = int(FS * HISTORY_SECONDS)  # 2000 Datenpunkte
# ==============================================================================

def butter_bandpass_filter(data: list, lowcut: float = 0.1, highcut: float = HIGHCUT_FREQ, fs: float = FS, order: int = 2) -> np.ndarray:
    """Kausaler Butterworth-Bandpassfilter für Latenzfreiheit im Live-Stream."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def save_csv_log(candidate_id: str, start_dt: datetime, end_dt: datetime, event_log: list):
    """Speichert den Lauf strukturiert mit Metadaten-Kopfzeilen und Phasendauern."""
    filename = f"atem_lauf_{candidate_id}_{start_dt.strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')

        # Metadaten-Header
        writer.writerow(["# Kandidaten_ID", candidate_id])
        writer.writerow(["# Startzeit", start_dt.strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["# Endzeit", end_dt.strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["# Gesamtdauer_Sekunden", round((end_dt - start_dt).total_seconds(), 2)])
        writer.writerow([])  # Leerzeile

        # Spaltenüberschriften
        writer.writerow(["Uhrzeit", "Gesamtlaufzeit_s", "Neuer_Zustand", "Dauer_Vorherige_Phase_s"])

        # Datenzeilen
        for row in event_log:
            writer.writerow(row)

    typer.secho(f"\n✔ Lauf erfolgreich gespeichert in: {filename}", fg=typer.colors.GREEN, bold=True)

def prompt_time(label: str, default_dt: datetime) -> datetime:
    """Fragt nach einer Uhrzeit (HH:MM); Tag wird übernommen, Sekunden = 0."""
    default_str = default_dt.strftime("%H:%M")
    user_input = typer.prompt(
        f"{label} eingeben (HH:MM)",
        default=default_str,
    ).strip()

    if not user_input:
        user_input = default_str

    try:
        parsed = datetime.strptime(user_input, "%H:%M")
    except ValueError:
        typer.secho(
            f"⚠ Ungültiges Format '{user_input}', verwende {default_str}",
            fg=typer.colors.YELLOW,
        )
        parsed = default_dt

    return parsed.replace(
        year=default_dt.year,
        month=default_dt.month,
        day=default_dt.day,
        second=0,
        microsecond=0,
    )

def start_live_graph(ip: str, port: int, candidate_id: str):
    """Startet das grafische 40s-Live-Diagramm mit Matplotlib."""
    start_datetime = datetime.now()
    start_perf_time = time.time()
    event_log = []

    raw_data = deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
    state_data = deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        sock.setblocking(False)
    except Exception as e:
        typer.secho(f"❌ Netzwerk-Fehler: {e}", fg=typer.colors.RED)
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    x_time = np.linspace(-HISTORY_SECONDS, 0, HISTORY_LENGTH)

    line_state, = ax.plot(x_time, np.zeros(HISTORY_LENGTH), color='limegreen', linewidth=2.0)

    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 (Ausatmen)", "1 (Einatmen)"])
    ax.set_xlabel("Zeit (Sekunden relativ zu JETZT)")
    ax.set_ylabel("Atemzustand")
    ax.set_title(f"M5Stick - Lauf: {candidate_id} (Start: {start_datetime.strftime('%H:%M:%S')})")
    ax.grid(True, linestyle='--', alpha=0.5)

    info_text = ax.text(0.02, 0.85, "Warte auf ersten Schwung...", transform=ax.transAxes,
                        fontsize=11, weight='bold', bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    current_state = 0.0
    last_toggle_time = time.time()
    last_interval = 0.0

    def update_frame(frame):
        nonlocal current_state, last_toggle_time, last_interval

        while True:
            try:
                data, _ = sock.recvfrom(1024)
                line_in = data.decode('utf-8').strip()
                parts = line_in.split(',')
                if len(parts) == 3:
                    raw_data.append(float(parts[2]))
            except BlockingIOError:
                break
            except (ValueError, UnicodeDecodeError):
                pass

        filtered = butter_bandpass_filter(list(raw_data))
        acc = filtered[-1]
        now = time.time()
        time_since_toggle = now - last_toggle_time

        # Entprellte Latch-Logik
        new_state = current_state
        if time_since_toggle >= MIN_HOLD_TIME:
            if acc < -THRESHOLD and current_state == 0.0:
                new_state = 1.0
            elif acc > THRESHOLD and current_state == 1.0:
                new_state = 0.0

        if new_state != current_state:
            last_interval = time_since_toggle
            last_toggle_time = now
            current_state = new_state

            # Event protokollieren
            elapsed_total = now - start_perf_time
            now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            event_log.append([now_str, round(elapsed_total, 2), int(current_state), round(last_interval, 2)])

        state_data.append(current_state)
        line_state.set_ydata(list(state_data))

        status_label = "EINATMEN (1)" if current_state == 1.0 else "AUSATMEN (0)"
        info_text.set_text(f"Zustand: {status_label}  |  Dauer Phase: {time_since_toggle:.1f} s  |  Letzter Wechsel: {last_interval:.2f} s")
        info_text.set_color("green" if current_state == 1.0 else "red")

        return line_state, info_text

    ani = animation.FuncAnimation(fig, update_frame, interval=20, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    # Nach dem Schließen des Diagramm-Fensters speichern
    end_datetime = datetime.now()
    sock.close()
    start_datetime = prompt_time("Startzeit", start_datetime)
    end_datetime = prompt_time("Endzeit", end_datetime)
    save_csv_log(candidate_id, start_datetime, end_datetime, event_log)

@app.command()
def run(
        ip: Annotated[
            str,
            typer.Option("--ip", "-i", help="IP-Adresse (0.0.0.0 lauscht auf allen Schnittstellen)")
        ] = "0.0.0.0",
        port: Annotated[
            int,
            typer.Option("--port", "-p", help="UDP-Port des M5Sticks (Standard: 1234)")
        ] = 1234,
        graph: Annotated[
            bool,
            typer.Option("--graph", "-g", help="Live-Grafik anzeigen statt reiner Terminal-Ausgabe")
        ] = True, # Standardmäßig auf True gesetzt
        candidate: Annotated[
            Optional[str],
            typer.Option("--candidate", "-c", help="Kandidaten- / Lauf-ID (z. B. K01_Lauf1)")
        ] = None,
) -> None:
    """
    Empfängt Atemdaten vom M5Stick via UDP, zeigt die Kurve an und protokolliert 1/0 Phasen.
    """
    candidate_id = candidate if candidate else typer.prompt("Bitte Kandidaten-ID / Lauf-Nummer eingeben", default="Lauf_Unbekannt")

    # 1. GRAFIK-MODUS (Standard)
    if graph:
        typer.secho(f"▶ Starte grafische Live-Analyse für '{candidate_id}'...", fg=typer.colors.MAGENTA, bold=True)
        start_live_graph(ip, port, candidate_id)
        return

    # 2. REINER TERMINAL-MODUS (OHNE GRAFIK)
    typer.secho(f"▶ Erstelle UDP-Socket auf {ip}:{port}...", fg=typer.colors.CYAN)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        sock.setblocking(False)

        typer.secho(f"✔ UDP-Server läuft für Lauf '{candidate_id}'!", fg=typer.colors.GREEN, bold=True)
        typer.secho("Hinweis: Beenden mit STRG+C speichert die CSV-Datei ab.\n", fg=typer.colors.YELLOW)

        raw_data = deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        event_log = []
        start_datetime = datetime.now()
        start_perf_time = time.time()

        current_state = 0.0
        last_toggle_time = time.time()

        while True:
            while True:
                try:
                    data, _ = sock.recvfrom(1024)
                    line_in = data.decode('utf-8').strip()
                    parts = line_in.split(',')
                    if len(parts) == 3:
                        raw_data.append(float(parts[2]))
                except BlockingIOError:
                    break
                except (ValueError, UnicodeDecodeError):
                    pass

            filtered = butter_bandpass_filter(list(raw_data))
            acc = filtered[-1]
            now = time.time()
            time_since_toggle = now - last_toggle_time

            new_state = current_state
            if time_since_toggle >= MIN_HOLD_TIME:
                if acc < -THRESHOLD and current_state == 0.0:
                    new_state = 1.0
                elif acc > THRESHOLD and current_state == 1.0:
                    new_state = 0.0

            if new_state != current_state:
                last_interval = time_since_toggle
                last_toggle_time = now
                current_state = new_state

                elapsed_total = now - start_perf_time
                now_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                event_log.append([now_str, round(elapsed_total, 2), int(current_state), round(last_interval, 2)])

                status_text = "EINATMUNG (1)" if current_state == 1.0 else "AUSATMUNG (0)"
                color = typer.colors.GREEN if current_state == 1.0 else typer.colors.RED

                typer.secho(
                    f"[{now_str}] Zustand: {status_text} | Dauer Phase: {last_interval:.2f} s | Gesamtzeit: {elapsed_total:.1f} s",
                    fg=color
                )

            time.sleep(0.01)

    except KeyboardInterrupt:
        end_datetime = datetime.now()
        sock.close()
        typer.secho("\n▶ Messung beendet.", fg=typer.colors.BLUE)
        start_datetime = prompt_time("Startzeit", start_datetime)
        end_datetime = prompt_time("Endzeit", end_datetime)
        save_csv_log(candidate_id, start_datetime, end_datetime, event_log)
    except Exception as e:
        typer.secho(f"❌ Netzwerk-Fehler: {e}", fg=typer.colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    app()