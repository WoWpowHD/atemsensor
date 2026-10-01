import sys
import socket
import errno
import csv
from datetime import datetime
from pathlib import Path
from typing import Annotated
import typer

# Typer-App initialisieren
app = typer.Typer(help="🚀 Uni Bremen - Breathing-Clip UDP Live Tracking")

@app.command()
def run(
    ip: Annotated[
        str, 
        typer.Option("--ip", "-i", help="Die IP-Adresse Ihres Laptops (0.0.0.0 lauscht auf allen Schnittstellen)")
    ] = "0.0.0.0",
    port: Annotated[
        int, 
        typer.Option("--port", "-p", help="Der UDP-Port aus dem Uni-Bremer-Repo (Standard: 1234)")
    ] = 1234, # Standard-Port aus dem GitHub-Repo "breathing-clip"
    graph: Annotated[
        bool,
        typer.Option("--graph", "-g", help="Live-Grafik anzeigen statt Terminal-Text")
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="CSV-Datei für die Aufzeichnung (Standard: atem_lauf_<zeitstempel>.csv)")
    ] = None,
) -> None:
    """
    Startet den UDP-Server und empfängt die Live-Atemdaten vom Breathing-Clip via WLAN.
    """
    # === HIER IST DIE GRAFIK-WEICHE ===
    if graph:
        typer.secho("📈 Starte grafische Live-Atemanalyse...", fg=typer.colors.MAGENTA)
        # Importiert die soeben erstellte live_breath.py Datei und startet das Plot-Fenster
        from live_breath import start_live_graph
        start_live_graph(ip, port)
        return

    # === STANDARD TERMINAL-MODUS (WENN KEIN --graph GEGEBEN IST) ===
    typer.secho(f"🌐 Erstelle UDP-Socket auf {ip}:{port}...", fg=typer.colors.CYAN)
    
    csv_file = None

    try:
        # UDP Socket erstellen
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Erlaubt sofortiges erneutes Binden nach dem Beenden (verhindert "Address already in use" bei schnellem Neustart)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((ip, port))
        
        typer.secho("🎯 UDP-Server läuft! Warte auf Atemdaten vom M5Stick...", fg=typer.colors.GREEN, bold=True)
        typer.secho("Hinweis: Beide Geräte müssen im selben Handy-Hotspot eingewählt sein!", fg=typer.colors.YELLOW, dim=True)
        typer.secho("Wichtig: Falls keine Daten kommen, drücke die SEITENTASTE am Stick!", fg=typer.colors.YELLOW)

        # CSV-Aufzeichnung vorbereiten
        if output is None:
            output = Path(f"atem_lauf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        csv_file = open(output, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "patient_id", "elapsed_time", "ac_sig"])
        csv_file.flush()
        typer.secho(f"💾 Zeichne Signal auf in: {output}", fg=typer.colors.CYAN)

        last_val = 0.0
        
        while True:
            data, addr = sock.recvfrom(1024)
            
            try:
                line = data.decode('utf-8').strip()
                parts = line.split(',')
                if len(parts) == 3:
                    patient_id = parts[0]
                    elapsed_time = parts[1]
                    ac_sig = float(parts[2])
                    
                    diff = ac_sig - last_val
                    last_val = ac_sig
                    
                    if diff < -15:
                        status = ">>> EINATMUNG (Brust hebt sich) >>>"
                        color = typer.colors.GREEN
                    elif diff > 15:
                        status = "<<< AUSATMUNG (Brust senkt sich) <<<"
                        color = typer.colors.RED
                    else:
                        status = "Atempause / Stillstand"
                        color = typer.colors.WHITE
                        
                    # Signal zusätzlich in CSV mitschreiben
                    csv_writer.writerow([
                        datetime.now().isoformat(timespec="milliseconds"),
                        patient_id,
                        elapsed_time,
                        ac_sig,
                    ])
                    csv_file.flush()

                    output = f"[{addr[0]}] ID: {patient_id} | Zeit: {elapsed_time}ms | Signal: {ac_sig:6.1f} | {status}"
                    # nl=False hält die Ausgabe auf einer Zeile (Carriage Return \r überschreibt die vorherige)
                    typer.secho(f"\r{output}", fg=color, nl=False)
                    
            except (UnicodeDecodeError, ValueError):
                continue
                
    except KeyboardInterrupt:
        if csv_file is not None:
            csv_file.close()
        typer.secho("\n👋 UDP-Server gestoppt. Messung beendet.", fg=typer.colors.BLUE)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            typer.secho(
                f"❌ Port {port} ist bereits belegt (Address already in use).",
                fg=typer.colors.RED,
            )
            typer.secho(
                "   Schließe das andere Programm/Fenster oder nutze einen anderen Port: --port <nummer>",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho(f"❌ Netzwerk-Fehler: {e}", fg=typer.colors.RED)
        sys.exit(1)
    except Exception as e:
        typer.secho(f"❌ Netzwerk-Fehler: {e}", fg=typer.colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    app()
