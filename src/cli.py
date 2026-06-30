import sys
import socket
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
    
    try:
        # UDP Socket erstellen
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        
        typer.secho("🎯 UDP-Server läuft! Warte auf Atemdaten vom M5Stick...", fg=typer.colors.GREEN, bold=True)
        typer.secho("Hinweis: Beide Geräte müssen im selben Handy-Hotspot eingewählt sein!", fg=typer.colors.YELLOW, dim=True)
        typer.secho("Wichtig: Falls keine Daten kommen, drücke die SEITENTASTE am Stick!", fg=typer.colors.YELLOW)
        
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
                    
                    if diff > 15: 
                        status = ">>> EINATMUNG (Brust hebt sich) >>>"
                        color = typer.colors.GREEN
                    elif diff < -15: 
                        status = "<<< AUSATMUNG (Brust senkt sich) <<<"
                        color = typer.colors.RED
                    else:
                        status = "Atempause / Stillstand"
                        color = typer.colors.WHITE
                        
                    output = f"[{addr[0]}] ID: {patient_id} | Zeit: {elapsed_time}ms | Signal: {ac_sig:6.1f} | {status}"
                    typer.secho(f"\r{output}", fg=color, replace_dest=sys.stdout)
                    
            except (UnicodeDecodeError, ValueError):
                continue
                
    except KeyboardInterrupt:
        typer.secho("\n👋 UDP-Server gestoppt. Messung beendet.", fg=typer.colors.BLUE)
    except Exception as e:
        typer.secho(f"❌ Netzwerk-Fehler: {e}", fg=typer.colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    app()
