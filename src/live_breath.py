import socket
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, lfilter

# Signal-Filter für die Atmung (0.1Hz bis 0.6Hz)
def butter_bandpass_filter(data: list, lowcut: float = 0.1, highcut: float = 0.6, fs: float = 50.0, order: int = 2) -> list[float]:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def start_live_graph(ip: str, port: int):
    history_length = 200
    raw_data = deque([0.0] * history_length, maxlen=history_length)

    # UDP-Socket erstellen statt seriellem Port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        sock.setblocking(False) # Verhindert, dass das Skript beim Warten auf Daten einfriert
        print(f"Grafik-Server läuft auf UDP {ip}:{port}...")
    except Exception as e:
        print(f"Netzwerk-Fehler: {e}")
        return

    # Diagramm-Fenster einrichten
    fig, ax = plt.subplots()
    x = np.arange(0, history_length)
    
    # Uni Bremen schickt Werte in "mG" (G-Force), daher skalieren wir die Y-Achse großzügiger auf ca. -500 bis 500
    line, = ax.plot(x, np.zeros(history_length), label="Atemkurve (Gefiltert)", color='dodgerblue')
    ax.set_ylim(-500, 500) 
    ax.set_title("M5Stick - Drahtlose Live-Atemanalyse (Uni Bremen)")
    ax.set_ylabel("Beschleunigungs-Amplitude (mG)")
    ax.set_xlabel("Zeit-Schritte (~50 Hz)")
    status_text = ax.text(5, 400, "Warte auf Daten (Seitentaste drücken!)...", fontsize=11, weight='bold')

    def update_frame(frame):
        # Alle wartenden Datenpakete aus der Luft abfangen
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                line_in = data.decode('utf-8').strip()
                
                # CSV-Format parsen: patient_id, elapsedTime, ac_sig
                parts = line_in.split(',')
                if len(parts) == 3:
                    ac_sig = float(parts[2]) # Der eigentliche Atemwert
                    raw_data.append(ac_sig)
            except BlockingIOError:
                # Keine weiteren Daten im UDP-Netzwerkpuffer vorhanden
                break
            except (ValueError, UnicodeDecodeError):
                pass

        # Filter anwenden
        data_list = list(raw_data)
        filtered = butter_bandpass_filter(data_list)
        current_value = filtered[-1]
        
        # Einatmung vs. Ausatmung anhand der gefilterten Amplitude erkennen
        if current_value > 20: # Schwellenwert angepasst an mG-Einheit der Uni
            status_text.set_text("STATUS: >>> EINATMUNG (Brust hebt sich) >>>")
            status_text.set_color("green")
        elif current_value < -20:
            status_text.set_text("STATUS: <<< AUSATMUNG (Brust senkt sich) <<<")
            status_text.set_color("red")
        else:
            status_text.set_text("STATUS: Atempause / Stillstand")
            status_text.set_color("orange")

        line.set_ydata(filtered)
        return line, status_text

    # Animation starten (ruft die Update-Schleife alle 20ms auf)
    ani = animation.FuncAnimation(fig, update_frame, interval=20, blit=True, cache_frame_data=False)
    plt.legend(loc="lower left")
    plt.show()
    
    sock.close()
    print("UDP-Verbindung geschlossen.")
