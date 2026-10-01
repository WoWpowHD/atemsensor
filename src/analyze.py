"""Auswertung der Studie "Mind Over Matter: Audio vs Haptic Breathing Guidance".

Vollstaendige, modular aufgebaute Auswertungs-Pipeline fuer die Atemsignale
(M5Stick-Beschleunigungssensor, ``ac_sig``) waehrend einer simultan
durchgefuehrten "Make 'n' Break"-Primaeraufgabe mit Box-Breathing-Schrittmacher
(4 s Einatmen / 4 s Halten / 4 s Ausatmen / 4 s Halten => 16 s Zyklus
=> 0.0625 Hz Ziel-Grundfrequenz).

Die Pipeline ist in klar getrennte Stufen gegliedert:

1.  Konfiguration (:class:`AnalysisConfig`) und Verzeichnis-Setup.
2.  Robustes Einlesen der Mess-CSV-Dateien und der Metadaten-Excel-Datei.
3.  Qualitaetssicherung / Rohdatenvisualisierung (Saettigung / Clipping).
4.  Preprocessing (uniformes Resampling, Trimming, Bandpass).
5.  Feature-Extraktion (SPR, normierte Kreuzkorrelation, Zyklus-CV).
6.  Zusammenfuehrung mit den Probandendaten.
7.  Inferenzstatistik (H1/H2/H3) inklusive Effektstaerken.
8.  Publikationsreife Abbildungen (300 DPI).

Ausfuehrung::

    python -m src.analyze
    python src/analyze.py --runs-dir runs --metadata metadaten.xlsx

Die Datei-Zuordnung erfolgt ueber den Dateinamen (``messung_N.csv`` -> Proband ``N``),
da die Spalte ``patient_id`` in den Rohdaten konstant ist.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # Headless-Backend fuer das Speichern von Abbildungen.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, ConfigDict, Field
from scipy import signal, stats

# --------------------------------------------------------------------------- #
# Konstanten
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("breath_analysis")

AUDIO: str = "audio"
HAPTIC: str = "haptic"
GROUP_ORDER: tuple[str, str] = (AUDIO, HAPTIC)
GROUP_PALETTE: dict[str, str] = {AUDIO: "#4C72B0", HAPTIC: "#DD8452"}

#: Zentrale Atem-Adhaerenz- und Performance-Metriken.
ADHERENCE_METRICS: tuple[str, ...] = ("spr", "max_cross_corr")
PERFORMANCE_METRIC: str = "cards_solved"
TIME_METRIC: str = "time_to_complete_sec"
EFFICIENCY_METRIC: str = "cards_per_minute"

#: Alle Performance-Metriken: geloeste Karten, benoetigte Zeit und Effizienz.
PERFORMANCE_METRICS: tuple[str, ...] = (PERFORMANCE_METRIC, TIME_METRIC, EFFICIENCY_METRIC)

#: Metriken, fuer die deskriptive Kennwerte exportiert werden.
DESCRIPTIVE_METRICS: tuple[str, ...] = (
    "spr",
    "max_cross_corr",
    "mean_cycle_duration_sec",
    "cv_cycle",
    PERFORMANCE_METRIC,
    TIME_METRIC,
    EFFICIENCY_METRIC,
)

#: Menschenlesbare Achsenbeschriftungen fuer die Abbildungen.
METRIC_LABELS: dict[str, str] = {
    "spr": "Spectral Power Ratio",
    "max_cross_corr": "Max Cross-Corr",
    "cv_cycle": "CV Zyklusdauer",
    "mean_cycle_duration_sec": "Mittlere Zyklusdauer [s]",
    PERFORMANCE_METRIC: "Geloeste Karten",
    TIME_METRIC: "Bearbeitungszeit [s]",
    EFFICIENCY_METRIC: "Karten pro Minute",
}

#: Abbildung der Fragebogen-Items auf sprechende Slugs.
QUESTIONNAIRE_SLUGS: tuple[tuple[str, str], ...] = (
    # Reihenfolge ist relevant: spezifischere Muster zuerst pruefen.
    ("bewusst wahrgenommen", "awareness_loss"),
    ("wahrgenommen", "signal_perception"),
    ("unterstuetzt", "signal_support"),
    ("schwierig", "dual_task_ease"),
    ("atemrhythmus", "rhythm_follow"),
    ("konzentration", "concentration_effect"),
    ("art der atemfuehrung", "comfort"),
)

#: Aliase, die in der Aufgabenstellung explizit genannt werden.
QUESTIONNAIRE_ALIASES: dict[str, str] = {
    "distraction": "concentration_effect",
    "clarity": "rhythm_follow",
}

#: Antwort-Mapping fuer binaere Ja/Nein-Items.
BINARY_MAP: dict[str, float] = {
    "ja": 1.0,
    "j": 1.0,
    "yes": 1.0,
    "nein": 0.0,
    "n": 0.0,
    "no": 0.0,
}


# --------------------------------------------------------------------------- #
# 1. Konfiguration
# --------------------------------------------------------------------------- #


class AnalysisConfig(BaseModel):
    """Zentrale, validierte Konfiguration der Auswertungs-Pipeline.

    Alle Pfade sind relativ zum Projektwurzelverzeichnis. Die Parameter
    beschreiben das komplette Preprocessing und die Feature-Definitionen.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Pfade
    runs_dir: Path = Field(default=Path("runs"), description="Ordner mit den Mess-CSV-Dateien.")
    metadata_path: Path = Field(
        default=Path("metadaten.xlsx"), description="Excel-Datei mit den Probandendaten."
    )
    plots_dir: Path = Field(default=Path("plots"), description="Zielordner fuer Abbildungen.")
    results_dir: Path = Field(default=Path("results"), description="Zielordner fuer Ergebnisse.")

    # Zeit / Resampling
    target_fs: float = Field(default=20.0, gt=0, description="Ziel-Abtastrate in Hz.")
    window_sec: float = Field(default=120.0, gt=0, description="Auswertungsfenster in Sekunden.")

    # Bandpass
    bandpass_low: float = Field(default=0.03, gt=0, description="Untere Grenzfrequenz in Hz.")
    bandpass_high: float = Field(default=0.35, gt=0, description="Obere Grenzfrequenz in Hz.")
    bandpass_order: int = Field(default=3, ge=1, description="Butterworth-Filterordnung.")

    # Spektralanalyse / Atemziel
    psd_nperseg: int = Field(default=2048, ge=64, description="Segmentlaenge des Welch-PSD.")
    spr_band_low: float = Field(default=0.05, gt=0, description="Untere Grenze des SPR-Zielbands.")
    spr_band_high: float = Field(
        default=0.075, gt=0, description="Obere Grenze des SPR-Zielbands."
    )
    target_freq: float = Field(default=0.0625, gt=0, description="Ziel-Grundfrequenz in Hz.")
    box_cycle_sec: float = Field(default=16.0, gt=0, description="Dauer eines Box-Breathing-Zyklus.")

    # Qualitaetssicherung
    saturation_abs: float = Field(
        default=998.0, gt=0, description="Betrag, ab dem ein Sample als gesaettigt gilt."
    )
    full_saturation_ratio: float = Field(
        default=0.9, gt=0, le=1, description="Anteil gesaettigter Samples ab dem eine Datei verworfen wird."
    )
    file_pattern: str = Field(
        default=r"messung_(\d+)\.csv", description="Regex zur Extraktion der Probanden-ID."
    )
    duration_format: str = Field(
        default="mm_ss",
        description=(
            "Interpretation der Spalte 'Dauer': 'mm_ss' (Standard, Minuten:Sekunden, "
            "wie im Datensatz: 01:36:00 = 96 s) oder 'hh_mm_ss'."
        ),
    )

    # Statistik / Plot-Ausgabe
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Signifikanzniveau.")
    dpi: int = Field(default=300, ge=72, description="Aufloesung der gespeicherten Abbildungen.")


def ensure_directories(config: AnalysisConfig) -> None:
    """Erstellt die Zielordner fuer Abbildungen und Ergebnisse.

    Args:
        config: Aktive Pipeline-Konfiguration.
    """
    for directory in (config.plots_dir, config.results_dir):
        directory.mkdir(parents=True, exist_ok=True)
        LOGGER.debug("Stelle Ordner sicher: %s", directory)


# --------------------------------------------------------------------------- #
# Datenklassen
# --------------------------------------------------------------------------- #


@dataclass
class Recording:
    """Rohe Messung einer einzelnen CSV-Datei inklusive QA-Status.

    Attributes:
        subject_id: Aus dem Dateinamen abgeleitete Probanden-ID.
        path: Pfad zur Quelldatei.
        raw: DataFrame mit den Spalten ``time_s`` (Sekunden) und ``ac_sig``.
        saturation_ratio: Anteil als gesaettigt markierter Samples.
        valid: ``True``, wenn die Datei die Qualitaetspruefung bestanden hat.
        notes: Menschenlesbare Anmerkungen (z. B. Ausschlussgrund).
    """

    subject_id: int
    path: Path
    raw: pd.DataFrame
    saturation_ratio: float = 0.0
    valid: bool = True
    notes: str = ""


@dataclass
class ProcessedSignal:
    """Preprozessiertes, gleichmaessig abgetastetes Atemsignal.

    Attributes:
        subject_id: Probanden-ID.
        time: Uniformes Zeitraster in Sekunden.
        raw: Resampletes, aber ungefiltertes Signal.
        filtered: Bandpass-gefiltertes Signal.
        fs: Abtastrate in Hz.
    """

    subject_id: int
    time: np.ndarray
    raw: np.ndarray
    filtered: np.ndarray
    fs: float


@dataclass
class BreathFeatures:
    """Extrahierte Atem- und Rhythmus-Kennzahlen einer Messung.

    Attributes:
        subject_id: Probanden-ID.
        spr: Spectral Power Ratio (Zielband / Gesamtband).
        max_cross_corr: Maximale normierte Kreuzkorrelation mit der Soll-Welle.
        mean_cycle_duration_sec: Mittlere Zyklusdauer in Sekunden.
        cv_cycle: Variationskoeffizient der Zyklusdauern.
        n_cycles: Anzahl detektierter Zyklen.
        dominant_freq_hz: Dominante Frequenz innerhalb des Analysebands.
        psd_freqs: Frequenzachse des Welch-Periodogramms.
        psd_values: Spektrale Leistungsdichte.
    """

    subject_id: int
    spr: float
    max_cross_corr: float
    mean_cycle_duration_sec: float
    cv_cycle: float
    n_cycles: int
    dominant_freq_hz: float
    psd_freqs: np.ndarray = field(repr=False)
    psd_values: np.ndarray = field(repr=False)

    def as_row(self) -> dict[str, Any]:
        """Liefert die skalaren Kennzahlen als Dict (ohne PSD-Arrays)."""
        return {
            "subject_id": self.subject_id,
            "spr": self.spr,
            "max_cross_corr": self.max_cross_corr,
            "mean_cycle_duration_sec": self.mean_cycle_duration_sec,
            "cv_cycle": self.cv_cycle,
            "n_cycles": self.n_cycles,
            "dominant_freq_hz": self.dominant_freq_hz,
        }


# --------------------------------------------------------------------------- #
# Numerische Hilfsfunktionen
# --------------------------------------------------------------------------- #


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Numerische Integration, kompatibel mit NumPy < 2.0 und >= 2.0."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # pragma: no cover - Fallback fuer alte NumPy


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Integriert die spektrale Leistungsdichte innerhalb eines Frequenzbands.

    Verwendet eine Rechteck-Integration (Bin-Breite x Summe der PSD-Werte).
    Diese ist robuster als die Trapezregel, da schmale Baender bei grober
    Frequenzaufloesung nur einen einzigen Bin enthalten koennen - dort liefert
    die Trapezregel faelschlich 0.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    bin_width = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0
    return float(np.sum(psd[mask]) * bin_width)


def p_to_stars(p_value: float, alpha: float = 0.05) -> str:
    """Wandelt einen p-Wert in gaengige Signifikanz-Sternchen um."""
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < alpha:
        return "*"
    return "n.s."


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Berechnet Cohen's d mit gepoolter Standardabweichung (a - b)."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(group_a) - np.mean(group_b)) / pooled)


def _shapiro_p(values: np.ndarray) -> float:
    """Shapiro-Wilk-Normalitaetstest; gibt ``nan`` bei zu kleinen Stichproben zurueck."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.allclose(np.std(values), 0.0):
        return float("nan")
    try:
        return float(stats.shapiro(values).pvalue)
    except ValueError:  # pragma: no cover - defensiv
        return float("nan")


# --------------------------------------------------------------------------- #
# 2. Einlesen der Rohdaten
# --------------------------------------------------------------------------- #


def _extract_subject_id(path: Path, pattern: str) -> int | None:
    """Extrahiert die Probanden-ID aus dem Dateinamen gemaess Regex."""
    match = re.search(pattern, path.name)
    if match is None:
        return None
    return int(match.group(1))


def _build_time_axis(df: pd.DataFrame) -> pd.Series:
    """Konstruiert eine belastbare Zeitachse in Sekunden.

    ``elapsed_time`` ist als Geraete-Uptime in Millisekunden mit einem
    abschliessenden ``l``-Suffix kodiert. Faellt dieses Feld aus, wird die
    Zeitachse aus dem ISO-Zeitstempel abgeleitet.

    Args:
        df: Rohes Mess-DataFrame mit ``elapsed_time`` und/oder ``timestamp``.

    Returns:
        Series mit monoton aufsteigender Zeit in Sekunden (Start bei 0).
    """
    time_s: pd.Series | None = None

    if "elapsed_time" in df.columns:
        cleaned = (
            df["elapsed_time"]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
        )
        elapsed_ms = pd.to_numeric(cleaned, errors="coerce")
        if elapsed_ms.notna().sum() >= max(2, int(0.5 * len(df))):
            time_s = (elapsed_ms - elapsed_ms.min()) / 1000.0

    if time_s is None and "timestamp" in df.columns:
        stamps = pd.to_datetime(df["timestamp"], errors="coerce")
        if stamps.notna().sum() >= max(2, int(0.5 * len(df))):
            time_s = (stamps - stamps.min()).dt.total_seconds()

    if time_s is None:
        # Letzter Fallback: Sample-Index bei angenommener Nennrate.
        LOGGER.warning("Keine verwertbare Zeitachse gefunden - nutze Sample-Index.")
        time_s = pd.Series(np.arange(len(df), dtype=float))

    return time_s.astype(float)


def _extract_acceleration(df: pd.DataFrame) -> np.ndarray:
    """Waehlt das Beschleunigungssignal (Betrag, Einzelachse oder Fallback)."""
    axes = {"x", "y", "z"}
    if axes.issubset(df.columns):
        return np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2).to_numpy(dtype=float)
    for candidate in ("ac_sig", "acc_z", "acc"):
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce").to_numpy(dtype=float)
    if df.shape[1] >= 2:
        return pd.to_numeric(df.iloc[:, -1], errors="coerce").to_numpy(dtype=float)
    raise ValueError("Kein verwertbares Beschleunigungssignal in der CSV-Datei gefunden.")


def load_recording(path: Path, config: AnalysisConfig) -> Recording:
    """Liest eine Mess-CSV ein und bereinigt Zeitachse und Signalwerte.

    Args:
        path: Pfad zur Mess-CSV.
        config: Aktive Pipeline-Konfiguration.

    Returns:
        Ein :class:`Recording` mit den Spalten ``time_s`` und ``ac_sig``.
    """
    frame = pd.read_csv(path)
    signal_values = _extract_acceleration(frame)
    time_s = _build_time_axis(frame)

    raw = pd.DataFrame({"time_s": np.asarray(time_s), "ac_sig": signal_values})
    raw = raw.dropna(subset=["time_s", "ac_sig"]).sort_values("time_s").reset_index(drop=True)

    subject_id = _extract_subject_id(path, config.file_pattern)
    if subject_id is None:
        LOGGER.warning("Konnte keine Probanden-ID aus %s ableiten.", path.name)
        subject_id = -1

    return Recording(subject_id=subject_id, path=path, raw=raw)


def load_all_recordings(config: AnalysisConfig) -> list[Recording]:
    """Laedt alle zum Muster passenden Mess-CSV-Dateien, sortiert nach Probanden-ID."""
    paths = sorted(
        (p for p in config.runs_dir.glob("*") if re.search(config.file_pattern, p.name)),
        key=lambda p: _extract_subject_id(p, config.file_pattern) or 0,
    )
    if not paths:
        LOGGER.error("Keine Messdateien in %s gefunden.", config.runs_dir)
        return []

    recordings = [load_recording(path, config) for path in paths]
    LOGGER.info("%d Messdateien geladen (%s).", len(recordings), config.runs_dir)
    return recordings


# --------------------------------------------------------------------------- #
# 3. Qualitaetssicherung und Rohdatenvisualisierung
# --------------------------------------------------------------------------- #


def assess_saturation(recording: Recording, config: AnalysisConfig) -> Recording:
    """Prueft ein Signal auf Sensor-Saettigung und markiert unbrauchbare Dateien.

    Samples mit ``|ac_sig| >= saturation_abs`` sind Artefakte der Sensor-
    Saettigung. Ueberschreitet ihr Anteil ``full_saturation_ratio``, wird die
    komplette Aufnahme als ungueltig verworfen.

    Args:
        recording: Die zu pruefende Aufnahme.
        config: Aktive Pipeline-Konfiguration.

    Returns:
        Die um Bereinigungsflags ergaenzte Aufnahme (in-place aktualisiert).
    """
    values = recording.raw["ac_sig"].to_numpy(dtype=float)
    saturated = np.abs(values) >= config.saturation_abs
    recording.saturation_ratio = float(np.mean(saturated)) if values.size else 1.0

    if recording.saturation_ratio >= config.full_saturation_ratio:
        recording.valid = False
        recording.notes = (
            f"Vollsaettigung ({recording.saturation_ratio:.1%} Samples) - Aufnahme verworfen."
        )
    elif values.size < config.target_fs * 5:
        recording.valid = False
        recording.notes = "Aufnahme zu kurz fuer eine Auswertung."
    else:
        recording.notes = f"Saettigungsanteil: {recording.saturation_ratio:.1%} (maskiert)."

    return recording


def plot_raw_overview(recordings: Sequence[Recording], config: AnalysisConfig) -> Path:
    """Erstellt ein Subplot-Uebersichtsbild aller Rohsignale.

    Gesaettigte Samples werden rot hervorgehoben; der Titel enthaelt die
    Probanden-ID, die Dauer und den Saettigungsanteil.

    Args:
        recordings: Alle geladenen Aufnahmen.
        config: Aktive Pipeline-Konfiguration.

    Returns:
        Pfad zur gespeicherten Abbildung.
    """
    if not recordings:
        raise ValueError("Keine Aufnahmen fuer die Rohdatenvisualisierung vorhanden.")

    n_cols = 2
    n_rows = int(np.ceil(len(recordings) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14, 2.6 * n_rows), squeeze=False, sharex=False
    )
    flat_axes = axes.ravel()

    for ax, recording in zip(flat_axes, recordings):
        time = recording.raw["time_s"].to_numpy(dtype=float)
        values = recording.raw["ac_sig"].to_numpy(dtype=float)
        saturated = np.abs(values) >= config.saturation_abs

        ax.plot(time, values, color="#4C72B0", linewidth=0.6, label="ac_sig")
        if np.any(saturated):
            ax.scatter(
                time[saturated],
                values[saturated],
                s=4,
                color="#C44E52",
                label="Saettigung",
                zorder=3,
            )

        status = "OK" if recording.valid else "VERWORFEN"
        ax.set_title(
            f"Proband {recording.subject_id} [{status}] - "
            f"Dauer {time[-1]:.0f}s - Saettigung {recording.saturation_ratio:.1%}",
            fontsize=10,
            color="black" if recording.valid else "#C44E52",
        )
        ax.set_xlabel("Zeit [s]")
        ax.set_ylabel("ac_sig")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)

    for ax in flat_axes[len(recordings):]:
        ax.axis("off")

    fig.suptitle("Rohdaten-Uebersicht aller Messungen (ac_sig ueber Zeit)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = config.plots_dir / "01_raw_overview.png"
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Rohdaten-Uebersicht gespeichert: %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 4. Preprocessing
# --------------------------------------------------------------------------- #


def resample_uniform(
    time_s: np.ndarray,
    values: np.ndarray,
    fs: float,
    saturation_abs: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Resamplet ein unregelmaessig abgetastetes Signal auf ein uniformes Raster.

    Gesaettigte Samples werden als ungueltig verworfen und das verbleibende
    Signal monotonisch sortiert. Anschliessend erfolgt eine lineare
    Interpolation auf ein aequidistantes Raster mit der Zielrate ``fs``.

    Args:
        time_s: Ursprungs-Zeitachse in Sekunden.
        values: Signalwerte.
        fs: Ziel-Abtastrate in Hz.
        saturation_abs: Betragsschwelle fuer die Saettigungsmaske.

    Returns:
        Tupel ``(zeit, werte)`` auf uniformem Raster oder ``None``, wenn das
        Signal nach der Bereinigung zu wenige gueltige Stuetzstellen besitzt.
    """
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(time_s) & np.isfinite(values) & (np.abs(values) < saturation_abs)
    time_s, values = time_s[valid], values[valid]
    if time_s.size < 2:
        return None

    order = np.argsort(time_s, kind="mergesort")
    time_s, values = time_s[order], values[order]
    unique_time, unique_idx = np.unique(time_s, return_index=True)
    time_s, values = unique_time, values[unique_idx]

    n_samples = int(np.floor((time_s[-1] - time_s[0]) * fs)) + 1
    if n_samples < 5:
        return None

    uniform_time = time_s[0] + np.arange(n_samples, dtype=float) / fs
    uniform_values = np.interp(uniform_time, time_s, values)
    return uniform_time, uniform_values


def trim_window(
    time_s: np.ndarray, values: np.ndarray, window_sec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Schneidet das relevante Versuchsfenster ab dem ersten Sample zu."""
    mask = time_s <= (time_s[0] + window_sec)
    return time_s[mask], values[mask]


def bandpass_filter(
    values: np.ndarray, fs: float, low: float, high: float, order: int
) -> np.ndarray:
    """Phasenlinearer Butterworth-Bandpass (filtfilt) zur Atem-Isolation.

    Der Filter entfernt DC-Offset, langsame Drift und schnelle
    Handbewegungsartefakte. Ist das Signal kuerzer als die Filter-Randzone,
    wird auf eine reine Mittelwertbefreiung zurueckgefallen.

    Args:
        values: Eingangssignal.
        fs: Abtastrate in Hz.
        low: Untere Grenzfrequenz in Hz.
        high: Obere Grenzfrequenz in Hz.
        order: Filterordnung.

    Returns:
        Gefiltertes Signal gleicher Laenge.
    """
    nyquist = 0.5 * fs
    low_norm, high_norm = low / nyquist, high / nyquist
    if not (0 < low_norm < high_norm < 1):
        raise ValueError("Ungueltige Bandpass-Grenzen fuer die gegebene Abtastrate.")

    sos = signal.butter(order, [low_norm, high_norm], btype="band", output="sos")
    try:
        filtered = signal.sosfiltfilt(sos, values)
    except ValueError:
        LOGGER.warning("Signal zu kurz fuer sosfiltfilt - nutze Mittelwertbefreiung.")
        filtered = values - np.mean(values)
    return np.asarray(filtered, dtype=float)


def preprocess_recording(recording: Recording, config: AnalysisConfig) -> ProcessedSignal | None:
    """Fuehrt die komplette Preprocessing-Kette fuer eine Aufnahme aus.

    Schritte: Resampling auf ``target_fs`` -> Trimmen auf ``window_sec`` ->
    Butterworth-Bandpass.

    Args:
        recording: Gueltige Rohaufnahme.
        config: Aktive Pipeline-Konfiguration.

    Returns:
        Ein :class:`ProcessedSignal` oder ``None``, wenn das Signal unbrauchbar ist.
    """
    resampled = resample_uniform(
        recording.raw["time_s"].to_numpy(dtype=float),
        recording.raw["ac_sig"].to_numpy(dtype=float),
        fs=config.target_fs,
        saturation_abs=config.saturation_abs,
    )
    if resampled is None:
        LOGGER.warning("Proband %s: Resampling fehlgeschlagen.", recording.subject_id)
        return None

    time_s, values = resampled
    time_s, values = trim_window(time_s, values, config.window_sec)
    if values.size < config.target_fs * 10:
        LOGGER.warning("Proband %s: zu wenig Daten nach dem Trimmen.", recording.subject_id)
        return None

    # Zeitachse fuer die Auswertung auf 0 normieren.
    time_s = time_s - time_s[0]
    filtered = bandpass_filter(
        values, config.target_fs, config.bandpass_low, config.bandpass_high, config.bandpass_order
    )
    return ProcessedSignal(
        subject_id=recording.subject_id,
        time=time_s,
        raw=values,
        filtered=filtered,
        fs=config.target_fs,
    )


# --------------------------------------------------------------------------- #
# 5. Feature-Extraktion
# --------------------------------------------------------------------------- #


def compute_psd(values: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Berechnet das Welch-Leistungsdichtespektrum des Signals."""
    effective_nperseg = int(min(nperseg, values.size))
    freqs, psd = signal.welch(
        values, fs=fs, nperseg=effective_nperseg, detrend="linear", scaling="density"
    )
    return freqs, psd


def spectral_power_ratio(
    freqs: np.ndarray, psd: np.ndarray, config: AnalysisConfig
) -> float:
    """Berechnet das Spectral Power Ratio (SPR) gemaess Studienprotokoll.

    ``SPR = Leistung(0.05-0.075 Hz) / Leistung(0.03-0.35 Hz)``.

    Args:
        freqs: Frequenzachse.
        psd: Spektrale Leistungsdichte.
        config: Aktive Pipeline-Konfiguration.

    Returns:
        SPR als Anteil in ``[0, 1]`` (0 bei fehlender Gesamtleistung).
    """
    target_power = _band_power(freqs, psd, config.spr_band_low, config.spr_band_high)
    total_power = _band_power(
        freqs, psd, config.bandpass_low, config.bandpass_high
    )
    if total_power <= 0:
        return 0.0
    return float(np.clip(target_power / total_power, 0.0, 1.0))


def box_breathing_reference(duration_sec: float, fs: float, cycle_sec: float) -> np.ndarray:
    """Synthetisiert die Soll-Welle fuer Box Breathing als Trapez-Periodizitaet.

    Ein Zyklus besteht aus vier gleich langen Phasen: linearer Anstieg
    (Einatmen), Halten oben, linearer Abfall (Ausatmen), Halten unten. Das
    Signal wird mittelwertbefreit ausgegeben.

    Args:
        duration_sec: Laenge des Referenzsignals in Sekunden.
        fs: Abtastrate in Hz.
        cycle_sec: Zyklusdauer in Sekunden (Standard: 16 s).

    Returns:
        Mittelwertbefreites Referenzsignal.
    """
    n_samples = int(round(duration_sec * fs))
    time = np.arange(n_samples, dtype=float) / fs
    phase = (time % cycle_sec) / cycle_sec  # normierte Phase in [0, 1)

    reference = np.piecewise(
        phase,
        [
            phase < 0.25,
            (phase >= 0.25) & (phase < 0.5),
            (phase >= 0.5) & (phase < 0.75),
            phase >= 0.75,
        ],
        [
            lambda p: p / 0.25,
            lambda p: np.ones_like(p),
            lambda p: 1.0 - (p - 0.5) / 0.25,
            lambda p: np.zeros_like(p),
        ],
    )
    return reference - np.mean(reference)


def max_normalized_cross_correlation(
    values: np.ndarray, reference: np.ndarray
) -> float:
    """Berechnet die maximale normierte Kreuzkorrelation zweier Signale.

    Beide Signale werden z-standardisiert; die Kreuzkorrelation wird auf die
    Signalbreite normiert, sodass der Koeffizient direkt interpretierbar ist.

    Args:
        values: Gefiltertes Atemsignal.
        reference: Soll-Referenzwelle.

    Returns:
        Maximum der normierten Kreuzkorrelation (auch ueber Zeitverschiebungen).
    """
    n = min(values.size, reference.size)
    if n < 2:
        return float("nan")
    values = values[:n]
    reference = reference[:n]

    values_std = np.std(values)
    reference_std = np.std(reference)
    if values_std == 0 or reference_std == 0:
        return float("nan")

    norm_values = (values - np.mean(values)) / values_std
    norm_reference = (reference - np.mean(reference)) / reference_std
    correlation = signal.correlate(norm_values, norm_reference, mode="full") / n
    return float(np.max(correlation))


def cycle_duration_features(
    values: np.ndarray, fs: float, cycle_sec: float
) -> tuple[float, float, int]:
    """Detektiert Atemzyklen und berechnet mittlere Dauer sowie deren CV.

    Args:
        values: Gefiltertes Atemsignal.
        fs: Abtastrate in Hz.
        cycle_sec: Erwartete Zyklusdauer (dient als Mindest-Peakabstand).

    Returns:
        Tupel ``(mittlere_zyklusdauer, cv, anzahl_zyklen)``. Bei weniger als
        zwei Peaks werden ``(nan, nan, n_peaks)`` zurueckgegeben.
    """
    std = np.std(values)
    prominence = 0.25 * std if std > 0 else None
    peaks, _ = signal.find_peaks(
        values,
        distance=int(0.5 * cycle_sec * fs),
        prominence=prominence,
    )
    if peaks.size < 2:
        return float("nan"), float("nan"), int(peaks.size)

    durations = np.diff(peaks) / fs
    mean_duration = float(np.mean(durations))
    cv = float(np.std(durations) / mean_duration) if mean_duration > 0 else float("nan")
    return mean_duration, cv, int(peaks.size)


def extract_features(signal_data: ProcessedSignal, config: AnalysisConfig) -> BreathFeatures:
    """Extrahiert alle Atem-Adhaerenz- und Rhythmus-Features einer Messung.

    Args:
        signal_data: Preprozessiertes Signal.
        config: Aktive Pipeline-Konfiguration.

    Returns:
        Ein :class:`BreathFeatures`-Objekt.
    """
    freqs, psd = compute_psd(signal_data.filtered, signal_data.fs, config.psd_nperseg)
    spr = spectral_power_ratio(freqs, psd, config)

    in_band = (freqs >= config.bandpass_low) & (freqs <= config.bandpass_high)
    dominant_freq = float(freqs[in_band][np.argmax(psd[in_band])]) if np.any(in_band) else float("nan")

    reference = box_breathing_reference(
        duration_sec=signal_data.time[-1], fs=signal_data.fs, cycle_sec=config.box_cycle_sec
    )
    max_corr = max_normalized_cross_correlation(signal_data.filtered, reference)

    mean_duration, cv_cycle, n_cycles = cycle_duration_features(
        signal_data.filtered, signal_data.fs, config.box_cycle_sec
    )

    return BreathFeatures(
        subject_id=signal_data.subject_id,
        spr=spr,
        max_cross_corr=max_corr,
        mean_cycle_duration_sec=mean_duration,
        cv_cycle=cv_cycle,
        n_cycles=n_cycles,
        dominant_freq_hz=dominant_freq,
        psd_freqs=freqs,
        psd_values=psd,
    )


# --------------------------------------------------------------------------- #
# 6. Metadaten einlesen und zusammenfuehren
# --------------------------------------------------------------------------- #


def _pick_sheet(sheet_names: Iterable[str], *keywords: str) -> str | None:
    """Waehlt das erste Arbeitsblatt, dessen Name alle Keywords enthaelt."""
    for name in sheet_names:
        lowered = name.lower()
        if all(keyword.lower() in lowered for keyword in keywords):
            return name
    return None


def _normalize_text(text: str) -> str:
    """Normalisiert Umlaute und Sonderzeichen fuer stabiles Keyword-Matching."""
    replacements = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "’": "'", "‚": "'"}
    normalized = text.lower()
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _slugify_question(question: str) -> str:
    """Transformiert einen Fragebogentext in einen kompakten, stabilen Slug."""
    lowered = _normalize_text(question)
    for keyword, slug in QUESTIONNAIRE_SLUGS:
        if keyword in lowered:
            return slug
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return f"q_{slug[:40]}" if slug else "q_unknown"


def _clean_response(series: pd.Series) -> pd.Series:
    """Bereinigt einen Fragebogen-Antwortvektor.

    Wandelt Ja/Nein-Antworten in 1/0 um und parst numerische Likert-Werte
    tolerant (Komma als Dezimaltrenner, ungueltige Eintraege -> ``NaN``).
    """
    text = series.astype(str).str.strip().str.lower()
    as_binary = text.map(BINARY_MAP)
    as_numeric = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")
    return as_binary.fillna(as_numeric)


def parse_duration_seconds(series: pd.Series, config: AnalysisConfig) -> pd.Series:
    """Parst eine Zeit-/Dauer-Spalte robust in Sekunden.

    Unterstuetz die im Datensatz vorkommenden Formate:

    - ``HH:MM:SS``/``MM:SS``-Text (z. B. ``01:36:00``), interpretiert gemaess
      ``config.duration_format`` (Standard ``mm_ss``: erster Block = Minuten).
    - Excel-Zeitbruchteil als Gleitkommazahl in ``[0, 1)`` (Anteil eines Tages).
    - Reine Zahlen (werden als Sekunden interpretiert).
    - Fehlende/ungueltige Eintraege (``-``, leer) werden zu ``NaN``.

    Args:
        series: Rohspalte mit Zeitangaben.
        config: Aktive Pipeline-Konfiguration (liefert ``duration_format``).

    Returns:
        Series mit Zeiten in Sekunden.
    """

    def _single(value: Any) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float("nan")
        if isinstance(value, (int, float, np.number)):
            numeric = float(value)
            return numeric * 86400.0 if 0 <= numeric < 1 else numeric

        text = str(value).strip()
        if text in {"", "-", "nan", "NaT", "None"}:
            return float("nan")
        parts = text.replace(",", ".").split(":")
        try:
            blocks = [float(part) for part in parts]
        except ValueError:
            return float("nan")

        if len(blocks) >= 3:  # HH:MM:SS dargestellt; erster Block je Format.
            if config.duration_format == "hh_mm_ss":
                return blocks[0] * 3600.0 + blocks[1] * 60.0 + blocks[2]
            return blocks[0] * 60.0 + blocks[1]
        if len(blocks) == 2:
            return blocks[0] * 60.0 + blocks[1]
        return blocks[0]

    return series.map(_single)


def load_metadata(config: AnalysisConfig) -> pd.DataFrame:
    """Laedt und bereinigt alle relevanten Arbeitsblaetter der Metadaten-Excel.

    Das Blatt ``Post Interview`` liegt transponiert vor (Fragen als Zeilen,
    Probanden als Spalten) und wird ins Langformat ueberfuehrt, bereinigt und
    wieder breit gepivotet.

    Args:
        config: Aktive Pipeline-Konfiguration.

    Returns:
        DataFrame mit ``subject_id`` als Index sowie Gruppen-, Performance- und
        Fragebogenspalten.
    """
    if not config.metadata_path.exists():
        raise FileNotFoundError(f"Metadatendatei nicht gefunden: {config.metadata_path}")

    workbook = pd.ExcelFile(config.metadata_path)
    sheet_names = workbook.sheet_names

    # --- Blatt 1: Stammdaten und Gruppenzuordnung ---------------------------------
    sheet_proband = _pick_sheet(sheet_names, "proband") or sheet_names[0]
    proband = pd.read_excel(workbook, sheet_name=sheet_proband)
    proband = proband.rename(columns={"Proband": "subject_id", "Art": "group"})
    proband["subject_id"] = pd.to_numeric(proband["subject_id"], errors="coerce")
    proband = proband.dropna(subset=["subject_id"])
    proband["subject_id"] = proband["subject_id"].astype(int)
    proband["group"] = (
        proband["group"].astype(str).str.strip().str.lower().replace({"haptik": HAPTIC})
    )

    # --- Blatt 2: Make 'n' Break Performance --------------------------------------
    sheet_performance = _pick_sheet(sheet_names, "make") or (
        sheet_names[1] if len(sheet_names) > 1 else sheet_proband
    )
    performance = pd.read_excel(workbook, sheet_name=sheet_performance, skiprows=2)
    performance = performance.rename(
        columns={
            "Proband": "subject_id",
            "Aufgaben gelöst": "cards_solved",
            "Davon falsch": "cards_wrong",
            "Dauer": "task_duration_raw",
            "Startzeit": "start_time_raw",
        }
    )
    if "subject_id" in performance.columns:
        performance["subject_id"] = pd.to_numeric(performance["subject_id"], errors="coerce")
        performance = performance.dropna(subset=["subject_id"])
        performance["subject_id"] = performance["subject_id"].astype(int)

        for numeric_column in ("cards_solved", "cards_wrong"):
            if numeric_column in performance.columns:
                performance[numeric_column] = pd.to_numeric(
                    performance[numeric_column], errors="coerce"
                )

        # Bearbeitungszeit ("Dauer") in Sekunden sowie abgeleitete Effizienz.
        if "task_duration_raw" in performance.columns:
            performance[TIME_METRIC] = parse_duration_seconds(
                performance["task_duration_raw"], config
            )
            minutes = performance[TIME_METRIC] / 60.0
            performance[EFFICIENCY_METRIC] = np.where(
                minutes > 0, performance["cards_solved"] / minutes, np.nan
            )

        perf_cols = [
            column
            for column in (
                "subject_id",
                "cards_solved",
                "cards_wrong",
                "task_duration_raw",
                TIME_METRIC,
                EFFICIENCY_METRIC,
            )
            if column in performance.columns
        ]
        performance = performance[perf_cols]
    else:  # pragma: no cover - defensiv
        performance = pd.DataFrame(columns=["subject_id", "cards_solved", "cards_wrong"])

    # --- Blatt 3: Post-Interview-Fragebogen (transponiert) -------------------------
    sheet_interview = _pick_sheet(sheet_names, "interview") or (
        sheet_names[2] if len(sheet_names) > 2 else None
    )
    questionnaire = pd.DataFrame()
    if sheet_interview is not None:
        raw_q = pd.read_excel(workbook, sheet_name=sheet_interview, skiprows=1)
        question_col = raw_q.columns[0]
        raw_q = raw_q.rename(columns={question_col: "question"})

        long_q = raw_q.melt(id_vars="question", var_name="subject_label", value_name="value")
        long_q["subject_id"] = long_q["subject_label"].astype(str).str.extract(r"(\d+)")
        long_q = long_q.dropna(subset=["subject_id"])
        long_q["subject_id"] = long_q["subject_id"].astype(int)

        columns: dict[str, pd.Series] = {}
        for question, group in long_q.groupby("question"):
            cleaned = _clean_response(group["value"])
            if cleaned.notna().mean() < 0.5:
                continue  # Freitext-Item ohne numerische Antworten.
            slug = _slugify_question(str(question))
            columns[slug] = pd.Series(
                cleaned.to_numpy(), index=group["subject_id"].to_numpy(), name=slug
            )
        if columns:
            questionnaire = pd.DataFrame(columns).reset_index(names="subject_id")

    # --- Zusammenfuehren ----------------------------------------------------------
    metadata = proband.merge(performance, on="subject_id", how="left")
    if not questionnaire.empty:
        metadata = metadata.merge(questionnaire, on="subject_id", how="left")

    # Aliase aus der Aufgabenstellung bereitstellen.
    for alias, source in QUESTIONNAIRE_ALIASES.items():
        if source in metadata.columns:
            metadata[alias] = metadata[source]

    LOGGER.info(
        "Metadaten geladen: %d Probanden (%d Audio / %d Haptik).",
        len(metadata),
        int((metadata["group"] == AUDIO).sum()),
        int((metadata["group"] == HAPTIC).sum()),
    )
    return metadata


def merge_dataset(
    features: Sequence[BreathFeatures], metadata: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    """Verknuepft extrahierte Atem-Features mit den Probandendaten (Inner Join).

    Args:
        features: Liste der Feature-Objekte.
        metadata: Bereinigte Metadaten mit ``subject_id``.
        config: Aktive Pipeline-Konfiguration.

    Returns:
        Zusammengefuehrter DataFrame, sortiert nach Probanden-ID.
    """
    features_df = pd.DataFrame([feature.as_row() for feature in features])
    if features_df.empty:
        return pd.DataFrame()

    merged = features_df.merge(metadata, on="subject_id", how="inner").sort_values("subject_id")

    unmatched = sorted(set(features_df["subject_id"]) - set(metadata["subject_id"]))
    if unmatched:
        LOGGER.warning("Probanden mit Atem-Features ohne Metadaten-Zuordnung: %s", unmatched)
    LOGGER.info(
        "Zusammengefuehrter Datensatz: %d Probanden (%d Audio / %d Haptik).",
        len(merged),
        int((merged["group"] == AUDIO).sum()),
        int((merged["group"] == HAPTIC).sum()),
    )
    return merged


# --------------------------------------------------------------------------- #
# 7. Inferenzstatistik
# --------------------------------------------------------------------------- #


def compare_groups(
    data: pd.DataFrame, metric: str, config: AnalysisConfig
) -> dict[str, Any] | None:
    """Vergleicht zwei unabhaengige Gruppen fuer eine Zielmetrik.

    Es wird automatisch zwischen Welch-t-Test (bei Normalverteilung beider
    Gruppen laut Shapiro-Wilk) und Mann-Whitney-U-Test (sonst) gewaehlt.
    Zusaetzlich wird eine Effektstaerke berechnet: Cohen's d bzw. die
    rang-biseriale Korrelation ``r``.

    Args:
        data: Zusammengefuehrter Datensatz mit ``group`` und ``metric``.
        metric: Name der zu vergleichenden Variable.
        config: Aktive Pipeline-Konfiguration.

    Returns:
        Ergebnis-Dict oder ``None``, wenn eine Gruppe weniger als zwei
        gueltige Beobachtungen besitzt.
    """
    if metric not in data.columns or "group" not in data.columns:
        return None

    subset = data[["group", metric]].dropna()
    group_a = subset.loc[subset["group"] == AUDIO, metric].to_numpy(dtype=float)
    group_b = subset.loc[subset["group"] == HAPTIC, metric].to_numpy(dtype=float)
    if group_a.size < 2 or group_b.size < 2:
        LOGGER.warning("Zu wenige Daten fuer den Gruppenvergleich von '%s'.", metric)
        return None

    shapiro_a = _shapiro_p(group_a)
    shapiro_b = _shapiro_p(group_b)
    normal_a = np.isfinite(shapiro_a) and shapiro_a > config.alpha
    normal_b = np.isfinite(shapiro_b) and shapiro_b > config.alpha

    if normal_a and normal_b:
        statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
        test_name, effect_name = "Welch t-test", "Cohen's d"
        effect_size = cohens_d(group_a, group_b)
    else:
        statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        test_name, effect_name = "Mann-Whitney U", "rank-biserial r"
        effect_size = float(2.0 * statistic / (group_a.size * group_b.size) - 1.0)

    return {
        "hypothesis": "",
        "metric": metric,
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_name": effect_name,
        "effect_size": float(effect_size),
        "mean_audio": float(np.mean(group_a)),
        "sd_audio": float(np.std(group_a, ddof=1)),
        "n_audio": int(group_a.size),
        "shapiro_audio": shapiro_a,
        "mean_haptic": float(np.mean(group_b)),
        "sd_haptic": float(np.std(group_b, ddof=1)),
        "n_haptic": int(group_b.size),
        "shapiro_haptic": shapiro_b,
        "significant": bool(p_value < config.alpha),
        "stars": p_to_stars(float(p_value), config.alpha),
    }


def spearman_association(
    data: pd.DataFrame, predictor: str, outcome: str, group: str | None = None
) -> dict[str, Any] | None:
    """Berechnet eine Spearman-Rangkorrelation zwischen zwei Variablen.

    Args:
        data: Datensatz.
        predictor: Praediktorvariable (z. B. Atem-Adhaerenz).
        outcome: Kriteriumsvariable (z. B. ``cards_solved``).
        group: Optionaler Gruppenfilter (``None`` = alle Probanden).

    Returns:
        Ergebnis-Dict oder ``None`` bei zu wenigen Datenpunkten.
    """
    subset = data if group is None else data[data["group"] == group]
    subset = subset[[predictor, outcome]].dropna()
    if len(subset) < 4:
        return None
    # Ohne Varianz in einer der Variablen ist der Rangkorrelationskoeffizient
    # nicht definiert (z. B. Deckeneffekt bei cards_solved).
    if subset[predictor].nunique() < 2 or subset[outcome].nunique() < 2:
        return None

    rho, p_value = stats.spearmanr(subset[predictor], subset[outcome])
    return {
        "predictor": predictor,
        "outcome": outcome,
        "group": group or "gesamt",
        "n": int(len(subset)),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "stars": p_to_stars(float(p_value)),
    }


def run_inference(data: pd.DataFrame, config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fuehrt die hypothesenbezogene Auswertung (H1, H2, H3) durch.

    - H1: Effekt der Modalitaet auf die Atem-Adhaerenz (SPR, Cross-Correlation).
    - H2: Effekt der Modalitaet auf die Spielleistung (``cards_solved``).
    - H3: Zusammenhang zwischen Atem-Adhaerenz und Spielleistung (Spearman).

    Args:
        data: Zusammengefuehrter Datensatz.
        config: Aktive Pipeline-Konfiguration.

    Returns:
        Tupel ``(gruppenvergleiche, korrelationen)`` als DataFrames.
    """
    comparisons: list[dict[str, Any]] = []
    metric_hypotheses = [("spr", "H1"), ("max_cross_corr", "H1")] + [
        (metric, "H2") for metric in PERFORMANCE_METRICS
    ]
    for metric, hypothesis in metric_hypotheses:
        result = compare_groups(data, metric, config)
        if result is not None:
            result["hypothesis"] = hypothesis
            comparisons.append(result)

    associations: list[dict[str, Any]] = []
    for predictor in ADHERENCE_METRICS:
        for outcome in PERFORMANCE_METRICS:
            for group in (None, AUDIO, HAPTIC):
                result = spearman_association(data, predictor, outcome, group)
                if result is not None:
                    result["hypothesis"] = "H3"
                    associations.append(result)

    comparison_df = pd.DataFrame(comparisons)
    association_df = pd.DataFrame(associations)
    return comparison_df, association_df


def format_inference_report(
    comparisons: pd.DataFrame, associations: pd.DataFrame, config: AnalysisConfig
) -> str:
    """Formatiert die Statistikergebnisse als lesbaren Konsolen-/Dateibericht."""
    lines: list[str] = ["=" * 78, "INFERENZSTATISTISCHE AUSWERTUNG", "=" * 78, ""]

    lines.append("H1/H2 - Gruppenvergleich Audio vs. Haptik")
    lines.append("-" * 78)
    if comparisons.empty:
        lines.append("Keine ausreichenden Daten fuer Gruppenvergleiche.")
    else:
        for row in comparisons.itertuples(index=False):
            lines.append(
                f"[{row.hypothesis}] {row.metric:<16} | {row.test:<15} | "
                f"p = {row.p_value:.4f} {row.stars:<4} | {row.effect_size_name} = {row.effect_size:+.3f}"
            )
            lines.append(
                f"    Audio : M = {row.mean_audio:8.4f}, SD = {row.sd_audio:7.4f}, n = {row.n_audio}"
                f" (Shapiro p = {row.shapiro_audio:.3f})"
            )
            lines.append(
                f"    Haptik: M = {row.mean_haptic:8.4f}, SD = {row.sd_haptic:7.4f}, n = {row.n_haptic}"
                f" (Shapiro p = {row.shapiro_haptic:.3f})"
            )
            lines.append("")

    lines.append("H3 - Zusammenhang Atem-Adhaerenz vs. Spielleistung (Spearman)")
    lines.append("-" * 78)
    if associations.empty:
        lines.append("Keine ausreichenden Daten fuer Korrelationsanalysen.")
    else:
        for row in associations.itertuples(index=False):
            lines.append(
                f"{row.predictor:<16} x {row.outcome:<14} [{row.group:<7}] | "
                f"rho = {row.spearman_rho:+.3f} | p = {row.p_value:.4f} {row.stars} | n = {row.n}"
            )

    lines.append("")
    lines.append("Hinweise zur Datenqualitaet")
    lines.append("-" * 78)
    lines.append(
        "Achtung: 'cards_solved' zeigt einen starken Deckeneffekt (viele Probanden am Maximum). "
        "Auch die Bearbeitungszeit ist bei mehreren Probanden auf das 2-Minuten-Limit (120 s) "
        "zensiert ('02:00:00'), was die Varianz und damit die Teststaerke fuer H2/H3 reduziert."
    )
    lines.append(
        "Hinweis: Die Spalte 'Dauer' wurde als Minuten:Sekunden interpretiert "
        "(z. B. '01:36:00' = 96 s). Bei Bedarf ueber 'duration_format=\"hh_mm_ss\"' umstellbar."
    )
    lines.append(
        f"Signifikanzniveau alpha = {config.alpha}; Sternchen: *** p<.001, ** p<.01, * p<{config.alpha}, n.s. nicht signifikant."
    )
    return "\n".join(lines)


def describe_groups(
    data: pd.DataFrame, metrics: Sequence[str], config: AnalysisConfig
) -> pd.DataFrame:
    """Berechnet deskriptive Kennwerte je Gruppe und Metrik.

    Diese Uebersicht wird immer exportiert - auch dann, wenn die Stichprobe
    fuer inferenzstatistische Tests zu klein ist.

    Args:
        data: Zusammengefuehrter Datensatz.
        metrics: Liste der zu beschreibenden Metriken.
        config: Aktive Pipeline-Konfiguration (ungenutzt, fuer Signaturkonsistenz).

    Returns:
        DataFrame mit n, Mittelwert, SD, Median und Spannweite je Zelle.
    """
    del config  # Signaturkonsistenz; aktuell werden keine Parameter benoetigt.
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in data.columns:
            continue
        for group in GROUP_ORDER:
            values = data.loc[data["group"] == group, metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            rows.append(
                {
                    "metric": metric,
                    "group": group,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def format_descriptives(descriptives: pd.DataFrame) -> str:
    """Formatiert die deskriptiven Kennwerte als lesbaren Textblock."""
    lines: list[str] = ["", "Deskriptive Kennwerte je Gruppe", "-" * 78]
    if descriptives.empty:
        lines.append("Keine deskriptiven Kennwerte verfuegbar.")
        return "\n".join(lines)

    for metric in descriptives["metric"].unique():
        lines.append(f"{metric}:")
        block = descriptives[descriptives["metric"] == metric]
        for row in block.itertuples(index=False):
            lines.append(
                f"    {row.group:<7} n = {row.n:<3} | M = {row.mean:8.4f} | SD = {row.sd:8.4f} | "
                f"Median = {row.median:8.4f} | [{row.minimum:.4f}, {row.maximum:.4f}]"
            )
    return "\n".join(lines)


def format_qa_summary(recordings: Sequence[Recording], config: AnalysisConfig) -> str:
    """Formatiert eine kompakte Rohdaten-Qualitaetsuebersicht."""
    lines: list[str] = [
        f"Rohdaten-Qualitaet (Saettigung ab |ac_sig| >= {config.saturation_abs:.0f})",
        "-" * 78,
    ]
    for recording in recordings:
        status = "OK" if recording.valid else "VERWORFEN"
        lines.append(
            f"Proband {recording.subject_id:>2} | {recording.path.name:<16} | "
            f"Saettigung {recording.saturation_ratio:6.1%} | {status:<9} | {recording.notes}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 8. Wissenschaftliche Visualisierungen
# --------------------------------------------------------------------------- #


def plot_group_boxplots(
    data: pd.DataFrame, comparisons: pd.DataFrame, config: AnalysisConfig
) -> Path:
    """Erstellt Vergleichs-Boxplots (Audio vs. Haptik) mit Signifikanzangaben."""
    metrics = [
        metric
        for metric in ("spr", "max_cross_corr", PERFORMANCE_METRIC, TIME_METRIC, EFFICIENCY_METRIC)
        if metric in data.columns
    ]
    if not metrics:
        raise ValueError("Keine Metriken fuer die Gruppen-Boxplots vorhanden.")

    n_cols = 3
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.8 * n_rows), squeeze=False)
    flat_axes = axes.ravel()

    for ax, metric in zip(flat_axes, metrics):
        label = METRIC_LABELS.get(metric, metric)
        available = data[["group", metric]].dropna()
        if available.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=available,
            x="group",
            y=metric,
            order=GROUP_ORDER,
            hue="group",
            palette=GROUP_PALETTE,
            legend=False,
            width=0.55,
            ax=ax,
        )
        sns.stripplot(
            data=available,
            x="group",
            y=metric,
            order=GROUP_ORDER,
            color="black",
            size=4,
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)

        if not comparisons.empty and metric in set(comparisons["metric"]):
            row = comparisons.loc[comparisons["metric"] == metric].iloc[0]
            y_max = available[metric].max()
            y_min = available[metric].min()
            offset = 0.08 * (y_max - y_min) if y_max > y_min else 1.0
            ax.text(
                0.5,
                1.02,
                f"p = {row['p_value']:.3f} {row['stars']}",
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
            )
            ax.set_ylim(y_min - offset, y_max + 2.5 * offset)

    for ax in flat_axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle("Modalitaetsvergleich: Audio vs. Haptik", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = config.plots_dir / "02_group_boxplots.png"
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Gruppen-Boxplots gespeichert: %s", out_path)
    return out_path


def plot_scatter_regression(data: pd.DataFrame, config: AnalysisConfig) -> Path:
    """Erstellt Scatterplots mit Regressionsgeraden (Atem-Adhaerenz vs. Leistung)."""
    adherence = [metric for metric in ADHERENCE_METRICS if metric in data.columns] or ["spr"]
    outcomes = [metric for metric in (PERFORMANCE_METRIC, TIME_METRIC) if metric in data.columns]
    if not outcomes:
        outcomes = [PERFORMANCE_METRIC]

    fig, axes = plt.subplots(
        len(adherence),
        len(outcomes),
        figsize=(7 * len(outcomes), 6 * len(adherence)),
        squeeze=False,
    )

    for row_index, predictor in enumerate(adherence):
        for col_index, outcome in enumerate(outcomes):
            ax = axes[row_index][col_index]
            subset = data[["group", predictor, outcome]].dropna()
            if subset.empty:
                ax.set_visible(False)
                continue

            sns.scatterplot(
                data=subset,
                x=predictor,
                y=outcome,
                hue="group",
                palette=GROUP_PALETTE,
                s=90,
                alpha=0.85,
                ax=ax,
            )
            for group in GROUP_ORDER:
                group_data = subset[subset["group"] == group]
                if len(group_data) >= 3:
                    sns.regplot(
                        data=group_data,
                        x=predictor,
                        y=outcome,
                        scatter=False,
                        color=GROUP_PALETTE[group],
                        ci=None,
                        ax=ax,
                    )

            association = spearman_association(data, predictor, outcome)
            title = (
                f"{METRIC_LABELS.get(predictor, predictor)} vs. "
                f"{METRIC_LABELS.get(outcome, outcome)}"
            )
            if association is not None:
                title += (
                    f"\nSpearman rho = {association['spearman_rho']:+.2f}, "
                    f"p = {association['p_value']:.3f} {association['stars']}"
                )
            ax.set_title(title)
            ax.set_xlabel(METRIC_LABELS.get(predictor, predictor))
            ax.set_ylabel(METRIC_LABELS.get(outcome, outcome))
            ax.grid(alpha=0.3)

    fig.suptitle("Atem-Adhaerenz und Aufgabenleistung", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = config.plots_dir / "03_scatter_regression.png"
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Scatterplot mit Regression gespeichert: %s", out_path)
    return out_path


def plot_mean_psd(
    features: Sequence[BreathFeatures], groups: dict[int, str], config: AnalysisConfig
) -> Path:
    """Stellt das mittlere Leistungsspektrum je Gruppe der Ziel-Frequenz gegenueber."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Gemeinsames Frequenzraster, da die PSD-Laengen je nach Signallaenge variieren.
    grid = np.linspace(config.bandpass_low, 0.2, 400)

    for group in GROUP_ORDER:
        members = [feature for feature in features if groups.get(feature.subject_id) == group]
        if not members:
            continue
        spectra = [np.interp(grid, feature.psd_freqs, feature.psd_values) for feature in members]
        stacked = np.vstack(spectra)
        mean_psd = stacked.mean(axis=0)
        sem = (
            stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            if stacked.shape[0] > 1
            else np.zeros_like(grid)
        )

        ax.plot(grid, mean_psd, label=f"{group.title()} (n={stacked.shape[0]})", color=GROUP_PALETTE[group])
        ax.fill_between(grid, mean_psd - sem, mean_psd + sem, color=GROUP_PALETTE[group], alpha=0.25)

    ax.axvline(
        config.target_freq,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Ziel {config.target_freq:.4f} Hz (16 s)",
    )
    ax.axvspan(config.spr_band_low, config.spr_band_high, color="grey", alpha=0.12, label="SPR-Zielband")
    ax.set_xlim(config.bandpass_low, 0.2)
    ax.set_xlabel("Frequenz [Hz]")
    ax.set_ylabel("Leistungsdichte [a.u.]")
    ax.set_title("Mittleres Leistungsspektrum der Atmung (Welch-PSD)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = config.plots_dir / "04_mean_psd.png"
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Mittleres Leistungsspektrum gespeichert: %s", out_path)
    return out_path


def plot_correlation_heatmap(data: pd.DataFrame, config: AnalysisConfig) -> Path:
    """Erstellt eine Spearman-Korrelations-Heatmap aller zentralen Variablen."""
    candidate_columns = [
        "spr",
        "max_cross_corr",
        "cv_cycle",
        "mean_cycle_duration_sec",
        PERFORMANCE_METRIC,
        TIME_METRIC,
        EFFICIENCY_METRIC,
        "distraction",
        "clarity",
        "signal_perception",
        "signal_support",
        "dual_task_ease",
        "comfort",
    ]
    columns = [column for column in candidate_columns if column in data.columns]
    correlation_data = data[columns].apply(pd.to_numeric, errors="coerce")
    correlation_data = correlation_data.dropna(axis=1, how="all")

    if correlation_data.shape[1] < 2:
        LOGGER.warning("Zu wenige Variablen fuer die Korrelations-Heatmap.")
        correlation_data = data.select_dtypes("number")
        correlation_data = correlation_data.dropna(axis=1, how="all")

    corr = correlation_data.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman rho"},
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Korrelations-Heatmap: Adhaerenz, Leistung und Fragebogen-Items")
    fig.tight_layout()
    out_path = config.plots_dir / "05_correlation_heatmap.png"
    fig.savefig(out_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Korrelations-Heatmap gespeichert: %s", out_path)
    return out_path


def create_all_figures(
    data: pd.DataFrame,
    comparisons: pd.DataFrame,
    features: Sequence[BreathFeatures],
    config: AnalysisConfig,
) -> list[Path]:
    """Erzeugt saemtliche Publikations-Abbildungen und gibt deren Pfade zurueck."""
    groups = dict(zip(data["subject_id"], data["group"])) if not data.empty else {}
    outputs = [
        plot_group_boxplots(data, comparisons, config),
        plot_scatter_regression(data, config),
        plot_mean_psd(features, groups, config),
        plot_correlation_heatmap(data, config),
    ]
    return outputs


# --------------------------------------------------------------------------- #
# 9. Orchestrierung
# --------------------------------------------------------------------------- #


def save_results(
    data: pd.DataFrame,
    comparisons: pd.DataFrame,
    associations: pd.DataFrame,
    descriptives: pd.DataFrame,
    report: str,
    config: AnalysisConfig,
) -> dict[str, Path]:
    """Schreibt den zusammengefuehrten Datensatz und die Statistikergebnisse."""
    outputs: dict[str, Path] = {}

    merged_path = config.results_dir / "merged_dataset.csv"
    data.to_csv(merged_path, index=False)
    outputs["merged_dataset"] = merged_path

    descriptives_path = config.results_dir / "descriptives.csv"
    descriptives.to_csv(descriptives_path, index=False)
    outputs["descriptives"] = descriptives_path

    comparison_columns = [
        "hypothesis",
        "metric",
        "test",
        "statistic",
        "p_value",
        "effect_size_name",
        "effect_size",
        "mean_audio",
        "sd_audio",
        "n_audio",
        "shapiro_audio",
        "mean_haptic",
        "sd_haptic",
        "n_haptic",
        "shapiro_haptic",
        "significant",
        "stars",
        "result_type",
    ]
    association_columns = [
        "hypothesis",
        "predictor",
        "outcome",
        "group",
        "n",
        "spearman_rho",
        "p_value",
        "significant",
        "stars",
        "result_type",
    ]
    schema = comparison_columns + [c for c in association_columns if c not in comparison_columns]

    statistics_path = config.results_dir / "statistics_results.csv"
    combined = pd.concat(
        [
            comparisons.assign(result_type="group_comparison"),
            associations.assign(result_type="association"),
        ],
        ignore_index=True,
    ).reindex(columns=schema)
    combined.to_csv(statistics_path, index=False)
    outputs["statistics_results"] = statistics_path

    report_path = config.results_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    outputs["report"] = report_path

    LOGGER.info("Ergebnisse exportiert: %s", ", ".join(str(path) for path in outputs.values()))
    return outputs


def run_analysis(config: AnalysisConfig | None = None) -> dict[str, Any]:
    """Fuehrt die komplette Auswertungs-Pipeline aus.

    Args:
        config: Optionale Konfiguration; andernfalls werden Defaults verwendet.

    Returns:
        Dict mit den wichtigsten Ergebnissen (Pfade, DataFrames, Report).
    """
    config = config or AnalysisConfig()
    ensure_directories(config)

    sns.set_theme(style="whitegrid", context="talk")

    # 2. Rohdaten laden und 3. auf Saettigung pruefen.
    recordings = load_all_recordings(config)
    if not recordings:
        return {"error": "Keine Messdateien gefunden."}
    recordings = [assess_saturation(recording, config) for recording in recordings]
    plot_raw_overview(recordings, config)

    for recording in recordings:
        LOGGER.info(
            "Proband %2d | %-28s | %s",
            recording.subject_id,
            recording.path.name,
            recording.notes,
        )

    # 4./5. Preprocessing und Feature-Extraktion (nur gueltige Aufnahmen).
    features: list[BreathFeatures] = []
    for recording in recordings:
        if not recording.valid:
            LOGGER.warning(
                "Proband %s wird uebersprungen (%s)", recording.subject_id, recording.notes
            )
            continue
        processed = preprocess_recording(recording, config)
        if processed is None:
            continue
        features.append(extract_features(processed, config))

    if not features:
        return {"error": "Keine verwertbaren Signale nach dem Preprocessing."}

    # 6. Metadaten laden und zusammenfuehren.
    metadata = load_metadata(config)
    data = merge_dataset(features, metadata, config)
    if data.empty:
        return {"error": "Keine Uebereinstimmung zwischen Messungen und Metadaten."}

    # 7. Inferenzstatistik und deskriptive Kennwerte.
    comparisons, associations = run_inference(data, config)
    descriptives = describe_groups(data, DESCRIPTIVE_METRICS, config)
    report = "\n".join(
        [
            format_qa_summary(recordings, config),
            "",
            format_inference_report(comparisons, associations, config),
            format_descriptives(descriptives),
        ]
    )
    print(report)

    # 8. Abbildungen.
    create_all_figures(data, comparisons, features, config)

    # Exporte.
    outputs = save_results(data, comparisons, associations, descriptives, report, config)

    return {
        "data": data,
        "comparisons": comparisons,
        "associations": associations,
        "descriptives": descriptives,
        "report": report,
        "outputs": outputs,
        "features": features,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Erstellt den CLI-Parser fuer die Pipeline."""
    parser = argparse.ArgumentParser(
        description="Auswertung: Audio vs. Haptic Breathing Guidance (Box Breathing, 0.0625 Hz)."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Ordner der Mess-CSV-Dateien.")
    parser.add_argument("--metadata", type=Path, default=Path("metadaten.xlsx"), help="Pfad zur Metadaten-Excel.")
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"), help="Zielordner der Abbildungen.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Zielordner der Ergebnisse.")
    parser.add_argument("--window-sec", type=float, default=120.0, help="Auswertungsfenster in Sekunden.")
    parser.add_argument("--log-level", default="INFO", help="Log-Level (DEBUG, INFO, WARNING, ERROR).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI-Einstiegspunkt der Auswertung."""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)-8s | %(message)s",
    )

    config = AnalysisConfig(
        runs_dir=args.runs_dir,
        metadata_path=args.metadata,
        plots_dir=args.plots_dir,
        results_dir=args.results_dir,
        window_sec=args.window_sec,
    )
    result = run_analysis(config)
    if "error" in result:
        LOGGER.error(result["error"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
