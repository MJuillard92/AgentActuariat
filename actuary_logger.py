"""
actuary_logger.py
=================
Singleton de logging structuré pour la bibliothèque actuarielle.

Chaque appel de fonction clé enregistre :
  - la fonction appelée
  - un message lisible en français
  - un dict de métriques numériques clés

Le logger est partagé entre tous les modules via sys.modules pour garantir
une seule instance même quand les modules sont chargés via importlib.

Usage dans un module actuariel :
    from actuary_logger import LOGGER
    LOGGER.log("compute_smr", "SMR calculé", {"smr": 1.04, "d_obs": 234})

Usage pour le RAG / rapport :
    chunks = LOGGER.to_chunks()     # pour l'indexation RAG
    text   = LOGGER.to_report()     # pour le rapport Word
    LOGGER.clear()                  # avant chaque nouvelle analyse
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

# Répertoire où les logs de chaque analyse sont sauvegardés
_LOG_DIR = Path(__file__).parent / "logs"


# ─────────────────────────────────────────────────────────────────────────────
# Entrée de log
# ─────────────────────────────────────────────────────────────────────────────

class LogEntry:
    __slots__ = ("ts", "function", "message", "metrics")

    def __init__(self, function: str, message: str, metrics: dict[str, Any]):
        self.ts       = datetime.now().isoformat(timespec="seconds")
        self.function = function
        self.message  = message
        self.metrics  = metrics or {}

    def to_text(self) -> str:
        """Représentation textuelle riche pour le RAG."""
        lines = [f"[{self.function}]  {self.message}"]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k} = {v:.4g}")
            elif isinstance(v, list) and len(v) > 8:
                lines.append(f"  {k} = {v[:4]} … {v[-2:]} (n={len(v)})")
            else:
                lines.append(f"  {k} = {v}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"LogEntry({self.function!r}, {self.message!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class ActuaryLogger:
    """Logger structuré thread-safe — singleton partagé entre tous les modules.

    Chaque analyse écrit ses logs dans logs/run_YYYYMMDD_HHMMSS.jsonl
    (une ligne JSON par entrée). Le fichier est créé au premier appel à log()
    après un clear() et reste ouvert jusqu'au prochain clear().
    Format JSONL : une entrée par ligne, lisible avec pandas.read_json(lines=True).
    """

    def __init__(self):
        self._entries: list[LogEntry] = []
        self._lock = threading.Lock()
        self._log_file: Path | None = None   # fichier du run courant

    def log(self, function: str, message: str, metrics: dict[str, Any] | None = None) -> None:
        """Enregistre une entrée et l'écrit immédiatement sur disque (JSONL).

        Args:
            function: Nom de la fonction (ex: "compute_smr").
            message:  Phrase en français décrivant le résultat.
            metrics:  Dict de métriques numériques ou textuelles clés.
        """
        entry = LogEntry(function, message, metrics or {})
        with self._lock:
            self._entries.append(entry)
            self._append_to_file(entry)

    def _append_to_file(self, entry: LogEntry) -> None:
        """Ouvre (ou crée) le fichier JSONL du run courant et ajoute l'entrée."""
        try:
            if self._log_file is None:
                _LOG_DIR.mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._log_file = _LOG_DIR / f"run_{ts}.jsonl"
            record = {
                "ts": entry.ts,
                "function": entry.function,
                "message": entry.message,
                "metrics": {
                    k: (float(v) if hasattr(v, "__float__") and not isinstance(v, (str, bool, list, dict)) else v)
                    for k, v in entry.metrics.items()
                },
            }
            with self._log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass   # le log sur disque est best-effort : on ne plante jamais l'analyse

    def clear(self) -> None:
        """Vide le log en mémoire et démarre un nouveau fichier au prochain log()."""
        with self._lock:
            self._entries.clear()
            self._log_file = None   # sera recréé au premier log() du prochain run

    def get_entries(self) -> list[LogEntry]:
        with self._lock:
            return list(self._entries)

    # ── Export RAG ────────────────────────────────────────────────────────────

    def to_chunks(self) -> list[dict]:
        """Retourne les entrées sous forme de chunks pour l'indexation RAG.

        Chaque chunk : {"text": str, "label": str}
        Les chunks d'une même fonction sont regroupés.
        """
        with self._lock:
            entries = list(self._entries)

        # Grouper par fonction
        groups: dict[str, list[LogEntry]] = {}
        for e in entries:
            groups.setdefault(e.function, []).append(e)

        chunks = []
        for fn, elist in groups.items():
            text = "\n\n".join(e.to_text() for e in elist)
            chunks.append({"text": text, "label": fn})
        return chunks

    def to_report(self) -> str:
        """Retourne une transcription textuelle complète pour le rapport Word."""
        with self._lock:
            entries = list(self._entries)
        if not entries:
            return ""
        lines = []
        for e in entries:
            lines.append(e.to_text())
            lines.append("")
        return "\n".join(lines)

    def summary_for_step(self, function_prefix: str) -> str:
        """Retourne le texte de log pour toutes les fonctions commençant par un préfixe."""
        with self._lock:
            matching = [e for e in self._entries
                        if e.function.startswith(function_prefix)]
        return "\n\n".join(e.to_text() for e in matching)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        return f"ActuaryLogger({len(self)} entries)"


# ─────────────────────────────────────────────────────────────────────────────
# Singleton — garanti unique même si le module est rechargé via importlib
# ─────────────────────────────────────────────────────────────────────────────

def _get_singleton() -> ActuaryLogger:
    mod = sys.modules.get("actuary_logger")
    if mod is not None and hasattr(mod, "LOGGER") and isinstance(mod.LOGGER, ActuaryLogger):
        return mod.LOGGER
    return ActuaryLogger()


LOGGER: ActuaryLogger = _get_singleton()
