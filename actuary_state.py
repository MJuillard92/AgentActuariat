"""
actuary_state.py
Stockage partagé du kernel actif et cache d'embeddings pour le RAG.

Le kernel est une référence vivante : quand le workflow/agent le modifie,
le RAG voit automatiquement les nouveaux DataFrames sans copie.

Patron singleton thread-safe
-----------------------------
L'objet STATE en bas du fichier est l'instance unique d'ActuaryState.
_get_singleton() vérifie que sys.modules["actuary_state"].STATE existe déjà
avant d'en créer un nouveau — cela garantit qu'un rechargement de module
(p. ex. via importlib.reload) ne détruira pas le kernel en cours d'analyse.

Pourquoi un singleton ?

- canvas_app.py (UI Streamlit), agent.py et rag.py tournent dans le même
  processus mais sont importés indépendamment. Sans singleton, chacun aurait
  sa propre copie du kernel et les DataFrames calculés par l'agent ne seraient
  pas visibles par le RAG.

Gestion de la concurrence
--------------------------
Un threading.Lock() protège toutes les lectures/écritures. Streamlit peut
invoquer des callbacks depuis plusieurs threads (reruns), et les embeddings
OpenAI sont calculés en tâche de fond — sans verrou, les deux pourraient
écraser le cache simultanément.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import threading
import types
from typing import Any

import numpy as np
import pandas as pd


class ActuaryState:
    """Singleton thread-safe : référence vivante sur le kernel + cache embeddings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kernel: dict[str, Any] = {}
        self._embed_cache: dict = {}
        self._rag_ns: dict[str, Any] = {}   # namespace persistant entre les tours RAG
        self._pdf_chunks: list[dict] = []   # chunks PDF uploadés par l'utilisateur
        self._report_template: dict | None = None  # template JSON analysé

    # ── Kernel / namespace ──────────────────────────────────────────────────

    def set_kernel(self, kernel: dict) -> None:
        """Pointe vers le kernel actif (référence directe, pas de copie).
        Réinitialise aussi le namespace RAG pour repartir sur une base propre."""
        with self._lock:
            self._kernel = kernel
            self._embed_cache = {}
            self._rag_ns = {}   # reset RAG namespace quand nouvelle analyse

    def get_exec_namespace(self) -> dict[str, Any]:
        """Namespace RAG persistant = kernel + variables calculées entre les tours.

        Fusion : le kernel de base + tout ce que le RAG a calculé précédemment.
        Les variables du RAG ont priorité (permettent de surcharger les valeurs).

        On retourne une COPIE superficielle du kernel pour que execute_python()
        dans le RAG ne modifie pas involontairement le kernel principal partagé.
        Les nouveaux objets créés (DataFrames intermédiaires, etc.) sont ensuite
        récupérés via update_rag_ns() et stockés dans _rag_ns.
        """
        with self._lock:
            ns = dict(self._kernel)
            ns.update(self._rag_ns)     # les calculs RAG persistent entre les questions
            return ns

    def update_rag_ns(self, ns: dict[str, Any]) -> None:
        """Persiste les nouvelles variables calculées par le RAG dans _rag_ns.

        On filtre délibérément les modules et callables pour ne conserver que
        des objets sérialisables/comparables. Les variables du kernel principal
        ne sont jamais écrasées : le RAG peut calculer des sous-ensembles mais
        ne doit pas modifier les tables de référence (df_exposure, df_smooth…).
        """
        import pandas as pd
        import numpy as np
        with self._lock:
            for k, v in ns.items():
                # Ne persister que les objets de données (pas les modules, fonctions, etc.)
                if k.startswith("_") or callable(v) or isinstance(v, type):
                    continue
                if k in self._kernel:
                    continue        # ne pas écraser les variables du kernel principal
                if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray,
                                  int, float, bool, str, list, dict)):
                    self._rag_ns[k] = v

    def reset_rag_ns(self) -> None:
        """Efface le namespace RAG (appelé quand l'utilisateur efface le chat)."""
        with self._lock:
            self._rag_ns = {}

    def clear_kernel(self) -> None:
        with self._lock:
            self._kernel = {}
            self._embed_cache = {}
            self._rag_ns = {}

    # ── Chunks PDF ──────────────────────────────────────────────────────────

    def add_pdf_chunks(self, chunks: list[dict]) -> None:
        """Ajoute des chunks PDF au contexte RAG (non effacés par set_kernel)."""
        with self._lock:
            self._pdf_chunks.extend(chunks)
        self.invalidate_cache()

    def get_pdf_chunks(self) -> list[dict]:
        with self._lock:
            return list(self._pdf_chunks)

    def clear_pdf_chunks(self) -> None:
        with self._lock:
            self._pdf_chunks = []
        self.invalidate_cache()

    # ── Template de rapport ──────────────────────────────────────────────────

    def set_template(self, template: dict) -> None:
        """Stocke le template JSON analysé (sections, tableaux, graphiques, prompts)."""
        with self._lock:
            self._report_template = template

    def get_template(self) -> dict | None:
        with self._lock:
            return self._report_template

    def clear_template(self) -> None:
        with self._lock:
            self._report_template = None

    # ── Cache d'embeddings ──────────────────────────────────────────────────

    def _hash(self, chunks: list[dict]) -> str:
        return hashlib.md5(
            json.dumps([c["text"][:80] for c in chunks]).encode()
        ).hexdigest()

    def set_embed_cache(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        h = self._hash(chunks)
        with self._lock:
            self._embed_cache = {
                "hash": h,
                "chunks": chunks,
                "embeddings": embeddings,
            }

    def get_embed_cache(self, chunks: list[dict]):
        """Retourne (chunks, embeddings) si le cache est valide, sinon None."""
        if not self._embed_cache:
            return None
        h = self._hash(chunks)
        with self._lock:
            if self._embed_cache.get("hash") == h:
                return self._embed_cache["chunks"], self._embed_cache["embeddings"]
        return None

    def invalidate_cache(self) -> None:
        with self._lock:
            self._embed_cache = {}

    # ── Résumé du namespace pour le LLM ─────────────────────────────────────

    @staticmethod
    def _describe_value(name: str, val: Any) -> str | None:
        """Retourne une ligne de description pour une valeur du kernel, ou None."""
        try:
            if isinstance(val, pd.DataFrame):
                cols = list(val.columns[:8])
                extra = "…" if len(val.columns) > 8 else ""
                return (
                    f"  • {name}: DataFrame {val.shape[0]}×{val.shape[1]}"
                    f" — colonnes: {cols}{extra}"
                )
            elif isinstance(val, pd.Series):
                return f"  • {name}: Series ({len(val)} éléments)"
            elif isinstance(val, np.ndarray):
                return f"  • {name}: array numpy {val.shape}"
            elif isinstance(val, (int, float, bool)):
                return f"  • {name} = {val!r}"
            elif isinstance(val, str) and len(val) < 120:
                return f"  • {name} = {val!r}"
        except Exception:
            pass
        return None

    def summary(self) -> str:
        lines = ["=== Objets disponibles dans le namespace de l'analyse ==="]
        with self._lock:
            for name, val in self._kernel.items():
                if name.startswith("_") or isinstance(val, type):
                    continue
                # Variables scalaires / DataFrames directes
                if not callable(val) and not isinstance(val, types.ModuleType):
                    desc = self._describe_value(name, val)
                    if desc:
                        lines.append(desc)
                # Modules actuariels : lister leurs DataFrames / Series / scalaires
                elif isinstance(val, types.ModuleType):
                    mod_attrs = []
                    for attr in dir(val):
                        if attr.startswith("_"):
                            continue
                        try:
                            aval = getattr(val, attr)
                        except Exception:
                            continue
                        desc = self._describe_value(f"{name}.{attr}", aval)
                        if desc:
                            mod_attrs.append(desc)
                    if mod_attrs:
                        lines.append(f"  [module {name}]")
                        lines.extend(mod_attrs)
        return (
            "\n".join(lines)
            if len(lines) > 1
            else "Aucun état disponible (lancez une analyse)."
        )


def _get_singleton() -> ActuaryState:
    """Retourne l'instance STATE existante si le module a déjà été chargé.

    Ce mécanisme protège contre les rechargements de module (importlib.reload
    ou imports multiples dans des contextes différents) : si STATE existe déjà
    dans sys.modules["actuary_state"], on le réutilise au lieu d'en créer un
    nouveau qui perdrait le kernel courant.
    """
    mod = sys.modules.get("actuary_state")
    if mod is not None and hasattr(mod, "STATE") and isinstance(mod.STATE, ActuaryState):
        return mod.STATE
    return ActuaryState()


# Instance unique partagée par toute l'application.
# Importée dans canvas_app.py, agent.py et rag.py via :
#   from actuary_state import STATE
STATE: ActuaryState = _get_singleton()
