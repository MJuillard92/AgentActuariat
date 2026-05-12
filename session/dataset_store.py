"""
session/dataset_store.py
Persistance du DataFrame initial — écriture unique, lecture seule ensuite.

Règles :
  - store() est idempotent : si l'artefact existe déjà pour cette session, ne fait rien.
  - Le DataFrame n'est jamais modifié après le chargement initial.
  - Seule la référence (DatasetMeta) circule dans l'état LangGraph.
"""
from __future__ import annotations

import hashlib
import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from session.session_state import DatasetMeta

_DATA_DIR      = Path(__file__).resolve().parent / "data"
_ARTIFACTS_DIR = _DATA_DIR / "artifacts"


class DatasetStore:
    """
    Gère la persistance du DataFrame initial d'une session.

    Usage :
        meta = DatasetStore.store(session_id, df)   # au chargement CSV
        df   = DatasetStore.load(meta)              # dans les nodes
    """

    @staticmethod
    def store(session_id: str, df: pd.DataFrame) -> DatasetMeta:
        """
        Persiste le DataFrame en Parquet.
        Idempotent : si le fichier existe déjà pour cette session, retourne
        directement le DatasetMeta existant sans réécrire.

        Args:
            session_id : identifiant de session (ex. "2604021636")
            df         : DataFrame à persister

        Returns:
            DatasetMeta avec path, hash, shape et colonnes.
        """
        _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = _ARTIFACTS_DIR / f"{session_id}_dataset.parquet"

        if path.exists():
            # Déjà stocké pour cette session — reconstruire le meta depuis le fichier
            existing_df = pd.read_parquet(path)
            sha = DatasetStore._sha256(existing_df)
            return DatasetMeta(
                path       = str(path),
                sha256     = sha,
                n_rows     = len(existing_df),
                n_cols     = len(existing_df.columns),
                columns    = list(existing_df.columns),
                created_at = datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            )

        # Première écriture
        df.to_parquet(path, index=False)
        sha = DatasetStore._sha256(df)
        return DatasetMeta(
            path       = str(path),
            sha256     = sha,
            n_rows     = len(df),
            n_cols     = len(df.columns),
            columns    = list(df.columns),
            created_at = datetime.datetime.now().isoformat(),
        )

    @staticmethod
    def load(meta: DatasetMeta) -> pd.DataFrame:
        """
        Charge le DataFrame depuis l'artefact Parquet.

        Raises:
            FileNotFoundError si l'artefact n'existe plus.
        """
        path = Path(meta.path)
        if not path.exists():
            raise FileNotFoundError(
                f"Artefact dataset introuvable : {meta.path}. "
                "Le fichier a peut-être été supprimé manuellement."
            )
        return pd.read_parquet(path)

    @staticmethod
    def load_by_session(session_id: str) -> Optional[pd.DataFrame]:
        """
        Charge le DataFrame d'une session depuis son session_id.
        Retourne None si aucun artefact n'existe.
        """
        path = _ARTIFACTS_DIR / f"{session_id}_dataset.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    @staticmethod
    def load_preferring_normalized(
        data_store: dict | None,
        session_id: str | None,
    ) -> Optional[pd.DataFrame]:
        """
        Charge le DataFrame en préférant le Parquet normalisé écrit par
        `maybe_normalize_records()` après validation UI des mappings.

        Ordre :
          1. `data_store["dataset_ref_normalized"]` si présent et fichier OK
          2. `<session_id>_dataset.parquet` (original) en fallback

        Retourne None si rien de chargeable.
        """
        ds = data_store or {}
        norm_path = ds.get("dataset_ref_normalized")
        if norm_path:
            p = Path(str(norm_path))
            if p.exists():
                try:
                    return pd.read_parquet(p)
                except Exception:
                    pass
        if not session_id:
            return None
        return DatasetStore.load_by_session(session_id)

    @staticmethod
    def exists(session_id: str) -> bool:
        """Vérifie si un artefact existe pour cette session."""
        return (_ARTIFACTS_DIR / f"{session_id}_dataset.parquet").exists()

    @staticmethod
    def _sha256(df: pd.DataFrame) -> str:
        """Hash SHA-256 sur les 12 premiers caractères du contenu CSV."""
        raw = df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]
