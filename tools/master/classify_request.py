"""
TOOL CONTRACT — master.classify_request
═══════════════════════════════════════

CATALOGUE METADATA
------------------
name          : master.classify_request
domain        : master
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-20

DESCRIPTION
-----------
Classifie une demande utilisateur en langage naturel vers un objectif
méthodologique parmi une liste fermée. V1 trivial : retourne toujours
'construction_table_mortalite' (seul objectif supporté aujourd'hui).

WHEN TO USE
-----------
- Au démarrage de session Master, pour inférer study_objective.

INPUTS
------
params:
  request:
    type    : string
    note    : Demande utilisateur en langage naturel.

OUTPUTS
-------
return_payload:
  objective    : string
  gender_mode  : string — unisex | by_sex
"""
# TODO: V2 — classification LLM multi-classes (provisionnement, tarification,
# best estimate, ...). Élargir l'enum `allowed` du YAML en conséquence.
from __future__ import annotations


_ALLOWED = ("construction_table_mortalite",)

_BY_SEX_PATTERNS = ("h/f", "h / f", "par sexe", "par genre", "masculin et féminin")


def run(data: dict, params: dict) -> dict:
    request = str(data.get("request", "")).lower()
    gender_mode = "by_sex" if any(p in request for p in _BY_SEX_PATTERNS) else "unisex"
    return {
        "objective":   _ALLOWED[0],
        "gender_mode": gender_mode,
    }
