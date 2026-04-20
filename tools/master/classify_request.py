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
  objective : string
"""
# TODO: V2 — classification LLM multi-classes (provisionnement, tarification,
# best estimate, ...). Élargir l'enum `allowed` du YAML en conséquence.
from __future__ import annotations


_ALLOWED = ("construction_table_mortalite",)


def run(data: dict, params: dict) -> dict:
    _ = data.get("request", "")
    return {"objective": _ALLOWED[0]}
