"""
workflow.py
Modèle de données du workflow d'orchestration :
- nœuds  = notebooks à exécuter
- arêtes = ordre d'exécution + conditions métier optionnelles
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Modèle de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkflowNode:
    """Représente un notebook dans le canvas."""
    id: str                          # identifiant unique (ex. "nb01")
    notebook_path: str               # chemin vers le .ipynb
    label: str                       # nom affiché sur le canvas
    description: str = ""            # sous-titre
    x: float = 100.0                 # position X sur le canvas
    y: float = 100.0                 # position Y sur le canvas
    color: str = "#4A90D9"           # couleur du nœud


@dataclass
class WorkflowEdge:
    """Lien entre deux nœuds, avec condition optionnelle."""
    id: str                          # identifiant unique (ex. "e01-02")
    source: str                      # id du nœud source
    target: str                      # id du nœud cible
    condition: Optional[str] = None  # ex. "SMR > 1.2" ou None (toujours exécuter)
    label: str = ""                  # label affiché sur l'arête


@dataclass
class Workflow:
    """Graphe complet du workflow."""
    name: str = "Analyse mortalité"
    nodes: list = field(default_factory=list)  # list[WorkflowNode]
    edges: list = field(default_factory=list)  # list[WorkflowEdge]

    # ── Sérialisation ────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Workflow":
        nodes = [WorkflowNode(**n) for n in d.get("nodes", [])]
        edges = [WorkflowEdge(**e) for e in d.get("edges", [])]
        return cls(name=d.get("name", "Workflow"), nodes=nodes, edges=edges)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Workflow":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ── Gestion des nœuds et arêtes ──────────────────────────────────────────
    def add_node(self, node: WorkflowNode) -> None:
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)

    def remove_node(self, node_id: str) -> None:
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]

    def add_edge(self, edge: WorkflowEdge) -> None:
        if not any(e.id == edge.id for e in self.edges):
            self.edges.append(edge)

    def remove_edge(self, edge_id: str) -> None:
        self.edges = [e for e in self.edges if e.id != edge_id]

    def update_node_position(self, node_id: str, x: float, y: float) -> None:
        for n in self.nodes:
            if n.id == node_id:
                n.x, n.y = x, y
                break

    # ── Topologie ─────────────────────────────────────────────────────────────
    def execution_order(self) -> list[str]:
        """Retourne les ids de nœuds dans l'ordre topologique (Kahn)."""
        in_degree = {n.id: 0 for n in self.nodes}
        adjacency = {n.id: [] for n in self.nodes}
        for e in self.edges:
            if e.source in adjacency and e.target in in_degree:
                adjacency[e.source].append(e.target)
                in_degree[e.target] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []
        while queue:
            nid = queue.pop(0)
            result.append(nid)
            for succ in adjacency[nid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return result

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        return next((n for n in self.nodes if n.id == node_id), None)

    def get_edges_from(self, node_id: str) -> list:
        return [e for e in self.edges if e.source == node_id]

    # ── Conversion format Cytoscape ──────────────────────────────────────────
    def to_cytoscape_elements(self) -> list[dict]:
        """Format attendu par dash-cytoscape."""
        elements = []
        for n in self.nodes:
            elements.append({
                "data": {
                    "id": n.id,
                    "label": n.label,
                    "description": n.description,
                    "notebook_path": n.notebook_path,
                    "color": n.color,
                },
                "position": {"x": n.x, "y": n.y},
                "classes": "notebook-node",
            })
        for e in self.edges:
            elements.append({
                "data": {
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "label": e.label or (e.condition or ""),
                    "condition": e.condition or "",
                },
                "classes": "conditional-edge" if e.condition else "default-edge",
            })
        return elements

    @classmethod
    def from_cytoscape_elements(cls, elements: list[dict], name: str = "Workflow") -> "Workflow":
        """Reconstruit le Workflow depuis les éléments Cytoscape (après drag)."""
        nodes, edges = [], []
        for el in elements:
            d = el.get("data", {})
            if "source" in d:  # arête
                edges.append(WorkflowEdge(
                    id=d.get("id", ""),
                    source=d["source"],
                    target=d["target"],
                    condition=d.get("condition") or None,
                    label=d.get("label", ""),
                ))
            else:  # nœud
                pos = el.get("position", {})
                nodes.append(WorkflowNode(
                    id=d.get("id", ""),
                    notebook_path=d.get("notebook_path", ""),
                    label=d.get("label", ""),
                    description=d.get("description", ""),
                    x=pos.get("x", 100),
                    y=pos.get("y", 100),
                    color=d.get("color", "#4A90D9"),
                ))
        return cls(name=name, nodes=nodes, edges=edges)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow par défaut (7 notebooks en séquence linéaire)
# ─────────────────────────────────────────────────────────────────────────────
_NOTEBOOK_META = [
    ("nb01", "01_chargement_donnees.ipynb",          "1. Chargement",       "Imports et données",          "#2196F3", 100, 200),
    ("nb02", "02_controle_qualite.ipynb",             "2. Contrôle qualité", "Validation des données",      "#4CAF50", 300, 200),
    ("nb03", "03_expositions_taux_bruts.ipynb",       "3. Expositions",      "E_x, D_x, q_x bruts",        "#FF9800", 500, 200),
    ("nb04", "04_lissage_whittaker_henderson.ipynb",  "4. Lissage WH",       "Whittaker-Henderson λ=100",   "#9C27B0", 700, 200),
    ("nb05", "05_visualisation.ipynb",                "5. Visualisation",    "Taux bruts vs lissés",        "#00BCD4", 900, 200),
    ("nb06", "06_smr.ipynb",                          "6. SMR",              "Ratio observé/attendu",       "#F44336", 1100, 200),
    ("nb07", "07_export.ipynb",                       "7. Export",           "Export CSV table finale",     "#607D8B", 1300, 200),
]


def default_workflow(notebooks_dir: str = "./notebooks") -> Workflow:
    """Crée le workflow linéaire standard (7 notebooks en séquence)."""
    wf = Workflow(name="Analyse mortalité — Workflow standard")
    ids = []
    for nid, fname, label, desc, color, x, y in _NOTEBOOK_META:
        path = str(Path(notebooks_dir) / fname)
        if Path(path).exists():
            wf.add_node(WorkflowNode(
                id=nid, notebook_path=path,
                label=label, description=desc,
                color=color, x=x, y=y,
            ))
            ids.append(nid)

    # Arêtes linéaires
    for i in range(len(ids) - 1):
        src, tgt = ids[i], ids[i + 1]
        wf.add_edge(WorkflowEdge(id=f"e{src}-{tgt}", source=src, target=tgt))

    return wf
