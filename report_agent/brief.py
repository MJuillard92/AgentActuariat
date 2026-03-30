"""
report_agent/brief.py
Contrat de données entre le WriterAgent (rédacteur maître) et le builder (esclave).

Le Brief JSON est produit par le rédacteur lors du dialogue utilisateur.
Il définit exactement ce que le builder doit calculer.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ─── Capacités du builder ──────────────────────────────────────────────────────
# Ce dictionnaire est la source de vérité sur ce que le builder sait faire.
BUILDER_CAPABILITIES: dict[str, dict] = {
    "mortalite_brute": {
        "disponible": True,
        "description": "Calcul des taux bruts q_brut = D_x / E_x",
        "variables": ["ages", "exposure", "deaths_observed", "q_brut"],
    },
    "lissage_wh": {
        "disponible": True,
        "description": "Lissage Whittaker-Henderson (minimisation critère pénalisé)",
        "variables": ["q_lisse", "ic_inf", "ic_sup"],
    },
    "lissage_gompertz": {
        "disponible": True,
        "description": "Ajustement Gompertz-Makeham (paramétrique)",
        "variables": ["q_lisse", "ic_inf", "ic_sup"],
    },
    "lissage_beard": {
        "disponible": True,
        "description": "Ajustement Beard (μx = (a+bc^x)/(1+dbc^x))",
        "variables": ["q_lisse", "ic_inf", "ic_sup"],
    },
    "lissage_noyau": {
        "disponible": True,
        "description": "Lissage par noyau gaussien",
        "variables": ["q_lisse", "ic_inf", "ic_sup"],
    },
    "validation_smr": {
        "disponible": True,
        "description": "SMR global et IC 95% (formule de Liddell)",
        "variables": ["smr_global", "smr_ic_inf", "smr_ic_sup"],
    },
    "chi2": {
        "disponible": True,
        "description": "Test chi2 d'adéquation à la table de référence",
        "variables": ["chi2_stat", "chi2_ddl", "chi2_pvalue"],
    },
    "abattement": {
        "disponible": True,
        "description": "Abattement αx = q_lisse / q_ref par âge",
        "variables": ["alpha"],
    },
    "deciles_exposition": {
        "disponible": True,
        "description": "SMR par déciles d'exposition cumulée (~10% chacun)",
        "variables": ["deciles"],
    },
    # ── Sections hors périmètre ──
    "chain_ladder": {
        "disponible": False,
        "raison": "Module non-vie non implémenté. Périmètre actuel : mortalité vie.",
    },
    "bornhuetter_ferguson": {
        "disponible": False,
        "raison": "Module non-vie non implémenté. Périmètre actuel : mortalité vie.",
    },
    "tarification_auto": {
        "disponible": False,
        "raison": "Hors périmètre actuariel vie. Le builder ne traite pas la tarification dommages.",
    },
    "tarification_iard": {
        "disponible": False,
        "raison": "Hors périmètre actuariel vie.",
    },
    "ibnr": {
        "disponible": False,
        "raison": "Module IBNR non implémenté.",
    },
}

# Map section_id du Brief → capability_id du builder
# La méthode de lissage peut ajuster le mapping dynamiquement.
SECTION_TO_CAPABILITY: dict[str, str] = {
    "mortalite_brute":  "mortalite_brute",
    "lissage":          "lissage_wh",      # défaut ; ajusté selon section.methode
    "validation_smr":   "validation_smr",
    "chi2":             "chi2",
    "abattement":       "abattement",
    "deciles":          "deciles_exposition",
    # Sections non-vie — seront non satisfaisables
    "chain_ladder":     "chain_ladder",
    "bornhuetter_ferguson": "bornhuetter_ferguson",
    "tarification_auto": "tarification_auto",
    "tarification_iard": "tarification_iard",
    "ibnr":             "ibnr",
}

# Map section_id → bloc payload requis (pour check_payload_against_brief)
SECTION_TO_PAYLOAD_BLOC: dict[str, str] = {
    "mortalite_brute":  "donnees_brutes",
    "lissage":          "lissage",
    "validation_smr":   "validation",
    "chi2":             "validation",
    "abattement":       "abattement",
    "deciles":          "deciles",
}

_METHODE_TO_CAPABILITY: dict[str, str] = {
    "whittaker_henderson": "lissage_wh",
    "gompertz_makeham":    "lissage_gompertz",
    "beard":               "lissage_beard",
    "noyau":               "lissage_noyau",
    "spline_cubique":      "lissage_wh",   # traité comme WH (disponible)
    "karup_king":          "lissage_wh",
}


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class BriefSection:
    id: str
    label: str
    priorite: str = "obligatoire"          # "obligatoire" | "optionnel"
    variables_requises: list = field(default_factory=list)
    methode: str | None = None             # pour la section "lissage"
    parametres: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "BriefSection":
        return cls(
            id=d["id"],
            label=d.get("label", d["id"]),
            priorite=d.get("priorite", "obligatoire"),
            variables_requises=d.get("variables_requises", []),
            methode=d.get("methode"),
            parametres=d.get("parametres", {}),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "priorite": self.priorite,
            "variables_requises": self.variables_requises,
            "methode": self.methode,
            "parametres": self.parametres,
        }


@dataclass
class Brief:
    version: str = "1.0"
    etude: dict = field(default_factory=dict)      # titre, description, contexte
    perimetre: dict = field(default_factory=dict)   # type_contrat, ages, segmentation, etc.
    sections: list = field(default_factory=list)    # list[BriefSection]
    format_sortie: str = "pdf_certification"
    source_pdf_reference: str | None = None         # base64 ou None

    @classmethod
    def from_dict(cls, d: dict) -> "Brief":
        sections = [
            BriefSection.from_dict(s) if isinstance(s, dict) else s
            for s in d.get("sections", [])
        ]
        return cls(
            version=d.get("version", "1.0"),
            etude=d.get("etude", {}),
            perimetre=d.get("perimetre", {}),
            sections=sections,
            format_sortie=d.get("format_sortie", "pdf_certification"),
            source_pdf_reference=d.get("source_pdf_reference"),
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "etude": self.etude,
            "perimetre": self.perimetre,
            "sections": [
                s.to_dict() if isinstance(s, BriefSection) else s
                for s in self.sections
            ],
            "format_sortie": self.format_sortie,
            "source_pdf_reference": self.source_pdf_reference,
        }


# ─── Fonctions de vérification ────────────────────────────────────────────────

def check_brief_satisfiability(brief: Brief) -> list[str]:
    """Retourne la liste des section_id NON réalisables par le builder actuel."""
    unsatisfiable = []
    for section in brief.sections:
        cap_id = SECTION_TO_CAPABILITY.get(section.id)
        if cap_id is None:
            # Section inconnue → non satisfaisable
            unsatisfiable.append(section.id)
            continue
        # Ajustement dynamique selon la méthode de lissage
        if section.id == "lissage" and section.methode:
            cap_id = _METHODE_TO_CAPABILITY.get(section.methode, "lissage_wh")
        cap = BUILDER_CAPABILITIES.get(cap_id, {})
        if not cap.get("disponible", False):
            unsatisfiable.append(section.id)
    return unsatisfiable


def check_payload_against_brief(brief: Brief, payload: dict) -> list[str]:
    """
    Retourne les section_id obligatoires du brief dont le bloc payload est absent.
    """
    missing = []
    payload_blocs = set(payload.keys())
    for section in brief.sections:
        if section.priorite != "obligatoire":
            continue
        expected_bloc = SECTION_TO_PAYLOAD_BLOC.get(section.id)
        if expected_bloc and expected_bloc not in payload_blocs:
            missing.append(section.id)
    return missing


def unsatisfiable_message(cannot_do: list[str]) -> str:
    """Formate un message lisible pour l'utilisateur."""
    if not cannot_do:
        return ""
    lines = [
        "⚠ Les sections suivantes ne peuvent pas être calculées par le builder :",
    ]
    for sid in cannot_do:
        cap_id = SECTION_TO_CAPABILITY.get(sid, sid)
        cap = BUILDER_CAPABILITIES.get(cap_id, {})
        raison = cap.get("raison", "Capacité non disponible")
        lines.append(f"  • {sid} — {raison}")
    lines.append(
        "Souhaitez-vous retirer ces sections du brief ou continuer sans elles ?"
    )
    return "\n".join(lines)
