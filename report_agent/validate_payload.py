"""
Validation du payload JSON contre le data contract.
Retourne un ValidationResult avec erreurs/warnings.
"""
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def refusal_message(self) -> str:
        if self.valid:
            return ""
        lines = [
            "RAPPORT NON GÉNÉRÉ — Données manquantes ou incohérentes",
            "", "Erreurs bloquantes :",
        ]
        for e in self.errors:
            lines.append(f"  ✗ {e}")
        if self.warnings:
            lines += ["", "Avertissements (non bloquants) :"]
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        lines += ["", "Action requise : l'agent builder doit corriger ces points et relancer."]
        return "\n".join(lines)


REQUIRED_BLOCS = {
    "portfolio": {
        "required_fields": [
            "n_assures", "n_contrats_actifs", "type_contrat",
            "periode_debut", "periode_fin", "age_min", "age_max",
            "segmentation", "table_reference",
        ],
        "description": "Métadonnées du portefeuille",
    },
    "donnees_brutes": {
        "required_fields": ["ages", "exposure", "deaths_observed", "q_brut"],
        "array_fields": ["ages", "exposure", "deaths_observed", "q_brut"],
        "description": "Vecteurs par âge : ages, exposure, deaths_observed, q_brut",
    },
    "lissage": {
        "required_fields": ["methode", "parametres", "q_lisse", "ic_inf", "ic_sup", "q_ref"],
        "array_fields": ["q_lisse", "ic_inf", "ic_sup", "q_ref"],
        "allowed_methodes": [
            "whittaker_henderson", "gompertz_makeham", "beard",
            "spline_cubique", "karup_king", "noyau",
        ],
        "description": "Méthode de lissage, paramètres, taux lissés, IC, taux référence",
    },
    "validation": {
        "required_fields": [
            "smr_global", "smr_ic_inf", "smr_ic_sup",
            "chi2_stat", "chi2_ddl", "chi2_pvalue", "abattement_global",
        ],
        "description": "SMR global + IC, χ², abattement global",
    },
    "deciles": {
        "is_array": True, "min_items": 3,
        "item_required_fields": [
            "tranche_label", "age_start", "age_end", "exposure",
            "deaths_observed", "deaths_expected", "smr", "smr_ic_inf", "smr_ic_sup",
        ],
        "description": "SMR par décile d'exposition (~10 items, quantiles d'exposition PAS tranches fixes)",
    },
    "abattement": {
        "required_fields": ["ages", "q_construit", "q_reference", "alpha"],
        "array_fields": ["ages", "q_construit", "q_reference", "alpha"],
        "description": "Abattement par âge",
    },
    "qualite_donnees": {
        "required_fields": ["traitements_appliques", "stats_annuelles"],
        "description": "Traitements appliqués et stats annuelles",
    },
}


def validate(payload: dict) -> ValidationResult:
    result = ValidationResult(valid=True)

    for bloc_name, schema in REQUIRED_BLOCS.items():
        if bloc_name not in payload:
            result.valid = False
            result.errors.append(f"Bloc '{bloc_name}' manquant — {schema['description']}")
            continue

        bloc = payload[bloc_name]

        if schema.get("is_array"):
            if not isinstance(bloc, list):
                result.valid = False
                result.errors.append(f"'{bloc_name}' doit être un array")
                continue
            if len(bloc) < schema.get("min_items", 1):
                result.valid = False
                result.errors.append(
                    f"'{bloc_name}' : {len(bloc)} items, minimum {schema.get('min_items')}"
                )
                continue
            for i, item in enumerate(bloc):
                for f in schema.get("item_required_fields", []):
                    if f not in item:
                        result.valid = False
                        result.errors.append(f"'{bloc_name}[{i}].{f}' manquant")
            continue

        if not isinstance(bloc, dict):
            result.valid = False
            result.errors.append(f"'{bloc_name}' doit être un objet")
            continue

        for f in schema.get("required_fields", []):
            if f not in bloc:
                result.valid = False
                result.errors.append(f"'{bloc_name}.{f}' manquant")

        if "allowed_methodes" in schema and "methode" in bloc:
            if bloc["methode"] not in schema["allowed_methodes"]:
                result.valid = False
                result.errors.append(
                    f"'{bloc_name}.methode' = '{bloc['methode']}' non reconnu. "
                    f"Autorisés : {schema['allowed_methodes']}"
                )

    # Cohérence des longueurs de donnees_brutes
    if "donnees_brutes" in payload and isinstance(payload["donnees_brutes"], dict):
        db = payload["donnees_brutes"]
        db_schema = REQUIRED_BLOCS["donnees_brutes"]
        lengths = {
            f: len(db[f])
            for f in db_schema.get("array_fields", [])
            if f in db and isinstance(db[f], (list, tuple))
        }
        if lengths and len(set(lengths.values())) > 1:
            result.valid = False
            result.errors.append(f"donnees_brutes : longueurs incohérentes — {lengths}")

        n_ages = lengths.get("ages")
        if n_ages:
            for bloc_key in ["lissage", "abattement"]:
                if bloc_key in payload and isinstance(payload[bloc_key], dict):
                    for f in REQUIRED_BLOCS.get(bloc_key, {}).get("array_fields", []):
                        if f in payload[bloc_key] and isinstance(
                            payload[bloc_key][f], (list, tuple)
                        ):
                            if len(payload[bloc_key][f]) != n_ages:
                                result.valid = False
                                result.errors.append(
                                    f"{bloc_key}.{f} : {len(payload[bloc_key][f])} éléments,"
                                    f" attendu {n_ages}"
                                )

    # Avertissement bloc optionnel
    if "trace" not in payload:
        result.warnings.append("Bloc 'trace' absent — section annexe réduite")

    return result
