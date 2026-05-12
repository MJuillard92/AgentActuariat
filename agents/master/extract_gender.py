"""
agents/master/extract_gender.py
Heuristique pure (sans LLM) pour détecter le mode de segmentation par sexe
dans une demande utilisateur en français.

Utilisé par le Master pour court-circuiter la question
"unisex ou by_sex ?" quand l'utilisateur a déjà exprimé son choix dans
sa formulation.
"""
from __future__ import annotations


_BY_SEX_KEYS: tuple[str, ...] = (
    "by_sex", "by sex", "h/f", " h /f", "h-f",
    "par sexe", "par genre", "par sex",
    "tables séparées", "tables separees",
    "séparé par sexe", "separe par sexe",
    "homme et femme", "hommes et femmes", "femmes et hommes",
    "hommes-femmes", "homme/femme", "differencié par sexe",
)

_UNISEX_KEYS: tuple[str, ...] = (
    "unisex", "uni-sex", "uni sex",
    "agrégé", "agrege", "table agrégée", "table agregee",
    "sans distinction de sexe", "sans distinction par sexe",
    "tous sexes confondus", "tous sexes",
)


def extract_gender_from_text(text: str) -> str | None:
    """Retourne 'unisex' / 'by_sex' / None selon les mots-clés présents.

    Couvre les expressions FR explicites :
      - by_sex : "H/F", "par sexe", "tables séparées", "hommes et femmes", ...
      - unisex : "unisex", "agrégé", "sans distinction de sexe", ...

    Si aucun mot-clé n'est trouvé, retourne None (la question sera posée
    via le pattern need_user_input côté master_node).
    """
    txt = (text or "").lower()
    if any(k in txt for k in _BY_SEX_KEYS):
        return "by_sex"
    if any(k in txt for k in _UNISEX_KEYS):
        return "unisex"
    return None
