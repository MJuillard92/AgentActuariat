"""
report_agent/test_writer_agent.py
Tests unitaires du WriterAgent et du module brief.
Aucun appel API réel — tout est testé en logique pure.

Lance avec :
    python -m report_agent.test_writer_agent
"""
from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from report_agent.brief import (
    Brief, BriefSection,
    check_brief_satisfiability,
    check_payload_against_brief,
    unsatisfiable_message,
    BUILDER_CAPABILITIES,
)
from report_agent.writer_agent import WriterAgent


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_brief(sections=None) -> Brief:
    sections = sections or [
        BriefSection(id="mortalite_brute", label="Taux bruts", priorite="obligatoire"),
        BriefSection(id="lissage", label="Lissage WH", priorite="obligatoire",
                     methode="whittaker_henderson", parametres={"lambda": 500, "ordre": 2}),
        BriefSection(id="validation_smr", label="SMR", priorite="obligatoire"),
        BriefSection(id="chi2", label="chi2", priorite="obligatoire"),
        BriefSection(id="abattement", label="Abattement", priorite="optionnel"),
    ]
    return Brief(
        etude={"titre": "Test", "description": "Étude de test", "contexte": ""},
        perimetre={
            "type_contrat": "vie_entiere", "age_min": 25, "age_max": 125,
            "segmentation": "global", "periode_debut": "2015-01-01",
            "periode_fin": "2023-12-31", "table_reference": "TH00-02",
        },
        sections=sections,
    )


# ─── Tests brief.py ──────────────────────────────────────────────────────────

def test_brief_roundtrip():
    """Brief.from_dict(brief.to_dict()) doit reproduire le même objet."""
    original = _make_brief()
    restored = Brief.from_dict(original.to_dict())
    assert restored.version == original.version
    assert restored.perimetre == original.perimetre
    assert len(restored.sections) == len(original.sections)
    assert restored.sections[0].id == original.sections[0].id
    assert restored.sections[1].parametres == {"lambda": 500, "ordre": 2}
    print("  ✓ Brief roundtrip OK")


def test_check_satisfiability_all_ok():
    """Toutes les sections standards doivent être réalisables."""
    brief = _make_brief()
    cannot_do = check_brief_satisfiability(brief)
    assert cannot_do == [], f"Sections non réalisables inattendues : {cannot_do}"
    print("  ✓ Satisfiabilité — toutes sections OK")


def test_check_satisfiability_chain_ladder():
    """chain_ladder doit être non réalisable."""
    brief = _make_brief(sections=[
        BriefSection(id="mortalite_brute", label="Taux bruts"),
        BriefSection(id="chain_ladder", label="Chain-Ladder", priorite="obligatoire"),
    ])
    cannot_do = check_brief_satisfiability(brief)
    assert "chain_ladder" in cannot_do, f"chain_ladder devrait être non réalisable, got {cannot_do}"
    print("  ✓ Satisfiabilité — chain_ladder détecté non réalisable")


def test_check_satisfiability_tarification_auto():
    """tarification_auto doit être non réalisable."""
    brief = _make_brief(sections=[
        BriefSection(id="tarification_auto", label="Tarification auto"),
    ])
    cannot_do = check_brief_satisfiability(brief)
    assert "tarification_auto" in cannot_do
    print("  ✓ Satisfiabilité — tarification_auto détectée non réalisable")


def test_check_satisfiability_unknown_section():
    """Section inconnue doit être non réalisable."""
    brief = _make_brief(sections=[
        BriefSection(id="section_inconnue_xyz", label="?"),
    ])
    cannot_do = check_brief_satisfiability(brief)
    assert "section_inconnue_xyz" in cannot_do
    print("  ✓ Satisfiabilité — section inconnue détectée")


def test_check_payload_against_brief_complete():
    """Payload complet — aucune section manquante."""
    brief = _make_brief()
    payload = {
        "donnees_brutes": {}, "lissage": {}, "validation": {},
        "abattement": {}, "deciles": [], "portfolio": {}, "qualite_donnees": {},
    }
    missing = check_payload_against_brief(brief, payload)
    assert missing == [], f"Sections manquantes inattendues : {missing}"
    print("  ✓ Payload complet — aucune section manquante")


def test_check_payload_against_brief_missing_lissage():
    """Payload sans bloc 'lissage' → section lissage manquante."""
    brief = _make_brief()
    payload = {"donnees_brutes": {}, "validation": {}, "abattement": {}}
    missing = check_payload_against_brief(brief, payload)
    assert "lissage" in missing, f"'lissage' devrait être manquant, got {missing}"
    print("  ✓ Payload incomplet — lissage manquant détecté")


def test_unsatisfiable_message():
    """unsatisfiable_message doit mentionner les sections et la raison."""
    msg = unsatisfiable_message(["chain_ladder", "tarification_auto"])
    assert "chain_ladder" in msg
    assert "tarification_auto" in msg
    assert "non-vie" in msg.lower() or "périmètre" in msg.lower()
    print("  ✓ Message non-réalisable formaté correctement")


# ─── Tests writer_agent.py ───────────────────────────────────────────────────

def test_extract_brief_from_response_valid():
    """Réponse avec bloc ```json valide contenant 'sections' → brief extrait."""
    response = """
Voici le brief de l'étude :

```json
{
  "version": "1.0",
  "etude": {"titre": "Test", "description": "desc", "contexte": ""},
  "perimetre": {"type_contrat": "vie_entiere", "age_min": 25, "age_max": 125,
                "segmentation": "global", "periode_debut": "2015-01-01",
                "periode_fin": "2023-12-31", "table_reference": "TH00-02"},
  "sections": [
    {"id": "mortalite_brute", "label": "Taux bruts", "priorite": "obligatoire",
     "variables_requises": ["ages", "exposure"]}
  ],
  "format_sortie": "pdf_certification",
  "source_pdf_reference": null
}
```

L'analyse portera sur la période 2015–2023.
"""
    result = WriterAgent._extract_brief_from_response(response)
    assert result is not None
    assert "sections" in result
    assert result["sections"][0]["id"] == "mortalite_brute"
    print("  ✓ Extraction brief depuis réponse valide")


def test_extract_brief_from_response_no_json():
    """Réponse sans JSON → None."""
    response = "Quel type de contrat avez-vous ?"
    result = WriterAgent._extract_brief_from_response(response)
    assert result is None
    print("  ✓ Extraction brief — pas de JSON → None")


def test_extract_brief_from_response_json_without_sections():
    """JSON sans clé 'sections' → None (pas un brief)."""
    response = '```json\n{"quelque_chose": "autre"}\n```'
    result = WriterAgent._extract_brief_from_response(response)
    assert result is None
    print("  ✓ Extraction brief — JSON sans 'sections' → None")


def test_extract_brief_from_response_invalid_json():
    """JSON malformé → None."""
    response = '```json\n{invalid json here\n```'
    result = WriterAgent._extract_brief_from_response(response)
    assert result is None
    print("  ✓ Extraction brief — JSON malformé → None")


def test_build_builder_prompt_contains_lambda():
    """Le prompt builder doit contenir le λ configuré dans le brief."""
    brief = _make_brief()  # λ=500 dans le fixture
    prompt = WriterAgent(model="gpt-4o").build_builder_prompt(brief)
    assert "500" in prompt, f"λ=500 devrait apparaître dans le prompt, got:\n{prompt[:500]}"
    assert "TH00-02" in prompt
    assert "whittaker_henderson" in prompt.lower() or "Whittaker" in prompt
    print("  ✓ Prompt builder contient λ et table de référence")


def test_build_builder_prompt_chi2():
    """Le prompt builder doit mentionner chi2 si la section est présente."""
    brief = _make_brief()
    prompt = WriterAgent(model="gpt-4o").build_builder_prompt(brief)
    assert "chi2" in prompt.lower() or "chi" in prompt.lower()
    print("  ✓ Prompt builder contient chi2")


def test_build_builder_prompt_chain_ladder_excluded():
    """chain_ladder ne doit pas générer d'instruction dans le prompt."""
    brief = _make_brief(sections=[
        BriefSection(id="mortalite_brute", label="Taux bruts"),
        BriefSection(id="chain_ladder", label="Chain-Ladder"),
    ])
    prompt = WriterAgent(model="gpt-4o").build_builder_prompt(brief)
    # chain_ladder n'a pas de handler dans build_builder_prompt → pas d'instruction générée
    assert "chain_ladder" not in prompt.lower()
    print("  ✓ Prompt builder — chain_ladder absent (non géré)")


def test_verify_capabilities():
    """verify_capabilities sépare correctement can_do et cannot_do."""
    brief = _make_brief(sections=[
        BriefSection(id="mortalite_brute", label="Taux bruts"),
        BriefSection(id="validation_smr", label="SMR"),
        BriefSection(id="chain_ladder", label="Chain-Ladder"),
        BriefSection(id="tarification_auto", label="Tarification"),
    ])
    writer = WriterAgent(model="gpt-4o")
    can_do, cannot_do = writer.verify_capabilities(brief)
    assert "mortalite_brute" in can_do
    assert "validation_smr" in can_do
    assert "chain_ladder" in cannot_do
    assert "tarification_auto" in cannot_do
    print(f"  ✓ verify_capabilities : {len(can_do)} réalisables, {len(cannot_do)} non réalisables")


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  WriterAgent — Tests unitaires")
    print("=" * 60 + "\n")

    # brief.py
    test_brief_roundtrip()
    test_check_satisfiability_all_ok()
    test_check_satisfiability_chain_ladder()
    test_check_satisfiability_tarification_auto()
    test_check_satisfiability_unknown_section()
    test_check_payload_against_brief_complete()
    test_check_payload_against_brief_missing_lissage()
    test_unsatisfiable_message()

    # writer_agent.py
    test_extract_brief_from_response_valid()
    test_extract_brief_from_response_no_json()
    test_extract_brief_from_response_json_without_sections()
    test_extract_brief_from_response_invalid_json()
    test_build_builder_prompt_contains_lambda()
    test_build_builder_prompt_chi2()
    test_build_builder_prompt_chain_ladder_excluded()
    test_verify_capabilities()

    print("\n" + "=" * 60)
    print("  TOUS LES TESTS PASSENT")
    print("=" * 60 + "\n")
