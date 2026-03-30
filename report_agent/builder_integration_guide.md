# Guide d'intégration — Report Agent

Ce guide explique comment appeler le report agent depuis n'importe quel pipeline de calcul actuariel.

---

## Architecture

```
Votre pipeline de calcul
        │
        ▼  (sérialiser avec build_payload)
  dict JSON-compatible  ←── report_data_contract.yaml
        │
        ▼  (valider avec validate)
  validate_payload.validate(payload)
        │
        ▼  (générer avec generate)
  generate_report.generate(payload)  →  rapport.pdf
```

---

## Intégration minimale

```python
from report_agent.generate_report import build_payload, generate

# 1. Construire le payload après vos calculs
payload = build_payload(
    steps=[],                    # liste de steps si vous en avez
    summary="Analyse terminée.", # synthèse finale
    user_message="Analyser la mortalité du portefeuille retraite",
    domain_label="mortality",
    study_ref="Portefeuille 2024-Q1",
)

# 2. Générer le PDF
pdf_path = generate(payload, output_path="/tmp/rapport.pdf")
print(f"Rapport généré : {pdf_path}")
```

---

## Avec figures pré-générées

```python
import matplotlib.pyplot as plt
import io

# Générer une figure matplotlib en mémoire
fig, ax = plt.subplots()
ax.plot(ages, q_lisse, label="Taux lissés")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
plt.close(fig)
exposure_png = buf.getvalue()

payload = build_payload(
    ...,
    prebuilt_figures={
        "exposure": exposure_png,   # bytes PNG
        "rates": rates_png,
        "smr": smr_png,
    },
)
```

Les clés standard pour `prebuilt_figures` sont :
| Clé          | Graphique                                  |
|--------------|--------------------------------------------|
| `exposure`   | Exposition et décès par âge                |
| `rates`      | Taux bruts vs lissés avec IC 95%           |
| `smr`        | SMR par tranche d'âge                      |
| `oa`         | Observés vs attendus par tranche           |
| `comparison` | Expérience vs table de référence           |

---

## Avec template encoder (writer prompt personnalisé)

```python
import json

# Charger un template produit par encoder_app.py
with open("template.json") as f:
    tpl = json.load(f)

payload = build_payload(
    ...,
    writer_prompt=tpl.get("agent_system_prompt"),
    template_sections=tpl.get("sections", []),
    methodology=tpl.get("methodology"),
    pdf_reference_path=tpl.get("source_pdf"),
)
```

---

## Avec steps du computation agent

Les steps sont des dicts produits par `run_agent_loop()`. Ils peuvent contenir des DataFrames via `display_outputs` — le report agent tentera de les parser pour en extraire les données :

```python
payload = build_payload(
    steps=agent_steps,     # list[dict] de run_agent_loop()
    summary=agent_summary, # str
    ...,
)
```

`build_payload()` encode automatiquement les figures bytes → base64.

---

## Validation seule

Pour tester un payload sans générer le PDF :

```python
from report_agent.validate_payload import validate

try:
    validate(payload)
    print("Payload valide")
except ValueError as e:
    print(f"Erreur : {e}")
```

---

## Contrat de données complet

Voir [report_data_contract.yaml](report_data_contract.yaml) pour la description exhaustive de tous les champs.
