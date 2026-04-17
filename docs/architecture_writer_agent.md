# Architecture WriterAgent — Génération de rapport PDF

```mermaid
flowchart TD
    A([Demande utilisateur\n'génère le rapport']) --> B

    subgraph DET1 ["① Déterministe — 01_load_plan.py"]
        B["load_yaml_template\n— Charge le YAML\n— Résout les placeholders\n— Produit un plan structuré\n  avec 1 prompt par section\n— Indique ready/missing par section"]
    end

    B --> C

    subgraph LLM1 ["② LLM — 02_validation_plan.py"]
        C["Pour chaque section du plan :\n— Vérifie que les données requises\n  sont présentes ET suffisantes\n  (pas juste leur existence)\n— Vérifie que les outils nécessaires\n  sont disponibles\n— Produit : liste OK / KO par section"]
    end

    C -->|"Sections KO\n(données manquantes\nou insuffisantes)"| D["MasterAgent\n→ BuilderAgent\n→ calcule les données manquantes\n→ retour en ①"]
    D --> B
    C -->|Toutes sections OK| E

    subgraph LLM2 ["③ LLM + RAG — 03_completion_plan.py"]
        E["Pour chaque section du plan :\n— Appelle search_exemplars\n  avec le prompt de la section\n— Si exemples trouvés → enrichit\n  le prompt avec les extraits RAG\n— Si corpus vide → pas grave,\n  on continue sans exemples\n— Produit : plan enrichi avec\n  exemples de rédaction par section"]
    end

    E --> F

    subgraph DET2 ["④ Déterministe + LLM — 04_redaction.py\n(boucle Python sur les sections)"]
        F["Pour chaque section du plan enrichi :\n— Appelle les tools tableaux/graphiques\n— Appelle GPT-4o avec :\n  • prompt section\n  • données data_store\n  • exemples RAG (si disponibles)\n  • résultats tableaux/graphiques\n— Stocke le résultat dans section_outputs"]
    end

    F --> G

    subgraph DET3 ["⑤ Déterministe — 05_assemble.py"]
        G["assemble_sections\n— Mise en page ReportLab\n— En-têtes, numérotation, TOC\n— Intégration tableaux + graphiques\n— Export PDF"]
    end

    G --> H

    subgraph LLM3 ["⑥ LLM — 06_validation.py"]
        H["Compare :\n• Demande initiale\n• Rapport rédigé\n• Standard professionnel\n\nCatégorise chaque anomalie :\n⚠ Mineure : style, longueur, formulation\n🔴 Majeure : section manquante,\n   chiffres incohérents,\n   demande non couverte"]
    end

    H -->|Aucune alerte| I([Rapport PDF livré\nà l'utilisateur ✓])

    H -->|"⚠ Alerte mineure\n(max 1 retry)"| J["Relance 04_redaction\nuniquement pour\nla section concernée\navec feedback du validateur\najouté au prompt"]
    J --> G

    H -->|"🔴 Alerte majeure"| K([Rapport PDF livré\navec flag explicite\n+ liste des points\nnon couverts\npour décision utilisateur])

    style DET1 fill:#E8F4E8,stroke:#4CAF50
    style DET2 fill:#E8F4E8,stroke:#4CAF50
    style DET3 fill:#E8F4E8,stroke:#4CAF50
    style LLM1 fill:#E3F2FD,stroke:#2196F3
    style LLM2 fill:#E3F2FD,stroke:#2196F3
    style LLM3 fill:#E3F2FD,stroke:#2196F3
```

## Légende

| Couleur | Rôle |
|---------|------|
| 🟢 Vert — Déterministe | Logique Python pure, résultat prévisible, zéro token LLM |
| 🔵 Bleu — LLM | GPT-4o intervient pour juger, enrichir ou rédiger |

## Les 6 étapes

| Fichier | Type | Rôle |
|---------|------|------|
| `01_load_plan.py` | Déterministe | Charge le YAML, résout les placeholders, produit le plan |
| `02_validation_plan.py` | LLM | Vérifie que données et outils sont présents ET suffisants |
| `03_completion_plan.py` | LLM + RAG | Enrichit chaque prompt de section avec des exemples de rédaction |
| `04_redaction.py` | Boucle Python + LLM | Rédige chaque section avec tables, graphiques et GPT-4o |
| `05_assemble.py` | Déterministe | Assemble le PDF final (ReportLab) |
| `06_validation.py` | LLM | Vérifie cohérence globale et qualité professionnelle |

## Gestion des alertes en ⑥

**Alerte mineure** (style, longueur, formulation vague)
→ 1 retry ciblé sur la section concernée uniquement, avec le feedback du validateur ajouté au prompt de rédaction. Maximum 1 retry — jamais de boucle infinie.

**Alerte majeure** (section manquante, chiffres incohérents, demande non couverte)
→ Le rapport est livré tel quel avec un flag explicite et la liste des points non couverts. Pas de retry automatique — une alerte majeure signale soit un problème de données (qui aurait dû être détecté en ②), soit une limite structurelle. L'utilisateur décide de la suite.

## Principes clés

**Le RAG est préparé en amont (③), pas pendant la rédaction (④)**
L'agent de rédaction reçoit déjà les exemples dans son prompt — il n'a pas besoin de décider d'aller chercher ou non. Cela simplifie la boucle ④ et évite des appels RAG aléatoires.

**La validation des données (②) est séparée de la rédaction (④)**
Si des données manquent, on le sait avant de commencer à rédiger. La boucle ④ peut tourner sans interruption.

**Un seul point de retour vers le MasterAgent**
Uniquement en ② si des données sont insuffisantes. La boucle ④ ne remonte jamais vers le Master.
```
