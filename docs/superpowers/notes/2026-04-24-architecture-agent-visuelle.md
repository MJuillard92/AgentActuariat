# Fonctionnement de l'agent — diagrammes

Vue synthétique de la cinématique après le refactor `report_mode` + `write` + anti-boucle.

---

## 1. Vue d'ensemble (haut niveau)

```
       ┌─────────────────────────────────────────────────────┐
       │                  UTILISATEUR (UI)                   │
       │                                                     │
       │  Upload CSV ──► mapping colonnes ──► chat          │
       └────────────────────────┬────────────────────────────┘
                                │
                                │ HumanMessage
                                ▼
       ┌─────────────────────────────────────────────────────┐
       │                  MASTER  (LangGraph node)           │
       │                                                     │
       │  1. WRITE_DONE ?    → done                          │
       │  2. BUILD_DONE + clés OK ?                          │
       │       └─ write=yes ─► Writer                        │
       │       └─ write=no  ─► done                          │
       │  3. NEED_DATA ?     → réinjecter au Builder         │
       │  4. Désambiguation colonnes/valeurs                 │
       │  5. classify_intent (LLM)                           │
       │       → (kind, write, report_mode)                  │
       │  6. Routing :                                       │
       │       kind=question     ─► LLM conversationnel      │
       │       write=ask (1x)    ─► pose la question PDF    │
       │       write=yes|no      ─► Builder                  │
       │       missing_keys=[]   ─► Writer direct (write=yes)│
       │  7. Compteur cycles > 3 ─► done (anti-boucle)       │
       └──────┬────────────────────────┬────────────────┬────┘
              │                        │                │
              │ instruction            │                │
              │ (sections + keys)      │                │
              ▼                        ▼                ▼
       ┌──────────────┐       ┌───────────────┐   ┌──────────┐
       │   BUILDER    │       │    WRITER     │   │   end    │
       │  (LangGraph) │       │ (3-étapes)    │   └──────────┘
       │              │       │               │
       │  ┌──────────┐│       │  load_plan    │
       │  │ Tool LLM ││       │  complete     │
       │  │   loop   ││──────►│  redaction    │
       │  └──────────┘│       │  → PDF        │
       │              │       │               │
       │ <BUILD_DONE> │       │ <WRITE_DONE>  │
       └──────┬───────┘       └───────┬───────┘
              │                       │
              └───────┐               │
                      ▼               ▼
                  Retour Master  Retour Master
```

---

## 2. Diagramme d'état (cycle complet)

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │ user message
                         ▼
              ┌──────────────────────┐
              │       MASTER         │
              │                      │
              │   classify_intent    │
              │   → kind, write,     │
              │     report_mode      │
              └──────────┬───────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        │ kind=question  │ kind=task      │ BUILD_DONE
        │                │                │ + write=yes
        ▼                ▼                ▼
   ┌─────────┐    ┌──────────────┐    ┌──────────┐
   │   LLM   │    │   write=?    │    │  WRITER  │
   │ conver- │    └──────┬───────┘    └─────┬────┘
   │ sation  │           │                  │
   └────┬────┘     ┌─────┼──────┐          │ WRITE_DONE
        │          │     │      │          ▼
        ▼          ▼     ▼      ▼      ┌───────┐
     ┌─────┐    ask   yes    no       │ done  │
     │ end │    │     │      │        └───────┘
     └─────┘    │     │      │
                │     │      │
                ▼     ▼      │
        ┌──────────┐  │      │
        │  Master  │  │      │
        │   pose   │  │      │
        │ question │  │      │
        │   PDF    │  │      │
        └────┬─────┘  │      │
             │        │      │
             │ END    │      │
             │ (user répond) │
             │        │      │
             └───┐    ▼      ▼
                 ▼  ┌────────────┐
               (re- │  BUILDER   │
                classify) └─┬───┘
                 │          │
                 │ BUILD_DONE + keys OK
                 │          │
                 │          ▼
                 │     ┌────────┐
                 │     │ MASTER │
                 │     └───┬────┘
                 │         │
                 │         ├─ write=yes ─► Writer ─► done
                 │         │
                 │         └─ write=no  ─► done
                 │
                 │ cycles > 3
                 │
                 └──────► done (anti-boucle)
```

---

## 3. Détail Master — branches de routing

```
                MASTER RECEIVES STATE
                        │
                        ▼
              ┌─────────────────────┐
              │ Extract data_store  │
              │ Read messages list  │
              └──────────┬──────────┘
                         │
           ┌─────────────┼─────────────────┐
           │             │                 │
      [1] WRITE_DONE   [2] BUILD_DONE    [3] autre
           │             │                 │
           ▼             ▼                 │
      ┌─────────┐   ┌─────────────┐        │
      │ cleanup │   │ check keys  │        │
      │         │   │ for mode    │        │
      │ return  │   └──────┬──────┘        │
      │  done   │          │               │
      └─────────┘     all present ?        │
                          │                │
                   ┌──────┼──────┐         │
                  yes              no      │
                   │                │      │
                   ▼                │      │
              ┌─────────┐           │      │
              │ write=? │           │      │
              └────┬────┘           │      │
                   │                │      │
             ┌─────┼─────┐          │      │
            yes         no          │      │
             │           │          │      │
             ▼           ▼          │      │
        ┌────────┐ ┌────────┐       │      │
        │ Writer │ │ done   │       │      │
        └────────┘ └────────┘       │      │
                                    │      │
                                    ▼      ▼
                         [3] détect NEED_DATA ou
                         continuer vers classify_intent
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ [4] Désambiguation   │
                         │  colonnes + valeurs  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ [5] classify_intent  │
                         │                      │
                         │  kind:        ?      │
                         │  write:       ?      │
                         │  report_mode: ?      │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              kind=question      task           task
                    │            write=ask      write=yes
                    │            (1ère fois)    ou write=no
                    │               │               │
                    ▼               ▼               ▼
             ┌──────────┐    ┌──────────┐    ┌──────────┐
             │   LLM    │    │ Poser la │    │ Compute  │
             │ response │    │ question │    │ active   │
             │          │    │ au user  │    │ sections │
             │  done    │    │          │    │    +     │
             └──────────┘    │   END    │    │ missing_ │
                             │          │    │   keys   │
                             └──────────┘    └────┬─────┘
                                                   │
                                        ┌──────────┼─────────┐
                                        │          │         │
                                 missing=[ ]  missing≠[] cycles>3
                                        │          │         │
                                        ▼          ▼         ▼
                                   ┌────────┐ ┌────────┐ ┌────────┐
                                   │ Writer │ │ Builder│ │ done   │
                                   │(yes)   │ │        │ │        │
                                   │  ou    │ │ instru-│ │(alerte)│
                                   │ done   │ │ ction  │ │        │
                                   │(no)    │ │ dérivée│ │        │
                                   └────────┘ └────────┘ └────────┘
```

---

## 4. Détail Builder — boucle LLM + tools

```
                      BUILDER NODE CALLED
                              │
                              ▼
                ┌─────────────────────────────┐
                │ [DÉTERMINISTE] raw_rates ?  │
                │                             │
                │ if report_mode=="raw_rates" │
                │    and qx_table present     │
                │    and not smoothed_table : │
                │ → data_store["smoothed_tabl │
                │    e"] = copy(qx_table)     │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │ _build_system_prompt()      │
                │                             │
                │ base = behavioral_contract  │
                │       + step1_planning      │
                │       + column_mapping      │
                │       + study_plan          │
                │       + _capabilities_block │
                │       + règles report_mode  │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │ Appel OpenAI gpt-4o         │
                │ (system + messages + tools) │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │ [GARDE-FOU]                 │
                │ _has_pending_decision ?     │
                │                             │
                │ si oui ET tool_calls :      │
                │ → lc_msg.tool_calls = []    │
                │ → forcer content si vide    │
                └──────────────┬──────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │ _should_continue_builder    │
                └──────────────┬──────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
    tool_calls            BUILD_DONE               no tool_calls,
    non vide              dans content              pas de signal
        │                      │                      │
        ▼                      ▼                      ▼
   ┌──────────┐          ┌──────────┐           ┌──────────┐
   │ ToolNode │          │   to_    │           │  END     │
   │ exécute  │          │  master  │           │ (pause)  │
   │  tools   │          │          │           │          │
   └────┬─────┘          └────┬─────┘           └──────────┘
        │                     │
        │ ToolMessage          │
        │ ajouté               │
        ▼                     ▼
   (boucle retour       (retour au Master)
    au Builder LLM)
```

---

## 5. Les 3 axes — effet sur le pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     AXE 1 : kind                                │
│                                                                 │
│   task    ────► pipeline complet Master → Builder → Writer     │
│   question ───► LLM conversationnel, aucun agent                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    AXE 2 : write                                │
│                                                                 │
│   yes  ────► Builder puis Writer (route automatique)           │
│   no   ────► Builder seul, pas de PDF                          │
│   ask  ────► Master pose la question AVANT le Builder          │
│              (calculs seulement lancés après réponse user)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 AXE 3 : report_mode                             │
│                                                                 │
│  full_report  ►─ Bloc A + B + C + D                            │
│                  ├─ A. Stats descriptives (nettoyage + stats)   │
│                  ├─ B. Taux bruts (exposure, crude_rates,       │
│                  │     diagnostics, validation sur qx)          │
│                  ├─ C. Taux lissés (smoothing + validation      │
│                  │     sur smoothed_table)                      │
│                  └─ D. Benchmarking réglementaire               │
│                                                                 │
│  raw_rates   ►── Bloc A + B + C' + D                           │
│                  └─ C'. Assimilation déterministe               │
│                         (copy qx → smoothed, pas de smoothing) │
│                                                                 │
│  description ►── Bloc A uniquement                             │
│                  └─ preamble, preprocessing, data_analysis      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Gestion de la mémoire (3 couches)

```
┌──────────────────── À CHAQUE TOUR LLM ─────────────────────────┐
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ SYSTEM PROMPT (reconstruit, couche 2)                    │  │
│  │                                                          │  │
│  │  ├─ behavioral_contract.md                               │  │
│  │  ├─ step1_planning.md (règles tools canoniques)          │  │
│  │  ├─ column mapping (CSV → canoniques)                    │  │
│  │  ├─ study_plan confirmé                                  │  │
│  │  ├─ résumé MemoryManager                                 │  │
│  │  ├─ _capabilities_block (sections YAML → clés/tools)     │  │
│  │  └─ règle report_mode (raw_rates skip smoothing, etc.)   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 20 DERNIERS MESSAGES (couche 1 - fenêtre glissante)      │  │
│  │                                                          │  │
│  │  HumanMessage / AIMessage / ToolMessage                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│                    ⬇  ENVOI gpt-4o                             │
└────────────────────────────────────────────────────────────────┘

┌──────────────────── PERSISTÉ (couche 3) ───────────────────────┐
│                                                                │
│  state["messages"]             → historique complet            │
│  state["data_store"]           → clés produites par les tools  │
│     ├─ cleaned_records                                         │
│     ├─ total_exposure                                          │
│     ├─ segmentations                                           │
│     ├─ serie / serie_h / serie_f                               │
│     ├─ ages                                                    │
│     ├─ _kind, _write, report_mode       (axes classification) │
│     ├─ _write_question_asked            (flag désambig ask)    │
│     ├─ _master_builder_cycles           (compteur anti-boucle) │
│     └─ _builder_turns                   (tours LLM par appel)  │
│                                                                │
│  session/{id}/records.parquet  → DataFrame CSV normalisé       │
│  session/{id}/memory.json      → résumé + contexte persisté    │
└────────────────────────────────────────────────────────────────┘
```

---

## 7. Anti-boucle — 3 niveaux de protection

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  NIVEAU 1  ─  _builder_turns  (graph.py)                       │
│                                                                │
│    Chaque appel LLM Builder incrémente _builder_turns.         │
│    Si ≥ 5 → retour forcé au Master.                            │
│    Protège contre une boucle LLM → tool → LLM infinie dans     │
│    UNE MÊME invocation du Builder.                             │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  NIVEAU 2  ─  _master_builder_cycles  (master_node.py)         │
│                                                                │
│    Chaque fois que Master route vers Builder avec des clés     │
│    manquantes, compteur incrémenté.                            │
│    Si > 3 → Master émet 'done' avec message d'alerte.          │
│    Protège contre un cycle Master ↔ Builder qui ne converge    │
│    jamais.                                                     │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  NIVEAU 3  ─  _has_pending_decision  (builder_node.py)         │
│                                                                │
│    Si un tool retourne un marqueur `decision_required`,        │
│    le prochain tour Builder écrase tout tool_call émis par    │
│    le LLM (même si ce dernier désobéit à son prompt).          │
│    Force une pause pour question utilisateur.                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. Séquence type — "construis-moi une table de mortalité"

```
User                 Master                 Builder                 Writer
 │                     │                       │                       │
 │── "construis..." ──►│                       │                       │
 │                     │                       │                       │
 │                     │ classify_intent       │                       │
 │                     │ → kind=task           │                       │
 │                     │   write=ask           │                       │
 │                     │   report_mode=        │                       │
 │                     │     full_report       │                       │
 │                     │                       │                       │
 │                     │ désambiguation        │                       │
 │                     │ colonnes/valeurs      │                       │
 │                     │                       │                       │
 │                     │ write=ask + !flag     │                       │
 │                     │ → _write_question_    │                       │
 │                     │   asked=True          │                       │
 │◄── "Voulez-vous un rapport PDF ?" ─┤        │                       │
 │                     │                       │                       │
 │                   (PAUSE, state END)        │                       │
 │                                             │                       │
 │── "oui" ──────────►│                       │                       │
 │                     │                       │                       │
 │                     │ reclassify            │                       │
 │                     │ → write=yes           │                       │
 │                     │                       │                       │
 │                     │ active_sections=      │                       │
 │                     │   [preamble,          │                       │
 │                     │    preprocessing,     │                       │
 │                     │    data_analysis_*,   │                       │
 │                     │    table_construction,│                       │
 │                     │    smoothing,         │                       │
 │                     │    validation,        │                       │
 │                     │    benchmarking,      │                       │
 │                     │    conclusion]        │                       │
 │                     │                       │                       │
 │                     │ missing_keys=...      │                       │
 │                     │ cycles=1              │                       │
 │                     │                       │                       │
 │                     │ HumanMessage ─────────►│                       │
 │                     │ "Mode: full_report    │                       │
 │                     │  Sections actives:    │                       │
 │                     │  ...                  │                       │
 │                     │  Reste à produire:    │                       │
 │                     │  [cleaned_records,    │                       │
 │                     │   total_exposure,     │                       │
 │                     │   ...]"               │                       │
 │                     │                       │                       │
 │                     │                       │ tool_call exposure    │
 │                     │                       │ tool_call crude_rates │
 │                     │                       │ tool_call diagnostics │
 │                     │                       │ tool_call smoothing   │
 │                     │                       │ tool_call validation  │
 │                     │                       │ tool_call benchmarking│
 │                     │                       │                       │
 │                     │                       │ <BUILD_DONE>          │
 │                     │ ◄─────────────────────┤                       │
 │                     │                       │                       │
 │                     │ check keys OK         │                       │
 │                     │ write=yes             │                       │
 │                     │ cycles=None (reset)   │                       │
 │                     │                       │                       │
 │                     │ ─────────────────────────────────────────────►│
 │                     │                       │                       │
 │                     │                       │                       │ load_plan
 │                     │                       │                       │ completion
 │                     │                       │                       │ redaction
 │                     │                       │                       │ → PDF
 │                     │                       │                       │
 │                     │                       │                       │ <WRITE_DONE>
 │                     │◄──────────────────────────────────────────────┤
 │                     │                       │                       │
 │                     │ cleanup               │                       │
 │                     │ done                  │                       │
 │                                                                     │
 │◄── PDF affiché ────────────────────────────────────────────────────┤
```

---

## 9. Fichiers clés pour comprendre le code

| Fichier | Rôle |
|---|---|
| [agents/mortality/agents/graph.py](agents/mortality/agents/graph.py) | Construction du graphe LangGraph + fonctions `_should_continue_*` |
| [agents/mortality/agents/master_node.py](agents/mortality/agents/master_node.py) | Nœud Master (routing, classify, désambiguation) |
| [agents/mortality/agents/builder_node.py](agents/mortality/agents/builder_node.py) | Nœud Builder (LLM + tools + garde-fou + raw_rates) |
| [agents/report/pipeline/_01_load_plan.py](agents/report/pipeline/_01_load_plan.py) | Étape 1 Writer — lit YAML, résout placeholders, filtre sections |
| [agents/report/pipeline/_03_completion_plan.py](agents/report/pipeline/_03_completion_plan.py) | Étape 2 Writer — enrichit avec RAG |
| [agents/report/pipeline/_04_redaction.py](agents/report/pipeline/_04_redaction.py) | Étape 3 Writer — rédaction LLM + hydratation visuels |
| [knowledge_base/report_template/mortality_template.yaml](knowledge_base/report_template/mortality_template.yaml) | Source de vérité : sections, narratives, visual_specs, activations |
| [knowledge_base/report_template/template_loader.py](knowledge_base/report_template/template_loader.py) | API unifiée : `build_manifest`, `load_section`, activations multi-clés |
