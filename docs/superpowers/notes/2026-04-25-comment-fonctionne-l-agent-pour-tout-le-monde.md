# Comment fonctionne l'agent — explication pour tout le monde

> Document destiné à une personne qui n'a aucune connaissance en intelligence artificielle ou en agents conversationnels. Aucun jargon, des exemples concrets, des analogies du quotidien.

---

## 1. De quoi parle-t-on exactement ?

L'**agent**, c'est un programme informatique qui aide un actuaire à construire des **tables de mortalité** et à rédiger un **rapport PDF**. L'utilisateur lui parle en français, comme à un collègue, et l'agent fait les calculs et la rédaction tout seul.

À l'intérieur de cet agent, il y a en fait **trois "personnages"** qui se passent le relais :

1. **Le Master** (le chef d'orchestre).
2. **Le Builder** (le calculateur).
3. **Le Writer** (le rédacteur).

C'est comme dans un bureau d'études : un chef de projet (Master) reçoit la demande du client, il confie les calculs à un actuaire (Builder), puis confie la rédaction à un consultant (Writer). Ces trois personnages **ne sont pas des humains** : ce sont des programmes qui chacun appellent un grand modèle de langage (gpt-5.4, gpt-5.4-mini, etc.) pour réfléchir.

## 2. Imaginons un cas concret

L'utilisateur charge un fichier Excel contenant 1 000 contrats d'assurance vie, puis tape :

> *"Construis-moi une table de mortalité avec les taux bruts et fais-moi le rapport."*

Voyons ce qui se passe, étape par étape.

---

## 3. Le Master prend la main d'abord

Le Master, c'est le **chef d'orchestre**. Il ne fait jamais de calcul, il ne rédige pas de rapport. Son seul rôle, c'est de **lire ce que l'utilisateur veut** et de **décider qui doit travailler ensuite**.

### Étape 1 — Comprendre la demande

Le Master pose 3 questions au modèle de langage (un mini-LLM rapide et bon marché) :

1. **C'est quoi cette demande ?** — Une vraie tâche métier (`task`) ou juste une question (`question`) ?
2. **Le client veut un rapport PDF ?** — Oui (`yes`), non (`no`), ou il n'a rien dit clairement (`ask`) ?
3. **Quel type de calcul ?** — Tout le pipeline (`full_report`), juste les taux bruts (`raw_rates`), ou juste une description (`description`) ?

Pour notre exemple *"Construis-moi une table de mortalité avec les taux bruts et fais-moi le rapport"*, le mini-LLM répond :
```
kind        = task
write       = yes        (le mot "rapport" est explicite)
report_mode = raw_rates  (le mot "taux bruts" est explicite)
```

Le Master enregistre ces 3 informations dans une **mémoire partagée** (qu'on appelle `data_store` dans le code).

### Étape 2 — Vérifier les colonnes du fichier

Le Master regarde le fichier Excel chargé. Si les colonnes ne sont pas reconnues automatiquement (par exemple, le fichier a une colonne "DateNais" au lieu de "date_naissance"), il pose la question à l'utilisateur via une fenêtre :

> *"J'ai vu que vous avez une colonne `DateNais`. C'est bien la date de naissance ?"*

L'utilisateur valide, et on passe à la suite. C'est ce qu'on appelle la **désambiguation**.

### Étape 3 — Choisir les sections du rapport

Le Master ouvre un **fichier de configuration** (`mortality_template.yaml`) qui décrit toutes les sections possibles d'un rapport actuariel :

- Préambule
- Retraitement des données
- Analyse descriptive
- Construction de la table
- Lissage
- Validation
- Comparaison réglementaire
- Conclusion

Chaque section dit dans quel mode elle est active. Comme l'utilisateur a demandé `raw_rates`, le Master active toutes les sections **sauf** une variante : la section "lissage" sera présente dans le PDF mais avec une mention spéciale du genre *"Les taux bruts ont été utilisés sans lissage à la demande du client"*.

Pour chaque section active, le Master collecte la liste des **données nécessaires** : par exemple, la section "préambule" a besoin de l'exposition totale, du nombre de décès, de la composition par sexe, de la série annuelle, etc. Au total, disons 10 chiffres clés à produire.

### Étape 4 — Donner ses ordres au Builder

Le Master rédige un **message d'instruction** à destination du Builder. Pas un message à l'utilisateur — un message interne entre deux programmes. Concrètement :

```
Mode de rapport : raw_rates
Sections actives : [preamble, data_preprocessing, data_analysis_unisex,
                    table_construction, smoothing, validation, benchmarking,
                    conclusion]
Déjà produit (NE PAS relancer) : []
Reste à produire : [cleaned_records, total_exposure, total_deaths,
                    segmentations, serie, ages, ...]
Émets <BUILD_DONE> quand toutes les clés ci-dessus sont dans le data_store.
```

Le Master place ce message dans la **conversation interne** (la liste de messages partagée entre tous les personnages) et passe la main au Builder.

> **Analogie** : c'est comme si le chef de projet glissait une note Post-it sur le bureau de l'actuaire : *"Voici ce que je veux que tu calcules. Quand t'as fini, accroche un drapeau rouge sur ton écran."*

---

## 4. Le Builder fait les calculs

Le Builder, c'est l'**actuaire-calculateur**. Il a une boîte à outils — la "boîte à tools" — avec des fonctions Python :

- `preprocessing.clean_records` : nettoie les données (supprime les lignes aberrantes).
- `statistical_analysis.portfolio_summary` : résume le portefeuille.
- `statistical_analysis.segmentation` : ventile par sexe, produit, etc.
- `statistical_analysis.time_series` : calcule les séries annuelles.
- `statistical_analysis.age_distribution` : pyramide des âges.
- `builder.exposure` : calcule l'exposition centrale par âge.
- `builder.crude_rates` : calcule les taux bruts.
- `builder.smoothing` : applique le lissage.
- `builder.validation` : teste statistiquement la table.
- `builder.benchmarking` : compare à une table réglementaire.

### Comment le Builder choisit quel outil utiliser ?

Le Builder a un **mode d'emploi** (un long texte appelé "system prompt") qui contient :

- Sa carte d'identité ("Tu es un actuaire senior…").
- Le **catalogue de tous les outils disponibles** avec leurs descriptions et leurs paramètres (en JSON).
- Une **table** qui dit, pour chaque section du rapport, quels outils utiliser.
- Les ordres reçus du Master (le message vu plus haut).

Quand le Builder réfléchit, il appelle un grand modèle de langage (gpt-5.4) en lui passant tout ce contexte. Le modèle décide alors **quelle fonction appeler en premier** et **avec quels paramètres**.

Par exemple, le Builder pourrait répondre :

```
Je vais commencer par nettoyer les données.
→ Appel de tool : preprocessing.clean_records
   avec : {observation_end: "31/12/2023"}
```

Un programme intermédiaire (qu'on appelle un **ToolNode**) récupère cette demande, exécute la fonction Python correspondante (qui filtre le DataFrame), et renvoie le résultat au Builder dans un nouveau message :

```
[ToolMessage] {"cleaned_records": [...], "exclusion_report": {...},
              "total_records": 950}
```

Le Builder lit ce résultat, l'enregistre dans la mémoire partagée, puis décide du **prochain outil**. Et ainsi de suite, jusqu'à ce qu'il ait produit toutes les données demandées.

### Le mode `raw_rates` — particularité

Souvenez-vous : l'utilisateur a demandé `raw_rates`. Pour respecter cette consigne :

1. Le Builder appelle quand même `builder.exposure`, `builder.crude_rates`, `builder.diagnostics`.
2. **Mais** un **petit programme déterministe** (sans LLM) prend le relais et copie directement les taux bruts dans le champ "table lissée" — c'est l'**assimilation**. Pas de vraie technique de lissage appliquée.
3. Le Builder continue avec `builder.validation` et `builder.benchmarking` sur cette table assimilée.

C'est plus rapide et moins coûteux qu'un vrai lissage, et c'est ce que l'utilisateur a demandé.

### Le drapeau rouge

Une fois toutes les données dans la mémoire partagée, le Builder écrit un message contenant le mot magique `<BUILD_DONE>` (notre "drapeau rouge"). Le programme principal détecte ce mot et **rend la main au Master**.

> **Analogie** : l'actuaire termine ses calculs, range les feuilles dans un classeur posé sur le bureau (la mémoire partagée), et accroche le drapeau rouge sur son écran. Le chef de projet voit le drapeau et reprend la main.

---

## 5. Comment les personnages se parlent ?

Tout passe par la **liste de messages**. C'est comme un fil de chat entre 4 participants :

- L'utilisateur.
- Le Master.
- Le Builder.
- Les outils (qui ont leur propre type de message, le `ToolMessage`).

Chaque message a un type :

| Type | Auteur | Exemple |
|---|---|---|
| `HumanMessage` | l'utilisateur | "Construis-moi une table de mortalité." |
| `HumanMessage` (interne) | le Master adressant le Builder | "Sections actives : … Reste à produire : …" |
| `AIMessage` | un personnage qui répond | "Je commence par nettoyer les données." |
| `ToolMessage` | le résultat brut d'un outil | `{"cleaned_records": [...], "total_records": 950}` |

Le programme principal qui orchestre tout ça s'appelle **LangGraph**. Sa logique est simple :

- *"Quel est le dernier message ? Si c'est un AIMessage qui demande un outil, j'appelle l'outil. Sinon, je passe la main au prochain personnage."*

LangGraph ne **comprend rien au métier**. Il fait juste circuler les messages entre les personnages selon des règles prédéfinies.

### La fenêtre des 20 derniers messages

Quand on demande à un grand modèle de langage de réfléchir, on ne peut pas lui donner toute la conversation depuis le début (trop long, trop cher). On lui donne **les 20 derniers messages**, plus le mode d'emploi du personnage. Au-delà, c'est oublié — sauf si on l'a stocké dans la mémoire partagée (`data_store`).

---

## 6. Le Master reprend la main après le Builder

Le Master voit le drapeau `<BUILD_DONE>`. Il vérifie deux choses :

1. **Toutes les données nécessaires sont-elles dans la mémoire partagée ?** Le Master a la liste, il check. Si oui → étape suivante. Si non → on retourne au Builder avec un message du genre *"il manque encore X et Y, finis le boulot"*.
2. **L'utilisateur veut-il un rapport PDF ?** Souvenez-vous : `write=yes`. Donc oui.

Conséquence : le Master passe la main au **Writer**.

### Petit aparté : et si l'utilisateur n'avait rien dit sur le rapport ?

Si l'utilisateur avait juste tapé *"Construis-moi une table"* (sans dire "rapport"), la classification aurait été `write=ask`. Dans ce cas, le Master aurait posé la question **avant même de lancer le Builder** :

> *"Voulez-vous que je génère un rapport PDF à la fin des calculs ?"*

L'utilisateur répond *"oui"* ou *"non"*, le Master reclassifie et continue dans la bonne branche. Pas de calculs inutiles : on demande l'avis de l'humain dès le départ.

---

## 7. Le Writer rédige le rapport

Le Writer est un **mini-pipeline** en plusieurs étapes :

### Étape A — Charger le plan

Le Writer ouvre le fichier `mortality_template.yaml` (le même que celui utilisé par le Master). Il lit, pour chaque section active :

- Le **texte narratif** avec des `{{ trous }}` à remplir (ex: *"Le portefeuille comporte {{ total_records }} contrats observés sur {{ num_observation_years }} années"*).
- Les **directives de rédaction** : ton (neutre, descriptif…), longueur cible (200-300 mots…), tags pour la recherche documentaire.
- Les **visuels à produire** : tableaux et graphiques avec leurs colonnes.

Pour chaque section, le Writer remplace les `{{ trous }}` par les vraies valeurs prises dans la mémoire partagée. Si la valeur manque, il met un tiret `—` et signale la section comme "incomplète".

### Étape B — Compléter avec une recherche documentaire

Pour les sections où le YAML demande une recherche (champ `rag_query`), le Writer interroge une base de connaissances (par exemple, des PDFs d'articles actuariels). Il récupère 2-3 paragraphes pertinents et les ajoute au contexte de rédaction.

### Étape C — Rédaction par le LLM

Pour chaque section, le Writer envoie au modèle de langage (gpt-5.4) :

- Le **rôle attendu** : *"Tu es actuaire senior, ton attendu : neutre et descriptif, longueur 250 mots, en français."*
- La **narrative de référence** avec les chiffres déjà résolus.
- Les **règles** : *"Ne cite QUE des chiffres présents dans la narrative ou les tableaux. Ne dépasse pas 10% de la longueur cible."*
- Les **tableaux et graphiques** disponibles (en données brutes).

Le modèle rédige un texte fluide, en restant fidèle aux chiffres et au ton demandés.

### Étape D — Hydratation des visuels

En parallèle de la rédaction, le Writer transforme chaque "spec" de visuel en un objet concret :

- Pour un **tableau** : il extrait les colonnes demandées depuis la mémoire partagée et formate les en-têtes et les lignes.
- Pour un **graphique** : il extrait les valeurs en X et en Y et appelle un programme qui produit une image PNG.

Exemple : la section "préambule" a un tableau "composition par sexe" qui pointe vers `segmentations.sexe`. Le Writer va lire `data_store["segmentations"]["sexe"]` et obtient une liste comme :
```
[{"valeur": "H", "nb_contrats": 500, "nb_deces": 25},
 {"valeur": "F", "nb_contrats": 500, "nb_deces": 17}]
```

Il transforme ça en tableau Markdown ou en image PNG selon le rendu cible.

### Étape E — Assemblage du PDF

Tous les morceaux (textes rédigés + tableaux + graphiques) sont assemblés en un **document PDF** par un programme dédié (qui utilise les bibliothèques classiques de mise en page).

Quand le PDF est généré et sauvegardé sur le disque, le Writer écrit un message contenant `<WRITE_DONE>` et rend la main au Master.

---

## 8. Fin de session

Le Master reçoit `<WRITE_DONE>`. Il nettoie quelques flags internes et émet un événement `done` qui dit *"c'est fini"*. L'interface utilisateur affiche alors un lien vers le PDF généré, et la session est prête pour une nouvelle demande.

---

## 9. Récap visuel — l'analogie du bureau d'études

```
              ┌──── BUREAU D'ÉTUDES ────┐
              │                         │
              │  CHEF DE PROJET (Master)│
   utilisateur│           │             │
       ──────►│           │  notes      │
       "fais  │           ▼             │
       table" │       ┌───────┐         │
              │       │ tâche │         │
              │       └───┬───┘         │
              │           │             │
              │           ▼             │
              │   ACTUAIRE (Builder)    │
              │       │     │           │
              │       │     ▼           │
              │       │  ┌──────┐       │
              │       │  │BOÎTE │       │
              │       │  │ TOOLS│       │
              │       │  └──────┘       │
              │       │     │           │
              │       └─────┘           │
              │           │             │
              │  classeur de calculs    │
              │  (data_store)           │
              │           │             │
              │           ▼             │
              │  CONSULTANT (Writer)    │
              │       │     │           │
              │       │     ▼           │
              │       │  ┌──────────┐   │
              │       │  │ CHARTE / │   │
              │       │  │ TEMPLATE │   │
              │       │  └──────────┘   │
              │       │     │           │
              │       └─────┘           │
              │           │             │
              │           ▼             │
              │       ┌──────┐          │
              │       │ PDF  │          │
              │       └──────┘          │
              │           │             │
              │           ▼             │
   utilisateur ◄─────────────           │
              │                         │
              └─────────────────────────┘
```

---

## 10. Ce que ça veut dire concrètement

- **Tout est piloté par un fichier YAML** (`mortality_template.yaml`). Si demain tu veux changer une formule de calcul, ajouter une section au rapport, ou modifier un tableau, tu touches ce fichier — pas le code Python.
- **Les modèles LLM sont configurables** dans `config/llm_models.yaml`. Tu peux passer le Master sur un mini moins cher, ou tester un nouveau modèle de raisonnement sur le Builder, sans toucher au code.
- **Les outils sont indépendants**. Chaque outil est un fichier Python isolé. On peut en ajouter un nouveau (par exemple un nouveau test statistique) en l'écrivant + en l'inscrivant dans le catalogue ; le Builder le découvrira tout seul.
- **La sécurité** : trois garde-fous empêchent l'agent de tourner en rond indéfiniment (compteur de tours, compteur de cycles, blocage automatique quand un outil demande une décision utilisateur).

---

## 11. Petit lexique

| Terme | Traduction simple |
|---|---|
| **Agent** | un programme qui dialogue + décide quoi faire |
| **LLM** | grand modèle de langage (gpt-5.4, etc.) — le "cerveau" qui réfléchit |
| **Tool** | une fonction Python que l'agent peut appeler pour calculer un truc précis |
| **Tool-calling** | le modèle décide d'appeler tel outil avec tels paramètres |
| **System prompt** | le mode d'emploi qu'on donne au modèle à chaque réflexion |
| **data_store** | la mémoire partagée entre le Master, le Builder et le Writer |
| **YAML** | un fichier de configuration lisible par un humain (texte structuré) |
| **LangGraph** | le programme qui fait circuler les messages entre les personnages |
| **`<BUILD_DONE>`** | un mot magique que le Builder écrit quand il a fini ses calculs |
| **`<WRITE_DONE>`** | un mot magique que le Writer écrit quand le PDF est prêt |
| **Désambiguation** | l'étape où le Master demande à l'utilisateur de lever une ambiguïté (colonnes, valeurs) |
| **Prompt cache** | optimisation OpenAI : un texte identique envoyé 2 fois en moins de 5 minutes coûte 10x moins cher la 2ème fois |

---

## 12. Pour aller plus loin

Si tu veux comprendre la logique exacte de routing entre Master, Builder et Writer, le document [2026-04-24-architecture-agent-visuelle.md](2026-04-24-architecture-agent-visuelle.md) contient les diagrammes de flux complets. Le document présent reste volontairement de haut niveau et orienté narration ; le diagramme entre dans le détail technique.

Pour le code, les 4 fichiers les plus importants à connaître :

| Fichier | Rôle |
|---|---|
| `agents/mortality/agents/master_node.py` | Le "chef d'orchestre" |
| `agents/mortality/agents/builder_node.py` | "L'actuaire-calculateur" |
| `agents/report/pipeline/_04_redaction.py` | "Le consultant rédacteur" |
| `knowledge_base/report_template/mortality_template.yaml` | La charte du rapport |
| `config/llm_models.yaml` | Les choix de modèles LLM |
