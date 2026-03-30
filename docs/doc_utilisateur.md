# Guide utilisateur — Agent Actuariat

## Lancer l'application

```bash
python canvas_app.py
```
Puis ouvrir http://localhost:8050 dans votre navigateur.

## Onglet "Rapport guidé"

### 1. Charger votre portefeuille

Glissez-déposez votre fichier CSV (ou cliquez "choisir un fichier").

Le fichier est analysé automatiquement :
- Les colonnes sont mappées aux rôles actuariels (date d'entrée, cause de sortie, sexe…)
- Un résumé s'affiche : nombre de lignes, colonnes reconnues (✓) et manquantes (✗)
- Le nombre de fonctions disponibles avec vos données est indiqué

**Formats acceptés** : CSV avec séparateur `;`, `,`, tabulation ou `|`. Encodages UTF-8 et Latin-1.

**Colonnes reconnues automatiquement** (noms insensibles à la casse) :
| Rôle | Noms reconnus |
|---|---|
| Date d'entrée | date_entree, ctreffet, entry_date |
| Date de sortie | date_sortie, exit_date |
| Date de naissance | date_naissance, clinaiss, dob |
| Cause de sortie | cause_sortie, statut, status |
| Sexe | sexe, sexeref, gender |
| Produit | cdprod, produit, product |

### 2. Dialoguer avec l'agent

Tapez votre demande dans la zone de texte et cliquez **Envoyer**.

**Exemples de demandes** :
- "Donne-moi un résumé de mon portefeuille"
- "Montre-moi la pyramide des âges par sexe"
- "Construis une table de mortalité d'expérience pour les hommes entre 30 et 80 ans"
- "Génère un rapport PDF descriptif"
- "Compare les méthodes de lissage Whittaker et Gompertz"

L'agent va :
1. Valider le mapping de vos colonnes (vous poser des questions si besoin)
2. Appeler les fonctions actuarielles nécessaires (visible dans le chat : icône ⚙️)
3. Afficher les résultats et graphiques
4. Rédiger une analyse en langage naturel

**Indicateurs dans le chat** :
- ⚙️ `statistical_analysis.portfolio_summary` : l'agent appelle une fonction
- ✓ `portfolio_summary → nb_contrats, nb_deces, …` : résultat reçu
- Graphiques s'affichent directement dans la conversation
- Badge **Prêt** (vert) : agent disponible | **En cours…** (orange) : calcul en cours

### 3. Décès reconnus automatiquement

La colonne cause de sortie est analysée automatiquement. Les valeurs suivantes sont interprétées comme des décès :
`deces`, `décès`, `dcd`, `d`, `dead`, `mort`, `1`, `true`, `oui`, `yes`

---

## Onglet "DEV"

### Onglet Capacités

Affiche toutes les fonctions disponibles par tool, avec :
- **✓** / **indisponible** : statut de la fonction
- **Code** : ouvre le fichier Python dans l'éditeur
- **Req** (rouge) : colonnes requises
- **Opt** (bleu) : colonnes optionnelles

**Bouton Ajouter** sur chaque tool : ouvre un formulaire pour créer une nouvelle fonction. Le code Python est généré automatiquement à partir des colonnes sélectionnées.

### Onglet Code

Arborescence des fichiers `report_agent/` :
- `dictionary/` : mappings colonnes
- `tools/statistical_analysis/` : analyses descriptives
- `tools/builder/` : construction table de mortalité
- `tools/graphs/` : graphiques
- `tools/reasoning/` : compréhension métier
- `tools/build_pdf/` : génération PDF

Cliquer sur un fichier l'ouvre dans l'éditeur. Modifier et cliquer **Sauvegarder** pour enregistrer.

---

## Pipeline de construction de table (ordre des appels)

```
1. builder.exposure        → calcule E_x et D_x par âge
2. builder.crude_rates     → estime q_x brut
3. builder.diagnostics     → évalue la crédibilité des données
4. builder.smoothing       → lisse q_x (Whittaker, Gompertz…)
5. builder.validation      → intervalles de confiance + chi2
6. builder.benchmarking    → comparaison vs table de référence (TH/TF 00-02…)
7. graphs.builder_plots    → graphiques (exposition, taux bruts vs lissés, SMR)
8. build_pdf.certification_report → rapport PDF (à venir)
```

---

## Paramètres actuariels

Les seuils et paramètres de la pipeline (lambda Whittaker, âges min/max, seuil de crédibilité…) sont dans `notebooks/actuarial_params.py`. Modifier ce fichier et redémarrer l'application.
