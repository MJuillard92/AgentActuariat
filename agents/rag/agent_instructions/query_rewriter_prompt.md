## Query Rewriter — Prompt LLM nano

Reformule la question utilisateur en une requête de recherche concise et
précise (max 15 mots), en explicitant les termes techniques actuariels
implicites.

Règles :
- N'écris pas une question — écris une **affirmation de recherche**.
- Explicite les acronymes (`KM` → `Kaplan-Meier`, `IC` → `intervalle de confiance`).
- Conserve les noms propres (Whittaker-Henderson, Lee-Carter, Cairns-Blake-Dowd).
- Pas de ponctuation finale.

### Exemples

| Question utilisateur | Requête de recherche |
|---|---|
| "c'est quoi le truc avec h en lissage ?" | paramètre lissage h méthode Whittaker-Henderson |
| "comment marche le KM ?" | estimateur Kaplan-Meier taux bruts survie |
| "explique-moi l'A132-18" | obligation A132-18 Code des assurances certification table |
| "différence table périodique prospective" | table mortalité périodique versus prospective définition |

### Format de sortie

Une seule ligne, l'affirmation de recherche. Rien d'autre.

Question : {user_text}
Requête de recherche :
