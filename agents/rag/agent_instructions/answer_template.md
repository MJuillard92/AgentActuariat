## Format de réponse — RAGAgent

### Règles strictes

1. **Synthèse rédigée** en 3 à 6 phrases, français professionnel.
2. **Citations inline** entre crochets : `[D03.02]`, `[D07.01]`.
   Pas de citation = pas d'affirmation autorisée.
3. **Section Sources** finale listant les chunks réellement utilisés.
4. **Formules en Unicode** (Σ, Δ, q̃, ²) — pas de LaTeX brut (`\sum`, `\Delta`).
5. **Si corpus insuffisant** : "Le corpus ne couvre pas ce point" — ne pas
   inventer.

### Exemple de sortie attendue

```
Le lissage Whittaker-Henderson est une méthode de régularisation qui pénalise
les différences finies d'ordre k entre les taux bruts adjacents [D03.02]. Le
paramètre h contrôle l'arbitrage biais-variance : h faible préserve la
structure locale, h élevé renforce la régularité [D03.04]. Le choix optimal
de h s'effectue typiquement par validation croisée ou critère AIC [D03.04].

Sources :
- D03.02 — Whittaker-Henderson 1D
- D03.04 — Sélection du paramètre h (Biessy 2023)
```

### Anti-patterns interdits

- ❌ Coller plusieurs chunks bruts juxtaposés (templating sans synthèse).
- ❌ Affirmer sans citation : "La méthode est X" sans `[Dxx.yy]`.
- ❌ Citer un doc_id qui n'apparaît pas dans les chunks retournés.
- ❌ LaTeX brut : `M(\tilde q) = h \sum_i ...`.
- ❌ Inventer une référence si le corpus ne couvre pas la question.
