# Guide de style — rapport de certification de table de mortalité

Ce guide résume les tournures, conventions typographiques et la structure
narrative observées dans le rapport Winter & Associés AF8796-TD3 (2012),
référence pour la rédaction des rapports de certification de l'agent.

## Ton général

- Formel, descriptif, interprétatif en fin de section.
- Actuariel senior : chaque chiffre est commenté, pas seulement cité.
- Voix passive ou première personne du pluriel sobre (« on notera »,
  « on trouve »), jamais la première personne du singulier.
- Le narrateur évalue mais ne porte pas de jugement catégorique :
  « la méthodologie n'appelle pas de commentaire particulier »,
  « ce caractère linéaire tranche avec les analyses observées ».

## Tournures récurrentes à réutiliser

- « On notera [une tendance / une stabilité / un écart] » — pour introduire un
  constat factuel.
- « Il est précisé que [clarification technique] » — pour une mise au point.
- « Au global, [synthèse chiffrée] » — pour agréger après une analyse fine.
- « Le fait que [observation] met en évidence [conclusion] » — pour un lien
  cause-conséquence rigoureux.
- « La méthodologie n'appelle pas de commentaire particulier » — formule
  standard quand le processus suivi est conforme.
- « On peut en retenir [synthèse] » — pour conclure un paragraphe analytique.
- « Cette analyse est complétée par [contrôle additionnel] » — pour chaîner
  les contrôles.
- « Ce point est conforté par [second indicateur] » — pour corroborer.
- « Classiquement observé en pratique » — pour marquer l'alignement avec les
  références actuarielles.
- « Il est à noter que » — pour une mise en garde ou une nuance.

## Conventions typographiques

- **Décimales** : virgule française (0,289% et non 0.289%).
- **Milliers** : espace insécable fine (253 067 lignes, 780 411 années).
- **Pourcentages** : signe % collé au chiffre, précédé du nombre avec virgule
  décimale française si nécessaire (14,5%).
- **Intervalles de confiance** : `IC 95%` ou `intervalle de confiance à 95 %`.
- **Guillemets** : guillemets français « … » pour les noms de produits et
  citations.
- **Formules mathématiques** : notation LaTeX en ligne ($q_x = D_x / E_x$) ou
  en bloc centrée (\[ … \]) quand la formule est longue.
- **Références de tableaux et figures** : numérotées et citées en italique
  dans le texte : « cf. Tableau 7 », « cf. Figure 8 ».

## Référencement des tableaux et figures

- Chaque tableau et figure est introduit dans la prose AVANT son affichage :
  « La comparaison par tranches d'âges représentant près de 10 % de
  l'exposition au risque vient confirmer cette observation : »
- Suivi immédiatement du titre numéroté (« Tableau 7 — Comparaison des décès
  observés et des décès modélisés (par classe d'âges) »).
- Après le tableau, un paragraphe COMMENTE ce que l'on y voit : quels âges
  sont aberrants, quelle est l'ampleur de l'écart, etc. Jamais un tableau
  laissé sans interprétation.

## Structure d'une section type

1. **Phrase d'ouverture** qui pose l'objectif de la section et rappelle le
   contexte (« Afin de mesurer la prudence de la table d'expérience, l'approche
   retenue a consisté à … »).
2. **Exposé méthodologique** court (1-2 paragraphes) précisant la procédure
   de calcul ou le choix retenu.
3. **Résultats chiffrés** cités textuellement dans la prose, accompagnés
   d'une lecture actuarielle (prudence, cohérence, écart significatif).
4. **Tableau ou figure** avec caption numérotée.
5. **Commentaire du tableau** : quels âges, quelle amplitude, quelle
   interprétation pour la certification.
6. **Phrase de transition** vers la section suivante ou de synthèse locale.

## Transitions typiques entre sections

- « Sur cette base on effectue quelques statistiques descriptives présentées
  ci-après. »
- « Cette analyse est complétée, pour chaque année de la période considérée,
  par … »
- « L'analyse par âge conforte ce constat : »
- « Au global, [synthèse de ce qui précède]. »

## Conclusion type

- Rappel synthétique du périmètre et de la méthodologie.
- Reprise des **indicateurs clés chiffrés** (décès observés/modélisés, SMR,
  intervalle de confiance, abattements).
- Énoncé du **verdict** : table certifiable ou non, durée de validité
  (5 ans classiquement).
- **Domaine de validité** : produits concernés, évolutions acceptées.
- **Dispositif de suivi** : indicateurs à produire annuellement
  (obs/modélisé par classe d'âge, positionnement IC 95 %, sex-ratio).

## À éviter dans la rédaction

- Pas de « [donnée non disponible] » dans la prose : si une statistique manque,
  omettre la phrase entière plutôt que de la signaler.
- Pas de placeholder brut type `{{ … }}` ou `[…]` : toujours résoudre ou
  supprimer la phrase.
- Pas de chiffre en notation scientifique (`2.14e-3`) dans le corps de texte :
  écrire « 0,00214 » ou « 2,14 ‰ ».
- Pas de superlatifs sans justification chiffrée (« très bonne prudence »
  doit être soutenu par un ratio, un IC, etc.).
- Pas de paraphrase plate du tableau : commenter, ne pas recopier.
