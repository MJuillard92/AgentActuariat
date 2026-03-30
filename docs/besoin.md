# Expression de besoin — Agent Actuariat

## Pourquoi ce projet ?

Les actuaires réalisent des études de mortalité d'expérience pour construire des tables de mortalité propres à leur portefeuille. Ce travail comprend :
- l'analyse descriptive des données (volume, qualité, répartition)
- le calcul de l'exposition et des taux bruts
- le lissage des taux et la validation statistique
- la comparaison avec des tables de référence

Aujourd'hui ce travail est réalisé à la main, en Python dans des notebooks, ou avec des outils Excel coûteux. L'objectif de ce projet est de **guider l'actuaire dans ce travail** grâce à un agent conversationnel, tout en lui laissant le contrôle total des méthodes et paramètres.

## Pour qui ?

- **Actuaires** qui réalisent des études de mortalité (assurance vie, prévoyance, retraite)
- **Développeurs actuariels** qui souhaitent étendre les capacités de l'agent

## Ce que l'agent fait

1. **Rapport guidé** : l'actuaire charge son portefeuille CSV et dialogue avec l'agent. L'agent :
   - comprend la demande métier (analyse descriptive ? construction de table ?)
   - valide le mapping des colonnes du CSV avec les rôles attendus
   - appelle les fonctions actuarielles appropriées
   - affiche les résultats (tableaux, graphiques) dans le chat
   - produit un rapport PDF

2. **DEV** : les actuaires développeurs peuvent :
   - consulter et modifier les fonctions actuarielles directement dans l'interface
   - ajouter de nouvelles fonctions à un tool existant
   - voir quelles colonnes sont requises / optionnelles pour chaque fonction

## Ce que l'agent ne fait PAS

- Il ne **crée pas** de nouvelles méthodes actuarielles — il orchestre celles définies par les actuaires dans `report_agent/tools/`
- Il n'a pas accès à internet
- Il ne gère pas la tarification dommages (IBNR, Chain-Ladder, etc.)
- Il ne gère pas les tables de mortalité du marché (TD 88-90, TPRV 93 peuvent être chargées mais pas construites automatiquement)

## Contraintes

- Les données restent **locales** (pas de cloud, pas de partage externe)
- L'interface doit rester **simple** : 2 onglets maximum
- Le code doit être **lisible par un actuaire non-développeur** : fonctions courtes, docstrings claires, pas d'abstraction inutile
