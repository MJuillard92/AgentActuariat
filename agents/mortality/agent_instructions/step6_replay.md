## Replay d'une session

Si le message du client contient `=== SESSION LOG` ou "rejoue cette session" ou "refais la même analyse" :
1. Chercher le bloc JSON sous `REPLAY — Bloc JSON`.
2. Exécuter les étapes dans l'ordre exact avec les paramètres indiqués.
3. Annoncer : "Je rejoue la session ({n} étapes détectées). Démarrage…"
4. Ne pas afficher le plan (déjà défini dans le log).
