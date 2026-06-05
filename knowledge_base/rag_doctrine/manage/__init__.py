"""
knowledge_base.rag_doctrine.manage — gestion du corpus doctrinal

Sous-commandes CLI (cli.py) :
  add      : injecte PDF/DOCX d'un dossier dans le corpus
  list     : liste les documents indexés
  delete   : supprime un doc_id (et ses chunks)
  rebuild  : reconstruit l'index FAISS depuis chunks_enriched.json

UI Dash (ui.py) : onglet "Doctrine RAG" pour canvas_app.py
  - panneau gauche : liste des documents
  - panneau droit  : chunks (texte complet) du doc sélectionné
  - upload PDF/DOCX → pipeline complet (extract → chunk → enrich → embed → FAISS)
"""
