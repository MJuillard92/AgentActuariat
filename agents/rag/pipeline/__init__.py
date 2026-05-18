"""
agents.rag.pipeline — Pipeline RAG en 5 étapes.

Composants :
- query_normalizer : correction déterministe des typos actuariels (Python pur)
- query_rewriter   : reformulation LLM nano pour optimiser le retrieval
- answer_generator : synthèse rédigée LLM mini avec citations groundées
- grounding_check  : vérification optionnelle LLM mini que les citations
                     sont bien supportées par les chunks
- run_pipeline     : orchestrateur logique pur (pas de LangGraph)
"""
