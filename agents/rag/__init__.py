"""
rag_agent — Agent RAG (Retrieval-Augmented Generation) pour les questions
doctrinales actuarielles.

Pair de `agents.mortality` et `agents.report` — chargé via le master
quand l'intent classifié est `question`.

Pipeline :
    Master → rag_node (LangGraph adapter)
           → agents.rag.pipeline.run_pipeline.run(state)
                ├─ query_normalizer.normalize    (Python pur)
                ├─ query_rewriter.rewrite        (LLM nano, opt.)
                ├─ search_doctrine.run           (FAISS+BM25+RRF)
                ├─ answer_generator.generate     (LLM mini)
                └─ grounding_check.verify        (LLM mini, opt.)

Usage :
    from agents.rag.pipeline.run_pipeline import run as run_rag
    result = run_rag(state)
    # → {"answer": str, "sources": list[dict], "stage_events": list[tuple]}
"""
__version__ = "1.0.0"
