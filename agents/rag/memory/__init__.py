"""
agents.rag.memory — Mémoire conversationnelle 3-niveaux RAG-only.

Composants :
- schemas         : RAGTurn, RAGSummary Pydantic
- rag_memory_store : buffer + vectorstore + lazy rebuild, RAM-only par session
- summarizer      : LLM nano synchrone pour compaction tour > SUMMARY_TRIGGER
"""
