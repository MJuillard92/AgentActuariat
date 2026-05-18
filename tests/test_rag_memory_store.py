"""Tests du RAGMemoryStore (buffer + vectorstore + lazy rebuild + append)."""
from __future__ import annotations

import pytest


def test_store_starts_empty():
    from agents.rag.memory.rag_memory_store import RAGMemoryStore
    s = RAGMemoryStore(session_id="test_001")
    assert s.get_buffer() == []
    assert s.get_summary() is None


def test_buffer_appends_in_order():
    from agents.rag.memory.rag_memory_store import RAGMemoryStore
    s = RAGMemoryStore(session_id="test_002")
    s._append_buffer_only("q1", "a1", [])
    s._append_buffer_only("q2", "a2", [])
    buf = s.get_buffer()
    assert len(buf) == 2
    assert buf[0].user_q == "q1"
    assert buf[1].user_q == "q2"


def test_buffer_ring_fifo_caps_at_buffer_size():
    """Au-delà de BUFFER_SIZE, les anciens tours sont évincés (FIFO)."""
    from agents.rag.memory.rag_memory_store import RAGMemoryStore, BUFFER_SIZE
    s = RAGMemoryStore(session_id="test_003")
    for i in range(BUFFER_SIZE + 3):
        s._append_buffer_only(f"q{i}", f"a{i}", [])
    buf = s.get_buffer()
    assert len(buf) == BUFFER_SIZE
    # Les BUFFER_SIZE derniers
    assert buf[0].user_q == f"q{3}"      # q0,q1,q2 évincés
    assert buf[-1].user_q == f"q{BUFFER_SIZE + 2}"


def test_get_buffer_with_n_limits_results():
    from agents.rag.memory.rag_memory_store import RAGMemoryStore
    s = RAGMemoryStore(session_id="test_004")
    for i in range(4):
        s._append_buffer_only(f"q{i}", f"a{i}", [])
    assert len(s.get_buffer(n=2)) == 2
    assert s.get_buffer(n=2)[-1].user_q == "q3"
