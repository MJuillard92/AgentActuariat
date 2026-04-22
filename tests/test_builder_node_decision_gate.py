"""Tests : garde-fou Python dans builder_node pour le gate decision_required.

Si le dernier ToolMessage contient le marqueur decision_required ET le
LLM émet à la fois du content et des tool_calls, les tool_calls doivent
être écrasés pour forcer le LLM à rendre la main (pause utilisateur).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.mortality.agents.builder_node import _has_pending_decision  # noqa: E402


class _Msg:
    """Stub de message langchain — content suffit pour nos tests."""
    def __init__(self, content):
        self.content = content


def test_no_pending_decision_when_no_messages():
    assert _has_pending_decision([]) is False


def test_no_pending_decision_when_tool_message_without_marker():
    from langchain_core.messages import ToolMessage
    msgs = [ToolMessage(content='{"smoothed_table": [], "n_non_monotone": 0}', tool_call_id="1")]
    assert _has_pending_decision(msgs) is False


def test_pending_decision_when_last_tool_message_has_marker():
    from langchain_core.messages import ToolMessage
    msgs = [
        ToolMessage(content='{"smoothed_table": [], "decision_required": {"options": []}}', tool_call_id="1"),
    ]
    assert _has_pending_decision(msgs) is True


def test_pending_decision_looks_at_last_tool_message_only():
    """Un decision_required antérieur suivi d'un ToolMessage propre = plus de gate."""
    from langchain_core.messages import ToolMessage, HumanMessage
    msgs = [
        ToolMessage(content='{"decision_required": {"options": []}}', tool_call_id="1"),
        HumanMessage(content="option A"),
        ToolMessage(content='{"smoothed_table": [], "n_non_monotone": 0}', tool_call_id="2"),
    ]
    assert _has_pending_decision(msgs) is False


def test_pending_decision_ignores_non_tool_messages():
    from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
    msgs = [
        ToolMessage(content='{"decision_required": {"options": []}}', tool_call_id="1"),
        AIMessage(content="Question?"),
        HumanMessage(content="réponse"),
    ]
    # Le dernier ToolMessage contient toujours le marqueur → gate actif
    assert _has_pending_decision(msgs) is True
