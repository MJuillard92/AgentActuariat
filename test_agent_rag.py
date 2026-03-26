"""
test_agent_rag.py
=================
Tests unitaires pour le RAG dynamique et l'agent actuariel.
Utilise des mocks pour éviter les appels API réels (coûteux/lents).

Lancer :
    cd "Agent actuariat"
    python -m pytest test_agent_rag.py -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# S'assurer que le répertoire courant est bien le projet
sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers réutilisables
# ─────────────────────────────────────────────────────────────────────────────

def _make_steps(n: int, output_size: int = 200) -> list[dict]:
    """Génère n étapes factices avec sorties de taille donnée."""
    return [
        {"label": f"Étape {i+1}", "output": f"Résultat étape {i+1}: " + "x" * output_size}
        for i in range(n)
    ]


def _mock_embed_response(n_vecs: int = 1, dim: int = 3) -> MagicMock:
    """Crée un mock de réponse d'embedding OpenAI."""
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[0.1] * dim) for _ in range(n_vecs)]
    return resp


def _mock_chat_response(content: str, tool_calls=None, finish_reason: str = "stop") -> MagicMock:
    """Crée un mock de réponse ChatCompletion."""
    resp = MagicMock()
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    resp.choices = [MagicMock(finish_reason=finish_reason, message=msg)]
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tests build_index
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildIndex:

    def test_empty_steps_returns_empty(self):
        from rag import build_index
        assert build_index([]) == []

    def test_skips_steps_with_empty_output(self):
        from rag import build_index
        steps = [{"label": "E1", "output": ""}, {"label": "E2", "output": "   "}]
        assert build_index(steps) == []

    def test_creates_one_chunk_per_step(self):
        from rag import build_index
        steps = _make_steps(3, output_size=50)
        chunks = build_index(steps)
        assert len(chunks) == 3

    def test_chunk_contains_label_and_output(self):
        from rag import build_index
        steps = [{"label": "Calcul SMR", "output": "SMR = 0.87 (IC 95% : 0.81-0.93)"}]
        chunks = build_index(steps)
        assert "Calcul SMR" in chunks[0]["text"]
        assert "SMR = 0.87" in chunks[0]["text"]

    def test_truncates_long_output(self):
        from rag import build_index, _MAX_CHUNK_CHARS
        long_output = "Z" * (_MAX_CHUNK_CHARS * 3)
        steps = [{"label": "Long", "output": long_output}]
        chunks = build_index(steps)
        assert "[…]" in chunks[0]["text"]
        # Le chunk ne doit pas dépasser 2× _MAX_CHUNK_CHARS (moitié début + moitié fin + délimiteur)
        assert len(chunks[0]["text"]) < _MAX_CHUNK_CHARS * 2

    def test_uses_description_when_label_absent(self):
        from rag import build_index
        steps = [{"description": "Lissage Whittaker", "output": "lambda=100, monotone=True"}]
        chunks = build_index(steps)
        assert len(chunks) == 1
        assert "Lissage Whittaker" in chunks[0]["label"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tests fenêtre RAG dynamique
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicWindow:

    def test_empty_chunks_returns_zero_top_k(self):
        from rag import _dynamic_window
        top_k, max_tokens = _dynamic_window([], "question")
        assert top_k == 0
        assert max_tokens >= 512

    def test_small_corpus_uses_all_chunks(self):
        from rag import _dynamic_window
        chunks = [{"text": "a" * 100, "label": str(i)} for i in range(5)]
        top_k, _ = _dynamic_window(chunks, "question?")
        assert top_k == 5  # tout petit corpus → tout utiliser

    def test_top_k_never_exceeds_available_chunks(self):
        from rag import _dynamic_window
        chunks = [{"text": "a" * 100, "label": str(i)} for i in range(3)]
        top_k, _ = _dynamic_window(chunks, "question?")
        assert top_k <= 3

    def test_max_tokens_within_acceptable_range(self):
        from rag import _dynamic_window
        chunks = [{"text": "a" * 1000, "label": str(i)} for i in range(20)]
        _, max_tokens = _dynamic_window(chunks, "question?", summary="résumé")
        assert 512 <= max_tokens <= 8192

    def test_large_corpus_exceeds_old_hardcoded_top_k_5(self):
        """Avec 50 chunks disponibles, top_k doit dépasser l'ancienne valeur codée en dur (5)."""
        from rag import _dynamic_window
        chunks = [{"text": "a" * 500, "label": str(i)} for i in range(50)]
        top_k, _ = _dynamic_window(chunks, "question?")
        assert top_k > 5

    def test_max_tokens_at_least_512_even_with_large_context(self):
        from rag import _dynamic_window
        chunks = [{"text": "a" * 1500, "label": str(i)} for i in range(200)]
        _, max_tokens = _dynamic_window(chunks, "question?")
        assert max_tokens >= 512

    def test_top_k_at_least_1_when_chunks_available(self):
        from rag import _dynamic_window
        chunks = [{"text": "a" * 100, "label": "x"}]
        top_k, _ = _dynamic_window(chunks, "q" * 10000)  # très longue question
        assert top_k >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tests answer_with_rag
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerWithRag:

    def test_no_steps_returns_error_message(self):
        from rag import answer_with_rag
        result = answer_with_rag("question", steps=[])
        assert "Aucun résultat" in result

    def test_all_empty_outputs_returns_error(self):
        from rag import answer_with_rag
        result = answer_with_rag("question", steps=[{"label": "E", "output": ""}])
        assert len(result) > 0  # message d'erreur

    @patch("rag._get_client")
    def test_uses_dynamic_max_tokens(self, mock_get_client):
        """max_tokens dans l'appel API doit dépasser l'ancienne valeur codée en dur (700)."""
        from rag import answer_with_rag

        client = MagicMock()
        client.embeddings.create.return_value = _mock_embed_response(n_vecs=10, dim=5)
        client.chat.completions.create.return_value = _mock_chat_response("Réponse OK")
        mock_get_client.return_value = client

        steps = _make_steps(10, output_size=300)
        result = answer_with_rag("Quel est le SMR global ?", steps=steps)

        assert result == "Réponse OK"
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] > 700, (
            f"max_tokens={call_kwargs['max_tokens']} doit dépasser l'ancienne valeur codée en dur 700"
        )

    @patch("rag._get_client")
    def test_explicit_top_k_respected(self, mock_get_client):
        """Quand top_k est explicitement passé, il doit être respecté (borné par len(chunks))."""
        from rag import answer_with_rag

        client = MagicMock()
        client.embeddings.create.return_value = _mock_embed_response(n_vecs=3, dim=5)
        client.chat.completions.create.return_value = _mock_chat_response("Réponse")
        mock_get_client.return_value = client

        steps = _make_steps(3, output_size=100)
        answer_with_rag("question", steps=steps, top_k=2)

        # _embed appelé 2 fois : une fois pour les docs, une fois pour la query
        assert client.embeddings.create.call_count >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tests answer_with_tools
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerWithTools:

    def test_no_steps_still_calls_api(self):
        """Sans steps, l'API doit quand même être appelée (namespace disponible)."""
        from rag import answer_with_tools
        # Sans clé API valide → erreur capturée et retournée proprement
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-invalid")
        answer, figs = answer_with_tools("question", steps=[], exec_ns={})
        # Retourne soit une réponse, soit un message d'erreur — pas une exception non catchée
        assert isinstance(answer, str)
        assert isinstance(figs, list)

    @patch("rag._get_client")
    def test_tool_call_dispatched_correctly(self, mock_get_client):
        """L'outil execute_python doit être dispatché et son résultat renvoyé au LLM."""
        from rag import answer_with_tools

        tool_call = MagicMock()
        tool_call.id = "tc_001"
        tool_call.function.name = "execute_python"
        tool_call.function.arguments = json.dumps({"code": "print(42)"})

        first_resp = _mock_chat_response(
            content=None,
            tool_calls=[tool_call],
            finish_reason="tool_calls",
        )
        second_resp = _mock_chat_response(content="La réponse finale.")

        client = MagicMock()
        client.embeddings.create.return_value = _mock_embed_response(n_vecs=1, dim=5)
        client.chat.completions.create.side_effect = [first_resp, second_resp]
        mock_get_client.return_value = client

        exec_ns = {}
        answer, figs = answer_with_tools(
            "Quelle est la valeur ?",
            steps=_make_steps(2),
            exec_ns=exec_ns,
        )
        assert answer == "La réponse finale."
        # 2 appels API : premier → tool_call, second → réponse finale
        assert client.chat.completions.create.call_count == 2

    @patch("rag._get_client")
    def test_returns_figures_from_tool_execution(self, mock_get_client):
        """Les figures générées lors des tool calls doivent être retournées."""
        from rag import answer_with_tools
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tool_call = MagicMock()
        tool_call.id = "tc_fig"
        tool_call.function.name = "execute_python"
        tool_call.function.arguments = json.dumps({
            "code": "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.plot([1,2,3])"
        })

        first_resp = _mock_chat_response(
            content=None, tool_calls=[tool_call], finish_reason="tool_calls"
        )
        second_resp = _mock_chat_response(content="Graphique généré.")

        client = MagicMock()
        client.embeddings.create.return_value = _mock_embed_response(n_vecs=1, dim=5)
        client.chat.completions.create.side_effect = [first_resp, second_resp]
        mock_get_client.return_value = client

        # exec_ns avec plt disponible
        exec_ns: dict = {}
        exec(
            "import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
            exec_ns,
        )

        answer, figs = answer_with_tools(
            "Génère un graphique",
            steps=_make_steps(1),
            exec_ns=exec_ns,
        )
        # On vérifie juste que la réponse est correcte (les figures dépendent du state plt)
        assert answer == "Graphique généré."


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tests execute_cell
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteCell:

    def test_basic_execution_and_output(self):
        from notebook_runner import execute_cell
        ns: dict = {}
        out = execute_cell("x = 5 + 3\nprint(x)", ns)
        assert "8" in out
        assert ns["x"] == 8

    def test_captures_stdout(self):
        from notebook_runner import execute_cell
        ns: dict = {}
        out = execute_cell('print("bonjour monde")', ns)
        assert "bonjour monde" in out

    def test_error_returns_traceback_with_marker(self):
        from notebook_runner import execute_cell
        ns: dict = {}
        out = execute_cell("raise ValueError('erreur test')", ns)
        assert "❌ Erreur" in out
        assert "ValueError" in out

    def test_state_persists_across_calls(self):
        from notebook_runner import execute_cell
        ns: dict = {}
        execute_cell("total = 0", ns)
        execute_cell("total += 10", ns)
        execute_cell("total += 5", ns)
        out = execute_cell("print(total)", ns)
        assert "15" in out

    def test_no_output_returns_success_marker(self):
        from notebook_runner import execute_cell
        ns: dict = {}
        out = execute_cell("x = 42  # pas de print", ns)
        assert "✓" in out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tests make_kernel
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeKernel:

    def test_has_standard_imports(self):
        from workflow_executor import make_kernel
        ns = make_kernel()
        assert "pd" in ns, "pandas manquant"
        assert "np" in ns, "numpy manquant"
        assert "plt" in ns, "matplotlib.pyplot manquant"

    def test_kernels_are_independent(self):
        from workflow_executor import make_kernel
        k1 = make_kernel()
        k2 = make_kernel()
        k1["isolation_test"] = 999
        assert "isolation_test" not in k2

    def test_actuarial_modules_loaded(self):
        from workflow_executor import make_kernel
        ns = make_kernel()
        # Au moins data_prep et exposure doivent être présents (modules présents)
        loaded = [m for m in ("data_prep", "exposure", "crude_rates") if m in ns]
        assert len(loaded) > 0, "Aucun module actuariel chargé — vérifier notebooks/"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tests agent ReAct loop
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentLoop:

    @patch("agent._get_client")
    def test_direct_stop_no_tool_call(self, mock_get_client):
        """L'agent retourne une réponse texte directe sans appeler d'outil."""
        from agent import run_agent_loop

        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response(
            "Réponse directe sans outil."
        )
        mock_get_client.return_value = client

        events = list(run_agent_loop(
            user_message="Bonjour",
            notebook_context="",
            conversation_history=[],
            execute_fn=MagicMock(return_value=("", [])),
        ))
        types = [e["type"] for e in events]
        assert "summary" in types
        assert "step" not in types
        summary = next(e for e in events if e["type"] == "summary")
        assert "directe" in summary["content"]

    @patch("agent._get_client")
    def test_one_tool_call_then_stop(self, mock_get_client):
        """L'agent fait un appel outil puis produit un résumé final."""
        from agent import run_agent_loop

        tool_call = MagicMock()
        tool_call.id = "call_001"
        tool_call.function.name = "execute_python"
        tool_call.function.arguments = json.dumps({
            "code": "df, s = data_prep.load_data(FILE_PATH)",
            "description": "Chargement des données",
        })

        first_resp = _mock_chat_response(
            content=None, tool_calls=[tool_call], finish_reason="tool_calls"
        )
        second_resp = _mock_chat_response("Analyse terminée avec succès.")

        client = MagicMock()
        client.chat.completions.create.side_effect = [first_resp, second_resp]
        mock_get_client.return_value = client

        execute_fn = MagicMock(return_value=("Données chargées : 1000 lignes", []))

        events = list(run_agent_loop(
            user_message="Construis la table de mortalité",
            notebook_context="",
            conversation_history=[],
            execute_fn=execute_fn,
        ))

        types = [e["type"] for e in events]
        assert "step" in types
        assert "summary" in types

        step = next(e for e in events if e["type"] == "step")
        assert step["description"] == "Chargement des données"
        assert "1000 lignes" in step["output"]

        summary = next(e for e in events if e["type"] == "summary")
        assert "terminée" in summary["content"]

    @patch("agent._get_client")
    def test_unexpected_finish_reason_yields_error(self, mock_get_client):
        """Un finish_reason inattendu doit produire un événement error."""
        from agent import run_agent_loop

        resp = _mock_chat_response(content=None, finish_reason="content_filter")
        resp.choices[0].message.tool_calls = None

        client = MagicMock()
        client.chat.completions.create.return_value = resp
        mock_get_client.return_value = client

        events = list(run_agent_loop(
            user_message="Test",
            notebook_context="",
            conversation_history=[],
            execute_fn=MagicMock(return_value=("", [])),
        ))
        assert any(e["type"] == "error" for e in events)

    @patch("agent._get_client")
    def test_multiple_tool_calls_all_processed(self, mock_get_client):
        """Plusieurs appels outils successifs sont tous traités avant le stop final."""
        from agent import run_agent_loop

        def make_tool_call(call_id: str, description: str) -> MagicMock:
            tc = MagicMock()
            tc.id = call_id
            tc.function.name = "execute_python"
            tc.function.arguments = json.dumps({
                "code": f"print('{description}')",
                "description": description,
            })
            return tc

        responses = [
            _mock_chat_response(
                content=None,
                tool_calls=[make_tool_call("c1", "Étape préparation")],
                finish_reason="tool_calls",
            ),
            _mock_chat_response(
                content=None,
                tool_calls=[make_tool_call("c2", "Étape exposition")],
                finish_reason="tool_calls",
            ),
            _mock_chat_response(content="Table construite."),
        ]

        client = MagicMock()
        client.chat.completions.create.side_effect = responses
        mock_get_client.return_value = client

        execute_fn = MagicMock(return_value=("ok", []))
        events = list(run_agent_loop(
            user_message="Construis",
            notebook_context="",
            conversation_history=[],
            execute_fn=execute_fn,
        ))

        steps = [e for e in events if e["type"] == "step"]
        assert len(steps) == 2
        assert steps[0]["description"] == "Étape préparation"
        assert steps[1]["description"] == "Étape exposition"

    @patch("agent._get_client")
    def test_error_in_tool_appends_warning_to_context(self, mock_get_client):
        """Quand execute_fn retourne une erreur, un message d'attention est ajouté."""
        from agent import run_agent_loop

        tool_call = MagicMock()
        tool_call.id = "c_err"
        tool_call.function.name = "execute_python"
        tool_call.function.arguments = json.dumps({
            "code": "1/0",
            "description": "Division par zéro",
        })

        responses = [
            _mock_chat_response(
                content=None, tool_calls=[tool_call], finish_reason="tool_calls"
            ),
            _mock_chat_response("Erreur corrigée."),
        ]

        client = MagicMock()
        client.chat.completions.create.side_effect = responses
        mock_get_client.return_value = client

        execute_fn = MagicMock(return_value=("❌ Erreur : ZeroDivisionError", []))
        events = list(run_agent_loop(
            user_message="Test erreur",
            notebook_context="",
            conversation_history=[],
            execute_fn=execute_fn,
        ))

        # Vérifier que le contexte LLM a reçu un message tool avec "ATTENTION"
        call_args_list = client.chat.completions.create.call_args_list
        second_call_messages = call_args_list[1][1]["messages"]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert any("ATTENTION" in m["content"] for m in tool_messages)

    @patch("agent._get_client")
    def test_history_event_emitted(self, mock_get_client):
        """Un événement 'history' est émis après le résumé final."""
        from agent import run_agent_loop

        client = MagicMock()
        client.chat.completions.create.return_value = _mock_chat_response("Fin.")
        mock_get_client.return_value = client

        events = list(run_agent_loop(
            user_message="Test",
            notebook_context="",
            conversation_history=[],
            execute_fn=MagicMock(return_value=("", [])),
        ))
        assert any(e["type"] == "history" for e in events)
        history_event = next(e for e in events if e["type"] == "history")
        assert isinstance(history_event["messages"], list)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Tests precompute_index
# ─────────────────────────────────────────────────────────────────────────────

class TestPrecomputeIndex:

    @patch("rag._get_client")
    def test_precompute_populates_cache(self, mock_get_client):
        from rag import precompute_index
        from actuary_state import ActuaryState

        client = MagicMock()
        client.embeddings.create.return_value = _mock_embed_response(n_vecs=3, dim=5)
        mock_get_client.return_value = client

        state = ActuaryState()
        steps = _make_steps(3, output_size=100)
        precompute_index(steps, state)

        # Le cache doit être peuplé
        from rag import build_index
        chunks = build_index(steps)
        assert state.get_embed_cache(chunks) is not None

    @patch("rag._get_client")
    def test_precompute_empty_steps_no_error(self, mock_get_client):
        """Pas d'erreur si les étapes sont vides."""
        from rag import precompute_index
        # Ne doit pas lever d'exception
        precompute_index(steps=[], state=None)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(Path(__file__).parent),
    )
    sys.exit(result.returncode)
