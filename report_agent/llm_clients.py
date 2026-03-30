"""
report_agent/llm_clients.py
Adaptateurs LLM pour generate_report.py.

Interface requise :
    client.complete(system_prompt: str, user_message: str) -> str

Usage :
    from report_agent.llm_clients import ClaudeClient
    client = ClaudeClient()
    narratif_json = generate_mortality_report(payload, llm_client=client)
"""
from __future__ import annotations


class ClaudeClient:
    """Client Anthropic Claude via SDK officiel."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Le package 'anthropic' est requis. Installez-le avec : pip install anthropic"
            ) from e
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system_prompt: str, user_message: str) -> str:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return r.content[0].text


class OpenAIClient:
    """Client OpenAI via SDK officiel."""

    def __init__(self, model: str = "gpt-4o"):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "Le package 'openai' est requis. Installez-le avec : pip install openai"
            ) from e
        self.client = openai.OpenAI()
        self.model = model

    def complete(self, system_prompt: str, user_message: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            max_tokens=8000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return r.choices[0].message.content

    def complete_with_history(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 8000,
    ) -> str:
        """Appel multi-tours avec historique complet.

        Args:
            system_prompt: Instruction système (chargée depuis un fichier .md).
            messages:      Historique [{role: user/assistant, content: str}, ...].
            max_tokens:    Limite de tokens en sortie.

        Returns:
            Texte de la réponse du modèle.
        """
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)
        r = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        return r.choices[0].message.content
