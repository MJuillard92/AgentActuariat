from __future__ import annotations

import json
import os
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

from offline.models import TextBlockArtifact

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"

try:
    import config

    DEFAULT_MODEL = getattr(config, "FORMATTER_MODEL", DEFAULT_MODEL)
except Exception:
    pass


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY manquante dans .env")
    return OpenAI(api_key=api_key)


def _batched(items: list[TextBlockArtifact], batch_size: int) -> Iterable[list[TextBlockArtifact]]:
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def _extract_json_payload(raw_text: str) -> list[dict]:
    raw_text = raw_text.strip()
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Réponse JSON introuvable dans la sortie du modèle.")
    return json.loads(raw_text[start:end + 1])


def describe_text_blocks(
    text_blocks: list[TextBlockArtifact],
    section_titles: dict[str, str],
    model: str | None = None,
    batch_size: int = 8,
) -> dict[str, str]:
    if not text_blocks:
        return {}

    client = _get_client()
    model_name = model or DEFAULT_MODEL
    results: dict[str, str] = {}

    for batch in _batched(text_blocks, batch_size=batch_size):
        payload = [
            {
                "block_id": text_block.block_id,
                "section_title": section_titles.get(text_block.section_id, "Section inconnue"),
                "page_number": text_block.page_number,
                "text": text_block.text,
            }
            for text_block in batch
        ]

        response = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu aides à construire un blueprint de rédaction à partir d'un PDF. "
                        "Pour chaque bloc de texte, produis une courte description en français, "
                        "orientée réutilisation par un agent. "
                        "La description doit expliquer le rôle analytique du bloc, "
                        "pas simplement le paraphraser. Réponds uniquement en JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Retourne un tableau JSON. Chaque objet doit avoir exactement les clés "
                        "`block_id` et `description`.\n\n"
                        f"Blocs à décrire :\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
                    ),
                },
            ],
            max_tokens=1800,
        )

        content = response.choices[0].message.content or "[]"
        parsed = _extract_json_payload(content)
        for item in parsed:
            block_id = item.get("block_id")
            description = (item.get("description") or "").strip()
            if block_id and description:
                results[block_id] = description

    return results


def describe_single_text_block(
    text_block: TextBlockArtifact,
    section_title: str,
    model: str | None = None,
) -> str:
    descriptions = describe_text_blocks(
        text_blocks=[text_block],
        section_titles={text_block.section_id: section_title},
        model=model,
        batch_size=1,
    )
    return descriptions.get(text_block.block_id, "")
