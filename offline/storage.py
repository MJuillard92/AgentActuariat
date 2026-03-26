from __future__ import annotations

import json
from pathlib import Path

from offline.models import ParsedDocument


def save_review_state(parsed_document: ParsedDocument, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "document": parsed_document.to_dict(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def load_review_state(input_path: str) -> ParsedDocument:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return ParsedDocument.from_dict(payload["document"])

