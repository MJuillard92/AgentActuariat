from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from offline.blueprint_builder import build_blueprint, save_blueprint
from offline.llm_guidance import DEFAULT_MODEL, build_agent_guidance, describe_section
from offline.models import ParsedDocument, SectionArtifact
from offline.pdf_parser import parse_pdf
from offline.storage import load_review_state, save_review_state

OUTPUTS_DIR = ROOT_DIR / "offline" / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"

SECTION_LIST_FIELDS = (
    "table_roles",
    "figure_roles",
    "narrative_guidance",
    "expected_outputs",
    "expected_tables",
    "expected_figures",
)
SECTION_TEXT_FIELDS = ("purpose", "analysis_logic", "agent_guidance")
GLOBAL_TEXT_FIELDS = ("report_type", "objective", "audience", "agent_prompt")
GLOBAL_LIST_FIELDS = ("ordered_sections", "global_narrative_rules")


def _init_state() -> None:
    defaults = {
        "offline_document": None,
        "offline_source_path": "",
        "offline_last_error": "",
        "offline_blueprint": None,
        "offline_loaded_review_path": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _available_pdf_files() -> list[str]:
    candidates = []
    for folder in (ROOT_DIR / "Portefeuille", ROOT_DIR / "uploads", ROOT_DIR):
        if folder.exists():
            candidates.extend(str(path) for path in sorted(folder.glob("*.pdf")))
    return candidates


def _available_review_files() -> list[str]:
    if not OUTPUTS_DIR.exists():
        return []
    return [str(path) for path in sorted(OUTPUTS_DIR.glob("*_review_state.json"))]


def _clear_review_widget_state() -> None:
    keys_to_delete = [
        key
        for key in list(st.session_state.keys())
        if key.startswith("section_") or key.startswith("global_")
    ]
    for key in keys_to_delete:
        del st.session_state[key]


def _load_document(path: str) -> None:
    _clear_review_widget_state()
    st.session_state.offline_document = parse_pdf(path)
    st.session_state.offline_source_path = path
    st.session_state.offline_blueprint = None
    st.session_state.offline_last_error = ""
    st.session_state.offline_loaded_review_path = ""


def _load_review(path: str) -> None:
    _clear_review_widget_state()
    st.session_state.offline_document = load_review_state(path)
    st.session_state.offline_source_path = st.session_state.offline_document.source_path
    st.session_state.offline_blueprint = None
    st.session_state.offline_last_error = ""
    st.session_state.offline_loaded_review_path = path


def _persist_uploaded_file(uploaded_file) -> str:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getvalue())
    return str(destination)


def _section_keep_key(section_id: str) -> str:
    return f"section_keep_{section_id}"


def _section_note_key(section_id: str) -> str:
    return f"section_note_{section_id}"


def _section_field_key(section_id: str, field_name: str) -> str:
    return f"section_field_{section_id}_{field_name}"


def _global_field_key(field_name: str) -> str:
    return f"global_field_{field_name}"


def _list_to_text(items) -> str:
    if not items:
        return ""
    if isinstance(items, str):
        return items
    return "\n".join(str(item) for item in items if str(item).strip())


def _text_to_list(value: str) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def _section_has_analysis(section: SectionArtifact) -> bool:
    analysis = section.analysis or {}
    return any((analysis.get(field) or "") for field in SECTION_TEXT_FIELDS + SECTION_LIST_FIELDS)


def _sync_document_from_state(parsed_document: ParsedDocument) -> None:
    for section in parsed_document.sections:
        keep_key = _section_keep_key(section.section_id)
        note_key = _section_note_key(section.section_id)
        if keep_key in st.session_state:
            section.keep = bool(st.session_state[keep_key])
        if note_key in st.session_state:
            section.description = st.session_state[note_key].strip()

        analysis = dict(section.analysis or {})
        for field in SECTION_TEXT_FIELDS:
            widget_key = _section_field_key(section.section_id, field)
            if widget_key in st.session_state:
                analysis[field] = st.session_state[widget_key].strip()
        for field in SECTION_LIST_FIELDS:
            widget_key = _section_field_key(section.section_id, field)
            if widget_key in st.session_state:
                analysis[field] = _text_to_list(st.session_state[widget_key])
        section.analysis = analysis

    global_guidance = dict(parsed_document.metadata.get("global_guidance", {}))
    for field in GLOBAL_TEXT_FIELDS:
        widget_key = _global_field_key(field)
        if widget_key in st.session_state:
            global_guidance[field] = st.session_state[widget_key].strip()
    for field in GLOBAL_LIST_FIELDS:
        widget_key = _global_field_key(field)
        if widget_key in st.session_state:
            global_guidance[field] = _text_to_list(st.session_state[widget_key])
    parsed_document.metadata["global_guidance"] = global_guidance


def _seed_section_widgets(section: SectionArtifact) -> None:
    keep_key = _section_keep_key(section.section_id)
    note_key = _section_note_key(section.section_id)
    if keep_key not in st.session_state:
        st.session_state[keep_key] = section.keep
    if note_key not in st.session_state:
        st.session_state[note_key] = section.description

    analysis = section.analysis or {}
    for field in SECTION_TEXT_FIELDS:
        widget_key = _section_field_key(section.section_id, field)
        if widget_key not in st.session_state:
            st.session_state[widget_key] = analysis.get(field, "")
    for field in SECTION_LIST_FIELDS:
        widget_key = _section_field_key(section.section_id, field)
        if widget_key not in st.session_state:
            st.session_state[widget_key] = _list_to_text(analysis.get(field, []))


def _seed_global_widgets(parsed_document: ParsedDocument) -> None:
    global_guidance = parsed_document.metadata.get("global_guidance", {})
    for field in GLOBAL_TEXT_FIELDS:
        widget_key = _global_field_key(field)
        if widget_key not in st.session_state:
            st.session_state[widget_key] = global_guidance.get(field, "")
    for field in GLOBAL_LIST_FIELDS:
        widget_key = _global_field_key(field)
        if widget_key not in st.session_state:
            st.session_state[widget_key] = _list_to_text(global_guidance.get(field, []))


def _apply_section_analysis(section: SectionArtifact, analysis: dict) -> None:
    section.analysis = analysis
    for field in SECTION_TEXT_FIELDS:
        st.session_state[_section_field_key(section.section_id, field)] = analysis.get(field, "")
    for field in SECTION_LIST_FIELDS:
        st.session_state[_section_field_key(section.section_id, field)] = _list_to_text(analysis.get(field, []))


def _apply_global_guidance(parsed_document: ParsedDocument, guidance: dict) -> None:
    parsed_document.metadata["global_guidance"] = guidance
    for field in GLOBAL_TEXT_FIELDS:
        st.session_state[_global_field_key(field)] = guidance.get(field, "")
    for field in GLOBAL_LIST_FIELDS:
        st.session_state[_global_field_key(field)] = _list_to_text(guidance.get(field, []))


def _generate_section_analyses(parsed_document: ParsedDocument, model_name: str, only_missing: bool = False) -> None:
    _sync_document_from_state(parsed_document)
    target_sections = [
        section
        for section in parsed_document.sections
        if section.keep and (not only_missing or not _section_has_analysis(section))
    ]
    for section in target_sections:
        analysis = describe_section(parsed_document=parsed_document, section=section, model=model_name)
        _apply_section_analysis(section, analysis)


def _overview_dataframe(parsed_document: ParsedDocument) -> pd.DataFrame:
    rows = []
    for section in parsed_document.sections:
        rows.append({
            "numero": section.section_number,
            "section": section.title,
            "niveau": section.level,
            "page_debut": section.page_start,
            "page_fin": section.page_end,
            "incluse": "oui" if section.keep else "non",
            "decrite": "oui" if _section_has_analysis(section) else "non",
            "blocs": len(section.text_block_ids),
            "tableaux": len(section.table_ids),
            "figures": len(section.figure_ids),
        })
    return pd.DataFrame(rows)


def _live_blueprint(parsed_document: ParsedDocument) -> dict:
    _sync_document_from_state(parsed_document)
    return build_blueprint(parsed_document)


def _save_current_blueprint(parsed_document: ParsedDocument) -> dict:
    blueprint = _live_blueprint(parsed_document)
    output_path = OUTPUTS_DIR / f"{Path(parsed_document.filename).stem}_blueprint.json"
    save_blueprint(blueprint, str(output_path))
    st.session_state.offline_blueprint = blueprint
    return blueprint


def _bbox_caption(bbox) -> str:
    if not bbox:
        return "bbox indisponible"
    x0, y0, x1, y1 = bbox
    return f"x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}"


def _safe_table_preview(preview_rows: list[list[str]]) -> pd.DataFrame:
    if not preview_rows:
        return pd.DataFrame()

    normalized_rows = []
    max_len = max(len(row) for row in preview_rows)
    for row in preview_rows:
        normalized_rows.append(list(row) + [""] * (max_len - len(row)))

    if len(normalized_rows) == 1:
        columns = [f"col_{index + 1}" for index in range(max_len)]
        return pd.DataFrame(normalized_rows, columns=columns)

    raw_headers = normalized_rows[0]
    seen: dict[str, int] = {}
    headers = []
    for index, header in enumerate(raw_headers, start=1):
        base = (header or "").strip() or f"col_{index}"
        count = seen.get(base, 0)
        headers.append(base if count == 0 else f"{base}_{count + 1}")
        seen[base] = count + 1

    return pd.DataFrame(normalized_rows[1:], columns=headers)


def _render_section_structure(section: SectionArtifact, parsed_document: ParsedDocument) -> None:
    block_map = {block.block_id: block for block in parsed_document.text_blocks}
    table_map = {table.table_id: table for table in parsed_document.tables}
    figure_map = {figure.figure_id: figure for figure in parsed_document.figures}

    with st.expander("Structure extraite", expanded=False):
        if section.text_block_ids:
            st.markdown("**Blocs texte**")
            for block_id in section.text_block_ids:
                if block_id not in block_map:
                    continue
                block = block_map[block_id]
                st.caption(f"{block.block_id} | page {block.page_number} | {_bbox_caption(block.bbox)}")
                st.write(block.text)

        if section.table_ids:
            st.markdown("**Tableaux**")
            for table_id in section.table_ids:
                if table_id not in table_map:
                    continue
                table = table_map[table_id]
                st.caption(
                    f"{table.table_id} | page {table.page_number} | {table.n_rows} lignes x {table.n_cols} colonnes | {_bbox_caption(table.bbox)}"
                )
                preview = _safe_table_preview(table.preview_rows)
                if not preview.empty:
                    st.dataframe(preview, use_container_width=True, hide_index=True)

        if section.figure_ids:
            st.markdown("**Figures**")
            for figure_id in section.figure_ids:
                if figure_id not in figure_map:
                    continue
                figure = figure_map[figure_id]
                st.caption(f"{figure.figure_id} | page {figure.page_number} | {_bbox_caption(figure.bbox)}")


def _render_section(section: SectionArtifact, parsed_document: ParsedDocument, model_name: str) -> None:
    _seed_section_widgets(section)
    analysis = section.analysis or {}

    section_label = f"{section.section_number or '—'} | {section.title} | pages {section.page_start}-{section.page_end}"
    with st.expander(section_label, expanded=section.level <= 1):
        top1, top2, top3, top4 = st.columns([1, 1, 1, 1.5])
        top1.metric("Blocs", len(section.text_block_ids))
        top2.metric("Tableaux", len(section.table_ids))
        top3.metric("Figures", len(section.figure_ids))
        top4.metric("Analyse LLM", "faite" if _section_has_analysis(section) else "à faire")

        st.checkbox(
            "Inclure cette section dans l'analyse et le blueprint",
            value=st.session_state[_section_keep_key(section.section_id)],
            key=_section_keep_key(section.section_id),
        )
        section.keep = bool(st.session_state[_section_keep_key(section.section_id)])
        if not section.keep:
            st.warning("Cette section sera exclue du blueprint et du prompt agent.")

        action1, action2 = st.columns([1, 1])
        if action1.button("Décrire cette section avec le LLM", key=f"describe_section_{section.section_id}", use_container_width=True):
            analysis = describe_section(parsed_document=parsed_document, section=section, model=model_name)
            _apply_section_analysis(section, analysis)
        if action2.button("Vider l'analyse de section", key=f"clear_section_{section.section_id}", use_container_width=True):
            _apply_section_analysis(section, {})

        st.text_area(
            "Note libre de section",
            value=st.session_state[_section_note_key(section.section_id)],
            key=_section_note_key(section.section_id),
            height=80,
            placeholder="Ajoutez ici une remarque métier ou une consigne spécifique.",
        )
        section.description = st.session_state[_section_note_key(section.section_id)].strip()

        purpose_col, logic_col = st.columns(2)
        purpose_col.text_input(
            "Purpose",
            value=st.session_state[_section_field_key(section.section_id, "purpose")],
            key=_section_field_key(section.section_id, "purpose"),
        )
        logic_col.text_area(
            "Analysis logic",
            value=st.session_state[_section_field_key(section.section_id, "analysis_logic")],
            key=_section_field_key(section.section_id, "analysis_logic"),
            height=120,
        )

        list_cols_1 = st.columns(2)
        list_cols_1[0].text_area(
            "Table roles",
            value=st.session_state[_section_field_key(section.section_id, "table_roles")],
            key=_section_field_key(section.section_id, "table_roles"),
            height=120,
            placeholder="Une ligne par rôle de tableau",
        )
        list_cols_1[1].text_area(
            "Figure roles",
            value=st.session_state[_section_field_key(section.section_id, "figure_roles")],
            key=_section_field_key(section.section_id, "figure_roles"),
            height=120,
            placeholder="Une ligne par rôle de figure",
        )

        list_cols_2 = st.columns(2)
        list_cols_2[0].text_area(
            "Narrative guidance",
            value=st.session_state[_section_field_key(section.section_id, "narrative_guidance")],
            key=_section_field_key(section.section_id, "narrative_guidance"),
            height=140,
            placeholder="Une ligne par guideline narrative",
        )
        list_cols_2[1].text_area(
            "Expected outputs",
            value=st.session_state[_section_field_key(section.section_id, "expected_outputs")],
            key=_section_field_key(section.section_id, "expected_outputs"),
            height=140,
            placeholder="Une ligne par output attendu",
        )

        list_cols_3 = st.columns(2)
        list_cols_3[0].text_area(
            "Expected tables",
            value=st.session_state[_section_field_key(section.section_id, "expected_tables")],
            key=_section_field_key(section.section_id, "expected_tables"),
            height=120,
            placeholder="Une ligne par tableau attendu",
        )
        list_cols_3[1].text_area(
            "Expected figures",
            value=st.session_state[_section_field_key(section.section_id, "expected_figures")],
            key=_section_field_key(section.section_id, "expected_figures"),
            height=120,
            placeholder="Une ligne par figure attendue",
        )

        st.text_area(
            "Agent guidance for this section",
            value=st.session_state[_section_field_key(section.section_id, "agent_guidance")],
            key=_section_field_key(section.section_id, "agent_guidance"),
            height=140,
        )

        _render_section_structure(section, parsed_document)


def main() -> None:
    st.set_page_config(page_title="Offline PDF Blueprint Builder", layout="wide")
    st.title("Offline PDF Blueprint Builder")
    st.caption("Charge un PDF, extrait sa structure, fait décrire chaque section par le LLM, puis construit un prompt de guidage pour un agent.")

    _init_state()

    with st.sidebar:
        st.subheader("Source PDF")
        local_files = _available_pdf_files()
        review_files = _available_review_files()

        selected_path = st.selectbox(
            "PDF local",
            options=[""] + local_files,
            index=0,
            format_func=lambda value: Path(value).name if value else "Choisir un .pdf",
        )
        selected_review = st.selectbox(
            "Ou reprendre une revue sauvegardée",
            options=[""] + review_files,
            index=0,
            format_func=lambda value: Path(value).name if value else "Choisir un état de revue",
        )
        uploaded_file = st.file_uploader("Ou déposer un PDF", type=["pdf"])

        if st.button("Charger le PDF", use_container_width=True):
            try:
                if uploaded_file is not None:
                    path = _persist_uploaded_file(uploaded_file)
                elif selected_path:
                    path = selected_path
                else:
                    raise ValueError("Sélectionnez un fichier PDF local ou déposez un document.")
                _load_document(path)
            except Exception as exc:
                st.session_state.offline_last_error = str(exc)

        if st.button("Charger l'état de revue", use_container_width=True):
            try:
                if not selected_review:
                    raise ValueError("Choisissez un état de revue JSON dans la liste.")
                _load_review(selected_review)
            except Exception as exc:
                st.session_state.offline_last_error = str(exc)

        st.divider()
        model_name = st.text_input("Modèle LLM", value=DEFAULT_MODEL)

        if st.button("Décrire toutes les sections incluses", use_container_width=True):
            try:
                parsed_document = st.session_state.offline_document
                if parsed_document is None:
                    raise ValueError("Chargez d'abord un PDF.")
                with st.spinner("Description des sections en cours..."):
                    _generate_section_analyses(parsed_document, model_name=model_name, only_missing=False)
            except Exception as exc:
                st.session_state.offline_last_error = str(exc)

        if st.button("Décrire seulement les sections manquantes", use_container_width=True):
            try:
                parsed_document = st.session_state.offline_document
                if parsed_document is None:
                    raise ValueError("Chargez d'abord un PDF.")
                with st.spinner("Description des sections manquantes en cours..."):
                    _generate_section_analyses(parsed_document, model_name=model_name, only_missing=True)
            except Exception as exc:
                st.session_state.offline_last_error = str(exc)

        if st.button("Construire le prompt agent global", use_container_width=True):
            try:
                parsed_document = st.session_state.offline_document
                if parsed_document is None:
                    raise ValueError("Chargez d'abord un PDF.")
                _sync_document_from_state(parsed_document)
                with st.spinner("Construction du prompt agent en cours..."):
                    guidance = build_agent_guidance(parsed_document=parsed_document, model=model_name)
                _apply_global_guidance(parsed_document, guidance)
            except Exception as exc:
                st.session_state.offline_last_error = str(exc)

        if st.button("Exporter le blueprint JSON", use_container_width=True):
            try:
                parsed_document = st.session_state.offline_document
                if parsed_document is None:
                    raise ValueError("Chargez d'abord un PDF.")
                _save_current_blueprint(parsed_document)
            except Exception as exc:
                st.session_state.offline_last_error = str(exc)

        if st.button("Sauvegarder l'état de revue", use_container_width=True):
            try:
                parsed_document = st.session_state.offline_document
                if parsed_document is None:
                    raise ValueError("Chargez d'abord un PDF.")
                _sync_document_from_state(parsed_document)
                review_path = OUTPUTS_DIR / f"{Path(parsed_document.filename).stem}_review_state.json"
                save_review_state(parsed_document, str(review_path))
                st.success(f"État sauvegardé dans {review_path}")
            except Exception as exc:
                st.session_state.offline_last_error = str(exc)

    if st.session_state.offline_last_error:
        st.error(st.session_state.offline_last_error)

    parsed_document = st.session_state.offline_document
    if parsed_document is None:
        st.info("Chargez un document `.pdf` depuis la barre latérale pour commencer.")
        return

    _seed_global_widgets(parsed_document)
    _sync_document_from_state(parsed_document)
    live_blueprint = _live_blueprint(parsed_document)
    global_guidance = parsed_document.metadata.get("global_guidance", {})

    metrics_1 = st.columns(5)
    metrics_1[0].metric("Pages", parsed_document.metadata.get("page_count", 0))
    metrics_1[1].metric("Sections", parsed_document.metadata.get("section_count", 0))
    metrics_1[2].metric("Sections incluses", live_blueprint["metadata"].get("kept_section_count", 0))
    metrics_1[3].metric("Sections décrites", sum(1 for section in parsed_document.sections if _section_has_analysis(section)))
    metrics_1[4].metric("Tableaux/Figures", f"{parsed_document.metadata.get('table_count', 0)} / {parsed_document.metadata.get('figure_count', 0)}")

    st.subheader(parsed_document.title)
    st.caption(parsed_document.source_path)
    if st.session_state.offline_loaded_review_path:
        st.caption(f"État de revue chargé : {st.session_state.offline_loaded_review_path}")

    overview_tab, review_tab, prompt_tab, blueprint_tab = st.tabs(
        ["Vue d'ensemble", "Revue des sections", "Prompt agent", "Aperçu blueprint"]
    )

    with overview_tab:
        st.markdown("**Structure détectée**")
        st.dataframe(_overview_dataframe(parsed_document), use_container_width=True, hide_index=True)

    with review_tab:
        status_filter = st.selectbox(
            "Afficher",
            options=["Toutes", "Inclues", "Exclues", "Décrites", "Non décrites"],
            index=0,
        )
        for section in parsed_document.sections:
            if status_filter == "Inclues" and not section.keep:
                continue
            if status_filter == "Exclues" and section.keep:
                continue
            if status_filter == "Décrites" and not _section_has_analysis(section):
                continue
            if status_filter == "Non décrites" and _section_has_analysis(section):
                continue
            _render_section(section, parsed_document, model_name=model_name)

    with prompt_tab:
        st.markdown("**Guidance globale pour l'agent**")
        for field in ("report_type", "objective", "audience"):
            st.text_input(
                field.replace("_", " ").title(),
                value=st.session_state[_global_field_key(field)],
                key=_global_field_key(field),
            )
        st.text_area(
            "Ordered sections",
            value=st.session_state[_global_field_key("ordered_sections")],
            key=_global_field_key("ordered_sections"),
            height=120,
            placeholder="Une ligne par section attendue",
        )
        st.text_area(
            "Global narrative rules",
            value=st.session_state[_global_field_key("global_narrative_rules")],
            key=_global_field_key("global_narrative_rules"),
            height=120,
            placeholder="Une ligne par règle narrative globale",
        )
        st.text_area(
            "Prompt final pour l'agent",
            value=st.session_state[_global_field_key("agent_prompt")],
            key=_global_field_key("agent_prompt"),
            height=260,
            placeholder="Le prompt de guidage final apparaîtra ici.",
        )
        _sync_document_from_state(parsed_document)
        if parsed_document.metadata.get("global_guidance", {}).get("agent_prompt"):
            st.download_button(
                "Télécharger le prompt agent",
                data=parsed_document.metadata["global_guidance"]["agent_prompt"].encode("utf-8"),
                file_name=f"{Path(parsed_document.filename).stem}_agent_prompt.txt",
                mime="text/plain",
            )

    with blueprint_tab:
        st.markdown("**Sections du blueprint**")
        for section in live_blueprint["sections"]:
            label = f"{section['section_number'] or '—'} | {section['title']} | pages {section['page_start']}-{section['page_end']}"
            with st.expander(label, expanded=section["level"] <= 1):
                st.write(section["purpose"])
                st.write(section["section_description"])
                if section["table_roles"]:
                    st.write("Rôles des tableaux :")
                    for item in section["table_roles"]:
                        st.write(f"- {item}")
                if section["figure_roles"]:
                    st.write("Rôles des figures :")
                    for item in section["figure_roles"]:
                        st.write(f"- {item}")
                if section["narrative_guidelines"]:
                    st.write("Guidelines narratives :")
                    for item in section["narrative_guidelines"]:
                        st.write(f"- {item}")
                if section["expected_outputs"]:
                    st.write("Outputs attendus :")
                    for item in section["expected_outputs"]:
                        st.write(f"- {item}")
                if section["agent_guidance"]:
                    st.write("Consigne agent :")
                    st.write(section["agent_guidance"])

        blueprint_json = json.dumps(live_blueprint, ensure_ascii=False, indent=2)
        st.download_button(
            "Télécharger le blueprint courant",
            data=blueprint_json.encode("utf-8"),
            file_name=f"{Path(parsed_document.filename).stem}_blueprint.json",
            mime="application/json",
        )
        st.code(blueprint_json, language="json")


if __name__ == "__main__":
    main()
