"""
validator.py — validation contractuelle YAML (Design 3) ↔ registry des tools.

Consommé par :
- scripts/check_template.py (US-3) : gate CI bloquant.
- agents/mortality/agents/master_node.py (US-19) : preflight Phase 0 au boot.

API
---
validate_template(yaml_path, registry) -> ValidationReport

Checks (cf. ADR §Validation) :
  Bloquants :
    1. YAML parse
    2. produced_by.tool ∈ registry
    3. inputs keys ⊆ signature du tool (param names)
       inputs values ∈ data_store keys (session_inputs ∪ data_contract)
    4. output_mapping keys ⊆ tool.outputs
       key ∈ values(output_mapping) OU key ∈ tool.outputs si pas de mapping
    5. {{ placeholders }} résolvent contre session_inputs ∪ data_contract
    6. type: date dans un shape a un format
    7. type: enum a allowed
    8. Pas de cycle dans le DAG des produced_by
    9. dependencies: [...] pointe vers des sections existantes
   10. Unicité de production (une clé = un seul produced_by)
  Warning :
    - Clé déclarée mais jamais consommée (ni placeholder, ni input d'un autre produced_by)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Issue:
    severity: str   # "error" | "warning"
    location: str
    message: str


@dataclass
class ValidationReport:
    errors:   list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, location: str, message: str) -> None:
        self.errors.append(Issue("error", location, message))

    def add_warning(self, location: str, message: str) -> None:
        self.warnings.append(Issue("warning", location, message))


_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_DATA_CONTRACT_BLOCKS = ("master_from_data", "master_from_modeling", "builder_outputs")


# ───────────────── Entrée principale ─────────────────

def validate_template(
    yaml_path: Path | str,
    registry: dict[str, dict],
) -> ValidationReport:
    report = ValidationReport()
    path = Path(yaml_path)

    # Check 1 : parse
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        report.add_error(str(path), f"YAML parse error: {exc}")
        return report
    except FileNotFoundError:
        report.add_error(str(path), "fichier introuvable")
        return report

    if not isinstance(doc, dict):
        report.add_error(str(path), "document YAML racine doit être un mapping")
        return report

    # Index des clés produites + shape des session_inputs
    produced_keys = _index_produced_keys(doc, report)  # check 10 (unicité) + collecte
    session_keys  = _collect_session_keys(doc)
    all_data_keys = session_keys | set(produced_keys.keys())

    # Checks 6 & 7 : type:date/enum dans les shapes
    _check_shapes(doc, report)

    # Checks 2, 3, 4 : produced_by cohérent avec registry
    _check_produced_by(doc, registry, all_data_keys, report)

    # Check 5 : placeholders
    _check_placeholders(doc, all_data_keys, report)

    # Check 8 : pas de cycle
    _check_dag_cycles(doc, report)

    # Check 9 : dependencies
    _check_dependencies(doc, report)

    # Check 11 : activation (syntaxe + couverture enum)
    _validate_activation(doc, report)

    # Warning : clés jamais consommées
    _warn_unused_keys(doc, produced_keys, report)

    return report


# ───────────────── Check étendu (--check-columns) ─────────────────

def check_table_columns(
    yaml_path: Path | str,
    registry: dict[str, dict],
    samples_provider=None,
) -> ValidationReport:
    """Vérifie que chaque `columns[].key` d'un visual_spec table existe
    bien dans les records produits par le tool référencé via produced_by.

    Stratégie : on appelle le tool avec des fixtures minimales (via
    `samples_provider(tool_name) -> dict | None`, optionnel) ou on parse
    les noms de champs depuis la docstring `OUTPUTS.data_store_keys_written`
    du tool. Le second mode ne nécessite aucune exécution réelle.

    Retourne un ValidationReport. Erreur si une columns[].key n'existe
    pas dans les records de la source attendue.
    """
    report = ValidationReport()
    path = Path(yaml_path)

    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, FileNotFoundError) as exc:
        report.add_error(str(path), f"lecture YAML : {exc}")
        return report

    # Map : key produite → docstring fields (extraits de OUTPUTS.data_store_keys_written)
    # Format : "ci_table : list[dict] — {age, q_x_lisse, ci_lower, ci_upper}"
    # On extrait les noms entre { } ou les colonnes mentionnées.
    import re as _re
    keys_to_fields: dict[str, set[str]] = {}
    for block, idx, entry in _iter_produced(doc):
        pb = entry.get("produced_by") or {}
        tool_name = pb.get("tool")
        if not tool_name:
            continue
        tool_spec = registry.get(tool_name) or {}
        key_name = entry.get("key")
        if not key_name:
            continue
        # Read tool docstring OUTPUTS.data_store_keys_written for the key
        path_to_tool = tool_spec.get("path")
        if not path_to_tool:
            continue
        try:
            src = Path(path_to_tool).read_text(encoding="utf-8")
        except Exception:
            continue
        # Trouver la ligne du data_store_keys_written matchant la clé
        # Format attendu : `  - key_name : type — {field1, field2, ...}`
        # ou `  - key_name : type — texte mentionnant les champs`
        pat = _re.compile(
            r"^\s*-\s*" + _re.escape(key_name) + r"\s*:\s*[^—\n]+—\s*(.+)$",
            _re.MULTILINE,
        )
        match = pat.search(src)
        if not match:
            continue
        desc = match.group(1)
        # Extraire les champs entre {…}
        brace_match = _re.search(r"\{([^}]+)\}", desc)
        if brace_match:
            fields_raw = brace_match.group(1)
            fields = {f.strip() for f in fields_raw.split(",") if f.strip()}
        else:
            # Fallback : mots-clés style "age, q_x" dans la description
            fields = set(_re.findall(r"\b([a-z_][a-z0-9_]*)\b", desc.lower()))
        if fields:
            keys_to_fields[key_name] = fields

    # Pour chaque visual_spec table, vérifier que columns[].key ∈ keys_to_fields[source]
    for section in doc.get("sections") or []:
        if not isinstance(section, dict):
            continue
        sid = section.get("id", "?")
        for j, spec in enumerate(section.get("visual_specs") or []):
            if not isinstance(spec, dict) or spec.get("type") != "table":
                continue
            src = spec.get("source") or ""
            if not src:
                continue
            # Si la source est sub-pathed (ex. `exclusion_report.rules`), on
            # ne sait pas typer le sub-record précisément depuis la docstring
            # globale — on skip (best-effort).
            if "." in src:
                continue
            declared_fields = keys_to_fields.get(src)
            if not declared_fields:
                # Pas d'info de schéma — on ne peut pas vérifier
                continue
            for k, col in enumerate(spec.get("columns") or []):
                col_key = col.get("key")
                if not col_key:
                    continue
                if col_key not in declared_fields:
                    report.add_error(
                        f"sections.{sid}.visual_specs[{j}].columns[{k}]",
                        f"colonne '{col_key}' absente du schéma de '{base}' "
                        f"(fields déclarés dans le tool : {sorted(declared_fields)})",
                    )

    return report


# ───────────────── Helpers : collecte ─────────────────

def _data_contract(doc: dict) -> dict:
    return doc.get("data_contract") or {}


def _iter_produced(doc: dict):
    """Yield (block_name, idx, entry) pour chaque entrée de data_contract.*"""
    dc = _data_contract(doc)
    for block in _DATA_CONTRACT_BLOCKS:
        for i, entry in enumerate(dc.get(block) or []):
            if isinstance(entry, dict):
                yield block, i, entry


def _collect_session_keys(doc: dict) -> set[str]:
    result: set[str] = set()
    for entry in doc.get("session_inputs") or []:
        if isinstance(entry, dict) and "key" in entry:
            result.add(entry["key"])
    return result


def _index_produced_keys(doc: dict, report: ValidationReport) -> dict[str, tuple[str, int]]:
    """Retourne {key: (block, idx)}. Signale les doublons (check 10)."""
    seen: dict[str, tuple[str, int]] = {}
    for block, idx, entry in _iter_produced(doc):
        key = entry.get("key")
        if not key:
            report.add_error(f"data_contract.{block}[{idx}]", "entrée sans clé 'key'")
            continue
        if key in seen:
            prev_block, prev_idx = seen[key]
            report.add_error(
                f"data_contract.{block}[{idx}]",
                f"clé '{key}' duplicate (déjà déclarée dans {prev_block}[{prev_idx}])",
            )
        else:
            seen[key] = (block, idx)
    return seen


# ───────────────── Check 6 & 7 : shapes ─────────────────

def _iter_shape_items(doc: dict):
    """Yield (location, item_dict) pour chaque entrée de shape dans le YAML."""
    for entry in doc.get("session_inputs") or []:
        if isinstance(entry, dict) and isinstance(entry.get("shape"), list):
            for j, item in enumerate(entry["shape"]):
                if isinstance(item, dict):
                    yield f"session_inputs.{entry.get('key', '?')}.shape[{j}]", item
    for block, idx, entry in _iter_produced(doc):
        if isinstance(entry.get("shape"), list):
            for j, item in enumerate(entry["shape"]):
                if isinstance(item, dict):
                    yield f"data_contract.{block}[{idx}].shape[{j}]", item


def _check_shapes(doc: dict, report: ValidationReport) -> None:
    for loc, item in _iter_shape_items(doc):
        t = item.get("type")
        key = item.get("key", "?")
        if t == "date" and "format" not in item:
            report.add_error(loc, f"champ '{key}' type:date doit déclarer un 'format'")
        if t == "enum" and "allowed" not in item:
            report.add_error(loc, f"champ '{key}' type:enum doit déclarer 'allowed'")


# ───────────────── Check 2, 3, 4 : produced_by ─────────────────

def _check_produced_by(
    doc: dict,
    registry: dict[str, dict],
    all_data_keys: set[str],
    report: ValidationReport,
) -> None:
    for block, idx, entry in _iter_produced(doc):
        key = entry.get("key")
        loc = f"data_contract.{block}[{idx}]"
        pb = entry.get("produced_by")
        if not isinstance(pb, dict):
            report.add_error(loc, f"'{key}' sans bloc 'produced_by' valide")
            continue

        tool_name = pb.get("tool")
        if not tool_name:
            report.add_error(loc, f"'{key}.produced_by' sans 'tool'")
            continue

        tool_spec = registry.get(tool_name)
        if tool_spec is None:
            report.add_error(loc, f"tool inconnu du registry : '{tool_name}'")
            continue

        # Inputs : param names ⊆ signature ; valeurs ∈ data_store keys
        inputs = pb.get("inputs") or {}
        if not isinstance(inputs, dict):
            report.add_error(loc, f"'{key}.produced_by.inputs' doit être un mapping")
        else:
            tool_inputs = tool_spec.get("inputs") or {}
            for param, ref in inputs.items():
                if param not in tool_inputs:
                    report.add_error(
                        loc,
                        f"param '{param}' inconnu pour le tool '{tool_name}' "
                        f"(signature: {sorted(tool_inputs)})",
                    )
                if isinstance(ref, str) and ref not in all_data_keys:
                    report.add_error(
                        loc,
                        f"input '{param}' référence une clé inexistante : '{ref}'",
                    )

        # Output mapping
        tool_outputs = tool_spec.get("outputs") or {}
        mapping = pb.get("output_mapping")
        if mapping:
            if not isinstance(mapping, dict):
                report.add_error(loc, f"'{key}.produced_by.output_mapping' doit être un mapping")
            else:
                for tool_field, data_key in mapping.items():
                    if tool_field not in tool_outputs:
                        report.add_error(
                            loc,
                            f"output_mapping référence un champ inexistant pour '{tool_name}' : "
                            f"'{tool_field}' (outputs: {sorted(tool_outputs)})",
                        )
                if key not in mapping.values():
                    report.add_error(
                        loc,
                        f"la clé produite '{key}' n'apparaît pas dans output_mapping.values()",
                    )
        else:
            # Sans mapping : la clé doit exister telle quelle dans les outputs du tool
            if key not in tool_outputs:
                report.add_error(
                    loc,
                    f"clé '{key}' absente des outputs du tool '{tool_name}' "
                    f"et pas d'output_mapping défini",
                )


# ───────────────── Check 5 : placeholders ─────────────────

def _iter_placeholder_texts(doc: dict):
    """Yield (location, text) pour chaque zone qui accepte des placeholders."""
    for i, section in enumerate(doc.get("sections") or []):
        if not isinstance(section, dict):
            continue
        sid = section.get("id", f"[{i}]")
        narrative = section.get("narrative")
        if isinstance(narrative, dict):
            txt = narrative.get("text")
            if isinstance(txt, str):
                yield f"sections.{sid}.narrative.text", txt
        for j, spec in enumerate(section.get("visual_specs") or []):
            if not isinstance(spec, dict):
                continue
            for k, col in enumerate(spec.get("columns") or []):
                if isinstance(col, dict) and isinstance(col.get("label"), str):
                    yield (
                        f"sections.{sid}.visual_specs[{j}].columns[{k}].label",
                        col["label"],
                    )


def _check_placeholders(
    doc: dict,
    all_data_keys: set[str],
    report: ValidationReport,
) -> None:
    for loc, text in _iter_placeholder_texts(doc):
        for m in _PLACEHOLDER_RE.finditer(text):
            name = m.group(1)
            if name not in all_data_keys:
                report.add_error(
                    loc,
                    f"placeholder {{{{ {name} }}}} non résolu "
                    f"(pas dans session_inputs ∪ data_contract)",
                )


# ───────────────── Check 8 : cycles ─────────────────

def _check_dag_cycles(doc: dict, report: ValidationReport) -> None:
    """
    DAG : pour chaque clé produite, les clés référencées dans inputs sont
    des prédécesseurs. On détecte un cycle par DFS à trois états.
    """
    edges: dict[str, list[str]] = {}
    for _block, _idx, entry in _iter_produced(doc):
        key = entry.get("key")
        if not key:
            continue
        pb = entry.get("produced_by") or {}
        inputs = pb.get("inputs") or {}
        refs = [v for v in inputs.values() if isinstance(v, str)]
        edges.setdefault(key, []).extend(refs)

    WHITE, GREY, BLACK = 0, 1, 2
    color: dict[str, int] = {k: WHITE for k in edges}

    def dfs(node: str, path: list[str]) -> list[str] | None:
        color[node] = GREY
        for nxt in edges.get(node, []):
            if nxt not in color:
                continue  # référence à une session_input, pas un nœud du DAG
            if color[nxt] == GREY:
                return path + [node, nxt]
            if color[nxt] == WHITE:
                cycle = dfs(nxt, path + [node])
                if cycle:
                    return cycle
        color[node] = BLACK
        return None

    for node in list(edges):
        if color[node] == WHITE:
            cycle = dfs(node, [])
            if cycle:
                report.add_error(
                    "data_contract",
                    f"cycle détecté dans le DAG produced_by : {' → '.join(cycle)}",
                )
                return


# ───────────────── Check 9 : dependencies ─────────────────

def _check_dependencies(doc: dict, report: ValidationReport) -> None:
    sections = doc.get("sections") or []
    ids = {s.get("id") for s in sections if isinstance(s, dict)}
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        sid = section.get("id", f"[{i}]")
        deps = section.get("dependencies") or []
        if not isinstance(deps, list):
            report.add_error(f"sections.{sid}.dependencies", "doit être une liste")
            continue
        for d in deps:
            if d not in ids:
                report.add_error(
                    f"sections.{sid}.dependencies",
                    f"section inconnue : '{d}'",
                )


# ───────────────── Check 11 : activation ─────────────────

def _validate_activation(template: dict, errors: ValidationReport) -> None:
    """Vérifie la syntaxe et la couverture d'enum des champs `activation`.

    Deux formats supportés :
      1. Ancien : {key: X, equals: Y}
      2. Nouveau : {field1: [values...], field2: [values...]}  (AND implicite)

    Pour la couverture d'enum, seuls les champs présents dans les enums
    master_* sont vérifiés (les autres sont ignorés — couvre par exemple le
    cas où `report_mode` est un champ logique géré par Master et non déclaré
    comme enum dans le data_contract).
    """
    # Index des enums dans master_from_data + master_from_modeling
    enums: dict[str, list[str]] = {}
    for group in ("master_from_data", "master_from_modeling"):
        for entry in template.get("data_contract", {}).get(group) or []:
            if isinstance(entry, dict) and entry.get("type") == "enum":
                enums[entry["key"]] = entry.get("allowed") or []

    # Collecter les activations par clé d'enum référencée
    covered: dict[str, set[str]] = {}
    for section in template.get("sections") or []:
        if not isinstance(section, dict):
            continue
        act = section.get("activation")
        if act is None:
            continue
        sid = section.get("id", "?")
        if not isinstance(act, dict):
            errors.add_error(
                f"sections.{sid}.activation",
                f"Section {sid} : activation doit être un dict",
            )
            continue

        # Format ancien
        if "key" in act and "equals" in act:
            key = act["key"]
            val = act["equals"]
            if key not in enums:
                errors.add_error(
                    f"sections.{sid}.activation",
                    f"Section {sid} : activation.key '{key}' absent des enums master_*",
                )
                continue
            if val not in enums[key]:
                errors.add_error(
                    f"sections.{sid}.activation",
                    f"Section {sid} : activation.equals '{val}' absent de allowed={enums[key]}",
                )
                continue
            covered.setdefault(key, set()).add(val)
            continue

        # Format nouveau : dict {field: [values]} — on vérifie chaque champ enum
        for field, values in act.items():
            if field not in enums:
                # Champ logique non déclaré comme enum (ex: report_mode géré
                # par le Master). On tolère.
                continue
            allowed_values = values if isinstance(values, list) else [values]
            for v in allowed_values:
                if v not in enums[field]:
                    errors.add_error(
                        f"sections.{sid}.activation",
                        f"Section {sid} : activation.{field} contient '{v}' "
                        f"absent de allowed={enums[field]}",
                    )
                    break
            else:
                covered.setdefault(field, set()).update(allowed_values)

    for key, seen in covered.items():
        missing = set(enums[key]) - seen
        if missing:
            errors.add_error(
                "data_contract",
                f"Enum '{key}' : valeurs sans section activable {sorted(missing)}",
            )


# ───────────────── Warning : clés jamais consommées ─────────────────

def _warn_unused_keys(
    doc: dict,
    produced_keys: dict[str, tuple[str, int]],
    report: ValidationReport,
) -> None:
    consumed: set[str] = set()

    # Placeholders
    for _loc, text in _iter_placeholder_texts(doc):
        consumed.update(_PLACEHOLDER_RE.findall(text))

    # Inputs d'autres produced_by
    for _block, _idx, entry in _iter_produced(doc):
        pb = entry.get("produced_by") or {}
        inputs = pb.get("inputs") or {}
        for v in inputs.values():
            if isinstance(v, str):
                consumed.add(v)

    # visual_specs.source (support sub-paths like "exclusion_report.rules")
    # + multi-series charts qui déclarent une source par série.
    for section in doc.get("sections") or []:
        if not isinstance(section, dict):
            continue
        for spec in section.get("visual_specs") or []:
            if not isinstance(spec, dict):
                continue
            # Source globale (mono-source)
            src = spec.get("source")
            if isinstance(src, str):
                consumed.add(src)
                base = src.split(".")[0]
                if base != src:
                    consumed.add(base)
            # Sources par série (multi_series)
            for s in (spec.get("series") or []):
                if isinstance(s, dict):
                    ssrc = s.get("source")
                    if isinstance(ssrc, str):
                        consumed.add(ssrc)
                        sbase = ssrc.split(".")[0]
                        if sbase != ssrc:
                            consumed.add(sbase)

    for key, (block, idx) in produced_keys.items():
        if key not in consumed:
            report.add_warning(
                f"data_contract.{block}[{idx}]",
                f"clé '{key}' déclarée mais jamais consommée",
            )
