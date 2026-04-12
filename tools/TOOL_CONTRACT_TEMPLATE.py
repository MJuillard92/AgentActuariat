"""
TOOL CONTRACT — [tool_name]
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : [domain].[function_name]
domain        : [mortality_experience | longevity | morbidity | descriptive]
version       : 1.0.0
author        : [name]
last_updated  : [YYYY-MM-DD]

DESCRIPTION
-----------
[What this tool does in 2-3 sentences. Plain language.]

WHEN TO USE
-----------
[Conditions under which the agent should call this tool.]

WHEN NOT TO USE
---------------
[Conditions where this tool should NOT be called, even if relevant.]

PREREQUISITES
-------------
required_tools:
  - [tool_name] → provides [key]
required_data_store_keys:
  - [key_name]

INPUTS
------
params:
  [param_name]:
    type    : [string | int | float | bool]
    values  : [accepted values]
    default : [default value]
    note    : [guidance for the agent on how to choose this value]

OUTPUTS
-------
data_store_keys_written:
  - [key] : [type and description]
return_payload:
  [field] : [type and description]

QUALITY GATES
-------------
BLOCKING:
  - [condition] → [what the agent must do before proceeding]
NON-BLOCKING:
  - [condition] → [what the agent must document]

ERROR HANDLING
--------------
error: "[error message as returned by the tool]"
  → cause  : [why this happens]
  → action : [what the agent should do — never retry identically]

AGENT GUIDANCE
--------------
reasoning_hint: >
  [Free text guidance for the agent's reasoning.]
exemplar_query: >
  [Query string for the exemplar RAG when facing a judgment decision.]

CATALOGUE METADATA
------------------
display_name      : [Human-readable name]
short_description : [One sentence for the catalogue]
domain            : [domain]
capability_group  : [table_construction | descriptive | reporting | graphs]
depends_on        : [[tool1, tool2]]
required_by       : [[tool3, tool4]]
client_visible    : [true | false]
"""

# ── Stub implementation ───────────────────────────────────────────────────────


def run(data, params=None):
    """
    Stub function — replace with actual implementation.

    Args:
        data   : dict | pd.DataFrame — input data or data_store
        params : dict | None         — tool parameters

    Returns:
        dict with result keys, or {"erreur": "..."} on failure
    """
    raise NotImplementedError("Replace this stub with the actual tool implementation.")
