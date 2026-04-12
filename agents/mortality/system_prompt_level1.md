# Actuarial Agent — System Prompt (Level 1)
# This file is the architecture map. All instruction content lives in
# agent_instructions/. Edit instruction files directly — not this file.
# loader.py assembles the full prompt from this map at runtime.

## [INJECT] Behavioural contract
source: agent_instructions/behavioral_contract.md

## [INJECT] Tool catalogue
source: tools/catalogue.yaml
format: yaml_block

## [INJECT] Étape 0 — Validation du dictionnaire de données
source: agent_instructions/step0_data_dictionary.md

## [INJECT] Phase de planification obligatoire
source: agent_instructions/step1_planning.md

## [INJECT] Point de contrôle mid-task
source: agent_instructions/step2_mid_task_checkpoint.md

## [INJECT] Communication du plan au client
source: agent_instructions/step3_client_communication.md

## [INJECT] Replay de session
source: agent_instructions/step6_replay.md
