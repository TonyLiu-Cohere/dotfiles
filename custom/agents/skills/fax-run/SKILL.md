---
name: fax-run
description: Plans, validates, summarizes, and explicitly launches fax/tax pretraining, SFT, and RLVR runs. Use only inside a fax/tax repository or when the user explicitly identifies the work as fax/tax training; do not apply to other projects or generic training workflows.
---

# Fax Run Planning

Do not start or submit a run by default. Inspect the repository and give the
user the exact command to run themselves. If the user explicitly asks the agent
to start or submit the run, the agent may execute `make debug`, `make submit`,
or `workspace uv run ... --submit start` as appropriate.

## Before proposing a command

1. Identify the task: pretraining, SFT, or RLVR.
2. Read the launch file, referenced config, and the relevant `Makefile` target
   or Python argument parser. Do not guess flag names or default values.
3. Give this short run card before the command:

   ```text
   Run: <pretrain | SFT | RLVR>; <debug | submit>
   Model: <model/name/size>
   Batch size: <per-device × grad-accumulation × data-parallel = global, if available>
   Config: <path and important overrides>
   Data: <dataset/source, split, mixture, and size/filter if available>
   ```

   Mark unavailable values as `not found`, rather than inferring them.
4. State the command on one line and list any required working directory or
   environment prerequisite separately.

## Command forms

- Pretraining: use the repository's `make debug` target for a debug run and
  `make submit` for a submitted run. Include only arguments the `Makefile`
  accepts.
- SFT and RLVR: use the repository's launch file with:

  ```bash
  workspace uv run python <file_name> --submit start
  ```

  Preserve any repository-defined arguments and their ordering. Replace
  `<file_name>` with the discovered launcher path.

## Debug minimal repro

When the user asks for debugging, propose the smallest repro that still tests
the suspected failure. Start from the real launch configuration and show every
change from it.

1. Keep the model, code path, and the one suspected setting; reduce unrelated
   dimensions first.
2. Use one process/device, one data shard or a fixed tiny sample, minimal
   sequence length, and the fewest steps needed to reproduce the symptom.
3. For data failures, preserve the failing examples and disable nondeterminism
   where the launcher supports it. For distributed failures, first reproduce
   on one device, then add workers one at a time.
4. Report the proposed reductions, the expected signal, and what result would
   justify scaling back up.

Give the debug command only after its reduced configuration is summarized in
the run card.

## Scope

This skill plans launches and minimal reproductions only. Do not inspect or
stream logs here; a separate low-token log-checking skill will handle that.
