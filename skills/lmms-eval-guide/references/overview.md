# lmms-eval Overview

## What This Repo Is

- `lmms-eval` is a multimodal model evaluation framework, not a training framework.
- The primary product surfaces are the unified CLI, task YAML system, model wrapper layer, evaluation loop, and optional HTTP eval service and Web UI.
- The most common usage is still one-shot command-line evaluation via `python -m lmms_eval --model ... --tasks ...` or `lmms-eval eval ...`.

## Canonical Entrypoints

- Use `pyproject.toml` as the source of truth for console entrypoints.
- `lmms-eval` routes through `lmms_eval.cli.dispatch:main`.
- `lmms-eval-ui` launches the Web UI.
- `lmms-eval-mcp` launches the MCP service.
- `setup.py` is thin and not the authoritative runtime map.

## Repo Mental Model

- Tasks live under `lmms_eval/tasks/**` and are YAML-first.
- A task is usually a YAML file plus optional `utils.py`.
- Prefer `doc_to_messages` for new tasks; keep `doc_to_text` plus `doc_to_visual` as fallback for simple-model paths.
- Models are wrapper classes that subclass `lmms_eval.api.model.lmms`.
- New model work usually means implementing `generate_until` around an existing runtime or API backend.
- Task discovery is automatic. `TaskManager` scans YAML files and does not download datasets during browse commands.
- Dataset loading happens later when tasks are instantiated and executed.

## Core Abstractions

- `lmms_eval.api.model.lmms`: base model contract.
- `lmms_eval.api.task.TaskConfig`, `ConfigurableTask`, `ConfigurableMessagesTask`: task config and task execution.
- `lmms_eval.models.registry_v2.ModelRegistryV2`: canonical model resolution.
- `lmms_eval.protocol.ChatMessages`: multimodal chat message protocol.

## Evaluation Data Flow

1. CLI parses args in `lmms_eval/cli/dispatch.py` and `lmms_eval/__main__.py`.
2. Config values merge in this order: defaults, config YAML, explicit CLI.
3. `simple_evaluate()` resolves the model via `lmms_eval.models.get_model(...).create_from_arg_string(...)`.
4. `TaskManager` loads task YAML definitions and `get_task_dict()` instantiates task objects.
5. `Task.build_all_requests()` produces `Instance` objects.
6. The model runs `generate_until()` or `loglikelihood()`.
7. The evaluator applies filters, strips reasoning tags when configured, and calls `process_results()`.
8. Metrics aggregate and `EvaluationTracker` writes aggregated JSON and optional sample JSONL files.

## Canonical Browse and Validation Commands

Run these in increasing cost order:

1. `lmms-eval version`
2. `lmms-eval tasks subtasks`
3. `lmms-eval models --aliases`
4. `uv run python -m pytest test/ -q --ignore=test/eval/test_usage_metrics.py`
5. `uv run python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct --tasks mme --batch_size 1 --limit 8`

Avoid browse commands that download datasets just to inspect the registry.

## Directory Map

- `README.md`: top-level product and usage overview.
- `pyproject.toml`: canonical scripts, dependencies, pytest config.
- `configs/`: run-level config examples.
- `docs/`: user and developer docs; useful, but not always current.
- `lmms_eval/__main__.py`: legacy eval parser and eval entry.
- `lmms_eval/cli/dispatch.py`: unified CLI router.
- `lmms_eval/evaluator.py`: evaluation main loop.
- `lmms_eval/tasks/__init__.py`: `TaskManager` and YAML discovery.
- `lmms_eval/api/task.py`: task config, dataset lifecycle, request construction.
- `lmms_eval/api/model.py`: model abstraction.
- `lmms_eval/models/__init__.py` and `lmms_eval/models/registry_v2.py`: model registration and resolution.
- `lmms_eval/protocol.py`: chat multimodal protocol.
- `lmms_eval/caching/response_cache.py`: response cache.
- `lmms_eval/entrypoints/http_server.py` and `lmms_eval/entrypoints/job_scheduler.py`: async eval service.
- `lmms_eval/tui/`: Web UI.
- `test/README.md`: test suite map and recommended commands.

## Best Starter Files

Read in this order:

1. `README.md`
2. `pyproject.toml`
3. `lmms_eval/cli/dispatch.py`
4. `lmms_eval/__main__.py`
5. `lmms_eval/evaluator.py`
6. `lmms_eval/tasks/__init__.py`
7. `lmms_eval/api/task.py`
8. `lmms_eval/api/model.py`
9. `lmms_eval/models/__init__.py`
10. `lmms_eval/models/registry_v2.py`
11. `lmms_eval/protocol.py`
12. `lmms_eval/tasks/mme/mme.yaml` and `lmms_eval/tasks/mme/utils.py`
13. `lmms_eval/tasks/mmmu/mmmu_val.yaml` and `lmms_eval/tasks/mmmu/utils.py`
14. `lmms_eval/models/chat/qwen2_5_vl.py` and `lmms_eval/models/simple/qwen2_5_vl.py`
15. `test/README.md`, `test/eval/test_task_pipeline.py`, `test/eval/test_construct_requests.py`, and `test/models/test_model_registry_v2.py`

## First-Week Workflow

1. Honor repo-root `AGENTS.md` and workspace execution constraints before running commands.
2. Start with no-download browse commands to learn tasks and models.
3. Read one simple task, one chat model, and `evaluator.py` before reading wide across the tree.
4. Run CPU tests before trying a smoke eval.
5. For new task work, stay inside `lmms_eval/tasks/<name>/` whenever possible.
6. For new model work, stay inside `lmms_eval/models/chat/<name>.py` plus registry edits whenever possible.
7. Leave cache, baseline compare, LLM judge, and HTTP server work until after the basic task/model path is clear.

## Common Pitfalls

- Docs drift exists. Prefer source and `pyproject.toml` over prose docs when they disagree.
- `group` exists historically, but prefer `tag` for new task grouping work.
- `@register_model(...)` is common in model modules, but CLI resolution still depends on `lmms_eval/models/__init__.py` building `ModelRegistryV2`.
- Avoid broad refactors while adding one task or one model.
- Avoid noisy stdout in runtime code paths.
