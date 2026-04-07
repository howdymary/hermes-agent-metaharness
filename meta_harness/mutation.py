"""Structured candidate mutation helpers for Meta-Harness search."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class MutationSpec:
    """One deterministic wrapper mutation around a seed candidate."""

    slug: str
    description: str
    prompt_prelude: str = ""
    prioritize_tools: Tuple[str, ...] = ()
    exclude_tools: Tuple[str, ...] = ()
    max_turns_delta: int = 0
    max_turns_cap: int = 0
    notes: Tuple[str, ...] = field(default_factory=tuple)


def builtin_mutations() -> Dict[str, MutationSpec]:
    """Return the built-in deterministic mutations used for structured search."""
    return {
        "plan_briefly": MutationSpec(
            slug="plan_briefly",
            description="Prepend a brief planning reminder to the first user prompt.",
            prompt_prelude=(
                "Start with a short plan, then act. Avoid repeating environment discovery "
                "if the prompt or bootstrap already gives you the needed context."
            ),
            notes=("Adds a concise planning instruction before the task prompt.",),
        ),
        "verify_before_finish": MutationSpec(
            slug="verify_before_finish",
            description="Prepend a verification reminder before the first user prompt.",
            prompt_prelude=(
                "Before finishing, run the smallest relevant verification step you can and "
                "summarize the concrete result."
            ),
            notes=("Encourages explicit verification before the agent stops.",),
        ),
        "terminal_first": MutationSpec(
            slug="terminal_first",
            description="Prioritize terminal-centric tools earlier in the tool list.",
            prioritize_tools=("terminal", "python", "shell"),
            notes=("Moves terminal-style tools earlier in the exposed tool order.",),
        ),
        "no_todo": MutationSpec(
            slug="no_todo",
            description="Hide todo-style tools to reduce overhead on short benchmark tasks.",
            exclude_tools=("todo_tool", "todo"),
            notes=("Removes optional todo tools when they are present.",),
        ),
        "shorter_loop": MutationSpec(
            slug="shorter_loop",
            description="Cap the rollout to fewer turns to reduce wandering.",
            max_turns_delta=-6,
            max_turns_cap=24,
            notes=("Reduces max turns to encourage earlier completion.",),
        ),
    }


def resolve_mutation_specs(selected: Sequence[str] | None) -> List[MutationSpec]:
    """Resolve mutation names to built-in specs."""
    builtins = builtin_mutations()
    if not selected:
        selected = ("plan_briefly", "verify_before_finish", "terminal_first", "shorter_loop")

    specs: List[MutationSpec] = []
    for slug in selected:
        if slug not in builtins:
            raise KeyError(f"Unknown mutation '{slug}'. Expected one of {sorted(builtins)}")
        specs.append(builtins[slug])
    return specs


def safe_slug(value: str) -> str:
    """Convert an arbitrary candidate or mutation name to a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "candidate"


def variant_candidate_name(seed_name: str, mutation: MutationSpec) -> str:
    """Stable human-readable name for a generated variant."""
    return f"{safe_slug(seed_name)}__{mutation.slug}"


def generate_variant_candidates(
    *,
    seed_candidate_path: Path,
    seed_candidate_name: str,
    output_dir: Path,
    mutations: Iterable[MutationSpec],
) -> List[Path]:
    """Generate wrapper candidate files for a set of mutations."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: List[Path] = []
    for mutation in mutations:
        variant_name = variant_candidate_name(seed_candidate_name, mutation)
        output_path = output_dir / f"{variant_name}.py"
        output_path.write_text(
            render_wrapper_candidate(
                seed_candidate_path=seed_candidate_path,
                variant_name=variant_name,
                mutation=mutation,
            ),
            encoding="utf-8",
        )
        generated_paths.append(output_path)
    return generated_paths


def render_wrapper_candidate(
    *,
    seed_candidate_path: Path,
    variant_name: str,
    mutation: MutationSpec,
) -> str:
    """Render a generated candidate that wraps a seed candidate."""
    mutation_payload = json.dumps(
        {
            "slug": mutation.slug,
            "description": mutation.description,
            "prompt_prelude": mutation.prompt_prelude,
            "prioritize_tools": list(mutation.prioritize_tools),
            "exclude_tools": list(mutation.exclude_tools),
            "max_turns_delta": mutation.max_turns_delta,
            "max_turns_cap": mutation.max_turns_cap,
            "notes": list(mutation.notes),
        },
        indent=2,
        sort_keys=True,
    )

    return f'''"""Generated Meta-Harness wrapper candidate.

This file was created by hermes-agent-metaharness and wraps a seed candidate
with one deterministic mutation.
"""

from __future__ import annotations

import copy
import json

from environments.meta_harness.candidate import BaseMetaHarnessCandidate
from environments.meta_harness.loader import load_candidate
from environments.meta_harness.types import LoopSettings, ToolSelection

_SEED_CANDIDATE_PATH = {str(seed_candidate_path.resolve())!r}
_VARIANT_NAME = {variant_name!r}
_MUTATION = json.loads({repr(mutation_payload)})


def _tool_name(tool_schema):
    return str(tool_schema.get("function", {{}}).get("name") or "")


class HarnessCandidate(BaseMetaHarnessCandidate):
    """Generated wrapper around a seed candidate."""

    name = _VARIANT_NAME
    description = _MUTATION["description"]

    def __init__(self):
        self._base_candidate, self._base_candidate_path = load_candidate(_SEED_CANDIDATE_PATH)

    def configure_tools(self, ctx, tool_schemas):
        selection = self._base_candidate.configure_tools(ctx, copy.deepcopy(tool_schemas))
        if selection is None:
            selection = ToolSelection(tool_schemas=list(tool_schemas))
        elif not isinstance(selection, ToolSelection):
            selection = ToolSelection(tool_schemas=list(selection))

        selected = list(selection.tool_schemas)
        notes = list(selection.notes)

        excluded = set(_MUTATION.get("exclude_tools") or [])
        if excluded:
            selected = [tool for tool in selected if _tool_name(tool) not in excluded]
            notes.append("Excluded tools: " + ", ".join(sorted(excluded)))

        prioritized = list(_MUTATION.get("prioritize_tools") or [])
        if prioritized:
            rank = {{name: index for index, name in enumerate(prioritized)}}
            selected = sorted(
                selected,
                key=lambda tool: (rank.get(_tool_name(tool), len(rank)), _tool_name(tool)),
            )
            notes.append("Prioritized tools: " + ", ".join(prioritized))

        notes.extend(_MUTATION.get("notes") or [])
        return ToolSelection(tool_schemas=selected, notes=notes)

    def pre_run_bootstrap(self, ctx, tool_context):
        return self._base_candidate.pre_run_bootstrap(ctx, tool_context)

    def build_system_prompt(self, ctx, base_system_prompt, bootstrap_artifacts):
        return self._base_candidate.build_system_prompt(
            ctx,
            base_system_prompt,
            bootstrap_artifacts,
        )

    def build_user_prompt(self, ctx, base_user_prompt, bootstrap_artifacts):
        base_prompt = self._base_candidate.build_user_prompt(
            ctx,
            base_user_prompt,
            bootstrap_artifacts,
        )
        prelude = str(_MUTATION.get("prompt_prelude") or "").strip()
        if not prelude:
            return base_prompt
        return f"{{prelude}}\\n\\n{{base_prompt}}"

    def loop_settings(self, ctx):
        settings = self._base_candidate.loop_settings(ctx) or LoopSettings(max_turns=ctx.default_max_turns)
        max_turns = settings.max_turns or ctx.default_max_turns

        max_turns_delta = int(_MUTATION.get("max_turns_delta") or 0)
        if max_turns_delta:
            max_turns = max(1, max_turns + max_turns_delta)

        max_turns_cap = int(_MUTATION.get("max_turns_cap") or 0)
        if max_turns_cap > 0:
            max_turns = min(max_turns, max_turns_cap)

        notes = list(settings.notes)
        notes.extend(_MUTATION.get("notes") or [])

        if max_turns_delta:
            notes.append(f"Adjusted max turns by {{max_turns_delta}}.")
        if max_turns_cap > 0:
            notes.append(f"Capped max turns at {{max_turns_cap}}.")

        return LoopSettings(
            max_turns=max_turns,
            loop_hooks=settings.loop_hooks,
            notes=notes,
        )
'''
