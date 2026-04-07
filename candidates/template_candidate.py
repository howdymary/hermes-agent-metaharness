"""Template Meta-Harness candidate for Hermes benchmark runs.

This file is loaded by Hermes, not by hermes-agent-metaharness itself.
"""

from environments.meta_harness.candidate import BaseMetaHarnessCandidate
from environments.meta_harness.types import BootstrapArtifact


class HarnessCandidate(BaseMetaHarnessCandidate):
    """Minimal example candidate."""

    name = "template_candidate"
    description = "Example candidate that prepends a small planning hint."

    def build_user_prompt(self, ctx, base_user_prompt: str, bootstrap_artifacts):
        return (
            "Before acting, make a short plan and avoid redundant environment probing.\n\n"
            "Task:\n"
            f"{base_user_prompt}"
        )

    def pre_run_bootstrap(self, ctx, tool_context):
        # Keep bootstraps bounded and best-effort.
        try:
            result = tool_context.terminal("pwd", timeout=5)
        except Exception:
            return []

        output = str((result or {}).get("output") or "").strip()
        if not output:
            return []

        return [
            BootstrapArtifact(
                name="working_directory",
                content=output,
                include_in_prompt=False,
            )
        ]
