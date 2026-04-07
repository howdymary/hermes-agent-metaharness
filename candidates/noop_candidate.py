"""No-op Meta-Harness candidate for baseline Hermes benchmark runs.

This candidate preserves the default prompt/tool/loop behavior exposed by
Hermes' benchmark runtime and is useful as a comparison baseline.
"""

from environments.meta_harness.candidate import BaseMetaHarnessCandidate


class HarnessCandidate(BaseMetaHarnessCandidate):
    """Baseline candidate that intentionally makes no harness changes."""

    name = "noop_candidate"
    description = "Pass-through baseline candidate with no bootstrap or prompt changes."
