# Candidate Files

This directory is for standalone Meta-Harness candidate files.

Candidate files are executed by Hermes benchmark environments, not by this repo directly.

That means candidate imports should target the Hermes runtime contract, for example:

```python
from environments.meta_harness.candidate import BaseMetaHarnessCandidate
from environments.meta_harness.types import BootstrapArtifact
```

Use this directory when you want to:

- keep candidate code under version control in this repo
- evaluate local candidate variants against Hermes
- compare candidate archives over time

You can evaluate a local candidate like this:

```bash
python -m meta_harness evaluate-candidate \
  --candidate candidates/template_candidate.py \
  --benchmark tblite
```
