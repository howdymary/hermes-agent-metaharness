# Config Examples

This directory contains small Hermes benchmark config examples that are useful
for local Meta-Harness testing.

Included configs:

- `tblite_ollama_local.yaml`: lightweight local Docker + Ollama smoke-test config
- `tblite_ollama_qwopus.yaml`: stronger local Docker + Ollama config using a larger Qwen-family model

These are examples, not defaults. For serious evaluation, point Meta-Harness at
your own Hermes benchmark config with the strongest coding backend you have
available, for example:

- an OpenRouter-backed coding model
- a local or remote vLLM deployment
- another OpenAI-compatible endpoint
