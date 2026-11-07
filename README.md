# compache

A harness-agnostic optimizer for long agent conversations: it keeps a conversation inside the context window without rewriting the provider's cached prefix by accident, and decides at every run end whether keeping that prefix warm is worth paying for.

Status: design phase, no code yet. See the [docs](docs/README.md).
