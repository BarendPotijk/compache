"""compache as Hermes' context engine: `context.engine: compache`.

Everything is in the package, see `compache.hermes`. Hermes looks for the words
ContextEngine or register_context_engine in this file to recognize it as one.
"""

from compache.hermes import register

__all__ = ["register"]
