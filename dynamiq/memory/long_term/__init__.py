"""Long-term, fact-shaped, user-scoped memory for Dynamiq agents.

See docs/superpowers/specs/2026-05-25-long-term-memory-design.md.
"""

from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact

__all__ = [
    "Fact",
    "LongTermMemoryBackend",
]
