"""Tests for agent parsing (deprecated).

Agents use function calling only; text/XML parsing was removed.
These tests are skipped. See integration tests for function-calling agent behavior.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Parser deprecated; agents use function calling only.")
