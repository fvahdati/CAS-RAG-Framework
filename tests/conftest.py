"""Test defaults: mock LLM allowed only in pytest runs."""

import os

# Must run before any cas_rag import that calls get_settings().
os.environ["CAS_RAG_ALLOW_MOCK"] = "1"
os.environ["CAS_RAG_LLM_PROVIDER"] = "mock"
os.environ["CAS_RAG_OFFLINE_MODE"] = "1"

# Clear settings singleton so pytest always sees the env above.
def pytest_configure(config):
    from cas_rag.config.settings import get_settings

    get_settings.cache_clear()
