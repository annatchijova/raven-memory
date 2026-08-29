"""
raven-memory — Adaptive Memory Field for Agentic Systems.

Public API:

    from raven import AdaptiveMemoryEngine, MemoryAgentOrchestrator, QwenConfig

Authors: Anna Tchijova + Claude (VIGÍA AI Collective)
License: Apache 2.0
"""

from .memory_engine import (
    AdaptiveMemoryEngine,
    LinkType,
    MemoryEntry,
    MemoryState,
    RecallResult,
    verify_audit_chain,
)
from .qwen_client import (
    EmbeddingProvider,
    MemoryAgentOrchestrator,
    QwenConfig,
    QwenLLMClient,
    resolve_dashscope_api_key,
)

__version__ = "1.2.0"

__all__ = [
    "AdaptiveMemoryEngine",
    "EmbeddingProvider",
    "LinkType",
    "MemoryAgentOrchestrator",
    "MemoryEntry",
    "MemoryState",
    "QwenConfig",
    "QwenLLMClient",
    "RecallResult",
    "resolve_dashscope_api_key",
    "verify_audit_chain",
    "__version__",
]
