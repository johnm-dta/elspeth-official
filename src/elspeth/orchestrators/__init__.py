"""Orchestrators for different execution strategies.

This package provides different orchestration strategies for running SDA cycles:
- StandardOrchestrator: Simple sequential execution of cycles
- ExperimentalOrchestrator: A/B testing with baseline comparison and statistical analysis
"""

from .experimental import ExperimentalOrchestrator
from .standard import StandardOrchestrator

__all__ = [
    "ExperimentalOrchestrator",
    "StandardOrchestrator",
]
