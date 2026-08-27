"""Pluggable point-in-time research workflow."""

from csi500_alpha.workflow.calibration import WalkForwardReturnCalibrationEngine
from csi500_alpha.workflow.components import (
    ResearchComponentRegistry,
    default_component_registry,
)
from csi500_alpha.workflow.orchestrator import ResearchWorkflow, WorkflowResult
from csi500_alpha.workflow.samples import ResearchSamplePolicy

__all__ = [
    "ResearchComponentRegistry",
    "ResearchWorkflow",
    "ResearchSamplePolicy",
    "WalkForwardReturnCalibrationEngine",
    "WorkflowResult",
    "default_component_registry",
]
