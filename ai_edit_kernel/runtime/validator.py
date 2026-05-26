"""Validation system for the AI Editing Kernel.

The Validator checks whether documents, actions, and results are safe and correct.
It should become one of the project's main differentiators because it enforces
mask safety, geometry consistency, layer correctness, and trace reliability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from ai_edit_kernel.document.document_state import DocumentState
from ai_edit_kernel.document.mask import Mask
from ai_edit_kernel.schema.actions import Action, ActionResult


class ValidationSeverity(str, Enum):
    """Severity level for a validation issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCheckType(str, Enum):
    """Categories of validation checks."""

    DOCUMENT_INVARIANT = "document_invariant"
    ACTION_SCHEMA = "action_schema"
    PRECONDITION = "precondition"
    MASK_INTEGRITY = "mask_integrity"
    LAYER_INTEGRITY = "layer_integrity"
    GEOMETRY = "geometry"
    PIXEL_PROTECTION = "pixel_protection"
    VISUAL_QUALITY = "visual_quality"
    TRACEABILITY = "traceability"


@dataclass(slots=True)
class ValidationIssue:
    """One validation finding."""

    type: ValidationCheckType
    severity: ValidationSeverity
    message: str
    code: str
    action_id: Optional[str] = None
    layer_id: Optional[str] = None
    mask_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationReport:
    """Validation result with zero or more issues."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Return True if any issue is ERROR or CRITICAL."""
        raise NotImplementedError

    def add_issue(self, issue: ValidationIssue) -> None:
        """Append an issue and update the `passed` flag if needed."""
        raise NotImplementedError

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        """Return a report combining this report and another report."""
        raise NotImplementedError

    def to_json(self) -> dict[str, Any]:
        """Serialize the report for trace logs and API responses."""
        raise NotImplementedError


@dataclass(slots=True)
class Validator:
    """Validate documents, actions, and execution results.

    The Executor should call this before and after every action. Some checks are
    deterministic and must be strict; visual-quality checks may be approximate or
    delegated to a model in later versions.
    """

    strict: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_document(self, document: DocumentState) -> ValidationReport:
        """Check all document invariants.

        Must verify canvas settings, layer IDs, layer shapes, mask dimensions,
        active layer references, active selection references, group references,
        and attached layer masks.
        """
        raise NotImplementedError

    def validate_action_schema(self, action: Action) -> ValidationReport:
        """Check that an action is well-formed before document-specific validation."""
        raise NotImplementedError

    def validate_preconditions(self, document: DocumentState, action: Action) -> ValidationReport:
        """Check that the document is ready for an action.

        Must confirm required layers/masks exist, target layers are unlocked,
        write masks exist for pixel writes, and dangerous full-canvas writes are
        explicitly allowed.
        """
        raise NotImplementedError

    def validate_result(
        self,
        before: DocumentState,
        after: DocumentState,
        action: Action,
        result: ActionResult,
    ) -> ValidationReport:
        """Check that an executed action produced an acceptable result.

        Must combine document invariant checks, expected-result checks, mask
        safety checks, layer changes, and action-specific geometry checks.
        """
        raise NotImplementedError

    def validate_mask(self, document: DocumentState, mask: Mask) -> ValidationReport:
        """Check mask dimensions, value range, hard/soft consistency, and metadata."""
        raise NotImplementedError

    def validate_layer_stack(self, document: DocumentState) -> ValidationReport:
        """Check layer order, uniqueness, group relationships, and active layer state."""
        raise NotImplementedError

    def validate_write_mask_required(self, action: Action) -> ValidationReport:
        """Ensure pixel-writing actions have a write mask or explicit full-canvas permission."""
        raise NotImplementedError

    def assert_outside_mask_unchanged(
        self,
        before_pixels: np.ndarray,
        after_pixels: np.ndarray,
        write_mask: Mask,
        tolerance: float = 0.0,
    ) -> ValidationReport:
        """Verify that pixels outside a write mask did not change.

        This is the critical safety check for localized editing. For hard masks,
        tolerance should usually be 0.0. For feathered masks, implementation must
        distinguish fully protected pixels from the transition band.
        """
        raise NotImplementedError

    def validate_expected_layers(
        self,
        before: DocumentState,
        after: DocumentState,
        action: Action,
    ) -> ValidationReport:
        """Check that expected layers were created, changed, preserved, or removed."""
        raise NotImplementedError

    def validate_geometry_expectations(
        self,
        document: DocumentState,
        action: Action,
    ) -> ValidationReport:
        """Check action-specific geometry expectations.

        Examples: square center matches previous circle center, square side length
        equals old circle diameter, or generated content remains inside a shape.
        """
        raise NotImplementedError

    def validate_visual_expectations(
        self,
        before_preview: np.ndarray,
        after_preview: np.ndarray,
        action: Action,
    ) -> ValidationReport:
        """Check visual-quality expectations.

        V0 can return a placeholder report. Later this may use heuristics or a
        vision model for seam detection, blotchiness, prompt satisfaction, and
        line-art preservation.
        """
        raise NotImplementedError
