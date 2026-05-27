"""Validation system for the AI Editing Kernel.

The validator turns document, action, and result checks into structured reports.
It does not mutate the document. Executors use it before and after actions to
enforce schema correctness, layer/mask existence, write-mask safety, lock flags,
and basic expected-result contracts.
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

    def to_json(self) -> dict[str, Any]:
        """Serialize this issue for trace logs and API responses."""
        return _drop_none(
            {
                "type": ValidationCheckType(self.type).value,
                "severity": ValidationSeverity(self.severity).value,
                "message": self.message,
                "code": self.code,
                "action_id": self.action_id,
                "layer_id": self.layer_id,
                "mask_id": self.mask_id,
                "details": _json_safe(self.details),
            }
        )


@dataclass(slots=True)
class ValidationReport:
    """Validation result with zero or more issues."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Return True if any issue is ERROR or CRITICAL."""
        return any(
            issue.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}
            for issue in self.issues
        )

    def add_issue(self, issue: ValidationIssue) -> None:
        """Append an issue and update the `passed` flag if needed."""
        if not isinstance(issue, ValidationIssue):
            raise TypeError("issue must be a ValidationIssue")
        self.issues.append(issue)
        if issue.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}:
            self.passed = False

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        """Return a report combining this report and another report."""
        if not isinstance(other, ValidationReport):
            raise TypeError("other must be a ValidationReport")
        return ValidationReport(
            passed=self.passed and other.passed,
            issues=[*self.issues, *other.issues],
            metrics={**self.metrics, **other.metrics},
            metadata={**self.metadata, **other.metadata},
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize the report for trace logs and API responses."""
        return {
            "passed": self.passed,
            "issues": [issue.to_json() for issue in self.issues],
            "metrics": _json_safe(self.metrics),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class Validator:
    """Validate documents, actions, and execution results."""

    strict: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_document(self, document: DocumentState) -> ValidationReport:
        """Check all document invariants."""
        report = ValidationReport(passed=True)
        if not isinstance(document, DocumentState):
            report.add_issue(
                _issue(
                    ValidationCheckType.DOCUMENT_INVARIANT,
                    ValidationSeverity.CRITICAL,
                    "document must be a DocumentState",
                    "document.type",
                )
            )
            return report
        try:
            document.validate()
        except Exception as exc:
            report.add_issue(
                _issue(
                    ValidationCheckType.DOCUMENT_INVARIANT,
                    ValidationSeverity.ERROR,
                    str(exc),
                    "document.invalid",
                )
            )
        return report

    def validate_action_schema(self, action: Action) -> ValidationReport:
        """Check that an action is well-formed before document-specific validation."""
        report = ValidationReport(passed=True)
        if not isinstance(action, Action):
            report.add_issue(
                _issue(
                    ValidationCheckType.ACTION_SCHEMA,
                    ValidationSeverity.CRITICAL,
                    "action must be an Action",
                    "action.type",
                )
            )
            return report
        try:
            action.validate_schema()
        except Exception as exc:
            report.add_issue(
                _issue(
                    ValidationCheckType.ACTION_SCHEMA,
                    ValidationSeverity.ERROR,
                    str(exc),
                    "action.invalid",
                    action_id=getattr(action, "id", None),
                )
            )
        return report

    def validate_preconditions(self, document: DocumentState, action: Action) -> ValidationReport:
        """Check that the document is ready for an action."""
        report = self.validate_action_schema(action).merge(self.validate_document(document))
        if report.has_errors():
            return report

        for layer_id in action.preconditions.required_layer_ids:
            if not _layer_exists(document, layer_id):
                report.add_issue(
                    _issue(
                        ValidationCheckType.PRECONDITION,
                        ValidationSeverity.ERROR,
                        f"required layer {layer_id!r} does not exist",
                        "precondition.layer_missing",
                        action_id=action.id,
                        layer_id=layer_id,
                    )
                )
        for mask_id in action.preconditions.required_mask_ids:
            if mask_id not in document.masks:
                report.add_issue(
                    _issue(
                        ValidationCheckType.PRECONDITION,
                        ValidationSeverity.ERROR,
                        f"required mask {mask_id!r} does not exist",
                        "precondition.mask_missing",
                        action_id=action.id,
                        mask_id=mask_id,
                    )
                )

        if action.preconditions.require_active_layer and document.active_layer_id is None:
            report.add_issue(
                _issue(
                    ValidationCheckType.PRECONDITION,
                    ValidationSeverity.ERROR,
                    "document has no active layer",
                    "precondition.active_layer_missing",
                    action_id=action.id,
                )
            )
        if action.preconditions.require_active_selection and document.active_selection_mask_id is None:
            report.add_issue(
                _issue(
                    ValidationCheckType.PRECONDITION,
                    ValidationSeverity.ERROR,
                    "document has no active selection",
                    "precondition.active_selection_missing",
                    action_id=action.id,
                )
            )

        if action.target.layer_id is not None:
            try:
                layer = document.get_layer(action.target.layer_id)
            except KeyError:
                report.add_issue(
                    _issue(
                        ValidationCheckType.PRECONDITION,
                        ValidationSeverity.ERROR,
                        f"target layer {action.target.layer_id!r} does not exist",
                        "precondition.target_layer_missing",
                        action_id=action.id,
                        layer_id=action.target.layer_id,
                    )
                )
            else:
                if not action.preconditions.allow_hidden_layers and not layer.visible:
                    report.add_issue(
                        _issue(
                            ValidationCheckType.PRECONDITION,
                            ValidationSeverity.ERROR,
                            f"target layer {layer.id!r} is hidden",
                            "precondition.target_layer_hidden",
                            action_id=action.id,
                            layer_id=layer.id,
                        )
                    )
                if action.preconditions.require_unlocked_target_layer:
                    if layer.locks.fully_locked or (action.requires_pixel_write() and layer.locks.pixels_locked):
                        report.add_issue(
                            _issue(
                                ValidationCheckType.PRECONDITION,
                                ValidationSeverity.ERROR,
                                f"target layer {layer.id!r} is locked",
                                "precondition.target_layer_locked",
                                action_id=action.id,
                                layer_id=layer.id,
                            )
                        )

        report = report.merge(self.validate_write_mask_required(action))
        if action.write_mask_id is not None and action.write_mask_id not in document.masks:
            report.add_issue(
                _issue(
                    ValidationCheckType.PRECONDITION,
                    ValidationSeverity.ERROR,
                    f"write mask {action.write_mask_id!r} does not exist",
                    "precondition.write_mask_missing",
                    action_id=action.id,
                    mask_id=action.write_mask_id,
                )
            )

        return report

    def validate_result(
        self,
        before: DocumentState,
        after: DocumentState,
        action: Action,
        result: ActionResult,
    ) -> ValidationReport:
        """Check that an executed action produced an acceptable result."""
        report = self.validate_document(after)
        report = report.merge(self.validate_expected_layers(before, after, action))

        if result.succeeded() and action.requires_pixel_write() and action.target.layer_id and action.write_mask_id:
            try:
                before_layer = before.get_layer(action.target.layer_id)
                after_layer = after.get_layer(action.target.layer_id)
                write_mask = after.get_mask(action.write_mask_id)
            except Exception as exc:
                report.add_issue(
                    _issue(
                        ValidationCheckType.PIXEL_PROTECTION,
                        ValidationSeverity.ERROR,
                        str(exc),
                        "pixel_protection.lookup_failed",
                        action_id=action.id,
                    )
                )
            else:
                if before_layer.pixels is not None and after_layer.pixels is not None:
                    report = report.merge(
                        self.assert_outside_mask_unchanged(before_layer.pixels, after_layer.pixels, write_mask)
                    )
        return report

    def validate_mask(self, document: DocumentState, mask: Mask) -> ValidationReport:
        """Check mask dimensions, value range, hard/soft consistency, and metadata."""
        report = ValidationReport(passed=True)
        try:
            mask.validate(document.canvas.width, document.canvas.height)
        except Exception as exc:
            report.add_issue(
                _issue(
                    ValidationCheckType.MASK_INTEGRITY,
                    ValidationSeverity.ERROR,
                    str(exc),
                    "mask.invalid",
                    mask_id=getattr(mask, "id", None),
                )
            )
        return report

    def validate_layer_stack(self, document: DocumentState) -> ValidationReport:
        """Check layer order, uniqueness, group relationships, and active layer state."""
        return self.validate_document(document)

    def validate_write_mask_required(self, action: Action) -> ValidationReport:
        """Ensure pixel-writing actions have a write mask."""
        report = ValidationReport(passed=True)
        if action.requires_pixel_write() and action.preconditions.require_write_mask and action.write_mask_id is None:
            report.add_issue(
                _issue(
                    ValidationCheckType.PRECONDITION,
                    ValidationSeverity.ERROR,
                    "pixel-writing action is missing write_mask_id",
                    "precondition.write_mask_required",
                    action_id=action.id,
                )
            )
        return report

    def assert_outside_mask_unchanged(
        self,
        before_pixels: np.ndarray,
        after_pixels: np.ndarray,
        write_mask: Mask,
        tolerance: float = 0.0,
    ) -> ValidationReport:
        """Verify that fully protected pixels outside a write mask did not change."""
        report = ValidationReport(passed=True)
        if before_pixels.shape != after_pixels.shape:
            report.add_issue(
                _issue(
                    ValidationCheckType.PIXEL_PROTECTION,
                    ValidationSeverity.ERROR,
                    "before and after pixel arrays have different shapes",
                    "pixel_protection.shape_mismatch",
                    mask_id=write_mask.id,
                    details={"before_shape": before_pixels.shape, "after_shape": after_pixels.shape},
                )
            )
            return report
        if write_mask.data.shape != before_pixels.shape[:2]:
            report.add_issue(
                _issue(
                    ValidationCheckType.PIXEL_PROTECTION,
                    ValidationSeverity.ERROR,
                    "write mask shape does not match pixel arrays",
                    "pixel_protection.mask_shape_mismatch",
                    mask_id=write_mask.id,
                )
            )
            return report

        protected = write_mask.data <= 0.0
        if not bool(np.any(protected)):
            report.metrics["max_protected_delta"] = 0.0
            return report

        deltas = np.abs(after_pixels[protected] - before_pixels[protected])
        max_delta = float(deltas.max()) if deltas.size else 0.0
        report.metrics["max_protected_delta"] = max_delta
        if max_delta > tolerance:
            report.add_issue(
                _issue(
                    ValidationCheckType.PIXEL_PROTECTION,
                    ValidationSeverity.ERROR,
                    "pixels outside the write mask changed",
                    "pixel_protection.protected_pixels_changed",
                    mask_id=write_mask.id,
                    details={"max_delta": max_delta, "tolerance": tolerance},
                )
            )
        return report

    def validate_expected_layers(
        self,
        before: DocumentState,
        after: DocumentState,
        action: Action,
    ) -> ValidationReport:
        """Check that expected layers were created or changed."""
        report = ValidationReport(passed=True)
        before_ids = {layer.id for layer in before.layers}
        after_ids = {layer.id for layer in after.layers}

        for layer_id in action.expected_result.changed_layer_ids:
            if layer_id not in after_ids:
                report.add_issue(
                    _issue(
                        ValidationCheckType.LAYER_INTEGRITY,
                        ValidationSeverity.ERROR,
                        f"expected changed layer {layer_id!r} does not exist after action",
                        "expected.layer_missing_after",
                        action_id=action.id,
                        layer_id=layer_id,
                    )
                )
            elif layer_id not in before_ids:
                report.add_issue(
                    _issue(
                        ValidationCheckType.LAYER_INTEGRITY,
                        ValidationSeverity.WARNING,
                        f"expected changed layer {layer_id!r} was created by the action",
                        "expected.layer_created_not_changed",
                        action_id=action.id,
                        layer_id=layer_id,
                    )
                )
        return report

    def validate_geometry_expectations(
        self,
        document: DocumentState,
        action: Action,
    ) -> ValidationReport:
        """Return a placeholder pass for geometry checks not yet modeled."""
        return ValidationReport(passed=True, metadata={"geometry_expectations": action.expected_result.geometry_expectations})

    def validate_visual_expectations(
        self,
        before_preview: np.ndarray,
        after_preview: np.ndarray,
        action: Action,
    ) -> ValidationReport:
        """Return a placeholder pass for visual quality checks."""
        return ValidationReport(
            passed=True,
            metadata={
                "visual_expectations": action.expected_result.visual_expectations,
                "before_shape": before_preview.shape,
                "after_shape": after_preview.shape,
            },
        )


def _issue(
    type: ValidationCheckType,
    severity: ValidationSeverity,
    message: str,
    code: str,
    action_id: Optional[str] = None,
    layer_id: Optional[str] = None,
    mask_id: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> ValidationIssue:
    """Create a validation issue with consistent defaults."""
    return ValidationIssue(
        type=type,
        severity=severity,
        message=message,
        code=code,
        action_id=action_id,
        layer_id=layer_id,
        mask_id=mask_id,
        details={} if details is None else details,
    )


def _layer_exists(document: DocumentState, layer_id: str) -> bool:
    """Return whether a layer ID exists without leaking lookup exceptions."""
    try:
        document.get_layer(layer_id)
    except KeyError:
        return False
    return True


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"array": {"shape": list(value.shape), "dtype": str(value.dtype)}}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
