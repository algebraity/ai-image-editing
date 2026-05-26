"""Action executor for the AI Editing Kernel.

The Executor is the only component that should mutate DocumentState. Planners
produce Actions; Validators check them; Executor applies them; TraceLogger records
what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ai_edit_kernel.document.document_state import DocumentState
from ai_edit_kernel.schema.actions import Action, ActionBatch, ActionResult


class DiffusionBackend(Protocol):
    """Protocol for pluggable diffusion/image-generation backends.

    Concrete adapters can wrap ComfyUI, Diffusers, hosted APIs, or internal
    models. The executor should call this protocol, not a specific backend.
    """

    def inpaint(self, job: dict[str, Any]) -> dict[str, Any]:
        """Generate pixels for a masked inpainting job and return result assets."""
        raise NotImplementedError

    def img2img(self, job: dict[str, Any]) -> dict[str, Any]:
        """Generate an image-to-image result and return result assets."""
        raise NotImplementedError

    def txt2img(self, job: dict[str, Any]) -> dict[str, Any]:
        """Generate a text-to-image result and return result assets."""
        raise NotImplementedError


class TraceSink(Protocol):
    """Minimal protocol implemented by TraceLogger."""

    def log_action_started(self, action: Action, document: DocumentState) -> None:
        """Record that action execution has started."""
        raise NotImplementedError

    def log_action_result(self, action: ActionResult, document: DocumentState) -> None:
        """Record the final result of an action."""
        raise NotImplementedError


@dataclass(slots=True)
class ExecutionOptions:
    """Runtime behavior switches for action execution."""

    dry_run: bool = False
    validate_before: bool = True
    validate_after: bool = True
    rollback_on_failure: bool = True
    allow_full_canvas_writes: bool = False
    record_intermediate_snapshots: bool = True
    strict_mask_guard: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionContext:
    """External dependencies and per-run settings for the Executor."""

    options: ExecutionOptions = field(default_factory=ExecutionOptions)
    diffusion_backend: Optional[DiffusionBackend] = None
    trace_sink: Optional[TraceSink] = None
    asset_store: Optional[Any] = None
    validator: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Executor:
    """Apply Actions to a DocumentState.

    All edits should flow through this class. The executor is responsible for
    dispatching action types, enforcing write masks, calling external backends,
    updating document revisions, and returning structured ActionResults.
    """

    context: ExecutionContext = field(default_factory=ExecutionContext)

    def execute_batch(self, document: DocumentState, batch: ActionBatch) -> list[ActionResult]:
        """Execute a sequence of actions against a document.

        Must validate batch schema, optionally snapshot the document for rollback,
        run each action in order, stop or continue based on batch.stop_on_error,
        log each action, and return one ActionResult per attempted action.
        """
        raise NotImplementedError

    def execute_action(self, document: DocumentState, action: Action) -> ActionResult:
        """Execute one action against a document.

        Must perform schema validation, precondition validation, before snapshot
        capture if needed, dispatch to the action handler, run post-validation,
        update revision, handle rollback on failure, and emit trace events.
        """
        raise NotImplementedError

    def dispatch(self, document: DocumentState, action: Action) -> ActionResult:
        """Route an action to its implementation method.

        This should contain no business logic beyond mapping ActionType values to
        private handler methods.
        """
        raise NotImplementedError

    def apply_write_mask(self, before_pixels: Any, proposed_pixels: Any, write_mask_id: str, document: DocumentState) -> Any:
        """Composite proposed pixels over old pixels using the action's write mask.

        This is the key safety primitive. Pixel-changing handlers should prepare a
        proposed result, then call this method so only write-mask pixels change.
        """
        raise NotImplementedError

    def create_rollback_snapshot(self, document: DocumentState) -> Any:
        """Create whatever snapshot is needed to restore the document on failure.

        V0 can use DocumentState.clone_deep(). Later versions may use copy-on-write
        snapshots or operation-level undo data for performance.
        """
        raise NotImplementedError

    def rollback(self, document: DocumentState, snapshot: Any) -> None:
        """Restore a document to a previous snapshot after a failed action."""
        raise NotImplementedError

    def _execute_create_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a new layer from action params and insert it into the document."""
        raise NotImplementedError

    def _execute_delete_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Remove a layer after lock and dependency checks."""
        raise NotImplementedError

    def _execute_duplicate_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a deep copy of a layer and insert it into the stack."""
        raise NotImplementedError

    def _execute_rename_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Rename a layer without changing its ID or pixels."""
        raise NotImplementedError

    def _execute_reorder_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Move a layer to a different stack index."""
        raise NotImplementedError

    def _execute_set_active_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Set the document's active layer."""
        raise NotImplementedError

    def _execute_select_rect(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a rectangular selection mask and set it active if requested."""
        raise NotImplementedError

    def _execute_select_ellipse(self, document: DocumentState, action: Action) -> ActionResult:
        """Create an elliptical selection mask and set it active if requested."""
        raise NotImplementedError

    def _execute_magic_wand_select(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a contiguous color-based selection from a seed point.

        The implementation should use deterministic image processing, not an LLM,
        and should store the resulting mask in DocumentState.
        """
        raise NotImplementedError

    def _execute_combine_masks(self, document: DocumentState, action: Action) -> ActionResult:
        """Union, intersect, or subtract masks and register the output mask."""
        raise NotImplementedError

    def _execute_feather_mask(self, document: DocumentState, action: Action) -> ActionResult:
        """Create a softened copy of a mask."""
        raise NotImplementedError

    def _execute_draw_shape(self, document: DocumentState, action: Action) -> ActionResult:
        """Rasterize or store a vector shape on the target layer.

        Must respect write masks, target layer locks, opacity, alpha behavior, and
        expected geometry parameters.
        """
        raise NotImplementedError

    def _execute_clear_region(self, document: DocumentState, action: Action) -> ActionResult:
        """Clear pixels or alpha inside the write mask on the target layer."""
        raise NotImplementedError

    def _execute_paint_bucket_fill(self, document: DocumentState, action: Action) -> ActionResult:
        """Fill a contiguous or preselected region with a color/texture.

        Must use a deterministic region mask and then clip through the write mask.
        """
        raise NotImplementedError

    def _execute_transform_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Move, scale, rotate, or align a layer according to action params."""
        raise NotImplementedError

    def _execute_inpaint_region(self, document: DocumentState, action: Action) -> ActionResult:
        """Call the diffusion backend for a masked region and composite the result.

        Must crop/prepare source context and mask, call the backend, receive the
        generated asset, clip it through the write mask, place it on the requested
        layer, and validate that protected pixels are unchanged.
        """
        raise NotImplementedError

    def _execute_img2img_to_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Call an image-to-image backend and place the result on a target layer."""
        raise NotImplementedError

    def _execute_txt2img_to_layer(self, document: DocumentState, action: Action) -> ActionResult:
        """Call a text-to-image backend and import the result as a new layer."""
        raise NotImplementedError
