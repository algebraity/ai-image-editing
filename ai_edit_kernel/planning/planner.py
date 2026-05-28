"""Model-agnostic AI planner orchestration.

The planner layer is the boundary between natural-language model output and the
editing kernel. Backends return a small planner-facing JSON object; this module
turns it into a canonical `ActionBatch`, fills kernel-owned fields, and can pass
the batch to the existing executor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ai_edit_kernel.document.document_state import DocumentState
from ai_edit_kernel.runtime.executor import Executor
from ai_edit_kernel.schema.actions import (
    SCHEMA_VERSION as ACTION_SCHEMA_VERSION,
    Action,
    ActionBatch,
    ActionPreconditions,
    ActionTarget,
    ExpectedResult,
)
from ai_edit_kernel.schema.actions import ActionType
from ai_edit_kernel.text import FontRegistry

from .action_catalog import (
    PLANNER_CATALOG_VERSION,
    PLANNER_OUTPUT_SCHEMA_VERSION,
    TargetFieldMode,
    available_action_specs,
    get_action_spec,
    planner_output_schema,
)


PLANNER_REQUEST_SCHEMA_VERSION = "ai_edit_planner_request.v1"


JsonObject = dict[str, Any]


class PlannerBackend(Protocol):
    """Protocol for backends that turn planner requests into planner JSON."""

    def plan(self, request: JsonObject) -> JsonObject | str:
        """Return planner output JSON, or a JSON string containing it."""
        ...


class PlannerError(RuntimeError):
    """Raised when planner output cannot be normalized after retries."""

    def __init__(self, message: str, errors: list[JsonObject]) -> None:
        super().__init__(message)
        self.errors = errors


@dataclass(slots=True)
class PlannerOptions:
    """Behavior switches for planner request building and normalization."""

    max_schema_retries: int = 2
    created_by: str = "planner"
    batch_id_prefix: str = "batch"
    action_id_prefix: str = "action"
    default_to_active_layer: bool = True
    auto_create_full_canvas_mask: bool = True
    full_canvas_mask_id: str = "mask_full_canvas"
    stop_on_error: bool = True
    metadata: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class PlannerResult:
    """A successfully normalized planner response."""

    request: JsonObject
    raw_output: JsonObject | str
    action_batch: ActionBatch
    attempts: list[JsonObject] = field(default_factory=list)
    metadata: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class PlannerExecutionResult:
    """Result of planning and executing one action batch."""

    planner_result: PlannerResult
    action_results: list[Any]

    def succeeded(self) -> bool:
        """Return whether every executed action succeeded."""
        return all(result.succeeded() for result in self.action_results)


@dataclass(slots=True)
class StaticPlannerBackend:
    """Simple backend useful for tests and manual prototypes."""

    outputs: list[JsonObject | str]
    index: int = 0

    def plan(self, request: JsonObject) -> JsonObject | str:
        """Return the next configured output."""
        if self.index >= len(self.outputs):
            raise RuntimeError("static planner backend has no remaining outputs")
        output = self.outputs[self.index]
        self.index += 1
        return output


@dataclass(slots=True)
class PlannerRequestBuilder:
    """Build the JSON object sent to a planner backend."""

    include_action_schemas: bool = True
    include_font_catalog: bool = True
    font_catalog_limit: int = 80
    font_registry: Optional[FontRegistry] = None

    def build(
        self,
        user_prompt: str,
        document: DocumentState,
        *,
        observations: Optional[list[Any]] = None,
        asset_refs: Optional[dict[str, str]] = None,
        previous_errors: Optional[list[JsonObject]] = None,
        metadata: Optional[JsonObject] = None,
    ) -> JsonObject:
        """Return a complete model-facing planner request."""
        if not isinstance(user_prompt, str):
            raise TypeError("user_prompt must be a string")
        if not isinstance(document, DocumentState):
            raise TypeError("document must be a DocumentState")
        return {
            "schema_version": PLANNER_REQUEST_SCHEMA_VERSION,
            "user_prompt": user_prompt,
            "document_summary": document.snapshot_summary(),
            "observations": [] if observations is None else list(observations),
            "asset_refs": {} if asset_refs is None else dict(asset_refs),
            "tool_catalog_version": PLANNER_CATALOG_VERSION,
            "available_actions": available_action_specs() if self.include_action_schemas else [],
            **({"font_catalog": self._font_catalog()} if self.include_font_catalog else {}),
            "output_contract": planner_output_schema(),
            "previous_errors": [] if previous_errors is None else list(previous_errors),
            "constraints": [
                "Return only planner output JSON.",
                "Do not include action IDs, preconditions, expected_result, trace events, or manifest data.",
                "Use existing layer and mask IDs from document_summary when targeting existing objects.",
                "Provide a semantic output_layer_id or mask_id only when later actions need to reference it.",
                "Use bbox_xyxy coordinates as half-open pixel bounds: [x0, y0, x1, y1].",
                "When planning text, prefer font_id values from font_catalog instead of font paths or invented font names.",
            ],
            "metadata": {} if metadata is None else dict(metadata),
        }

    def _font_catalog(self) -> JsonObject:
        """Return the cached font catalog included in planner requests."""
        registry = self.font_registry if self.font_registry is not None else FontRegistry.default()
        return registry.catalog(limit=self.font_catalog_limit, include_paths=False)


@dataclass(slots=True)
class ActionBatchNormalizer:
    """Convert planner output into canonical executable action batches."""

    options: PlannerOptions = field(default_factory=PlannerOptions)

    def normalize(self, raw_output: JsonObject | str, document: DocumentState, user_prompt: str) -> ActionBatch:
        """Return an executable `ActionBatch` from planner output."""
        data = _coerce_json_object(raw_output)
        if data.get("schema_version") == ACTION_SCHEMA_VERSION:
            return ActionBatch.from_json(data)
        _reject_unknown_keys(data, "planner output", {"schema_version", "actions", "description", "stop_on_error", "metadata"})
        if "schema_version" in data and data["schema_version"] != PLANNER_OUTPUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported planner output schema_version {data['schema_version']!r}")
        actions_data = data.get("actions")
        if not isinstance(actions_data, list) or len(actions_data) == 0:
            raise ValueError("planner output requires a non-empty actions list")

        known_layer_ids = {layer.id for layer in document.layers}
        known_mask_ids = set(document.masks)
        layer_aliases: dict[str, str] = {}
        mask_aliases: dict[str, str] = {}
        pending_payloads: list[JsonObject] = []
        full_canvas_mask_planned = self.options.full_canvas_mask_id in known_mask_ids

        for item in actions_data:
            payload = self._normalize_action(
                item,
                document,
                known_layer_ids,
                known_mask_ids,
                layer_aliases,
                mask_aliases,
            )
            if self._needs_generated_full_canvas_mask(payload, full_canvas_mask_planned):
                pending_payloads.append(self._full_canvas_mask_payload(document))
                known_mask_ids.add(self.options.full_canvas_mask_id)
                mask_aliases[self.options.full_canvas_mask_id] = self.options.full_canvas_mask_id
                full_canvas_mask_planned = True
            pending_payloads.append(payload)
            target = payload.get("target", {})
            if isinstance(target, dict):
                if isinstance(target.get("output_layer_id"), str):
                    known_layer_ids.add(target["output_layer_id"])
                if isinstance(target.get("mask_id"), str) and self._creates_or_replaces_mask(payload):
                    known_mask_ids.add(target["mask_id"])

        canonical_actions = []
        for index, payload in enumerate(pending_payloads, start=1):
            payload["id"] = f"{self.options.action_id_prefix}_{index:03d}"
            payload["created_by"] = self.options.created_by
            payload.setdefault("preconditions", self._preconditions_for(payload))
            payload.setdefault("expected_result", self._expected_result_for(payload))
            canonical_actions.append(Action.from_json(payload))

        batch = ActionBatch(
            id=f"{self.options.batch_id_prefix}_{_safe_id_fragment(user_prompt, fallback='plan')}",
            user_prompt=user_prompt,
            description=data.get("description"),
            stop_on_error=_bool_or_default(data.get("stop_on_error"), self.options.stop_on_error),
            actions=canonical_actions,
            metadata={**self.options.metadata, **_mapping_or_empty(data.get("metadata"), "planner output metadata")},
        )
        batch.validate_schema()
        return batch

    def _normalize_action(
        self,
        item: Any,
        document: DocumentState,
        known_layer_ids: set[str],
        known_mask_ids: set[str],
        layer_aliases: dict[str, str],
        mask_aliases: dict[str, str],
    ) -> JsonObject:
        """Normalize one model-facing action object."""
        if not isinstance(item, dict):
            raise TypeError("each planner action must be an object")
        _reject_unknown_keys(item, "planner action", {"type", "target", "write_mask_id", "params", "description", "metadata"})
        action_type = ActionType(_required_string(item, "type", "planner action.type"))
        spec = get_action_spec(action_type)
        target = _mapping_or_empty(item.get("target"), "planner action.target")
        params = _mapping_or_empty(item.get("params"), "planner action.params")

        normalized_target: JsonObject = {}
        for field_name, raw_value in target.items():
            if field_name not in {"document_id", "layer_id", "layer_name", "mask_id", "selection_id", "output_layer_id"}:
                raise ValueError(f"planner action.target contains unknown key {field_name!r}")
            if not isinstance(raw_value, str) or raw_value == "":
                raise ValueError(f"planner action.target.{field_name} must be a non-empty string")
            normalized_target[field_name] = self._resolve_target_value(field_name, raw_value, layer_aliases, mask_aliases)

        for field_name, mode in spec.target_fields.items():
            mode = TargetFieldMode(mode)
            if field_name in normalized_target:
                continue
            if mode == TargetFieldMode.REQUIRED:
                raise ValueError(f"{action_type.value} requires target.{field_name}")
            if mode == TargetFieldMode.DEFAULT_ACTIVE_LAYER:
                if self.options.default_to_active_layer and document.active_layer_id is not None:
                    normalized_target[field_name] = document.active_layer_id
            elif mode == TargetFieldMode.DEFAULT_ACTIVE_SELECTION:
                if document.active_selection_mask_id is not None:
                    normalized_target[field_name] = document.active_selection_mask_id
            elif mode == TargetFieldMode.GENERATED:
                generated_id = self._generated_target_id(field_name, action_type, params, known_layer_ids, known_mask_ids)
                if generated_id is not None:
                    normalized_target[field_name] = generated_id

        if "output_layer_id" in normalized_target:
            _add_aliases(target.get("output_layer_id"), normalized_target["output_layer_id"], layer_aliases)
        if self._mask_target_is_output(action_type) and "mask_id" in normalized_target:
            _add_aliases(target.get("mask_id"), normalized_target["mask_id"], mask_aliases)

        write_mask_id = item.get("write_mask_id")
        if write_mask_id is not None:
            if not isinstance(write_mask_id, str) or write_mask_id == "":
                raise ValueError("planner action.write_mask_id must be a non-empty string")
            write_mask_id = mask_aliases.get(write_mask_id, write_mask_id)
        elif TargetFieldMode(spec.write_mask) == TargetFieldMode.GENERATED:
            write_mask_id = self.options.full_canvas_mask_id

        payload: JsonObject = {
            "type": action_type.value,
            "target": normalized_target,
            "params": params,
            "metadata": _mapping_or_empty(item.get("metadata"), "planner action.metadata"),
        }
        if write_mask_id is not None:
            payload["write_mask_id"] = write_mask_id
        if item.get("description") is not None:
            payload["description"] = _string_value(item["description"], "planner action.description")
        return payload

    def _generated_target_id(
        self,
        field_name: str,
        action_type: ActionType,
        params: JsonObject,
        known_layer_ids: set[str],
        known_mask_ids: set[str],
    ) -> Optional[str]:
        """Generate an output target ID for fields the kernel owns."""
        if field_name == "output_layer_id":
            base = params.get("output_layer_name") or params.get("name") or action_type.value
            return _unique_id("layer", str(base), known_layer_ids)
        if field_name == "mask_id":
            base = params.get("name") or params.get("source_mask_id") or action_type.value
            return _unique_id("mask", str(base), known_mask_ids)
        return None

    def _resolve_target_value(
        self,
        field_name: str,
        value: str,
        layer_aliases: dict[str, str],
        mask_aliases: dict[str, str],
    ) -> str:
        """Resolve model-provided semantic IDs to generated canonical IDs."""
        if field_name in {"layer_id", "output_layer_id"}:
            return layer_aliases.get(value, value)
        if field_name in {"mask_id", "selection_id"}:
            return mask_aliases.get(value, value)
        return value

    def _needs_generated_full_canvas_mask(self, payload: JsonObject, already_planned: bool) -> bool:
        """Return whether a full-canvas write mask action should be inserted."""
        if already_planned or not self.options.auto_create_full_canvas_mask:
            return False
        return payload.get("write_mask_id") == self.options.full_canvas_mask_id

    def _full_canvas_mask_payload(self, document: DocumentState) -> JsonObject:
        """Return a canonical payload for the generated full-canvas write mask."""
        return {
            "type": ActionType.CREATE_MASK_FROM_SHAPE.value,
            "target": {"mask_id": self.options.full_canvas_mask_id},
            "params": {
                "name": "full canvas",
                "kind": "write_guard",
                "shape": {
                    "type": "rectangle",
                    "bbox_xyxy": [0, 0, document.canvas.width, document.canvas.height],
                    "corner_radius": 0,
                },
                "set_active": False,
            },
            "description": "Generated full-canvas write mask.",
            "metadata": {"generated_by": "planner_normalizer"},
        }

    def _creates_or_replaces_mask(self, payload: JsonObject) -> bool:
        """Return whether `payload` creates a mask at target.mask_id."""
        try:
            action_type = ActionType(payload["type"])
        except Exception:
            return False
        return self._mask_target_is_output(action_type)

    def _mask_target_is_output(self, action_type: ActionType) -> bool:
        """Return whether target.mask_id names a newly written mask."""
        return action_type in {
            ActionType.SELECT_RECT,
            ActionType.SELECT_ELLIPSE,
            ActionType.SELECT_POLYGON,
            ActionType.SELECT_FREEHAND,
            ActionType.SELECT_FROM_ALPHA,
            ActionType.SELECT_COLOR_RANGE,
            ActionType.MAGIC_WAND_SELECT,
            ActionType.SAVE_SELECTION_AS_MASK,
            ActionType.CREATE_MASK_FROM_SHAPE,
            ActionType.GROW_MASK,
            ActionType.SHRINK_MASK,
            ActionType.FEATHER_MASK,
            ActionType.INVERT_MASK,
            ActionType.COMBINE_MASKS,
            ActionType.REFINE_SELECTION,
            ActionType.REMOVE_SMALL_ISLANDS,
            ActionType.FILL_MASK_HOLES,
            ActionType.ADD_LAYER_MASK,
            ActionType.SEGMENT_OBJECT,
            ActionType.ESTIMATE_DEPTH,
            ActionType.EXTRACT_LINE_ART,
        }

    def _preconditions_for(self, payload: JsonObject) -> JsonObject:
        """Generate conservative action preconditions from target and params."""
        target = _mapping_or_empty(payload.get("target"), "action target")
        params = _mapping_or_empty(payload.get("params"), "action params")
        action_type = ActionType(payload["type"])
        layer_ids: list[str] = []
        mask_ids: list[str] = []

        if isinstance(target.get("layer_id"), str):
            layer_ids.append(target["layer_id"])
        for layer_id in params.get("layer_ids", []):
            if isinstance(layer_id, str):
                layer_ids.append(layer_id)

        if isinstance(payload.get("write_mask_id"), str):
            mask_ids.append(payload["write_mask_id"])
        if isinstance(params.get("source_mask_id"), str):
            mask_ids.append(params["source_mask_id"])
        for mask_id in params.get("mask_ids", []):
            if isinstance(mask_id, str):
                mask_ids.append(mask_id)

        require_active_selection = False
        if action_type == ActionType.SAVE_SELECTION_AS_MASK and "source_mask_id" not in params:
            require_active_selection = True
        if action_type == ActionType.ADD_LAYER_MASK and params.get("mode", "from_selection") == "from_selection":
            require_active_selection = True

        allow_hidden_layers = action_type in {
            ActionType.DELETE_LAYER,
            ActionType.REORDER_LAYER,
            ActionType.SET_LAYER_VISIBILITY,
            ActionType.SET_LAYER_OPACITY,
            ActionType.SET_BLEND_MODE,
            ActionType.MERGE_LAYERS,
        }
        return ActionPreconditions(
            required_layer_ids=_unique_preserving_order(layer_ids),
            required_mask_ids=_unique_preserving_order(mask_ids),
            require_active_selection=require_active_selection,
            require_unlocked_target_layer=True,
            require_write_mask=True,
            allow_hidden_layers=allow_hidden_layers,
        ).to_json()

    def _expected_result_for(self, payload: JsonObject) -> JsonObject:
        """Generate basic expected-result metadata."""
        target = _mapping_or_empty(payload.get("target"), "action target")
        params = _mapping_or_empty(payload.get("params"), "action params")
        action_type = ActionType(payload["type"])
        changed_layer_ids: list[str] = []
        created_layer_names: list[str] = []
        created_mask_names: list[str] = []
        if isinstance(target.get("layer_id"), str) and action_type not in {
            ActionType.COPY,
            ActionType.DETECT_SHAPE,
            ActionType.DETECT_OBJECTS,
        }:
            changed_layer_ids.append(target["layer_id"])
        if isinstance(target.get("output_layer_id"), str):
            created_layer_names.append(str(params.get("name") or params.get("output_layer_name") or target["output_layer_id"]))
        if self._mask_target_is_output(action_type) and isinstance(target.get("mask_id"), str):
            created_mask_names.append(str(params.get("name") or target["mask_id"]))
        return ExpectedResult(
            changed_layer_ids=_unique_preserving_order(changed_layer_ids),
            created_layer_names=created_layer_names,
            created_mask_names=created_mask_names,
        ).to_json()


@dataclass(slots=True)
class AIPlanner:
    """Coordinate planner requests, normalization, tracing, and execution."""

    backend: PlannerBackend
    request_builder: PlannerRequestBuilder = field(default_factory=PlannerRequestBuilder)
    normalizer: ActionBatchNormalizer = field(default_factory=ActionBatchNormalizer)
    options: PlannerOptions = field(default_factory=PlannerOptions)
    trace_sink: Optional[Any] = None

    def __post_init__(self) -> None:
        """Keep planner and normalizer options aligned."""
        self.normalizer.options = self.options

    def plan(
        self,
        user_prompt: str,
        document: DocumentState,
        *,
        observations: Optional[list[Any]] = None,
        asset_refs: Optional[dict[str, str]] = None,
        metadata: Optional[JsonObject] = None,
    ) -> PlannerResult:
        """Ask the backend for planner output and normalize it into an action batch."""
        previous_errors: list[JsonObject] = []
        attempts: list[JsonObject] = []
        last_request: JsonObject = {}
        last_raw_output: JsonObject | str = {}

        for attempt_index in range(self.options.max_schema_retries + 1):
            request = self.request_builder.build(
                user_prompt,
                document,
                observations=observations,
                asset_refs=asset_refs,
                previous_errors=previous_errors,
                metadata=metadata,
            )
            last_request = request
            _log_if_available(self.trace_sink, "log_planner_input", document, request, asset_refs or {})
            raw_output = self.backend.plan(request)
            last_raw_output = raw_output
            try:
                batch = self.normalizer.normalize(raw_output, document, user_prompt)
            except Exception as exc:
                error = {
                    "attempt": attempt_index + 1,
                    "stage": "normalize",
                    "message": str(exc),
                }
                previous_errors.append(error)
                attempts.append({"request": request, "raw_output": _json_safe(raw_output), "error": error})
                _log_if_available(
                    self.trace_sink,
                    "log_planner_output_raw",
                    document,
                    raw_output,
                    "invalid",
                    [error],
                )
                continue

            _log_if_available(self.trace_sink, "log_planner_output_raw", document, raw_output, "parsed", [])
            return PlannerResult(
                request=request,
                raw_output=raw_output,
                action_batch=batch,
                attempts=[*attempts, {"request": request, "raw_output": _json_safe(raw_output), "error": None}],
                metadata={"schema_retry_count": attempt_index},
            )

        raise PlannerError(
            "planner output could not be normalized",
            [*previous_errors, {"stage": "final", "request": last_request, "raw_output": _json_safe(last_raw_output)}],
        )

    def plan_and_execute(
        self,
        user_prompt: str,
        document: DocumentState,
        executor: Executor,
        *,
        observations: Optional[list[Any]] = None,
        asset_refs: Optional[dict[str, str]] = None,
        metadata: Optional[JsonObject] = None,
    ) -> PlannerExecutionResult:
        """Plan an action batch, log it when possible, and execute it."""
        planner_result = self.plan(
            user_prompt,
            document,
            observations=observations,
            asset_refs=asset_refs,
            metadata=metadata,
        )
        _log_if_available(self.trace_sink, "log_action_batch_planned", planner_result.action_batch, document)
        action_results = executor.execute_batch(document, planner_result.action_batch)
        return PlannerExecutionResult(planner_result=planner_result, action_results=action_results)


def _coerce_json_object(raw_output: JsonObject | str) -> JsonObject:
    """Return a dictionary from a backend response."""
    if isinstance(raw_output, str):
        parsed = json.loads(raw_output)
    else:
        parsed = raw_output
    if not isinstance(parsed, dict):
        raise TypeError("planner output must be a JSON object")
    return dict(parsed)


def _reject_unknown_keys(data: JsonObject, field_name: str, allowed: set[str]) -> None:
    """Reject unknown object keys with a clear error."""
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{field_name} contains unknown keys {unknown!r}")


def _required_string(data: JsonObject, key: str, field_name: str) -> str:
    """Return a required non-empty string."""
    if key not in data:
        raise ValueError(f"{field_name} is required")
    return _string_value(data[key], field_name)


def _string_value(value: Any, field_name: str) -> str:
    """Validate and return a non-empty string."""
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _mapping_or_empty(value: Any, field_name: str) -> JsonObject:
    """Return a mapping, treating None as an empty mapping."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)


def _bool_or_default(value: Any, default: bool) -> bool:
    """Validate an optional boolean."""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError("stop_on_error must be a boolean")
    return value


def _unique_id(prefix: str, base: str, existing_ids: set[str]) -> str:
    """Return a stable generated ID not present in `existing_ids`."""
    root = f"{prefix}_{_safe_id_fragment(base, fallback=prefix)}"
    candidate = root
    index = 2
    while candidate in existing_ids:
        candidate = f"{root}_{index}"
        index += 1
    existing_ids.add(candidate)
    return candidate


def _safe_id_fragment(value: str, *, fallback: str) -> str:
    """Return a conservative ID fragment from user/model text."""
    fragment = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    fragment = "_".join(part for part in fragment.split("_") if part)
    return fragment or fallback


def _add_aliases(raw_alias: Any, canonical_id: str, aliases: dict[str, str]) -> None:
    """Record raw planner aliases for later target resolution."""
    aliases[canonical_id] = canonical_id
    if isinstance(raw_alias, str) and raw_alias:
        aliases[raw_alias] = canonical_id


def _unique_preserving_order(values: list[str]) -> list[str]:
    """Return unique strings while preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _log_if_available(trace_sink: Any, method_name: str, *args: Any) -> None:
    """Call a trace method when the active sink supports it."""
    if trace_sink is None:
        return
    method = getattr(trace_sink, method_name, None)
    if method is not None:
        method(*args)


def _json_safe(value: Any) -> Any:
    """Return JSON-safe data for planner diagnostics."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value
