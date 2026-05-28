"""Bridge between service edit requests and the AI Editing Kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ai_edit_kernel.document.document_state import CanvasSpec, DocumentMetadata, DocumentState
from ai_edit_kernel.document.layer import BlendMode, Layer, LayerKind
from ai_edit_kernel.document.mask import Mask, MaskKind
from ai_edit_kernel.planning import AIPlanner
from ai_edit_kernel.planning.planner import ActionBatchNormalizer, PlannerBackend
from ai_edit_kernel.runtime.executor import DiffusionBackend, ExecutionContext, Executor
from ai_edit_kernel.schema.actions import ActionBatch
from ai_edit_kernel.trace.trace_logger import TraceLogger
from ai_edit_service.assets import AssetStore
from ai_edit_service.models import EditRequest, EditResult, JobStatus, PayloadEncoding, ResultLayer, TraceRef
from ai_edit_service.payloads import decode_image_payload, decode_mask_payload, encode_image_payload, encode_mask_payload, encode_png_bytes


@dataclass(slots=True)
class KernelRunnerOptions:
    """Policy for converting service requests into kernel executions."""

    default_document_id: str = "service_document"
    selection_mask_id: str = "host_selection"
    result_layer_prefix: str = "result"
    include_flat_preview: bool = True
    include_trace_refs: bool = True
    output_encoding: str = "png_base64"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KernelRunner:
    """Execute host-neutral edit requests against the kernel.

    This class is intentionally the only service layer that imports
    `ai_edit_kernel`. HTTP, jobs, and assets stay generic so Photoshop, Krita,
    GIMP, Canva, tests, or CLI tools can all use the same service protocol.
    """

    planner_backend: Optional[PlannerBackend] = None
    diffusion_backend: Optional[DiffusionBackend] = None
    asset_store: Optional[AssetStore] = None
    trace_root: Optional[Path] = None
    options: KernelRunnerOptions = field(default_factory=KernelRunnerOptions)

    def run_edit(self, request: EditRequest) -> EditResult:
        """Build a kernel document, execute an edit, and return host results."""
        document = self.build_document(request)
        logger = self.create_trace_logger(request)
        if logger is not None:
            logger.start_session(request.prompt)
            logger.log_document_snapshot(document, "initial")

        action_batch = self.plan_actions(request, document, logger)
        executor = self.create_executor(logger)
        action_results = executor.execute_batch(document, action_batch)

        if logger is not None:
            logger.log_document_snapshot(document, "final")
            session = logger.end_session()
        else:
            session = None

        return self.build_result(request, document, action_results, session)

    def build_document(self, request: EditRequest) -> DocumentState:
        """Convert service layers and masks into a kernel `DocumentState`."""
        layers: list[Layer] = []
        masks: dict[str, Mask] = {}
        for item in request.layers:
            layer_mask_id = None
            if item.mask is not None:
                layer_mask_id = f"{item.id}_mask"
                masks[layer_mask_id] = Mask(
                    id=layer_mask_id,
                    name=item.mask.name or f"{item.name} mask",
                    data=decode_mask_payload(item.mask, self.resolve_asset),
                    kind=MaskKind.LAYER_ALPHA,
                    hard=False,
                    source=f"host_layer:{item.id}",
                    metadata=dict(item.mask.metadata),
                )
            layers.append(
                Layer(
                    id=item.id,
                    name=item.name,
                    kind=self._layer_kind(item.kind),
                    pixels=decode_image_payload(item.image, self.resolve_asset),
                    opacity=float(item.opacity),
                    visible=bool(item.visible),
                    blend_mode=self._blend_mode(item.blend_mode),
                    mask_id=layer_mask_id,
                    metadata={**dict(item.metadata), "host_layer_id": item.id},
                )
            )

        active_selection_mask_id = None
        if request.selection_mask is not None:
            active_selection_mask_id = self.options.selection_mask_id
            masks[active_selection_mask_id] = Mask(
                id=active_selection_mask_id,
                name=request.selection_mask.name or "host selection",
                data=decode_mask_payload(request.selection_mask, self.resolve_asset),
                kind=MaskKind.SELECTION,
                hard=False,
                source="host_selection",
                metadata=dict(request.selection_mask.metadata),
            )

        document = DocumentState(
            id=request.host.document_id or self.options.default_document_id,
            canvas=CanvasSpec(width=request.canvas_width, height=request.canvas_height, color_space=request.color_space),
            layers=layers,
            masks=masks,
            active_layer_id=request.active_layer_id or (layers[-1].id if layers else None),
            active_selection_mask_id=active_selection_mask_id,
            metadata=DocumentMetadata(
                title=request.host.document_name,
                source_file=request.asset_refs.get("source_file"),
                tags=["service_request", request.host.host_name],
                custom={
                    "request_id": request.request_id,
                    "host": request.host.to_json(),
                    "service_metadata": dict(request.metadata),
                },
            ),
        )
        document.validate()
        return document

    def plan_actions(self, request: EditRequest, document: DocumentState, trace_sink: Optional[TraceLogger]) -> ActionBatch:
        """Return the action batch for this request.

        Early adapters can pass an `action_batch` or `planner_output` object in
        `request.options`. The production path should supply `planner_backend`
        so the service can ask a model/backend for planner JSON.
        """
        if "action_batch" in request.options:
            return ActionBatch.from_json(request.options["action_batch"])
        if "planner_output" in request.options:
            return ActionBatchNormalizer().normalize(request.options["planner_output"], document, request.prompt)
        if self.planner_backend is None:
            raise RuntimeError("KernelRunner requires planner_backend, request.options.action_batch, or request.options.planner_output")

        planner_asset_refs = dict(request.asset_refs)
        current_image_ref = self.store_current_image_for_planner(document)
        if current_image_ref is not None:
            planner_asset_refs["current_image"] = current_image_ref
            planner_asset_refs["current_image_role"] = "attached_vision_input"

        planner = AIPlanner(backend=self.planner_backend, trace_sink=trace_sink)
        result = planner.plan(
            request.prompt,
            document,
            observations=request.observations,
            asset_refs=planner_asset_refs,
            metadata={"host": request.host.to_json(), "request_id": request.request_id},
        )
        if trace_sink is not None:
            trace_sink.log_action_batch_planned(result.action_batch, document)
        return result.action_batch

    def create_executor(self, trace_sink: Optional[TraceLogger]) -> Executor:
        """Create the kernel executor with service-owned dependencies."""
        return Executor(ExecutionContext(diffusion_backend=self.diffusion_backend, trace_sink=trace_sink, asset_store=self.asset_store))

    def store_current_image_for_planner(self, document: DocumentState) -> Optional[str]:
        """Store the current visual state as an asset for vision-capable planners."""
        if self.asset_store is None:
            return None
        try:
            pixels = document.flatten_preview()
            source = "flattened_preview"
        except NotImplementedError:
            active = document.get_layer(document.active_layer_id) if document.active_layer_id is not None else None
            if active is None or active.pixels is None:
                return None
            pixels = active.pixels
            source = "active_layer"
        record = self.asset_store.put_bytes(
            encode_png_bytes(pixels),
            suffix=".png",
            media_type="image/png",
            metadata={"kind": "planner_current_image", "source": source, "document_id": document.id},
        )
        return f"asset:{record.id}"

    def create_trace_logger(self, request: EditRequest) -> Optional[TraceLogger]:
        """Create a trace logger for this request when tracing is configured."""
        if self.trace_root is None:
            return None
        return TraceLogger(
            self.trace_root,
            metadata={
                "source": "ai_edit_service",
                "request_id": request.request_id,
                "host": request.host.to_json(),
                "service_options": self.options.metadata,
            },
        )

    def build_result(self, request: EditRequest, document: DocumentState, action_results: list[Any], session: Any) -> EditResult:
        """Convert final kernel document state into host-neutral result payloads."""
        succeeded = all(result.succeeded() for result in action_results if hasattr(result, "succeeded"))
        changed_layer_ids = self._changed_or_created_layer_ids(action_results)
        if not changed_layer_ids and document.active_layer_id is not None:
            changed_layer_ids = [document.active_layer_id]

        result_layers: list[ResultLayer] = []
        for index, layer_id in enumerate(changed_layer_ids, start=1):
            layer = document.get_layer(layer_id)
            if layer is None or layer.pixels is None:
                continue
            result_layers.append(
                ResultLayer(
                    id=f"{self.options.result_layer_prefix}_{index:03d}_{layer.id}",
                    name=layer.name,
                    image=encode_image_payload(layer.pixels, encoding=self._output_encoding(), color_space=request.color_space),
                    opacity=layer.opacity,
                    blend_mode=layer.blend_mode.value,
                    mask=None if layer.mask_id is None else self._result_mask(document, layer.mask_id),
                    metadata={"source_layer_id": layer.id},
                )
            )

        preview = None
        if self.options.include_flat_preview:
            try:
                preview = encode_image_payload(
                    document.flatten_preview(),
                    encoding=self._output_encoding(),
                    color_space=request.color_space,
                )
            except NotImplementedError:
                preview = None

        return EditResult(
            request_id=request.request_id,
            status=JobStatus.SUCCEEDED if succeeded else JobStatus.FAILED,
            layers=result_layers,
            preview=preview,
            trace=self._trace_ref(session),
            action_results=[result.to_json() if hasattr(result, "to_json") else {"result": str(result)} for result in action_results],
            document_summary=document.snapshot_summary(),
            metadata={"host": request.host.to_json()},
        )

    def resolve_asset(self, asset_id: str) -> bytes:
        """Read an asset referenced by an incoming payload."""
        if self.asset_store is None:
            raise ValueError("asset references require an AssetStore")
        return self.asset_store.get_bytes(asset_id)

    def _result_mask(self, document: DocumentState, mask_id: str):
        mask = document.masks.get(mask_id)
        if mask is None:
            return None
        return encode_mask_payload(mask.data, name=mask.name)

    def _trace_ref(self, session: Any) -> Optional[TraceRef]:
        if session is None or not self.options.include_trace_refs:
            return None
        trace_dir = self.trace_root / session.id if self.trace_root is not None else None
        return TraceRef(
            session_id=getattr(session, "id", None),
            local_trace_dir=None if trace_dir is None else str(trace_dir),
        )

    def _output_encoding(self) -> PayloadEncoding:
        return PayloadEncoding(self.options.output_encoding)

    def _changed_or_created_layer_ids(self, action_results: list[Any]) -> list[str]:
        output: list[str] = []
        for result in action_results:
            for layer_id in getattr(result, "changed_layer_ids", []):
                if layer_id not in output:
                    output.append(layer_id)
            for layer_id in getattr(result, "created_layer_ids", []):
                if layer_id not in output:
                    output.append(layer_id)
        return output

    def _layer_kind(self, value: str) -> LayerKind:
        try:
            return LayerKind(value)
        except ValueError:
            return LayerKind.RASTER

    def _blend_mode(self, value: str) -> BlendMode:
        try:
            return BlendMode(value)
        except ValueError:
            return BlendMode.NORMAL
