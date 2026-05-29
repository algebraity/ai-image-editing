# AI Editing Kernel JSON Formats

This document defines the prototype JSON contracts used by the AI Editing
Kernel. These formats are intentionally explicit and ID-based so sessions can be
replayed, validated, and converted into training data.

Version identifiers:

- Trace format: `ai_edit_trace.v1`
- Action batch format: `ai_edit_actions.v1`
- Document summary format: `ai_edit_document.v1`
- Training example format: `ai_edit_training_example.v1`
- Tool catalog format: `tools.v1`

Coordinates use pixel space with origin at the top-left of the canvas. Shape
bounding boxes use `bbox_xyxy`: `[x0, y0, x1, y1]`, where `x1` and `y1` are
excluded, matching Python slice semantics.

## Trace Directory

Each session is stored as one directory:

```text
session_2026_05_27_abc123/
  manifest.json
  events.jsonl
  previews/
  layers/
  masks/
  snapshots/
  diffusion/
  artifacts/
```

`manifest.json` is the session index. `events.jsonl` is the ordered replay log.
Asset folders contain files referenced by manifest and event `asset_refs`.

## Manifest Format

Required fields should be present for every saved session. Optional fields may be
`null` or omitted when the information is not available.

```json
{
  "schema_version": "ai_edit_trace.v1",
  "session_id": "session_2026_05_27_abc123",
  "created_at": "2026-05-27T14:32:10.441Z",
  "ended_at": "2026-05-27T14:32:27.983Z",
  "user_prompt": "Change the icon from a circle to a square.",
  "task_type": "shape_replacement",
  "status": "completed",
  "success": true,
  "kernel": {
    "kernel_version": "0.1.0",
    "action_schema_version": "ai_edit_actions.v1",
    "document_schema_version": "ai_edit_document.v1",
    "tool_catalog_version": "planner_tools.v1"
  },
  "environment": {
    "python_version": "3.12",
    "platform": "local",
    "deterministic_mode": true,
    "random_seed": 123456
  },
  "planner": {
    "planner_type": "manual",
    "planner_model": null,
    "planner_version": null,
    "temperature": null
  },
  "assets": {
    "root": ".",
    "previews_dir": "previews",
    "layers_dir": "layers",
    "masks_dir": "masks",
    "snapshots_dir": "snapshots",
    "diffusion_dir": "diffusion",
    "artifacts_dir": "artifacts"
  },
  "summary": {
    "initial_snapshot": "snapshots/doc_rev_0000.json",
    "final_snapshot": "snapshots/doc_rev_0004.json",
    "initial_preview": "previews/doc_rev_0000.png",
    "final_preview": "previews/doc_rev_0004.png",
    "action_count": 6,
    "validation_passed": true,
    "human_rating": null
  },
  "rights": {
    "source_image_license": "unknown",
    "user_provided_content": true,
    "allowed_for_training": true,
    "contains_personal_data": false
  },
  "metadata": {
    "notes": "Prototype circle-to-square demo."
  }
}
```

`status` is one of `running`, `completed`, `failed`, or `aborted`.

## Event Envelope

Every line in `events.jsonl` is one JSON object with this envelope:

```json
{
  "event_id": "evt_000001",
  "type": "user_prompt",
  "timestamp": "2026-05-27T14:32:10.451Z",
  "document_id": "doc_abc123",
  "document_revision_before": null,
  "document_revision_after": 0,
  "action_id": null,
  "payload": {},
  "asset_refs": {},
  "metadata": {}
}
```

`payload` contains event-specific structured data. `asset_refs` maps short names
to paths relative to the trace directory.

## Event Payloads

### `session_started`

```json
{
  "event_id": "evt_000001",
  "type": "session_started",
  "timestamp": "2026-05-27T14:32:10.441Z",
  "document_id": null,
  "document_revision_before": null,
  "document_revision_after": null,
  "action_id": null,
  "payload": {
    "session_id": "session_2026_05_27_abc123",
    "schema_version": "ai_edit_trace.v1"
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `user_prompt`

```json
{
  "event_id": "evt_000002",
  "type": "user_prompt",
  "timestamp": "2026-05-27T14:32:10.451Z",
  "document_id": "doc_abc123",
  "document_revision_before": null,
  "document_revision_after": 0,
  "action_id": null,
  "payload": {
    "prompt": "Change the icon from a circle to a square."
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `document_snapshot`

```json
{
  "event_id": "evt_000003",
  "type": "document_snapshot",
  "timestamp": "2026-05-27T14:32:10.500Z",
  "document_id": "doc_abc123",
  "document_revision_before": 0,
  "document_revision_after": 0,
  "action_id": null,
  "payload": {
    "label": "initial",
    "document_summary": {
      "schema_version": "ai_edit_document.v1",
      "id": "doc_abc123",
      "revision": 0,
      "canvas": {
        "width": 1024,
        "height": 1024,
        "color_space": "srgb"
      },
      "active_layer_id": "layer_icon",
      "active_selection_mask_id": null,
      "layers": [],
      "masks": []
    }
  },
  "asset_refs": {
    "snapshot": "snapshots/doc_rev_0000.json",
    "preview": "previews/doc_rev_0000.png"
  },
  "metadata": {}
}
```

### `observation`

```json
{
  "event_id": "evt_000004",
  "type": "observation",
  "timestamp": "2026-05-27T14:32:11.000Z",
  "document_id": "doc_abc123",
  "document_revision_before": 0,
  "document_revision_after": 0,
  "action_id": null,
  "payload": {
    "label": "detected_shapes",
    "observations": [
      {
        "id": "shape_circle_001",
        "type": "circle",
        "center": [512.0, 512.0],
        "radius": 312.0,
        "bbox_xyxy": [200.0, 200.0, 824.0, 824.0],
        "stroke_width_estimate": 24.0
      }
    ]
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `planner_input`

```json
{
  "event_id": "evt_000005",
  "type": "planner_input",
  "timestamp": "2026-05-27T14:32:11.100Z",
  "document_id": "doc_abc123",
  "document_revision_before": 0,
  "document_revision_after": 0,
  "action_id": null,
  "payload": {
    "schema_version": "ai_edit_planner_request.v1",
    "user_prompt": "Change the icon from a circle to a square.",
    "document_summary": {},
    "observations": [],
    "tool_catalog_version": "planner_tools.v1",
    "available_actions": [
      {
        "name": "draw_shape",
        "category": "paint",
        "summary": "Draw a deterministic rectangle or ellipse.",
        "planner_schema": {
          "type": "object",
          "required": ["type", "params"],
          "properties": {
            "type": {"const": "draw_shape"},
            "target": {
              "type": "object",
              "properties": {
                "layer_id": {
                  "type": "string",
                  "description": "Optional; defaults to the active layer when omitted."
                }
              }
            },
            "write_mask_id": {
              "type": "string",
              "description": "Optional; generated as a full-canvas write mask when omitted."
            },
            "params": {}
          }
        },
        "kernel_filled_fields": ["id", "created_by", "preconditions", "expected_result"]
      }
    ],
    "output_contract": {},
    "previous_errors": []
  },
  "asset_refs": {
    "visible_preview": "previews/doc_rev_0000.png"
  },
  "metadata": {}
}
```

Planner backends return the smaller schema `ai_edit_planner_output.v1`, not a
full trace and not a canonical action batch. The planner layer normalizes this
into `ai_edit_actions.v1` by generating action IDs, output IDs when omitted,
preconditions, expected-result metadata, and default write masks.

```json
{
  "schema_version": "ai_edit_planner_output.v1",
  "description": "Create a border layer and draw a black rectangle.",
  "stop_on_error": true,
  "actions": [
    {
      "type": "create_layer",
      "target": {
        "output_layer_id": "layer_border"
      },
      "params": {
        "name": "border",
        "color": "#00000000"
      }
    },
    {
      "type": "draw_shape",
      "target": {
        "layer_id": "layer_border"
      },
      "params": {
        "shape": {
          "type": "rectangle",
          "bbox_xyxy": [128, 128, 896, 896],
          "corner_radius": 0
        },
        "stroke": {
          "color": "#000000",
          "width": 24
        },
        "fill": null
      }
    }
  ],
  "metadata": {}
}
```

If a generated layer or mask will be used by later planner actions, the model
should provide a semantic `target.output_layer_id` or `target.mask_id` and reuse
that same string later. The planner treats those strings as aliases and carries
them into the canonical action batch.

### `planner_output_raw`

```json
{
  "event_id": "evt_000006",
  "type": "planner_output_raw",
  "timestamp": "2026-05-27T14:32:11.600Z",
  "document_id": "doc_abc123",
  "document_revision_before": 0,
  "document_revision_after": 0,
  "action_id": null,
  "payload": {
    "raw_text": null,
    "raw_json": {},
    "parser_status": "parsed",
    "errors": []
  },
  "asset_refs": {},
  "metadata": {}
}
```

Do not store hidden reasoning, private system prompts, API secrets, or unrelated
personal data in this event.

### `action_batch_planned`

```json
{
  "event_id": "evt_000007",
  "type": "action_batch_planned",
  "timestamp": "2026-05-27T14:32:11.700Z",
  "document_id": "doc_abc123",
  "document_revision_before": 0,
  "document_revision_after": 0,
  "action_id": null,
  "payload": {
    "action_batch": {
      "schema_version": "ai_edit_actions.v1",
      "id": "batch_001",
      "stop_on_error": true,
      "actions": []
    }
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `action_started`

```json
{
  "event_id": "evt_000008",
  "type": "action_started",
  "timestamp": "2026-05-27T14:32:12.000Z",
  "document_id": "doc_abc123",
  "document_revision_before": 0,
  "document_revision_after": 0,
  "action_id": "action_001",
  "payload": {
    "action": {}
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `action_result`

```json
{
  "event_id": "evt_000009",
  "type": "action_result",
  "timestamp": "2026-05-27T14:32:12.100Z",
  "document_id": "doc_abc123",
  "document_revision_before": 0,
  "document_revision_after": 1,
  "action_id": "action_001",
  "payload": {
    "result": {
      "action_id": "action_001",
      "status": "executed",
      "document_id": "doc_abc123",
      "before_revision": 0,
      "after_revision": 1,
      "created_layer_ids": [],
      "created_mask_ids": [],
      "changed_layer_ids": ["layer_icon"],
      "output_assets": {},
      "metadata": {}
    }
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `validation_report`

```json
{
  "event_id": "evt_000010",
  "type": "validation_report",
  "timestamp": "2026-05-27T14:32:12.150Z",
  "document_id": "doc_abc123",
  "document_revision_before": 1,
  "document_revision_after": 1,
  "action_id": "action_001",
  "payload": {
    "report": {
      "passed": true,
      "issues": [],
      "metrics": {
        "max_protected_delta": 0.0
      },
      "metadata": {}
    }
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `diffusion_job_started`

```json
{
  "event_id": "evt_000011",
  "type": "diffusion_job_started",
  "timestamp": "2026-05-27T14:32:13.000Z",
  "document_id": "doc_abc123",
  "document_revision_before": 2,
  "document_revision_after": 2,
  "action_id": "action_inpaint_001",
  "payload": {
    "job_id": "diffusion_job_001",
    "operation": "inpaint",
    "backend": "example_backend",
    "seed": 123456,
    "params": {
      "prompt": "replace damaged pixels",
      "denoise": 0.65
    }
  },
  "asset_refs": {
    "source_preview": "diffusion/job_001/source.png",
    "mask": "diffusion/job_001/mask.png"
  },
  "metadata": {}
}
```

### `diffusion_job_result`

```json
{
  "event_id": "evt_000012",
  "type": "diffusion_job_result",
  "timestamp": "2026-05-27T14:32:16.000Z",
  "document_id": "doc_abc123",
  "document_revision_before": 2,
  "document_revision_after": 2,
  "action_id": "action_inpaint_001",
  "payload": {
    "job_id": "diffusion_job_001",
    "status": "completed",
    "metrics": {}
  },
  "asset_refs": {
    "generated_image": "diffusion/job_001/generated.png"
  },
  "metadata": {}
}
```

### `human_feedback`

```json
{
  "event_id": "evt_000013",
  "type": "human_feedback",
  "timestamp": "2026-05-27T14:32:20.000Z",
  "document_id": "doc_abc123",
  "document_revision_before": 4,
  "document_revision_after": 4,
  "action_id": null,
  "payload": {
    "accepted": true,
    "rating": {
      "overall": 4,
      "goal_satisfaction": 5,
      "mask_accuracy": 5,
      "visual_quality": 4,
      "layer_structure": 5
    },
    "comments": null
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `session_ended`

```json
{
  "event_id": "evt_000014",
  "type": "session_ended",
  "timestamp": "2026-05-27T14:32:27.983Z",
  "document_id": "doc_abc123",
  "document_revision_before": 4,
  "document_revision_after": 4,
  "action_id": null,
  "payload": {
    "session_id": "session_2026_05_27_abc123",
    "status": "completed",
    "success": true,
    "summary": {
      "action_count": 6,
      "validation_passed": true
    }
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `error`

```json
{
  "event_id": "evt_000015",
  "type": "error",
  "timestamp": "2026-05-27T14:32:18.000Z",
  "document_id": "doc_abc123",
  "document_revision_before": 3,
  "document_revision_after": 3,
  "action_id": "action_003",
  "payload": {
    "code": "execution.exception",
    "message": "target layer 'layer_missing' does not exist",
    "details": {}
  },
  "asset_refs": {},
  "metadata": {}
}
```

### `note`

```json
{
  "event_id": "evt_000016",
  "type": "note",
  "timestamp": "2026-05-27T14:32:18.100Z",
  "document_id": "doc_abc123",
  "document_revision_before": 3,
  "document_revision_after": 3,
  "action_id": null,
  "payload": {
    "message": "Manual checkpoint before export.",
    "details": {}
  },
  "asset_refs": {},
  "metadata": {}
}
```

## Action Batch Envelope

```json
{
  "schema_version": "ai_edit_actions.v1",
  "id": "batch_001",
  "user_prompt": "Draw a square border.",
  "description": "Prototype deterministic edit batch.",
  "stop_on_error": true,
  "actions": [],
  "metadata": {}
}
```

Every action uses this envelope:

```json
{
  "id": "action_001",
  "type": "draw_shape",
  "target": {
    "layer_id": "layer_border"
  },
  "write_mask_id": "mask_full_canvas",
  "params": {},
  "preconditions": {
    "required_layer_ids": ["layer_border"],
    "required_mask_ids": ["mask_full_canvas"],
    "require_active_layer": false,
    "require_active_selection": false,
    "require_unlocked_target_layer": true,
    "require_write_mask": true,
    "allow_hidden_layers": false,
    "custom": {}
  },
  "expected_result": {
    "changed_layer_ids": ["layer_border"],
    "created_layer_names": [],
    "created_mask_names": [],
    "protected_mask_id": null,
    "geometry_expectations": {},
    "visual_expectations": {},
    "custom": {}
  },
  "description": "Draw a black rectangular border.",
  "created_by": "planner",
  "metadata": {}
}
```

## Prototype Action Formats

These are the executable prototype action types.

### `new_document`

```json
{
  "id": "action_new_document",
  "type": "new_document",
  "target": {
    "document_id": "doc_new"
  },
  "params": {
    "width": 1024,
    "height": 1024,
    "color_space": "srgb",
    "background_color": "#00000000",
    "dpi": null,
    "title": "Untitled"
  }
}
```

`new_document` resets the current in-memory `DocumentState` to an empty canvas.
It clears layers, masks, active references, and annotations, replaces document
metadata from the supplied params, and starts the new document at revision `1`
after executor revision handling. If `target.document_id` is omitted, the
existing document ID is preserved.

### `resize_canvas`

```json
{
  "id": "action_resize_canvas",
  "type": "resize_canvas",
  "params": {
    "width": 1280,
    "height": 960,
    "anchor": "center",
    "fill_color": "#00000000"
  }
}
```

The prototype executor resizes every layer pixel array and every mask around the
canvas center. Expanded layer pixels are filled with `fill_color`, which defaults
to transparent black. Expanded mask pixels are filled with `0.0`.

### `crop`

```json
{
  "id": "action_crop_document",
  "type": "crop",
  "params": {
    "bbox_xyxy": [64, 64, 704, 704],
    "scope": "document"
  }
}
```

`scope` is `document`, `layer`, or `mask`. A document crop changes the canvas
size and crops every layer and mask. A layer crop keeps the canvas size and
fills pixels outside the bounding box with `fill_color`, which defaults to
transparent black. A mask crop keeps the canvas size and sets mask values outside
the bounding box to `0.0`.

### `import_image_as_layer`

```json
{
  "id": "action_import_image",
  "type": "import_image_as_layer",
  "target": {
    "output_layer_id": "layer_source"
  },
  "params": {
    "path": "inputs/source.png",
    "name": "source image",
    "x": 64,
    "y": 64,
    "opacity": 1.0,
    "blend_mode": "normal",
    "set_active": true
  }
}
```

The image is loaded as RGBA and placed into a full-canvas raster layer at
integer pixel offset `[x, y]`. The imported image must fit within the current
canvas.

### `import_vector_as_raster`

```json
{
  "id": "action_import_vector",
  "type": "import_vector_as_raster",
  "target": {
    "output_layer_id": "layer_logo"
  },
  "params": {
    "path": "inputs/logo.svg",
    "name": "logo",
    "x": 64,
    "y": 64,
    "width": 512,
    "height": 512,
    "opacity": 1.0,
    "blend_mode": "normal",
    "set_active": true,
    "background_color": null
  }
}
```

The vector asset is rasterized into straight-alpha RGBA pixels and placed into a
full-canvas raster layer at integer pixel offset `[x, y]`. The rasterized image
must fit within the current canvas. `width` and `height` are optional positive
integer output dimensions; if one dimension is omitted, SVG aspect ratio is
preserved. `background_color`, when supplied, composites the rasterized result
over that color before insertion.

The prototype supports SVG inputs. If CairoSVG is installed, it is used as the
renderer. Otherwise the kernel uses a conservative built-in SVG renderer for
basic shapes, simple line paths, solid fills and strokes, opacity, and common
transforms. The resulting layer is always raster; native vector editing is not
part of this action.

### `rasterize_vector_asset`

```json
{
  "id": "action_rasterize_vector",
  "type": "rasterize_vector_asset",
  "params": {
    "path": "inputs/logo.svg",
    "output_path": "artifacts/logo.png",
    "width": 512,
    "height": 512,
    "background_color": "#ffffff"
  }
}
```

The vector asset is rasterized to a standalone artifact without changing the
document. `output_path` must end in `.png` or `.npy`. PNG outputs store an RGBA
image; NPY outputs store the kernel's float RGBA pixel array. The same SVG
renderer rules and optional `background_color` compositing behavior used by
`import_vector_as_raster` apply here.

### `create_layer`

```json
{
  "id": "action_create_layer",
  "type": "create_layer",
  "target": {
    "output_layer_id": "layer_border"
  },
  "params": {
    "name": "border",
    "kind": "raster",
    "color": "#00000000",
    "opacity": 1.0,
    "blend_mode": "normal",
    "insert_index": null,
    "set_active": true
  }
}
```

Optional `color_rgba` may be used instead of `color`:

```json
"color_rgba": [0.0, 0.0, 0.0, 0.0]
```

Prototype layer `width` and `height`, if supplied, must match the document
canvas.

### `delete_layer`

```json
{
  "id": "action_delete_layer",
  "type": "delete_layer",
  "target": {
    "layer_id": "layer_draft"
  },
  "params": {}
}
```

The layer is removed from the document stack. Document invariants clear stale
active-layer and group-child references when necessary.

### `duplicate_layer`

```json
{
  "id": "action_duplicate_layer",
  "type": "duplicate_layer",
  "target": {
    "layer_id": "layer_source",
    "output_layer_id": "layer_source_copy"
  },
  "params": {
    "name": "source copy",
    "insert_index": 2,
    "set_active": true
  }
}
```

Duplication creates an independent deep copy of the source layer, including
pixel data and layer metadata.

### `rename_layer`

```json
{
  "id": "action_rename_layer",
  "type": "rename_layer",
  "target": {
    "layer_id": "layer_source"
  },
  "params": {
    "name": "clean source"
  }
}
```

The layer ID remains stable; only the display name changes.

### `reorder_layer`

```json
{
  "id": "action_reorder_layer",
  "type": "reorder_layer",
  "target": {
    "layer_id": "layer_border"
  },
  "params": {
    "index": 3
  }
}
```

Layer stack indices are bottom-to-top and zero-based.

### `set_active_layer`

```json
{
  "id": "action_set_active_layer",
  "type": "set_active_layer",
  "target": {
    "layer_id": "layer_border"
  },
  "params": {}
}
```

### `set_layer_visibility`

```json
{
  "id": "action_hide_reference",
  "type": "set_layer_visibility",
  "target": {
    "layer_id": "layer_reference"
  },
  "params": {
    "visible": false
  }
}
```

Visibility controls preview compositing and visible-layer merge operations.

### `set_layer_opacity`

```json
{
  "id": "action_set_opacity",
  "type": "set_layer_opacity",
  "target": {
    "layer_id": "layer_overlay"
  },
  "params": {
    "opacity": 0.5
  }
}
```

Opacity is a unit value in `[0, 1]`.

### `set_blend_mode`

```json
{
  "id": "action_set_blend_mode",
  "type": "set_blend_mode",
  "target": {
    "layer_id": "layer_shadow"
  },
  "params": {
    "blend_mode": "normal"
  }
}
```

Supported blend-mode metadata values are `normal`, `multiply`, `screen`,
`overlay`, `add`, and `subtract`. The current merge executor composites only
`normal` layers.

### `merge_layers`

```json
{
  "id": "action_merge_down",
  "type": "merge_layers",
  "target": {
    "layer_id": "layer_highlight"
  },
  "params": {
    "mode": "down",
    "output_layer_name": "merged highlight"
  }
}
```

```json
{
  "id": "action_merge_selected",
  "type": "merge_layers",
  "target": {
    "output_layer_id": "layer_merged"
  },
  "params": {
    "mode": "selected",
    "layer_ids": ["layer_base", "layer_shadow", "layer_highlight"],
    "output_layer_name": "merged visible detail"
  }
}
```

`mode` is `down`, `visible`, `selected`, or `flatten`. `down` merges the target
layer into the layer immediately below it and preserves the lower layer's ID.
The other modes create `target.output_layer_id`. Prototype merging uses
straight-alpha source-over compositing, supports normal blend mode, and requires
identity layer transforms. `flatten` discards hidden layers and writes an opaque
final alpha channel.

### `select_rect`

```json
{
  "id": "action_select_rect",
  "type": "select_rect",
  "target": {
    "mask_id": "mask_inner"
  },
  "params": {
    "name": "inner",
    "bbox_xyxy": [128, 128, 896, 896],
    "set_active": true
  }
}
```

### `select_ellipse`

```json
{
  "id": "action_select_ellipse",
  "type": "select_ellipse",
  "target": {
    "mask_id": "mask_soft_eye_region"
  },
  "params": {
    "name": "eye region",
    "bbox_xyxy": [160, 140, 260, 240],
    "set_active": true
  }
}
```

The ellipse is rasterized inside `bbox_xyxy` as a hard selection mask.

### `select_color_range` / `select_by_color`

```json
{
  "id": "action_select_eye_color",
  "type": "select_by_color",
  "target": {
    "layer_id": "layer_source",
    "mask_id": "mask_eye_color"
  },
  "params": {
    "name": "eye color",
    "color": "#62beb4",
    "threshold": 15,
    "criterion": "composite",
    "antialias": true,
    "bbox_xyxy": [160, 160, 260, 260],
    "alpha_min": 0.9,
    "kind": "selection",
    "set_active": true
  }
}
```

The mask includes pixels on `target.layer_id` whose color is close to `color`.
For RGB/composite selection, the kernel follows GIMP's max-channel color
difference and can produce a soft antialiased selection. `threshold` uses GIMP's
0-255 units; legacy `tolerance` is still accepted as a normalized value.
`bbox_xyxy`, when supplied, confines the selection to a rectangular region.

### `fuzzy_select` / `magic_wand_select`

```json
{
  "id": "action_select_background",
  "type": "fuzzy_select",
  "target": {
    "layer_id": "layer_source",
    "mask_id": "mask_background"
  },
  "params": {
    "name": "connected background",
    "seed_points": [[70, 70], [560, 70]],
    "threshold": 15,
    "criterion": "composite",
    "antialias": true,
    "alpha_min": 0.9,
    "diagonal": false,
    "kind": "selection",
    "set_active": true
  }
}
```

Each seed selects the contiguous region whose pixels are similar to the clicked
seed color. The implementation is intentionally close to GIMP fuzzy select:
`threshold` is in 0-255 units, `criterion: "composite"` uses max-channel color
difference, and antialiasing returns partial mask values near the boundary.
`diagonal=false` uses four-neighbor connectivity. Multiple seeds are unioned;
for per-click operations, use `clicks` entries with `operation` set to
`replace`, `add`, `subtract`, or `intersect`.

### `create_mask_from_shape`

```json
{
  "id": "action_create_mask",
  "type": "create_mask_from_shape",
  "target": {
    "mask_id": "mask_full_canvas"
  },
  "params": {
    "name": "full canvas",
    "kind": "write_guard",
    "shape": {
      "type": "rectangle",
      "bbox_xyxy": [0, 0, 1024, 1024],
      "corner_radius": 0
    },
    "set_active": false
  }
}
```

Supported prototype shapes are `rectangle` and `ellipse`. Rounded rectangles are
reserved for a later executor implementation.

### `combine_masks`

```json
{
  "id": "action_combine_masks",
  "type": "combine_masks",
  "target": {
    "mask_id": "mask_combined"
  },
  "params": {
    "operation": "union",
    "mask_ids": ["mask_a", "mask_b"],
    "name": "combined mask"
  }
}
```

`operation` is one of `union`, `intersect`, or `subtract`. `subtract` requires
exactly two masks and computes `mask_ids[0] - mask_ids[1]`.

### `grow_mask`

```json
{
  "id": "action_grow_selection",
  "type": "grow_mask",
  "target": {
    "mask_id": "mask_selection_grown"
  },
  "params": {
    "source_mask_id": "mask_selection",
    "pixels": 4,
    "name": "selection grown 4px",
    "set_active": true
  }
}
```

### `shrink_mask`

```json
{
  "id": "action_shrink_selection",
  "type": "shrink_mask",
  "target": {
    "mask_id": "mask_selection_shrunk"
  },
  "params": {
    "source_mask_id": "mask_selection",
    "pixels": 4,
    "name": "selection shrunk 4px",
    "set_active": true
  }
}
```

Grow and shrink create new masks using a disk-shaped pixel footprint. Hard masks
remain hard; soft masks use gray-scale morphology.

### `feather_mask`

```json
{
  "id": "action_feather_mask",
  "type": "feather_mask",
  "target": {
    "mask_id": "mask_inner_soft"
  },
  "params": {
    "source_mask_id": "mask_inner",
    "radius": 2.0,
    "name": "inner soft"
  }
}
```

`radius` is a nonnegative Gaussian sigma in pixels.

### `invert_mask`

```json
{
  "id": "action_invert_selection",
  "type": "invert_mask",
  "target": {
    "mask_id": "mask_selection_inverse"
  },
  "params": {
    "source_mask_id": "mask_selection",
    "name": "selection inverse",
    "set_active": true
  }
}
```

The output mask uses `1.0 - source_value` for every pixel.

### `draw_shape`

```json
{
  "id": "action_draw_border",
  "type": "draw_shape",
  "target": {
    "layer_id": "layer_border"
  },
  "write_mask_id": "mask_full_canvas",
  "params": {
    "shape": {
      "type": "rectangle",
      "bbox_xyxy": [128, 128, 896, 896],
      "corner_radius": 0
    },
    "stroke": {
      "color": "#000000",
      "width": 24
    },
    "fill": null
  },
  "preconditions": {
    "required_layer_ids": ["layer_border"],
    "required_mask_ids": ["mask_full_canvas"],
    "require_unlocked_target_layer": true,
    "require_write_mask": true
  },
  "expected_result": {
    "changed_layer_ids": ["layer_border"]
  },
  "description": "Draw a black rectangular border."
}
```

At least one of `stroke` or `fill` must be non-null. Colors are `#RRGGBB`,
`#RRGGBBAA`, or RGBA lists in `[0, 1]`.

### `paint_bucket_fill`

```json
{
  "id": "action_paint_background",
  "type": "paint_bucket_fill",
  "target": {
    "layer_id": "layer_source"
  },
  "write_mask_id": "mask_background",
  "params": {
    "color": "#ffd6ea",
    "mode": "replace_rgb_preserve_alpha"
  },
  "preconditions": {
    "required_layer_ids": ["layer_source"],
    "required_mask_ids": ["mask_background"],
    "require_write_mask": true
  },
  "expected_result": {
    "changed_layer_ids": ["layer_source"]
  }
}
```

Supported prototype modes are `replace_rgb_preserve_alpha`, `replace_rgba`, and
`source_over`. The executor applies the action through `write_mask_id`, so soft
masks produce blended paint edges.

### `blur_region`

```json
{
  "id": "action_blur_skin",
  "type": "blur_region",
  "target": {
    "layer_id": "layer_portrait"
  },
  "write_mask_id": "mask_skin",
  "params": {
    "radius": 2.0,
    "channels": "rgb",
    "edge_mode": "nearest"
  }
}
```

`channels` may be `rgb`, `alpha`, `rgba`, one channel name, or a list of channel
names drawn from `r`, `g`, `b`, and `a`. `edge_mode` is passed to the Gaussian
filter and may be `reflect`, `constant`, `nearest`, `mirror`, or `wrap`.

### `clear_region`

```json
{
  "id": "action_clear_inner",
  "type": "clear_region",
  "target": {
    "layer_id": "layer_icon"
  },
  "write_mask_id": "mask_old_circle_border",
  "params": {
    "mode": "alpha_to_zero",
    "preserve_rgb": false
  },
  "preconditions": {
    "required_layer_ids": ["layer_icon"],
    "required_mask_ids": ["mask_old_circle_border"],
    "require_unlocked_target_layer": true,
    "require_write_mask": true
  },
  "expected_result": {
    "changed_layer_ids": ["layer_icon"]
  }
}
```

`mode` is `alpha_to_zero` or `rgba_to_zero`.

### `export_flat`

```json
{
  "id": "action_export_flat",
  "type": "export_flat",
  "params": {
    "path": "artifacts/final_preview.npy"
  }
}
```

The prototype executor supports `.npy` directly and `.png` when Pillow is
available.

### `export_layered_bundle`

```json
{
  "id": "action_export_bundle",
  "type": "export_layered_bundle",
  "params": {
    "path": "exports/session_bundle",
    "include_preview": true,
    "include_hidden": true,
    "overwrite": true
  }
}
```

The action writes a directory bundle without mutating the document:

```text
session_bundle/
  manifest.json
  document_snapshot.json
  preview.png
  layers/
    layer_0000_layer_background.png
  masks/
    mask_full_canvas.png
```

`manifest.json` uses schema version `ai_edit_layered_bundle.v1` and records the
canvas, active references, layer entries, mask entries, preview path, and
snapshot path. Layer and preview images are RGBA PNGs. Masks are grayscale PNGs.
`document_snapshot.json` contains the document's JSON-compatible
`snapshot_summary()`.

### `no_op`

```json
{
  "id": "action_no_op",
  "type": "no_op",
  "params": {}
}
```

## Additional Raster Action Families

The executable prototype also includes the following raster-first action
families. They use the same canonical action envelope shown above and prefer
stable IDs over names.

- Transform actions: `move_layer`, `scale_layer`, `rotate_layer`, `flip_layer`,
  `transform_layer`, and `align_layer`.
- Layer-mask actions: `add_layer_mask`, `apply_layer_mask`, and
  `remove_layer_mask`.
- Selection and mask-cleanup actions: `fuzzy_select`, `select_by_color`,
  `select_polygon`, `select_freehand`, `select_from_alpha`,
  `save_selection_as_mask`, `refine_selection`, `remove_small_islands`, and
  `fill_mask_holes`.
- Drawing and fill actions: `draw_path`, `brush_stroke`, `erase_stroke`,
  `gradient_fill`, and `pattern_fill`.
- Clipboard and region-transfer actions: `copy`, `cut`, `paste`,
  `paste_as_new_layer`, and `duplicate_region_to_layer`.
- Color-adjustment actions: `adjust_brightness_contrast`,
  `adjust_hue_saturation`, `adjust_levels`, `adjust_curves`, `colorize`,
  `replace_color`, and `desaturate`.
- Filter and style actions: `sharpen_region`, `noise_reduce`, `median_filter`,
  `edge_detect`, `drop_shadow`, and `stroke_selection`.
- Text actions: `create_text_layer`, `edit_text_layer`, and
  `rasterize_text_layer`. Text layers store rasterized pixels plus editable text
  metadata.
- Perception bridge actions: `detect_shape`, `detect_objects`,
  `segment_object`, `estimate_depth`, `extract_line_art`, and
  `decompose_to_layers`. These are deterministic prototype helpers, not trained
  perception models.
- Diffusion bridge actions: `txt2img_to_layer`, `img2img_to_layer`,
  `inpaint_region`, and `outpaint_region`. These require a configured diffusion
  backend and keep diffusion as a subroutine rather than the whole edit process.

All current `ActionType` enum members have executable prototype behavior.

## Training Example Format

Training examples are derived artifacts, not the trace source of truth. They
should reference the source session and use canonical action JSON in the target.
`input.available_tools` contains the same detailed machine-readable action
catalog entries used by `planner_input.available_actions`; abbreviated entries
in examples stand in for those full catalog objects.

```json
{
  "schema_version": "ai_edit_training_example.v1",
  "example_id": "ex_session_abc123_plan_001",
  "source_session_id": "session_2026_05_27_abc123",
  "example_type": "planner_imitation",
  "task_type": "shape_replacement",
  "split": "train",
  "input": {
    "user_prompt": "Change the icon from a circle to a square.",
    "document_summary": {
      "schema_version": "ai_edit_document.v1",
      "canvas": {
        "width": 1024,
        "height": 1024,
        "color_space": "srgb"
      },
      "layers": [
        {
          "id": "layer_icon",
          "name": "circle icon",
          "kind": "raster",
          "visible": true,
          "opacity": 1.0,
          "stack_index": 1,
          "content_bbox_xyxy": [200, 200, 824, 824]
        }
      ],
      "masks": [],
      "active_layer_id": "layer_icon"
    },
    "observations": [
      {
        "id": "shape_circle_001",
        "type": "circle",
        "center": [512.0, 512.0],
        "radius": 312.0,
        "bbox_xyxy": [200.0, 200.0, 824.0, 824.0],
        "stroke_width_estimate": 24.0
      }
    ],
    "available_tools": [
      {
        "name": "resize_canvas",
        "description": "Resize the canvas around its center."
      },
      {
        "name": "crop",
        "description": "Crop the document, or clear outside a crop on one layer or mask."
      },
      {
        "name": "import_image_as_layer",
        "description": "Import an image file into a full-canvas raster layer."
      },
      {
        "name": "import_vector_as_raster",
        "description": "Rasterize a vector asset and import it as a full-canvas raster layer."
      },
      {
        "name": "rasterize_vector_asset",
        "description": "Rasterize a vector asset to a standalone PNG or NPY artifact."
      },
      {
        "name": "create_layer",
        "description": "Create a new full-canvas layer."
      },
      {
        "name": "delete_layer",
        "description": "Remove a layer from the document stack."
      },
      {
        "name": "duplicate_layer",
        "description": "Create a deep copy of a layer."
      },
      {
        "name": "rename_layer",
        "description": "Rename a layer without changing its ID."
      },
      {
        "name": "reorder_layer",
        "description": "Move a layer to a new stack index."
      },
      {
        "name": "set_active_layer",
        "description": "Set the active layer."
      },
      {
        "name": "set_layer_visibility",
        "description": "Show or hide a layer."
      },
      {
        "name": "set_layer_opacity",
        "description": "Set a layer's opacity."
      },
      {
        "name": "set_blend_mode",
        "description": "Set a layer's blend mode metadata."
      },
      {
        "name": "merge_layers",
        "description": "Merge layers using normal source-over compositing."
      },
      {
        "name": "select_rect",
        "description": "Create a rectangular selection mask."
      },
      {
        "name": "select_ellipse",
        "description": "Create an elliptical selection mask."
      },
      {
        "name": "select_color_range",
        "description": "Create a mask from pixels close to a target color."
      },
      {
        "name": "magic_wand_select",
        "description": "Create a contiguous color-based selection from seed points."
      },
      {
        "name": "create_mask_from_shape",
        "description": "Create a mask from a deterministic geometric shape."
      },
      {
        "name": "grow_mask",
        "description": "Grow a mask by a pixel radius."
      },
      {
        "name": "shrink_mask",
        "description": "Shrink a mask by a pixel radius."
      },
      {
        "name": "invert_mask",
        "description": "Invert a mask."
      },
      {
        "name": "combine_masks",
        "description": "Combine masks with union, intersect, or subtract."
      },
      {
        "name": "feather_mask",
        "description": "Create a softened copy of a mask."
      },
      {
        "name": "draw_shape",
        "description": "Draw a deterministic geometric shape on a target layer."
      },
      {
        "name": "paint_bucket_fill",
        "description": "Fill the current write mask on a target layer with a color."
      },
      {
        "name": "blur_region",
        "description": "Apply Gaussian blur to selected channels through a write mask."
      },
      {
        "name": "clear_region",
        "description": "Clear pixels or alpha inside a write mask on a target layer."
      },
      {
        "name": "export_flat",
        "description": "Export a flattened preview image."
      },
      {
        "name": "no_op",
        "description": "Execute no document mutation."
      }
    ],
    "asset_refs": {
      "visible_preview": "previews/doc_rev_0000.png"
    }
  },
  "target": {
    "action_batch": {
      "schema_version": "ai_edit_actions.v1",
      "id": "batch_001",
      "stop_on_error": true,
      "actions": [
        {
          "id": "action_create_square_layer",
          "type": "create_layer",
          "target": {
            "output_layer_id": "layer_square_border"
          },
          "params": {
            "name": "square border",
            "kind": "raster",
            "color": "#00000000",
            "opacity": 1.0,
            "blend_mode": "normal",
            "insert_index": 2,
            "set_active": true
          }
        },
        {
          "id": "action_draw_square_border",
          "type": "draw_shape",
          "target": {
            "layer_id": "layer_square_border"
          },
          "write_mask_id": "mask_full_canvas",
          "params": {
            "shape": {
              "type": "rectangle",
              "bbox_xyxy": [200, 200, 824, 824],
              "corner_radius": 0
            },
            "stroke": {
              "color": "#0D0D0DFF",
              "width": 24
            },
            "fill": null
          },
          "preconditions": {
            "required_layer_ids": ["layer_square_border"],
            "required_mask_ids": ["mask_full_canvas"],
            "require_unlocked_target_layer": true,
            "require_write_mask": true
          },
          "expected_result": {
            "changed_layer_ids": ["layer_square_border"]
          }
        }
      ]
    }
  },
  "labels": {
    "success": true,
    "validation_passed": true,
    "human_accepted": true,
    "human_rating": {
      "overall": 4,
      "goal_satisfaction": 5,
      "mask_accuracy": 5,
      "visual_quality": 4,
      "layer_structure": 5
    },
    "metrics": {
      "outside_write_mask_changed_pixels": 0,
      "geometry_error_px": 0.0,
      "created_expected_layers": true
    }
  },
  "provenance": {
    "trace_path": "traces/session_2026_05_27_abc123",
    "source": "synthetic_demo",
    "allowed_for_training": true
  }
}
```
