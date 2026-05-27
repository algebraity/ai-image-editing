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
    "tool_catalog_version": "tools.v1"
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
    "user_prompt": "Change the icon from a circle to a square.",
    "document_summary": {},
    "observations": [],
    "available_tools": [
      {
        "name": "draw_shape",
        "description": "Draw a deterministic geometric shape on a target layer."
      }
    ]
  },
  "asset_refs": {
    "visible_preview": "previews/doc_rev_0000.png"
  },
  "metadata": {}
}
```

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
    "parser_status": "parsed"
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

### `select_color_range`

```json
{
  "id": "action_select_eye_color",
  "type": "select_color_range",
  "target": {
    "layer_id": "layer_source",
    "mask_id": "mask_eye_color"
  },
  "params": {
    "name": "eye color",
    "color": "#62beb4",
    "tolerance": 0.36,
    "bbox_xyxy": [160, 160, 260, 260],
    "alpha_min": 0.9,
    "kind": "selection",
    "set_active": true
  }
}
```

The mask includes pixels on `target.layer_id` whose RGB distance from `color` is
within `tolerance`. `bbox_xyxy`, when supplied, confines the selection to a
rectangular region.

### `magic_wand_select`

```json
{
  "id": "action_select_background",
  "type": "magic_wand_select",
  "target": {
    "layer_id": "layer_source",
    "mask_id": "mask_background"
  },
  "params": {
    "name": "connected background",
    "seed_points": [[70, 70], [560, 70]],
    "tolerance": 0.08,
    "alpha_min": 0.9,
    "diagonal": false,
    "kind": "selection",
    "set_active": true
  }
}
```

Each seed selects the contiguous region whose pixels are within `tolerance` of
the seed pixel's RGB value. `diagonal=false` uses four-neighbor connectivity.

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

### `no_op`

```json
{
  "id": "action_no_op",
  "type": "no_op",
  "params": {}
}
```

## Future Action Families

These action families exist in the enum but do not yet have finalized executable
prototype schemas:

- new-document and canvas resize actions
- layer delete, duplicate, rename, reorder, visibility, opacity, blend mode, and
  merge actions
- polygon and alpha selections
- path, brush, gradient, cut, copy, paste, transform, and align actions
- perception actions
- diffusion actions

When finalized, each should use the same action envelope and should prefer
stable IDs over names.

## Training Example Format

Training examples are derived artifacts, not the trace source of truth. They
should reference the source session and use canonical action JSON in the target.

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
        "name": "import_image_as_layer",
        "description": "Import an image file into a full-canvas raster layer."
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
        "name": "paint_bucket_fill",
        "description": "Fill the current write mask on a target layer with a color."
      },
      {
        "name": "clear_region",
        "description": "Clear pixels or alpha inside a write mask on a target layer."
      },
      {
        "name": "create_layer",
        "description": "Create a new layer."
      },
      {
        "name": "draw_shape",
        "description": "Draw a deterministic geometric shape on a target layer."
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
