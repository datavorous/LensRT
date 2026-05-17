# GraphLens

Static TFLite graph analyzer for QNN partition diagnostics.

## Install

```bash
git clone <repo-url>
cd graphlens
uv sync
```

## Quick Start

```python
from graphlens import Inspector

inspector = Inspector.from_tflite(
    "model.tflite",
    op_support_path="analysis/opSupportMap.csv",
)
inspector.analyze()
inspector.report_rank_violations()
inspector.report_dynamic_shapes()
inspector.report_cross_signature_divergence()
inspector.report_partitions()
inspector.report_seams(context=2, kind="CPU")
```

Run: `uv run python analyze.py`

For `.litertlm` files:

```python
inspector = Inspector.from_litertlm(
    "model.litertlm",
    op_support_path="analysis/opSupportMap.csv",
    dump_dir="./litertlm_dump", # extracted .tflite sections
)
```

## What It Does

Flags partition blockers:

1. No QNN builder for op
2. Input rank exceeds QNN cap
3. Dynamic shape or index input
4. Inferred dimension (-1 in RESHAPE, PAD, BROADCAST_TO, TILE)
5. Op fragments across multiple subgraph signatures

Each check is deterministic and fixable at export time.

## How To Use It

1. Run tool on new model. Takes seconds.
2. Fix everything it flags. These are exporter-level issues.
3. Run apply_plugin_main. Remaining fallbacks are dtype/attribute/SDK.
4. Feed apply_plugin_main output to compare harness for next iteration.

The tool clears 60-70% of fallbacks without SDK setup or hardware. It does not replace apply_plugin_main. It filters what gets sent to it.

## Limitations

- No LiteRT/QNN execution
- No runtime backend placement prediction
- opSupportMap.csv legalization does not guarantee NPU placement
