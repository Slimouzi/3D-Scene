"""Read-only ERP seam diagnostic for an existing run (never writes into the run).

    python scripts/seam_diagnostic.py Output/runs/salon-sam3-001 R0010009

Rebuilds an approximation of the reflective candidate mask from labels.png (glass,
unknown_glass, mirror, unknown_reflective) and compares the former raw seam metric with
the circular rule of fusion.seam_continuity. Validated reflection without glass/mirror
has no label of its own, so the approximation can only under-count candidates.
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theta_pipeline.segmentation import LABELS  # noqa: E402
from theta_pipeline.segmentation.fusion import seam_continuity  # noqa: E402


def main(run, pano):
    labels = np.asarray(Image.open(Path(run) / 'segmentation/fused' / pano / 'labels.png'))
    mask = np.isin(labels, [LABELS[k] for k in ('glass', 'unknown_glass', 'mirror', 'unknown_reflective')])
    rows = np.flatnonzero(mask[:, -1] != mask[:, 0])
    near = int(mask.shape[1] / 180)                     # 2 degrees in columns
    result = seam_continuity(mask)
    result['mismatched_rows'] = rows.tolist()
    result['mismatched_latitude_deg'] = [round(((r + .5) / mask.shape[0] - .5) * 180, 2) for r in rows]
    result['candidate_columns_within_2deg'] = int(mask[:, list(range(-near, near))].any(axis=0).sum())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:3])
