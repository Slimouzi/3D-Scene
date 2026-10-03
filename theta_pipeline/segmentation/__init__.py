"""Zero-click semantic segmentation on perspective faces, never on the full ERP.

This module is imported by both environments: keep it free of cv2, pycolmap and torch.
GPU inference lives in `sam3_segmenter` / `__main__`; CPU fusion lives in `fusion`.
"""
MODEL_ID = 'facebook/sam3'
TRIAL_PANORAMA = 'R0010004'

PROMPTS = ('window', 'glass window', 'sliding glass door', 'large glass pane',
           'window frame', 'mirror', 'reflection', 'floor', 'wall', 'furniture',
           'person', 'tripod')

# Prompts are fused per group: a face votes for a group if any of its prompts fires.
GROUPS = {
    'glass': ('window', 'glass window', 'sliding glass door', 'large glass pane'),
    'mirror': ('mirror',),
    'reflection': ('reflection',),
    'window_frame': ('window frame',),
    'dynamic': ('person', 'tripod'),
    'floor': ('floor',),
    'wall': ('wall',),
    'furniture': ('furniture',),
}

# Versioned decision rules. Thresholds are NOT calibrated on annotated data yet:
# every decision derived from them is experimental (CTO/05, section 4).
POLICY = {
    'version': 'seg-policy-1',
    'calibrated': False,
    'confidence_threshold': 0.5,
    'face_border_px': 8,
    'min_area_fraction': 0.0005,
    'min_face_overlap_fraction': 0.2,
    'min_consistent_projections': 2,
    'consensus_fraction': 0.5,
    'fusion_max_width': 2048,
    'geometry_margin_deg': 1.0,
    'geometry_policy': (
        'Exclude from SfM: validated (>=2 agreeing faces, majority) glass, mirror, reflection '
        'and dynamic regions; enclosed holes of validated glass/mirror; unvalidated candidates '
        'connected to a validated region; pixels with no successful inference; all dilated by '
        'geometry_margin_deg. Isolated unvalidated candidates and single-projection pixels stay '
        'usable for SfM texture, are marked in unknown_mask and blocked for navigation.'),
    'appearance_weight_reflective': 128,
    'max_hole_fraction': 0.05,
    'max_seam_mismatch': 0.02,
    'neighbors': 2,
    'trial': {'min_two_view_coverage': 0.99, 'max_failed_faces': 0,
              'min_pairwise_iou': 0.5, 'max_unknown_fraction': 0.35},
}

LABELS = {'other': 0, 'no_inference': 1, 'dynamic': 2, 'mirror': 3, 'glass': 4,
          'unknown_glass': 5, 'unknown_reflective': 6, 'window_frame': 7,
          'floor': 8, 'wall': 9, 'furniture': 10}


def slug(prompt):
    return prompt.replace(' ', '_')
