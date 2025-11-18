# Paper / Formula Implementation Guide

This document preserves the detailed code ↔ math walkthrough formerly in IMPLEMENTATION_GUIDE.md. It is now secondary to the root README.

## Algorithm Skeleton
Refer to README Section 2 for pipeline diagram; core mathematical correspondences remain:
- Encoders φ_g: `dro_hetero_anchors/src/encoders/`
- Anchors μ_c: `dro_hetero_anchors/src/model/anchors.py`
- W₂ distance: `dro_hetero_anchors/src/model/wasserstein.py`
- Fit & separation losses: `dro_hetero_anchors/src/model/losses.py`
- GroupDRO: `dro_hetero_anchors/src/model/groupdro.py`

## Notes
See archived COMPLETE_GUIDE.md and REPO_OVERVIEW.md if historical context required.
