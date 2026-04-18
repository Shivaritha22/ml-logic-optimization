# MTCNN Decision Points Map

## Code Locations
- Network definitions: `facenet_pytorch/models/mtcnn.py`
- Decision logic: `facenet_pytorch/utils/detect_face.py`

## Default Parameters
- P-Net threshold: 0.6
- R-Net threshold: 0.7
- O-Net threshold: 0.7
- Scale NMS IOU: 0.5
- Cross-scale NMS IOU: 0.7
- R-Net NMS IOU: 0.7
- O-Net NMS IOU: 0.7

## Decision Points

| # | Stage | Type | Value | Input | Key Variable |
|---|-------|------|-------|-------|--------------|
| 1 | P-Net | Threshold | 0.6 | ~50,000-200,000 | mask = probs >= thresh |
| 2a | P-Net | NMS (per scale) | IOU=0.5 | Post-threshold boxes | batched_nms(..., 0.5) |
| 2b | P-Net | NMS (cross scale) | IOU=0.7 | All scale boxes combined | batched_nms(..., 0.7) |
| 3 | R-Net | Threshold | 0.7 | ~200-2000 | ipass = score > threshold[1] |
| 4 | R-Net | NMS | IOU=0.7 | Post-threshold boxes | batched_nms(..., 0.7) |
| 5 | O-Net | Threshold | 0.7 | ~20-200 | ipass = score > threshold[2] |
| 6 | O-Net | NMS (Min) | IOU=0.7 | Post-threshold boxes | batched_nms_numpy(..., 'Min') |

## Key Observations

1. P-Net has TWO NMS calls, not one
   - Per-scale NMS (IOU=0.5): removes overlaps within each scale
   - Cross-scale NMS (IOU=0.7): removes overlaps across all scales

2. R-Net and O-Net have identical thresholds (both 0.7)
   - Could one be redundant given the other?

3. O-Net uses Min strategy NMS, different from P-Net and R-Net
   - More aggressive - removes boxes contained within larger boxes
   - Makes sense for final precise detection

4. Redundancy hypothesis:
   - R-Net threshold (0.7) sees only boxes that survived P-Net (0.6)
   - R-Net only filters scores in range [0.6, 0.7]
   - If this range is rare in real data, R-Net threshold is redundant
   - Same logic applies to O-Net threshold vs R-Net threshold