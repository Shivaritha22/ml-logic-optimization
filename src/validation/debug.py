# Debug image 490
import json
import numpy as np

with open('baseline/wider_face_baseline.json') as f:
    baseline = json.load(f)

key = '0--Parade\\0_Parade_Parade_0_490.jpg'
orig_boxes = np.array(baseline[key]['boxes'])

print(f"Total boxes: {len(orig_boxes)}")
print(f"\nAll orig boxes sorted by x1 then y1:")
orig_idx = np.lexsort((orig_boxes[:, 1], orig_boxes[:, 0]))
orig_sorted = orig_boxes[orig_idx]
for i, box in enumerate(orig_sorted):
    print(f"  [{i}] x1={box[0]:.1f} y1={box[1]:.1f} x2={box[2]:.1f} y2={box[3]:.1f}")