"""
Decision point tracer for MTCNN pipeline.
Logs input/output counts at each decision point for analysis.
"""

import json
import os
from datetime import datetime


class DecisionTracer:
    """
    Logs every decision point in the MTCNN pipeline.
    
    Each log entry records:
    - Which image was being processed
    - Which stage (pnet, rnet, onet)
    - Which decision type (threshold, nms)
    - How many boxes entered the decision
    - How many boxes survived the decision
    - How many boxes were rejected
    - The threshold or IOU value used
    """

    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'mtcnn_trace_{timestamp}.json')
        self.entries = []
        self.current_image_id = None

    def set_image(self, image_id):
        """Call this before processing each image."""
        self.current_image_id = image_id

    def log(self, stage, decision_type, input_count, output_count, value):
        """
        Log one decision point.

        Args:
            stage:         'pnet_threshold', 'pnet_nms_scale', 'pnet_nms_cross',
                           'rnet_threshold', 'rnet_nms',
                           'onet_threshold', 'onet_nms'
            decision_type: 'threshold' or 'nms'
            input_count:   number of boxes entering this decision
            output_count:  number of boxes surviving this decision
            value:         threshold value (0.6, 0.7) or IOU value (0.5, 0.7)
        """
        entry = {
            'image_id':      self.current_image_id,
            'stage':         stage,
            'decision_type': decision_type,
            'input_count':   int(input_count),
            'output_count':  int(output_count),
            'rejected_count': int(input_count - output_count),
            'value':         float(value),
        }
        self.entries.append(entry)

    def save(self):
        """Write all entries to disk as a single formatted JSON file."""
        payload = {
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'entry_count': len(self.entries),
            'entries': self.entries,
        }
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(f"Trace saved to {self.log_path}")
        print(f"Total entries: {len(self.entries)}")