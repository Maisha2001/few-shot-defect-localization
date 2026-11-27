from pathlib import Path
import tempfile
import unittest

import numpy as np

from inspection.data import RESERVED
from inspection.final import SPEC, read, selected_images, threshold_metrics, verify_lock


class FinalProtocolTests(unittest.TestCase):
    def test_published_evidence_matches_frozen_digests(self):
        from inspection.data import file_hash
        folder = SPEC.parents[1]/"results"
        report = read(folder/"final.json")
        audit = read(folder/"final-audit.json")
        self.assertEqual(file_hash(folder/"final.json"),audit["report_sha256"])
        self.assertEqual(file_hash(folder/"final-protocol.json"),report["protocol_sha256"])
        self.assertEqual(file_hash(SPEC),report["spec_sha256"])
        self.assertEqual(report["predictions"]["files"]["outputs.npz"],audit["prediction_sha256"])
        self.assertEqual(report["heads"],read(SPEC)["heads"])
        self.assertEqual(set(report["arms"]),{"active-32","random-32","reference-bank"})

    def test_selection_is_complete_and_independent_of_labels(self):
        rows = [{"object":p,"split":"test","image":p+"/01"} for p in RESERVED]
        expected = selected_images(rows)
        extra = [{"object":RESERVED[0],"split":"train","image":"unused"},
                 {"object":"pcb1","split":"test","image":"unused2"}]
        self.assertEqual(selected_images(list(reversed(rows)) + extra), expected)
        self.assertEqual(len(expected), 3)
        self.assertTrue(all(not s.mask and not s.anomaly for s in expected))
        with self.assertRaises(ValueError):
            selected_images(rows + [rows[0]])

    def test_threshold_ties_partial_cells_and_failures(self):
        prob = np.array([[[.2,.5]], [[.9,.1]], [[.1,.0]]])
        total = np.full(prob.shape,10)
        positive = np.array([[[0,2]], [[0,0]], [[1,0]]])
        m = threshold_metrics(prob,total,positive,.5)
        self.assertEqual(m["true_positive_pixels"],2)
        self.assertAlmostEqual(m["pixel_precision"],.1)
        self.assertAlmostEqual(m["pixel_recall"],2/3)
        self.assertEqual(m["normal_images_with_foreground"],1)
        self.assertEqual(m["defective_images_without_overlap"],1)
        self.assertIsNone(threshold_metrics(prob,total,positive,1.)["pixel_precision"])

    def test_protocol_mutation_is_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/"protocol.json").write_text(json.dumps({"spec_sha256":"changed","spec":read(SPEC)}))
            with self.assertRaises(ValueError):
                verify_lock(path)
