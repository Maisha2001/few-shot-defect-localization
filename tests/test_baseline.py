import io
import tarfile
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from inspection.data import Archive, DEVELOPMENT, FIT, RESERVED, Sample, select_samples
from inspection.evaluate import fingerprint, near_reference
from inspection.metrics import native_counts, summarize
from inspection.model import anomaly_scores, pixels, reference_indices

class BaselineTests(unittest.TestCase):
    def test_protocol_keeps_product_families_and_uses_normal_support(self):
        rows = [dict(object=p, split="train", label="normal", image=p+"/good", mask="") for p in FIT]
        rows += [dict(object=p, split="test", label="anomaly", image=p+"/bad", mask=p+"/mask") for p in DEVELOPMENT]
        rows += [dict(object=p, split="test", label="anomaly", image=p+"/secret", mask="secret") for p in RESERVED]
        fit, dev = select_samples(reversed(rows), per_product=1)
        self.assertEqual({s.product for s in fit}, set(FIT))
        self.assertEqual({s.product for s in dev}, set(DEVELOPMENT))
        self.assertTrue(all(not s.anomaly for s in fit))
        with self.assertRaisesRegex(ValueError, "normal"):
            select_samples([dict(rows[0],label="anomaly"), *rows[1:]], per_product=1)

    def test_native_metrics_match_explicit_pixel_ranking_with_ties(self):
        mask = np.zeros((3,5), dtype=np.uint8)
        mask[2,4] = 1
        t, p = native_counts((5,3), Image.fromarray(mask), cells=2)
        self.assertEqual((t.sum(),p.sum()), (15,1))
        heat = np.array([[0.1,0.2],[0.6,0.6]])
        native = heat[(np.arange(3)*2//3)[:,None],(np.arange(5)*2//5)[None,:]]
        labels = np.r_[mask.ravel(), np.zeros(15)]
        scores = np.r_[native.ravel(), native.ravel()]
        result = summarize([1,0], [0.8,0.1], [heat,heat], [t,t], [p,p*0])
        self.assertAlmostEqual(result["pixel"]["average_precision"], average_precision_score(labels,scores))
        self.assertAlmostEqual(result["pixel"]["auroc"], roc_auc_score(labels,scores))
        self.assertEqual(result["image"]["auroc"],1.)

    def test_invalid_masks_fail(self):
        for size in [(4,3), (5,3)]:
            with self.assertRaises(ValueError):
                native_counts(size,Image.new("L",(5,3)))

    def test_known_bank_distances_and_non_square_input(self):
        patches = torch.tensor([[[1.,0.]]*252+[[0.,1.]]*4])
        scores, heat = anomaly_scores(patches, torch.tensor([[1.,0.]]))
        self.assertAlmostEqual(float(scores[0]), 1.)
        self.assertEqual(heat.shape,(1,56,56))
        self.assertEqual(float(heat[0,0,0]),0.)
        self.assertEqual(float(heat[0,-1,-1]),1.)
        self.assertEqual(tuple(pixels(Image.new("RGB",(345,210))).shape),(3,224,224))
        self.assertEqual(len(set(reference_indices("product/000.JPG"))),16)

    def test_cross_partition_near_duplicate_guard(self):
        template = np.arange(256,dtype=np.uint8).reshape(16,16)
        im = Image.fromarray(template)
        self.assertTrue(near_reference(fingerprint(im.resize((128,128))),[fingerprint(im)]))
        self.assertFalse(near_reference(fingerprint(Image.fromarray(255-template)),[fingerprint(im)]))

    def test_tar_rejects_duplicate_missing_and_link_members(self):
        sample = Sample("candle","image.png","",False)
        for kind in ("duplicate","missing","link"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                path=Path(d)/"sample.tar"
                with tarfile.open(path,"w") as out:
                    for _ in range(2 if kind=="duplicate" else 1):
                        info=tarfile.TarInfo("other" if kind=="missing" else sample.image)
                        if kind=="link":
                            info.type=tarfile.SYMTYPE
                            info.linkname="../outside"
                        else:
                            info.size=1
                        out.addfile(info,io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    Archive(path,[sample])

if __name__ == "__main__":
    unittest.main()
