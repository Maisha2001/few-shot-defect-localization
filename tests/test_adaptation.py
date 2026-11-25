import json
from pathlib import Path
import unittest

import numpy as np

from inspection.adapt import PatchHead, calibration_metrics, dice_threshold, fit_head, next_image, probabilities

class AdaptationTests(unittest.TestCase):
    def test_fit_uses_positive_mass_in_mixed_patches(self):
        x=np.array([[[-1.,0.],[1.,0.]], [[-1.,0.],[1.,0.]]],dtype=np.float32)
        total=np.full((2,2),100)
        head=fit_head(x,total,np.array([[0,2],[0,3]]))
        self.assertGreater(head.raw(x)[0,1],head.raw(x)[0,0])
        fallback=fit_head(x[:,:1],total[:,:1],np.zeros((2,1)))
        self.assertTrue(np.isfinite(fallback.raw(x)).all())
        self.assertGreater(fallback.raw(x)[0,1],fallback.raw(x)[0,0])

    def test_acquisition_excludes_reviewed_and_uses_diversity(self):
        patches=np.zeros((3,256,2),dtype=np.float32)
        desc=np.array([[1.,0.],[1.,0.],[0.,1.]])
        head=PatchHead([0.,0.],0.)
        self.assertEqual(next_image("active",[0],patches,head,desc),2)
        self.assertEqual(next_image("random",[0],patches,head,desc),1)

    def test_threshold_keeps_ties_and_pixel_counts(self):
        prob=np.array([.1,.4,.4,.9])
        total=np.array([10,10,10,10]);pos=np.array([0,8,4,7])
        threshold=dice_threshold(prob,total,pos)
        m=calibration_metrics(prob,total,pos,threshold)
        self.assertEqual(threshold,.4)
        self.assertAlmostEqual(m["dice"],38/49)
        individual=np.repeat(prob,total)
        ys=np.concatenate([np.r_[np.ones(k),np.zeros(n-k)] for n,k in zip(total,pos)])
        self.assertAlmostEqual(m["pixel_brier"],np.mean((individual-ys)**2))

    def test_inference_uses_matching_patch_geometry(self):
        x=np.zeros((1,256,2),dtype=np.float32)
        x[:,252:,0]=1.
        head=PatchHead([1.,0.],-.2,2.,-.4)
        score,heat=probabilities(head,x)
        self.assertEqual(heat.shape,(1,56,56))
        self.assertAlmostEqual(float(score[0]),float(heat[0,-1,-1]),places=6)
        self.assertGreater(score[0],heat[0,0,0])

    def test_published_budget_experiment_has_nested_and_equal_costs(self):
        report=json.loads((Path(__file__).resolve().parents[1]/"results"/"adaptation.json").read_text())
        for policy in ("active","random"):
            before=[]
            for budget in (4,8,16,32):
                arm=report["arms"][f"{policy}-{budget}"]
                order=arm["acquisition_indices"]
                self.assertEqual(len(set(order)),budget)
                self.assertEqual(order[:len(before)],before)
                before=order
                pooled=arm["groups"]["pooled"]
                self.assertEqual(pooled["images"],560)
                self.assertEqual(pooled["defect_pixels"],1012410)
                self.assertTrue(0<=pooled["dice"]<=1)
        self.assertEqual(report["calibration_images"],8)
