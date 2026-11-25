"""A fractional-mask patch classifier and its pixel-probability calibration."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch.nn import functional as F


@dataclass
class PatchHead:
    weight: list
    bias: float
    slope: float = 1.
    offset: float = 0.
    threshold: float = .5

    def raw(self, patches):
        return patches @ np.asarray(self.weight, dtype=np.float32) + self.bias

    def to_dict(self):
        return asdict(self)


def fit_head(features, total, positive):
    """Fit class-balanced logistic loss to foreground occupancy within each patch."""
    x = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    p = positive.ravel().astype(np.float64)
    n = total.ravel().astype(np.float64) - p
    if np.any(p < 0) or np.any(n < 0) or not np.all(np.isfinite(x)) or n.sum() <= 0:
        raise ValueError("Invalid patch counts or features")
    if p.sum() == 0:
        center = np.average(x, axis=0, weights=n)
        return PatchHead((-center / max(np.linalg.norm(center), 1e-12)).tolist(), 1.)
    p *= .5 / p.sum()
    n *= .5 / n.sum()
    def objective(params):
        w, b = params[:-1], params[-1]
        z = x @ w + b
        loss = p @ np.logaddexp(0., -z) + n @ np.logaddexp(0., z) + .0005 * (w @ w)
        err = (p+n)*expit(z) - p
        return loss, np.r_[x.T @ err + .001*w, err.sum()]
    result = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method="L-BFGS-B",
                      options={"maxiter":200, "ftol":1e-12, "gtol":1e-7})
    if not result.success:
        raise RuntimeError(result.message)
    return PatchHead(result.x[:-1].tolist(), float(result.x[-1]))


def interpolate(raw):
    t = torch.from_numpy(np.array(raw,dtype=np.float32)).reshape(-1,1,16,16)
    return F.interpolate(t, size=(56,56), mode="bilinear", align_corners=False)[:,0].numpy()


def dice_threshold(scores, total, pos):
    scores, total, pos = [np.asarray(a).ravel() for a in (scores,total,pos)]
    if pos.sum() == 0:
        return 1.
    idx = np.argsort(-scores,kind="stable")
    s = scores[idx]
    ends = np.flatnonzero(np.r_[s[:-1] != s[1:], True])
    dice = 2 * pos[idx].cumsum()[ends] / (pos.sum() + total[idx].cumsum()[ends])
    return float(s[ends[np.argmax(dice)]])


def calibrate(head, patches, total, pos):
    raw = interpolate(head.raw(patches)).astype(np.float64)
    mass = total.sum()
    p, n = pos.ravel()/mass, (total-pos).ravel()/mass
    r = raw.ravel()
    def objective(params):
        slope, offset = params
        z = slope*r + offset
        e = (p+n)*expit(z)-p
        return (p@np.logaddexp(0.,-z)+n@np.logaddexp(0.,z)+1e-6*(slope-1)**2,
                np.array([e@r+2e-6*(slope-1),e.sum()]))
    result = minimize(objective, [1.,-5.], jac=True, bounds=[(.01,30.),(-25.,15.)],
                      method="L-BFGS-B", options={"ftol":1e-14,"gtol":1e-10})
    if not result.success:
        raise RuntimeError(result.message)
    head.slope, head.offset = map(float,result.x)
    prob = probabilities(head,patches)[1]
    head.threshold = dice_threshold(prob,total,pos)


def probabilities(head, patches):
    raw = head.raw(patches)
    top = np.partition(raw,-4,axis=1)[:,-4:].mean(1)
    image_score = expit(head.slope*top + head.offset)
    return image_score, expit(head.slope*interpolate(raw) + head.offset)


def next_image(policy, selected, patches, head, descriptors):
    """Random control uses the pre-shuffled order. Active queries only see features."""
    available = np.setdiff1d(np.arange(len(patches)),selected)
    if not len(available): raise ValueError("Acquisition pool exhausted")
    if policy == "random" or head is None:
        return int(available[0])
    if policy != "active":
        raise ValueError("Unknown acquisition policy")
    # Class-balanced training logits give margins before prevalence calibration.
    margins = np.abs(head.raw(patches[available]))
    ambiguity = np.exp(-np.partition(margins,7,axis=1)[:,:8].mean(axis=1))
    novelty = 1 - (descriptors[available] @ descriptors[selected].T).max(axis=1)
    return int(available[np.argmax(ambiguity + novelty)])


def calibration_metrics(prob, total, pos, threshold):
    prob = np.asarray(prob,dtype=np.float64)
    n = total-pos
    mass = total.sum()
    pred = prob >= threshold
    tp = pos[pred].sum()
    bins = np.minimum((prob*10).astype(int),9)
    bin_mass = np.bincount(bins.ravel(),weights=total.ravel(),minlength=10)
    gap = np.bincount(bins.ravel(),weights=(prob*total-pos).ravel(),minlength=10)
    return {"pixel_brier":float(((1-prob)**2*pos+prob**2*n).sum()/mass),
            "pixel_ece10":float(np.abs(gap[bin_mass>0]).sum()/mass),
            "dice":float(2*tp / max(1, total[pred].sum()+pos.sum())),
            "predicted_positive_pixels":int(total[pred].sum())}
