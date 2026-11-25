import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def native_counts(size, mask, cells=56):
    """Count native pixels per prediction cell so even a one-pixel flaw contributes."""
    w, h = size
    if min(w, h) < 1:
        raise ValueError("Empty image")
    xbin = np.arange(w) * cells // w
    ybin = np.arange(h) * cells // h
    total = np.outer(np.bincount(ybin, minlength=cells), np.bincount(xbin, minlength=cells))
    if mask is None:
        return total, np.zeros_like(total)
    if mask.size != size:
        raise ValueError("Mask dimensions differ from the image")
    ys, xs = np.nonzero(np.asarray(mask))
    if not len(xs):
        raise ValueError("Empty anomaly mask")
    pos = np.bincount(ybin[ys] * cells + xbin[xs], minlength=cells*cells)
    return total, pos.reshape(cells, cells)

def ranking(labels, scores, weights=None):
    return {"auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
            "average_precision": float(average_precision_score(labels, scores, sample_weight=weights))}

def summarize(labels, image_scores, maps, totals, positives):
    labels, image_scores = np.asarray(labels), np.asarray(image_scores)
    scores = np.asarray(maps).ravel()
    pos, total = np.asarray(positives).ravel(), np.asarray(totals).ravel()
    if not len(scores) == len(total) == len(pos) or (total < pos).any() or (pos < 0).any():
        raise ValueError("Inconsistent pixel counts")
    # Each cell has a single score but can contain both positive and negative pixels.
    weights = np.concatenate((pos, total - pos))
    keep = weights > 0
    y = np.repeat([1, 0], len(scores))[keep]
    return {"images": len(labels), "anomalous_images": int(labels.sum()),
            "native_pixels": int(total.sum()), "defect_pixels": int(pos.sum()),
            "image": ranking(labels, image_scores),
            "pixel": ranking(y, np.tile(scores, 2)[keep], weights[keep])}
