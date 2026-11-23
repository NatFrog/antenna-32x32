#Training losses on dB patterns.

import torch

from metrics import loss_fn, pixel_weights
from model import DB_FLOOR


def l2db(pred, tgt, th):
    w = pixel_weights(tgt, th)
    return (w * (pred.clamp_min(DB_FLOOR) - tgt.clamp_min(DB_FLOOR)).pow(2)).mean() / 10


def l1db_relfloor(pred, tgt, th, rel=55.0):
    # Floor relative to each pattern's peak instead of an absolute -60 dBi.
    f = tgt.amax((1, 2), keepdim=True) - rel
    w = pixel_weights(tgt, th)
    return (w * (torch.maximum(pred, f) - torch.maximum(tgt, f)).abs()).mean()


def l1db_flat(pred, tgt, th):
    # Solid-angle weighting only (no main-lobe boost).
    w = pixel_weights(tgt, th, main_boost=0.0)
    return (w * (pred.clamp_min(DB_FLOOR) - tgt.clamp_min(DB_FLOOR)).abs()).mean()


def l1db_boost10(pred, tgt, th):
    # 10x (instead of 3x) emphasis on pixels within 30 dB of the peak.
    w = pixel_weights(tgt, th, main_boost=10.0)
    return (w * (pred.clamp_min(DB_FLOOR) - tgt.clamp_min(DB_FLOOR)).abs()).mean()


LOSSES = {"l1db": loss_fn, "l2db": l2db, "l1db_relfloor": l1db_relfloor, "l1db_flat": l1db_flat,
          "l1db_boost10": l1db_boost10}
