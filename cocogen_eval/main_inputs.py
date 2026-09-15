"""Apply FM4PDE's task gate to archived candidate observation masks."""
from __future__ import annotations

from .inputs import historical_input as archived_input


def historical_input(record):
    fields, masks, ids, params = archived_input(record)
    # FM4PDE stores BOTH candidate field masks even for forward/inverse tasks.
    # Its observation loss gates the inactive field by `task` (losses.py).
    # CoCoGen's hard imputation must receive only the effective observations.
    masks = masks.clone()
    if record['pde'] != 'burger':
        if record['task'] == 'forward':
            masks[:, 1:] = False
        elif record['task'] == 'inverse':
            masks[:, :1] = False
        elif record['task'] != 'both':
            raise ValueError(record['task'])
    return fields, masks, ids, params
