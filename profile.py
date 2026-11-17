"""Existing gap-free independent profile evidence on equal-length peptides."""
import math
import numpy as np

ALPHABET = "ACDE"


def analyze(model, query):
    theta = np.array(model["theta"], dtype=np.float64)
    channel = np.asarray(model["substitution"], dtype=np.float64)
    length = len(theta)
    if len(query) != length:
        raise ValueError("gap-free input requires equal lengths")
    rows = theta * channel[:, [ALPHABET.index(c) for c in query]].T
    normalizers = rows.sum(axis=1)
    weight = model["start"][0] * model["end"][0]
    weight *= model["transition"][0][0] ** (length - 1)
    if not weight or np.any(normalizers == 0):
        return {"status": "no_path", "log_partition": -math.inf,
                "residue_posterior": None}
    return {"status": "ok", "log_partition": float(np.log(normalizers).sum() + math.log(weight)),
            "residue_posterior": (rows / normalizers[:, None]).tolist()}
