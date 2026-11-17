"""Profile evidence for short peptides with insertions and deletions.

The model is a small weighted alignment model with three states:

    M (0) consume one profile position and one query position
    D (1) consume one profile position only
    I (2) consume one query position only

Every step has a path weight: ``start[state]`` for the first step,
``transition[previous, state]`` thereafter and ``end[state]`` after the last
step.  These are path weights, *not* row-normalised probabilities.  M and D
also choose a latent true residue for the profile position that is being
consumed; I only emits the observed query residue from ``background``.

All sums are done in log space, so legal tiny probabilities do not underflow
to a false ``no_path`` result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

ALPHABET = "ACDE"
AA_INDEX = {a: k for k, a in enumerate(ALPHABET)}
M, D, I = 0, 1, 2
STATE_NAMES = ("M", "D", "I")

_PROB_TOL = 1e-12
_MIN_POSITIVE = 1e-12
_MAX_PATH_WEIGHT = 5.0
_NEG_INF = -math.inf


def _logsumexp(values: Any) -> float:
    """Stable log(sum(exp(values))) that treats -inf as an empty term."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    if vals.size == 0:
        return _NEG_INF
    m = float(np.max(vals))
    if not math.isfinite(m):
        return m
    with np.errstate(divide="ignore"):
        return m + float(np.log(np.sum(np.exp(vals - m))))


def _as_float_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        arr = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def _prepare_probability_row(values: np.ndarray, name: str) -> np.ndarray:
    if np.any((values > 0.0) & (values < _MIN_POSITIVE)):
        raise ValueError(f"{name}: positive probabilities must be at least 1e-12")
    total = float(np.sum(values))
    if abs(total - 1.0) > _PROB_TOL:
        raise ValueError(f"{name}: row sums must differ from 1 by at most 1e-12")
    if total <= 0.0:
        raise ValueError(f"{name}: row sum must be positive")
    return values / total


def _prepare_model(model: Any) -> dict[str, np.ndarray]:
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")
    required = ("theta", "substitution", "background", "start", "transition", "end")
    missing = [key for key in required if key not in model]
    if missing:
        raise ValueError(f"model is missing keys: {missing}")

    theta_shape = np.shape(model["theta"])
    if len(theta_shape) != 2 or theta_shape[1] != 4:
        raise ValueError("theta must be a 2-D array with 4 columns")
    theta = _as_float_array(model["theta"], "theta", theta_shape)
    length = theta.shape[0]
    if not 1 <= length <= 10:
        raise ValueError("theta must have between 1 and 10 positions")

    subst = _as_float_array(model["substitution"], "substitution", (4, 4))
    background = _as_float_array(model["background"], "background", (4,))
    start = _as_float_array(model["start"], "start", (3,))
    transition = _as_float_array(model["transition"], "transition", (3, 3))
    end = _as_float_array(model["end"], "end", (3,))

    for row in range(length):
        theta[row] = _prepare_probability_row(theta[row], f"theta[{row}]")
    for row in range(4):
        subst[row] = _prepare_probability_row(subst[row], f"substitution[{row}]")
    background = _prepare_probability_row(background, "background")

    for name, arr in (("start", start), ("transition", transition), ("end", end)):
        positive = arr[arr > 0.0]
        if np.any(positive < _MIN_POSITIVE) or np.any(positive > _MAX_PATH_WEIGHT):
            raise ValueError(
                f"{name}: positive path weights must be in [1e-12, 5]"
            )

    return {
        "theta": theta,
        "substitution": subst,
        "background": background,
        "start": start,
        "transition": transition,
        "end": end,
    }


def _prepare_query(query: Any) -> tuple[str, int]:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if not 1 <= len(query) <= 14:
        raise ValueError("query length must be between 1 and 14")
    for char in query:
        if char not in AA_INDEX:
            raise ValueError(f"unknown query letter: {char!r}")
    return query, len(query)


def _log(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, _NEG_INF, dtype=np.float64)
    positive = values > 0.0
    out[positive] = np.log(values[positive])
    return out


def analyze(model: Any, query: Any) -> dict[str, Any]:
    """Analyse ``query`` against ``model`` with the full M/D/I state space.

    The result contains the log partition function, one joint-MAP path
    (states *and* latent residues), and the requested posteriors.  If no
    positive-mass complete path exists, ``status`` is ``"no_path"`` and every
    posterior is ``None``.
    """
    params = _prepare_model(model)
    query, m_length = _prepare_query(query)

    theta = params["theta"]
    subst = params["substitution"]
    background = params["background"]
    start = params["start"]
    transition = params["transition"]
    end = params["end"]
    length = theta.shape[0]

    log_theta = _log(theta)
    log_subst = _log(subst)
    log_background = _log(background)
    log_start = _log(start)
    log_transition = _log(transition)
    log_end = _log(end)

    obs = np.array([AA_INDEX[c] for c in query], dtype=np.int64)

    # ------------------------------------------------------------------
    # Local log masses for actions at source coordinates (i, j).
    # ------------------------------------------------------------------
    # D action at profile index i.
    log_d_local = np.full(length, _NEG_INF, dtype=np.float64)
    for i in range(length):
        log_d_local[i] = _logsumexp(log_theta[i])

    # I action at query index j.
    log_i_local = np.full(m_length, _NEG_INF, dtype=np.float64)
    for j in range(m_length):
        log_i_local[j] = log_background[obs[j]]

    # M action at profile index i and query index j; per-residue terms are
    # also needed for the residue posterior.
    log_m_local = np.full((length, m_length), _NEG_INF, dtype=np.float64)
    log_m_residue = np.full((length, m_length, 4), _NEG_INF, dtype=np.float64)
    for i in range(length):
        for j in range(m_length):
            terms = log_theta[i] + log_subst[:, obs[j]]
            log_m_residue[i, j] = terms
            log_m_local[i, j] = _logsumexp(terms)

    # ------------------------------------------------------------------
    # Forward pass.  fwd[s][i, j] is the total log mass of paths that have
    # consumed (i, j) and whose last action was state s.  pre[s][i, j] is
    # the start/transition factor entering action s from source (i, j),
    # excluding that action's local mass.
    # ------------------------------------------------------------------
    fwd = np.full((3, length + 1, m_length + 1), _NEG_INF, dtype=np.float64)
    pre = np.full((3, length + 1, m_length + 1), _NEG_INF, dtype=np.float64)

    for i in range(length + 1):
        for j in range(m_length + 1):
            if i == length and j == m_length:
                continue
            if i == 0 and j == 0:
                pre[:, i, j] = log_start
            else:
                for state in range(3):
                    pre[state, i, j] = _logsumexp(
                        [fwd[prev, i, j] + log_transition[prev, state]
                         for prev in range(3)]
                    )

            if (
                i < length
                and j < m_length
                and log_m_local[i, j] != _NEG_INF
                and pre[M, i, j] != _NEG_INF
            ):
                fwd[M, i + 1, j + 1] = pre[M, i, j] + log_m_local[i, j]
            if (
                i < length
                and log_d_local[i] != _NEG_INF
                and pre[D, i, j] != _NEG_INF
            ):
                fwd[D, i + 1, j] = pre[D, i, j] + log_d_local[i]
            if (
                j < m_length
                and log_i_local[j] != _NEG_INF
                and pre[I, i, j] != _NEG_INF
            ):
                fwd[I, i, j + 1] = pre[I, i, j] + log_i_local[j]

    # ------------------------------------------------------------------
    # Backward pass.  bwd[s][i, j] is the total log mass of all
    # continuations after arriving at (i, j) with last state s, including
    # the final end weight.  bwd[s][L, M] = log end[s].
    # ------------------------------------------------------------------
    bwd = np.full((3, length + 1, m_length + 1), _NEG_INF, dtype=np.float64)
    bwd[:, length, m_length] = log_end

    for i in range(length, -1, -1):
        for j in range(m_length, -1, -1):
            if i == length and j == m_length:
                continue
            for state in range(3):
                terms: list[float] = []
                if (
                    i < length
                    and j < m_length
                    and log_m_local[i, j] != _NEG_INF
                ):
                    terms.append(
                        log_transition[state, M]
                        + log_m_local[i, j]
                        + bwd[M, i + 1, j + 1]
                    )
                if i < length and log_d_local[i] != _NEG_INF:
                    terms.append(
                        log_transition[state, D]
                        + log_d_local[i]
                        + bwd[D, i + 1, j]
                    )
                if j < m_length and log_i_local[j] != _NEG_INF:
                    terms.append(
                        log_transition[state, I]
                        + log_i_local[j]
                        + bwd[I, i, j + 1]
                    )
                bwd[state, i, j] = _logsumexp(terms)

    log_partition = _logsumexp(
        [fwd[state, length, m_length] + bwd[state, length, m_length]
         for state in range(3)]
    )

    # ------------------------------------------------------------------
    # Joint MAP (Viterbi) path.  Residue choices do not interact across
    # positions, so each M/D action takes its own best latent residue.
    # ------------------------------------------------------------------
    vit = np.full((3, length + 1, m_length + 1), _NEG_INF, dtype=np.float64)
    back: list[list[list[Any]]] = [
        [[None for _ in range(m_length + 1)] for _ in range(length + 1)]
        for _ in range(3)
    ]

    best_m_res = np.full((length, m_length), -1, dtype=np.int64)
    best_m_score = np.full((length, m_length), _NEG_INF, dtype=np.float64)
    for i in range(length):
        for j in range(m_length):
            a = int(np.argmax(log_m_residue[i, j]))
            if log_m_residue[i, j, a] != _NEG_INF:
                best_m_res[i, j] = a
                best_m_score[i, j] = log_m_residue[i, j, a]
    best_d_res = np.argmax(log_theta, axis=1).astype(np.int64)
    best_d_score = np.full(length, _NEG_INF, dtype=np.float64)
    for i in range(length):
        if log_theta[i, best_d_res[i]] != _NEG_INF:
            best_d_score[i] = log_theta[i, best_d_res[i]]
        else:
            best_d_res[i] = -1

    def _best_prev(i: int, j: int, state: int) -> tuple[float, int | None]:
        if i == 0 and j == 0:
            return float(log_start[state]), None
        best_val = _NEG_INF
        best_prev_state: int | None = None
        for prev in range(3):
            value = vit[prev, i, j] + log_transition[prev, state]
            if value > best_val:
                best_val = value
                best_prev_state = prev
        return float(best_val), best_prev_state

    for i in range(length + 1):
        for j in range(m_length + 1):
            if i == length and j == m_length:
                continue
            if i < length and j < m_length and best_m_score[i, j] != _NEG_INF:
                value, prev = _best_prev(i, j, M)
                if value != _NEG_INF:
                    vit[M, i + 1, j + 1] = value + best_m_score[i, j]
                    back[M][i + 1][j + 1] = (i, j, prev, int(best_m_res[i, j]))
            if i < length and best_d_score[i] != _NEG_INF:
                value, prev = _best_prev(i, j, D)
                if value != _NEG_INF:
                    vit[D, i + 1][j] = value + best_d_score[i]
                    back[D][i + 1][j] = (i, j, prev, int(best_d_res[i]))
            if j < m_length and log_i_local[j] != _NEG_INF:
                value, prev = _best_prev(i, j, I)
                if value != _NEG_INF:
                    vit[I][i][j + 1] = value + log_i_local[j]
                    back[I][i][j + 1] = (i, j, prev, None)

    best_log_score = _NEG_INF
    best_final_state: int | None = None
    for state in range(3):
        score = vit[state, length, m_length] + log_end[state]
        if score > best_log_score:
            best_log_score = float(score)
            best_final_state = state

    if log_partition == _NEG_INF:
        return {
            "status": "no_path",
            "log_partition": _NEG_INF,
            "best": [],
            "best_log_score": _NEG_INF,
            "residue_posterior": None,
            "match_posterior": None,
            "delete_posterior": None,
            "insert_posterior": None,
        }

    best_path: list[dict[str, Any]] = []
    if best_final_state is not None and best_log_score != _NEG_INF:
        state: int | None = best_final_state
        i, j = length, m_length
        while state is not None:
            entry = back[state][i][j]
            if entry is None:
                break
            src_i, src_j, prev_state, residue = entry
            step: dict[str, Any] = {
                "state": STATE_NAMES[state],
                "state_index": int(state),
                "i": int(src_i),
                "j": int(src_j),
            }
            if residue is not None:
                step["residue"] = ALPHABET[int(residue)]
            best_path.append(step)
            i, j = int(src_i), int(src_j)
            state = prev_state
        best_path.reverse()

    # ------------------------------------------------------------------
    # Posteriors.
    # ------------------------------------------------------------------
    match_posterior = np.zeros((length, m_length), dtype=np.float64)
    delete_posterior = np.zeros((length, m_length + 1), dtype=np.float64)
    insert_posterior = np.zeros((length + 1, m_length), dtype=np.float64)

    for i in range(length):
        for j in range(m_length):
            if log_m_local[i, j] != _NEG_INF:
                contribution = (
                    pre[M, i, j] + log_m_local[i, j] + bwd[M, i + 1, j + 1]
                )
                match_posterior[i, j] = math.exp(contribution - log_partition)
        for j in range(m_length + 1):
            if log_d_local[i] != _NEG_INF:
                contribution = pre[D, i, j] + log_d_local[i] + bwd[D, i + 1, j]
                delete_posterior[i, j] = math.exp(contribution - log_partition)
    for i in range(length + 1):
        for j in range(m_length):
            if log_i_local[j] != _NEG_INF:
                contribution = pre[I, i, j] + log_i_local[j] + bwd[I, i, j + 1]
                insert_posterior[i, j] = math.exp(contribution - log_partition)

    residue_posterior = np.zeros((length, 4), dtype=np.float64)
    for i in range(length):
        log_numerators = np.full(4, _NEG_INF, dtype=np.float64)
        for a in range(4):
            if log_theta[i, a] == _NEG_INF:
                continue
            terms: list[float] = []
            for j in range(m_length):
                if log_m_residue[i, j, a] != _NEG_INF:
                    terms.append(
                        pre[M, i, j]
                        + log_m_residue[i, j, a]
                        + bwd[M, i + 1, j + 1]
                    )
            for j in range(m_length + 1):
                terms.append(pre[D, i, j] + log_theta[i, a] + bwd[D, i + 1, j])
            log_numerators[a] = _logsumexp(terms)
        residue_posterior[i] = np.exp(log_numerators - log_partition)

    return {
        "status": "ok",
        "log_partition": float(log_partition),
        "best": best_path,
        "best_log_score": float(best_log_score),
        "residue_posterior": residue_posterior.tolist(),
        "match_posterior": match_posterior.tolist(),
        "delete_posterior": delete_posterior.tolist(),
        "insert_posterior": insert_posterior.tolist(),
    }


def gapfree(model: Any, query: Any) -> dict[str, Any]:
    """Ordinary equal-length, no-gap matching kept for comparison.

    This is the degenerate case where every profile position is matched
    (all-M path).  It returns the same simple posterior shape as the original
    starter but is now log-safe.
    """
    params = _prepare_model(model)
    query, m_length = _prepare_query(query)
    length = params["theta"].shape[0]
    if m_length != length:
        raise ValueError("gap-free matching requires equal query length")

    theta = params["theta"]
    subst = params["substitution"]
    log_start = _log(params["start"])
    log_transition = _log(params["transition"])
    log_end = _log(params["end"])

    rows = np.zeros((length, 4), dtype=np.float64)
    for i in range(length):
        obs = AA_INDEX[query[i]]
        rows[i] = theta[i] * subst[:, obs]

    log_weight = float(log_start[M]) + float(log_end[M])
    if length > 1:
        log_weight += (length - 1) * float(log_transition[M, M])
    normalizers = np.sum(rows, axis=1)
    if log_weight == _NEG_INF or np.any(normalizers == 0.0):
        return {
            "status": "no_path",
            "log_partition": _NEG_INF,
            "residue_posterior": None,
        }
    log_partition = float(np.sum(np.log(normalizers)) + log_weight)
    return {
        "status": "ok",
        "log_partition": log_partition,
        "residue_posterior": (rows / normalizers[:, None]).tolist(),
    }
