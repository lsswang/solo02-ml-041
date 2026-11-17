"""Global optimal DNA coding under external residue probabilities.

``design`` chooses one allowed synonymous codon per residue, so the DNA is
exactly ``3 * L`` bases long.  It minimises

    sum_i (-log P_i(selected residue)) + sum_c usage_cost(selected codon)

over every residue and codon choice jointly, subject to forbidden DNA
substrings (checked across codon boundaries) and a rolling GC window plus a
total-GC constraint.  The search is exact: it is a dynamic program over
(codon position, last <=5 DNA characters, recent GC bits, total GC) and it
never fixes a consensus residue first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

ALPHABET = "ACDE"
AA_INDEX = {a: k for k, a in enumerate(ALPHABET)}
GENETIC_CODE = {
    "A": ("GCT", "GCC", "GCA", "GCG"),
    "C": ("TGT", "TGC"),
    "D": ("GAT", "GAC"),
    "E": ("GAA", "GAG"),
}
_GENETIC_SETS = {aa: frozenset(codons) for aa, codons in GENETIC_CODE.items()}
_PROB_TOL = 1e-12
_MAX_FORBIDDEN_LEN = 6
_DNA_LETTERS = frozenset("ACGT")

State = tuple[str, tuple[int, ...], int]
# (cost, previous_state, codon, residue_index)


def _validate_probabilities(residue_probabilities: Any) -> np.ndarray:
    try:
        probs = np.array(residue_probabilities, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("residue_probabilities must be a numeric array") from exc
    if probs.ndim != 2 or probs.shape[1] != 4:
        raise ValueError("residue_probabilities must be a 2-D [L, 4] array")
    length = probs.shape[0]
    if not 1 <= length <= 10:
        raise ValueError("residue_probabilities must have between 1 and 10 rows")
    if not np.all(np.isfinite(probs)):
        raise ValueError("residue_probabilities must be finite")
    if np.any(probs < 0.0):
        raise ValueError("residue_probabilities must be non-negative")
    for row in range(length):
        total = float(np.sum(probs[row]))
        if abs(total - 1.0) > _PROB_TOL:
            raise ValueError(
                f"residue_probabilities[{row}] must sum to 1 within 1e-12"
            )
        probs[row] = probs[row] / total
    return probs


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_policy(policy: Any, length: int) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be a mapping")

    codons_input = policy.get("codons")
    if not isinstance(codons_input, Mapping):
        raise ValueError("policy.codons must be a mapping")
    missing_aa = [aa for aa in ALPHABET if aa not in codons_input]
    if missing_aa:
        raise ValueError(f"policy.codons is missing amino acids: {missing_aa}")

    codons: dict[str, tuple[str, ...]] = {}
    for aa in ALPHABET:
        listed = codons_input[aa]
        if isinstance(listed, (str, bytes)) or not isinstance(listed, (list, tuple)):
            raise ValueError(f"policy.codons[{aa}] must be a list of codons")
        if not listed:
            raise ValueError(f"policy.codons[{aa}] must be non-empty")
        seen: set[str] = set()
        values: list[str] = []
        for codon in listed:
            if not isinstance(codon, str):
                raise ValueError(f"policy.codons[{aa}] must contain strings")
            if len(codon) != 3 or any(ch not in _DNA_LETTERS for ch in codon):
                raise ValueError(f"invalid codon {codon!r} for {aa}")
            if codon not in _GENETIC_SETS[aa]:
                raise ValueError(f"codon {codon!r} does not encode {aa}")
            if codon in seen:
                raise ValueError(f"duplicate codon {codon!r} for {aa}")
            seen.add(codon)
            values.append(codon)
        codons[aa] = tuple(values)

    usage_input = policy.get("usage_cost")
    if not isinstance(usage_input, Mapping):
        raise ValueError("policy.usage_cost must be a mapping")
    allowed_codons = {c for values in codons.values() for c in values}
    usage_cost: dict[str, float] = {}
    for codon in sorted(allowed_codons):
        if codon not in usage_input:
            raise ValueError(f"usage_cost missing allowed codon {codon!r}")
        value = usage_input[codon]
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"usage_cost[{codon}] must be a number")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 20.0:
            raise ValueError(f"usage_cost[{codon}] must be finite in [0, 20]")
        usage_cost[codon] = numeric

    forbidden_input = policy.get("forbidden", [])
    if isinstance(forbidden_input, (str, bytes)) or not isinstance(
        forbidden_input, (list, tuple)
    ):
        raise ValueError("policy.forbidden must be a list of DNA substrings")
    if len(forbidden_input) > 4:
        raise ValueError("policy.forbidden may contain at most 4 substrings")
    forbidden: list[str] = []
    seen_forbidden: set[str] = set()
    for motif in forbidden_input:
        if not isinstance(motif, str):
            raise ValueError("forbidden motifs must be strings")
        if not 2 <= len(motif) <= _MAX_FORBIDDEN_LEN:
            raise ValueError("forbidden motifs must have length 2 to 6")
        if any(ch not in _DNA_LETTERS for ch in motif):
            raise ValueError(f"forbidden motif {motif!r} has invalid DNA letters")
        if motif in seen_forbidden:
            raise ValueError(f"duplicate forbidden motif {motif!r}")
        seen_forbidden.add(motif)
        forbidden.append(motif)

    width = _require_int(policy.get("window_width"), "window_width")
    if not 3 <= width <= 6:
        raise ValueError("window_width must be an integer in [3, 6]")

    window_gc_min = _require_int(policy.get("window_gc_min"), "window_gc_min")
    window_gc_max = _require_int(policy.get("window_gc_max"), "window_gc_max")
    if not 0 <= window_gc_min <= window_gc_max <= width:
        raise ValueError("require 0 <= window_gc_min <= window_gc_max <= window_width")

    dna_length = 3 * length
    total_gc_min = _require_int(policy.get("total_gc_min"), "total_gc_min")
    total_gc_max = _require_int(policy.get("total_gc_max"), "total_gc_max")
    if not 0 <= total_gc_min <= total_gc_max <= dna_length:
        raise ValueError("require 0 <= total_gc_min <= total_gc_max <= 3*L")

    return {
        "codons": codons,
        "usage_cost": usage_cost,
        "forbidden": tuple(forbidden),
        "window_width": width,
        "window_gc_min": window_gc_min,
        "window_gc_max": window_gc_max,
        "total_gc_min": total_gc_min,
        "total_gc_max": total_gc_max,
    }


def _gc(base: str) -> int:
    return 1 if base in ("G", "C") else 0


def _step_state(
    suffix: str,
    gc_bits: tuple[int, ...],
    base: str,
    forbidden: tuple[str, ...],
    keep: int,
    width: int,
    wmin: int,
    wmax: int,
) -> tuple[str, tuple[int, ...]] | None:
    """Advance the finite-state memory by one DNA base, or reject it."""
    merged = suffix + base
    for motif in forbidden:
        if motif in merged:
            return None
    bits = gc_bits + (_gc(base),)
    if len(bits) >= width:
        window = sum(bits[-width:])
        if not wmin <= window <= wmax:
            return None
    new_suffix = merged[-keep:] if keep else ""
    new_bits = bits[-(width - 1):] if width > 1 else ()
    return new_suffix, new_bits


def design(residue_probabilities: Any, policy: Any) -> dict[str, Any]:
    """Return a globally optimal coding DNA for the supplied probabilities.

    The function is independent of the profile model: it consumes only the
    ``[L, 4]`` probability table and ``policy``.  A return status of
    ``"infeasible"`` means the complete state space was exhausted with no
    legal DNA, not that a heuristic search gave up.
    """
    probs = _validate_probabilities(residue_probabilities)
    cfg = _validate_policy(policy, probs.shape[0])

    length = probs.shape[0]
    dna_length = 3 * length
    width = cfg["window_width"]
    wmin = cfg["window_gc_min"]
    wmax = cfg["window_gc_max"]
    forbidden = cfg["forbidden"]
    keep = min(
        max((len(m) for m in forbidden), default=1) - 1, _MAX_FORBIDDEN_LEN - 1
    )
    keep = max(keep, 0)

    def advance(suffix, gc_bits, base):
        return _step_state(suffix, gc_bits, base, forbidden, keep, width, wmin, wmax)

    # cost_dp[k][state] = (best_cost, previous_state, codon, residue_index).
    # Layers are retained: the state space per layer is small for L <= 10.
    initial: State = ("", (), 0)
    layers: list[dict[State, tuple[float, State | None, str, int]]] = [
        {initial: (0.0, None, "", -1)}
    ]

    for pos in range(length):
        layer = layers[-1]
        nxt: dict[State, tuple[float, State | None, str, int]] = {}
        for state, (cost_so_far, _prev, _codon, _res) in layer.items():
            suffix, gc_bits, total_gc = state
            for a_index in range(4):
                residue_prob = float(probs[pos, a_index])
                if residue_prob == 0.0:
                    continue
                residue_cost = -math.log(residue_prob)
                aa = ALPHABET[a_index]
                for codon in cfg["codons"][aa]:
                    cur_suffix, cur_bits = suffix, gc_bits
                    cur_gc = total_gc
                    valid = True
                    for base in codon:
                        built = advance(cur_suffix, cur_bits, base)
                        if built is None:
                            valid = False
                            break
                        cur_suffix, cur_bits = built
                        cur_gc += _gc(base)
                    if not valid:
                        continue
                    new_state: State = (cur_suffix, cur_bits, cur_gc)
                    cost = cost_so_far + residue_cost + cfg["usage_cost"][codon]
                    old = nxt.get(new_state)
                    if old is None or cost < old[0]:
                        nxt[new_state] = (cost, state, codon, a_index)
        layers.append(nxt)
        if not nxt:
            return _infeasible()

    final = layers[-1]
    best_state: State | None = None
    best_cost = math.inf
    for state, (cost, _prev, _codon, _res) in final.items():
        if not cfg["total_gc_min"] <= state[2] <= cfg["total_gc_max"]:
            continue
        if cost < best_cost:
            best_cost = cost
            best_state = state
    if best_state is None:
        return _infeasible()

    codons_chosen: list[str] = []
    residues_chosen: list[int] = []
    state: State | None = best_state
    for pos in range(length - 1, -1, -1):
        _cost, prev, codon, a_index = layers[pos + 1][state]
        codons_chosen.append(codon)
        residues_chosen.append(int(a_index))
        state = prev
    codons_chosen.reverse()
    residues_chosen.reverse()

    dna = "".join(codons_chosen)
    translation = "".join(ALPHABET[k] for k in residues_chosen)
    probability_cost = 0.0
    for i, k in enumerate(residues_chosen):
        probability_cost += -math.log(float(probs[i, k]))
    usage_total = sum(cfg["usage_cost"][c] for c in codons_chosen)
    total_gc = sum(_gc(b) for b in dna)
    window_gc = [
        sum(_gc(b) for b in dna[j:j + width])
        for j in range(dna_length - width + 1)
    ]

    return {
        "status": "ok",
        "dna": dna,
        "translation": translation,
        "codons": codons_chosen,
        "probability_cost": float(probability_cost),
        "usage_cost": float(usage_total),
        "cost": float(probability_cost + usage_total),
        "total_gc": int(total_gc),
        "window_gc": window_gc,
    }


def _infeasible() -> dict[str, Any]:
    return {
        "status": "infeasible",
        "dna": "",
        "translation": "",
        "codons": [],
        "probability_cost": None,
        "usage_cost": None,
        "cost": None,
        "total_gc": None,
        "window_gc": [],
    }


def unconstrained(residue_probabilities: Any, policy: Any) -> dict[str, Any]:
    """Convenience wrapper that disables all DNA constraints.

    Keeps the ordinary "no forbidden motif / no GC constraint" coding entry
    while still choosing codons globally by codon usage cost for the best
    residue combination.
    """
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be a mapping")
    probs = _validate_probabilities(residue_probabilities)
    length = probs.shape[0]
    relaxed = dict(policy)
    relaxed["forbidden"] = []
    relaxed["window_width"] = 3
    relaxed["window_gc_min"] = 0
    relaxed["window_gc_max"] = 3
    relaxed["total_gc_min"] = 0
    relaxed["total_gc_max"] = 3 * length
    return design(probs, relaxed)
