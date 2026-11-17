"""Independent brute-force references used only by the test suite.

These routines enumerate every complete alignment path (and every latent
residue choice) and every complete codon string of the short problems the
tests use.  They deliberately do not import ``profile`` or ``coding`` so the
main solvers are checked against a separate implementation.
"""

from __future__ import annotations

import itertools
import math

ALPHABET = "ACDE"
AA_INDEX = {a: k for k, a in enumerate(ALPHABET)}
GENETIC_CODE = {
    "A": ("GCT", "GCC", "GCA", "GCG"),
    "C": ("TGT", "TGC"),
    "D": ("GAT", "GAC"),
    "E": ("GAA", "GAG"),
}


def _all_paths(length: int, m_length: int):
    paths = []

    def walk(i, j, seq):
        if i == length and j == m_length:
            paths.append(tuple(seq))
            return
        if i < length and j < m_length:
            seq.append((0, i, j))
            walk(i + 1, j + 1, seq)
            seq.pop()
        if i < length:
            seq.append((1, i, j))
            walk(i + 1, j, seq)
            seq.pop()
        if j < m_length:
            seq.append((2, i, j))
            walk(i, j + 1, seq)
            seq.pop()

    walk(0, 0, [])
    return paths


def brute_profile(model, query):
    theta = model["theta"]
    subst = model["substitution"]
    background = model["background"]
    start = model["start"]
    transition = model["transition"]
    end = model["end"]
    length = len(theta)
    m_length = len(query)

    total = 0.0
    best = -math.inf
    residue_numer = [[0.0] * 4 for _ in range(length)]
    match = [[0.0] * m_length for _ in range(length)]
    delete = [[0.0] * (m_length + 1) for _ in range(length)]
    insert = [[0.0] * m_length for _ in range(length + 1)]

    for path in _all_paths(length, m_length):
        n_latent = sum(1 for s, _, _ in path if s in (0, 1))
        for residues in itertools.product(range(4), repeat=n_latent):
            weight = start[path[0][0]]
            for k in range(1, len(path)):
                weight *= transition[path[k - 1][0]][path[k][0]]
            weight *= end[path[-1][0]]
            chosen = {}
            r_index = 0
            for state, i, j in path:
                if state == 0:
                    a = residues[r_index]
                    r_index += 1
                    weight *= theta[i][a] * subst[a][AA_INDEX[query[j]]]
                    chosen[i] = a
                elif state == 1:
                    a = residues[r_index]
                    r_index += 1
                    weight *= theta[i][a]
                    chosen[i] = a
                else:
                    weight *= background[AA_INDEX[query[j]]]
            if weight == 0.0:
                continue
            total += weight
            if weight > best:
                best = weight
            for state, i, j in path:
                if state == 0:
                    match[i][j] += weight
                elif state == 1:
                    delete[i][j] += weight
                else:
                    insert[i][j] += weight
            for i, a in chosen.items():
                residue_numer[i][a] += weight

    if total == 0.0:
        return None
    return {
        "log_partition": math.log(total),
        "best_log_score": math.log(best),
        "residue_posterior": [
            [value / total for value in row] for row in residue_numer
        ],
        "match_posterior": [[v / total for v in row] for row in match],
        "delete_posterior": [[v / total for v in row] for row in delete],
        "insert_posterior": [[v / total for v in row] for row in insert],
    }


def _gc(char):
    return 1 if char in ("G", "C") else 0


def brute_coding(probabilities, policy):
    length = len(probabilities)
    window = policy["window_width"]
    options = []
    for i in range(length):
        per_position = []
        for a_index, aa in enumerate(ALPHABET):
            if probabilities[i][a_index] == 0:
                continue
            residue_cost = -math.log(probabilities[i][a_index])
            for codon in policy["codons"][aa]:
                per_position.append(
                    (codon, a_index, residue_cost + policy["usage_cost"][codon])
                )
        options.append(per_position)

    best_cost = math.inf
    best = None
    for combo in itertools.product(*options):
        dna = "".join(item[0] for item in combo)
        if any(motif in dna for motif in policy["forbidden"]):
            continue
        total_gc = sum(_gc(ch) for ch in dna)
        if not policy["total_gc_min"] <= total_gc <= policy["total_gc_max"]:
            continue
        if any(
            not policy["window_gc_min"]
            <= sum(_gc(ch) for ch in dna[j:j + window])
            <= policy["window_gc_max"]
            for j in range(len(dna) - window + 1)
        ):
            continue
        cost = sum(item[2] for item in combo)
        if cost < best_cost:
            best_cost = cost
            best = combo
    if best is None:
        return None
    return {
        "cost": best_cost,
        "dna": "".join(item[0] for item in best),
        "translation": "".join(ALPHABET[item[1]] for item in best),
        "codons": [item[0] for item in best],
    }
