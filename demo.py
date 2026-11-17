"""Repeatable demo: profile evidence drives the globally optimal coding DNA.

Run ``python demo.py``.  Everything is synthetic, uses only the declared
A/C/D/E alphabet, downloads nothing, and claims no expression, fitness or
laboratory effect.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import coding
import profile

ROOT = Path(__file__).resolve().parent
FIXTURES = json.loads((ROOT / "fixtures" / "cases.json").read_text())


def _round(value):
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return str(value)
        return round(value, 6)
    if isinstance(value, list):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def profile_summary(result):
    if result["status"] != "ok":
        return {"status": result["status"],
                "log_partition": _round(result["log_partition"]),
                "best_log_score": _round(result["best_log_score"]),
                "best": result["best"]}
    return {
        "status": result["status"],
        "log_partition": _round(result["log_partition"]),
        "best_log_score": _round(result["best_log_score"]),
        "best": result["best"],
        "residue_posterior": _round(result["residue_posterior"]),
        "match_posterior": _round(result["match_posterior"]),
        "delete_posterior": _round(result["delete_posterior"]),
        "insert_posterior": _round(result["insert_posterior"]),
    }


def coding_summary(result):
    if result["status"] != "ok":
        return {"status": result["status"], "dna": result["dna"]}
    return {
        "status": result["status"],
        "dna": result["dna"],
        "translation": result["translation"],
        "codons": result["codons"],
        "probability_cost": _round(result["probability_cost"]),
        "usage_cost": _round(result["usage_cost"]),
        "cost": _round(result["cost"]),
        "total_gc": result["total_gc"],
        "window_gc": result["window_gc"],
    }


def main() -> None:
    report = {}

    # 1. Ordinary gap-free matching and unconstrained coding.
    plain = profile.gapfree(FIXTURES["model"], FIXTURES["query_before"])
    plain_probs = [[1.0 if k == 0 else 0.0 for k in range(4)],
                   [0.6, 0.4, 0.0, 0.0]]
    report["ordinary"] = {
        "profile_gapfree": {"status": plain["status"],
                            "log_partition": _round(plain["log_partition"])},
        "coding_unconstrained": coding_summary(
            coding.unconstrained(plain_probs, FIXTURES["ordinary_policy"])),
    }

    # 2. Full profile analysis on the same model: insertions/deletions allowed.
    full_before = profile.analyze(FIXTURES["model"], FIXTURES["query_before"])
    full_after = profile.analyze(FIXTURES["model"], FIXTURES["query_after"])
    report["profile"] = {
        "before_query": profile_summary(full_before),
        "after_query": profile_summary(full_after),
    }

    # 3. Combined: this profile's residue_posterior feeds the same coding
    #    design; changing the observed peptide changes the chosen DNA.
    policy = FIXTURES["coupling_policy"]
    dna_before = coding.design(full_before["residue_posterior"], policy)
    dna_after = coding.design(full_after["residue_posterior"], policy)
    # Re-score the frozen (old) DNA under the new target probabilities.
    frozen_cost = None
    if dna_before["status"] == "ok" and dna_after["status"] == "ok":
        frozen_residues = [
            coding.AA_INDEX[
                [a for a, cs in coding.GENETIC_CODE.items() if c in cs][0]]
            for c in dna_before["codons"]
        ]
        frozen_cost = sum(
            -math.log(full_after["residue_posterior"][i][k])
            for i, k in enumerate(frozen_residues)
        ) + sum(policy["usage_cost"][c] for c in dna_before["codons"])
    report["combined"] = {
        "policy_forbidden": policy["forbidden"],
        "before_query": coding_summary(dna_before),
        "after_query": coding_summary(dna_after),
        "frozen_old_table_cost_under_new_target": _round(frozen_cost),
        "new_optimal_cost": _round(dna_after["cost"]),
    }

    # 4. Named edge cases.
    report["edge_cases"] = {
        "joint_map_vs_marginal": profile_summary(
            profile.analyze(FIXTURES["joint_map_case"]["model"],
                            FIXTURES["joint_map_case"]["query"])),
        "deletion_uncertainty": profile_summary(
            profile.analyze(FIXTURES["deletion_case"]["model"],
                            FIXTURES["deletion_case"]["query"])),
        "zero_mass": profile_summary(
            profile.analyze(FIXTURES["zero_mass_case"]["model"],
                            FIXTURES["zero_mass_case"]["query"])),
        "cross_codon_forbidden": coding_summary(
            coding.design(FIXTURES["cross_codon_motif"]["residue_probabilities"],
                          FIXTURES["cross_codon_motif"]["policy"])),
        "rolling_window_seam": coding_summary(
            coding.design(
                FIXTURES["rolling_window_seam"]["residue_probabilities"],
                FIXTURES["rolling_window_seam"]["policy"])),
        "total_gc": coding_summary(
            coding.design(FIXTURES["total_gc_case"]["residue_probabilities"],
                          FIXTURES["total_gc_case"]["policy"])),
        "infeasible": coding_summary(
            coding.design(FIXTURES["infeasible_coding"]["residue_probabilities"],
                          FIXTURES["infeasible_coding"]["policy"])),
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
