"""Tests for profile evidence and coding DNA design.

Run with ``python -m unittest discover -s tests -v`` (or ``python -m pytest``).
The brute-force reference in ``reference.py`` is an independent enumerator;
the main solvers are never used to validate each other.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

import coding
import profile
from tests import reference

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = json.loads((ROOT / "fixtures" / "cases.json").read_text())


def _rand_model(rng, length, m_length):
    theta = rng.dirichlet([0.7] * 4, size=length)
    subst = rng.dirichlet([0.7] * 4, size=4)
    background = rng.dirichlet([0.7] * 4)
    start = rng.uniform(0.05, 4.0, size=3)
    transition = rng.uniform(0.05, 4.0, size=(3, 3))
    end = rng.uniform(0.05, 4.0, size=3)
    return {
        "theta": theta,
        "substitution": subst,
        "background": background,
        "start": start,
        "transition": transition,
        "end": end,
    }


class ProfileTests(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-9, msg=""):
        self.assertTrue(
            np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                        atol=atol, rtol=0.0),
            f"{msg}: {a} != {b}",
        )

    def test_random_against_enumeration(self):
        rng = np.random.default_rng(20240501)
        for _ in range(120):
            length = int(rng.integers(1, 4))
            m_length = int(rng.integers(1, 5))
            model = _rand_model(rng, length, m_length)
            query = "".join(rng.choice(list("ACDE"), size=m_length))
            result = profile.analyze(model, query)
            expected = reference.brute_profile(model, query)
            self.assertEqual(result["status"], "ok")
            self.assert_close(result["log_partition"], expected["log_partition"])
            self.assert_close(result["best_log_score"], expected["best_log_score"])
            for key in ("residue_posterior", "match_posterior",
                        "delete_posterior", "insert_posterior"):
                self.assert_close(result[key], expected[key], msg=key)

    def test_posterior_marginals_are_consistency_identities(self):
        rng = np.random.default_rng(7)
        for _ in range(40):
            model = _rand_model(rng, int(rng.integers(1, 5)),
                                int(rng.integers(1, 6)))
            query = "".join(rng.choice(list("ACDE"), size=len(model["theta"]) + 1))
            result = profile.analyze(model, query)
            match = np.array(result["match_posterior"])
            delete = np.array(result["delete_posterior"])
            insert = np.array(result["insert_posterior"])
            # Each profile row: M plus D sum to one.
            self.assert_close(match.sum(axis=1) + delete.sum(axis=1),
                              np.ones(match.shape[0]))
            # Each query column: M plus I sum to one.
            self.assert_close(match.sum(axis=0) + insert.sum(axis=0),
                              np.ones(match.shape[1]))
            # Residue posteriors are proper distributions.
            self.assert_close(np.array(result["residue_posterior"]).sum(axis=1),
                              np.ones(match.shape[0]))

    def test_joint_map_may_differ_from_marginal_choice(self):
        case = FIXTURES["joint_map_case"]
        result = profile.analyze(case["model"], case["query"])
        marginal = [profile.ALPHABET[k] for k in
                    np.argmax(result["residue_posterior"], axis=1)]
        map_residues = [step["residue"] for step in result["best"]
                        if "residue" in step]
        # The MAP path picks D on position 0 even though the marginal argmax is
        # A, so they are not the same construction.
        self.assertNotEqual(marginal[0], map_residues[0])
        expected = reference.brute_profile(case["model"], case["query"])
        self.assert_close(result["log_partition"], expected["log_partition"])
        self.assert_close(result["best_log_score"], expected["best_log_score"])

    def test_deleted_positions_still_have_residue_posterior(self):
        case = FIXTURES["deletion_case"]
        result = profile.analyze(case["model"], case["query"])
        expected = reference.brute_profile(case["model"], case["query"])
        # Position 1 is deleted with probability mass; its residue posterior is
        # still a full distribution over the four residues, not a fifth state.
        self.assert_close(result["residue_posterior"], expected["residue_posterior"])
        self.assertAlmostEqual(sum(result["residue_posterior"][1]), 1.0, places=12)
        self.assertGreater(np.array(result["delete_posterior"]).sum(), 0.0)

    def test_no_path_returns_full_shape(self):
        case = FIXTURES["zero_mass_case"]
        result = profile.analyze(case["model"], case["query"])
        self.assertEqual(result["status"], "no_path")
        self.assertEqual(result["log_partition"], -math.inf)
        self.assertEqual(result["best_log_score"], -math.inf)
        self.assertEqual(result["best"], [])
        for key in ("residue_posterior", "match_posterior",
                    "delete_posterior", "insert_posterior"):
            self.assertIsNone(result[key])

    def test_tiny_positive_mass_is_not_underflow(self):
        tiny = 1e-12
        model = {
            "theta": [[tiny, 1 - tiny, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
            "substitution": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "background": [0.25, 0.25, 0.25, 0.25],
            "start": [tiny, tiny, tiny],
            "transition": [[tiny] * 3, [tiny] * 3, [tiny] * 3],
            "end": [tiny, tiny, tiny],
        }
        result = profile.analyze(model, "A")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(math.isfinite(result["log_partition"]))
        self.assertLess(result["log_partition"], -50)

    def test_gapfree_matches_original_formula(self):
        model = FIXTURES["model"]
        result = profile.gapfree(model, "AA")
        theta = np.array(model["theta"])
        channel = np.array(model["substitution"])
        rows = theta * channel[:, [0, 0]].T
        normalizers = rows.sum(axis=1)
        expected = (np.log(normalizers).sum()
                    + math.log(model["start"][0] * model["end"][0]
                               * model["transition"][0][0]))
        self.assert_close(result["log_partition"], expected)
        self.assert_close(result["residue_posterior"], rows / normalizers[:, None])

    def test_input_validation(self):
        good = FIXTURES["model"]
        with self.assertRaises(ValueError):
            profile.analyze(good, "AX")
        with self.assertRaises(ValueError):
            profile.analyze(good, "")  # empty query
        bad = json.loads(json.dumps(good))
        bad["theta"][0][0] += 0.5
        with self.assertRaises(ValueError):
            profile.analyze(bad, "AA")
        bad = json.loads(json.dumps(good))
        bad["start"] = [0.0, 0.0, 0.0]  # all-zero start -> normal no_path
        self.assertEqual(profile.analyze(bad, "AA")["status"], "no_path")
        bad = json.loads(json.dumps(good))
        bad["transition"][0][0] = 5.5
        with self.assertRaises(ValueError):
            profile.analyze(bad, "AA")
        bad = json.loads(json.dumps(good))
        bad["theta"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            profile.analyze(bad, "AA")

    def test_input_is_not_modified(self):
        model = FIXTURES["model"]
        snapshot = json.loads(json.dumps(model))
        profile.analyze(model, "AA")
        self.assertEqual(model, snapshot)


class CodingTests(unittest.TestCase):
    def assert_close(self, a, b, atol=1e-8, rtol=1e-9):
        self.assertTrue(math.isclose(a, b, abs_tol=atol, rel_tol=rtol), (a, b))

    def _policy(self, forbidden=(), width=3, wmin=0, wmax=3, tmin=0, tmax=6,
                usage=None):
        allc = ["GCT", "GCC", "GCA", "GCG", "TGT", "TGC",
                "GAT", "GAC", "GAA", "GAG"]
        costs = {c: 0.0 for c in allc}
        if usage:
            costs.update(usage)
        return {
            "codons": {aa: list(cs) for aa, cs in coding.GENETIC_CODE.items()},
            "usage_cost": costs,
            "forbidden": list(forbidden),
            "window_width": width,
            "window_gc_min": wmin,
            "window_gc_max": wmax,
            "total_gc_min": tmin,
            "total_gc_max": tmax,
        }

    def _check_against_reference(self, probabilities, policy):
        result = coding.design(probabilities, policy)
        expected = reference.brute_coding(probabilities, policy)
        if expected is None:
            self.assertEqual(result["status"], "infeasible")
            self.assertEqual(result["dna"], "")
            return None
        self.assertEqual(result["status"], "ok")
        self.assert_close(result["cost"], expected["cost"])
        self.assertEqual(result["translation"], expected["translation"])
        self._check_facts(result, probabilities, policy)
        return result

    def _check_facts(self, result, probabilities, policy):
        dna = result["dna"]
        L = len(probabilities)
        self.assertEqual(len(dna), 3 * L)
        self.assertEqual(len(result["codons"]), L)
        translation = ""
        for i, codon in enumerate(result["codons"]):
            aa = [a for a, cs in coding.GENETIC_CODE.items() if codon in cs][0]
            indices = coding.AA_INDEX[aa]
            self.assertGreater(
                probabilities[i][indices] if not isinstance(probabilities, np.ndarray)
                else probabilities[i][indices], 0.0)
            translation += aa
        self.assertEqual(translation, result["translation"])
        self.assertEqual(dna, "".join(result["codons"]))
        # forbidden substring check over the whole string (cross-codon too)
        for motif in policy["forbidden"]:
            self.assertNotIn(motif, dna)
        # window GC
        width = policy["window_width"]
        windows = [sum(ch in "GC" for ch in dna[j:j + width])
                   for j in range(len(dna) - width + 1)]
        self.assertEqual(windows, result["window_gc"])
        for value in windows:
            self.assertTrue(policy["window_gc_min"] <= value
                            <= policy["window_gc_max"])
        total = sum(ch in "GC" for ch in dna)
        self.assertEqual(total, result["total_gc"])
        self.assertTrue(policy["total_gc_min"] <= total <= policy["total_gc_max"])
        recomputed = sum(
            -math.log(
                probabilities[i][coding.AA_INDEX[r]]
                if not isinstance(probabilities, np.ndarray)
                else probabilities[i][coding.AA_INDEX[r]]
            )
            for i, r in enumerate(result["translation"])
        )
        recomputed += sum(policy["usage_cost"][c] for c in result["codons"])
        self.assert_close(result["cost"], recomputed)

    def test_random_against_enumeration(self):
        rng = np.random.default_rng(99)
        for trial in range(120):
            length = int(rng.integers(1, 4))
            probs = rng.dirichlet([1.0] * 4, size=length)
            width = int(rng.integers(3, 7))
            wmin = int(rng.integers(0, width + 1))
            wmax = int(rng.integers(wmin, width + 1))
            tmin = int(rng.integers(0, 3 * length + 1))
            tmax = int(rng.integers(tmin, 3 * length + 1))
            forbidden = [[], ["CTG"], ["TA"], ["GCG"], ["GCA", "TT"]][
                int(rng.integers(0, 5))
            ]
            usage = {c: float(rng.uniform(0, 4))
                     for c in ("GCT", "GCC", "GCA", "GCG", "TGT", "TGC",
                               "GAT", "GAC", "GAA", "GAG")}
            policy = self._policy(forbidden=forbidden, width=width, wmin=wmin,
                                  wmax=wmax, tmin=tmin, tmax=tmax, usage=usage)
            self._check_against_reference(probs.tolist(), policy)

    def test_zero_probability_removes_residue(self):
        result = coding.design([[0.0, 0.0, 0.0, 1.0]],
                               self._policy(tmax=3))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["translation"], "E")
        self.assertIn(result["dna"], ("GAA", "GAG"))

    def test_tiny_probability_accepted(self):
        probs = [[1e-300, 1 - 1e-300, 0.0, 0.0]]
        policy = self._policy(tmax=3)
        result = coding.design(probs, policy)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["translation"], "C")
        self.assertAlmostEqual(result["cost"], -math.log(1 - 1e-300), places=8)

    def test_cross_codon_forbidden_motif(self):
        case = FIXTURES["cross_codon_motif"]
        result = self._check_against_reference(
            case["residue_probabilities"], case["policy"])
        self.assertNotIn("CTG", result["dna"])

    def test_cross_window_seam(self):
        case = FIXTURES["rolling_window_seam"]
        result = self._check_against_reference(
            case["residue_probabilities"], case["policy"])
        # Greedy "cheapest codon first" would choose GCT+GCT with a GC-4 seam
        # window; the true optimum moves the second codon to satisfy GC 5.
        self.assertEqual(result["dna"], "GCTGCC")
        self.assertEqual(result["window_gc"], [5])

    def test_total_gc_constraint(self):
        case = FIXTURES["total_gc_case"]
        result = self._check_against_reference(
            case["residue_probabilities"], case["policy"])
        self.assertEqual(result["total_gc"], 4)

    def test_infeasible_is_real(self):
        case = FIXTURES["infeasible_coding"]
        result = coding.design(case["residue_probabilities"], case["policy"])
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["dna"], "")
        self.assertIsNone(result["cost"])
        self.assertIsNone(reference.brute_coding(
            case["residue_probabilities"], case["policy"]))

    def test_input_validation(self):
        policy = self._policy(tmax=3)
        with self.assertRaises(ValueError):
            coding.design([[0.5, 0.5, 0.1, 0.0]], policy)  # bad sum
        with self.assertRaises(ValueError):
            coding.design([[1.0, 0.0, 0.0, 0.0, 0.0]], policy)  # shape
        with self.assertRaises(ValueError):
            coding.design([[-0.1, 0.6, 0.5, 0.0]], policy)  # negative
        bad = self._policy(tmax=3)
        bad["codons"] = dict(bad["codons"])
        bad["codons"]["A"] = ["GCT", "GCT"]
        with self.assertRaises(ValueError):
            coding.design([[1, 0, 0, 0]], bad)  # duplicate codon
        bad = self._policy(tmax=3)
        bad["codons"] = dict(bad["codons"])
        bad["codons"]["A"] = ["GCT", "AAA"]
        with self.assertRaises(ValueError):
            coding.design([[1, 0, 0, 0]], bad)  # wrong genetic code
        bad = self._policy(tmax=3)
        bad["forbidden"] = ["AA", "AA"]
        with self.assertRaises(ValueError):
            coding.design([[1, 0, 0, 0]], bad)  # duplicate motif
        bad = self._policy(tmax=3)
        bad["window_width"] = 7
        with self.assertRaises(ValueError):
            coding.design([[1, 0, 0, 0]], bad)
        bad = self._policy(tmax=3)
        bad["usage_cost"] = dict(bad["usage_cost"])
        bad["usage_cost"]["GCT"] = 21.0
        with self.assertRaises(ValueError):
            coding.design([[1, 0, 0, 0]], bad)

    def test_input_is_not_modified(self):
        probs = [[1, 0, 0, 0], [0, 0, 1, 0]]
        policy = self._policy(tmax=6)
        snapshot = json.loads(json.dumps(policy))
        coding.design(probs, policy)
        self.assertEqual(policy, snapshot)

    def test_unconstrained_wrapper_matches_reference(self):
        probs = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
        policy = self._policy(tmax=6)
        relaxed = dict(policy)
        relaxed["forbidden"] = []
        relaxed["window_gc_min"] = 0
        relaxed["window_gc_max"] = 3
        relaxed["total_gc_min"] = 0
        relaxed["total_gc_max"] = 6
        result = coding.unconstrained(probs, policy)
        expected = reference.brute_coding(probs, relaxed)
        self.assert_close(result["cost"], expected["cost"])


class CombinedTests(unittest.TestCase):
    def test_posterior_drives_dna_and_frozen_table_is_worse(self):
        policy = FIXTURES["coupling_policy"]
        before = profile.analyze(FIXTURES["model"], "AA")
        after = profile.analyze(FIXTURES["model"], "CC")
        dna_new = coding.design(after["residue_posterior"], policy)
        dna_old = coding.design(before["residue_posterior"], policy)
        self.assertEqual(dna_new["translation"], "CC")
        self.assertEqual(dna_old["translation"], "AA")
        # Evaluate the old DNA under the new target probabilities.
        codons = [dna_old["dna"][3 * i:3 * i + 3]
                  for i in range(len(after["residue_posterior"]))]
        residues = [coding.AA_INDEX[
            [a for a, cs in coding.GENETIC_CODE.items() if c in cs][0]]
            for c in codons]
        frozen_cost = sum(
            -math.log(after["residue_posterior"][i][k])
            for i, k in enumerate(residues)
        ) + sum(policy["usage_cost"][c] for c in codons)
        self.assertGreater(frozen_cost, dna_new["cost"] + 1e-6)

    def test_no_path_does_not_fabricate_probabilities(self):
        case = FIXTURES["zero_mass_case"]
        result = profile.analyze(case["model"], case["query"])
        self.assertEqual(result["status"], "no_path")
        self.assertIsNone(result["residue_posterior"])


if __name__ == "__main__":
    unittest.main()
