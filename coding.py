"""Existing per-position consensus and first-codon encoder."""
import math
import numpy as np

ALPHABET = "ACDE"


def design(residue_probabilities, policy):
    identities = [int(np.argmax(row)) for row in residue_probabilities]
    translation = "".join(ALPHABET[k] for k in identities)
    codons = [policy["codons"][aa][0] for aa in translation]
    probability_cost = -sum(math.log(residue_probabilities[i][k])
                            for i, k in enumerate(identities))
    usage_cost = sum(policy["usage_cost"][c] for c in codons)
    dna = "".join(codons)
    width = policy["window_width"]
    return {"status": "ok", "dna": dna, "translation": translation,
            "codons": codons, "probability_cost": probability_cost,
            "usage_cost": usage_cost, "cost": probability_cost + usage_cost,
            "total_gc": sum(c in "GC" for c in dna),
            "window_gc": [sum(c in "GC" for c in dna[j:j + width])
                          for j in range(len(dna) - width + 1)]}
