# Short profile evidence and coding DNA

Synthetic, CPU-only Python 3.11 examples for two problems:

1. **Profile analysis with insertions and deletions.** `profile.analyze` sums
   every complete M/D/I alignment path of a short peptide against a profile,
   keeps all latent-residue choices, and returns the log partition function,
   one true joint-MAP path, and residue/event posteriors.
2. **Global DNA coding under external residue probabilities.**
   `coding.design` chooses a residue and a synonymous codon at every position
   jointly, minimising probability plus codon-usage cost while honouring
   forbidden DNA substrings (also across codon boundaries), rolling GC
   windows, and a total-GC bound.

Only the declared amino acids **A, C, D, E** and the fixed genetic code are
used. No model is downloaded and no expression, fitness, or laboratory
outcome is modeled or claimed.

## Commands

```bash
pip install -r requirements.txt          # NumPy only
python demo.py                           # structured JSON report
python -m unittest discover -s tests -v  # independent brute-force checks
```

The demo prints the gap-free/unconstrained baseline, the full profile result
for two observed peptides, the coding DNA induced by each posterior, a frozen
old-table control, and one named case for each required edge condition.

## Profile semantics (`profile.py`)

A step is one of `M` (consume one profile position `i` and one query position
`j`), `D` (consume `i` only), or `I` (consume `j` only). Coordinates run from
`(0,0)` to `(L,M)` with no skip edges and no restart/duration terms.

* `M` local mass: `theta[i,a] * Q[a, query[j]]`.
* `D` local mass: `theta[i,a]` (a deleted profile position still chooses a
  latent residue).
* `I` local mass: `background[query[j]]` (no invented profile residue).
* Path weight: `start[first] * ... transition[prev,next] ... * end[last]`.
  These are path weights, not normalised probabilities, and path mass is not
  described as a distribution over all peptide strings.

`analyze(model, query)` returns:

* `status` — `"ok"` or `"no_path"`;
* `log_partition` — log of the total positive path-plus-residue mass;
* `best` — a genuine joint-MAP step list (`state`, `state_index`, pre-action
  `i`/`j`, and the latent `residue` for M/D), not a stitch of per-position
  marginal argmaxes;
* `best_log_score` — its log mass;
* `residue_posterior[L][4]` — conditional true-residue distribution per
  profile position, including positions that are only ever deleted;
* `match_posterior[L][M]`, `delete_posterior[L][M+1]`,
  `insert_posterior[L+1][M]` — event posteriors at the pre-action coordinates,
  summed over latent residues.

Identities checked by the tests: each profile row `M + D = 1`; each query
column `M + I = 1`; residue rows sum to 1. When no complete path has positive
mass, all posteriors are `None` and both log scores are `-inf`.

Everything is accumulated with a stable log-sum-exp, so legal probabilities as
small as `1e-12` are not mistaken for "no path" by multiplicative underflow.
Note that path weights are used exactly as supplied: a weight matrix need not
be row-normalised and its total is not presented as a distribution.

`profile.gapfree` reproduces the original equal-length, no-gap calculation
(all-M path) for comparison.

## Coding semantics (`coding.py`)

Input is only an independent `[L][4]` residue-probability table and a policy.
The profile is not consulted, so probabilities need not be generatable by the
analyzer and may be far below `1e-12`.

A candidate is exactly `L` allowed codons (`3L` DNA bases). Its cost is

```
sum_i -log(residue_probabilities[i, chosen residue])
+ sum_i usage_cost[chosen codon]
```

and the minimum is taken over **all residue and synonymous-codon choices
jointly**. A greedy "pick the highest-probability residue, then fix codons"
solution is not used. Constraints:

* `forbidden`: 0–4 distinct motifs of length 2–6, matched only as given
  (no reverse complement), across codon boundaries too;
* `window_width` 3–6 with inclusive `window_gc_min..window_gc_max` on every
  complete window (no padded end windows);
* `total_gc_min..total_gc_max` on the whole DNA.

The exact solver is a dynamic program over `(codon position, trailing ≤5 DNA
bases, last width-1 GC bits, running total GC)`. The trailing DNA suffix is
what catches motifs that straddle codons; the GC bits and running total catch
rolling windows and global GC without ever re-reading the DNA. A state space
of this size is tiny for `L ≤ 10` (measured well under a second), and complete
per-layer storage gives an exact traceback. Status `"infeasible"` means the
whole state space was exhausted; it is never a heuristic giving up.

`design` returns `status`, `dna`, `translation`, `codons`, recomputable
`probability_cost` and `usage_cost`, `cost`, `total_gc`, and `window_gc`.
`coding.unconstrained` is a convenience wrapper that clears all DNA
constraints (for the ordinary coding demo).

## Combination

The demo passes the **same** `profile.analyze(...)["residue_posterior"]`
straight into `coding.design(...)`. Keeping the codon set, usage costs,
forbidden motifs and GC policy fixed, changing the observed peptide changes the
posterior and therefore the optimal DNA/translation. Re-scoring the old
("frozen") DNA under the new posterior is strictly worse, which shows the
probability table is genuinely re-optimised rather than only displayed.
Analysis never uses coding feasibility to prune the original distribution, and
when analysis returns `no_path` no probability is fabricated.

## Verification

`tests/reference.py` contains independent brute-force references: full path
and latent-residue enumeration for tiny models, and full codon-string
enumeration for short DNA. `tests/test_suite.py` compares the main solvers to
those referents (they are never used to validate one another), plus targeted
cases for:

* marginal-vs-joint MAP disagreement,
* insertion/deletion and deleted-position residue uncertainty,
* log-safe tiny mass and genuine zero mass (`no_path`),
* cross-codon forbidden motifs and the rolling-window seam,
* total-GC bounds and true infeasibility,
* input validation and input immutability.

## Trade-offs

* **Alignment:** the forward/backward DP reuses the same per-action local
  masses for partition, Viterbi, and all posteriors. Cost is
  `O(L·M·3·3·4)`; storage is the three `3 × (L+1) × (M+1)` tables. Coordinates
  are small, so clarity beats further compression.
* **Coding:** the suffix/GC/total-GC state is the price of exactness. Storing
  all layers costs a few thousand dictionary entries at the limits but makes
  traceback exact and keeps cross-codon and rolling-window constraints
  first-class, rather than pouched onto per-codon checks.
