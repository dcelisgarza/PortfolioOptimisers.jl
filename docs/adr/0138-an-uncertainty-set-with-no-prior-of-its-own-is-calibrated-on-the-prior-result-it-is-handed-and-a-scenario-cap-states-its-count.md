---
status: accepted
---

# An uncertainty set with no prior of its own is calibrated on the prior result it is handed, and a Scenario Cap states its count

## Context

[Map #861](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/861) lifts every layer above
the moments onto the online seam.
[ADR 0137](0137-an-optimiser-forwards-to-its-prior-alone-and-a-read-out-reconstitutes-the-carrier-and-runs-the-batch-path.md)
put the optimiser on it: the step forwards the observation to the prior alone, and `optimise(opt)`
rebuilds the carrier and runs the ordinary batch path, so every uncertainty set is fitted at the
read-out exactly as batch fits it, and no set gains a method.
[Issue #868](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/868) asked what that
costs each set, and what the identity a test pins is.

### Where a set's prior comes from

`src/14_UncertaintySets/` holds five estimator families. Four of them —
`DeltaUncertaintySet`, `NormalUncertaintySet`, `ARCHUncertaintySet` and
`CharacteristicUncertaintySet` — hold `pe::AbstractLowOrderPriorEstimator = EmpiricalPrior()` and
open every fit with `pr = prior(ue.pe, X, F)`. The optimiser's call site is the three-argument
`mu_ucs(uc, rd, pr)` (`01_Base_UncertaintySets.jl`), and for these four it **drops `pr`** and fits
`ue.pe` on `rd.X`. Only `OrthogonalUncertaintySet`, an `AbstractPriorUncertaintySetEstimator`,
reads the optimisation's own `pr`.

So the default `MeanRisk(; pe = EmpiricalPrior(), r = UncertaintySetVariance())` fits
`EmpiricalPrior` twice in batch. Under ADR 0137 the online read-out folds one and refits the other
from zero over the buffered rows at every step: `O(t · N²)` per set per step, growing with `t`,
for a set whose statistic is a function of `mu`, `sigma` and a count and therefore has an exact
recursion by composition. And by
[ADR 0050](0050-an-uncertainty-set-carries-the-quantity-it-bounds.md) the set's own centre wins,
so `DeltaUncertaintySet()` inside `MeanRisk(; pe = FactorPrior())` centres the return term on the
empirical mean. ADR 0050 named the trap: example 11 "puts the characteristic in the outer prior's
`me`. That forces the two fits to agree by hand, and it re-centres every other consumer of `pr.mu`
as a side effect." A set calibrated on the prior the optimiser is solving on is not expressible
today: the `pe` bound refuses a `TimeDependent`, an `Online` and a prior result alike.

### What a count reader sees under a Scenario Cap

Nine sites read the number of observations behind a prior result off the shape of its returns
matrix: the normal set's `choose_scaling_parameter` reads `ue.ens`, else `pr.ens`, else
`size(pr.X, 1)`; `effective_sample_size` in `06_CalibrationRules.jl` reads a Kish count from the
rule's weights, else `size(pr.X, 1)`; `RateSignificance`, `EntropyBudget`, `RateRadius` and
`compact_radius_sample_size` read `size(pr.X, 1)` bare. That was right while `X` was the whole
sample. ADR 0136's Scenario Cap, `EmpiricalPrior(; max_scenarios = w)`, fits `mu` and `sigma` over
every observation and carries the last `w` rows, so under a cap every count reader prices `w`
observations for moments fitted over `t`. After 1250 observations with `max_scenarios = 250`, the
normal set's `sigma / T` is five times the sampling variance and its ellipsoid `√5 ≈ 2.24` times
too wide. Batch and online agree on that number, so it is a defect of the cap, not of the seam,
and it is `dev`-only: `max_scenarios` has no occurrence on `main`.

`ens` is produced today by five entropy-pooling sites as `exp(entropy(w))` and read by the normal
set alone. Its meaning to its reader is "the effective count the prior carries … because a weighted
or a shrunk prior carries fewer effective observations than it has rows". The effective sample
size of `mu` and `sigma` fitted over `t` unweighted observations is `t`, whatever `X` carries
afterwards. A cap does not change the sample behind the moments; it changes the rows the result
carries.

### What the reference does

Its empirical and bootstrap sets fit their own `prior_estimator` on `X` inside their `fit`, and
have no `partial_fit`, so an online optimiser configured with one raises. Its orthogonal set reads
the optimiser's return distribution and ignores `X`. It reads the observation count off the buffer
**after** its history cap truncates it, with a static `n_eff` as the caller's override. A caller
of the reference cannot say "the set of the prior I am optimising on", and cannot recover the
folded count without typing it.

## Decision

### No set takes a step, and a set with no prior of its own is calibrated on the prior result it is handed

ADR 0137 stands: no uncertainty set gains `partial_fit!`, and a set with its own `pe` is refitted
over the buffered rows at the read-out through the batch path, bit-identical to batch because the
set's fit *is* a batch verb over the same rows. A seeded bootstrap is bit-identical too, because
`resolve_rng` reseeds a private copy on every fit; an unseeded one advances the shared task
generator per fit, in batch and online alike, and no test pins it.

The four returns-data families widen `pe` to `Option{<:AbstractLowOrderPriorEstimator}`. **`nothing`
means the set is calibrated on a prior result it is handed** — the contract
`OrthogonalUncertaintySet` already has, reached by dispatch on the field:

```julia
NormalUncertaintySet(; pe = nothing)                                  # opt-in
ucs(ue::NormalUncertaintySet{Nothing}, pr::AbstractPriorResult; kwargs...)   # the new arm, one per family
ucs(ue::NormalUncertaintySet{Nothing}, X::MatNum, F = nothing)               # refused by name
ucs(uc, rd, pr) → ucs(uc, pr; rd = rd)                                # the optimiser's call, routed as the orthogonal set's is
```

Inside an optimiser the set reads the `pr` the optimiser is solving on, so its centre and the
objective's `mu` are the same number by construction, its statistic folds exactly by composition
through the **one** prior, and the step costs it nothing — no state, no second buffer, no second
fit. Standalone it takes `ucs(ue, prior(pe, X))`, which is the optimiser's call spelled out. The
two-argument `ucs(ue, X)` refuses by name, pointing at the prior-result form and at `pe`, because
`nothing` says one thing: this set reads a prior result. It does not resolve to an `EmpiricalPrior`
over `X` at the fit — that would calibrate the same estimator on two different priors depending on
the call site, which is the ADR 0050 fault in a new coat.

**The default stays `EmpiricalPrior()`.** No released number moves. A caller who wants the set
around the prior at hand writes `pe = nothing`; the map's closing test does.

The bootstrap set under `nothing` resamples the `pr.X` it is handed and refits `me` and `ce` per
resample, so under a Scenario Cap it resamples the carried rows. That is inherent — a bootstrap can
only resample what is carried — and the docstring says so.

### A Scenario Cap states its count

`EmpiricalPrior` sets `ens` to the number of observations its moments were fitted over when
`max_scenarios` cuts, in batch and at the folded read-out alike; without a cap `ens` stays
`nothing`, which every reader takes as `size(pr.X, 1)`. `effective_sample_size(pr, w)` gains the
middle arm — a Kish count from the rule's weights, else the `ens` the result states **beside no
`w`**, else the shape — and the four bare `size(pr.X, 1)` count reads route through it. The
middle arm is narrower than "else `pr.ens`" for one reason: `ens` is bound to `w` as a diagnostic
of it, and an entropy-pooling prior writes `exp(entropy(w))` there. The three rate rules read the
raw row count by design and ignore the weights they are handed, so they take the verb with no
weights, and a reader that ignores a weighting must ignore its diagnostic too; the only `ens` a
result states with no `w` is the one a cap writes. Under entropy pooling, therefore, a rule that
reads the weights takes the Kish arm as before and a rule that reads the rows still reads the
rows, so no released number moves; the normal set already reads `pr.ens`.

The `@set` guard that binds `ens` to `w` stays right: a caller who patches a weighting onto a
capped result must restate `ens`, and a new weighting does have a new effective count.

### Calibration rules take their host's route

Every calibration rule is a callable of the prior result its host hands it, so a rule takes
whatever route its host's prior takes and holds no state. Which quantity of `pr` a rule reads —
a count, a moment, or an order statistic of `pr.X` — decides nothing about the seam.

### The identity a test pins

After `t` steps of `partial_fit!(opt, rd)`:

- a set with its own `pe` read through `optimise(opt)` equals the set batch reads through
  `optimise(opt, rd[1:t])` **bit for bit**, the seeded bootstrap included;
- a set with `pe = nothing` equals the set built from the batch `pr` to the moment layer's
  tolerance, because the optimiser's own prior is folded;
- under `max_scenarios = w`, both read `ens = t`, so the normal set's `T` and every count-priced
  rule agree with an uncapped fit over the same observations.

## Considered options

1. **Refit every set's own `pe` at the read-out, as ADR 0137 left it.** Zero code, bit-identical,
   already ahead of the reference. Rejected as the *only* route because the map's destination
   says a member with an exact recursion folds without a refit from zero, and the delta, normal
   and characteristic sets are exact by composition; it stays as the route of a set with its own
   `pe`.
2. **Forward the step to every set's `pe`.** Exact, but `N + M` carry buffers — the copies of `X`
   that ADR 0136 refused — a walk over the return and risk vectors at every step, and four types
   gaining `partial_fit!`. Rejected.
3. **`pe = nothing` reads the prior result at hand** (chosen). One route, the orthogonal set's;
   no state; exact by composition; the double fit and the split centre go for a caller who asks.
4. **`nothing` as the default.** A set would mean "around what I am optimising" unless told
   otherwise, which is the reading ADR 0050 wanted and had to fake in example 11. Rejected because
   it moves the number of a default set inside a non-empirical optimiser prior, a case ADR 0050
   ruled correct as it stands.
5. **A new `nobs` field on the carrier** for the capped count. Rejected: `ens` already means the
   effective count behind the moments to the one reader that reads it, every wrapping prior
   already forwards it, and `port_opt_view` already carries it.
6. **Leave the capped count as the rows carried, as the reference does.** Rejected: a cap would
   silently mis-price every count reader by `t / w`.

## Consequences

- Four estimator families gain an `Option` on `pe`, one prior-result arm each, and one refusal
  each on the returns-data arm, written once in `ucs_prior`. The returns-data arm of each family
  fits the set's own `pe` and hands the result to the prior-result arm of the same set with its
  `pe` set to `nothing`, so the two routes share one body per shape. Which argument the
  three-argument routing, `ucs_risk_measure` and the Pipeline's uncertainty step hand an
  estimator is decided by one per-type predicate, `reads_prior_result`, which the prior-reading
  root and the four `{Nothing}` families answer `true`. Nothing released moves, because the
  default is unchanged and `nothing` is opt-in.
- `EmpiricalPrior` writes `ens` under a cap, `effective_sample_size` reads it, and four bare
  count reads route through that verb. Every path without a cap is bit-identical; entropy pooling
  is untouched; a capped fit, which is `dev`-only, prices `t`.
- ADR 0050 is amended, not rewritten: a set with no prior of its own carries the quantity it was
  handed, so "the carried quantity wins" and "the set is calibrated on the objective's prior" are
  the same statement for it.
- The `max_scenarios` docstring and the Scenario Cap glossary entry say what a cap does state: the
  count its moments were fitted over.
- Two ESS formulas coexist for one weighting — the normal set reads `exp(entropy(w))` off `pr.ens`
  and the calibration rules read a Kish count off `w`. This ADR does not reconcile them; it is
  recorded so the next reader does not rediscover it.
