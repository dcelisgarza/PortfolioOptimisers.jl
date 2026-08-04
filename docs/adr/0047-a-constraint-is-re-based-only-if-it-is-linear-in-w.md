---
status: accepted
---

# A constraint is re-based only if it is linear in `w`

## Context

Every constraint in the library resolves against the asset universe. A practitioner's mandate rarely
does: "at most 30% momentum exposure", "market-neutral to value", "at least 20% in the quality and
low-volatility factors combined". Those are constraints on the *factor* weights

```math
\boldsymbol{w}_f = \mathbf{M}^\intercal \boldsymbol{w}_a
```

where `M` is the loadings matrix a factor model already computes.

Today the only way to express one is to fit the regression by hand, compute `M` outside the library,
and write asset-level rows against it. Under cross-validation that is not merely tedious, it is
wrong: the loadings are refit per fold, and rows computed once against a full-sample `M` describe a
factor exposure the fold's model does not have.

The library also already contained an unresolved ambiguity about what "my factors" means.
[`FactorBlackLittermanPrior`](../../src/13_Prior/08_FactorBlackLittermanPrior.jl),
[`BayesianBlackLittermanPrior`](../../src/13_Prior/07_BayesianBlackLittermanPrior.jl),
[`AugmentedBlackLittermanPrior`](../../src/13_Prior/09_AugmentedBlackLittermanPrior.jl) and
[`FactorRiskBudgeting`](../../src/20_Optimisation/14_RiskBudgeting.jl) each take an `AssetSets` whose
*asset* key holds factor names, validated against `size(F, 2)`. The type says assets; the contents
are factors; only the length check knows the difference.

That ambiguity carries a latent defect. `port_opt_view` slices key-prefixed entries by asset index,
so a factor-flavoured sets in a `@vprop` field would be sliced along the wrong axis.
`AugmentedBlackLittermanPrior` escapes it only because `f_sets` was left un-`@vprop`'d **by hand**,
next to an `a_sets` that is annotated — a correctness property held in place by an annotation nobody
is obliged to get right.

## Decision

**A constraint can be re-based into another basis if and only if it is a linear form in `w`, and the
re-basis is declared by a wrapper type rather than by a field, so that constraints which are not
linear forms cannot represent one.**

Four consequences follow, and each was chosen against a real alternative.

### The factor axis is declared, not inferred

`AssetSets` becomes `UniverseSets` and gains `fkey`/`ufkey` alongside the renamed `xkey`/`uxkey`
(defaults `"nf"`, `"uf"`, `"nx"`, `"ux"` — the field letter now matches the default's letter, which
is why `xkey` was preferred to `akey`). The factor axis is optional; requiring it would invalidate
every existing sets object.

`rd.nf` was the alternative, and it has one genuine advantage: it is the *only* thing guaranteed
column-aligned with `M`. But it is `Option`, it carries no groups, and it cannot be authored by a
user who wants a factor taxonomy. It is kept as a **cross-check** instead —
`processed_jump_optimiser_attributes` throws when `sets.dict[fkey] != rd.nf`, and, since the same
gap exists on the other axis and has always existed, when `sets.dict[xkey] != rd.nx`. The redundancy
is the point: it converts the one silent-wrong-answer failure mode into an error and costs nothing
when the user copies the names across.

A separate `FactorSets` type was rejected: it duplicates the group-expansion machinery, forces every
generic constraint function to widen to an `AbstractSets`, and needs an `fsets` field on
`JuMPOptimiser` — three costs to avoid two fields.

Declaring the axis also makes the view exemption a property of the *data* rather than of each
field's annotation, which is what closes the latent defect above.

### The rule has teeth because of what does *not* get the field

`LinearConstraintEstimator` feeds three optimiser slots: `lcse`, `gcarde` and `sgcarde`. The latter
two build rows over the binary *held indicators*, not over `w`. A projected row `M'a` is neither
integral nor an index into those indicators, so a factor-space cardinality constraint is not a
feature the implementation lacks — it is a category error. The same holds for `ThresholdEstimator`
(MIP buy-in over the same indicators) and for `WeightBoundsEstimator`, whose per-asset box has no
factor counterpart because `lb <= M'w <= ub` *is* a linear constraint and already has a home.

So `LinearConstraintEstimator` is left exactly as it was — `val`, `key`, and no space field. Being
unmarked is what keeps it the only thing `gcarde`/`sgcarde` admit, and therefore what makes the
category error **unrepresentable** rather than validated. Putting a `space` field on it and
rejecting the illegal combinations in the `JuMPOptimiser` constructor was the alternative; it buys
the same safety at the cost of a check that must be kept in sync with three field bounds, across
scalar, vector and `TimeDependent` shapes.

The re-basis lives on a new decorator:

```julia
abstract type AbstractConstraintSpace <: AbstractAlgorithm end
struct FactorSpace <: AbstractConstraintSpace end

ExposureConstraintEstimator(; lce::LcE_Lc_VecLcE_Lc,
                              space::AbstractConstraintSpace)   # required, no default
```

It wraps rather than reimplements, so parsing, `val`/`key` validation and group expansion are
inherited. Its bound is *exactly* what `lcse` accepts — an estimator, a vector of them, or a
precomputed `LinearConstraint` — so no shape reaches `lcse` unre-based.

There is deliberately **no `AssetSpace`**. The asset frame is the *absence* of a re-basis, spelled
`LinearConstraintEstimator`; an `AssetSpace` member would make the decorator a no-op computing
bit-for-bit what it wraps, which is the same redundancy in another shape. The abstract type survives
with one member because the family is open: any linear change of basis in `w` — currency, sector —
lands here without a new struct.

### The loadings are `M`, and the projection happens at generation time

`rr.M` is the loadings over the *named original* factors. `rr.L` is the same information in the
reduced basis: under
[`DimensionReductionRegression`](../../src/08_Moments/23_DimensionReductionRegression.jl) the two are
the two sides of one projection, `M = Vp * β_pc ⊘ σ` and `L = pinv(Vp) * (M ⊙ σ)`, so each is
recoverable from the other.

Which one a consumer wants is therefore not a matter of taste but of what it is doing.
[`FactorRiskContribution`](../../src/20_Optimisation/12_FactorRiskContribution.jl),
[`FactorRiskBudgeting`](../../src/20_Optimisation/14_RiskBudgeting.jl) and
[`RegressionFeatures`](../../src/13_Prior/13_FeaturePrior.jl) want **`L`**, because risk must be
attributed in the basis its covariance was actually estimated in — the orthogonal reduced one. A
constraint wants **`M`**, because a constraint is *written*, and only `M`'s columns carry names a
user can put in an equation; `L`'s are principal components. The divergence is correct in both
directions and is a consequence of the same projection, not a drift between two conventions.

The projection is applied while the row is being assembled:

```julia
At += vec(sum(view(M, :, Ai), dims = 2)) * c        # estimator path
A * transpose(M)                                     # precomputed-constraint path
```

so what leaves constraint generation is an **ordinary asset-space `LinearConstraint`**. It flows into
the existing `lcsr` slot and the existing `set_linear_weight_constraints!`. There is no `flcse`, no
`flcsr`, no entry in `jump_optimiser_from_attributes`'s rename table, and no second constraint
pathway — which means Near Optimal Centering, time-dependent schedules, and every JuMP optimiser
sharing `JuMPOptimiser` get factor exposure constraints without knowing they exist.

Carrying factor-length rows into the model and registering `w_f = M' * w` as a JuMP expression was
the alternative. It is numerically identical. It costs new fields on two structs and a second
pathway, in exchange for a named model expression that nothing currently asks for.

### Missing loadings throw, regardless of `strict`

`strict` governs *unknown names*: a per-row, recoverable condition where the offending row is dropped
with a warning and the rest of the problem is still the problem the caller described. A missing
regression is not that. It makes **every** factor row unbuildable, and dropping them silently
produces a feasible, plausible-looking portfolio carrying none of the requested exposure — the same
failure class ADR 0046 was written about. So it throws either way, reusing
[`prior_regression_remedy`](../../src/13_Prior/01_Base_Prior.jl) so the diagnosis and its remedy
match the rest of the library.

## Consequences

**The four ambiguous consumers do not migrate here.** `FactorBlackLittermanPrior`,
`BayesianBlackLittermanPrior`, `AugmentedBlackLittermanPrior` and `FactorRiskBudgeting` keep reading
their factor names from the asset key until each is migrated on its own. Until then the library has
two ways to say "these are my factors", which this ADR only *ends* once those land. Each needs its
own proof rather than a sweep, and `FactorRiskBudgeting` shows why: it is correctly **not** `@vprop`
today *because* it holds only factors, and must **become** `@vprop` once it can carry both axes.
A field gaining view participation as a consequence of gaining a second axis is not something a
find-and-replace would notice.

**The asset-axis order check is new behaviour on a path that has nothing to do with factors.** It may
surface existing user misconfigurations as errors where there were previously silently wrong
constraints. That is intended. `_update_asset_sets` already reconciles the two for the Nested
Clustered synthetic universe, so that path is unaffected.

**A pipeline step pins its projection.** `ExposureConstraintEstimator` is a valid bare pipeline step
reading the `:prior` slot, and its result routes to `:lcse` by existing result-type routing. But the
rows it produces were computed against the loadings the *pipeline's* prior carried; a downstream
optimiser that refits its own prior will not re-project them. The optimiser-field route stays the
default advice, because there the projection is recomputed per fold with the prior actually in use.

**Factor group expansion needs no code.** `replace_group_by_assets` expands any name found in `dict`
and is axis-blind, so a factor group works unchanged. The same blindness means a factor constraint
naming an *asset* group degrades to unknown-variable warnings rather than to an error, which is
detectable but not prevented.

**`ufkey` earns its place on symmetry alone.** On the asset axis the prefix conventions serve views:
`xkey`-prefixed groups are sliced, `uxkey`-prefixed ones recomputed from them. Factors are never
sliced, so on the factor side the identical rules buy only length validation and one shared mental
model. That was judged worth the validation code the view side will never exercise.

## Amendment (2026-08-03): what the implementation found

Five statements above were written from the design and turned out to be too strong, or to have
missed a case. Each is corrected here; the decision itself stands.

### Near Optimal Centering does *not* compose for free by default

"Near Optimal Centering, time-dependent schedules, and every JuMP optimiser sharing
`JuMPOptimiser` get factor exposure constraints without knowing they exist" is true of the
projection — nothing downstream needs a factor concept — but it overstated what NOC does with a
linear constraint.
[`UnconstrainedNearOptimalCentering`](../../src/20_Optimisation/13_NearOptimalCentering.jl), which
is the **default**, builds its centering model with weight bounds, budget, risk and return only. It
never calls `set_linear_weight_constraints!`, so `lcsr` is dropped from the model whose solution is
returned. This is by design, is long-standing, and applies to an asset-space linear constraint
exactly as much as to a re-based one — the re-basis changed nothing here. But it means a factor
mandate written under a default `NearOptimalCentering` does not hold in the reported weights, so
the documentation shows `ConstrainedNearOptimalCentering` wherever NOC and a mandate appear
together.

### The axis-order check runs only where both sides exist

"`processed_jump_optimiser_attributes` throws when `sets.dict[fkey] != rd.nf`" describes the check
as unconditional. As written that throws a `KeyError` on the common case: every factor-prior run
that declares no factor axis has an `rd.nf` and no `sets.dict[fkey]`, and the factor axis is
*optional* by the decision two paragraphs above it.
[`assert_universe_axis_order`](../../src/20_Optimisation/10_JuMPOptimiser.jl) therefore skips an
axis when **either** side is absent, and checks the pair only when both are declared. It also runs
*before* the prior is fitted, so a misaligned universe is reported without paying for a regression
first.

### A re-basis creates a third diagnosis

The design recognised two failure modes for a row — an unknown name, and an empty row — and both
already had messages. Projection adds a third that neither can express: **every name resolved, and
the loadings annihilated the row** (a factor no asset loads on, or a long/short combination whose
loadings cancel). The all-zero coefficient check cannot tell this apart from "no name matched",
because both leave a zero row. Assembly therefore carries a `matched` flag, and
`empty_projected_row_msg` reports the case with the remedy it actually needs — inspect the
loadings, not the spelling.

### The `lcse` type bound was not the whole enforcement

"`ExposureConstraintEstimator`'s bound is exactly what `lcse` accepts" is a statement about types,
and it holds. It does not follow that widening `lcse` was enough. The `sets`-required guard in
`JuMPOptimiser`'s inner constructor is a **hand-written `isa` chain**, not derived from the field
bounds, so widening the bound alone left `JuMPOptimiser(; lcse = ece)` with no `sets` constructing
happily and failing later. The decorator is now named there **unconditionally** — including when it
wraps a *precomputed* `LinearConstraint`, which needs no names but still needs `sets`, because
`constraint_space_basis` reads `sets.fkey` before it looks at what is wrapped.

### The pipeline step's axis comes from the returns, by construction

The consequence "a pipeline step pins its projection" stands, and is documented on the estimator
and on `run_step`. Two details were missing. First, the step was **unreachable as specified**:
`pipeline_asset_sets` declared only the asset axis, so every `FactorSpace` row would have died on
the missing-`fkey` error. It now declares every axis the returns carry, taking the factor names
from `rd.nf` — which makes the axis and the loadings agree *by construction*, and makes
`FactorSpace`'s missing-axis error unreachable from a pipeline. Second, the step declares
`pipe_reads = (:returns, :prior)`, and `Pipeline`'s constructor checks declared reads, so a
pipeline carrying the step but no prior step is rejected at **construction** rather than at run
time.

## Amendment (2026-08-04): the declared factor axis names `M`'s columns, and a factor risk budget indexes `L`'s

"`FactorRiskBudgeting` wants `L`" and "`FactorRiskBudgeting` migrates onto the declared factor
axis" were both written above, one paragraph apart, and they are in tension the ADR did not notice.
The declared axis names the **original** factors — the columns of `F`, and so of `M`. A factor risk
budget is a vector over `w1`, the factor weights, whose length is `size(rr.L, 2)`. Under
`StepwiseRegression` `L` is unset and falls back to `M`, the two agree, and nothing is visible.
Under `DimensionReductionRegression` `L`'s columns are the retained principal components: fewer
than the named factors, and carrying no names at all. There is nothing to name them *with*, so a
**named** factor budget is well defined only when the reduced basis coincides with the declared
one.

This is not a new restriction — the pre-migration shape had the same arithmetic and simply failed
further down, at the bare `length(rb) == N` check, with a message about two numbers. What the
migration changes is where it is diagnosed:
[`risk_budget_universe_key`](../../src/20_Optimisation/14_RiskBudgeting.jl) checks the declared axis
against `size(rr.L, 2)` through the shared `factor_universe`, so the message names `rr.L`, the axis
key, and the two lengths. Reading the axis off `M` instead would accept a budget of the wrong length
and mis-attribute it silently, which is the failure this ADR exists to prevent one level up.

The corollary is that the axis a named budget resolves against belongs to the **algorithm**, not to
the sets: `AssetRiskBudgeting` and `FactorRiskBudgeting` share one `risk_budget_constraints`, one
`_set_risk_budgeting_constraints!` and one `UniverseSets`, and differ only in the key they pass.
The asset path passes `nothing` and is bit-identical to what it was.

The consequence recorded above — that `FactorRiskBudgeting.sets` must **become** `@vprop` once it
carries both axes — held exactly as written, and is now the case. `rkb` deliberately does not: an
asset index has no meaning on a factor budget, and unlike `AssetRiskBudgeting.rkb` it must not be
sliced.
