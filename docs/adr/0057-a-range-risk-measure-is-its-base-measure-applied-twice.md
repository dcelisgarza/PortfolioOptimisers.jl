---
status: accepted
---

# A range risk measure is its base measure applied twice

## Context

A **Range** variant of a tail risk measure is the base measure evaluated on the losses at
`alpha`, plus the same measure evaluated on the gains at a second level. That is the whole of
what a range is. The value-level functors already said so:

```julia
(r::EntropicValueatRiskRange)(x) = ERM(x, r.slv, r.alpha, r.w) + ERM(-x, r.slv, r.beta, r.w)
(r::RelativisticValueatRiskRange)(x) = RRM(x, r.slv, r.alpha, r.kappa_a, r.w) +
                                       RRM(-x, r.slv, r.beta, r.kappa_b, r.w)
```

The JuMP side did not. Every single-tail `set_risk_constraints!` takes a `loss::Bool` keyword
that negates the net portfolio returns, and `GenericValueatRiskRange` was built as two calls
through that seam plus a sum — twelve lines. Seven of the nine other range members ignored the
seam and re-stated their base measure's whole variable, cone and expression block a second time
inline, with `_l` and `_h` suffixes on every name.

Those seven copies had drifted from the measure they duplicate:

- `RelativisticValueatRiskRange` built **both** power cones from `kappa_a`. It computed
    `opk_b`, `omk_b`, `ik_b`, `iopk_b` and `iomk_b` and used none of them, so `kappa_b` reached
    only the scalar coefficient of the risk expression.
- `PowerNormValueatRiskRange` built **both** power cones with `PowerCone(ipa)`, so `pb`
    reached only the scalar coefficient. Its upper-tail cone was also never registered in the
    Model State: four constraints were destructured into three names.

Both defects are invisible at the default parameters, where the two tails are symmetric, and
both make the optimiser disagree with the measure's own functor whenever they are not.

## Decision

A range risk measure **declares its two tails and builds them through the single-tail seam**.

`range_tails(r) -> (; loss, gain)` returns the two point measures a range is the sum of. Each
tail carries `RiskMeasureSettings(; rke = false)`: an upper bound and a contribution to the
objective belong to the range as a whole, which registers them once from the composite
expression.

`set_range_risk_constraints!(model, i, r, name, opt, pr, args...)` is the one builder. It reads
the two tails, calls `set_risk_constraints!` on `loss` with `loss = true` and on `gain` with
`loss = false`, registers the sum under `name` at index `i` through the Model State interface
(ADR 0037), and hands it to `set_risk_bounds_and_expression!`. Every range member's
`set_risk_constraints!` is now that one call, and `GenericValueatRiskRange` takes the same
path — its tails are *given* by the caller rather than derived, which is the only difference
between it and the rest.

The `alpha`/`beta` split is therefore a property of the measure, resolved once in
`range_tails`, and no longer a fact each builder re-encodes.

## Consequences

- Seven duplicated builder bodies become seven declarations plus one shared builder.
    `20_RiskMeasureConstraints/` loses about 630 lines.
- `RelativisticValueatRiskRange` and `PowerNormValueatRiskRange` now shape each tail with its
    own `kappa` and `p`. Results change for those two measures whenever `kappa_a != kappa_b` or
    `pa != pb`, and they change **towards** the value-level functor, which was always right.
- A base measure's fix reaches its range for free. A new range costs a `range_tails` method
    and a four-line builder.
- Model State entry names change: the tail expressions are registered under
    `nested_index(:loss_, i)` and `nested_index(:gain_, i)` rather than with `_l_` and `_h_`
    infixes, so `:cvar_risk_l_1` is now `:cvar_risk_loss_1`. The **composite** key of every
    range measure is unchanged, and no consumer inside or outside the package read the tail
    keys.
- `set_asset_neg_returns_plus_one!` loses its only caller. The DR-CVaR gain tail reaches the
    same matrix through the point builder, which negates `X` and namespaces it under a
    `gain_` prefix. The helper is kept, documented, on its API page.

## Scope

Two members are **excluded**, and they are excluded for the same reason: they *fuse* their two
tails rather than duplicating one, so there are no two sub-models to build.

- `OrderedWeightsArrayRange` under `ExactOrderedWeightsArray` collapses to a single bilinear
    constraint on `w1 - w2`.
- `ValueatRiskRange` under `DistributionValueatRisk` shares one `g_var` cone between tails.

`range_tails` is undefined for both. A measure that fuses declares no tails rather than
returning a decomposition that does not describe it.

## Notes

The MIP formulation of `ValueatRisk` had a pre-existing defect in its gain tail that this
change neither introduces nor repairs: `loss = false` negates the binary indicator vector along
with the returns, which makes the cardinality constraint vacuous and collapses the tail to the
best realisation. The hand-written `ValueatRiskRange` MIP body had the *same* degeneracy by a
different route — its upper-tail big-M term carried the wrong sign — so the two agreed
numerically, and `mr_block6` has asserted that agreement all along. Routing the range through
the seam preserves the agreement exactly. Correcting the tail itself is a separate change to
`ValueatRisk`, which would move `ValueatRiskRange` and `GenericValueatRiskRange` together.

## Related

- [0005](0005-prefix-namespaced-risk-state.md) — the prefix namespacing the tails build under.
- [0007](0007-precomputed-returns-functor-contract.md) — the functor contract the tails mirror.
- [0037](0037-model-state-accessor-interface.md) — the Model State keys the tails register.

## Amendment (2026-09-25) — `set_asset_neg_returns_plus_one!` is deleted

The consequences above keep `set_asset_neg_returns_plus_one!`, with no caller, on its API page.
The helper is now deleted (#1336), because a unit with no caller is a dead end (the ruling
on #1212). The DR-CVaR gain tail still reaches `1 - X` through `set_asset_returns_plus_one!` with
`-X` under its `gain_` prefix, so no result changes. The key `:nXap1` has no builder now.
ADR [0037](0037-model-state-accessor-interface.md) records the same deletion for four accessor
pairs.

## Amendment (2026-10-29) — an OWA range is spelled only as `OrderedWeightsArrayRange`

The verdict of #1894 removes the two fused range builders,
`OrderedWeightsArrayConditionalValueatRiskRange` and `OrderedWeightsArrayTailGiniRange` (#1896).
An OWA range is now spelled only as `OrderedWeightsArrayRange`, and the aliases `OWA_RG`,
`OWA_CVaR_RG` and `OWA_TG_RG` build it with one builder or function per tail.

The reason is the approximate formulation. Under `ApproxOrderedWeightsArray`, this ADR builds
each tail through the single-tail seam, and the two tails are a tighter relaxation than one
`OrderedWeightsArray` over the fused vector. In a minimum-risk `MeanRisk` on seeded returns
with `T = 100` and `N = 8`, the fused optimum had an exact risk 4.6 % (CVaR range), 2.1 % (tail
Gini range) and 9.6 % (`owa_rg`) above the exact optimum. The two tails reached the exact
optimum for the CVaR range and for the range, and came to 0.2 % above it for the tail Gini
range. Under `ExactOrderedWeightsArray` and `SortingNetworkOrderedWeightsArray`, the two
spellings give the same model. The fused functions `owa_rg`, `owa_cvarrg`, `owa_wcvarrg` and
`owa_tgrg` stay, and their docstrings state the gap.

`OrderedWeightsArrayRange` loses its `rev` field and keyword. `w2` defaults to `nothing`, and
only then does the constructor store the reversal of `w1`: `reverse(w1)` for a vector and
`reverse ∘ w1` for a callable. A `w2` that the caller gives is stored as given, so the caller
gives it reversed. The stored `w2` is never `nothing`, so a positional rebuild reverses nothing
a second time.

## Amendment (2026-11-14) — the one builder is a method of `set_risk_constraints!`

The decision above names `set_range_risk_constraints!(model, i, r, name, opt, pr, args...)` as
the one builder, and each range method of `set_risk_constraints!` as one call of it with a key.
The rulings of #2072 read that builder as a second name for the job of `set_risk_constraints!`:
a helper that holds a part of the job of a function for several forms of input. The ruling
"One owner, one name" gives each form to the owner as a method, and "Several methods share one
body" writes one method on the abstract type of the family (#2120).

The reason of this ADR is unchanged. A range declares its two tails once in `range_tails`, and
the single-tail seam builds them. Only the name of the builder and the place of the key change:

- `RangeRiskMeasure <: RiskMeasure` is the abstract type of the Range variants. All nine
    `…Range` types subtype it, the fused ones and `Range` too.
- `set_risk_constraints!(model, i, r::RangeRiskMeasure, opt, pr, args...)` holds the body of
    the old builder. A fused formulation keeps its own method, which is more specific.
- `range_risk_key(r)` is a public contract with one method for each formulation that
    decomposes. It gives the Model State key of the sum, by the convention
    `<base>_range_risk_`. Three keys move to that convention: `:drcvar_risk_range_` is now
    `:drcvar_range_risk_`, `:evar_risk_range_` is now `:evar_range_risk_`, and `:genvar_range_`
    is now `:genvar_range_risk_`.
- `OrderedWeightsArrayRange` under `ApproxOrderedWeightsArray` checks `w1` and `w2` under their
    names, and `GenericValueatRiskRange` of two `DistributionValueatRisk` tails builds the cone
    that the tails share. Each of the two holds a copy of the six lines of the build, and its
    `# Algorithm` states that the copy agrees with the method for a `RangeRiskMeasure`.
- `range_tails` takes its throwing fallback on `RangeRiskMeasure`. A measure that is not a range
    has no method, so a call raises a `MethodError`. A range that fuses still raises the
    `ArgumentError` that names its type.

A new range that decomposes now costs a `range_tails` method and a one-line `range_risk_key`
method, and a range of a user takes the same path. Before this amendment, a range of a user had
no builder of its own to call.

## Amendment (2026-11-25) — a range can be a tail, and a gain tail is a measure that has that side

The decision above builds `loss` with `loss = true` and `gain` with `loss = false`. Each builder
then splatted its keywords after that value, so a range that was itself a tail gave the side of
the outer range to both of its tails. The model optimised a value that was not the value of the
functor (#2328). The three builders now take `loss::Bool = true`. They build the loss tail on that
side and the gain tail on the other, and they record the owner of each frontier entry by the same
side.

A builder that takes no `loss` keyword gave the loss tail in place of the gain tail, and nothing
objected. Each such measure now has one of three answers:

- `OrderedWeightsArray` and `OrderedWeightsArrayRange` under `ExactOrderedWeightsArray` and
    `SortingNetworkOrderedWeightsArray`, and `ValueatRiskRange` under `DistributionValueatRisk`,
    build the side. They take no second block: an OWA builder negates and reverses its weight
    vector, and the parametric range negates the mean and reflects the law over the same cone.
- `Range`, the variances, `Kurtosis`, `BrownianDistanceVariance` and a full `LowOrderMoment`
    about the mean have one value on the two sides, so their builders need no keyword.
- Every other measure has no gain tail. `supports_gain_tail` is a public trait with the fallback
    `false` on `RiskMeasure`, and `RangeTails` refuses a gain tail whose answer is `false`. The
    configuration alone decides the refusal, so it is in the constructor.

`NegativeSkewness` has one value on `w` and `-w`, but its matrix reads the negative side of the
coskewness alone, so it has no gain tail. The matrix of a variance or of a `Kurtosis` is the choice
of the caller, so a lower semi-moment in that matrix keeps its side. A range that decomposes answers
for its loss tail, which moves to the negated portfolio with the range.

The trait fails closed. A measure of a user whose builder reads `loss` declares it, where no
declaration was necessary before. A default of `true` would keep the silent defect for each measure
that a user or a later ticket adds.
