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
