---
status: accepted
---

# The prior `factory` resolves Deferred Quantities last, after every selection

## Context

`@propagatable`'s prior channel emits `factory(x::T, pr::AbstractPriorResult, args...)` for every
type that tags a field `@pprop` or `@cprop` (ADR 0010, ADR 0012, ADR 0061). The method does two
things: it **selects** each tagged field against a source — a moment of the prior result for
`@pprop`, the threaded optimiser solver for `@cprop` — and it **resolves** the Deferred Quantities
that ADR 0051 put in the slots.

It used to do them in that order reversed. The body bound

```julia
xr = resolve_deferred_quantities(x, pr)
```

first, and every selection then read its field off `xr`. **No ADR wrote that order down.** ADR 0051
owns the Deferred-Quantity channel and names its three resolution points; ADR 0061 owns the tag
table. Neither says which of the two steps runs first, because until the calibration slot nothing
depended on it.

The calibration slot made it load-bearing.
[#581](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/581) asked for a Calibration Rule
that may call `ERM` or `RRM`, and such a rule needs a solver. A solver reaches a risk measure through
the `@cprop slv` selection, so under the old order the rule ran **before** the value it needs
existed: a measure that states no solver of its own — the common case, because the optimiser's
solver is what it is meant to use — would hand its rule `nothing`.

## Decision

**The generated prior `factory` runs every selection first, off `x`, and hands the selected struct to
`resolve_deferred_quantities`.** The resolution runs last.

```julia
factory(x::T, pr::AbstractPriorResult, args...) =
    resolve_deferred_quantities(T(; f = sel(getfield(x, :f), …), …), pr, …)
```

Two reasons, both of them about what a slot sees when it resolves:

1. **The solver must be settled before the resolution.** It is the one input a rule cannot derive
   from the prior result, and the selection is what settles it.
2. **A deferred slot should see everything else already settled** — the solver, the observation
   weights and the composed children in the state the optimisation put them in. Under the old order
   a fit ran against a struct that was still half the caller's and half nobody's.

### `sel` keeps a stated method

Because the selection now runs first, a `@pprop` slot that admits a Deferred Quantity reaches `sel`
still holding the estimator, where the prior's same-named field would overwrite it. Two arms refuse
that:

```julia
sel(risk_variable::DeferredQuantity, ::Any) = risk_variable
sel(risk_variable::AbstractCalibrationEstimator, ::Any) = risk_variable
```

**A slot the caller filled with the method that computes the value is a stated slot.** The caller
named the method, and the resolution that follows replaces it with the value that method produced,
so the prior must not fill it in between. `ThirdCentralMoment.mu` is the only field in the library
that is both `@pprop` and deferrable, and it is where the first arm bites.

### The solver is threaded, not selected, on the other route

`resolve_deferred_quantities(x, pr, slv = nothing)` and
`resolve_calibration_slot(slot, key, pr, w, slv = nothing)` both carry the effective solver. The
`JuMP` model builders never call `factory`, so no selection runs there: `set_risk_constraints!`
threads `opt.opt.slv` into the resolution and the owning measure settles it as `sel(x.slv, slv)`
([#591](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/591)). That is how the two
routes are made to agree about which solver a rule sees.

## Rejected alternatives

**Keep the old order, and let a rule read the measure's own stated solver.** No change to the macro.
Rejected because a measure that states no solver is the common case, and its rule would see `nothing`
while the optimisation around it holds one. The rule would then work only for the caller who
duplicated the solver onto every measure.

**Resolve twice, once before the selection and once after.** Rejected because a resolution fits
estimators. ADR 0051 records that each of the three resolution points resolves once and not once per
evaluation, for the same reason.

**A new field tag for a slot the prior must not fill.** Rejected because `sel` dispatches on what the
slot holds, which is the same information the tag would carry, and the tag table stays as ADR 0061
left it.

## Consequences

- **The order is now written down, and it is part of the contract.** A channel added later that also
  rewrites fields must say where it sits relative to the resolution.
- **A rule may call `ERM` or `RRM`.** Map
  [#580](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/580) listed "a rule that needs
  a solver" as out of scope, and this lifts it. ADR 0095 records the calibration channel itself.
- **ADR 0051's three resolution points are unchanged.** Only what happens inside the `factory` point
  moved, and `resolve_deferred_quantities` gained a third argument. ADR 0051 carries the amendment.
- **What a type's own resolution method reads has changed under it.** It now sees selected fields, so
  a method that wants the caller's stated value rather than the settled one no longer has it. No
  shipped method wants that, and the calibration resolutions read the settled value by design.
