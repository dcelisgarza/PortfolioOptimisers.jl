---
status: accepted
---

# The Asset Panel owns its Panel Fields, and the Feature Matrix is derived from them

## Context

Two consumers read per-asset data that is not a return series.

A cross-sectional factor prior fits on a **point-in-time panel**: several numeric fields per
observation and asset (market capitalisation, book equity), a categorical classification, an
optional labelled three-dimensional field, and two universe masks. An asset that lists late has
no market capitalisation on the earlier dates, and a delisted one has none on the later dates, so
the raw panel has blank cells by construction. Its descriptors and exposures read **one field at a
time** and want that field in its own shape.

The clustering and network stack reads a **Feature Matrix**, `assets × features`, or
`observations × assets × features` when it varies in time, and measures a distance between its rows
(ADR 0045). A sector taxonomy, a fitted loadings matrix and a graph neighbourhood all take that
shape. This consumer reads **many fields at once** and wants one matrix.

[Issue #646](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/646) first decided the
carrier, and [issue #664](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/664) built
it, both under map [#643](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/643): the
panel's values rode the carrier's dense `Z`, and `AssetPanel` held only an integer field index and
the masks. [Issue #803](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/803), under
map [#802](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/802), reworked the stack
from zero and reversed that split. This ADR records the reworked decision. It is a draft on `dev`,
so it is rewritten in place rather than amended.

Three facts decided the reworked shape.

 1. **A dense matrix plus an integer index is two things that must agree.** Every check the first
    shape carried existed to keep them agreeing: unique, positive, disjoint columns; a column
    bound against the feature axis; a walk from a column back to its owner; and two naming
    conventions, `"<field>=<level>"` and `"<field>::observed"`, which the first shape rejected as
    the index and then kept to name the columns.
 2. **The reference implementation stores per-field values.** Its panel is a dictionary of fields,
    each with its own array and element type, a categorical as integer codes over labels, and two
    boolean masks. It has no verb that stacks fields into one matrix, and no distance or
    clustering consumer, so it never needed one.
 3. **The distance already allocates.** A time-varying Feature Matrix collapses along the
    observation axis before the metric runs (ADR 0045, decision 7), so the distance builds a
    small `assets × features` matrix per call in either layout. Stacking selected fields into that
    matrix costs the same allocation.

## Decision

### A Panel Field owns its values and its observed mask

Three field types, under one unexported supertype.

- `NumericPanelField(name, vals, omsk)`: one number per observation and asset. `vals` is
  `observations × assets`, or `assets` in the static shape.
- `CategoricalPanelField(name, levels, codes, omsk)`: one label per observation and asset, stored
  as an integer code over declared `levels`. A code is what a cross-sectional group label is, so
  the consumer that groups reads it as it stands, and the one-hot form is built only where a
  matrix is needed.
- `TensorPanelField(name, axis, labels, groups, vals, omsk)`: a labelled third axis, such as the
  factors of an exposure tensor, with optional groups such as Factor Families. `vals` is
  `observations × assets × labels`, or `assets × labels` in the static shape.

`omsk` is the **observed mask**: a boolean array of the same shape as the values, `true` where the
raw input carried a value and `false` where a fill policy wrote one. A field whose policy refuses
blanks carries `nothing` there. The tensor's mask keeps the per-entry resolution the raw input had.

The dense layout was rejected on fact 1. A field is now one object, and there is nothing left to
disagree. The integer index, its four checks, the two naming conventions and the column-owner walk
are all gone.

### The Asset Panel is a vector of fields and two masks

`AssetPanel(pf, amsk, emsk)`. `pf` is a non-empty vector of fields with unique names, all of one
shape class and one size. `amsk` is the active mask, `emsk` the estimation mask, each
`observations × assets`. `panel_field(pnl, name)` looks a field up by name and returns it.

A tuple of fields was rejected. The lookup dispatches once per field, which costs nothing that
matters, and a tuple would compile once per panel composition.

### One type, two shapes, and the masks say which

A static panel has no observation axis: a taxonomy, a loadings matrix, an adjacency. Its fields
are one dimension lower, and its masks are `nothing`, because "active at an observation" has no
meaning without an observation. The constructor holds the coupling: masks are `nothing` if and
only if the fields are static. The static type is `AssetPanel{PF, Nothing, Nothing}` by
parameter, so a consumer that needs masks dispatches on the mask type and never branches.

This follows ADR 0045, decision 2, which admits both shapes of a Feature Matrix with `ndims` as
the switch and rejected wrapper types. Two sibling panel types were rejected for the same reason.
A single time-varying shape with `T = 1` standing in for static was rejected because the carrier
checks the observation count against `X`, so the stand-in needs a broadcast rule, which is a
special case in a different place.

### The Feature Matrix is derived, and nothing stores it

One verb stacks the fields a selector names into the matrix a distance measures. A numeric field
gives one column, a categorical field one `0`/`1` column per level, a tensor field one column per
label, and an observed mask one `0`/`1` column. The verb's name and its selector argument are
decided by the selector ticket of map #802; the natural name is `feature_matrix`, free once the
producer ticket deletes the `FeaturePrior` verb of that name.

`nz` therefore has no home. The field names are the vector of `f.name`, and column labels are the
stacking verb's output when a consumer needs them.

### The carriers hold one field, `pnl`

`ReturnsResult` holds `nx, X, nf, F, nb, B, ts, iv, ivpa, pnl`, and `PricesResult` holds
`X, F, B, iv, ivpa, pnl`. `nz` and `Z` leave both. A time-varying panel must match `size(X, 1)`
and the asset count; a static one must match the asset count. `LowOrderPrior` loses `Z`, and
`HighOrderPrior` forwards nothing of it. The `Pr_RR` bridge stays, because `x_src` stays and the
bridge serves it.

`pnl` follows the library's abbreviated-type-word pattern (`alg`, `opt`, `sim`, `sel`). The strict
initialism `ap` was considered; `pnl` reads as *panel* and is what map #643's readers already say
at about 54 sites. `CONTEXT.md` rules out the profit-and-loss reading.

### A blank never reaches a carrier

`asset_panel(inputs; amsk, emsk)` takes the raw, blank-carrying form of each Panel Field with a
per-field fill policy, and returns the `AssetPanel` with every blank resolved. So a numeric
field's values stay finite, and no consumer sees a `NaN` or a `Missing`.

An input whose `vals` has one dimension fewer builds a static panel. There, `ForwardPanelFill` and
`BackwardPanelFill` are refused, because there is no observation axis to fill along;
`ConstantPanelFill` and `NoPanelFill` are admitted; and the mask keywords must be `nothing`.

Two routes were rejected, as before. A `NaN` reaches `FeatureDistance` and `ClusterGroups` as a
`NaN` distance. A `Missing` element type widens a library-wide numeric bound and stops the array
being concrete. This mirrors what the library already does for a missing **return**: a
preprocessing step resolves it before the returns carrier exists.

### The builder is a function, not a preprocessing estimator

A preprocessing estimator is fitted inside a fold, so it would need a carrier for the *unfilled*
panel, and the finite rule gives it none. The blank-carrying form is therefore a plain argument to
a function, and it never enters a carrier at all.

### A forward-looking fill policy is documented, not refused

`asset_panel` runs once, over the whole history, and has no fold machinery.

- `ForwardPanelFill` looks **backward** along the observation axis. A fold that starts later reads
  only rows it already holds, so it is safe.
- `BackwardPanelFill` looks **forward**. It carries a value from an observation into an earlier
  one, and a fold that ends before the source row still sees what that row supplied.

The leak is stated in `BackwardPanelFill`'s own docstring rather than refused. Map #643's
governing rule is that a decision may add capability or simplify the design and may never remove
a mode, and the reference implementation offers a backward fill. A panel built outside any fold
looks forward into nothing, so the mode is real.

### The subset invariant is checked, not coerced

The estimation mask is a subset of the active mask: an asset that is not listed at an observation
cannot enter that observation's estimate. The reference implementation coerces silently. A
coercion allocates a new mask, and `port_opt_view` must return views, so the rule is **checked**
and the caller writes `emsk .& amsk` when they want the coercion. A slice of two masks that
satisfy the rule satisfies it again, so a view never has to re-establish it and never throws.

**The per-observation non-empty checks are deliberately not ported.** The reference refuses a
panel in which some observation has no active asset. An asset view can produce exactly that, and
a view must not throw. The rule would also be a refusal rather than a capability, so dropping it
removes no mode.

### What the fold ticket owns

Both `port_opt_view` arities return views of the fields and the masks. The square case, an
adjacency whose labels are the asset names, becomes a tensor field whose label axis a view slices
when the labels equal `nx`. The meta-optimiser collapse, preselection and the assembly seam are
decided by the fold ticket of map #802. Per-field storage makes one question visible that the
dense layout hid: a convex combination of one-hot columns is a membership fraction, but a convex
combination of integer codes means nothing, so the fold ticket must say what a collapsed
categorical field is.

## Consequences

- **`ReturnsResult` loses two fields and `PricesResult` loses two.** Every printed block of a
  carrier loses its `nz` and `Z` rows, and every doctest that prints one is regenerated.
- **`asset_panel` returns one object**, not a triple, and the carriers take one keyword where they
  took three.
- **Map #643's descriptors and exposures change one seam.** `panel_field_values` reads `f.vals`
  with `NaN` where `omsk` is false; `cross_sectional_groups` returns a categorical field's codes
  and loses its `Z` argument; the one-hot exposure builds its tensor from `codes` and `levels`.
  The readers of `rd.pnl`'s masks change nothing.
- **The `field_dict` entry `:nz_feat` loses both users and is deleted.**
- **The stacking verb is the one new surface**, and the selector ticket owns its signature.
- **Panel persistence is not built.** It is in scope for map #643 and does not gate its close.
