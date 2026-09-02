---
status: accepted
---

# The Asset Panel rides the feature matrix, and carries only its structure

## Context

A cross-sectional factor prior fits on a **point-in-time panel**: several numeric fields per
observation and asset (returns, market capitalisation, book equity), a categorical classification,
an optional labelled three-dimensional field, and two universe masks. An asset that lists late has
no market capitalisation on the earlier dates, and a delisted one has none on the later dates, so
the raw panel has blank cells by construction.

[Issue #646](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/646) decided what carries
it, and [issue #664](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/664) built it.
Both sit under map [#643](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/643).

Three facts about the library decided the shape.

 1. **`ReturnsResult.Z` is already the panel's shape.** It admits `observations × assets ×
    features`, it is canonically assets-major, `nz` names the feature axis, and
    `feature_matrix_view` returns views. `port_opt_view` slices it in step with `X` on both the
    asset arity and the observations-then-assets arity that cross-validation uses, and
    `prices_to_returns` already takes it and already slices it when it drops rows or columns.
 2. **283 sites in `src/`, `ext/` and `test/` name the concrete `ReturnsResult`**, across 37 files.
    They include `prior(pe, rd::ReturnsResult)`, all of `20_Optimisation/02_CrossValidation/`,
    `24_Plotting.jl` and the meta-optimisers. A sibling type under `AbstractReturnsResult` is
    refused by every one of them.
 3. **A boolean is numeric and finite**, so a mask is one feature slice of `Z`, sliced correctly on
    both axes for free.

## Decision

### The values ride `nz` and `Z`; the structure rides one new field

`ReturnsResult` gains exactly **one** field, `pnl`, holding an `AssetPanel`. The panel's numbers
stay in `rd.Z` and its column names in `rd.nz`.

`AssetPanel` holds two things.

- **The field index**, a vector of `PanelField`. Each names one Panel Field, its kind, its value
  columns of `nz`, and its observed-mask columns.
- **The two masks**, each `observations × assets`: the active mask says the asset is listed at that
  observation, and the estimation mask says it enters that observation's cross-sectional estimate.

A dedicated container beside `ReturnsResult` was rejected on fact 2. Two separate slots — an index
and a mask pair — were rejected because one panel would then be described by two things that must
agree.

### A blank never reaches a carrier

`asset_panel` takes the raw, blank-carrying form of each Panel Field with a per-field fill policy,
and returns `nz`, `Z` and `pnl` with every blank resolved. So `Z` keeps its `assert_all_finite`
guarantee and no existing consumer of a feature matrix loses one.

Each Panel Field whose policy can fill contributes an **observed-mask** column beside the field it
belongs to, recording which cells the fill touched. A field whose policy is `NoPanelFill` contributes
none, because a blank there is an error rather than a fact.

Two routes were rejected. A `NaN` in `Z` reaches `FeatureDistance` and `ClusterGroups` as a `NaN`
distance. A `Missing` element type widens a library-wide numeric bound and stops the array being
concrete.

This mirrors what the library already does for a missing **return**: a preprocessing step resolves
it before the returns carrier exists.

### The builder is a function, not a preprocessing estimator

A preprocessing estimator is fitted inside a fold, so it would need a carrier for the *unfilled*
panel, and the all-finite rule on `Z` gives it none. The blank-carrying form is therefore a plain
argument to a function, and it never enters a carrier at all.

### A forward-looking fill policy is documented, not refused

`asset_panel` runs once, over the whole history, and has no fold machinery.

- `ForwardPanelFill` looks **backward** along the observation axis. A fold that starts later reads
  only rows it already holds, so it is safe.
- `BackwardPanelFill` looks **forward**. It carries a value from an observation into an earlier one,
  and a fold that ends before the source row still sees what that row supplied.

The leak is stated in `BackwardPanelFill`'s own docstring, where the policy is named, rather than
refused. Map #643's governing rule is that a decision may add capability or simplify the design and
may never remove a mode, and the reference implementation offers a backward fill. A panel built
outside any fold looks forward into nothing, so the mode is real.

### A categorical Panel Field is one-hot, and the index is not a naming convention

A categorical Panel Field takes one column per level, named `"<field>=<level>"` — the convention
`asset_sets_features` already writes a taxonomy under. No side table of integer codes is needed,
because a one-hot column is numeric and finite.

An observed-mask column takes the value column's name with `"::observed"` appended, or
`"<field>::observed"` when the Panel Field has a single observable.

**A bare naming convention on `nz` was rejected as the index**: it has nowhere to hold the mask
invariant, and a field whose own name carries the convention's punctuation would collide silently.
The `PanelField` index is the cure, because a consumer looks a Panel Field up and reads its columns
as **integers**. Nothing parses a column name. The conventions still have to produce a unique `nz`,
so `asset_panel` checks the whole feature axis and refuses a collision naming both Panel Fields.

### The subset invariant is checked, not coerced

`AssetPanel`'s constructor holds one invariant: the estimation mask is a subset of the active mask.
An asset that is not listed at an observation cannot enter that observation's estimate.

The reference implementation coerces silently, with `estimation_mask &= active_mask`. A coercion
allocates a new mask, and `port_opt_view` must return views, so the rule is **checked** and the
caller writes `emsk .& amsk` when they want the coercion. Nothing is lost: a slice of two masks that
satisfy the rule satisfies it again, so a view never has to re-establish it and never throws.

**The per-observation non-empty checks are deliberately not ported.** The reference refuses a panel
in which some observation has no active asset. An asset view can produce exactly that — a cluster of
assets none of which was listed at the first observation — and a view must not throw. The rule would
also be a refusal rather than a capability, so dropping it removes no mode.

### Both `port_opt_view` arities slice the masks, and the field index is untouched

- `port_opt_view(rd, i)` selects assets, so it slices the mask columns.
- `port_opt_view(rd, i, j, k)` selects observations then assets, so it slices both mask axes.
- Neither copies.

The field index addresses the **feature** axis, which no asset view and no observation view reaches:
`feature_matrix_view` slices the feature axis only in the square case, `nz == nx`, and a panel's
column names are not asset names.

### `LowOrderPrior` is unchanged

The panel is *carried input*, and no carried input travels onto a prior result: the result holds no
`nx`, `ts`, `F`, `iv` or `nz`. Every prior-driven consumer in `src/` is already passed `rd`
explicitly, and cross-validation slices `rd` into `rdi` and stores it on `PredictionResult` and
`TimeDependentContext`, so each fold carries that fold's panel.

## Consequences

- **`ReturnsResult` grows one field.** Every printed block of it gains a `pnl` line, and the nine
  doctests that print one were regenerated.
- **`prices_to_returns` gains one keyword and one slice line**, beside the two that already slice
  `Z` by `feature_row_indices` and `acols`.
- **The `Pipeline` Data Slot needs nothing new.** `PipelineContext.returns` is typed
  `Option{<:AbstractReturnsResult}`, and the panel rides `ReturnsResult`.
- **The feature stack must learn name-based selection.** `FeatureDistance` swallows the whole
  feature axis and carries no column selector, so a panel-carrying `Z` presents the observed masks
  and the one-hot levels as clustering features. [Issue #665](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/665)
  owns the fix, which serves every `Z` consumer and not only the panel.
- **Panel persistence is not built.** It is in scope for map #643 and does not gate its close.
