---
status: accepted
---

# A door mints a Non-Investable Axis on the asset sets, and every view drops it

## Context

[ADR 0115](0115-every-optimisation-estimator-reduces-once-at-its-entry-and-its-result-carries-the-investable-mask.md)
reduces every optimisation to the Investable Mask at its entry: a `port_opt_view` of the optimiser
slices every constraint the caller stated by one asset index, and the result carries the reduced
objects beside the mask. `UniverseSets` is one of the things that view slices, so by the time the
shared JuMP prelude resolves a name-keyed estimator, `opt.sets` names the investable assets alone.

Ticket [#911](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/911) found what that
costs. A caller states a constraint over the universe they were handed. The data then delists an
asset, the prior cannot estimate it, and the mask drops it. The name the caller wrote is now absent
from `sets`, so the term is dropped with a warning under `strict = false` and **refused with an
`ArgumentError` under `strict = true`** — on a name that was correct when it was written. `strict`
exists to catch a typo, and a caller cannot foresee which asset a prior will fail to estimate, so
the refusal reports something no one can act on.

This is the resolution order, not a property of any one estimator, so it reaches
`weight_bounds_constraints`, the six `threshold_constraints`, `linear_constraints` for `lcse`,
`gcarde` and `sgcarde`, `asset_sets_matrix` and `turnover_constraints` alike.

The fee half of the same ticket was answered first, and differently: `fees_constraints` was
**hoisted above the door** so it resolves on the caller's own `sets`, with `investable_fees_view`
placing the resolved fee on the two axes the mask leaves. That works for fees because a resolved
`Fees` can be sliced onto both axes afterwards. It does not generalise. `port_opt_view` of a
precomputed `LinearConstraint` is deliberately the identity — position is the only link between a
column of `A` and an asset, so no column can be dropped without changing what `Ax ≤ B` means — so
hoisting the linear family would produce a full-width row over a reduced weight vector.

## Decision

### `UniverseSets` gains a seventh key prefix, `nikey`, naming the Non-Investable Axis

`dict[nikey]` holds the names the Investable Mask left out, in the order the complement of the mask
visits them. It is the fourth axis the type declares, beside the asset axis and the two factor
axes, and it is validated as they are: no two of the seven prefixes may be a prefix of one another,
which is 42 ordered checks rather than 30.

Two rules are its own. Its entries must be **unique**, because a departure happens once. And they
must be **disjoint from `dict[xkey]`**, because an asset is investable or it is not, and a name on
both axes would be priced twice — once as a holding, once as a forced exit.

The axis is **bare**: no `nikey`-prefixed partition and no unique-entry twin. Its entries are
unique by construction, so a unique-entry group over it would summarise nothing, and a plain group
is axis-blind already — `resolve_axis_name` expands it to names and `axis_name_indices` keeps
whichever land on the axis being resolved, so a group resolves on the Non-Investable Axis with no
machinery of its own.

### Only a door mints the axis, and every view drops it

`non_investable_sets` is the only verb that declares it. A door — `investable_reduction` for a head
that fits a prior, `coverage_reduction` for one that does not — reads the departed names off the
*unreduced* `rd.nx`, takes the view, and declares the axis on what comes back. It declares it
**after** the view because `port_opt_view(::UniverseSets, i)` **drops** it.

The drop is the load-bearing half. `port_opt_view` of a `UniverseSets` is called for a cluster of a
`NestedClustered`, a candidate of a `Stacking` and a subset of a `SubsetResampling`, none of which
is a departure. Carrying the axis into them would let a sub-problem inherit its parent's departures
and charge every one of them again, once per cluster. So the invariant is stated as an invariant: a
`UniverseSets` that carries a Non-Investable Axis was reduced by exactly one door, for exactly one
problem. The key is matched **exactly** rather than by prefix, so a caller's plain group whose name
merely starts with `ni` — `"nikkei225"` under the default — is not dropped with it.

A caller may still declare the axis by hand. Outside a door that is the only way to resolve a
forced-liquidation rate with no optimisation around it. Inside one the mask is the truth, so a
hand-authored entry is **overwritten**: the two can only disagree, and the mask is the one derived
from the data.

The per-type verb that writes it onto an estimator, `non_investable_universe`, has a generic method
that returns its argument untouched — the right answer for an estimator that carries no sets — and
one line per head that owns one, forwarding through the nested optimiser where the sets lives on
`opt`. It is written per type rather than derived by reflection, for the reason `port_opt_view` is:
a field's meaning is the type's to state.

### A name on the counterpart axis is skipped in silence

`name_to_val!` takes the **counterpart axis**: resolving on the asset universe it is the
Non-Investable Axis, and resolving on the Non-Investable Axis it is the asset universe. A name
found there is skipped, under `strict` and without `strict` alike, and a group's departed members
are struck from the missing-member report so that a group whose losses are all accounted for is
silent while one holding a genuine typo still names it.

That is the whole of what `strict` gives up, and it gives up nothing it was for: a name on neither
axis is still refused, with the same message and the same suggestion pool.

### The departure is announced once, by the door

`announce_non_investable` emits one `@info` naming the assets that left. It is not a warning and
not a `strict_diagnostic`: nothing is wrong, the data moved, and the optimisation is proceeding
correctly over what is left. Making it raise under `strict` would put back the refusal this path
exists to remove. It is said at the door rather than per constraint because a departure is one
event, and the door is the only place that knows it as an event rather than as a shape.

### A precomputed linear constraint is refused by name

`assert_investable_constraint_width` refuses a precomputed `LinearConstraint` whose `A` is wider
than the investable universe, and says why and what to do: state it as a `LinearConstraintEstimator`
so it is re-resolved by name. The alternative is a bare `DimensionMismatch` between two numbers
deep inside the model, with nothing connecting either to the asset that delisted. This is a
diagnostic, not a repair — the row genuinely cannot follow the door.

## Consequences

- A constraint, bound, threshold or rate stated for an asset that later delists no longer refuses
  under `strict = true`, in any family, at any door.
- Every optimisation over a departure logs one `@info`. A long walk-forward over a delisting logs
  it once per window, which is silenced with a logger.
- `UniverseSets` prints an eighth field, so every doctest that shows one was regenerated.
- A precomputed `LinearConstraint` over a universe that later loses an asset now fails at the seam
  with a message naming the repair, where it previously failed inside the model.

## Alternatives refused

- **Hoist every builder above the door**, as the fee half was hoisted. Refused because the linear
  family cannot follow: a precomputed row is bound to its columns by position, so a hoisted
  full-width `A` would meet a reduced weight vector. It would also reorder the shared prelude and
  the same pattern at the other thirteen fit sites.
- **Mint the axis inside `port_opt_view` from the width of `X`**, mirroring how `Fees` derives its
  complement. Refused because the `(i, X)` arity is not a door: `SubsetResampling` and
  `MultipleRandomised` pass `X` for a subset and a cross-validation fold, and both would mint a
  false axis naming a sibling cluster.
- **Carry the axis through views**, and strip it per cluster by hand. Refused because it inverts
  the default: every future nested head would have to remember, and forgetting double-charges in
  silence.
- **A standalone field beside `dict`** rather than a seventh prefix. Refused because it puts a
  second asset axis outside the grammar that validates the first, and no group could resolve on it.
- **Full symmetry with the asset axis** — `ni`-prefixed partitions and a `unikey` twin. Refused as
  machinery no consumer asks for, over an axis whose entries are unique by construction.
- **Warn per dropped name**, or downgrade the refusal to a warning under `strict`. Refused because
  a delisting is routine and unactionable: a walk-forward would warn several times per window, for
  an event the door already announced once.
