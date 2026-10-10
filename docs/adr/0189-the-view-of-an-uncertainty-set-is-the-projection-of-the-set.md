---
status: accepted
---

# The view of an uncertainty set is the projection of the set

## Context

A hierarchical optimiser restricts every Result to the assets of a cluster with
`port_opt_view(x, j)`. For an uncertainty set, two rules can answer that call:

- **The projection of the set.** The view states the worst case of a cluster portfolio `w_j` as
  the portfolio of the full universe that holds `w_j` on the cluster and nothing elsewhere.
- **A refit.** The view states the set that the same estimator would fit on the cluster alone.

For the box, the ellipsoid, the L1 set and the covariance norm ball, the two rules give the same
shape, and every view took the projection. The two sets of `OrthogonalUncertaintySet` are
different, because their geometry depends on the factor span of the whole cross-section, and
their views followed opposite rules:

- The mean `NormBallUncertaintySet` sliced the rows of its map `L` and kept its radius. That is
  the projection, and #729 stated it.
- The `CompactCovarianceUncertaintySet` sliced the rows of its basis `Q` and re-orthonormalised
  them, so its penalty became the projection onto `col(Q_j)`. That is the refit, and step 3 of
  #653 stated it: "the restricted penalty is the projection onto the sliced span, not the slice
  of the projection". #659 stated the same rule for a subspace basis.

The parity measure of map #1375 (#1390) found the difference, and #1424 records it. The
covariance view equalled the oracle's refit on a sub-universe to 7.8e-16, and the mean view
differed from the oracle's refit by 0.47.

The worst case of a padded portfolio under the fitted covariance set is
`w_j' Σ_jj w_j + κ w_j' C_j (I − Q_j Q_j') C_j w_j`: the principal block of the penalty matrix.
The refit pays `C_j (I − P_j) C_j`, with `P_j` the projector onto `col(Q_j)`. Because
`Q_j Q_j' ⪯ P_j`, the refit pays less. On the two parity panels, the refit paid 6.9 % and 14.9 %
of the block's penalty at the equal-weight portfolio of the cluster. On the directions that the
refit spares, it paid 0 %.

The refit also broke an invariant that the estimator states: both sets spare the same
portfolios, `w ∈ col(W B)`. A cluster portfolio in `col(Q_j)` paid 2e-31 on the covariance view,
0.64 on the mean view, and 0.27 on the principal block.

## Decision

**The view of a fitted uncertainty set is the projection of the set.** It is not a refit. This
holds for every set of the library, and it reverses step 3 of #653 for the compact covariance set.

The compact set gains a field `R`. A fitted set has no row in it. The view keeps the sliced rows
`Q_j` as they are, and takes `R` as the triangular factor of the `qr` of the rows it drops,
stacked on the `R` of the set:

```text
R_view = qr([Q[dropped, :]; R]).R        so   R_view' R_view = I − Q_j' Q_j
penalty = min_z ‖C w − Q z‖² + ‖R z‖²    =    w' C (I − Q Q') C w
```

The stacked matrix `[Q_j; R]` has orthonormal columns, so the inner problem gives the principal
block exactly. The JuMP model adds `R z` to the residual of its second-order cone, and
`ucs_variance` solves the same stacked least-squares problem. A view of a view composes exactly,
because the Gram matrix of the stack is the Gram matrix of every row dropped so far. A dropped
row that is zero adds nothing to the Gram matrix, so the view leaves it out, and the view at the
Investable Mask of a set that `expand_investable_ucs` wrote recovers the fitted set.

The factor comes from the `qr` of the dropped rows, not from the square root of `I − Q_j' Q_j`.
The subtraction from the identity loses digits when a factor lives inside the cluster: the error
against a `BigFloat` truth was 7.1e-15 there, against 5.5e-16 from the `qr`.

**The estimator route still refits.** An `OrthogonalUncertaintySet` estimator inside the inner
optimiser of a cluster fits on the viewed prior, so it takes the factor span of the cluster's own
loadings and sizes its radius on that span. That is what an estimator does with the data it
receives. A pair of sets fitted before states the uncertainty of the full universe, and its view
restricts that statement. The two routes give different sets on purpose, and the docstring of
`OrthogonalUncertaintySet` says which route gives which set.

## Why the projection

1. **It is exact.** The worst case of a cluster portfolio does not depend on whether the set is
   viewed first. A view that pays less than the set it came from states a smaller uncertainty
   than the Result holds.
2. **The radius stays exact.** A Result holds its radius as a number and no rule, so a view
   cannot size it again. Under a refit the radius goes stale: it was sized on the dimension of the
   full span, and the refit moves that dimension. The projection of a set keeps the radius of the
   set.
3. **The two orthogonal sets spare the same portfolios again**, the padded portfolios in
   `col(W B)`.
4. **It is the rule of every other view.** The view of a Prior Result slices `mu` and `sigma`,
   which is the projection of the moments, and the other sets already took the projection.

A refit for both sets was the other consistent option. The mean set would need a result type that
knows its loadings, and both views would still carry a stale radius, so a refit through a view is
only half a refit. A caller who wants the refit fits the estimator on the viewed prior.

## Consequences

- `CompactCovarianceUncertaintySet` has five fields: `kappa`, `C`, `Q`, `R` and `val`. The keyword
  `R` defaults to a matrix with no row, so every earlier constructor call keeps its meaning.
- `orthonormalise_basis` lost its last caller and is deleted.
- A hierarchical optimiser that receives a pre-built compact set charges a cluster portfolio more
  than before, and exactly what the full set charges it.
- `test/test_10d_parity_uncertainty_sets.jl` pins the refit route against the oracle's refit, and
  pins the view against the principal block.
- ADR 0111 and ADR 0127 carry an amendment that points here.
