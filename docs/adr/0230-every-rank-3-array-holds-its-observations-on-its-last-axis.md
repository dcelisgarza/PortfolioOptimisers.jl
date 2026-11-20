---
status: accepted
---

# Every rank-3 array holds its observations on its last axis

## Context

Julia stores an array in column-major order: the first index moves fastest. A slice of an array
is one contiguous block only when it fixes the last indices. BLAS reads a matrix only when its
first stride is one, so a product on a slice that is not contiguous falls back to the generic
kernel of Julia.

Until #1973 every rank-3 array of the library held its observations on its **first** axis: the
exposure history `observations × assets × factors`, the Gram history `observations × factors ×
factors`, the score stack of a Return Forecast, the tensor Panel Fields `observations × assets ×
labels`, and the time-varying Feature Matrix. ADR 0045 fixed the rule for the Feature Matrix
("observation axis always leading"), so that a rank-3 array read like the returns matrix
`observations × assets`. That reason is a convention, not mathematics.

The work on a slice is per observation. The fit of a `CrossSectionalFactorPrior` regresses each
observation on the exposures of that observation, a diagnostic reads the Gram matrix of each
observation, and a feature distance measures the features of each observation. With the
observations first, the matrix of one observation `A[t, :, :]` jumps `T` entries between two
neighbours. Each kernel then copied the slice before a product, or ran the generic product on a
strided view. In the case of #1971 (T = 500, N = 1600), parts A to D of #1973 removed the copies,
and the strided layout was the largest cost that remained.

## Decision

**Every rank-3 array of the library holds its observations on its last axis.** The matrix of one
observation, `view(A, :, :, t)`, is one contiguous block, and it reaches BLAS with no copy.

| Array | Shape |
| --- | --- |
| Exposure history `Ms`, the design of a cross-sectional regression, the observed exposures, a one-hot or a currency block | `assets × factors × observations` |
| Gram history, factor covariance stacks, the stack of an expanded or a reduced covariance | `factors × factors × observations` |
| Reconstruction weights of the dropped factors | `constrained families × reduced factors × observations` |
| Descriptor score stack of a Return Forecast | `assets × descriptors × observations` |
| Time-varying tensor Panel Field, its masks, and its raw input | `assets × labels × observations` |
| Time-varying Feature Matrix | `assets × features × observations` (`features × assets × observations` at `dims = 2`) |

1. **A matrix over observations keeps them on its first axis.** The returns, the weights, the
   universe masks, the factor returns, the residuals, a one-factor exposure and a numeric Panel
   Field stay `observations × assets`. The whole library reads them so, and a time-series kernel
   reads one asset, a column, with a unit stride. So the rule depends on the rank, and
   `observation_axis(A)` states it once: `1` for a vector or a matrix, `3` for a rank-3 array. A
   helper that cuts or appends the rows of a history of any rank reads the axis there.
2. **The label axis of a tensor Panel Field is its second axis in both shapes**: `assets × labels`
   when static and `assets × labels × observations` when time-varying. The static shape did not
   change.
3. **A universe mask that a reader of a rank-3 array meets keeps the shape of the panel axes**,
   `observations × assets`, as the active mask does. A reader that meets the two transposes the
   mask, through `panel_column_transpose` or `panel_tensor_mask`. So the readers of the universe
   masks do not change. The observed mask and the placeholder mask of a tensor Panel Field are
   not universe masks: each has the size of the values, so a time-varying one is `assets × labels
   × observations`.
4. **The lazy lift of a static Panel Field puts the observation axis where the rule puts it.**
   `RepeatedObservations` replaces `RepeatedLeading`: a lifted vector is `observations × assets`,
   and a lifted matrix is `assets × labels × observations`.

## Considered Options

| Option | Why not |
| --- | --- |
| Keep the observations first, and copy each slice before its product | The cost that #1973 measured. The copies were the largest allocations of the fit. |
| Put the observations last in the cross-sectional factor model only | Two conventions for one kind of array. A Feature Matrix and a tensor Panel Field feed the same kind of per-observation kernel, and a reader would have to know which subsystem made an array. |
| Put the observations last in every array, matrices too | A matrix over observations is the returns layout of the whole public API, and a time-series kernel reads its columns. The row reads that remain in the cross-sectional fit are #1975. |

## Consequences

- **The bits.** A kernel that reads a slice cell by cell, or sums over the same index in the same
  order, gives the same bits. The probe of #1973, 211 fits of a `CrossSectionalFactorPrior`
  (batch, carry fold, every solve algorithm, mixed element types), is bit-identical. A kernel
  whose contiguous slice now reaches BLAS, or that reduces along another axis, can move the last
  bits: the stack of `expand_factor_covariance`, the slices of a realised attribution, the panel
  collapse of a meta-optimiser (a relative difference up to `5e-15`), and the mean of
  `AggregateFeatures` (up to `5e-16`). A distance near zero amplifies the change of that mean:
  on a panel of 60 assets, 24 features and 300 observations, the angular distance of
  `FeatureDistance(; alg = AggregateFeatures())` moves by `1e-13` relative to its largest entry,
  where the mean moves by `4e-16`. `AggregateDistances` at `dims = 2` moves by `3e-15` (R38 of
  #1817).
- **The speed.** On the case of #1971, the fit of a `CrossSectionalFactorPrior` runs 8 to 12 %
  faster than with parts A to D alone, with the same allocations: the default regression 1.43 s
  to 1.32 s, and the normal equations without the leverage check 0.97 s to 0.86 s (medians of 5
  fits, one thread, both trees in one run).
- **A breaking change.** Every public verb that takes or returns a rank-3 array changes its
  layout: `cross_sectional_regression`, `predict`, `cross_sectional_r2` and
  `mean_cross_sectional_r2` (`Z`), `cs_gram`, the exposure and the regression diagnostics over a
  history, `factor_exposure` of a member of several factors, `cross_sectional_groups`,
  `factor_family_basis`, `reduce_exposures`, `reduce_loadings`, `reduce_factor_covariance` and
  `expand_factor_covariance` over a stack, `dropped_factor_weights` with no observation index,
  `descriptor_scores`, `forecast_factor_correlation` and its plot, `factor_attribution` over an
  exposure history, `TensorPanelField`, `TensorPanelInput` and `panel_field_values` of a tensor
  field, and `feature_matrix`, `panel_feature_matrix`, `distance` and `cor_and_dist` of a
  time-varying panel. The migration guide lists them. The docstring of each interface that a user implements (`cross_sectional_regression`,
  `unseen_member_design`, `factor_exposure` of an observed member, `attribution_leverage_split`)
  states the break.
- **Out of scope.** The bootstrap covariance stacks `assets × assets × draws`, the lattice of an
  estimated regime bias, and the co-moment arrays hold their varying axis last already, and none
  of them has an observation axis. #1975 records them, and the strided row reads of the matrices
  in the cross-sectional fit.

This ADR amends ADR 0045 (decision 7, the layout of the time-varying Feature Matrix), ADR 0102
(the shapes of a tensor Panel Field and of the Feature Matrix), ADR 0103 (the shape of the design)
and ADR 0112 (the shape named in its options).
