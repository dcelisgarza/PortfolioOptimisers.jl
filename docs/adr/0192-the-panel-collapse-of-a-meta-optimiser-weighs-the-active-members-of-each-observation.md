---
status: accepted
---

# The panel collapse of a meta-optimiser weighs the active members of each observation

## Context

A meta-optimiser collapses its Asset Panel onto its sub-portfolios (`collapse_asset_panel`), so
that an outer `FeatureDistance` measures the clusters of `NestedClustered` and the inner
portfolios of `Stacking`. The collapse starts from the normalised inner weights `W̃`
(`synthetic_asset_weights`), which divide out the gross exposure `s_k` of each sub-portfolio. A
sub-portfolio holds each member with one weight for the whole history, and a time-varying panel
lists and delists the members. The old collapse formed `A W̃` over every member at every
observation, so the value that the panel stores at an inactive cell entered the feature of the
sub-portfolio. ADR 0102 keeps that value finite, and its content is arbitrary.

The census of issue #1411 wrote `1e6` into every inactive cell and found the defect: the poison
took every angular distance between three sub-portfolios from 0.32 to 0.74 down to about 1e-7.
Issue #1451 measured it and the maintainer decided the rule in a grilling session. Issue #1456
built it.

Two rules are coherent. The outer returns of a meta-optimiser are `X w` on the returns of the
prior, which hold a zero at an inactive cell of an admitted asset (ADR 0118 and the coverage
fill): the weight of that asset sits in cash. So a collapse can divide the cash out, or read it
as cash with a zero feature. Under the default `AngularDist` the two rules give the same distance
on a rectangular field, because the cash rule scales each row by the active share. They differ
by up to 24 % under a standardisation of the columns or under `Euclidean`, and on a square field.

## Decision

**At each observation, the collapse weighs only the members that are active and hold data
there.** A field
`pcol` on `NestedClustered` and on `Stacking` holds the rule, an `AbstractPanelCollapseAlgorithm`,
which stays unexported. The two rules are exported.

| Rule | Collapsed value at observation `t` |
| --- | --- |
| `RenormaliseActive()`, the default | `a°_tk = Σ_i W̃_ik r_ti a_ti / Σ_i W̃_ik r_ti` |
| `InactiveAsCash()` | `a°_tk = Σ_i W̃_ik r_ti a_ti` |

`r = m .& u` marks the cells that the collapse reads: `m` is the active mask of the panel, and `u`
is the data mask of the field, `.!pmsk` (#1508, #1631). A member is read where it is active and its
cell holds data: a value that the raw input carried, or that a fill policy wrote. A placeholder is
not data, and the collapse skips it. A field with no placeholder mask reads `r = m`. **The default renormalises** because the collapse already
divides out the gross exposure: leverage, short positions and the cash that a budget leaves. The
cash of an inactive member is one more part of the portfolio outside the universe, and the
default divides it out in the same way, so each value stays a convex combination, as the
docstring of `collapse_asset_panel` requires. The cash rule is one keyword away because it
matches the outer returns for an additive feature, such as a loading.

| Part | Rule |
| --- | --- |
| Every kind of field | A numeric field, the one-hot block of a categorical field and a rectangular tensor field take the restricted weights `D_t` of each row, `D_t,ik = W̃_ik r_ti / d_tk`. A tensor field reads the data mask of each label. A square tensor field reads a pair of members with the weight `W̃_ik W̃_jl` where both are active and the cell holds data, and divides by the weight of the read pairs, so it stays symmetric. Where the read pairs are separable, this is `S°_t = D_tᵀ S_t D_t`. |
| A lifted field | A static input that a time-varying panel holds as a `RepeatedLeading` takes the rule of each row, so its collapse varies over time. |
| A static panel | It has no mask and no inactive member, and it does not change under either rule. |
| The divisor | `d_tk` is the weight of the read members under the default, and one under the cash rule. It is also one where a sub-portfolio has no read member, whose values are then the zero of its empty column, and where every member with weight is read. |
| The observed mask | A collapsed cell is observed where a member that is active and observed has weight: the collapse reads `m .& omsk`, not `omsk`. A field with no observed mask keeps none. |
| The placeholder mask | A collapsed cell holds a placeholder where no member that is active and holds data has weight: `pmsk° = .!(collapse of m .& u)`. A field with no placeholder mask keeps none. An observed collapsed cell is never a placeholder. |
| The universe masks | They keep their rule: a sub-portfolio is active where one member with weight is, and so for the estimation mask. |
| The cross-validated path | `rebuild_asset_panel` stacks the collapsed active and estimation masks of each fold, in place of all-`true` masks, so a sub-portfolio is inactive on the same rows on both paths. A static panel keeps all-`true` masks. |
| `iv` | `prepare_outer_rd` collapses it with the same rule, row by row, through `collapse_rate`. |
| `ivpa` | A vector collapses over the members active at the last row of the outer window, the row at which the outer fit stands. A scalar stays as it is. |

**The cross-validated path collapses the rates again.** Each fold collapses `iv` and `ivpa` in
`reconstruct_rd` with its own weights and no panel, so it weighs every member. On a time-varying
panel, `rebuild_fold_rates` collapses the original `rd.iv` and `rd.ivpa` again for each fold,
with the weights and the active rows of that fold, as `rebuild_asset_panel` does for the panel.
So `cv` changes execution only, and the rates follow the rule on both paths. The resolution of
issue #1451 named the fold-less path, and this build carried the rule to the cross-validated
one, as the same resolution requires of the masks.

**A panel with no inactive member keeps its old result bit for bit.** The divisor of a
sub-portfolio whose weighted members are all active is exactly one, not a sum that rounds to
one, and the vector form of `ivpa` keeps the product of the old code. #1456 compared 16
optimisations (`NestedClustered` and `Stacking`, with and without `cv`, on no panel, a static
panel, a time-varying panel with every cell active and the census panel under the default prior)
and 8 calls of `prepare_outer_rd` with the old code: every result was bit-equal, except the calls
of `prepare_outer_rd` on the census panel, which has inactive members.

## Consequences

- `prepare_outer_rd`, `collapse_asset_panel`, `fold_asset_panel`, `rebuild_asset_panel` and
  `rebuild_returns_result` take the rule as a positional argument with no default, so no path
  reads a different rule from the meta-optimiser's. The fold-less `predict_outer_returns` reads
  `opt.pcol`.
- A time-varying `ivpa` is outside this decision, in issue #1455.
- `NestedClustered` and `Stacking` print one more field.
