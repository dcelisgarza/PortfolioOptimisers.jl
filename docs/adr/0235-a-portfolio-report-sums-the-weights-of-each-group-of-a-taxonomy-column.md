---
status: accepted
---

# A Portfolio Report sums the weights of each group of a taxonomy column

## Context

A Portfolio Report (ADR 0228) shows the weights of each asset of a candidate. A reader who
compares candidates also asks how much of each candidate sits in a group: a sector, a currency, a
venue. A constraint can cap such a group, but a report did not show the exposure that a fit gave,
so the reader summed the weights table by hand.

The library already holds the groups. A **taxonomy column** is a key of a `UniverseSets`, such as
`nx_sector`, that holds one group label for each asset of `sets.dict[sets.xkey]`, in that order.
`asset_sets_matrix` and `panel_input` read it through `taxonomy_column`, and the view of a
`UniverseSets` cuts it by position, so it stays correct after a filter step.

Three facts bore on the design.

1. **The weights of an entry are on names, not on positions.** `prediction_weights` puts the folds
   of a path and the paths of a population on the union of their names (ADR 0228). A fold can hold
   a part of the universe, so a position in the weights is not a position in the taxonomy column.
2. **A label can be absent.** A caller who builds a taxonomy column from data that has gaps puts
   `missing` for an asset with no label, and an asset of the weights can be absent from the
   universe of the sets.
3. **`statistics_table` sets the layout.** One row of the summary gives a table of two columns,
   and many rows give a sortable table with one row for each path and member.

## Decision

**1. Two keywords.** `portfolio_report` takes `sets`, one `UniverseSets` for every prediction or
one for each prediction with `nothing` for none, as `allocs` takes one result for each prediction.
It takes `taxonomy`, the names of the taxonomy columns, and each name gives one subsection of an
Exposure section. No sets, or an empty `taxonomy`, gives no section, so a report that does not
ask pays nothing.

**2. An asset maps to its label by name.** The section reads the rows of `prediction_weights`,
and it maps each asset to its label through `sets.dict[sets.xkey]`. An asset that the universe
does not name, or whose label is `missing`, counts in the group `(none)`. So the groups of a row
sum to its weights, and no weight is lost.

**3. The section computes only the sums.** For each group, the long sum adds `max(x, 0)` and the
short sum adds `min(x, 0)`, so a short shows on the short side. The net is their sum, and the gross
exposure is the long sum less the short sum. A `NaN` weight gives a `NaN` on both sides of its
group, and the section does not hide it. The sums take the element type of the weights.

**4. The layout follows `statistics_table`.** One row of weights, a single fold, gives a sortable
table with one row for each group and the columns `long`, `short` and `net`, and the caption gives
the gross exposure. Many rows, a path or a population, give a sortable table with one row for each
fold or path, one column of net exposure for each group, and the column `gross`. A reader then
sees how stable the tilt of a candidate is over its folds. A prediction with many members gives
one table for each member, as the Weights section does.

**5. The allocation reads the realised weights.** With a finite allocation, a single fold gives
the columns `alloc_long`, `alloc_short` and `alloc_net`, and a path gives a last row `allocation`.
The allocation is the allocation of the first member, on the names of the last fold, as in the
Allocation section. The Allocation section is built first, so its checks run before the Exposure
section reads it.

**6. The section comes after the Weights section**, because it sums that section.

## Rejected

- **Every key of the sets that starts with `xkey`.** A key such as `nx_weight_cap` holds a number
  for each asset and is not a taxonomy. The caller names the columns.
- **The plain groups of the sets.** A plain group is a list of asset names, and nothing states
  which plain groups belong to one taxonomy or that they cover the universe. A taxonomy column
  gives one label for each asset, so its groups cover the universe with `(none)`.
- **A statistic of each group, such as its risk contribution.** The report computes no statistic
  of its own (ADR 0228). A risk contribution by group can come from the risk contribution of the
  assets, in a figure.
- **Positions in place of names.** The weights of a path are on the union of the fold names, so a
  position gives a wrong label when a fold holds a part of the universe.
