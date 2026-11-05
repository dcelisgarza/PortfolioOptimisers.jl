---
status: accepted
---

# A Panel Field Scorer reduces the product of its Panel Fields, and its statistic states the orientation

## Context

[ADR 0225](0225-an-asset-scorer-scores-the-assets-from-the-whole-returns-data-and-each-fold-scores-its-own-coverage-universe.md)
made `AbstractAssetScorer` the seam for a score that a risk measure on one column of returns
cannot state. Its two members read the returns. A selection by liquidity or by size reads neither:
it reads the traded amount or the market capitalisation of each asset, and these are Panel Fields
of the Asset Panel `rd.pnl`. Before this decision, no scorer read a Panel Field. `EWVolumeRatio`
and `DaysToCover` read the volume, but for a Factor Exposure, not for a selection.

A floor of liquidity must move with the window. A selector inside a `Pipeline` fits on the
training window of each fold, and `port_opt_view` cuts the panel with the returns. So a scorer
that reads `rd.pnl` reads only the observations of the window, and a cut of the universe before
the fit cannot do this.

## Decision

### `PanelFieldScorer(; fields, stat = MedianValue(), bib = nothing, min_obs = 1)`

- **`fields` is a vector of names, read as their product.** `EWVolumeRatio` set the precedent: its
  product side reads a vector of names as their product, through `ew_ratio_values`, because a
  traded amount is a price times a volume, and no single Panel Field holds it. The scorer calls the
  same reader, so one rule states what a vector of names means, and the reader refuses a missing
  panel, a field that is not numeric, and a static panel, with the messages of
  `descriptor_field_values`. A vector of one name reads that field.
- **`stat` follows the callable pattern of `Num_VecToScaM`.** It takes a `VectorToScalarMeasure`
  or a plain `Function`, and `vec_to_real_measure` applies it. The bound
  `Union{VectorToScalarMeasure, Function}` is the one of `oriented_score` (ADR 0231). A number is
  not a statistic, so the bound leaves it out.
- **A value counts only where it is finite.** The reader writes `NaN` outside the universe and at
  each cell that a fill policy wrote, because a filled volume is not a trade. The statistic reads
  the finite values of each asset in the order of the observations, through
  `finite_vec_to_real_measure`, the helper that the default drop score of a redundancy selection
  uses. It cuts the observation weights of the statistic to the finite values, so a weighted
  statistic keeps the weight of each value.
- **`min_obs` is the count of `EWMean` and `DaysToCover`.** An asset with fewer finite values has
  the score `NaN`, through `min_folds_scores`, and every rule drops it (ADR 0225).

### The orientation is a trait of the statistic, and `bib` overrides it

The maintainer chose this over a required `bib::Bool`. `bigger_is_better` gains methods on the
statistic. A `VectorToScalarMeasure` answers `true` by default: a statistic of location or of
size, such as a median traded amount, is better when it is bigger. `StdValue` and `VarValue`
answer `false`, because a dispersion is a risk, as a variance is for a risk measure. A plain
`Function` answers `true`, because it has no type that can state a trait. The orientation is a
trait of the type, as for a risk measure, so a call on a concrete scorer folds to a constant.

The orientation of a Panel Field can differ from that of its statistic: a bigger median leverage
is worse. The field `bib::Option{<:Bool}` then overrides the trait. `nothing`, the default, takes
`bigger_is_better(stat)`, which is the use of `nothing` for a trivial default of ADR 0222.

## Considered options

| Question | Refused | Why |
| --- | --- | --- |
| The fields | One `field` name. | A traded amount needs two fields, and the product precedent already reads a vector of names. |
| The fields | A second measure `comb` that joins the fields at each cell. | The precedent of `EWVolumeRatio` reads a vector of names as their product. A sum of fields has the `name => coefficient` form of `PanelFieldRatio`, and no selection needs it yet. |
| The orientation | A required `bib::Bool`, after ADR 0225: a guess cuts the wrong end. | The statistic states the orientation, so it is not a guess, and the override covers a field whose orientation differs. |
| The orientation | `bib = true` as a default. | It is a guess for a statistic of dispersion. |
| A static panel | Read a static field as one observation. | A score over the window reads a time-varying panel, and the reader of the precedent refuses a static one with a message that names it. |

## Consequences

- A liquidity floor is `ScoreSelector(; score = PanelFieldScorer(; fields = [price, volume]),
  rule = ThresholdRule(; lo))`, and a floor of size is the same with a market capitalisation. A
  `RankRule` keeps the most liquid assets.
- `bigger_is_better` now answers for a `VectorToScalarMeasure` and for a `Function`. The
  `# Interfaces` section of `VectorToScalarMeasure` names the method as optional.
- A statistic with observation weights, such as `MedianValue(; w)`, holds one weight for each
  observation of the returns data that it scores, and a gap in the panel cuts the weight of the
  gap.
