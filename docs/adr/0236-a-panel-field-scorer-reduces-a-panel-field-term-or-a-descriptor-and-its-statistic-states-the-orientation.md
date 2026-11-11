---
status: accepted
---

# A Panel Field Scorer reduces a Panel Field term or a Descriptor, and its statistic states the orientation

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

- **`fields` is a `PanelTermType` or a Descriptor Estimator, and `descriptor_field_values` reads
  it.** A `PanelTermType` is one of the three Panel Field terms that a side of `EWVolumeRatio`
  takes: one name, a vector of `name => coefficient` pairs read as their sum, or a vector of names
  read as their product. A traded amount is a price times a volume, and no single Panel Field holds
  it, so the product form is the liquidity floor. A Descriptor Estimator, such as `EWVolumeRatio`
  or `EWAmihudIlliquidity`, gives its Descriptor, so a smoothed turnover or an illiquidity is a
  score with no new type.
- **One reader and one check state every form.** `descriptor_field_values` reads all five forms:
  `nothing` (the absolute returns of a side of `EWVolumeRatio`), the three Panel Field terms, and a
  Descriptor Estimator, through `descriptor`. `assert_panel_terms` checks them. The scorer calls
  both with no helper of its own. Before this decision, `ew_ratio_values` read the forms of a side
  of `EWVolumeRatio`, and `assert_ew_ratio_side` checked them. Each one sent two forms to the
  generic function and held the other two, so both go, and the generic functions take their
  methods. The product promotes the types of its fields, as the sum does, so the type of the first
  field does not cut the precision. The scorer leaves `nothing` out of its bound, because the
  absolute returns are not a Panel Field.
- **`stat` follows the callable pattern of `Num_VecToScaM`.** It takes a `VectorToScalarMeasure`
  or a plain `Function`, and `vec_to_real_measure` applies it. The bound
  `Union{VectorToScalarMeasure, Function}` is `Num_VecToScaM` with no `Number`. A number is not a
  statistic, so the bound leaves it out. It is also `Sca_VecToScaM` of `oriented_score`
  (ADR 0231) with no `Scalariser`, because a `Scalariser` reduces the values of several measures
  through `scalarise`, and `vec_to_real_measure` does not apply it.
- **A value counts only where it is finite.** The reader writes `NaN` outside the universe and at
  each cell that a fill policy wrote, because a filled volume is not a trade. A Descriptor is also
  `NaN` before its warm-up ends, and it reads the window alone, so its recursion starts again in
  each window. The statistic reads
  the finite values of each asset in the order of the observations, through
  `finite_vec_to_real_measure`, the helper that the default drop score of a redundancy selection
  uses. It cuts the observation weights of the statistic to the finite values, so a weighted
  statistic keeps the weight of each value.
- **`min_obs` is the count of `EWMean` and `DaysToCover`.** An asset with fewer finite values has
  the score `NaN`, through `min_folds_scores`, and every rule drops it (ADR 0225).

### `PanelFieldTarget` takes the same forms, and a target compares by configuration

The forward target of a forecast evaluation has the same job: it names the history that a window
reads. So it takes the bound of the scorer, `Union{PanelTermType, AbstractDescriptorEstimator}`,
and the field name of the scorer, `fields`. The released field and keyword `name` go, because a
Descriptor Estimator is not a name. This is a break: `PanelFieldTarget(; name = "market_cap")`
becomes `PanelFieldTarget(; fields = "market_cap")`, and the positional form does not change.
`forecast_target_history` reads it through `descriptor_field_values`. The sum form gives an
excess return, such as `["total_return" => 1, "rf" => -1]`, with no field of its own.

A Descriptor in a target exposed a defect of the summary. `forecast_summary_same_target` compared
the fields of two targets with `isequal`. The library defines no equality on an estimator, so
`isequal` falls back to `===`, which compares a vector field by identity. Two calls of
`EWAmihudIlliquidity()` build two copies of each vector, so two equal targets differed, and the
summary refused a pair of evaluations that answer one question. `same_configuration` replaces it:
it compares two estimators or algorithms field by field, two arrays, tuples or named tuples entry
by entry, and every other pair with `isequal`.

### The orientation is a trait of the statistic, and `bib` overrides it

The maintainer chose this over a required `bib::Bool`. `bigger_is_better` gains methods on the
statistic. A `VectorToScalarMeasure` answers `true` by default: a statistic of location or of
size, such as a median traded amount, is better when it is bigger. `StdValue` and `VarValue`
answer `false`, because a dispersion is a risk, as a variance is for a risk measure. A plain
`Function` answers `true`, because it has no type that can state a trait. The orientation is a
trait of the type, as for a risk measure, so a call on a concrete scorer folds to a constant.

The orientation of a Panel Field or of a Descriptor can differ from that of its statistic: a
bigger median leverage is worse, and so is a bigger median illiquidity. The field `bib::Option{<:Bool}` then overrides the trait. `nothing`, the default, takes
`bigger_is_better(stat)`, which is the use of `nothing` for a trivial default of ADR 0222.

## Considered options

| Question | Refused | Why |
| --- | --- | --- |
| The fields | A vector of names alone. | A single name is the common case, a floor of size, and the sum and the Descriptor are already forms of the reader. One bound that `EWVolumeRatio` shares states them all. |
| The fields | A second measure `comb` that joins the fields at each cell. | The Panel Field terms already state a sum and a product. |
| The fields | A field of its own for a Descriptor, or a second scorer type. | A Descriptor gives one matrix of values, as a term does, so the same field and the same reader take it. |
| The target | Keep the released field and keyword `name`. | The field holds a sum, a product or a Descriptor Estimator, which are not names, and the sibling with the same job calls it `fields`. The maintainer chose the rename. |
| The comparison | Define `==` on every estimator. | It changes the equality of every type of the library for one check of a summary. |
| The reader | Keep `ew_ratio_values`, or add a reader of the scorer. | Each one only sends a form to `descriptor_field_values` or to `descriptor`. One generic function with one method for each form states the rule once. |
| The orientation | A required `bib::Bool`, after ADR 0225: a guess cuts the wrong end. | The statistic states the orientation, so it is not a guess, and the override covers a field whose orientation differs. |
| The orientation | `bib = true` as a default. | It is a guess for a statistic of dispersion. |
| A static panel | Read a static field as one observation. | A score over the window reads a time-varying panel, and the reader of the precedent refuses a static one with a message that names it. |

## Consequences

- A liquidity floor is `ScoreSelector(; score = PanelFieldScorer(; fields = [price, volume]),
  rule = ThresholdRule(; lo))`, and a floor of size is the same with a market capitalisation. A
  `RankRule` keeps the most liquid assets. `PanelFieldScorer(; fields = EWAmihudIlliquidity(),
  bib = false)` keeps the assets with the smallest median illiquidity.
- `ew_ratio_values` and `assert_ew_ratio_side` are gone. `descriptor_field_values` and
  `assert_panel_terms` read and check every side of `EWVolumeRatio`, whose bounds take
  `Option{<:PanelTermType}`.
- `bigger_is_better` now answers for a `VectorToScalarMeasure` and for a `Function`. The
  `# Interfaces` section of `VectorToScalarMeasure` names the method as optional.
- A statistic with observation weights, such as `MedianValue(; w)`, holds one weight for each
  observation of the returns data that it scores, and a gap in the panel cuts the weight of the
  gap.
