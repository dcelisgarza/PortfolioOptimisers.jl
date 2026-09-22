```@meta
Description = "Every breaking change between releases of PortfolioOptimisers.jl and the replacement to write, one section per version."
```

# [Migration guide](@id migration)

This page lists every change that can break code written against an earlier release, with the replacement to write. Each version has its own section; read the one for the version you are upgrading from, and every section after it.

## From v0.30 to v0.31

v0.31 is a large release. The changes fall into seven groups, and most code needs at most the first three:

 1. [Names that were removed](@ref migration-0-31-removed), and what replaces each.
 2. [Loading prices](@ref migration-0-31-prices): `prices_to_returns` now converts prices and nothing else. Joining, filling and filtering are separate steps.
 3. [Asset features](@ref migration-0-31-features): the feature matrix `Z` is replaced by the Asset Panel.
 4. [Fees](@ref migration-0-31-fees): fixed fees are charged once per holding period, and `calc_fees` returns a pair.
 5. [Renamed keywords and changed defaults](@ref migration-0-31-renames).
 6. [Calls that now raise](@ref migration-0-31-raises) where they used to answer.
 7. [Results that move](@ref migration-0-31-numbers): configurations whose numbers differ from v0.30, and why.

A last section covers [code that extends the library](@ref migration-0-31-extending) with its own estimators, and [links into the documentation](@ref migration-0-31-docs).

Julia 1.11, 1.12 and 1.13 are supported.

### [Names that were removed](@id migration-0-31-removed)

| Removed | Write instead |
| --- | --- |
| `Imputer`, `ImputerResult`, and the `Impute` extension | `PriceGapFill(; fill = CarriedPrice())` on prices, or `MissingDataFilter` to drop rather than fill. See [Loading prices](@ref migration-0-31-prices). |
| `FeaturePrior` | Nothing wraps the prior any more. Put the features on the returns as an Asset Panel and select them on the distance. See [Asset features](@ref migration-0-31-features). |
| `RegressionFeatures` | `FeatureDistance(; ape = RegressionPanel())` |
| `PhylogenyFeatures` | `FeatureDistance(; ape = PhylogenyPanel(; pl, alg))` |
| `AssetSetsFeatures`, `asset_sets_features`, `asset_sets_feature_names`, `resolve_feature_value` | `panel_input(sets, key)` builds a Panel Field input from a `UniverseSets` key; `asset_panel(inputs)` builds the panel. |
| `Scale` | A graded classification is a `TensorPanelInput` with its own `labels`. |
| `MuEllipsoidalUncertaintySet`, `SigmaEllipsoidalUncertaintySet` | `MuUncertaintySetClass`, `SigmaUncertaintySetClass` (the same tags, renamed because they now also class the norm-ball sets). |
| The `nz` and `Z` keywords of `prices_to_returns`, `ReturnsResult` and `PricesResult`; the `Z` field of `LowOrderPrior` | The `pnl` keyword, an `AssetPanel`. |
| The `z_src` keyword of `JuMPOptimiser`, `HierarchicalOptimiser`, `NestedClustered` and every optimiser that took it | Gone. Features reach a distance through the panel on the returns, or through `FeatureDistance(; ape)`. |
| The `zkey` keyword and field of `UniverseSets` | Gone. A classification enters through `panel_input`. |
| `feature_matrix(ze, pr, X, F, sets)` | `feature_matrix(pnl)` and `feature_matrix(pnl, sel)` stack an Asset Panel; `feature_labels` names the columns. |
| `calc_fees(w, p, …)`, `calc_asset_fees(w, p, …)` and the other price-carrying fee methods | The finite allocation charges its fees inside its own model, on the shares it buys. Read them off `DiscreteAllocationResult.fees` or `GreedyAllocationResult.fees`. |

### [Loading prices](@id migration-0-31-prices)

`prices_to_returns` used to do five jobs with one keyword list: join the factor and benchmark tables onto the assets, collapse to a lower frequency, impute or delete gaps, apply a function, and compute returns. It now computes returns, and each of the other jobs is a named step you compose. The keywords that survive are `ret_method` and `padding`, plus a new `gap_return_alg`.

```julia
# v0.30
rd = prices_to_returns(X, F; B = B, iv = iv, ivpa = ivpa,
                       join_method = :outer, collapse_args = (week, last),
                       missing_col_percent = 0.9, missing_row_percent = 0.9,
                       impute_method = Impute.Interpolate(), map_func = f)

# v0.31
pr = price_ingestion(PriceIngestion(; join_method = :outer, collapse_args = (week, last)), X;
                     F = F, B = B, iv = iv, ivpa = ivpa)
mdf = fit_preprocessing(MissingDataFilter(; col_thr = 0.9, row_thr = 0.9), pr)  # delete, or
mdf = fit_preprocessing(PriceGapFill(; fill = CarriedPrice()), pr)               # fill
rd = prices_to_returns(apply_preprocessing(mdf, pr))
```

For the one-line form, `prices_to_returns(X; kwargs...)` still works: it ingests with the default `PriceIngestion()` and forwards `kwargs` to the conversion. Its factor, benchmark and implied-volatility tables now enter through `price_ingestion`.

What changed underneath, and what it does to a result:

- **Gaps are no longer deleted.** An asset that lists late, delists, or is suspended stays in the table with `NaN` where it had no price, and the `ReturnsResult` carries an `AssetPanel` whose two masks say which assets are in the universe at each date and which can be estimated. Every optimiser reads those masks, fits on the assets it can trade, and returns an exact zero weight for the rest. A walk-forward over such a table therefore runs on each fold's own universe, where v0.30 either dropped the asset from the whole run or held it after it delisted. On a table with no gaps, nothing changes.
- **The join is `:left` by default**, where it was `:outer`. A factor or benchmark table with more dates than the assets no longer extends the observation clock; it is padded onto the asset dates and the padding is reported. Pass `PriceIngestion(; join_method = :outer)` for the old clock.
- **`missing_row_percent = nothing`** (the modal-history cut) has no counterpart. `MissingDataFilter(; col_thr = 0.0)` names the condition it approximated.
- **`map_func`** has no counterpart: apply the function to the `TimeArray` before ingestion.
- **`nan_to_missing`** is gone because every absence is spelled `NaN` on ingestion.
- **A price of `Inf`** is refused at ingestion. It used to reach the conversion and produce a `-100 %` return.
- **A column name shared by two tables**, or shared with the timestamp column, is refused. It used to be resolved by position.

If you hand-build a `PricesResult` or `ReturnsResult`, the `nz`/`Z` keywords are replaced by `pnl`, and `PricesResult` gains `span`, the listing calendar the conversion reads.

### [Asset features](@id migration-0-31-features)

A feature used to reach a clustering optimiser as a matrix `Z` and its labels `nz`, stored on the returns, with the optimiser's `z_src` choosing between the data and a `FeaturePrior` that produced them. All of that is replaced by one object, the Asset Panel, which lives on the returns as `rd.pnl` and holds each feature as a named Panel Field: numeric, categorical, or a tensor with its own labels.

```julia
# v0.30: a classification from the asset sets, through a feature prior
rd  = prices_to_returns(X; nz = ["nx_sector"], Z = Z)
pe  = FeaturePrior(; pe = EmpiricalPrior(), ze = AssetSetsFeatures(; vals = ["nx_sector"]),
                   sets = sets)
opt = HierarchicalOptimiser(; pe = pe, cle = ClustersEstimator(; de = FeatureDistance()),
                            z_src = :prior)

# v0.31: the classification is a Panel Field, and the distance selects it by name
rd  = ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts,
                    pnl = asset_panel([panel_input(sets, "nx_sector")]))
opt = HierarchicalOptimiser(; pe = EmpiricalPrior(),
                            cle = ClustersEstimator(; de = FeatureDistance(; sel = ["sector"])))
```

A field built from `UniverseSets` is named by stripping the key prefix, so `"nx_sector"` becomes `"sector"`. A numeric feature you hold as a vector or matrix is a `NumericPanelInput(; name, vals)`; a graded classification with its own column labels is a `TensorPanelInput(; name, vals, axis, labels)`.

- `FeatureDistance()` with no `sel` stacks every field of the panel. `sel` takes field names, `name => [levels]`, `name => level` or `name => :observed`; the integer column index of an earlier `sel` is gone, so `sel = [3]` becomes the column's name.
- The two feature producers become the `ape` keyword: `FeatureDistance(; ape = RegressionPanel())` clusters on the factor loadings of the prior's regression, `FeatureDistance(; ape = PhylogenyPanel(; pl, alg))` on a phylogeny built at the point of use.
- A blank in a feature (`missing`, `nothing` or `NaN`) is filled by the field's fill policy — `NoPanelFill`, `ConstantPanelFill`, `ForwardPanelFill`, `BackwardPanelFill` — and never reaches a consumer. A categorical field's fill policy must name a level: `CategoricalPanelInput(; alg = ForwardPanelFill())` without a value is refused where it used to mint a level `"0.0"`.
- `feature_matrix(pnl)` and `feature_labels(pnl)` stack and name a panel; the labels are themselves a selector, so `feature_matrix(pnl, feature_labels(pnl))` rebuilds the matrix.

### [Fees](@id migration-0-31-fees)

The fee clock is now explicit. The proportional fees `l`, `s` and the turnover fee `tn` are charged on every observation; the fixed fees `fl` and `fs` are charged once for the holding period. v0.30 charged the fixed fees on every observation and the turnover fee once, so a fold of `T` rows paid a fixed fee `T` times. **Any backtest with a fixed fee reports a different net return.**

- `calc_fees(w, fees)` is now `calc_fees(w, T, fees)` and returns the pair `(periodic, one_off)`. The one number a caller used to read is `calc_total_fees(w, T, fees)`, the cost of the whole holding period; `calc_total_asset_fees` is its per-asset twin. On one observation, `calc_total_fees(w, 1, fees)` reproduces the old sum when `fees.fa` is `nothing`.
- `Fees(; fa = AmortisedFees())` spreads the one-off charge over the period instead; `FirstObservationFees()` and `nothing` charge it on the first observation. A cross-validation scheme's own `fa` overrides the fee's on the fold's return series.
- `Fees` and `FeesEstimator` gain `lq` and `flq`, a proportional and a fixed charge on the position of an asset that leaves the universe during a fold, each a `Turnover` stated over the full universe. A walk-forward over a delisting reports a lower return by the liquidation charge of each exit.
- The finite allocation (`DiscreteAllocation`, `GreedyAllocation`) prices its fee on the shares it buys, inside its own model. The price-carrying `calc_fees(w, p, …)` family is deleted; the result's `fees` field holds the charge.

### [Renamed keywords and changed defaults](@id migration-0-31-renames)

| Where | v0.30 | v0.31 |
| --- | --- | --- |
| `UniverseSets` | `fkey`, `ufkey` | `tfkey`, `utfkey` (time-series factors). `cfkey`, `ucfkey` (cross-sectional factors) and `nikey` (assets that left the universe) are new. The default key strings are unchanged, so a `dict` built for v0.30 still resolves. |
| `ConditionalValueatRiskRange` and every other `…Range` measure | `beta = 0.05` regardless of `alpha` | `beta = alpha`. A call that states `alpha` and not `beta` now gets a symmetric range; state both to keep the old measure. |
| `RelativisticValueatRiskRange` | `kappa_b = 0.3` | `kappa_b = kappa_a`, likewise. |
| `RegimeAdjustedExpWeightedVariance` | `regime_lohi_mult = (0.7, 1.6)`, but the clamp never ran | `regime_lohi_mult = nothing`. A default-constructed estimator answers the same variance; a caller who set the bounds explicitly now gets the clamp they asked for, and a different variance. |
| `RegimeAdjustedExpWeightedVariance` | A covariance estimator | A variance estimator, with `std`. It still fits every `ve` slot; `cov` and `cor` on it are a `MethodError`, and `RegimeAdjustedExpWeightedCovariance` is the covariance form. |
| `PriceIngestion` (was `prices_to_returns`) | `join_method = :outer` | `join_method = :left` |
| `ResourceLimits` | — | `max_ep_grid = 10_000` caps the `K` of a grid entropy-pooling view; a larger `K` is refused at construction. |
| Printing | A field holding `nothing` printed as `w ┴ nothing` | The field is hidden. `PortfolioOptimisers.set_show_nothing_fields!(true)` restores it. |
| `PredictionReturnsResult` | `iv`, `ivpa` documented as "investment vehicle" | Documented as implied volatility. The fields did not change. |

### [Calls that now raise](@id migration-0-31-raises)

Each of these returned an answer in v0.30 that did not honour the configuration. The raise names the cause and, where there is one, the way out.

| Call | v0.30 | v0.31 |
| --- | --- | --- |
| A moment estimator (`cov`, `mean`, …) on a sample holding `NaN` | Some estimators framed the `NaN`, some raised from inside LAPACK, and `GerberIQCovariance` leaked it into the other assets | Refused by name. Use the exponentially weighted family (`ExpWeightedExpectedReturns`, `ExpWeightedVariance`, `ExpWeightedCovariance`), which handles a masked sample, or a `CoveragePolicy` on the estimator, or let the prior fit on the assets priced at every row of the window (the default). |
| A bound vector shorter than the universe on `WeightBounds` | Assets past the end were unbounded | `DimensionMismatch` |
| `WeightBounds(; lb = Inf)` or `(; ub = -Inf)` | The infinite bound resolved to the free bound of its side, so the tightest bound became the loosest | The bound is the bound. |
| `risk_budget_constraints` with a vector for the `:rkb` target | Accepted | `MethodError`; the target takes a scalar |
| `JuMPOptimiser` with an empty `l2`, `lp` or `lpc` vector | Silent no-op | `ArgumentError` |
| `ShrunkExpectedReturns` on a degenerate sample | `NaN`, or a wrong-signed coefficient | `DomainError` |
| `SmythBrobyCovariance` with a negative severity exponent | Accepted | `DomainError` |
| A dimension-reduction regression whose `maxoutdim` equals the factor count | A bare LAPACK error from inside the fit | `DomainError` naming `maxoutdim` and the factor count |
| A Black-Litterman view set in which no name resolves, under `strict = false` | `FieldError` | `IsNothingError` naming the universe and the `strict` escape |
| An entropy-pooling covariance or correlation view naming an unknown asset, under `strict = false` | Raised, with a message naming an empty equation | The row is dropped with a report, as the linear view families already did |
| `CategoricalPanelInput(; alg = ForwardPanelFill())` | Minted a level `"0.0"` | `ArgumentError`; name the level |
| A walk-forward with `test_size = 0` | Ran forever | `ArgumentError` |
| A hyperparameter search key not rooted at `steps` on a Pipeline | Silently matched nothing | `ArgumentError` |
| A calibration rule that reads the sample length under a dynamic observation weight | `MethodError` | `ObservationWeightsError` |
| A price of `Inf`, or a column name two price tables share | A `-100 %` return; a positional guess | `DomainError`; `ArgumentError` |
| A `predict` on a fold that still holds an asset with a missing return (a **held gap**) | The `NaN` propagated | The missing weight sits in cash for that row, and the fold warns once. `strict = true` on the scheme makes it an `ArgumentError`. |

### [Results that move](@id migration-0-31-numbers)

A result from v0.30 is **approximate** where a correction moved a number, and **wrong** where the old answer did not honour the configuration. Recompute any of the following.

**Wrong in v0.30:**

- `AugmentedBlackLittermanPrior` in any configuration: the regression intercept was added twice. `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` with a non-zero `rf` are approximate.
- An entropy-pooling value-at-risk view whose coefficient is not one selected the wrong tail; a view with a target of `0` was dropped in silence; `OptimEntropyPooling` ignored a non-default `sc1`.
- `WeightBounds` with an infinite scalar bound, or a vector bound shorter than the universe (now refused, above).
- A weighted pipeline holding `ImpliedVolatilityRegression` computed the realised volatility unweighted.
- A hyperparameter search over `"opti[1].opt.l2"`: the penalty never entered the model, and the search reported success.
- The effective-asset norm ceiling at any `p != 2`: the ceiling was weaker than asked.
- DBHT read the wrong component of a `CartesianIndex`, so its second output was unrelated to any path; three further DBHT defects moved clusters on a tie and on a repeated call.
- A `KFold` run through the Pipeline threaded the previous fold's weights where the optimiser-level run did not; `FeesEstimator.tn` was the identity, so a turnover fee was charged against stale reference weights on every fold.
- `MultipleRandomised` with a seed draws different subsets, because it now draws from the assets live in each window.

**Approximate in v0.30:**

- A moment estimator with observation weights (`SimpleVariance`, `Covariance`, `Coskewness`, `Cokurtosis`) now weights its centre as well as its spread.
- A co-movement matrix (Gerber, Smyth-Broby) carries a unit diagonal, and a return of exactly zero at a zero band edge is neutral rather than a crossing.
- `GerberIQCovariance` under `Gerber2`: the bound is read per pair.
- A histogram edge is widened by one ulp rather than `eps`, so mutual information and every reader of `calc_hist_data` on standardised data, prices or levels moves in the last digits.
- `cor` on an implied-volatility estimator; `collapse_features` with `MedianCollapse` over a narrow element type; `pre_order` with a custom traversal strategy (output element type).
- A weighted `RelativisticValueatRisk`, its range, and the two relativistic drawdown measures.
- A `Fees` resolved from a `FeesEstimator` carrying a non-default `kwargs`.
- An upper-bound `GridRelativisticValueatRiskView`.
- An `EntropyPoolingPrior` over a `FactorPrior` with a `StepwiseRegression`: the nested prior now fits first, so the factors selected are the ones the `FactorPrior` selects alone.
- Every backtest with a fixed fee, a turnover fee, or a delisting: see [Fees](@ref migration-0-31-fees).
- A walk-forward search (`GridSearch`, `RandomisedSearch`) over an optimiser with a `Turnover`, tracking or fee term that reads the previous weights, or a `TimeDependent` schedule: each candidate is now scored through the one fold loop it declared, rather than as independent refits.
- A walk-forward in which a fold failed and a later fold read the previous weights: it now reads the last solved fold's weights instead of raising.

**Meaning changed, number unchanged:**

- The `Max` and log-sum-exp scalarisers put an upper bound on `model[:risk]`, not the aggregate; read the exact figure back with `expected_risk`.
- `number_effective_assets` under an order-`p` norm reads `(sum_i |w_i|^p)^(1/(1 - p))`.
- `res.fb` on an optimisation result records the fallback chain, the `(estimator, result)` pairs that failed before the answer, where it was always `nothing`.
- Every optimisation result carries `imsk`, the assets it was allowed to trade. On a table with no gaps it is all `true`.

### [If you extend the library](@id migration-0-31-extending)

- **A prior estimator's returns-matrix method takes the Asset Panel as its third positional argument**: `prior(pe, X, F = nothing, pnl = nothing; kwargs...)`. The library calls it with all three, so a method with a fixed two-argument signature raises a `MethodError` on the carrier route until it accepts (and may ignore) the third.
- **An uncertainty-set estimator that answers `reads_prior_result` as `true`** receives an `rd` keyword it may ignore.
- **`LowOrderPrior.rr`** is bound to the loadings root, so a result stored under the old `AbstractRegressionResult` bound and handed to a consumer meets a `MethodError`.
- **`Regression`** gains a fourth field, `esigma`, the idiosyncratic variances, when the regression is fitted with `rsd = true`. Code that constructs or compares it positionally sees the extra field.
- **`SubsetResamplingResult`, `NestedClusteredResult` and `StackingResult`** carry `imsk` before `fb`. Keyword construction is unaffected.
- **Weight bounds** are validated against the stated asset count with `DimensionMismatch`, so a constraint generator outside the library that relied on truncation raises.

### [Links into the documentation](@id migration-0-31-docs)

The API pages are renumbered to follow the new source layout, one page per source file. A bookmark into `api/24_Plotting`, `api/25_Aliases`, `api/19_RiskMeasures` or `api/20_Optimisation` on the `stable` site resolves to the new number (`22_Plotting`, `23_Aliases`, `16_RiskMeasures`, `17_Optimisation`) after this release; navigate from the [API introduction](@ref) rather than by number.

## From v0.31 to v0.32

v0.32 adds the online portfolio selection family and touches released code in four places: one name and one keyword are removed, the naive optimisers gain a `fees` field, three Results gain fields, and two released numbers move.

### [Names that were removed](@id migration-0-32-removed)

| Removed | Write instead |
| --- | --- |
| `OnlineStep`, and the `ff` keyword of `IndexWalkForward` and `DateWalkForward` | The online step is a scheme wrapped in `Online`, built by its own constructor: `OnlineIndexWalkForward(train_size, test_size; …)` and `OnlineDateWalkForward(train_size, test_size; …)` take every keyword of the plain scheme except `expand_train`, which an online run sets. `IndexWalkForward(60, 1; ff = OnlineStep())` becomes `OnlineIndexWalkForward(60, 1)`. `Online(cv)` written by hand on a scheme is refused with the three constructors named. |
| The derived `expand_train` (`nothing` on the two walk-forwards) | `expand_train` is a plain `Bool` keyword again, `false` by default, as in v0.30. |
| `fold_fit`, `cv_online_info`, `cv_resume_info` (unexported) | Nothing. The fold loop reads whether a scheme steps off the scheme's type, and the online arm announces nothing. |

### [Renamed keywords and changed defaults](@id migration-0-32-renames)

- **The naive optimisers charge fees.** `EqualWeighted`, `InverseVolatility`, `RandomWeighted` and `PreviousWeights` take `fees`, and `NaiveOptimisationResult` carries it. Keyword construction is unaffected; the positional constructor of `NaiveOptimisationResult` takes `fees` as its third argument, `NaiveOptimisationResult(pr, wb, fees, retcode, w, imsk, fb)`.
- **`PerformanceSummaryResult`** has four more fields — `excess_ret`, `tracking_error`, `information_ratio` and `turnover` — so its positional constructor takes sixteen arguments. `performance_summary` takes a `benchmark` and fills the first three; without one they are `NaN`, and `turnover` is `nothing` without a held weight path.
- **`plot_performance_summary`** is one method over an array, an `OptimisationResult` or a prediction Result, with `benchmark` as a keyword, where v0.31 had six arities. Every call written against v0.31 still resolves.
- **The risk-measure builders' `opt` slot** is a `RiskConstraintOwner`: a JuMP optimiser or a programme Allocation Set. Every method keeps its argument list; a method you added under the old bound still dispatches.

### [Results that move](@id migration-0-32-numbers)

**Wrong in v0.31:**

- A walk-forward that charged a turnover fee on a naive head priced the fee against the constructor's weights on every fold, never against the weights the fold held: buy-and-hold paid the most and a constant rebalanced portfolio paid nothing. The fee is now charged against the previous weights the loop threads, and the first fold's fee is the entry trade from the head's own start.

**Approximate in v0.31:**

- The t-statistic of every information-coefficient summary read the forward windows as independent rows. Under overlapping windows — `forecast_holding_period` from its second row on, `forecast_evaluation_summary` at any `step < horizon`, `exposure_ic_summary` on a block at any `horizon > 1` — it now reads a Newey–West variance at the known overlap order and reports a smaller statistic; `ic_ir` does not move.

### [If you extend the library](@id migration-0-32-extending)

- **A risk-measure builder** `set_risk_constraints!(model, i, r, opt, pr, …)` is called with `opt::RiskConstraintOwner`; a method bound to `RiskJuMPOptimisationEstimator` alone is not reached from a programme Allocation Set.
