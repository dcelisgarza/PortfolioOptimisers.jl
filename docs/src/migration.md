```@meta
Description = "Every breaking change between releases of PortfolioOptimisers.jl and the replacement to write, one section per version."
```

# [Migration guide](@id migration)

This page lists every change that can break code written against an earlier release, and the code to write instead. Each version has its own section. Read the section for the version you upgrade from, and every section after it.

## From v0.30 to v0.31

v0.31 is a large release. The changes fall into seven groups, and most code needs at most the first three.

 1. [Names that were removed](@ref migration-0-31-removed), and what replaces each.
 2. [Loading prices](@ref migration-0-31-prices). `prices_to_returns` now converts prices and nothing else. Joining, filling and filtering are separate steps.
 3. [Asset features](@ref migration-0-31-features). An asset panel, one object that holds every feature of every asset, replaces the feature matrix `Z`.
 4. [Fees](@ref migration-0-31-fees). The fixed fees are charged once per holding period, and `calc_fees` returns a pair.
 5. [Renamed keywords and changed defaults](@ref migration-0-31-renames).
 6. [Calls that now raise](@ref migration-0-31-raises) an error where they used to return a value.
 7. [Results that move](@ref migration-0-31-numbers). These are the configurations whose numbers differ from v0.30, and the reason for each.

Two more sections cover [code that extends the library](@ref migration-0-31-extending) with its own estimators, and [links into the documentation](@ref migration-0-31-docs).

v0.31 supports Julia 1.11, 1.12 and 1.13.

### [Names that were removed](@id migration-0-31-removed)

| Removed | Write instead |
| --- | --- |
| `Imputer`, `ImputerResult`, and the `Impute` extension | `PriceGapFill(; fill = CarriedPrice())` on prices, or `MissingDataFilter` to drop rather than fill. See [Loading prices](@ref migration-0-31-prices). |
| `FeaturePrior` | Nothing wraps the prior any more. Put the features on the returns as an asset panel, and select them on the distance. See [Asset features](@ref migration-0-31-features). |
| `RegressionFeatures` | `FeatureDistance(; ape = RegressionPanel())` |
| `PhylogenyFeatures` | `FeatureDistance(; ape = PhylogenyPanel(; pl, alg))` |
| `AssetSetsFeatures`, `asset_sets_features`, `asset_sets_feature_names`, `resolve_feature_value` | `panel_input(sets, key)` builds the input of one panel field from a `UniverseSets` key, and `asset_panel(inputs)` builds the panel. |
| `Scale` | A graded classification is a `TensorPanelInput` with its own `labels`. |
| `MuEllipsoidalUncertaintySet`, `SigmaEllipsoidalUncertaintySet` | `MuUncertaintySetClass`, `SigmaUncertaintySetClass`. These are the same tags, renamed because they now also class the norm-ball sets. |
| The `nz` and `Z` keywords of `prices_to_returns`, `ReturnsResult` and `PricesResult`, and the `Z` field of `LowOrderPrior` | The `pnl` keyword, an `AssetPanel`. |
| The `z_src` keyword of `JuMPOptimiser`, `HierarchicalOptimiser`, `NestedClustered` and every optimiser that took it | Gone. Features reach a distance through the panel on the returns, or through `FeatureDistance(; ape)`. |
| The `zkey` keyword and field of `UniverseSets` | Gone. A classification enters through `panel_input`. |
| `feature_matrix(ze, pr, X, F, sets)` | `feature_matrix(pnl)` and `feature_matrix(pnl, sel)` stack an asset panel, and `feature_labels` names the columns. |
| `calc_fees(w, p, …)`, `calc_asset_fees(w, p, …)` and the other fee methods that take prices | The finite allocation charges its fees inside its own model, on the shares it buys. Read them from `DiscreteAllocationResult.fees` or `GreedyAllocationResult.fees`. |

### [Loading prices](@id migration-0-31-prices)

`prices_to_returns` used to do five jobs with one keyword list: join the factor and benchmark tables onto the assets, collapse to a lower frequency, impute or delete gaps, apply a function, and compute returns. It now computes returns, and each of the other jobs is a named step you compose. The keywords that remain are `ret_method` and `padding`, plus a new `gap_return_alg`.

In v0.30:

```julia
rd = prices_to_returns(X, F; B = B, iv = iv, ivpa = ivpa,
                       join_method = :outer, collapse_args = (week, last),
                       missing_col_percent = 0.9, missing_row_percent = 0.9,
                       impute_method = Impute.Interpolate(), map_func = f)
```

In v0.31:

```julia
pr = price_ingestion(PriceIngestion(; join_method = :outer, collapse_args = (week, last)), X;
                     F = F, B = B, iv = iv, ivpa = ivpa)
#! Keep one of the next two lines: the first drops the assets and rows with too many gaps, the second fills the gaps.
mdf = fit_preprocessing(MissingDataFilter(; col_thr = 0.9, row_thr = 0.9), pr)
mdf = fit_preprocessing(PriceGapFill(; fill = CarriedPrice()), pr)
rd = prices_to_returns(apply_preprocessing(mdf, pr))
```

The one-line form `prices_to_returns(X; kwargs...)` still works. It ingests the prices with the default `PriceIngestion()` and passes `kwargs` to the conversion. The factor, benchmark and implied-volatility tables now go through `price_ingestion`.

These are the changes to the conversion, and what each one does to a result.

- **Gaps are no longer deleted.** An asset that lists late, delists, or is suspended stays in the table, with `NaN` where it had no price. The `ReturnsResult` holds an `AssetPanel` whose two masks say which assets are in the universe at each date, and which of them can be estimated. Every optimiser reads those masks, fits on the assets it can trade, and gives the other assets a weight of exactly zero. So a walk-forward over such a table runs each fold on the assets listed in that fold. v0.30 either dropped the asset from the whole run, or held it after it delisted. On a table with no gaps, nothing changes.
- **The join is `:left` by default**, where it was `:outer`. A factor or benchmark table with more dates than the assets no longer adds dates to the observations. The ingestion pads it onto the asset dates and reports the padding. Pass `PriceIngestion(; join_method = :outer)` for the old dates.
- **`missing_row_percent = nothing`** has no counterpart. It kept the columns whose count of missing rows equalled the most common count. `MissingDataFilter(; col_thr = 0.0)` states the condition that it approximated.
- **`map_func`** has no counterpart. Apply the function to the `TimeArray` before the ingestion.
- **`nan_to_missing`** is gone, because the ingestion writes every missing price as `NaN`.
- **A price of `Inf`** raises an error at the ingestion. In v0.30 it reached the conversion and gave a `-100 %` return.
- **A column name that two tables share**, or that a table shares with the timestamp column, raises an error. v0.30 matched such columns by position.

If you build a `PricesResult` or a `ReturnsResult` by hand, the `pnl` keyword replaces the `nz` and `Z` keywords. `PricesResult` also gains `span`, a Boolean matrix that says which assets are listed at each observation, and the conversion reads it.

### [Asset features](@id migration-0-31-features)

In v0.30 a feature reached a clustering optimiser as a matrix `Z` and its labels `nz`, stored on the returns. The `z_src` keyword of the optimiser chose between the data and a `FeaturePrior` that made the features. One object replaces all of that. The asset panel is stored on the returns as `rd.pnl`, and it holds each feature as a named field. A field is numeric, categorical, or a tensor with its own labels.

In v0.30:

```julia
rd  = prices_to_returns(X; nz = ["nx_sector"], Z = Z)
pe  = FeaturePrior(; pe = EmpiricalPrior(), ze = AssetSetsFeatures(; vals = ["nx_sector"]),
                   sets = sets)
opt = HierarchicalOptimiser(; pe = pe, cle = ClustersEstimator(; de = FeatureDistance()),
                            z_src = :prior)
```

In v0.31:

```julia
rd  = ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts,
                    pnl = asset_panel([panel_input(sets, "nx_sector")]))
opt = HierarchicalOptimiser(; pe = EmpiricalPrior(),
                            cle = ClustersEstimator(; de = FeatureDistance(; sel = ["sector"])))
```

The name of a field built from `UniverseSets` is the key without its prefix, so `"nx_sector"` becomes `"sector"`. A numeric feature that you hold as a vector or a matrix is a `NumericPanelInput(; name, vals)`. A graded classification with its own column labels is a `TensorPanelInput(; name, vals, axis, labels)`.

- `FeatureDistance()` with no `sel` stacks every field of the panel. `sel` takes field names, `name => [levels]`, `name => level` or `name => :observed`. It no longer takes an integer column index, so `sel = [3]` becomes the name of the column.
- The `ape` keyword replaces the two feature producers. `FeatureDistance(; ape = RegressionPanel())` clusters on the factor loadings of the prior's regression. `FeatureDistance(; ape = PhylogenyPanel(; pl, alg))` clusters on a phylogeny that `PhylogenyPanel` builds when the distance is computed.
- A blank in a feature is `missing`, `nothing` or `NaN`. The field's fill policy fills it, so no blank reaches the code that reads the panel. The policies are `NoPanelFill`, `ConstantPanelFill`, `ForwardPanelFill` and `BackwardPanelFill`. The fill policy of a categorical field must name a level. `CategoricalPanelInput(; alg = ForwardPanelFill())` without a value raises an error. The table of [calls that now raise](@ref migration-0-31-raises) gives the v0.30 behaviour.
- `feature_matrix(pnl)` stacks a panel, and `feature_labels(pnl)` names its columns. The labels are also a selector, so `feature_matrix(pnl, feature_labels(pnl))` builds the same matrix again.

### [Fees](@id migration-0-31-fees)

v0.31 changes when each fee is charged. The proportional fees `l` and `s` and the turnover fee `tn` are charged on every observation. The fixed fees `fl` and `fs` are charged once for the holding period. v0.30 charged the fixed fees on every observation and the turnover fee once, so a fold of `T` rows paid a fixed fee `T` times. **Any backtest with a fixed fee reports a different net return.**

- `calc_fees(w, fees)` is now `calc_fees(w, T, fees)`, and it returns the pair `(periodic, one_off)`. The one number that a caller used to read is `calc_total_fees(w, T, fees)`, the cost of the whole holding period. `calc_total_asset_fees` gives the same cost per asset. On one observation, `calc_total_fees(w, 1, fees)` gives the old sum when `fees.fa` is `nothing`.
- `Fees(; fa = AmortisedFees())` spreads the one-off charge over the period instead. `FirstObservationFees()` and `nothing` charge it on the first observation. A cross-validation scheme's own `fa` overrides the fee's `fa` on each fold's return series.
- `Fees` and `FeesEstimator` gain `lq` and `flq`, a proportional and a fixed charge on the position of an asset that leaves the universe during a fold. Each one is a `Turnover` stated over the full universe. A walk-forward over a delisting reports a return that is lower by the liquidation charge of each exit.
- The finite allocations, `DiscreteAllocation` and `GreedyAllocation`, price their fees on the shares they buy, inside their own models. The `calc_fees(w, p, …)` methods that took prices are deleted, and the `fees` field of the result holds the charge.

### [Renamed keywords and changed defaults](@id migration-0-31-renames)

| Where | v0.30 | v0.31 |
| --- | --- | --- |
| `UniverseSets` | `fkey`, `ufkey` | `tfkey`, `utfkey` (time-series factors). `cfkey`, `ucfkey` (cross-sectional factors) and `nikey` (assets that left the universe) are new. The default key strings are unchanged, so a `dict` built for v0.30 still resolves. |
| `ConditionalValueatRiskRange` and every other `…Range` measure | `beta = 0.05` regardless of `alpha` | `beta = alpha`. A call that states `alpha` and not `beta` now gets a symmetric range. State both to keep the old measure. |
| `RelativisticValueatRiskRange` | `kappa_b = 0.3` | `kappa_b = kappa_a`, likewise. |
| `RegimeAdjustedExpWeightedVariance` | `regime_lohi_mult = (0.7, 1.6)`, but the clamp never ran | `regime_lohi_mult = nothing`. An estimator built with the defaults returns the same variance. A caller who set the bounds now gets the clamp they asked for, and a different variance. |
| `RegimeAdjustedExpWeightedVariance` | A covariance estimator | A variance estimator, with `std`. You can still pass it to every `ve` keyword. A call of `cov` or `cor` on it throws a `MethodError`, and `RegimeAdjustedExpWeightedCovariance` is the covariance form. |
| `PriceIngestion` (was `prices_to_returns`) | `join_method = :outer` | `join_method = :left` |
| `ResourceLimits` | No such keyword | `max_ep_grid = 10_000` caps the `K` of a grid entropy-pooling view. A larger `K` raises an error at construction. |
| Printing | A field holding `nothing` printed as `w ┴ nothing` | The field is hidden. `PortfolioOptimisers.set_show_nothing_fields!(true)` restores it. |
| `PredictionReturnsResult` | `iv`, `ivpa` documented as "investment vehicle" | Documented as implied volatility. The fields did not change. |

### [Calls that now raise](@id migration-0-31-raises)

In v0.30 each of these calls returned a value that did not follow the configuration. In v0.31 each call raises an error that names the cause and, where one exists, the fix, or it does what the table says.

| Call | v0.30 | v0.31 |
| --- | --- | --- |
| A moment estimator (`cov`, `mean`, …) on a sample that contains `NaN` | Some estimators passed the `NaN` into the result, some raised an error from inside LAPACK, and `GerberIQCovariance` spread it into the other assets | An `IsNonFiniteError` that gives the count and the first position of the non-finite entries. Use an estimator of the exponentially weighted family (`ExpWeightedExpectedReturns`, `ExpWeightedVariance`, `ExpWeightedCovariance`), which handles a masked sample. You can also put a `CoveragePolicy` on the estimator, or let the prior fit on the assets that have a price at every row of the window, which is the default. |
| A bound vector shorter than the universe on `WeightBounds` | Assets past the end were unbounded | `DimensionMismatch` |
| `WeightBounds(; lb = Inf)` or `(; ub = -Inf)` | The infinite bound became the free bound of its side, so the tightest bound became the loosest | The bound keeps the value you give it. |
| `risk_budget_constraints` with a vector for the `:rkb` target | Accepted | `MethodError`. The target takes a scalar. |
| `JuMPOptimiser` with an empty `l2`, `lp` or `lpc` vector | Silent no-op | `ArgumentError` |
| `ShrunkExpectedReturns` on a degenerate sample | `NaN`, or a wrong-signed coefficient | `DomainError` |
| `SmythBrobyCovariance` with a negative severity exponent | Accepted | `DomainError` |
| A dimension-reduction regression whose `maxoutdim` equals the factor count | A bare LAPACK error from inside the fit | `DomainError` naming `maxoutdim` and the factor count |
| A Black-Litterman view set in which no name resolves, under `strict = false` | `FieldError` | An `IsNothingError` that names the universe and the `strict` keyword |
| An entropy-pooling covariance or correlation view that names an unknown asset, under `strict = false` | Raised an error whose message named an empty equation | The view drops the row and reports it, as the linear views already did |
| `CategoricalPanelInput(; alg = ForwardPanelFill())` | Made up a level `"0.0"` | `ArgumentError`. Name the level. |
| A walk-forward with `test_size = 0` | Ran forever | `ArgumentError` |
| A hyperparameter search key on a Pipeline that does not start with `steps` | Silently matched nothing | `ArgumentError` |
| A calibration rule that reads the sample length under a dynamic observation weight | `MethodError` | `ObservationWeightsError` |
| A price of `Inf`, or a column name that two price tables share | A `-100 %` return for the price, and a match by position for the name | `DomainError` for the price, and `ArgumentError` for the name |
| A `predict` on a fold that still holds an asset with a missing return | The `NaN` propagated | The weight of that asset goes to cash for that row, and the fold warns once. `strict = true` on the scheme makes it an `ArgumentError`. |

### [Results that move](@id migration-0-31-numbers)

A result from v0.30 is **approximate** where a correction moved a number, and **wrong** where the old answer did not follow the configuration. Recompute any of the following.

These results were wrong in v0.30:

- `AugmentedBlackLittermanPrior` in any configuration, because the prior added the regression intercept twice. `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` with a non-zero `rf` are approximate.
- An entropy-pooling value-at-risk view whose coefficient is not one selected the wrong tail. A view with a target of `0` was dropped with no warning. `OptimEntropyPooling` ignored a `sc1` other than the default.
- `WeightBounds` with an infinite scalar bound, or a vector bound shorter than the universe (now an error, see the table above).
- A weighted pipeline holding `ImpliedVolatilityRegression` computed the realised volatility unweighted.
- A hyperparameter search over `"opti[1].opt.l2"`. The penalty never entered the model, and the search reported success.
- The effective-asset norm ceiling at any `p != 2`. The ceiling was weaker than asked.
- `distance_wei`, the private shortest-path function of DBHT, returned a wrong edge count for each path in `B`, its second output. DBHT uses only the path lengths, and this defect moved no cluster. Three more DBHT defects moved clusters on a tie and on a repeated call.
- A `KFold` run through the Pipeline passed the weights of the previous fold to the next fold, where a run of the optimiser alone did not. `FeesEstimator.tn` passed the turnover through unchanged, so every fold charged the turnover fee against old reference weights.
- `MultipleRandomised` with a seed draws different subsets, because it now draws from the assets listed in each window.

These results were approximate in v0.30:

- A moment estimator with observation weights (`SimpleVariance`, `Covariance`, `Coskewness`, `Cokurtosis`) now weights its centre as well as its spread.
- A co-movement matrix of the Gerber or Smyth-Broby kind has a diagonal of ones, and a return of exactly zero at a zero band edge is neutral rather than a crossing.
- `GerberIQCovariance` under `Gerber2` now reads the bound for each pair.
- The library widens a histogram edge by one ulp rather than by `eps`. So mutual information, and every caller of `calc_hist_data` on standardised data, prices or levels, moves in the last digits.
- `cor` on an implied-volatility estimator, `collapse_features` with `MedianCollapse` over a narrow element type, and the output element type of `pre_order` with a custom traversal strategy.
- A weighted `RelativisticValueatRisk`, its range, and the two relativistic drawdown measures.
- A `Fees` resolved from a `FeesEstimator` carrying a non-default `kwargs`.
- An upper-bound `GridRelativisticValueatRiskView`.
- An `EntropyPoolingPrior` over a `FactorPrior` with a `StepwiseRegression`. The nested prior now fits first, so it selects the same factors as the `FactorPrior` alone.
- Every backtest with a fixed fee, a turnover fee, or a delisting. See [Fees](@ref migration-0-31-fees).
- A walk-forward search (`GridSearchCrossValidation`, `RandomisedSearchCrossValidation`) over an optimiser with a `TimeDependent` schedule, or with a `Turnover`, tracking or fee term that reads the previous weights. The search now scores each candidate with one run of its folds in order, and no longer fits each fold on its own.
- A walk-forward in which a fold failed and a later fold read the previous weights. The later fold now reads the last solved fold's weights instead of raising an error.

These results keep their numbers, but their meaning changed:

- The `Max` and log-sum-exp scalarisers put an upper bound on `model[:risk]`, so `model[:risk]` is not the aggregate. Read the exact value with `expected_risk`.
- The order-`p` effective number of assets behind an `lpc` ceiling is `(sum_i |w_i|^p)^(1/(1 - p))`, where v0.30 used `1 / sum_i |w_i|^p`. To ask for at least `m` such assets, set `val = m^(1/p - 1)`, not `m^(-1/p)`.
- `res.fb` on an optimisation result holds the fallbacks that failed before the answer, as `(estimator, result)` pairs. In v0.30 it was always `nothing`.
- Every optimisation result has `imsk`, the assets it was allowed to trade. On a table with no gaps it is all `true`.

### [If you extend the library](@id migration-0-31-extending)

- **A prior estimator's method on a returns matrix takes the asset panel as its third positional argument.** The signature is `prior(pe, X, F = nothing, pnl = nothing; kwargs...)`. When you pass a `ReturnsResult`, the library calls this method with all three arguments. A method that takes only two arguments raises a `MethodError`. Give it a third argument, even one it ignores.
- **An uncertainty-set estimator for which `reads_prior_result` returns `true`** receives an `rd` keyword, which it can ignore.
- **`LowOrderPrior.rr`** now takes an `AbstractLoadingsRegressionResult`. A regression result that is an `AbstractRegressionResult` but not a loadings result raises a `MethodError` when you store it there.
- **`Regression`** gains a fourth field, `esigma`, the idiosyncratic variances, when the regression is fitted with `rsd = true`. Code that constructs or compares it by position must handle the extra field.
- **`SubsetResamplingResult`, `NestedClusteredResult` and `StackingResult`** have `imsk` before `fb`. Keyword construction is unaffected.
- **Weight bounds** are validated against the stated asset count with `DimensionMismatch`, so a constraint generator outside the library that relied on truncation now raises an error.

### [Links into the documentation](@id migration-0-31-docs)

The API pages follow the new source layout, and each source file has a public page under `public_api/` and a private page under `private_api/`. The build writes no page under `api/`, and a bookmark into `api/24_Plotting`, `api/25_Aliases`, `api/19_RiskMeasures` or `api/20_Optimisation` on the `stable` site finds no page. The public pages are now `public_api/22_Plotting`, `public_api/23_Aliases`, and the directories `public_api/16_RiskMeasures` and `public_api/17_Optimisation`. Find a page from the [API introduction](@ref) rather than by its number.

## From v0.31 to v0.32

v0.32 adds the online portfolio selection family. It also removes `OnlineStep`, the `ff` keyword, the `ddof` keyword of `LInfNorm` and three unexported functions. It gives the naive optimisers a `fees` field and adds fields to four results. Five calls that ran in v0.31 now raise an error, four calls that raised now answer, and the results in [Results that move](@ref migration-0-32-numbers) change.

### [Names that were removed](@id migration-0-32-removed)

| Removed | Write instead |
| --- | --- |
| `OnlineStep`, and the `ff` keyword of `IndexWalkForward` and `DateWalkForward` | The online step is a scheme wrapped in `Online`, and each scheme has its own constructor. `OnlineIndexWalkForward(train_size, test_size; …)` and `OnlineDateWalkForward(train_size, test_size; …)` take every keyword of the plain scheme except `expand_train`, which an online run sets. `IndexWalkForward(60, 1; ff = OnlineStep())` becomes `OnlineIndexWalkForward(60, 1)`. An `Online(cv)` that you write by hand on a scheme raises an error that names the three constructors. |
| The derived `expand_train` (`nothing` on the two walk-forwards) | `expand_train` is a plain `Bool` keyword again, `false` by default, as in v0.30. |
| `fold_fit`, `cv_online_info`, `cv_resume_info` (unexported) | Nothing replaces them. The fold loop reads the type of the scheme to decide whether it steps online. |
| The `ddof` keyword and field of `LInfNorm` | `LInfNorm()`. The largest single-period difference carries no degrees of freedom, so the norm is not scaled. `LInfNorm(; ddof = 1)` raises a `MethodError`. |

### [Renamed keywords and changed defaults](@id migration-0-32-renames)

- **The naive optimisers charge fees.** `EqualWeighted`, `InverseVolatility`, `RandomWeighted` and `PreviousWeights` take `fees`, and `NaiveOptimisationResult` stores it. The keyword constructor does not change. The positional constructor of `NaiveOptimisationResult` takes `fees` as its third argument, `NaiveOptimisationResult(pr, wb, fees, retcode, w, imsk, fb)`.
- **`PerformanceSummaryResult`** has four more fields, `excess_ret`, `tracking_error`, `information_ratio` and `turnover`, so its positional constructor takes sixteen arguments. `performance_summary` takes a `benchmark` and fills the first three. Without a benchmark they are `NaN`, and `turnover` is `nothing` unless the result holds the path of the weights.
- **`plot_performance_summary`** is one method over an array, an `OptimisationResult` or a prediction result, with `benchmark` as a keyword. v0.31 had six methods with different numbers of arguments. Every call written against v0.31 still works.
- **The `opt` argument of the risk-measure builders** is a `RiskConstraintOwner`, which is a JuMP optimiser or a `ProgrammeAllocationSet`. Every method keeps its argument list, and a method that you added under the old type bound still dispatches.
- **`LpNorm` defaults to `ddof = 1`**, where v0.31 used `0`. So `LpNorm(; p = 2)` equals `L2Norm()`. At `T = 252` the factor of `LpNorm()` is `251^(1/3)`, not `252^(1/3)`. State `ddof = 0` to keep the v0.31 error. `L1Norm` gains a `ddof` keyword whose default `0` keeps the v0.31 factor `T`.
- **`GridEntropicValueatRiskView` and `GridRelativisticValueatRiskView`** take `M = 1`, where v0.31 used `10`. `M` now multiplies the smallest constant that releases each row, so it must be at least `1`.
- **An upper bound over several assets on an EVaR or RLVaR view**, and an equality below the prior value, take the sequential formulation when `alg = nothing`. v0.31 sent them to the grid, which holds one asset, so the view raised an `ArgumentError`. A group lower bound stays conic.
- **`plot_histogram(…; reference = true)`** draws the pdf of the Normal fitted to the returns, and `reference = false` draws no curve. v0.31 drew a kernel density under the label "Normal" for `true`, and the Normal for `false`.
- **`plot_factor_risk_contribution`** names its last bar "Off-factor", where v0.31 named it "Constant". The factor names apply only when they count the columns of the loadings. Otherwise the bars are numbered.
- **`plot_performance_summary`** draws four more bars: the excess return, the tracking error, the information ratio and the turnover.

### [Calls that now raise](@id migration-0-32-raises)

In v0.31 each of these calls returned a value that did not follow the configuration. In v0.32 each call raises an error that names the cause and the fix.

| Call | v0.31 | v0.32 |
| --- | --- | --- |
| `cross_val_predict` on `HierarchicalRiskParity`, `HierarchicalEqualRiskContribution` or `SchurComplementHierarchicalRiskParity` whose `opt.pe` is a prior result | Every fold used the prior of the full sample, so each fold read its own test rows. Under `KFold` all five folds gave the same weights. | `ArgumentError`. Give an estimator, such as `EmpiricalPrior()`, so each fold fits its own prior. |
| `search_cross_validation` on an optimiser whose prior is a prior result, or a grid that writes one | The search scored every candidate on the prior of the full sample | `ArgumentError`, before any candidate is scored |
| A `TimeDependent` schedule whose entries or `default` hold a prior result, in the fold loop | Accepted, and every fold read the full-sample prior | `ArgumentError`. A callable schedule still runs. |
| A walk-forward on a `Frontier` sweep with a term that reads the previous weights, such as `Turnover(; w)` | Ran, and charged every fold against the stated `w`, because no one portfolio of the sweep is the previous one | `ArgumentError`. A `fixed = true` turnover still runs. |
| `GridEntropicValueatRiskView(; M)` or `GridRelativisticValueatRiskView(; M)` with `M < 1` | Accepted | `DomainError` |

### [Calls that now answer](@id migration-0-32-answers)

| Call | v0.31 | v0.32 |
| --- | --- | --- |
| A library estimator that holds a bare `StatsBase` covariance estimator, such as `Covariance(; ce = SimpleCovariance())`, inside a prior or an optimiser | `MethodError`: the bare estimator refused the library's keywords and the asset panel | Runs. A working path keeps its numbers. |
| A mask-aware moment estimator (`ExpWeightedExpectedReturns`, `ExpWeightedVariance`, `ExpWeightedCovariance` and the two regime-adjusted forms) on a transposed sample with an asset panel, `dims = 2` | `DimensionMismatch` | The same answer as `dims = 1` |
| An upper bound over several assets on an EVaR or RLVaR view, with `alg = nothing` | `ArgumentError` | The sequential formulation meets the view. |
| An upper-bound grid EVaR view at half the prior value | On a 100-row example, a `DomainError` on the posterior weights | The posterior meets the bound to `1e-10`. |

### [Results that move](@id migration-0-32-numbers)

These results were wrong in v0.31:

- An `LInfNorm` tracking error divided the largest difference by `T - ddof`. With 252 rows, `err = 2e-2` let one day differ by `5.04`, so the bound did not bind. The error is now the largest difference, and `err` is the bound on one period.
- `MonotonicSchurComplement` returned the weights of a `gamma` other than the one it reported. The difference reached `8e-6` in a weight. The bisection also returned the weights of a midpoint that it rejected.
- `SchurComplementHierarchicalRiskParity` dropped its fee. The result had no `fees` field, so `calc_net_returns` gave the gross return, and a fold charged no forced exit. The result now carries the fee.
- After `factory`, the skewness term of a `VarianceSkewKurtosis` had scale `1` and no floor, so `expected_risk` of the stored measure did not match the model. The weights did not move.
- `IntegerConditionalValueatRiskView` did not reach the posterior of least divergence. On a 60-row example, the divergence falls from `0.0189` to `0.0175` for an upper bound, and from `0.0858` to `0.0415` for an equality. The view now warns when its window of `sbar` losses binds.
- An `IterativeWeightFinaliser` that stalled or diverged reported success with weights that broke the bounds. It now returns the Euclidean projection of its input. A bound set that cannot hold the budget now fails, so the fallback chain runs.
- A walk-forward that charged a turnover fee on a naive optimiser priced the fee against the reference weights `w` of its `Turnover` on every fold, never against the weights that the fold held. Buy-and-hold paid the most, and a constant rebalanced portfolio paid nothing. The fee is now charged against the previous weights that the fold loop passes on. The first fold's fee is the trade from those reference weights.

These results were approximate in v0.31:

- A tracking error on `LpNorm()`, whose default `ddof` moved from `0` to `1`.
- The t-statistic of every information-coefficient summary treated the forward windows as independent rows. The windows overlap in three cases: `forecast_holding_period` from its second row on, `forecast_evaluation_summary` at any `step < horizon`, and `exposure_ic_summary` on a block at any `horizon > 1`. In these cases the statistic now uses a Newey-West variance at the known overlap order, so it is smaller. `ic_ir` does not move.

This result keeps its numbers, but its meaning changed:

- `PopulationPredictionResult` gives each member whose `id` is `nothing` its position, so a scorer that selects a path names it.

### [If you extend the library](@id migration-0-32-extending)

- **A risk-measure builder** `set_risk_constraints!(model, i, r, opt, pr, …)` receives `opt::RiskConstraintOwner`. A method bound to `RiskJuMPOptimisationEstimator` alone is not called for a `ProgrammeAllocationSet`.
- **`SchurComplementHierarchicalRiskParityResult`** has a `fees` field after `clr`, so its positional constructor takes one more argument. The keyword constructor needs `fees` too. Pass `fees = nothing` for a result with no fee.
- **`L1Norm`** has a field, `ddof`, and **`LInfNorm`** has none. Code that constructs either by position must follow.
