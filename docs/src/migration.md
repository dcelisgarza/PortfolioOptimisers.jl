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

v0.32 adds the online portfolio selection family. It also removes `OnlineStep`, the `ff` keyword, the `ddof` keyword of `LInfNorm`, the `dims` keyword of every call on a `ReturnsResult` or a prior result, and some unexported functions. It gives the naive optimisers a `fees` field and adds fields to some results. The rank statistics give a tie its mean rank. Some calls that ran in v0.31 now raise an error, some calls that raised now answer, and the results in [Results that move](@ref migration-0-32-numbers) change.

### [Names that were removed](@id migration-0-32-removed)

| Removed | Write instead |
| --- | --- |
| `OnlineStep`, and the `ff` keyword of `IndexWalkForward` and `DateWalkForward` | The online step is a scheme wrapped in `Online`, and each scheme has its own constructor. `OnlineIndexWalkForward(train_size, test_size; …)` and `OnlineDateWalkForward(train_size, test_size; …)` take every keyword of the plain scheme except `expand_train`, which an online run sets. `IndexWalkForward(60, 1; ff = OnlineStep())` becomes `OnlineIndexWalkForward(60, 1)`. An `Online(cv)` that you write by hand on a scheme raises an error that names the three constructors. |
| The derived `expand_train`, which was `nothing` on the two walk-forwards | `expand_train` is a plain `Bool` keyword again, `false` by default, as in v0.30. |
| `fold_fit`, `cv_online_info`, `cv_resume_info` (unexported) | Nothing replaces them. The fold loop reads the type of the scheme to decide whether it steps online. |
| The `ddof` keyword and field of `LInfNorm` | `LInfNorm()`. The largest single-period difference carries no degrees of freedom, so the norm is not scaled. `LInfNorm(; ddof = 1)` raises a `MethodError`. |
| The `dims` keyword of `optimise` on every optimiser, of `prior`, `ucs`, `mu_ucs` and `sigma_ucs` on a `ReturnsResult`, and of `clusterise`, `phylogeny_matrix`, `phylogeny_constraints`, `centrality_vector`, `average_centrality`, `asset_phylogeny`, `centrality_constraints`, `plot_dendrogram` and `plot_clusters` on a prior result | Nothing. A `ReturnsResult` and a prior result hold the observations along the rows, so `dims = 1` is the one correct value. These calls ignore a `dims` that you pass. A method that takes a matrix keeps `dims`. |
| `find_complete_indices` (unexported) | `CompleteAssetSelector` keeps the assets with no gap, and `MissingDataFilter` drops rows and columns. The function counted a column with an `Inf` as complete. |
| `assert_returns_result_dims` (unexported) | Nothing replaces it, because no call takes `dims` on a `ReturnsResult`. |
| `has_X`, `get_X`, `has_Xap1`, `get_Xap1`, `has_ddap1`, `get_ddap1`, `has_dd`, `get_dd` and `set_asset_neg_returns_plus_one!` (unexported) | Keep the value that the builder of the entry returns. For the negated returns, pass `-X` to `set_asset_returns_plus_one!` under a prefix of your own. No code in the library called these functions. |

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
- **The rank statistics give a tie its mean rank.** `forecast_evaluation`, `exposure_ic`, `exposure_ic_summary`, `idio_vol_ic` and `idio_vol_residual_dependence` take `ties = :average` by default, and `forecast_ic`, `forecast_factor_correlation` and the `:rank` book of `forecast_portfolio` read the `ties` of the evaluation. v0.31 ranked a tie by the order of the assets, so a constant forecast scored a Spearman coefficient of `1` or `-1`. State `ties = :ordinal` to keep the v0.31 numbers.
- **`covariance_forecast_compare`** caps its default `lags` at the number of steps less one. So a run with fewer steps than the horizon no longer refuses its own default.
- **A finite allocation below a unit budget** splits the cash by the new collateral algorithm, `ProceedsCollateral` by default. See [Results that move](@ref migration-0-32-numbers).
- **The keyword `N` of the plots that keep the largest assets** counts the effective number of assets from the shares of the magnitudes. So the count no longer changes when you scale the weights.

### [Calls that now raise](@id migration-0-32-raises)

In v0.31 each of these calls returned a value that did not follow the configuration, or raised an error that did not name the cause. In v0.32 each call raises an error that names the cause.

| Call | v0.31 | v0.32 |
| --- | --- | --- |
| `cross_val_predict` on `HierarchicalRiskParity`, `HierarchicalEqualRiskContribution` or `SchurComplementHierarchicalRiskParity` whose `opt.pe` is a prior result | Every fold used the prior of the full sample, so each fold read its own test rows. Under `KFold` all five folds gave the same weights. | `ArgumentError`. Give an estimator, such as `EmpiricalPrior()`, so each fold fits its own prior. |
| `search_cross_validation` on an optimiser whose prior is a prior result, or a grid that writes one | The search scored every candidate on the prior of the full sample | `ArgumentError`, before any candidate is scored |
| A `TimeDependent` schedule whose entries or `default` hold a prior result, in the fold loop | Accepted, and every fold read the full-sample prior | `ArgumentError`. A callable schedule still runs. |
| A walk-forward on a `Frontier` sweep with a term that reads the previous weights, such as `Turnover(; w)` | Ran, and charged every fold against the stated `w`, because no one portfolio of the sweep is the previous one | `ArgumentError`. A `fixed = true` turnover still runs. |
| `GridEntropicValueatRiskView(; M)` or `GridRelativisticValueatRiskView(; M)` with `M < 1` | Accepted | `DomainError` |
| `ExpWeightedExpectedReturns`, `ExpWeightedVariance`, `ExpWeightedCovariance`, `RegimeAdjustedExpWeightedVariance` or `RegimeAdjustedExpWeightedCovariance` with a `decay` outside `(0, 1)` | `decay > 1` was accepted, and the estimate could be negative. On one example, `decay = 1.5` gave the variances `-1.5e12` and `-1.24e12`. `decay = 1` raised `InexactError: Int64(Inf)`, and a negative `decay` raised a `DomainError` of `log2` that did not name `decay`. | `DomainError` that names `decay` |
| `Fees` with a `tn`, `lq` or `flq` whose `val` is not finite | Accepted. `Inf` gave a model with an infinite coefficient and `NaN` weights. | `DomainError` that names the field |
| `GridSearchCrossValidation` or `RandomisedSearchCrossValidation` with an empty vector of values, such as `"opt.l1" => Float64[]` | The search fitted nothing, and it raised an error that said that every candidate failed | `IsEmptyError` that names the key, before the search builds the grid |
| `AssetPanel` whose fields derive one column name of the feature matrix twice, such as `"beta=size"` beside a tensor field `beta` with the label `size` | `panel_feature_matrix` gave the name twice | `ArgumentError` that names the column and both fields |
| `ReturnsResult(; X, nb)` with no `B` | Accepted | `IsNothingError` that names `nb` and `B` |
| `UncertaintySetVariance` with a `sigma` that is not square | Accepted, and a box set ran | `DimensionMismatch`, as `Variance` gives |
| `DistributionallyRobustConditionalValueatRisk(; w)` with a negative or non-finite weight | Accepted | `DomainError` |
| `prior(CrossSectionalFactorPrior(…), rd)` whose regression fits an intercept | Accepted. `mu`, `sigma` and the scenarios dropped the share of the intercept. | `ArgumentError` that names the remedy |
| `cross_sectional_factor_sets` with a factor label that starts with a key prefix of the universe, or that equals its `nikey` | A label such as `"ni"` became an axis. Other labels gave an unrelated `DimensionMismatch` or `KeyError`. | `ArgumentError` that names the label |
| `IntegerPhylogeny` in the `frc_ple` of `FactorRiskContribution` when the asset bounds are `nothing` or not finite | The phylogeny had no effect | `ArgumentError`. With finite bounds the phylogeny now binds the factor weights, and it needs a MIP solver. |
| `ResidualInflation` with no `dof` on a hand-built `Regression` that records no `edof` | The radius took a count of degrees of freedom from the type of the block | `IsNothingError` |
| `MissingDataFilter` applied to a window that lacks a fitted asset | The filter dropped the asset with no error | `ArgumentError` |
| `merge_states` of two returns buffers with different caps | Merged to the smaller cap | `ArgumentError` |
| `PriceIngestion(; join_method)` with `:right` or a misspelt symbol | Accepted. `:right` let a covariate set the clock. | `ArgumentError` |
| `panel_dataframe` on a panel whose field name or asset name equals a column that the table writes, such as `"asset"` or `"observation"` | The table replaced the column with no error | `ArgumentError` that names the column |
| A `Pipeline` with a capped `Online` prior two levels down, such as `opt.pe.pe`, after a `PriceGapFill` step | Ran, although the same prior one level down raised an error | `ArgumentError` |
| `partial_fit!` on a `Pipeline` whose `Online` step no warm-up resolved | The step dropped the wrapper and its cap. `Online(EmpiricalPrior(); max_history = 30)` folded 80 rows. | `ArgumentError` |
| `plot_rolling_drawdowns` with a window longer than the series | Drew an empty plot | `DomainError` |
| `RollingLogReturn`, `Reversal`, `RollingMax` or `MaxReturn` on returns that hold an infinite value | Accepted. `RollingLogReturn` gave `NaN` in every later window, also in a window that does not hold the infinite return, and `RollingMax` gave `Inf`. | `DomainError` that names the observation and the asset |
| `EWResidualVolatility(; beta_half_life = Inf)` | `DomainError` that named `half_life` | `DomainError` that names `beta_half_life` |
| `factor_family_basis` when the dropped member of a family has a zero benchmark-weighted exposure | `DivideError` on `Rational` data, and on floating-point data an `IsNonFiniteError` that did not name the member | `IsNonFiniteError` that names the member, the family and the observation |
| `PredictionReturnsResult(; iv)` with no `X` | `MethodError` | `IsNothingError` that names `X` and `iv` |

### [Calls that now answer](@id migration-0-32-answers)

| Call | v0.31 | v0.32 |
| --- | --- | --- |
| A library estimator that holds a bare `StatsBase` covariance estimator, such as `Covariance(; ce = SimpleCovariance())`, inside a prior or an optimiser | `MethodError`, because the bare estimator refused the library's keywords and the asset panel | Runs. A working path keeps its numbers. |
| `ExpWeightedExpectedReturns`, `ExpWeightedVariance`, `ExpWeightedCovariance` or one of the two regime-adjusted forms on a transposed sample with an asset panel, `dims = 2` | `DimensionMismatch` | The same answer as `dims = 1` |
| An upper bound over several assets on an EVaR or RLVaR view, with `alg = nothing` | `ArgumentError` | The sequential formulation meets the view. |
| An upper-bound grid EVaR view at half the prior value | On a 100-row example, a `DomainError` on the posterior weights | The posterior meets the bound to `1e-10`. |
| `optimise(opt, rd; dims = 2)` on `EqualWeighted`, `RandomWeighted` or a hierarchical optimiser, and `plot_dendrogram` or `plot_clusters` on a prior result with `dims = 2` | `ConflictingArgumentError`, `DimensionMismatch` or `BoundsError` | The call ignores `dims`, and it gives the answer of `dims = 1`. |
| Integer returns in `HierarchicalRiskParity`, `HierarchicalEqualRiskContribution`, `SchurComplementHierarchicalRiskParity`, `InverseVolatility`, `EqualWeighted`, `RandomWeighted`, `PreviousWeights`, the three exponentially weighted moment estimators, `RegimeAdjustedExpWeightedVariance`, `RegimeAdjustedExpWeightedCovariance`, `phylogeny_features`, `factor_model_summary`, the exposure, regression and idiosyncratic diagnostics, `CrossSectionalFactorPrior`, `factor_family_basis` and `OrthogonalUncertaintySet`. In `descriptor`: the rolling Descriptors, such as `RollingLogReturn` and `RollingMax`, `EWMean` and the named Descriptors that it builds, such as `EWMomentum`, `EWMarketBeta` with `agg_obs > 1`, and `EWMacroSensitivity`, also with an integer `ref` | `InexactError`, or `MethodError` for `eps(::Type{Int64})` | The answer of the `Float64` copy of the data. `Float32` data stays `Float32`. |
| An optimiser whose fallback `fb` is a precomputed result, such as `fb = optimise(EqualWeighted(), rd)`, when the first solve fails | `MethodError` | The fallback returns that result. A failed precomputed result ends the chain. |
| `MonotonicSchurComplement` on a bracket narrower than about `0.18 * tol` | The search did not iterate. It warned that it did not converge, or it raised under `strict = true`. | The search stops when the bracket is at most `tol` wide. |
| `feature_matrix(…, rows)` with a `Bool` mask | `DimensionMismatch` | The stack holds the rows where the mask is `true`. |
| The compact covariance uncertainty set on a singular covariance | `PosDefException` | Runs, as `Variance` does |
| `mean(StandardDeviationExpectedReturns(SimpleCovariance()), X)`, the same for `VarianceExpectedReturns`, and either one inside `EmpiricalPrior(; me = …)` | `MethodError` | Runs |
| `covariance_forecast_evaluation` of a `PortfolioOptimisersCovariance` or `CorrelationCovariance` that holds a `StatsBase` estimator | `MethodError` | Runs |
| `covariance_forecast_summary` on a one-step evaluation, or on a step with a `NaN` standardised return | `ArgumentError` about quantiles in the presence of `NaN` | The bias columns are `NaN`. |
| `FactorRiskContribution` with an estimator in `frc_ple`, when `optimise` receives a keyword that the optimiser does not read | `MethodError` | Runs |
| `FactorRiskContribution` with a vector `frc_ple` | The model solved, then a `TypeError` stopped the result | Runs, and `frc_plr` holds the vector |
| An `sgcarde` whose column count equals the row count of `sgmtx` when the row count does not, or on an optimiser where an asset is not investable | `DimensionMismatch` | Runs. An asset that departed removes a column of `sgmtx`, not a sub-group. |
| A vector `sglt` or `sgst` with no `card` or `gcard` | `MethodError` | Runs |
| `NearestQuantilePrediction`, `sort_by_measure`, `quantile_by_measure`, `expected_risk(r, ppred)` or `rolling_window_measure(r, ppred, window)` on a `PopulationPredictionResult` of single-fold `PredictionResult`s | `FieldError` or `MethodError` | Runs. Each member takes its own method. |
| `train_test_split` or `TrainTestSplit` with a `Rational` fraction, such as `train_size = 1 // 5` | `MethodError` | Runs |
| `MissingDataFilter` with a vector `ivpa` and no `iv`, when the filter drops a column | `DimensionMismatch` | Runs |
| `merge_states` of the empty sample buffer that an `Online` wrapper seeds with a folded buffer, and `port_opt_view` of an empty buffer | `DimensionMismatch` or `BoundsError` | A copy of the folded buffer, and a view |
| `price_ingestion` on a table of `Union{Missing, Rational{Int}}` that holds no `missing` | `DomainError` | Runs |
| A `price_ingestion` collapse whose timestamp function makes new timestamps, on a time-varying panel | `ArgumentError` | Runs |
| `linear_constraints` with a group that sheds every member while the other side is not empty | `BoundsError` | Runs |
| `predict` of a `Frontier` optimisation on a `ReturnsResult` with `iv` and no `ivpa`, and `PredictionReturnsResult` with one `iv` vector for each member and no `ivpa` | `MethodError` | Runs. The `iv` of each member is `iv * (abs.(w) / sum(abs, w))`. |
| `CrossSectionalFactorPrior(; neutralise)` with a `Symbol` target, such as `"size" => :value` | The constructor accepted it, and the fit raised a `MethodError` | The answer of the `String` target |
| `plot_factor_mu`, `plot_factor_sigma`, `plot_factor_forecast_correlation` and `plot_factor_forecast_volatilities` on a `HighOrderPrior` over a `FactorPrior` | `FieldError` | Runs |
| `plot_composition` of a prediction with a masked fold, `plot_dendrogram` and `plot_clusters` on a prior with gaps, `plot_measures` on a `PredictionResult`, and `plot_asset_cumulative_returns(…; N = 0.5)` | An error. A masked fold gave a `BoundsError`, a prior with gaps an `IsNonFiniteError`, and a fractional `N` a `TypeError`. | Runs |

### [Results that move](@id migration-0-32-numbers)

These results were wrong in v0.31:

- An `LInfNorm` tracking error divided the largest difference by `T - ddof`. With 252 rows, `err = 2e-2` let one day differ by `5.04`, so the bound did not bind. The error is now the largest difference, and `err` is the bound on one period.
- `MonotonicSchurComplement` returned the weights of a `gamma` other than the one it reported. The difference reached `8e-6` in a weight. The bisection also returned the weights of a midpoint that it rejected.
- `SchurComplementHierarchicalRiskParity` dropped its fee. The result had no `fees` field, so `calc_net_returns` gave the gross return, and a fold charged no forced exit. The result now carries the fee.
- After `factory`, the skewness term of a `VarianceSkewKurtosis` had scale `1` and no floor, so `expected_risk` of the stored measure did not match the model. The weights did not move.
- `IntegerConditionalValueatRiskView` did not reach the posterior of least divergence. On a 60-row example, the divergence falls from `0.0189` to `0.0175` for an upper bound, and from `0.0858` to `0.0415` for an equality. The view now warns when its window of `sbar` losses binds.
- An `IterativeWeightFinaliser` that stalled or diverged reported success with weights that broke the bounds. It now returns the Euclidean projection of its input. A bound set that cannot hold the budget now fails, so the fallback chain runs.
- A walk-forward that charged a turnover fee on a naive optimiser priced the fee against the reference weights `w` of its `Turnover` on every fold, never against the weights that the fold held. Buy-and-hold paid the most, and a constant rebalanced portfolio paid nothing. The fee is now charged against the previous weights that the fold loop passes on. The first fold's fee is the trade from those reference weights.
- `optimise(opt, rd; dims = 2)` on `MeanRisk`, `InverseVolatility` or another optimiser that fits a prior fitted the prior on the transpose, and gave one weight per observation. `prior(pe, rd; dims = 2)` and the clustering calls on a prior result also read the transpose. Now `dims` has no effect.
- Under `MaximumRatio`, the squared L2 penalty of `RSOCRiskExpr`, `SquaredSOCRiskExpr` and `QuadRiskExpr` scaled as `k^2 ‖w‖^2`, so its strength moved with `ohf`. The weights moved by `0.25` between `ohf` and `10 ohf`. The penalty is now `k ‖w‖^2`, of degree one, as every other penalty is.
- With a fee, a moment risk measure whose target is per asset centred the net returns on the gross mean, so each deviation carried the mean fee. A target is per asset when you state a `mu` vector, or when `factory` fills the prior mean. The target now subtracts the mean fee of each period, in the model and in the functor. This covers the low- and high-order moments, `Kurtosis`, `MedianAbsoluteDeviation`, the third central moment and the skewness.
- `LowOrderMoment(; w, alg = SecondMoment())`, `HighOrderMoment` with a standardised algorithm and `Skewness(; w)` did not pass `w` to their variance estimator. So the functor ignored the weights in the variance. On one example it gave `2.7574`, and the optimiser gave `2.9726`.
- `expected_risk` of a `RiskTrackingRiskMeasure` with `IndependentVariableTracking` and a fee charged the fee of `w - wb`, which is zero at `wb`, while the model charged the fee of `w`. With `Fees(; l = 0.002)` and a CVaR at `w = wb`, the functor gave about `0` and the model `0.002`. The functor now reads the series of the model. The weights do not move.
- A `SemiDefinitePhylogeny` dropped its penalty `p tr(W)` when any variance was in the model. It now drops the penalty only when the objective minimises that variance: `MinimumRisk`, `MaximumUtility` with `l > 0`, and the return form of `MaximumRatio`. A variance ceiling, `MaximumReturn`, `MaximumUtility` with `l = 0` and the risk form of `MaximumRatio` keep the penalty, so their weights move. The same rule now holds for a `SemiDefinitePhylogeny` in the `frc_ple` of `FactorRiskContribution`, which v0.31 always penalised.
- A `RiskTrackingRiskMeasure` with `DependentVariableTracking` of a semidefinite `Variance`, and a dependent `RiskTrackingError`, built the tracking variance on a second lifted matrix, which the rows of a `SemiDefinitePhylogeny` did not bind. The tracking variance now reads the lifted matrix of the optimiser.
- An `IntegerPhylogeny` in the `frc_ple` of `FactorRiskContribution` had no effect. It now binds the factor weights.
- Under `FactorRiskContribution(; flag = true)`, the variance read the factor part `b1 * w1` alone and omitted the off-factor weights. On one example the model value was 43 % below the variance of the returned weights. The variance now covers the whole decision vector.
- Under `RelaxedRiskBudgeting` with `FactorRiskBudgeting(; flag = true)`, the relaxed risk priced `b1 * w1` alone. On one example `psi` was `0.219`, and the standard deviation of the weights was `1.88`. The marginal risks now read the asset weights.
- `RelaxedRiskBudgeting` with a `RiskBudget` whose values do not sum to one did not meet the budget. `RiskBudget(; val = 20:-1:1)` gave contribution ratios from `0.19` to `1.64` against a target of `20`. The cones now read `val / sum(val)`.
- Under `MaximumRatio`, an upper bound on the compact covariance uncertainty set bounded the worst-case variance by `ub / k`, so a bound that `MaximumReturn` meets made the model infeasible. The set now bounds the square root by `sqrt(ub)`.
- When `smtx` and `sgmtx` were the same matrix, the model ignored `sglt` and `sgst`. It now keeps them unless they are the same objects as `slt` and `sst`.
- `risk_contribution` of a vector of risk measures under `MaxScalariser` or `MinScalariser` took the wrong measure when the measures had different degrees. The chain-rule weights now come from the scaled risks, and the log-sum-exp contributions move too. `SumScalariser` does not move.
- The Laplace z-score of a parametric `ValueatRiskRange` was wrong. At `beta = 0.05` it was `-0.454` where the quantile of the unit-variance Laplace is `-1.628`, so the range was about `2.08` standard deviations in place of `3.26`. The lower tail at `alpha > 1/2` was also wrong.
- The empirical `ValueatRisk`, `ValueatRiskRange`, `DrawdownatRisk` and `RelativeDrawdownatRisk` took the order statistic `ceil(alpha T)`. When `alpha T` rounds above an integer, as `0.07 × 100` does, the functor returned the 8th smallest return and the MIP model the 7th. The functor now takes the order statistic of the model.
- `HierarchicalEqualRiskContribution` chose the nodes above the cut by their height. When two heights tied at the cut, every weight was `NaN`, and the result was a failure. It now takes the nodes that `cutree` leaves. Under `LogSumExpScalariser` the shares inside a cluster did not sum to one, and at `gamma = 1e-3` the weights moved by up to `5.6e-3`.
- DBHT clustering turned `Inf * 0` into `NaN` where two vertices have no path, and then chose the `NaN`. The bubble assignment and the tree move on such a network.
- `ExpWeightedCovariance`, and `RegimeAdjustedExpWeightedCovariance` with one decay, let the variance of an asset decay on a holiday of another asset while their covariance stayed. So the estimate was not positive semidefinite. Two equal series, one of them with five holidays, gave an implied correlation of `2.37` under `ExpWeightedCovariance` and `2.44` under the regime-adjusted form. Each asset now ages its observations on its own clock, and a holiday holds the correlation of each pair that contains its asset. A sample with no holiday keeps its numbers.
- `RegimeAdjustedExpWeightedCovariance` with a separate `cor_decay` divided each pair by its own count of common observations. With holidays, the smallest eigenvalue reached `-0.1075` times the largest entry. A holiday now holds the correlation of each pair that contains its asset. A sample with no holiday keeps its numbers.
- `IdiosyncraticVarianceScaling` read only the diagonal of a full idiosyncratic covariance, which a `CrossSectionalFactorPrior` with `th != 0` writes. On a six-asset example with a correlated `esigma`, the scaling missed the full covariance by `0.0032`. It now reads the whole matrix.
- A `PreviousWeights` fallback kept its whole `w` under an asset subset. So a cluster of `NestedClustered`, or the asset subset of `MultipleRandomised` or `SubsetResampling`, whose solve failed gave weights of the wrong length. A view now slices `w`, `fees` and `fb`, and it does not rescale the slice.
- `DiscreteAllocation` had three defects. It multiplied the weights of a side by the cash of the side, which already held the budget of the side. So when a side budget `b` is not `1`, the targets summed to `b^2` times the cash. The two relative formulations bounded `x C / (w p) - 1`, where `x` is the count of shares, `p` the price, `w` the weight and `C` the cash. The correct relative error is `x p / (w C) - 1`. The relative objective added a unitless error to the cash. On `w = [0.4, 0.4]`, `p = [10, 100]` and `cash = 1250`, it bought `[40, 4]` where the exact book is `[50, 5]`.
- Below a unit budget, the long side of a finite allocation took the cash that the short side did not use. So a book could spend more than its cash. On `[1.2, -0.5]` a book spent `170` of a cash of `100`, and a market-neutral book became all long. When the short side paid a fee, the long side also received an excess of twice that fee. The default `ProceedsCollateral` now bounds the net money, and `CashCollateral` bounds the gross money.
- `GreedyAllocation` bought an asset with a zero target weight when its first pass bought nothing, and a side whose targets are all zero gave `NaN` shares and cash.
- A fixed fee on a MIP optimiser charged the binary variable, which does not scale with `k`, so under `MaximumRatio` the weights depended on `ohf`. The fee now charges the binary times `k`. A fee with fixed terms alone did not reach the net returns, so a risk measure on the net returns did not see it.
- `prices_to_returns` let the gap-return algorithm of the caller overwrite the `-1`, `-Inf` and `Inf` returns next to an observed zero price. Those cells read two observed prices, and the algorithm no longer writes them. `CatchUpGapReturn` writes the same values as before.
- `prices_to_returns(…; padding = true)` on a view of a `ListingSpan` made the first return row active for an asset listed before the window, with a `NaN` return. The row is now inactive.
- `PriceGapFill` with `CarriedPrice` carried a price across an absence in the span of the caller, and across an absence at the end of the training window. So the return after the gap was an active move across the absence. An absence now clears the carry.
- A `price_ingestion` collapse such as `(Dates.week, first, last)` took the prices of each period from its last row, but the values of the panel fields from its first row. Each period now takes the values of its last row.
- `MissingDataFilter` dropped a row whose share of missing assets equals `row_thr`, and a `Float32` `col_thr` dropped a column at its threshold. The apply step kept the column order of the window, not the fitted order.
- `train_test_split` and `TrainTestSplit` with a fraction lost a row to round-off. For example, `train_size = 0.29` on 100 rows gave 28 rows. The count is now the largest `n` with `n / N <= s`.
- `merge_states` of two capped returns buffers trimmed `X` to `max_history`, but not `ts` or `B`.
- `panel_dataframe` shared memory with the panel in the static long layout, so a write to the table changed the panel. It now copies.
- Under `ResetCoverage`, the semi-covariance, the coskewness and the cokurtosis gave the answer of `DecayCoverage`. They now drop the history before the last delisting.
- The online answer of a `HighOrderPriorEstimator` refitted a co-moment that does not fold over the capped or zero-filled `pr.X`. Under a cap of 25 scenarios on 80 rows, the coskewness missed the batch fit by 24 %. It now refits over the rows that the embedded prior folded.
- `linear_constraints` read only the first two factors of a product: `"2*3*x <= 1"` lost `x`, and `"2*x*3 <= 1"` gave `2x`. `UniformValues` under `datatype = Rational{Int}` gave `6004799503160661//18014398509481984` for `1//3`.
- `covariance_forecast_evaluation` of a prior centred the test rows on the `mu` that the prior publishes. For an `EmpiricalPrior` whose `me` shrinks the mean, this read a bias that the forecast did not have. The centre is now the centre of the `sigma` of the prior. The default pair does not move.
- `covariance_forecast_evaluation` of a `PortfolioOptimisersCovariance` that holds a mask-aware estimator dropped an asset that listed after the start of the sample. Under an online cross-validation scheme with `store_forecasts = true`, every stored location was the vector of the last fold, so `covariance_forecast_portfolio` of such a run was off by up to `0.76`.
- `performance_summary` of a constant return series gave a `sharpe` of about `2e16`, and a constant excess series gave an `information_ratio` of `2e16`. Each is now `NaN`, as the docstring states.
- `forecast_portfolio` under `:zscore` turned a flat cross-section into a short book of 200 % gross. The row now stays zero. The series summary gave a constant series an information ratio of about `2e16`, which is now `NaN`.
- `forecast_coverage` and its summary counted an asset with an infinite weight, which the Pearson column drops. They now admit `0 < u < Inf`.
- `factor_model_summary` of a constant factor series gave a Sharpe ratio up to `1.3e17` and an autocorrelation of `±1`. The exposure diagnostics counted a zero weight toward a pair, and one `NaN` weight made an observation `NaN`. A constant cross-section gave an exposure correlation of `0.0` and a `cs_regression_r2` up to `8.1e31`. Each of these is now `NaN`.
- `idio_skewness` and `idio_kurtosis` of a constant cross-section gave a finite value up to `2.45`, where the docstring states `NaN`. `idio_tail_rate` counted an infinite entry in the numerator and not in the denominator, so `[Inf 0.5 4 NaN -5]` gave `1.0` where the rate is `2/3`.
- A `PanelFieldRatio` whose `num` or `den` sums several Panel Fields took the number type of the first term. So a `Float32` first term rounded a `Float64` term. With `num = ["f32" => 1, "f64" => 1]`, a `Float64` term of `1e-6` became `9.54e-7`, and the reverse order gave `1e-6`. The sum now takes the type that every term promotes to.

These results were approximate in v0.31:

- A tracking error on `LpNorm()`, whose default `ddof` moved from `0` to `1`.
- The t-statistic of every information-coefficient summary treated the forward windows as independent rows. The windows overlap in three cases: `forecast_holding_period` from its second row on, `forecast_evaluation_summary` at any `step < horizon`, and `exposure_ic_summary` on a block at any `horizon > 1`. In these cases the statistic now uses a Newey-West variance at the known overlap order, so it is smaller. `ic_ir` does not move.
- `MIPValueatRisk`, `DrawdownatRisk` and `ValueatRiskRange` with `b = nothing` used a big-M constant of `1000`. A solver integrality tolerance of `1e-6` then loosened a row by about `1e-3`, so the model risk and the measure disagreed. The builder now derives the smallest exact constant from the weight bounds and the spread of the losses. It keeps `1000` for a free weight scale, a tracking shift, fees or a `NaN` in `X`. On one example, a `NestedClustered` VaR portfolio falls from `0.013824` to `0.013200`.
- The `ResidualInflation` radius took one count of degrees of freedom from the type of the block, so its level fell to `0.904` at `T = 30`, `K = 3` and to `0.769` at `K = 8`, against `0.95`. The fit now records the count of each asset, and the level is `0.942` and `0.946`.
- The three exponentially weighted estimators divided the cold-start correction by `max(1 - decay^n, eps)`. The floor is gone. It changed an answer only where `1 - decay < eps`.
- A bin edge of `forecast_calibration` that falls on an order statistic could land one ulp above it, and move a pair to the bin below.
- Under Optim 2.3 and later, the entropy-pooling solve takes `Fminbox(; mu0 = 1e-5)`, as it does under Optim 2.0.1 to 2.2. v0.31 took the default `mu0` of Optim there, which can give a `NaN`.

This result keeps its numbers, but its meaning changed:

- `PopulationPredictionResult` gives each member whose `id` is `nothing` its position, so a scorer that selects a path names it.

These results keep their numbers, but their number type changed:

- `EqualWeighted` on `Float32` returns gives `Float32` weights. v0.31 gave `Float64`.
- `factor_attribution` and the realised attribution family take the number type of the data. v0.31 gave `Float64` volatilities and errors for `Float32` data.
- `forecast_evaluation_summary(fe; quantiles = 0.2)` stores `quantiles` as a vector of one. v0.31 stored a 0-dimensional array.

### [If you extend the library](@id migration-0-32-extending)

- **A risk-measure builder** `set_risk_constraints!(model, i, r, opt, pr, …)` receives `opt::RiskConstraintOwner`. A method bound to `RiskJuMPOptimisationEstimator` alone is not called for a `ProgrammeAllocationSet`.
- **`SchurComplementHierarchicalRiskParityResult`** has a `fees` field after `clr`, so its positional constructor takes one more argument. The keyword constructor needs `fees` too. Pass `fees = nothing` for a result with no fee.
- **`L1Norm`** has a field, `ddof`, and **`LInfNorm`** has none. Code that constructs either by position must follow.
- **`ForecastEvaluationResult`** has a `ties` field after `min_count`, so its positional constructor takes one more argument.
- **`Regression`** and **`CrossSectionalFactorModel`** have the fields `edof` and `ediv`, so their positional constructors take two more arguments.
- **`FactorRiskBudgeting`** has a `hedge` field, `false` by default, so its positional constructor takes one more argument.
- **`FiniteAllocationInput`** has a field `ca`, the collateral algorithm, so its positional constructor takes one more argument.
