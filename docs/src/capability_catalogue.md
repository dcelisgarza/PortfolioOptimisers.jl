# [Capability catalogue](@id capability-catalogue)

Everything `PortfolioOptimisers.jl` can do, grouped by the job it does rather
than by the file it lives in. Each entry links to its docstring.

This page is generated (see
[docs/generate_capability_catalogue.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/docs/generate_capability_catalogue.jl)):
the grouping is curated in `docs/capability_catalogue.jl`, and every description
is the first sentence of the corresponding docstring, so the two can never
disagree. A test asserts that every estimator and algorithm in the package
appears here, so the page cannot fall behind the code.

For the same types arranged by subtyping rather than by capability, see the
[type hierarchy](@ref type-hierarchy-AbstractEstimator).

## Core abstractions

Every component is an **Estimator** (a configuration encoding a method and its hyperparameters), an **Algorithm** (a behaviour selector consumed through an Estimator), or a **Result** (computed output). Estimators and Algorithms are what you choose; Results are what you get back.

Because every struct is immutable, runtime values are propagated down a composed estimator tree by rebuilding it.

- No-op factory function for constructing objects with a uniform interface. [`factory`](@ref)

## Preprocessing

- Convert `TimeSeries.TimeArray` price data to returns. [`prices_to_returns`](@ref) and [`ReturnsResult`](@ref)
- A container for aligned, time-indexed price-level data. [`PricesResult`](@ref)
- ::: details Preprocessing estimator converting price-level data into returns-level data. [`PricesToReturns`](@ref), [`fit_preprocessing`](@ref), and [`apply_preprocessing`](@ref)
  - Preprocessing estimator dropping assets and observations with excessive missing data from price-level data. [`MissingDataFilter`](@ref) and [`MissingDataFilterResult`](@ref)
  - Preprocessing estimator imputing missing price observations from per-asset statistics fitted on the training window. [`Imputer`](@ref) and [`ImputerResult`](@ref)
- ::: details Asset selection
  - Asset selector that scores every asset with a risk measure and keeps the assets a rule admits. [`ScoreSelector`](@ref), [`ZeroVarianceFilter`](@ref), and [`CompleteAssetSelector`](@ref)
  - Asset selector that discards assets which duplicate information already carried by others. [`RedundancySelector`](@ref)
  - Fitted result of any [`AbstractAssetSelector`](@ref). [`AssetSelectorResult`](@ref)
  - ::: details Selection rules
    - [`RankRule`](@ref) with the tail sizes given as *fractions* of the asset universe. [`QuantileRule`](@ref)
    - Take `best` and/or `worst` assets from the tails of the score ordering, then keep or drop them. [`RankRule`](@ref)
    - Keep assets whose score falls strictly inside the band `(lo, hi)`. [`ThresholdRule`](@ref)
- Cut price- or returns-level data into a training window (the head) and a held-out test window (the tail). [`train_test_split`](@ref), [`TrainTestSplit`](@ref), and [`TrainTestSplitResult`](@ref)
- Return a `ReturnsResult` appropriate for benchmark-tracking optimisations. [`returns_result_picker`](@ref)

## Matrix processing

- Projects a matrix to the nearest positive definite matrix, typically used for co-moment matrices. [`Posdef`](@ref), [`posdef!`](@ref), and [`posdef`](@ref)
- ::: details Configures and applies denoising algorithms to covariance or correlation matrices. [`Denoise`](@ref), [`denoise!`](@ref), and [`denoise`](@ref)
  - Denoises by setting the smallest `num_factors` eigenvalues to zero. [`SpectralDenoise`](@ref)
  - Denoises by replacing the smallest `num_factors` eigenvalues with their average. [`FixedDenoise`](@ref)
  - Denoises by shrinking the smallest `num_factors` eigenvalues towards the diagonal. [`ShrunkDenoise`](@ref)
- Removes the largest `n` principal components (market modes) from a covariance or correlation matrix. [`Detone`](@ref), [`detone!`](@ref), and [`detone`](@ref)
- Configures and applies matrix processing routines. [`MatrixProcessing`](@ref), [`matrix_processing!`](@ref), [`matrix_processing_step!`](@ref), and [`matrix_processing`](@ref)

## Regression models

Factor prior models and implied volatility use [`regression`](@ref) in their estimation, which return a [`Regression`](@ref) object.

### Regression targets

- Regression target type for standard linear models. [`LinearModel`](@ref)
- Regression target type for generalised linear models (GLMs). [`GeneralisedLinearModel`](@ref)

### Regression types

- ::: details Estimator for stepwise regression-based moment estimation. [`StepwiseRegression`](@ref)
  - ::: details Algorithms
    - Stepwise regression algorithm: forward selection. [`ForwardSelection`](@ref)
    - Stepwise regression algorithm: backward elimination. [`BackwardElimination`](@ref)
  - ::: details Selection criteria
    - Stepwise regression criterion based on p-value thresholding. [`PValue`](@ref)
    - Akaike Information Criterion (AIC) for stepwise regression. [`AIC`](@ref)
    - Corrected Akaike Information Criterion (AICC) for stepwise regression. [`AICC`](@ref)
    - Bayesian Information Criterion (BIC) for stepwise regression. [`BIC`](@ref)
    - Coefficient of determination (R²) for stepwise regression. [`RSquared`](@ref)
    - Adjusted coefficient of determination (Adjusted R²) for stepwise regression. [`AdjustedRSquared`](@ref)
- ::: details Estimator for dimension reduction regression-based moment estimation. [`DimensionReductionRegression`](@ref)
  - Principal Component Analysis (PCA) dimension reduction target. [`PCA`](@ref)
  - Probabilistic Principal Component Analysis (PPCA) dimension reduction target. [`PPCA`](@ref)

## Moment estimation

### Expected returns

Overloads `Statistics.mean`.

- A simple expected returns estimator for `PortfolioOptimisers.jl`, representing the sample mean with optional observation weights. [`SimpleExpectedReturns`](@ref)
- Container type for equilibrium expected returns estimators. [`EquilibriumExpectedReturns`](@ref)
- Container type for excess expected returns estimators. [`ExcessExpectedReturns`](@ref)
- ::: details Container type for shrinkage-based expected returns estimators. [`ShrunkExpectedReturns`](@ref)
  - ::: details Algorithms
    - James-Stein [`JamesStein`](@ref)
    - Bayes-Stein [`BayesStein`](@ref)
    - Bodnar-Okhrin-Parolya [`BodnarOkhrinParolya`](@ref)
  - ::: details Targets: all algorithms can have any of the following targets
    - Grand Mean [`GrandMean`](@ref)
    - Volatility Weighted [`VolatilityWeighted`](@ref)
    - Mean Squared Error [`MeanSquaredError`](@ref)
- Expected returns estimator that returns the asset standard deviations. [`StandardDeviationExpectedReturns`](@ref)
- Expected returns estimator that returns the asset variances. [`VarianceExpectedReturns`](@ref)
- Expected returns estimator that returns the optionally weighted asset medians. [`MedianExpectedReturns`](@ref)
- Expected returns estimator that returns custom values for each asset. [`CustomValueExpectedReturns`](@ref)
- Expected returns estimator that restricts computation to a rolling or indexed observation window. [`WindowedExpectedReturns`](@ref)

### Variance and standard deviation

Overloads `Statistics.var` and `Statistics.std`.

- A flexible variance estimator for `PortfolioOptimisers.jl` supporting optional expected returns estimators, observation weights, and bias correction. [`SimpleVariance`](@ref)
- Variance estimator that restricts computation to a rolling or indexed observation window. [`WindowedVariance`](@ref)

### Covariance and correlation

Overloads `Statistics.cov` and `Statistics.cor`.

- A simple wrapper around a [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator), optional [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/), and an optional index. [`GeneralCovariance`](@ref)
- ::: details Estimates the covariance matrix of asset returns from a centring estimator, a covariance estimator, and a moment algorithm. [`Covariance`](@ref)
  - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
  - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
- ::: details Configures and applies Gerber covariance estimators. [`GerberCovariance`](@ref)
  - Implements the original Gerber covariance algorithm. [`Gerber0`](@ref)
  - Implements the first variant of the Gerber covariance algorithm. [`Gerber1`](@ref)
  - Implements the second variant of the Gerber covariance algorithm. [`Gerber2`](@ref)
- ::: details Configures and applies Smyth-Broby covariance estimators. [`SmythBrobyCovariance`](@ref)
  - Implements the original Smyth-Broby covariance algorithm. [`SmythBroby0`](@ref)
  - Implements the first variant of the Smyth-Broby covariance algorithm. [`SmythBroby1`](@ref)
  - Implements the second variant of the Smyth-Broby covariance algorithm. [`SmythBroby2`](@ref)
  - Implements the original Smyth-Broby covariance algorithm scaled by vote counts. [`SmythBrobyGerber0`](@ref)
  - Implements the first variant of the Smyth-Broby covariance algorithm scaled by vote counts. [`SmythBrobyGerber1`](@ref)
  - Implements the second variant of the Smyth-Broby covariance algorithm scaled by vote counts. [`SmythBrobyGerber2`](@ref)
  - Implements the original Smyth-Broby covariance algorithm using vote counts only. [`SmythBrobyCount0`](@ref)
  - Implements the first variant of the Smyth-Broby covariance algorithm using vote counts only. [`SmythBrobyCount1`](@ref)
  - Implements the second variant of the Smyth-Broby covariance algorithm using vote counts only. [`SmythBrobyCount2`](@ref)
- ::: details Gerber Information Quality [`GerberIQCovariance`](@ref) with custom variance, demeaning, temporal decay and numerator + denominator estimators
  - Implements the basic Gerber IQ covariance template. [`BasicGerberIQ`](@ref)
  - Gerber Information Quality template with asymmetric thresholds. [`PartialGerberIQ`](@ref)
  - Gerber Information Quality template with fine-grained asymmetric thresholds. [`FullGerberIQ`](@ref)
  - Exponential Gerber IQ temporal decay. [`ExpGerberIQDecay`](@ref)
  - Scales the threshold parameters using the individual asset volatilities. [`AssetVolatilityGerberIQScaler`](@ref)
- Configures and applies distance-based covariance estimators. [`DistanceCovariance`](@ref)
- Lower tail dependence covariance estimator. [`LowerTailDependenceCovariance`](@ref)
- ::: details Rank covariances
  - Robust covariance estimator based on Kendall's tau rank correlation. [`KendallCovariance`](@ref)
  - Robust covariance estimator based on Spearman's rho rank correlation. [`SpearmanCovariance`](@ref)
- ::: details Covariance estimator based on mutual information. [`MutualInfoCovariance`](@ref)
  - ::: details Abstract supertype for all histogram binning algorithms based on a bin width selection rule. [`BinWidthBins`](@ref)
    - Knuth's optimal bin width [`Knuth`](@ref)
    - Freedman Diaconis bin width [`FreedmanDiaconis`](@ref)
    - Scott's bin width [`Scott`](@ref)
  - Histogram binning algorithm using the Hacine-Gharbi–Ravier rule. [`HacineGharbiRavier`](@ref)
  - Predefined number of bins
- Convenience constructor. [`DenoiseCovariance`](@ref)
- Convenience constructor. [`DetoneCovariance`](@ref)
- Convenience constructor. [`ProcessedCovariance`](@ref)
- ::: details Covariance estimator based on implied volatility scaling. [`ImpliedVolatility`](@ref)
  - Implied volatility algorithm that scales implied volatility by a user-supplied premium factor. [`ImpliedVolatilityPremium`](@ref)
  - Implied volatility algorithm that predicts realised volatility via regression on implied volatility. [`ImpliedVolatilityRegression`](@ref)
- Composite covariance estimator with post-processing. [`PortfolioOptimisersCovariance`](@ref)
- A covariance estimator that returns the correlation matrix as both the covariance and correlation. [`CorrelationCovariance`](@ref)
- Covariance estimator that restricts computation to a rolling or indexed observation window. [`WindowedCovariance`](@ref)
- ::: details Regime-adjusted covariance and variance
  - Online exponentially weighted covariance estimator with regime-state adjustment. [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - Online exponentially weighted variance estimator with regime-state adjustment. [`RegimeAdjustedExpWeightedVariance`](@ref)
  - ::: details Regime adjustment methods
    - Regime adjustment method that scales variance by the ratio of the mean absolute deviation of standardised returns to the first-moment normalisation constant `x`. [`FirstMomentRegimeAdjusted`](@ref)
    - Regime adjustment method that scales variance exponentially with the smoothed log-deviation of standardised squared returns from its expected value under stationarity. [`LogRegimeAdjusted`](@ref)
    - Regime adjustment method that scales variance by the square root of the mean of the standardised squared returns. [`RootMeanSquaredAdjusted`](@ref)
  - ::: details Shrinkage targets
    - Regime-adjustment target that uses a diagonal baseline covariance structure. [`DiagonalTarget`](@ref)
    - Regime-adjustment target that uses a Mahalanobis-distance-based baseline covariance structure. [`MahalanobisTarget`](@ref)
    - Regime-adjustment target that uses a portfolio-weighted baseline covariance structure. [`PortfolioTarget`](@ref)
  - ::: details Demeaning
    - Centres the returns series using the (weighted) mean before computing the Median Absolute Deviation. [`MeanCentering`](@ref)
    - Centres the returns series using the (weighted) median before computing the Median Absolute Deviation. [`MedianCentering`](@ref)
  - ::: details Correlation smoothing
    - Greedy pairwise correlation pruning: drop assets until no surviving pair exceeds `t`. [`PairwiseCorrelation`](@ref)
    - Group assets by connected component of the over-threshold correlation graph, and keep the best-scoring member of each. [`CorrelationComponents`](@ref)

### [Coskewness](@id catalogue-coskewness)

Implements [`coskewness`](@ref).

- ::: details Container type for coskewness estimators. [`Coskewness`](@ref)
  - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
  - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
- Coskewness estimator that restricts computation to a rolling or indexed observation window. [`WindowedCoskewness`](@ref)

### [Cokurtosis](@id catalogue-cokurtosis)

Implements [`cokurtosis`](@ref).

- ::: details Container type for cokurtosis estimators. [`Cokurtosis`](@ref)
  - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
  - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
- Cokurtosis estimator that restricts computation to a rolling or indexed observation window. [`WindowedCokurtosis`](@ref)

### Windowed moments

Every windowed estimator wraps a base moment estimator and recomputes it over a trailing window, so a moment can vary across the folds of a cross-validation scheme. The window is set by a fixed length or by a [`WindowSizeEstimator`](@ref).

- Abstract supertype for estimators that determine the rolling window size. [`WindowSizeEstimator`](@ref)

## Distance matrices

Implements [`distance`](@ref) and [`cor_and_dist`](@ref).

- If power is not `nothing`, computes the generalised distance estimator. [`Distance`](@ref)
- Distance-of-distances estimator for portfolio optimization. [`DistanceDistance`](@ref)

The distance estimators are used together with various distance matrix algorithms.

- Simple distance algorithm for portfolio optimization. [`SimpleDistance`](@ref)
- Simple absolute distance algorithm for portfolio optimization. [`SimpleAbsoluteDistance`](@ref)
- Logarithmic distance algorithm for portfolio optimization. [`LogDistance`](@ref)
- Correlation distance algorithm for portfolio optimization. [`CorrelationDistance`](@ref)
- ::: details Variation of Information (VI) distance algorithm for portfolio optimization. [`VariationInfoDistance`](@ref)
  - ::: details Abstract supertype for all histogram binning algorithms based on a bin width selection rule. [`BinWidthBins`](@ref)
    - Knuth's optimal bin width [`Knuth`](@ref)
    - Freedman Diaconis bin width [`FreedmanDiaconis`](@ref)
    - Scott's bin width [`Scott`](@ref)
  - Histogram binning algorithm using the Hacine-Gharbi–Ravier rule. [`HacineGharbiRavier`](@ref)
  - Predefined number of bins
- Canonical distance algorithm for portfolio optimization. [`CanonicalDistance`](@ref)

## Phylogeny

`PortfolioOptimisers.jl` can make use of asset relationships to perform optimisations, define constraints, and compute relatedness characteristics of portfolios.

### Clustering

Phylogeny constraints and clustering optimisations make use of clustering algorithms via [`ClustersEstimator`](@ref), [`Clusters`](@ref), and [`clusterise`](@ref). Most clustering algorithms come from [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl).

- ::: details Estimator type for selecting the optimal number of clusters. [`OptimalNumberClusters`](@ref) and [`VectorToScalarMeasure`](@ref)
  - Algorithm type for estimating the optimal number of clusters using the second-order difference method. [`SecondOrderDifference`](@ref)
  - Algorithm type for estimating the optimal number of clusters using the standardised silhouette score. [`SilhouetteScore`](@ref)
  - Predefined number of clusters.
  - Select the optimal number of clusters for a hierarchical clustering tree. [`optimal_number_clusters`](@ref)
- Get the vector of cluster indices for each point. [`assignments`](@ref)

#### Hierarchical

- Algorithm type for hierarchical clustering. [`HClustAlgorithm`](@ref)
- ::: details Direct Bubble Hierarchical Trees [`DBHT`](@ref) and Local Global sparsification of the covariance matrix [`LoGo`](@ref), [`logo!`](@ref), and [`logo`](@ref)
  - ::: details Root selection
    - A DBHT root selection method that enforces a unique root in the hierarchy. [`UniqueRoot`](@ref)
    - A DBHT root selection method that creates a root from the adjacency tree of all root candidates. [`EqualRoot`](@ref)

#### Non-hierarchical

Non-hierarchical clustering algorithms are incompatible with hierarchical clustering optimisations, but they can be used for phylogeny constraints and [`NestedClustered`](@ref) optimisations.

- K-means clustering algorithm configuration for non-hierarchical clustering. [`KMeansAlgorithm`](@ref)

### Networks

#### Adjacency matrices

Adjacency matrices encode asset relationships either with clustering or graph theory via [`phylogeny_matrix`](@ref) and [`PhylogenyResult`](@ref).

- ::: details Network adjacency [`NetworkEstimator`](@ref) with custom tree algorithms, covariance, and distance estimators
  - Algorithm type for Kruskal's minimum spanning tree (MST). [`KruskalTree`](@ref), [`BoruvkaTree`](@ref), and [`PrimTree`](@ref)
  - ::: details Triangulated Maximally Filtered Graph with various similarity matrix estimators
    - Maximum distance similarity [`MaximumDistanceSimilarity`](@ref)
    - Exponential similarity [`ExponentialSimilarity`](@ref)
    - General exponential similarity [`GeneralExponentialSimilarity`](@ref)
- Estimator type for clustering. [`ClustersEstimator`](@ref) and [`Clusters`](@ref)
- Estimator type for network-based phylogeny analysis. [`NetworkClustersEstimator`](@ref)
- Group assets by clustering them, and keep the best-scoring member of each cluster. [`ClusterGroups`](@ref)

#### Centrality and phylogeny measures

- ::: details Centrality estimator [`CentralityEstimator`](@ref) with custom adjacency matrix estimators (clustering and network) and centrality measures
  - Betweenness [`BetweennessCentrality`](@ref)
  - Closeness [`ClosenessCentrality`](@ref)
  - Degree [`DegreeCentrality`](@ref)
  - Eigenvector [`EigenvectorCentrality`](@ref)
  - Katz [`KatzCentrality`](@ref)
  - Pagerank [`Pagerank`](@ref)
  - Radiality [`RadialityCentrality`](@ref)
  - Stress [`StressCentrality`](@ref)
- Fallback no-op for returning a validated centrality vector result as-is. [`centrality_vector`](@ref)
- Compute the weighted average centrality for a network and centrality algorithm. [`average_centrality`](@ref)
- Compute the asset phylogeny score for a set of weights and a phylogeny matrix. [`asset_phylogeny`](@ref)

#### Cluster trees

Hierarchical clustering produces a tree of [`ClusterNode`](@ref)s, walked by [`to_tree`](@ref), [`pre_order`](@ref), and [`is_leaf`](@ref).

- Preorder traversal strategy that visits nodes by their ID. [`PreorderTreeByID`](@ref)

## Optimisation constraints

Non clustering optimisers support a wide range of constraints, while naive and clustering optimisers only support weight bounds. Furthermore, entropy pooling prior supports a variety of views constraints. It is therefore important to provide users with the ability to generate constraints manually and/or programmatically. We therefore provide a wide, robust, and extensible range of types such as [`AbstractEstimatorValueAlgorithm`](@ref) and [`UniformValues`](@ref), and functions that make this easy, fast, and safe.

Constraints can be defined via their estimators or directly by their result types. Some using estimators need to map key-value pairs to the asset universe, this is done by defining the assets and asset groups in [`AssetSets`](@ref). Internally, `PortfolioOptimisers.jl` uses all the information and calls [`group_to_val!`](@ref), and [`replace_group_by_assets`](@ref) to produce the appropriate arrays.

- Equation parsing [`parse_equation`](@ref) and [`ParsingResult`](@ref)
- No-op fallback for returning an existing `LinearConstraint` object or `nothing`. [`linear_constraints`](@ref), [`LinearConstraintEstimator`](@ref), [`PartialLinearConstraint`](@ref), and [`LinearConstraint`](@ref)
- No-op fallback for risk budget constraint generation. [`risk_budget_constraints`](@ref), [`RiskBudgetEstimator`](@ref), and [`RiskBudget`](@ref)
- Generate phylogeny-based portfolio constraints from an estimator or result. [`phylogeny_constraints`](@ref), [`centrality_constraints`](@ref), [`SemiDefinitePhylogenyEstimator`](@ref), [`SemiDefinitePhylogeny`](@ref), [`IntegerPhylogenyEstimator`](@ref), [`IntegerPhylogeny`](@ref), and [`CentralityConstraint`](@ref)
- Generate portfolio weight bounds constraints from a `WeightBoundsEstimator` and asset set. [`weight_bounds_constraints`](@ref), [`WeightBoundsEstimator`](@ref), and [`WeightBounds`](@ref)
- Container for asset set and group information used in constraint generation. [`AssetSets`](@ref)
- ::: details Budget constraints [`BudgetEstimator`](@ref) and [`BudgetRange`](@ref)
  - Budget constraint that accounts for linear transaction costs. [`BudgetCosts`](@ref)
  - Budget constraint that accounts for non-linear (power-law) market impact costs. [`BudgetMarketImpact`](@ref)
- ::: details Constraint values [`AbstractEstimatorValueAlgorithm`](@ref)

  Where a constraint takes one value per asset or group, these algorithms say how to derive it from data rather than stating it outright.

  - Custom weight bounds constraint for uniformly distributing asset weights, `1/N` for lower bounds and `1` for upper bounds, where `N` is the number of assets. [`UniformValues`](@ref)
  - Return value for assets or groups, based on a mapping and asset sets. [`estimator_to_val`](@ref)
  - Algorithm for reducing a vector of real values to its minimum. [`MinValue`](@ref)
  - Algorithm for reducing a vector of real values to its maximum. [`MaxValue`](@ref)
  - Algorithm for reducing a vector of real values to its optionally weighted mean. [`MeanValue`](@ref)
  - Algorithm for reducing a vector of real values to its optionally weighted median. [`MedianValue`](@ref)
  - Algorithm for reducing a vector of real values to its mode. [`ModeValue`](@ref)
  - Algorithm for reducing a vector of real values to its sum. [`SumValue`](@ref)
  - Algorithm for reducing a vector of real values to its product. [`ProdValue`](@ref)
  - Algorithm for reducing a vector of real values to its optionally weighted standard deviation. [`StdValue`](@ref)
  - Algorithm for reducing a vector of real values to its optionally weighted variance. [`VarValue`](@ref)
  - Algorithm for reducing a vector of real values to its optionally weighted mean divided by its optionally weighted standard deviation. [`StandardisedValue`](@ref)
  - Marker for "no default here", used in the two places a fold-less value may be missing. [`NoDefault`](@ref)
- ::: details Time-dependent constraint: an optimiser input whose value changes across the folds of a cross-validation scheme. [`TimeDependent`](@ref)

  A time-dependent input takes a different value in each fold of a cross-validation scheme, and is inert outside one.

  - Abstract supertype for callable structs used as time-dependent constraint values. [`TimeDependentCallable`](@ref)
  - Abstract supertype for callable structs whose per-fold value is an *optimiser*. [`TimeDependentOptimiserCallable`](@ref)
  - Per-fold context handed to time-dependent constraints when they are resolved. [`TimeDependentContext`](@ref)
  - Wrapper marking a callable time-dependent constraint entry as requiring the previous optimisation's weights. [`PreviousWeightsFunction`](@ref)
- Construct a binary asset-group membership matrix from asset set groupings. [`asset_sets_matrix`](@ref) and [`AssetSetsMatrixEstimator`](@ref)
- Propagate or pass through buy-in threshold portfolio constraints. [`threshold_constraints`](@ref), [`ThresholdEstimator`](@ref), and [`Threshold`](@ref)

## Prior statistics

Many optimisations and constraints use prior statistics computed via [`prior`](@ref).

- ::: details Container type for low order prior results. [`LowOrderPrior`](@ref)
  - Empirical prior estimator for asset returns. [`EmpiricalPrior`](@ref)
  - Factor-based prior estimator for asset returns. [`FactorPrior`](@ref)
  - ::: details Black-Litterman
    - Unified interface for constructing or passing through Black-Litterman investor views. [`black_litterman_views`](@ref)
    - Black-Litterman prior estimator for asset returns. [`BlackLittermanPrior`](@ref)
    - Bayesian Black-Litterman prior estimator for asset returns. [`BayesianBlackLittermanPrior`](@ref)
    - Factor Black-Litterman prior estimator for asset returns. [`FactorBlackLittermanPrior`](@ref)
    - Augmented Black-Litterman prior estimator for asset returns. [`AugmentedBlackLittermanPrior`](@ref)
  - ::: details Entropy pooling prior estimator for asset returns. [`EntropyPoolingPrior`](@ref)

    Entropy pooling reweights the observations so that the posterior satisfies the stated views while staying as close as possible to the prior.

    - Container for Black-Litterman investor views in canonical matrix form. [`BlackLittermanViews`](@ref)
    - ::: details View constraint algorithms
      - One-shot entropy pooling. [`H0_EntropyPooling`](@ref)
      - Uses the initial probabilities to optimise the posterior probabilities at every step. [`H1_EntropyPooling`](@ref)
      - Uses the previous step's probabilities to optimise the next step's probabilities. [`H2_EntropyPooling`](@ref)
      - Conditional Value-at-Risk (CVaR) entropy pooling optimiser. [`CVaREntropyPooling`](@ref)
    - ::: details Divergence formulations
      - Exponential entropy pooling optimisation algorithm. [`ExpEntropyPooling`](@ref)
      - Logarithmic entropy pooling optimisation algorithm. [`LogEntropyPooling`](@ref)
    - ::: details Optimisers
      - [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl)-based entropy pooling optimiser. [`OptimEntropyPooling`](@ref)
      - [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl)-based entropy pooling optimiser. [`JuMPEntropyPooling`](@ref)
  - ::: details Opinion pooling prior estimator for asset returns. [`OpinionPoolingPrior`](@ref)
    - Linear opinion pooling algorithm for consensus prior estimation. [`LinearOpinionPooling`](@ref)
    - Logarithmic opinion pooling algorithm for consensus prior estimation. [`LogarithmicOpinionPooling`](@ref)
- ::: details Container type for high order prior results. [`HighOrderPrior`](@ref)
  - High order prior estimator for asset returns. [`HighOrderPriorEstimator`](@ref)
  - Represents the High Order Factor Prior Estimator. [`HighOrderFactorPriorEstimator`](@ref)

## Uncertainty sets

In order to make optimisations more robust to noise and measurement error, it is possible to define uncertainty sets on the expected returns and covariance. These can be used in optimisations which use either of these two quantities. These are implemented via [`ucs`](@ref), [`mu_ucs`](@ref), and [`sigma_ucs`](@ref).

`PortfolioOptimisers.jl` implements two types of uncertainty sets.

- Represents a box uncertainty set for risk or prior statistics in portfolio optimisation. [`BoxUncertaintySet`](@ref) and [`BoxUncertaintySetAlgorithm`](@ref)
- ::: details [`EllipsoidalUncertaintySet`](@ref) and [`EllipsoidalUncertaintySetAlgorithm`](@ref) with various algorithms for computing the scaling parameter via [`k_ucs`](@ref)
  - Algorithm for computing the scaling parameter `k` for ellipsoidal uncertainty sets under the assumption of normally distributed returns in portfolio optimisation. [`NormalKUncertaintyAlgorithm`](@ref)
  - Computes the ellipsoidal uncertainty set scaling parameter `k` as `sqrt((1 - q) / q)`. [`GeneralKUncertaintyAlgorithm`](@ref)
  - Algorithm for computing the scaling parameter `k` for ellipsoidal uncertainty sets using the chi-squared distribution in portfolio optimisation. [`ChiSqKUncertaintyAlgorithm`](@ref)
  - Predefined scaling parameter

It also implements various estimators for the uncertainty sets, the following two can generate box and ellipsoidal sets.

- Estimator for box or ellipsoidal uncertainty sets under the assumption of normally distributed returns in portfolio optimisation. [`NormalUncertaintySet`](@ref)
- ::: details Bootstrapping via Autoregressive Conditional Heteroscedasticity [`ARCHUncertaintySet`](@ref) via [`arch`](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html)
  - Circular [`CircularBootstrap`](@ref)
  - Moving [`MovingBootstrap`](@ref)
  - Stationary [`StationaryBootstrap`](@ref)

The following estimator can only generate box sets.

- Estimator for box uncertainty sets using delta bounds on mean and covariance statistics in portfolio optimisation. [`DeltaUncertaintySet`](@ref)

Quintile portfolios are expressed as an uncertainty set on the characteristic vector rather than as an optimiser of their own (ADR 0032).

- ::: details Estimator for $\ell_1$ uncertainty sets on the characteristic vector. [`CharacteristicUncertaintySet`](@ref)
  - $\ell_1$ (cross-polytope) uncertainty set on the characteristic vector. [`L1UncertaintySet`](@ref) and [`L1UncertaintySetAlgorithm`](@ref)
  - Signed $\ell_1$ uncertainty set on the characteristic vector, with a separate error budget per sign. [`SignedL1UncertaintySet`](@ref) and [`SignedL1UncertaintySetAlgorithm`](@ref)
  - Radius algorithm that calibrates the $\ell_1$ uncertainty radius to a target number of active assets. [`ActiveAssetsUncertaintyAlgorithm`](@ref)
- ::: details Ellipsoidal set classes
  - Represents the class identifier for mean ellipsoidal uncertainty sets in portfolio optimisation. [`MuEllipsoidalUncertaintySet`](@ref)
  - Represents the class identifier for covariance ellipsoidal uncertainty sets in portfolio optimisation. [`SigmaEllipsoidalUncertaintySet`](@ref)

## Turnover

The turnover is defined as the element-wise absolute difference between the vector of current weights and a vector of benchmark weights. It can be used as a constraint, method for fee calculation, and risk measure. These are all implemented using [`turnover_constraints`](@ref), [`TurnoverEstimator`](@ref), and [`Turnover`](@ref).

## Fees

Fees are a non-negligible aspect of active investing. As such `PortfolioOptimiser.jl` has the ability to account for them in all optimisations but the naive ones. They can also be used to adjust expected returns calculations via [`calc_fees`](@ref) and [`calc_asset_fees`](@ref).

- Generate portfolio transaction fee constraints from a `FeesEstimator` and asset set. [`fees_constraints`](@ref)
- Compute the fixed portfolio fees for assets that have been allocated. [`calc_fixed_fees`](@ref) and [`calc_asset_fixed_fees`](@ref)
- ::: details Estimator for portfolio transaction fees constraints. [`FeesEstimator`](@ref) and [`Fees`](@ref)
  - Proportional long
  - Proportional short
  - Fixed long
  - Fixed short
  - Turnover

## Portfolio returns and drawdowns

Various risk measures and analyses require the computation of simple and cumulative portfolio returns and drawdowns both in aggregate and per-asset. These are computed by [`calc_net_returns`](@ref), [`calc_net_asset_returns`](@ref), [`cumulative_returns`](@ref), [`drawdowns`](@ref).

## [Tracking](@id catalogue-tracking)

It is often useful to create portfolios that track the performance of an index, indicator, or another portfolio.

- ::: details Compute the benchmark portfolio returns for a weights-based tracking algorithm. [`tracking_benchmark`](@ref) and [`TrackingError`](@ref)
  - Returns-based tracking algorithm. [`ReturnsTracking`](@ref)
  - Asset weights-based tracking algorithm. [`WeightsTracking`](@ref)

The error can be computed using different algorithms using [`norm_error`](@ref).

- ::: details Norm tracking algorithms
  - Norm-one (NOC) error formulation. [`L1Norm`](@ref)
  - Second-order cone (SOC) norm-based error formulation. [`L2Norm`](@ref)
  - Second-order cone (SOC) squared norm-based error formulation. [`SquaredL2Norm`](@ref)
  - L-p norm error estimator. [`LpNorm`](@ref)
  - L-infinity norm (maximum absolute deviation) error estimator. [`LInfNorm`](@ref)

It is also possible to track the error in with risk measures [`RiskTrackingError`](@ref) using [`WeightsTracking`](@ref), which allows for two approaches.

- Dependent variable-based tracking formulation. [`DependentVariableTracking`](@ref)
- Independent variable-based tracking formulation. [`IndependentVariableTracking`](@ref)

## Risk measures

`PortfolioOptimisers.jl` provides a wide range of risk measures. These are broadly categorised into two types based on the type of optimisations that support them.

### Risk measures for traditional optimisation

These are all subtypes of [`RiskMeasure`](@ref), and are supported by all optimisation estimators.

- ::: details Represents the portfolio variance using a covariance matrix. [`Variance`](@ref)
  - ::: details Traditional optimisations also support:
    - Risk contribution
    - ::: details Formulations
      - Direct quadratic risk expression optimisation formulation for variance-like risk measures. [`QuadRiskExpr`](@ref)
      - Squared second-order cone risk expression optimisation formulation for applicable risk measures. [`SquaredSOCRiskExpr`](@ref)
- Represents the portfolio standard deviation using a covariance matrix. [`StandardDeviation`](@ref)
- Uncertainty set variance [`UncertaintySetVariance`](@ref) (same as variance when used in non-traditional optimisation)
- ::: details Represents a low-order moment risk measure. [`LowOrderMoment`](@ref)
  - Represents the first lower moment risk measure algorithm. [`FirstLowerMoment`](@ref)
  - Represents the mean absolute deviation risk measure algorithm. [`MeanAbsoluteDeviation`](@ref)
  - ::: details Represents a second moment (variance or standard deviation) risk measure algorithm. [`SecondMoment`](@ref)
    - ::: details Second squared moments
      - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
      - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
      - ::: details Traditional optimisation formulations
        - Direct quadratic risk expression optimisation formulation for variance-like risk measures. [`QuadRiskExpr`](@ref)
        - Squared second-order cone risk expression optimisation formulation for applicable risk measures. [`SquaredSOCRiskExpr`](@ref)
        - Rotated second-order cone risk expression optimisation formulation for applicable risk measures. [`RSOCRiskExpr`](@ref)
    - ::: details Second-order cone risk expression optimisation formulation for applicable risk measures. [`SOCRiskExpr`](@ref)
      - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
      - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
- ::: details Represents the square root kurtosis risk measure. [`Kurtosis`](@ref)
  - Actual kurtosis
    - ::: details FullMoment and semi-kurtosis are supported in traditional optimisers via the `kt` field. Risk calculation uses
      - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
      - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
    - ::: details Traditional optimisation formulations
      - Direct quadratic risk expression optimisation formulation for variance-like risk measures. [`QuadRiskExpr`](@ref)
      - Squared second-order cone risk expression optimisation formulation for applicable risk measures. [`SquaredSOCRiskExpr`](@ref)
      - Rotated second-order cone risk expression optimisation formulation for applicable risk measures. [`RSOCRiskExpr`](@ref)
  - ::: details Second-order cone risk expression optimisation formulation for applicable risk measures. [`SOCRiskExpr`](@ref)
    - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
    - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
- ::: details Represents the Negative Skewness risk measure. [`NegativeSkewness`](@ref)
  - ::: details Squared negative skewness
    - ::: details FullMoment and semi-skewness are supported in traditional optimisers via the `sk` and `V` fields. Risk calculation uses
      - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
      - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
    - ::: details Traditional optimisation formulations
      - Direct quadratic risk expression optimisation formulation for variance-like risk measures. [`QuadRiskExpr`](@ref)
      - Squared second-order cone risk expression optimisation formulation for applicable risk measures. [`SquaredSOCRiskExpr`](@ref)
    - Second-order cone risk expression optimisation formulation for applicable risk measures. [`SOCRiskExpr`](@ref)
- ::: details Represents the Value-at-Risk (VaR) risk measure. [`ValueatRisk`](@ref)
  - ::: details Traditional optimisation formulations
    - Mixed-integer programming (MIP) formulation for Value-at-Risk. [`MIPValueatRisk`](@ref)
    - Distribution-based formulation for Value-at-Risk. [`DistributionValueatRisk`](@ref)
- ::: details Represents the Value-at-Risk Range risk measure. [`ValueatRiskRange`](@ref)
  - ::: details Traditional optimisation formulations
    - Mixed-integer programming (MIP) formulation for Value-at-Risk. [`MIPValueatRisk`](@ref)
    - Distribution-based formulation for Value-at-Risk. [`DistributionValueatRisk`](@ref)
- Represents the Drawdown-at-Risk (DaR) risk measure. [`DrawdownatRisk`](@ref)
- Represents the Conditional Value-at-Risk (CVaR) risk measure, also known as Expected Shortfall (ES). [`ConditionalValueatRisk`](@ref)
- Distributionally Robust Conditional Value at Risk [`DistributionallyRobustConditionalValueatRisk`](@ref) (same as conditional value at risk when used in non-traditional optimisation)
- Represents the Conditional Value-at-Risk Range (CVaR Range) risk measure. [`ConditionalValueatRiskRange`](@ref)
- Distributionally Robust Conditional Value at Risk Range [`DistributionallyRobustConditionalValueatRiskRange`](@ref) (same as conditional value at risk range when used in non-traditional optimisation)
- Represents the Conditional Drawdown-at-Risk (CDaR) risk measure, also known as Expected Maximum Drawdown. [`ConditionalDrawdownatRisk`](@ref)
- Distributionally Robust Conditional Drawdown at Risk [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)(same as conditional drawdown at risk when used in non-traditional optimisation)
- Represents the Entropic Value-at-Risk (EVaR) risk measure. [`EntropicValueatRisk`](@ref)
- Represents the Entropic Value-at-Risk Range (EVaR Range) risk measure. [`EntropicValueatRiskRange`](@ref)
- Represents the Entropic Drawdown-at-Risk (EDaR) risk measure. [`EntropicDrawdownatRisk`](@ref)
- Represents the Relativistic Value-at-Risk (RVaR) risk measure. [`RelativisticValueatRisk`](@ref)
- Represents the Relativistic Value-at-Risk Range (RVaR Range) risk measure. [`RelativisticValueatRiskRange`](@ref)
- Represents the Relativistic Drawdown-at-Risk (RLDaR) risk measure. [`RelativisticDrawdownatRisk`](@ref)
- ::: details Ordered Weights Array
  - ::: details Risk measures
    - Ordered Weights Array (OWA) risk measure. [`OrderedWeightsArray`](@ref)
    - Ordered Weights Array Range (OWA Range) risk measure. [`OrderedWeightsArrayRange`](@ref)
  - ::: details Traditional optimisation formulations
    - OWA formulation that computes exact OWA weights by solving a linear programme. [`ExactOrderedWeightsArray`](@ref)
    - OWA formulation that approximates OWA weights using a set of p-norm parameters. [`ApproxOrderedWeightsArray`](@ref)
    - Estimator type for OWA weights using JuMP-based optimization. [`OWAJuMP`](@ref)
  - ::: details One-call OWA measures
    - Callable OWA weight estimator for the Conditional Value at Risk (CVaR) risk measure. [`OrderedWeightsArrayConditionalValueatRisk`](@ref)
    - Callable OWA weight estimator for the Conditional Value at Risk Range risk measure. [`OrderedWeightsArrayConditionalValueatRiskRange`](@ref)
    - Callable OWA weight estimator for the tail Gini risk measure. [`OrderedWeightsArrayTailGini`](@ref)
    - Callable OWA weight estimator for the tail Gini range risk measure. [`OrderedWeightsArrayTailGiniRange`](@ref)
  - ::: details Array functions
    - Gini Mean Difference [`owa_gmd`](@ref)
    - Worst Realisation [`owa_wr`](@ref)
    - Range [`owa_rg`](@ref)
    - Conditional Value at Risk [`owa_cvar`](@ref)
    - Weighted Conditional Value at Risk [`owa_wcvar`](@ref)
    - Conditional Value at Risk Range [`owa_cvarrg`](@ref)
    - Weighted Conditional Value at Risk Range [`owa_wcvarrg`](@ref)
    - Tail Gini [`owa_tg`](@ref)
    - Tail Gini Range [`owa_tgrg`](@ref)
    - ::: details Linear moments (L-moments)
      - Compute the linear moment weights for the linear moments convex risk measure (CRM). [`owa_l_moment`](@ref)
      - ::: details Compute Ordered Weights Array (OWA) linear moment convex risk measure (CRM) weights using various estimation methods. [`owa_l_moment_crm`](@ref)
        - ::: details L-moment combination formulations
          - ::: details Represents the Maximum Entropy algorithm for Ordered Weights Array (OWA) estimation. [`MaximumEntropy`](@ref)
            - Entropy formulation for [`MaximumEntropy`](@ref) OWA that uses the exponential cone entropy constraint in JuMP. [`ExponentialConeEntropy`](@ref)
            - Entropy formulation for [`MaximumEntropy`](@ref) OWA that uses the relative entropy cone constraint in JuMP. [`RelativeEntropy`](@ref)
          - Represents the Minimum Squared Distance algorithm for Ordered Weights Array (OWA) estimation. [`MinimumSquaredDistance`](@ref)
          - Represents the Minimum Sum of Squares algorithm for Ordered Weights Array (OWA) estimation. [`MinimumSumSquares`](@ref)
- Represents the Average Drawdown risk measure. [`AverageDrawdown`](@ref)
- Represents the Ulcer Index risk measure. [`UlcerIndex`](@ref)
- Represents the Maximum Drawdown risk measure. [`MaximumDrawdown`](@ref)
- ::: details Represents the Brownian Distance Variance (BDVar) risk measure. [`BrownianDistanceVariance`](@ref)
  - ::: details Traditional optimisation formulations
    - ::: details Distance matrix constraint formulations
      - Norm-one cone formulation for the Brownian Distance Variance optimisation constraint. [`NormOneConeBrownianDistanceVariance`](@ref)
      - Inequality formulation for the Brownian Distance Variance optimisation constraint. [`IneqBrownianDistanceVariance`](@ref)
    - ::: details Risk formulation
      - Direct quadratic risk expression optimisation formulation for variance-like risk measures. [`QuadRiskExpr`](@ref)
      - Rotated second-order cone risk expression optimisation formulation for applicable risk measures. [`RSOCRiskExpr`](@ref)
- Represents the Worst Realisation risk measure. [`WorstRealisation`](@ref)
- Represents the Range risk measure. [`Range`](@ref)
- Represents the Turnover risk measure. [`TurnoverRiskMeasure`](@ref)
- ::: details Represents the Tracking Error risk measure. [`TrackingRiskMeasure`](@ref)
  - Norm-one (NOC) error formulation. [`L1Norm`](@ref)
  - Second-order cone (SOC) norm-based error formulation. [`L2Norm`](@ref)
  - Second-order cone (SOC) squared norm-based error formulation. [`SquaredL2Norm`](@ref)
  - L-p norm error estimator. [`LpNorm`](@ref)
  - L-infinity norm (maximum absolute deviation) error estimator. [`LInfNorm`](@ref)
- ::: details Risk Tracking Risk Measure
  - Dependent variable-based tracking formulation. [`DependentVariableTracking`](@ref)
  - Independent variable-based tracking formulation. [`IndependentVariableTracking`](@ref)
- Represents the Power Norm Value-at-Risk (PNVaR) risk measure. [`PowerNormValueatRisk`](@ref)
- Represents the Power Norm Value-at-Risk Range (PNVaRRange) risk measure. [`PowerNormValueatRiskRange`](@ref)
- Represents the Power Norm Drawdown-at-Risk (PNDDaR) risk measure. [`PowerNormDrawdownatRisk`](@ref)
- Represents a generic Value-at-Risk range risk measure that combines any pair of XatRisk-type measures applied to the loss and gain sides of the return distribution. [`GenericValueatRiskRange`](@ref)
- Represents the Risk Tracking risk measure. [`RiskTrackingRiskMeasure`](@ref)
- Risk measure that contributes no risk. [`NoRisk`](@ref)
- ::: details Risk measure settings

  Every risk measure carries a settings object saying how it enters the problem: as the objective, as a constraint with an upper bound, and with what scale.

  - Settings type for configuring risk measure estimators. [`RiskMeasureSettings`](@ref)
  - Settings type for configuring hierarchical risk measure estimators. [`HierarchicalRiskMeasureSettings`](@ref)
  - Settings type for configuring risk measures that expose a lower bound (maximisation direction). [`MaxRiskMeasureSettings`](@ref)

### Risk measures for hierarchical optimisation

These are all subtypes of [`HierarchicalRiskMeasure`](@ref), and are only supported by hierarchical optimisation estimators.

- ::: details Represents a high-order moment risk measure. [`HighOrderMoment`](@ref)
  - Represents the unstandardised semi-skewness risk measure algorithm. [`ThirdLowerMoment`](@ref)
  - Represents a standardised high-order moment risk measure algorithm. [`StandardisedHighOrderMoment`](@ref) and [`ThirdLowerMoment`](@ref)
  - ::: details Represents the unstandardised fourth moment (kurtosis or semi-kurtosis) risk measure algorithm. [`FourthMoment`](@ref)
    - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
    - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
  - ::: details Represents a standardised high-order moment risk measure algorithm. [`StandardisedHighOrderMoment`](@ref) and [`FourthMoment`](@ref)
    - `FullMoment` is used to indicate that all deviations are included in the moment estimation process. [`FullMoment`](@ref)
    - `SemiMoment` is used for semi-moment estimators, where only observations below a target are considered. [`SemiMoment`](@ref)
- Represents the Relative Drawdown-at-Risk risk measure for hierarchical optimisation. [`RelativeDrawdownatRisk`](@ref)
- Represents the Relative Conditional Drawdown-at-Risk risk measure for hierarchical optimisation. [`RelativeConditionalDrawdownatRisk`](@ref)
- Represents the Relative Entropic Drawdown-at-Risk (Relative EDaR) risk measure for hierarchical optimisation. [`RelativeEntropicDrawdownatRisk`](@ref)
- Represents the Relative Relativistic Drawdown-at-Risk (Relative RLDaR) risk measure for hierarchical optimisation. [`RelativeRelativisticDrawdownatRisk`](@ref)
- Represents the Relative Average Drawdown risk measure for hierarchical optimisation. [`RelativeAverageDrawdown`](@ref)
- Represents the Relative Ulcer Index risk measure for hierarchical optimisation. [`RelativeUlcerIndex`](@ref)
- Represents the Relative Maximum Drawdown risk measure for hierarchical optimisation. [`RelativeMaximumDrawdown`](@ref)
- Represents the Relative Power Norm Drawdown-at-Risk (Relative PNDDaR) risk measure for hierarchical optimisation. [`RelativePowerNormDrawdownatRisk`](@ref)
- Represents a risk ratio risk measure for hierarchical portfolio optimisation. [`RiskRatio`](@ref)
- Represents the Equal Risk Measure for hierarchical portfolio optimisation. [`EqualRisk`](@ref)
- Represents the Median Absolute Deviation (MAD) risk measure for hierarchical portfolio optimisation. [`MedianAbsoluteDeviation`](@ref)
- Composite risk measure combining variance, skewness, and kurtosis into a single expression. [`VarianceSkewKurtosis`](@ref)
- Represents an even-order moment risk measure algorithm. [`EvenMoment`](@ref)
- Callable estimator that generates OWA linear moment convex risk measure (CRM) weights for a given number of observations. [`LinearMoment`](@ref)

### Non-optimisation risk measures

These risk measures are unsuitable for optimisation because they can return negative values. However, they can be used for performance metrics.

- Represents a simple mean return measure for use in non-optimisation contexts. [`MeanReturn`](@ref)
- Represents the Third Central Moment risk measure. [`ThirdCentralMoment`](@ref)
- Represents the standardised Skewness risk measure. [`Skewness`](@ref)
- Return-based risk measure. [`ExpectedReturn`](@ref)
- Ratio-based risk measure. [`ExpectedReturnRiskRatio`](@ref)
- Represents a mean return to risk ratio measure. [`MeanReturnRiskRatio`](@ref)
- Represents a non-optimisation risk ratio measure. [`NonOptimisationRiskRatio`](@ref)

## Performance metrics

- Compute the expected value of a risk measure. [`expected_risk`](@ref)
- Compute the effective number of assets (Herfindahl-Hirschman inverse index). [`number_effective_assets`](@ref)
- ::: details Risk contribution
  - Compute the risk contribution of each asset to the total portfolio risk using numerical differentiation. [`risk_contribution`](@ref)
  - Compute the risk contribution of each factor (and the idiosyncratic component) to the total portfolio risk using a factor regression. [`factor_risk_contribution`](@ref)
- ::: details Compute the expected portfolio return using the specified return estimator. [`expected_return`](@ref)
  - Arithmetic [`ArithmeticReturn`](@ref)
  - Logarithmic [`LogarithmicReturn`](@ref)
- Compute the expected risk of a measure from a precomputed net-return series. [`expected_risk_from_returns`](@ref)
- Compute the expected risk of a risk measure over rolling windows of the returns data. [`rolling_window_measure`](@ref)
- Sort the successful paths in a [`PopulationPredictionResult`](@ref) by their expected risk under `r`. [`sort_by_measure`](@ref)
- Compute the expected risk-adjusted return ratio for a portfolio. [`expected_ratio`](@ref) and [`expected_risk_ret_ratio`](@ref)
- Compute the risk-adjusted ratio information criterion (SRIC) for a portfolio. [`expected_sric`](@ref) and [`expected_risk_ret_sric`](@ref)
- Compute Brinson performance attribution aggregated per asset class [brinson_attribution](@cite). [`brinson_attribution`](@ref)

## Portfolio optimisation

Optimisations are implemented via [`optimise`](@ref). Optimisations consume an estimator and return a result.

### Naive

These return a [`NaiveOptimisationResult`](@ref).

- Inverse Volatility portfolio optimiser. [`InverseVolatility`](@ref)
- Equal-weighted portfolio optimiser. [`EqualWeighted`](@ref)
- Random-weighted portfolio optimiser. [`RandomWeighted`](@ref)

#### Naive optimisation features

- Estimator for portfolio weight bounds constraints. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- ::: details Weight finalisers
  - Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)
  - ::: details Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)
    - Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
    - Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

### Traditional

These optimisations are implemented as `JuMP` problems and make use of [`JuMPOptimiser`](@ref), which encodes all supported constraints.

#### Objective function optimisations

These optimisations support a variety of objective functions.

- Container for configuring a JuMP solver and its settings. [`Solver`](@ref)
- Main JuMP-based portfolio optimiser configuration. [`JuMPOptimiser`](@ref)
- ::: details Objective functions
  - Minimum risk [`MinimumRisk`](@ref)
  - Maximum utility [`MaximumUtility`](@ref)
  - Maximum return over risk ratio [`MaximumRatio`](@ref)
  - Maximum return [`MaximumReturn`](@ref)
- ::: details Mean-Risk portfolio optimiser. [`MeanRisk`](@ref) and [`NearOptimalCentering`](@ref)
  - ::: details Defines the number of points on the efficient frontier (Pareto Front). [`Frontier`](@ref)
    - Return based
    - Risk based
    - ::: details Bound spacing [`FrontierBoundEstimator`](@ref)
      - Passes bound values through unchanged (identity transformation). [`LinearBound`](@ref)
      - Applies a square-root transformation to bound values before enforcing them. [`SquareRootBound`](@ref)
      - Applies a squaring transformation to bound values before enforcing them. [`SquaredBound`](@ref)
- ::: details Optimisation estimators
  - Mean-Risk [`MeanRisk`](@ref) returns a [`MeanRiskResult`](@ref)
  - Near Optimal Centering [`NearOptimalCentering`](@ref) returns a [`NearOptimalCenteringResult`](@ref)
  - Factor Risk Contribution [`FactorRiskContribution`](@ref) returns a [`FactorRiskContributionResult`](@ref)
- ::: details Near Optimal Centering formulations [`NearOptimalCentering`](@ref)
  - Constrained Near Optimal Centering algorithm. [`ConstrainedNearOptimalCentering`](@ref)
  - Unconstrained Near Optimal Centering algorithm. [`UnconstrainedNearOptimalCentering`](@ref)
  - Intermediate result type storing the setup data for Near Optimal Centering. [`NearOptimalSetup`](@ref)

#### Risk budgeting optimisations

These optimisations attempt to achieve weight values according to a risk budget vector. This vector can be provided on a per asset or per factor basis.

- ::: details Budget targets
  - Asset-level Risk Budgeting algorithm. [`AssetRiskBudgeting`](@ref)
  - ::: details Fromulations
    - Log-barrier formulation for Risk Budgeting. [`LogRiskBudgeting`](@ref)
    - Mixed-integer formulation for Risk Budgeting. [`MixedIntegerRiskBudgeting`](@ref)
  - Factor-level Risk Budgeting algorithm. [`FactorRiskBudgeting`](@ref)
- ::: details Optimisation estimators
  - Risk Budgeting [`RiskBudgeting`](@ref) returns a [`RiskBudgetingResult`](@ref)
  - ::: details Relaxed Risk Budgeting [`RelaxedRiskBudgeting`](@ref) returns a [`RiskBudgetingResult`](@ref)
    - Basic Relaxed Risk Budgeting formulation. [`BasicRelaxedRiskBudgeting`](@ref)
    - Regularised Relaxed Risk Budgeting formulation. [`RegularisedRelaxedRiskBudgeting`](@ref)
    - Regularised and penalised Relaxed Risk Budgeting formulation. [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)

#### Traditional optimisation features

- Abstract supertype for custom JuMP objective implementations. [`CustomJuMPObjective`](@ref)
- Abstract supertype for custom JuMP constraint implementations. [`CustomJuMPConstraint`](@ref)
- Estimator for portfolio weight bounds constraints. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- ::: details Budget
  - ::: details Directionality
    - Long
    - Short
  - ::: details Type
    - Exact
    - Specifies the portfolio budget constraint as a closed interval $[\mathrm{lb}, \mathrm{ub}]$ on the sum of weights. [`BudgetRange`](@ref)
- ::: details Estimator for buy-in threshold portfolio constraints. [`ThresholdEstimator`](@ref) and [`Threshold`](@ref)
  - ::: details Directionality
    - Long
    - Short
  - ::: details Type
    - Asset
    - Estimator for constructing asset set membership matrices from asset groupings. [`AssetSetsMatrixEstimator`](@ref)
- Container for one or more linear constraint equations to be parsed and converted into constraint matrices. [`LinearConstraintEstimator`](@ref) and [`LinearConstraint`](@ref)
- Estimator type for centrality-based analysis. [`CentralityEstimator`](@ref)
- ::: details Cardinality
  - Asset
  - Container for one or more linear constraint equations to be parsed and converted into constraint matrices. [`LinearConstraintEstimator`](@ref) and [`LinearConstraint`](@ref)
  - Set(s)
  - Container for one or more linear constraint equations to be parsed and converted into constraint matrices. [`LinearConstraintEstimator`](@ref) and [`LinearConstraint`](@ref)
- Estimator for turnover portfolio constraints. [`TurnoverEstimator`](@ref) and [`Turnover`](@ref)
- Estimator for portfolio transaction fees constraints. [`FeesEstimator`](@ref) and [`Fees`](@ref)
- Tracking error result type. [`TrackingError`](@ref)
- Estimator for generating integer phylogeny-based constraints. [`IntegerPhylogenyEstimator`](@ref) and [`SemiDefinitePhylogenyEstimator`](@ref)
- ::: details Portfolio returns
  - ::: details Arithmetic [`ArithmeticReturn`](@ref)
    - Represents a box uncertainty set for risk or prior statistics in portfolio optimisation. [`BoxUncertaintySet`](@ref), [`BoxUncertaintySetAlgorithm`](@ref), [`EllipsoidalUncertaintySet`](@ref), and [`EllipsoidalUncertaintySetAlgorithm`](@ref)
    - Custom expected returns vector
  - Logarithmic [`LogarithmicReturn`](@ref)
- ::: details Risk vector scalarisation
  - Scalariser that combines multiple risk measures using a weighted sum. [`SumScalariser`](@ref)
  - Scalariser that selects the risk expression whose scaled value is the largest. [`MaxScalariser`](@ref)
  - Scalariser that aggregates multiple risk measures using the log-sum-exp function. [`LogSumExpScalariser`](@ref)
- Custom constraint
- Number of effective assets
- ::: details Regularisation penalty
  - L1
  - L2-norm regularisation term added to the optimisation objective. [`L2Regularisation`](@ref)
  - Lp-norm regularisation term added to the optimisation objective. [`LpRegularisation`](@ref)
  - L-Inf
- ::: details Weight-norm constraints

  Where a regularisation penalty prices a norm in the objective, these bound it instead.

  - L2 (`l2c`)
  - Lp (`lpc`)
  - L-Inf (`linfc`)
- Estimator type for normalised constant relative risk aversion (CRRA) OWA weights. [`NormalisedConstantRelativeRiskAversion`](@ref)
- Scalariser that selects the risk expression whose scaled value is the largest. [`MinScalariser`](@ref)

### Clustering optimisation

Clustering optimisations make use of asset relationships to either minimise the risk exposure by breaking the asset universe into subsets which are hierarchically or individually optimised.

- Base configuration for hierarchical clustering-based portfolio optimisers. [`HierarchicalOptimiser`](@ref)

#### Hierarchical clustering optimisation

These optimisations minimise risk by hierarchically splitting the asset universe into subsets, computing the risk of each subset, and combining them according to their hierarchy.

- Hierarchical Risk Parity [`HierarchicalRiskParity`](@ref) returns a [`HierarchicalResult`](@ref)
- Hierarchical Equal Risk Contribution [`HierarchicalEqualRiskContribution`](@ref) returns a [`HierarchicalResult`](@ref)

##### Hierarchical clustering optimisation features

- Estimator for portfolio weight bounds constraints. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- Estimator for portfolio transaction fees constraints. [`FeesEstimator`](@ref) and [`Fees`](@ref)
- ::: details Risk vector scalarisation
  - Scalariser that combines multiple risk measures using a weighted sum. [`SumScalariser`](@ref)
  - Scalariser that selects the risk expression whose scaled value is the largest. [`MaxScalariser`](@ref)
  - Scalariser that aggregates multiple risk measures using the log-sum-exp function. [`LogSumExpScalariser`](@ref)
- ::: details Weight finalisers
  - Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)
  - ::: details Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)
    - Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
    - Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

#### Schur complementary optimisation

Schur complementary hierarchical risk parity provides a bridge between mean variance optimisation and hierarchical risk parity by using an interpolation parameter. It converges to hierarchical risk parity, and approximates mean variance by adjusting this parameter. It uses the Schur complement to adjust the weights of a portfolio according to how much more useful information is gained by assigning more weight to a group of assets.

- Schur Complementary Hierarchical Risk Parity [`SchurComplementHierarchicalRiskParity`](@ref) returns a [`SchurComplementHierarchicalRiskParityResult`](@ref)
- ::: details Parameters for the Schur Complement step of SCHRP. [`SchurComplementParams`](@ref)
  - Monotonic Schur Complement algorithm variant for SCHRP. [`MonotonicSchurComplement`](@ref)
  - Non-monotonic Schur Complement algorithm variant for SCHRP. [`NonMonotonicSchurComplement`](@ref)

##### Schur complementary optimisation features

- Estimator for portfolio weight bounds constraints. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- Estimator for portfolio transaction fees constraints. [`FeesEstimator`](@ref) and [`Fees`](@ref)
- ::: details Weight finalisers
  - Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)
  - ::: details Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)
    - Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
    - Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

#### Nested clusters optimisation

Nested clustered optimisation breaks the asset universe of size `N` into `C` smaller subsets and treats every subset as an individual portfolio. The weights assigned to each asset are placed in an `N × C` matrix. In each column, non-zero values correspond to assets assigned to that subset, this means that assets only contribute to the column (and therefore synthetic asset) corresponding to their assigned subset. In other words, each row of the matrix contains a single non-zero value and each row contains as many non-zero values as there are assets in that subset.

From here there are two options:

1. Compute the returns matrix of the synthetic assets directly by multiplying the original `T × N` matrix by the `N × C` matrix of asset weights to produce a `T × C` matrix of predicted returns, where `T` is the number of observations.
2. For each subset perform a cross validation prediction, yielding a vector of returns for that subset. These vectors are then horizontally concatenated into a `Y × C` matrix of cross-validation predicted returns, where `Y ≤ T` because the cross validation may not use the full history.

This matrix of predicted returns is then used by the outer optimisation estimator to generate an optimisation of the synthetic assets. This produces a `C × 1` vector, essentially optimising a portfolio of asset clusters. The final weights are the product of the original `N × C` matrix of asset weights per cluster by the `C × 1` vector of optimal synthetic asset weights to produce the final `N × 1` vector of asset weights.

- Nested Clustered [`NestedClustered`](@ref) returns a [`NestedClusteredResult`](@ref)

#### Nested clusters optimisation features

- Any features supported by the inner and outer estimators.
- Estimator for portfolio weight bounds constraints. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- Estimator for portfolio transaction fees constraints. [`FeesEstimator`](@ref) and [`Fees`](@ref)
- ::: details Weight finalisers
  - Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)
  - ::: details Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)
    - Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
    - Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
- Cross validation predictor for the outer estimator

### Ensemble optimisation

This works similarly to the Nested Clustered estimator, only instead of breaking the asset universe into subsets, a list of inner estimators is provided. The procedure is then exactly the same as the nested clusters optimisation, only instead of an `N × C` matrix of asset weights where each column corresponds to a subset of assets, each column corresponds to a completely independent and isolated inner estimator, which also means there is no enforced sparsity pattern on this matrix.

- Stacking [`Stacking`](@ref) returns a [`StackingResult`](@ref)

#### Ensemble optimisation features

- Any features supported by the inner and outer estimators.
- Estimator for portfolio transaction fees constraints. [`FeesEstimator`](@ref) and [`Fees`](@ref)
- Estimator for portfolio weight bounds constraints. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- ::: details Weight finalisers
  - Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)
  - ::: details Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)
    - Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
    - Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
- Cross validation predictor for the outer estimator

### Subset resampling optimisation

This optimiser takes ideas from [`MultipleRandomised`](@ref) cross validation to randomly sample the asset universe and optimise each sample individually using a given optimiser. The final asset weights are the average weight per asset across all samples, if an asset does not appear in a sample, it is taken to be zero.

- [`SubsetResampling`](@ref) returns a [`SubsetResamplingResult`](@ref)

#### Subset resampling optimisation features

- Any features supported by the inner estimator.
- Estimator for portfolio transaction fees constraints. [`FeesEstimator`](@ref) and [`Fees`](@ref)
- Estimator for portfolio weight bounds constraints. [`WeightBoundsEstimator`](@ref), [`UniformValues`](@ref), and [`WeightBounds`](@ref)
- ::: details Weight finalisers
  - Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)
  - ::: details Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)
    - Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
    - Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
    - Minimises the L2 norm (squared) of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)

### Finite allocation optimisation

Unlike all other estimators, finite allocation does not yield an "optimal" value, but rather the optimal attainable solution based on a finite amount of capital. They use the result of other estimations, the latest prices, and a cash amount.

- ::: details Discrete Allocation portfolio optimiser. [`DiscreteAllocation`](@ref)
  - ::: details Weight finalisers
    - Iteratively projects weights into the feasible region defined by weight bounds. [`IterativeWeightFinaliser`](@ref)
    - ::: details Uses a JuMP optimisation model to enforce weight bounds. [`JuMPWeightFinaliser`](@ref)
      - Minimises the L1 norm of relative weight deviations when enforcing weight bounds. [`RelativeErrorWeightFinaliser`](@ref)
      - Minimises the L2 norm (squared) of relative weight deviations when enforcing weight bounds. [`SquaredRelativeErrorWeightFinaliser`](@ref)
      - Minimises the L1 norm of absolute weight deviations when enforcing weight bounds. [`AbsoluteErrorWeightFinaliser`](@ref)
      - Minimises the L2 norm (squared) of absolute weight deviations when enforcing weight bounds. [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
- Greedy Allocation portfolio optimiser. [`GreedyAllocation`](@ref)
- Problem data fed to a finite allocation optimiser. [`FiniteAllocationInput`](@ref)

## Cross validation

- Prediction on unseen data [`PredictionReturnsResult`](@ref), [`PredictionResult`](@ref), [`MultiPeriodPredictionResult`](@ref), [`PopulationPredictionResult`](@ref) via [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref), [`fit_and_predict`](@ref)
- Union of concrete [`PredictionScorer`](@ref) subtypes and plain functions that score a [`PopulationPredictionResult`](@ref). [`PredictionCrossValScorer`](@ref), [`NearestQuantilePrediction`](@ref), and [`quantile_by_measure`](@ref)
- Run cross-validated portfolio optimisation and return predictions over all folds. [`cross_val_predict`](@ref)
- Fit optimisation estimator `opt` on returns data `rd` and immediately produce a [`PredictionResult`](@ref) for the same data. [`fit_predict`](@ref)
- Return the number of cross-validation splits (folds) that would be produced by `cv` for the given returns data `rd`. [`n_splits`](@ref)
- Find the optimal `(n_folds, n_test_folds)` pair for combinatorial cross-validation by minimising a weighted cost that balances the average training size against the number of test paths. [`optimal_number_folds`](@ref)
- ::: details Split `str` into an array of substrings on occurrences of the delimiter(s) `dlm`. [`split`](@ref) and [`fit_and_predict`](@ref)
  - K-Fold [`KFold`](@ref) returns a [`KFoldResult`](@ref)
  - Combinatorial [`CombinatorialCrossValidation`](@ref) returns a [`CombinatorialCrossValidationResult`](@ref)
  - ::: details Walk forward [`WalkForwardEstimator`](@ref) return a [`WalkForwardResult`](@ref)
    - Implements index-based walk-forward cross-validation for time series, supporting purging and flexible train/test windowing. [`IndexWalkForward`](@ref) and [`DateWalkForward`](@ref)
  - Multiple randomised [`MultipleRandomised`](@ref) returns a [`MultipleRandomisedResult`](@ref)
- ::: details Performs grid search cross-validation for portfolio optimisation estimators. [`search_cross_validation`](@ref)
  - Performs grid search cross-validation for portfolio optimisation estimators. [`GridSearchCrossValidation`](@ref)
  - Randomised search cross-validation estimator for portfolio optimisation. [`RandomisedSearchCrossValidation`](@ref)
  - ::: details Scoring a parameter set [`CrossValidationSearchScorer`](@ref)
    - A [`CrossValidationSearchScorer`](@ref) that selects the parameter set with the highest mean score across cross-validation splits. [`HighestMeanScore`](@ref)
- Wraps a cross-validation scheme and an optional scorer to form a complete optimisation cross-validation pipeline. [`OptimisationCrossValidation`](@ref)
- Abstract supertype for estimators that determine the number of random subsets to draw. [`NumberSubsetsEstimator`](@ref) and [`SubsetSizeEstimator`](@ref)

## [Pipeline](@id catalogue-pipeline)

A [`Pipeline`](@ref) reifies an end-to-end workflow as data: an ordered list of steps run left-to-right over a [`PipelineContext`](@ref), so preprocessing, priors, and the optimiser travel together as one estimator and can be cross-validated or tuned as a unit.

- A reified end-to-end portfolio workflow: an ordered list of steps executed left-to-right over a [`PipelineContext`](@ref). [`Pipeline`](@ref) and [`PipelineResult`](@ref)
- Explicit pipeline step wrapper — used when a step's slots or its routing intent must be stated rather than inferred. [`PipelineStep`](@ref)
- The accumulating blackboard threaded through a pipeline's steps. [`PipelineContext`](@ref)
- The mu/sigma pair held by the `uncertainty` slot of a [`PipelineContext`](@ref). [`PipelineUncertaintySets`](@ref)

## Plotting

Visualising the results is quite a useful way of summarising the portfolio characteristics or evolution. To this extent we provide a few plotting functions with more to come.

- ::: details Simple or compound cumulative returns.
  - Plot the cumulative returns of a portfolio. [`plot_portfolio_cumulative_returns`](@ref)
  - Plot the cumulative returns of individual assets, selecting the most relevant via `N`. [`plot_asset_cumulative_returns`](@ref)
- ::: details Portfolio composition.
  - Plot portfolio composition as a bar chart of asset weights. [`plot_composition`](@ref)
  - ::: details Multi portfolio.
    - Plot portfolio composition as a stacked bar chart. [`plot_stacked_bar_composition`](@ref)
    - Plot portfolio composition as a stacked area chart. [`plot_stacked_area_composition`](@ref)
- ::: details Risk contribution.
  - Plot per-asset risk contribution as a bar chart. [`plot_risk_contribution`](@ref)
  - Plot per-factor risk contribution as a bar chart, including the constant (idiosyncratic) term. [`plot_factor_risk_contribution`](@ref)
- Plot a hierarchical clustering dendrogram with coloured cluster regions. [`plot_dendrogram`](@ref)
- Plot a reordered correlation/covariance heatmap with flanking dendrograms and coloured cluster boxes. [`plot_clusters`](@ref)
- Plot portfolio drawdown over time. [`plot_drawdowns`](@ref)
- Line plot of the rolling maximum drawdown over a sliding window. [`plot_rolling_drawdowns`](@ref)
- Plot a histogram of portfolio returns with vertical risk-measure lines and an optional fitted Normal distribution. [`plot_histogram`](@ref)
- Scatter plot of risk/return measures across a collection of portfolio weight vectors. [`plot_measures`](@ref)
- Line plot of a risk or return measure evaluated over a rolling window of portfolio returns. [`plot_rolling_measure`](@ref)
- Sort a collection of portfolio results by risk (`x`), connect them with a line to trace the efficient frontier, and optionally annotate the minimum-risk and maximum-score portfolios. [`plot_efficient_frontier`](@ref)
- Box plot of per-asset weight distributions across cross-validation folds or population members. [`plot_weight_stability`](@ref)
- Line plot of portfolio turnover (L1 weight change) over time. [`plot_turnover`](@ref)
- Overlay portfolio cumulative returns against one or more benchmark return series from `rd.B`. [`plot_benchmark`](@ref)
- ::: details Moments and priors
  - Bar chart of per-asset expected returns (μ vector). [`plot_mu`](@ref)
  - Bar chart of per-asset volatility (√diag(Σ)). [`plot_sigma`](@ref)
  - Standalone correlation (or covariance) heatmap without clustering or dendrograms. [`plot_correlation`](@ref)
  - Heatmap of the coskewness matrix (N × N²) from a [`HighOrderPrior`](@ref). [`plot_coskewness`](@ref)
  - Eigenvalue spectrum of the cokurtosis matrix (N² × N²) from a [`HighOrderPrior`](@ref). [`plot_cokurtosis`](@ref)
  - Bar chart of eigenvalues of the covariance/correlation matrix, sorted in descending order. [`plot_eigenspectrum`](@ref)
  - Three-panel composite plot summarising a prior result: [`plot_prior`](@ref)
- ::: details Factor models
  - Bar chart of per-factor expected returns (the `f_mu` vector from a factor model prior). [`plot_factor_mu`](@ref)
  - Correlation/covariance heatmap of the factor covariance matrix (`pr.f_sigma`). [`plot_factor_sigma`](@ref)
  - Heatmap of the factor loadings matrix B (assets × factors) from a prior with a regression model. [`plot_factor_loadings`](@ref)
- ::: details Phylogeny
  - Plot the asset network (MST, PMFG, TMFG, or adjacency) as a graph using `GraphRecipes.graphplot`. [`plot_network`](@ref)
  - Bar chart of asset centrality scores, sorted in descending order. [`plot_centrality`](@ref)
- ::: details Cross validation
  - Bar chart of cross-validation scores (one bar per fold or population member). [`plot_cv_scores`](@ref)
  - Four-panel composite plot for a walk-forward cross-validation result: [`plot_cv_dashboard`](@ref)
- ::: details Dashboards
  - Four-panel composite plot for a single optimisation result: [`plot_portfolio_dashboard`](@ref)
  - Bar chart of annualised portfolio performance metrics: annualised return, annualised volatility, Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown %, and CVaR %. [`plot_performance_summary`](@ref)
