# The Capability Catalogue: the curated grouping of everything a user of
# `PortfolioOptimisers.jl` can construct. See ADR 0040.
#
# WHAT LIVES HERE, AND WHAT DOES NOT
#
# Only the *grouping* is curated: which capabilities exist, how they nest, and
# in what order. A `Cap`'s one-line description is NOT written here -- it is
# derived at build time from the first sentence of the type's docstring (the
# paragraph following `$(DocStringExtensions.TYPEDEF)`), so there is exactly one
# description of each type in the repo and it cannot drift from the API page.
# Pass `label` only where the docstring genuinely reads worse in a bullet list.
#
# This file is plain data with no dependencies, because it is `include`d from
# two different environments: `docs/generate_capability_catalogue.jl` renders it,
# and `test/test_26_docs.jl` checks it for coverage.
#
# COVERAGE
#
# Every concrete leaf subtype of `AbstractEstimator` / `AbstractAlgorithm` must
# appear in at least one `Cap`. Adding a new estimator therefore forces a
# placement decision here rather than letting the page quietly fall behind.

"""
    Cap(names...; label = nothing)

One capability: the names a user writes to reach it. The first name supplies the
description unless `label` overrides it; any further names are companions bound
to the same idea (a type and its verbs, e.g. `Posdef` / `posdef!` / `posdef`).

A name is a `Symbol` resolved against `PortfolioOptimisers` (so a typo fails at
load, not silently at render), or a `String` for a `@ref` target that is not a
bare identifier, such as a method signature.

Normally the description is rendered first and the links are appended after it.
If `label` already contains `@ref` links, it is instead used verbatim and no
links are appended -- for entries whose sentence reads through its links
("Mean-Risk [`MeanRisk`](@ref) returns a [`MeanRiskResult`](@ref)"), where
hoisting them to the end would change what the sentence claims. `names` still
lists every link, so coverage sees them; the check asserts the two agree.
"""
struct Cap
    names::Vector{Union{Symbol, String}}
    label::Union{String, Nothing}
end
function Cap(names::Union{Symbol, String}...; label::Union{String, Nothing} = nothing)
    return Cap(collect(Union{Symbol, String}, names), label)
end

"""
    Note(text, children = [])

A capability with no type behind it -- a choice the library offers that is
spelled as a value rather than a struct ("Predefined number of bins", "Long").
Carries no name, so it is invisible to the coverage check.

May carry children, for a plain bullet that heads a sub-list without collapsing
it. Prefer `Group` when the sub-list is long enough to want folding away.
"""
struct Note
    text::String
    children::Vector
end
Note(text::String) = Note(text, [])

"""
    Group(head, children)

A collapsible sub-list, rendered as a `::: details` block. `head` is either a
`Cap` (the group is itself a capability that contains others) or a plain
`String` (a pure heading, e.g. "Algorithms").
"""
struct Group
    head::Union{Cap, String}
    children::Vector
end

"""
    Section(title, children)

A markdown heading. Nesting depth sets the heading level, so moving a subtree
re-levels it automatically.
"""
struct Section
    title::String
    children::Vector
end

"""
    Prose(text)

A free paragraph between lists -- the narrative connective tissue that explains
what a section is for.
"""
struct Prose
    text::String
end

"""
    NOT_A_FEATURE

Exported functions that deliberately have no catalogue entry, each with the
reason it is out of scope.

Checked in both directions: a name here that is no longer exported fails just as
loudly as an exported function that is neither catalogued nor listed. That is
what stops this from becoming a denylist that quietly rots -- the usual failure
of the hand-maintained page this one replaced.

  - `:alias` -- a short constructor form for a measure that is already
    catalogued under its full name; the alias table in the risk-measure guide
    is the place to look these up.
  - `:base_overload` -- an overload of a `Base` / `Statistics` / `StatsAPI`
    verb. Narrated in section prose ("Overloads `Statistics.mean`") rather than
    listed as a capability, because the capability is the estimator passed to
    it, not the verb.
  - `:trait` -- a dispatch predicate the library asks about a risk measure
    while building a problem. Extension authors implement these; callers do not
    call them.
  - `:internal` -- exported for extension authors, not for callers.
"""
const NOT_A_FEATURE = Dict{Symbol, Symbol}(
                                           # Alias constructors (`src/25_Aliases.jl`).
                                           :DnCov => :alias, :DtCov => :alias,
                                           :ECM => :alias, :ELM => :alias, :FLM => :alias,
                                           :FTCM => :alias, :FTLM => :alias, :KT => :alias,
                                           :MAD => :alias, :PrCov => :alias, :SCM => :alias,
                                           :SKT => :alias, :SLM => :alias, :SSK => :alias,
                                           :TLM => :alias, :OWA_CVaR => :alias,
                                           :OWA_CVaR_RG => :alias, :OWA_GMD => :alias,
                                           :OWA_LMoment => :alias, :OWA_RG => :alias,
                                           :OWA_TG => :alias, :OWA_TG_RG => :alias,
                                           :OWA_WR => :alias,
                                           # Overloads of foreign verbs; the estimator is the capability, not the verb.
                                           :cov => :base_overload, :cor => :base_overload,
                                           :mean => :base_overload, :var => :base_overload,
                                           :std => :base_overload, :fit => :base_overload,
                                           :predict => :base_overload,
                                           # Dispatch predicates consulted while assembling a problem.
                                           :bounds_returns_estimator => :trait,
                                           :bounds_risk_measure => :trait,
                                           :no_bounds_risk_measure => :trait,
                                           :no_risk_expr_risk_measure => :trait,
                                           :no_bounds_no_risk_expr_risk_measure => :trait,
                                           :supported_risk_measures => :trait,
                                           :supports_risk_measure => :trait,
                                           # Extension-author plumbing.
                                           :concrete_typed_array => :internal)

"""
    CATALOGUE

The catalogue itself, in reading order.
"""
const CATALOGUE = [Section("Core abstractions",
                           [Prose("Every component is an **Estimator** (a configuration encoding a method and its hyperparameters), an **Algorithm** (a behaviour selector consumed through an Estimator), or a **Result** (computed output). Estimators and Algorithms are what you choose; Results are what you get back."),
                            Prose("Because every struct is immutable, runtime values are propagated down a composed estimator tree by rebuilding it."),
                            Cap(:factory)]),
                   Section("Preprocessing",
                           [Cap(:prices_to_returns, :ReturnsResult), Cap(:PricesResult),
                            Group(Cap(:PricesToReturns, :fit_preprocessing,
                                      :apply_preprocessing),
                                  [Cap(:MissingDataFilter, :MissingDataFilterResult),
                                   Cap(:Imputer, :ImputerResult)]),
                            Group("Asset selection",
                                  [Cap(:ScoreSelector, :ZeroVarianceFilter,
                                       :CompleteAssetSelector), Cap(:RedundancySelector),
                                   Cap(:AssetSelectorResult),
                                   Group("Selection rules",
                                         [Cap(:QuantileRule), Cap(:RankRule),
                                          Cap(:ThresholdRule)])]),
                            Cap(:train_test_split, :TrainTestSplit, :TrainTestSplitResult),
                            Cap(:returns_result_picker)]),
                   Section("Matrix processing",
                           [Cap(:Posdef, :posdef!, :posdef),
                            Group(Cap(:Denoise, :denoise!, :denoise),
                                  [Cap(:SpectralDenoise), Cap(:FixedDenoise),
                                   Cap(:ShrunkDenoise)]), Cap(:Detone, :detone!, :detone),
                            Cap(:MatrixProcessing, :matrix_processing!,
                                :matrix_processing_step!, :matrix_processing)]),
                   Section("Regression models",
                           [Prose("Factor prior models and implied volatility use [`regression`](@ref) in their estimation, which return a [`Regression`](@ref) object."),
                            Section("Regression targets",
                                    [Cap(:LinearModel), Cap(:GeneralisedLinearModel)]),
                            Section("Regression types",
                                    [Group(Cap(:StepwiseRegression),
                                           [Group("Algorithms",
                                                  [Cap(:ForwardSelection),
                                                   Cap(:BackwardElimination)]),
                                            Group("Selection criteria",
                                                  [Cap(:PValue), Cap(:AIC), Cap(:AICC),
                                                   Cap(:BIC), Cap(:RSquared),
                                                   Cap(:AdjustedRSquared)])]),
                                     Group(Cap(:DimensionReductionRegression),
                                           [Cap(:PCA), Cap(:PPCA)])])]),
                   Section("Moment estimation",
                           [Section("Expected returns",
                                    [Prose("Overloads `Statistics.mean`."),
                                     Cap(:SimpleExpectedReturns),
                                     Cap(:EquilibriumExpectedReturns),
                                     Cap(:ExcessExpectedReturns),
                                     Group(Cap(:ShrunkExpectedReturns),
                                           [Group("Algorithms",
                                                  [Cap(:JamesStein; label = "James-Stein"),
                                                   Cap(:BayesStein; label = "Bayes-Stein"),
                                                   Cap(:BodnarOkhrinParolya;
                                                       label = "Bodnar-Okhrin-Parolya")]),
                                            Group("Targets: all algorithms can have any of the following targets",
                                                  [Cap(:GrandMean; label = "Grand Mean"),
                                                   Cap(:VolatilityWeighted;
                                                       label = "Volatility Weighted"),
                                                   Cap(:MeanSquaredError;
                                                       label = "Mean Squared Error")])]),
                                     Cap(:StandardDeviationExpectedReturns),
                                     Cap(:VarianceExpectedReturns),
                                     Cap(:MedianExpectedReturns),
                                     Cap(:CustomValueExpectedReturns),
                                     Cap(:WindowedExpectedReturns)]),
                            Section("Variance and standard deviation",
                                    [Prose("Overloads `Statistics.var` and `Statistics.std`."),
                                     Cap(:SimpleVariance), Cap(:WindowedVariance)]),
                            Section("Covariance and correlation",
                                    [Prose("Overloads `Statistics.cov` and `Statistics.cor`."),
                                     Cap(:GeneralCovariance),
                                     Group(Cap(:Covariance),
                                           [Cap(:FullMoment), Cap(:SemiMoment)]),
                                     Group(Cap(:GerberCovariance),
                                           [Cap(:Gerber0), Cap(:Gerber1), Cap(:Gerber2)]),
                                     Group(Cap(:SmythBrobyCovariance),
                                           [Cap(:SmythBroby0), Cap(:SmythBroby1),
                                            Cap(:SmythBroby2), Cap(:SmythBrobyGerber0),
                                            Cap(:SmythBrobyGerber1),
                                            Cap(:SmythBrobyGerber2), Cap(:SmythBrobyCount0),
                                            Cap(:SmythBrobyCount1), Cap(:SmythBrobyCount2)]),
                                     Group(Cap(:GerberIQCovariance;
                                               label = "Gerber Information Quality [`GerberIQCovariance`](@ref) with custom variance, demeaning, temporal decay and numerator + denominator estimators"),
                                           [Cap(:BasicGerberIQ), Cap(:PartialGerberIQ),
                                            Cap(:FullGerberIQ), Cap(:ExpGerberIQDecay),
                                            Cap(:AssetVolatilityGerberIQScaler)]),
                                     Cap(:DistanceCovariance),
                                     Cap(:LowerTailDependenceCovariance),
                                     Group("Rank covariances",
                                           [Cap(:KendallCovariance),
                                            Cap(:SpearmanCovariance)]),
                                     Group(Cap(:MutualInfoCovariance),
                                           [Group(Cap(:BinWidthBins),
                                                  [Cap(:Knuth;
                                                       label = "Knuth's optimal bin width"),
                                                   Cap(:FreedmanDiaconis;
                                                       label = "Freedman Diaconis bin width"),
                                                   Cap(:Scott; label = "Scott's bin width")]),
                                            Cap(:HacineGharbiRavier),
                                            Note("Predefined number of bins")]),
                                     Cap(:DenoiseCovariance), Cap(:DetoneCovariance),
                                     Cap(:ProcessedCovariance),
                                     Group(Cap(:ImpliedVolatility),
                                           [Cap(:ImpliedVolatilityPremium),
                                            Cap(:ImpliedVolatilityRegression)]),
                                     Cap(:PortfolioOptimisersCovariance),
                                     Cap(:CorrelationCovariance), Cap(:WindowedCovariance),
                                     Group("Regime-adjusted covariance and variance",
                                           [Cap(:RegimeAdjustedExpWeightedCovariance),
                                            Cap(:RegimeAdjustedExpWeightedVariance),
                                            Group("Regime adjustment methods",
                                                  [Cap(:FirstMomentRegimeAdjusted),
                                                   Cap(:LogRegimeAdjusted),
                                                   Cap(:RootMeanSquaredAdjusted)]),
                                            Group("Shrinkage targets",
                                                  [Cap(:DiagonalTarget),
                                                   Cap(:MahalanobisTarget),
                                                   Cap(:PortfolioTarget)]),
                                            Group("Demeaning",
                                                  [Cap(:MeanCentering),
                                                   Cap(:MedianCentering)]),
                                            Group("Correlation smoothing",
                                                  [Cap(:PairwiseCorrelation),
                                                   Cap(:CorrelationComponents)])])]),
                            Section("[Coskewness](@id catalogue-coskewness)",
                                    [Prose("Implements [`coskewness`](@ref)."),
                                     Group(Cap(:Coskewness),
                                           [Cap(:FullMoment), Cap(:SemiMoment)]),
                                     Cap(:WindowedCoskewness)]),
                            Section("[Cokurtosis](@id catalogue-cokurtosis)",
                                    [Prose("Implements [`cokurtosis`](@ref)."),
                                     Group(Cap(:Cokurtosis),
                                           [Cap(:FullMoment), Cap(:SemiMoment)]),
                                     Cap(:WindowedCokurtosis)]),
                            Section("Windowed moments",
                                    [Prose("Every windowed estimator wraps a base moment estimator and recomputes it over a trailing window, so a moment can vary across the folds of a cross-validation scheme. The window is set by a fixed length or by a [`WindowSizeEstimator`](@ref)."),
                                     Cap(:WindowSizeEstimator)])]),
                   Section("Distance matrices",
                           [Prose("Implements [`distance`](@ref) and [`cor_and_dist`](@ref)."),
                            Cap(:Distance), Cap(:DistanceDistance),
                            Prose("The distance estimators are used together with various distance matrix algorithms."),
                            Cap(:SimpleDistance), Cap(:SimpleAbsoluteDistance),
                            Cap(:LogDistance), Cap(:CorrelationDistance),
                            Group(Cap(:VariationInfoDistance),
                                  [Group(Cap(:BinWidthBins),
                                         [Cap(:Knuth; label = "Knuth's optimal bin width"),
                                          Cap(:FreedmanDiaconis;
                                              label = "Freedman Diaconis bin width"),
                                          Cap(:Scott; label = "Scott's bin width")]),
                                   Cap(:HacineGharbiRavier),
                                   Note("Predefined number of bins")]),
                            Cap(:CanonicalDistance)]),
                   Section("Phylogeny",
                           [Prose("`PortfolioOptimisers.jl` can make use of asset relationships to perform optimisations, define constraints, and compute relatedness characteristics of portfolios."),
                            Section("Clustering",
                                    [Prose("Phylogeny constraints and clustering optimisations make use of clustering algorithms via [`ClustersEstimator`](@ref), [`Clusters`](@ref), and [`clusterise`](@ref). Most clustering algorithms come from [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl)."),
                                     Group(Cap(:OptimalNumberClusters,
                                               :VectorToScalarMeasure),
                                           [Cap(:SecondOrderDifference),
                                            Cap(:SilhouetteScore),
                                            Note("Predefined number of clusters."),
                                            Cap(:optimal_number_clusters)]),
                                     Cap(:assignments),
                                     Section("Hierarchical",
                                             [Cap(:HClustAlgorithm),
                                              Group(Cap(:DBHT, :LoGo, :logo!, :logo;
                                                        label = "Direct Bubble Hierarchical Trees [`DBHT`](@ref) and Local Global sparsification of the covariance matrix [`LoGo`](@ref), [`logo!`](@ref), and [`logo`](@ref)"),
                                                    [Group("Root selection",
                                                           [Cap(:UniqueRoot),
                                                            Cap(:EqualRoot)])])]),
                                     Section("Non-hierarchical",
                                             [Prose("Non-hierarchical clustering algorithms are incompatible with hierarchical clustering optimisations, but they can be used for phylogeny constraints and [`NestedClustered`](@ref) optimisations."),
                                              Cap(:KMeansAlgorithm)])]),
                            Section("Networks",
                                    [Section("Adjacency matrices",
                                             [Prose("Adjacency matrices encode asset relationships either with clustering or graph theory via [`phylogeny_matrix`](@ref) and [`PhylogenyResult`](@ref)."),
                                              Group(Cap(:NetworkEstimator;
                                                        label = "Network adjacency [`NetworkEstimator`](@ref) with custom tree algorithms, covariance, and distance estimators"),
                                                    [Cap(:KruskalTree, :BoruvkaTree,
                                                         :PrimTree),
                                                     Group("Triangulated Maximally Filtered Graph with various similarity matrix estimators",
                                                           [Cap(:MaximumDistanceSimilarity;
                                                                label = "Maximum distance similarity"),
                                                            Cap(:ExponentialSimilarity;
                                                                label = "Exponential similarity"),
                                                            Cap(:GeneralExponentialSimilarity;
                                                                label = "General exponential similarity")])]),
                                              Cap(:ClustersEstimator, :Clusters),
                                              Cap(:NetworkClustersEstimator),
                                              Cap(:ClusterGroups)]),
                                     Section("Centrality and phylogeny measures",
                                             [Group(Cap(:CentralityEstimator;
                                                        label = "Centrality estimator [`CentralityEstimator`](@ref) with custom adjacency matrix estimators (clustering and network) and centrality measures"),
                                                    [Cap(:BetweennessCentrality;
                                                         label = "Betweenness"),
                                                     Cap(:ClosenessCentrality;
                                                         label = "Closeness"),
                                                     Cap(:DegreeCentrality;
                                                         label = "Degree"),
                                                     Cap(:EigenvectorCentrality;
                                                         label = "Eigenvector"),
                                                     Cap(:KatzCentrality; label = "Katz"),
                                                     Cap(:Pagerank; label = "Pagerank"),
                                                     Cap(:RadialityCentrality;
                                                         label = "Radiality"),
                                                     Cap(:StressCentrality;
                                                         label = "Stress")]),
                                              Cap(:centrality_vector),
                                              Cap(:average_centrality),
                                              Cap(:asset_phylogeny)]),
                                     Section("Cluster trees",
                                             [Prose("Hierarchical clustering produces a tree of [`ClusterNode`](@ref)s, walked by [`to_tree`](@ref), [`pre_order`](@ref), and [`is_leaf`](@ref)."),
                                              Cap(:PreorderTreeByID)])])]),
                   Section("Optimisation constraints",
                           [Prose("Non clustering optimisers support a wide range of constraints, while naive and clustering optimisers only support weight bounds. Furthermore, entropy pooling prior supports a variety of views constraints. It is therefore important to provide users with the ability to generate constraints manually and/or programmatically. We therefore provide a wide, robust, and extensible range of types such as [`AbstractEstimatorValueAlgorithm`](@ref) and [`UniformValues`](@ref), and functions that make this easy, fast, and safe."),
                            Prose("Constraints can be defined via their estimators or directly by their result types. Some using estimators need to map key-value pairs to the asset universe, this is done by defining the assets and asset groups in [`AssetSets`](@ref). Internally, `PortfolioOptimisers.jl` uses all the information and calls [`group_to_val!`](@ref), and [`replace_group_by_assets`](@ref) to produce the appropriate arrays."),
                            Cap(:parse_equation, :ParsingResult;
                                label = "Equation parsing"),
                            Cap(:linear_constraints, :LinearConstraintEstimator,
                                :PartialLinearConstraint, :LinearConstraint),
                            Cap(:risk_budget_constraints, :RiskBudgetEstimator,
                                :RiskBudget),
                            Cap(:phylogeny_constraints, :centrality_constraints,
                                :SemiDefinitePhylogenyEstimator, :SemiDefinitePhylogeny,
                                :IntegerPhylogenyEstimator, :IntegerPhylogeny,
                                :CentralityConstraint),
                            Cap(:weight_bounds_constraints, :WeightBoundsEstimator,
                                :WeightBounds), Cap(:AssetSets),
                            Group(Cap(:BudgetEstimator, :BudgetRange;
                                      label = "Budget constraints"),
                                  [Cap(:BudgetCosts), Cap(:BudgetMarketImpact)]),
                            Group("Constraint values [`AbstractEstimatorValueAlgorithm`](@ref)",
                                  [Prose("Where a constraint takes one value per asset or group, these algorithms say how to derive it from data rather than stating it outright."),
                                   Cap(:UniformValues), Cap(:estimator_to_val),
                                   Cap(:MinValue), Cap(:MaxValue), Cap(:MeanValue),
                                   Cap(:MedianValue), Cap(:ModeValue), Cap(:SumValue),
                                   Cap(:ProdValue), Cap(:StdValue), Cap(:VarValue),
                                   Cap(:StandardisedValue), Cap(:NoDefault)]),
                            Group(Cap(:TimeDependent),
                                  [Prose("A time-dependent input takes a different value in each fold of a cross-validation scheme, and is inert outside one."),
                                   Cap(:TimeDependentCallable),
                                   Cap(:TimeDependentOptimiserCallable),
                                   Cap(:TimeDependentContext),
                                   Cap(:PreviousWeightsFunction)]),
                            Cap(:asset_sets_matrix, :AssetSetsMatrixEstimator),
                            Cap(:threshold_constraints, :ThresholdEstimator, :Threshold)]),
                   Section("Prior statistics",
                           [Prose("Many optimisations and constraints use prior statistics computed via [`prior`](@ref)."),
                            Group(Cap(:LowOrderPrior),
                                  [Cap(:EmpiricalPrior), Cap(:FactorPrior),
                                   Group("Black-Litterman",
                                         [Cap(:black_litterman_views),
                                          Cap(:BlackLittermanPrior),
                                          Cap(:BayesianBlackLittermanPrior),
                                          Cap(:FactorBlackLittermanPrior),
                                          Cap(:AugmentedBlackLittermanPrior)]),
                                   Group(Cap(:EntropyPoolingPrior),
                                         [Prose("Entropy pooling reweights the observations so that the posterior satisfies the stated views while staying as close as possible to the prior."),
                                          Cap(:BlackLittermanViews),
                                          Group("View constraint algorithms",
                                                [Cap(:H0_EntropyPooling),
                                                 Cap(:H1_EntropyPooling),
                                                 Cap(:H2_EntropyPooling),
                                                 Cap(:CVaREntropyPooling)]),
                                          Group("Divergence formulations",
                                                [Cap(:ExpEntropyPooling),
                                                 Cap(:LogEntropyPooling)]),
                                          Group("Optimisers",
                                                [Cap(:OptimEntropyPooling),
                                                 Cap(:JuMPEntropyPooling)])]),
                                   Group(Cap(:OpinionPoolingPrior),
                                         [Cap(:LinearOpinionPooling),
                                          Cap(:LogarithmicOpinionPooling)])]),
                            Group(Cap(:HighOrderPrior),
                                  [Cap(:HighOrderPriorEstimator),
                                   Cap(:HighOrderFactorPriorEstimator)])]),
                   Section("Uncertainty sets",
                           [Prose("In order to make optimisations more robust to noise and measurement error, it is possible to define uncertainty sets on the expected returns and covariance. These can be used in optimisations which use either of these two quantities. These are implemented via [`ucs`](@ref), [`mu_ucs`](@ref), and [`sigma_ucs`](@ref)."),
                            Prose("`PortfolioOptimisers.jl` implements two types of uncertainty sets."),
                            Cap(:BoxUncertaintySet, :BoxUncertaintySetAlgorithm),
                            Group(Cap(:EllipsoidalUncertaintySet,
                                      :EllipsoidalUncertaintySetAlgorithm, :k_ucs;
                                      label = "[`EllipsoidalUncertaintySet`](@ref) and [`EllipsoidalUncertaintySetAlgorithm`](@ref) with various algorithms for computing the scaling parameter via [`k_ucs`](@ref)"),
                                  [Cap(:NormalKUncertaintyAlgorithm),
                                   Cap(:GeneralKUncertaintyAlgorithm),
                                   Cap(:ChiSqKUncertaintyAlgorithm),
                                   Note("Predefined scaling parameter")]),
                            Prose("It also implements various estimators for the uncertainty sets, the following two can generate box and ellipsoidal sets."),
                            Cap(:NormalUncertaintySet),
                            Group(Cap(:ARCHUncertaintySet;
                                      label = "Bootstrapping via Autoregressive Conditional Heteroscedasticity [`ARCHUncertaintySet`](@ref) via [`arch`](https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html)"),
                                  [Cap(:CircularBootstrap; label = "Circular"),
                                   Cap(:MovingBootstrap; label = "Moving"),
                                   Cap(:StationaryBootstrap; label = "Stationary")]),
                            Prose("The following estimator can only generate box sets."),
                            Cap(:DeltaUncertaintySet),
                            Prose("Quintile portfolios are expressed as an uncertainty set on the characteristic vector rather than as an optimiser of their own (ADR 0032)."),
                            Group(Cap(:CharacteristicUncertaintySet),
                                  [Cap(:L1UncertaintySet, :L1UncertaintySetAlgorithm),
                                   Cap(:SignedL1UncertaintySet,
                                       :SignedL1UncertaintySetAlgorithm),
                                   Cap(:ActiveAssetsUncertaintyAlgorithm)]),
                            Group("Ellipsoidal set classes",
                                  [Cap(:MuEllipsoidalUncertaintySet),
                                   Cap(:SigmaEllipsoidalUncertaintySet)])]),
                   Section("Turnover",
                           [Prose("The turnover is defined as the element-wise absolute difference between the vector of current weights and a vector of benchmark weights. It can be used as a constraint, method for fee calculation, and risk measure. These are all implemented using [`turnover_constraints`](@ref), [`TurnoverEstimator`](@ref), and [`Turnover`](@ref).")]),
                   Section("Fees",
                           [Prose("Fees are a non-negligible aspect of active investing. As such `PortfolioOptimiser.jl` has the ability to account for them in all optimisations but the naive ones. They can also be used to adjust expected returns calculations via [`calc_fees`](@ref) and [`calc_asset_fees`](@ref)."),
                            Cap(:fees_constraints),
                            Cap(:calc_fixed_fees, :calc_asset_fixed_fees),
                            Group(Cap(:FeesEstimator, :Fees),
                                  [Note("Proportional long"), Note("Proportional short"),
                                   Note("Fixed long"), Note("Fixed short"),
                                   Note("Turnover")])]),
                   Section("Portfolio returns and drawdowns",
                           [Prose("Various risk measures and analyses require the computation of simple and cumulative portfolio returns and drawdowns both in aggregate and per-asset. These are computed by [`calc_net_returns`](@ref), [`calc_net_asset_returns`](@ref), [`cumulative_returns`](@ref), [`drawdowns`](@ref).")]),
                   Section("[Tracking](@id catalogue-tracking)",
                           [Prose("It is often useful to create portfolios that track the performance of an index, indicator, or another portfolio."),
                            Group(Cap(:tracking_benchmark, :TrackingError),
                                  [Cap(:ReturnsTracking), Cap(:WeightsTracking)]),
                            Prose("The error can be computed using different algorithms using [`norm_error`](@ref)."),
                            Group("Norm tracking algorithms",
                                  [Cap(:L1Norm), Cap(:L2Norm), Cap(:SquaredL2Norm),
                                   Cap(:LpNorm), Cap(:LInfNorm)]),
                            Prose("It is also possible to track the error in with risk measures [`RiskTrackingError`](@ref) using [`WeightsTracking`](@ref), which allows for two approaches."),
                            Cap(:DependentVariableTracking),
                            Cap(:IndependentVariableTracking)]),
                   Section("Risk measures",
                           [Prose("`PortfolioOptimisers.jl` provides a wide range of risk measures. These are broadly categorised into two types based on the type of optimisations that support them."),
                            Section("Risk measures for traditional optimisation",
                                    [Prose("These are all subtypes of [`RiskMeasure`](@ref), and are supported by all optimisation estimators."),
                                     Group(Cap(:Variance),
                                           [Group("Traditional optimisations also support:",
                                                  [Note("Risk contribution"),
                                                   Group("Formulations",
                                                         [Cap(:QuadRiskExpr),
                                                          Cap(:SquaredSOCRiskExpr)])])]),
                                     Cap(:StandardDeviation),
                                     Cap(:UncertaintySetVariance;
                                         label = "Uncertainty set variance [`UncertaintySetVariance`](@ref) (same as variance when used in non-traditional optimisation)"),
                                     Group(Cap(:LowOrderMoment),
                                           [Cap(:FirstLowerMoment),
                                            Cap(:MeanAbsoluteDeviation),
                                            Group(Cap(:SecondMoment),
                                                  [Group("Second squared moments",
                                                         [Cap(:FullMoment),
                                                          Cap(:SemiMoment),
                                                          Group("Traditional optimisation formulations",
                                                                [Cap(:QuadRiskExpr),
                                                                 Cap(:SquaredSOCRiskExpr),
                                                                 Cap(:RSOCRiskExpr)])]),
                                                   Group(Cap(:SOCRiskExpr),
                                                         [Cap(:FullMoment),
                                                          Cap(:SemiMoment)])])]),
                                     Group(Cap(:Kurtosis),
                                           [Note("Actual kurtosis",
                                                 [Group("FullMoment and semi-kurtosis are supported in traditional optimisers via the `kt` field. Risk calculation uses",
                                                        [Cap(:FullMoment),
                                                         Cap(:SemiMoment)]),
                                                  Group("Traditional optimisation formulations",
                                                        [Cap(:QuadRiskExpr),
                                                         Cap(:SquaredSOCRiskExpr),
                                                         Cap(:RSOCRiskExpr)])]),
                                            Group(Cap(:SOCRiskExpr),
                                                  [Cap(:FullMoment), Cap(:SemiMoment)])]),
                                     Group(Cap(:NegativeSkewness),
                                           [Group("Squared negative skewness",
                                                  [Group("FullMoment and semi-skewness are supported in traditional optimisers via the `sk` and `V` fields. Risk calculation uses",
                                                         [Cap(:FullMoment),
                                                          Cap(:SemiMoment)]),
                                                   Group("Traditional optimisation formulations",
                                                         [Cap(:QuadRiskExpr),
                                                          Cap(:SquaredSOCRiskExpr)]),
                                                   Cap(:SOCRiskExpr)])]),
                                     Group(Cap(:ValueatRisk),
                                           [Group("Traditional optimisation formulations",
                                                  [Cap(:MIPValueatRisk),
                                                   Cap(:DistributionValueatRisk)])]),
                                     Group(Cap(:ValueatRiskRange),
                                           [Group("Traditional optimisation formulations",
                                                  [Cap(:MIPValueatRisk),
                                                   Cap(:DistributionValueatRisk)])]),
                                     Cap(:DrawdownatRisk), Cap(:ConditionalValueatRisk),
                                     Cap(:DistributionallyRobustConditionalValueatRisk;
                                         label = "Distributionally Robust Conditional Value at Risk [`DistributionallyRobustConditionalValueatRisk`](@ref) (same as conditional value at risk when used in non-traditional optimisation)"),
                                     Cap(:ConditionalValueatRiskRange),
                                     Cap(:DistributionallyRobustConditionalValueatRiskRange;
                                         label = "Distributionally Robust Conditional Value at Risk Range [`DistributionallyRobustConditionalValueatRiskRange`](@ref) (same as conditional value at risk range when used in non-traditional optimisation)"),
                                     Cap(:ConditionalDrawdownatRisk),
                                     Cap(:DistributionallyRobustConditionalDrawdownatRisk;
                                         label = "Distributionally Robust Conditional Drawdown at Risk [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)(same as conditional drawdown at risk when used in non-traditional optimisation)"),
                                     Cap(:EntropicValueatRisk),
                                     Cap(:EntropicValueatRiskRange),
                                     Cap(:EntropicDrawdownatRisk),
                                     Cap(:RelativisticValueatRisk),
                                     Cap(:RelativisticValueatRiskRange),
                                     Cap(:RelativisticDrawdownatRisk),
                                     Group("Ordered Weights Array",
                                           [Group("Risk measures",
                                                  [Cap(:OrderedWeightsArray),
                                                   Cap(:OrderedWeightsArrayRange)]),
                                            Group("Traditional optimisation formulations",
                                                  [Cap(:ExactOrderedWeightsArray),
                                                   Cap(:ApproxOrderedWeightsArray),
                                                   Cap(:OWAJuMP)]),
                                            Group("One-call OWA measures",
                                                  [Cap(:OrderedWeightsArrayConditionalValueatRisk),
                                                   Cap(:OrderedWeightsArrayConditionalValueatRiskRange),
                                                   Cap(:OrderedWeightsArrayTailGini),
                                                   Cap(:OrderedWeightsArrayTailGiniRange)]),
                                            Group("Array functions",
                                                  [Cap(:owa_gmd;
                                                       label = "Gini Mean Difference"),
                                                   Cap(:owa_wr;
                                                       label = "Worst Realisation"),
                                                   Cap(:owa_rg; label = "Range"),
                                                   Cap(:owa_cvar;
                                                       label = "Conditional Value at Risk"),
                                                   Cap(:owa_wcvar;
                                                       label = "Weighted Conditional Value at Risk"),
                                                   Cap(:owa_cvarrg;
                                                       label = "Conditional Value at Risk Range"),
                                                   Cap(:owa_wcvarrg;
                                                       label = "Weighted Conditional Value at Risk Range"),
                                                   Cap(:owa_tg; label = "Tail Gini"),
                                                   Cap(:owa_tgrg;
                                                       label = "Tail Gini Range"),
                                                   Group("Linear moments (L-moments)",
                                                         [Cap(:owa_l_moment),
                                                          Group(Cap(:owa_l_moment_crm),
                                                                [Group("L-moment combination formulations",
                                                                       [Group(Cap(:MaximumEntropy),
                                                                              [Cap(:ExponentialConeEntropy),
                                                                               Cap(:RelativeEntropy)]),
                                                                        Cap(:MinimumSquaredDistance),
                                                                        Cap(:MinimumSumSquares)])])])])]),
                                     Cap(:AverageDrawdown), Cap(:UlcerIndex),
                                     Cap(:MaximumDrawdown),
                                     Group(Cap(:BrownianDistanceVariance),
                                           [Group("Traditional optimisation formulations",
                                                  [Group("Distance matrix constraint formulations",
                                                         [Cap(:NormOneConeBrownianDistanceVariance),
                                                          Cap(:IneqBrownianDistanceVariance)]),
                                                   Group("Risk formulation",
                                                         [Cap(:QuadRiskExpr),
                                                          Cap(:RSOCRiskExpr)])])]),
                                     Cap(:WorstRealisation), Cap(:Range),
                                     Cap(:TurnoverRiskMeasure),
                                     Group(Cap(:TrackingRiskMeasure),
                                           [Cap(:L1Norm), Cap(:L2Norm), Cap(:SquaredL2Norm),
                                            Cap(:LpNorm), Cap(:LInfNorm)]),
                                     Group("Risk Tracking Risk Measure",
                                           [Cap(:DependentVariableTracking),
                                            Cap(:IndependentVariableTracking)]),
                                     Cap(:PowerNormValueatRisk),
                                     Cap(:PowerNormValueatRiskRange),
                                     Cap(:PowerNormDrawdownatRisk),
                                     Cap(:GenericValueatRiskRange),
                                     Cap(:RiskTrackingRiskMeasure), Cap(:NoRisk),
                                     Group("Risk measure settings",
                                           [Prose("Every risk measure carries a settings object saying how it enters the problem: as the objective, as a constraint with an upper bound, and with what scale."),
                                            Cap(:RiskMeasureSettings),
                                            Cap(:HierarchicalRiskMeasureSettings),
                                            Cap(:MaxRiskMeasureSettings)])]),
                            Section("Risk measures for hierarchical optimisation",
                                    [Prose("These are all subtypes of [`HierarchicalRiskMeasure`](@ref), and are only supported by hierarchical optimisation estimators."),
                                     Group(Cap(:HighOrderMoment),
                                           [Cap(:ThirdLowerMoment),
                                            Cap(:StandardisedHighOrderMoment,
                                                :ThirdLowerMoment),
                                            Group(Cap(:FourthMoment),
                                                  [Cap(:FullMoment), Cap(:SemiMoment)]),
                                            Group(Cap(:StandardisedHighOrderMoment,
                                                      :FourthMoment),
                                                  [Cap(:FullMoment), Cap(:SemiMoment)])]),
                                     Cap(:RelativeDrawdownatRisk),
                                     Cap(:RelativeConditionalDrawdownatRisk),
                                     Cap(:RelativeEntropicDrawdownatRisk),
                                     Cap(:RelativeRelativisticDrawdownatRisk),
                                     Cap(:RelativeAverageDrawdown),
                                     Cap(:RelativeUlcerIndex),
                                     Cap(:RelativeMaximumDrawdown),
                                     Cap(:RelativePowerNormDrawdownatRisk), Cap(:RiskRatio),
                                     Cap(:EqualRisk), Cap(:MedianAbsoluteDeviation),
                                     Cap(:VarianceSkewKurtosis), Cap(:EvenMoment),
                                     Cap(:LinearMoment)]),
                            Section("Non-optimisation risk measures",
                                    [Prose("These risk measures are unsuitable for optimisation because they can return negative values. However, they can be used for performance metrics."),
                                     Cap(:MeanReturn), Cap(:ThirdCentralMoment),
                                     Cap(:Skewness), Cap(:ExpectedReturn),
                                     Cap(:ExpectedReturnRiskRatio),
                                     Cap(:MeanReturnRiskRatio),
                                     Cap(:NonOptimisationRiskRatio)])]),
                   Section("Performance metrics",
                           [Cap(:expected_risk), Cap(:number_effective_assets),
                            Group("Risk contribution",
                                  [Cap(:risk_contribution), Cap(:factor_risk_contribution)]),
                            Group(Cap(:expected_return),
                                  [Cap(:ArithmeticReturn; label = "Arithmetic"),
                                   Cap(:LogarithmicReturn; label = "Logarithmic")]),
                            Cap(:expected_risk_from_returns), Cap(:rolling_window_measure),
                            Cap(:sort_by_measure),
                            Cap(:expected_ratio, :expected_risk_ret_ratio),
                            Cap(:expected_sric, :expected_risk_ret_sric),
                            Cap(:brinson_attribution)]),
                   Section("Portfolio optimisation",
                           [Prose("Optimisations are implemented via [`optimise`](@ref). Optimisations consume an estimator and return a result."),
                            Section("Naive",
                                    [Prose("These return a [`NaiveOptimisationResult`](@ref)."),
                                     Cap(:InverseVolatility), Cap(:EqualWeighted),
                                     Cap(:RandomWeighted),
                                     Section("Naive optimisation features",
                                             [Cap(:WeightBoundsEstimator, :UniformValues,
                                                  :WeightBounds),
                                              Group("Weight finalisers",
                                                    [Cap(:IterativeWeightFinaliser),
                                                     Group(Cap(:JuMPWeightFinaliser),
                                                           [Cap(:RelativeErrorWeightFinaliser),
                                                            Cap(:SquaredRelativeErrorWeightFinaliser),
                                                            Cap(:AbsoluteErrorWeightFinaliser),
                                                            Cap(:SquaredAbsoluteErrorWeightFinaliser)])])])]),
                            Section("Traditional",
                                    [Prose("These optimisations are implemented as `JuMP` problems and make use of [`JuMPOptimiser`](@ref), which encodes all supported constraints."),
                                     Section("Objective function optimisations",
                                             [Prose("These optimisations support a variety of objective functions."),
                                              Cap(:Solver), Cap(:JuMPOptimiser),
                                              Group("Objective functions",
                                                    [Cap(:MinimumRisk;
                                                         label = "Minimum risk"),
                                                     Cap(:MaximumUtility;
                                                         label = "Maximum utility"),
                                                     Cap(:MaximumRatio;
                                                         label = "Maximum return over risk ratio"),
                                                     Cap(:MaximumReturn;
                                                         label = "Maximum return")]),
                                              Group(Cap(:MeanRisk, :NearOptimalCentering),
                                                    [Group(Cap(:Frontier),
                                                           [Note("Return based"),
                                                            Note("Risk based"),
                                                            Group(Cap(:FrontierBoundEstimator;
                                                                      label = "Bound spacing"),
                                                                  [Cap(:LinearBound),
                                                                   Cap(:SquareRootBound),
                                                                   Cap(:SquaredBound)])])]),
                                              Group("Optimisation estimators",
                                                    [Cap(:MeanRisk, :MeanRiskResult;
                                                         label = "Mean-Risk [`MeanRisk`](@ref) returns a [`MeanRiskResult`](@ref)"),
                                                     Cap(:NearOptimalCentering,
                                                         :NearOptimalCenteringResult;
                                                         label = "Near Optimal Centering [`NearOptimalCentering`](@ref) returns a [`NearOptimalCenteringResult`](@ref)"),
                                                     Cap(:FactorRiskContribution,
                                                         :FactorRiskContributionResult;
                                                         label = "Factor Risk Contribution [`FactorRiskContribution`](@ref) returns a [`FactorRiskContributionResult`](@ref)")]),
                                              Group(Cap(:NearOptimalCentering;
                                                        label = "Near Optimal Centering formulations"),
                                                    [Cap(:ConstrainedNearOptimalCentering),
                                                     Cap(:UnconstrainedNearOptimalCentering),
                                                     Cap(:NearOptimalSetup)])]),
                                     Section("Risk budgeting optimisations",
                                             [Prose("These optimisations attempt to achieve weight values according to a risk budget vector. This vector can be provided on a per asset or per factor basis."),
                                              Group("Budget targets",
                                                    [Cap(:AssetRiskBudgeting),
                                                     Group("Fromulations",
                                                           [Cap(:LogRiskBudgeting),
                                                            Cap(:MixedIntegerRiskBudgeting)]),
                                                     Cap(:FactorRiskBudgeting)]),
                                              Group("Optimisation estimators",
                                                    [Cap(:RiskBudgeting,
                                                         :RiskBudgetingResult;
                                                         label = "Risk Budgeting [`RiskBudgeting`](@ref) returns a [`RiskBudgetingResult`](@ref)"),
                                                     Group(Cap(:RelaxedRiskBudgeting,
                                                               :RiskBudgetingResult;
                                                               label = "Relaxed Risk Budgeting [`RelaxedRiskBudgeting`](@ref) returns a [`RiskBudgetingResult`](@ref)"),
                                                           [Cap(:BasicRelaxedRiskBudgeting),
                                                            Cap(:RegularisedRelaxedRiskBudgeting),
                                                            Cap(:RegularisedPenalisedRelaxedRiskBudgeting)])])]),
                                     Section("Traditional optimisation features",
                                             [Cap(:CustomJuMPObjective),
                                              Cap(:CustomJuMPConstraint),
                                              Cap(:WeightBoundsEstimator, :UniformValues,
                                                  :WeightBounds),
                                              Group("Budget",
                                                    [Group("Directionality",
                                                           [Note("Long"), Note("Short")]),
                                                     Group("Type",
                                                           [Note("Exact"),
                                                            Cap(:BudgetRange)])]),
                                              Group(Cap(:ThresholdEstimator, :Threshold),
                                                    [Group("Directionality",
                                                           [Note("Long"), Note("Short")]),
                                                     Group("Type",
                                                           [Note("Asset"),
                                                            Cap(:AssetSetsMatrixEstimator)])]),
                                              Cap(:LinearConstraintEstimator,
                                                  :LinearConstraint),
                                              Cap(:CentralityEstimator),
                                              Group("Cardinality",
                                                    [Note("Asset"),
                                                     Cap(:LinearConstraintEstimator,
                                                         :LinearConstraint), Note("Set(s)"),
                                                     Cap(:LinearConstraintEstimator,
                                                         :LinearConstraint)]),
                                              Cap(:TurnoverEstimator, :Turnover),
                                              Cap(:FeesEstimator, :Fees),
                                              Cap(:TrackingError),
                                              Cap(:IntegerPhylogenyEstimator,
                                                  :SemiDefinitePhylogenyEstimator),
                                              Group("Portfolio returns",
                                                    [Group(Cap(:ArithmeticReturn;
                                                               label = "Arithmetic"),
                                                           [Cap(:BoxUncertaintySet,
                                                                :BoxUncertaintySetAlgorithm,
                                                                :EllipsoidalUncertaintySet,
                                                                :EllipsoidalUncertaintySetAlgorithm),
                                                            Note("Custom expected returns vector")]),
                                                     Cap(:LogarithmicReturn;
                                                         label = "Logarithmic")]),
                                              Group("Risk vector scalarisation",
                                                    [Cap(:SumScalariser),
                                                     Cap(:MaxScalariser),
                                                     Cap(:LogSumExpScalariser)]),
                                              Note("Custom constraint"),
                                              Note("Number of effective assets"),
                                              Group("Regularisation penalty",
                                                    [Note("L1"), Cap(:L2Regularisation),
                                                     Cap(:LpRegularisation), Note("L-Inf")]),
                                              Group("Weight-norm constraints",
                                                    [Prose("Where a regularisation penalty prices a norm in the objective, these bound it instead."),
                                                     Note("L2 (`l2c`)"), Note("Lp (`lpc`)"),
                                                     Note("L-Inf (`linfc`)")]),
                                              Cap(:NormalisedConstantRelativeRiskAversion),
                                              Cap(:MinScalariser)])]),
                            Section("Clustering optimisation",
                                    [Prose("Clustering optimisations make use of asset relationships to either minimise the risk exposure by breaking the asset universe into subsets which are hierarchically or individually optimised."),
                                     Cap(:HierarchicalOptimiser),
                                     Section("Hierarchical clustering optimisation",
                                             [Prose("These optimisations minimise risk by hierarchically splitting the asset universe into subsets, computing the risk of each subset, and combining them according to their hierarchy."),
                                              Cap(:HierarchicalRiskParity,
                                                  :HierarchicalResult;
                                                  label = "Hierarchical Risk Parity [`HierarchicalRiskParity`](@ref) returns a [`HierarchicalResult`](@ref)"),
                                              Cap(:HierarchicalEqualRiskContribution,
                                                  :HierarchicalResult;
                                                  label = "Hierarchical Equal Risk Contribution [`HierarchicalEqualRiskContribution`](@ref) returns a [`HierarchicalResult`](@ref)"),
                                              Section("Hierarchical clustering optimisation features",
                                                      [Cap(:WeightBoundsEstimator,
                                                           :UniformValues, :WeightBounds),
                                                       Cap(:FeesEstimator, :Fees),
                                                       Group("Risk vector scalarisation",
                                                             [Cap(:SumScalariser),
                                                              Cap(:MaxScalariser),
                                                              Cap(:LogSumExpScalariser)]),
                                                       Group("Weight finalisers",
                                                             [Cap(:IterativeWeightFinaliser),
                                                              Group(Cap(:JuMPWeightFinaliser),
                                                                    [Cap(:RelativeErrorWeightFinaliser),
                                                                     Cap(:SquaredRelativeErrorWeightFinaliser),
                                                                     Cap(:AbsoluteErrorWeightFinaliser),
                                                                     Cap(:SquaredAbsoluteErrorWeightFinaliser)])])])]),
                                     Section("Schur complementary optimisation",
                                             [Prose("Schur complementary hierarchical risk parity provides a bridge between mean variance optimisation and hierarchical risk parity by using an interpolation parameter. It converges to hierarchical risk parity, and approximates mean variance by adjusting this parameter. It uses the Schur complement to adjust the weights of a portfolio according to how much more useful information is gained by assigning more weight to a group of assets."),
                                              Cap(:SchurComplementHierarchicalRiskParity,
                                                  :SchurComplementHierarchicalRiskParityResult;
                                                  label = "Schur Complementary Hierarchical Risk Parity [`SchurComplementHierarchicalRiskParity`](@ref) returns a [`SchurComplementHierarchicalRiskParityResult`](@ref)"),
                                              Group(Cap(:SchurComplementParams),
                                                    [Cap(:MonotonicSchurComplement),
                                                     Cap(:NonMonotonicSchurComplement)]),
                                              Section("Schur complementary optimisation features",
                                                      [Cap(:WeightBoundsEstimator,
                                                           :UniformValues, :WeightBounds),
                                                       Cap(:FeesEstimator, :Fees),
                                                       Group("Weight finalisers",
                                                             [Cap(:IterativeWeightFinaliser),
                                                              Group(Cap(:JuMPWeightFinaliser),
                                                                    [Cap(:RelativeErrorWeightFinaliser),
                                                                     Cap(:SquaredRelativeErrorWeightFinaliser),
                                                                     Cap(:AbsoluteErrorWeightFinaliser),
                                                                     Cap(:SquaredAbsoluteErrorWeightFinaliser)])])])]),
                                     Section("Nested clusters optimisation",
                                             [Prose("Nested clustered optimisation breaks the asset universe of size `N` into `C` smaller subsets and treats every subset as an individual portfolio. The weights assigned to each asset are placed in an `N × C` matrix. In each column, non-zero values correspond to assets assigned to that subset, this means that assets only contribute to the column (and therefore synthetic asset) corresponding to their assigned subset. In other words, each row of the matrix contains a single non-zero value and each row contains as many non-zero values as there are assets in that subset."),
                                              Prose("From here there are two options:"),
                                              Prose("1. Compute the returns matrix of the synthetic assets directly by multiplying the original `T × N` matrix by the `N × C` matrix of asset weights to produce a `T × C` matrix of predicted returns, where `T` is the number of observations.\n2. For each subset perform a cross validation prediction, yielding a vector of returns for that subset. These vectors are then horizontally concatenated into a `Y × C` matrix of cross-validation predicted returns, where `Y ≤ T` because the cross validation may not use the full history."),
                                              Prose("This matrix of predicted returns is then used by the outer optimisation estimator to generate an optimisation of the synthetic assets. This produces a `C × 1` vector, essentially optimising a portfolio of asset clusters. The final weights are the product of the original `N × C` matrix of asset weights per cluster by the `C × 1` vector of optimal synthetic asset weights to produce the final `N × 1` vector of asset weights."),
                                              Cap(:NestedClustered, :NestedClusteredResult;
                                                  label = "Nested Clustered [`NestedClustered`](@ref) returns a [`NestedClusteredResult`](@ref)")]),
                                     Section("Nested clusters optimisation features",
                                             [Note("Any features supported by the inner and outer estimators."),
                                              Cap(:WeightBoundsEstimator, :UniformValues,
                                                  :WeightBounds),
                                              Cap(:FeesEstimator, :Fees),
                                              Group("Weight finalisers",
                                                    [Cap(:IterativeWeightFinaliser),
                                                     Group(Cap(:JuMPWeightFinaliser),
                                                           [Cap(:RelativeErrorWeightFinaliser),
                                                            Cap(:SquaredRelativeErrorWeightFinaliser),
                                                            Cap(:AbsoluteErrorWeightFinaliser),
                                                            Cap(:SquaredAbsoluteErrorWeightFinaliser)])]),
                                              Note("Cross validation predictor for the outer estimator")])]),
                            Section("Ensemble optimisation",
                                    [Prose("This works similarly to the Nested Clustered estimator, only instead of breaking the asset universe into subsets, a list of inner estimators is provided. The procedure is then exactly the same as the nested clusters optimisation, only instead of an `N × C` matrix of asset weights where each column corresponds to a subset of assets, each column corresponds to a completely independent and isolated inner estimator, which also means there is no enforced sparsity pattern on this matrix."),
                                     Cap(:Stacking, :StackingResult;
                                         label = "Stacking [`Stacking`](@ref) returns a [`StackingResult`](@ref)"),
                                     Section("Ensemble optimisation features",
                                             [Note("Any features supported by the inner and outer estimators."),
                                              Cap(:FeesEstimator, :Fees),
                                              Cap(:WeightBoundsEstimator, :UniformValues,
                                                  :WeightBounds),
                                              Group("Weight finalisers",
                                                    [Cap(:IterativeWeightFinaliser),
                                                     Group(Cap(:JuMPWeightFinaliser),
                                                           [Cap(:RelativeErrorWeightFinaliser),
                                                            Cap(:SquaredRelativeErrorWeightFinaliser),
                                                            Cap(:AbsoluteErrorWeightFinaliser),
                                                            Cap(:SquaredAbsoluteErrorWeightFinaliser)])]),
                                              Note("Cross validation predictor for the outer estimator")])]),
                            Section("Subset resampling optimisation",
                                    [Prose("This optimiser takes ideas from [`MultipleRandomised`](@ref) cross validation to randomly sample the asset universe and optimise each sample individually using a given optimiser. The final asset weights are the average weight per asset across all samples, if an asset does not appear in a sample, it is taken to be zero."),
                                     Cap(:SubsetResampling, :SubsetResamplingResult;
                                         label = "[`SubsetResampling`](@ref) returns a [`SubsetResamplingResult`](@ref)"),
                                     Section("Subset resampling optimisation features",
                                             [Note("Any features supported by the inner estimator."),
                                              Cap(:FeesEstimator, :Fees),
                                              Cap(:WeightBoundsEstimator, :UniformValues,
                                                  :WeightBounds),
                                              Group("Weight finalisers",
                                                    [Cap(:IterativeWeightFinaliser),
                                                     Group(Cap(:JuMPWeightFinaliser),
                                                           [Cap(:RelativeErrorWeightFinaliser),
                                                            Cap(:SquaredRelativeErrorWeightFinaliser),
                                                            Cap(:AbsoluteErrorWeightFinaliser),
                                                            Cap(:SquaredAbsoluteErrorWeightFinaliser)])])])]),
                            Section("Finite allocation optimisation",
                                    [Prose("Unlike all other estimators, finite allocation does not yield an \"optimal\" value, but rather the optimal attainable solution based on a finite amount of capital. They use the result of other estimations, the latest prices, and a cash amount."),
                                     Group(Cap(:DiscreteAllocation),
                                           [Group("Weight finalisers",
                                                  [Cap(:IterativeWeightFinaliser),
                                                   Group(Cap(:JuMPWeightFinaliser),
                                                         [Cap(:RelativeErrorWeightFinaliser),
                                                          Cap(:SquaredRelativeErrorWeightFinaliser),
                                                          Cap(:AbsoluteErrorWeightFinaliser),
                                                          Cap(:SquaredAbsoluteErrorWeightFinaliser)])])]),
                                     Cap(:GreedyAllocation), Cap(:FiniteAllocationInput)])]),
                   Section("Cross validation",
                           [Cap(:PredictionReturnsResult, :PredictionResult,
                                :MultiPeriodPredictionResult, :PopulationPredictionResult,
                                "predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)",
                                :fit_and_predict;
                                label = "Prediction on unseen data [`PredictionReturnsResult`](@ref), [`PredictionResult`](@ref), [`MultiPeriodPredictionResult`](@ref), [`PopulationPredictionResult`](@ref) via [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref), [`fit_and_predict`](@ref)"),
                            Cap(:PredictionCrossValScorer, :NearestQuantilePrediction,
                                :quantile_by_measure), Cap(:cross_val_predict),
                            Cap(:fit_predict), Cap(:n_splits), Cap(:optimal_number_folds),
                            Group(Cap(:split, :fit_and_predict),
                                  [Cap(:KFold, :KFoldResult;
                                       label = "K-Fold [`KFold`](@ref) returns a [`KFoldResult`](@ref)"),
                                   Cap(:CombinatorialCrossValidation,
                                       :CombinatorialCrossValidationResult;
                                       label = "Combinatorial [`CombinatorialCrossValidation`](@ref) returns a [`CombinatorialCrossValidationResult`](@ref)"),
                                   Group(Cap(:WalkForwardEstimator, :WalkForwardResult;
                                             label = "Walk forward [`WalkForwardEstimator`](@ref) return a [`WalkForwardResult`](@ref)"),
                                         [Cap(:IndexWalkForward, :DateWalkForward)]),
                                   Cap(:MultipleRandomised, :MultipleRandomisedResult;
                                       label = "Multiple randomised [`MultipleRandomised`](@ref) returns a [`MultipleRandomisedResult`](@ref)")]),
                            Group(Cap(:search_cross_validation),
                                  [Cap(:GridSearchCrossValidation),
                                   Cap(:RandomisedSearchCrossValidation),
                                   Group(Cap(:CrossValidationSearchScorer;
                                             label = "Scoring a parameter set"),
                                         [Cap(:HighestMeanScore)])]),
                            Cap(:OptimisationCrossValidation),
                            Cap(:NumberSubsetsEstimator, :SubsetSizeEstimator)]),
                   Section("[Pipeline](@id catalogue-pipeline)",
                           [Prose("A [`Pipeline`](@ref) reifies an end-to-end workflow as data: an ordered list of steps run left-to-right over a [`PipelineContext`](@ref), so preprocessing, priors, and the optimiser travel together as one estimator and can be cross-validated or tuned as a unit."),
                            Cap(:Pipeline, :PipelineResult), Cap(:PipelineStep),
                            Cap(:PipelineContext), Cap(:PipelineUncertaintySets)]),
                   Section("Plotting",
                           [Prose("Visualising the results is quite a useful way of summarising the portfolio characteristics or evolution. To this extent we provide a few plotting functions with more to come."),
                            Group("Simple or compound cumulative returns.",
                                  [Cap(:plot_portfolio_cumulative_returns),
                                   Cap(:plot_asset_cumulative_returns)]),
                            Group("Portfolio composition.",
                                  [Cap(:plot_composition),
                                   Group("Multi portfolio.",
                                         [Cap(:plot_stacked_bar_composition),
                                          Cap(:plot_stacked_area_composition)])]),
                            Group("Risk contribution.",
                                  [Cap(:plot_risk_contribution),
                                   Cap(:plot_factor_risk_contribution)]),
                            Cap(:plot_dendrogram), Cap(:plot_clusters),
                            Cap(:plot_drawdowns), Cap(:plot_rolling_drawdowns),
                            Cap(:plot_histogram), Cap(:plot_measures),
                            Cap(:plot_rolling_measure), Cap(:plot_efficient_frontier),
                            Cap(:plot_weight_stability), Cap(:plot_turnover),
                            Cap(:plot_benchmark),
                            Group("Moments and priors",
                                  [Cap(:plot_mu), Cap(:plot_sigma), Cap(:plot_correlation),
                                   Cap(:plot_coskewness), Cap(:plot_cokurtosis),
                                   Cap(:plot_eigenspectrum), Cap(:plot_prior)]),
                            Group("Factor models",
                                  [Cap(:plot_factor_mu), Cap(:plot_factor_sigma),
                                   Cap(:plot_factor_loadings)]),
                            Group("Phylogeny", [Cap(:plot_network), Cap(:plot_centrality)]),
                            Group("Cross validation",
                                  [Cap(:plot_cv_scores), Cap(:plot_cv_dashboard)]),
                            Group("Dashboards",
                                  [Cap(:plot_portfolio_dashboard),
                                   Cap(:plot_performance_summary)])])]
