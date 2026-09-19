@testset "Exported abstract type census: the export list is an allow-list" begin
    using PortfolioOptimisers, Test

    #=
    `CLAUDE.md`: "Never export an abstract type unless explicitly told to." The rule was
    stated and gated nowhere, and six abstract types gained an export across the feature
    matrix, prior, constraint generation, asset sets and similarity work before anyone
    counted: `AbstractFeatureMatrixEstimator`, `AbstractPhylogenyFeatureAlgorithm`,
    `AbstractConstraintSpace`, `AbstractSimilarityMatrixAlgorithm`
    and `AbstractNonNegativeSimilarityMatrixAlgorithm`. The exported abstract surface
    almost doubled, from seven names to thirteen, over five separate pieces of work.

    Neither gate in `test_26_docs.jl` can see an abstract type. `leaf_types` skips one by
    construction (`if !isabstracttype(T) && parentmodule(T) === PortfolioOptimisers`), and
    the "every exported function is accounted for" gate keeps only a `Function`. An
    exported abstract type is in neither population, so all six passed silently.

    This census closes that hole. Every exported abstract type must appear in the
    allow-list below, so an export becomes a deliberate edit to this file and never an
    accident. An export is public API: adding an entry is the maintainer's call, not a
    decision made in passing while landing a feature.

    The list held seven names when the census was written. `TimeDependentCallable` and
    `TimeDependentOptimiserCallable` left it on 2026-08-19, when the time-dependent callable
    family was reclassified (ADR 0030, amendment): the family root moved to
    `AbstractEstimator` and gained a third member, `TimeDependentConstraintCallable`, and
    none of the three is exported. Five names remain.
    =#
    allowed_export = Set([:AbstractCentralityAlgorithm, :AbstractUncertaintyEpsAlgorithm,
                          :HierarchicalRiskMeasure, :RegimeAdjustedTarget, :RiskMeasure])

    #=
    `public` is the weaker declaration, and `names(PortfolioOptimisers)` returns both, so a
    census that reads that list alone reports names it should not. Two are deliberate from
    the start: a caller subtypes `CustomJuMPObjective` or `CustomJuMPConstraint` to write a
    custom objective or constraint (ADR 0036), and the two vector aliases are `AbstractVector`
    over those families, which `isabstracttype` also answers `true` for. ADR 0154 adds a
    second route: an abstract type whose docstring carries a `# Interfaces` section earns a
    `public` declaration on the type and on every verb the section names, one
    promotion-ticket directory at a time (issue #553) — `VectorToScalarMeasure` joined this
    way on 2026-09-17 (issue #1127), the nine `src/01_Base/` types joined the same day (issue
    #1126), the twenty-five `src/05_Moments/` types joined the same day (issue #1130), the five
    `src/03_InputData/` types joined the same day too (issue #1128), the six
    `src/04_MatrixProcessing/` types below joined the same day again (issue #1129),
    `AbstractPreorderBy` from `src/08_Phylogeny/` joined the same day (issue #1131), the
    three `src/09_ConstraintGeneration/` types joined the same day as well (issue #1132), the
    fifteen `src/11_UncertaintySets/` types joined the same day (issue #1134), the three
    `src/10_Prior/` types joined the same day (issue #1133), the four
    `src/16_RiskMeasures/` types below joined the same day too (issue #1137), the seventeen
    `src/17_Optimisation/` types below joined the same day again (issue #1138), the two
    `src/20_AssetSelection.jl` types below joined the same day once more (issue #1139), and
    the four `src/10_Prior/` family types below — `AbstractPriorEstimator`'s own section
    tells an author to subtype one of them — joined on 2026-09-17 too (issue #1146), and
    the five online-selection types below — the rule, the geometry, the set, the slack and
    the price-level statistic — joined on 2026-09-18 with their `# Interfaces` sections
    (issue #1161). They are held to their own list for the same reason — public is API too.
    =#
    allowed_public = Set([:ARCHBootstrapSet, :AbstractAmbiguityRadiusCalibrationAlgorithm,
                          :AbstractAmbiguityTailWeightCalibrationAlgorithm, :AbstractBins,
                          :AbstractCompactRadiusAlgorithm, :AbstractConfidenceUpdate,
                          :AbstractConstraintEstimator, :AbstractConstraintResult,
                          :AbstractConstraintSpace, :AbstractCovarianceEstimator,
                          :AbstractCoverageAlgorithm,
                          :AbstractCrossSectionalRegressionEstimator,
                          :AbstractCrossSectionalTransform,
                          :AbstractCrossSectionalWeightsAlgorithm,
                          :AbstractDeformationCalibrationAlgorithm,
                          :AbstractDenoiseAlgorithm, :AbstractDenoiseEstimator,
                          :AbstractDescriptorEstimator, :AbstractDetoneEstimator,
                          :AbstractEstimatorValueAlgorithm,
                          :AbstractExpectedReturnsEstimator, :AbstractExposureEstimator,
                          :AbstractForecastTarget, :AbstractGapReturnAlgorithm,
                          :AbstractHighOrderPriorEstimator_F,
                          :AbstractLowOrderPriorEstimator_A,
                          :AbstractLowOrderPriorEstimator_AF,
                          :AbstractLowOrderPriorEstimator_F,
                          :AbstractMatrixProcessingAlgorithm,
                          :AbstractMatrixProcessingEstimator,
                          :AbstractNormCeilingCalibrationAlgorithm,
                          :AbstractOptimisationEstimator, :AbstractAllocationSet,
                          :AbstractOnlinePortfolioSelectionAlgorithm,
                          :AbstractOrderedWeightsArrayFunction, :AbstractOrthogonalScaling,
                          :AbstractOrthogonalityMetric, :AbstractPanelField,
                          :AbstractPanelFieldInput, :AbstractPanelFillAlgorithm,
                          :AbstractPartialFitState, :AbstractPassiveAggressiveSlack,
                          :AbstractPosdefEstimator, :AbstractPreorderBy,
                          :AbstractPreviousWeightsSource, :AbstractPriceLevelStatistic,
                          :AbstractPriorEstimator, :AbstractPriorResult,
                          :AbstractProjectionGeometry,
                          :AbstractPriorUncertaintySetEstimator, :AbstractRealisedTarget,
                          :AbstractRedundancyAlgorithm, :AbstractReturnForecastEstimator,
                          :AbstractRiskMeasureSettings,
                          :AbstractSearchCrossValidationResult, :AbstractSelectionRule,
                          :AbstractTrendTest, :AbstractSignificanceCalibrationAlgorithm,
                          :AbstractTimeSeriesRegressionEstimator,
                          :AbstractTrackingAlgorithm, :AbstractUncertaintyKAlgorithm,
                          :AbstractUncertaintySetAlgorithm, :AbstractUncertaintySetClass,
                          :AbstractUncertaintySetEstimator, :AbstractUncertaintySetResult,
                          :AbstractVarianceEstimator, :BaseGerberCovariance,
                          :BaseGerberIQCovariance, :BaseHierarchicalOptimisationResult,
                          :BaseOptimisationEstimator, :BinWidthBins,
                          :BootstrapUncertaintySetEstimator, :CokurtosisEstimator,
                          :CoskewnessEstimator, :CrossValidationSearchScorer,
                          :CustomExpectedReturnsValueAlgorithm, :CustomJuMPConstraint,
                          :CustomJuMPObjective, :DynamicAbstractWeights,
                          :FrontierBoundEstimator, :GerberCovarianceAlgorithm,
                          :GerberIQCovarianceAlgorithm, :GerberIQDecayEstimator,
                          :GerberIQEpsEstimator, :GerberIQGammaEstimator,
                          :GerberIQScalerEstimator, :HierarchicalOptimisationResult,
                          :ImpliedVolatilityAlgorithm, :JuMPWeightFinaliserFormulation,
                          :NonFiniteAllocationOptimisationEstimator,
                          :NonFiniteAllocationOptimisationResult,
                          :NonJuMPOptimisationResult, :NormError, :OpinionPoolingAlgorithm,
                          :OptimisationAlgorithm, :OptimisationEstimator,
                          :OptimisationModelResult, :OptimisationResult,
                          :OptimisationReturnCode, :RegimeAdjustedMethod, :Scalariser,
                          :TimeDependentCallable, :TimeDependentConstraintCallable,
                          :TimeDependentOptimiserCallable, :VecJuMPConstr, :VecJuMPObj,
                          :VectorAbstractEstimatorValueAlgorithm, :VectorToScalarMeasure,
                          :WeightFinaliser])

    is_abstract(n) = isdefined(PortfolioOptimisers, n) &&
                     isa(getfield(PortfolioOptimisers, n), Type) &&
                     isabstracttype(getfield(PortfolioOptimisers, n))
    abstract_names = filter(is_abstract, names(PortfolioOptimisers))
    exported = Set(filter(n -> Base.isexported(PortfolioOptimisers, n), abstract_names))
    published = Set(filter(n -> !Base.isexported(PortfolioOptimisers, n), abstract_names))

    # Both directions are named, so a failure says which name moved and which way.
    for (label, found, allowed) in
        (("exported", exported, allowed_export), ("public", published, allowed_public))
        added = sort!(collect(setdiff(found, allowed)))
        dropped = sort!(collect(setdiff(allowed, found)))
        @test ("$label abstract types missing from the allow-list", added) ==
              ("$label abstract types missing from the allow-list", Symbol[])
        @test ("allow-list entries no longer $label", dropped) ==
              ("allow-list entries no longer $label", Symbol[])
    end

    #=
    Two floors on the census itself. A predicate that quietly stopped matching would satisfy
    every assertion above with an empty set on each side, so the shape has to be proven
    alive: the module still defines far more abstract types than it publishes, and the
    published ones are a small minority of them.
    =#
    defined = filter(names(PortfolioOptimisers; all = true)) do n
        return is_abstract(n) &&
               parentmodule(getfield(PortfolioOptimisers, n)) === PortfolioOptimisers
    end
    @test length(defined) > 200
    @test length(exported) < length(defined) / 10

    #=
    The four names stay reachable through the module prefix, which is what an extension
    needs to subtype them, and each keeps its docstring and its private mirror-page entry.
    Unexported is not undocumented. They are pinned by name so the regression cannot come
    back quietly. `AbstractConstraintSpace` left this list on 2026-09-17: its docstring
    carries a `# Interfaces` section, so ADR 0154 promotes it to `public` (issue #1132), and
    it now sits on `allowed_public` above. The time-dependent callable family
    (`TimeDependentCallable`, `TimeDependentConstraintCallable`,
    `TimeDependentOptimiserCallable`) left the same way on 2026-09-17 (issue #1138): the
    2026-08-19 amendment to ADR 0030 kept the family unexported by design, but did not
    anticipate ADR 0154 -- the root's own `# Interfaces` section already states "subtype one
    of the two children, not this root", so promoting the root to `public` alongside its
    children restates that rule as the public contract rather than contradicting it.
    =#
    for n in (:AbstractAssetPanelEstimator, :AbstractPhylogenyFeatureAlgorithm,
              :AbstractSimilarityMatrixAlgorithm, :AbstractNonNegativeSimilarityMatrixAlgorithm)
        @test is_abstract(n)
        @test !Base.isexported(PortfolioOptimisers, n)
        @test n ∉ names(PortfolioOptimisers)
    end
end
