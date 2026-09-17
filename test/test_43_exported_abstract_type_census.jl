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
    #1126), the twenty-five `src/05_Moments/` types joined the same day (issue #1130), and the
    five `src/03_InputData/` types below joined the same day too (issue #1128). They are held
    to their own list for the same reason — public is API too.
    =#
    allowed_public = Set([:AbstractBins, :AbstractCoverageAlgorithm,
                          :AbstractCovarianceEstimator,
                          :AbstractCrossSectionalRegressionEstimator,
                          :AbstractCrossSectionalTransform,
                          :AbstractCrossSectionalWeightsAlgorithm,
                          :AbstractDescriptorEstimator, :AbstractEstimatorValueAlgorithm,
                          :AbstractExpectedReturnsEstimator, :AbstractExposureEstimator,
                          :AbstractForecastTarget, :AbstractGapReturnAlgorithm,
                          :AbstractOptimisationEstimator, :AbstractPanelField,
                          :AbstractPanelFieldInput, :AbstractPanelFillAlgorithm,
                          :AbstractPartialFitState, :AbstractReturnForecastEstimator,
                          :AbstractTimeSeriesRegressionEstimator,
                          :AbstractVarianceEstimator, :BaseGerberCovariance,
                          :BaseGerberIQCovariance, :BinWidthBins, :CokurtosisEstimator,
                          :CoskewnessEstimator, :CustomExpectedReturnsValueAlgorithm,
                          :CustomJuMPConstraint, :CustomJuMPObjective,
                          :DynamicAbstractWeights, :GerberCovarianceAlgorithm,
                          :GerberIQCovarianceAlgorithm, :GerberIQDecayEstimator,
                          :GerberIQEpsEstimator, :GerberIQGammaEstimator,
                          :GerberIQScalerEstimator, :ImpliedVolatilityAlgorithm,
                          :NonFiniteAllocationOptimisationEstimator, :NormError,
                          :OptimisationEstimator, :RegimeAdjustedMethod, :VecJuMPConstr,
                          :VecJuMPObj, :VectorAbstractEstimatorValueAlgorithm,
                          :VectorToScalarMeasure])

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
    The nine names stay reachable through the module prefix, which is what an extension needs
    to subtype them, and each keeps its docstring and its `docs/src/api/` entry. Unexported
    is not undocumented. They are pinned by name so the regression cannot come back quietly.
    The last three are the time-dependent callable family, whose classification is stated in
    the type tree rather than in the export list.
    =#
    for n in (:AbstractAssetPanelEstimator, :AbstractPhylogenyFeatureAlgorithm,
              :AbstractConstraintSpace, :AbstractSimilarityMatrixAlgorithm,
              :AbstractNonNegativeSimilarityMatrixAlgorithm, :TimeDependentCallable,
              :TimeDependentConstraintCallable, :TimeDependentOptimiserCallable)
        @test is_abstract(n)
        @test !Base.isexported(PortfolioOptimisers, n)
        @test n ∉ names(PortfolioOptimisers)
    end
end
