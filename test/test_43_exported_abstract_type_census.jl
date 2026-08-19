@testset "Exported abstract type census: the export list is an allow-list" begin
    using PortfolioOptimisers, Test

    #=
    `CLAUDE.md`: "Never export an abstract type unless explicitly told to." The rule was
    stated and gated nowhere, and six abstract types gained an export across the feature
    matrix, prior, constraint generation, asset sets and similarity work before anyone
    counted: `AbstractFeatureMatrixEstimator`, `AbstractPhylogenyFeatureAlgorithm`,
    `AbstractConstraintSpace`, `AbstractFeatureValue`, `AbstractSimilarityMatrixAlgorithm`
    and `AbstractNonNegativeSimilarityMatrixAlgorithm`. The exported abstract surface
    almost doubled, from seven names to thirteen, over five separate pieces of work.

    Neither gate in `test_26_docs.jl` can see an abstract type. `leaf_types` skips one by
    construction (`if !isabstracttype(T) && parentmodule(T) === PortfolioOptimisers`), and
    the "every exported function is accounted for" gate keeps only a `Function`. An
    exported abstract type is in neither population, so all six passed silently.

    This census closes that hole. Every exported abstract type must appear in the
    allow-list below, so an export becomes a deliberate edit to this file and never an
    accident. The list is the seven that held before the feature matrix work started. An
    export is public API: adding an entry is the maintainer's call, not a decision made in
    passing while landing a feature.
    =#
    allowed_export = Set([:AbstractCentralityAlgorithm, :AbstractUncertaintyEpsAlgorithm,
                          :HierarchicalRiskMeasure, :RegimeAdjustedTarget, :RiskMeasure,
                          :TimeDependentCallable, :TimeDependentOptimiserCallable])

    #=
    `public` is the weaker declaration, and `names(PortfolioOptimisers)` returns both, so a
    census that reads that list alone reports four names it should not. These four are
    deliberate: a caller subtypes `CustomJuMPObjective` or `CustomJuMPConstraint` to write a
    custom objective or constraint (ADR 0036), and the two vector aliases are `AbstractVector`
    over those families, which `isabstracttype` also answers `true` for. They are held to
    their own list for the same reason — public is API too.
    =#
    allowed_public = Set([:CustomJuMPConstraint, :CustomJuMPObjective, :VecJuMPConstr,
                          :VecJuMPObj])

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
    The six names stay reachable through the module prefix, which is what an extension needs
    to subtype them, and each keeps its docstring and its `docs/src/api/` entry. Unexported
    is not undocumented. They are pinned by name so the regression cannot come back quietly.
    =#
    for n in (:AbstractFeatureMatrixEstimator, :AbstractPhylogenyFeatureAlgorithm,
              :AbstractConstraintSpace, :AbstractFeatureValue,
              :AbstractSimilarityMatrixAlgorithm, :AbstractNonNegativeSimilarityMatrixAlgorithm)
        @test is_abstract(n)
        @test !Base.isexported(PortfolioOptimisers, n)
        @test n ∉ names(PortfolioOptimisers)
    end
end
