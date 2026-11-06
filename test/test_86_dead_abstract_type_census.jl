#=
`code_health/CodeHealth.jl` holds the one parser and the one reference counter that every census
reads the source text with. The load sits OUTSIDE the `@testset` on purpose: `include` defines
methods, and a method defined inside one top-level statement is not visible to a call in that same
statement. The module wrapper keeps that module's own names out of the worker module.
`test_70_trivial_unit_census.jl` loads it the same way.
=#
module DeadAbstractTypeCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Dead abstract type census: an abstract type that nothing codes against is on the kept list (ADR 0222)" begin
    using PortfolioOptimisers, Test

    #=
    Issue #1868, the last build of the map #1852. An abstract type that no method dispatches
    on, and that no field, type parameter, `Union` or `isa` names, does nothing that its members
    do not do. The research of #1855 found 46 such types, and the verdicts of #1859, #1873,
    #1874, #1875 and #1876 ruled on each one under the keep rule of ADR 0222. Some were removed,
    some gained a dispatch site, and the rest stay. This census stops a new one.

    A type is dead when `CodeHealth.reference_counts` counts no reference to it in `src/` or
    `ext/`. Its own declaration and the declarations of its subtypes are not references. A use in
    `test/`, `docs/` or `examples/` is not a use. `test_70_trivial_unit_census.jl` flags the
    narrower shape of a dead type with one concrete subtype, and holds its own kept list under
    ADR 0168.

    A flagged type is kept only by an entry below with its reason. ADR 0222 keeps a type by
    default, so a dead type is a question for the maintainer, not a type to remove at once:

    1. A type that callers subtype is a `public` extension point with an `# Interfaces` section.
       Its reason starts with "public", and the census checks both parts.
    2. A private type that an ADR names deliberate meets check 4 of the keep rule. Its reason
       names the ADR.
    3. A private type that a verdict kept names a concept of the library. Its reason names the
       verdict.

    ADR 0222 reads a `Union` that spells all the members of a dead type as a defect. Name the
    abstract type in the `Union`, and the type is no longer dead. An entry whose type is no longer
    flagged is stale and fails, so the list can only shrink on its own.
    =#
    CH = DeadAbstractTypeCensusHealth.CodeHealth
    PO = PortfolioOptimisers
    root = normpath(joinpath(@__DIR__, ".."))

    files = String[]
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") && push!(files, joinpath(r, f))
    end
    sort!(files)
    trees = [CH.parse_file(f; root) for f in files]
    refs = CH.reference_counts(trees)

    # The loaded module is the population, so a type that an extension module declares is not
    # in it. No file under `ext/` declares one. A file that does fails here, and the census then
    # reads that module too.
    ext_abstract = String[]
    for (f, t) in zip(files, trees)
        startswith(f, joinpath(root, "ext")) || continue
        CH.walk_ast(t) do e
            e.head === :abstract && push!(ext_abstract, CH.definition_name(e))
            return nothing
        end
    end
    @test ("abstract types declared under ext/", ext_abstract) ==
          ("abstract types declared under ext/", String[])

    declared = CH.declared_types(PO)
    abstract_types = [n for (n, T) in declared if isabstracttype(T)]
    flagged = sort!([n for n in abstract_types if get(refs, n, 0) == 0])

    #! format: off
    kept = Dict{Symbol, String}(
        # Public extension points. A caller subtypes each one through its `# Interfaces` section
        # (ADR 0154), and a caller's method is a use that the census cannot see.
        :AbstractHighOrderPriorEstimator => "public, the fifth prior family that `AbstractPriorEstimator` tells an author to subtype (#531)",
        :AbstractPenalisedRegressionTarget => "public, the family of the penalised regression targets (ADR 0223)",
        :AbstractRiskMeasureSettings => "public, the family of the settings of a risk measure",
        :AbstractSearchCrossValidationResult => "public, the family of the results of a search cross-validation",
        :BaseGerberCovariance => "public, the family of the Gerber covariance estimators",
        :BaseGerberIQCovariance => "public, the family of the Gerber IQ covariance estimators",
        :BaseHierarchicalOptimisationResult => "public, the root of the hierarchical optimisation results (ADR 0011)",
        :BaseSmythBrobyCovariance => "public, the family of the Smyth-Broby covariance estimators (#1874)",
        :BootstrapUncertaintySetEstimator => "public, the family of the bootstrap uncertainty set estimators",
        :HierarchicalOptimisationResult => "public, the family of the hierarchical optimisation results (ADR 0011)",
        :NonJuMPOptimisationResult => "public, the family of the results that keep no JuMP model (ADR 0011)",
        :OptimisationAlgorithm => "public, the family of the optimisation algorithms",
        :OptimisationModelResult => "public, the family of the model results of an optimisation",
        :TimeDependentConstraintCallable => "public, the family of the time-dependent constraint callables (ADR 0030)",
        # Private types that an ADR names deliberate: check 4 of the keep rule (ADR 0222).
        :AbstractCentralityPolarity => "ADR 0048 keeps the fields unbounded by it on purpose",
        :AbstractCrossSectionalRegressionResult => "ADR 0103, a second cross-sectional regression result, a robust or a constrained fit",
        :AbstractRegressionEstimator => "ADR 0103, the umbrella of the regression estimators",
        :AbstractRegressionResult => "ADR 0103, the umbrella of the regression results",
        :BaseJuMPOptimisationResult => "ADR 0011, a second JuMP optimisation result beside `JuMPOptimisationResult`",
        :HierarchicalScalariser => "ADR 0052, the max and sum scalarisers beside `MinScalariser`",
        :NonOptimisationCrossValidationEstimator => "ADR 0143, the family of the cross-validation estimators that are not optimisers",
        # Private types that a verdict of the map #1852 kept (ADR 0222).
        :AbstractCrossValidationScorer => "#1875, the family of the cross-validation scorers, with `CrossValidationSearchScorer` under it (#1876)",
        :AbstractEntropyPoolingViewEstimator => "#1875, the view groups that carry their own significance level",
        :AbstractJuMPResult => "#1874, a second JuMP result carrier, one that keeps the model",
        :AbstractParsingResult => "#1875, the one canonical shape of a parsed equation",
        :AbstractRegressionAlgorithm => "#1875, the umbrella of the five algorithm families of a regression estimator (ADR 0103)",
        :AbstractRegularisationEstimator => "#1875, the family of the norm penalties on the weights",
        :AbstractShrunkExpectedReturnsEstimator => "#1859, a second shrinkage estimator beside `ShrunkExpectedReturns`",
        :InverseMatrixSparsificationAlgorithm => "#1874, a second sparsification beside `LoGo`, a graphical lasso",
        :NonOptimisationCrossValidationResult => "#1875, the result side of the estimator family that ADR 0143 names",
        :ProcessedAttributes => "#1875, the bundle that one optimiser family makes for each `optimise` call",
    )
    #! format: on

    # Floors on the census itself. A parser that counted every name would flag nothing, and the
    # stale check below catches that. A parser that counted no name would flag every type.
    # Measured 2026-10-28: 330 abstract types, of which 31 flagged.
    @test length(abstract_types) > 300
    @test length(flagged) < length(abstract_types) / 5

    # Both directions are named, so a failure says which type moved and which way.
    unkept = [n for n in flagged if !haskey(kept, n)]
    stale = sort!([n for n in keys(kept) if !(n in flagged)])
    @test ("abstract types that nothing codes against, outside the kept list", unkept) ==
          ("abstract types that nothing codes against, outside the kept list", Symbol[])
    @test ("kept list entries no longer flagged", stale) ==
          ("kept list entries no longer flagged", Symbol[])

    # A "public" reason holds only while the type is `public` and carries the section that tells
    # a caller how to subtype it. Every other reason names the ADR or the verdict that keeps it.
    for (n, why) in kept
        if startswith(why, "public")
            doc = string(Base.Docs.doc(getfield(PO, n)))
            @test (n, Base.ispublic(PO, n), CH.interfaces_section(doc) !== nothing) ==
                  (n, true, true)
        else
            @test (n, occursin(r"^(ADR \d{4}|#\d+)", why)) == (n, true)
        end
    end
end
