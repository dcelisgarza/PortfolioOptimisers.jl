#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with:
`walk_ast`, `isdocstring` and `docstring_text`. The load sits OUTSIDE the `@testset` on
purpose: `include` defines methods, and a method defined inside one top-level statement is
not visible to a call in that same statement. The module wrapper keeps that module's own
names out of the worker module. `test_41_constructor_docstring_drift.jl` loads it the same way.
=#
module InterfacesCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Interfaces census: every `# Interfaces` section names methods and types that exist" begin
    using PortfolioOptimisers, Test

    #=
    Issue #1057. `AbstractPanelFieldInput`'s docstring carried a `# Interfaces` section that
    asked a new subtype to implement `panel_input_kind(inp, vals) -> AbstractPanelFieldKind`
    and `panel_write!(Z, kind, vals, cols)`. Neither `AbstractPanelFieldKind` nor
    `panel_write!` existed at the head: both left with #803's rework of the panel (ADR 0102),
    which replaced them by `panel_input_is_static`, `panel_resolve` and `panel_input_field`.
    A `panel_input_kind` did exist, in `09_ConstraintGeneration/06_AssetSetsMatrix.jl`, with
    a different signature and a different job, which is how the stale name survived a name
    search. A reader implementing a new input type from the docstring would have written two
    methods nothing calls and missed one the builder needs.

    Nothing gated it. `test_41` checks a `# Constructors` block against the signatures,
    `test_26` checks `@ref`s and `@docs` entries, and the `# Interfaces` prose is neither.
    This census reads that prose. It walks every docstring under `src/` and `ext/`, takes
    the `# Interfaces` section of each one that carries it, and asserts three things:

    1. every `## \`verb\`` heading names a function the package defines;
    2. every call-shaped bullet, `- \`verb(args...) -> Ret\``, names a function or a type the
       package defines, names the heading's verb when it sits under a verb heading, and every
       capitalised name in a `::T` annotation or in the `-> Ret` return is a type the package
       defines or reaches through its imports;
    3. every abstract type that is the direct supertype of a concrete type the package
       defines carries a `# Interfaces` section, or is on the debt list below.

    The docstring is read from the source text with the parser `code_health/CodeHealth.jl`
    holds, so an interpolated piece (`$(arg_dict[:X])`) is a space, and a section heading
    is a literal line. A name is then resolved against the LOADED module, which is the
    only reader that can tell a function from a type and a defined name from a dead one.
    =#
    CH = InterfacesCensusHealth.CodeHealth
    PO = PortfolioOptimisers
    root = normpath(joinpath(@__DIR__, ".."))

    # ------------------------------------------------------------- the doc side

    # Every docstring under `src/` and `ext/`, as (file, binding, text). The binding is the
    # name an `abstract type` declares, or `nothing` for any other unit; only an abstract
    # type is counted by check 3, but every section is read by checks 1 and 2.
    function abstract_binding(def)
        Meta.isexpr(def, :abstract) || return nothing
        n = def.args[1]
        while isa(n, Expr)
            n = n.args[1]
        end
        return isa(n, Symbol) ? n : nothing
    end

    files = String[]
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") && push!(files, joinpath(r, f))
    end
    sort!(files)

    docs = Tuple{String, Union{Nothing, Symbol}, String}[]
    for f in files
        CH.walk_ast(CH.parse_file(f; root)) do node
            if CH.isdocstring(node)
                push!(docs,
                      (relpath(f, root), abstract_binding(node.args[end]),
                       CH.docstring_text(node)))
            end
            return nothing
        end
    end

    # The lines of the `# Interfaces` section, up to the next top-level heading. Shared with
    # `test/test_66_public_declaration_census.jl` as `CH.interfaces_section`, so the two censuses
    # can never disagree about where the section starts and ends.
    interfaces_section = CH.interfaces_section

    # A verb heading is `## \`verb\``, optionally followed by prose (`## \`k_ucs\` interface`).
    # A bullet is `- \`…\``, and it is call-shaped when the code opens with a name and a paren.
    heading_re = r"^## `([^`]+)`"
    bullet_re = r"^\s*-\s+`([^`]+)`"
    call_re = r"^[A-Za-z_][\w!]*(\.[A-Za-z_][\w!]*)*\("

    # ------------------------------------------------------------- resolution

    # The binding a bare or module-qualified name reaches from inside the package, or
    # `nothing`. `Base.copy` and `Statistics.var` resolve through the module the package
    # imports, so a verb the package extends rather than owns still resolves.
    function resolve(n)
        if isa(n, Symbol)
            return isdefined(PO, n) ? getfield(PO, n) : nothing
        elseif Meta.isexpr(n, :., 2) && isa(n.args[2], QuoteNode)
            m = resolve(n.args[1])
            isa(m, Module) || return nothing
            s = n.args[2].value
            return isdefined(m, s) ? getfield(m, s) : nothing
        end
        return nothing
    end

    # The names in type position under a `::T`, a `<:T`, a `-> Ret` or a `where`. A
    # `curly`, a `Union{…}` and a tuple are walked; a name bound by `where` is skipped.
    function typenames!(out, t, bound)
        if isa(t, Symbol)
            t in bound || push!(out, t)
        elseif Meta.isexpr(t, :.)
            push!(out, t)
        elseif Meta.isexpr(t, :curly) || Meta.isexpr(t, :tuple) || Meta.isexpr(t, :braces)
            for a in t.args
                typenames!(out, a, bound)
            end
        elseif Meta.isexpr(t, :<:) || Meta.isexpr(t, :(::))
            typenames!(out, t.args[end], bound)
        elseif Meta.isexpr(t, :where)
            typenames!(out, t.args[1], bound)
        end
        return out
    end

    # The names in type position across the arguments of a call, positional and keyword.
    function argtypes!(out, call, bound)
        for a in call.args[2:end]
            if Meta.isexpr(a, :parameters)
                argtypes!(out, Expr(:call, :_, a.args...), bound)
            elseif Meta.isexpr(a, :kw) || Meta.isexpr(a, :...)
                argtypes!(out, Expr(:call, :_, a.args[1]), bound)
            elseif Meta.isexpr(a, :(::))
                typenames!(out, a.args[end], bound)
            end
        end
        return out
    end

    #=
    A name in type position is checked only when it is capitalised, so a return written as
    a tuple of values (`-> (basis, key)`) or as `nothing` is prose, not a type. Two
    capitalised names are exempt by convention: `My…` is the reader's own subtype, the
    placeholder the abstract-type template in
    `.github/instructions/julia-docstrings.instructions.md` writes (`MyState`,
    `MyOptimiser`), and `Vararg` is a type constructor rather than a `Type`.
    =#
    is_placeholder(t) = isa(t, Symbol) && (!isuppercase(first(string(t))) ||
                                           occursin(r"^My[A-Z]", string(t)) ||
                                           t === :Vararg)
    is_type(x) = isa(x, Type) || isa(x, TypeVar)

    # ------------------------------------------------------------- the census

    nsections = 0
    nheadings = 0
    nbullets = 0
    dead_headings = Tuple{String, String}[]
    dead_verbs = Tuple{String, String}[]
    dead_types = Tuple{String, String, Any}[]
    unparsable = Tuple{String, String, String}[]
    strays = Tuple{String, Symbol, String}[]
    with_section = Set{Symbol}()

    for (rel, binding, text) in docs
        section = interfaces_section(text)
        section === nothing && continue
        nsections += 1
        binding === nothing || push!(with_section, binding)
        heading = nothing
        for l in section
            m = match(heading_re, l)
            if m !== nothing
                nheadings += 1
                heading = Meta.parse(m.captures[1])
                v = resolve(heading)
                isa(v, Function) || push!(dead_headings, (rel, m.captures[1]))
                continue
            elseif startswith(l, "## ")
                heading = nothing
                continue
            end
            m = match(bullet_re, l)
            m === nothing && continue
            code = m.captures[1]
            occursin(call_re, code) || continue
            nbullets += 1
            ex = try
                Meta.parse(code)
            catch e
                push!(unparsable, (rel, code, sprint(showerror, e)))
                continue
            end
            # `f(x) -> Ret` parses to an `->` whose right side is a block holding `Ret`;
            # `f(x)::Ret` to a `::`; either may sit under a `where`.
            ret = nothing
            if Meta.isexpr(ex, :->)
                ret = ex.args[2]
                if Meta.isexpr(ret, :block)
                    ret = filter(a -> !isa(a, LineNumberNode), ret.args)[end]
                end
                ex = ex.args[1]
            end
            bound = Symbol[]
            while Meta.isexpr(ex, :where)
                for b in ex.args[2:end]
                    push!(bound, isa(b, Symbol) ? b : b.args[1])
                end
                ex = ex.args[1]
            end
            if Meta.isexpr(ex, :(::), 2)
                ret = ex.args[2]
                ex = ex.args[1]
            end
            if !Meta.isexpr(ex, :call)
                push!(unparsable, (rel, code, "not a call"))
                continue
            end
            verb = ex.args[1]
            v = resolve(verb)
            isa(v, Function) || isa(v, Type) || push!(dead_verbs, (rel, code))
            if heading !== nothing && verb != heading
                push!(strays, (rel, Symbol(heading), code))
            end
            ts = argtypes!(Any[], ex, bound)
            ret === nothing || typenames!(ts, ret, bound)
            for t in ts
                is_placeholder(t) && continue
                is_type(resolve(t)) || push!(dead_types, (rel, code, t))
            end
        end
    end

    #=
    Three floors on the census itself. A section reader that quietly stopped matching
    would satisfy every assertion below with an empty list, so the shape is proven alive.
    The figures are from 2026-09-14: 98 sections, 34 verb headings and 99 call-shaped
    bullets. The floors are loose enough to survive a rewrite of a family's contract.
    =#
    @test nsections >= 80
    @test nheadings >= 25
    @test nbullets >= 75

    # 1. A verb heading names a function the package defines.
    @test dead_headings == Tuple{String, String}[]

    # 2. A call-shaped bullet parses, names a function or a type the package defines, names
    #    the heading's verb when it sits under one, and names only types that exist.
    @test unparsable == Tuple{String, String, String}[]
    @test dead_verbs == Tuple{String, String}[]
    @test strays == Tuple{String, Symbol, String}[]
    @test dead_types == Tuple{String, String, Any}[]

    # ------------------------------------------------------------- check 3

    # Every type the package defines, keyed by the name it was declared under. Shared with
    # `test/test_66_public_declaration_census.jl` as `CH.declared_types`, an alias (`VecNum`) is
    # skipped, because it does not carry the name it is bound to.
    declared = CH.declared_types(PO)
    abstracts = Set(n for (n, T) in declared if isabstracttype(T))

    # The direct supertype of every concrete type the package defines, when that
    # supertype is one of its own abstract types. `subtypes` is not used: the suite runs
    # every test file in one process, so it would also see the subtypes the tests declare.
    parents = Set{Symbol}()
    for (n, T) in declared
        isabstracttype(T) && continue
        S = supertype(T)
        S === Any && continue
        S0 = Base.unwrap_unionall(S)
        parentmodule(S0) === PO && nameof(S0) in abstracts && push!(parents, nameof(S0))
    end

    #=
    The debt list: the abstract types that had a concrete member and no `# Interfaces`
    section on 2026-09-14, when the census was written. Most are classification roots (an
    `AbstractEstimator`, an `AbstractResult`) or a family whose contract is stated on a
    sibling, and each is a name the systematic sweep of #404 decides. The list may only
    SHRINK: a type that gains a section is removed from it in the same edit, and a type
    that gains a concrete member without a section is a failure, never an addition here.
    =#
    debt = Set([:AbstractAlgorithm, :AbstractAssetPanelEstimator,
                :AbstractCalibrationSeries, :AbstractCentralityConstraint,
                :AbstractCentralityEstimator, :AbstractCentralityPolarity,
                :AbstractClusteringResult, :AbstractClustersEstimator,
                :AbstractCollapseAlgorithm, :AbstractConditionalValueatRiskViewFormulation,
                :AbstractCrossSectionalRegressionResult,
                :AbstractCrossSectionalSolveAlgorithm, :AbstractDecompositionContract,
                :AbstractDistanceAlgorithm, :AbstractDistanceEstimator,
                :AbstractDrawdownSeries, :AbstractEntropicValueatRiskViewFormulation,
                :AbstractEntropyPoolingAlgorithm, :AbstractEntropyPoolingOptAlgorithm,
                :AbstractEntropyPoolingOptimiser, :AbstractEntropyPoolingTailView,
                :AbstractEntropyPoolingTailViewEstimator,
                :AbstractEntropyPoolingViewEstimator, :AbstractEstimator,
                :AbstractFactorFamilyBasis, :AbstractFeatureCollapseAlgorithm,
                :AbstractFeeAmortisation, :AbstractForecastUnit,
                :AbstractHierarchicalClusteringAlgorithm, :AbstractHighOrderPriorEstimator,
                :AbstractJuMPResult, :AbstractLoadingsRegressionResult,
                :AbstractMIPIndicators, :AbstractMIPSpace, :AbstractNetworkEstimator,
                :AbstractNonHierarchicalClusteringAlgorithm,
                :AbstractNonNegativeSimilarityMatrixAlgorithm,
                :AbstractOptimalNumberClustersAlgorithm,
                :AbstractOptimalNumberClustersEstimator,
                :AbstractOrderedWeightsArrayAlgorithm,
                :AbstractOrderedWeightsArrayEstimator, :AbstractParsingResult,
                :AbstractPhylogenyConstraintEstimator, :AbstractPhylogenyConstraintResult,
                :AbstractPhylogenyFeatureAlgorithm, :AbstractPhylogenyResult,
                :AbstractPipelineEstimator, :AbstractPipelineResult,
                :AbstractPredictionResult, :AbstractRegressionTarget,
                :AbstractRegularisationEstimator,
                :AbstractRelativisticValueatRiskViewFormulation, :AbstractResult,
                :AbstractReturnForecastResult, :AbstractRiskSeriesAlgorithm,
                :AbstractSearchCrossValidationEstimator, :AbstractSeparationAlgorithm,
                :AbstractSeparationDecayAlgorithm, :AbstractSequentialTailViewConstraint,
                :AbstractShrunkExpectedReturnsAlgorithm,
                :AbstractShrunkExpectedReturnsEstimator,
                :AbstractShrunkExpectedReturnsTarget, :AbstractSimilarityMatrixAlgorithm,
                :AbstractStepwiseRegressionAlgorithm, :AbstractStepwiseRegressionCriterion,
                :AbstractTracking, :AbstractTreeType, :AbstractWeightDrift,
                :BaseClusteringOptimisationEstimator, :BaseJuMPOptimisationEstimator,
                :BaseJuMPOptimisationResult, :BaseSmythBrobyCovariance,
                :BaseStackingOptimisationEstimator,
                :BaseSubsetResamplingOptimisationEstimator,
                :BrownianDistanceVarianceFormulation, :BudgetCostEstimator,
                :BudgetEstimator, :ClusteringOptimisationEstimator, :DBHTRootMethod,
                :DimensionReductionTarget, :EntropyFormulation,
                :FiniteAllocationOptimisationEstimator, :FiniteAllocationOptimisationResult,
                :HierarchicalScalariser, :HighOrderMomentMeasureAlgorithm,
                :HopCountAlgorithm, :InverseMatrixSparsificationAlgorithm,
                :JuMPOptimisationEstimator, :JuMPReturnsEstimator, :JuMPRiskMeasureSettings,
                :LowOrderMomentMeasureAlgorithm, :MedianCenteringFunction,
                :NaiveOptimisationEstimator, :NearOptimalCenteringAlgorithm,
                :NonHierarchicalScalariser, :NonOptimisationRiskMeasure,
                :NonOptimisationSequentialCrossValidationEstimator,
                :NonOptimisationSequentialCrossValidationResult,
                :NonRiskJuMPOptimisationResult, :NonSequentialCrossValidationEstimator,
                :NonSequentialCrossValidationResult, :ObjectiveFunction,
                :OrderedWeightsArrayFormulation, :PathLengthAlgorithm,
                :PortfolioOptimisersError, :PredictionScorer, :ProcessedAttributes,
                :ProcessedRiskBudgetingAttributes, :RankCovarianceEstimator,
                :RelaxedRiskBudgetingAlgorithm, :RiskBudgetingAlgorithm,
                :RiskBudgetingFormulation, :RiskInputKind, :RiskJuMPOptimisationEstimator,
                :RiskJuMPOptimisationResult, :SecondMomentFormulation,
                :SequentialCrossValidationResult, :SmythBrobyCovarianceAlgorithm,
                :SquaredOrderedWeightsArrayAlgorithm,
                :UnstandardisedHighOrderMomentMeasureAlgorithm,
                :UnstandardisedLowOrderMomentMeasureAlgorithm, :ValueatRiskFormulation,
                :VariableTracking, :VarianceFormulation, :WalkForwardEstimator])

    # Floors on the walk: the package declares far more abstract types than carry a
    # section, and most of them parent a concrete type.
    @test length(abstracts) > 200
    @test length(parents) > 150
    @test length(with_section) > 80

    # Both directions are named, so a failure says which name moved and which way.
    uncontracted = sort!(collect(setdiff(parents, with_section, debt)))
    paid = sort!(collect(intersect(debt, with_section)))
    childless = sort!(collect(setdiff(debt, parents)))
    @test ("abstract types with a concrete member and no `# Interfaces` section",
           uncontracted) ==
          ("abstract types with a concrete member and no `# Interfaces` section", Symbol[])
    @test ("debt entries that now carry a `# Interfaces` section", paid) ==
          ("debt entries that now carry a `# Interfaces` section", Symbol[])
    @test ("debt entries with no concrete member, or no longer declared", childless) ==
          ("debt entries with no concrete member, or no longer declared", Symbol[])

    # A verb heading and a bullet resolve through the package, so a name the package no
    # longer defines is dead. Pinned on the defect the census was written for, which the
    # Q3 piece of the PR 625 review corrected by hand.
    @test resolve(:panel_input_is_static) isa Function
    @test resolve(:panel_resolve) isa Function
    @test resolve(:panel_input_field) isa Function
    @test resolve(:panel_write!) === nothing
    @test resolve(:AbstractPanelFieldKind) === nothing
end
