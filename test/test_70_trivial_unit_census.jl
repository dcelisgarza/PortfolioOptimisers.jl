#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with. The load
sits OUTSIDE the `@testset` on purpose: `include` defines methods, and a method defined inside one
top-level statement is not visible to a call in that same statement. The module wrapper keeps that
module's own names out of the worker module. `test_61_interfaces_section_census.jl` loads it the
same way.
=#
module TrivialUnitCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Trivial unit census: a pure forward with one caller is inlined, and a kept one is an allow-list entry (ADR 0168)" begin
    using PortfolioOptimisers, Test

    #=
    Issue #1208, decided by ADR 0168 and built by #1212. A unit that offers nothing but
    indirection is a dead-end: a name between a caller and the one expression it forwards
    to. `diagonal_geometry` was the shape, one method, one caller, and the body
    `DiagonalProjection(; h = h)`. The rule flags exactly that shape and nothing wider:

    1. **one method**, because two methods are dispatch, and a `Nothing`/value pair or a
       two-family pair saves its caller an `isa` branch and keeps the caller's inferred
       type concrete;
    2. a body that is **one depth-one forward**: a bare argument, a literal, a field of an
       argument, or one call (or constructor) whose callee is a name and whose arguments
       are each of those, a global name, a splat of one or a keyword forward. A decision,
       an operator, an index, a nested call, a comprehension or a string template is logic
       the unit owns, and a unit that reduces its caller's complexity earns its keep;
    3. **at most one reference** in `src/` and `ext/` outside its own definition. A
       reference in value position counts as a call does. A call from `test/`, `docs/` or
       `examples/` is not a use, and a name with no reference at all is dead.

    The type-level case is an abstract type with one concrete subtype that nothing codes
    against: no dispatch on it, no field bound, no union alias, so its only reference is the
    subtype's own declaration. An abstract type with a realistic future application stays,
    and the allow-list below is where that application is named.

    Two shapes are not candidates: a constructor method named after its type, the
    library's keyword-constructor idiom, and a method that extends another module's
    function, such as `Base.show`, which is an interface by definition.

    The census is the record. The source is parsed with the parser `code_health/CodeHealth.jl`
    holds (`Meta.parseall`, which is JuliaSyntax since Julia 1.10), docstrings, comments,
    `export`/`public` lines and symbols are not references, and a name is resolved against
    the LOADED module to tell a public binding from a private one and an owned function
    from an extended one. A flagged name is kept only by a hand-written entry below, with a
    one-line reason, and an entry whose name is no longer flagged is stale and fails, so
    the list can only shrink on its own.
    =#
    CH = TrivialUnitCensusHealth.CodeHealth
    PO = PortfolioOptimisers
    root = normpath(joinpath(@__DIR__, ".."))

    files = String[]
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") && push!(files, joinpath(r, f))
    end
    sort!(files)
    trees = Dict(f => CH.parse_file(f; root) for f in files)

    # ------------------------------------------------------------- definitions

    strip_lnn(args) = filter(a -> !(a isa LineNumberNode), args)

    # The name under a signature, and whether it is module-qualified (`Base.show`). A
    # functor `(r::T)(x)` has no name and is never a candidate.
    function sig_name(sig)
        while sig isa Expr && sig.head in (:where, :(::))
            sig = sig.args[1]
        end
        sig isa Expr && sig.head === :call || return (nothing, false)
        f = sig.args[1]
        f isa Symbol && return (f, false)
        if f isa Expr && f.head === :. && f.args[end] isa QuoteNode
            return (f.args[end].value, true)
        end
        if f isa Expr && f.head === :curly
            return sig_name(Expr(:call, f.args[1]))
        end
        return (nothing, false)
    end

    # A leaf a forward may pass on: a name, a literal, `nothing`, a field chain on a name,
    # a splat of one, a keyword forward or a tuple of leaves.
    function is_leaf(e)
        e isa Symbol && return true
        e isa Expr || return true
        if e.head === :. && length(e.args) == 2 && e.args[2] isa QuoteNode
            return is_leaf(e.args[1])
        elseif e.head === :(...) || e.head === :kw
            return is_leaf(e.args[end])
        elseif e.head === :parameters || e.head === :tuple
            return all(is_leaf, e.args)
        end
        return false
    end
    is_operator_name(f) = f isa Symbol && Base.isoperator(f)
    # One depth-one forward, as rule 2 states it.
    function is_forward(e)
        is_leaf(e) && return true
        e isa Expr && e.head === :call || return false
        f = e.args[1]
        callee_ok = (f isa Symbol && !is_operator_name(f)) ||
                    (f isa Expr && f.head === :. && f.args[end] isa QuoteNode && is_leaf(f))
        return callee_ok && all(is_leaf, e.args[2:end])
    end
    function body_trivial(body)
        stmts = body isa Expr && body.head === :block ? strip_lnn(body.args) : Any[body]
        length(stmts) == 1 || return false
        s = stmts[1]
        if s isa Expr && s.head === :return
            isempty(s.args) && return true
            s = s.args[1]
        end
        return is_forward(s)
    end

    # name => [(file, line, trivial, qualified)], and the names of every declared struct.
    defs = Dict{Symbol, Vector{Tuple{String, Int, Bool, Bool}}}()
    structs = Set{Symbol}()
    function record_def!(e, file, line)
        if CH.isdocstring(e)
            e.args[2] isa LineNumberNode && (line = e.args[2].line)
            e = CH.unwrap(e)
        end
        e isa Expr || return nothing
        if e.head === :function || (e.head === :(=) && CH.is_signature(e.args[1]))
            length(e.args) >= 2 || return nothing
            name, qual = sig_name(e.args[1])
            name === nothing && return nothing
            push!(get!(defs, name, Tuple{String, Int, Bool, Bool}[]),
                  (file, line, body_trivial(e.args[2]), qual))
        elseif e.head === :struct
            n = CH.defname(e.args[2])
            isempty(n) || push!(structs, Symbol(n))
        elseif e.head === :macrocall
            l = e.args[2] isa LineNumberNode ? e.args[2].line : line
            record_def!(e.args[end], file, l)
        elseif e.head === :module || e.head === :block || e.head === :toplevel
            for a in (e.head === :module ? e.args[end].args : e.args)
                a isa LineNumberNode && (line = a.line; continue)
                record_def!(a, file, line)
            end
        end
        return nothing
    end
    for f in files
        record_def!(trees[f], f, 0)
    end

    # ------------------------------------------------------------- references

    #=
    Every `Symbol` in reference position, counted once per occurrence. The positions that
    are NOT references: a docstring's prose, an `export`/`public`/`import` line, a
    definition's own name and argument names, a keyword's name, the left side of an
    assignment, the name bound by `x::T`, a struct's field names and its supertype, and a
    function's own name inside its own body. `Mod.name` counts `name`, and so does `x.field`,
    because the two cannot be told apart here, which errs on the side of a use. The bounds of
    a struct's own type parameters count: they are uses of the types they name.
    =#
    refs = Dict{Symbol, Int}()
    count!(s::Symbol) = (refs[s] = get(refs, s, 0) + 1)
    function count_refs!(e, self)
        if e isa Symbol
            e === self || count!(e)
            return nothing
        end
        e isa Expr || return nothing
        h = e.head
        if h === :macrocall && CH.isdocstring(e)
            count_refs!(e.args[end], self)
        elseif h in (:export, :public, :import, :using, :inert, :abstract)
            nothing
        elseif h === :const
            count_refs!(e.args[1].args[end], self)
        elseif h === :function || (h === :(=) && CH.is_signature(e.args[1]))
            name, _ = sig_name(e.args[1])
            count_sig_refs!(e.args[1], name)
            length(e.args) >= 2 && count_refs!(e.args[2], name)
        elseif h === :struct
            count_struct_refs!(e)
        elseif h === :kw
            count_refs!(e.args[2], self)
        elseif h === :(=) &&
               (e.args[1] isa Symbol || (e.args[1] isa Expr && e.args[1].head === :tuple))
            count_refs!(e.args[2], self)
        elseif h === :(::) && length(e.args) == 2 && e.args[1] isa Symbol
            count_refs!(e.args[2], self)
        elseif h === :.
            count_refs!(e.args[1], self)
            if length(e.args) == 2
                if e.args[2] isa QuoteNode
                    e.args[2].value isa Symbol && count_refs!(e.args[2].value, self)
                else
                    # A broadcast call `f.(args...)`: the tuple holds the arguments.
                    count_refs!(e.args[2], self)
                end
            end
        elseif h === :macrocall
            foreach(a -> count_refs!(a, self), e.args[2:end])
        else
            foreach(a -> count_refs!(a, self), e.args)
        end
        return nothing
    end
    function count_sig_refs!(sig, self)
        while sig isa Expr && sig.head in (:where, :(::))
            if sig.head === :(::)
                count_refs!(sig.args[end], self)
            else
                foreach(w -> count_refs!(w, self), sig.args[2:end])
            end
            sig = sig.args[1]
        end
        sig isa Expr && sig.head === :call || return nothing
        foreach(a -> count_arg_refs!(a, self), sig.args[2:end])
        return nothing
    end
    function count_arg_refs!(a, self)
        if a isa Symbol
            nothing
        elseif a isa Expr && a.head === :parameters
            foreach(x -> count_arg_refs!(x, self), a.args)
        elseif a isa Expr && a.head === :kw
            count_arg_refs!(a.args[1], self)
            count_refs!(a.args[2], self)
        elseif a isa Expr && a.head === :(...)
            count_arg_refs!(a.args[1], self)
        elseif a isa Expr && a.head === :(::)
            if length(a.args) == 2
                count_refs!(a.args[2], self)
            else
                count_refs!(a.args[1], self)
            end
        else
            count_refs!(a, self)
        end
        return nothing
    end
    function count_struct_refs!(e)
        head = e.args[2]
        head isa Expr && head.head === :<: && (head = head.args[1])
        if head isa Expr && head.head === :curly
            for a in head.args[2:end]
                a isa Expr && a.head === :<: && count_refs!(a.args[end], nothing)
            end
        end
        for a in e.args[3].args
            a isa Expr || continue
            if a.head === :(::)
                count_refs!(a.args[end], nothing)
            elseif a.head === :function ||
                   (a.head === :(=) && CH.is_signature(a.args[1])) ||
                   a.head === :macrocall
                count_refs!(a, nothing)
            end
        end
        return nothing
    end
    for f in files
        count_refs!(trees[f], nothing)
    end

    # ------------------------------------------------------------- the function census

    function extends_other(name)
        isdefined(PO, name) || return false
        v = getfield(PO, name)
        return v isa Function && parentmodule(v) !== PO
    end
    flagged = Symbol[]
    for (name, ds) in defs
        name in structs && continue
        any(d -> d[4], ds) && continue
        extends_other(name) && continue
        length(ds) == 1 && ds[1][3] && get(refs, name, 0) <= 1 && push!(flagged, name)
    end
    sort!(flagged)

    #=
    The allow-list. A public name is public API, and a user's call is a use the census
    cannot see, so every public forward is listed by name and the list shows every one the
    library carries. A private name is listed with the reason it earns its keep.
    =#
    kept = Dict{Symbol, String}(#
                                # Public descriptor factories: one keyword call each, the
                                # user's door to a named exposure (ADR 0135).
                                :ReturnOnAssets => "public descriptor factory",
                                :ReturnOnEquity => "public descriptor factory",
                                :AssetTurnover => "public descriptor factory",
                                :CashFlowToAssets => "public descriptor factory",
                                :SalesToEnterpriseValue => "public descriptor factory",
                                :LogMarketCap => "public descriptor factory",
                                :BookToPrice => "public descriptor factory",
                                :CashFlowToPrice => "public descriptor factory",
                                :SalesToPrice => "public descriptor factory",
                                :EarningsToPrice => "public descriptor factory",
                                :ForwardEarningsToPrice => "public descriptor factory",
                                :EbitdaToEnterpriseValue => "public descriptor factory",
                                :DebtToAssets => "public descriptor factory",
                                :AssetsGrowthRate => "public descriptor factory",
                                :SalesGrowthRate => "public descriptor factory",
                                :IssuanceGrowthRate => "public descriptor factory",
                                :EarningsChangeToPrice => "public descriptor factory",
                                :CapexToAssetsChangeInIntensity => "public descriptor factory",
                                :EWMomentum => "public descriptor factory",
                                :EWShareTurnover => "public descriptor factory",
                                :EWAmihudIlliquidity => "public descriptor factory",
                                :EWMarketBeta => "public descriptor factory",
                                :RollingMomentum => "public descriptor factory",
                                :Reversal => "public descriptor factory",
                                :MaxReturn => "public descriptor factory",
                                # Public aliases of the ordered-weights risk measures.
                                :OWA_GMD => "public alias constructor",
                                :OWA_WR => "public alias constructor",
                                :OWA_RG => "public alias constructor",
                                # The value form of a public seam, `nothing` arm; the
                                # `# Interfaces` section of `matrix_processing_algorithm!`
                                # names it for a subtype to answer.
                                :matrix_processing_algorithm => "public seam, the out-of-place arm of an Open Family",
                                # Private names that are simple and convenient.
                                :compact_radius_sample_size => "one rule, Kish's size, named where three calibration rules read it",
                                :pipe_config_field => "the fallback of a family `@pipe_delegates` generates methods for",
                                :cv_sequential_info => "the one message of the sequential fold loop, named beside the rule that sends a run there",
                                :_expr_to_lens => "the base case of the lens-building recursion of `expr_to_lens_chain`",
                                :attribution_no_errors => "the empty answer of an attribution, named where the Result reads three fields from it")

    # ------------------------------------------------------------- the type census

    # An abstract type the package declares with exactly one concrete subtype and no
    # reference outside the declarations. `subtypes` is not used: the suite runs every
    # test file in one process, so it would also see the subtypes the tests declare.
    declared = CH.declared_types(PO)
    children = Dict{Symbol, Vector{Symbol}}()
    for (n, T) in declared
        S = supertype(T)
        S === Any && continue
        S0 = Base.unwrap_unionall(S)
        parentmodule(S0) === PO &&
            haskey(declared, nameof(S0)) &&
            push!(get!(children, nameof(S0), Symbol[]), n)
    end
    flagged_types = Symbol[]
    for (n, T) in declared
        isabstracttype(T) || continue
        kids = get(children, n, Symbol[])
        length(kids) == 1 &&
            !isabstracttype(declared[kids[1]]) &&
            get(refs, n, 0) == 0 &&
            push!(flagged_types, n)
    end
    sort!(flagged_types)

    #=
    The allow-list of abstract types. Each names the realistic future application the type
    is the seam for, or the `# Interfaces` section that already states its contract. A type
    that gains a second subtype, or a method that dispatches on it, is no longer flagged and
    its entry is stale.
    =#
    kept_types = Dict{Symbol, String}(#
                                      :AbstractCrossSectionalRegressionResult => "a second cross-sectional regression result, a robust or a constrained fit",
                                      :AbstractJuMPResult => "a second JuMP result carrier, one that keeps the model",
                                      :AbstractPipelineEstimator => "a second pipeline shape beside `Pipeline`",
                                      :AbstractPipelineResult => "the result of a second pipeline shape",
                                      :AbstractSearchCrossValidationResult => "carries an `# Interfaces` section: an Open Family",
                                      :BaseClusteringOptimisationEstimator => "a second clustering optimiser beside `HierarchicalOptimiser`",
                                      :BaseGerberIQCovariance => "carries an `# Interfaces` section: an Open Family",
                                      :BaseHierarchicalOptimisationResult => "carries an `# Interfaces` section: an Open Family",
                                      :BaseJuMPOptimisationResult => "a second JuMP optimisation result beside `JuMPOptimisationResult`",
                                      :BaseSmythBrobyCovariance => "a second Smyth-Broby family member, as `BaseGerberIQCovariance` has",
                                      :BaseStackingOptimisationEstimator => "a second stacking estimator beside `Stacking`",
                                      :BaseSubsetResamplingOptimisationEstimator => "a second resampling estimator beside `SubsetResampling`",
                                      :BootstrapUncertaintySetEstimator => "carries an `# Interfaces` section: an Open Family",
                                      :BudgetEstimator => "a second budget estimator beside `BudgetRange`, a solver-driven one",
                                      :HierarchicalScalariser => "the max and sum scalarisers beside `MinScalariser`",
                                      :InverseMatrixSparsificationAlgorithm => "a second sparsification beside `LoGo`, a graphical lasso",
                                      :NonOptimisationSequentialCrossValidationEstimator => "a second sequential non-optimisation scheme beside `MultipleRandomised`",
                                      :NonOptimisationSequentialCrossValidationResult => "the result of a second sequential non-optimisation scheme",
                                      :OptimisationModelResult => "carries an `# Interfaces` section: an Open Family")

    # ------------------------------------------------------------- the assertions

    # Floors on the census itself: a parser that quietly stopped matching would satisfy
    # every assertion below with an empty list. Measured 2026-09-21: 2765 names, of which
    # 34 flagged, and 19 flagged abstract types.
    @test length(defs) > 2000
    @test sum(values(refs)) > 50_000
    @test length(flagged) >= 20
    @test length(flagged_types) >= 10

    # A flagged name outside the allow-list is a pure forward with at most one caller, and
    # is inlined at that caller; a dead one is deleted. Both directions are named, so a
    # failure says which name moved and which way.
    unkept = [n for n in flagged if !haskey(kept, n)]
    stale = sort!([n for n in keys(kept) if !(n in flagged)])
    @test ("pure forwards with at most one caller, outside the allow-list", unkept) ==
          ("pure forwards with at most one caller, outside the allow-list", Symbol[])
    @test ("allow-list entries no longer flagged", stale) ==
          ("allow-list entries no longer flagged", Symbol[])
    unkept_types = [n for n in flagged_types if !haskey(kept_types, n)]
    stale_types = sort!([n for n in keys(kept_types) if !(n in flagged_types)])
    @test ("single-subtype abstract types nothing codes against, outside the allow-list",
           unkept_types) ==
          ("single-subtype abstract types nothing codes against, outside the allow-list",
           Symbol[])
    @test ("allow-list types no longer flagged", stale_types) ==
          ("allow-list types no longer flagged", Symbol[])

    # A public entry is public, and a private entry carries a reason. An entry that says
    # "public" for a name that is no longer declared so has lost its justification.
    for (n, why) in kept
        @test !isempty(why)
        if startswith(why, "public")
            @test (n, Base.ispublic(PO, n)) == (n, true)
        end
    end

    # Pinned on the three units ADR 0168 named and the one dead-end the build removed.
    @test !isdefined(PO, :diagonal_geometry)
    @test !isdefined(PO, :assert_diagonal_bound)
    @test !isdefined(PO, :resolve_turnover)
    @test !isdefined(PO, :set_budget_costs!)
    @test !hasfield(AdaptiveSubgradient, :proj)
    @test !Base.ispublic(PO, :DiagonalProjection)
end
