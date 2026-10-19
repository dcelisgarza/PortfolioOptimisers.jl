#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with. The load
sits OUTSIDE the `@testset` on purpose: `include` defines methods, and a method defined inside one
top-level statement is not visible to a call in that same statement. The module wrapper keeps that
module's own names out of the worker module. `test_77_correction_keyword_census.jl` loads it the same way.
=#
module TypeDecidedEarlyReturnCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Type-decided early return census: a branch that a type decides is a method (#1752)" begin
    using Test

    #=
    Issue #1752. `set_sdp_variance_risk!` returned early on a condition that `r.ratio_alg`
    decided, and the maintainer ruled that such a branch is a method of its own.
    `.github/instructions/julia-source-code.instructions.md` § *Multiple Dispatch* is the
    Authority.

    A branch on a type costs nothing at run time: the compiler knows the type of each argument
    of a method, and it removes the dead arm. Measured on `set_sdp_variance_risk!` under
    `LinearRatio`, the typed body of the early-return form held no call to the lift of the
    objective. So the rule is about the source. A method states one case, and the reader finds
    that case by its signature, not inside the body of another case.

    The breach this census reads is an early exit at the top level of a definition, which more
    statements follow: `if c … return … end` with no `else`, `c && return …` or
    `c || return …`. The condition `c` is made of the type tests `isnothing(x)`,
    `x === nothing`, `x !== nothing`, `isa(x, T)` and `x isa T`, and a dispatch splits it into
    two methods. `x` is a positional argument or a field of one, and the body assigns no value
    to its name.

    The census does not read these shapes, because each is a run-time decision or no dispatch
    can take it in two methods:

    - a condition that also tests a value, such as `isnothing(x) || isempty(x)`;
    - a mix of type tests that needs more than two methods, such as
      `isnothing(x) || isnothing(y)`;
    - a test of a keyword argument, because a keyword does not dispatch;
    - a test of a local, or of an argument name that the body assigns;
    - an `if` with an `else`, a ternary, and an exit inside a loop.

    `RUN_TIME` holds the sites whose tested value has a type that only the run time decides,
    each with its reason. A field that declares a `Union` is the usual case. Those sites stay.
    `DEBT` holds the sites that existed when the census was written. Each one becomes methods,
    and its entry then leaves the list, so the list can only shrink.
    =#
    CH = TypeDecidedEarlyReturnCensusHealth.CodeHealth
    root = normpath(joinpath(@__DIR__, ".."))

    # The positional argument names of a signature. A keyword does not dispatch, so it is not
    # read.
    argname(a::Symbol) = a
    function argname(a::Expr)
        if a.head === :(::)
            return length(a.args) == 2 ? argname(a.args[1]) : nothing
        end
        return a.head in (:kw, :(...)) ? argname(a.args[1]) : nothing
    end
    argname(a) = nothing
    function positional_args(sig)
        s = sig
        while s isa Expr && s.head in (:where, :(::))
            s = s.args[1]
        end
        if !(s isa Expr && s.head === :call)
            return Symbol[]
        end
        out = Symbol[]
        for a in s.args[2:end]
            n = argname(a)
            if !isnothing(n)
                push!(out, n)
            end
        end
        return out
    end

    # Every name the body assigns, so a local that reuses an argument name is not read.
    assigned!(out, lhs::Symbol) = push!(out, lhs)
    function assigned!(out, lhs::Expr)
        if lhs.head in (:tuple, :parameters)
            foreach(a -> assigned!(out, a), lhs.args)
        elseif lhs.head === :(::)
            assigned!(out, lhs.args[1])
        end
        return out
    end
    assigned!(out, lhs) = out
    function assigned_names(body)
        out = Set{Symbol}()
        CH.walk_ast(body) do ex
            if ex.head in (:(=), :+=, :-=, :*=, :/=) && !CH.is_signature(ex.args[1])
                assigned!(out, ex.args[1])
            elseif ex.head === :for
                spec = ex.args[1]
                for it in (spec.head === :block ? spec.args : [spec])
                    if it isa Expr && it.head === :(=)
                        assigned!(out, it.args[1])
                    end
                end
            end
        end
        return out
    end

    # An argument, or a field chain of one.
    rooted(x::Symbol, args) = x in args
    rooted(x::Expr, args) = x.head === :. && rooted(x.args[1], args)
    rooted(x, args) = false

    callee(f::Symbol) = f
    function callee(f::Expr)
        return f.head === :. && f.args[end] isa QuoteNode ? f.args[end].value : nothing
    end
    callee(f) = nothing

    #=
    The census reads a condition that a dispatch splits into exactly two methods: a conjunction
    of positive type tests, one method whose signature intersects them; a disjunction of
    positive type tests of one value, one method on their `Union`; and the negation of either,
    which swaps the two methods. Any other mix, such as `isnothing(a) && !isa(b, T)`, needs a
    method for each combination and more methods to settle their ambiguities, so it stays a
    branch.
    =#

    # The value that a positive type test reads, or `nothing`.
    tested(c, args) = nothing
    function tested(c::Expr, args)
        if c.head !== :call
            return nothing
        end
        f = callee(c.args[1])
        x = if f === :isnothing && length(c.args) == 2
            c.args[2]
        elseif f === :(===) && length(c.args) == 3
            if c.args[3] === :nothing
                c.args[2]
            elseif c.args[2] === :nothing
                c.args[3]
            else
                nothing
            end
        elseif f === :isa && length(c.args) == 3
            c.args[2]
        else
            nothing
        end
        return !isnothing(x) && rooted(x, args) ? x : nothing
    end

    function leaves!(out, c, head)
        if c isa Expr && c.head === head
            foreach(a -> leaves!(out, a, head), c.args)
        else
            push!(out, c)
        end
        return out
    end

    function positive_split(c, args)
        if c isa Expr && c.head in (:&&, :||)
            xs = [tested(l, args) for l in leaves!([], c, c.head)]
            if any(isnothing, xs)
                return false
            end
            return c.head === :&& || allequal(xs)
        end
        return !isnothing(tested(c, args))
    end

    # Whether a dispatch splits the condition into two methods.
    function type_test(c, args)
        if c isa Expr && c.head === :call && length(c.args) == 2 && callee(c.args[1]) === :!
            return positive_split(c.args[2], args)
        end
        if c isa Expr &&
           c.head === :call &&
           length(c.args) == 3 &&
           callee(c.args[1]) === :(!==)
            return positive_split(Expr(:call, :(===), c.args[2], c.args[3]), args)
        end
        return positive_split(c, args)
    end

    has_return(x) = false
    has_return(x::Expr) = x.head === :return || any(has_return, x.args)

    # The condition of an early exit, or `nothing`.
    exit_condition(st) = nothing
    function exit_condition(st::Expr)
        if st.head === :if && length(st.args) == 2 && has_return(st.args[2])
            return st.args[1]
        end
        if !(st.head in (:&&, :||)) || !(st.args[end] isa Expr)
            return nothing
        end
        tail = st.args[end]
        if tail.head === :return
            return st.args[1]
        end
        # The parser nests `a && b && return x` as `a && (b && return x)`.
        c = tail.head === st.head ? exit_condition(tail) : nothing
        return isnothing(c) ? nothing : Expr(st.head, st.args[1], c)
    end

    function isdefinition(ex::Expr)
        return ex.head === :function && length(ex.args) == 2 ||
               (ex.head === :(=) && CH.is_signature(ex.args[1]))
    end

    function census(ast, rel)
        local sites = String[]
        CH.walk_ast(ast) do ex
            if isdefinition(ex)
                body = ex.args[2]
                stmts = if body isa Expr && body.head === :block
                    filter(s -> !(s isa LineNumberNode), body.args)
                else
                    Any[]
                end
                # A short-form definition wraps a parenthesised body in a second block.
                if length(stmts) == 1 && stmts[1] isa Expr && stmts[1].head === :block
                    stmts = filter(s -> !(s isa LineNumberNode), stmts[1].args)
                end
                args = setdiff(positional_args(ex.args[1]), assigned_names(body))
                for st in stmts[1:(end - 1)]
                    c = exit_condition(st)
                    if !isnothing(c) && type_test(c, args)
                        push!(sites, string(rel, ": ", CH.defname(ex.args[1]), ": ", c))
                    end
                end
                return CH.PRUNE
            end
        end
        return sites
    end

    # A witness for each branch. The early exit on the type of an argument, of a field, and on
    # two type tests is the breach. A test that also reads a value, a keyword, a local that
    # reuses an argument name, an `if` with an `else` and a last statement are not.
    witness(src) = census(Meta.parseall(src), "w")
    @test witness("function f(x, y)\n    isnothing(x) && return y\n    return x + y\nend") ==
          ["w: f: isnothing(x)"]
    @test witness("function f(r)\n    if !(r.a isa Int)\n        return r\n    end\n    return g(r)\nend") ==
          ["w: f: !(r.a isa Int)"]
    @test witness("f(a, b) = (isnothing(a) && b === nothing && return 0; a + b)") ==
          ["w: f: isnothing(a) && b === nothing"]
    @test witness("function f(x)\n    x !== nothing && return x\n    return 0\nend") ==
          ["w: f: x !== nothing"]
    @test witness("function f(x)\n    (isnothing(x) || x isa Int) && return 0\n    return g(x)\nend") ==
          ["w: f: isnothing(x) || x isa Int"]
    @test isempty(witness("function f(x, y)\n    (isnothing(x) || isnothing(y)) && return 0\n    return x + y\nend"))
    @test isempty(witness("function f(x, y)\n    isnothing(x) && !(y isa Int) && return 0\n    return g(x, y)\nend"))
    @test isempty(witness("function f(x)\n    isnothing(x) || isempty(x) && return 0\n    return 1\nend"))
    @test isempty(witness("function f(x; by = nothing)\n    isnothing(by) && return x\n    return g(x, by)\nend"))
    @test isempty(witness("function f(x)\n    x = g(x)\n    isnothing(x) && return 0\n    return x\nend"))
    @test isempty(witness("function f(x)\n    if isnothing(x)\n        return 0\n    else\n        g()\n    end\n    return x\nend"))
    @test isempty(witness("function f(x)\n    g(x)\n    if isnothing(x)\n        return 0\n    end\nend"))

    #=
    The sites whose tested value has a type that only the run time decides. A dispatch on such a
    value is a dynamic dispatch, which costs more than the branch, so the branch stays.
    =#
    RUN_TIME = [
                # `SampleBufferState.G` declares `Union{Nothing, Vector{Bool}}`, so that the type
                # of the buffer does not depend on whether a block has a gap.
                "src/01_Base/15_Online_a.jl: buffer_kept_rows: isnothing(state.G)",
                "src/01_Base/15_Online_a.jl: merge_buffer_gaps: isnothing(a.G) && isnothing(b.G)",
                "src/01_Base/15_Online_a.jl: shared_series_buffer: isnothing(state.G)"]

    #=
    The sites that existed when the census was written. Turn each one into methods, then remove
    its entry. Do not add an entry: a new early exit on a type is a method from the start.
    =#
    DEBT = ["src/05_Moments/30_RegimeAdjusted/02_RegimeAdjustedExpWeightedCovariance_a.jl: process_observation!: isnothing(active_mask)",
            "src/05_Moments/30_RegimeAdjusted/02_RegimeAdjustedExpWeightedCovariance_a.jl: update_regime!: isnothing(ce.regime_method)",
            "src/05_Moments/30_RegimeAdjusted/03_MahalanobisRegimeBias.jl: mahalanobis_transform: isnothing(grid.cut)",
            "src/05_Moments/32_CrossSectionalFactorModel/03_CrossSectionalFactorModel.jl: assert_cs_block_rows: isnothing(idx) && isnothing(ts)",
            "src/05_Moments/32_CrossSectionalFactorModel/04_FactorExposures/14_ObservedExposure.jl: currency_level_columns: isnothing(xe.base)",
            "src/05_Moments/34_PriceLevelExpectedReturns.jl: push_memory: isnothing(hist)",
            "src/09_ConstraintGeneration/08_ExposureConstraintGeneration.jl: factor_space_regression: isnothing(re)",
            "src/09_ConstraintGeneration/08_ExposureConstraintGeneration.jl: factor_space_regression: re isa AbstractLoadingsRegressionResult",
            "src/10_Prior/07_OpinionPoolingPrior.jl: prior_probability_factory: isnothing(pe.pe1)",
            "src/10_Prior/07_OpinionPoolingPrior.jl: prior_probability_factory: isnothing(pe.pe1)",
            "src/10_Prior/15_CrossSectionalPriorFold.jl: cross_sectional_fold_choice: isnothing(pe.families)",
            "src/10_Prior/16_CrossSectionalForecastHistoryCarry.jl: cross_sectional_carry_standardised: isnothing(st.csr)",
            "src/17_Optimisation/02_CrossValidation/01_Base_CrossValidation_b.jl: assert_prediction_series: isnothing(A)",
            "src/17_Optimisation/05_JuMP/02_JuMPConstraints/03_BudgetConstraints.jl: assert_gross_budget_admissible: isnothing(gbgt) || gbgt isa TimeDependent",
            "src/17_Optimisation/05_JuMP/03_JuMPOptimiser_b.jl: assert_universe_axis_order: isnothing(sets)",
            "src/17_Optimisation/05_JuMP/07_RiskBudgeting.jl: risk_budget_universe_key: !(rba.rkb isa RiskBudgetEstimator)",
            "src/17_Optimisation/06_Meta/03_Stacking.jl: narrow_optimiser_vector: opti isa VecOptE_Opt_TD",
            "src/17_Optimisation/09_OnlineOptimisation.jl: batch_from_state: isnothing(opt.cache)",
            "src/17_Optimisation/09_OnlineOptimisation.jl: batch_from_state: isnothing(opt.cache)",
            "src/17_Optimisation/10_OnlinePortfolioSelection/02_OnlineSelectionRules.jl: expert_start_allocation: isnothing(alg.w)",
            "src/17_Optimisation/10_OnlinePortfolioSelection/02_OnlineSelectionRules.jl: expert_start_weights: isnothing(alg.p0) && isnothing(alg.eset)",
            "src/17_Optimisation/10_OnlinePortfolioSelection/03_OnlinePortfolioSelection.jl: update_online_estimator: !(opt.fb isa OptimisationEstimator)",
            "src/17_Optimisation/10_OnlinePortfolioSelection/05_ForecastRules.jl: holds_second_moment: est isa Union{<:AbstractCovarianceEstimator, <:StatsBase.CovarianceEstimator, <:AbstractVarianceEstimator, <:AbstractPriorEstimator}",
            "src/17_Optimisation/10_OnlinePortfolioSelection/12_AllocationSetProgramme.jl: add_allocation_set_bounds!: isnothing(sbgt) && isnothing(gbgt)",
            "src/21_Pipeline/02_StepExecution.jl: resolve_constraint_target: isnothing(target)",
            "src/21_Pipeline/02_StepExecution.jl: run_step: ps.est isa AbstractConstraintEstimator",
            "src/21_Pipeline/02_StepExecution.jl: run_step: ps.est isa AbstractUncertaintySetEstimator",
            "src/21_Pipeline/02_StepExecution.jl: run_step: ps.est isa Function",
            "src/21_Pipeline/03_Pipeline.jl: inject_context: isnothing(ctx.prior) && (isnothing(ctx.phylogeny) && (isnothing(ctx.uncertainty) && isnothing(ctx.constraints)))",
            "src/21_Pipeline/03_Pipeline.jl: maybe_inject_step: ps.est isa OptimisationEstimator",
            "src/21_Pipeline/06_OnlinePipeline.jl: copy_step_states: ps.est isa Function",
            "src/21_Pipeline/06_OnlinePipeline.jl: merge_states: isnothing(b.data)",
            "src/21_Pipeline/06_OnlinePipeline.jl: online_entry_state: !(isnothing(p.cache))",
            "src/21_Pipeline/06_OnlinePipeline.jl: update_online_estimator: p.cache isa PipelineBufferState",
            "src/21_Pipeline/06_OnlinePipeline.jl: update_online_step: ps.est isa Function"]

    found = String[]
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") || continue
        append!(found,
                census(CH.parse_file(joinpath(r, f); root), relpath(joinpath(r, f), root)))
    end

    # The sites of `a` that `b` does not hold, counted with their multiplicity, because one
    # definition can hold two equal exits.
    function minus(a, b)
        out = copy(a)
        for x in b
            k = findfirst(==(x), out)
            if !isnothing(k)
                deleteat!(out, k)
            end
        end
        return sort!(out)
    end

    # A census that parses nothing passes on empty lists, so the runtime sites must be found.
    @test length(found) >= length(RUN_TIME) > 0

    # A new entry here is an early exit that a type decides. Split the definition into methods
    # by dispatch on that type. If only the run time decides the type, add the site to
    # `RUN_TIME` with its reason.
    @test minus(found, vcat(RUN_TIME, DEBT)) == String[]

    # A stale entry names a site that is gone. Remove it, so the lists can only shrink.
    @test minus(vcat(RUN_TIME, DEBT), found) == String[]
end
