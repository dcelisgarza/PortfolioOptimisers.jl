#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with. The load
sits OUTSIDE the `@testset` on purpose: `include` defines methods, and a method defined inside one
top-level statement is not visible to a call in that same statement. The module wrapper keeps that
module's own names out of the worker module. `test_87_any_container_census.jl` loads it the same
way.
=#
module JuMPBuildFormCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "JuMP build form census: a JuMP model is built in the measured forms (#2062)" begin
    using Test

    #=
    Issue #2062. `.github/instructions/julia-source-code.instructions.md` § *A JuMP model is built
    in the measured forms* is the Authority, and it states the measurement of each form. This
    census reads the parse tree of every file under `src/` and `ext/`, so a docstring, a doctest
    and a comment are out of its reach. It reads five spellings:

    - `plural`: a plural macro, `@variables`, `@constraints` or `@expressions`. A plural block
      builds the same model as one singular macro for each line, at the same cost, and the
      formatter aligns its lines far to the right.
    - `axis_sum`: a call of `sum`, `mean`, `reduce` or `mapreduce` with a `dims` keyword, in a
      definition that holds a JuMP macro. Base reduces a JuMP array over an axis with a plain
      `+`, which copies each partial sum.
    - `update`: an updating assignment `+=`, `-=`, `.+=` or `.-=`, in a definition that holds a
      JuMP macro. On a JuMP expression it copies the expression at each step.
    - `broadcast_sum`: `sum(a .* b)` in a definition that holds a JuMP macro. It builds the
      vector of products before the sum.
    - `broadcast_row`: a `@constraint` whose comparison is broadcast, `.<=`, `.>=` or `.==`.
      One vector-valued row builds and copies to a solver for less than one scalar row for each
      entry (R51 of #1817 measured 27 % less on the path of `path_budget_constraint!`).

    A definition that holds a JuMP macro can also do the same arithmetic on numbers, where it is
    correct. `KEEP` holds each such definition with the count of its sites and the reason. The
    census asserts each count exactly, so a count that falls is a stale entry.
    =#
    CH = JuMPBuildFormCensusHealth.CodeHealth
    root = normpath(joinpath(@__DIR__, ".."))

    PLURAL = ("@variables", "@constraints", "@expressions")
    JUMP = ("@variable", "@constraint", "@expression", "@objective", PLURAL...)
    REDUCERS = (:sum, :mean, :reduce, :mapreduce)
    UPDATES = (:+=, :-=, :.+=, :.-=)
    DOTTED = (:.<=, :.>=, :.==, :.≤, :.≥)

    bare(x::Symbol) = x
    bare(x::Expr) = x.head === :. && x.args[end] isa QuoteNode ? x.args[end].value : nothing
    bare(x) = nothing
    isjump(ex) = ex isa Expr && ex.head === :macrocall && CH.macro_name(ex.args[1]) in JUMP
    # `true` when `a` is a comparison with a broadcast operator, as `x .>= 0` or `l .<= x .<= u`.
    function dotted(a)
        a isa Expr || return false
        a.head === :call && return a.args[1] in DOTTED
        a.head === :comparison && return any(in(DOTTED), a.args[2:2:end])
        return false
    end
    function broadcast_row(ex::Expr)
        return ex.head === :macrocall &&
               CH.macro_name(ex.args[1]) == "@constraint" &&
               any(dotted, ex.args[2:end])
    end

    # `true` when a keyword argument named `dims` is among the arguments of the call `ex`.
    function has_dims(ex::Expr)
        for a in ex.args[2:end]
            if a isa Expr && a.head === :parameters
                any(p -> p isa Expr && p.head === :kw && p.args[1] === :dims, a.args) &&
                    return true
            elseif a isa Expr && a.head === :kw && a.args[1] === :dims
                return true
            end
        end
        return false
    end
    # `hit` and not `found`: a closure assigns an outer local of the same name.
    function holds_jump(ex)
        hit = false
        CH.walk_ast(ex) do x
            hit |= isjump(x)
            return nothing
        end
        return hit
    end

    # Every site of one tree, as `(definition, line, rule)`.
    function sites(tree)
        out = Tuple{String, Int, String}[]
        line = 0
        name = "<toplevel>"
        function visit(ex, jumpdef::Bool)
            if ex isa LineNumberNode
                line = ex.line
                return nothing
            end
            ex isa Expr || return nothing
            if CH.isdocstring(ex)
                visit(ex.args[end], jumpdef)
                return nothing
            end
            if ex.head === :macrocall && CH.macro_name(ex.args[1]) in PLURAL
                push!(out, (name, line, "plural"))
            end
            if jumpdef && ex.head === :call && bare(ex.args[1]) in REDUCERS && has_dims(ex)
                push!(out, (name, line, "axis_sum"))
            end
            if jumpdef && ex.head in UPDATES
                push!(out, (name, line, "update"))
            end
            if broadcast_row(ex)
                push!(out, (name, line, "broadcast_row"))
            end
            if jumpdef &&
               ex.head === :call &&
               bare(ex.args[1]) === :sum &&
               length(ex.args) == 2 &&
               ex.args[2] isa Expr &&
               ex.args[2].head === :call &&
               ex.args[2].args[1] === :.*
                push!(out, (name, line, "broadcast_sum"))
            end
            foreach(a -> visit(a, jumpdef), ex.args)
            return nothing
        end
        for top in tree.args
            n = CH.definition_name(top)
            name = isempty(n) ? "<toplevel>" : n
            visit(top, holds_jump(top))
        end
        return out
    end

    #=
    Each definition that keeps a site, keyed by `file => definition`, with the count of its sites
    and the reason. A broadcast row stays where its vector form fails the stop rule of the
    Authority, as R51 of #1817 measured it with the JuMP build and the copy to Clarabel and SCS.
    =#
    KEEP = Dict{Tuple{String, String}, Tuple{Int, String}}(("src/17_Optimisation/05_JuMP/09_RiskMeasureConstraints/01_BaseRiskConstraints.jl",
                                                            "scalarise_risk_expression!") =>
                                                               (1,
                                                                "`risk_ms` holds one row for each risk measure of the vector, a handful in a model of thousands of rows, so its vector form is below the stop rule"),
                                                           ("src/17_Optimisation/10_OnlinePortfolioSelection/09_AdaptiveOptimistic.jl",
                                                            "solve_projection_programme!") =>
                                                               (1,
                                                                "`proj_fix` of the second stage: the vector form saved 2.5 % of the build and the copy at 100 assets and cost 2.9 % more at 20 assets"),
                                                           ("src/17_Optimisation/10_OnlinePortfolioSelection/12_AllocationSetProgramme.jl",
                                                            "set_projection_objective!") =>
                                                               (1,
                                                                "`proj_zero` of the entropic projection: the vector form cost 1.6 % more at 100 assets and 4.7 % more at 20 assets"))

    found = Dict{Tuple{String, String}, Vector{String}}()
    for f in CH.source_files(; root)
        for (name, line, rule) in sites(CH.parse_file(f; root))
            push!(get!(found, (f, name), String[]), "$f:$line ($name): $rule")
        end
    end

    # A site outside `KEEP`, or a definition whose count moved, fails the census.
    unlisted = String[]
    for (key, hits) in found
        if get(KEEP, key, (0, ""))[1] != length(hits)
            append!(unlisted, hits)
        end
    end
    if !isempty(unlisted)
        @info "JuMP build forms that the measured rules refuse. Write one singular macro for each entry, reduce over an axis with one `sum(view(…))` or `dot` for each row or column, build a loop with `JuMP.add_to_expression!`, write a dot product as `LinearAlgebra.dot`, and write a vector of rows as one row with an unbroadcast comparison. A site that does the arithmetic on numbers goes into `KEEP` with its reason:\n" *
              join(sort!(unlisted), "\n")
    end
    @test isempty(unlisted)
    stale = sort!([k for k in keys(KEEP) if !haskey(found, k)])
    if !isempty(stale)
        @info "Entries of `KEEP` that no longer name a site:\n" * join(string.(stale), "\n")
    end
    @test isempty(stale)
    @test all(v -> !isempty(v[2]), values(KEEP))

    # The matcher reads each spelling, and it lets the near shapes past.
    function rules(s)
        return last.(sites(Meta.parseall(s)))
    end
    @test rules("f(m) = JuMP.@variables(m, begin\n x\n y\n end)") == ["plural"]
    @test rules("f(m) = @constraints(m, begin\n x >= 0\n end)") == ["plural"]
    @test rules("f(m) = JuMP.@expressions(m, begin\n e, x + y\n end)") == ["plural"]
    @test isempty(rules("f(m) = JuMP.@variable(m, x)"))
    @test rules("function f(m, u, A)\n c = vec(sum(u .* A; dims = 2))\n JuMP.@constraint(m, c <= 0)\nend") ==
          ["axis_sum"]
    @test rules("function f(m, u)\n c = Statistics.mean(u; dims = 1)\n JuMP.@constraint(m, c <= 0)\nend") ==
          ["axis_sum"]
    # A sum over an axis in a definition with no JuMP macro acts on numbers.
    @test isempty(rules("f(A) = vec(sum(A; dims = 2))"))
    @test isempty(rules("function f(m, u)\n c = sum(u)\n JuMP.@constraint(m, c <= 0)\nend"))
    @test rules("function f(m, w)\n e = JuMP.AffExpr(0.0)\n for x in w\n e += x\n end\n JuMP.@constraint(m, e <= 0)\nend") ==
          ["update"]
    @test isempty(rules("function f(x)\n s = 0\n for v in x\n s += v\n end\n return s\nend"))
    @test rules("function f(m, e, w)\n e .+= w\n JuMP.@constraint(m, e <= 0)\nend") ==
          ["update"]
    @test rules("function f(m, u)\n c = vec(reduce(+, u; dims = 2))\n JuMP.@constraint(m, c <= 0)\nend") ==
          ["axis_sum"]
    @test rules("f(m, a, w) = JuMP.@expression(m, sum(a .* w))") == ["broadcast_sum"]
    @test rules("function f(m, a, w)\n s = sum(a .* w)\n JuMP.@constraint(m, s <= 0)\nend") ==
          ["broadcast_sum"]
    @test isempty(rules("f(m, a, w) = JuMP.@expression(m, LinearAlgebra.dot(a, w))"))
    @test isempty(rules("f(a, w) = sum(a .* w)"))
    @test rules("f(m, x) = JuMP.@constraint(m, x .>= 0)") == ["broadcast_row"]
    @test rules("f(m, x) = JuMP.@constraint(m, c, x .== 1)") == ["broadcast_row"]
    @test rules("f(m, x, l, u) = JuMP.@constraint(m, l .<= x .<= u)") == ["broadcast_row"]
    @test isempty(rules("f(m, x) = JuMP.@constraint(m, x >= 0)"))
    @test isempty(rules("f(m, x, z) = JuMP.@constraint(m, x - z .* 2 >= 0)"))
    # A docstring is text.
    @test isempty(rules("\"\"\"\n    JuMP.@variables(m, begin x end)\n\"\"\"\nf(m) = 1"))
end
