#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with. The load
sits OUTSIDE the `@testset` on purpose: `include` defines methods, and a method defined inside one
top-level statement is not visible to a call in that same statement. The module wrapper keeps that
module's own names out of the worker module. `test_75_concrete_struct_census.jl` loads it the same way.
=#
module CorrectionKeywordCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Correction keyword census: `ddof` is subtracted by the library, `corrected` is applied upstream (#1525)" begin
    using Test

    #=
    Issue #1525. The library named the divisor of a spread with two keywords, `ddof::Integer`
    and `corrected::Bool`. The maintainer ruled that both stay, each where it means one thing.
    `.github/instructions/julia-source-code.instructions.md` § *A divisor's correction is named
    by who makes it* is the Authority.

    - `ddof` is an integer the library subtracts from the number of observations itself,
      `T - ddof`.
    - `corrected` is the flag of `Statistics` and `StatsBase`, which apply it, and which give it
      a meaning for each weight type that no integer states.

    The breach this census reads is one definition that names `ddof` and also hands the
    correction to `Statistics` or `StatsBase`: it names `corrected`, or it calls one of the
    upstream spreads. Such a site states an integer and applies a flag, so `ddof = 2` either
    raises or silently means `ddof = 1`. The first plan for the weight spread of the realised
    factor attribution had this shape, which is why the census exists.

    The converse is not read. A partial fit reproduces a correction that `StatsBase` applies on
    the batch path, so it subtracts a `corrected` flag itself, and it keeps the name of the
    field it reads.
    =#
    CH = CorrectionKeywordCensusHealth.CodeHealth
    root = normpath(joinpath(@__DIR__, ".."))

    # The spreads of `Statistics` and `StatsBase` that apply a bias correction of their own.
    UPSTREAM = (:std, :var, :cov, :cor, :stdm, :varm, :varcorrection, :mean_and_std,
                :mean_and_var, :mean_and_cov)

    callee(f::Symbol) = f
    function callee(f::Expr)
        return f.head === :. && f.args[end] isa QuoteNode ? f.args[end].value : nothing
    end
    callee(f) = nothing

    function names_symbol(node, s::Symbol)
        node === s && return true
        node isa QuoteNode && return node.value === s
        node isa Expr || return false
        return any(a -> names_symbol(a, s), node.args)
    end

    function delegates(node)
        node isa Expr || return false
        if node.head === :call && callee(node.args[1]) in UPSTREAM
            return true
        end
        return names_symbol(node, :corrected) || any(delegates, node.args)
    end

    function isdefinition(ex::Expr)
        return ex.head === :function || (ex.head === :(=) && CH.is_signature(ex.args[1]))
    end

    # Every definition that names `ddof`, and those of them that also delegate the correction.
    function census(ast, rel)
        local sites = String[]
        local breaches = String[]
        CH.walk_ast(ast) do ex
            if isdefinition(ex)
                if names_symbol(ex, :ddof)
                    site = string(rel, ": ", CH.defname(ex.args[1]))
                    push!(sites, site)
                    delegates(ex) && push!(breaches, site)
                end
                return CH.PRUNE
            end
        end
        return sites, breaches
    end

    # A witness for each branch. The library subtracts the integer itself, so it passes. The same
    # integer beside an upstream flag, or beside an upstream spread, is the breach.
    ours = Meta.parseall("spread(x; ddof::Integer = 1) = sqrt(sum(abs2, x .- mean(x)) / (length(x) - ddof))")
    flag = Meta.parseall("function spread(W; ddof::Integer = 1)\n    return std(W; dims = 1, corrected = ddof == 1)\nend")
    call = Meta.parseall("spread(r) = Statistics.var(r.x) * (length(r.x) - 1) / (length(r.x) - r.ddof)")
    @test census(ours, "ours") == (["ours: spread"], String[])
    @test census(flag, "flag")[2] == ["flag: spread"]
    @test census(call, "call")[2] == ["call: spread"]

    named = String[]
    mixed = String[]
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") || continue
        n, m = census(CH.parse_file(joinpath(r, f); root), relpath(joinpath(r, f), root))
        append!(named, n)
        append!(mixed, m)
    end

    # A census that parses nothing passes on an empty set. 20 definitions named `ddof` when the
    # census was written.
    @test length(named) >= 18

    # A new entry here names `ddof` and hands the correction upstream. Subtract the integer from
    # the number of observations yourself, or take `corrected::Bool` and pass it upstream.
    @test sort!(mixed) == String[]
end
