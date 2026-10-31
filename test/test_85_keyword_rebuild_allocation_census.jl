@testset "Keyword rebuild allocation census: a keyword rebuild costs what a positional call costs" begin
    using Test, PortfolioOptimisers, StatsBase, Logging

    #=
    `rebuild_estimator` rebuilds an estimator through its keyword constructor (#1806), and the
    online fold calls it on every step (#1807). A keyword argument whose type annotation holds a
    `Union` with a bounded member, such as `Union{<:Real, <:AbstractLearningRateSchedule}`, or a
    `Tuple` with a bounded parameter, such as `Option{<:Tuple{<:Number, <:Number}}`, makes Julia
    build that type again on each keyword call. The plain spellings, `Union{Real, AbstractLearningRateSchedule}`
    and `Option{<:Tuple{Number, Number}}`, are the same types and cost nothing.

    The review of #1817 (R6) measured 16 types whose keyword rebuild allocated more than a
    positional call with the same fields: 352 bytes for each online step of the two
    regime-adjusted estimators, and up to 784 bytes for each rebuild of the others. #1824 wrote
    every such annotation of `src/` and `ext/` in the plain spelling, 1041 sites in 166 files.
    `.github/instructions/julia-source-code.instructions.md` § *A `Union` or a `Tuple` names its
    members without a bound* states the rule.

    The census has two parts. The first reads the source text, so it reaches every annotation,
    also of a function that is not a constructor. The second measures the keyword rebuild of
    every estimator that it can build: the default instance of each struct with a constructor
    of no arguments, every value of the library nested in one, and the fixtures below.
    =#
    PO = PortfolioOptimisers

    #=
    `true` when the `Union{` or the `Tuple{` that ends just before the byte `i` holds a member
    that starts with `<:` at its top level. The scan reads bytes: every delimiter it counts is
    ASCII, so a multi-byte character of a docstring never matches one.
    =#
    function holds_bounded_member(cu, i::Int)
        depth = 0
        at_start = true
        j = i
        while j <= length(cu)
            c = cu[j]
            if at_start
                if c in (UInt8(' '), UInt8('\n'), UInt8('\t'), UInt8('\r'))
                    j += 1
                    continue
                end
                c == UInt8('<') && j < length(cu) && cu[j + 1] == UInt8(':') && return true
                at_start = false
            end
            if c in (UInt8('{'), UInt8('('), UInt8('['))
                depth += 1
            elseif c in (UInt8('}'), UInt8(')'), UInt8(']'))
                depth == 0 && return false
                depth -= 1
            elseif c == UInt8(',') && depth == 0
                at_start = true
            end
            j += 1
        end
        return false
    end
    union_or_tuple = r"(?<![A-Za-z0-9_])(?:Union|Tuple)\{"
    function bounded_lines(s::AbstractString)
        cu = codeunits(s)
        lines = Int[]
        for m in eachmatch(union_or_tuple, s)
            holds_bounded_member(cu, m.offset + ncodeunits(m.match)) || continue
            push!(lines, count(==(UInt8('\n')), view(cu, 1:(m.offset))) + 1)
        end
        return lines
    end

    @testset "the scan flags a bounded member of a `Union` or a `Tuple`, and no other bound" begin
        for s in
            ("x::Union{<:Real, <:A}", "x::Union{Real, <:A}", "isa(m, Union{<:A, B, C})",
             "x::Option{<:Tuple{<:Number, <:Number}}", "Option{<:Tuple{Number, <:Number}}",
             "x::Union{Nothing,\n           <:A}", "Tuple{<:Integer}",
             "AbstractVector{<:Union{<:Real, A}}", "Union{A{<:B}, <:C}")
            @test (s, bounded_lines(s)) == (s, [1])
        end
        for s in ("x::Option{<:Integer}", "AbstractVector{<:Union{Real, A}}",
                  "NTuple{N, <:Number} where {N}", "Pair{<:Symbol, <:Real}", "Union{A, B{<:C}}",
                  "Union{Nothing, AbstractVector{<:Real}}", "MyUnion{<:A, <:B}",
                  "Tuple{Int, NTuple{8, <:Number}}", "Union{A, Tuple{B, C}}",
                  "where {T1 <: Union{A, B}}")
            @test (s, bounded_lines(s)) == (s, Int[])
        end
        @test bounded_lines("a\nb::Union{A,\n  <:B}\nc::Tuple{<:D}") == [2, 4]
    end

    @testset "no annotation of `src/` or `ext/` names a bounded member of a `Union` or a `Tuple`" begin
        root = pkgdir(PO)
        sites = String[]
        nfiles = 0
        for dir in ("src", "ext"), (d, _, fs) in walkdir(joinpath(root, dir)), f in fs
            endswith(f, ".jl") || continue
            p = joinpath(d, f)
            nfiles += 1
            for l in bounded_lines(read(p, String))
                push!(sites, string(relpath(p, root), ":", l))
            end
        end
        @test nfiles > 300
        @test ("annotations with a bounded member", sites) ==
              ("annotations with a bounded member", String[])
    end

    function walk!(seen, x, depth = 0)
        depth > 8 && return nothing
        if x isa Array && !isbitstype(eltype(x))
            foreach(v -> walk!(seen, v, depth + 1), Iterators.take(x, 4))
            return nothing
        elseif x isa AbstractDict
            foreach(v -> walk!(seen, v, depth + 1), Iterators.take(values(x), 4))
            return nothing
        end
        S = typeof(x)
        (S in keys(seen) || !isstructtype(S)) && return nothing
        (parentmodule(S) === PO || S <: Tuple || S <: NamedTuple) || return nothing
        seen[S] = x
        for i in 1:fieldcount(S)
            isdefined(x, i) && walk!(seen, getfield(x, i), depth + 1)
        end
        return nothing
    end

    seen = Dict{DataType, Any}()
    Logging.with_logger(Logging.NullLogger()) do
        for nm in names(PO; all = true)
            isdefined(PO, nm) || continue
            T = getfield(PO, nm)
            T isa Type || continue
            U = Base.unwrap_unionall(T)
            U isa DataType || continue
            (isabstracttype(U) ||
             !isstructtype(U) ||
             parentmodule(U) !== PO ||
             U <: Function) && continue
            x = try
                Base.invokelatest(T)
            catch
                continue
            end
            walk!(seen, x)
        end
        return nothing
    end
    #=
    A struct that needs an argument has no instance with no arguments, so the loop above
    cannot build it. Each fixture builds one such struct as its docstring example does, and the
    walk reaches the values nested in it.
    =#
    for x in (MeanRisk(; opt = JuMPOptimiser(; slv = Solver(; solver = nothing))),
              FactorBlackLittermanPrior(;
                                        sets = UniverseSets(;
                                                            dict = Dict("nx" => ["A", "B", "C"],
                                                                        "nf" => ["F1", "F2"])),
                                        views = LinearConstraintEstimator(;
                                                                          val = ["F1 == 0.03",
                                                                                 "F2 == 0.04"])),
              TargetReturnForecast(;
                                   scores = DescriptorScores(;
                                                             descriptors = [Passthrough(;
                                                                                        field = "a")]),
                                   target_outlier = nothing))
        walk!(seen, x)
    end

    # Each call is measured through a function, so the call itself is what is counted. The
    # function names the module by its global name: `PO` is a local of the test set, and a call
    # through it resolves the field at run time and allocates by itself.
    keyword_rebuild(x) = PortfolioOptimisers.rebuild_estimator(x, (;))
    function positional_call(x)
        T = typeof(x)
        return T.name.wrapper(ntuple(i -> getfield(x, i), Val(fieldcount(T)))...)
    end
    allocs(f, x) = (f(x); f(x); @allocated f(x))

    #=
    A positional call that fails, for a type whose inner constructor does not take every field,
    counts as zero bytes, so the keyword rebuild of that type must allocate nothing. A type that
    `rebuild_estimator` cannot rebuild at all is skipped: `Frontier` and `LogRegimeAdjusted`
    derive a field that their keyword constructor does not take, and no step reaches them.
    =#
    measured = DataType[]
    costly = Tuple{Symbol, Int, Int}[]
    for (S, x) in seen
        x isa
        Union{PO.AbstractEstimator, PO.AbstractAlgorithm, StatsBase.CovarianceEstimator} ||
            continue
        Logging.with_logger(Logging.NullLogger()) do
            k = try
                allocs(keyword_rebuild, x)
            catch
                return nothing
            end
            p = try
                allocs(positional_call, x)
            catch
                0
            end
            push!(measured, S)
            k > p && push!(costly, (nameof(S), k, p))
            return nothing
        end
    end

    # A floor on the walk: the package has far more such types than this.
    @test length(measured) > 500
    # The fixtures reach estimators that need an argument.
    for T in (MeanRisk, JuMPOptimiser, FactorBlackLittermanPrior, TargetReturnForecast)
        @test any(S -> S <: T, measured)
    end
    sort!(costly)
    @test ("types whose keyword rebuild allocates more than a positional call", costly) ==
          ("types whose keyword rebuild allocates more than a positional call",
           empty(costly))
end
