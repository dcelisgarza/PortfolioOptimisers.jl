@testset "Keyword rebuild allocation census: a keyword rebuild costs what a positional call costs" begin
    using Test, PortfolioOptimisers, StatsBase, Logging, InteractiveUtils

    #=
    `rebuild_estimator` rebuilds an estimator through its keyword constructor (#1806), and the
    online fold calls it on every step (#1807). A keyword argument whose type annotation holds a
    `Union` of bounded type variables, such as `Union{<:Real, <:AbstractLearningRateSchedule}`,
    or a bound inside a bound, such as `Option{<:Tuple{<:Number, <:Number}}`, makes Julia build
    that type again on each keyword call. The plain spellings, `Union{Real, AbstractLearningRateSchedule}`
    and `Option{<:Tuple{Number, Number}}`, admit the same values and cost nothing.

    The review of #1817 (R6) measured 16 types whose keyword rebuild allocated more than a
    positional call with the same fields: 352 bytes for each online step of the two
    regime-adjusted estimators, and up to 784 bytes for each rebuild of the others. This census
    holds the measure for every concrete type of the package that has a constructor of no
    arguments. A type with no such constructor is not reached here, and #1824 fixes the spelling
    in the rest of the library.

    A positional call that fails, for a type whose inner constructor does not take every field,
    counts as zero bytes, so the keyword rebuild of that type must allocate nothing. A type that
    `rebuild_estimator` cannot rebuild at all is skipped.
    =#
    PO = PortfolioOptimisers

    function concrete_types(T, acc = Set{Any}())
        for S in InteractiveUtils.subtypes(T)
            (isconcretetype(S) || (S isa UnionAll && !isabstracttype(S))) && push!(acc, S)
            isabstracttype(S) && concrete_types(S, acc)
        end
        return acc
    end
    types = Set{Any}()
    for r in (PO.AbstractEstimator, PO.AbstractAlgorithm, StatsBase.CovarianceEstimator)
        concrete_types(r, types)
    end
    types = [t
             for t in types
             if parentmodule(t isa UnionAll ? Base.unwrap_unionall(t) : t) === PO]

    # Each call is measured through a function, so the call itself is what is counted. The
    # function names the module by its global name: `PO` is a local of the test set, and a call
    # through it resolves the field at run time and allocates by itself.
    keyword_rebuild(x) = PortfolioOptimisers.rebuild_estimator(x, (;))
    function positional_call(x)
        T = typeof(x)
        return T.name.wrapper(ntuple(i -> getfield(x, i), Val(fieldcount(T)))...)
    end
    allocs(f, x) = (f(x); f(x); @allocated f(x))

    measured = 0
    costly = Tuple{Symbol, Int, Int}[]
    for T in types
        x = try
            Logging.with_logger(Logging.NullLogger()) do
                return Base.invokelatest(T)
            end
        catch
            continue
        end
        Logging.with_logger(Logging.NullLogger()) do
            # `Frontier` and `LogRegimeAdjusted` derive a field that their keyword constructor
            # does not take, so `rebuild_estimator` cannot rebuild them. No step reaches them.
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
            measured += 1
            k > p && push!(costly, (nameof(T), k, p))
            return nothing
        end
    end

    # A floor on the walk: the package has far more such types than this.
    @test measured > 500
    sort!(costly)
    @test ("types whose keyword rebuild allocates more than a positional call", costly) ==
          ("types whose keyword rebuild allocates more than a positional call",
           empty(costly))
end
