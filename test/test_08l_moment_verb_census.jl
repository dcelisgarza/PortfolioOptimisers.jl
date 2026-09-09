#=
Choice Surface membership means the verbs exist.

Subtyping `AbstractCovarianceEstimator` and exporting the leaf is the whole contract a caller
relies on. Two exported leaves satisfy the type and not the behaviour, and nothing said so
until the caller ran them:

```julia-repl
julia> cov(RegimeAdjustedExpWeightedCovariance(), X)
ERROR: StackOverflowError:
```

The surface's own fallbacks are what turn the omission into that crash rather than into a
`MethodError`. `cov(ce::AbstractCovarianceEstimator, X)` reads `cor(ce, X)`; a leaf with no
`cor` of its own reaches `StatsBase`'s generic `cor(::CovarianceEstimator, ::AbstractMatrix)`,
which reads `cov(ce, X)` straight back. The pair is a cycle with no base case, so the interface
is the same width as every sibling's and the implementation behind it is empty.

This census closes the promise. Every concrete leaf of the three moment families answers its
family's verbs, and the answer is a method `PortfolioOptimisers` declares below the surface.
The rule names no leaf, so a leaf added in future is covered the day it is written -- the
closed polarity of ADR 0037's rules, which ADR 0058's two censuses already copy.

`moment_family_setup.jl` holds the split and the ownership predicate, because
`test_08d_dims_guard.jl` reads the same two and the two files must not drift.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, InteractiveUtils

include(joinpath(@__DIR__, "moment_family_setup.jl"))

@testset "Choice Surface membership means the verbs exist" begin
    families = moment_families()

    # A `subtypes` walk that answered nothing would make every check below vacuously green.
    for family in (:er, :ve, :ce)
        @test !isempty(getproperty(families, family))
    end

    # ------------------------------------------------- 1. every leaf answers its family

    offenders = String[]
    for family in (:er, :ve, :ce), S in getproperty(families, family)
        if nameof(S) in VERB_EXEMPT || answers_family(family, S)
            continue
        end
        verbs = join(string.(nameof.(family_verbs(family))), "/")
        push!(offenders, "$(nameof(S))  [$(family) family: $(verbs)]")
    end

    @test isempty(offenders)
    if !isempty(offenders)
        println("Moment estimators that subtype a Choice Surface and do not answer its ",
                "verbs. A caller reaches a stack overflow rather than a result. Declare the ",
                "family's verbs for each leaf:")
        for o in offenders
            println("  ", o)
        end
    end

    # ------------------------------------------------- 2. the exemption is empty

    #=
    An exemption that has been paid must be deleted, or the next reader takes it for a standing
    decision. Issue #637 paid the only one this census ever carried, so the tuple is empty and
    the first half above names every leaf without exception.
    =#
    @test isempty(VERB_EXEMPT)

    # ------------------------------------------- 3. the Correlation Rescale keeps its word

    #=
    A leaf may take `cov(ce::AbstractCovarianceEstimator, X)` in place of its own `cov`. That
    fallback reads `ce.ve`, which no type bound states, so a leaf that takes it without
    carrying a variance estimator raises a `FieldError` from inside the library. Four leaves
    take it today, and this holds the fallback's precondition for all of them.
    =#
    rescaled = filter(S -> takes_correlation_rescale(S), families.ce)
    @test !isempty(rescaled)
    for S in rescaled
        @test hasfield(typeof(S()), :ve)
        @test owns_verb(Statistics.cor, S)
    end
end

#=
The asset-axis tripwire: a per-asset field is declared, never inferred.

Under the reduce-and-expand contract of ADR 0115 an optimiser takes a `port_opt_view` of its
moment estimators at the Investable Mask. The seam slices the SAMPLE; it cannot know which of an
estimator's own fields are one entry per asset of the full universe, so the estimator declares
them -- with an `@vprop` tag on a field that is always on the asset axis, or with a
`port_opt_view` method on the parameterisation whose field is.

The two methods on the surfaces are the identity, which is right for the many estimators that
carry no asset-axis field and wrong, silently, for one that does. This census names the leaves
that take the identity, so that a leaf added with a per-asset field reds the build until someone
either declares the axis or states here why it has none. Closed polarity, as ADR 0037's rules
and the censuses of ADR 0058 have it.

The probe asks about `typeof(S())` rather than `S`, for the reason `dispatched_method` gives: a
parametric leaf arrives as a `UnionAll`, and `which` on one reads the fallback whatever the leaf
declares.
=#

# The slots a `port_opt_view` resolves AT rather than below. `StatsBase.CovarianceEstimator`
# joins the three surfaces because the covariance identity is declared on it, not on
# `AbstractCovarianceEstimator`.
const VIEW_SURFACES = (MOMENT_SURFACES..., StatsBase.CovarianceEstimator)

# `true` when `port_opt_view` on `me` reaches one of those slots.
function instance_takes_view_identity(me)
    m = try
        which(PortfolioOptimisers.port_opt_view,
              Tuple{typeof(me), Vector{Int}, Matrix{Float64}})
    catch
        return false
    end
    if !(m.module === PortfolioOptimisers)
        return false
    end
    return any(surface -> m.sig.parameters[2] === surface, VIEW_SURFACES)
end

# The same question of a leaf TYPE, asked of the instance its default constructor gives. A leaf
# whose asset axis depends on a type parameter answers only for that default shape, which is why
# `CustomValueExpectedReturns` is on the list below and its three shapes are pinned one by one.
function takes_view_identity(S::Type)
    me = try
        S()
    catch
        return false
    end
    return instance_takes_view_identity(me)
end

#=
Every leaf that takes the identity, and why it carries no asset-axis field.

Measured under #667, by building each estimator and taking a `port_opt_view` of it.

  - `MedianExpectedReturns` and `DistanceCovariance` carry `w::Option{<:ObsWeights}` alone. Those
    are OBSERVATION weights, tagged `@wprop`, and `obs_weights_view` is their verb; the asset axis
    holds nothing.
  - `CustomValueExpectedReturns` defaults to `val = 0.0`, so the shape its default constructor
    gives is the scalar one, and a scalar is universe-independent. Its per-asset shape,
    `CustomValueExpectedReturns{<:VecNum}`, declares its axis with a `port_opt_view` method, and
    the three shapes are pinned one by one at the end of this testset. A leaf whose asset axis
    depends on a type parameter is the one case this census cannot read off the default instance.
  - The five exponentially weighted and regime-adjusted estimators carry `cache`, bound to
    `Option{<:AbstractPartialFitState}`. Its arrays ARE one entry per asset, and slicing them
    would still change no answer: no consumer reads the cache, which is the exemption ADR 0106
    grants and the reason ADR 0105 hides the field. `mean`, `var` and `cov` rebuild from the
    sample they are handed, measured equal to the same verb on an estimator carrying no cache.

A name added here must carry its own reason, and a leaf with a genuine per-asset field belongs in
a `port_opt_view` method instead -- `CustomValueExpectedReturns{<:VecNum}` is the worked example.
=#
const VIEW_IDENTITY_ALLOWED = Set([:MedianExpectedReturns, :DistanceCovariance,
                                   :CustomValueExpectedReturns, :ExpWeightedExpectedReturns,
                                   :ExpWeightedVariance, :ExpWeightedCovariance,
                                   :RegimeAdjustedExpWeightedVariance,
                                   :RegimeAdjustedExpWeightedCovariance])

@testset "The asset-axis tripwire: a per-asset field is declared, never inferred" begin
    families = moment_families()
    for family in (:er, :ve, :ce)
        @test !isempty(getproperty(families, family))
    end

    takers = Set{Symbol}()
    for family in (:er, :ve, :ce), S in getproperty(families, family)
        if takes_view_identity(S)
            push!(takers, nameof(S))
        end
    end

    # The census is worthless if it walked nothing, and the identity is the common case.
    @test length(takers) >= 5

    # A leaf that takes the identity and is not accounted for.
    unexplained = sort(collect(setdiff(takers, VIEW_IDENTITY_ALLOWED)))
    @test isempty(unexplained)
    if !isempty(unexplained)
        println("Moment estimators that take the `port_opt_view` identity and are not ",
                "accounted for. Under ADR 0115 a reduction slices the sample alone, so a field ",
                "of one entry per asset reaches a narrower sample than it describes. Declare ",
                "the asset axis with an `@vprop` tag or a `port_opt_view` method, or state in ",
                "`VIEW_IDENTITY_ALLOWED` why the leaf has no asset-axis field:")
        for u in unexplained
            println("  ", u)
        end
    end

    # And the reverse: a leaf that gained a declaration leaves the list, so the reason stops
    # being carried for a type that no longer needs one.
    stale = sort(collect(setdiff(VIEW_IDENTITY_ALLOWED, takers)))
    @test isempty(stale)
    if !isempty(stale)
        println("Names in `VIEW_IDENTITY_ALLOWED` that no longer take the identity. Each now ",
                "declares its asset axis, so remove it from the list:")
        for s in stale
            println("  ", s)
        end
    end

    # The worked example, shape by shape. Only the per-asset one declares an axis.
    i = [1, 2, 4]
    X = rand(8, 3)

    # A stored vector is one entry per asset, so it is sliced, and `mean` then answers the
    # sample it was handed rather than throwing a `DimensionMismatch` against it.
    mev = CustomValueExpectedReturns(; val = [0.1, 0.2, 0.3, 0.4])
    @test !instance_takes_view_identity(mev)
    @test collect(PortfolioOptimisers.port_opt_view(mev, i, X).val) == [0.1, 0.2, 0.4]
    @test_throws DimensionMismatch Statistics.mean(mev, X)
    @test Statistics.mean(PortfolioOptimisers.port_opt_view(mev, i, X), X) == [0.1 0.2 0.4]

    # A scalar is universe-independent, so the reduction carries it through whole.
    mes = CustomValueExpectedReturns(; val = 0.05)
    @test instance_takes_view_identity(mes)
    @test PortfolioOptimisers.port_opt_view(mes, i, X) === mes

    # As is a callable, which answers one value per column of the matrix it is handed.
    mef = CustomValueExpectedReturns(;
                                     val = (X; dims = 1, kwargs...) -> fill(0.01,
                                                                            size(X, 2)))
    @test instance_takes_view_identity(mef)
    @test PortfolioOptimisers.port_opt_view(mef, i, X) === mef
end
