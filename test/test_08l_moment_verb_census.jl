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

    # ------------------------------------------------- 2. the exemption is not stale

    #=
    An exemption that has been paid must be deleted, or the next reader takes it for a
    standing decision. Each name is still a leaf of the covariance family, and each still
    fails the check the list excuses it from. The day issue #637 lands the missing
    mathematics, this half reds the build and the tuple empties.
    =#
    for name in VERB_EXEMPT
        S = getfield(PortfolioOptimisers, name)
        @test S in families.ce
        @test !answers_family(:ce, S)
    end

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
