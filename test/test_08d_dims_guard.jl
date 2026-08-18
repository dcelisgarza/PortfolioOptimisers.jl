#=
The `dims` guard is one decision, not one decision per leaf.

`dims_oriented` fuses the guard (`assert_dims`) with the orientation (`transpose` when
`dims == 2`), so a leaf cannot orient a matrix without validating `dims` first. The two used
to be spelled by hand at 42 sites, and five leaves spelled neither: one user error had four
different outcomes, and `mean(SimpleExpectedReturns(), X; dims = 3)` answered a `4 × 3`
input with that same raw `4 × 3` matrix instead of a `1 × 3` statistic.

Two locks hold the seam shut:

  1. BEHAVIOUR. Every concrete moment estimator answers `dims = 3` with the canonical
     `DomainError`, not with a `StatsBase` message, a `mapslices` message, or the input.
     The census derives its type list from `subtypes`, so it names no estimators and covers
     one added in future the day it is written.

  2. CONSTRUCTION. No file in `src/` spells the orientation branch or the guard by hand.
     This rule names no sites either, so a leaf that re-decides the guard fails here rather
     than in a caller's result.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, InteractiveUtils

const PO = PortfolioOptimisers

# Every concrete subtype of `T`, at any depth.
function concrete_leaves(T::Type)
    out = Type[]
    for S in subtypes(T)
        isabstracttype(S) ? append!(out, concrete_leaves(S)) : push!(out, S)
    end
    return out
end

@testset "dims_oriented: the guard and the orientation are one call" begin
    X = reshape(collect(1.0:12.0), 4, 3)
    F = reshape(collect(1.0:8.0), 4, 2)

    # `dims == 1` is the canonical orientation, so the input comes back untouched.
    @test PO.dims_oriented(1, X) === X
    # `dims == 2` puts the observations on the rows.
    @test PO.dims_oriented(2, X) == transpose(X)
    @test size(PO.dims_oriented(2, X)) == (3, 4)
    # More than one matrix comes back as a tuple, each one oriented.
    @test PO.dims_oriented(1, X, F) === (X, F)
    @test PO.dims_oriented(2, X, F) == (transpose(X), transpose(F))
    @test PO.dims_oriented(2, X, F, F) == (transpose(X), transpose(F), transpose(F))
    # An absent optional matrix needs no branch of its own.
    @test PO.dims_oriented(2, X, nothing) == (transpose(X), nothing)
    @test isnothing(PO.dims_oriented(2, nothing))
    @test isnothing(PO.dims_oriented(1, nothing))
    # The guard cannot be skipped, at any arity.
    @test_throws DomainError PO.dims_oriented(3, X)
    @test_throws DomainError PO.dims_oriented(0, X)
    @test_throws DomainError PO.dims_oriented(3, X, F)
    @test_throws DomainError PO.dims_oriented(3, X, nothing)
    @test_throws DomainError PO.dims_oriented(-1, nothing)
end

@testset "every moment estimator answers dims = 3 with a DomainError" begin
    X = reshape(collect(1.0:12.0), 4, 3)

    # A variance estimator is also a covariance estimator in the hierarchy, but it answers
    # `var`/`std` rather than `cov`/`cor`, so each family is driven by its own verbs.
    ve_types = concrete_leaves(PO.AbstractVarianceEstimator)
    ce_types = setdiff(concrete_leaves(PO.AbstractCovarianceEstimator), ve_types)

    # `RegimeAdjustedExpWeightedCovariance` and `RegimeAdjustedExpWeightedVariance` answer
    # neither `cov` nor `cor` at ANY `dims`: neither declares one, so the generic
    # `cov(::AbstractCovarianceEstimator, X)` falls through to `cor`, which falls back to
    # `cov`, and the call overflows the stack. That is a missing method rather than a `dims`
    # hole -- it reproduces at `dims = 1` -- so it is reported separately. Delete this skip
    # when those two declare their verbs.
    no_verb = (PO.RegimeAdjustedExpWeightedCovariance, PO.RegimeAdjustedExpWeightedVariance)
    ce_types = setdiff(ce_types, no_verb)

    families = [(concrete_leaves(PO.AbstractExpectedReturnsEstimator), [Statistics.mean]),
                (ve_types, [Statistics.var, Statistics.std]),
                (ce_types, [Statistics.cov, Statistics.cor])]

    checked = 0
    for (types, verbs) in families
        @test !isempty(types)
        for S in types, verb in verbs
            est = S()
            # `ImpliedVolatility` reads its own series; the guard runs before it does.
            kwargs = isa(est, PO.ImpliedVolatility) ? (; iv = X) : (;)
            @test_throws DomainError verb(est, X; dims = 3, kwargs...)
            @test_throws DomainError verb(est, X; dims = 0, kwargs...)
            checked += 1
        end
    end
    # The census is closed, so this number moves when an estimator is added. It is here to
    # show that the loop above ran over a real universe rather than an empty one.
    @test checked == 41
end

@testset "no leaf spells the dims guard or the orientation by hand" begin
    srcdir = normpath(joinpath(@__DIR__, "..", "src"))
    canonical = "01_Base.jl"        # the file that owns `assert_dims` and `dims_oriented`
    offenders = String[]
    for (root, _, files) in walkdir(srcdir), f in files
        # `NOTRACK_*` files are not part of the module.
        if !endswith(f, ".jl") || startswith(f, "NOTRACK")
            continue
        end
        path = joinpath(root, f)
        owner = basename(path) == canonical
        for (i, line) in enumerate(eachline(path))
            s = strip(line)
            # The orientation branch: `dims_oriented` owns it, and it takes the guard with
            # it. A hand-written branch is how a leaf ends up orienting without validating.
            hand_orientation = s == "if dims == 2"
            # A hand-rolled guard: the message drifts from the canonical one, which is the
            # second way one user error grows a second outcome.
            hand_guard = !owner && occursin(r"@argcheck\(\s*!?\(?dims\b", s)
            if hand_orientation || hand_guard
                push!(offenders, "$(relpath(path, srcdir)):$i  $s")
            end
        end
    end
    @test isempty(offenders)
    if !isempty(offenders)
        @info "hand-spelled dims guard or orientation" offenders
    end
end
