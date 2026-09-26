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

Lock 1 can only drive a leaf that answers its family's verbs, because a leaf that answers
none of them overflows the stack before it reaches a guard.
`test_08l_moment_verb_census.jl` is the census that holds that set, and
`moment_family_setup.jl` holds the one predicate both files read.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, InteractiveUtils

const PO = PortfolioOptimisers

# The family split and the ownership predicate. `test_08l_moment_verb_census.jl` reads the
# same two, and a second copy of either is how the two censuses drift apart.
include(joinpath(@__DIR__, "moment_family_setup.jl"))

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

    #=
    Each family is driven by its own verbs, and only a leaf that ANSWERS them can be driven
    at all. A leaf that owns neither `cov` nor `cor` reaches the surface's fallback, which
    reaches `StatsBase`'s generic `cor`, which reaches the fallback again: the call overflows
    the stack at any `dims`, so it reaches no guard to test. That is a missing method rather
    than a `dims` hole -- it reproduces at `dims = 1` -- and
    `test_08l_moment_verb_census.jl` is the census that owns it. This file names no leaf: it
    takes the same predicate, so the day that census admits a leaf, this one drives it.
    =#
    leaves = moment_families()
    families = [(filter(S -> answers_family(f, S), getproperty(leaves, f)),
                 family_verbs(f)) for f in (:er, :ve, :ce)]

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
    # show that the loop above ran over a real universe rather than an empty one. Issue #854
    # added `ExpWeightedExpectedReturns`, `ExpWeightedVariance` and `ExpWeightedCovariance`,
    # which carry one, two and two verbs, so the number rose by five. The online portfolio
    # selection family added `PriceLevelExpectedReturns` and `PriorExpectedReturns` (one verb
    # each) and `RankOneCovariance` (two), so it rose by four. `GeodesicShrinkageCovariance`
    # carries two verbs, so it rose by two.
    @test checked == 56
end

@testset "a panel arm answers the same at both orientations" begin
    #=
    An Asset Panel carries its masks as `observations × assets` whatever `dims` says, and
    every mask-aware root reads a mask shaped as its sample is: it orients the two together
    with `dims_oriented`. So the panel arm of a verb must turn the mask where the caller
    turns the sample. An arm that hands the panel's mask through unturned compares a
    `T × N` mask with an `N × T` sample, and it throws a `DimensionMismatch` at
    `dims = 2` alone, while the reduce-and-expand root of the same verb answers. One verb
    then answered a transposed sample for a plain estimator and refused it for a mask-aware
    one (issue #1249).

    The rule belongs to the family rather than to the leaf, so this census drives the same
    pairings the `dims = 3` census drives, and it names no leaf either. The two orientations
    must agree, in the answer and in the refusal alike: a leaf this one fixture cannot drive
    refuses at both, and a leaf it can drive answers the same statistic at both.
    =#
    T, N = 30, 5
    R = 0.02 .* reshape(sinpi.(collect(1:(T * N)) ./ 7), T, N)
    amsk = trues(T, N)
    amsk[1:8, N] .= false                # the last asset lists at observation 9
    R[.!amsk] .= NaN
    pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk))
    Rt = permutedims(R)

    leaves = moment_families()
    families = [(filter(S -> answers_family(f, S), getproperty(leaves, f)),
                 family_verbs(f)) for f in (:er, :ve, :ce)]
    for (types, verbs) in families
        @test !isempty(types)
        for S in types, verb in verbs
            est = S()
            # `ImpliedVolatility` reads its own series, which is oriented as the sample is.
            kw1 = isa(est, PO.ImpliedVolatility) ? (; iv = R) : (;)
            kw2 = isa(est, PO.ImpliedVolatility) ? (; iv = Rt) : (;)
            a1 = try
                verb(est, R, pnl; dims = 1, kw1...)
            catch err
                err
            end
            a2 = try
                verb(est, Rt, pnl; dims = 2, kw2...)
            catch err
                err
            end
            if isa(a1, Exception)
                @test isa(a2, typeof(a1))
            else
                @test !isa(a2, Exception)
                # The reduce-and-expand root sums a transposed block in another order, so
                # the two answers agree to the last few bits rather than bit for bit. The
                # mask-aware arms answer bit for bit, and a mask that arrives unturned is
                # a `DimensionMismatch` rather than a small difference, so the tolerance
                # costs the census nothing.
                @test isapprox(vec(a1), vec(a2); nans = true)
            end
        end
    end
end

@testset "no leaf spells the dims guard or the orientation by hand" begin
    srcdir = normpath(joinpath(@__DIR__, "..", "src"))
    canonical = "10_Assertions.jl"  # the file that owns `assert_dims` and `dims_oriented`
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
