@testset "Meta-optimiser scalar weight bounds are asset-dimensioned" begin
    using Test, PortfolioOptimisers, StableRNGs

    #=
    A scalar weight bound means "this bound applies to every asset". Meta-optimisers used to
    expand it to the wrong length: `Stacking` passed `N = Ni` (the number of inner
    optimisers) and `NestedClustered` passed `N = clr.k` (the number of clusters) to
    `weight_bounds_constraints`. But `outer_optimisation_finaliser` maps the outer weights
    back through `wi` into asset space *before* applying the bound, so the bound must be
    asset-length.

    That produced two failures, and the first is why these tests assert on weights rather
    than only on lengths:

      1. Silently wrong answers. `opt_weight_bounds` guards with
         `map((x, y) -> x > y, lb, w)`, and `map` truncates to its shorter argument, so only
         the first `Ni` (or `clr.k`) assets were ever checked. A bound violated by a later
         asset was neither detected nor enforced, and the optimiser returned success.
      2. An opaque crash. Once the truncated guard *did* see a violation, the clamping loop
         reached a real broadcast (`w .< ub .&& w .> lb`) and threw `DimensionMismatch`,
         naming neither the bound nor the optimiser.

    The default bounds are `nothing`, which expands to ±Inf and can never be violated, so
    the early return fired before either path — which is why no existing test caught this.
    =#
    N = 5
    rng = StableRNG(20260721)
    X = randn(rng, 250, N) ./ 100
    #= Assets 4 and 5 get much lower volatility, so InverseVolatility overweights them and
    the largest weights land *beyond* the truncation point. Without that the truncated guard
    would have caught the violation by luck and the regression would not reproduce. =#
    X[:, 5] ./= 8
    X[:, 4] ./= 4
    rd = ReturnsResult(; nx = string.("A", 1:N), X = X)

    ub = 0.30
    lb = 0.15

    @testset "Stacking" begin
        Ni = 2
        unbounded = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                      opto = EqualWeighted()), rd)
        # Guard: the test only exercises truncation if the violation is past index Ni.
        @test argmax(unbounded.w) > Ni
        @test maximum(unbounded.w) > ub

        res = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                opto = EqualWeighted(),
                                wb = WeightBounds(; lb = 0.0, ub = ub)), rd)
        # Pre-fix: unenforced past index Ni, so this returned maximum(w) ≈ 0.366.
        @test all(res.w .<= ub + 1e-10)
        @test length(res.wb.ub) == N
        @test length(res.wb.lb) == N
        @test isapprox(sum(res.w), 1.0)
        @test isa(res.retcode, OptimisationSuccess)

        # Violated within the first Ni: pre-fix this reached the clamping broadcast and threw.
        resl = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                 opto = EqualWeighted(),
                                 wb = WeightBounds(; lb = lb, ub = 1.0)), rd)
        @test all(resl.w .>= lb - 1e-10)
        @test isapprox(sum(resl.w), 1.0)
        @test isa(resl.retcode, OptimisationSuccess)
    end

    @testset "NestedClustered" begin
        unbounded = optimise(NestedClustered(; opti = InverseVolatility(),
                                             opto = EqualWeighted()), rd)
        k = unbounded.clr.k
        # Guard: the same reasoning, with clr.k as the truncation point.
        @test k < N
        @test argmax(unbounded.w) > k
        @test maximum(unbounded.w) > ub

        res = optimise(NestedClustered(; opti = InverseVolatility(), opto = EqualWeighted(),
                                       wb = WeightBounds(; lb = 0.0, ub = ub)), rd)
        @test all(res.w .<= ub + 1e-10)
        @test length(res.wb.ub) == N
        @test length(res.wb.lb) == N
        @test isapprox(sum(res.w), 1.0)
        @test isa(res.retcode, OptimisationSuccess)
    end

    @testset "vector bounds are unaffected" begin
        #=
        `weight_bounds_constraints_side(wb::VecNum, args...) = wb` ignores N entirely, so
        vector bounds never had the defect. This pins that the fix did not change them —
        and covers everything a Pipeline routes in, which is always asset-length.
        =#
        vec_res = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                    opto = EqualWeighted(),
                                    wb = WeightBounds(; lb = fill(0.0, N),
                                                      ub = fill(ub, N))), rd)
        sca_res = optimise(Stacking(; opti = [EqualWeighted(), InverseVolatility()],
                                    opto = EqualWeighted(),
                                    wb = WeightBounds(; lb = 0.0, ub = ub)), rd)
        # A scalar bound must now be exactly equivalent to the vector spelling of it.
        @test isapprox(vec_res.w, sca_res.w)
    end
end

@testset "Weight bound resolution keeps its sign and its length" begin
    using Test, PortfolioOptimisers, StableRNGs

    #=
    Two defects that issue #517 found while it swept
    `src/12_ConstraintGeneration/05_WeightBoundsConstraintGeneration.jl`.

      1. An infinite scalar bound lost its sign. `weight_bounds_constraints_side(wb::Number,
         N, val)` filled with `val` whenever `isinf(wb)`, and `val` is the free bound of the
         side: `-Inf` below and `Inf` above. So a lower bound of `+Inf`, which admits no
         weight at all, resolved to `-Inf`, which admits every weight. The pair
         `WeightBounds(; lb = Inf, ub = Inf)` passes validation, because `Inf <= Inf`, so a
         caller reached it. The branch exists because `range(Inf, Inf; length = 3)` collects
         to `NaN`, and it is kept with `wb` in place of `val`.
      2. A vector bound of the wrong length passed through unchecked. Every reader compares a
         bound with the weights through `map`, which truncates to its shorter argument, so
         each asset past the end of the bound went unchecked and the optimiser reported
         success on a portfolio that broke the bound. Measured before the fix, with five
         assets and a two-entry `ub` of `0.3`: asset five took `0.646`.
    =#
    @testset "an infinite scalar bound keeps its sign" begin
        both_up = weight_bounds_constraints(WeightBounds(; lb = Inf, ub = Inf); N = 3)
        # Pre-fix: `lb` came back as `[-Inf, -Inf, -Inf]`.
        @test all(isequal(Inf), both_up.lb)
        @test all(isequal(Inf), both_up.ub)

        both_down = weight_bounds_constraints(WeightBounds(; lb = -Inf, ub = -Inf); N = 3)
        # Pre-fix: `ub` came back as `[Inf, Inf, Inf]`.
        @test all(isequal(-Inf), both_down.lb)
        @test all(isequal(-Inf), both_down.ub)

        # The free box is unchanged, and it is the case every default takes.
        free = weight_bounds_constraints(WeightBounds(; lb = -Inf, ub = Inf); N = 3)
        @test all(isequal(-Inf), free.lb)
        @test all(isequal(Inf), free.ub)
        @test !any(isnan, free.lb)
        @test !any(isnan, free.ub)

        # A finite scalar still expands to a constant range rather than to an `Array`.
        finite = weight_bounds_constraints(WeightBounds(0.0, 1.0); N = 3)
        @test finite.lb == zeros(3)
        @test finite.ub == ones(3)
        @test !isa(finite.lb, Array)
    end

    @testset "a vector bound must match the asset count" begin
        @test_throws DimensionMismatch weight_bounds_constraints(WeightBounds([0.0, 0.0],
                                                                              [0.3, 0.3]);
                                                                 N = 5)
        # The mixed pair takes the other method, and it is guarded too.
        @test_throws DimensionMismatch weight_bounds_constraints(WeightBounds([0.0, 0.0],
                                                                              0.3); N = 5)
        @test_throws DimensionMismatch PortfolioOptimisers.weight_bounds_constraints_side([0.1,
                                                                                           0.2],
                                                                                          5,
                                                                                          -Inf)
        # A matching length passes, and so does a caller that names no asset count.
        matched = weight_bounds_constraints(WeightBounds([0.0, 0.0], [0.3, 0.3]); N = 2)
        @test matched.lb == [0.0, 0.0]
        @test weight_bounds_constraints(WeightBounds([0.0, 0.0], [0.3, 0.3])).ub ==
              [0.3, 0.3]
        @test PortfolioOptimisers.weight_bounds_constraints_side([0.1, 0.2]) == [0.1, 0.2]
    end

    @testset "the truncated guard is now a raise" begin
        #= The end-to-end shape of defect 2. Asset five carries the largest weight, so a
        two-entry bound never reaches it. =#
        N = 5
        rng = StableRNG(20260826)
        X = randn(rng, 200, N) ./ 100
        X[:, 5] ./= 8
        rd = ReturnsResult(; nx = string.("A", 1:N), X = X)

        full = optimise(InverseVolatility(; wb = WeightBounds(fill(0.0, N), fill(0.3, N))),
                        rd)
        @test all(full.w .<= 0.3 + 1e-10)
        @test isa(full.retcode, OptimisationSuccess)

        # Pre-fix this returned `OptimisationSuccess` with `w[5] ≈ 0.646`.
        @test_throws DimensionMismatch optimise(InverseVolatility(;
                                                                  wb = WeightBounds([0.0,
                                                                                     0.0],
                                                                                    [0.3,
                                                                                     0.3])),
                                                rd)
    end
end

@testset "validate_bounds dispatches every method it declares" begin
    using Test, PortfolioOptimisers

    #=
    Seven methods, and the pair of argument types alone decides which one runs. Two of the
    seven exist to catch a `nothing` on one side, and the catch-all closes the family with no
    check at all. Each block below selects one method and drives its raises.
    =#
    vb = PortfolioOptimisers.validate_bounds

    @testset "(Number, Number)" begin
        @test isnothing(vb(0.0, 1.0))
        @test_throws DomainError vb(0.3, 0.2)
    end

    @testset "(VecNum, Number)" begin
        @test isnothing(vb([0.1, 0.2], 0.5))
        @test_throws PortfolioOptimisers.IsEmptyError vb(Float64[], 0.5)
        @test_throws DomainError vb([0.1, 0.9], 0.5)
    end

    @testset "(Number, VecNum)" begin
        @test isnothing(vb(0.05, [0.1, 0.2]))
        @test_throws PortfolioOptimisers.IsEmptyError vb(0.5, Float64[])
        @test_throws DomainError vb(0.5, [0.9, 0.1])
    end

    @testset "(VecNum, VecNum)" begin
        @test isnothing(vb([0.1, 0.2], [0.3, 0.4]))
        @test_throws PortfolioOptimisers.IsEmptyError vb(Float64[], [0.1])
        @test_throws PortfolioOptimisers.IsEmptyError vb([0.1], Float64[])
        # The length check runs before the comparison, so `map` never truncates a pair here.
        @test_throws DimensionMismatch vb([0.1, 0.2], [0.3])
        @test_throws DomainError vb([0.1, 0.9], [0.3, 0.5])
    end

    @testset "(VecNum, Any) and (Any, VecNum)" begin
        @test isnothing(vb([0.1, 0.2], nothing))
        @test isnothing(vb(nothing, [0.1, 0.2]))
        @test_throws PortfolioOptimisers.IsEmptyError vb(Float64[], nothing)
        @test_throws PortfolioOptimisers.IsEmptyError vb(nothing, Float64[])
    end

    @testset "the catch-all checks nothing" begin
        @test isnothing(vb(nothing, nothing))
        @test isnothing(vb(0.9, nothing))
        @test isnothing(vb(nothing, 0.1))
    end

    @testset "the estimator route validates the resolved pair" begin
        sets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        #= The estimator's own constructor compares a `Dict` with a number with nothing,
        because neither side is resolved yet. `weight_bounds_constraints` builds a
        `WeightBounds` from the resolved pair, and that constructor compares them. =#
        loose = WeightBoundsEstimator(; lb = Dict("A" => 0.9), ub = 0.5)
        @test_throws DomainError weight_bounds_constraints(loose, sets)

        # `dlb` and `dub` reach every asset the estimator does not name.
        named = weight_bounds_constraints(WeightBoundsEstimator(; lb = Dict("A" => 0.1),
                                                                ub = Dict("A" => 0.5),
                                                                dlb = 0.02, dub = 0.7),
                                          sets)
        @test named.lb == [0.1, 0.02, 0.02]
        @test named.ub == [0.5, 0.7, 0.7]

        # A `nothing` side survives the resolution as `nothing`.
        @test isnothing(weight_bounds_constraints(WeightBoundsEstimator(; lb = nothing,
                                                                        ub = 1.0), sets).lb)
        @test isnothing(weight_bounds_constraints(WeightBoundsEstimator(; lb = 0.0,
                                                                        ub = nothing),
                                                  sets).ub)
    end

    @testset "the estimator compares two vector bounds" begin
        #= The branch the estimator's constructor takes when both sides are numbers or
        vectors of numbers. A `Dict`, a `Pair` or an algorithmic rule never reaches it.
        The constructor states no check of its own here: it hands the pair to
        `validate_bounds`, which owns the length check and the entry-by-entry
        comparison. =#
        both = WeightBoundsEstimator(; lb = [0.0, 0.1], ub = [0.8, 0.9])
        @test both.lb == [0.0, 0.1]
        @test both.ub == [0.8, 0.9]
        @test_throws DimensionMismatch WeightBoundsEstimator(; lb = [0.0, 0.1], ub = [0.8])
        @test_throws DomainError WeightBoundsEstimator(; lb = [0.9, 0.1], ub = [0.8, 0.9])

        # A scalar pair reaches the same branch, and `validate_bounds` compares it too.
        @test_throws DomainError WeightBoundsEstimator(; lb = 0.9, ub = 0.5)
        # `dlb` and `dub` are compared with each other whenever both are given.
        @test_throws DomainError WeightBoundsEstimator(; dlb = 0.9, dub = 0.5)

        #= The constructor's own empty guards name the side they refuse, the way the
        `validate_bounds` siblings do. A bare exception type leaves that to ArgCheck's
        generated text, and the two shapes then read differently for one rule. =#
        err_lb = try
            WeightBoundsEstimator(; lb = Float64[])
            nothing
        catch e
            e
        end
        @test isa(err_lb, PortfolioOptimisers.IsEmptyError)
        @test err_lb.msg == "lb cannot be empty"
        err_ub = try
            WeightBoundsEstimator(; ub = Float64[])
            nothing
        catch e
            e
        end
        @test isa(err_ub, PortfolioOptimisers.IsEmptyError)
        @test err_ub.msg == "ub cannot be empty"
    end

    @testset "the default asset count builds empty bounds and is refused" begin
        @test_throws PortfolioOptimisers.IsEmptyError weight_bounds_constraints(WeightBounds(0.0,
                                                                                             1.0))
        @test_throws PortfolioOptimisers.IsEmptyError weight_bounds_constraints(nothing)
    end
end
