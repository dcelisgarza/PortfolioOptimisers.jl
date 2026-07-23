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
