@testset "Stacking's Combination Weight reaches the combination" begin
    using Test, PortfolioOptimisers, StableRNGs

    #=
    ADR 0053 makes `scale` a Combination Weight and enumerates four sites.
    `Stacking.scale` is a fifth, and before the ADR 0053 amendment the weight reached no
    consumer that kept it.

    `_optimise` built `swi = wi .* transpose(st.scale)` and handed it to
    `predict_outer_st_estimator_returns`. The outer returns matrix ignored it (its columns
    come from `calc_net_returns` on the inner *results*), `iv`, `ivpa` and the feature
    matrix all cancelled it through `synthetic_asset_weights`, and the recombination used
    the unscaled `wi`. The one surviving effect was a tilt of the benchmark column. Both
    cross-validation methods took the matrix and never referenced it, so under any `cv`
    the field was entirely inert.

    The weight now acts at the combination, after the outer solve. The outer *problem*
    never sees it, which is what makes a cross-validated run and a fold-less one agree.
    =#

    PO = PortfolioOptimisers
    rng = StableRNG(987654321)
    X = randn(rng, 300, 8) ./ 100 .+ 0.0005
    ts = PO.Dates.Date(2020, 1, 1) .+ PO.Dates.Day.(0:299)
    rd = ReturnsResult(; nx = string.(1:8), X = X, ts = ts)
    opti = [InverseVolatility(), EqualWeighted(), InverseVolatility()]
    scale = [3.0, 1.0, 1.0]
    stack(s, cv) = Stacking(; opti = opti, opto = InverseVolatility(), scale = s, cv = cv)

    @testset "combination_weights" begin
        v = [0.5, 0.3, 0.2]
        # `nothing` is the identity, on a point and on a frontier alike.
        @test PO.combination_weights(nothing, v) === v
        @test PO.combination_weights(nothing, [v, 0.9 * v]) == [v, 0.9 * v]
        # A uniform weight is neutral whatever it sums to: only ratios carry meaning.
        @test PO.combination_weights(fill(1 / 3, 3), v) ≈ v
        @test PO.combination_weights(fill(9.0, 3), v) ≈ v
        # One element is not a combination, so the weight is inert on a lone element.
        @test PO.combination_weights([5.0], [0.9]) ≈ [0.9]
        # A common factor cancels, which is why the field needs no normalised form.
        @test PO.combination_weights([1.0, 2.0, 3.0], v) ≈
              PO.combination_weights([10.0, 20.0, 30.0], v)
        # The rescale preserves the total the outer optimiser chose, not the literal 1 —
        # rescaling to 1 would silently overrule a `bgt` of 0.9.
        @test sum(PO.combination_weights([1.0, 2.0, 3.0], v)) ≈ sum(v)
        @test sum(PO.combination_weights([1.0, 2.0, 3.0], 0.9 * v)) ≈ 0.9 * sum(v)
        # A frontier applies the same map point by point.
        @test PO.combination_weights([1.0, 2.0, 3.0], [v, 0.9 * v]) ==
              [PO.combination_weights([1.0, 2.0, 3.0], v),
               PO.combination_weights([1.0, 2.0, 3.0], 0.9 * v)]
        # Degenerate combinations stay finite. A zero-total outer allocation gives a zero
        # factor, and a tilt that cancels one gives a non-finite factor; both leave the
        # tilted coefficients unrescaled rather than collapsing or overflowing.
        z = [0.5, -0.3, -0.2]
        @test PO.combination_weights([1.0, 2.0, 3.0], z) ≈ [0.5, -0.6, -0.6]
        @test PO.combination_weights([1.0, 1.0, 1.0], z) ≈ z
        @test all(isfinite, PO.combination_weights([1.0, 2.0, 3.0], z))
    end

    @testset "the weight acts, and acts identically under cv" begin
        for cv in (nothing, OptimisationCrossValidation(; cv = KFold(; n = 4)))
            r0 = optimise(stack(nothing, cv), rd)
            rs = optimise(stack(scale, cv), rd)
            # It is no longer inert. This is the whole defect under `cv`, where the two
            # prediction methods received the scaled matrix and never referenced it.
            @test !isapprox(r0.w, rs.w)
            # The outer problem never sees it, so the outer solve is unchanged. This is
            # what makes the two paths agree: the effect of the weight is one function of
            # one argument, and both arguments are scale-free.
            @test r0.reso.w ≈ rs.reso.w
            wi = hcat([r.w for r in rs.resi]...)
            @test rs.w ≈ wi * PO.combination_weights(scale, r0.reso.w)
            # The combination stays a combination.
            @test sum(rs.w) ≈ sum(r0.w)
            # A uniform weight is neutral end to end, and any common factor of it is too.
            @test optimise(stack(fill(1 / 3, 3), cv), rd).w ≈ r0.w
            @test optimise(stack(fill(9.0, 3), cv), rd).w ≈ r0.w
            @test optimise(stack(30 * scale, cv), rd).w ≈ rs.w
        end
    end

    @testset "a lone inner optimiser is inert" begin
        lone(s) = optimise(Stacking(; opti = [InverseVolatility()],
                                    opto = InverseVolatility(), scale = s), rd).w
        @test lone([5.0]) ≈ lone(nothing)
    end

    @testset "validation is unchanged" begin
        @test_throws DimensionMismatch Stacking(; opti = opti, opto = InverseVolatility(),
                                                scale = [1.0, 2.0])
        @test_throws PO.IsNonFiniteError Stacking(; opti = opti, opto = InverseVolatility(),
                                                  scale = [1.0, 2.0, Inf])
        # The field is stored as written: the rescale makes a common factor unobservable,
        # so there is nothing for a construction-time normalisation to buy.
        @test Stacking(; opti = opti, opto = InverseVolatility(), scale = scale).scale ==
              scale
    end
end
