@testset "Combination weight: scale is inert on a lone term" begin
    using PortfolioOptimisers, Test, StableRNGs, Clarabel, JuMP, LinearAlgebra

    #=
    `scale` is a combination weight: it says how much an element contributes to an
    expression built from several elements. One element is not a combination, so the weight
    has nothing to weigh and both axes drop it before the expression is built.

    These assertions are structural, so they need no solver and cannot drift with a
    tolerance. They fail loudly if a future builder reintroduces the multiplier.
    =#

    rng = StableRNG(987654321)
    X = randn(rng, 300, 10) ./ 100 .+ 0.0005
    slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    pr = prior(EmpiricalPrior(), X)
    scales = (1, 2, 50)

    ret_opt(sc) = JuMPOptimiser(; pe = pr, slv = slv,
                                ret = ArithmeticReturn(;
                                                       settings = JuMPReturnsSettings(;
                                                                                      scale = sc)))
    risk_r(sc) = StandardDeviation(; settings = RiskMeasureSettings(; scale = sc))

    @testset "Return axis: one term" begin
        # A lone term contributes its own expression, never a multiple of it.
        for sc in scales
            res = optimise(MeanRisk(; r = StandardDeviation(), obj = MinimumRisk(),
                                    opt = ret_opt(sc)); save = true)
            @test JuMP.isequal_canonical(res.model[:ret], res.model[:ret_1])
        end
    end

    @testset "Return axis: several terms still combine" begin
        # The weight must survive where there really is a combination.
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            ret = [ArithmeticReturn(;
                                                    settings = JuMPReturnsSettings(;
                                                                                   scale = 2)),
                                   ArithmeticReturn(;
                                                    settings = JuMPReturnsSettings(;
                                                                                   scale = 3))])
        res = optimise(MeanRisk(; r = StandardDeviation(), obj = MinimumRisk(), opt = opt);
                       save = true)
        @test !JuMP.isequal_canonical(res.model[:ret], res.model[:ret_1])
    end

    @testset "Risk axis: one measure" begin
        # `StandardDeviation` puts an epigraph variable in `:risk`, so a surviving weight
        # would show up as a coefficient other than one.
        for sc in scales
            res = optimise(MeanRisk(; r = risk_r(sc), obj = MinimumRisk(),
                                    opt = ret_opt(1)); save = true)
            @test all(isone, values(res.model[:risk].terms))
        end
    end

    @testset "Risk axis: several measures still combine" begin
        res = optimise(MeanRisk(; r = [risk_r(2), ConditionalValueatRisk()],
                                obj = MinimumRisk(), opt = ret_opt(1)); save = true)
        @test !all(isone, values(res.model[:risk].terms))
    end

    @testset "Numerical: the portfolio does not move" begin
        #=
        `MaximumUtility` is the objective that makes a lone weight observable: it optimises
        `ret - l * risk`, so a weight on either axis changes the trade-off rather than
        cancelling. `MinimumRisk`, `MaximumReturn` and `MaximumRatio` are argmax-invariant,
        which is why the return axis carried the defect unnoticed.
        =#
        for obj in (MinimumRisk(), MaximumUtility(), MaximumRatio(), MaximumReturn())
            w_ret = map(scales) do sc
                optimise(MeanRisk(; r = StandardDeviation(), obj = obj, opt = ret_opt(sc))).w
            end
            @test all(w -> w == first(w_ret), w_ret)

            w_risk = map(scales) do sc
                optimise(MeanRisk(; r = risk_r(sc), obj = obj, opt = ret_opt(1))).w
            end
            @test all(w -> w == first(w_risk), w_risk)
        end
    end
end
