# The sweep of the turnover risk measure (#608). The functor is the two-way turnover, the full
# sum of the absolute changes, which the code does not halve. A measure always holds reference
# weights, and `factory` with no weights keeps them. The model of an optimisation reports the
# same number, scaled by the budget variable of a ratio problem. One defect: the constructor
# accepted a `NaN` or an `Inf` reference weight, which `Turnover` and `WeightsTracking` refuse.
using Clarabel, JuMP

const PO = PortfolioOptimisers

const _w0_608 = [0.5, 0.25, 0.25]
const _w_608 = [0.25, 0.5, 0.25]

@testset "Turnover risk measure" begin
    @testset "Two-way turnover against a hand computation" begin
        r = TurnoverRiskMeasure(; w = _w0_608)
        # A weight of 0.25 moves from the first asset to the second: 0.25 out plus 0.25 in.
        @test r(_w_608) == 0.5
        @test r(_w_608) == sum(abs.(_w_608 - _w0_608))
        # The one-way convention halves the sum, and the measure does not.
        @test r(_w_608) != sum(abs.(_w_608 - _w0_608)) / 2
        @test r(_w0_608) == 0
        @test TurnoverRiskMeasure(; w = _w_608)(_w0_608) == r(_w_608)
        @test expected_risk(r, _w_608, rand(StableRNG(608), 10, 3)) == r(_w_608)
    end
    @testset "Reference weights and previous weights" begin
        @test_throws UndefKeywordError TurnoverRiskMeasure()
        r = TurnoverRiskMeasure(; w = _w0_608)
        rf = TurnoverRiskMeasure(; w = _w0_608, fixed = true)
        @test PO.needs_previous_weights(r)
        @test !PO.needs_previous_weights(rf)
        # A free measure takes the new weights, and a fixed measure keeps its target.
        r1 = PO.factory(r, _w_608)
        @test r1.w == _w_608
        @test r1.settings === r.settings
        @test !r1.fixed
        @test PO.factory(rf, _w_608) === rf
        # A call with no weights keeps the reference weights.
        @test PO.factory(r, nothing) === r
        @test PO.factory(r, nothing, nothing, nothing) === r
        # The full context reads only the weights in the fifth position.
        r5 = PO.factory(r, nothing, nothing, nothing, _w_608)
        @test r5.w == _w_608
        @test r5(_w_608) == 0
        @test PO.factory(rf, nothing, nothing, nothing, _w_608) === rf
    end
    @testset "Validation" begin
        @test_throws IsEmptyError TurnoverRiskMeasure(; w = Float64[])
        @test_throws DomainError TurnoverRiskMeasure(; w = [0.5, NaN])
        @test_throws DomainError TurnoverRiskMeasure(; w = [Inf, 0.5])
        @test_throws DomainError PO.factory(TurnoverRiskMeasure(; w = _w0_608),
                                            [NaN, 0.5, 0.5])
        @test_throws DimensionMismatch TurnoverRiskMeasure(; w = _w0_608)([0.5, 0.5])
    end
    @testset "View" begin
        r = PO.port_opt_view(TurnoverRiskMeasure(; w = _w0_608), [1, 3])
        @test r.w == [0.5, 0.25]
        @test r([0.25, 0.25]) == 0.25
    end
    @testset "The model reports the functor" begin
        rng = StableRNG(608)
        X = randn(rng, 200, 4) * 0.01 .+ 0.001
        pr = prior(EmpiricalPrior(), X)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false))
        # Outside the long-only simplex: the cheapest move sells the two shorts (0.2) and cuts
        # the longs by 0.2, so the least turnover is 0.4.
        w0 = [0.6, 0.6, -0.1, -0.1]
        for obj in (MinimumRisk(), MaximumRatio(; rf = 0.0))
            r = TurnoverRiskMeasure(; w = w0)
            sol = optimise(MeanRisk(; r = r, obj = obj,
                                    opt = JuMPOptimiser(; pe = pr, slv = slv)))
            @test isa(sol.retcode, OptimisationSuccess)
            k = isa(obj, MaximumRatio) ? value(sol.model[:k]) : 1.0
            @test value(sol.model[:risk]) / k ≈ r(sol.w) atol = 1e-6
            @test r(sol.w) ≈ 0.4 atol = 1e-6
            if isa(obj, MaximumRatio)
                @test k > 1
            end
        end
    end
    @testset "A bound holds the turnover" begin
        rng = StableRNG(6080)
        X = randn(rng, 200, 4) * 0.01 .+ [0.004 0.001 0.002 0.003]
        pr = prior(EmpiricalPrior(), X)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false))
        w0 = fill(0.25, 4)
        for obj in (MaximumReturn(), MaximumRatio(; rf = 0.0))
            r = TurnoverRiskMeasure(;
                                    settings = RiskMeasureSettings(; ub = 0.3, rke = false),
                                    w = w0)
            sol = optimise(MeanRisk(; r = [Variance(), r], obj = obj,
                                    opt = JuMPOptimiser(; pe = pr, slv = slv)))
            @test isa(sol.retcode, OptimisationSuccess)
            m = sol.model
            @test haskey(m, :turnover_risk_2)
            @test haskey(m, :turnover_r_2)
            @test haskey(m, :cturnover_r_noc_2)
            @test haskey(m, :turnover_risk_2_ub)
            k = isa(obj, MaximumRatio) ? value(m[:k]) : 1.0
            @test value(m[:turnover_risk_2]) / k <= 0.3 + 1e-6
            @test r(sol.w) <= 0.3 + 1e-6
            @test r(sol.w) <= value(m[:turnover_risk_2]) / k + 1e-6
        end
        # The bound binds: without it the return-seeking portfolio moves further.
        sol = optimise(MeanRisk(; obj = MaximumReturn(),
                                opt = JuMPOptimiser(; pe = pr, slv = slv)))
        @test TurnoverRiskMeasure(; w = w0)(sol.w) > 0.3
    end
end
