include(joinpath(@__DIR__, "test18_setup.jl"))

# The base layer of the risk measure constraints: the scalariser rows, the drawdown rows and the
# number bound. Each testset reads the model back and checks it against the closed form that
# the docstring of the builder states.

@testset "The scalarisers reduce risk_vec to the stated risk" begin
    r = [StandardDeviation(; settings = RiskMeasureSettings(; scale = 2.0)),
         ConditionalValueatRisk(; settings = RiskMeasureSettings(; scale = 0.5))]
    gamma = 50.0
    for (sca, f) in ((SumScalariser(), sum), (MaxScalariser(), maximum),
                     (LogSumExpScalariser(; gamma = gamma), x -> log(sum(exp.(gamma * x))) / gamma))
        opt = JuMPOptimiser(; pe = pr, slv = slv, sca = sca)
        res = optimise(MeanRisk(; r = r, opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        risk_vec = JuMP.value.(res.model[:risk_vec])
        risk = JuMP.value(res.model[:risk])
        # A minimum risk objective pulls `risk` down onto the scalarised value.
        @test isapprox(risk, f(risk_vec); rtol = 1e-6)
        # The rows hold `risk` at or above every entry.
        @test all(risk_vec .<= risk + 1e-8)
    end
end

@testset "A scalariser registers nothing when no measure reaches risk_vec" begin
    rt = optimise(MeanRisk(; r = StandardDeviation(), obj = MaximumReturn(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv)))
    ub = expected_risk(factory(StandardDeviation(), pr), rt.w, pr) / 2
    r = StandardDeviation(; settings = RiskMeasureSettings(; rke = false, ub = ub))
    for sca in (SumScalariser(), MaxScalariser(), LogSumExpScalariser())
        opt = JuMPOptimiser(; pe = pr, slv = slv, sca = sca)
        res = optimise(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test !haskey(res.model, :risk_vec)
        @test !haskey(res.model, :risk)
        # The measure reaches the model through its bound alone.
        @test expected_risk(factory(StandardDeviation(), pr), res.w, pr) <= ub * (1 + 1e-6)
    end
end

@testset "The drawdown rows bound the drawdown from above" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    drawdown(w) = begin
        c = [zero(eltype(w)); cumsum(pr.X * w)]
        [maximum(view(c, 1:t)) - c[t] for t in eachindex(c)]
    end
    # A sum over the entries pulls every entry onto the drawdown.
    res = optimise(MeanRisk(; r = AverageDrawdown(), opt = opt))
    dd = JuMP.value.(res.model[:dd])
    @test abs(dd[1]) < 1e-10
    @test isapprox(dd, drawdown(res.w); atol = 1e-8)
    # A maximum pulls on the largest entry alone. Every entry stays at or above the drawdown.
    res = optimise(MeanRisk(; r = MaximumDrawdown(), opt = opt))
    dd = JuMP.value.(res.model[:dd])
    DD = drawdown(res.w)
    @test all(dd .>= DD .- 1e-8)
    @test isapprox(maximum(dd), maximum(DD); rtol = 1e-6)
end

@testset "A number bound holds on the weights a ratio objective returns" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    rsd = factory(StandardDeviation(), pr)
    sdmin = expected_risk(rsd, optimise(MeanRisk(; r = StandardDeviation(), opt = opt)).w,
                          pr)
    ub = 1.05 * sdmin
    r = StandardDeviation(; settings = RiskMeasureSettings(; ub = ub))
    res = optimise(MeanRisk(; r = r, obj = MaximumRatio(), opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    # The row is `sc * (risk - ub * k) <= 0`, and the head divides the weights by `k`.
    @test !isapprox(JuMP.value(res.model[:k]), 1; rtol = 1e-3)
    @test isapprox(expected_risk(rsd, res.w, pr), ub; rtol = 1e-6)
end
