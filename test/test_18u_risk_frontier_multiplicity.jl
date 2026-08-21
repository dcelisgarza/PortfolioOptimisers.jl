include(joinpath(@__DIR__, "test18_setup.jl"))

# A risk measure registers a `:risk_frontier` entry only when its `settings.ub` is a
# `Front_NumVec`, so the registry is shorter than the risk measure vector whenever some
# measure carries no frontier bound. These tests fix the two properties that follow: an entry
# names the measure that owns it, and an entry is rebuilt from itself.

@testset "Risk frontier multiplicity: an unbounded measure does not shift the registry" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    rf = ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = Frontier(; N = 3)))
    # Measure 1 carries no bound. The only registry entry belongs to measure 2, so the span
    # must be measure 2's own risk between the two corner portfolios.
    res = optimise(MeanRisk(; r = [Variance(), rf], obj = MaximumReturn(), opt = opt))
    @test all(x -> isa(x, PortfolioOptimisers.OptimisationSuccess), res.retcode)

    entry = only(res.model[:risk_frontier])
    @test entry.second[4] == 2

    r_cvar = factory(ConditionalValueatRisk(), pr)
    res_min = optimise(MeanRisk(; r = [Variance(), ConditionalValueatRisk()], opt = opt))
    res_max = optimise(MeanRisk(; r = [Variance(), ConditionalValueatRisk()],
                                obj = MaximumReturn(), opt = opt))
    span = range(expected_risk(r_cvar, res_min.w, pr), expected_risk(r_cvar, res_max.w, pr);
                 length = 3)
    got = [expected_risk(r_cvar, w, pr) for w in res.w]
    @test issorted(got)
    # Each sweep point sits at its own bound, because the objective maximises return and the
    # conditional value at risk bound is what holds it back.
    @test isapprox(got, collect(span); rtol = 1e-4)
end

@testset "Risk frontier multiplicity: two frontiers bound two different expressions" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    rv = Variance(; settings = RiskMeasureSettings(; ub = Frontier(; N = 2)))
    rc = ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = Frontier(; N = 2)))
    res = optimise(MeanRisk(; r = [rv, rc], obj = MaximumReturn(), opt = opt))
    @test length(res.w) == 4
    @test all(x -> isa(x, PortfolioOptimisers.OptimisationSuccess), res.retcode)

    registry = res.model[:risk_frontier]
    @test length(registry) == 2
    @test [e.second[4] for e in registry] == [1, 2]
    # Each entry keeps its own expression. Rebuilding entry 2 from entry 1 bounds the
    # variance twice and leaves the conditional value at risk unbounded.
    @test registry[1].second[1] !== registry[2].second[1]

    r_var = factory(Variance(), pr)
    r_cvar = factory(ConditionalValueatRisk(), pr)
    res_min = optimise(MeanRisk(; r = [Variance(), ConditionalValueatRisk()], opt = opt))
    res_max = optimise(MeanRisk(; r = [Variance(), ConditionalValueatRisk()],
                                obj = MaximumReturn(), opt = opt))
    var_ub = expected_risk(r_var, res_max.w, pr)
    cvar_ub = expected_risk(r_cvar, res_max.w, pr)
    tol = sqrt(eps())
    # Every sweep point honours both bounds, so neither expression is left unconstrained.
    @test all(w -> expected_risk(r_var, w, pr) <= var_ub + tol, res.w)
    @test all(w -> expected_risk(r_cvar, w, pr) <= cvar_ub + tol, res.w)
    # The conditional value at risk bound binds on at least one point, which it cannot do
    # while it is written against the variance expression.
    cvar_lo = expected_risk(r_cvar, res_min.w, pr)
    @test any(w -> isapprox(expected_risk(r_cvar, w, pr), cvar_lo; rtol = 1e-4), res.w)
end
