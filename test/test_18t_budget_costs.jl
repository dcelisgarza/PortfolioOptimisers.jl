using Clarabel

#=
`BudgetCosts` and `BudgetMarketImpact` are exported, are admitted by `JuMPOptimiser`'s own
`bgt` type bound, and had no test of any kind. Three defects survived because of that, and
each of the testsets below pins one:

  1. `set_weight_constraints!` declared `bgt::Option{<:Num_BgtRg}`, which admits a number and
     a `BudgetRange` alone. Neither cost estimator is a `BudgetRange`, so every `optimise`
     that carried one raised a `MethodError` before a single constraint was built.

  2. `set_budget_constraints!(model, ::BudgetMarketImpact, w)` passed the power-cone
     variables `wip`/`win` into `set_cost_budget_constraints!`'s *coefficient* slots, so the
     charged cost was `dot(wip, wp) + dot(win, wn)` -- a quadratic expression that discarded
     `vp` and `vn` outright. `add_market_impact_cost!` then raised
     `MethodError: no method matching add_to_expression!(::AffExpr, ::QuadExpr)`, so the type
     could not build a model either. Equation 9.23 of the source charges `delta' iota`.

  3. The constructor admitted `0 <= beta <= 1`, and the power cone is degenerate at both
     endpoints: a solver answers `SLOW_PROGRESS` for `beta = 0` and for `beta = 1`.

The budget identities below are recomputed from the returned weights alone, so they check
the model rather than restating it.
=#

rng = StableRNG(987654321)
X = randn(rng, 200, 5) * 0.01 .+ 0.001
rd = ReturnsResult(; nx = string.(1:5), X = X)
pr = prior(EmpiricalPrior(), rd)
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
w0 = fill(0.2, 5)

@testset "BudgetCosts: the linear cost budget" begin
    bc = BudgetCosts(; bgt = 1.0, w = w0, vp = 0.01, vn = 0.01, up = 0.5, un = 0.5)
    # (1) The estimator reaches the model builder at all.
    res = optimise(MeanRisk(; obj = MaximumReturn(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bc)))
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    w = res.w
    # Equation 9.15: `c = 1'w + v_p' w_p + v_n' w_n`, with `w_p - w_n = w - w_0`.
    cost = 0.01 * sum(max.(w .- w0, 0)) + 0.01 * sum(max.(w0 .- w, 0))
    @test isapprox(sum(w) + cost, 1.0; atol = 1e-7)
    # The coefficients are read: a larger charge buys a smaller invested budget.
    bc2 = BudgetCosts(; bgt = 1.0, w = w0, vp = 0.5, vn = 0.5, up = 0.5, un = 0.5)
    res2 = optimise(MeanRisk(; obj = MaximumReturn(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bc2)))
    w2 = res2.w
    cost2 = 0.5 * sum(max.(w2 .- w0, 0)) + 0.5 * sum(max.(w0 .- w2, 0))
    @test isapprox(sum(w2) + cost2, 1.0; atol = 1e-7)
    @test cost2 > cost
    # A `BudgetRange` cost budget takes the other overload.
    bc3 = BudgetCosts(; bgt = BudgetRange(; lb = 0.8, ub = 1.0), w = w0, vp = 0.01,
                      vn = 0.01, up = 0.5, un = 0.5)
    res3 = optimise(MeanRisk(; obj = MaximumReturn(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bc3)))
    w3 = res3.w
    cost3 = 0.01 * sum(max.(w3 .- w0, 0)) + 0.01 * sum(max.(w0 .- w3, 0))
    @test 0.8 - 1e-7 <= sum(w3) + cost3 <= 1.0 + 1e-7
end

@testset "BudgetMarketImpact: the power-law budget" begin
    beta = 2 / 3
    bmi = BudgetMarketImpact(; bgt = 1.0, w = w0, vp = 0.01, vn = 0.01, up = 0.5, un = 0.5,
                             beta = beta)
    # (2) The estimator builds a model and the charge is linear in the cone variables.
    res = optimise(MeanRisk(; obj = MaximumReturn(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bmi)))
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    w = res.w
    # Equation 9.23: the cone gives `iota >= w_p^(1/beta)`, so the realised power law is 3/2.
    cost = 0.01 * sum(max.(w .- w0, 0) .^ (1 / beta)) +
           0.01 * sum(max.(w0 .- w, 0) .^ (1 / beta))
    @test isapprox(sum(w) + cost, 1.0; atol = 1e-7)
    # `vp` and `vn` are read, and they were discarded before the fix. A dearer unit charge
    # buys less trading, so the portfolio stays nearer the reference one.
    bmi2 = BudgetMarketImpact(; bgt = 1.0, w = w0, vp = 10.0, vn = 10.0, up = 0.5, un = 0.5,
                              beta = beta)
    res2 = optimise(MeanRisk(; obj = MaximumReturn(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bmi2)))
    w2 = res2.w
    cost2 = 10.0 * sum(max.(w2 .- w0, 0) .^ (1 / beta)) +
            10.0 * sum(max.(w0 .- w2, 0) .^ (1 / beta))
    @test isapprox(sum(w2) + cost2, 1.0; atol = 1e-7)
    @test sum(abs.(w2 .- w0)) < sum(abs.(w .- w0))
    # The charge also reaches the return expression, and `settings.mic` removes it.
    res3 = optimise(MeanRisk(; obj = MaximumReturn(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bmi,
                                                 ret = ArithmeticReturn(;
                                                                        settings = JuMPReturnsSettings(;
                                                                                                       mic = false)))))
    @test isa(res3.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test !isapprox(res3.w, w; atol = 1e-8)
end

@testset "BudgetMarketImpact: beta is the reciprocal exponent" begin
    # (3) Both endpoints of the old `0 <= beta <= 1` guard are degenerate.
    @test_throws DomainError BudgetMarketImpact(; w = w0, beta = 0.0)
    @test_throws DomainError BudgetMarketImpact(; w = w0, beta = 1.0)
    @test_throws DomainError BudgetMarketImpact(; w = w0, beta = -0.1)
    @test_throws DomainError BudgetMarketImpact(; w = w0, beta = 1.1)
    @test BudgetMarketImpact(; w = w0, beta = 0.5).beta == 0.5
    @test BudgetMarketImpact(; w = w0).beta == 2 / 3
    # The budget identity holds at the exponent `1 / beta`, and not at `beta`. This is the
    # decisive check: the cone is `[iota, 1, w_p] in PowerCone(beta)`, which bounds `iota`
    # below by `w_p ^ (1 / beta)`, so a reader who takes `beta` for the power law is wrong.
    bmi = BudgetMarketImpact(; bgt = 1.0, w = w0, vp = 0.1, vn = 0.1, up = 0.5, un = 0.5,
                             beta = 0.5)
    w = optimise(MeanRisk(; obj = MaximumReturn(),
                          opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bmi))).w
    dw = abs.(w .- w0)
    @test isapprox(sum(w) + 0.1 * sum(dw .^ 2), 1.0; atol = 1e-7)
    @test !isapprox(sum(w) + 0.1 * sum(dw .^ 0.5), 1.0; atol = 1e-7)
end
