include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Formulations" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    r = factory(Variance(), pr)
    res_min = optimise(MeanRisk(; r = r, opt = opt))
    res_max = optimise(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
    rk_min = expected_risk(r, res_min.w, pr)
    rk_max = expected_risk(r, res_max.w, pr)
    rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
    rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
    res1 = optimise(MeanRisk(;
                             r = Variance(;
                                          settings = RiskMeasureSettings(;
                                                                         ub = Frontier(;
                                                                                       N = 5))),
                             obj = MaximumReturn(), opt = opt))
    res2 = optimise(MeanRisk(;
                             r = Variance(; alg = QuadRiskExpr(),
                                          settings = RiskMeasureSettings(;
                                                                         ub = Frontier(;
                                                                                       N = 5))),
                             obj = MaximumReturn(), opt = opt))
    res = isapprox(hcat(res1.w...), hcat(res2.w...); rtol = 5e-4)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res1.w...), hcat(res2.w...))
    end
    @test res
    rks = expected_risk.(r, res1.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res1.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    res3 = optimise(MeanRisk(;
                             r = Variance(;
                                          settings = RiskMeasureSettings(;
                                                                         ub = range(;
                                                                                    start = rk_min,
                                                                                    stop = rk_max,
                                                                                    length = 5))),
                             obj = MaximumReturn(), opt = opt))
    res4 = optimise(MeanRisk(;
                             r = Variance(; alg = QuadRiskExpr(),
                                          settings = RiskMeasureSettings(;
                                                                         ub = range(;
                                                                                    start = rk_min,
                                                                                    stop = rk_max,
                                                                                    length = 5))),
                             obj = MaximumReturn(), opt = opt))
    res = isapprox(hcat(res3.w...), hcat(res4.w...); rtol = 1e-6)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res3.w...), hcat(res4.w...))
    end
    @test res
    rks = expected_risk.(r, res3.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res3.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        ret = ArithmeticReturn(; lb = Frontier(; N = 5)))
    res5 = optimise(MeanRisk(; r = Variance(;), opt = opt))
    res6 = optimise(MeanRisk(; r = Variance(; alg = QuadRiskExpr()), opt = opt))
    res = isapprox(hcat(res5.w...), hcat(res6.w...); rtol = 1e-3)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res5.w...), hcat(res6.w...))
    end
    @test res
    rks = expected_risk.(r, res5.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res5.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        ret = ArithmeticReturn(;
                                               lb = range(; start = rt_min, stop = rt_max,
                                                          length = 5)))
    res7 = optimise(MeanRisk(; r = Variance(;), opt = opt))
    res8 = optimise(MeanRisk(; r = Variance(; alg = QuadRiskExpr()), opt = opt))
    res = isapprox(hcat(res7.w...), hcat(res8.w...); rtol = 1e-3)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res7.w...), hcat(res8.w...))
    end
    @test res
    rks = expected_risk.(r, res7.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res7.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv)
    r = factory(LowOrderMoment(; alg = SecondMoment(; alg2 = QuadRiskExpr())), pr)
    res_min = optimise(MeanRisk(; r = r, opt = opt))
    res_max = optimise(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
    rk_min = expected_risk(r, res_min.w, pr)
    rk_max = expected_risk(r, res_max.w, pr)
    rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
    rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
    res1 = optimise(MeanRisk(;
                             r = LowOrderMoment(;
                                                settings = RiskMeasureSettings(;
                                                                               ub = Frontier(;
                                                                                             N = 5)),
                                                alg = SecondMoment(; alg2 = QuadRiskExpr())),
                             obj = MaximumReturn(), opt = opt))
    res2 = optimise(MeanRisk(;
                             r = LowOrderMoment(;
                                                settings = RiskMeasureSettings(;
                                                                               ub = Frontier(;
                                                                                             N = 5)),
                                                alg = SecondMoment(; alg2 = RSOCRiskExpr())),
                             obj = MaximumReturn(), opt = opt))
    res = isapprox(hcat(res1.w...), hcat(res2.w...); rtol = 5e-3)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res1.w...), hcat(res2.w...))
    end
    @test res
    rks = expected_risk.(r, res1.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res1.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    res3 = optimise(MeanRisk(;
                             r = LowOrderMoment(;
                                                settings = RiskMeasureSettings(;
                                                                               ub = range(;
                                                                                          start = rk_min,
                                                                                          stop = rk_max,
                                                                                          length = 5)),
                                                alg = SecondMoment(; alg2 = QuadRiskExpr())),
                             obj = MaximumReturn(), opt = opt))
    res4 = optimise(MeanRisk(;
                             r = LowOrderMoment(;
                                                settings = RiskMeasureSettings(;
                                                                               ub = range(;
                                                                                          start = rk_min,
                                                                                          stop = rk_max,
                                                                                          length = 5)),
                                                alg = SecondMoment(; alg2 = RSOCRiskExpr())),
                             obj = MaximumReturn(), opt = opt))
    res = isapprox(hcat(res3.w...), hcat(res4.w...); rtol = 5e-6)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res3.w...), hcat(res4.w...))
    end
    @test res
    rks = expected_risk.(r, res3.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res3.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        ret = ArithmeticReturn(; lb = Frontier(; N = 5)))
    res5 = optimise(MeanRisk(;
                             r = LowOrderMoment(;
                                                alg = SecondMoment(; alg2 = QuadRiskExpr())),
                             opt = opt))
    res6 = optimise(MeanRisk(;
                             r = LowOrderMoment(;
                                                alg = SecondMoment(; alg2 = RSOCRiskExpr())),
                             opt = opt))
    res = isapprox(hcat(res5.w...), hcat(res6.w...); rtol = 5e-3)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res5.w...), hcat(res6.w...))
    end
    @test res
    rks = expected_risk.(r, res5.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res5.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        ret = ArithmeticReturn(;
                                               lb = range(; start = rt_min, stop = rt_max,
                                                          length = 5)))
    res7 = optimise(MeanRisk(;
                             r = LowOrderMoment(; settings = RiskMeasureSettings(;),
                                                alg = SecondMoment(; alg2 = QuadRiskExpr())),
                             opt = opt))
    res8 = optimise(MeanRisk(;
                             r = LowOrderMoment(; settings = RiskMeasureSettings(;),
                                                alg = SecondMoment(; alg2 = RSOCRiskExpr())),
                             opt = opt))
    res = isapprox(hcat(res7.w...), hcat(res8.w...); rtol = 5e-3)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res7.w...), hcat(res8.w...))
    end
    @test res
    rks = expected_risk.(r, res7.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res7.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv)
    res9 = optimise(MeanRisk(;
                             r = ValueatRisk(;
                                             alg = DistributionValueatRisk(;
                                                                           dist = Laplace())),
                             opt = opt))
    res10 = optimise(MeanRisk(;
                              r = ValueatRisk(;
                                              alg = DistributionValueatRisk(;
                                                                            dist = TDist(5))),
                              opt = opt))
    @test isapprox(res9.w, res10.w; rtol = 5e-2)

    res11 = optimise(MeanRisk(;
                              r = ValueatRiskRange(;
                                                   alg = DistributionValueatRisk(;
                                                                                 dist = Laplace())),
                              opt = opt))
    res12 = optimise(MeanRisk(;
                              r = ValueatRiskRange(;
                                                   alg = DistributionValueatRisk(;
                                                                                 dist = TDist(5))),
                              opt = opt))
    @test isapprox(res11.w, res12.w; rtol = 5e-4)

    opt = JuMPOptimiser(; pe = pr, slv = slv[7:end])
    r = factory(NegativeSkewness(; alg = SquaredSOCRiskExpr()), pr)
    res_min = optimise(MeanRisk(; r = r, opt = opt))
    res_max = optimise(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
    rk_min = expected_risk(r, res_min.w, pr)
    rk_max = expected_risk(r, res_max.w, pr)
    rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
    rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
    res1 = optimise(MeanRisk(;
                             r = NegativeSkewness(; alg = SquaredSOCRiskExpr(),
                                                  settings = RiskMeasureSettings(;
                                                                                 ub = Frontier(;
                                                                                               N = 5))),
                             obj = MaximumReturn(), opt = opt))
    res2 = optimise(MeanRisk(;
                             r = NegativeSkewness(; alg = QuadRiskExpr(),
                                                  settings = RiskMeasureSettings(;
                                                                                 ub = Frontier(;
                                                                                               N = 5))),
                             obj = MaximumReturn(), opt = opt))
    res = isapprox(hcat(res1.w...), hcat(res2.w...); rtol = 5e-4)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res1.w...), hcat(res2.w...))
    end
    @test res
    rks = expected_risk.(r, res1.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res1.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv)
    res3 = optimise(MeanRisk(;
                             r = NegativeSkewness(; alg = SquaredSOCRiskExpr(),
                                                  settings = RiskMeasureSettings(;
                                                                                 ub = range(;
                                                                                            start = rk_min,
                                                                                            stop = rk_max,
                                                                                            length = 5))),
                             obj = MaximumReturn(), opt = opt))
    res4 = optimise(MeanRisk(;
                             r = NegativeSkewness(; alg = QuadRiskExpr(),
                                                  settings = RiskMeasureSettings(;
                                                                                 ub = range(;
                                                                                            start = rk_min,
                                                                                            stop = rk_max,
                                                                                            length = 5))),
                             obj = MaximumReturn(), opt = opt))
    res = isapprox(hcat(res3.w...), hcat(res4.w...); rtol = 1e-6)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res3.w...), hcat(res4.w...))
    end
    @test res
    rks = expected_risk.(r, res3.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res3.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv[7:end],
                        ret = ArithmeticReturn(; lb = Frontier(; N = 5)))
    res5 = optimise(MeanRisk(; r = NegativeSkewness(; alg = SquaredSOCRiskExpr()),
                             opt = opt))
    res6 = optimise(MeanRisk(; r = NegativeSkewness(; alg = QuadRiskExpr()), opt = opt))
    res = isapprox(hcat(res5.w...), hcat(res6.w...); rtol = 5e-4)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res5.w...), hcat(res6.w...))
    end
    @test res
    rks = expected_risk.(r, res5.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res5.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv[7:end],
                        ret = ArithmeticReturn(;
                                               lb = range(; start = rt_min, stop = rt_max,
                                                          length = 5)))
    res7 = optimise(MeanRisk(; r = NegativeSkewness(; alg = SquaredSOCRiskExpr()),
                             opt = opt))
    res8 = optimise(MeanRisk(; r = NegativeSkewness(; alg = QuadRiskExpr()), opt = opt))
    res = isapprox(hcat(res7.w...), hcat(res8.w...); rtol = 5e-4)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res7.w...), hcat(res8.w...))
    end
    @test res
    rks = expected_risk.(r, res7.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res7.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr2, slv = slv)
    mr = MeanRisk(; r = BrownianDistanceVariance(; alg2 = IneqBrownianDistanceVariance()),
                  opt = opt)
    res9 = optimise(mr)

    mr = MeanRisk(; r = BrownianDistanceVariance(; alg1 = RSOCRiskExpr()), opt = opt)
    res10 = optimise(mr)

    mr = MeanRisk(;
                  r = BrownianDistanceVariance(; alg1 = RSOCRiskExpr(),
                                               alg2 = IneqBrownianDistanceVariance()),
                  opt = opt)
    res11 = optimise(mr)
    @test isapprox(res9.w,
                   CSV.read(joinpath(@__DIR__, "./assets/MeanRiskBDV.csv.gz"), DataFrame)[!,
                                                                                          1],
                   rtol = 5e-4)
    @test isapprox(res9.w, res10.w; rtol = 5e-4)
    @test isapprox(res9.w, res11.w; rtol = 1e-3)

    opt = JuMPOptimiser(; pe = pr, slv = slv)
    r = factory(LowOrderMoment(; mu = VecScalar(; v = pr.mu, s = 4.2 / 252 / 100)), pr)
    res_min = optimise(MeanRisk(; r = r, opt = opt))
    res_max = optimise(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
    rk_min = expected_risk(r, res_min.w, pr)
    rk_max = expected_risk(r, res_max.w, pr)
    rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
    rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
    res12 = optimise(MeanRisk(;
                              r = LowOrderMoment(;
                                                 mu = VecScalar(; v = pr.mu,
                                                                s = 4.2 / 252 / 100),
                                                 settings = RiskMeasureSettings(;
                                                                                ub = Frontier(;
                                                                                              N = 5))),
                              obj = MaximumReturn(), opt = opt))
    res13 = optimise(MeanRisk(;
                              r = LowOrderMoment(; mu = pr.mu .- 4.2 / 252 / 100,
                                                 settings = RiskMeasureSettings(;
                                                                                ub = Frontier(;
                                                                                              N = 5))),
                              obj = MaximumReturn(), opt = opt))

    res = !isapprox(hcat(res12.w...), hcat(res13.w...); rtol = 1e-2)
    if !res
        println("Frontier formulation failed")
        find_tol(hcat(res12.w...), hcat(res13.w...))
    end
    @test res
    rks = expected_risk.(r, res12.w, pr)
    @test issorted(rks)
    @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
    rts = expected_return.(ArithmeticReturn(), res12.w, pr)
    @test issorted(rts)
    @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))
end
@testset "Generic X at Range" begin
    mr_block6()
end
@testset "L2 regularisation formulations reach affine objectives" begin
    # Regression: a quadratic penalty (QuadRiskExpr, SquaredSOCRiskExpr) added to an affine
    # objective (MaximumReturn) needs `obj_expr` promoted to a QuadExpr. The promotion
    # allocates a new expression, so `add_penalty_to_objective!` must hand it back to the
    # caller — otherwise the penalty is built, registered, and silently dropped.
    opt_l2(alg) = JuMPOptimiser(; pe = pr, slv = slv,
                                l2 = L2Regularisation(; val = 0.5, alg = alg))
    w_of(alg) = optimise(MeanRisk(; obj = MaximumReturn(), opt = opt_l2(alg))).w
    w_none = optimise(MeanRisk(; obj = MaximumReturn(),
                               opt = JuMPOptimiser(; pe = pr, slv = slv))).w
    w_soc = w_of(SOCRiskExpr())
    w_ssoc = w_of(SquaredSOCRiskExpr())
    w_quad = w_of(QuadRiskExpr())
    w_rsoc = w_of(RSOCRiskExpr())
    # Every formulation must actually penalise: an unpenalised MaximumReturn concentrates.
    @test norm(w_none, 2) > 0.9
    for w in (w_soc, w_ssoc, w_quad, w_rsoc)
        @test norm(w, 2) < norm(w_none, 2)
    end
    # SquaredSOCRiskExpr, QuadRiskExpr and RSOCRiskExpr all penalise ‖w‖₂², so they agree.
    @test isapprox(w_ssoc, w_quad; rtol = 5e-4)
    @test isapprox(w_quad, w_rsoc; rtol = 5e-4)
    # SOCRiskExpr penalises ‖w‖₂ instead, so it does not.
    @test !isapprox(w_soc, w_quad; rtol = 5e-4)
end
@testset "Weight norm constraints bound the effective number of assets" begin
    # wn2/wnp/wninf are lower bounds on diversification, not upper bounds on the norm.
    opt(; kwargs...) = JuMPOptimiser(; pe = pr, slv = slv, kwargs...)
    ena_p(w, p) = inv(sum(abs.(w) .^ p))
    w_none = optimise(MeanRisk(; obj = MaximumReturn(), opt = opt())).w
    @test number_effective_assets(w_none) < 1.5    # concentrated without a constraint
    for v in (4, 8)
        w = optimise(MeanRisk(; obj = MaximumReturn(), opt = opt(; wn2 = v))).w
        @test number_effective_assets(w) >= v - 1e-4
    end
    for v in (4, 8)
        w = optimise(MeanRisk(; obj = MaximumReturn(),
                              opt = opt(; wnp = LpRegularisation(; p = 3, val = v)))).w
        @test ena_p(w, 3) >= v - 1e-3
    end
    for v in (4, 8)
        w = optimise(MeanRisk(; obj = MaximumReturn(), opt = opt(; wninf = v))).w
        @test maximum(abs, w) <= inv(v) + 1e-4
    end
end
