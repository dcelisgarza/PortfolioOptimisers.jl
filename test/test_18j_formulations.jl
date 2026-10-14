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
                        ret = ArithmeticReturn(;
                                               settings = JuMPReturnsSettings(;
                                                                              lb = Frontier(;
                                                                                            N = 5))))
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
                                               settings = JuMPReturnsSettings(;
                                                                              lb = range(;
                                                                                         start = rt_min,
                                                                                         stop = rt_max,
                                                                                         length = 5))))
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
                        ret = ArithmeticReturn(;
                                               settings = JuMPReturnsSettings(;
                                                                              lb = Frontier(;
                                                                                            N = 5))))
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
                                               settings = JuMPReturnsSettings(;
                                                                              lb = range(;
                                                                                         start = rt_min,
                                                                                         stop = rt_max,
                                                                                         length = 5))))
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
                        ret = ArithmeticReturn(;
                                               settings = JuMPReturnsSettings(;
                                                                              lb = Frontier(;
                                                                                            N = 5))))
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
                                               settings = JuMPReturnsSettings(;
                                                                              lb = range(;
                                                                                         start = rt_min,
                                                                                         stop = rt_max,
                                                                                         length = 5))))
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
@testset "Range measures equal their two tails when the tails differ" begin
    mr_block6_asymmetric()
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
@testset "Weight norm constraints double as a diversification floor" begin
    # l2c/lpc/linfc are upper bounds on the respective norm. Via the reciprocal recipe they
    # act as diversification floors: l2c = 1/sqrt(m) and lpc val = m^(1/p - 1) require at
    # least m order-p effective assets; linfc = 1/m caps the largest weight at a 1/m share.
    # The order-p count is `number_effective_assets` taken to an arbitrary order, so an
    # equal-weight portfolio over m assets reports exactly m at every order.
    opt(; kwargs...) = JuMPOptimiser(; pe = pr, slv = slv, kwargs...)
    ena_p(w, p) = sum(abs.(w) .^ p)^inv(1 - p)
    w_none = optimise(MeanRisk(; obj = MaximumReturn(), opt = opt())).w
    @test number_effective_assets(w_none) < 1.5    # concentrated without a constraint
    for v in (4, 8)
        w = optimise(MeanRisk(; obj = MaximumReturn(), opt = opt(; l2c = inv(sqrt(v))))).w
        @test number_effective_assets(w) >= v - 1e-4
    end
    for v in (4, 8)
        w = optimise(MeanRisk(; obj = MaximumReturn(),
                              opt = opt(;
                                        lpc = LpRegularisation(; p = 3,
                                                               val = v^(1 / 3 - 1))))).w
        @test ena_p(w, 3) >= v - 1e-3
    end
    for v in (4, 8)
        w = optimise(MeanRisk(; obj = MaximumReturn(), opt = opt(; linfc = inv(v)))).w
        @test maximum(abs, w) <= inv(v) + 1e-4
    end
end
@testset "Model scales reach the model under their own names" begin
    # Regression: `set_model_scales!` declared its parameters `(so, sc)` while every head —
    # and every test — calls it `(opt.sc, opt.so)`, so `model[:sc]` held `opt.so` and
    # `model[:so]` held `opt.sc`. Both default to 1, which is why the swap stayed invisible.
    res = optimise(MeanRisk(; r = Variance(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sc = 5, so = 7));
                   save = true)
    @test value(res.model[:sc]) == 5
    @test value(res.model[:so]) == 7
end
@testset "The rotated second-order cone is constraint-scale invariant" begin
    # Regression: `[sc * t; 0.5; sc * x] in RotatedSecondOrderCone()` reads
    # `2 (sc t) (1/2) >= ‖sc x‖²`, i.e. `t >= sc ‖x‖²`, because the middle entry carried no
    # `sc`. The penalty and the reported second-moment risk therefore scaled with a knob
    # that is meant to change conditioning only. The middle entry now carries `sc`, which
    # makes the cone homogeneous: `2 (sc t) (sc/2) >= ‖sc x‖²` is `t >= ‖x‖²`.
    wv = [0.4, -0.3, 0.2, 0.7, -0.1]
    function rsoc_penalty(sc)
        model = JuMP.Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, w[1:length(wv)])
        @constraint(model, w .== wv)
        # The builder divides the square by the homogenisation variable, #1318.
        @expression(model, k, 1)
        PortfolioOptimisers.set_model_scales!(model, sc, 1.0)
        PortfolioOptimisers.set_l2_regularisation!(model,
                                                   L2Regularisation(; val = 1.0,
                                                                    alg = RSOCRiskExpr()))
        pen = model[:op]
        @objective(model, Min, pen)
        optimize!(model)
        return value(pen)
    end
    for sc in (0.5, 1.0, 2.0, 10.0)
        @test isapprox(rsoc_penalty(sc), norm(wv, 2)^2; rtol = 1e-6)
    end
    function rsoc_second_moment(sc)
        model = JuMP.Model(Clarabel.Optimizer)
        set_silent(model)
        PortfolioOptimisers.set_model_scales!(model, sc, 1.0)
        @variable(model, y[1:length(wv)])
        @constraint(model, y .== wv)
        expr, _ = PortfolioOptimisers.set_second_moment_risk!(model, RSOCRiskExpr(), 1, 1.0,
                                                              y, :t_sm_, :c_sm_)
        @objective(model, Min, expr)
        optimize!(model)
        return value(expr)
    end
    for sc in (0.5, 1.0, 2.0, 10.0)
        @test isapprox(rsoc_second_moment(sc), norm(wv, 2)^2; rtol = 1e-6)
    end
    # The whole solve agrees too: an RSOC-encoded penalty must not move with `sc`.
    w_of(sc) = optimise(MeanRisk(; obj = MaximumReturn(),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, sc = sc,
                                                     l2 = L2Regularisation(; val = 0.5,
                                                                           alg = RSOCRiskExpr())))).w
    @test isapprox(w_of(1), w_of(4); rtol = 5e-4)
end
@testset "Under MaximumRatio ScaleCornerLift maximises return per unit of variance" begin
    # Issue #1321, kept as an option by #1676. On the lift with the corner `k`,
    # W ⪰ y y' / k, the semidefinite variance tr(Σ W) has degree one in (y, k). The SOC
    # variance y' Σ y has degree two. So MaximumRatio maximises mu' w / w' Σ w on the first
    # and the Sharpe ratio on the second. By Dinkelbach's theorem the maximiser of
    # mu' w / w' Σ w also maximises mu' w - l * w' Σ w at l equal to the largest ratio, which
    # is a MaximumUtility on the SOC variance.
    ohf = mean(abs.(pr.mu))
    mv(w) = dot(pr.mu, w) / dot(w, pr.sigma, w)
    sharpe(w) = dot(pr.mu, w) / sqrt(dot(w, pr.sigma, w))
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    wsoc = optimise(MeanRisk(; r = Variance(), obj = MaximumRatio(; ohf = ohf), opt = opt)).w
    wstd = optimise(MeanRisk(; r = StandardDeviation(), obj = MaximumRatio(; ohf = ohf),
                             opt = opt)).w
    @test isapprox(wsoc, wstd; atol = 1e-3)
    # Rows that never bind, and a phylogeny that links no pair: each only moves the variance
    # to the semidefinite formulation.
    rc = LinearConstraintEstimator(; val = ["$a <= 1000" for a in rd.nx])
    plg = SemiDefinitePhylogeny(; A = zeros(length(pr.mu), length(pr.mu)), p = 0)
    for (r, kw) in ((Variance(; rc = rc, lift = ScaleCornerLift()), (; sets = sets)),
                    (Variance(; lift = ScaleCornerLift()), (; ple = plg)))
        res_of(ohf) = optimise(MeanRisk(; r = r, obj = MaximumRatio(; ohf = ohf),
                                        opt = JuMPOptimiser(; pe = pr, slv = slv, kw...));
                               save = true)
        res = res_of(ohf)
        @test isa(res.retcode, OptimisationSuccess)
        @test haskey(res.model, :W) && !haskey(res.model, :lift_W)
        @test haskey(res.model, :sr_ret)
        w = res.w
        wu = optimise(MeanRisk(; r = Variance(), obj = MaximumUtility(; l = mv(w)),
                               opt = opt)).w
        @test isapprox(w, wu; atol = 2e-4)
        # The maximiser is free of ohf. The phylogeny's solve at 10 * ohf stops 1.3e-3 away
        # in the weights, so compare the ratio, which moves by 1e-4.
        @test isapprox(mv(w), mv(res_of(10 * ohf).w); rtol = 5e-4)
        @test norm(w - wsoc, Inf) > 0.1
        @test mv(w) > mv(wsoc)
        @test sharpe(w) < sharpe(wsoc)
    end
    # The risk form reaches the same maximiser. A logarithmic return selects it.
    lr(w) = expected_return(LogarithmicReturn(), w, pr)
    res = optimise(MeanRisk(; r = Variance(; rc = rc, lift = ScaleCornerLift()),
                            obj = MaximumRatio(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                                                ret = LogarithmicReturn())); save = true)
    @test haskey(res.model, :sr_risk)
    w = res.w
    wu = optimise(MeanRisk(; r = Variance(),
                           obj = MaximumUtility(; l = lr(w) / dot(w, pr.sigma, w)),
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               ret = LogarithmicReturn()))).w
    @test isapprox(w, wu; atol = 2e-4)
end
@testset "Under MaximumRatio a lifted variance maximises the Sharpe ratio by default" begin
    # Issue #1676. `ConstantCornerLift` states the objective term of a lifted variance on a
    # second lift with a constant corner c, [Wc y; y' c] ⪰ 0, so tr(Σ Wc) ≥ y' Σ y / c has
    # degree two and MaximumRatio maximises the Sharpe ratio, as the SOC variance does. Rows
    # that never bind, a phylogeny that links no pair, and an uncertainty set of radius zero
    # then give the tangency weights of `Variance()`. `ScaleCornerLift` keeps the degree-one
    # ratio, which differs.
    ratio(r; ohf = nothing, ret = ArithmeticReturn(), kw...) = optimise(MeanRisk(; r = r,
                                                                                 obj = MaximumRatio(;
                                                                                                    ohf = ohf),
                                                                                 opt = JuMPOptimiser(;
                                                                                                     pe = pr,
                                                                                                     slv = slv,
                                                                                                     ret = ret,
                                                                                                     kw...));
                                                                        save = true)
    sharpe(w) = dot(pr.mu, w) / sqrt(dot(w, pr.sigma, w))
    wsoc = ratio(Variance()).w
    # Measured 1.0e-3 at most from `wsoc`, on the box, and 6.7e-4 at most under a stated
    # `ohf`. `ScaleCornerLift` stays 0.15 away.
    tol = 2e-3
    N = length(pr.mu)
    cls = SigmaUncertaintySetClass()
    rc = LinearConstraintEstimator(; val = ["$a <= 1000" for a in rd.nx])
    plg = SemiDefinitePhylogeny(; A = zeros(N, N), p = 0)
    box = BoxUncertaintySet(; lb = pr.sigma, ub = pr.sigma)
    ell = EllipsoidalUncertaintySet(; sigma = Matrix(1.0I, N^2, N^2), k = 1e-12,
                                    class = cls)
    nb = NormBallUncertaintySet(; kappa = 0.0, L = zeros(N^2, 0), class = cls)
    cases = [("rc", lift -> Variance(; rc = rc, lift = lift), (; sets = sets)),
             ("phylogeny", lift -> Variance(; lift = lift), (; ple = plg)),
             ("box", lift -> UncertaintySetVariance(; ucs = box, lift = lift), (;)),
             ("ellipsoid", lift -> UncertaintySetVariance(; ucs = ell, lift = lift), (;)),
             ("norm ball", lift -> UncertaintySetVariance(; ucs = nb, lift = lift), (;))]
    for (name, mk, kw) in cases
        res = ratio(mk(ConstantCornerLift()); kw...)
        @test isa(res.retcode, OptimisationSuccess)
        @test haskey(res.model, :lift_W) && haskey(res.model, :sr_ret)
        @test res.model[:lift_corner] == 1
        @test isapprox(res.w, wsoc; atol = tol)
        res1 = ratio(mk(ScaleCornerLift()); kw...)
        @test !haskey(res1.model, :lift_W)
        @test norm(res1.w - wsoc, Inf) > 0.05
        @test sharpe(res1.w) < sharpe(res.w)
    end
    # The phylogeny reads the lift of the objective, so it needs no penalty.
    res = ratio(Variance(); ple = SemiDefinitePhylogeny(; A = zeros(N, N), p = 1.0))
    @test !haskey(res.model, :sdp_plg_p_1)
    @test isapprox(res.w, wsoc; atol = tol)
    # A stated `ohf` far from the size of a return moves `k` far from one. The derived corner
    # follows it, so the solve stays well conditioned.
    for ohf in (1.0, 1e-5)
        res = ratio(Variance(; rc = rc); ohf = ohf, sets = sets)
        @test isapprox(res.model[:lift_corner], ohf / min(1e3, max(1e-3, mean(abs, pr.mu))))
        @test isapprox(res.w, wsoc; atol = tol)
    end
    # A stated corner gives the same weights, and a corner that the measures disagree on is
    # refused, because they share one lift.
    @test isapprox(ratio(Variance(; rc = rc, corner = 3.0); sets = sets).w, wsoc;
                   atol = tol)
    @test_throws ArgumentError ratio([Variance(; rc = rc, corner = 3.0),
                                      UncertaintySetVariance(; ucs = box)]; sets = sets)
    @test_throws DomainError Variance(; corner = 0)
    @test_throws DomainError UncertaintySetVariance(; corner = Inf)
    # The risk form reaches the Sharpe ratio too. A logarithmic return selects it.
    res = ratio(Variance(; rc = rc); ret = LogarithmicReturn(), sets = sets)
    @test haskey(res.model, :sr_risk) && haskey(res.model, :lift_W)
    @test isapprox(res.w, ratio(Variance(); ret = LogarithmicReturn()).w; atol = tol)
    # Every other objective reads one lift.
    res = optimise(MeanRisk(; r = UncertaintySetVariance(; ucs = box),
                            opt = JuMPOptimiser(; pe = pr, slv = slv)); save = true)
    @test !haskey(res.model, :lift_W)
end
