include(joinpath(@__DIR__, "test12_setup.jl"))

@testset "ExpEntropyPooling" begin
    pr0 = prior(EmpiricalPrior(), rd)
    jopt = JuMPEntropyPooling(; slv = slv)

    mu_views = LinearConstraintEstimator(; val = "AAPL == 0.002")
    pr = prior(EntropyPoolingPrior(; w = w, sets = sets, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 1e-7)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             mu_views = mu_views), rd).w, rtol = 5e-6)

    pr = prior(EntropyPoolingPrior(;
                                   pe = FactorPrior(;
                                                    re = StepwiseRegression(; crit = BIC())),
                                   sets = sets, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(;
                                             pe = FactorPrior(;
                                                              re = StepwiseRegression(;
                                                                                      crit = BIC())),
                                             sets = sets, opt = jopt, mu_views = mu_views),
                         rd).w, rtol = 5e-6)

    pr = prior(EntropyPoolingPrior(;
                                   pe = FactorPrior(;
                                                    re = DimensionReductionRegression(;
                                                                                      retgt = GeneralisedLinearModel())),
                                   sets = sets, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(;
                                             pe = FactorPrior(;
                                                              re = DimensionReductionRegression(;
                                                                                                retgt = GeneralisedLinearModel())),
                                             sets = sets, opt = jopt, mu_views = mu_views),
                         rd).w, rtol = 5e-6)

    pr = prior(EntropyPoolingPrior(; w = StatsBase.pweights(range(iT, iT; length = T)),
                                   alg = H0_EntropyPooling(),
                                   pe = FactorPrior(;
                                                    re = StepwiseRegression(; crit = BIC())),
                                   sets = sets, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; alg = H0_EntropyPooling(),
                                             pe = FactorPrior(;
                                                              re = StepwiseRegression(;
                                                                                      crit = BIC())),
                                             sets = sets, opt = jopt, mu_views = mu_views),
                         rd).w, rtol = 5e-6)

    pr = prior(EntropyPoolingPrior(; alg = H0_EntropyPooling(),
                                   pe = FactorPrior(;
                                                    re = DimensionReductionRegression(;
                                                                                      retgt = GeneralisedLinearModel())),
                                   sets = sets, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; alg = H0_EntropyPooling(),
                                             pe = FactorPrior(;
                                                              re = DimensionReductionRegression(;
                                                                                                retgt = GeneralisedLinearModel())),
                                             sets = sets, opt = jopt, mu_views = mu_views),
                         rd).w, rtol = 5e-6)

    mu_views = LinearConstraintEstimator(; val = "AAPL >= 0.0025")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views), rd)
    @test pr.mu[1] >= 0.0025
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             mu_views = mu_views), rd).w, rtol = 5e-6)

    mu_views = LinearConstraintEstimator(; val = "AAPL <= 0.001")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views), rd)
    @test pr.mu[1] <= 0.001
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             mu_views = mu_views), rd).w, rtol = 5e-6)

    var_views = LinearConstraintEstimator(; val = "AAPL == 0.03264496113282452")
    pr = prior(EntropyPoolingPrior(; sets = sets, var_views = var_views), rd)
    @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) == ValueatRisk(;)(rd.X[:, 1])
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = var_views), rd).w, rtol = 1e-6)

    var_views = LinearConstraintEstimator(; val = "AAPL >= 1.15*prior(AAPL)")
    pr = prior(EntropyPoolingPrior(; sets = sets, var_views = var_views), rd)
    @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) >= 1.15 * ValueatRisk(;)(rd.X[:, 1])
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = var_views), rd).w, rtol = 1e-6)

    var_views = LinearConstraintEstimator(; val = "AAPL == 0.12865204867438676")
    pr = prior(EntropyPoolingPrior(; sets = sets, var_views = var_views), rd)
    @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) == WorstRealisation()(rd.X[:, 1])
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, var_views = var_views,
                                             opt = jopt), rd).w, rtol = 1e-6)

    var_views = LinearConstraintEstimator(; val = ["AAPL == 0.028", "XOM >= 0.027"])
    pr = prior(EntropyPoolingPrior(; sets = sets, var_alpha = 0.07, var_views = var_views),
               rd)
    @test isapprox(ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, 1]), 0.028, rtol = 7e-3)
    @test ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, end]) >= 0.027
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt, var_alpha = 0.07,
                                             var_views = var_views), rd).w, rtol = 1e-4)

    sigma_views = LinearConstraintEstimator(; val = "AAPL == 0.0007")
    pr = prior(EntropyPoolingPrior(; sets = sets, sigma_views = sigma_views), rd)
    r = LowOrderMoment(; w = pr.w, mu = pr.mu[1],
                       alg = SecondMoment(; ve = SimpleVariance(; w = pr.w)))
    @test isapprox(r([1], reshape(pr.X[:, 1], :, 1)), 0.0007, rtol = 1e-3)
    @test isapprox(pr.sigma[1, 1], r([1], reshape(pr.X[:, 1], :, 1)))
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, sigma_views = sigma_views,
                                             opt = jopt), rd).w, rtol = 1e-2)

    mu_views = LinearConstraintEstimator(; val = "AAPL == 1.7*prior(AAPL)")
    sigma_views = LinearConstraintEstimator(; val = "AAPL == 0.0008")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                   sigma_views = sigma_views), rd)
    @test isapprox(pr.mu[1], pr0.mu[1] * 1.7, rtol = 5e-6)
    @test isapprox(pr.sigma[1, 1], 0.0008, rtol = 1e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt, mu_views = mu_views,
                                             sigma_views = sigma_views), rd).w, rtol = 5e-4)

    sk_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*2")
    pr = prior(EntropyPoolingPrior(; sets = sets, sk_views = sk_views), rd)
    @test isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                         reshape(pr.X[:, 1],
                                                                                 :, 1)),
                   2 * Skewness()([1], reshape(pr0.X[:, 1], :, 1)), rtol = 2e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             sk_views = sk_views), rd).w, rtol = 5e-3)

    mu_views = LinearConstraintEstimator(; val = "AAPL==1.5*prior(AAPL)")
    sigma_views = LinearConstraintEstimator(; val = "AAPL==1.3prior(AAPL)")
    sk_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*2")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                   sigma_views = sigma_views, sk_views = sk_views), rd)
    @test isapprox(pr.mu[1], 1.5 * pr0.mu[1], rtol = ifelse(Sys.islinux(), 1e-3, 1e-6))
    @test isapprox(pr.sigma[1, 1], 1.3 * pr0.sigma[1, 1], rtol = 5e-3)
    @test isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                         reshape(pr.X[:, 1],
                                                                                 :, 1)),
                   2 * Skewness()([1], reshape(pr0.X[:, 1], :, 1)), rtol = 5e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                             sigma_views = sigma_views,
                                             sk_views = sk_views), rd).w)

    kt_views = LinearConstraintEstimator(; val = "AAPL == 7.5")
    pr = prior(EntropyPoolingPrior(; sets = sets, kt_views = kt_views), rd)
    @test isapprox(HighOrderMoment(; w = pr.w,
                                   alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                                     ve = SimpleVariance(;
                                                                                         w = pr.w)))([1],
                                                                                                     reshape(pr.X[:,
                                                                                                                  1],
                                                                                                             :,
                                                                                                             1)),
                   7.5, rtol = 5e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             kt_views = kt_views), rd).w, rtol = 1e-2)

    mu_views = LinearConstraintEstimator(; val = "AAPL<=1.5*prior(AAPL)")
    sigma_views = LinearConstraintEstimator(; val = "AAPL==0.7prior(AAPL)")
    kt_views = LinearConstraintEstimator(; val = "AAPL >= prior(AAPL)*0.87")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                   sigma_views = sigma_views, kt_views = kt_views), rd)
    @test pr.mu[1] <= 1.5 * pr0.mu[1]
    @test isapprox(pr.sigma[1, 1], 0.7 * pr0.sigma[1, 1], rtol = 1e-3)
    @test isapprox(HighOrderMoment(; w = pr.w,
                                   alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                                     ve = SimpleVariance(;
                                                                                         w = pr.w)))([1],
                                                                                                     reshape(pr.X[:,
                                                                                                                  1],
                                                                                                             :,
                                                                                                             1)),
                   HighOrderMoment(;
                                   alg = StandardisedHighOrderMoment(;
                                                                     alg = FourthMoment()))([1],
                                                                                            reshape(pr.X[:,
                                                                                                         1],
                                                                                                    :,
                                                                                                    1)) *
                   0.87, rtol = 5e-6)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt, mu_views = mu_views,
                                             sigma_views = sigma_views,
                                             kt_views = kt_views), rd).w, rtol = 5e-3)

    cov_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == prior(AAPL, XOM)*1.1")
    pr = prior(EntropyPoolingPrior(; sets = sets, cov_views = cov_views), rd)
    @test isapprox(pr.sigma[1, end], pr0.sigma[1, end] * 1.1, rtol = 1e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             cov_views = cov_views), rd).w, rtol = 5e-6)

    rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == 0.35")
    pr = prior(EntropyPoolingPrior(; sets = sets, rho_views = rho_views), rd)
    @test isapprox(StatsBase.cov2cor(pr.sigma)[1, end], 0.35, rtol = 5e-6)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             rho_views = rho_views), rd).w, rtol = 1e-2)

    rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == prior(AAPL,XOM)*0.94")
    pr = prior(EntropyPoolingPrior(; sets = sets, rho_views = rho_views), rd)
    @test isapprox(StatsBase.cov2cor(pr.sigma)[1, end],
                   StatsBase.cov2cor(pr0.sigma)[1, end] * 0.94, rtol = 5e-6)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             rho_views = rho_views), rd).w, rtol = 5e-2)

    pr = prior(HighOrderPriorEstimator(;
                                       pe = EntropyPoolingPrior(; alg = H2_EntropyPooling(),
                                                                sets = sets,
                                                                mu_views = LinearConstraintEstimator(;
                                                                                                     val = ["AAPL<=0.92*prior(AAPL)",
                                                                                                            "XOM >= 0.83*prior(XOM)"]),
                                                                sigma_views = LinearConstraintEstimator(;
                                                                                                        val = ["AAPL==1.2prior(AAPL)",
                                                                                                               "WMT==1.4prior(WMT)"]),
                                                                rho_views = LinearConstraintEstimator(;
                                                                                                      val = "(AAPL, XOM) == 0.35"))),
               rd)
    @test pr.mu[1] <= 0.92 * pr0.mu[1] + sqrt(eps()) * length(pr0.mu) * 2
    @test pr.mu[end] >= 0.83 * pr0.mu[end] - sqrt(eps()) * length(pr0.mu) * 22
    @test isapprox(pr.sigma[1, 1], 1.2 * pr0.sigma[1, 1], rtol = 1e-2)
    @test isapprox(pr.sigma[19, 19], 1.4 * pr0.sigma[19, 19], rtol = 5e-3)
    @test isapprox(StatsBase.cov2cor(pr.sigma)[1, end], 0.35, rtol = 1e-3)

    cvar_views = LinearConstraintEstimator(; val = "AAPL == 0.07")
    pr = prior(EntropyPoolingPrior(; sets = sets, cvar_views = cvar_views), rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.07, rtol = 1e-6)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             cvar_views = cvar_views), rd).w, rtol = 5e-5)

    cvar_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*1.37")
    pr = prior(EntropyPoolingPrior(; sets = sets, cvar_views = cvar_views), rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]),
                   ConditionalValueatRisk(;)(rd.X[:, 1]) * 1.37, rtol = 1e-6)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             cvar_views = cvar_views), rd).w, rtol = 5e-5)

    cvar_views = LinearConstraintEstimator(; val = ["AAPL == 0.053", "XOM==0.045"])
    pr = prior(HighOrderPriorEstimator(;
                                       pe = EntropyPoolingPrior(; sets = sets,
                                                                alg = H2_EntropyPooling(),
                                                                cvar_views = cvar_views)),
               rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.053, rtol = 5e-5)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, end]), 0.045, rtol = 1e-4)
    @test isapprox(pr.w,
                   prior(HighOrderPriorEstimator(;
                                                 pe = EntropyPoolingPrior(; sets = sets,
                                                                          alg = H1_EntropyPooling(),
                                                                          cvar_views = cvar_views)),
                         rd).w, rtol = 5e-3)

    mu_views = LinearConstraintEstimator(;
                                         val = ["AAPL<=0.75*prior(AAPL)",
                                                "XOM >= 0.4*prior(XOM)"])
    sigma_views = LinearConstraintEstimator(;
                                            val = ["AAPL==0.2prior(AAPL)",
                                                   "WMT==1.4prior(WMT)"])
    cov_views = LinearConstraintEstimator(; val = "(MSFT, PEP) <= prior(MSFT, PEP)*0.8")
    rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == 0.35")
    kt_views = LinearConstraintEstimator(; val = "AAPL >= prior(AAPL)*0.3")
    sk_views = LinearConstraintEstimator(; val = "WMT == prior(WMT)*1.4")
    pr = prior(HighOrderPriorEstimator(;
                                       pe = EntropyPoolingPrior(; alg = H0_EntropyPooling(),
                                                                sets = sets,
                                                                mu_views = mu_views,
                                                                sigma_views = sigma_views,
                                                                cov_views = cov_views,
                                                                rho_views = rho_views,
                                                                kt_views = kt_views,
                                                                sk_views = sk_views)), rd)
    @test isapprox(pr.mu[1], 0.75 * pr0.mu[1], rtol = 1e-5)
    @test pr.mu[end] >= 0.4 * pr0.mu[end]
    @test isapprox(pr.sigma[1, 1], 0.2 * pr0.sigma[1, 1], rtol = 1e-2)
    @test isapprox(pr.sigma[19, 19], 1.4 * pr0.sigma[19, 19], rtol = 5e-3)
    @test !isapprox(StatsBase.cov2cor(pr.sigma)[1, end], 0.35; rtol = 5e-4)
    @test pr0.sigma[13, 14] * 0.8 >= pr.sigma[13, 14]
    @test HighOrderMoment(; w = pr.w,
                          alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                            ve = SimpleVariance(; w = pr.w)))([1],
                                                                                              reshape(pr.X[:,
                                                                                                           1],
                                                                                                      :,
                                                                                                      1)) >=
          HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = FourthMoment()))([1],
                                                                                       reshape(pr.X[:,
                                                                                                    1],
                                                                                               :,
                                                                                               1)) *
          0.3
    @test !isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                          reshape(pr.X[:,
                                                                                       end - 1],
                                                                                  :, 1)),
                    1.4 * Skewness()([1], reshape(pr0.X[:, end - 1], :, 1)); rtol = 5e-3)
end
