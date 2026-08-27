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
                                                    re = StepwiseRegression(; crit = :bic)),
                                   sets = sets, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(;
                                             pe = FactorPrior(;
                                                              re = StepwiseRegression(;
                                                                                      crit = :bic)),
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
                                                    re = StepwiseRegression(; crit = :bic)),
                                   sets = sets, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; alg = H0_EntropyPooling(),
                                             pe = FactorPrior(;
                                                              re = StepwiseRegression(;
                                                                                      crit = :bic)),
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
    pr = prior(EntropyPoolingPrior(; sets = sets,
                                   var_views = ValueatRiskView(; views = var_views)), rd)
    @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) == ValueatRisk(;)(rd.X[:, 1])
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = ValueatRiskView(;
                                                                         views = var_views)),
                         rd).w, rtol = 1e-6)

    var_views = LinearConstraintEstimator(; val = "AAPL >= 1.15*prior(AAPL)")
    pr = prior(EntropyPoolingPrior(; sets = sets,
                                   var_views = ValueatRiskView(; views = var_views)), rd)
    @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) >= 1.15 * ValueatRisk(;)(rd.X[:, 1])
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = ValueatRiskView(;
                                                                         views = var_views)),
                         rd).w, rtol = 1e-6)

    var_views = LinearConstraintEstimator(; val = "AAPL == 0.12865204867438676")
    pr = prior(EntropyPoolingPrior(; sets = sets,
                                   var_views = ValueatRiskView(; views = var_views)), rd)
    @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) == WorstRealisation()(rd.X[:, 1])
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = ValueatRiskView(;
                                                                         views = var_views)),
                         rd).w, rtol = 1e-6)

    #=
    Issue #537. `OptimEntropyPooling`'s `g!` wrote the UNSCALED gradient into `G` and returned
    `sc1 * G`. `Optim` reads the mutated `G` and discards the return value, so `sc1` scaled
    the objective and not the gradient. The two disagreed, and `sc1` reached nothing: every
    value of it returned the same posterior to the last bit. With the gradient scaled to
    match, `sc1` is the knob its description promises, and raising it tightens both the view
    and the sum to one.
    =#
    epc_sc1 = Dict{Symbol, Any}()
    PortfolioOptimisers.ep_mu_views!(LinearConstraintEstimator(; val = "AAPL == 0.002"),
                                     epc_sc1, pr0, sets)
    w_sc1 = fill(inv(size(rd.X, 1)), size(rd.X, 1))
    for alg in (ExpEntropyPooling(), LogEntropyPooling())
        lo = PortfolioOptimisers.entropy_pooling(w_sc1, epc_sc1,
                                                 OptimEntropyPooling(; sc1 = 1, alg = alg))
        hi = PortfolioOptimisers.entropy_pooling(w_sc1, epc_sc1,
                                                 OptimEntropyPooling(; sc1 = 1e6,
                                                                     alg = alg))
        @test abs(dot(hi, rd.X[:, 1]) - 0.002) < abs(dot(lo, rd.X[:, 1]) - 0.002)
        @test abs(sum(hi) - 1) < abs(sum(lo) - 1)
    end

    #=
    Issue #537. A `>=` value at risk view reaches `ep_var_views!` normalised to `A * p <= B`,
    so `B` is the negated target and never positive. The sign rule read `B[i] >= 0` as a `<=`
    view, which only a ZERO target can satisfy, and flipped that row: the view `AAPL >= 0`
    wrote `sum(p) <= alpha` over the losing observations rather than `>= alpha`. That row is
    already met at the prior, so the posterior stayed on the prior and the view was silently
    dropped.

    The row itself is tested rather than the quantile it states. The view drives the tail
    mass to exactly `alpha`, and a weighted quantile read at exactly `alpha` sits on a knife
    edge between two neighbouring observations, which the two optimisers land on either side
    of. `alpha = 0.5` is used because the prior tail mass of AAPL falls short of it there,
    which makes the view binding.
    =#
    var_views = LinearConstraintEstimator(; val = "AAPL >= 0")
    idx = rd.X[:, 1] .<= 0
    @test sum(fill(inv(size(rd.X, 1)), size(rd.X, 1))[idx]) < 0.5
    pr = prior(EntropyPoolingPrior(; sets = sets,
                                   var_views = ValueatRiskView(; alpha = 0.5,
                                                               views = var_views)), rd)
    @test isapprox(sum(pr.w[idx]), 0.5, rtol = 1e-5)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = ValueatRiskView(; alpha = 0.5,
                                                                         views = var_views)),
                         rd).w, rtol = 1e-6)

    var_views = LinearConstraintEstimator(; val = ["AAPL == 0.028", "XOM >= 0.027"])
    pr = prior(EntropyPoolingPrior(; sets = sets,
                                   var_views = ValueatRiskView(; alpha = 0.07,
                                                               views = var_views)), rd)
    @test isapprox(ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, 1]), 0.028, rtol = 7e-3)
    @test ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, end]) >= 0.027
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = ValueatRiskView(; alpha = 0.07,
                                                                         views = var_views)),
                         rd).w, rtol = 1e-4)

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

    # A view over a pair of groups emits one constraint row per spanned pair, so a
    # `prior(gA, gB)` reference must give each row that pair's own prior value. Issue #403.
    gsets = UniverseSets(;
                         dict = Dict("nx" => rd.nx, "gA" => rd.nx[1:2], "gB" => rd.nx[3:4]))
    rho0 = StatsBase.cov2cor(pr0.sigma)
    @test PortfolioOptimisers.get_pr_value(pr0, [1, 2], [3, 4], Val(:rho)) ==
          [rho0[1, 3], rho0[2, 4]]
    @test PortfolioOptimisers.get_pr_value(pr0, [1, 2], [3, 4], Val(:cov)) ==
          [pr0.sigma[1, 3], pr0.sigma[2, 4]]
    # The `:cov` tag answers with a covariance, never a correlation.
    @test PortfolioOptimisers.get_pr_value(pr0, [1, 2], [3, 4], Val(:cov)) !=
          PortfolioOptimisers.get_pr_value(pr0, [1, 2], [3, 4], Val(:rho))

    rho_views = LinearConstraintEstimator(; val = "(gA, gB) == prior(gA, gB)*1.1")
    pr = prior(EntropyPoolingPrior(; sets = gsets, rho_views = rho_views), rd)
    rho1 = StatsBase.cov2cor(pr.sigma)
    @test isapprox(rho1[1, 3], rho0[1, 3] * 1.1, rtol = 5e-5)
    @test isapprox(rho1[2, 4], rho0[2, 4] * 1.1, rtol = 5e-5)
    # The two pairs take different targets, so a single aggregate cannot serve both.
    @test !isapprox(rho1[1, 3], rho1[2, 4]; rtol = 1e-2)

    cov_views = LinearConstraintEstimator(; val = "(gA, gB) == prior(gA, gB)*1.1")
    pr = prior(EntropyPoolingPrior(; sets = gsets, cov_views = cov_views), rd)
    @test isapprox(pr.sigma[1, 3], pr0.sigma[1, 3] * 1.1, rtol = 5e-3)
    @test isapprox(pr.sigma[2, 4], pr0.sigma[2, 4] * 1.1, rtol = 5e-3)

    # Every entry of a vector right-hand side must stay inside [-1, 1].
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = gsets,
                                                         rho_views = LinearConstraintEstimator(;
                                                                                               val = "(gA, gB) == prior(gA, gB)*3")),
                                     rd)
    # A vector right-hand side needs one value per spanned pair.
    @test_throws DimensionMismatch RhoParsingResult(["(A, B)"], [1.0], "==", [0.1, 0.2],
                                                    "1.0*(A, B) == [0.1, 0.2]", [(1, 2)])
    @test_throws DimensionMismatch RhoParsingResult(["([A, B], [C, D])"], [1.0], "==",
                                                    [0.1, 0.2, 0.3],
                                                    "1.0*([A, B], [C, D]) == [0.1, 0.2, 0.3]",
                                                    [([1, 2], [3, 4])])

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
    pr = prior(MeucciEntropyPoolingPrior(; sets = sets,
                                         cvar_views = ConditionalValueatRiskView(;
                                                                                 views = cvar_views)),
               rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.07, rtol = 1e-6)
    @test isapprox(pr.w,
                   prior(MeucciEntropyPoolingPrior(; sets = sets, opt = jopt,
                                                   cvar_views = ConditionalValueatRiskView(;
                                                                                           views = cvar_views)),
                         rd).w, rtol = 5e-5)

    cvar_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*1.37")
    pr = prior(MeucciEntropyPoolingPrior(; sets = sets,
                                         cvar_views = ConditionalValueatRiskView(;
                                                                                 views = cvar_views)),
               rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]),
                   ConditionalValueatRisk(;)(rd.X[:, 1]) * 1.37, rtol = 1e-6)
    @test isapprox(pr.w,
                   prior(MeucciEntropyPoolingPrior(; sets = sets, opt = jopt,
                                                   cvar_views = ConditionalValueatRiskView(;
                                                                                           views = cvar_views)),
                         rd).w, rtol = 5e-5)

    cvar_views = LinearConstraintEstimator(; val = ["AAPL == 0.053", "XOM==0.045"])
    pr = prior(HighOrderPriorEstimator(;
                                       pe = MeucciEntropyPoolingPrior(; sets = sets,
                                                                      alg = H2_EntropyPooling(),
                                                                      cvar_views = ConditionalValueatRiskView(;
                                                                                                              views = cvar_views))),
               rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.053, rtol = 5e-5)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, end]), 0.045, rtol = 1e-4)
    @test isapprox(pr.w,
                   prior(HighOrderPriorEstimator(;
                                                 pe = MeucciEntropyPoolingPrior(;
                                                                                sets = sets,
                                                                                alg = H1_EntropyPooling(),
                                                                                cvar_views = ConditionalValueatRiskView(;
                                                                                                                        views = cvar_views))),
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

# A covariance or correlation view carries its operator through a sign flip, and its
# coefficient through both sides of the row. Until #379 the flip reached only the
# `mu_i * mu_j` term, so a `>=` view targeted the NEGATED value and was vacuous against a
# prior that already sat above it; and the coefficient reached only the left-hand side, so
# a view whose coefficient was not one landed off target. Both are read off the posterior
# matrix, never off a residual the solver reports.
@testset "Covariance and correlation view operators and coefficients" begin
    pr0 = prior(EmpiricalPrior(), rd)
    rho0 = StatsBase.cov2cor(pr0.sigma)
    ep_rho(v) = StatsBase.cov2cor(prior(EntropyPoolingPrior(; sets = sets,
                                                            rho_views = LinearConstraintEstimator(;
                                                                                                  val = v)),
                                        rd).sigma)
    ep_cov(v) = prior(EntropyPoolingPrior(; sets = sets,
                                          cov_views = LinearConstraintEstimator(; val = v)),
                      rd).sigma

    # The prior correlation of AAPL and XOM is 0.3364, so a `>=` view above it has to move
    # the posterior and a `<=` view below it has to move it the other way. A correlation is a
    # ratio of two covariances, so the `T / (T - 1)` factor of the recomputed `pr.sigma`
    # cancels and the posterior lands on the target itself.
    @test rho0[1, end] < 0.45
    @test isapprox(ep_rho("(AAPL, XOM) >= 0.45")[1, end], 0.45, rtol = 5e-5)
    @test isapprox(ep_rho("(AAPL, XOM) == 0.45")[1, end], 0.45, rtol = 5e-5)
    @test rho0[1, end] > 0.2
    @test isapprox(ep_rho("(AAPL, XOM) <= 0.2")[1, end], 0.2, rtol = 5e-5)

    # The coefficient scales both sides, so these two views are the same view.
    @test isapprox(ep_rho("2*(AAPL, XOM) >= 0.9")[1, end],
                   ep_rho("(AAPL, XOM) >= 0.45")[1, end], rtol = 5e-5)

    # The covariance twin of each check. A covariance is not a ratio, so the constraint pins
    # the UNCORRECTED weighted second moment and `pr.sigma` reports the corrected one: the
    # posterior lands exactly `T / (T - 1)` above the target.
    T = size(pr0.X, 1)
    bessel = T / (T - 1)
    tgt = pr0.sigma[13, 14] * 1.5
    @test pr0.sigma[13, 14] < tgt
    @test isapprox(ep_cov("(MSFT, PEP) >= $(tgt)")[13, 14], tgt * bessel, rtol = 1e-5)
    @test isapprox(ep_cov("2*(MSFT, PEP) >= $(2 * tgt)")[13, 14],
                   ep_cov("(MSFT, PEP) >= $(tgt)")[13, 14], rtol = 5e-5)

    # A view naming two groups sets EVERY pair the groups span, on the same scale as the
    # single-asset view. Until #379 the group branch scaled by the Frobenius norm of the
    # VARIANCE products, which is neither a per-pair scale nor a standard deviation, so a
    # correlation target of 0.3 landed three orders of magnitude below it.
    gsets = UniverseSets(; xkey = "nx",
                         dict = Dict("nx" => sets.dict[sets.xkey],
                                     "gA" => sets.dict[sets.xkey][1:2],
                                     "gX" => fill(sets.dict[sets.xkey][end], 2)))
    rg = StatsBase.cov2cor(prior(EntropyPoolingPrior(; sets = gsets,
                                                     rho_views = LinearConstraintEstimator(;
                                                                                           val = "(gA, gX) == 0.3")),
                                 rd).sigma)
    @test isapprox(rg[1, end], 0.3, rtol = 5e-4)
    @test isapprox(rg[2, end], 0.3, rtol = 5e-4)
end

# `sc2` bounds the slack of the relaxed fixed equalities, so a negative value inverts the
# box. `JuMPEntropyPooling` has always rejected it; `OptimEntropyPooling` documented the
# same rule and never checked it (#379).
@testset "OptimEntropyPooling validates sc2" begin
    @test_throws DomainError OptimEntropyPooling(; sc2 = -1)
    @test_throws DomainError OptimEntropyPooling(; sc1 = -1)
    @test OptimEntropyPooling(; sc2 = 0).sc2 == 0
end
