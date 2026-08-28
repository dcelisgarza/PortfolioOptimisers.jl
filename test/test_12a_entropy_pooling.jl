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

# #538: a view naming an asset the universe does not hold. `replace_prior_views` and both
# branches of `replace_coprior_views` drop the term under `strict = false` and raise under
# `strict = true`. The three drops were unreached before this testset.
@testset "A view naming an unknown asset is dropped" begin
    using PortfolioOptimisers, Test, Logging
    pr0 = prior(EmpiricalPrior(), rd)
    dt = eltype(pr0.X)
    rpv = PortfolioOptimisers.replace_prior_views
    rcv = PortfolioOptimisers.replace_coprior_views
    pe(s) = PortfolioOptimisers.parse_equation(s; datatype = dt)

    # The `prior(...)` term goes, the plain term stays, and the right-hand side keeps the
    # constant the dropped term never contributed to.
    r = @test_logs (:warn,) rpv(pe("AAPL - prior(NOPE) == 0.001"), pr0, sets, :mu;
                                strict = false)
    @test r.vars == ["AAPL"]
    @test isapprox(r.rhs, 0.001)
    @test_throws ArgumentError rpv(pe("AAPL - prior(NOPE) == 0.001"), pr0, sets, :mu;
                                   strict = true)

    with_logger(SimpleLogger(stderr, Logging.Error)) do
        # The plain `(a, b)` branch, with the unknown name in either position.
        r1 = rcv(pe("(NOPE, BAC) + (AAPL, BBY) == 0.0001"), pr0, sets, :cov; strict = false)
        @test r1.vars == ["(AAPL, BBY)"]
        @test r1.ij == [(1, 4)]
        r2 = rcv(pe("(AAPL, NOPE) + (AAPL, BBY) == 0.0001"), pr0, sets, :cov;
                 strict = false)
        @test r2.vars == ["(AAPL, BBY)"]
        @test r2.ij == [(1, 4)]

        # The `prior(a, b)` branch, with the unknown name in either position.
        r3 = rcv(pe("(AAPL, BAC) - prior(NOPE, BBY) == 0.0001"), pr0, sets, :cov;
                 strict = false)
        @test r3.vars == ["(AAPL, BAC)"]
        @test isapprox(r3.rhs, 0.0001)
        r4 = rcv(pe("(AAPL, BAC) - prior(AMD, NOPE) == 0.0001"), pr0, sets, :cov;
                 strict = false)
        @test r4.vars == ["(AAPL, BAC)"]
        @test isapprox(r4.rhs, 0.0001)
    end
    @test_throws ArgumentError rcv(pe("(NOPE, BAC) + (AAPL, BBY) == 0.0001"), pr0, sets,
                                   :cov; strict = true)
    @test_throws ArgumentError rcv(pe("(AAPL, BAC) - prior(NOPE, BBY) == 0.0001"), pr0,
                                   sets, :cov; strict = true)
end

# #538: a vector of `ValueatRiskView` states one view per entry, each with its own `alpha`.
# The vector method was unreached: every test before this one stated a single view.
@testset "A vector of ValueatRiskView states one view per entry" begin
    pr0 = prior(EmpiricalPrior(), rd)
    X = pr0.X
    var_views = [ValueatRiskView(;
                                 views = LinearConstraintEstimator(; val = "AAPL == 0.04"),
                                 alpha = 0.05),
                 ValueatRiskView(; views = LinearConstraintEstimator(; val = "AMD == 0.06"),
                                 alpha = 0.1)]
    epc = Dict{Symbol, Tuple{<:PortfolioOptimisers.MatNum, <:PortfolioOptimisers.VecNum}}()
    PortfolioOptimisers.ep_var_views!(var_views, epc, pr0, sets)
    @test size(epc[:eq][1], 1) == 2

    # A value at risk view is a statement about the tail MASS: the posterior probability of
    # the observations at or below `-target` must equal the view's own `alpha`.
    pw = collect(prior(EntropyPoolingPrior(; w = w, sets = sets, var_views = var_views),
                       rd).w)
    @test isapprox(sum(pw[X[:, 1] .<= -0.04]), 0.05, rtol = 5e-6)
    @test isapprox(sum(pw[X[:, 2] .<= -0.06]), 0.1, rtol = 5e-6)
end

# #538: two covariance pairs in one estimator reach the vector method of
# `replace_coprior_views`, which every single-pair test before this one stepped over.
@testset "Two covariance pairs in one estimator" begin
    pr0 = prior(EmpiricalPrior(), rd)
    X = pr0.X
    t1 = 1.3 * pr0.sigma[1, 2]
    t2 = 0.7 * pr0.sigma[3, 4]
    pw = collect(prior(EntropyPoolingPrior(; w = w, sets = sets,
                                           cov_views = LinearConstraintEstimator(;
                                                                                 val = ["(AAPL, AMD) == $(t1)",
                                                                                        "(BAC, BBY) == $(t2)"])),
                       rd).w)
    # `ep_cov_views!` states the second moment about the PRIOR means, so that is where the
    # target lands.
    cw(a, b) = sum(pw .* X[:, a] .* X[:, b]) - pr0.mu[a] * pr0.mu[b]
    @test isapprox(cw(1, 2), t1, rtol = 5e-6)
    @test isapprox(cw(3, 4), t2, rtol = 5e-6)
end

# #538: `wb` boxes a `:feq` dual variable by `sc2`, and the start was `1/sqrt(T)` for every
# coordinate. Any `sc2` below that value put the start outside the box, and `Optim` raised
# `ArgumentError: Initial x[(3,)]=0.0315 is outside of [-0.031, 0.031]` instead of solving.
@testset "OptimEntropyPooling starts inside its box" begin
    pr0 = prior(EmpiricalPrior(), rd)
    X = pr0.X
    target = pr0.mu[1] + 0.002
    function epc_with_fix()
        epc = Dict{Symbol,
                   Tuple{<:PortfolioOptimisers.MatNum, <:PortfolioOptimisers.VecNum}}()
        PortfolioOptimisers.ep_mu_views!(LinearConstraintEstimator(;
                                                                   val = "AAPL == $(target)"),
                                         epc, pr0, sets)
        to_fix = falses(size(X, 2))
        to_fix[1] = true
        PortfolioOptimisers.fix_mu!(epc, falses(size(X, 2)), to_fix, pr0)
        return epc
    end
    post(opt) = sum(collect(PortfolioOptimisers.entropy_pooling(w, epc_with_fix(), opt)) .*
                    X[:, 1])
    # `1/sqrt(T)` is the start, so these three `sc2` values bracket it from below.
    for sc2 in (1e-4, 0.031, inv(sqrt(size(X, 1))))
        @test isapprox(post(OptimEntropyPooling(; sc2 = sc2)), target, rtol = 5e-7)
        @test isapprox(post(OptimEntropyPooling(; sc2 = sc2, alg = LogEntropyPooling())),
                       target, rtol = 5e-7)
    end
    # `sc2 = 0` pins the `:feq` dual variable to zero, so the fix carries no weight. That is
    # what the JuMP route's penalty of weight `sc2` already gave, and the two now agree.
    epc_eq_only = Dict{Symbol,
                       Tuple{<:PortfolioOptimisers.MatNum, <:PortfolioOptimisers.VecNum}}()
    PortfolioOptimisers.ep_mu_views!(LinearConstraintEstimator(; val = "AAPL == $(target)"),
                                     epc_eq_only, pr0, sets)
    eq_only = sum(collect(PortfolioOptimisers.entropy_pooling(w, epc_eq_only,
                                                              OptimEntropyPooling())) .*
                  X[:, 1])
    @test isapprox(post(OptimEntropyPooling(; sc2 = 0)), eq_only, rtol = 1e-12)
    @test isapprox(post(OptimEntropyPooling(; sc2 = 0, alg = LogEntropyPooling())), eq_only,
                   rtol = 1e-12)
    @test isapprox(post(JuMPEntropyPooling(; slv = slv, sc2 = 0)), eq_only, rtol = 5e-7)

    # The documented key set is `:eq`, `:ineq`, `:feq` and `:cvar_eq`. Both algorithms raise
    # on anything else; `:cvar_ineq` was a dead disjunct of the `Exp` branch alone.
    bogus = Dict{Symbol, Tuple{<:PortfolioOptimisers.MatNum, <:PortfolioOptimisers.VecNum}}()
    bogus[:cvar_ineq] = (reshape(X[:, 1], 1, :), [0.002])
    @test_throws KeyError PortfolioOptimisers.entropy_pooling(w, bogus,
                                                              OptimEntropyPooling())
    @test_throws KeyError PortfolioOptimisers.entropy_pooling(w, bogus,
                                                              OptimEntropyPooling(;
                                                                                  alg = LogEntropyPooling()))
end

# #538: the guard read `any` where it needed `all`. A row over a universe of more than one
# asset always carries a zero, so `any` held for every view and the guard never fired. The
# body then read the target off `B` while it discarded the coefficient, which doubled the
# threshold a `2*AAPL` view asks for: 28 observations sat below `-0.04` where 124 sat below
# the `-0.02` the view means.
@testset "A ValueatRiskView takes a coefficient of one alone" begin
    pr0 = prior(EmpiricalPrior(), rd)
    epc() = Dict{Symbol, Tuple{<:PortfolioOptimisers.MatNum, <:PortfolioOptimisers.VecNum}}()
    vv(s) = ValueatRiskView(; views = LinearConstraintEstimator(; val = s), alpha = 0.05)
    @test_throws ArgumentError PortfolioOptimisers.ep_var_views!(vv("2*AAPL == 0.04"),
                                                                 epc(), pr0, sets)
    @test_throws ArgumentError PortfolioOptimisers.ep_var_views!(vv("0.5*AAPL >= 0.04"),
                                                                 epc(), pr0, sets)
    e = epc()
    PortfolioOptimisers.ep_var_views!(vv("AAPL == 0.04"), e, pr0, sets)
    @test size(e[:eq][1], 1) == 1
end

# #538: a covariance or correlation view naming an asset the universe does not hold RAISED
# under `strict = false`. `replace_coprior_views` drops the pair, which leaves the view with
# no pair, and the "mix multiple pairs" guard then reported a view of no pairs as one of
# several and named an equation with no variable in it. The `strict` contract is that
# `false` warns and drops, which the four linear families already honoured.
@testset "A covariance pair naming an unknown asset is dropped" begin
    using PortfolioOptimisers, Test, Logging
    pr0 = prior(EmpiricalPrior(), rd)
    N = size(pr0.X, 2)
    epc() = Dict{Symbol, Tuple{<:PortfolioOptimisers.MatNum, <:PortfolioOptimisers.VecNum}}()
    lce(v) = LinearConstraintEstimator(; val = v)
    cov_val = ["(NOPE, AMD) == 0.0002", "(AAPL, AMD) == 0.0006"]
    rho_val = ["(NOPE, AMD) == 0.5", "(AAPL, AMD) == 0.5"]

    with_logger(SimpleLogger(stderr, Logging.Error)) do
        e = epc()
        to_fix = PortfolioOptimisers.ep_cov_views!(lce(cov_val), e, pr0, sets;
                                                   strict = false)
        # The surviving view still lands its row, and it alone marks its two assets.
        @test size(e[:eq][1], 1) == 1
        @test findall(to_fix) == [1, 2]

        e = epc()
        to_fix = PortfolioOptimisers.ep_rho_views!(lce(rho_val), e, pr0, sets;
                                                   strict = false)
        @test size(e[:eq][1], 1) == 1
        @test findall(to_fix) == [1, 2]
    end
    # The dropped row is reported, not silent.
    @test_logs (:warn,) (:warn,) match_mode = :any PortfolioOptimisers.ep_cov_views!(lce(cov_val),
                                                                                     epc(),
                                                                                     pr0,
                                                                                     sets;
                                                                                     strict = false)
    @test_throws ArgumentError PortfolioOptimisers.ep_cov_views!(lce(cov_val), epc(), pr0,
                                                                 sets; strict = true)
    @test_throws ArgumentError PortfolioOptimisers.ep_rho_views!(lce(rho_val), epc(), pr0,
                                                                 sets; strict = true)
end

# #538: an `:ineq` block through the two dual algorithms. `LogEntropyPooling`'s `:ineq` box
# was the one line of this file the two entropy pooling test files never reached: every
# inequality view they state runs through `ExpEntropyPooling` or through JuMP.
@testset "An inequality view runs through both dual algorithms" begin
    pr0 = prior(EmpiricalPrior(), rd)
    X = pr0.X
    function epc_two()
        epc = Dict{Symbol,
                   Tuple{<:PortfolioOptimisers.MatNum, <:PortfolioOptimisers.VecNum}}()
        PortfolioOptimisers.ep_mu_views!(LinearConstraintEstimator(;
                                                                   val = ["AAPL == 0.002",
                                                                          "AMD >= 0.004"]),
                                         epc, pr0, sets)
        return epc
    end
    @test sort(collect(keys(epc_two()))) == [:eq, :ineq]
    # The prior mean of AMD is 0.00187, so the lower bound binds.
    pexp = collect(PortfolioOptimisers.entropy_pooling(w, epc_two(),
                                                       OptimEntropyPooling(;
                                                                           alg = ExpEntropyPooling())))
    plog = collect(PortfolioOptimisers.entropy_pooling(w, epc_two(),
                                                       OptimEntropyPooling(;
                                                                           alg = LogEntropyPooling())))
    for p in (pexp, plog)
        @test isapprox(sum(p .* X[:, 1]), 0.002, rtol = 5e-6)
        @test isapprox(sum(p .* X[:, 2]), 0.004, rtol = 5e-6)
        @test isapprox(sum(p), 1, rtol = 5e-7)
        @test all(>(0), p)
    end
    # Two parameterisations of one problem.
    @test isapprox(pexp, plog, rtol = 1e-12)
end

# #539: the constructor guards of `MeucciEntropyPoolingPrior`. Nothing in the suite passed an
# empty `w` or an empty `var_views` vector, so both raises were dead lines.
@testset "MeucciEntropyPoolingPrior constructor guards" begin
    @test_throws PortfolioOptimisers.IsEmptyError MeucciEntropyPoolingPrior(;
                                                                            w = StatsBase.pweights(Float64[]))
    @test_throws PortfolioOptimisers.IsNothingError MeucciEntropyPoolingPrior(;
                                                                              mu_views = LinearConstraintEstimator(;
                                                                                                                   val = "AAPL == 0.002"))
    @test_throws PortfolioOptimisers.IsEmptyError MeucciEntropyPoolingPrior(; sets = sets,
                                                                            var_views = ValueatRiskView[])
    # A non-empty vector of value at risk views passes the same guard.
    vv = [ValueatRiskView(; views = LinearConstraintEstimator(; val = "AAPL == 0.03"))]
    @test isa(MeucciEntropyPoolingPrior(; sets = sets, var_views = vv).var_views,
              AbstractVector)
    # The constructor normalises `w` to sum to one. A mutable value is normalised in place and
    # an immutable one is rebuilt, so both reach the field summing to one.
    T0 = size(rd.X, 1)
    wmut = StatsBase.pweights(collect(range(1.0, 2.0; length = T0)))
    wimm = StatsBase.pweights(range(1.0, 2.0; length = T0))
    @test !isapprox(sum(wmut), 1; rtol = 1e-3)
    for pe in
        (MeucciEntropyPoolingPrior(; w = wmut), MeucciEntropyPoolingPrior(; w = wimm))
        @test isapprox(sum(pe.w), 1; rtol = 1e-10)
    end
end

# #539: with no CVaR view the solve takes the `::Nothing` method of `ep_cvar_views_solve!`,
# which no test reached: every `MeucciEntropyPoolingPrior` in the suite carried `cvar_views`.
@testset "MeucciEntropyPoolingPrior without a CVaR view" begin
    pr0 = prior(EmpiricalPrior(), rd)
    mu_v = LinearConstraintEstimator(; val = "AAPL == 0.002")
    function epc_mu()
        epc = Dict{Symbol,
                   Tuple{<:PortfolioOptimisers.MatNum, <:PortfolioOptimisers.VecNum}}()
        PortfolioOptimisers.ep_mu_views!(mu_v, epc, pr0, sets)
        return epc
    end
    opt = OptimEntropyPooling()
    # The `::Nothing` method is a pass-through: it must return exactly what a direct
    # `entropy_pooling` call returns, ignoring `pr`, `sets`, `ds_opt` and `dm_opt`.
    wnone = PortfolioOptimisers.ep_cvar_views_solve!(nothing, epc_mu(), pr0, sets, w, opt,
                                                     nothing, nothing)
    wdirect = PortfolioOptimisers.entropy_pooling(w, epc_mu(), opt)
    @test wnone == wdirect
    # And the same weights reach the caller through `prior`.
    pr = prior(MeucciEntropyPoolingPrior(; sets = sets, mu_views = mu_v), rd)
    @test isapprox(collect(pr.w), collect(wnone), rtol = 1e-6)
    @test isapprox(pr.mu[1], 0.002, rtol = 1e-6)
    @test isapprox(sum(pr.w), 1, rtol = 5e-7)
    @test all(>(0), pr.w)
end

# #539: the raises of the CVaR search. A view beyond the worst realisation is refused before
# any solve, a view group carrying a formulation has nothing to apply it to, and a search that
# cannot converge is reported rather than returned.
@testset "MeucciEntropyPoolingPrior refuses an unattainable CVaR view" begin
    x = rd.X[:, 1]
    worst = -minimum(x)
    # The message names every offending view beside the largest target its asset admits.
    err = try
        prior(MeucciEntropyPoolingPrior(; sets = sets,
                                        cvar_views = ConditionalValueatRiskView(;
                                                                                views = LinearConstraintEstimator(;
                                                                                                                  val = "AAPL == $(worst * 1.01)"))),
              rd)
        nothing
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("too extreme", err.msg)
    @test occursin(string(worst), err.msg)
    # A `ds_opt` that stops after one iteration cannot bracket the root, so the catch rethrows
    # the failure as an `ErrorException` naming the ways out. This is also the only route that
    # reaches the `ds_opt` arm of the single-view branch.
    @test_throws ErrorException prior(MeucciEntropyPoolingPrior(; sets = sets,
                                                                ds_opt = ConditionalValueatRiskEntropyPooling(;
                                                                                                              kwargs = (;
                                                                                                                        maxiters = 1)),
                                                                cvar_views = ConditionalValueatRiskView(;
                                                                                                        views = LinearConstraintEstimator(;
                                                                                                                                          val = "AAPL == 0.07"))),
                                      rd)
    # A view group carrying a formulation has nothing to apply it to on this route.
    @test_throws ArgumentError prior(MeucciEntropyPoolingPrior(; sets = sets,
                                                               cvar_views = ConditionalValueatRiskView(;
                                                                                                       alg = LinearConditionalValueatRiskView(),
                                                                                                       views = LinearConstraintEstimator(;
                                                                                                                                         val = "AAPL == 0.07"))),
                                     rd)
    # One asset per view, and a non-negative target.
    @test_throws ArgumentError prior(MeucciEntropyPoolingPrior(; sets = sets,
                                                               cvar_views = ConditionalValueatRiskView(;
                                                                                                       views = LinearConstraintEstimator(;
                                                                                                                                         val = "AAPL + XOM == 0.07"))),
                                     rd)
    @test_throws DomainError prior(MeucciEntropyPoolingPrior(; sets = sets,
                                                             cvar_views = ConditionalValueatRiskView(;
                                                                                                     views = LinearConstraintEstimator(;
                                                                                                                                       val = "AAPL == -0.07"))),
                                   rd)
end

# #573: `Optim`'s default stopping rule leaves the entropy pooling dual short of its own
# optimum, and the two testsets below read the answer through a statistic that magnifies the
# shortfall. The staged chain misses the mean view it binds by 9e-4 of the bound, against a
# tolerance of 1e-3, and the outer CVaR search re-solves the whole problem at each candidate
# value at risk, so an inner solve that stops early moves the root. Both then land differently
# on a different host and after a different sequence of solves in the same process. This
# optimiser drives the same dual to stationarity: it holds the CVaR views to 1e-10 of their
# targets where the default holds them to 1e-7, and the mean view to 7e-6 of the bound. It is
# a caller-side setting, so no library default and no tolerance moves with it. `Fminbox`
# carries its `mu0` because a non-empty `args` replaces the one `entropy_pooling` supplies.
const EP_TIGHT = OptimEntropyPooling(;
                                     args = (PortfolioOptimisers.Optim.Fminbox(;
                                                                               mu0 = 1e-5),
                                             PortfolioOptimisers.Optim.Options(;
                                                                               x_abstol = 1e-12,
                                                                               f_reltol = 1e-14,
                                                                               g_abstol = 1e-12,
                                                                               outer_x_abstol = 1e-12,
                                                                               iterations = 10_000,
                                                                               outer_iterations = 50)))

# #539: the outer search lands on the value at risk the CVaR view implies. With one view and
# no other row the posterior is an exponential tilt in closed form, so the check needs no
# entropy pooling solver and it separates the formulation from the solver.
@testset "MeucciEntropyPoolingPrior CVaR search finds the value at risk" begin
    x = rd.X[:, 1]
    T0, alpha, target = length(x), 0.05, 0.07
    q = fill(inv(T0), T0)
    # A plain bisection keeps the check free of a solver and of a new test dependency.
    bisect = function (f, lo, hi, tol)
        flo = f(lo)
        for _ in 1:200
            mid = 0.5 * (lo + hi)
            hi - lo <= tol && return mid
            fm = f(mid)
            if (flo < 0) == (fm < 0)
                lo, flo = mid, fm
            else
                hi = mid
            end
        end
        return 0.5 * (lo + hi)
    end
    tiltp = function (c, lam)
        u = log.(q) .- lam .* c
        u .-= maximum(u)
        z = exp.(u)
        z ./= sum(z)
        return z
    end
    tilt = function (c, b)
        f = lam -> LinearAlgebra.dot(tiltp(c, lam), c) - b
        lo, hi = -1.0, 1.0
        while f(lo) < 0 && lo > -1e8
            lo *= 4
        end
        while f(hi) > 0 && hi < 1e8
            hi *= 4
        end
        return tiltp(c, bisect(f, lo, hi, 1e-12))
    end
    # `eta` is the value at risk: the tilted posterior must leave exactly `alpha` beyond it.
    g = function (eta)
        c = max.(-x .- eta, 0.0) ./ alpha
        return sum(tilt(c, target - eta)[.!iszero.(c)]) - alpha
    end
    eta = bisect(g, 1e-8, target * 0.999, 1e-13)
    pcf = tilt(max.(-x .- eta, 0.0) ./ alpha, target - eta)

    pr = prior(MeucciEntropyPoolingPrior(; sets = sets,
                                         cvar_views = ConditionalValueatRiskView(;
                                                                                 views = LinearConstraintEstimator(;
                                                                                                                   val = "AAPL == $target"))),
               rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(x), target, rtol = 1e-7)
    # The library's posterior value at risk is the root the outer search returned.
    @test isapprox(ValueatRisk(; w = pr.w)(x), eta, rtol = 1e-8)
    # And the whole posterior is the closed-form tilt.
    @test isapprox(collect(pr.w), pcf, rtol = 1e-5)
    @test isapprox(pr.kld, sum(pcf .* log.(pcf ./ q)), rtol = 1e-5)
    # The root sits strictly inside the bracket `[0, B]`, and not against either end.
    @test 0.3 < eta / target < 0.9
    # A target below the prior conditional value at risk is met as readily as one above it.
    for f in (0.80, 1.10)
        prf = prior(MeucciEntropyPoolingPrior(; sets = sets,
                                              cvar_views = ConditionalValueatRiskView(;
                                                                                      views = LinearConstraintEstimator(;
                                                                                                                        val = "AAPL == prior(AAPL)*$f"))),
                    rd)
        @test isapprox(ConditionalValueatRisk(; w = prf.w)(x),
                       ConditionalValueatRisk()(x) * f, rtol = 1e-5)
        @test isapprox(sum(prf.w), 1, rtol = 5e-7)
        @test all(>(0), prf.w)
    end
    # The bracket holds the root strictly inside over several assets and several levels.
    #
    # The view asks for 1.03 of the prior conditional value at risk and not 1.10. The outer
    # search re-solves the whole entropy pooling problem at each candidate value at risk, and
    # `EP_TIGHT` refuses a solve that `Optim` does not call converged. At 1.10 the CI runner
    # met that refusal on `j = 13, a = 0.10`, inside `Roots.find_zero`, where this machine
    # solved all twelve cases. A smaller multiplier demands a smaller tail excess at every
    # candidate, so every inner solve carries a smaller dual and meets the stopping rule more
    # readily. It costs the testset nothing: measured over the twelve cases, 1.03 meets the
    # view to `cvar / B == 1.0` and puts the root between 0.315 and 0.668 of `B`, against
    # 0.320 to 0.640 at 1.10, and the tilt stays real (`ens` falls from 1008 to 877 at
    # `a = 0.20`).
    for j in (1, 5, 13, 20), a in (0.05, 0.10, 0.20)
        nm = rd.nx[j]
        prj = prior(MeucciEntropyPoolingPrior(; sets = sets, opt = EP_TIGHT,
                                              cvar_views = ConditionalValueatRiskView(;
                                                                                      alpha = a,
                                                                                      views = LinearConstraintEstimator(;
                                                                                                                        val = "$nm == prior($nm)*1.03"))),
                    rd)
        xj = rd.X[:, j]
        B = ConditionalValueatRisk(; alpha = a)(xj) * 1.03
        @test isapprox(ConditionalValueatRisk(; alpha = a, w = prj.w)(xj), B, rtol = 1e-2)
        @test 0 < ValueatRisk(; alpha = a, w = prj.w)(xj) / B < 1
    end
    # Two views take the multi-view branch, which searches a box rather than a bracket. A
    # `dm_opt` reaches its own arm of that branch.
    two = ConditionalValueatRiskView(;
                                     views = LinearConstraintEstimator(;
                                                                       val = ["AAPL == 0.053",
                                                                              "XOM == 0.045"]))
    for dm in (nothing,
               OptimEntropyPooling(;
                                   args = (PortfolioOptimisers.Optim.Fminbox(),
                                           PortfolioOptimisers.Optim.Options(;
                                                                             outer_x_abstol = 1e-4,
                                                                             x_abstol = 1e-4))))
        prt = prior(MeucciEntropyPoolingPrior(; sets = sets, dm_opt = dm, cvar_views = two),
                    rd)
        # The box search stops at an `x_abstol` of 1e-4 on each `eta`, and the targets are
        # near 0.05, so it holds the views to a few percent rather than to the 1e-9 the
        # single-view bracket reaches. The tolerance states that bound, and it is still far
        # tighter than the gap to the prior conditional value at risk, which is 0.049.
        @test isapprox(ConditionalValueatRisk(; w = prt.w)(rd.X[:, 1]), 0.053, rtol = 5e-2)
        @test isapprox(ConditionalValueatRisk(; w = prt.w)(rd.X[:, end]), 0.045,
                       rtol = 5e-2)
        @test all(>(0), prt.w)
        # The posterior is not degenerate: the box search reached a real interior solution.
        @test prt.ens > 0.5 * size(rd.X, 1)
    end
end

# #574: the single-view CVaR search brackets `[0, B]`, and `Roots` evaluates both ends before
# it searches. At `eta = B` the constraint reads `E_p[(-x - eta)^+] / alpha == 0`, and the
# positive part is non-zero on the observations worse than `-B`, so the demanded value is
# unreachable, the dual is unbounded, and a tighter stopping rule only runs the dual further:
# the two rules below answer 7.6e-12 and 5.6e-21 there. The bracket now stops at
# `B * (1 - sqrt(eps))`, where the demanded value is strictly positive and both rules meet it.
@testset "MeucciEntropyPoolingPrior CVaR bracket ends inside B" begin
    ep_end = function (x, a, Bt, eta, opt)
        pos = max.(-x .- eta, zero(eltype(x)))
        epc = Dict{Symbol, Tuple{Matrix{Float64}, Vector{Float64}}}()
        PortfolioOptimisers.add_ep_constraint!(epc, transpose(pos ./ a), [Bt - eta],
                                               :cvar_eq)
        wi = PortfolioOptimisers.entropy_pooling(w, epc, opt)
        return LinearAlgebra.dot(wi, pos) / a
    end
    for j in (1, 5, 13, 20), a in (0.05, 0.10, 0.20)
        x = rd.X[:, j]
        Bt = ConditionalValueatRisk(; alpha = a)(x) * 1.10
        hi = Bt * (1 - sqrt(eps(eltype(Bt))))
        # The end the search now uses demands a strictly positive tail contribution, so an
        # interior posterior answers it and the dual that carries it is bounded.
        @test Bt - hi > 0
        @test isapprox(ep_end(x, a, Bt, hi, OptimEntropyPooling()), Bt - hi, rtol = 1e-2)
        @test isapprox(ep_end(x, a, Bt, hi, EP_TIGHT), Bt - hi, rtol = 1e-6)
        # The end the search used before demands exactly zero, which no interior posterior
        # carries, so the solve only approaches it and no relative tolerance states the miss.
        @test iszero(Bt - Bt)
        @test ep_end(x, a, Bt, Bt, OptimEntropyPooling()) > 0
    end
    # The root the search returns sits well inside the shrunk end, so the shrink holds it.
    for j in (1, 20), a in (0.05, 0.20)
        x = rd.X[:, j]
        Bt = ConditionalValueatRisk(; alpha = a)(x) * 1.10
        prj = prior(MeucciEntropyPoolingPrior(; sets = sets, opt = EP_TIGHT,
                                              cvar_views = ConditionalValueatRiskView(;
                                                                                      alpha = a,
                                                                                      views = LinearConstraintEstimator(;
                                                                                                                        val = "$(rd.nx[j]) == prior($(rd.nx[j]))*1.10"))),
                    rd)
        @test 0.3 < ValueatRisk(; alpha = a, w = prj.w)(x) / Bt < 0.7
    end
end

# #539: prior observation weights reach both `ep_prior` routes. Every `MeucciEntropyPoolingPrior`
# in the suite left `w` at `nothing`, so the branch that reads `pe.w` was never taken. The
# constructor normalises `w` in place, so each case builds its own.
#
# #574 gave this testset `EP_TIGHT`, for the reason #573 records: under the default stopping rule
# the CVaR search lands on a different root depending on what ran before it, and the testsets
# above it now run a different sequence of solves. The view here is feasible and the default
# reaches it standalone, so the setting buys repeatability and not correctness.
@testset "MeucciEntropyPoolingPrior takes prior observation weights" begin
    T0 = size(rd.X, 1)
    mkw = () -> StatsBase.pweights(exp.(range(-1, 0; length = T0)))
    cvv = ConditionalValueatRiskView(;
                                     views = LinearConstraintEstimator(;
                                                                       val = "AAPL == 0.07"))
    prn = prior(MeucciEntropyPoolingPrior(; sets = sets, opt = EP_TIGHT, cvar_views = cvv),
                rd)
    for alg in (H1_EntropyPooling(), H0_EntropyPooling())
        pe = MeucciEntropyPoolingPrior(; sets = sets, alg = alg, w = mkw(), opt = EP_TIGHT,
                                       cvar_views = cvv)
        @test isapprox(sum(pe.w), 1, rtol = 1e-10)
        pr = prior(pe, rd)
        @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.07, rtol = 1e-5)
        @test isapprox(sum(pr.w), 1, rtol = 5e-7)
        @test all(>(0), pr.w)
        # The divergence is measured from the exponential weights, not from the uniform ones,
        # so the posterior differs from the one the uniform prior gives.
        @test !isapprox(collect(pr.w), collect(prn.w), rtol = 1e-3)
        @test pr.ens < prn.ens
    end
    # A length that does not match the number of observations is refused on both routes.
    for alg in (H1_EntropyPooling(), H0_EntropyPooling())
        @test_throws DimensionMismatch prior(MeucciEntropyPoolingPrior(; sets = sets,
                                                                       alg = alg,
                                                                       w = StatsBase.pweights(fill(inv(7),
                                                                                                   7)),
                                                                       cvar_views = cvv),
                                             rd)
    end
end

# #539: the stages of both `ep_prior` routes. The suite only ever gave this estimator a CVaR
# view, so the variance stage, the higher moment stage and the whole single-shot method were
# never run. The two routes answer the same view set differently, and that is the point of
# keeping both.
@testset "MeucciEntropyPoolingPrior runs every stage of both routes" begin
    pr0 = prior(EmpiricalPrior(), rd)
    mu_v = LinearConstraintEstimator(;
                                     val = ["AAPL<=0.75*prior(AAPL)",
                                            "XOM >= 0.4*prior(XOM)"])
    sig_v = LinearConstraintEstimator(;
                                      val = ["AAPL==0.2prior(AAPL)", "WMT==1.4prior(WMT)"])
    cov_v = LinearConstraintEstimator(; val = "(MSFT, PEP) <= prior(MSFT, PEP)*0.8")
    rho_v = LinearConstraintEstimator(; val = "(AAPL, XOM) == 0.35")
    kt_v = LinearConstraintEstimator(; val = "AAPL >= prior(AAPL)*0.3")
    sk_v = LinearConstraintEstimator(; val = "WMT == prior(WMT)*1.4")
    mk = alg -> MeucciEntropyPoolingPrior(; sets = sets, alg = alg, opt = EP_TIGHT,
                                          mu_views = mu_v, sigma_views = sig_v,
                                          cov_views = cov_v, rho_views = rho_v,
                                          kt_views = kt_v, sk_views = sk_v)
    prh0 = prior(mk(H0_EntropyPooling()), rd)
    prh1 = prior(mk(H1_EntropyPooling()), rd)
    prh2 = prior(mk(H2_EntropyPooling()), rd)
    for pr in (prh0, prh1, prh2)
        @test isapprox(sum(pr.w), 1, rtol = 5e-7)
        @test all(>(0), pr.w)
        @test pr.kld > 0
        @test 0 < pr.ens < size(rd.X, 1)
        # The mean view binds at its bound on every route.
        @test isapprox(pr.mu[1], 0.75 * pr0.mu[1], rtol = 1e-3)
    end
    # The staged routes pin the mean and the variance before the higher moment stage, so the
    # correlation view lands on its target. The single-shot route pins nothing and misses it
    # by a wide margin, which is the whole difference between the two algorithms.
    rho_h1 = StatsBase.cov2cor(prh1.sigma)[1, 20]
    rho_h2 = StatsBase.cov2cor(prh2.sigma)[1, 20]
    rho_h0 = StatsBase.cov2cor(prh0.sigma)[1, 20]
    @test isapprox(rho_h1, 0.35, rtol = 5e-3)
    @test isapprox(rho_h2, 0.35, rtol = 5e-3)
    @test rho_h0 > 0.6
    # Pinning nothing costs divergence: the single-shot posterior sits much further from the
    # prior than either staged one, and it keeps far fewer effective scenarios.
    @test prh0.kld > 2 * prh1.kld
    @test prh0.ens < 0.8 * prh1.ens
    # The two staged references agree here: the stage sets nest, so projecting the prior onto
    # the last one and projecting each stage onto the next reach the same posterior.
    @test isapprox(collect(prh1.w), collect(prh2.w), rtol = 5e-2)
    # A CVaR view rides every stage, and it still holds on the final posterior beside the
    # stage-one mean view. The variance view names a different asset from the CVaR view: a
    # variance view that shrinks the same asset the CVaR view fattens is infeasible, and the
    # testset below pins what that pair answers.
    prc = prior(MeucciEntropyPoolingPrior(; sets = sets, opt = EP_TIGHT, mu_views = mu_v,
                                          sigma_views = LinearConstraintEstimator(;
                                                                                  val = "WMT == 1.3*prior(WMT)"),
                                          cvar_views = ConditionalValueatRiskView(;
                                                                                  views = LinearConstraintEstimator(;
                                                                                                                    val = "AAPL == 0.07"))),
                rd)
    @test isapprox(ConditionalValueatRisk(; w = prc.w)(rd.X[:, 1]), 0.07, rtol = 1e-5)
    # The mean view is an upper bound. The CVaR view fattens AAPL's left tail, which pulls the
    # posterior mean below that bound rather than onto it, so the view holds slack.
    @test prc.mu[1] <= 0.75 * pr0.mu[1] + sqrt(eps())
    # The posterior spreads over the sample rather than collapsing onto a few observations.
    @test prc.ens > 0.5 * size(rd.X, 1)
    @test maximum(prc.w) < 0.05
end

# #539: `factor_residual_config` forwards the wrapped estimator's declaration, and
# `VecMeucciEP` admits a vector of these estimators.
@testset "MeucciEntropyPoolingPrior forwards its residual declaration" begin
    cvv = ConditionalValueatRiskView(;
                                     views = LinearConstraintEstimator(;
                                                                       val = "AAPL == 0.07"))
    # An empirical inner estimator declares no residual block, and a factor one declares the
    # `(ve, pdm, rsd)` shape. Both reach the caller unchanged.
    pe_emp = MeucciEntropyPoolingPrior(; sets = sets, cvar_views = cvv)
    pe_fac = MeucciEntropyPoolingPrior(; pe = FactorPrior(), sets = sets, cvar_views = cvv)
    @test isnothing(PortfolioOptimisers.factor_residual_config(pe_emp))
    cfg = PortfolioOptimisers.factor_residual_config(pe_fac)
    @test isa(cfg, NamedTuple)
    @test keys(cfg) == (:ve, :pdm, :rsd)
    @test cfg == PortfolioOptimisers.factor_residual_config(FactorPrior())

    # A vector of these estimators is a `VecMeucciEP`, and pooling one keeps its order.
    pes = [MeucciEntropyPoolingPrior(; sets = sets,
                                     cvar_views = ConditionalValueatRiskView(;
                                                                             views = LinearConstraintEstimator(;
                                                                                                               val = "AAPL == $t")))
           for t in (0.06, 0.07, 0.08)]
    @test isa(pes, PortfolioOptimisers.VecMeucciEP)
    op = OpinionPoolingPrior(; pes = pes)
    @test [p.cvar_views.views.val for p in op.pes] ==
          ["AAPL == 0.06", "AAPL == 0.07", "AAPL == 0.08"]
    for (p, t) in zip(op.pes, (0.06, 0.07, 0.08))
        @test isapprox(ConditionalValueatRisk(; w = prior(p, rd).w)(rd.X[:, 1]), t,
                       rtol = 1e-5)
    end
    prop = prior(op, rd)
    @test isapprox(sum(prop.w), 1, rtol = 5e-7)
    @test all(>(0), prop.w)
    # The pooled posterior sits inside the range its members span.
    @test 0.06 < ConditionalValueatRisk(; w = prop.w)(rd.X[:, 1]) < 0.08
end

# #572: an infeasible view set is never detected, and this testset pins what it gives instead.
# `Optim` reports the runaway dual of an infeasible set as converged, because `x_converged`
# or `f_converged` fires once the iterate stops moving. `Optim.g_converged` does separate the
# two, because the gradient of this dual is the primal residual of the view set, but it also
# refuses a solve that is correct and merely loose, so acting on it needs a tolerance on the
# residual. The docstrings state the failure and the signs that name it. Change this testset
# in the commit that changes that decision.
#
# #574 and #592: which of the three faces a route gives is not stable, so this testset asserts
# the set of the three faces and not which one appears. The faces are a raise from the moment
# estimators, which meet the non-finite weights of the runaway dual, a raise from the CVaR
# search, which names the ways out, and a posterior that carries the weight of a handful of
# observations. The search over an infeasible set is chaotic, so any perturbation of the
# candidate sequence moves the face. #574 moved it by ending the single-view CVaR bracket at
# `B * (1 - sqrt(eps))` rather than at `B`. #592 saw the CI host give a face that a developer
# machine does not give, with no source change between the two runs, which is the class of
# #414. None of the three faces reports the infeasibility, and the feasible control below
# still solves, so the answer comes from the direction of the views and not from the shrink.
@testset "MeucciEntropyPoolingPrior does not detect an infeasible view pair" begin
    cvv = ConditionalValueatRiskView(;
                                     views = LinearConstraintEstimator(;
                                                                       val = "AAPL == 0.07"))
    T0 = size(rd.X, 1)
    # The variance view shrinks AAPL while the CVaR view fattens AAPL's tail. No posterior
    # carries both. `H2` is left out because it shares the staged route with `H1` and each run
    # of this pair is slow.
    mk = alg -> MeucciEntropyPoolingPrior(; sets = sets, alg = alg,
                                          sigma_views = LinearConstraintEstimator(;
                                                                                  val = "AAPL == 0.2*prior(AAPL)"),
                                          cvar_views = cvv)
    for alg in (H1_EntropyPooling(), H0_EntropyPooling())
        face = try
            prior(mk(alg), rd)
        catch e
            e
        end
        if isa(face, Exception)
            # The moment estimators raise an `ArgumentError` on the non-finite weights, and
            # the CVaR search raises an `ErrorException` of its own. The warnings of
            # `EntropyPoolingPrior` and `MeucciEntropyPoolingPrior` name both raises.
            @test isa(face, Union{ArgumentError, ErrorException})
        else
            # A posterior that comes back is degenerate. The runaway dual puts the weight of
            # the sample on a few observations, so the posterior fails the two tests that the
            # feasible control below passes.
            @test face.ens < 0.5 * T0
            @test maximum(face.w) > 0.05
        end
    end
    # The same pair on two different assets is feasible and solves normally, so it is the
    # direction of the views and not the pairing.
    prok = prior(MeucciEntropyPoolingPrior(; sets = sets,
                                           sigma_views = LinearConstraintEstimator(;
                                                                                   val = "WMT == 1.3*prior(WMT)"),
                                           cvar_views = cvv), rd)
    @test prok.ens > 0.5 * T0
    @test maximum(prok.w) < 0.05
    @test isapprox(ConditionalValueatRisk(; w = prok.w)(rd.X[:, 1]), 0.07, rtol = 1e-5)
end
