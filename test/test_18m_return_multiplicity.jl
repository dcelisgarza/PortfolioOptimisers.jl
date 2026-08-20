include(joinpath(@__DIR__, "test18_setup.jl"))

# The `JuMPOptimiser.ret` slot takes one return term or several, weighted-summed into the
# model's single scalar `ret` expression. These tests fix the properties the multiplicity was
# built to have, and the ones it was built NOT to have.

@testset "Return multiplicity: the settings bundle" begin
    s = JuMPReturnsSettings()
    @test s.scale == 1.0
    @test isnothing(s.lb)
    @test s.rte
    @test s.fee
    @test s.mic
    @test_throws PortfolioOptimisers.IsNonFiniteError JuMPReturnsSettings(; scale = NaN)
    @test_throws PortfolioOptimisers.IsNonFiniteError JuMPReturnsSettings(; lb = Inf)
    @test_throws PortfolioOptimisers.IsEmptyError JuMPReturnsSettings(; lb = Float64[])
    @test_throws PortfolioOptimisers.IsNonFiniteError JuMPReturnsSettings(; lb = [0.1, NaN])
    # The bundle sits first, matching `Variance(settings, sigma, chol, rc, alg)`.
    @test fieldnames(ArithmeticReturn)[1] === :settings
    @test fieldnames(LogarithmicReturn)[1] === :settings
    # `lb` lives on the bundle, not on the term.
    @test !(:lb in fieldnames(ArithmeticReturn))
    @test !(:lb in fieldnames(LogarithmicReturn))
end

@testset "Return multiplicity: one term builds the same problem as before" begin
    # The single-term case must be unchanged. The check is structural rather than a stored
    # weight vector, because it holds for every problem rather than for one: with one term
    # at `scale = 1` the aggregate `:ret` is *equal as an affine function* to the term's own
    # `:ret_1`, which is the expression the old code registered as `:ret`; and for a bare
    # `ArithmeticReturn` the return builder contributes no variables and no constraints. A
    # problem with the same rows and the same objective has the same solution.
    for opt in (MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv)),
                MeanRisk(; obj = MaximumUtility(), opt = JuMPOptimiser(; pe = pr, slv = slv)),
                RiskBudgeting(; r = Variance(), opt = JuMPOptimiser(; pe = pr, slv = slv)))
        res = optimise(opt)
        mdl = res.model
        @test mdl[:ret] == mdl[:ret_1]
        # `:ret_vec` is a plain Julia vector of expressions, not a model row.
        @test filter(k -> occursin("ret", lowercase(string(k))), keys(mdl.obj_dict)) ⊆
              [:ret, :ret_1, :ret_vec]
    end
end

@testset "Return multiplicity: bounds_returns_estimator pairs term by term" begin
    r1 = ArithmeticReturn()
    r2 = LogarithmicReturn()
    @test bounds_returns_estimator(r1, 0.01).settings.lb == 0.01
    @test isnothing(bounds_returns_estimator(bounds_returns_estimator(r1, 0.01), nothing).settings.lb)
    rs = bounds_returns_estimator([r1, r2], [0.01, 0.02])
    @test [r.settings.lb for r in rs] == [0.01, 0.02]
    # A scalar `nothing` clears all of them.
    @test all(r -> isnothing(r.settings.lb), bounds_returns_estimator(rs, nothing))
    # A scalar number against several terms is refused: the terms need not share a unit.
    @test_throws ArgumentError bounds_returns_estimator([r1, r2], 0.01)
    @test_throws DimensionMismatch bounds_returns_estimator([r1, r2], [0.01])
end

@testset "Return multiplicity: a mean uncertainty set never broadcasts" begin
    # One set is a neighbourhood of one quantity (ADR 0050), so the routing target refuses a
    # vector rather than applying the same ball to every term.
    cfg = JuMPOptimiser(; pe = pr, slv = slv,
                        ret = [ArithmeticReturn(), ArithmeticReturn()])
    @test_throws ArgumentError PortfolioOptimisers.pipe_route(cfg, Val(:mu_ucs),
                                                              L1UncertaintySet(; eps = 0.1))
end

@testset "Return multiplicity: sum-scalarising set-less terms collapses exactly" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    mu1 = pr.mu .* 0.6
    mu2 = pr.mu .* 0.4
    for obj in (MinimumRisk(), MaximumUtility(), MaximumReturn(), MaximumRatio())
        a = optimise(MeanRisk(; obj = obj,
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  ret = [ArithmeticReturn(; mu = mu1),
                                                         ArithmeticReturn(; mu = mu2)])))
        b = optimise(MeanRisk(; obj = obj,
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  ret = ArithmeticReturn(; mu = mu1 .+ mu2))))
        @test isa(a.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test isapprox(a.w, b.w; rtol = 1e-6)
    end
    # `scale` is the weight in the sum, so it re-weights the collapse.
    c = optimise(MeanRisk(; obj = MaximumUtility(),
                          opt = JuMPOptimiser(; pe = pr, slv = slv,
                                              ret = [ArithmeticReturn(; mu = pr.mu,
                                                                      settings = JuMPReturnsSettings(;
                                                                                                     scale = 0.25)),
                                                     ArithmeticReturn(; mu = pr.mu,
                                                                      settings = JuMPReturnsSettings(;
                                                                                                     scale = 0.75))])))
    d = optimise(MeanRisk(; obj = MaximumUtility(),
                          opt = JuMPOptimiser(; pe = pr, slv = slv,
                                              ret = ArithmeticReturn(; mu = pr.mu))))
    @test isapprox(c.w, d.w; rtol = 1e-6)
end

@testset "Return multiplicity: two l1 terms do not collapse" begin
    # `- eps_i * ||sd_i .* w||_inf` summed over terms is a single infinity norm only when
    # every `sd_i` matches, so a pair with different `sd` is not any single term.
    sd1 = vec(std(pr.X; dims = 1))
    sd2 = reverse(sd1) .* 5
    l1a = L1UncertaintySet(; eps = 1.5, sd = sd1, mu = pr.mu)
    l1b = L1UncertaintySet(; eps = 1.5, sd = sd2, mu = pr.mu)
    opt2 = JuMPOptimiser(; pe = pr, slv = slv,
                         ret = [ArithmeticReturn(; ucs = l1a),
                                ArithmeticReturn(; ucs = l1b)])
    res2 = optimise(MeanRisk(; obj = MaximumUtility(), opt = opt2))
    @test isa(res2.retcode, PortfolioOptimisers.OptimisationSuccess)
    # Structurally: one epigraph variable and one cone per term, which no single term emits.
    @test haskey(res2.model, :t_l1ucs_1)
    @test haskey(res2.model, :t_l1ucs_2)
    @test haskey(res2.model, :ret_1)
    @test haskey(res2.model, :ret_2)
    # Numerically: no single term over a sweep of the two radii reproduces the pair.
    best = Inf
    for eps in range(0.1, 5.0; length = 25), s in (sd1, sd2)
        res1 = optimise(MeanRisk(; obj = MaximumUtility(),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                     ret = ArithmeticReturn(;
                                                                            ucs = L1UncertaintySet(;
                                                                                                   eps = eps,
                                                                                                   sd = s,
                                                                                                   mu = pr.mu)))))
        best = min(best, maximum(abs.(res2.w .- res1.w)))
    end
    @test best > 1e-5
end

@testset "Return multiplicity: rte and the constraint-only term" begin
    mu2 = pr.mu .* 2
    # `rte = false` keeps the term out of `ret`, so the objective is the first term alone.
    a = optimise(MeanRisk(; obj = MaximumUtility(),
                          opt = JuMPOptimiser(; pe = pr, slv = slv,
                                              ret = [ArithmeticReturn(; mu = pr.mu),
                                                     ArithmeticReturn(; mu = mu2,
                                                                      settings = JuMPReturnsSettings(;
                                                                                                     rte = false))])))
    b = optimise(MeanRisk(; obj = MaximumUtility(),
                          opt = JuMPOptimiser(; pe = pr, slv = slv,
                                              ret = ArithmeticReturn(; mu = pr.mu))))
    @test isapprox(a.w, b.w; rtol = 1e-6)
    # …and its own bound still binds, which is what makes a constraint-only term expressible.
    lb = dot(mu2, b.w) * 1.1
    c = optimise(MeanRisk(; obj = MaximumUtility(),
                          opt = JuMPOptimiser(; pe = pr, slv = slv,
                                              ret = [ArithmeticReturn(; mu = pr.mu),
                                                     ArithmeticReturn(; mu = mu2,
                                                                      settings = JuMPReturnsSettings(;
                                                                                                     rte = false,
                                                                                                     lb = lb))])))
    @test isa(c.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test dot(mu2, c.w) >= lb - sqrt(eps())
    @test !isapprox(c.w, b.w; rtol = 1e-6)
    # The value-level twin skips an excluded term too.
    rets = [ArithmeticReturn(; mu = pr.mu),
            ArithmeticReturn(; mu = mu2, settings = JuMPReturnsSettings(; rte = false))]
    @test isapprox(expected_return(rets, b.w, pr),
                   expected_return(ArithmeticReturn(; mu = pr.mu), b.w, pr))
end

@testset "Return multiplicity: the bound binds per term, pre-scale" begin
    b = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv)))
    lb = dot(pr.mu, b.w) * 1.2
    # `scale` does not move the bound: it binds on the term's own expression.
    for scale in (0.5, 1.0, 2.0)
        res = optimise(MeanRisk(;
                                opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                    ret = ArithmeticReturn(;
                                                                           settings = JuMPReturnsSettings(;
                                                                                                          scale = scale,
                                                                                                          lb = lb)))))
        @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test dot(pr.mu, res.w) >= lb - sqrt(eps())
    end
end

@testset "Return multiplicity: the return frontier is a product over terms" begin
    mu1 = pr.mu .* 0.6
    mu2 = pr.mu .* 0.4
    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        ret = [ArithmeticReturn(; mu = mu1,
                                                settings = JuMPReturnsSettings(;
                                                                               lb = Frontier(;
                                                                                             N = 3))),
                               ArithmeticReturn(; mu = mu2,
                                                settings = JuMPReturnsSettings(;
                                                                               lb = Frontier(;
                                                                                             N = 4)))])
    res = optimise(MeanRisk(; opt = opt))
    @test length(res.w) == 12
    @test all(x -> isa(x, PortfolioOptimisers.OptimisationSuccess), res.retcode)
    # One term sweeping alone is the single-term frontier, unchanged.
    opt1 = JuMPOptimiser(; pe = pr, slv = slv,
                         ret = ArithmeticReturn(;
                                                settings = JuMPReturnsSettings(;
                                                                               lb = Frontier(;
                                                                                             N = 5))))
    res1 = optimise(MeanRisk(; opt = opt1))
    @test length(res1.w) == 5
    @test issorted(expected_return.(ArithmeticReturn(), res1.w, pr))
    # Each term's span ascends, because it is read off a corner that maximised that term
    # alone rather than off the aggregate maximum-return corner.
    rtf = res.model[:ret_frontier]
    @test length(rtf) == 2
    for (_, vals) in rtf
        @test issorted(vals[2])
    end
    # A product across terms contains infeasible corners whenever the terms disagree: each
    # term's span is read off a corner that left the other terms unconstrained, so demanding
    # the high end of two conflicting terms at once has no solution. The risk-side product
    # behaves the same way, and the sweep reports the failures rather than refusing to run.
    opp = JuMPOptimiser(; pe = pr, slv = slv,
                        ret = [ArithmeticReturn(; mu = pr.mu,
                                                settings = JuMPReturnsSettings(;
                                                                               lb = Frontier(;
                                                                                             N = 3))),
                               ArithmeticReturn(; mu = reverse(pr.mu),
                                                settings = JuMPReturnsSettings(;
                                                                               lb = Frontier(;
                                                                                             N = 4)))])
    resopp = optimise(MeanRisk(; opt = opp))
    @test length(resopp.w) == 12
    @test any(x -> isa(x, PortfolioOptimisers.OptimisationSuccess), resopp.retcode)
    @test any(x -> isa(x, PortfolioOptimisers.OptimisationFailure), resopp.retcode)
end

@testset "Return multiplicity: fees and market impact are per term" begin
    fees = Fees(; l = 0.01, s = 0.01)
    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = fees)
    # Two terms at scale 1 charge the fee twice; the same pair at scale 0.5 charges it once.
    twice = optimise(MeanRisk(; obj = MaximumUtility(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv, fees = fees,
                                                  ret = [ArithmeticReturn(),
                                                         ArithmeticReturn()])))
    blend = optimise(MeanRisk(; obj = MaximumUtility(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv, fees = fees,
                                                  ret = [ArithmeticReturn(;
                                                                          settings = JuMPReturnsSettings(;
                                                                                                         scale = 0.5)),
                                                         ArithmeticReturn(;
                                                                          settings = JuMPReturnsSettings(;
                                                                                                         scale = 0.5))])))
    once = optimise(MeanRisk(; obj = MaximumUtility(), opt = opt))
    @test isapprox(blend.w, once.w; rtol = 1e-6)
    @test !isapprox(twice.w, once.w; rtol = 1e-4)
    # `fee = false` takes the charge out of the term, and the scalar twin follows.
    nofee = ArithmeticReturn(; settings = JuMPReturnsSettings(; fee = false))
    @test isapprox(expected_return(nofee, once.w, pr, fees),
                   expected_return(ArithmeticReturn(), once.w, pr))
    @test expected_return(ArithmeticReturn(), once.w, pr, fees) <
          expected_return(nofee, once.w, pr, fees)
end

@testset "Return multiplicity: MaximumRatio is aggregate-level" begin
    # The ratio constraint is hoisted, so it registers exactly one model-global name.
    res = optimise(MeanRisk(; obj = MaximumRatio(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                ret = [ArithmeticReturn(;
                                                                        mu = pr.mu .* 0.5),
                                                       ArithmeticReturn(;
                                                                        mu = pr.mu .* 0.5)])))
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test haskey(res.model, :sr_ret) ⊻ haskey(res.model, :sr_risk)
    # A logarithmic term has no per-asset quantity, so it forces the risk form — and it now
    # registers `sr_risk` like every other shape, which is what collapsed the two objective
    # methods into one.
    reslog = optimise(MeanRisk(; obj = MaximumRatio(),
                               opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                   ret = LogarithmicReturn())))
    @test isa(reslog.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test haskey(reslog.model, :sr_risk)
    @test !haskey(reslog.model, :sr_elog_ret_risk)
    # An empty numerator is refused rather than solved to an arbitrary feasible point.
    @test_throws ArgumentError optimise(MeanRisk(; obj = MaximumRatio(),
                                                 opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                     ret = [ArithmeticReturn(;
                                                                                             settings = JuMPReturnsSettings(;
                                                                                                                            rte = false))])))
end

@testset "Return multiplicity: every JuMP optimiser reaches the shared seam" begin
    # None of the five optimisers is edited: they all build `ret` through the one
    # `set_return_constraints!` call in the shared Model Assembly (or NOC's own copy of it).
    mu1 = pr.mu .* 0.6
    mu2 = pr.mu .* 0.4
    rets = [ArithmeticReturn(; mu = mu1), ArithmeticReturn(; mu = mu2)]
    one = ArithmeticReturn(; mu = mu1 .+ mu2)
    r = Variance()
    # `FactorRiskContribution` needs factor data, so it gets its own fixture.
    rd_f = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                       timestamp = :Date)[(end - 252):end],
                             TimeArray(CSV.File(joinpath(@__DIR__,
                                                         "./assets/Factors.csv.gz"));
                                       timestamp = :Date)[(end - 252):end])
    pr_f = prior(EmpiricalPrior(), rd_f)
    mu1f = pr_f.mu .* 0.6
    mu2f = pr_f.mu .* 0.4
    resf = optimise(FactorRiskContribution(; r = r,
                                           opt = JuMPOptimiser(; pe = pr_f, slv = slv,
                                                               ret = [ArithmeticReturn(;
                                                                                       mu = mu1f),
                                                                      ArithmeticReturn(;
                                                                                       mu = mu2f)])),
                    rd_f)
    resf1 = optimise(FactorRiskContribution(; r = r,
                                            opt = JuMPOptimiser(; pe = pr_f, slv = slv,
                                                                ret = ArithmeticReturn(;
                                                                                       mu = mu1f .+
                                                                                            mu2f))),
                     rd_f)
    @test isa(resf.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test haskey(resf.model, :ret_1)
    @test haskey(resf.model, :ret_2)
    @test haskey(resf.model, :ret)
    @test isapprox(resf.w, resf1.w; rtol = 1e-5)
    for (mk, mk1, tol) in
        ((() -> MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets)),
          () -> MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv, ret = one)), 1e-5),
         (() -> RiskBudgeting(; r = r,
                              opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets)),
          () -> RiskBudgeting(; r = r, opt = JuMPOptimiser(; pe = pr, slv = slv, ret = one)),
          1e-5),
         (() -> RelaxedRiskBudgeting(;
                                     opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets)),
          () -> RelaxedRiskBudgeting(; opt = JuMPOptimiser(; pe = pr, slv = slv, ret = one)),
          1e-5),
         (() -> NearOptimalCentering(; r = StandardDeviation(),
                                     opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets)),
          () -> NearOptimalCentering(; r = StandardDeviation(),
                                     opt = JuMPOptimiser(; pe = pr, slv = slv, ret = one)),
          # `NearOptimalCentering` stacks four solves, so the two routes agree
          # to solver tolerance rather than to machine precision.
          1e-3))
        res = optimise(mk())
        res1 = optimise(mk1())
        @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test haskey(res.model, :ret_1)
        @test haskey(res.model, :ret_2)
        @test haskey(res.model, :ret)
        @test isapprox(res.w, res1.w; rtol = tol)
    end
end
