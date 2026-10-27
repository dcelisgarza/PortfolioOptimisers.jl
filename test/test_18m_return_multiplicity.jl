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
    # The bundle sits first, as `settings` does on every risk measure.
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
    rd_f = prices_to_returns(price_ingestion(PriceIngestion(),
                                             TimeArray(CSV.File(joinpath(@__DIR__,
                                                                         "./assets/SP500.csv.gz"));
                                                       timestamp = :Date)[(end - 252):end];
                                             F = TimeArray(CSV.File(joinpath(@__DIR__,
                                                                             "./assets/Factors.csv.gz"));
                                                           timestamp = :Date)[(end - 252):end]))
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

# The frontier sweep is a product: every swept return term and every swept risk measure joins
# the same `Iterators.product`, so `k` bounds of `N` points each cost `N^k` full solves.
# `Frontier`'s constructor only caps the `N` of one bound, so the product is capped separately
# at Model Assembly by `assert_frontier_sweep_cap`.

const PO = PortfolioOptimisers

@testset "Frontier sweep cap: counting one bound" begin
    @test PO.frontier_point_count(Frontier(; N = 7)) == 7
    @test PO.frontier_point_count([0.1, 0.2, 0.3]) == 3
    @test PO.frontier_point_count(range(0.1, 0.2; length = 4)) == 4
end

@testset "Frontier sweep cap: two risk frontiers at the ceiling" begin
    # This is the configuration that passes today: `Frontier`'s constructor accepts each `N`
    # on its own, and nothing sees the 10^10-solve product.
    cap = PO.RESOURCE_LIMITS[].max_frontier
    r = [Variance(; settings = RiskMeasureSettings(; ub = Frontier(; N = cap))),
         ConditionalValueatRisk(;
                                settings = RiskMeasureSettings(; ub = Frontier(; N = cap)))]
    mr = MeanRisk(; r = r, obj = MaximumReturn(), opt = JuMPOptimiser(; pe = pr, slv = slv))
    err = try
        optimise(mr)
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    # The message names the product, the factors that made it, and the knob.
    @test err.val == big(cap)^2
    @test occursin("$(big(cap)^2)", err.msg)
    @test length(collect(eachmatch(Regex("_ub = $cap"), err.msg))) == 2
    @test occursin("max_frontier", err.msg)
    @test occursin("set_resource_limits!", err.msg)
    @test occursin("with_resource_limits", err.msg)
end

@testset "Frontier sweep cap: two return frontiers at the ceiling" begin
    cap = PO.RESOURCE_LIMITS[].max_frontier
    ret = [ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = cap))),
           ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = cap)))]
    mr = MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret))
    err = try
        optimise(mr)
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    @test err.val == big(cap)^2
    @test occursin("ret_lb_1", err.msg)
    @test occursin("ret_lb_2", err.msg)
    @test occursin("max_frontier", err.msg)
end

@testset "Frontier sweep cap: the two sides multiply together" begin
    # One bound on each side, each well inside the per-`Frontier` cap, whose product is not.
    cap = PO.RESOURCE_LIMITS[].max_frontier
    mr = MeanRisk(;
                  r = Variance(;
                               settings = RiskMeasureSettings(; ub = Frontier(; N = cap))),
                  opt = JuMPOptimiser(; pe = pr, slv = slv,
                                      ret = ArithmeticReturn(;
                                                             settings = JuMPReturnsSettings(;
                                                                                            lb = Frontier(;
                                                                                                          N = cap)))))
    err = try
        optimise(mr)
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    @test err.val == big(cap)^2
    @test occursin("ret_lb_1 = $cap", err.msg)
    @test length(collect(eachmatch(Regex("_ub = $cap"), err.msg))) == 1
end

@testset "Frontier sweep cap: NearOptimalCentering is guarded through the same seam" begin
    cap = PO.RESOURCE_LIMITS[].max_frontier
    ret = [ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = cap))),
           ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = cap)))]
    noc = NearOptimalCentering(; opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret))
    @test_throws DomainError optimise(noc)
end

@testset "Frontier sweep cap: a stated vector counts by its length" begin
    # A vector of bounds is swept exactly like a `Frontier`, and `Frontier`'s constructor
    # never sees it at all, so the product is the only guard it meets.
    PO.with_resource_limits(; max_frontier = 3) do
        mr = MeanRisk(;
                      r = Variance(;
                                   settings = RiskMeasureSettings(;
                                                                  ub = [1e-4, 2e-4, 3e-4,
                                                                        4e-4])),
                      obj = MaximumReturn(), opt = JuMPOptimiser(; pe = pr, slv = slv))
        err = try
            optimise(mr)
            nothing
        catch e
            e
        end
        @test isa(err, DomainError)
        @test err.val == 4
        @test occursin("RESOURCE_LIMITS[].max_frontier = 3", err.msg)
    end
end

@testset "Frontier sweep cap: an unswept model never meets the guard" begin
    # No frontier registry, no factors, no check — the ordinary path is untouched even at the
    # tightest ceiling the caps admit.
    PO.with_resource_limits(; max_frontier = 1) do
        res = optimise(MeanRisk(; r = Variance(),
                                opt = JuMPOptimiser(; pe = pr, slv = slv)))
        @test isa(res.retcode, PO.OptimisationSuccess)
        @test isa(PO.frontier_sweep_points(res.model),
                  Tuple{BigInt, Vector{Pair{Symbol, Int}}})
        @test PO.frontier_sweep_points(res.model) == (big(1), Pair{Symbol, Int}[])
    end
end

@testset "Frontier sweep cap: a product inside the ceiling still sweeps" begin
    # Both registries populated, 3 x 3 = 9 sweep points against a ceiling of 12. The point is
    # that the two sides multiply rather than being capped one at a time: 3 and 3 each pass a
    # per-bound check trivially, and 9 is what the guard actually compares.
    PO.with_resource_limits(; max_frontier = 12) do
        mr = MeanRisk(;
                      r = Variance(;
                                   settings = RiskMeasureSettings(; ub = Frontier(; N = 3))),
                      opt = JuMPOptimiser(; pe = pr, slv = slv,
                                          ret = ArithmeticReturn(;
                                                                 settings = JuMPReturnsSettings(;
                                                                                                lb = Frontier(;
                                                                                                              N = 3)))))
        res = optimise(mr)
        @test length(res.w) == 9
        total, factors = PO.frontier_sweep_points(res.model)
        @test total == 9
        # The return term is counted first, then the risk measure; both count 3.
        @test last.(factors) == [3, 3]
        @test first(factors).first === :ret_lb_1
        @test endswith(string(last(factors).first), "_ub")
        # One more point on either side would have been refused.
        @test_throws DomainError optimise(MeanRisk(;
                                                   r = Variance(;
                                                                settings = RiskMeasureSettings(;
                                                                                               ub = Frontier(;
                                                                                                             N = 4))),
                                                   opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                       ret = ArithmeticReturn(;
                                                                                              settings = JuMPReturnsSettings(;
                                                                                                                             lb = Frontier(;
                                                                                                                                           N = 4))))))
    end
end

using Test, PortfolioOptimisers

const PO = PortfolioOptimisers

# The frontier sweep's flat order is load-bearing, not cosmetic (ADR 0062).
# `NearOptimalCentering` solves its anchors as one `MeanRisk` sweep over the same two
# frontiers and pairs anchor `i` with sweep point `i`, so the two sweeps must enumerate
# their points in the same order. `frontier_sweep_axes` is the one place that order is
# stated. These tests need no solver and no model: they drive the seam on synthetic
# registries.

# A registry entry is `(bound_var_key, bound_key) => (expr, points, …)`. `frontier_axis`
# reads only the first key and the points, so the expression is `nothing` here.
ret_registry = [(:ret_lb_var_1, :ret_lb_1) => (nothing, [10.0, 20.0], 1)]
risk_registry = [(:sd_ub_var, :sd_ub) => (nothing, [1.0, 2.0, 3.0], true)]

@testset "Frontier sweep order: one axis enumerates its own points" begin
    axis = PO.frontier_axis(ret_registry)
    pts = PO.frontier_sweep_axes(axis, nothing)
    @test length(pts) == 2
    # Each point is a tuple of one `(keys, bounds)` pair, and each pair holds one entry.
    @test [only(p[1][1]) for p in pts] == [:ret_lb_var_1, :ret_lb_var_1]
    @test [only(p[1][2]) for p in pts] == [10.0, 20.0]
end

@testset "Frontier sweep order: several entries on one axis are a product" begin
    two = [(:ret_lb_var_1, :ret_lb_1) => (nothing, [10.0, 20.0], 1),
           (:ret_lb_var_2, :ret_lb_2) => (nothing, [30.0, 40.0, 50.0], 2)]
    pts = PO.frontier_sweep_axes(PO.frontier_axis(two), nothing)
    # Two bounds of 2 and 3 points cost 6 solves, which is what `frontier_sweep_points`
    # counts.
    @test length(pts) == 6
    # Iterate exactly as `frontier_sweep!` does, so this asserts the traversal and not the
    # shape a comprehension over a product would give back.
    order = Vector{Float64}[]
    for p in pts
        push!(order, collect(p[1][2]))
    end
    @test order == [[10.0, 30.0], [20.0, 30.0], [10.0, 40.0], [20.0, 40.0], [10.0, 50.0],
                    [20.0, 50.0]]
end

@testset "Frontier sweep order: the risk axis varies fastest" begin
    ret_axis = PO.frontier_axis(ret_registry)
    risk_axis = PO.frontier_axis(risk_registry)
    pts = PO.frontier_sweep_axes(ret_axis, risk_axis)
    @test length(pts) == 6
    # `p[1]` is the risk axis and `p[2]` the return axis. The return bound is held while the
    # risk bound runs, so the flat order is return-outer and risk-inner.
    order = Tuple{Float64, Float64}[]
    for p in pts
        push!(order, (only(p[2][2]), only(p[1][2])))
    end
    @test order ==
          [(10.0, 1.0), (10.0, 2.0), (10.0, 3.0), (20.0, 1.0), (20.0, 2.0), (20.0, 3.0)]
end

@testset "Frontier sweep order: a sweep with no frontier still has its points" begin
    # The unconstrained `NearOptimalCentering` sweep: no bound to write, one solve per
    # anchor.
    pts = Iterators.repeated((), 4)
    @test length(pts) == 4
    @test all(p -> p === (), pts)
end

# A `:risk_frontier` entry is `(bound_var_key, bound_key) => (expr, points, flag, owner)`.
# The registry is **not** parallel to the risk measure vector: a measure registers an entry
# only when its `settings.ub` is a `Front_NumVec`. So an entry holds the measure that owns it,
# and the rebuild reads both the owner and the expression off the entry it is rebuilding.
@testset "Frontier rebuild: an entry is rebuilt from itself, not from entry 1" begin
    X = [0.01 0.02 -0.01; -0.02 0.01 0.03; 0.03 -0.01 0.02; -0.01 0.02 -0.02;
         0.02 -0.03 0.01; 0.01 0.01 0.01; -0.03 0.02 -0.01; 0.02 0.01 0.02]
    pr = prior(EmpiricalPrior(), X)
    w_min = fill(1 / 3, 3)
    w_max = [0.7, 0.2, 0.1]
    front = Frontier(; N = 3)
    rs = factory([Variance(), ConditionalValueatRisk()], pr)
    registry = [(:a_ub_var, :a_ub) => (:EXPR_ONE, front, true, rs[1]),
                (:b_ub_var, :b_ub) => (:EXPR_TWO, front, false, rs[2])]

    entry = PO._rebuild_risk_frontier(pr, nothing, rs[2], registry, w_min, w_max, 2)
    # The keys, the expression, the polarity and the owner are entry 2's own. Only the bound
    # changes. Reading the expression off entry 1 bounds the first measure twice and leaves
    # the second measure unbounded.
    @test entry.first == (:b_ub_var, :b_ub)
    @test entry.second[1] === :EXPR_TWO
    @test entry.second[3] === false
    @test entry.second[4] === rs[2]
    # The span is measure 2's own risk at the two corner portfolios.
    @test collect(entry.second[2]) ≈
          range(expected_risk(rs[2], w_min, X), expected_risk(rs[2], w_max, X); length = 3)

    entry1 = PO._rebuild_risk_frontier(pr, nothing, rs[1], registry, w_min, w_max, 1)
    @test entry1.second[1] === :EXPR_ONE
    @test entry1.second[3] === true
    @test entry1.second[4] === rs[1]
end

@testset "Frontier rebuild: an owner is written once, into the entries that have none" begin
    model = JuMP.Model()
    x = JuMP.@variable(model)
    front = Frontier(; N = 2)
    rv, rc, rs = Variance(), ConditionalValueatRisk(), StandardDeviation()
    model[:risk_frontier] = Pair{Tuple{Symbol, Symbol},
                                 Tuple{JuMP.AbstractJuMPScalar, PO.Front_NumVec, Bool,
                                       Union{Nothing, PO.AbstractBaseRiskMeasure}}}[(:a_ub_var, :a_ub) => (1.0 * x,
                                                                                                           front,
                                                                                                           true,
                                                                                                           rv),
                                                                                    (:b_ub_var, :b_ub) => (1.0 * x,
                                                                                                           front,
                                                                                                           true,
                                                                                                           nothing),
                                                                                    (:c_ub_var, :c_ub) => (1.0 * x,
                                                                                                           front,
                                                                                                           true,
                                                                                                           rc)]
    PO.set_risk_frontier_owner!(model, 0, rs)
    # A composite records the measure of a term before the whole measure claims the rest, so
    # the owners of entries 1 and 3 stay. Entry 2 had none, and takes `rs`.
    @test [e.second[4] for e in model[:risk_frontier]] == [rv, rs, rc]
    # Entries at or before `first` belong to an earlier measure and are not read.
    PO.set_risk_frontier_owner!(model, 3, rv)
    @test model[:risk_frontier][2].second[4] === rs
end

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
    @test entry.second[4] isa ConditionalValueatRisk

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
    @test registry[1].second[4] isa Variance
    @test registry[2].second[4] isa ConditionalValueatRisk
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
