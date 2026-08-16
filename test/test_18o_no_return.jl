include(joinpath(@__DIR__, "test18_setup.jl"))

# `NoReturn` is the return-side mirror of `NoRisk`: a term that contributes a zero return
# expression, so an optimiser whose formulation never reads `:ret` can say so instead of
# building a term and discarding it. These tests fix its shape, its coherence table, and the
# one place where it deliberately diverges from the risk side.

@testset "NoReturn: the shape" begin
    r = NoReturn()
    @test isa(r, PortfolioOptimisers.JuMPReturnsEstimator)
    @test fieldnames(NoReturn) == (:settings,)
    @test isa(r.settings, JuMPReturnsSettings)
    @test r.settings.scale == 1.0
    @test isnothing(r.settings.lb)
    # The bundle is the only field, and `lb` lives on it, exactly as on the other two terms.
    @test !(:lb in fieldnames(NoReturn))
    s = JuMPReturnsSettings(; scale = 0.5, lb = 0.01, rte = false)
    @test NoReturn(; settings = s).settings === s
    # The bounds helpers treat it like any other term.
    @test bounds_returns_estimator(r, 0.02).settings.lb == 0.02
    @test isnothing(PortfolioOptimisers.no_bounds_returns_estimator(NoReturn(;
                                                                             settings = s)).settings.lb)
    @test isa(PortfolioOptimisers.no_bounds_returns_estimator(NoReturn(; settings = s)),
              NoReturn)
    # Everything else on the bundle survives the bound being cleared.
    nb = PortfolioOptimisers.no_bounds_returns_estimator(NoReturn(; settings = s))
    @test nb.settings.scale == 0.5
    @test !nb.settings.rte
end

@testset "NoReturn: the predicate uses `all`, where the risk side uses `any`" begin
    @test PortfolioOptimisers.zero_return_expression_flag(NoReturn())
    @test PortfolioOptimisers.zero_return_expression_flag([NoReturn(), NoReturn()])
    @test !PortfolioOptimisers.zero_return_expression_flag(ArithmeticReturn())
    # This is the whole divergence: `:ret` is a sum, so one real term beside a `NoReturn`
    # leaves it non-zero, and every degeneracy the guards catch needs it identically zero.
    @test !PortfolioOptimisers.zero_return_expression_flag([ArithmeticReturn(), NoReturn()])
    @test !PortfolioOptimisers.zero_return_expression_flag([NoReturn(),
                                                            LogarithmicReturn()])
    @test !PortfolioOptimisers.zero_return_expression_flag(PortfolioOptimisers.JuMPReturnsEstimator[])
    # The risk side quantifies the other way on its *type* half, and that is deliberate.
    @test PortfolioOptimisers.norisk_flag([Variance(), NoRisk()])
end

@testset "NoReturn: the expression is identically zero" begin
    res = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv, ret = NoReturn())))
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    mdl = res.model
    @test iszero(mdl[:ret_1])
    @test iszero(mdl[:ret])
    @test mdl[:ret] == mdl[:ret_1]
    # A `MinimumRisk` problem with `NoReturn` is the same problem as one with the default
    # term, because a bare `ArithmeticReturn` contributes to `:ret` alone and `:ret` is not
    # read by this objective.
    ref = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv)))
    @test isapprox(res.w, ref.w; rtol = 5e-5)
    # `MaximumUtility` reads `:ret`, and the term drops out of it, leaving `-l * risk`.
    resu = optimise(MeanRisk(; obj = MaximumUtility(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv, ret = NoReturn())))
    @test isa(resu.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test iszero(resu.model[:ret])
    # The term contributes exactly zero, so the problem is the one a zero characteristic
    # states. The weights are bit-identical, which no solver tolerance is doing the work of.
    refu = optimise(MeanRisk(; obj = MaximumUtility(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                 ret = ArithmeticReturn(;
                                                                        mu = zeros(length(pr.mu))))))
    @test resu.w == refu.w
    # Its argmin is `MinimumRisk`'s. The two are differently-scaled problems, so the weights
    # agree only to solver precision while the risk they achieve agrees far more tightly.
    @test isapprox(expected_risk(Variance(), resu.w, pr),
                   expected_risk(Variance(), ref.w, pr); rtol = 1e-5)
end

@testset "NoReturn: the vestigial term it removes" begin
    # `RiskBudgeting` never reads `:ret`, yet the shared Model Assembly builds it. A
    # `LogarithmicReturn` there raises `T` variables and `T` exponential cones that nothing
    # consumes; `NoReturn` raises none of them, and the answer is the same either way.
    opt_log = RiskBudgeting(; r = Variance(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                ret = LogarithmicReturn()))
    opt_no = RiskBudgeting(; r = Variance(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv, ret = NoReturn()))
    res_log = optimise(opt_log)
    res_no = optimise(opt_no)
    @test isa(res_log.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test isa(res_no.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test JuMP.num_variables(res_no.model) < JuMP.num_variables(res_log.model)
    @test !haskey(res_no.model, :t_elog_ret_1)
    @test haskey(res_log.model, :t_elog_ret_1)
    @test isapprox(res_no.w, res_log.w; rtol = 5e-4)
    # A mean uncertainty set is the same argument with cones instead of variables.
    mu_set = mu_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(),
                                         rng = StableRNG(987654321),
                                         alg = BoxUncertaintySetAlgorithm()), rd.X)
    res_ucs = optimise(RiskBudgeting(; r = Variance(),
                                     opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                         ret = ArithmeticReturn(;
                                                                                ucs = mu_set))))
    @test haskey(res_ucs.model, :bucs_ret_1)
    @test !haskey(res_no.model, :bucs_ret_1)
    @test isapprox(res_no.w, res_ucs.w; rtol = 5e-4)
end

@testset "NoReturn: the three optimisers that never read `:ret`" begin
    # This is the type's main use, so all three must take it.
    res_rb = optimise(RiskBudgeting(; r = Variance(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                        ret = NoReturn())))
    @test isa(res_rb.retcode, PortfolioOptimisers.OptimisationSuccess)
    res_rrb = optimise(RelaxedRiskBudgeting(;
                                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                ret = NoReturn())))
    @test isa(res_rrb.retcode, PortfolioOptimisers.OptimisationSuccess)
    rd_f = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                       timestamp = :Date)[(end - 252):end],
                             TimeArray(CSV.File(joinpath(@__DIR__,
                                                         "./assets/Factors.csv.gz"));
                                       timestamp = :Date)[(end - 252):end])
    pr_f = prior(EmpiricalPrior(), rd_f)
    res_frc = optimise(FactorRiskContribution(; r = Variance(),
                                              opt = JuMPOptimiser(; pe = pr_f, slv = slv,
                                                                  ret = NoReturn())), rd_f)
    @test isa(res_frc.retcode, PortfolioOptimisers.OptimisationSuccess)
end

@testset "NoReturn: the refusals" begin
    # The objective-level refusals live at the shared model-build seam, not at a constructor,
    # so they arrive at `optimise` time (ADR 0054). Construction itself succeeds.
    mr_ret = MeanRisk(; obj = MaximumReturn(),
                      opt = JuMPOptimiser(; pe = pr, slv = slv, ret = NoReturn()))
    @test isa(mr_ret, MeanRisk)
    # The objective would be identically zero, so every feasible portfolio is optimal and the
    # solver returns an arbitrary one while reporting success.
    @test_throws ArgumentError optimise(mr_ret)
    # The ratio's numerator vanishes.
    @test_throws ArgumentError optimise(MeanRisk(; obj = MaximumRatio(),
                                                 opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                     ret = NoReturn())))
    # A vector of nothing but `NoReturn` is the same configuration.
    @test_throws ArgumentError optimise(MeanRisk(; obj = MaximumReturn(),
                                                 opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                     ret = [NoReturn(),
                                                                            NoReturn()])))
    # NOC is infeasible rather than degenerate: its barrier constrains
    # `exp(log_ret) <= ret - rt`, and with no return term both sides are zero. This one is a
    # *formulation* guard, so it stays at the constructor.
    @test_throws ArgumentError NearOptimalCentering(; r = StandardDeviation(),
                                                    opt = JuMPOptimiser(; pe = pr,
                                                                        slv = slv,
                                                                        ret = NoReturn()))
    # The two permitted objectives are not refused, and they solve.
    @test isa(optimise(MeanRisk(;
                                opt = JuMPOptimiser(; pe = pr, slv = slv, ret = NoReturn()))).retcode,
              PortfolioOptimisers.OptimisationSuccess)
    @test isa(optimise(MeanRisk(; obj = MaximumUtility(),
                                opt = JuMPOptimiser(; pe = pr, slv = slv, ret = NoReturn()))).retcode,
              PortfolioOptimisers.OptimisationSuccess)
end

@testset "NoReturn: one real term beside it is coherent everywhere" begin
    # The `all` quantifier exists for exactly this: `any` would refuse configurations that
    # solve correctly, at all three refusal sites.
    rets = [ArithmeticReturn(), NoReturn()]
    res_ret = optimise(MeanRisk(; obj = MaximumReturn(),
                                opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets)))
    @test isa(res_ret.retcode, PortfolioOptimisers.OptimisationSuccess)
    res_ratio = optimise(MeanRisk(; obj = MaximumRatio(),
                                  opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets)))
    @test isa(res_ratio.retcode, PortfolioOptimisers.OptimisationSuccess)
    # `StandardDeviation`, not `Variance`: NOC warns that a `JuMP.QuadExpr` risk expression
    # is not guaranteed to work in its barrier, and it fails there for every return term.
    res_noc = optimise(NearOptimalCentering(; r = StandardDeviation(),
                                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                ret = rets)))
    @test isa(res_noc.retcode, PortfolioOptimisers.OptimisationSuccess)
    # The `NoReturn` contributes nothing, so the pair solves the single-term problem.
    res_one = optimise(MeanRisk(; obj = MaximumReturn(),
                                opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                    ret = ArithmeticReturn())))
    @test isapprox(res_ret.w, res_one.w; rtol = 5e-5)
end

@testset "NoReturn: the ratio's empty numerator is caught by type or by flag" begin
    # The predicate is FUSED, so it sees a vector that mixes the two routes. A composed
    # `all(isa NoReturn) || all(!rte)` would return `false` here and let the degeneracy
    # through (ADR 0054).
    rets = [NoReturn(), ArithmeticReturn(; settings = JuMPReturnsSettings(; rte = false))]
    @test PortfolioOptimisers.zero_return_expression_flag(rets)
    mr = MeanRisk(; obj = MaximumRatio(),
                  opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets))
    @test isa(mr, MeanRisk)
    @test_throws ArgumentError optimise(mr)
    # A `NoReturn` with `rte = true` is still out of the numerator: it contributes zero.
    @test_throws ArgumentError optimise(MeanRisk(; obj = MaximumRatio(),
                                                 opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                     ret = [NoReturn(),
                                                                            NoReturn()])))
end

@testset "NoReturn: the value-level twin" begin
    w = fill(inv(size(pr.X, 2)), size(pr.X, 2))
    @test iszero(expected_return(NoReturn(), w, pr))
    # No charge is applied, so a fees configuration does not make it non-zero. This matches
    # the model, where `settings.fee` and `settings.mic` are inert on this term.
    fees = Fees(; l = 0.01)
    @test iszero(expected_return(NoReturn(), w, pr, fees))
    @test iszero(expected_return(NoReturn(; settings = JuMPReturnsSettings(; scale = 3.0)),
                                 w, pr, fees))
    # The aggregate over a vector adds nothing for it, so the pair equals the single term.
    ar = ArithmeticReturn()
    @test isapprox(expected_return([ar, NoReturn()], w, pr),
                   expected_return(ar, w, pr, nothing))
    @test iszero(expected_return([NoReturn(), NoReturn()], w, pr))
end

@testset "NoReturn: a bound binds on a quantity that is always zero" begin
    # Legal, and pointless, exactly as `NoRisk`'s `settings.ub` is. A non-positive bound is
    # satisfied by the zero expression; a positive one makes the model infeasible.
    ok = optimise(MeanRisk(;
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               ret = NoReturn(;
                                                              settings = JuMPReturnsSettings(;
                                                                                             lb = -0.1)))))
    @test isa(ok.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test haskey(ok.model, :ret_lb_1)
    bad = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                ret = NoReturn(;
                                                               settings = JuMPReturnsSettings(;
                                                                                              lb = 0.1)))))
    @test isa(bad.retcode, PortfolioOptimisers.OptimisationFailure)
end
