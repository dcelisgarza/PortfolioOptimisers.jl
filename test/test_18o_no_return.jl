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
    rd_f = prices_to_returns(price_ingestion(PriceIngestion(),
                                             TimeArray(CSV.File(joinpath(@__DIR__,
                                                                         "./assets/SP500.csv.gz"));
                                                       timestamp = :Date)[(end - 252):end];
                                             F = TimeArray(CSV.File(joinpath(@__DIR__,
                                                                             "./assets/Factors.csv.gz"));
                                                           timestamp = :Date)[(end - 252):end]))
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

@testset "Combination weight: scale is inert on a lone term" begin
    using PortfolioOptimisers, Test, StableRNGs, Clarabel, JuMP, LinearAlgebra

    #=
    `scale` is a combination weight: it says how much an element contributes to an
    expression built from several elements. One element is not a combination, so the weight
    has nothing to weigh and both axes drop it before the expression is built.

    These assertions are structural, so they need no solver and cannot drift with a
    tolerance. They fail loudly if a future builder reintroduces the multiplier.
    =#

    rng = StableRNG(987654321)
    X = randn(rng, 300, 10) ./ 100 .+ 0.0005
    slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    pr = prior(EmpiricalPrior(), X)
    scales = (1, 2, 50)

    ret_opt(sc) = JuMPOptimiser(; pe = pr, slv = slv,
                                ret = ArithmeticReturn(;
                                                       settings = JuMPReturnsSettings(;
                                                                                      scale = sc)))
    risk_r(sc) = StandardDeviation(; settings = RiskMeasureSettings(; scale = sc))

    @testset "Return axis: one term" begin
        # A lone term contributes its own expression, never a multiple of it.
        for sc in scales
            res = optimise(MeanRisk(; r = StandardDeviation(), obj = MinimumRisk(),
                                    opt = ret_opt(sc)); save = true)
            @test JuMP.isequal_canonical(res.model[:ret], res.model[:ret_1])
        end
    end

    @testset "Return axis: several terms still combine" begin
        # The weight must survive where there really is a combination.
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            ret = [ArithmeticReturn(;
                                                    settings = JuMPReturnsSettings(;
                                                                                   scale = 2)),
                                   ArithmeticReturn(;
                                                    settings = JuMPReturnsSettings(;
                                                                                   scale = 3))])
        res = optimise(MeanRisk(; r = StandardDeviation(), obj = MinimumRisk(), opt = opt);
                       save = true)
        @test !JuMP.isequal_canonical(res.model[:ret], res.model[:ret_1])
    end

    @testset "Risk axis: one measure" begin
        # `StandardDeviation` puts an epigraph variable in `:risk`, so a surviving weight
        # would show up as a coefficient other than one.
        for sc in scales
            res = optimise(MeanRisk(; r = risk_r(sc), obj = MinimumRisk(),
                                    opt = ret_opt(1)); save = true)
            @test all(isone, values(res.model[:risk].terms))
        end
    end

    @testset "Risk axis: several measures still combine" begin
        res = optimise(MeanRisk(; r = [risk_r(2), ConditionalValueatRisk()],
                                obj = MinimumRisk(), opt = ret_opt(1)); save = true)
        @test !all(isone, values(res.model[:risk].terms))
    end

    @testset "Numerical: the portfolio does not move" begin
        #=
        `MaximumUtility` is the objective that makes a lone weight observable: it optimises
        `ret - l * risk`, so a weight on either axis changes the trade-off rather than
        cancelling. `MinimumRisk`, `MaximumReturn` and `MaximumRatio` are argmax-invariant,
        which is why the return axis carried the defect unnoticed.
        =#
        for obj in (MinimumRisk(), MaximumUtility(), MaximumRatio(), MaximumReturn())
            w_ret = map(scales) do sc
                optimise(MeanRisk(; r = StandardDeviation(), obj = obj, opt = ret_opt(sc))).w
            end
            @test all(w -> w == first(w_ret), w_ret)

            w_risk = map(scales) do sc
                optimise(MeanRisk(; r = risk_r(sc), obj = obj, opt = ret_opt(1))).w
            end
            @test all(w -> w == first(w_risk), w_risk)
        end
    end
end

# A degeneracy guard tests the STATE of the expression, not the TYPE of the term (ADR 0054).
# Two routes reach a zero expression on each axis: the sentinel type (`NoReturn` / `NoRisk`)
# and the inclusion flag (`rte` / `rke`). These tests fix both routes, on both axes, and pin
# the configurations that must keep working.

const rte_off = JuMPReturnsSettings(; rte = false)
rke_off(; kwargs...) = RiskMeasureSettings(; rke = false, kwargs...)

@testset "Degeneracy guard: the return predicate is fused" begin
    # `:ret` is zero exactly when every term is out of it, by either route.
    @test PortfolioOptimisers.zero_return_expression_flag(NoReturn())
    @test PortfolioOptimisers.zero_return_expression_flag(ArithmeticReturn(;
                                                                           settings = rte_off))
    @test PortfolioOptimisers.zero_return_expression_flag([ArithmeticReturn(;
                                                                            settings = rte_off),
                                                           LogarithmicReturn(;
                                                                             settings = rte_off)])
    # The fusion is the whole point. A composed `all(isa NoReturn) || all(!rte)` returns
    # `false` on this vector, though every term is out of `:ret`.
    mixed = [NoReturn(), ArithmeticReturn(; settings = rte_off)]
    @test PortfolioOptimisers.zero_return_expression_flag(mixed)
    @test !(all(x -> isa(x, NoReturn), mixed) || all(x -> !x.settings.rte, mixed))
    # One included term leaves the expression non-zero, by either route.
    @test !PortfolioOptimisers.zero_return_expression_flag([NoReturn(), ArithmeticReturn()])
    @test !PortfolioOptimisers.zero_return_expression_flag([ArithmeticReturn(;
                                                                             settings = rte_off),
                                                            ArithmeticReturn()])
    # An empty vector is not degenerate; it is refused separately.
    @test !PortfolioOptimisers.zero_return_expression_flag(PortfolioOptimisers.JuMPReturnsEstimator[])
end

@testset "Degeneracy guard: the risk predicate composes" begin
    # The two halves carry different quantifiers, so they compose where the return axis fuses.
    @test PortfolioOptimisers.zero_risk_expression_flag(NoRisk())
    @test PortfolioOptimisers.zero_risk_expression_flag(Variance(; settings = rke_off()))
    @test PortfolioOptimisers.zero_risk_expression_flag([Variance(; settings = rke_off()),
                                                         StandardDeviation(;
                                                                           settings = rke_off())])
    # `any` on the type: a `NoRisk` beside a real measure is refused on its own terms.
    @test PortfolioOptimisers.zero_risk_expression_flag([Variance(), NoRisk()])
    # `all` on the state: one included measure leaves `:risk` non-zero.
    @test !PortfolioOptimisers.zero_risk_expression_flag([Variance(),
                                                          Variance(; settings = rke_off())])
    @test !PortfolioOptimisers.zero_risk_expression_flag(Variance())
end

@testset "Degeneracy guard: `rte = false` on every term is refused" begin
    # This is the defect the rule removes. Before it, `MaximumReturn` reported success on an
    # identically-zero objective and returned an arbitrary feasible portfolio.
    all_off = ArithmeticReturn(; settings = rte_off)
    mr = MeanRisk(; obj = MaximumReturn(),
                  opt = JuMPOptimiser(; pe = pr, slv = slv, ret = all_off))
    @test isa(mr, MeanRisk)
    @test_throws ArgumentError optimise(mr)
    # The vector form, and the form that mixes the two routes.
    @test_throws ArgumentError optimise(MeanRisk(; obj = MaximumReturn(),
                                                 opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                     ret = [all_off,
                                                                            NoReturn()])))
    @test_throws ArgumentError optimise(MeanRisk(; obj = MaximumRatio(),
                                                 opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                     ret = [all_off,
                                                                            NoReturn()])))
    # NOC's formulation guard is the same predicate, and it stays at the constructor.
    @test_throws ArgumentError NearOptimalCentering(; r = StandardDeviation(),
                                                    opt = JuMPOptimiser(; pe = pr,
                                                                        slv = slv,
                                                                        ret = all_off))
end

@testset "Degeneracy guard: a constraint-only return term still works" begin
    # `rte = false` beside an included term is the deliberate constraint-only shape. The
    # refusal only ever bites when EVERY term is out, so nothing here is narrowed.
    rets = [ArithmeticReturn(),
            ArithmeticReturn(; settings = JuMPReturnsSettings(; rte = false, lb = -0.1))]
    res = optimise(MeanRisk(; obj = MaximumReturn(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets)))
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test haskey(res.model, :ret_lb_2)
    # An all-`rte = false` configuration is legitimate under the objectives that never read
    # `:ret`, so those are not refused.
    off = ArithmeticReturn(; settings = rte_off)
    for obj in (MinimumRisk(), MaximumUtility())
        r = optimise(MeanRisk(; obj = obj,
                              opt = JuMPOptimiser(; pe = pr, slv = slv, ret = off)))
        @test isa(r.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test iszero(r.model[:ret])
    end
end

@testset "Degeneracy guard: FactorRiskContribution gets a return-side guard" begin
    # It carries its own `obj` and had NO return-side guard at all, so it was refused under
    # `MaximumRatio` (a model-build check) and accepted under `MaximumReturn` (a `MeanRisk`
    # constructor check) — uneven against itself. The shared seam closes both.
    rd_f = prices_to_returns(price_ingestion(PriceIngestion(),
                                             TimeArray(CSV.File(joinpath(@__DIR__,
                                                                         "./assets/SP500.csv.gz"));
                                                       timestamp = :Date)[(end - 252):end];
                                             F = TimeArray(CSV.File(joinpath(@__DIR__,
                                                                             "./assets/Factors.csv.gz"));
                                                           timestamp = :Date)[(end - 252):end]))
    pr_f = prior(EmpiricalPrior(), rd_f)
    for ret in (NoReturn(), ArithmeticReturn(; settings = rte_off))
        frc = FactorRiskContribution(; r = Variance(), obj = MaximumReturn(),
                                     opt = JuMPOptimiser(; pe = pr_f, slv = slv, ret = ret))
        @test_throws ArgumentError optimise(frc, rd_f)
    end
    # `MinimumRisk` never reads `:ret`, so this is the case the type exists for.
    ok = optimise(FactorRiskContribution(; r = Variance(),
                                         opt = JuMPOptimiser(; pe = pr_f, slv = slv,
                                                             ret = NoReturn())), rd_f)
    @test isa(ok.retcode, PortfolioOptimisers.OptimisationSuccess)
end

@testset "Degeneracy guard: MaximumElementReturn is per index and ignores `rte`" begin
    obj_i = PortfolioOptimisers.MaximumElementReturn
    # A `NoReturn` at the named index makes `ret_i` itself zero, whatever the other terms do.
    @test_throws ArgumentError PortfolioOptimisers.assert_no_return_objective_compatibility([NoReturn(),
                                                                                             ArithmeticReturn()],
                                                                                            obj_i(1))
    # Index 2 is a real term, so the same vector is accepted there.
    @test isnothing(PortfolioOptimisers.assert_no_return_objective_compatibility([NoReturn(),
                                                                                  ArithmeticReturn()],
                                                                                 obj_i(2)))
    # `rte` is NOT consulted: it removes a term from the summed `:ret`, while this objective
    # reads `ret_i` directly, and the builder registers `ret_i` whatever the flag says.
    @test isnothing(PortfolioOptimisers.assert_no_return_objective_compatibility([ArithmeticReturn(;
                                                                                                   settings = rte_off)],
                                                                                 obj_i(1)))
    # An index that names no term is a DomainError, not a downstream `KeyError`, and the
    # check runs FIRST, because the `NoReturn` test indexes `ret[i]`. This is the upper half
    # of the domain whose lower half the constructor already refuses (#321).
    @test_throws DomainError PortfolioOptimisers.assert_no_return_objective_compatibility([ArithmeticReturn()],
                                                                                          obj_i(2))
    @test_throws DomainError PortfolioOptimisers.assert_no_return_objective_compatibility([NoReturn(),
                                                                                           NoReturn()],
                                                                                          obj_i(3))
    @test_throws DomainError PortfolioOptimisers.assert_no_return_objective_compatibility(ArithmeticReturn(),
                                                                                          obj_i(2))
    @test_throws DomainError obj_i(0)
    # A lone estimator registers `ret_1`, so index 1 is valid there: the predicate is one
    # `1 <= i <= length` with no singular/vector split.
    @test isnothing(PortfolioOptimisers.assert_no_return_objective_compatibility(ArithmeticReturn(),
                                                                                 obj_i(1)))
end

@testset "Degeneracy guard: the internally-generated indices keep passing" begin
    # `return_term_ends` builds a full `MeanRisk(; obj = MaximumElementReturn(i), …)`, so it
    # goes through model assembly and meets the range check on every corner solve. Its `i`
    # comes from the swept indices, so it is valid by construction — but the check must not
    # be written so as to refuse it.
    rets = [ArithmeticReturn(), LogarithmicReturn()]
    res = optimise(NearOptimalCentering(; r = StandardDeviation(),
                                        opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                            ret = rets)))
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    # The frontier sweep swaps the objective on an already-built model rather than
    # reassembling, so its `MaximumElementReturn` never reaches the guard at all. Its `i`
    # comes from the `:ret_frontier` registry, so it is valid by construction either way.
    swept = [ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = 3))),
             ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = 3)))]
    fr = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv, ret = swept)))
    # Two `Frontier` bounds give a 3 x 3 sweep, and every point is solved.
    @test length(fr.retcode) == 9
    @test all(x -> isa(x, PortfolioOptimisers.OptimisationSuccess), fr.retcode)
end

@testset "Degeneracy guard: `rke = false` on every measure is refused" begin
    # The risk axis had no `rke` check anywhere in `src/`, inherited unguarded from PR #21.
    off = Variance(; settings = rke_off())
    @test_throws ArgumentError MeanRisk(; r = off, obj = MinimumRisk(),
                                        opt = JuMPOptimiser(; pe = pr, slv = slv))
    @test_throws ArgumentError MeanRisk(;
                                        r = [off,
                                             StandardDeviation(; settings = rke_off())],
                                        obj = MaximumRatio(),
                                        opt = JuMPOptimiser(; pe = pr, slv = slv))
    @test_throws ArgumentError RiskBudgeting(; r = off,
                                             opt = JuMPOptimiser(; pe = pr, slv = slv))
    @test_throws ArgumentError NearOptimalCentering(;
                                                    r = StandardDeviation(;
                                                                          settings = rke_off()),
                                                    opt = JuMPOptimiser(; pe = pr,
                                                                        slv = slv))
    # `MaximumReturn` never reads `:risk`, so a zero risk expression is legitimate there.
    ok = optimise(MeanRisk(; r = off, obj = MaximumReturn(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = 1.0,
                                               wb = WeightBounds(; lb = 0.0, ub = 1.0))))
    @test isa(ok.retcode, PortfolioOptimisers.OptimisationSuccess)
end

@testset "Degeneracy guard: a constraint-only risk measure still works" begin
    # The state half quantifies with `all` for exactly this: a measure that binds a `ub`
    # without entering the objective. It is the risk-side twin of the constraint-only return
    # term, and `any` would refuse it.
    ub = expected_risk(Variance(), fill(inv(size(pr.X, 2)), size(pr.X, 2)), pr)
    r = [Variance(), Variance(; settings = rke_off(; ub = ub))]
    @test !PortfolioOptimisers.zero_risk_expression_flag(r)
    res = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv)))
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
end

@testset "Degeneracy guard: `rke` is inert in the clustering optimisers" begin
    # HRP and HERC never reach the JuMP risk builders, so `rke` never touched them. Widening
    # their predicate would refuse configurations that solve correctly today, so they keep
    # `norisk_flag`.
    off = Variance(; settings = rke_off())
    res_hrp = optimise(HierarchicalRiskParity(; r = off), rd)
    @test isa(res_hrp.retcode, PortfolioOptimisers.OptimisationSuccess)
    # The answer is the flag-free one, which is what "inert" means.
    ref_hrp = optimise(HierarchicalRiskParity(; r = Variance()), rd)
    @test isapprox(res_hrp.w, ref_hrp.w)
    res_herc = optimise(HierarchicalEqualRiskContribution(; ri = off), rd)
    @test isa(res_herc.retcode, PortfolioOptimisers.OptimisationSuccess)
    # The type route is still refused there, because a `NoRisk` really does measure nothing.
    @test_throws ArgumentError HierarchicalRiskParity(; r = NoRisk())
    @test_throws ArgumentError HierarchicalEqualRiskContribution(; ri = NoRisk())
end

@testset "Degeneracy guard: the ratio's scale floor closes the homogenised ray (#924)" begin
    # Homogenisation carries a second degeneracy, and it is the mirror of the two above: the
    # objective is not identically zero here, it is non-positive, and the supremum of a
    # homogeneous objective that never rises above zero is zero at the origin. A mean
    # uncertainty set of a large enough radius puts the model there, because no feasible
    # portfolio's worst-case return can then beat `rf`. Every constraint the model writes is
    # homogeneous, so all of them hold on that ray, and `w = y / k` came back breaking the
    # bound the caller wrote while the result reported success.
    T, N = size(pr.X)
    # A hand-built set rather than a calibrated one: the radius is the whole fixture, and a
    # calibration would obscure which side of the collapse the test is on.
    ue(kappa) = EllipsoidalUncertaintySet(; sigma = Matrix(Diagonal(diag(pr.sigma) ./ T)),
                                          k = kappa, class = MuUncertaintySetClass())
    # `group1` is a dense row over half the universe; the single-asset row is the sparse one.
    # A denser row carries more of the violation, so both are pinned.
    function ratio(kappa; obj = MaximumRatio(), val = "$(rd.nx[1]) >= 0.09")
        return optimise(MeanRisk(; r = Variance(), obj = obj,
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = 1.0,
                                                     wb = WeightBounds(; lb = 0.0,
                                                                       ub = 0.1),
                                                     sets = sets,
                                                     lcse = LinearConstraintEstimator(;
                                                                                      val = val),
                                                     ret = ArithmeticReturn(;
                                                                            ucs = ue(kappa)))),
                        rd)
    end
    kof(res) = JuMP.value(PortfolioOptimisers.get_k(res.jr.model))
    # The floor the `nothing` rule sizes, re-derived from the same two numbers it reads.
    ohf = min(1e3, max(1e-3, mean(abs.(pr.mu))))
    kfloor = 1e-4 * ohf / max(ohf, maximum(pr.mu))

    @testset "It is inert where the ratio has a tangency portfolio to find" begin
        # A radius small enough that the worst case still beats `rf` pins `k` through the
        # risk cap alone, and the floor is orders below it. The proof that it is inert is
        # that a floor a million times smaller gives the same answer.
        tight = ratio(1e-6)
        loose = ratio(1e-6; obj = MaximumRatio(; kmin = kfloor * 1e-6))
        @test isa(tight.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test kof(tight) > 1e3 * kfloor
        @test isapprox(kof(tight), kof(loose); rtol = 1e-6)
        @test isapprox(tight.w, loose.w; atol = 1e-7)
    end

    @testset "It closes the ray, and the caller's constraints survive it" begin
        for val in ("$(rd.nx[1]) >= 0.09", "group1 >= 0.5")
            res = ratio(1e3; val = val)
            @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
            # `k` comes back sitting on the floor: nothing else pins it here.
            @test isapprox(kof(res), kfloor; rtol = 1e-3)
            # Which is the whole point. The recovered weights meet the row that was written,
            # and the bound that was not, both of which the collapsed ray broke. The
            # tolerance is what a floored `k` costs: `w = y / k` carries the solver's
            # residual on `y` divided by `k`, and `k` sits at `kfloor` here, five orders
            # below the scale a tangency portfolio would have pinned.
            @test all(x -> x <= 0.1 + 1e-4, res.w)
            @test isapprox(sum(res.w), 1.0; atol = 1e-7)
        end
        @test ratio(1e3).w[1] >= 0.09 - 1e-4
        # The dense row is the one that carried the larger violation before the floor.
        gidx = [findfirst(==(x), rd.nx) for x in sets.dict["group1"]]
        @test sum(ratio(1e3; val = "group1 >= 0.5").w[gidx]) >= 0.5 - 1e-4
    end

    @testset "The floor is caller-visible, and a `k` on it is the signal" begin
        # A floor the caller sets is used as written, which is how a model that wants the
        # old scale asks for it.
        res = ratio(1e3; obj = MaximumRatio(; kmin = 0.05))
        @test isapprox(kof(res), 0.05; rtol = 1e-5)
        # And it is a floor, not a target: raising it on a model that has a tangency
        # portfolio above it leaves that answer alone.
        @test isapprox(kof(ratio(1e-6; obj = MaximumRatio(; kmin = 0.05))),
                       kof(ratio(1e-6)); rtol = 1e-5)
    end

    @testset "`kmin` is validated where `ohf` is" begin
        @test isnothing(MaximumRatio().kmin)
        @test MaximumRatio(; kmin = 0.5).kmin == 0.5
        # A stated zero turns the floor off and keeps only `k >= 0`.
        @test MaximumRatio(; kmin = 0.0).kmin == 0.0
        @test_throws DomainError MaximumRatio(; kmin = -1e-8)
    end
end

@testset "Value-level multiplicity: the singular case is unchanged" begin
    using PortfolioOptimisers, Test, StableRNGs, Clarabel, JuMP, LinearAlgebra, Statistics

    #=
    The whole value-level `expected_*` surface now takes one risk measure or a vector of
    them. The first thing that must hold is that nothing moved for a caller who passes one
    measure, so these assertions are `==` and not `isapprox`: the singular path must be
    **bit-identical**, not merely close.
    =#

    rng = StableRNG(20260816)
    X = randn(rng, 250, 8) ./ 100 .+ 0.0004
    w = fill(inv(8), 8)
    pr = prior(EmpiricalPrior(), X)

    rv = Variance()
    rc = ConditionalValueatRisk()
    rs = StandardDeviation()

    @testset "one measure gives the same number through every entry" begin
        for r in (rv, rc, rs)
            # The prior route and the matrix route are separate methods; both stayed put.
            @test expected_risk(r, w, pr) == expected_risk(factory(r, pr), w, pr.X)
        end
    end

    @testset "`sca` on a singular call is inert" begin
        # A scalariser over one element returns that element, so every scalariser agrees
        # with the un-scalarised call. It is swallowed by `kwargs...`, never consulted.
        base = expected_risk(rv, w, pr)
        for sca in
            (SumScalariser(), MaxScalariser(), MinScalariser(), LogSumExpScalariser())
            @test expected_risk(rv, w, pr; sca = sca) == base
        end
    end

    @testset "a one-element vector under `SumScalariser` equals the singular call" begin
        #=
        Load-bearing identity. #299 leaned on it when it accepted the reported-versus-
        optimised mismatch in silence: a caller who names no scalariser anywhere gets a
        matched figure, because the sum is the identity on one element and
        `JuMPOptimiser.sca` defaults to `SumScalariser()` too.
        =#
        for r in (rv, rc, rs)
            @test expected_risk([r], w, pr) == expected_risk(r, w, pr)
        end
    end

    @testset "`scale` is a combination weight on the value level" begin
        #=
        Inert on a scalar, applied on a vector, under **every** scalariser. This mirrors
        the model, where `set_risk_expression!` pushes `scale * r_expr` before any
        scalariser runs.
        =#
        base = expected_risk(rv, w, pr)
        for sc in (1.0, 2.0, 50.0)
            scaled = Variance(; settings = RiskMeasureSettings(; scale = sc))
            # Inert on the scalar entry.
            @test expected_risk(scaled, w, pr) == base
            # Applied on the vector entry.
            @test expected_risk([scaled], w, pr) ≈ sc * base
            # And under a non-summing scalariser too.
            @test expected_risk([scaled], w, pr; sca = MaxScalariser()) ≈ sc * base
        end
    end

    @testset "the scalarisers combine as they say they do" begin
        a = expected_risk(rv, w, pr)
        b = expected_risk(rc, w, pr)
        v = [rv, rc]
        @test expected_risk(v, w, pr; sca = SumScalariser()) ≈ a + b
        @test expected_risk(v, w, pr; sca = MaxScalariser()) ≈ max(a, b)
        # `MinScalariser` is admitted at the value level even though no `MeanRisk`
        # optimisation can express it: the value level evaluates numbers, and convexity
        # does not bind numbers.
        @test expected_risk(v, w, pr; sca = MinScalariser()) ≈ min(a, b)
    end
end

@testset "Value-level multiplicity: the live break closes" begin
    using PortfolioOptimisers, Test, StableRNGs, Clarabel, JuMP, LinearAlgebra

    #=
    `JuMPOptimiser.ret` has taken a vector of return terms since #272, and the bundle
    stores the resolved value, so `res.ret` is a vector the moment a user optimises against
    several terms. Documented example sites pass `res.ret` straight into `ExpectedReturn`,
    and that call raised a `MethodError` until `rt` widened.
    =#

    rng = StableRNG(1234321)
    X = randn(rng, 250, 6) ./ 100 .+ 0.0005
    slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    pr = prior(EmpiricalPrior(), X)

    rets = [ArithmeticReturn(),
            ArithmeticReturn(; settings = JuMPReturnsSettings(; scale = 2.0))]
    res = optimise(MeanRisk(; r = Variance(), obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rets)))
    @test isa(res.retcode, OptimisationSuccess)
    @test isa(res.ret, AbstractVector)

    @testset "the documented route out of a result works" begin
        # This is the call 32 documented example sites make.
        er = ExpectedReturn(; rt = res.ret)
        @test isa(expected_risk(er, res.w, pr), Number)

        errr = ExpectedReturnRiskRatio(; rt = res.ret, rk = res.r, rf = 0.0)
        @test isa(expected_risk(errr, res.w, pr), Number)

        # And the documented tuple route.
        rk, rt, sr = expected_risk_ret_ratio(res.r, res.ret, res.w, res.pr; sca = res.sca)
        @test isfinite(rk) && isfinite(rt) && isfinite(sr)
    end

    @testset "a vector `rt` sums at its terms' own scales" begin
        # 1.0 + 2.0 = 3.0 times the lone term, and no scalariser is involved: the return
        # axis sums and never scalarises.
        one_term = expected_risk(ExpectedReturn(; rt = ArithmeticReturn()), res.w, pr)
        many = expected_risk(ExpectedReturn(; rt = rets), res.w, pr)
        @test many ≈ 3 * one_term
    end

    @testset "the result carries the measure and the scalariser it ran under" begin
        # A resolved measure is fitted state, so it belongs on the Result.
        @test isa(res.sca, PortfolioOptimisers.Scalariser)
        @test !isnothing(res.r)
        # Stored resolved: reporting against it reproduces the optimised figure without a
        # refit, and it passes `assert_resolved_slots`.
        @test expected_risk(res.r, res.w, res.pr; sca = res.sca) ≈
              expected_risk(res.r, res.w, res.pr.X; sca = res.sca)
    end

    @testset "the result overload resolves the caller's own measure" begin
        #=
        `expected_risk(r, res, pr)` extracts the prior and then forwards it whole. It used
        to unwrap `pr.X`, so the caller's own measure met the kernel with an unstated slot.
        This overload had no test at all, because every other call site in the suite passes
        `res.w` explicitly.

        The invariant it must satisfy: it agrees with the explicit `res.w` route on the same
        measure, in every slot state. Stated this way the assertion does not depend on
        whether a bare `Covariance()` equals the prior's own estimator.
        =#
        fees = PortfolioOptimisers.extract_fees(res, nothing)
        for r in (Variance(), Variance(; sigma = Covariance()), res.r)
            # Both arms: the prior extracted from the result, and one passed explicitly.
            @test expected_risk(r, res) ≈ expected_risk(r, res.w, res.pr, fees)
            @test expected_risk(r, res, res.pr) ≈ expected_risk(r, res.w, res.pr, fees)
        end
        # The already-resolved measure reproduces the optimised figure through the overload.
        @test expected_risk(res.r, res; sca = res.sca) ≈
              expected_risk(res.r, res.w, res.pr; sca = res.sca)
        # The matrix arm opts out of resolution, so it needs a fully stated measure.
        @test expected_risk(factory(Variance(), res.pr), res, res.pr.X) ≈
              expected_risk(factory(Variance(), res.pr), res.w, res.pr.X, fees)
    end
end

@testset "Value-level multiplicity: the field beats a caller's keyword" begin
    using PortfolioOptimisers, Test, StableRNGs, LinearAlgebra

    #=
    A wrapper is a value, so it must carry its own scalariser: `plot_measures(w, pr; x, y,
    c)` has one shared keyword channel serving three measures, and a wrapper stored in
    `opt.r` is evaluated with no call site to carry a scalariser at all.

    The wrapper therefore splats `kwargs...` **first** and pins its own field last. Without
    that order a caller's `sca` would flow through `expected_ratio` into the inner vector
    call and silently win.
    =#

    rng = StableRNG(55555)
    X = randn(rng, 200, 5) ./ 100 .+ 0.0004
    w = fill(0.2, 5)
    pr = prior(EmpiricalPrior(), X)
    # Unresolved: `expected_ratio` hands the risk axis the prior itself, so an unstated
    # slot takes the prior's field. No `factory` call is needed at the call site.
    rks = [Variance(), ConditionalValueatRisk()]

    errr_sum = ExpectedReturnRiskRatio(; rk = rks, sca = SumScalariser())
    errr_max = ExpectedReturnRiskRatio(; rk = rks, sca = MaxScalariser())

    v_sum = expected_risk(errr_sum, w, pr)
    v_max = expected_risk(errr_max, w, pr)
    @test v_sum != v_max

    # A caller's keyword cannot change either answer.
    @test expected_risk(errr_sum, w, pr; sca = MaxScalariser()) == v_sum
    @test expected_risk(errr_max, w, pr; sca = SumScalariser()) == v_max

    @testset "`NonOptimisationRiskRatio` carries one scalariser per axis" begin
        # It is the only type in the family with two independent risk vectors. This testset
        # evaluates on the matrix, so it resolves the measures itself.
        rrks = factory(rks, pr)
        r = NonOptimisationRiskRatio(; r1 = rrks, sca1 = SumScalariser(), r2 = rrks,
                                     sca2 = MaxScalariser())
        num = expected_risk(rrks, w, pr.X; sca = SumScalariser())
        den = expected_risk(rrks, w, pr.X; sca = MaxScalariser())
        @test expected_risk(r, w, pr.X) ≈ num / den
        # On a precomputed series the functor divides the two scalarised risks of the series,
        # so it agrees with the matrix route at constant weights.
        rs2 = [ConditionalValueatRisk(), MaximumDrawdown()]
        r2 = NonOptimisationRiskRatio(; r1 = rs2, sca1 = SumScalariser(), r2 = rs2,
                                      sca2 = MaxScalariser())
        @test r2(pr.X * w) ≈ expected_risk(r2, w, pr.X)
    end

    @testset "the ratio family routes its risk axis through the prior" begin
        #=
        `expected_ratio` used to unwrap `pr.X` for the risk axis while handing the return
        axis `pr`, two lines apart. So an unstated slot reached the kernel as `nothing`, and
        a Deferred Quantity was refused for want of a prior that was in the argument list.
        The `VecVecNum` method resolved first, so the vector twin answered where the
        singular one threw.
        =#
        unstated = ExpectedReturnRiskRatio(; rk = Variance())
        deferred = ExpectedReturnRiskRatio(; rk = Variance(; sigma = Covariance()))
        stated = ExpectedReturnRiskRatio(; rk = factory(Variance(), pr))

        ref = expected_risk(stated, w, pr)
        # All three slot states agree: the prior supplies the same matrix each way.
        @test expected_risk(unstated, w, pr) ≈ ref
        @test expected_risk(deferred, w, pr) ≈ ref
        # The singular method and the `VecVecNum` twin agree on every state.
        for m in (unstated, deferred, stated)
            @test expected_risk(m, [w], pr)[1] ≈ expected_risk(m, w, pr)
        end
        # The four public ratio entry points take an unstated slot directly.
        @test expected_ratio(Variance(), ArithmeticReturn(), w, pr) ≈ ref
        @test expected_ratio(Variance(; sigma = Covariance()), ArithmeticReturn(), w, pr) ≈
              ref
        rk, rt, rr = expected_risk_ret_ratio(Variance(), ArithmeticReturn(), w, pr)
        @test rr ≈ ref
        @test rk ≈ expected_risk(factory(Variance(), pr), w, pr.X)
        @test rt ≈ expected_return(ArithmeticReturn(), w, pr)
        @test expected_sric(Variance(), ArithmeticReturn(), w, pr) < ref
        srk, srt, ssr = expected_risk_ret_sric(Variance(), ArithmeticReturn(), w, pr)
        @test (srk, srt) == (rk, rt)
        @test ssr < rr
    end
end

@testset "Value-level multiplicity: polarity is refused on a mixed vector" begin
    using PortfolioOptimisers, Test

    #=
    `[Variance(), ExpectedReturn()]` scalarises to a number with no defined orientation.
    Neither `all` nor `any` is right, and both are wrong in silence: under `all` the vector
    answers `false` and `RankRule(; best = 20)` silently keeps the wrong twenty assets.
    Refusing by name is the only answer that cannot be wrong.
    =#

    bib = PortfolioOptimisers.bigger_is_better

    @test bib([Variance(), ConditionalValueatRisk()]) == false
    @test bib([ExpectedReturn(), MeanReturn()]) == true
    @test_throws ArgumentError bib([Variance(), ExpectedReturn()])
    @test_throws ArgumentError bib([ExpectedReturn(), Variance()])

    @testset "the refusal names the offending measures" begin
        msg = try
            bib([Variance(), ExpectedReturn()])
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("Variance", msg)
        @test occursin("ExpectedReturn", msg)
    end
end

@testset "Value-level multiplicity: risk contribution decomposes the aggregate" begin
    using PortfolioOptimisers, Test, StableRNGs, LinearAlgebra

    #=
    `adjust_risk_contribution` is a homogeneity correction, so a mixed vector has no single
    degree and adjusting the aggregate is impossible. Adjusting each element **before** the
    scalariser restores Euler's identity exactly, and that is a positive argument for the
    widening rather than a caveat on it.
    =#

    rng = StableRNG(24680)
    X = randn(rng, 250, 6) ./ 100 .+ 0.0004
    w = fill(inv(6), 6)
    pr = prior(EmpiricalPrior(), X)

    rv = Variance()                    # degree 2
    rc = ConditionalValueatRisk()      # degree 1

    @testset "the singular case is unchanged" begin
        @test sum(risk_contribution(rv, w, pr)) ≈ expected_risk(rv, w, pr) rtol=1e-5
        @test risk_contribution([rv], w, pr) ≈ risk_contribution(rv, w, pr)
    end

    @testset "Euler's identity survives a mixture of homogeneity degrees" begin
        mixed = [rv, rc]
        @test sum(risk_contribution(mixed, w, pr)) ≈ expected_risk(mixed, w, pr) rtol=1e-5
    end

    @testset "and it survives the combination weights too" begin
        mixed = [Variance(; settings = RiskMeasureSettings(; scale = 3.0)), rc]
        @test sum(risk_contribution(mixed, w, pr)) ≈ expected_risk(mixed, w, pr) rtol=1e-5
    end

    @testset "a maximum or a minimum decomposes the element it picks" begin
        #=
        The scaled variance sits above the CVaR, and below it once it is divided by its
        degree. The scalariser used to pick its element after the homogeneity correction, so
        the maximum decomposed the CVaR and the minimum decomposed the variance, and neither
        sum matched the figure `expected_risk` reports.
        =#
        v = expected_risk(rv, w, pr)
        c = expected_risk(rc, w, pr)
        s = 1.3 * c / v
        mixed = [Variance(; settings = RiskMeasureSettings(; scale = s)), rc]
        for sca in (MaxScalariser(), MinScalariser())
            @test sum(risk_contribution(mixed, w, pr; sca = sca)) ≈
                  expected_risk(mixed, w, pr; sca = sca) rtol=1e-5
        end
        # The log-sum-exp is not homogeneous, so the contributions sum to the
        # softmax-weighted mean of the scaled risks and not to the aggregate.
        gamma = 200.0
        vals = [s * v, c]
        p = exp.(gamma .* (vals .- maximum(vals)))
        p ./= sum(p)
        lse = LogSumExpScalariser(; gamma = gamma)
        @test sum(risk_contribution(mixed, w, pr; sca = lse)) ≈ dot(p, vals) rtol=1e-5
        @test dot(p, vals) < expected_risk(mixed, w, pr; sca = lse)
    end
end

@testset "Value-level multiplicity: `measure_label` never says \"Vector\"" begin
    using PortfolioOptimisers, Test

    #=
    `string(nameof(typeof(rs)))` on a vector evaluates to `"Vector"` — a wrong axis label,
    silently, with no error. Seven plot sites spelled that expression inline.
    =#
    ml = PortfolioOptimisers.measure_label
    @test ml(Variance()) == "Variance"
    @test ml([Variance()]) == "Variance"
    @test ml([Variance(), ConditionalValueatRisk()]) == "Variance + ConditionalValueatRisk"
    @test !occursin("Vector", ml([Variance(), ConditionalValueatRisk()]))
end

@testset "Value-level multiplicity: the hierarchical split forwards every property" begin
    using PortfolioOptimisers, Test, StableRNGs, Clustering, LinearAlgebra

    #=
    `HierarchicalResult` split into a shared core embedded as `hr` and one leaf per
    estimator. Because `getproperty` forwards through the core, every reader of `res.w`,
    `res.pr`, `res.clr`, `res.wb`, `res.fees` and `res.retcode` stays source-compatible.
    Only code that **names** the type, and positional construction, may break.
    =#

    rng = StableRNG(13579)
    X = randn(rng, 250, 8) ./ 100 .+ 0.0004
    pr = prior(EmpiricalPrior(), X)
    opt = HierarchicalOptimiser(; pe = pr)

    hrp = optimise(HierarchicalRiskParity(; r = Variance(), opt = opt))
    herc = optimise(HierarchicalEqualRiskContribution(; ri = Variance(), opt = opt))

    @test isa(hrp, PortfolioOptimisers.HierarchicalRiskParityResult)
    @test isa(herc, PortfolioOptimisers.HierarchicalEqualRiskContributionResult)

    @testset "the core's properties still read off the leaf" begin
        for res in (hrp, herc)
            for p in (:w, :pr, :clr, :wb, :fees, :retcode)
                @test hasproperty(res, p)
            end
            @test isa(res.retcode, OptimisationSuccess)
            @test sum(res.w) ≈ 1
            # `hr` is the embedded core, and it carries no `fb` — ADR 0011 keeps `fb` last
            # on each concrete result instead.
            @test isa(res.hr, PortfolioOptimisers.HierarchicalResult)
            @test !hasfield(typeof(res.hr), :fb)
            @test hasfield(typeof(res), :fb)
        end
    end

    @testset "each leaf carries its own estimator's arity" begin
        # HRP: one measure, one scalariser.
        @test !isnothing(hrp.r)
        @test isa(hrp.sca, PortfolioOptimisers.Scalariser)
        # HERC: two measures, two scalarisers.
        @test !isnothing(herc.ri)
        @test !isnothing(herc.ro)
        @test isa(herc.scai, PortfolioOptimisers.Scalariser)
        @test isa(herc.scao, PortfolioOptimisers.Scalariser)
    end

    @testset "the measure on the result reproduces a reported figure" begin
        # Stored resolved, so it needs no refit and passes `assert_resolved_slots`.
        @test isa(expected_risk(hrp.r, hrp.w, hrp.pr; sca = hrp.sca), Number)
    end
end

@testset "Value-level multiplicity: a vector measure does not zip against the results" begin
    using PortfolioOptimisers, Test, StableRNGs, Clarabel, StatsPlots, GraphRecipes,
          LinearAlgebra

    #=
    `plot_measures(res_vec, pr; …)` **broadcasts** over the results. A single measure is not
    iterable, so it broadcasts as a scalar and the call is correct. A **vector** of measures
    is iterable, so without `Ref` it would zip elementwise against `pr`, `w` and `fees` — a
    `DimensionMismatch` when the lengths differ and a silent **wrong answer** when they
    happen to match.

    Three measures against three results is exactly the case that would pass silently, so it
    is the case this test pins.
    =#

    ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
    is_plot(x) = x isa Plots.Plot || x isa Plots.AbstractLayout

    rng = StableRNG(97531)
    X = randn(rng, 200, 5) ./ 100 .+ 0.0004
    slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    pr = prior(EmpiricalPrior(), X)

    res_vec = [optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv)))
               for r in (Variance(), StandardDeviation(), ConditionalValueatRisk())]
    @test length(res_vec) == 3

    # Three measures, three results: the lengths match, so a zip would not raise.
    xs = [Variance(), StandardDeviation(), ConditionalValueatRisk()]
    @test is_plot(plot_measures(res_vec, pr; x = xs, y = ExpectedReturn()))

    # The figure must report the **scalarised** measure at every result, not measure `i` at
    # result `i`. Check the numbers the plot is built from rather than the plot object.
    ws = getproperty.(res_vec, :w)
    scalarised = [expected_risk(xs, wi, pr, nothing) for wi in ws]
    zipped = [expected_risk(xs[i], ws[i], pr, nothing) for i in eachindex(ws)]
    @test scalarised != zipped
    @test all(scalarised .> zipped)  # a sum of three positives exceeds any one of them

    # A single measure keeps working unchanged.
    @test is_plot(plot_measures(res_vec, pr; x = Variance(), y = ExpectedReturn()))
end

@testset "Value-level: an empty slot the functor reads is refused by name (#1079)" begin
    using PortfolioOptimisers, Test, StableRNGs, LinearAlgebra, Statistics, Dates
    PO = PortfolioOptimisers
    rng = StableRNG(1079)
    X = randn(rng, 100, 5)
    w = fill(0.2, 5)
    pr = prior(EmpiricalPrior(), X)
    hpr = prior(HighOrderPriorEstimator(), X)
    rd = ReturnsResult(; nx = string.("A", 1:5), X = X, ts = Date(2020, 1, 1) .+ Day.(0:99))
    sigma = cov(PortfolioOptimisersCovariance(), X)
    message(f) =
        try
            f()
            ""
        catch e
            sprint(showerror, e)
        end

    @testset "the matrix route names the measure, the slot and both ways out" begin
        # `Variance()` is built with `sigma` at `nothing`, and the functor is
        # `dot(w, r.sigma, w)`. Before #1079 this was a `MethodError` from inside
        # `LinearAlgebra` naming neither.
        @test_throws IsNothingError expected_risk(Variance(), w, X)
        @test_throws IsNothingError risk_contribution(Variance(), w, X)
        @test_throws IsNothingError expected_risk(StandardDeviation(), w, X)
        msg = message(() -> expected_risk(Variance(), w, X))
        @test occursin("`Variance.sigma` is `nothing`", msg)
        @test occursin("prior(EmpiricalPrior(), X)", msg)
        @test occursin("Variance(; sigma = ", msg)
        # The `ReturnsResult` route carries no moments either, so it refuses on the same
        # terms.
        @test_throws IsNothingError expected_risk(Variance(), w, rd)
        @test_throws IsNothingError risk_contribution(Variance(), w, rd)
        # A vector is refused element by element, so the third element is named.
        @test_throws IsNothingError expected_risk([ConditionalValueatRisk(), Variance()], w,
                                                  X)
        @test occursin("`Variance.sigma`",
                       message(() -> expected_risk([ConditionalValueatRisk(), Variance()],
                                                   w, X)))
    end

    @testset "both ways out work, and agree" begin
        # The prior route fills the slot through `factory`, and the stated slot needs no
        # filling. The two agree exactly when the prior is empirical, and the matrix route
        # never fills the slot itself, so they are free to disagree when it is not.
        @test expected_risk(Variance(), w, pr) ==
              expected_risk(Variance(; sigma = sigma), w, X)
        @test risk_contribution(Variance(), w, pr) ==
              risk_contribution(Variance(; sigma = sigma), w, X)
        @test expected_risk(StandardDeviation(), w, pr) ==
              expected_risk(StandardDeviation(; sigma = sigma), w, X)
    end

    @testset "every declared slot refuses, and none that has a sample reading" begin
        # `mu` at `nothing` means the sample mean on a value-level route, so the moment
        # measures pass and evaluate.
        for r in (LowOrderMoment(), HighOrderMoment(), MedianAbsoluteDeviation(),
                  ThirdCentralMoment(), ThirdCentralMoment(; ve = SimpleVariance()))
            @test isnothing(PO.assert_resolved_slots(r))
            @test isa(expected_risk(r, w, X), Number)
        end
        # `Kurtosis` and `Skewness` read their comoment matrix as it stands, so a bare
        # returns matrix cannot fill it (#1846).
        for (r, name) in ((Kurtosis(), "`Kurtosis.kt`"), (Skewness(), "`Skewness.sk`"))
            @test_throws IsNothingError expected_risk(r, w, X)
            @test occursin(name, message(() -> expected_risk(r, w, X)))
            @test isa(expected_risk(r, w, hpr), Number)
        end
        @test_throws IsNothingError expected_risk(UncertaintySetVariance(), w, X)
        @test_throws IsNothingError expected_risk(NegativeSkewness(), w, X)
        @test occursin("`NegativeSkewness.sk`",
                       message(() -> expected_risk(NegativeSkewness(), w, X)))
        # The parametric formulation reads `mu` and `sigma` off `alg`, and the walk reaches
        # it through the measure's `alg` slot.
        rv = ValueatRisk(; alg = DistributionValueatRisk())
        @test_throws IsNothingError expected_risk(rv, w, X)
        @test occursin("`DistributionValueatRisk.mu`",
                       message(() -> expected_risk(rv, w, X)))
        @test isa(expected_risk(rv, w, pr), Number)
        # A container reaches each child through its slot, so the child's own name is what
        # the refusal reads. `vr` is refused first, and with `vr` stated the refusal names
        # `sk`, which the child `Skewness` reads.
        vsk = VarianceSkewKurtosis()
        @test_throws IsNothingError expected_risk(vsk, w, X)
        @test occursin("`Variance.sigma`", message(() -> expected_risk(vsk, w, X)))
        vsk2 = VarianceSkewKurtosis(; vr = Variance(; sigma = sigma))
        @test_throws IsNothingError expected_risk(vsk2, w, X)
        @test occursin("`Skewness.sk`", message(() -> expected_risk(vsk2, w, X)))
        # A Deferred Quantity in a child is named as a Deferred Quantity, not as an empty
        # slot.
        vsk3 = VarianceSkewKurtosis(; vr = Variance(; sigma = sigma),
                                    sk = Skewness(; sk = HighOrderPriorEstimator()))
        @test_throws ArgumentError expected_risk(vsk3, w, X)
        @test occursin("holds a Deferred Quantity",
                       message(() -> expected_risk(vsk3, w, X)))
        # A low-order prior carries no co-moment tensor, so the prior route leaves the
        # child empty and the same refusal names it; the high-order prior fills it.
        @test_throws IsNothingError expected_risk(vsk, w, pr)
        @test occursin("`Skewness.sk`", message(() -> expected_risk(vsk, w, pr)))
        @test isa(expected_risk(vsk, w, hpr), Number)
        @test isa(expected_risk(NegativeSkewness(), w, hpr), Number)
        # A child measure held by a tracking measure is reached through its slot.
        rt = RiskTrackingRiskMeasure(; r = Variance(), tr = WeightsTracking(; w = w))
        @test_throws IsNothingError expected_risk(rt, w, X)
        @test isa(expected_risk(rt, w, pr), Number)
    end
end

@testset "Value-level multiplicity: risk_gradient is the derivative of the reported figure" begin
    using PortfolioOptimisers, Test, StableRNGs, LinearAlgebra, Statistics

    #=
    `risk_gradient` differentiates the figure `expected_risk` reports, with no homogeneity
    correction, so it is twice the marginal contribution on a `Variance` and equal to it
    on a degree-one measure. The three closed forms are checked against the shared
    finite-difference kernel at `delta = 1e-6`, and the vector route against the chain rule
    through each scalariser written by hand.
    =#

    po = PortfolioOptimisers
    rng = StableRNG(1182)
    X = randn(rng, 120, 5) ./ 100 .+ 0.0004
    w = [0.3, 0.25, 0.2, 0.15, 0.1]
    pr = prior(EmpiricalPrior(), X)
    fd(r; kw...) = po.finite_difference_gradient(v -> expected_risk(r, v, pr; kw...), w,
                                                 1e-6)

    @testset "the closed forms against the finite difference" begin
        @test risk_gradient(Variance(), w, pr) == 2 .* (pr.sigma * w)
        @test isapprox(risk_gradient(Variance(), w, pr), fd(Variance()); atol = 1e-9)
        @test isapprox(risk_gradient(StandardDeviation(), w, pr),
                       pr.sigma * w ./ sqrt(dot(w, pr.sigma, w)); atol = 1e-15)
        @test isapprox(risk_gradient(StandardDeviation(), w, pr), fd(StandardDeviation());
                       atol = 1e-8)
        @test isapprox(risk_gradient(MeanReturn(), w, pr), vec(mean(X; dims = 1));
                       atol = 1e-15)
        @test isapprox(risk_gradient(MeanReturn(), w, pr), fd(MeanReturn()); atol = 1e-9)
        # The log flag: the row's weight is divided by `1 + xᵀw`, the log-wealth gradient
        # averaged over the rows.
        lg = vec(transpose(X) * (inv(size(X, 1)) ./ (1 .+ X * w)))
        @test isapprox(risk_gradient(MeanReturn(; flag = true), w, pr), lg; atol = 1e-15)
        @test isapprox(risk_gradient(MeanReturn(; flag = true), w, pr),
                       fd(MeanReturn(; flag = true)); atol = 1e-9)
        # Observation weights normalise to one.
        ow = StatsBase.pweights(collect(1.0:120.0))
        @test isapprox(risk_gradient(MeanReturn(; w = ow), w, pr),
                       vec(transpose(X) * (ow ./ sum(ow))); atol = 1e-15)
        # A measure with no closed form is the finite difference of its own figure.
        @test risk_gradient(ConditionalValueatRisk(), w, pr) == fd(ConditionalValueatRisk())
        # The contribution is the Euler-adjusted figure: half the gradient on a variance,
        # the gradient itself on a degree-one measure.
        @test isapprox(risk_contribution(Variance(), w, pr; marginal = true),
                       risk_gradient(Variance(), w, pr) ./ 2; atol = 1e-9)
        @test isapprox(risk_contribution(ConditionalValueatRisk(), w, pr; marginal = true),
                       risk_gradient(ConditionalValueatRisk(), w, pr); atol = 1e-12)
    end

    @testset "the matrix route and its refusals" begin
        sigma = cov(X)
        @test risk_gradient(Variance(; sigma = sigma), w, X) == 2 .* (sigma * w)
        @test risk_gradient(MeanReturn(), w, X) == risk_gradient(MeanReturn(), w, pr)
        # At a zero variance the square root has no gradient, so the closed form falls
        # to the chord, which is the zero vector on a zero covariance.
        @test risk_gradient(StandardDeviation(; sigma = zeros(5, 5)), w, X) == zeros(5)
        # A `Variance()` on a bare matrix has no covariance to read, as `expected_risk`
        # refuses it.
        @test_throws IsNothingError risk_gradient(Variance(), w, X)
        # A stated matrix on the measure survives the prior route.
        @test risk_gradient(Variance(; sigma = sigma), w, pr) == 2 .* (sigma * w)
        # A fee makes the mean return non-linear, so it takes the difference.
        fees = Fees(; l = 0.001)
        @test isapprox(risk_gradient(MeanReturn(), w, X, fees),
                       po.finite_difference_gradient(v -> expected_risk(MeanReturn(), v, X,
                                                                        fees), w, 1e-6);
                       atol = 1e-15)
    end

    @testset "the chain rule through each scalariser" begin
        rv = Variance(; settings = RiskMeasureSettings(; scale = 2.0))
        rm = MeanReturn(; settings = HierarchicalRiskMeasureSettings(; scale = -1.0))
        rc = ConditionalValueatRisk()
        gv, gm, gc = risk_gradient(rv, w, pr), risk_gradient(rm, w, pr),
                     risk_gradient(rc, w, pr)
        # Sum: each at its scale, so the mean–variance utility is `-μ + 2 · 2Σw`.
        @test isapprox(risk_gradient([rm, rv], w, pr), -gm .+ 2 .* gv; atol = 1e-15)
        # Max and min: the winning scaled element's own gradient.
        vals = [2 * expected_risk(rv, w, pr), -expected_risk(rm, w, pr),
                expected_risk(rc, w, pr)]
        gs = [2 .* gv, -gm, gc]
        @test risk_gradient([rv, rm, rc], w, pr; sca = MaxScalariser()) == gs[argmax(vals)]
        @test risk_gradient([rv, rm, rc], w, pr; sca = MinScalariser()) == gs[argmin(vals)]
        # At a tie the earliest wins, which is the kink where the subgradient is a set.
        @test risk_gradient([Variance(), Variance()], w, pr; sca = MaxScalariser()) ==
              risk_gradient(Variance(), w, pr)
        # Log-sum-exp: the softmax of the scaled values.
        gamma = 50.0
        p = exp.(gamma .* vals)
        p ./= sum(p)
        @test isapprox(risk_gradient([rv, rm, rc], w, pr;
                                     sca = LogSumExpScalariser(; gamma = gamma)),
                       sum(p .* gs); atol = 1e-12)
        # Against the finite difference of the aggregate itself, for every scalariser.
        for sca in (SumScalariser(), MaxScalariser(), MinScalariser(),
                    LogSumExpScalariser(; gamma = gamma))
            @test isapprox(risk_gradient([rv, rm, rc], w, pr; sca = sca),
                           fd([rv, rm, rc]; sca = sca); atol = 1e-8)
        end
    end

    @testset "the prior route reduces to the investable mask" begin
        Xg = copy(X)
        Xg[:, 3] .= NaN
        prg = prior(EmpiricalPrior(), Xg)
        wg = [0.4, 0.3, 0.0, 0.2, 0.1]
        g = risk_gradient(Variance(), wg, prg)
        @test length(g) == 5 && g[3] == 0
        @test isapprox(g[[1, 2, 4, 5]],
                       2 .* (prg.sigma[[1, 2, 4, 5], [1, 2, 4, 5]] * wg[[1, 2, 4, 5]]);
                       atol = 1e-15)
    end
end

@testset "A measure whose value is a solve differentiates its Worst-Case Distribution (#1835)" begin
    using PortfolioOptimisers, Test, StableRNGs, LinearAlgebra, StatsBase, Clarabel

    #=
    The finite difference of a measure whose value is a solve differenced two solves. The
    change that the step `1e-6` makes is about `1e-8`, and the solver stops at a gap of about
    `1e-8`, so the contributions missed the risk: by 0.005 % on the EVaR, 5.4 % on the RLVaR
    and 2.2 % on the EDaR, on the data below. Each measure is the largest expected loss over a
    set of distributions, so its gradient is the gradient of the expected loss under the
    Worst-Case Distribution of one solve, which has no solver noise.
    =#

    po = PortfolioOptimisers
    X = randn(StableRNG(987456), 200, 5) * 0.01 .+ 0.0005
    pr = prior(EmpiricalPrior(), X)
    w = [0.3, 0.25, 0.2, 0.15, 0.1]
    x = X * w
    slv = Solver(; solver = Clarabel.Optimizer, settings = Dict("verbose" => false))
    # Even observations have weight zero, so their rows leave every programme.
    ow = StatsBase.Weights([iseven(t) ? 0.0 : 1.0 + t / 200 for t in 1:200])
    ratio(r; kw...) = sum(risk_contribution(r, w, pr; kw...)) /
                      expected_risk(r, w, pr; kw...)

    @testset "every homogeneous measure of the three families sums to its risk" begin
        # `alpha = 0.2` and `p = 1.5` put the power norm measures on their programme. At the
        # defaults they are the largest loss on 200 observations.
        pn = (; alpha = 0.2, p = 1.5)
        rs = [EntropicValueatRisk(), EntropicValueatRiskRange(), EntropicDrawdownatRisk(),
              RelativisticValueatRisk(), RelativisticValueatRiskRange(),
              RelativisticDrawdownatRisk(), PowerNormValueatRisk(; pn...),
              PowerNormValueatRiskRange(; alpha = 0.2, beta = 0.2, pa = 1.5, pb = 1.5),
              PowerNormDrawdownatRisk(; pn...), EntropicValueatRisk(; w = ow),
              RelativisticValueatRisk(; w = ow), PowerNormValueatRisk(; w = ow, pn...),
              EntropicValueatRisk(; alpha = 0), PowerNormValueatRisk()]
        for r in rs
            @test isapprox(ratio(factory(r, pr, slv)), 1; rtol = 1e-6)
        end
        # A vector of them decomposes its aggregate, and a measure with no solve sits beside
        # them.
        mixed = [factory(EntropicValueatRisk(), pr, slv), Variance(),
                 factory(RelativisticValueatRisk(), pr, slv)]
        @test isapprox(ratio(mixed), 1; rtol = 1e-6)
        @test isapprox(ratio(mixed; sca = MaxScalariser()), 1; rtol = 1e-6)
        # A proportional fee keeps the measure homogeneous.
        fees = Fees(; l = fill(0.0004, 5))
        r = factory(EntropicValueatRisk(), pr, slv)
        @test isapprox(sum(risk_contribution(r, w, pr, fees)) /
                       expected_risk(r, w, pr, fees), 1; rtol = 1e-6)
    end

    @testset "the gradient is that of the measure" begin
        r = factory(EntropicValueatRisk(), pr, slv)
        q = po.ERM_worst_case(x, slv, 0.05)
        @test isapprox(risk_gradient(r, w, pr), -transpose(X) * q; rtol = 1e-8)
        # A compounded drawdown is not homogeneous, so its contributions do not sum to its
        # risk. Its gradient is the expected loss under the Worst-Case Distribution of one
        # solve, through the Jacobian `J` of the drawdown series. `J` is in closed form, so
        # the check solves at `w` alone. A finite difference of the measure solved at shifted
        # weights, and a solve that fails returns `NaN` on some hosts (#1916).
        J = similar(X)
        cp = peak = one(eltype(x))
        dcp = zeros(size(X, 2))
        dpeak = zeros(size(X, 2))
        for t in axes(X, 1)
            cp *= 1 + x[t]
            dcp .= dcp .* (1 + x[t]) .+ (cp / (1 + x[t])) .* view(X, t, :)
            if cp > peak
                peak = cp
                dpeak .= dcp
            end
            J[t, :] .= dcp ./ peak .- (cp / peak^2) .* dpeak
        end
        dd = po.relative_drawdown_vec(x)
        for (r, q) in
            ((RelativeEntropicDrawdownatRisk(), po.ERM_worst_case(dd, slv, 0.05)),
             (RelativeRelativisticDrawdownatRisk(), po.RRM_worst_case(dd, slv, 0.05, 0.3)),
             (RelativePowerNormDrawdownatRisk(; alpha = 0.2, p = 1.5),
              po.PRM_worst_case(dd, slv, 0.2, 1.5)))
            @test isapprox(risk_gradient(factory(r, pr, slv), w, pr), -transpose(J) * q;
                           rtol = 1e-6)
        end
    end

    @testset "a composite carries the distributions of the measures it wraps" begin
        rl = factory(RelativisticValueatRisk(), pr, slv)
        ev = factory(EntropicValueatRisk(), pr, slv)
        sd = factory(StandardDeviation(), pr, slv)
        quot(a, b) = (risk_gradient(a, w, pr) .* expected_risk(b, w, pr) .-
                      expected_risk(a, w, pr) .* risk_gradient(b, w, pr)) ./
                     expected_risk(b, w, pr)^2
        @test isapprox(risk_gradient(RiskRatio(; r1 = rl, r2 = StandardDeviation()), w, pr),
                       quot(rl, sd); rtol = 1e-6)
        @test isapprox(risk_gradient(MeanReturnRiskRatio(; rk = rl), w, pr),
                       quot(MeanReturn(), rl); rtol = 1e-6)
        top = expected_risk(rl, w, pr) >= expected_risk(ev, w, pr) ? rl : ev
        nr = NonOptimisationRiskRatio(; r1 = [rl, ev], sca1 = MaxScalariser(),
                                      r2 = StandardDeviation())
        @test isapprox(risk_gradient(nr, w, pr), quot(top, sd); rtol = 1e-6)
        # A ratio of measures with no solve keeps the finite difference of its own figure.
        r = RiskRatio(; r1 = Variance(), r2 = ConditionalValueatRisk())
        @test risk_gradient(r, w, pr) ==
              po.finite_difference_gradient(v -> expected_risk(r, v, pr), w, 1e-6)
        # The independent tracking measure is the risk of the active weights, so Euler's
        # theorem holds on them. The finite difference of the measure itself meets failed
        # solves here, and returns `NaN`.
        wb = fill(0.2, 5)
        fees = Fees(; l = fill(0.0004, 5))
        t = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = rl)
        @test isapprox(dot(w - wb, risk_gradient(t, w, pr)), expected_risk(t, w, pr);
                       rtol = 1e-6)
        @test all(isfinite, risk_gradient(t, w, pr, fees))
        # An inner measure with no solve keeps the finite difference of the tracking measure.
        t = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb),
                                    r = factory(ConditionalValueatRisk(), pr))
        @test risk_gradient(t, w, pr, fees) ==
              po.finite_difference_gradient(v -> expected_risk(t, v, pr, fees), w, 1e-6)
        t = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = sd)
        @test risk_gradient(t, w, pr, fees) ==
              po.finite_difference_gradient(v -> expected_risk(t, v, pr, fees), w, 1e-6)
        t = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = rl,
                                    alg = DependentVariableTracking())
        for f in (nothing, fees)
            gfd = po.finite_difference_gradient(v -> expected_risk(t, v, pr, f), w, 1e-3)
            @test isapprox(risk_gradient(t, w, pr, f), gfd; rtol = 1e-3)
        end
    end

    @testset "the distribution of each programme" begin
        for (q, v) in ((po.ERM_worst_case(x, slv, 0.05), po.ERM(x, slv, 0.05)),
                       (po.RRM_worst_case(x, slv, 0.05, 0.3), po.RRM(x, slv, 0.05, 0.3)),
                       (po.PRM_worst_case(x, slv, 0.2, 1.5), po.PRM(x, slv, 0.2, 1.5)))
            @test isapprox(sum(q), 1; rtol = 1e-6)
            @test isapprox(-dot(q, x), v; rtol = 1e-6)
        end
        # An observation of weight zero has no probability.
        for (q, v) in ((po.ERM_worst_case(x, slv, 0.05, ow), po.ERM(x, slv, 0.05, ow)),
                       (po.RRM_worst_case(x, slv, 0.05, 0.3, ow), po.RRM(x, slv, 0.05, 0.3, ow)),
                       (po.PRM_worst_case(x, slv, 0.2, 1.5, ow), po.PRM(x, slv, 0.2, 1.5, ow)))
            @test length(q) == 200 && all(iszero, q[2:2:end])
            @test isapprox(-dot(q, x), v; rtol = 1e-6)
        end
        # Without a programme, the mass sits on the largest loss of positive weight, and a
        # tie shares it by weight.
        @test po.ERM_worst_case(x, slv, 0, ow) == po.worst_loss_distribution(x, ow)
        @test po.PRM_worst_case(x, slv) == po.worst_loss_distribution(x, nothing)
        @test po.worst_loss_distribution([-0.1, 0.2, -0.1, 0.0], nothing) ==
              [0.5, 0, 0.5, 0]
        @test po.worst_loss_distribution([-0.1, 0.2, -0.1, -0.1], [1, 1, 3, 0]) ==
              [0.25, 0, 0.75, 0]
        # A failed solve gives no distribution, so the contribution falls back to the
        # finite difference of the measure itself.
        bad = Solver(; solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false, "max_iter" => 1))
        @test isnothing(po.ERM_worst_case(x, bad, 0.05))
        @test isnothing(po.RRM_worst_case(x, bad, 0.05, 0.3))
        @test isnothing(po.PRM_worst_case(x, bad, 0.2, 1.5))
        @test isnothing(po.worst_case_distribution(EntropicValueatRiskRange(; slv = bad),
                                                   x))
        @test isnothing(po.worst_case_distribution(Variance(), x))
        @test isnothing(po.worst_case_distribution(Variance(; sigma = cov(X)), w, X,
                                                   nothing))
        # The route on the weights checks the slots as `expected_risk` does.
        @test_throws IsNothingError po.worst_case_distribution(Variance(), w, X, nothing)
        @test_throws IsEmptyError po.ERM_worst_case(x, Solver[])
        @test_throws IsEmptyError po.RRM_worst_case(x, Solver[])
        @test_throws IsEmptyError po.PRM_worst_case(x, Solver[])
    end
end

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
    # decisive check: the cone is `[iota, k, w_p] in PowerCone(beta)`, with `k = 1` here, which
    # bounds `iota` below by `w_p ^ (1 / beta)`, so a reader who takes `beta` for the power law
    # is wrong.
    bmi = BudgetMarketImpact(; bgt = 1.0, w = w0, vp = 0.1, vn = 0.1, up = 0.5, un = 0.5,
                             beta = 0.5)
    w = optimise(MeanRisk(; obj = MaximumReturn(),
                          opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bmi))).w
    dw = abs.(w .- w0)
    @test isapprox(sum(w) + 0.1 * sum(dw .^ 2), 1.0; atol = 1e-7)
    @test !isapprox(sum(w) + 0.1 * sum(dw .^ 0.5), 1.0; atol = 1e-7)
    # The constraint scale scales all three entries of the cone. The middle entry `k` without
    # `sc` gave `iota^beta * k^(1 - beta) >= sc^(1 - beta) * w_p`: at `sc = 100` the identity
    # read 0.99999995, and the weights stayed at `w0`.
    ws = optimise(MeanRisk(; obj = MaximumReturn(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bmi, sc = 100))).w
    @test isapprox(sum(ws) + 0.1 * sum(abs.(ws .- w0) .^ 2), 1.0; atol = 1e-7)
    @test isapprox(ws, w; rtol = 1e-4)
end

function maximum_ratio_weights(bgt)
    return map((2.5e-4, 1e-3, 1e-2, 1.0)) do ohf
        return optimise(MeanRisk(; obj = MaximumRatio(; ohf = ohf),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = bgt)))
    end
end
@testset "BudgetCosts and BudgetMarketImpact are homogeneous under MaximumRatio" begin
    #=
    `MaximumRatio` solves for `y = k w` and needs every row homogeneous in `(y, k)`. The cost
    rows held `w - w0`, `wp <= up` and the power-cone constant `1` without `k`, so the answer
    moved with `ohf`: `BudgetCosts` gave `sum(w) + cost` = 0.877 at `ohf = 2.5e-4` and 0.970
    at `ohf = 1e-3`, and both types failed to solve from `ohf = 1e-2` up. The start
    portfolio, the caps and the cone constant now scale with `k`, so every `ohf` solves to
    one ratio. The optimum is flat, and weights 1e-3 apart reach one ratio to 1e-5, so the
    test reads the ratio and not the weights.
    =#
    sr(w) = dot(pr.mu, w) / sqrt(dot(w, pr.sigma, w))
    rbc = maximum_ratio_weights(BudgetCosts(; bgt = 1.0, w = w0, vp = 0.01, vn = 0.01,
                                            up = 0.5, un = 0.5))
    @test all(r -> isa(r.retcode, PortfolioOptimisers.OptimisationSuccess), rbc)
    @test all(r -> isapprox(sr(r.w), sr(rbc[1].w); rtol = 1e-5), rbc)
    # The market-impact charge also leaves the return, and its budget identity is tight.
    rmi = maximum_ratio_weights(BudgetMarketImpact(; bgt = 1.0, w = w0, vp = 0.01,
                                                   vn = 0.01, up = 0.5, un = 0.5))
    mic(w) = 0.01 * sum(abs.(w .- w0) .^ 1.5)
    @test all(r -> isa(r.retcode, PortfolioOptimisers.OptimisationSuccess), rmi)
    @test all(r -> isapprox(sum(r.w) + mic(r.w), 1.0; atol = 1e-6), rmi)
    @test all(r -> isapprox((dot(pr.mu, r.w) - mic(r.w)) / sqrt(dot(r.w, pr.sigma, r.w)),
                            (dot(pr.mu, rmi[1].w) - mic(rmi[1].w)) /
                            sqrt(dot(rmi[1].w, pr.sigma, rmi[1].w)); rtol = 1e-5), rmi)
end
