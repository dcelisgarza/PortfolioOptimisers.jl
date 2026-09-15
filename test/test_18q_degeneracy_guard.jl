include(joinpath(@__DIR__, "test18_setup.jl"))

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
        @test_throws DomainError MaximumRatio(; kmin = 0.0)
        @test_throws DomainError MaximumRatio(; kmin = -1e-8)
    end
end
