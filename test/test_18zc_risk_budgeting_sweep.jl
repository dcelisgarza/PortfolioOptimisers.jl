using Clarabel, HiGHS, JuMP, Pajarito
const PO = PortfolioOptimisers

# The sweep of `src/17_Optimisation/05_JuMP/07_RiskBudgeting.jl`. Each formulation is checked
# against the risk budgeting condition of its original source, x_i (Sigma x)_i / x' Sigma x =
# b_i: the log constraint of bruderroncalli2012 equation 9, the orthant form of bai2016
# problem 2.13 with the general budget of mosek2023c problem 9.7, the mixed-integer form of
# mosek2023c problem 9.9, and the factor form of roncalliweisang2012 problem 8.

const RBS_X, RBS_F = let rng = StableRNG(689)
    T, N = 300, 5
    F = 0.01 .* randn(rng, T, 2)
    L = [0.9 0.2; 0.5 0.8; 1.1 -0.3; 0.3 0.6; 0.7 0.4]
    F * L' .+ 0.008 .* randn(rng, T, N) .+ 0.0005, F
end
const RBS_RD = ReturnsResult(; nx = ["a$i" for i in 1:5], X = RBS_X, nf = ["f1", "f2"],
                             F = RBS_F)
const RBS_PR = prior(EmpiricalPrior(), RBS_RD)
const RBS_B = [0.1, 0.15, 0.2, 0.25, 0.3]
const RBS_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       check_sol = (; allow_local = true, allow_almost = true),
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-11,
                                       "tol_gap_rel" => 1e-11, "tol_feas" => 1e-11))
const RBS_MIP = Solver(; name = :pajarito,
                       solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                          "verbose" => false,
                                                          "oa_solver" =>
                                                              optimizer_with_attributes(HiGHS.Optimizer,
                                                                                        JuMP.MOI.Silent() =>
                                                                                            true),
                                                          "conic_solver" =>
                                                              optimizer_with_attributes(Clarabel.Optimizer,
                                                                                        "verbose" =>
                                                                                            false)),
                       check_sol = (; allow_local = true, allow_almost = true))
# The share of the variance that each asset contributes. It is the same for the variance
# and for the standard deviation, because the two differ by a power.
rbs_share(w) = w .* (RBS_PR.sigma * w) ./ dot(w, RBS_PR.sigma * w)
function rbs_solve(alg; b = RBS_B, r = Variance(), slv = RBS_SLV, kwargs...)
    opt = JuMPOptimiser(; pe = RBS_PR, slv = slv, kwargs...)
    rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = b), alg = alg)
    return optimise(RiskBudgeting(; opt = opt, r = r, rba = rba), RBS_RD)
end

@testset "RiskBudgeting sweep: the log constraint gives the budgeted contributions" begin
    for r in (Variance(), StandardDeviation())
        res = rbs_solve(LogRiskBudgeting(); r = r)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
        @test isapprox(sum(res.w), 1; atol = 1e-8)
        # Appendix A.2 of maillard2008: the solution cannot be zero, so the log constraint
        # binds, and the weights are the solution divided by its sum.
        y = value.(res.model[:w])
        @test isapprox(dot(RBS_B, log.(y)), 0; atol = 1e-6)
        @test isapprox(value(res.model[:k]), sum(y); rtol = 1e-8)
    end
    # Section 2.4 of bruderroncalli2012: any convex risk measure with the Euler decomposition.
    r = factory(ConditionalValueatRisk(), RBS_PR, RBS_SLV)
    res = rbs_solve(LogRiskBudgeting(); r = r)
    rc = risk_contribution(r, res.w, RBS_PR.X)
    @test isapprox(rc / sum(rc), RBS_B; atol = 5e-3)
end

@testset "RiskBudgeting sweep: every formulation meets a budget other than one" begin
    z = [1, 1, -1, 1, 1]
    rg = BudgetRange(; lb = 0.4, ub = 0.6)
    # With no short budget the long budget row fixes the sum of the long parts. Under the
    # mixed-integer form the parts are the weights, so a budget of 0.5 keeps the book long,
    # and a range bounds the shorts by 0.2 k, a row that binds.
    cases = ((LogRiskBudgeting(), RBS_SLV, (;), (0.5, rg)),
             (LogRiskBudgeting(; z = z), RBS_SLV,
              (; wb = WeightBounds(; lb = -1.0, ub = 1.0), sbgt = 1.0), (0.5, rg)),
             (MixedIntegerRiskBudgeting(), RBS_MIP, (; wb = nothing, so = 10_000), (0.5,)))
    for (alg, slv, kw, bgts) in cases, bgt in bgts
        res = rbs_solve(alg; slv = slv, bgt = bgt, kw...)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
        if isa(bgt, Number)
            @test isapprox(sum(res.w), bgt; atol = 1e-8)
        else
            @test bgt.lb - 1e-8 <= sum(res.w) <= bgt.ub + 1e-8
        end
    end
end

@testset "RiskBudgeting sweep: the orthant vector sets the signs of the weights" begin
    kw = (; wb = WeightBounds(; lb = -1.0, ub = 1.0), sbgt = 1.0)
    z = [1, 1, -1, 1, 1]
    res = rbs_solve(LogRiskBudgeting(; z = z); kw...)
    @test sign.(res.w) == z
    @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
    # The solution in the orthant of `z` has a negative sum, so the budget row needs k < 0,
    # which the short budget forbids. The solve returns a portfolio of that orthant that
    # misses the budget, or fails. Lemma 2.2 of bai2016: the normalised portfolio of that
    # orthant lies in the complementary orthant, and `-z` reaches it. This book has no
    # bound and a short budget that does not bind, so the budget holds there.
    kw = (; wb = WeightBounds(; lb = -Inf, ub = Inf), sbgt = 1000.0)
    z = [-1, 1, 1, -1, -1]
    res = rbs_solve(LogRiskBudgeting(; z = z); kw...)
    @test !isa(res.retcode, OptimisationSuccess) ||
          maximum(abs.(rbs_share(res.w) - RBS_B)) > 1e-2
    res = rbs_solve(LogRiskBudgeting(; z = -z); kw...)
    @test sign.(res.w) == -z
    # Measured gap 6.4e-5 on this book.
    @test isapprox(rbs_share(res.w), RBS_B; atol = 2e-4)
end

@testset "RiskBudgeting sweep: the mixed-integer form takes the orthant of least risk" begin
    S = RBS_PR.sigma
    kw = (; wb = nothing, bgt = nothing, sbgt = nothing)
    # The risk of the solution in each orthant, at the scale where b' ln|y| = 0. The first
    # sign is fixed, because the orthants z and -z hold the same solution up to its sign.
    zs = [vcat(1, collect(t)) for t in Iterators.product(fill((1, -1), 4)...)][:]
    risk = map(zs) do z
        res = rbs_solve(LogRiskBudgeting(; z = z); kw...)
        w = res.w
        # No row fixes `k` here, and `k >= 0` keeps the orthant of `z` (#1844).
        @test value(res.model[:k]) > 0
        @test sign.(w) == z
        @test isapprox(rbs_share(w), RBS_B; atol = 5e-5)
        u = w * exp(-dot(RBS_B, log.(abs.(w))))
        return dot(u, S * u)
    end
    res = rbs_solve(MixedIntegerRiskBudgeting(); slv = RBS_MIP, so = 10_000, kw...)
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
    zm = Int.(sign.(res.w))
    @test zm * zm[1] == zs[argmin(risk)]
    # Finite bounds that do not bind reach the same orthant. The held bit, read as a long
    # sign, forbade every short under them (#2372). With nothing that reads a held bit the
    # pin takes one sign bit for each asset, and a cardinality takes a long and a short bit.
    fkw = merge(kw, (; wb = WeightBounds(; lb = -1.0, ub = 1.0)))
    for (extra, kind) in (((;), PO.SignIndicators), ((; card = 5), PO.LongShortIndicators))
        resb = rbs_solve(MixedIntegerRiskBudgeting(); slv = RBS_MIP, so = 10_000, fkw...,
                         extra...)
        @test isa(resb.retcode, OptimisationSuccess)
        @test PO.mip_indicators(resb.model) isa kind
        @test isapprox(rbs_share(resb.w), RBS_B; atol = 5e-5)
        zb = Int.(sign.(resb.w))
        @test zb * zb[1] == zs[argmin(risk)]
    end
    # Under a budget, a short budget range leaves the short exposure free, so the search
    # reaches the same orthant.
    res = rbs_solve(MixedIntegerRiskBudgeting(); slv = RBS_MIP, so = 10_000,
                    wb = WeightBounds(; lb = -Inf, ub = Inf),
                    sbgt = BudgetRange(; lb = 0.0, ub = 10.0))
    @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
    @test isapprox(sum(res.w), 1; atol = 1e-8)
    @test Int.(sign.(res.w)) == zs[argmin(risk)]
end

@testset "RiskBudgeting sweep: a gate of a long-short bundle scales with the budget (#2275)" begin
    # The model solves for the weights times `k`, so the bound, the threshold and the gate of
    # a bit scale with `k`. An equal budget on this book holds a smallest weight of about
    # 0.15, so a threshold of 0.18 binds.
    b = fill(0.2, 5)
    kw = (; b = b, r = StandardDeviation(), slv = RBS_MIP, so = 10_000,
          wb = WeightBounds(; lb = 0.0, ub = 1.0))
    th = Threshold(; val = 0.18)
    free = rbs_solve(MixedIntegerRiskBudgeting(); kw...)
    held = rbs_solve(MixedIntegerRiskBudgeting(); kw..., lt = th)
    # The bounds of each asset give the sign of its held bit, so a long-short bundle needs an
    # asset whose bounds admit both signs (#2372). The budget keeps the book long.
    both = rbs_solve(MixedIntegerRiskBudgeting();
                     merge(kw, (; wb = WeightBounds(; lb = -1.0, ub = 1.0)))..., lt = th,
                     st = th)
    @test minimum(free.w) < 0.18 - 1e-3
    for res in (held, both)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(minimum(res.w), 0.18; rtol = 1e-5)
        @test maximum(res.w) > 0.2 + 1e-3
    end
    @test PO.mip_indicators(held.model) isa PO.HeldIndicators
    @test PO.mip_indicators(both.model) isa PO.LongShortIndicators
    @test isapprox(both.w, held.w; rtol = 1e-3)
    k = value(both.model[:k])
    @test k > 2
    @test isapprox(value.(both.model[:il]), k .* value.(both.model[:ilb]); rtol = 1e-5)

    # A book with a short side: each held weight meets its threshold and its bound.
    res = rbs_solve(MixedIntegerRiskBudgeting(); b = b, r = StandardDeviation(),
                    slv = RBS_MIP, so = 10_000, wb = WeightBounds(; lb = -0.6, ub = 0.8),
                    sbgt = BudgetRange(; lb = 0.0, ub = 1.0), lt = th, st = th)
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(sum(res.w), 1; atol = 1e-6)
    @test all(x -> 0.18 - 1e-5 <= x <= 0.8 + 1e-5 || -0.6 - 1e-5 <= x <= -0.18 + 1e-5,
              res.w)
end

@testset "RiskBudgeting sweep: a number target of an even moment scales with the budget (#2275)" begin
    # `k` is the sum of the unscaled weights, so a target `mu` is the target zero on the
    # returns less `mu`.
    mu = 0.002
    prs = prior(EmpiricalPrior(), RBS_X .- mu)
    solve(pr, t, alg) = optimise(RiskBudgeting(; r = LowOrderMoment(; mu = t, alg = alg),
                                               opt = JuMPOptimiser(; pe = pr,
                                                                   slv = RBS_SLV)))
    for alg in (EvenMoment(), EvenMoment(; alg = SemiMoment()))
        w = solve(RBS_PR, mu, alg).w
        @test isapprox(w, solve(prs, 0.0, alg).w; rtol = 1e-3)
        @test !isapprox(w, solve(RBS_PR, 0.0, alg).w; rtol = 1e-2)
    end
end

@testset "RiskBudgeting sweep: a bound on a risk of degree above one bounds the risk of the weights (#2279)" begin
    prh = prior(HighOrderPriorEstimator(), RBS_RD)
    S = prh.sigma
    plg = PO.SemiDefinitePhylogeny(;
                                   A = [0.0 1 0 0 0; 1 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0;
                                        0 0 0 0 0], p = 1e-6)
    box = BoxUncertaintySet(; lb = S - 0.1 * abs.(S), ub = S + 0.1 * abs.(S))
    ell = EllipsoidalUncertaintySet(; sigma = Matrix(1e-12 * I(25)), k = 1.0,
                                    class = PO.SigmaUncertaintySetClass())
    cases = [("variance with a phylogeny", s -> Variance(; settings = s), plg, 1e4),
             ("box", s -> UncertaintySetVariance(; ucs = box, settings = s), nothing, 1e4),
             ("ellipsoid", s -> UncertaintySetVariance(; ucs = ell, settings = s), nothing,
              1e4),
             ("kurtosis", s -> Kurtosis(; alg = SOCRiskExpr(), settings = s), nothing, 1e4),
             ("approximate kurtosis",
              s -> Kurtosis(; N = 5, alg = SOCRiskExpr(), settings = s), nothing, 1e4),
             ("even moment", s -> LowOrderMoment(; alg = EvenMoment(), settings = s),
              nothing, 1),
             ("even lower moment",
              s -> LowOrderMoment(; alg = EvenMoment(; alg = SemiMoment()), settings = s),
              nothing, 1),
             ("weighted even moment",
              s -> LowOrderMoment(; w = pweights(range(0.5, 1.5; length = 300)),
                                  alg = EvenMoment(), settings = s), nothing, 1),
             ("variance, skewness and kurtosis", s -> VarianceSkewKurtosis(; settings = s),
              nothing, 1e4)]
    function solve(r, plg, so)
        opt = JuMPOptimiser(; pe = prh, slv = RBS_SLV, ple = plg, so = so)
        return optimise(RiskBudgeting(; opt = opt, r = r), RBS_RD)
    end
    for (name, f, pl, so) in cases
        @testset "$name" begin
            r0 = f(RiskMeasureSettings())
            R(w) = expected_risk(factory(r0, prh, RBS_SLV), w, RBS_X)
            free = solve(r0, pl, so)
            @test isa(free.retcode, OptimisationSuccess)
            # A bound that does not bind leaves the book, and one that binds holds the
            # risk of the weights at the bound.
            loose = solve(f(RiskMeasureSettings(; ub = 1.5 * R(free.w))), pl, so)
            @test isa(loose.retcode, OptimisationSuccess)
            @test isapprox(loose.w, free.w; rtol = 2e-3)
            ub = 0.97 * R(free.w)
            tight = solve(f(RiskMeasureSettings(; ub = ub)), pl, so)
            @test isa(tight.retcode, OptimisationSuccess)
            @test isapprox(R(tight.w), ub; rtol = 1e-3)
            # A measure that is only a bound, beside the risk that the budget reads.
            only = solve([StandardDeviation(),
                          f(RiskMeasureSettings(; ub = ub, rke = false))], pl, so)
            @test isa(only.retcode, OptimisationSuccess)
            @test R(only.w) <= ub * (1 + 2e-3)
        end
    end
end

@testset "RiskBudgeting sweep: only a bounded risk of degree above one pays for its bound (#2279)" begin
    size_of(m) = (JuMP.num_variables(m),
                  JuMP.num_constraints(m; count_variable_in_set_constraints = true))
    prh = prior(HighOrderPriorEstimator(), RBS_RD)
    opt = JuMPOptimiser(; pe = prh, slv = RBS_SLV)
    added(f, head) = size_of(head(f(RiskMeasureSettings(; ub = 1.0))).model) .-
                     size_of(head(f(RiskMeasureSettings())).model)
    rb(r) = optimise(RiskBudgeting(; opt = opt, r = r), RBS_RD)
    mr(r) = optimise(MeanRisk(; opt = opt, r = r), RBS_RD)
    ratio(r) = optimise(MeanRisk(; opt = opt, r = r, obj = MaximumRatio()), RBS_RD)
    even = s -> LowOrderMoment(; alg = EvenMoment(), settings = s)
    kurt = s -> Kurtosis(; settings = s)
    # The bound of a measure of degree one, and of a second-order cone variance, is one row.
    for f in (s -> StandardDeviation(; settings = s), s -> Variance(; settings = s),
              s -> ConditionalValueatRisk(; settings = s),
              s -> LowOrderMoment(; alg = SecondMoment(), settings = s))
        @test added(f, rb) == (0, 1)
    end
    # Under another head the bound of every measure is one row.
    for f in (even, kurt), head in (mr, ratio)
        @test added(f, head) == (0, 1)
    end
    # A bounded risk of degree above one pays for a second model under risk budgeting.
    @test all(added(even, rb) .> (0, 1))
    @test all(added(kurt, rb) .> (0, 1))
    # A measure that is only a bound builds the model of its bound alone.
    m = rb([StandardDeviation(), even(RiskMeasureSettings(; ub = 1.0, rke = false))]).model
    @test haskey(m, :bound_even_moment_risk_2)
    @test !haskey(m, :even_moment_risk_2)
    @test haskey(m, :even_moment_risk_2_ub)
    # Two measures on one weight vector share the lift of the bound.
    m = rb([kurt(RiskMeasureSettings(; ub = 1.0)),
            Kurtosis(; N = 5, settings = RiskMeasureSettings(; ub = 1.0))]).model
    @test size(m[:bound_W]) == size(m[:W]) == (5, 5)
end

@testset "RiskBudgeting sweep: the bounds of the terms of VarianceSkewKurtosis (#2279)" begin
    prh = prior(HighOrderPriorEstimator(), RBS_RD)
    opt = JuMPOptimiser(; pe = prh, slv = RBS_SLV, so = 1e4)
    solve(r) = optimise(RiskBudgeting(; opt = opt, r = r), RBS_RD)
    free = solve(VarianceSkewKurtosis())
    vr = dot(free.w, prh.sigma, free.w)
    res = solve(VarianceSkewKurtosis(;
                                     vr = Variance(;
                                                   settings = RiskMeasureSettings(;
                                                                                  ub = 0.97 *
                                                                                       vr,
                                                                                  rke = false))))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(dot(res.w, prh.sigma, res.w), 0.97 * vr; rtol = 1e-3)
end

@testset "A bound on a tracked risk bounds the risk of the weights (#2283)" begin
    prh = prior(HighOrderPriorEstimator(), RBS_RD)
    S = prh.sigma
    plg = PO.SemiDefinitePhylogeny(;
                                   A = [0.0 1 0 0 0; 1 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0;
                                        0 0 0 0 0], p = 1e-6)
    box = UncertaintySetVariance(;
                                 ucs = BoxUncertaintySet(; lb = S - 0.1 * abs.(S),
                                                         ub = S + 0.1 * abs.(S)))
    rsoc = LowOrderMoment(; alg = SecondMoment(; alg2 = RSOCRiskExpr()))
    soc = LowOrderMoment(; alg = SecondMoment(; alg2 = SOCRiskExpr()))
    even = LowOrderMoment(; alg = EvenMoment())
    kurt = Kurtosis(; alg = SquaredSOCRiskExpr())
    sql2 = TrackingRiskMeasure(; tr = WeightsTracking(; w = [0.0, 0, 0, 0, 0.5]),
                               alg = SquaredL2Norm())
    ind, dep = IndependentVariableTracking(), DependentVariableTracking()
    track(r, wb, alg; kw...) = RiskTrackingRiskMeasure(;
                                                       settings = RiskMeasureSettings(;
                                                                                      kw...),
                                                       tr = WeightsTracking(; w = wb),
                                                       r = r, alg = alg)
    nested(alg) = track(Variance(), [0.1, 0, 0, 0, 0], alg)
    # The second solver takes the models on which the first one stops.
    slv = [RBS_SLV,
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false))]
    opts(tr, pl) = JuMPOptimiser(; pe = prh, slv = slv, tr = tr, ple = pl, so = 1e3)
    utility(r, tr, pl) = MeanRisk(; opt = opts(tr, pl), r = r,
                                  obj = MaximumUtility(; l = 0.5))
    ratio(r, tr, pl) = MeanRisk(; opt = opts(tr, pl), r = r, obj = MaximumRatio())
    budget(r, tr, pl) = RiskBudgeting(; opt = opts(tr, pl), r = r)
    cases = [(utility, Variance(), nothing, (ind, dep)), (utility, rsoc, nothing, (dep,)),
             (ratio, Variance(), nothing, (ind, dep)), (ratio, Variance(), plg, (ind,)),
             (ratio, Variance(; alg = QuadRiskExpr()), nothing, (dep,)),
             (ratio, rsoc, nothing, (ind, dep)), (ratio, soc, nothing, (ind, dep)),
             (ratio, kurt, nothing, (ind, dep)),
             (ratio, NegativeSkewness(; alg = SquaredSOCRiskExpr()), nothing, (ind, dep)),
             (ratio, sql2, nothing, (ind, dep)), (ratio, nested(ind), nothing, (ind, dep)),
             (ratio, nested(dep), nothing, (ind,)),
             (budget, StandardDeviation(), nothing, (ind, dep)),
             # Clarabel stops on an independent bound row that does not divide by its bound.
             (budget, UlcerIndex(), nothing, (ind,)),
             (budget, Variance(), nothing, (ind, dep)), (budget, Variance(), plg, (ind,)),
             (budget, box, nothing, (ind, dep)), (budget, rsoc, nothing, (ind, dep)),
             (budget, even, nothing, (ind,)), (budget, kurt, nothing, (ind, dep)),
             (budget, VarianceSkewKurtosis(), nothing, (ind, dep))]
    base = StandardDeviation()
    solve(head, r, tr, pl) = optimise(head(r, tr, pl), RBS_RD)
    w0s = Dict(head => solve(head, base, nothing, nothing).w
               for head in (utility, ratio, budget))
    # The least risky book leaves the dependent bound room below the book with no bound.
    function benchmark(ri, alg)
        if alg === ind
            return fill(0.2, 5)
        end
        return optimise(MeanRisk(; r = ri, opt = opts(nothing, nothing)), RBS_RD).w
    end
    for (head, ri, pl, algs) in cases, alg in algs
        @testset "$head, $(nameof(typeof(ri))), $(nameof(typeof(alg)))" begin
            rf = factory(ri, prh, RBS_SLV)
            R(w) = expected_risk(rf, w, RBS_X)
            wb = benchmark(ri, alg)
            f = factory(track(ri, wb, alg), prh, RBS_SLV)
            q(w) = alg === dep ? R(w) - R(wb) : expected_risk(f, w, RBS_X)
            e0 = q(w0s[head])
            @test e0 > 0
            # A bound that does not bind leaves the book, and one that binds holds the
            # tracked risk of the weights at the bound.
            for c in (1.5, 0.5)
                tre = RiskTrackingError(; tr = WeightsTracking(; w = wb), r = ri,
                                        err = c * e0, alg = alg)
                trm = track(ri, wb, alg; ub = c * e0, rke = false)
                for res in
                    (solve(head, base, tre, pl), solve(head, [base, trm], nothing, pl))
                    @test isa(res.retcode, OptimisationSuccess)
                    @test isapprox(q(res.w), min(c, 1) * e0; rtol = 2e-2)
                end
            end
        end
    end
    # A bound that does not bind leaves the term of a measure that the objective reads.
    for (head, ri, alg) in
        ((ratio, Variance(), ind), (ratio, Variance(), dep), (budget, Variance(), ind),
         (budget, box, ind), (budget, Kurtosis(; alg = SOCRiskExpr()), ind),
         (budget, VarianceSkewKurtosis(), ind), (budget, Variance(), dep))
        wb = benchmark(ri, alg)
        f = factory(track(ri, wb, alg), prh, RBS_SLV)
        free = solve(head, [base, track(ri, wb, alg; scale = 5.0)], nothing, nothing)
        @test isa(free.retcode, OptimisationSuccess)
        ub = 1.5 * expected_risk(f, free.w, RBS_X)
        loose = solve(head, [base, track(ri, wb, alg; scale = 5.0, ub = ub)], nothing,
                      nothing)
        @test isa(loose.retcode, OptimisationSuccess)
        @test isapprox(loose.w, free.w; rtol = 5e-3)
    end
end

@testset "A tracked risk states its bound on one expression (#2283)" begin
    prh = prior(HighOrderPriorEstimator(), RBS_RD)
    opt = JuMPOptimiser(; pe = prh, slv = RBS_SLV)
    wb = WeightsTracking(; w = fill(0.2, 5))
    ind, dep = IndependentVariableTracking(), DependentVariableTracking()
    track(r, alg; kw...) = RiskTrackingRiskMeasure(;
                                                   settings = RiskMeasureSettings(; kw...),
                                                   tr = wb, r = r, alg = alg)
    # A bound of the independent mode is one row, and it keeps the key of the tracking risk.
    nrow(r, ub) = begin
        m = optimise(MeanRisk(; opt = opt,
                              r = [StandardDeviation(), track(r, ind; ub = ub)]), RBS_RD).model
        (num_variables(m), num_constraints(m; count_variable_in_set_constraints = true),
         haskey(m, :tracking_risk_2_ub))
    end
    for r in (StandardDeviation(), Variance(), ConditionalValueatRisk())
        a, b = nrow(r, 1.0), nrow(r, nothing)
        @test a[1:2] == b[1:2] .+ (0, 1)
        @test a[3] && !b[3]
    end
    # Under a unit budget the objective keeps its term, and the bound takes a second model.
    rb(r) = optimise(RiskBudgeting(; opt = opt, r = r), RBS_RD).model
    kurt = Kurtosis(; alg = SOCRiskExpr())
    m = rb([StandardDeviation(), track(kurt, ind; ub = 1.0)])
    @test haskey(m, :tr_iv_2_kurtosis_risk_1) && haskey(m, :tr_iv_2_bound_kurtosis_risk_1)
    m = rb([StandardDeviation(), track(kurt, ind; ub = 1.0, rke = false)])
    @test !haskey(m, :tr_iv_2_kurtosis_risk_1) && haskey(m, :tr_iv_2_bound_kurtosis_risk_1)
    # An even moment takes its units from the benchmark risk in the dependent mode, and a
    # bound of zero gives it no units.
    even = LowOrderMoment(; alg = EvenMoment())
    m = rb([StandardDeviation(), track(even, dep; ub = 1.0, rke = false)])
    @test haskey(m, :tr_dv_2_bound_even_moment_risk_1)
    @test haskey(m, :tr_dv_2_bound_ceven_moment_s_1)
    m = optimise(RiskBudgeting(;
                               opt = JuMPOptimiser(; pe = prh, slv = RBS_SLV,
                                                   tr = RiskTrackingError(; tr = wb,
                                                                          r = even)),
                               r = StandardDeviation()), RBS_RD).model
    @test haskey(m, :tr_ir_1_bound_ceven_moment_s_1)
    # A request converts as the bound of its measure does.
    ask = PO.TrackedRiskBound(:p_, true, 4.0, false)
    @test PO.risk_bound_level(ask) == (4.0, false)
    @test PO.risk_bound_level(2.0) == (2.0, false)
    @test PO.risk_bound_level([1.0, 2.0]) == PO.risk_bound_level(nothing) == (0, true)
    sq = PO.variance_risk_bounds_val(PO.SquareRootBound(), ask)
    @test PO.variance_risk_bounds_val(sq, 4.0) == 2.0
    @test PO.variance_risk_bounds_val(sq, [4.0, 9.0]) == [2.0, 3.0]
    @test isnothing(PO.variance_risk_bounds_val(sq, nothing))
    fr = PO.variance_risk_bounds_val(sq, Frontier(; N = 3))
    @test fr.N == 3 && fr.factor == 1 && fr.bound === PO.SquareRootBound()
    sm = PO.second_moment_bound_val(RSOCRiskExpr(), ask, 0.5)
    @test PO.variance_risk_bounds_val(sm, 4.0) ==
          PO.second_moment_bound_val(RSOCRiskExpr(), 4.0, 0.5)
    @test PO.variance_risk_bounds_val(PO.variance_risk_bounds_val(sq, ask), 9.0) == 3.0
    @test PO.reads_risk_term(RiskMeasureSettings(; rke = false, ub = ask))
    @test !PO.reads_risk_term(RiskMeasureSettings(; rke = false, ub = 1.0))
end

@testset "A sweep on a tracked risk takes the units of its bound (#2283)" begin
    base = StandardDeviation()
    for alg in (IndependentVariableTracking(), DependentVariableTracking())
        wb = if isa(alg, DependentVariableTracking)
            optimise(MeanRisk(; r = base,
                              opt = JuMPOptimiser(; pe = RBS_PR, slv = RBS_SLV)), RBS_RD).w
        else
            fill(0.2, 5)
        end
        track(ub) = RiskTrackingRiskMeasure(;
                                            settings = RiskMeasureSettings(; ub = ub,
                                                                           rke = false),
                                            tr = WeightsTracking(; w = wb), r = Variance(),
                                            alg = alg)
        f = factory(track(nothing), RBS_PR, RBS_SLV)
        q(w) = expected_risk(f, w, RBS_X)
        solve(ub) = optimise(MeanRisk(; r = [base, track(ub)], obj = MaximumReturn(),
                                      opt = JuMPOptimiser(; pe = RBS_PR, slv = RBS_SLV,
                                                          so = 1e3)), RBS_RD)
        lo, hi = q(solve(1e-7).w), q(solve(nothing).w)
        # A vector of levels: the tracked risk of each book is at its level.
        levels = [0.3, 0.6] .* hi
        res = solve(levels)
        @test all(x -> isa(x, OptimisationSuccess), res.retcode)
        @test isapprox(q.(res.w), levels; rtol = 1e-2)
        # A sweep: the levels ascend, and the last one is the tracked risk of the free book.
        res = solve(Frontier(; N = 4))
        @test all(x -> isa(x, OptimisationSuccess), res.retcode)
        qs = q.(res.w)
        @test issorted(qs)
        @test isapprox(qs[end], hi; rtol = 1e-2)
        # The levels of the independent mode are evenly spaced in the deviation.
        if isa(alg, IndependentVariableTracking)
            d = diff(sqrt.(qs))
            @test isapprox(d, fill(sum(d) / 3, 3); rtol = 2e-2)
        end
    end
end

@testset "FactorRiskContribution tracks the variance of the weight difference (#2283)" begin
    wb = WeightsTracking(; w = fill(0.2, 5))
    te(w) = dot(w .- 0.2, RBS_PR.sigma, w .- 0.2)
    solve(obj, tr) = optimise(FactorRiskContribution(; r = StandardDeviation(), flag = true,
                                                     obj = obj,
                                                     opt = JuMPOptimiser(; pe = RBS_PR,
                                                                         slv = RBS_SLV,
                                                                         tr = tr)), RBS_RD)
    for obj in (MinimumRisk(), MaximumRatio())
        e0 = te(solve(obj, nothing).w)
        for c in (1.5, 0.5), alg in (SquaredSOCRiskExpr(), QuadRiskExpr())
            res = solve(obj,
                        RiskTrackingError(; tr = wb, r = Variance(; alg = alg),
                                          err = c * e0))
            @test isa(res.retcode, OptimisationSuccess)
            @test isapprox(te(res.w), min(c, 1) * e0; rtol = 1e-2)
        end
    end
    # The factor rows of `rc` have no meaning on the weight difference.
    rc = LinearConstraintEstimator(; val = "f1 <= 0.5")
    @test_throws ArgumentError solve(MinimumRisk(),
                                     RiskTrackingError(; tr = wb, r = Variance(; rc = rc),
                                                       err = 1.0))
end

@testset "RiskBudgeting sweep: the factor form gives the budgeted factor contributions" begin
    S = RBS_PR.sigma
    bf = [0.3, 0.7]
    for (flag, hedge) in ((false, false), (true, false), (true, true))
        opt = JuMPOptimiser(; pe = RBS_PR, slv = RBS_SLV,
                            wb = WeightBounds(; lb = -Inf, ub = Inf), sbgt = 1000.0)
        rba = FactorRiskBudgeting(; rkb = RiskBudget(; val = bf), flag = flag,
                                  hedge = hedge)
        res = optimise(RiskBudgeting(; opt = opt, rba = rba), RBS_RD)
        @test isa(res.retcode, OptimisationSuccess)
        w = res.w
        L = res.rbr.rr.L
        # Theorem 1 of roncalliweisang2012: the contribution of factor j is y_j (L^+ dR/dx)_j,
        # with the factor exposures y = L' w.
        y = transpose(L) * w
        rc = y .* (pinv(L) * (S * w))
        @test isapprox(rc / sum(rc), bf; atol = 5e-5)
        @test isapprox(value.(res.model[:w1]) / value(res.model[:k]), y; rtol = 1e-8)
        # The off-factor weights are free, so at the optimum they carry no marginal risk:
        # the separation principle of section 4.1.1. With `flag = false` the weights lie in
        # the range of the loadings, and the factors take the whole variance.
        g = transpose(nullspace(Matrix(transpose(L)))) * (S * w)
        if flag
            @test norm(g) <= 1e-4 * norm(S * w)
        else
            @test isapprox(sum(rc), dot(w, S * w); rtol = 1e-8)
        end
    end
    # A long-only bound that binds keeps the off-factor weights from their free optimum. The
    # hedge rows hold the first-order condition all the same.
    opt = JuMPOptimiser(; pe = RBS_PR, slv = RBS_SLV)
    rba = FactorRiskBudgeting(; rkb = RiskBudget(; val = bf), hedge = true)
    res = optimise(RiskBudgeting(; opt = opt, rba = rba), RBS_RD)
    L = res.rbr.rr.L
    g = transpose(nullspace(Matrix(transpose(L)))) * (S * res.w)
    @test norm(g) <= 1e-8 * norm(S * res.w)
end

@testset "RiskBudgeting sweep: the floor of k" begin
    # The derived floor is 1e-4 over the upper bound of the budget, and zero when no finite
    # budget ties `k` to the sum of the weights. A stated `kmin` replaces it.
    rb_of(; kmin = nothing, kw...) = RiskBudgeting(;
                                                   opt = JuMPOptimiser(; slv = RBS_SLV,
                                                                       kw...), kmin = kmin)
    floor_of(; kw...) = PO.risk_budgeting_scale_floor(rb_of(; kw...))
    @test floor_of() == 1e-4
    @test floor_of(; bgt = 0.5) == 2e-4
    @test floor_of(; bgt = BudgetRange(; lb = 0.4, ub = 0.8)) ≈ 1.25e-4
    @test iszero(floor_of(; bgt = nothing))
    @test iszero(floor_of(; bgt = BudgetRange(; lb = 0.4, ub = nothing)))
    @test floor_of(; kmin = 1e-2) == 1e-2
    @test floor_of(; kmin = 1e-2, bgt = nothing) == 1e-2
    # A stated zero turns the floor off and keeps only `k >= 0`.
    @test iszero(floor_of(; kmin = 0.0))
    @test_throws DomainError rb_of(; kmin = -1)
    @test PO.port_opt_view(rb_of(; kmin = 1e-3), 1:3, RBS_X).kmin == 1e-3
    # The solution in the orthant of `z` has a negative sum, and no weight bound or short
    # budget bounds the scale. No portfolio of that orthant meets a positive budget, so the
    # scale collapses onto the floor, and the weights miss the budget. Measured at the
    # tolerance of the solver alone, before the floor: k = 2.2e-6 and a gross exposure of
    # 5e6, with `OptimisationSuccess`.
    kw = (; wb = WeightBounds(; lb = -Inf, ub = Inf),
          sbgt = BudgetRange(; lb = 0.0, ub = nothing))
    z = [-1, 1, 1, -1, -1]
    for (bgt, kmin) in ((1.0, nothing), (0.5, nothing), (1.0, 1e-2))
        opt = JuMPOptimiser(; pe = RBS_PR, slv = RBS_SLV, bgt = bgt, kw...)
        rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = RBS_B),
                                 alg = LogRiskBudgeting(; z = z))
        rb = RiskBudgeting(; opt = opt, r = Variance(), rba = rba, kmin = kmin)
        res = optimise(rb, RBS_RD)
        kf = PO.risk_budgeting_scale_floor(rb)
        @test lower_bound(res.model[:k]) == kf
        # Measured on the floor to 1.3e-5.
        @test isapprox(value(res.model[:k]), kf; rtol = 1e-4)
        @test sign.(res.w) == z
        @test isapprox(sum(res.w), bgt; rtol = 1e-6)
        @test maximum(abs.(rbs_share(res.w) - RBS_B)) > 1e-1
    end
    # The complementary orthant has a positive sum, so its scale stays far above the floor.
    # Its k = 0.459 is the smallest scale that a well-posed model fixed over the risk
    # budgeting tests.
    res = rbs_solve(LogRiskBudgeting(; z = -z); kw...)
    @test value(res.model[:k]) > 1e3 * floor_of()
    @test sign.(res.w) == -z
    @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
end

@testset "RiskBudgeting sweep: views, validation and the result" begin
    a = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting())
    @test PO.port_opt_view(a, 1:3).alg === MixedIntegerRiskBudgeting()
    a = AssetRiskBudgeting(; alg = LogRiskBudgeting())
    @test PO.port_opt_view(a, 1:3).alg === a.alg
    a = AssetRiskBudgeting(; alg = LogRiskBudgeting(; z = [1, -1, 1, 1, -1]))
    @test PO.port_opt_view(a, [2, 5]).alg.z == [-1, -1]
    rb = RiskBudgeting(; opt = JuMPOptimiser(; slv = RBS_SLV),
                       rba = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting()))
    @test isa(PO.port_opt_view(rb, 1:3, RBS_X).rba.alg, MixedIntegerRiskBudgeting)
    # A budget with no sets names no asset, so a non-investable list leaves it unchanged.
    rb2 = PO.non_investable_universe(rb, ["a2"])
    @test isnothing(rb2.rba.sets)
    @test isa(rb2.rba.alg, MixedIntegerRiskBudgeting)
    @test_throws IsEmptyError RiskBudgeting(; opt = JuMPOptimiser(; slv = RBS_SLV),
                                            r = PO.RiskMeasure[])
    @test_throws IsEmptyError LogRiskBudgeting(; z = Int[])
    @test_throws ArgumentError LogRiskBudgeting(; z = [1, 2])
    res = rbs_solve(LogRiskBudgeting())
    @test isa(res.rbr, PO.ResolvedAssetRiskBudgeting)
    @test res.rkb.val == RBS_B
    res2 = PO.set_retcode(res, OptimisationFailure(; res = nothing))
    @test isa(res2.retcode, OptimisationFailure)
    @test res2.w == res.w
end
