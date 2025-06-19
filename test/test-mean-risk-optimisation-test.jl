# @safetestset "MeanRisk Optimisation" begin
using PortfolioOptimisers, CSV, DataFrames, Test, StableRNGs, Random, Clarabel, StatsBase,
      LinearAlgebra, Pajarito, HiGHS, JuMP, TimeSeries, Clustering, Distributions
function find_tol(a1, a2; name1 = :a1, name2 = :a2)
    for rtol ∈
        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
         5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0, 1.4e0,
         1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
        if isapprox(a1, a2; rtol = rtol)
            println("isapprox($name1, $name2, rtol = $(rtol))")
            break
        end
    end
    for atol ∈
        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
         5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0, 1.4e0,
         1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
        if isapprox(a1, a2; atol = atol)
            println("isapprox($name1, $name2, atol = $(atol))")
            break
        end
    end
end
X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
              timestamp = :timestamp)
F = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/factor_prices.csv"));
              timestamp = :timestamp)
rd = prices_to_returns(X[(end - 252):end], F[(end - 252):end])
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.75))]
mip_slv = Solver(; name = :clarabel,
                 solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                    "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                             MOI.Silent() => true),
                                                    "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                "verbose" => false,
                                                                                                "max_step_fraction" => 0.75)),
                 check_sol = (; allow_local = true, allow_almost = true))
pr = prior(HighOrderPriorEstimator(;
                                   pe = FactorPriorEstimator(;
                                                             re = DimensionReductionRegression())),
           rd)
rf = 4.34 / 100 / 252
clr = clusterise(ClusteringEstimator(), pr.X)
clusters = cutree(clr.clustering; k = clr.k)
sets = DataFrame(; Assets = colnames(X), Clusters = clusters)
T, N = size(pr.X)
eqw = range(; start = inv(N), stop = inv(N), length = N)
@testset "MeanRisk measures" begin
    T, N = size(pr.X)
    ew = eweights(1:T, inv(T); scale = true)
    w1 = fill(inv(N), N)
    sigma = cov(GerberCovariance(), pr.X)
    mu = vec(mean(ShrunkExpectedReturns(; ce = GerberCovariance()), pr.X))
    sk, V = coskewness(Coskewness(;), rd.X)
    kt = cokurtosis(Cokurtosis(;), rd.X)
    pr.sk .= sk
    pr.V .= V
    pr.kt .= kt
    sk, V = coskewness(Coskewness(; alg = Semi()), rd.X)
    kt = cokurtosis(Cokurtosis(; alg = Semi()), rd.X)
    rng = StableRNG(987456321)
    ucs1 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                   rng = rng,
                                                   alg = BoxUncertaintySetAlgorithm(),
                                                   seed = 987654321), pr.X)
    ucs2 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                   rng = rng,
                                                   alg = EllipseUncertaintySetAlgorithm(),
                                                   seed = 987654321), pr.X)
    df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk1.csv"), DataFrame)
    pw = pweights(collect(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                                length = size(pr.X, 1))))
    rs = [Variance(; sigma = sigma, formulation = QuadRiskExpr()), Variance(),
          UncertaintySetVariance(; sigma = sigma, ucs = ucs1),
          UncertaintySetVariance(; ucs = ucs2), StandardDeviation(; sigma = sigma),
          StandardDeviation(), LowOrderMoment(; mu = mu), LowOrderMoment(; mu = rf),
          LowOrderMoment(; w = pw), LowOrderMoment(),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = SqrtRiskExpr())),
                         mu = mu),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = SqrtRiskExpr())),
                         mu = rf),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 ve = SimpleVariance(; corrected = false,
                                                                     w = pw),
                                                 alg = SecondLowerMoment(;
                                                                         formulation = SqrtRiskExpr())),
                         w = pw),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = SqrtRiskExpr()))),
          LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()), mu = mu),
          LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()), mu = rf),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 ve = SimpleVariance(; corrected = false,
                                                                     w = pw),
                                                 alg = SecondLowerMoment()), w = pw),
          LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment())),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = QuadRiskExpr())),
                         mu = mu),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = QuadRiskExpr())),
                         mu = rf),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 ve = SimpleVariance(; corrected = false,
                                                                     w = pw),
                                                 alg = SecondLowerMoment(;
                                                                         formulation = QuadRiskExpr())),
                         w = pw),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = QuadRiskExpr()))),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = RSOCRiskExpr())),
                         mu = mu),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = RSOCRiskExpr())),
                         mu = rf),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 ve = SimpleVariance(; corrected = false,
                                                                     w = pw),
                                                 alg = SecondLowerMoment(;
                                                                         formulation = RSOCRiskExpr())),
                         w = pw),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = RSOCRiskExpr()))),
          LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = mu),
          LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = rf),
          LowOrderMoment(; alg = MeanAbsoluteDeviation(), w = ew),
          LowOrderMoment(; alg = MeanAbsoluteDeviation()), SquareRootKurtosis(),
          SquareRootKurtosis(; kt = kt), SquareRootKurtosis(; N = 2),
          SquareRootKurtosis(; kt = kt, N = 2), NegativeSkewness(),
          NegativeSkewness(; alg = QuadRiskExpr()), NegativeSkewness(; sk = sk, V = V),
          NegativeSkewness(; alg = QuadRiskExpr(), sk = sk, V = V),
          ValueatRisk(; formulation = DistributionValueatRisk()),
          ValueatRisk(; formulation = DistributionValueatRisk(; dist = TDist(30))),
          ValueatRisk(; formulation = DistributionValueatRisk(; dist = Laplace())),
          ValueatRiskRange(; beta = 0.2, formulation = DistributionValueatRisk()),
          ValueatRiskRange(; beta = 0.2,
                           formulation = DistributionValueatRisk(; dist = TDist(30))),
          ValueatRiskRange(; beta = 0.2,
                           formulation = DistributionValueatRisk(; dist = Laplace())),
          ConditionalValueatRisk(;), ConditionalValueatRisk(; w = pw),
          DistributionallyRobustConditionalValueatRisk(; l = 1.7, r = 3e-4),
          DistributionallyRobustConditionalValueatRisk(; l = 1.7, r = 3e-4, w = pw),
          ConditionalValueatRiskRange(), ConditionalValueatRiskRange(; w = pw),
          DistributionallyRobustConditionalValueatRiskRange(; l_a = 1.7, r_a = 3e-4,
                                                            l_b = 1.7, r_b = 3e-4),
          DistributionallyRobustConditionalValueatRiskRange(; l_a = 1.7, r_a = 3e-4,
                                                            l_b = 1.7, r_b = 3e-4, w = pw),
          ConditionalDrawdownatRisk(), EntropicValueatRisk(), EntropicValueatRisk(; w = pw),
          EntropicValueatRiskRange(), EntropicValueatRiskRange(; w = pw),
          EntropicDrawdownatRisk(), RelativisticValueatRisk(),
          RelativisticValueatRisk(; w = pw), RelativisticValueatRiskRange(),
          RelativisticValueatRiskRange(; w = pw), RelativisticDrawdownatRisk(),
          AverageDrawdown(), AverageDrawdown(; w = pw), UlcerIndex(), MaximumDrawdown(),
          WorstRealisation(), Range(),
          [ConditionalValueatRisk(), TurnoverRiskMeasure(; w = w1)],
          [ConditionalValueatRisk(; settings = RiskMeasureSettings(; scale = 0.5)),
           TrackingRiskMeasure(; tracking = WeightsTracking(; w = w1))],
          [ConditionalValueatRisk(; settings = RiskMeasureSettings(; scale = 0.5)),
           TrackingRiskMeasure(; tracking = ReturnsTracking(; w = pr.X * w1))]]
    objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(; ohf = 1, rf = rf),
            MaximumReturn()]
    rets = [ArithmeticReturn(), KellyReturn()]
    i = 1
    for r ∈ rs
        for obj ∈ objs
            for ret ∈ rets
                opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
                sol = optimise!(MeanRisk(; r = r, obj = obj, opt = opt))
                if isa(sol.retcode, OptimisationFailure)
                    continue
                end
                w = sol.w
                wt = df[!, "$i"]
                rtol = if i ∈
                          (6, 10, 14, 17, 18, 19, 20, 21, 22, 30, 46, 62, 70, 78, 158, 179,
                           187, 189, 213, 221, 229, 237, 245, 253, 261, 267, 275, 285, 291,
                           293, 301, 309, 317, 325, 333, 341, 349, 357, 365, 368, 370, 371,
                           374, 376, 377, 385, 393, 396, 399, 400, 403, 407, 415, 418, 419,
                           420, 423, 426, 427, 428, 430, 431, 434, 435, 436, 437, 438, 441,
                           442, 443, 444, 446, 449, 450, 451, 457, 462, 469, 470, 486, 518,
                           526, 534, 542, 556, 558, 564, 566) ||
                          Sys.isapple() &&
                          i ∈ (38, 54, 60, 140, 156, 171, 173, 203, 259, 269)
                    1
                else
                    1e-6
                end
                res = isapprox(w, wt; rtol = rtol)
                if !res
                    println("$i failed:\n$(typeof(r))\n$(typeof(obj))\n$(typeof(ret)).")
                    find_tol(w1, wt; name1 = :w1, name2 = :wt)
                    display([w wt])
                end
                @test res
                i += 1
            end
        end
    end
end
@testset "Returns lower bounds and uncertainty sets" begin
    rng = StableRNG(123456789)
    ucs1 = mu_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                                alg = BoxUncertaintySetAlgorithm(),
                                                seed = 987654321), pr.X)
    ucs2 = mu_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(), rng = rng,
                                                alg = EllipseUncertaintySetAlgorithm(),
                                                seed = 987654321), pr.X)

    ret = ArithmeticReturn()
    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res1 = optimise!(mre)

    ret = ArithmeticReturn(; lb = dot(res1.w, pr.mu))
    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res2 = optimise!(mre)
    @test dot(res2.w, pr.mu) >= ret.lb - sqrt(eps())

    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
    mre = MeanRisk(; obj = MaximumRatio(), opt = opt)
    res3 = optimise!(mre)
    @test dot(res3.w, pr.mu) >= ret.lb - sqrt(eps())

    ret = KellyReturn()
    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res1 = optimise!(mre)

    ret = KellyReturn(; lb = mean(log1p.(pr.X * res1.w)))
    opt = JuMPOptimiser(; pe = pr, ret = ret,
                        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                                     check_sol = (; allow_local = true,
                                                  allow_almost = true),
                                     settings = Dict("max_step_fraction" => 0.65,
                                                     "verbose" => false)))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res2 = optimise!(mre)
    @test mean(log1p.(pr.X * res2.w)) >= ret.lb - sqrt(eps())

    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
    mre = MeanRisk(; obj = MaximumRatio(), opt = opt)
    res3 = optimise!(mre)
    @test mean(log1p.(pr.X * res3.w)) >= ret.lb - sqrt(eps())

    ew = eweights(1:size(pr.X, 1), inv(size(pr.X, 1)); scale = true)
    ret = KellyReturn(; w = ew)
    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res1 = optimise!(mre)

    ret = KellyReturn(; lb = mean(log1p.(pr.X * res1.w), ew))
    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res2 = optimise!(mre)
    @test mean(log1p.(pr.X * res2.w)) >= ret.lb - sqrt(eps())

    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res3 = optimise!(mre)
    @test mean(log1p.(pr.X * res3.w)) >= ret.lb - sqrt(eps())

    ucss = (ucs1, ucs2)
    objs = (MinimumRisk(), MaximumRatio(; rf = rf))
    df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk-UncertaintyReturns.csv"), DataFrame)
    i = 1
    for ucs ∈ ucss
        for obj ∈ objs
            ret = ArithmeticReturn(; ucs = ucs)
            opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
            mre = MeanRisk(; obj = obj, opt = opt)
            res1 = optimise!(mre)
            w = res1.w
            wt = df[!, i]
            rtol = 1e-6
            res = isapprox(w, wt; rtol = rtol)
            if !res
                println("Iteration $i failed.")
                find_tol(w, wt; name1 = :w, name2 = :wt)
                display([w wt])
            end
            @test res
            i += 1
        end
    end
end
@testset "Budget" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = BudgetRange(; lb = 0.8, ub = 1.5))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w = optimise!(mre, rd).w
    @test 0.8 <= sum(w) <= 1.5

    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test 0.8 <= sum(w) <= 1.5

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.1, ub = 0.15), bgt = 1.3,
                        sbgt = nothing)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
    @test isapprox(sum(w), 1.3)
    @test isapprox(sum(w[w .< 0]), 0)

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.1, ub = 0.15), bgt = nothing,
                        sbgt = 0.4)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
    @test isapprox(sum(w[w .< 0]), -0.4, rtol = 5e-5)

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.1, ub = 0.15), bgt = 1.3,
                        sbgt = 0.3)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
    @test isapprox(sum(w), 1.3)
    @test isapprox(sum(w[w .< 0]), -0.3, rtol = 5e-5)

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                        bgt = BudgetRange(; lb = 0.6, ub = 0.8), sbgt = 0.25)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
    @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())
    @test 0.6 + 0.25 - sqrt(eps()) <= sum(w[w .> 0]) <= 0.8 + 0.25 + sqrt(eps())

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                        sbgt = BudgetRange(; lb = 0.2, ub = 0.6), bgt = 0.9)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
    @test isapprox(sum(w), 0.9)
    @test -0.6 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                        bgt = BudgetRange(; lb = 0.6, ub = 0.8), sbgt = nothing)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
    @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                        sbgt = BudgetRange(; lb = 0.2, ub = 0.6), bgt = nothing)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
    @test -0.6 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                        bgt = BudgetRange(; lb = 0.6, ub = 0.8),
                        sbgt = BudgetRange(; lb = 0.2, ub = 0.5))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
    @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())
    @test 0.2 + 0.6 - sqrt(eps()) <= sum(w[w .>= 0]) <= 0.8 + 0.5 + sqrt(eps())
    @test -0.5 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                        bgt = BudgetRange(; lb = 0.6, ub = 0.8),
                        sbgt = BudgetRange(; lb = 0.2, ub = nothing))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
    @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())
    @test (0.2 + 0.6 - 375 * sqrt(eps())) <= sum(w[w .>= 0]) <= 0.8 + sqrt(eps())
    @test isapprox(sum(w[w .< 0]), -0.2, rtol = 5e-5)

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                        bgt = BudgetRange(; lb = 0.6, ub = 0.8),
                        sbgt = BudgetRange(; lb = nothing, ub = 0.5))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
    @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())
    @test 0.2 + 0.6 - sqrt(eps()) <= sum(w[w .>= 0]) <= 0.8 + 0.5 + sqrt(eps())
    @test -0.5 - sqrt(eps()) <= sum(w[w .< 0])

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                        bgt = BudgetRange(; lb = nothing, ub = 0.8),
                        sbgt = BudgetRange(; lb = 0.2, ub = 0.5))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
    @test sum(w) < 0.8 + sqrt(eps())
    @test sum(w[w .>= 0]) <= 0.8 + 0.5 + sqrt(eps())
    @test -0.5 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                        bgt = BudgetRange(; lb = 0.6, ub = nothing),
                        sbgt = BudgetRange(; lb = 0.2, ub = 0.5))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
    @test 0.6 - sqrt(eps()) < sum(w)
    @test 0.2 + 0.6 - sqrt(eps()) <= sum(w[w .>= 0])
    @test -0.5 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())
end
@testset "Cardinality" begin
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, card = 5)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test round(inv(dot(w, w))) <= 5

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                        wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1, sbgt = 0.3,
                        card = 7)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    lw = w[w .>= 0]
    sw = w[w .< 0]
    @test round(1.3^2 / dot(lw, lw)) + round(0.3^2 / dot(sw, sw)) <= 7

    gcard = CardinalityConstraint(;
                                  A = CardinalityConstraintSide(; group = :Clusters,
                                                                name = 1), B = 5,
                                  comp = LEQ())
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                        wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1, sbgt = 0.2,
                        gcard = gcard, sets = sets)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    swc = w[sets[!, :Clusters] .== 1]
    @test round(sum(swc) * sum(swc) / dot(swc, swc)) <= 5

    gcard = linear_constraints(gcard, sets)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                        wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1, sbgt = 0.2,
                        gcard = gcard)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    @test isapprox(w, optimise!(mre, rd).w)

    gcard = CardinalityConstraint(;
                                  A = CardinalityConstraintSide(;
                                                                group = [:Clusters,
                                                                         :Clusters,
                                                                         :Clusters],
                                                                name = [1, 2, 3],
                                                                coef = [1, 1, 1]), B = 2,
                                  comp = LEQ())
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                        wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1, sbgt = 0.2,
                        gcard = gcard, sets = sets)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test sum([abs(sum(w[sets[!, :Clusters] .== i])) >= sqrt(eps()) for i ∈ 1:3]) <= 2

    gnames = [:T, :GOOG, :GILD, :MSFT, :NFLX]
    gcard = CardinalityConstraint(;
                                  A = CardinalityConstraintSide(;
                                                                group = [:Assets, :Assets,
                                                                         :Assets, :Assets,
                                                                         :Assets],
                                                                name = gnames,
                                                                coef = [1, 1, 1, 1, 1]),
                                  B = 1, comp = LEQ())
    gcard = linear_constraints(gcard, sets)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, gcard = gcard, sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    idx = [findfirst(x -> x == string(n), rd.nx) for n ∈ gnames]
    @test sum(w[idx] .>= eps()) == 1

    gnames = [:T, :GOOG, :GILD, :MSFT, :NFLX, :TSLA]
    gcard = CardinalityConstraint(;
                                  A = CardinalityConstraintSide(;
                                                                group = [:Assets, :Assets,
                                                                         :Assets, :Assets,
                                                                         :Assets, :Assets],
                                                                name = gnames,
                                                                coef = [1, 1, 1, 1, 1, 1]),
                                  B = 1, comp = LEQ())
    gcard = linear_constraints(gcard, sets)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, gcard = gcard, sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    idx = [findfirst(x -> x == string(n), rd.nx) for n ∈ gnames]
    @test sum(w[idx] .>= sqrt(eps())) == 1

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1, smtx = :Clusters, sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test isapprox([sum(w[res.smtx[i, :]]) for i ∈ axes(res.smtx, 1)], [0, 0, 1])

    sgcard = LinearConstraint(;
                              A = LinearConstraintSide(; group = [:Clusters, :Clusters],
                                                       name = [1, 3],
                                                       coef = [inv(6), inv(11)]), B = 1,
                              comp = LEQ())
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sgcard = sgcard, smtx = :Clusters,
                        sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test sum([sum(w[res.smtx[i, :]]) >= sqrt(eps()) for i ∈ [1, 3]]) <= 1

    sgcard = LinearConstraint(;
                              A = LinearConstraintSide(; group = [:Clusters, :Clusters],
                                                       name = [1, 2],
                                                       coef = [inv(6), inv(13)]), B = 1,
                              comp = LEQ())
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sgcard = sgcard, smtx = :Clusters,
                        sets = sets, wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                        sbgt = 0.2)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test sum([sum(w[res.smtx[i, :]]) >= sqrt(eps()) for i ∈ [1, 2]]) <= 1

    sgcard = LinearConstraint(;
                              A = LinearConstraintSide(; group = [:Clusters, :Clusters],
                                                       name = [1, 3],
                                                       coef = [inv(6), inv(11)]), B = 1,
                              comp = LEQ())
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sgcard = sgcard, smtx = :Clusters,
                        sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test sum([sum(w[res.smtx[i, :]]) >= sqrt(eps()) for i ∈ [1, 3]]) <= 1

    sgcard = LinearConstraint(;
                              A = LinearConstraintSide(; group = [:Clusters, :Clusters],
                                                       name = [2, 3],
                                                       coef = [inv(11), inv(13)]), B = 1,
                              comp = LEQ())
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sgcard = sgcard, smtx = :Clusters,
                        sets = sets, wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                        sbgt = 0.2)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test sum([sum(w[res.smtx[i, :]]) >= sqrt(eps()) for i ∈ [2, 3]]) <= 1

    plc = IntegerPhilogenyConstraintEstimator(; pe = NetworkEstimator(), B = 1)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, bgt = 1, cplg = plc,
                        wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.2)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w

    res.cplg.A
    @test all(res.cplg.A * value.(res.model[:ib]) .<= 1)

    plc = IntegerPhilogenyConstraintEstimator(; pe = ClusteringEstimator(), B = 2)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, bgt = 1, nplg = plc,
                        wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.2)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test all(res.nplg.A * value.(res.model[:ib]) .<= 2)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = 0.2)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test all(w[w .>= sqrt(eps())] .>= 0.2 - sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = 0.5)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test all(w[w .>= sqrt(eps())] .>= 0.5 - sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = 0.2, st = 0.2, sbgt = 1, bgt = 1,
                        wb = WeightBoundsResult(; lb = -1, ub = 1))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test all(w[w .>= sqrt(eps())] .>= 0.2 - sqrt(eps()))
    @test all(w[w .<= -sqrt(eps())] .<= -0.2 + sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = 0.5, st = 0.5, sbgt = 1, bgt = 1,
                        wb = WeightBoundsResult(; lb = -1, ub = 1))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test all(w[w .>= sqrt(eps())] .>= 0.5 - sqrt(eps()))
    @test all(w[w .<= -sqrt(eps())] .<= -0.5 + sqrt(eps()))
end
@testset "L1 and L2 penalties" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w0 = optimise!(mre, rd).w

    opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 5e-6,
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w1 = optimise!(mre, rd).w

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l1 = 1e-4,
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w2 = optimise!(mre, rd).w
    @test sum(w0[w0 .< 0]) <= sum(w1[w1 .< 0]) <= sum(w2[w2 .< 0])

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w3 = optimise!(mre, rd).w

    opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 1e-4,
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w4 = optimise!(mre, rd).w

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l1 = 1e-3,
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w5 = optimise!(mre, rd).w
    @test sum(w3[w3 .< 0]) <= sum(w4[w4 .< 0]) <= sum(w5[w5 .< 0])

    N = size(pr.X, 2)
    wt = eqw
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l2 = 1e-4,
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w6 = optimise!(mre, rd).w

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l2 = 1e-3,
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w7 = optimise!(mre, rd).w
    @test rmsd(w0, wt) >= rmsd(w6, wt) >= rmsd(w7, wt)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l2 = 1e-2,
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w8 = optimise!(mre, rd).w

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l2 = 1e-2,
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w9 = optimise!(mre, rd).w
    @test rmsd(w3, wt) >= rmsd(w8, wt) >= rmsd(w9, wt)
end
@testset "Fees" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; l = 10),
                        wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .>= 5e-5) == 1
    @test isapprox(sum(res.w[res.w .>= 5e-5]), 1, rtol = 5.0e-5)

    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; s = 10),
                        wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.6, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .<= -5e-5) == 3
    @test isapprox(sum(res.w[res.w .<= -5e-5]), -0.6, rtol = 5e-4)

    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; l = 0.0034375),
                        wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                   opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .>= sqrt(eps())) == 1
    @test isapprox(sum(res.w[res.w .>= sqrt(eps())]), 1)

    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; s = 0.008056640625),
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                   opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .<= -sqrt(eps())) == 1
    @test isapprox(sum(res.w[res.w .<= -sqrt(eps())]), -1)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fl = 10),
                        wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .>= sqrt(eps())) == 1
    @test isapprox(sum(res.w[res.w .>= sqrt(eps())]), 1)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fs = 10),
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .<= -sqrt(eps())) == 0
    @test isapprox(sum(res.w[res.w .<= -sqrt(eps())]), 0)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fl = 10),
                        wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                   opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .>= sqrt(eps())) == 1
    @test isapprox(sum(res.w[res.w .>= sqrt(eps())]), 1)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fs = 10),
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                   opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .<= -sqrt(eps())) == 0
    @test isapprox(sum(res.w[res.w .<= -sqrt(eps())]), 0)

    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; l = 10), ret = KellyReturn(),
                        wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .>= 5e-5) == 1
    @test isapprox(sum(res.w[res.w .>= 5e-5]), 1, rtol = 5e-5)

    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; s = 10), ret = KellyReturn(),
                        wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.6, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .<= -5e-5) == 3
    @test isapprox(sum(res.w[res.w .<= -5e-5]), -0.6, rtol = 5e-4)

    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; l = 10), ret = KellyReturn(),
                        wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                   opt = opt)
    res = optimise!(mre, rd)
    @test rmsd(res.w, eqw) <= 0.015

    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; s = 10), ret = KellyReturn(),
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                   opt = opt)
    res = optimise!(mre, rd)
    @test rmsd(res.w, eqw) <= 0.017

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fl = 10),
                        ret = KellyReturn(), wb = WeightBoundsResult(; lb = 0, ub = 1),
                        sbgt = 0, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .>= 5e-5) == 1
    @test isapprox(sum(res.w[res.w .>= 5e-5]), 1, rtol = 5e-5)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fs = 10),
                        ret = KellyReturn(), wb = WeightBoundsResult(; lb = -0.2, ub = 1),
                        sbgt = 0.6, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .<= -5e-5) == 0
    @test isapprox(sum(res.w[res.w .<= -5e-5]), 0, rtol = 5e-4)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fl = 10),
                        ret = KellyReturn(), wb = WeightBoundsResult(; lb = 0, ub = 1),
                        sbgt = 0, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                   opt = opt)
    res = optimise!(mre, rd)
    @test rmsd(res.w, eqw) <= 0.46

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fs = 10),
                        ret = KellyReturn(), wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1)
    mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                   opt = opt)
    res = optimise!(mre, rd)
    @test count(res.w .<= -5e-5) == 0
    @test isapprox(sum(res.w[res.w .<= -5e-5]), 0, rtol = 5e-4)
end
@testset "Cone constraints" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, nea = 7.5,
                        wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
    mre = MeanRisk(; obj = MaximumReturn(), opt = opt)
    w = optimise!(mre, rd).w
    @test isapprox(number_effective_assets(w), opt.nea, rtol = 1e-7)

    mre = MeanRisk(; obj = MaximumRatio(), opt = opt)
    w = optimise!(mre, rd).w
    @test isapprox(number_effective_assets(w), opt.nea, rtol = 5e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, tn = Turnover(; w = eqw, val = 0))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w = optimise!(mre, rd).w
    @test isapprox(w, eqw)
    @test all(abs.(w - eqw) .<= sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, tn = Turnover(; w = eqw, val = 3e-2))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w = optimise!(mre, rd).w
    @test all(abs.(w - eqw) .- 3e-2 .<= sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, tn = Turnover(; w = eqw, val = 1e-1))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test all(abs.(w - eqw) .- 1e-1 .<= sqrt(eps()))

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1,
                        te = TrackingError(; err = 0,
                                           tracking = WeightsTracking(; w = eqw)))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w = optimise!(mre, rd).w
    @test isapprox(norm(pr.X * (eqw - w), 2) / sqrt(T), 0, atol = 1e-10)

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1,
                        te = TrackingError(; err = 6e-3,
                                           tracking = WeightsTracking(; w = eqw)))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w = optimise!(mre, rd).w
    @test norm(pr.X * (w - eqw), 2) / sqrt(T) <= 6e-3

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1,
                        te = TrackingError(; err = 6e-3,
                                           tracking = WeightsTracking(; w = eqw)))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test norm(pr.X * (w - eqw), 2) / sqrt(T) <= 6e-3

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1,
                        te = TrackingError(; formulation = NOCTracking(), err = 0,
                                           tracking = WeightsTracking(; w = eqw)))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w = optimise!(mre, rd).w
    @test isapprox(norm(pr.X * (eqw - w), 1) / T, 0, atol = 1e-10)

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1,
                        te = TrackingError(; formulation = NOCTracking(), err = 6e-3,
                                           tracking = WeightsTracking(; w = eqw)))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w = optimise!(mre, rd).w
    @test norm(pr.X * (w - eqw), 1) / T <= 6e-3

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1,
                        te = TrackingError(; formulation = NOCTracking(), err = 6e-3,
                                           tracking = WeightsTracking(; w = eqw)))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test norm(pr.X * (w - eqw), 1) / T <= 6e-3
end
@testset "Linear constraints" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w0 = optimise!(mre, rd).w

    lcs = [LinearConstraint(;
                            A = LinearConstraintSide(; group = :Assets, name = :ADI,
                                                     coef = 1), B = -0.05, comp = EQ()),
           LinearConstraint(;
                            A = LinearConstraintSide(; group = [:Assets, :Assets],
                                                     name = [:ADI, :TXN], coef = [0.5, -1]),
                            B = 0, comp = LEQ()),
           LinearConstraint(;
                            A = LinearConstraintSide(; group = [:Assets, :Assets, :Assets],
                                                     name = [:ADP, :AMGN, :MSFT],
                                                     coef = [1, 1, -1]), comp = GEQ(),
                            B = 0)]
    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, lcs = lcs, sets = sets)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w = optimise!(mre, rd).w
    @test isapprox(w[rd.nx .== "ADI"][1], -0.05)
    @test 0.5 * w[rd.nx .== "ADI"][1] <= w[rd.nx .== "TXN"][1]
    @test w[rd.nx .== "ADP"][1] + w[rd.nx .== "AMGN"][1] >= w[rd.nx .== "MSFT"][1]

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, lcs = linear_constraints(lcs, sets))
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test isapprox(w[rd.nx .== "ADI"][1], -0.05)
    @test 0.5 * w[rd.nx .== "ADI"][1] <= w[rd.nx .== "TXN"][1]
    @test w[rd.nx .== "ADP"][1] + w[rd.nx .== "AMGN"][1] >= w[rd.nx .== "MSFT"][1]
end
#=
@testset "SDP philogeny constraints" begin
    plc = SemiDefinitePhilogenyConstraintEstimator(; pe = NetworkEstimator())
    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -0.2, ub = 1),
                        bgt = 1, sbgt = 0.2, cplg = plc)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test isapprox(w,
                   [1.9182464594380404e-9, -5.486929531582411e-9, 0.25294011598798327,
                    0.30736755395924736, 1.4314131117487336e-9, -0.035502895237015414,
                    -1.8881132164854558e-10, 0.17572595977320954, 0.29946926729917145,
                    5.434866179443442e-10])

    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf), opt = opt)
    w = optimise!(mre, rd).w
    @test isapprox(w,
                   [-2.1268204115045244e-9, -0.06023046002084164, 0.2601463681020507,
                    0.4010094500291993, -1.499386434008056e-10, -0.13976952342299712,
                    -1.5207511190183943e-8, 0.23777409644795705, 0.3010700864415278,
                    -9.263056126010764e-11])

    plc = SemiDefinitePhilogenyConstraintEstimator(; pe = NetworkEstimator(), p = 6)
    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -0.2, ub = 1),
                        bgt = 1, sbgt = 0.2, nplg = philogeny_constraints(plc, pr.X))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    w = optimise!(mre, rd).wprintln(card_flag, "\n", gcard_flag, "\n", n_flag, "\n", c_flag,
                                    "\n", lt_flag, "\n", st_flag, "\n", ffl_flag, "\n",
                                    ffs_flag, "\n")

    mre = MeanRisk(;
                   r = UncertaintySetVariance(;
                                              ucs = NormalUncertaintySetEstimator(;
                                                                                  pe = EmpiricalPriorEstimator(),
                                                                                  rng = rng,
                                                                                  alg = BoxUncertaintySetAlgorithm(),
                                                                                  seed = 987654321)),
                   obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    @test !haskey(res.model, :sdp_nplg_p)

    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBoundsResult(; lb = -0.2, ub = 1),
                        bgt = 1, sbgt = 0.2, cplg = philogeny_constraints(plc, pr.X))
    mre = MeanRisk(;
                   r = UncertaintySetVariance(;
                                              ucs = sigma_ucs(NormalUncertaintySetEstimator(;
                                                                                            pe = EmpiricalPriorEstimator(),
                                                                                            rng = rng,
                                                                                            alg = BoxUncertaintySetAlgorithm(),
                                                                                            seed = 987654321),
                                                              pr.X)), obj = MinimumRisk(),
                   opt = opt)
    res = optimise!(mre, rd)
    @test !haskey(res.model, :sdp_cplg_p)
end
@testset "Centrality" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
    ce = CentralityConstraintEstimator(; A = CentralityEstimator(), B = MinValue(),
                                       comp = LEQ())
    sets = DataFrame(; Assets = 1:10, Clusters = [1, 2, 2, 2, 3, 2, 2, 1, 3, 3])
    opt = JuMPOptimiser(; pe = pr, slv = slv, cent = ce, sets = sets)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test all(abs.(res.cent.A_ineq * w .- res.cent.B_ineq) .<= sqrt(eps()))
    @test isapprox(w,
                   [0.22435864236807318, 0.11900511795941895, 0.2042024901056272,
                    0.21713501321050613, -1.1594442746740814e-10, 0.23529873506572008,
                    -1.461171082669015e-10, -1.3517306778692135e-10, 1.1400266353179912e-9,
                    5.478625248874811e-10])

    ce = [CentralityConstraintEstimator(; A = CentralityEstimator(), B = MedianValue(),
                                        comp = EQ())]
    opt = JuMPOptimiser(; pe = pr, slv = slv, cent = ce, sets = sets)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise!(mre, rd)
    w = res.w
    @test all(abs.(res.cent.A_eq * w .- res.cent.B_eq) .<= sqrt(eps()))
    @test isapprox(w,
                   [1.5080697736395485e-9, 1.0676961354967205e-10, 0.20143237340958933,
                    0.37944774611460697, 2.08976824723604e-9, 4.915607815862469e-10,
                    3.260477423319088e-10, 0.08088011921478026, 0.3382397556825816,
                    1.0562258217209712e-9])
end
=#
# end
