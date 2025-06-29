@safetestset "NearOptimalCentering Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, Random, Clarabel, TimeSeries, JuMP,
          Pajarito, HiGHS, StableRNGs, StatsBase
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
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
                     solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                        "verbose" => false,
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
    T, N = size(pr.X)
    eqw = range(; start = inv(N), stop = inv(N), length = N)
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(; rf = rf), MaximumReturn()]
    bins = [1, 5, 10, 20, nothing, 50]
    @testset "Efficient Frontier" begin
        rng = StableRNG(987456321)
        ucs1 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                       rng = rng,
                                                       alg = BoxUncertaintySetAlgorithm(),
                                                       seed = 987654321), pr.X)
        pw = pweights(collect(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                                    length = size(pr.X, 1))))
        rets1 = [ArithmeticReturn(), KellyReturn()]
        rets2 = [ArithmeticReturn(; lb = Frontier(; N = 3)),
                 KellyReturn(; lb = Frontier(; N = 3))]
        risks1 = [StandardDeviation(;),
                  LowOrderMoment(;
                                 alg = LowOrderDeviation(;
                                                         alg = SecondLowerMoment(;
                                                                                 formulation = SqrtRiskExpr()))),
                  LowOrderMoment(;
                                 alg = LowOrderDeviation(;
                                                         alg = SecondCentralMoment(;
                                                                                   formulation = SqrtRiskExpr()))),
                  NegativeSkewness(; settings = RiskMeasureSettings(;)),
                  ConditionalValueatRisk(;)]
        risks2 = [StandardDeviation(;
                                    settings = RiskMeasureSettings(;
                                                                   ub = Frontier(; N = 3))),
                  LowOrderMoment(; settings = RiskMeasureSettings(; ub = Frontier(; N = 3)),
                                 alg = LowOrderDeviation(;
                                                         alg = SecondLowerMoment(;
                                                                                 formulation = SqrtRiskExpr()))),
                  LowOrderMoment(; settings = RiskMeasureSettings(; ub = Frontier(; N = 3)),
                                 alg = LowOrderDeviation(;
                                                         alg = SecondCentralMoment(;
                                                                                   formulation = SqrtRiskExpr()))),
                  NegativeSkewness(;
                                   settings = RiskMeasureSettings(; ub = Frontier(; N = 3))),
                  ConditionalValueatRisk(;
                                         settings = RiskMeasureSettings(;
                                                                        ub = Frontier(;
                                                                                      N = 3)))]
        slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false)),
               Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
               Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
               Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false, "max_step_fraction" => 0.9,
                                      "max_iter" => 500, "equilibrate_min_scaling" => 1e-5,
                                      "equilibrate_max_scaling" => 1e5,
                                      "linesearch_backtrack_step" => 0.95,
                                      "equilibrate_max_iter" => 100))]
        i = 1
        for (ret1, ret2) ∈ zip(rets1, rets2)
            opt1 = JuMPOptimiser(; pe = pr, ret = ret1, slv = slv)
            opt2 = JuMPOptimiser(; pe = pr, ret = ret2, slv = slv)
            for (r1, r2) ∈ zip(risks1, risks2)
                sol_min = optimise!(NearOptimalCentering(; r = r1, obj = MinimumRisk(),
                                                         opt = opt1))
                w_min = sol_min.w
                sol_max = optimise!(NearOptimalCentering(; r = r1, obj = MaximumReturn(),
                                                         opt = opt1))
                w_max = sol_max.w
                sol_fnt1 = optimise!(NearOptimalCentering(; r = r1, obj = MinimumRisk(),
                                                          opt = opt2))
                w_fnt1 = sol_fnt1.w
                sol_fnt2 = optimise!(NearOptimalCentering(; r = r2, obj = MaximumReturn(),
                                                          opt = opt1))
                w_fnt2 = sol_fnt2.w
                r1 = PortfolioOptimisers.factory(r1, pr, slv)
                r2 = PortfolioOptimisers.factory(r2, pr, slv)
                rk_min = expected_risk(r1, w_min, pr.X)
                rk_max = expected_risk(r1, w_max, pr.X)
                rk_fnt = expected_risk(r1, w_fnt1, pr.X)
                rt_min = expected_return(ret1, w_min, pr)
                rt_max = expected_return(ret1, w_max, pr)
                rt_fnt = expected_return(ret1, w_fnt1, pr)
                rk_rtol = if i ∈ (1, 3, 6)
                    5e-6
                elseif i == 2
                    1e-5
                elseif i ∈ (4, 8)
                    5e-5
                elseif i == 5
                    5e-2
                elseif i == 9
                    5e-4
                elseif i == 7
                    1e-4
                else
                    1e-6
                end
                rt_rtol = if i == 10
                    5e-6
                elseif i ∈ (2, 8, 9)
                    5e-4
                elseif i == 1
                    1e-5
                elseif i ∈ (3, 4)
                    5e-5
                elseif i == 5
                    5e-3
                elseif i == 7
                    5e-3
                elseif i == 6
                    1e-3
                else
                    1e-6
                end
                res = isapprox(rk_fnt[1], rk_min; rtol = rk_rtol)
                if !res
                    println(i)
                    find_tol(rk_fnt[1], rk_min; name1 = "rk_fnt[1]", name2 = "rk_min")
                end
                @test res
                @test rk_min <= rk_fnt[2] <= rk_max
                res = isapprox(rk_fnt[3], rk_max; rtol = rk_rtol)
                if !res
                    find_tol(rk_fnt[3], rk_max; name1 = "rk_fnt[3]", name2 = "rk_max")
                end
                @test res
                res = isapprox(rt_fnt[1], rt_min; rtol = rt_rtol)
                if !res
                    println(i)
                    find_tol(rt_fnt[1], rt_min; name1 = "rt_fnt[1]", name2 = "rt_min")
                end
                @test res
                @test rt_min <= rt_fnt[2] <= rt_max
                res = isapprox(rt_fnt[3], rt_max; rtol = rt_rtol)
                if !res
                    println(i)
                    find_tol(rt_fnt[3], rt_max; name1 = "rt_fnt[3]", name2 = "rt_max")
                end
                @test res
                i += 1
            end
        end
    end
    @testset "NearOptimalCentering" begin
        w_min = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(),
                                   opt = opt), rd).w
        w_max = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumReturn(),
                                   opt = opt), rd).w
        df = CSV.read(joinpath(@__DIR__, "./assets/Unconstrained-NearOptimalCentering.csv"),
                      DataFrame)
        i = 1
        for obj ∈ objs
            for bin ∈ bins
                wt = df[!, i]
                noc1 = NearOptimalCentering(; bins = bin, r = ConditionalValueatRisk(),
                                            obj = obj, opt = opt)
                w1 = optimise!(noc1, rd).w
                res1 = if i ∈ (1, 4, 5, 7, 13, 14, 15, 19)
                    isapprox(w1, wt; rtol = 1e-4)
                elseif i ∈ (2, 3, 6, 8, 9, 11)
                    isapprox(w1, wt; rtol = 5e-4)
                elseif i ∈ (10, 16, 17, 20)
                    isapprox(w1, wt; rtol = 5e-5)
                elseif i ∈ (12, 18, 21)
                    isapprox(w1, wt; rtol = 5e-6)
                elseif i ∈ (22, 23)
                    isapprox(w1, wt; rtol = 1e-5)
                else
                    isapprox(w1, wt; rtol = 1e-6)
                end
                if !res1
                    println("NOC unconstrained failed: iter: $i\nobj: $obj\nbin: $bin.")
                    find_tol(w1, wt; name1 = :w1, name2 = :wt)
                end
                @test res1

                w_opt = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = obj,
                                           opt = opt), rd).w
                noc2 = NearOptimalCentering(; w_min = w_min, w_max = w_max, w_opt = w_opt,
                                            bins = bin, r = ConditionalValueatRisk(),
                                            obj = obj, opt = opt)
                w2 = optimise!(noc2, rd).w
                res2 = isapprox(w2, w1)
                if !res2
                    println("NOC unconstrained initial values failed: iter: $i\nobj: $obj\nbin: $bin.")
                    find_tol(w2, w1; name1 = :w2, name2 = :w1)
                end
                @test res2

                noc3 = NearOptimalCentering(;
                                            alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                            bins = bin, r = ConditionalValueatRisk(),
                                            obj = obj, opt = opt)
                w3 = optimise!(noc3, rd).w
                res3 = isapprox(w3, w1)
                if !res3
                    println("NOC constrained failed: iter: $i\nobj: $obj\nbin: $bin.")
                    find_tol(w3, w1; name1 = :w3, name2 = :wt)
                end
                @test res3

                w_opt = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = obj,
                                           opt = opt), rd).w
                noc4 = NearOptimalCentering(;
                                            alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                            w_min = w_min, w_max = w_max, w_opt = w_opt,
                                            bins = bin, r = ConditionalValueatRisk(),
                                            obj = obj, opt = opt)
                w4 = optimise!(noc4, rd).w
                w4 = optimise!(noc4, rd).w
                res2 = isapprox(w4, w2)
                if !res2
                    println("NOC constrained initial values failed: iter: $i\nobj: $obj\nbin: $bin.")
                    find_tol(w4, w2; name1 = :w4, name2 = :w3)
                end
                @test res2
                i += 1
            end
        end
        r = ConditionalValueatRisk()
        mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w
        ub = expected_risk(r, w1, rd.X)
        lb = expected_return(ArithmeticReturn(), w1, pr)

        noc1 = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w2 = optimise!(noc1, rd).w

        noc2 = NearOptimalCentering(;
                                    r = ConditionalValueatRisk(;
                                                               settings = RiskMeasureSettings(;
                                                                                              ub = ub)),
                                    obj = MaximumReturn(), opt = opt)
        w3 = optimise!(noc2, rd).w
        @test isapprox(w2, w3, rtol = 0.5)

        opt = JuMPOptimiser(; ret = ArithmeticReturn(; lb = lb), pe = pr, slv = slv)
        noc3 = NearOptimalCentering(; r = r, obj = MinimumRisk(), opt = opt)
        w4 = optimise!(noc3, rd).w
        @test isapprox(w2, w4, rtol = 5e-4)

        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = ConditionalValueatRisk()
        mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w

        r = ConditionalDrawdownatRisk()
        mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w2 = optimise!(mr, rd).w

        r = ConditionalValueatRisk()
        mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w
        ub = expected_risk(r, w1, rd.X)
        lb = expected_return(ArithmeticReturn(), w1, pr)

        noc1 = NearOptimalCentering(; alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                    r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w2 = optimise!(noc1, rd).w

        noc2 = NearOptimalCentering(; alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                    r = ConditionalValueatRisk(;
                                                               settings = RiskMeasureSettings(;
                                                                                              ub = ub)),
                                    obj = MaximumReturn(), opt = opt)
        sol = optimise!(noc2, rd)
        @test value(sol.model[:cvar_risk_1]) <= ub + sqrt(eps())

        opt = JuMPOptimiser(; ret = ArithmeticReturn(; lb = lb), pe = pr, slv = slv)
        noc3 = NearOptimalCentering(; alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                    r = r, obj = MinimumRisk(), opt = opt)
        sol = optimise!(noc3, rd)
        @test value(sol.model[:ret]) >= lb - sqrt(eps())
    end

    @testset "Scalarisers" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w
        @test isapprox(w1,
                       [0.006879323942612698, 0.011528211408637554, 0.010260412140232775,
                        0.018382683329409316, 0.009127291978483786, 0.010761818638247933,
                        0.020207341361297926, 0.00922666167356374, 0.017779308743719347,
                        0.017692788756226114, 0.058397531196042005, 0.009343801851355197,
                        0.012162201004275433, 0.0209180174690969, 0.011203550267400927,
                        0.012425255664557402, 0.01667289478834348, 0.005956860095134718,
                        0.011185665217225929, 0.25765219470828155, 0.02078675880421926,
                        0.02385481833356798, 0.014376852276222856, 0.00703008032762147,
                        0.013449081435994293, 0.17700511436847838, 0.03729883908556524,
                        0.13764501268074134, 0.00950406983452321, 0.011285556107478728],
                       rtol = 5e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = MaxScalariser())
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w2 = optimise!(mr, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser(; gamma = 1e-2))
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w3 = optimise!(mr, rd).w
        @test isapprox(w3, w1, rtol = 5e-4)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            sce = LogSumExpScalariser(; gamma = 2.907e1))
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w4 = optimise!(mr, rd).w
        @test isapprox(w4, w2, rtol = 1.3e-1)
    end
end
