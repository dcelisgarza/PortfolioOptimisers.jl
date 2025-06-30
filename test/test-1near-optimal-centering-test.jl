@safetestset "NearOptimalCentering Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, Random, Clarabel, TimeSeries, JuMP,
          Pajarito, HiGHS, StableRNGs, StatsBase
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    function get_rtol(a, b)
        rtol = 0.0
        for rtol in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
            res = isapprox(a, b; rtol = rtol)
            if res
                rtol = _rtol
                break
            end
        end
        return rtol
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
        for (ret1, ret2) in zip(rets1, rets2)
            opt1 = JuMPOptimiser(; pe = pr, ret = ret1, slv = slv)
            opt2 = JuMPOptimiser(; pe = pr, ret = ret2, slv = slv)
            for (r1, r2) in zip(risks1, risks2)
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
                sol_fnt3 = optimise!(NearOptimalCentering(;
                                                          alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                                          r = r1, obj = MinimumRisk(),
                                                          opt = opt2))
                w_fnt3 = sol_fnt3.w
                sol_fnt4 = optimise!(NearOptimalCentering(;
                                                          alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                                          r = r2, obj = MaximumReturn(),
                                                          opt = opt1))
                w_fnt4 = sol_fnt4.w
                r1 = PortfolioOptimisers.factory(r1, pr, slv)
                r2 = PortfolioOptimisers.factory(r2, pr, slv)
                rk_min = expected_risk(r1, w_min, pr.X)
                rk_max = expected_risk(r1, w_max, pr.X)
                rk_fnt_1 = expected_risk(r1, w_fnt1, pr.X)
                rk_fnt_2 = expected_risk(r1, w_fnt2, pr.X)
                rk_fnt_3 = expected_risk(r1, w_fnt3, pr.X)
                rk_fnt_4 = expected_risk(r1, w_fnt4, pr.X)
                rt_min = expected_return(ret1, w_min, pr)
                rt_max = expected_return(ret1, w_max, pr)
                rt_fnt_1 = expected_return(ret1, w_fnt1, pr)
                rt_fnt_2 = expected_return(ret1, w_fnt2, pr)
                rt_fnt_3 = expected_return(ret1, w_fnt3, pr)
                rt_fnt_4 = expected_return(ret1, w_fnt4, pr)
                rtols = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
                rk_rtol_1 = max(get_rtol(rk_fnt_1[1], rk_min),
                                get_rtol(rk_fnt_1[3], rk_max))
                rt_rtol_1 = max(get_rtol(rt_fnt_1[1], rt_min),
                                get_rtol(rt_fnt_1[3], rt_max))
                rk_rtol_2 = max(get_rtol(rk_fnt_2[1], rk_min),
                                get_rtol(rk_fnt_2[3], rk_max))
                rt_rtol_2 = max(get_rtol(rt_fnt_2[1], rt_min),
                                get_rtol(rt_fnt_2[3], rt_max))
                rk_rtol_3 = max(get_rtol(rk_fnt_3[1], rk_min),
                                get_rtol(rk_fnt_3[3], rk_max))
                rt_rtol_3 = max(get_rtol(rt_fnt_3[1], rt_min),
                                get_rtol(rt_fnt_3[3], rt_max))
                rk_rtol_4 = max(get_rtol(rk_fnt_4[1], rk_min),
                                get_rtol(rk_fnt_4[3], rk_max))
                rt_rtol_4 = max(get_rtol(rt_fnt_4[1], rt_min),
                                get_rtol(rt_fnt_4[3], rt_max))
                res = isapprox(rk_fnt_1[1], rk_min; rtol = rk_rtol_1)
                if !res
                    println(i)
                    find_tol(rk_fnt_1[1], rk_min; name1 = "rk_fnt_1[1]", name2 = "rk_min")
                end
                @test res
                @test rk_min <= rk_fnt_1[2] <= rk_max
                res = isapprox(rk_fnt_1[3], rk_max; rtol = rk_rtol_1)
                if !res
                    find_tol(rk_fnt_1[3], rk_max; name1 = "rk_fnt_1[3]", name2 = "rk_max")
                end
                @test res
                res = isapprox(rt_fnt_1[1], rt_min; rtol = rt_rtol_1)
                if !res
                    println(i)
                    find_tol(rt_fnt_1[1], rt_min; name1 = "rt_fnt_1[1]", name2 = "rt_min")
                end
                @test res
                @test rt_min <= rt_fnt_1[2] <= rt_max
                res = isapprox(rt_fnt_1[3], rt_max; rtol = rt_rtol_1)
                if !res
                    println(i)
                    find_tol(rt_fnt_1[3], rt_max; name1 = "rt_fnt_1[3]", name2 = "rt_max")
                end
                @test res
                res = isapprox(rk_fnt_2[1], rk_min; rtol = rk_rtol_2)
                if !res
                    println(i)
                    find_tol(rk_fnt_2[1], rk_min; name1 = "rk_fnt_2[1]", name2 = "rk_min")
                end
                @test res
                @test rk_min <= rk_fnt_2[2] <= rk_max
                res = isapprox(rk_fnt_2[3], rk_max; rtol = rk_rtol_2)
                if !res
                    find_tol(rk_fnt_2[3], rk_max; name1 = "rk_fnt_2[3]", name2 = "rk_max")
                end
                @test res
                res = isapprox(rt_fnt_2[1], rt_min; rtol = rt_rtol_2)
                if !res
                    println(i)
                    find_tol(rt_fnt_2[1], rt_min; name1 = "rt_fnt_2[1]", name2 = "rt_min")
                end
                @test res
                @test rt_min <= rt_fnt_2[2] <= rt_max
                res = isapprox(rt_fnt_2[3], rt_max; rtol = rt_rtol_2)
                if !res
                    println(i)
                    find_tol(rt_fnt_2[3], rt_max; name1 = "rt_fnt_2[3]", name2 = "rt_max")
                end
                @test res
                res = isapprox(rk_fnt_3[1], rk_min; rtol = rk_rtol_3)
                if !res
                    println(i)
                    find_tol(rk_fnt_3[1], rk_min; name1 = "rk_fnt_3[1]", name2 = "rk_min")
                end
                @test res
                @test rk_min <= rk_fnt_3[2] <= rk_max
                res = isapprox(rk_fnt_3[3], rk_max; rtol = rk_rtol_3)
                if !res
                    find_tol(rk_fnt_3[3], rk_max; name1 = "rk_fnt_3[3]", name2 = "rk_max")
                end
                @test res
                res = isapprox(rt_fnt_3[1], rt_min; rtol = rt_rtol_3)
                if !res
                    println(i)
                    find_tol(rt_fnt_3[1], rt_min; name1 = "rt_fnt_3[1]", name2 = "rt_min")
                end
                @test res
                @test rt_min <= rt_fnt_3[2] <= rt_max
                res = isapprox(rt_fnt_3[3], rt_max; rtol = rt_rtol_3)
                if !res
                    println(i)
                    find_tol(rt_fnt_3[3], rt_max; name1 = "rt_fnt_3[3]", name2 = "rt_max")
                end
                @test res
                res = isapprox(rk_fnt_4[1], rk_min; rtol = rk_rtol_4)
                if !res
                    println(i)
                    find_tol(rk_fnt_4[1], rk_min; name1 = "rk_fnt_4[1]", name2 = "rk_min")
                end
                @test res
                @test rk_min <= rk_fnt_4[2] <= rk_max
                res = isapprox(rk_fnt_4[3], rk_max; rtol = rk_rtol_4)
                if !res
                    find_tol(rk_fnt_4[3], rk_max; name1 = "rk_fnt_4[3]", name2 = "rk_max")
                end
                @test res
                res = isapprox(rt_fnt_4[1], rt_min; rtol = rt_rtol_4)
                if !res
                    println(i)
                    find_tol(rt_fnt_4[1], rt_min; name1 = "rt_fnt_4[1]", name2 = "rt_min")
                end
                @test res
                @test rt_min <= rt_fnt_4[2] <= rt_max
                res = isapprox(rt_fnt_4[3], rt_max; rtol = rt_rtol_4)
                if !res
                    println(i)
                    find_tol(rt_fnt_4[3], rt_max; name1 = "rt_fnt_4[3]", name2 = "rt_max")
                end
                @test res
                i += 1
            end
        end
        risks3 = [StandardDeviation(;
                                    settings = RiskMeasureSettings(;
                                                                   ub = range(;
                                                                              start = sqrt(6.866856440213463e-5),
                                                                              stop = sqrt(0.0020770775356002027),
                                                                              length = 3))),
                  LowOrderMoment(;
                                 settings = RiskMeasureSettings(;
                                                                ub = range(;
                                                                           start = 0.0049444908053744695,
                                                                           stop = 0.02159931505620857,
                                                                           length = 3)),
                                 alg = LowOrderDeviation(;
                                                         alg = SecondLowerMoment(;
                                                                                 formulation = SqrtRiskExpr()))),
                  LowOrderMoment(;
                                 settings = RiskMeasureSettings(;
                                                                ub = range(;
                                                                           start = 0.0066631114898292945,
                                                                           stop = 0.030531423662781672,
                                                                           length = 3)),
                                 alg = LowOrderDeviation(;
                                                         alg = SecondCentralMoment(;
                                                                                   formulation = SqrtRiskExpr()))),
                  NegativeSkewness(;
                                   settings = RiskMeasureSettings(;
                                                                  ub = range(;
                                                                             start = 0.0004845554751596269,
                                                                             stop = 0.00400658322849469,
                                                                             length = 3))),
                  ConditionalValueatRisk(;
                                         settings = RiskMeasureSettings(;
                                                                        ub = range(;
                                                                                   start = 0.013440511085279036,
                                                                                   stop = 0.06941842129425456,
                                                                                   length = 3)))]
        ret1 = rets1[1]
        opt1 = JuMPOptimiser(; pe = pr, ret = ret1, slv = slv)
        i = 1
        for r1 in risks3
            sol_fnt_1 = optimise!(NearOptimalCentering(; r = r1, obj = MaximumReturn(),
                                                       opt = opt1))
            w_fnt_1 = sol_fnt_1.w
            sol_fnt_2 = optimise!(NearOptimalCentering(;
                                                       alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                                       r = r1, obj = MaximumReturn(),
                                                       opt = opt1))
            w_fnt_2 = sol_fnt_2.w
            r1 = PortfolioOptimisers.factory(r1, pr, slv)
            rk_fnt_1 = expected_risk(r1, w_fnt_1, pr.X)
            rk_fnt_2 = expected_risk(r1, w_fnt_2, pr.X)
            ub = r1.settings.ub
            rtol_1 = max(get_rtol(rk_fnt_1[1], ub[1]), get_rtol(rk_fnt_1[2], ub[2]),
                         get_rtol(rk_fnt_1[3], ub[3]))
            rtol_2 = max(get_rtol(rk_fnt_2[1], ub[1]), get_rtol(rk_fnt_2[2], ub[2]),
                         get_rtol(rk_fnt_2[3], ub[3]))
            res = isapprox(rk_fnt_1[1], ub[1]; rtol = rtol_1)
            if !res
                println(i)
                find_tol(rk_fnt_1[1], ub[1]; name1 = "rk_fnt_1[1]", name2 = "ub[1]")
            end
            @test res
            res = isapprox(rk_fnt_1[2], ub[2]; rtol = rtol_1)
            if !res
                println(i)
                find_tol(rk_fnt_1[2], ub[2]; name1 = "rk_fnt_1[2]", name2 = "ub[2]")
            end
            @test res
            res = isapprox(rk_fnt_1[3], ub[3]; rtol = rtol_1)
            if !res
                println(i)
                find_tol(rk_fnt_1[3], ub[3]; name1 = "rk_fnt_1[3]", name2 = "ub[3]")
            end
            @test res
            res = isapprox(rk_fnt_2[1], ub[1]; rtol = rtol_2)
            if !res
                println(i)
                find_tol(rk_fnt_2[1], ub[1]; name1 = "rk_fnt_2[1]", name2 = "ub[1]")
            end
            @test res
            res = isapprox(rk_fnt_2[2], ub[2]; rtol = rtol_2)
            if !res
                println(i)
                find_tol(rk_fnt_2[2], ub[2]; name1 = "rk_fnt_2[2]", name2 = "ub[2]")
            end
            @test res
            res = isapprox(rk_fnt_2[3], ub[3]; rtol = rtol_2)
            if !res
                println(i)
                find_tol(rk_fnt_2[3], ub[3]; name1 = "rk_fnt_2[3]", name2 = "ub[3]")
            end
            @test res
            i += 1
        end
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
