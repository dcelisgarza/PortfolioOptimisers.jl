#=
@safetestset "Stacking Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, Random, Clarabel, StatsBase,
          TimeSeries, CovarianceEstimation, FLoops
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
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                  timestamp = :timestamp)
    F = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/factor_prices.csv"));
                  timestamp = :timestamp)
    rd = prices_to_returns(X[(end - 252):end], F[(end - 252):end])
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false)),
           Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.6,
                                  "max_iter" => 1500, "tol_gap_abs" => 1e-4,
                                  "tol_gap_rel" => 1e-4, "tol_ktratio" => 1e-3,
                                  "tol_feas" => 1e-4, "tol_infeas_abs" => 1e-4,
                                  "tol_infeas_rel" => 1e-4, "reduced_tol_gap_abs" => 1e-4,
                                  "reduced_tol_gap_rel" => 1e-4,
                                  "reduced_tol_ktratio" => 1e-3, "reduced_tol_feas" => 1e-4,
                                  "reduced_tol_infeas_abs" => 1e-4,
                                  "reduced_tol_infeas_rel" => 1e-4))]
    pr = prior(HighOrderPriorEstimator(;
                                       pe = FactorPrior(;
                                                                 re = DimensionReductionRegression())),
               rd)
    rf = 4.34 / 100 / 252
    clr = clusterise(ClusteringEstimator(), rd.X)
    jopt = JuMPOptimiser(; slv = slv)
    hopt = HierarchicalOptimiser(; slv = slv)
    res1 = optimise!(Stacking(; pe = pr,
                              opti = [Stacking(;
                                               opti = [MeanRisk(;
                                                                r = ConditionalValueatRisk(),
                                                                obj = MaximumRatio(;
                                                                                   rf = rf),
                                                                opt = jopt),
                                                       RiskBudgetting(;
                                                                      r = LowOrderMoment(;
                                                                                         alg = MeanAbsoluteDeviation()),
                                                                      opt = jopt)],
                                               opto = HierarchicalRiskParity(;
                                                                             r = ConditionalDrawdownatRisk(),
                                                                             opt = hopt),
                                               threads = SequentialEx()),
                                      InverseVolatility()],
                              opto = Stacking(;
                                              opti = [HierarchicalRiskParity(;
                                                                             r = ConditionalDrawdownatRisk(),
                                                                             opt = hopt),
                                                      NestedClustering(;
                                                                       opto = NearOptimalCentering(;
                                                                                                   r = ConditionalValueatRisk(),
                                                                                                   obj = MaximumRatio(;
                                                                                                                      rf = rf),
                                                                                                   opt = jopt),
                                                                       opti = HierarchicalEqualRiskContribution(;
                                                                                                                ri = ConditionalDrawdownatRisk(),
                                                                                                                ro = StandardDeviation(),
                                                                                                                opt = hopt),
                                                                       threads = SequentialEx())],
                                              opto = MeanRisk(;
                                                              r = ConditionalValueatRisk(),
                                                              obj = MaximumRatio(; rf = rf),
                                                              opt = jopt),
                                              threads = SequentialEx()),
                              threads = SequentialEx()), rd)
    res2 = optimise!(Stacking(; pe = pr,
                              opti = [Stacking(;
                                               opti = [MeanRisk(;
                                                                r = ConditionalValueatRisk(),
                                                                obj = MaximumRatio(;
                                                                                   rf = rf),
                                                                opt = jopt),
                                                       RiskBudgetting(;
                                                                      r = LowOrderMoment(;
                                                                                         alg = MeanAbsoluteDeviation()),
                                                                      opt = jopt)],
                                               opto = HierarchicalRiskParity(;
                                                                             r = ConditionalDrawdownatRisk(),
                                                                             opt = hopt),
                                               threads = ThreadedEx()),
                                      InverseVolatility()],
                              opto = Stacking(;
                                              opti = [HierarchicalRiskParity(;
                                                                             r = ConditionalDrawdownatRisk(),
                                                                             opt = hopt),
                                                      NestedClustering(;
                                                                       opto = NearOptimalCentering(;
                                                                                                   r = ConditionalValueatRisk(),
                                                                                                   obj = MaximumRatio(;
                                                                                                                      rf = rf),
                                                                                                   opt = jopt),
                                                                       opti = HierarchicalEqualRiskContribution(;
                                                                                                                ri = ConditionalDrawdownatRisk(),
                                                                                                                ro = StandardDeviation(),
                                                                                                                opt = hopt),
                                                                       threads = ThreadedEx())],
                                              opto = MeanRisk(;
                                                              r = ConditionalValueatRisk(),
                                                              obj = MaximumRatio(; rf = rf),
                                                              opt = jopt),
                                              threads = ThreadedEx()),
                              threads = ThreadedEx()), rd)
    @test isapprox(res1.w,
                   [0.01059410362915954, 0.010640989138181081, 0.0063999742693907945,
                    0.015431859703639948, 0.012831666066042956, 0.009396976984868565,
                    0.010003588495565223, 0.01654126099004405, 0.014009946519036835,
                    0.013406604878895011, 0.16693619855098685, 0.011876811087741987,
                    0.01346919910794874, 0.008574059280093594, 0.016052988416936066,
                    0.009959142367567307, 0.009081555951171149, 0.022672939077843384,
                    0.012607455503540475, 0.19723995022244734, 0.0075938504186526465,
                    0.07011279834000951, 0.008720310975705331, 0.006522503363721369,
                    0.010360434391982665, 0.2570596260934086, 0.015312750334714631,
                    0.01659707226998855, 0.007459717877791127, 0.012533667528154022],
                   rtol = 5e-6)
    @test isapprox(res1.w, res2.w)

    res3 = optimise!(NestedClustering(; pe = pr, cle = clr,
                                      opti = Stacking(;
                                                      opti = [MeanRisk(;
                                                                       r = ConditionalValueatRisk(),
                                                                       obj = MaximumRatio(;
                                                                                          rf = rf),
                                                                       opt = jopt),
                                                              RiskBudgetting(;
                                                                             r = LowOrderMoment(;
                                                                                                alg = MeanAbsoluteDeviation()),
                                                                             opt = jopt)],
                                                      opto = HierarchicalRiskParity(;
                                                                                    r = ConditionalDrawdownatRisk(),
                                                                                    opt = hopt),
                                                      threads = SequentialEx()),
                                      opto = Stacking(;
                                                      opti = [HierarchicalRiskParity(;
                                                                                     r = ConditionalDrawdownatRisk(),
                                                                                     opt = hopt),
                                                              NestedClustering(;
                                                                               opto = NearOptimalCentering(;
                                                                                                           r = ConditionalValueatRisk(),
                                                                                                           obj = MaximumRatio(;
                                                                                                                              rf = rf),
                                                                                                           opt = jopt),
                                                                               opti = HierarchicalEqualRiskContribution(;
                                                                                                                        ri = ConditionalDrawdownatRisk(),
                                                                                                                        ro = StandardDeviation(),
                                                                                                                        opt = hopt),
                                                                               threads = SequentialEx())],
                                                      opto = MeanRisk(;
                                                                      r = ConditionalValueatRisk(),
                                                                      obj = MaximumRatio(;
                                                                                         rf = rf),
                                                                      opt = jopt),
                                                      threads = SequentialEx()),
                                      threads = SequentialEx()), rd)
    res4 = optimise!(NestedClustering(; pe = pr, cle = clr,
                                      opti = Stacking(;
                                                      opti = [MeanRisk(;
                                                                       r = ConditionalValueatRisk(),
                                                                       obj = MaximumRatio(;
                                                                                          rf = rf),
                                                                       opt = jopt),
                                                              RiskBudgetting(;
                                                                             r = LowOrderMoment(;
                                                                                                alg = MeanAbsoluteDeviation()),
                                                                             opt = jopt)],
                                                      opto = HierarchicalRiskParity(;
                                                                                    r = ConditionalDrawdownatRisk(),
                                                                                    opt = hopt),
                                                      threads = ThreadedEx()),
                                      opto = Stacking(;
                                                      opti = [HierarchicalRiskParity(;
                                                                                     r = ConditionalDrawdownatRisk(),
                                                                                     opt = hopt),
                                                              NestedClustering(;
                                                                               opto = NearOptimalCentering(;
                                                                                                           r = ConditionalValueatRisk(),
                                                                                                           obj = MaximumRatio(;
                                                                                                                              rf = rf),
                                                                                                           opt = jopt),
                                                                               opti = HierarchicalEqualRiskContribution(;
                                                                                                                        ri = ConditionalDrawdownatRisk(),
                                                                                                                        ro = StandardDeviation(),
                                                                                                                        opt = hopt),
                                                                               threads = ThreadedEx())],
                                                      opto = MeanRisk(;
                                                                      r = ConditionalValueatRisk(),
                                                                      obj = MaximumRatio(;
                                                                                         rf = rf),
                                                                      opt = jopt),
                                                      threads = ThreadedEx()),
                                      threads = ThreadedEx()), rd)
    @test isapprox(res3.w,
                   [0.012607444682938344, 0.010808223770579987, 0.006596087610098528,
                    0.01791006029555872, 0.037621537748847245, 0.008014682964922727,
                    0.01116300314219038, 0.014890678246639966, 0.015218811326447704,
                    0.04661383268020077, 0.2026392183797597, 0.010340250619013877,
                    0.015548665559460498, 0.009112902995485185, 0.020589329377531038,
                    0.011595877509750415, 0.008165380212374717, 0.037635555670233196,
                    0.011117899145165262, 0.20301218454539227, 0.005158618961883212,
                    0.07657722619885293, 0.008397589791387276, 0.0065238500329574955,
                    0.011079335862400747, 0.11411728552872605, 0.01748327586412157,
                    0.004689916642243212, 0.007815485635875965, 0.03695578899896097],
                   rtol = 5e-6)
    @test isapprox(res3.w, res4.w)
end
=#
