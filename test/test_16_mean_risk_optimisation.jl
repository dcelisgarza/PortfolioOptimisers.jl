@safetestset "Mean Risk Optimisation" begin
    using Test, PortfolioOptimisers, DataFrames, CSV, TimeSeries, Clarabel, HiGHS, Pajarito,
          JuMP, StatsBase, StableRNGs, LinearAlgebra, Distributions
    function find_tol(a1, a2; name1 = :lhs, name2 = :rhs)
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
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = "verbose" => false),
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = ["verbose" => false, "max_step_fraction" => 0.95]),
           Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
           Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.85)),
           Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.80)),
           Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
           Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.7)),
           Solver(; name = :clarabel8, solver = Clarabel.Optimizer,
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
    mip_slv = [Solver(; name = :mip1,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip2,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.95)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip3,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.90)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip4,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.85)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip5,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.80)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip6,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.75)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip7,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.7)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip8,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.6,
                                                                                                     "max_iter" => 1500,
                                                                                                     "tol_gap_abs" => 1e-4,
                                                                                                     "tol_gap_rel" => 1e-4,
                                                                                                     "tol_ktratio" => 1e-3,
                                                                                                     "tol_feas" => 1e-4,
                                                                                                     "tol_infeas_abs" => 1e-4,
                                                                                                     "tol_infeas_rel" => 1e-4,
                                                                                                     "reduced_tol_gap_abs" => 1e-4,
                                                                                                     "reduced_tol_gap_rel" => 1e-4,
                                                                                                     "reduced_tol_ktratio" => 1e-3,
                                                                                                     "reduced_tol_feas" => 1e-4,
                                                                                                     "reduced_tol_infeas_abs" => 1e-4,
                                                                                                     "reduced_tol_infeas_rel" => 1e-4)),
                      check_sol = (; allow_local = true, allow_almost = true))]
    sets = AssetSets(;
                     dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                                 "group2" => rd.nx[2:2:end],
                                 "clusters1" => [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                                                 3, 3, 3, 3, 3, 3],
                                 "clusters2" => [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,
                                                 3, 1, 2, 3, 1, 2], "c1" => rd.nx[1:3:end],
                                 "c2" => rd.nx[2:3:end], "c3" => rd.nx[3:3:end]))
    pr = prior(HighOrderPriorEstimator(), rd)
    clr = clusterise(ClusteringEstimator(), pr)
    w0 = range(; start = inv(size(pr.X, 2)), stop = inv(size(pr.X, 2)),
               length = size(pr.X, 2))
    wp = pweights(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                        length = size(pr.X, 1)))
    ucs1 = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(),
                                          rng = StableRNG(987654321),
                                          alg = BoxUncertaintySetAlgorithm()), rd.X)
    ucs2 = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(),
                                          rng = StableRNG(987654321),
                                          alg = EllipseUncertaintySetAlgorithm()), rd.X)
    rf = 4.2 / 100 / 252
    rd2 = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                      timestamp = :Date)[(end - 50):end])
    pr2 = prior(EmpiricalPrior(), rd2)
    @testset "Mean Risk" begin
        objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(; rf = rf)]
        rets = [ArithmeticReturn(), KellyReturn()]
        rs = [StandardDeviation(), Variance(), UncertaintySetVariance(; ucs = ucs1),
              UncertaintySetVariance(; ucs = ucs2), LowOrderMoment(),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             alg = SqrtRiskExpr()))),
              LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment())),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondCentralMoment(;
                                                                               alg = SqrtRiskExpr()))),
              LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondCentralMoment())),
              LowOrderMoment(; alg = MeanAbsoluteDeviation()), WorstRealisation(), Range(),
              ConditionalValueatRisk(), ConditionalValueatRiskRange(),
              EntropicValueatRisk(), EntropicValueatRiskRange(), RelativisticValueatRisk(),
              RelativisticValueatRiskRange(), MaximumDrawdown(), AverageDrawdown(),
              UlcerIndex(), ConditionalDrawdownatRisk(), EntropicDrawdownatRisk(),
              RelativisticDrawdownatRisk(), SquareRootKurtosis(; N = 2),
              SquareRootKurtosis(), OrderedWeightsArray(; alg = ExactOrderedWeightsArray()),
              OrderedWeightsArray(),
              OrderedWeightsArrayRange(; alg = ExactOrderedWeightsArray()),
              OrderedWeightsArrayRange(), NegativeSkewness(),
              NegativeSkewness(; alg = QuadRiskExpr()),
              DistributionallyRobustConditionalValueatRisk(),
              ValueatRisk(; alg = DistributionValueatRisk()),
              DistributionallyRobustConditionalValueatRiskRange(),
              ValueatRiskRange(; alg = DistributionValueatRisk()),
              TurnoverRiskMeasure(; w = w0),
              TrackingRiskMeasure(; tracking = WeightsTracking(; w = w0)),
              TrackingRiskMeasure(; tracking = WeightsTracking(; w = w0),
                                  alg = NOCTracking())]

        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk1.csv.gz"), DataFrame)
        i = 1
        for r in rs, obj in objs, ret in rets
            opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret)
            mr = MeanRisk(; r = r, obj = obj, opt = opt)
            res = optimise!(mr, rd)
            @test isa(res.retcode, OptimisationSuccess)
            rtol = if i in
                      (4, 10, 22, 76, 86, 91, 92, 96, 97, 99, 101, 103, 105, 133, 135, 141,
                       148, 154, 175, 186, 196)
                5e-5
            elseif i in
                   (6, 16, 28, 36, 38, 40, 46, 52, 93, 108, 126, 139, 163, 165, 167, 177,
                    179, 204, 214, 216)
                5e-6
            elseif i in (18, 157, 158, 174, 228)
                5e-4
            elseif i in (48, 58, 88, 90, 94, 98, 134, 140, 159, 176, 184)
                1e-5
            elseif i in (160, 164, 180)
                5e-3
            elseif i in (162, 178)
                1e-3
            elseif i in (198, 210)
                5e-2
            elseif i in (208, 234)
                1e-4
            else
                1e-6
            end
            success = isapprox(res.w, df[!, i]; rtol = rtol)
            if !success
                println("Counter: $i")
                find_tol(res.w, df[!, i])
            end
            @test success
            if isa(obj, MaximumRatio)
                rk = expected_risk(factory(r, pr, slv), res.w, rd.X)
                rt = expected_return(ret, res.w, pr)
                opt1 = JuMPOptimiser(; pe = pr, slv = slv,
                                     ret = bounds_returns_estimator(ret, rt))
                mr = MeanRisk(; r = r, opt = opt1)
                res = optimise!(mr, rd)
                rt1 = expected_return(ret, res.w, pr)
                @test rt1 >= rt || abs(rt1 - rt) < 1e-10
                mr = MeanRisk(; r = bounds_risk_measure(r, rk), obj = MaximumReturn(),
                              opt = opt)
                res = optimise!(mr, rd)
                rk1 = expected_risk(factory(r, pr, slv), res.w, rd.X)
                if !isa(r, SquareRootKurtosis) ||
                   isa(r, SquareRootKurtosis) && isnothing(r.N)
                    tol = if i == 161
                        5e-10
                    elseif i == 203
                        0.00014
                    elseif i == 204
                        0.00022
                    else
                        1e-10
                    end
                    @test rk1 <= rk || abs(rk1 - rk) < tol
                else
                    @test rk1 / rk < 1.07
                end
            end
            i += 1
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk1DT.csv.gz"), DataFrame)
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        tracking = WeightsTracking(; w = w0)
        for (i, r) in pairs(rs)
            r1 = RiskTrackingRiskMeasure(; tracking = tracking, r = r,
                                         alg = DependentVariableTracking())
            mr = MeanRisk(; r = r1, opt = opt)
            res = optimise!(mr, rd)
            if isa(r, PortfolioOptimisers.QuadExpressionRiskMeasures)
                @test isa(res.retcode, OptimisationFailure)
                continue
            else
                @test isa(res.retcode, OptimisationSuccess)
            end
            rtol = 1e-6
            success = isapprox(res.w, df[!, i]; rtol = rtol)
            if !success
                println("Counter: $i")
                find_tol(res.w, df[!, i])
            end
            @test success
        end

        res = optimise!(MeanRisk(; wi = w0,
                                 opt = JuMPOptimiser(; pe = pr,
                                                     slv = Solver(;
                                                                  solver = Clarabel.Optimizer,
                                                                  settings = ["verbose" => false,
                                                                              "max_iter" => 1])),
                                 fallback = InverseVolatility(; pe = pr)))
        @test isapprox(res.w, optimise!(InverseVolatility(; pe = pr)).w)

        r = BrownianDistanceVariance()
        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk1BDV.csv.gz"), DataFrame)
        i = 1
        for obj in objs, ret in rets
            opt = JuMPOptimiser(; pe = pr2, slv = slv, ret = ret)
            mr = MeanRisk(; r = r, obj = obj, opt = opt)
            res = optimise!(mr, rd2)
            @test isa(res.retcode, OptimisationSuccess)
            rtol = if i == 4
                5e-6
            elseif i == 6
                5e-5
            else
                1e-6
            end
            success = isapprox(res.w, df[!, i]; rtol = rtol)
            if !success
                println("Counter: $i")
                find_tol(res.w, df[!, i])
            end
            @test success
            i += 1
        end
    end
    @testset "Formulations" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = factory(Variance(), pr)
        res_min = optimise!(MeanRisk(; r = r, opt = opt))
        res_max = optimise!(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        rk_min = expected_risk(r, res_min.w, pr)
        rk_max = expected_risk(r, res_max.w, pr)
        rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
        rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
        res1 = optimise!(MeanRisk(;
                                  r = Variance(;
                                               settings = RiskMeasureSettings(;
                                                                              ub = Frontier(;
                                                                                            N = 5))),
                                  obj = MaximumReturn(), opt = opt))
        res2 = optimise!(MeanRisk(;
                                  r = Variance(; alg = QuadRiskExpr(),
                                               settings = RiskMeasureSettings(;
                                                                              ub = Frontier(;
                                                                                            N = 5))),
                                  obj = MaximumReturn(), opt = opt))
        res = isapprox(hcat(res1.w...), hcat(res2.w...); rtol = 5e-4)
        if !res
            println("Frontier formulation failed")
            find_tol(hcat(res1.w...), hcat(res2.w...))
        end
        rks = expected_risk.(Ref(r), res1.w, Ref(pr))
        @test issorted(rks)
        @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
        rts = expected_return.(ArithmeticReturn(), res1.w, Ref(pr))
        @test issorted(rts)
        @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

        res3 = optimise!(MeanRisk(;
                                  r = Variance(;
                                               settings = RiskMeasureSettings(;
                                                                              ub = range(;
                                                                                         start = rk_min,
                                                                                         stop = rk_max,
                                                                                         length = 5))),
                                  obj = MaximumReturn(), opt = opt))
        res4 = optimise!(MeanRisk(;
                                  r = Variance(; alg = QuadRiskExpr(),
                                               settings = RiskMeasureSettings(;
                                                                              ub = range(;
                                                                                         start = rk_min,
                                                                                         stop = rk_max,
                                                                                         length = 5))),
                                  obj = MaximumReturn(), opt = opt))
        res = isapprox(hcat(res3.w...), hcat(res4.w...); rtol = 1e-6)
        if !res
            println("Frontier formulation failed")
            find_tol(hcat(res3.w...), hcat(res4.w...))
        end
        rks = expected_risk.(Ref(r), res3.w, Ref(pr))
        @test issorted(rks)
        @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
        rts = expected_return.(ArithmeticReturn(), res3.w, Ref(pr))
        @test issorted(rts)
        @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            ret = ArithmeticReturn(; lb = Frontier(; N = 5)))
        res5 = optimise!(MeanRisk(; r = Variance(;), opt = opt))
        res6 = optimise!(MeanRisk(; r = Variance(; alg = QuadRiskExpr()), opt = opt))
        res = isapprox(hcat(res5.w...), hcat(res6.w...); rtol = 5e-4)
        if !res
            println("Frontier formulation failed")
            find_tol(hcat(res5.w...), hcat(res6.w...))
        end
        rks = expected_risk.(Ref(r), res5.w, Ref(pr))
        @test issorted(rks)
        @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
        rts = expected_return.(ArithmeticReturn(), res5.w, Ref(pr))
        @test issorted(rts)
        @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            ret = ArithmeticReturn(;
                                                   lb = range(; start = rt_min,
                                                              stop = rt_max, length = 5)))
        res7 = optimise!(MeanRisk(; r = Variance(;), opt = opt))
        res8 = optimise!(MeanRisk(; r = Variance(; alg = QuadRiskExpr()), opt = opt))
        res = isapprox(hcat(res7.w...), hcat(res8.w...); rtol = 5e-4)
        if !res
            println("Frontier formulation failed")
            find_tol(hcat(res7.w...), hcat(res8.w...))
        end
        rks = expected_risk.(Ref(r), res7.w, Ref(pr))
        @test issorted(rks)
        @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
        rts = expected_return.(ArithmeticReturn(), res7.w, Ref(pr))
        @test issorted(rts)
        @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = factory(LowOrderMoment(;
                                   alg = LowOrderDeviation(;
                                                           alg = SecondCentralMoment(;
                                                                                     alg = QuadRiskExpr()))),
                    pr)
        res_min = optimise!(MeanRisk(; r = r, opt = opt))
        res_max = optimise!(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        rk_min = expected_risk(r, res_min.w, pr)
        rk_max = expected_risk(r, res_max.w, pr)
        rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
        rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
        res1 = optimise!(MeanRisk(;
                                  r = LowOrderMoment(;
                                                     settings = RiskMeasureSettings(;
                                                                                    ub = Frontier(;
                                                                                                  N = 5)),
                                                     alg = LowOrderDeviation(;
                                                                             alg = SecondCentralMoment(;
                                                                                                       alg = QuadRiskExpr()))),
                                  obj = MaximumReturn(), opt = opt))
        res2 = optimise!(MeanRisk(;
                                  r = LowOrderMoment(;
                                                     settings = RiskMeasureSettings(;
                                                                                    ub = Frontier(;
                                                                                                  N = 5)),
                                                     alg = LowOrderDeviation(;
                                                                             alg = SecondCentralMoment(;
                                                                                                       alg = RSOCRiskExpr()))),
                                  obj = MaximumReturn(), opt = opt))
        res = isapprox(hcat(res1.w...), hcat(res2.w...); rtol = 5e-3)
        if !res
            println("Frontier formulation failed")
            find_tol(hcat(res1.w...), hcat(res2.w...))
        end
        rks = expected_risk.(Ref(r), res1.w, Ref(pr))
        @test issorted(rks)
        @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
        rts = expected_return.(ArithmeticReturn(), res1.w, Ref(pr))
        @test issorted(rts)
        @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

        res3 = optimise!(MeanRisk(;
                                  r = LowOrderMoment(;
                                                     settings = RiskMeasureSettings(;
                                                                                    ub = range(;
                                                                                               start = rk_min,
                                                                                               stop = rk_max,
                                                                                               length = 5)),
                                                     alg = LowOrderDeviation(;
                                                                             alg = SecondCentralMoment(;
                                                                                                       alg = QuadRiskExpr()))),
                                  obj = MaximumReturn(), opt = opt))
        res4 = optimise!(MeanRisk(;
                                  r = LowOrderMoment(;
                                                     settings = RiskMeasureSettings(;
                                                                                    ub = range(;
                                                                                               start = rk_min,
                                                                                               stop = rk_max,
                                                                                               length = 5)),
                                                     alg = LowOrderDeviation(;
                                                                             alg = SecondCentralMoment(;
                                                                                                       alg = RSOCRiskExpr()))),
                                  obj = MaximumReturn(), opt = opt))
        res = isapprox(hcat(res3.w...), hcat(res4.w...); rtol = 5e-6)
        if !res
            println("Frontier formulation failed")
            find_tol(hcat(res3.w...), hcat(res4.w...))
        end
        rks = expected_risk.(Ref(r), res3.w, Ref(pr))
        @test issorted(rks)
        @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
        rts = expected_return.(ArithmeticReturn(), res3.w, Ref(pr))
        @test issorted(rts)
        @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            ret = ArithmeticReturn(; lb = Frontier(; N = 5)))
        res5 = optimise!(MeanRisk(;
                                  r = LowOrderMoment(;
                                                     alg = LowOrderDeviation(;
                                                                             alg = SecondCentralMoment(;
                                                                                                       alg = QuadRiskExpr()))),
                                  opt = opt))
        res6 = optimise!(MeanRisk(;
                                  r = LowOrderMoment(;
                                                     alg = LowOrderDeviation(;
                                                                             alg = SecondCentralMoment(;
                                                                                                       alg = RSOCRiskExpr()))),
                                  opt = opt))
        res = isapprox(hcat(res5.w...), hcat(res6.w...); rtol = 5e-3)
        if !res
            println("Frontier formulation failed")
            find_tol(hcat(res5.w...), hcat(res6.w...))
        end
        rks = expected_risk.(Ref(r), res5.w, Ref(pr))
        @test issorted(rks)
        @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
        rts = expected_return.(ArithmeticReturn(), res5.w, Ref(pr))
        @test issorted(rts)
        @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            ret = ArithmeticReturn(;
                                                   lb = range(; start = rt_min,
                                                              stop = rt_max, length = 5)))
        res7 = optimise!(MeanRisk(;
                                  r = LowOrderMoment(; settings = RiskMeasureSettings(;),
                                                     alg = LowOrderDeviation(;
                                                                             alg = SecondCentralMoment(;
                                                                                                       alg = QuadRiskExpr()))),
                                  opt = opt))
        res8 = optimise!(MeanRisk(;
                                  r = LowOrderMoment(; settings = RiskMeasureSettings(;),
                                                     alg = LowOrderDeviation(;
                                                                             alg = SecondCentralMoment(;
                                                                                                       alg = RSOCRiskExpr()))),
                                  opt = opt))
        res = isapprox(hcat(res7.w...), hcat(res8.w...); rtol = 5e-3)
        if !res
            println("Frontier formulation failed")
            find_tol(hcat(res7.w...), hcat(res8.w...))
        end
        rks = expected_risk.(Ref(r), res7.w, Ref(pr))
        @test issorted(rks)
        @test all(rk_min - sqrt(eps()) .<= rks .<= rk_max + sqrt(eps()))
        rts = expected_return.(ArithmeticReturn(), res7.w, Ref(pr))
        @test issorted(rts)
        @test all(rt_min - sqrt(eps()) .<= rts .<= rt_max + sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv)
        res9 = optimise!(MeanRisk(;
                                  r = ValueatRisk(;
                                                  alg = DistributionValueatRisk(;
                                                                                dist = Laplace())),
                                  opt = opt))
        res10 = optimise!(MeanRisk(;
                                   r = ValueatRisk(;
                                                   alg = DistributionValueatRisk(;
                                                                                 dist = TDist(5)),),
                                   opt = opt))
        @test isapprox(res9.w, res10.w; rtol = 5e-2)

        res11 = optimise!(MeanRisk(;
                                   r = ValueatRiskRange(;
                                                        alg = DistributionValueatRisk(;
                                                                                      dist = Laplace())),
                                   opt = opt))
        res12 = optimise!(MeanRisk(;
                                   r = ValueatRiskRange(;
                                                        alg = DistributionValueatRisk(;
                                                                                      dist = TDist(5)),),
                                   opt = opt))
        @test isapprox(res11.w, res12.w; rtol = 5e-4)

        opt = JuMPOptimiser(; pe = pr2, slv = slv)
        mr = MeanRisk(;
                      r = BrownianDistanceVariance(; algc = IneqBrownianDistanceVariance()),
                      opt = opt)
        res1 = optimise!(mr)

        mr = MeanRisk(; r = BrownianDistanceVariance(; alg = RSOCRiskExpr()), opt = opt)
        res2 = optimise!(mr)

        mr = MeanRisk(;
                      r = BrownianDistanceVariance(; alg = RSOCRiskExpr(),
                                                   algc = IneqBrownianDistanceVariance()),
                      opt = opt)
        res3 = optimise!(mr)
        @test isapprox(res1.w,
                       CSV.read(joinpath(@__DIR__, "./assets/MeanRisk1BDV.csv.gz"),
                                DataFrame)[!, 1], rtol = 5e-4)
        @test isapprox(res1.w, res2.w; rtol = 5e-4)
        @test isapprox(res1.w, res3.w; rtol = 1e-3)
    end
    @testset "Scalarisers" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, opt = opt)
        w1 = optimise!(mr, rd).w
        @test isapprox(w1,
                       [1.7074698994991376e-10, 8.433104973224101e-11,
                        1.9860067068611146e-9, 3.1027970850564853e-10, 0.09898828657505643,
                        0.0038933979989648256, 6.636136386259997e-10, 0.35967323582529487,
                        0.012150886312492075, 0.10067191014130493, 2.964806359249512e-10,
                        0.14595634518390374, 3.891455764455336e-10, 0.14289553677354302,
                        5.362751594199594e-10, 0.03009102863550913, 1.1386315235757199e-10,
                        1.1094461786914318e-9, 0.07216633866035838, 0.033513028233384],
                       rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = MaxScalariser())
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, opt = opt)
        w2 = optimise!(mr, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser(; gamma = 1e-3))
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, opt = opt)
        w3 = optimise!(mr, rd).w
        @test isapprox(w3, w1, rtol = 5e-2)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser(; gamma = 1e5))
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, opt = opt)
        w4 = optimise!(mr, rd).w
        @test isapprox(w4, w2, rtol = 1e-4)
    end
    @testset "Arithmetic return uncertainty set" begin
        rng = StableRNG(123456789)
        ucs1 = mu_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng,
                                           alg = BoxUncertaintySetAlgorithm()), pr.X)
        ucs2 = mu_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng,
                                           alg = EllipseUncertaintySetAlgorithm()), pr.X)
        ucss = [ucs1, ucs2]
        objs = [MinimumRisk(), MaximumRatio(; rf = rf), MaximumReturn()]
        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRiskUncertainty.csv.gz"), DataFrame)
        i = 1
        for ucs in ucss
            for obj in objs
                ret = ArithmeticReturn(; ucs = ucs)
                opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
                mre = MeanRisk(; obj = obj, opt = opt)
                res = optimise!(mre)
                @test isa(res.retcode, OptimisationSuccess)
                rtol = 1e-6
                success = isapprox(res.w, df[!, i]; rtol = rtol)
                if !success
                    println("Counter: $i")
                    find_tol(res.w, df[!, i])
                end
                @test success
                i += 1
            end
        end
    end
    @testset "Weight bounds" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBoundsEstimator(;
                                                       lb = ["group1" => -1,
                                                             "group2" => 0.1],
                                                       ub = Dict("group1" => -0.1,
                                                                 "group2" => 1)))
        mr = MeanRisk(; opt = opt)
        res1 = optimise!(mr)
        @test isapprox(sum(res1.w), 1)
        @test isapprox(sum(res1.w[res1.w .< zero(eltype(res1.w))]), -1)
        @test isapprox(sum(res1.w[res1.w .>= zero(eltype(res1.w))]), 2)
        @test all(res1.pa.wb.lb[1:2:end] .<= res1.w[1:2:end])
        @test all(abs.(res1.w[1:2:end] .- res1.pa.wb.ub[1:2:end]) .< 5e-10)
        @test all(res1.pa.wb.lb[2:2:end] .<= res1.w[2:2:end] .<= res1.pa.wb.ub[2:2:end])
    end
    @testset "Budget" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumReturn(), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w), 1)
        @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -1, rtol = 1e-6)
        @test isapprox(sum(res.w[res.w .>= zero(eltype(res.w))]), 2, rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 0.15, bgt = 0.5,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w), 0.5)
        @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -0.15, rtol = 1e-4)
        @test isapprox(sum(res.w[res.w .>= zero(eltype(res.w))]), 0.65, rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                            sbgt = BudgetRange(; lb = 0.15, ub = 0.15),
                            bgt = BudgetRange(; lb = 0.3, ub = 0.45),
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test 0.1 <= sum(res.w) <= 0.45
        @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -0.15, rtol = 5e-5)
        @test 0.45 <= sum(res.w[res.w .>= zero(eltype(res.w))]) <= 0.60

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = nothing, bgt = 1.7,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w), 1.7)
        @test all(res.w .>= 0)
        @test !haskey(res.model, :sbgt)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1.4, bgt = nothing,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w[res.w .< 0]), -1.4, rtol = 1e-3)
        @test !haskey(res.model, :bgt)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                            sbgt = BudgetRange(; lb = 0.41, ub = 0.63), bgt = 0.87,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test isapprox(sum(res.w), 0.87)
        @test -0.63 <= sum(res.w[res.w .< zero(eltype(res.w))]) <= -0.41
        @test 0.87 + 0.41 <= sum(res.w[res.w .>= zero(eltype(res.w))]) <= 0.87 + 0.63

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 0.61,
                            bgt = BudgetRange(; lb = 0.4, ub = 0.79),
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumUtility(), opt = opt)
        res = optimise!(mr)
        @test 0.4 <= sum(res.w) <= 0.79
        @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -0.61, rtol = 5e-5)
        @test 0.61 + 0.4 <= sum(res.w[res.w .> zero(eltype(res.w))]) <= 0.61 + 0.79

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                            sbgt = BudgetRange(; lb = 0.41, ub = 0.63), bgt = nothing,
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test -0.63 <= sum(res.w[res.w .< zero(eltype(res.w))]) <= -0.41
        @test !haskey(res.model, :bgt)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = nothing,
                            bgt = BudgetRange(; lb = 0.4, ub = 0.79),
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumUtility(), opt = opt)
        res = optimise!(mr)
        @test 0.4 <= sum(res.w) <= 0.79
        @test !haskey(res.model, :sbgt)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                            sbgt = BudgetRange(; lb = nothing, ub = 0.23),
                            bgt = BudgetRange(; lb = 0.41, ub = nothing),
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test sum(res.w) >= 0.41
        @test sum(res.w[res.w .< 0]) >= -0.23
        @test sum(res.w[res.w .>= 0]) >= 0.64

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                            sbgt = BudgetRange(; lb = 0.35, ub = nothing),
                            bgt = BudgetRange(; lb = nothing, ub = 0.65),
                            wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test sum(res.w) <= 0.41
        @test sum(res.w[res.w .< 0]) <= -0.35
        @test sum(res.w[res.w .>= 0]) <= 0.76

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = nothing,
                            bgt = nothing, wb = WeightBounds(; lb = -1, ub = 1))
        mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mr)
        @test !haskey(res.model, :sbgt)
        @test !haskey(res.model, :lbgt)
    end
    @testset "Cardinality" begin
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, card = 3)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        w = res.w
        @test count(w .> 1e-10) <= 3

        opt = JuMPOptimiser(; l2 = 0.1, pe = pr, slv = mip_slv, card = 3)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre)
        w = res.w
        @test count(w .> 1e-10) <= 3

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1, card = 7)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        w = res.w
        @test count(abs.(w) .> 1e-10) <= 7

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            gcard = LinearConstraintEstimator(;
                                                              val = [:(XOM + MRK + WMT <= 2),
                                                                     :(group2 == 5)]),
                            sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        w = res.w
        @test rd.nx[.!iszero.(vec(res.gcard.A_ineq[1, :]))] == ["MRK", "WMT", "XOM"]
        @test rd.nx[.!iszero.(vec(res.gcard.A_eq[1, :]))] == rd.nx[2:2:end]
        @test count(w[.!iszero.(vec(res.gcard.A_ineq[1, :]))] .> 1e-10) <= 2
        @test count(w[.!iszero.(vec(res.gcard.A_eq[1, :]))] .> 1e-10) == 5

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1,
                            gcard = LinearConstraintEstimator(;
                                                              val = [:(XOM + MRK + WMT <= 2),
                                                                     :(group2 == 3)]),
                            sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        w = res.w
        @test rd.nx[.!iszero.(vec(res.gcard.A_ineq[1, :]))] == ["MRK", "WMT", "XOM"]
        @test rd.nx[.!iszero.(vec(res.gcard.A_eq[1, :]))] == rd.nx[2:2:end]
        @test count(w[.!iszero.(vec(res.gcard.A_ineq[1, :]))] .> 1e-10) <= 2
        @test count(w[.!iszero.(vec(res.gcard.A_eq[1, :]))] .> 1e-10) == 3

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1,
                            smtx = AssetSetsMatrixEstimator(; val = "clusters1"),
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test sum(.!iszero.([sum(w[res.smtx[i, :]]) for i in axes(res.smtx, 1)])) == 1

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 2,
                            smtx = AssetSetsMatrixEstimator(; val = "clusters2"),
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test sum(.!iszero.([sum(w[res.smtx[i, :]]) for i in axes(res.smtx, 1)])) == 2

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = [1, 1],
                            smtx = [AssetSetsMatrixEstimator(; val = "clusters1"),
                                    AssetSetsMatrixEstimator(; val = "clusters2")],
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w

        i = 1
        dict = Dict{Tuple{Int, Int}, Int}()
        clusters3 = Int[]
        for cs in zip(sets.dict["clusters1"], sets.dict["clusters2"])
            if !haskey(dict, cs)
                dict[cs] = i
                i += 1
            end
            push!(clusters3, dict[cs])
        end
        sets.dict["clusters3"] = clusters3
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1,
                            smtx = AssetSetsMatrixEstimator(; val = "clusters3"),
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        @test isapprox(res.w, optimise!(mre, rd).w)
    end
    @testset "Buy-in threshold" begin
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            lt = BuyInThresholdEstimator(;
                                                         val = ["WMT" => 0.23,
                                                                "group2" => 0.48]),
                            sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        @test res.w[findfirst(x -> x == "WMT", rd.nx)] >= 0.23
        @test res.w[2:2:end][res.w[2:2:end] .> 1e-9][1] >= 0.48

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = BuyInThreshold(; val = 0.15),
                            sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        res.w[res.w .>= 1e-10] .>= 0.15

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            lt = BuyInThreshold(; val = fill(0.15, size(pr.X, 2))),
                            sets = sets)
        mre = MeanRisk(; opt = opt)
        @test isapprox(res.w, optimise!(mre).w)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1, st = BuyInThreshold(; val = 0.25),
                            lt = BuyInThreshold(; val = 0.4), sets = sets)
        mre = MeanRisk(; opt = opt)
        res = optimise!(mre)
        @test all(res.w[res.w .> 0][res.w[res.w .>= 0] .>= 1e-10] .>= 0.4)
        if Sys.isapple()
            @test all(res.w[res.w .< 0][res.w[res.w .< 0] .<= -1e-10] .<=
                      -0.25 + sqrt(eps()))
        else
            @test all(res.w[res.w .< 0][res.w[res.w .< 0] .<= -1e-10] .<= -0.25)
        end

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1, st = BuyInThreshold(; val = 0.25),
                            lt = BuyInThreshold(; val = 0.4), sets = sets)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre)
        @test all(res.w[res.w .> 0][res.w[res.w .>= 0] .>= 1e-10] .>= 0.4)
        @test all(res.w[res.w .< 0][res.w[res.w .< 0] .<= -1e-10] .<= -0.25)
    end
    @testset "Linear weight constraints" begin
        sets.dict["group4"] = ["AMD", "BAC"]
        sets.dict["group5"] = ["PG", "XOM"]
        lcs = LinearConstraintEstimator(;
                                        val = ["AAPL >= 0.2*MRK", "group4 >= 3*group5",
                                               "RRC <=-0.02", "MSFT==0.01"])
        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), lcs = lcs)
        mr = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mr)
        @test res.w[findfirst(x -> x == "AAPL", rd.nx)] >=
              0.2 * res.w[findfirst(x -> x == "MRK", rd.nx)]
        @test sum(res.w[[findfirst(x -> x == a, rd.nx) for a in sets.dict["group4"]]]) >=
              3 * sum(res.w[[findfirst(x -> x == a, rd.nx) for a in sets.dict["group5"]]])
        @test res.w[findfirst(x -> x == "RRC", rd.nx)] <= -0.02
        @test res.w[findfirst(x -> x == "MSFT", rd.nx)] == 0.01

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), lcs = res.lcs)
        @test isapprox(res.w, optimise!(MeanRisk(; obj = MinimumRisk(), opt = opt)).w)
    end
    @testset "Regularisation" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), l1 = 5e-6)
        mr = MeanRisk(; opt = opt)
        res = optimise!(mr)
        @test isapprox(res.w,
                       [-0.08259934606632031, -0.01735523094219388, -6.306917873785564e-9,
                        -0.019964241234679967, 0.08433288155125772, 0.04082860469386889,
                        0.0245444126670805, 0.38468746514362956, 0.025975231635603585,
                        0.11323392647659213, -0.0455153966134236, 0.1769386350615763,
                        0.04466740435373562, 0.12535190765089996, -0.01382516940599513,
                        0.021879232823390448, -0.01259642910877665, -3.789411268851941e-9,
                        0.09356110720420865, 0.05585501420587528], rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), l1 = 1)
        mr = MeanRisk(; opt = opt)
        res = optimise!(mr)
        @test isapprox(res.w,
                       [4.1633941065577117e-7, 2.6448742701527e-7, 4.477941406289695e-6,
                        4.854555411641452e-7, 0.07424358812530261, 0.00792596837274629,
                        1.352652412857295e-6, 0.36980684793744134, 0.007584348893331609,
                        0.11118617081059556, 3.753498265381272e-7, 0.17467158275776748,
                        9.348289673318389e-7, 0.08974785581907543, 8.391727846180317e-7,
                        0.023971871590369637, 3.3459673922392774e-7, 1.3947538587917836e-6,
                        0.0935238000646315, 0.04732709005036381], rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), l2 = 5e-6)
        mr = MeanRisk(; opt = opt)
        res = optimise!(mr)
        @test isapprox(res.w,
                       [-0.11064349447254161, -0.017854196186798618, -0.04397298675384868,
                        -0.03145351033102496, 0.09495263689036014, 0.05053896454111151,
                        0.03699197653220963, 0.38114357883139005, 0.06929867727498179,
                        0.11865327291902569, -0.06633836583807416, 0.19538041855641222,
                        0.07381248663128888, 0.13328230798627264, -0.037943963731425036,
                        0.029972928788686973, -0.017281968260805896, -0.008700415062896462,
                        0.09297968577516197, 0.05718196591051368], rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), l2 = 1e-4)
        mr = MeanRisk(; opt = opt)
        res = optimise!(mr)
        @test isapprox(res.w,
                       [-0.03076204518648503, -0.03965358415790933, 0.01741465312222848,
                        -0.014071169950968142, 0.08160008886554988, 0.038967358792986795,
                        0.03853758826706406, 0.17170470580976588, 0.03806574632010554,
                        0.10075437963820232, 0.024679044550725296, 0.1375839532075256,
                        0.024805294188283956, 0.10194818468441112, 0.03329884602273654,
                        0.08767866540531458, -0.0157785635897723, 0.03858247320319518,
                        0.09776343657139772, 0.06688094423564199], rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), l1 = 5e-6, l2 = 5e-6)
        mr = MeanRisk(; opt = opt)
        res = optimise!(mr)
        @test isapprox(res.w,
                       [-0.07591829609679655, -0.019207462900668874, -4.164287783441495e-9,
                        -0.019697945739295775, 0.08623890830132815, 0.03970246984743602,
                        0.024849760269582157, 0.3562105498789526, 0.028493593064519384,
                        0.11264585349973763, -0.03577564886530365, 0.1762813917849517,
                        0.03994057834010078, 0.12239469833440599, -0.007813742040459417,
                        0.03415102550994622, -0.01302979483152005, -5.177412500119382e-10,
                        0.09499371625016041, 0.05554035007495228], rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), l1 = 1, l2 = 1e-4)
        mr = MeanRisk(; opt = opt)
        res = optimise!(mr)
        @test isapprox(res.w,
                       [2.0232629822283223e-6, 9.760082045106613e-7, 0.0025796058703603294,
                        2.436499442593761e-6, 0.07216922707119579, 0.017185343248064092,
                        0.012997513915337116, 0.17907277860116347, 0.03127342741481768,
                        0.09864631045614426, 0.021928766640300995, 0.15045440958368198,
                        4.894112955006514e-6, 0.09912274998209057, 0.035519395576107456,
                        0.08660290480901828, 1.711128248182528e-6, 0.03477208922064647,
                        0.09904023885895782, 0.05862319774028108], rtol = 1e-6)
    end
    @testset "Turnover" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), tn = Turnover(; w = w0))
        mr = MeanRisk(; opt = opt)
        @test isapprox(w0, optimise!(mr).w)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1),
                            tn = [TurnoverEstimator(; w = w0,
                                                    val = ["AAPL" => 0, "MRK" => 0.05],
                                                    default = Inf)])
        mr = MeanRisk(; opt = opt)
        res = optimise!(mr)
        @test isapprox(res.w,
                       [0.049999999999993605, -0.04520519208370696, -0.04869401288598486,
                        -0.04327195569636555, 0.11569334228575458, 0.0414229664404061,
                        0.03303019511733334, 0.4429818998364995, 0.07583334507776132,
                        0.10762664262865197, -0.04825139998117526, 0.09999541531046201,
                        -0.0016373970470112344, 0.10510925375967227, -0.023144445601095607,
                        0.043540146321535515, -0.01798434663243727, -0.023893575203165842,
                        0.09199157017254597, 0.04485754818032653], rtol = 1e-6)
    end
    @testset "Number of effective assets" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, nea = 10)
        res = optimise!(MeanRisk(; obj = MinimumRisk(), opt = opt))
        @test round(inv(dot(res.w, res.w))) >= 10

        opt = JuMPOptimiser(; pe = pr, slv = slv, nea = 15)
        res = optimise!(MeanRisk(; obj = MaximumUtility(), opt = opt))
        @test round(inv(dot(res.w, res.w))) >= 15
    end
    @testset "Tracking" begin
        rdb = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__,
                                                            "./assets/SP500_idx.csv.gz"));
                                          timestamp = :Date)[(end - 252):end])
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            te = TrackingError(;
                                               tracking = ReturnsTracking(; w = vec(rdb.X)),
                                               err = 3e-3))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre)
        @test norm(rd.X * res.w - vec(rdb.X)) / sqrt(size(rd.X, 1)) <= 3e-3

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            te = TrackingError(;
                                               tracking = ReturnsTracking(; w = vec(rdb.X)),
                                               err = 2.5e-3, alg = NOCTracking()))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre)
        @test norm(rd.X * res.w - vec(rdb.X), 1) / size(rd.X, 1) <= 2.5e-3

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            te = TrackingError(; tracking = WeightsTracking(; w = w0),
                                               err = 2e-3))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre)
        @test norm(rd.X * (res.w - w0)) / sqrt(size(rd.X, 1)) <= 2e-3

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            te = [TrackingError(; tracking = WeightsTracking(; w = w0),
                                                err = 2e-3, alg = NOCTracking())])
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre)
        @test norm(rd.X * (res.w - w0), 1) / size(rd.X, 1) <= 2e-3
    end
    @testset "Phylogeny" begin
        plc = IntegerPhylogenyEstimator(; pe = NetworkEstimator(), B = 1)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, cplg = plc,
                            wb = WeightBounds(; lb = -1, ub = 1), l2 = 0.001)
        res = optimise!(MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt))
        @test all(value.(res.cplg.A * res.model[:ib]) .<= res.cplg.B)
        idx = [BitVector(res.cplg.A[:, i]) for i in axes(res.cplg.A, 2)]
        @test all([(count(abs.(getindex(res.w, i)) .> 1e-10) <= 1) for i in idx])
        @test isapprox(res.w,
                       [-9.431976408001725e-15, -0.8782741689961527, -9.379228352184268e-15,
                        -4.001985368795484e-15, 7.514829720676815e-15,
                        -5.128614882277353e-15, -7.875395433680138e-15,
                        -3.2602604140836353e-15, -6.293827026255587e-15,
                        0.20017499085628088, 3.6505444231065125e-15, 0.6781009262835807,
                        -9.805831650414789e-15, -2.2703996977265388e-15,
                        -6.760480819135996e-15, -5.3508991062443464e-15,
                        7.766000981009452e-15, -2.3067244875711246e-15,
                        -3.831552897938868e-15, 0.999998251856358], rtol = 1e-6)

        plc = IntegerPhylogenyEstimator(; pe = NetworkEstimator(), B = 2)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, nplg = plc,
                            wb = WeightBounds(; lb = -1, ub = 1), l2 = 0.0001)
        res = optimise!(MeanRisk(; obj = MinimumRisk(), opt = opt))
        @test all(value.(res.nplg.A * res.model[:ib]) .<= res.nplg.B)
        idx = [BitVector(res.nplg.A[:, i]) for i in axes(res.nplg.A, 2)]
        @test all([(count(abs.(getindex(res.w, i)) .> 1e-10) <= 2) for i in idx])
        success = isapprox(res.w,
                           [-5.444667507634538e-13, -0.04740153354475791,
                            0.04898447042002428, -4.731467916273829e-13,
                            -1.130957000167328e-12, 0.0597898659303373,
                            0.052442100113306786, 0.2536590009832417,
                            -8.955374909922694e-13, -1.3470633211428974e-12,
                            -1.3099269328335245e-12, 0.20037416198655156,
                            0.030961713958730052, -1.3571762187763267e-12,
                            -1.3360278929964757e-12, 0.15024992711136434,
                            -3.250354013486661e-13, -1.2704328733491568e-12,
                            0.1299170519311761, 0.12102324112001549]; rtol = 1e-6)
        if success
            @test success
        else
            @test isapprox(res.w,
                           [-7.681434332864639e-13, 7.405561598382192e-14,
                            0.014250247570361224, -6.974708088175452e-13,
                            0.08537332610615428, -1.0150718449032514e-12,
                            0.03983561006114424, 0.2502122641236305, 0.05404071746483757,
                            -1.5666916399782238e-12, -1.538861209090204e-12,
                            0.2025433723191622, -0.0014220443665369262,
                            -1.5802206832408362e-12, -1.5503975446319445e-12,
                            0.15784094556549422, -4.439084812281355e-13,
                            -1.4843816960768885e-12, 0.1276616300735639,
                            0.0696639310927601]; rtol = 1.0e-6)
        end
        plc = SemiDefinitePhylogenyEstimator(; pe = clr)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, cplg = plc,
                            wb = WeightBounds(; lb = -1, ub = 1))
        res = optimise!(MeanRisk(; obj = MinimumRisk(), opt = opt))
        @test isapprox(value.(res.cplg.A .* res.model[:W]), zeros(size(pr.sigma)),
                       atol = 1e-10)

        plc = SemiDefinitePhylogenyEstimator(; pe = ClusteringEstimator(), p = 1000)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, nplg = plc,
                            wb = WeightBounds(; lb = -1, ub = 1))
        @test isapprox(res.w, optimise!(MeanRisk(; obj = MinimumRisk(), opt = opt)).w)

        plc = phylogeny_constraints(SemiDefinitePhylogenyEstimator(; pe = clr, p = 10),
                                    rd.X)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, cplg = plc,
                            wb = WeightBounds(; lb = -1, ub = 1))
        @test isapprox(res.w, optimise!(MeanRisk(; obj = MinimumRisk(), opt = opt)).w)

        plc = SemiDefinitePhylogenyEstimator(; pe = clr)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, cplg = plc,
                            wb = WeightBounds(; lb = -1, ub = 1))
        res1 = optimise!(MeanRisk(; r = ConditionalValueatRisk(),
                                  obj = MaximumRatio(; rf = rf), opt = opt))
        @test isapprox(value.(res1.cplg.A .* res1.model[:W]), zeros(size(pr.sigma)),
                       atol = 1e-10)
        @test isapprox(res1.w,
                       [2.630737181170687e-10, -0.19525137327795888, 3.3177497633887114e-10,
                        0.09271805982152269, 4.871382112111268e-9, 2.2671963930992266e-9,
                        2.6584599289863144e-9, 1.6537639325487554e-9, 2.290703448901857e-9,
                        2.6855365271274973e-9, 9.376695285858486e-9, 0.5825009836198609,
                        2.6508193716385394e-10, 2.015718914440205e-9, 3.243922600581835e-10,
                        1.0969736485116875e-9, 1.2954614055481436e-9, 1.6973876211606508e-9,
                        1.0651967980173403e-9, 0.5200322956777765], rtol = 1e-6)

        plc = SemiDefinitePhylogenyEstimator(; pe = clr, p = 5)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, nplg = plc,
                            wb = WeightBounds(; lb = -1, ub = 1))
        res2 = optimise!(MeanRisk(; r = ConditionalValueatRisk(),
                                  obj = MaximumRatio(; rf = rf), opt = opt))
        @test isapprox(value.(res2.nplg.A .* res2.model[:W]), zeros(size(pr.sigma)),
                       atol = 1e-10)
        @test !isapprox(res1.w, res2.w; rtol = 0.25)
        @test isapprox(res2.w,
                       [7.563768267060532e-12, -0.011053444722900886,
                        1.2744522552340983e-11, 0.13873005981734254, 1.5677466758219746e-11,
                        1.8524261660343e-10, 2.3274378326879765e-11, 6.492695715288878e-12,
                        8.40443486629485e-11, 7.959007116833366e-12, 3.029139212113755e-11,
                        0.38059378591301724, 6.013140331476557e-12, 7.0627517783096286e-12,
                        3.653187579176551e-12, 4.366490736778463e-12,
                        1.2755045210929593e-11, 6.849289656481212e-12,
                        5.543601714607396e-12, 0.4917295985730075], rtol = 1e-6)

        plc = SemiDefinitePhylogenyEstimator(; pe = clr)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, cplg = plc,
                            wb = WeightBounds(; lb = -1, ub = 1))
        res1 = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumUtility(),
                                  opt = opt))
        @test isapprox(value.(res1.cplg.A .* res1.model[:W]), zeros(size(pr.sigma)),
                       atol = 1e-10)
        @test isapprox(res1.w,
                       [4.427061986438287e-10, -1.5714922493260265e-9,
                        1.5665474393730487e-9, 6.464906678384926e-10, 0.23327154351074547,
                        6.883372665023997e-10, 0.10470164777556355, 0.2657824913228072,
                        0.13530435887996797, 1.5888393849711381e-9, 2.076666772942388e-9,
                        0.25666270299728206, 2.8318950831949517e-9, 1.4710404832889636e-9,
                        6.474222570587401e-10, 1.4440379282232375e-9,
                        3.9753828604501704e-10, 7.999899264306004e-10,
                        2.0696138171794054e-9, 0.004277240413999178], rtol = 1e-6)

        plc = SemiDefinitePhylogenyEstimator(; pe = clr, p = 5)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, nplg = plc,
                            wb = WeightBounds(; lb = -1, ub = 1))
        res2 = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumUtility(),
                                  opt = opt))
        @test isapprox(value.(res2.nplg.A .* res2.model[:W]), zeros(size(pr.sigma)),
                       atol = 1e-10)
        @test isapprox(res2.w,
                       [2.17025563455507e-10, 6.087556118806929e-11, 7.438771978806892e-10,
                        4.138039205626746e-10, 0.0687583562872231, 1.9489026251735372e-10,
                        0.09508279540606476, 0.04281573130518989, 0.2378844989667673,
                        2.088540397904443e-10, 2.9156834912252903e-10, 0.2931290462533088,
                        4.4571619234844804e-10, 2.1531814553024086e-10,
                        1.2031301343541658e-10, 2.5715975023297707e-10,
                        9.683684671271267e-11, 1.1898969632271358e-10,
                        1.7783204947614032e-10, 0.26232956821838505], rtol = 1e-6)
    end
    @testset "Centrality" begin
        ces = [CentralityConstraint(; A = CentralityEstimator(), B = MinValue(),
                                    comp = GEQ()),
               CentralityConstraint(;
                                    A = CentralityEstimator(;
                                                            cent = EigenvectorCentrality()),
                                    B = MeanValue(), comp = LEQ()),
               CentralityConstraint(;
                                    A = CentralityEstimator(; cent = ClosenessCentrality()),
                                    B = MedianValue(), comp = EQ()),
               CentralityConstraint(; A = CentralityEstimator(; cent = StressCentrality()),
                                    B = MaxValue(), comp = EQ()),
               CentralityConstraint(;
                                    A = CentralityEstimator(; cent = RadialityCentrality()),
                                    B = 0.63, comp = EQ())]

        res = optimise!(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1,
                                                     bgt = 1, cent = ces[1],
                                                     wb = WeightBounds(; lb = -1, ub = 1))))
        @test average_centrality(ces[1].A, res.w, pr.X) >=
              minimum(centrality_vector(ces[1].A, pr.X).X)

        res = optimise!(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1,
                                                     bgt = 1, cent = ces[2],
                                                     wb = WeightBounds(; lb = -1, ub = 1))))
        @test average_centrality(ces[2].A, res.w, pr.X) <=
              mean(centrality_vector(ces[2].A, pr.X).X)

        res = optimise!(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1,
                                                     bgt = 1, cent = ces[3],
                                                     wb = WeightBounds(; lb = -1, ub = 1))))
        @test isapprox(average_centrality(ces[3].A, res.w, pr.X),
                       median(centrality_vector(ces[3].A, pr.X).X))

        res = optimise!(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1,
                                                     bgt = 1, cent = ces[4],
                                                     wb = WeightBounds(; lb = -1, ub = 1))))
        @test isapprox(average_centrality(ces[4].A, res.w, pr.X),
                       maximum(centrality_vector(ces[4].A, pr.X).X))

        res = optimise!(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1,
                                                     bgt = 1, cent = ces[5],
                                                     wb = WeightBounds(; lb = -1, ub = 1))))
        @test isapprox(average_centrality(ces[5].A, res.w, pr.X), 0.63)

        @test isapprox(res.w,
                       optimise!(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                          opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                              sbgt = 1, bgt = 1,
                                                              cent = centrality_constraints(ces[5],
                                                                                            pr.X),
                                                              wb = WeightBounds(; lb = -1,
                                                                                ub = 1)))).w)
    end
    @testset "Fees" begin
        r = ConditionalDrawdownatRisk()
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1))
        w1 = optimise!(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt)).w

        fees = FeesEstimator(;
                             tn = TurnoverEstimator(; w = w0,
                                                    val = Dict("XOM" => 0.3 / 252)),
                             l = Dict("JNJ" => 0.1 / 252), s = "BBY" => 0.2 / 252,
                             fl = ["HD" => 0.016 / 252], fs = "PFE" => 0.03 / 252)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), fees = fees, sets = sets)
        res = optimise!(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt))
        @test isapprox(res.w,
                       [-0.09154878925123214, -0.000997603027571535, -0.10856305089134274,
                        -0.08390917925486328, 0.039259568394976366, -0.02505835967729264,
                        0.148768780820147, 0.3885052932250303, 0.20819565141904295,
                        0.18472171066329013, 0.13841620449649236, 0.15039298390094433,
                        0.0742223213372, 0.33271399336428914, -0.415849263322291,
                        -0.199084396381982, -0.07498935819338508, 0.10263523977758801,
                        0.051767300799461936, 0.1804009518014979], rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1),
                            fees = fees_constraints(fees, sets))
        @test isapprox(res.w,
                       optimise!(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt)).w)

        fees = FeesEstimator(; tn = TurnoverEstimator(; w = w0, val = Dict("XOM" => 1)),
                             l = Dict("JNJ" => 1), s = "BBY" => 1, fl = ["HD" => 1],
                             fs = "PFE" => 1)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                            wb = WeightBounds(; lb = -1, ub = 1), fees = fees, sets = sets)
        res = optimise!(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt))
        @test isapprox(res.w[findfirst(x -> x == "XOM", rd.nx)],
                       w0[findfirst(x -> x == "XOM", rd.nx)])
        @test isapprox(res.w[findfirst(x -> x == "JNJ", rd.nx)], 0)
        @test isapprox(res.w[findfirst(x -> x == "BBY", rd.nx)], 0)
        @test isapprox(res.w[findfirst(x -> x == "HD", rd.nx)], 0)
        @test isapprox(res.w[findfirst(x -> x == "PFE", rd.nx)], 0)
        @test isapprox(res.w,
                       [0.05201790942682152, -0.004972845913212141, -0.5888432775595224,
                        0.0, 0.22730252177448815, -0.02316379567148528, -0.0, 0.0,
                        0.4423716947104361, 0.17060216343485277, -0.030706210074928315,
                        0.2813132780743069, -0.03468309934130539, 0.40833226855001803, 0.0,
                        -0.22737493979523524, -0.061270023409634534, 0.2636697569240946,
                        0.07540459887030522, 0.05], rtol = 1e-6)

        fees = FeesEstimator(; fl = ["JNJ" => 1])
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, bgt = 1, fees = fees, sets = sets)
        res = optimise!(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt))
        @test isapprox(res.w[findfirst(x -> x == "JNJ", rd.nx)], 0)
        @test isapprox(res.w,
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0,
                        0.10628880639076618, 0.37971339847731767, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0996419548234818, 0.21860531676987255, 0.1957505235385615],
                       rtol = 1e-6)
    end
    @testset "Variance risk contribution" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets)
        lcs = LinearConstraintEstimator(;
                                        val = [["$a <= 0.2" for a in rd.nx];
                                               ["c3 <= 0.1"]])
        r = Variance(; rc = lcs)
        obj = MaximumRatio()
        mr = MeanRisk(; obj = obj, opt = opt, r = r)
        res = optimise!(mr)
        rkc = risk_contribution(factory(r, pr, slv), res.w, pr)
        rkc = rkc / sum(rkc)
        @test all(rkc .- 0.2 .- 20 * sqrt(eps()) .< 0)
        @test abs(sum(rkc[3:3:end]) - 0.1) < 20 * sqrt(eps())
    end
    @testset "Weighted risk expressions" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        mip_opt = JuMPOptimiser(; pe = pr, slv = mip_slv)
        rs1 = [LowOrderMoment(; mu = 0),
               LowOrderMoment(; mu = 0,
                              alg = LowOrderDeviation(; alg = SecondLowerMoment())),
               LowOrderMoment(; mu = 0,
                              alg = LowOrderDeviation(; alg = SecondCentralMoment())),
               ConditionalValueatRisk(), EntropicValueatRisk(),
               ConditionalValueatRiskRange(), EntropicValueatRiskRange(),
               DistributionallyRobustConditionalValueatRisk(; l = 1e-1, r = 1e-3),
               DistributionallyRobustConditionalValueatRiskRange(; l_a = 1e-1, r_a = 1e-3,
                                                                 l_b = 1e-1, r_b = 1e-3),
               ValueatRisk(), ValueatRiskRange(),
               ValueatRisk(; alg = DistributionValueatRisk()),
               ValueatRiskRange(; alg = DistributionValueatRisk())]
        rs2 = [LowOrderMoment(; mu = 0, w = wp),
               LowOrderMoment(; mu = 0, w = wp,
                              alg = LowOrderDeviation(; alg = SecondLowerMoment())),
               LowOrderMoment(; mu = 0, w = wp,
                              alg = LowOrderDeviation(; alg = SecondCentralMoment())),
               ConditionalValueatRisk(; w = wp), EntropicValueatRisk(; w = wp),
               ConditionalValueatRiskRange(; w = wp), EntropicValueatRiskRange(; w = wp),
               DistributionallyRobustConditionalValueatRisk(; l = 1e-1, r = 1e-3, w = wp),
               DistributionallyRobustConditionalValueatRiskRange(; l_a = 1e-1, r_a = 1e-3,
                                                                 l_b = 1e-1, r_b = 1e-3,
                                                                 w = wp),
               ValueatRisk(; w = wp), ValueatRiskRange(; w = wp),
               ValueatRisk(; alg = DistributionValueatRisk(), w = wp),
               ValueatRiskRange(; alg = DistributionValueatRisk(), w = wp)]
        for (i, (r1, r2)) in enumerate(zip(rs1, rs2))
            res1, res2 = if isa(r1,
                                Union{<:ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk},
                                      <:ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                         <:MIPValueatRisk}})
                optimise!(MeanRisk(; r = r1, opt = mip_opt)),
                optimise!(MeanRisk(; r = r2, opt = mip_opt))
            else
                optimise!(MeanRisk(; r = r1, opt = opt)),
                optimise!(MeanRisk(; r = r2, opt = opt))
            end
            rtol = if i ∈ (2, 7)
                5e-5
            elseif i == 3
                5e-3
            elseif i == 5
                1e-4
            else
                1e-6
            end
            res = isapprox(res1.w, res2.w; rtol = rtol)
            if !res
                println("Iteration $i failed:")
                find_tol(res1.w, res2.w)
                display([res1.w res2.w res1.w - res2.w])
            end
            @test res
        end
    end
end
