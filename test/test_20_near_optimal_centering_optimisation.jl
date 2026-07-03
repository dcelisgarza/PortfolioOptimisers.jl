@testset "NearOptimalCentering Optimisation" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames, StableRNGs,
          LinearAlgebra
    ts = Date.(["2021-12-29", "2021-12-30", "2021-12-31", "2022-01-03", "2022-01-04",
                "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-10", "2022-01-11",
                "2022-01-12", "2022-01-13", "2022-01-14", "2022-01-18", "2022-01-19",
                "2022-01-20", "2022-01-21", "2022-01-24", "2022-01-25", "2022-01-26",
                "2022-01-27", "2022-01-28", "2022-01-31", "2022-02-01", "2022-02-02",
                "2022-02-03", "2022-02-04", "2022-02-07", "2022-02-08", "2022-02-09",
                "2022-02-10", "2022-02-11", "2022-02-14", "2022-02-15", "2022-02-16",
                "2022-02-17", "2022-02-18", "2022-02-22", "2022-02-23", "2022-02-24",
                "2022-02-25", "2022-02-28", "2022-03-01", "2022-03-02", "2022-03-03",
                "2022-03-04", "2022-03-07", "2022-03-08", "2022-03-09", "2022-03-10",
                "2022-03-11", "2022-03-14", "2022-03-15", "2022-03-16", "2022-03-17",
                "2022-03-18", "2022-03-21", "2022-03-22", "2022-03-23", "2022-03-24",
                "2022-03-25", "2022-03-28", "2022-03-29", "2022-03-30", "2022-03-31",
                "2022-04-01", "2022-04-04", "2022-04-05", "2022-04-06", "2022-04-07",
                "2022-04-08", "2022-04-11", "2022-04-12", "2022-04-13", "2022-04-14",
                "2022-04-18", "2022-04-19", "2022-04-20", "2022-04-21", "2022-04-22",
                "2022-04-25", "2022-04-26", "2022-04-27", "2022-04-28", "2022-04-29",
                "2022-05-02", "2022-05-03", "2022-05-04", "2022-05-05", "2022-05-06",
                "2022-05-09", "2022-05-10", "2022-05-11", "2022-05-12", "2022-05-13",
                "2022-05-16", "2022-05-17", "2022-05-18", "2022-05-19", "2022-05-20",
                "2022-05-23", "2022-05-24", "2022-05-25", "2022-05-26", "2022-05-27",
                "2022-05-31", "2022-06-01", "2022-06-02", "2022-06-03", "2022-06-06",
                "2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10", "2022-06-13",
                "2022-06-14", "2022-06-15", "2022-06-16", "2022-06-17", "2022-06-21",
                "2022-06-22", "2022-06-23", "2022-06-24", "2022-06-27", "2022-06-28",
                "2022-06-29", "2022-06-30", "2022-07-01", "2022-07-05", "2022-07-06",
                "2022-07-07", "2022-07-08", "2022-07-11", "2022-07-12", "2022-07-13",
                "2022-07-14", "2022-07-15", "2022-07-18", "2022-07-19", "2022-07-20",
                "2022-07-21", "2022-07-22", "2022-07-25", "2022-07-26", "2022-07-27",
                "2022-07-28", "2022-07-29", "2022-08-01", "2022-08-02", "2022-08-03",
                "2022-08-04", "2022-08-05", "2022-08-08", "2022-08-09", "2022-08-10",
                "2022-08-11", "2022-08-12", "2022-08-15", "2022-08-16", "2022-08-17",
                "2022-08-18", "2022-08-19", "2022-08-22", "2022-08-23", "2022-08-24",
                "2022-08-25", "2022-08-26", "2022-08-29", "2022-08-30", "2022-08-31",
                "2022-09-01", "2022-09-02", "2022-09-06", "2022-09-07", "2022-09-08",
                "2022-09-09", "2022-09-12", "2022-09-13", "2022-09-14", "2022-09-15",
                "2022-09-16", "2022-09-19", "2022-09-20", "2022-09-21", "2022-09-22",
                "2022-09-23", "2022-09-26", "2022-09-27", "2022-09-28", "2022-09-29",
                "2022-09-30", "2022-10-03", "2022-10-04", "2022-10-05", "2022-10-06",
                "2022-10-07", "2022-10-10", "2022-10-11", "2022-10-12", "2022-10-13",
                "2022-10-14", "2022-10-17", "2022-10-18", "2022-10-19", "2022-10-20",
                "2022-10-21", "2022-10-24", "2022-10-25", "2022-10-26", "2022-10-27",
                "2022-10-28", "2022-10-31", "2022-11-01", "2022-11-02", "2022-11-03",
                "2022-11-04", "2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10",
                "2022-11-11", "2022-11-14", "2022-11-15", "2022-11-16", "2022-11-17",
                "2022-11-18", "2022-11-21", "2022-11-22", "2022-11-23", "2022-11-25",
                "2022-11-28", "2022-11-29", "2022-11-30", "2022-12-01", "2022-12-02",
                "2022-12-05", "2022-12-06", "2022-12-07", "2022-12-08", "2022-12-09",
                "2022-12-12", "2022-12-13", "2022-12-14", "2022-12-15", "2022-12-16",
                "2022-12-19", "2022-12-20", "2022-12-21", "2022-12-22", "2022-12-23",
                "2022-12-27", "2022-12-28"])
    iv = TimeArray(ts, rand(StableRNG(123), 252, 20))
    ivpa = rand(StableRNG(123), 20)
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end]; iv = iv,
                           ivpa = ivpa)
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = false, allow_almost = false),
                  settings = Dict("verbose" => false)),
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = false, allow_almost = false),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.95)),
           Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = false, allow_almost = false),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
           Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = false, allow_almost = false),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.85)),
           Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = false, allow_almost = false),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.80)),
           Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = false, allow_almost = false),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
           Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = false, allow_almost = false),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.7)),
           Solver(; name = :clarabel8, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = false, allow_almost = false),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.6,
                                  "max_iter" => 1500, "tol_gap_abs" => 1e-4,
                                  "tol_gap_rel" => 1e-4, "tol_ktratio" => 1e-3,
                                  "tol_feas" => 1e-4, "tol_infeas_abs" => 1e-4,
                                  "tol_infeas_rel" => 1e-4, "reduced_tol_gap_abs" => 1e-4,
                                  "reduced_tol_gap_rel" => 1e-4,
                                  "reduced_tol_ktratio" => 1e-3, "reduced_tol_feas" => 1e-4,
                                  "reduced_tol_infeas_abs" => 1e-4,
                                  "reduced_tol_infeas_rel" => 1e-4))]
    pr = prior(HighOrderPriorEstimator(), rd)
    rf = 4.2 / 100 / 252
    @testset "Unconstrained" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier1.csv.gz"),
                      DataFrame)
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = factory(StandardDeviation(), pr)
        res_min = optimise(MeanRisk(; r = r, opt = opt))
        res_max = optimise(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        rk_min = expected_risk(r, res_min.w, pr)
        rk_max = expected_risk(r, res_max.w, pr)
        rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
        rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
        pred = fit_predict(NearOptimalCentering(;
                                                r = StandardDeviation(;
                                                                      settings = RiskMeasureSettings(;
                                                                                                     ub = Frontier(;
                                                                                                                   N = 5))),
                                                obj = MaximumReturn(), opt = opt), rd)
        res1 = pred.res
        @test pred.rd.nx == rd.nx
        @test pred.rd.X == calc_net_returns(pred.res, rd.X)
        @test pred.rd.ts == rd.ts
        @test pred.rd.iv == [rd.iv * w for w in pred.res.w]
        @test pred.rd.ivpa == [dot(rd.ivpa, w) for w in pred.res.w]
        res2 = optimise(NearOptimalCentering(;
                                             r = StandardDeviation(;
                                                                   settings = RiskMeasureSettings(;
                                                                                                  ub = range(;
                                                                                                             start = rk_min,
                                                                                                             stop = rk_max,
                                                                                                             length = 5))),
                                             obj = MaximumReturn(), opt = opt))
        @test all(isapprox.(res1.w, res2.w))
        success = isapprox(Matrix(df), hcat(res1.w...); rtol = 1e-4)
        if !success
            find_tol(Matrix(df), hcat(res1.w...))
        end
        @test success

        df = CSV.read(joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier2.csv.gz"),
                      DataFrame)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            ret = ArithmeticReturn(; lb = Frontier(; N = 5)))
        res3 = optimise(NearOptimalCentering(; r = StandardDeviation(), opt = opt))
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            ret = ArithmeticReturn(;
                                                   lb = range(; start = rt_min,
                                                              stop = rt_max, length = 5)))
        res4 = optimise(NearOptimalCentering(; r = StandardDeviation(), opt = opt))
        @test all(isapprox.(res3.w, res4.w))
        success = isapprox(Matrix(df), hcat(res3.w...); rtol = 1e-4)
        if !success
            find_tol(Matrix(df), hcat(res3.w...))
        end
        @test success

        res = optimise(NearOptimalCentering(;
                                            opt = JuMPOptimiser(; pe = pr,
                                                                slv = Solver(;
                                                                             solver = Clarabel.Optimizer,
                                                                             settings = ["verbose" =>
                                                                                             false,
                                                                                         "max_iter" =>
                                                                                             1])),
                                            fb = InverseVolatility(; pe = pr)))
        @test isapprox(res.w, optimise(InverseVolatility(; pe = pr)).w)

        w0 = range(; start = inv(length(pr.mu)), stop = inv(length(pr.mu)),
                   length = length(pr.mu))
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        res = optimise(NearOptimalCentering(; w_min_ini = w0,
                                            w_min = optimise(MeanRisk(; opt = opt)).w,
                                            w_opt_ini = w0,
                                            w_opt = optimise(MeanRisk(;
                                                                      obj = MaximumUtility(),
                                                                      opt = opt)).w,
                                            w_max_ini = w0,
                                            w_max = optimise(MeanRisk(;
                                                                      obj = MaximumReturn(),
                                                                      opt = opt)).w,
                                            bins = 20,
                                            opt = JuMPOptimiser(; pe = pr,
                                                                slv = Solver(;
                                                                             solver = Clarabel.Optimizer,
                                                                             settings = ["verbose" =>
                                                                                             false,
                                                                                         "max_iter" =>
                                                                                             1])),
                                            fb = InverseVolatility(; pe = pr)))
        @test isapprox(res.w, optimise(InverseVolatility(; pe = pr)).w)
    end
    @testset "Constrained" begin
        ivpa = rand(StableRNG(123))
        rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__,
                                                           "./assets/SP500.csv.gz"));
                                         timestamp = :Date)[(end - 252):end]; iv = iv,
                               ivpa = ivpa)
        pr = prior(HighOrderPriorEstimator(), rd)
        df = CSV.read(joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier3.csv.gz"),
                      DataFrame)
        opt = JuMPOptimiser(; pe = pr, slv = reverse(slv))
        r = factory(StandardDeviation(), pr)
        res_min = optimise(MeanRisk(; r = r, opt = opt))
        res_max = optimise(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        rk_min = expected_risk(r, res_min.w, pr)
        rk_max = expected_risk(r, res_max.w, pr)
        rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
        rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
        pred = fit_predict(NearOptimalCentering(;
                                                r = StandardDeviation(;
                                                                      settings = RiskMeasureSettings(;
                                                                                                     ub = Frontier(;
                                                                                                                   N = 5))),
                                                obj = MaximumReturn(), opt = opt,
                                                alg = ConstrainedNearOptimalCentering()),
                           rd)
        res1 = pred.res
        @test all(x -> x == ivpa, pred.rd.ivpa)
        res2 = optimise(NearOptimalCentering(;
                                             r = StandardDeviation(;
                                                                   settings = RiskMeasureSettings(;
                                                                                                  ub = range(;
                                                                                                             start = rk_min,
                                                                                                             stop = rk_max,
                                                                                                             length = 5))),
                                             obj = MaximumReturn(), opt = opt,
                                             alg = ConstrainedNearOptimalCentering()))
        @test all(isapprox.(res1.w, res2.w))
        success = isapprox(Matrix(df), hcat(res1.w...); rtol = 5e-5)
        if !success
            find_tol(Matrix(df), hcat(res1.w...))
        end
        @test success

        df = CSV.read(joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier4.csv.gz"),
                      DataFrame)
        opt = JuMPOptimiser(; pe = pr, slv = reverse(slv),
                            ret = ArithmeticReturn(; lb = Frontier(; N = 5)))
        res3 = optimise(NearOptimalCentering(; r = StandardDeviation(), opt = opt,
                                             alg = ConstrainedNearOptimalCentering()))
        opt = JuMPOptimiser(; pe = pr, slv = reverse(slv),
                            ret = ArithmeticReturn(;
                                                   lb = range(; start = rt_min,
                                                              stop = rt_max, length = 5)))
        res4 = optimise(NearOptimalCentering(; r = StandardDeviation(), opt = opt,
                                             alg = ConstrainedNearOptimalCentering()))
        @test all(isapprox.(res3.w, res4.w))
        success = isapprox(Matrix(df), hcat(res3.w...); rtol = 1e-4)
        if !success
            find_tol(Matrix(df), hcat(res3.w...))
        end
        @test success

        res = optimise(NearOptimalCentering(; alg = ConstrainedNearOptimalCentering(),
                                            opt = JuMPOptimiser(; pe = pr,
                                                                slv = Solver(;
                                                                             solver = Clarabel.Optimizer,
                                                                             settings = ["verbose" =>
                                                                                             false,
                                                                                         "max_iter" =>
                                                                                             1])),
                                            fb = InverseVolatility(; pe = pr)))
        @test isapprox(res.w, optimise(InverseVolatility(; pe = pr)).w)
    end
    @testset "Pareto Surface" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        df = CSV.read(joinpath(@__DIR__,
                               "./assets/NearOptimalCenteringParetoSurface1.csv.gz"),
                      DataFrame)
        r1 = StandardDeviation(;
                               settings = RiskMeasureSettings(; scale = 2e2,
                                                              ub = Frontier(; N = 3)))
        r2 = ConditionalValueatRisk(;
                                    settings = RiskMeasureSettings(;
                                                                   ub = Frontier(; N = 5)))
        res1 = optimise(NearOptimalCentering(; r = [r1, r2], obj = MaximumReturn(),
                                             opt = opt))
        success = isapprox(Matrix(df), hcat(res1.w...); rtol = 5e-5)
        if !success
            find_tol(Matrix(df), hcat(res1.w...))
        end
        @test success

        r1 = factory(StandardDeviation(; settings = RiskMeasureSettings(; scale = 2e2)), pr)
        r2 = ConditionalValueatRisk(; settings = RiskMeasureSettings(;))
        res_min = optimise(MeanRisk(; r = [r1, r2], opt = opt))
        res_max = optimise(MeanRisk(; r = [r1, r2], obj = MaximumReturn(), opt = opt))
        rk1_min = expected_risk(r1, res_min.w, pr)
        rk1_max = expected_risk(r1, res_max.w, pr)
        rk2_min = expected_risk(r2, res_min.w, pr)
        rk2_max = expected_risk(r2, res_max.w, pr)
        rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
        rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)

        df = CSV.read(joinpath(@__DIR__,
                               "./assets/NearOptimalCenteringParetoSurface2.csv.gz"),
                      DataFrame)
        opt = JuMPOptimiser(; pe = pr,
                            ret = ArithmeticReturn(;
                                                   lb = range(; start = rt_min,
                                                              stop = rt_min +
                                                                     0.0009316063452440891,
                                                              length = 3)), slv = slv)
        r1 = StandardDeviation(;
                               settings = RiskMeasureSettings(; scale = 2e2,
                                                              ub = range(;
                                                                         start = rk1_max -
                                                                                 0.006363402327019335,
                                                                         stop = rk1_max,
                                                                         length = 3)))
        r2 = ConditionalValueatRisk(;
                                    settings = RiskMeasureSettings(;
                                                                   ub = range(;
                                                                              start = rk2_max -
                                                                                      0.014823231590460057,
                                                                              stop = rk2_max,
                                                                              length = 3)))
        res2 = optimise(NearOptimalCentering(; r = [r1, r2], obj = MaximumUtility(),
                                             opt = opt))
        success = isapprox(Matrix(df), hcat(res2.w...); rtol = 1e-4)
        if !success
            find_tol(Matrix(df), hcat(res2.w...))
        end
        @test success

        r1 = factory(StandardDeviation(), pr)
        r2 = ConditionalValueatRisk(; settings = RiskMeasureSettings(;))
        res_min = optimise(MeanRisk(; r = r1, opt = opt))
        res_max = optimise(MeanRisk(; r = r1, obj = MaximumReturn(), opt = opt))
        rk1_min = expected_risk(r1, res_min.w, pr)
        rk1_max = expected_risk(r1, res_max.w, pr)
        rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
        rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
        drk1 = mean(diff(rk1_min))
        drt = mean(diff(rt_min))
        df = CSV.read(joinpath(@__DIR__,
                               "./assets/NearOptimalCenteringParetoSurfaceRetRk.csv.gz"),
                      DataFrame)
        res3 = optimise(NearOptimalCentering(; obj = MaximumUtility(),
                                             r = StandardDeviation(;
                                                                   settings = RiskMeasureSettings(;
                                                                                                  ub = rk1_min .+
                                                                                                       drk1)),
                                             opt = JuMPOptimiser(; pe = pr,
                                                                 ret = ArithmeticReturn(;
                                                                                        lb = rt_min .-
                                                                                             drt),
                                                                 slv = slv),
                                             alg = ConstrainedNearOptimalCentering()))
        success = isapprox(Matrix(df), hcat(res3.w...); rtol = 5e-4)
        if !success
            find_tol(Matrix(df), hcat(res3.w...))
        end
        @test success

        df = CSV.read(joinpath(@__DIR__,
                               "./assets/NearOptimalCenteringParetoSurface3.csv.gz"),
                      DataFrame)
        opt = JuMPOptimiser(; pe = pr, slv = slv, sca = MaxScalariser())
        r1 = StandardDeviation(;
                               settings = RiskMeasureSettings(; scale = 2e2,
                                                              ub = Frontier(; N = 3)))
        r2 = ConditionalValueatRisk(;
                                    settings = RiskMeasureSettings(;
                                                                   ub = Frontier(; N = 5)))
        res4 = optimise(NearOptimalCentering(; r = [r1, r2], obj = MaximumReturn(),
                                             opt = opt,
                                             alg = ConstrainedNearOptimalCentering()))
        success = isapprox(Matrix(df), hcat(res4.w...); rtol = 5e-5)
        if !success
            find_tol(Matrix(df), hcat(res4.w...))
        end
        @test success
    end
    @testset "Scalarisers" begin
        r1 = NegativeSkewness(; alg = SOCRiskExpr(),
                              settings = RiskMeasureSettings(; scale = 50))
        r2 = ConditionalValueatRisk(; settings = RiskMeasureSettings(;))

        res_m1 = optimise(NearOptimalCentering(; r = r1, obj = MaximumRatio(; rf = rf),
                                               opt = JuMPOptimiser(; pe = pr, slv = slv)))
        res_m2 = optimise(NearOptimalCentering(; r = r2, obj = MaximumRatio(; rf = rf),
                                               opt = JuMPOptimiser(; pe = pr, slv = slv)))

        res1 = optimise(NearOptimalCentering(; r = [r1, r2], obj = MaximumRatio(; rf = rf),
                                             opt = JuMPOptimiser(; pe = pr, slv = slv)))
        res2 = optimise(NearOptimalCentering(; r = [r1, r2], obj = MaximumRatio(; rf = rf),
                                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                 sca = MaxScalariser())))
        res3 = optimise(NearOptimalCentering(; r = [r1, r2], obj = MaximumRatio(; rf = rf),
                                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                 sca = LogSumExpScalariser(;
                                                                                           gamma = 8.5e-4))))
        res4 = optimise(NearOptimalCentering(; r = [r1, r2], obj = MaximumRatio(; rf = rf),
                                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                 sca = LogSumExpScalariser(;
                                                                                           gamma = 500))))
        @test isapprox(res2.w, res_m1.w, rtol = 1e-4)
        @test isapprox(res1.w, res3.w, rtol = 1e-3)
        @test isapprox(res4.w, res_m1.w, rtol = 1e-4)
    end
    @testset "UncertaintySetVariance barrier target parity" begin
        # Regression: the barrier risk targets used to be computed from the nominal
        # variance functor while the model risk expression is the worst-case variance,
        # making the noc_opt sub-problem provably infeasible. Solver-level stalls
        # (slow progress or numerical error, seen with ellipsoidal sets) are acceptable;
        # a proven INFEASIBLE noc_opt is the bug and must fail.
        function noc_solver_stalled(rc)
            isa(rc, OptimisationFailure) || return false
            err = string(get(get(rc.res, :clarabel1, Dict()), :err, ""))
            return occursin("SLOW_PROGRESS", err) || occursin("NUMERICAL_ERROR", err)
        end
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        w0 = fill(inv(size(pr.X, 2)), size(pr.X, 2))
        ucs1 = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(),
                                              rng = StableRNG(987654321),
                                              alg = BoxUncertaintySetAlgorithm()), rd.X)
        ucs2 = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(),
                                              rng = StableRNG(987654321),
                                              alg = EllipsoidalUncertaintySetAlgorithm()),
                         rd.X)
        ucs3 = NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(987654321))
        @test PortfolioOptimisers.ucs_variance(ucs1, pr.sigma, w0) >= dot(w0, pr.sigma, w0)
        @test PortfolioOptimisers.ucs_variance(ucs2, pr.sigma, w0) >= dot(w0, pr.sigma, w0)
        # Functor is worst-case with a fitted set, nominal with an estimator.
        for ucs in (ucs1, ucs2)
            rf1 = factory(UncertaintySetVariance(; ucs = ucs), pr, slv)
            @test rf1(w0) == PortfolioOptimisers.ucs_variance(ucs, pr.sigma, w0)
        end
        rf2 = factory(UncertaintySetVariance(; ucs = ucs3), pr, slv)
        @test rf2(w0) == dot(w0, pr.sigma, w0)
        # Overall failure retcode carries the sub-problem retcodes and diagnostics.
        rc = PortfolioOptimisers.get_overall_retcode(OptimisationFailure(;
                                                                         res = Dict(:a => 1)),
                                                     OptimisationSuccess(),
                                                     OptimisationSuccess(),
                                                     OptimisationSuccess())
        @test isa(rc, OptimisationFailure)
        @test occursin("w_min failed", rc.res.msg)
        @test isa(rc.res.w_min, OptimisationFailure) && rc.res.w_min.res[:a] == 1
        @test isa(rc.res.noc_opt, OptimisationSuccess)
        for ucs in (ucs1, ucs2, ucs3)
            r = UncertaintySetVariance(; ucs = ucs)
            res1 = optimise(NearOptimalCentering(; r = r, opt = opt), rd)
            @test isa(res1.retcode, OptimisationSuccess) ||
                  noc_solver_stalled(res1.noc_retcode)
            res2 = optimise(NearOptimalCentering(; r = r, opt = opt,
                                                 alg = ConstrainedNearOptimalCentering()),
                            rd)
            @test isa(res2.retcode, OptimisationSuccess) ||
                  noc_solver_stalled(res2.noc_retcode)
        end
    end
end
