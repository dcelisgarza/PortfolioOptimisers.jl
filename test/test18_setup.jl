# Shared fixtures + Mean Risk block functions for the test_18 split files.
# Not a test file (no `test_` prefix); excluded from discovery, included by
# each split file. Mean Risk blocks are sliced by `rs[idx]`; the absolute `i`
# counter is preserved so the hardcoded per-`i` tolerance tables stay valid.
# See ADR 0003.

using Test, PortfolioOptimisers, DataFrames, CSV, TimeSeries, Clarabel, HiGHS, Pajarito,
      JuMP, StatsBase, StableRNGs, LinearAlgebra, Distributions, StableRNGs, SCS
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
                                 timestamp = :Date)[(end - 252):end];
                       B = TimeArray(CSV.File(joinpath(@__DIR__,
                                                       "./assets/SP500_idx.csv.gz"));
                                     timestamp = :Date), iv = iv, ivpa = ivpa)
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
                              "reduced_tol_gap_rel" => 1e-4, "reduced_tol_ktratio" => 1e-3,
                              "reduced_tol_feas" => 1e-4, "reduced_tol_infeas_abs" => 1e-4,
                              "reduced_tol_infeas_rel" => 1e-4))]
mip_slv = [Solver(; name = :mip1,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip2,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.95)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip3,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.90)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip4,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.85)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip5,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.80)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip6,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.75)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip7,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.7)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip8,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.6,
                                                                                   "max_iter" =>
                                                                                       1500,
                                                                                   "tol_gap_abs" =>
                                                                                       1e-4,
                                                                                   "tol_gap_rel" =>
                                                                                       1e-4,
                                                                                   "tol_ktratio" =>
                                                                                       1e-3,
                                                                                   "tol_feas" =>
                                                                                       1e-4,
                                                                                   "tol_infeas_abs" =>
                                                                                       1e-4,
                                                                                   "tol_infeas_rel" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_gap_abs" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_gap_rel" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_ktratio" =>
                                                                                       1e-3,
                                                                                   "reduced_tol_feas" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_infeas_abs" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_infeas_rel" =>
                                                                                       1e-4)),
                  check_sol = (; allow_local = true, allow_almost = true))]

scs_slv = Solver(; name = :scs1, solver = SCS.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
sets = AssetSets(;
                 dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                             "group2" => rd.nx[2:2:end],
                             "clusters1" =>
                                 [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                  3],
                             "clusters2" =>
                                 [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
                                  2], "c1" => rd.nx[1:3:end], "c2" => rd.nx[2:3:end],
                             "c3" => rd.nx[3:3:end],
                             "nx_industries" => ["Technology", "Technology", "Financials",
                                                 "Consumer_Discretionary", "Energy", "Industrials",
                                                 "Consumer_Discretionary", "Healthcare", "Financials",
                                                 "Consumer_Staples", "Healthcare", "Healthcare",
                                                 "Technology", "Consumer_Staples", "Healthcare",
                                                 "Consumer_Staples", "Energy", "Healthcare",
                                                 "Consumer_Staples", "Energy"],
                             "ux_industries" =>
                                 ["Technology", "Financials", "Consumer_Discretionary",
                                  "Energy", "Industrials", "Healthcare",
                                  "Consumer_Staples"]))

pr = prior(HighOrderPriorEstimator(), rd)
clr = clusterise(ClustersEstimator(), pr)
w0 = range(; start = inv(size(pr.X, 2)), stop = inv(size(pr.X, 2)), length = size(pr.X, 2))
wp = StatsBase.pweights(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                              length = size(pr.X, 1)))
ucs1 = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(987654321),
                                      alg = BoxUncertaintySetAlgorithm()), rd.X)
ucs2 = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(987654321),
                                      alg = EllipsoidalUncertaintySetAlgorithm()), rd.X)
rf = 4.2 / 100 / 252
rd2 = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                  timestamp = :Date)[(end - 50):end])
pr2 = prior(HighOrderPriorEstimator(), rd2)

objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(; rf = rf)]
rets = [ArithmeticReturn(), LogarithmicReturn()]
rs = [StandardDeviation(), Variance(), UncertaintySetVariance(; ucs = ucs1),
      UncertaintySetVariance(; ucs = ucs2), LowOrderMoment(),
      LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment(), alg2 = SOCRiskExpr())),
      LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment())),
      LowOrderMoment(; alg = SecondMoment(; alg1 = FullMoment(), alg2 = SOCRiskExpr())),
      LowOrderMoment(; alg = SecondMoment()),
      LowOrderMoment(; alg = MeanAbsoluteDeviation()), WorstRealisation(), Range(),
      ConditionalValueatRisk(), ConditionalValueatRiskRange(), EntropicValueatRisk(),
      EntropicValueatRiskRange(), RelativisticValueatRisk(), RelativisticValueatRiskRange(),
      MaximumDrawdown(), AverageDrawdown(), UlcerIndex(), ConditionalDrawdownatRisk(),
      EntropicDrawdownatRisk(), RelativisticDrawdownatRisk(), Kurtosis(; N = 2), Kurtosis(),
      OrderedWeightsArray(; alg = ExactOrderedWeightsArray()), OrderedWeightsArray(),
      OrderedWeightsArrayRange(; alg = ExactOrderedWeightsArray()),
      OrderedWeightsArrayRange(), NegativeSkewness(),
      NegativeSkewness(; alg = SquaredSOCRiskExpr()),
      DistributionallyRobustConditionalValueatRisk(),
      ValueatRisk(; alg = DistributionValueatRisk()),
      DistributionallyRobustConditionalValueatRiskRange(),
      ValueatRiskRange(; alg = DistributionValueatRisk()), TurnoverRiskMeasure(; w = w0),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0)),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = L1Norm()),
      DistributionallyRobustConditionalDrawdownatRisk(), PowerNormValueatRisk(),
      PowerNormValueatRiskRange(), PowerNormDrawdownatRisk(),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = LpNorm(; p = 2)),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = LInfNorm()),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = LpNorm(; p = 10)),
      LowOrderMoment(; alg = EvenMoment()),
      LowOrderMoment(; alg = EvenMoment(; alg = SemiMoment()))]
tr = WeightsTracking(; w = w0)

function mr_block1(idx)
    df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk1.csv.gz"), DataFrame)
    i = (first(idx) - 1) * 6 + 1
    for r in rs[idx], obj in objs, ret in rets
        if i == 174
            i += 1
            continue
        end
        opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret)
        mr = MeanRisk(; r = r, obj = obj, opt = opt)
        res = optimise(mr, rd)
        @test isa(res.retcode, OptimisationSuccess)
        df[!, "$i"] = res.w
        rtol = if i == 22 && Sys.islinux()
            1e-2
        elseif i in
               (4, 10, 22, 76, 86, 91, 92, 96, 97, 99, 101, 103, 105, 133, 135, 141, 148,
                154, 175, 184, 196, 252, 276, 279, 281, 283, 284, 285)
            5e-5
        elseif i in
               (6, 16, 28, 36, 38, 40, 46, 52, 93, 108, 126, 139, 163, 165, 167, 177, 179,
                192, 204, 214, 216, 254, 264, 278, 282, 286)
            5e-6
        elseif i in (18, 157, 158, 174, 228, 270)
            5e-4
        elseif i in (48, 58, 88, 90, 94, 98, 134, 140, 159, 176, 263, 266, 268, 288)
            1e-5
        elseif i in (160, 164, 180, 287)
            5e-3
        elseif i in (162, 178)
            1e-3
        elseif i in (198, 210)
            5e-2
        elseif i in (208, 234, 246, 269)
            1e-4
        elseif i == 240
            0.25
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
            rkd = zero(eltype(rd.X))
            rtd = zero(eltype(rd.X))
            rk = expected_risk(factory(r, pr, slv), res.w, rd.X)
            rt = expected_return(ret, res.w, pr)
            if i in (23, 24, 174, 209)
                rkd = 7.5e-1 * rk
            elseif i == 197
                rkd = 8e-1 * rk
            elseif i == 198
                rkd = rk
                rtd = rt
            elseif i == 210
                rkd = 8e-1 * rk
                rtd = 9e-1 * rt
            elseif i == 239
                rkd = 7.5e-1 * rk
                # rtd = 1e-6 * rt
            elseif i == 240
                i += 1
                continue
            end
            opt1 = JuMPOptimiser(; pe = pr, slv = slv,
                                 ret = bounds_returns_estimator(ret, rt - rtd))
            mr = MeanRisk(; r = r, opt = opt1)
            res = optimise(mr, rd)
            rt1 = expected_return(ret, res.w, pr)
            if !(isa(r, Kurtosis) && isnothing(r.N))
                tol = if i == 161
                    1e-9
                else
                    1e-10
                end
                flag = rt1 >= rt - rtd || abs(rt1 - rt + rtd) < tol
                if !flag
                    println("Counter: $i")
                    println("rt1: $rt1")
                    println("rt: $rt")
                    println("abs(rt1 - rt): $(abs(rt1 - rt))")
                end
                @test flag
            end
            mr = MeanRisk(; r = bounds_risk_measure(r, rk + rkd), obj = MaximumReturn(),
                          opt = opt)
            res = optimise(mr, rd)
            rk1 = expected_risk(factory(r, pr, slv), res.w, rd.X)
            if !(isa(r, Kurtosis) && isnothing(r.N))
                tol = if i == 161
                    1e-9
                elseif i == 203
                    0.00014
                elseif i == 204
                    0.00022
                elseif i in (149, 150)
                    5e-5
                else
                    1e-10
                end
                flag = rk1 <= rk + rkd || abs(rk1 - rk - rkd) < tol
                if !flag
                    println("Counter: $i")
                    println("rk1: $rk1")
                    println("rk: $rk")
                    println("abs(rk1 - rk): $(abs(rk1 - rk))")
                end
                @test flag
            else
                @test rk1 / rk < 1.15
            end
        end
        if isa(r, Union{<:TurnoverRiskMeasure, <:TrackingRiskMeasure})
            @test PortfolioOptimisers.needs_previous_weights(mr)
        else
            @test !PortfolioOptimisers.needs_previous_weights(mr)
        end
        i += 1
    end
end

function mr_block2(idx)
    df = CSV.read(joinpath(@__DIR__, "./assets/MeanRiskDT.csv.gz"), DataFrame)
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    i = first(idx)
    for r in rs[idx]
        r1 = RiskTrackingRiskMeasure(; tr = tr, r = r, alg = DependentVariableTracking())
        mr = MeanRisk(; r = r1, obj = MaximumRatio(; rf = rf), opt = opt)
        @test PortfolioOptimisers.needs_previous_weights(mr)
        res = optimise(mr, rd)
        if isa(r, PortfolioOptimisers.QuadExpressionRiskMeasures)
            @test isa(res.retcode, OptimisationFailure)
            i += 1
            continue
        else
            @test isa(res.retcode, OptimisationSuccess)
        end
        rtol = if i in (12, 14, 30)
            5e-4
        elseif i in (13, 16, 17, 18)
            0.05
        elseif i in (15, 28, 41, 42)
            0.1
        elseif i == 29
            5e-6
        elseif i in (18, 24)
            5e-3
        elseif i == 22 && Sys.islinux()
            1e-2
        elseif i in (22, 47)
            1e-4
        elseif i == 23
            1e-3
        elseif i in (26, 43, 44, 48)
            5e-5
        elseif i == 4
            1e-5
        else
            1e-6
        end
        success = isapprox(res.w, df[!, "$i"]; rtol = rtol)
        if !success
            println("Counter: $i")
            find_tol(res.w, df[!, "$i"])
            display([res.w df[!, "$i"]])
        end
        @test success
        i += 1
    end
end

function mr_block3(idx)
    df = CSV.read(joinpath(@__DIR__, "./assets/MeanRiskIT.csv.gz"), DataFrame)
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    i = first(idx)
    for r in rs[idx]
        r1 = RiskTrackingRiskMeasure(; tr = tr, r = r, alg = IndependentVariableTracking())
        mr = MeanRisk(; r = r1, obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise(mr, rd)
        @test isa(res.retcode, OptimisationSuccess)
        rtol = if i in (16, 30)
            5e-5
        elseif i == 24
            5e-6
        elseif i in (27, 44)
            5e-5
        elseif i == 47
            5e-2
        elseif i == 48
            5e-3
        else
            1e-6
        end
        success = isapprox(res.w, df[!, "$i"]; rtol = rtol)
        if !success
            println("Counter: $i")
            find_tol(res.w, df[!, "$i"])
            display([res.w df[!, "$i"]])
        end
        @test success
        i += 1
    end
end

function mr_inline()
    res = optimise(MeanRisk(; wi = w0,
                            opt = JuMPOptimiser(; pe = pr,
                                                slv = Solver(; solver = Clarabel.Optimizer,
                                                             settings = ["verbose" => false,
                                                                         "max_iter" => 1])),
                            fb = InverseVolatility(; pe = pr)))
    @test isapprox(res.w, optimise(InverseVolatility(; pe = pr)).w)
end

function mr_block4()
    r = BrownianDistanceVariance()
    df = CSV.read(joinpath(@__DIR__, "./assets/MeanRiskBDV.csv.gz"), DataFrame)
    i = 1
    for obj in objs, ret in rets
        opt = JuMPOptimiser(; pe = pr2, slv = slv, ret = ret)
        mr = MeanRisk(; r = r, obj = obj, opt = opt)
        res = optimise(mr, rd2)
        @test isa(res.retcode, OptimisationSuccess)
        rtol = if i in (4, 6)
            5e-5
        elseif i == 47
            1e-3
        elseif i == 48
            5e-3
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

function mr_block5()
    r = VarianceSkewKurtosis(;
                             sk = Skewness(;
                                           settings = MaxRiskMeasureSettings(; scale = 2)),
                             kt = Kurtosis(; settings = RiskMeasureSettings(; scale = 3)))
    df = CSV.read(joinpath(@__DIR__, "./assets/MeanRiskVarianceSkewKurtosis.csv.gz"),
                  DataFrame)
    i = 1
    for obj in objs, ret in rets
        if i == 6
            i += 1
            continue
        end
        opt = JuMPOptimiser(; pe = pr2, slv = scs_slv, ret = ret)
        mr = MeanRisk(; r = r, obj = obj, opt = opt)
        res = optimise(mr, rd2)
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

function mr_block6()
    rs1 = [GenericValueatRiskRange(; loss = ValueatRisk(), gain = ValueatRisk()),
           GenericValueatRiskRange(; loss = ValueatRisk(; alg = DistributionValueatRisk()),
                                   gain = ValueatRisk(; alg = DistributionValueatRisk())),
           GenericValueatRiskRange(; loss = ConditionalValueatRisk(),
                                   gain = ConditionalValueatRisk()),
           GenericValueatRiskRange(; loss = DistributionallyRobustConditionalValueatRisk(),
                                   gain = DistributionallyRobustConditionalValueatRisk()),
           GenericValueatRiskRange(; loss = EntropicValueatRisk(),
                                   gain = EntropicValueatRisk()),
           GenericValueatRiskRange(; loss = RelativisticValueatRisk(),
                                   gain = RelativisticValueatRisk()),
           GenericValueatRiskRange(; loss = WorstRealisation(), gain = WorstRealisation()),
           GenericValueatRiskRange(; loss = PowerNormValueatRisk(),
                                   gain = PowerNormValueatRisk())]
    rs2 = [ValueatRiskRange(), ValueatRiskRange(; alg = DistributionValueatRisk()),
           ConditionalValueatRiskRange(),
           DistributionallyRobustConditionalValueatRiskRange(), EntropicValueatRiskRange(),
           RelativisticValueatRiskRange(), Range(), PowerNormValueatRiskRange()]
    for (i, (r1, r2)) in enumerate(zip(rs1, rs2))
        opt = JuMPOptimiser(; pe = pr2, slv = mip_slv)
        mr1 = MeanRisk(; r = r1, opt = opt)
        mr2 = MeanRisk(; r = r2, opt = opt)
        res1 = optimise(mr1, rd2)
        res2 = optimise(mr2, rd2)
        @test isa(res1.retcode, OptimisationSuccess)
        rtol = if i == 2
            5e-4
        else
            1e-6
        end
        success = isapprox(res1.w, res2.w; rtol = rtol)
        if !success
            println("Counter: $i")
            find_tol(res1.w, res2.w)
            display([res1.w res2.w])
        end
        @test success
    end
    return nothing
end
