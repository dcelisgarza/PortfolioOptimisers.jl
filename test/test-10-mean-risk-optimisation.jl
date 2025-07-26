@safetestset "Mean Risk Optimisation" begin
    using Test, PortfolioOptimisers, DataFrames, CSV, TimeSeries, Clarabel, StatsBase,
          StableRNGs
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
                  settings = Dict("verbose" => false)),
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.95)),
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
    pr = prior(HighOrderPriorEstimator(), rd)
    clr = clusterise(ClusteringEstimator(), pr)
    w0 = range(; start = inv(size(pr.X, 2)), stop = inv(size(pr.X, 2)),
               length = size(pr.X, 2))
    wp = pweights(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                        length = size(pr.X, 1)))
    ucs1 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                   rng = StableRNG(987654321),
                                                   alg = BoxUncertaintySetAlgorithm()),
                     rd.X)
    ucs2 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                   rng = StableRNG(987654321),
                                                   alg = EllipseUncertaintySetAlgorithm()),
                     rd.X)
    rf = 4.2 / 100 / 252
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
          ConditionalValueatRisk(), ConditionalValueatRiskRange(), EntropicValueatRisk(),
          EntropicValueatRiskRange(), RelativisticValueatRisk(),
          RelativisticValueatRiskRange(), MaximumDrawdown(), AverageDrawdown(),
          UlcerIndex(), ConditionalDrawdownatRisk(), EntropicDrawdownatRisk(),
          RelativisticDrawdownatRisk(), SquareRootKurtosis(; N = 2), SquareRootKurtosis(),
          OrderedWeightsArray(; alg = ExactOrderedWeightsArray()), OrderedWeightsArray(),
          OrderedWeightsArrayRange(; alg = ExactOrderedWeightsArray()),
          OrderedWeightsArrayRange(), NegativeSkewness(),
          NegativeSkewness(; alg = QuadRiskExpr())]

    df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk1.csv.gz"), DataFrame)
    i = 1
    for obj in objs, ret in rets, r in rs
        opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret)
        mr = MeanRisk(; r = r, obj = obj, opt = opt)
        res = optimise!(mr, rd)
        @test isa(res.retcode, OptimisationSuccess)
        rtol = if i ∈ (27, 59, 163, 189)
            5e-4
        elseif i ∈ (123, 187, 190) || Sys.isapple() && i == 60
            5e-3
        elseif i == 126
            1e-3
        else
            1e-4
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
