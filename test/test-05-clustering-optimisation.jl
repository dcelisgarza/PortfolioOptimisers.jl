@safetestset "Clustering optimisation" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames
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
    opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv)
    w0 = range(; start = inv(size(pr.X, 2)), stop = inv(size(pr.X, 2)),
               length = size(pr.X, 2))

    rs = [EqualRiskMeasure(), Variance(), StandardDeviation(), UncertaintySetVariance(),
          LowOrderMoment(), HighOrderMoment(), SquareRootKurtosis(), NegativeSkewness(),
          ValueatRisk(), ValueatRiskRange(), ConditionalValueatRisk(),
          DistributionallyRobustConditionalValueatRisk(), ConditionalValueatRiskRange(),
          DistributionallyRobustConditionalValueatRiskRange(), EntropicValueatRisk(),
          EntropicValueatRiskRange(), EntropicDrawdownatRisk(),
          RelativeEntropicDrawdownatRisk(), RelativisticValueatRisk(),
          RelativisticValueatRiskRange(), RelativisticDrawdownatRisk(),
          RelativeRelativisticDrawdownatRisk(), AverageDrawdown(),
          RelativeAverageDrawdown(), TurnoverRiskMeasure(; w = w0),
          TrackingRiskMeasure(; tracking = WeightsTracking(; w = w0)),
          RiskTrackingRiskMeasure(; r = StandardDeviation(),
                                  tracking = WeightsTracking(; w = w0)),
          RiskRatioRiskMeasure()]
    @testset "HierarchicalRiskParity" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/HierarchicalRiskParity1.csv.gz"),
                      DataFrame)
        for (i, r) in pairs(rs)
            res = optimise!(HierarchicalRiskParity(; r = r, opt = opt))
            @test isa(res.retcode, OptimisationSuccess)
            success = isapprox(res.w, df[!, i])
            if !success
                find_tol(res.w, df[!, i]; name1 = :lhs, name2 = :rhs)
            end
            @test success
        end
    end
end
