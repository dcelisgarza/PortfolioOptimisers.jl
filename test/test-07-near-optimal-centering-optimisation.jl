@safetestset "NearOptimalCentering Optimisation" begin
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
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    @testset "Unconstrained Efficient Frontier" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier1.csv.gz"),
                      DataFrame)
        r = factory(StandardDeviation(), pr)
        res_min = optimise!(MeanRisk(; r = r, opt = opt))
        res_max = optimise!(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        rk_min = expected_risk(r, res_min.w, pr)
        rk_max = expected_risk(r, res_max.w, pr)
        res1 = optimise!(NearOptimalCentering(;
                                              r = StandardDeviation(;
                                                                    settings = RiskMeasureSettings(;
                                                                                                   ub = Frontier(;
                                                                                                                 N = 5))),
                                              obj = MaximumReturn(), opt = opt))
        res2 = optimise!(NearOptimalCentering(;
                                              r = StandardDeviation(;
                                                                    settings = RiskMeasureSettings(;
                                                                                                   ub = range(;
                                                                                                              start = rk_min,
                                                                                                              stop = rk_max,
                                                                                                              length = 5))),
                                              obj = MaximumReturn(), opt = opt))
        @test all(isapprox.(res1.w, res2.w))
        @test isapprox(Matrix(df), hcat(res1.w...))
    end
    @testset "Constrained Efficient Frontier" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier2.csv.gz"),
                      DataFrame)
        r = factory(StandardDeviation(), pr)
        res_min = optimise!(MeanRisk(; r = r, opt = opt))
        res_max = optimise!(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        rk_min = expected_risk(r, res_min.w, pr)
        rk_max = expected_risk(r, res_max.w, pr)
        res1 = optimise!(NearOptimalCentering(;
                                              r = StandardDeviation(;
                                                                    settings = RiskMeasureSettings(;
                                                                                                   ub = Frontier(;
                                                                                                                 N = 5))),
                                              obj = MaximumReturn(), opt = opt,
                                              alg = ConstrainedNearOptimalCenteringAlgorithm()))
        res2 = optimise!(NearOptimalCentering(;
                                              r = StandardDeviation(;
                                                                    settings = RiskMeasureSettings(;
                                                                                                   ub = range(;
                                                                                                              start = rk_min,
                                                                                                              stop = rk_max,
                                                                                                              length = 5))),
                                              obj = MaximumReturn(), opt = opt,
                                              alg = ConstrainedNearOptimalCenteringAlgorithm()))
        @test all(isapprox.(res1.w, res2.w))
        @test isapprox(Matrix(df), hcat(res1.w...))
    end
end