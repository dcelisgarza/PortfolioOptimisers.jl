@safetestset "Risk measures" begin
    using PortfolioOptimisers, Test, DataFrames, TimeSeries, CSV, Clarabel, StatsBase
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    w = fill(inv(size(rd.X, 2)), size(rd.X, 2))
    wt = pweights(fill(inv(size(rd.X, 1)), size(rd.X, 1)))
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
    @testset "X at Risk" begin
        rs = [(ValueatRisk(; alpha = 1e-6), ValueatRisk(; alpha = 1e-6, w = wt)),
              (ValueatRisk(;), ValueatRisk(; w = wt)),
              (ValueatRisk(; alpha = 1 - 1e-6), ValueatRisk(; alpha = 1 - 1e-6, w = wt)),
              (ValueatRiskRange(; alpha = 1e-6, beta = 1e-6),
               ValueatRiskRange(; alpha = 1e-6, beta = 1e-6, w = wt)),
              (ValueatRiskRange(;), ValueatRiskRange(; w = wt)),
              (ValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6),
               ValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, w = wt)),
              (ConditionalValueatRisk(; alpha = 1e-6),
               ConditionalValueatRisk(; alpha = 1e-6, w = wt)),
              (ConditionalValueatRisk(;), ConditionalValueatRisk(; w = wt)),
              (ConditionalValueatRisk(; alpha = 1 - 1e-6),
               ConditionalValueatRisk(; alpha = 1 - 1e-6, w = wt)),
              (ConditionalValueatRiskRange(; alpha = 1e-6, beta = 1e-6),
               ConditionalValueatRiskRange(; alpha = 1e-6, beta = 1e-6, w = wt)),
              (ConditionalValueatRiskRange(;), ConditionalValueatRiskRange(; w = wt)),
              (ConditionalValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6),
               ConditionalValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, w = wt)),
              (EntropicValueatRisk(; alpha = 1e-6, slv = slv),
               EntropicValueatRisk(; alpha = 1e-6, w = wt, slv = slv)),
              (EntropicValueatRisk(; slv = slv), EntropicValueatRisk(; w = wt, slv = slv)),
              (EntropicValueatRisk(; alpha = 1 - 1e-6, slv = slv),
               EntropicValueatRisk(; alpha = 1 - 1e-6, w = wt, slv = slv)),
              (EntropicValueatRiskRange(; alpha = 1e-6, beta = 1e-6, slv = slv),
               EntropicValueatRiskRange(; alpha = 1e-6, beta = 1e-6, w = wt, slv = slv)),
              (EntropicValueatRiskRange(; slv = slv),
               EntropicValueatRiskRange(; w = wt, slv = slv)),
              (EntropicValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, slv = slv),
               EntropicValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, w = wt,
                                        slv = slv)),
              (RelativisticValueatRisk(; alpha = 1e-6, slv = slv),
               RelativisticValueatRisk(; alpha = 1e-6, w = wt, slv = slv)),
              (RelativisticValueatRisk(; slv = slv),
               RelativisticValueatRisk(; w = wt, slv = slv)),
              (RelativisticValueatRisk(; alpha = 1 - 1e-6, slv = slv),
               RelativisticValueatRisk(; alpha = 1 - 1e-6, w = wt, slv = slv)),
              (RelativisticValueatRiskRange(; alpha = 1e-6, beta = 1e-6, slv = slv),
               RelativisticValueatRiskRange(; alpha = 1e-6, beta = 1e-6, w = wt, slv = slv)),
              (RelativisticValueatRiskRange(; slv = slv),
               RelativisticValueatRiskRange(; w = wt, slv = slv)),
              (RelativisticValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, slv = slv),
               RelativisticValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, w = wt,
                                            slv = slv))]
        df = CSV.read(joinpath(@__DIR__, "./assets/XValueatRisk.csv.gz"), DataFrame)
        for (i, r) in enumerate(rs)
            r1 = expected_risk(r[1], w, rd.X)
            r2 = expected_risk(r[2], w, rd.X)
            rtol = if i == 15
                5e-3
            elseif i ∈ (16, 18, 21)
                5e-2
            elseif i ∈ (20, 23, 24)
                0.25
            else
                5e-7
            end
            success = isapprox(r1, r2; rtol = rtol)
            if !success
                println("Iteration $i fails")
                find_tol(r1, r2)
            end
            @test success

            success = isapprox(df[i, 1], r1)
            if !success
                println("Iteration $i r1 fails")
                find_tol(r1, df[i, 1])
            end
            @test success

            success = isapprox(df[i, 2], r2)
            if !success
                println("Iteration $i r2 fails")
                find_tol(r2, df[i, 2])
            end
            @test success
        end
    end
end