@safetestset "Risk measures" begin
    using PortfolioOptimisers, Test, DataFrames, TimeSeries, CSV, Clarabel, StatsBase,
          LinearAlgebra
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
    pr = prior(HighOrderPriorEstimator(; pe = EmpiricalPrior()), rd)
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
                5e-2
            elseif Sys.isapple() && i ∈ (18, 21)
                1e-1
            elseif i ∈ (16, 18, 21)
                5e-2
            elseif i ∈ (20, 23, 24)
                0.25
            else
                1e-6
            end
            success = isapprox(r1, r2; rtol = rtol)
            if !success
                println("Iteration $i fails")
                find_tol(r1, r2)
            end
            @test success

            success = isapprox(df[i, 1], r1; rtol = rtol)
            if !success
                println("Iteration $i r1 fails")
                find_tol(r1, df[i, 1])
            end
            @test success

            success = isapprox(df[i, 2], r2; rtol = rtol)
            if !success
                println("Iteration $i r2 fails")
                find_tol(r2, df[i, 2])
            end
            @test success
        end
    end
    @testset "Expected risk" begin
        r1 = factory(NegativeSkewness(), pr)
        r2 = factory(NegativeSkewness(; alg = QuadRiskExpr()), pr)
        @test isapprox(expected_risk(r1, w, rd.X), sqrt(expected_risk(r2, w, rd.X)))
        @test isapprox(expected_risk(SquareRootKurtosis(; alg = Semi()), w, rd.X),
                       0.0002291596657404573)
        @test isapprox(expected_risk(SquareRootKurtosis(;), w, rd.X),
                       expected_risk(SquareRootKurtosis(; mu = pr.mu), w, rd.X))
        @test isapprox(expected_risk(SquareRootKurtosis(; mu = dot(w, pr.mu)), w, rd.X),
                       expected_risk(SquareRootKurtosis(;), w, rd.X))
        @test isapprox(expected_risk(SquareRootKurtosis(; w = wt), w, rd.X),
                       expected_risk(SquareRootKurtosis(;), w, rd.X))
        @test isapprox(expected_risk(LowOrderMoment(;
                                                    alg = LowOrderDeviation(;
                                                                            alg = SecondLowerMoment(;
                                                                                                    alg = SqrtRiskExpr()))),
                                     w, rd.X), 0.009123864007588172)
        @test isapprox(expected_risk(LowOrderMoment(; mu = dot(w, pr.mu),
                                                    alg = LowOrderDeviation(;
                                                                            alg = SecondLowerMoment(;
                                                                                                    alg = SqrtRiskExpr()))),
                                     w, rd.X),
                       sqrt(expected_risk(LowOrderMoment(;
                                                         alg = LowOrderDeviation(;
                                                                                 alg = SecondLowerMoment(;
                                                                                                         alg = QuadRiskExpr()))),
                                          w, rd.X)))
        @test isapprox(expected_risk(LowOrderMoment(;
                                                    alg = LowOrderDeviation(;
                                                                            alg = SecondCentralMoment(;
                                                                                                      alg = SqrtRiskExpr()))),
                                     w, rd.X), 0.012828296955991162)
        @test isapprox(expected_risk(LowOrderMoment(; mu = dot(w, pr.mu),
                                                    alg = LowOrderDeviation(;
                                                                            alg = SecondCentralMoment(;
                                                                                                      alg = SqrtRiskExpr()))),
                                     w, rd.X),
                       sqrt(expected_risk(LowOrderMoment(;
                                                         alg = LowOrderDeviation(;
                                                                                 alg = SecondCentralMoment(;
                                                                                                           alg = QuadRiskExpr()))),
                                          w, rd.X)))
        @test isapprox(expected_risk(LowOrderMoment(; alg = MeanAbsoluteDeviation()), w,
                                     rd.X), 0.009807328313217291)
        @test isapprox(expected_risk(LowOrderMoment(; alg = MeanAbsoluteDeviation()), w,
                                     rd.X),
                       expected_risk(LowOrderMoment(; w = wt,
                                                    alg = MeanAbsoluteDeviation()), w,
                                     rd.X))
        @test isapprox(expected_risk(HighOrderMoment(; alg = FourthLowerMoment()), w, rd.X),
                       5.251415240227812e-8)
        @test isapprox(expected_risk(HighOrderMoment(; mu = dot(w, pr.mu),
                                                     alg = FourthLowerMoment()), w, rd.X),
                       expected_risk(HighOrderMoment(; alg = FourthLowerMoment()), w, rd.X))

        @test isapprox(expected_risk(HighOrderMoment(; alg = FourthCentralMoment()), w,
                                     rd.X), 9.793810102468416e-8)
        @test isapprox(expected_risk(HighOrderMoment(; mu = dot(w, pr.mu),
                                                     alg = FourthCentralMoment()), w, rd.X),
                       expected_risk(HighOrderMoment(; alg = FourthCentralMoment()), w,
                                     rd.X))
        @test isapprox(expected_risk(HighOrderMoment(;
                                                     alg = HighOrderDeviation(;
                                                                              alg = ThirdLowerMoment())),
                                     w, rd.X), 2.4944191180382487)
        @test isapprox(expected_risk(HighOrderMoment(; mu = dot(w, pr.mu),
                                                     alg = HighOrderDeviation(;
                                                                              alg = ThirdLowerMoment())),
                                     w, rd.X),
                       expected_risk(HighOrderMoment(;
                                                     alg = HighOrderDeviation(;
                                                                              alg = ThirdLowerMoment())),
                                     w, rd.X))
        @test isapprox(expected_risk(HighOrderMoment(;
                                                     alg = HighOrderDeviation(;
                                                                              alg = FourthLowerMoment())),
                                     w, rd.X), 7.5781142136319515)
        @test isapprox(expected_risk(HighOrderMoment(; mu = dot(w, pr.mu),
                                                     alg = HighOrderDeviation(;
                                                                              alg = FourthLowerMoment())),
                                     w, rd.X),
                       expected_risk(HighOrderMoment(;
                                                     alg = HighOrderDeviation(;
                                                                              alg = FourthLowerMoment())),
                                     w, rd.X))
        @test isapprox(expected_risk(HighOrderMoment(;
                                                     alg = HighOrderDeviation(;
                                                                              alg = FourthCentralMoment())),
                                     w, rd.X), 3.616393337050389)
        @test isapprox(expected_risk(HighOrderMoment(; mu = dot(w, pr.mu),
                                                     alg = HighOrderDeviation(;
                                                                              alg = FourthCentralMoment())),
                                     w, rd.X),
                       expected_risk(HighOrderMoment(;
                                                     alg = HighOrderDeviation(;
                                                                              alg = FourthCentralMoment())),
                                     w, rd.X))
        @test isapprox(expected_risk(AverageDrawdown(), w, rd.X), 0.048143525862128035)
        @test isapprox(expected_risk(AverageDrawdown(), w, rd.X),
                       expected_risk(AverageDrawdown(; w = wt), w, rd.X))
        @test isapprox(expected_risk(RelativeAverageDrawdown(), w, rd.X),
                       0.05118499953858111)
        @test isapprox(expected_risk(RelativeAverageDrawdown(), w, rd.X),
                       expected_risk(RelativeAverageDrawdown(; w = wt), w, rd.X))
        @test isapprox(expected_risk(RelativeUlcerIndex(), w, rd.X), 0.06356737835751593)
        @test isapprox(expected_risk(RelativeMaximumDrawdown(), w, rd.X),
                       0.14712227931904298)
        @test isapprox(expected_risk(BrownianDistanceVariance(), w, rd.X),
                       0.0005291680154419391)
    end
end
