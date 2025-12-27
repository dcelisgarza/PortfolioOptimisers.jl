@safetestset "Clustering optimisation" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames, FLoops
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
    opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv)
    rs = [EqualRiskMeasure(), Variance(), StandardDeviation(), UncertaintySetVariance(),
          LowOrderMoment(), HighOrderMoment(), Kurtosis(), NegativeSkewness(),
          ValueatRisk(), ValueatRiskRange(), ConditionalValueatRisk(),
          DistributionallyRobustConditionalValueatRisk(), ConditionalValueatRiskRange(),
          DistributionallyRobustConditionalValueatRiskRange(), EntropicValueatRisk(),
          EntropicValueatRiskRange(), EntropicDrawdownatRisk(),
          RelativeEntropicDrawdownatRisk(), RelativisticValueatRisk(),
          RelativisticValueatRiskRange(), RelativisticDrawdownatRisk(),
          RelativeRelativisticDrawdownatRisk(), AverageDrawdown(),
          RelativeAverageDrawdown(), TurnoverRiskMeasure(; w = w0),
          TrackingRiskMeasure(; tr = WeightsTracking(; w = w0)),
          RiskTrackingRiskMeasure(; r = StandardDeviation(),
                                  tr = WeightsTracking(; w = w0)),
          TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = NOCTracking()),
          RiskTrackingRiskMeasure(; r = StandardDeviation(), tr = WeightsTracking(; w = w0),
                                  alg = DependentVariableTracking()),
          RiskRatioRiskMeasure(), MedianAbsoluteDeviation()]
    @testset "HierarchicalRiskParity" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/HierarchicalRiskParity1.csv.gz"),
                      DataFrame)
        for (i, r) in pairs(rs)
            res = optimise(HierarchicalRiskParity(; r = r, opt = opt))
            @test isa(res.retcode, OptimisationSuccess)
            success = isapprox(res.w, df[!, i]; rtol = 5e-7)
            if !success
                println("Counter: $i")
                find_tol(res.w, df[!, i])
            end
            @test success
        end
    end
    df = CSV.read(joinpath(@__DIR__, "./assets/HierarchicalRiskParity1.csv.gz"), DataFrame)

    @testset "HierarchicalRiskParity vector rm" begin
        sces = [SumScalariser(), MaxScalariser(), LogSumExpScalariser(; gamma = 1.2e2),
                LogSumExpScalariser(; gamma = 1e6)]
        df = CSV.read(joinpath(@__DIR__, "./assets/HierarchicalRiskParity2.csv.gz"),
                      DataFrame)
        for (i, sca) in pairs(sces)
            res = optimise(HierarchicalRiskParity(;
                                                  r = [ConditionalValueatRisk(),
                                                       Variance(;
                                                                settings = RiskMeasureSettings(;
                                                                                               scale = 2e2))],
                                                  opt = opt, sca = sca))
            @test isa(res.retcode, OptimisationSuccess)
            success = isapprox(res.w, df[!, i])
            if !success
                println("Counter: $i")
                find_tol(res.w, df[!, i])
            end
            @test success
        end
    end
    @testset "HierarchicalEqualRiskContribution" begin
        w1 = [0.00908374873880388, 0.0030813159109630405, 0.010918199959460204,
              0.005595309855503248, 0.022247736637590203, 0.009462220791969355,
              0.01173118247048863, 0.15335141162301058, 0.012897796620250668,
              0.1202420807244618, 0.06262619007540955, 0.11709370371475523,
              0.009275523953337955, 0.12297190754199298, 0.06382989742924242,
              0.09641530015038421, 0.006088453496011643, 0.0783381784854751,
              0.06517803092186192, 0.01957181089902754]
        res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ri = Variance(),
                                                         ro = Variance(; sigma = pr.sigma),
                                                         opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ri = [Variance()], opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ri = [Variance()],
                                                         ro = [Variance()], opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ri = [Variance()],
                                                         ro = Variance(), opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ri = Variance(),
                                                         ro = [Variance()], opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)

        res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                         opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ri = Variance(),
                                                         ro = Variance(; sigma = pr.sigma),
                                                         opt = opt,
                                                         ex = FLoops.SequentialEx()))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                         ri = [Variance()], opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                         ri = [Variance()],
                                                         ro = [Variance()], opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                         ri = [Variance()], ro = Variance(),
                                                         opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
        res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                         ri = Variance(), ro = [Variance()],
                                                         opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, w1)
    end
    @testset "HierarchicalEqualRiskContribution scalarisers" begin
        sces = [SumScalariser(), MaxScalariser(), LogSumExpScalariser(; gamma = 1e-3),
                LogSumExpScalariser(; gamma = 1e2)]
        df = CSV.read(joinpath(@__DIR__,
                               "./assets/HierarchicalEqualRiskContribution2.csv.gz"),
                      DataFrame)
        for (i, sca) in pairs(sces)
            res = optimise(HierarchicalEqualRiskContribution(;
                                                             ri = [ConditionalValueatRisk(),
                                                                   Variance(;
                                                                            settings = RiskMeasureSettings(;
                                                                                                           scale = 1e1))],
                                                             opt = opt, scai = sca))
            @test isa(res.retcode, OptimisationSuccess)
            success = isapprox(res.w, df[!, i])
            if !success
                println("Counter parallel: $i")
                find_tol(res.w, df[!, i])
            end
            @test success
            res = optimise(HierarchicalEqualRiskContribution(;
                                                             ri = [ConditionalValueatRisk(),
                                                                   Variance(;
                                                                            settings = RiskMeasureSettings(;
                                                                                                           scale = 1e1))],
                                                             opt = opt, scai = sca,
                                                             ex = FLoops.SequentialEx()))
            @test isa(res.retcode, OptimisationSuccess)
            success = isapprox(res.w, df[!, i])
            if !success
                println("Counter serial: $i")
                find_tol(res.w, df[!, i])
            end
            @test success
        end
    end
    @testset "Weight bounds" begin
        w1 = [0.00908374873880388, 0.0030813159109630405, 0.010918199959460204,
              0.005595309855503248, 0.022247736637590203, 0.009462220791969355,
              0.01173118247048863, 0.15335141162301058, 0.012897796620250668,
              0.1202420807244618, 0.06262619007540955, 0.11709370371475523,
              0.009275523953337955, 0.12297190754199298, 0.06382989742924242,
              0.09641530015038421, 0.006088453496011643, 0.0783381784854751,
              0.06517803092186192, 0.01957181089902754]
        sets = AssetSets(; dict = Dict("nx" => rd.nx, "group1" => ["AAPL", "MSFT"]))
        eqn = WeightBoundsEstimator(; lb = ["JNJ" => 0.03, "group1" => 0.02],
                                    ub = Dict("PEP" => 0.08, "JNJ" => 0.03))
        opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn)
        res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w[findfirst(x -> x == "JNJ", sets.dict[sets.key])], 0.03)
        @test all(abs.(res.w[[findfirst(x -> x == i, sets.dict[sets.key])
                              for i in sets.dict["group1"]]] .- 0.02) .<= 1e-10)
        @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.key])] - 0.08) < 5e-10

        opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn,
                                    cwf = JuMPWeightFinaliser(;
                                                              alg = RelativeErrorWeightFinaliser(),
                                                              slv = slv))
        res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w[findfirst(x -> x == "JNJ", sets.dict[sets.key])], 0.03)
        @test all(abs.(res.w[[findfirst(x -> x == i, sets.dict[sets.key])
                              for i in sets.dict["group1"]]] .- 0.02) .<= 1e-10)
        @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.key])] - 0.08) < 5e-10

        opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn,
                                    cwf = JuMPWeightFinaliser(;
                                                              alg = SquareRelativeErrorWeightFinaliser(),
                                                              slv = slv))
        res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w[findfirst(x -> x == "JNJ", sets.dict[sets.key])], 0.03)
        @test all(abs.(res.w[[findfirst(x -> x == i, sets.dict[sets.key])
                              for i in sets.dict["group1"]]] .- 0.02) .<= 1e-10)
        @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.key])] - 0.08) < 5e-10

        opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn,
                                    cwf = JuMPWeightFinaliser(;
                                                              alg = AbsoluteErrorWeightFinaliser(),
                                                              slv = slv))
        res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w[findfirst(x -> x == "JNJ", sets.dict[sets.key])], 0.03)
        @test all(res.w[[findfirst(x -> x == i, sets.dict[sets.key])
                         for i in sets.dict["group1"]]] .>= 0.02)
        @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.key])] - 0.08) < 5e-10

        opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn,
                                    cwf = JuMPWeightFinaliser(;
                                                              alg = SquareAbsoluteErrorWeightFinaliser(),
                                                              slv = slv))
        res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w[findfirst(x -> x == "JNJ", sets.dict[sets.key])], 0.03)
        @test all(res.w[[findfirst(x -> x == i, sets.dict[sets.key])
                         for i in sets.dict["group1"]]] .>= 0.02)
        @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.key])] - 0.08) < 5e-10
    end
    @testset "SchurComplementHierarchicalRiskParity" begin
        r = factory(Variance(), pr)
        hrp = HierarchicalRiskParity(; r = r, opt = opt)
        res0 = optimise(hrp)
        rk0 = expected_risk(r, res0.w, pr)

        sch = SchurComplementHierarchicalRiskParity(;
                                                    params = SchurComplementParams(;
                                                                                   gamma = 0,
                                                                                   alg = NonMonotonicSchurComplement()),
                                                    opt = opt)
        res = optimise(sch)
        rk = expected_risk(r, res.w, pr)
        @test isapprox(res0.w, res.w)
        @test isapprox(rk0, rk)

        sch = SchurComplementHierarchicalRiskParity(;
                                                    params = SchurComplementParams(;
                                                                                   gamma = 1,
                                                                                   alg = MonotonicSchurComplement()),
                                                    opt = opt)
        res = optimise(sch)
        rk = expected_risk(r, res.w, pr)
        @test rk <= rk0

        sch = SchurComplementHierarchicalRiskParity(;
                                                    params = SchurComplementParams(;
                                                                                   gamma = 0.675,
                                                                                   alg = NonMonotonicSchurComplement()),
                                                    opt = opt)
        res = optimise(sch)
        rk = expected_risk(r, res.w, pr)
        @test rk >= rk0

        rk0 = -Inf
        for gamma in range(; start = 0.0, stop = 0.20, length = 10)
            sch = SchurComplementHierarchicalRiskParity(;
                                                        params = SchurComplementParams(;
                                                                                       gamma = 0.675,
                                                                                       alg = NonMonotonicSchurComplement()),
                                                        opt = opt)
            res = optimise(sch)
            rk = expected_risk(r, res.w, pr)
            @test rk >= rk0
            rk0 = rk
        end

        sch = SchurComplementHierarchicalRiskParity(;
                                                    params = SchurComplementParams(;
                                                                                   gamma = 0.05,
                                                                                   alg = NonMonotonicSchurComplement()),
                                                    opt = opt)
        res0 = optimise(sch)
        sch = SchurComplementHierarchicalRiskParity(;
                                                    params = SchurComplementParams(;
                                                                                   r = StandardDeviation(),
                                                                                   gamma = 0.1,
                                                                                   alg = NonMonotonicSchurComplement()),
                                                    opt = opt)
        res1 = optimise(sch)

        sch = SchurComplementHierarchicalRiskParity(;
                                                    params = [SchurComplementParams(;
                                                                                    gamma = 0.05,
                                                                                    alg = NonMonotonicSchurComplement()),
                                                              SchurComplementParams(;
                                                                                    r = StandardDeviation(;
                                                                                                          settings = RiskMeasureSettings(;
                                                                                                                                         scale = 2)),
                                                                                    gamma = 0.1,
                                                                                    alg = NonMonotonicSchurComplement())],
                                                    opt = opt)
        res2 = optimise(sch)

        w2 = res0.w + 2 * res1.w
        w2 /= sum(w2)
        @test isapprox(res2.w, w2)
    end
end
