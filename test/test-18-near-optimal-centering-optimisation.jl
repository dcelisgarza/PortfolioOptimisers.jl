@safetestset "NearOptimalCentering Optimisation" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames
    function find_tol(a1, a2; name1 = :lhs, name2 = :rhs)
        for rtol in [
            1e-10,
            5e-10,
            1e-9,
            5e-9,
            1e-8,
            5e-8,
            1e-7,
            5e-7,
            1e-6,
            5e-6,
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
            2.5e-1,
            5e-1,
            1e0,
            1.1e0,
            1.2e0,
            1.3e0,
            1.4e0,
            1.5e0,
            1.6e0,
            1.7e0,
            1.8e0,
            1.9e0,
            2e0,
            2.5e0,
        ]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol in [
            1e-10,
            5e-10,
            1e-9,
            5e-9,
            1e-8,
            5e-8,
            1e-7,
            5e-7,
            1e-6,
            5e-6,
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
            2.5e-1,
            5e-1,
            1e0,
            1.1e0,
            1.2e0,
            1.3e0,
            1.4e0,
            1.5e0,
            1.6e0,
            1.7e0,
            1.8e0,
            1.9e0,
            2e0,
            2.5e0,
        ]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    rd = prices_to_returns(
        TimeArray(
            CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
            timestamp = :Date,
        )[(end-252):end],
    )
    slv = [
        Solver(;
            name = :clarabel1,
            solver = Clarabel.Optimizer,
            check_sol = (; allow_local = true, allow_almost = true),
            settings = Dict("verbose" => false),
        ),
        Solver(;
            name = :clarabel2,
            solver = Clarabel.Optimizer,
            check_sol = (; allow_local = true, allow_almost = true),
            settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
        ),
        Solver(;
            name = :clarabel3,
            solver = Clarabel.Optimizer,
            check_sol = (; allow_local = true, allow_almost = true),
            settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
        ),
        Solver(;
            name = :clarabel4,
            solver = Clarabel.Optimizer,
            check_sol = (; allow_local = true, allow_almost = true),
            settings = Dict("verbose" => false, "max_step_fraction" => 0.85),
        ),
        Solver(;
            name = :clarabel5,
            solver = Clarabel.Optimizer,
            check_sol = (; allow_local = true, allow_almost = true),
            settings = Dict("verbose" => false, "max_step_fraction" => 0.80),
        ),
        Solver(;
            name = :clarabel6,
            solver = Clarabel.Optimizer,
            check_sol = (; allow_local = true, allow_almost = true),
            settings = Dict("verbose" => false, "max_step_fraction" => 0.75),
        ),
        Solver(;
            name = :clarabel7,
            solver = Clarabel.Optimizer,
            check_sol = (; allow_local = true, allow_almost = true),
            settings = Dict("verbose" => false, "max_step_fraction" => 0.7),
        ),
        Solver(;
            name = :clarabel8,
            solver = Clarabel.Optimizer,
            check_sol = (; allow_local = true, allow_almost = true),
            settings = Dict(
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
                "reduced_tol_infeas_rel" => 1e-4,
            ),
        ),
    ]
    pr = prior(HighOrderPriorEstimator(), rd)
    rf = 4.2 / 100 / 252
    @testset "Unconstrained Efficient Frontier" begin
        df = CSV.read(
            joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier1.csv.gz"),
            DataFrame,
        )
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = factory(StandardDeviation(), pr)
        res_min = optimise!(MeanRisk(; r = r, opt = opt))
        res_max = optimise!(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        rk_min = expected_risk(r, res_min.w, pr)
        rk_max = expected_risk(r, res_max.w, pr)
        rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
        rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
        res1 = optimise!(
            NearOptimalCentering(;
                r = StandardDeviation(;
                    settings = RiskMeasureSettings(; ub = Frontier(; N = 5)),
                ),
                obj = MaximumReturn(),
                opt = opt,
            ),
        )
        res2 = optimise!(
            NearOptimalCentering(;
                r = StandardDeviation(;
                    settings = RiskMeasureSettings(;
                        ub = range(; start = rk_min, stop = rk_max, length = 5),
                    ),
                ),
                obj = MaximumReturn(),
                opt = opt,
            ),
        )
        @test all(isapprox.(res1.w, res2.w))
        success = isapprox(Matrix(df), hcat(res1.w...); rtol = 1e-4)
        if !success
            find_tol(Matrix(df), hcat(res1.w...))
        end
        @test success

        df = CSV.read(
            joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier2.csv.gz"),
            DataFrame,
        )
        opt = JuMPOptimiser(;
            pe = pr,
            slv = slv,
            ret = ArithmeticReturn(; lb = Frontier(; N = 5)),
        )
        res3 = optimise!(NearOptimalCentering(; r = StandardDeviation(), opt = opt))
        opt = JuMPOptimiser(;
            pe = pr,
            slv = slv,
            ret = ArithmeticReturn(;
                lb = range(; start = rt_min, stop = rt_max, length = 5),
            ),
        )
        res4 = optimise!(NearOptimalCentering(; r = StandardDeviation(), opt = opt))
        @test all(isapprox.(res3.w, res4.w))
        success = isapprox(Matrix(df), hcat(res3.w...); rtol = 1e-4)
        if !success
            find_tol(Matrix(df), hcat(res3.w...))
        end
        @test success
    end
    @testset "Constrained Efficient Frontier" begin
        df = CSV.read(
            joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier3.csv.gz"),
            DataFrame,
        )
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = factory(StandardDeviation(), pr)
        res_min = optimise!(MeanRisk(; r = r, opt = opt))
        res_max = optimise!(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        rk_min = expected_risk(r, res_min.w, pr)
        rk_max = expected_risk(r, res_max.w, pr)
        rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
        rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)
        res1 = optimise!(
            NearOptimalCentering(;
                r = StandardDeviation(;
                    settings = RiskMeasureSettings(; ub = Frontier(; N = 5)),
                ),
                obj = MaximumReturn(),
                opt = opt,
                alg = ConstrainedNearOptimalCentering(),
            ),
        )
        res2 = optimise!(
            NearOptimalCentering(;
                r = StandardDeviation(;
                    settings = RiskMeasureSettings(;
                        ub = range(; start = rk_min, stop = rk_max, length = 5),
                    ),
                ),
                obj = MaximumReturn(),
                opt = opt,
                alg = ConstrainedNearOptimalCentering(),
            ),
        )
        @test all(isapprox.(res1.w, res2.w))
        success = isapprox(Matrix(df), hcat(res1.w...); rtol = 5e-5)
        if !success
            find_tol(Matrix(df), hcat(res1.w...))
        end
        @test success

        df = CSV.read(
            joinpath(@__DIR__, "./assets/NearOptimalCenteringFrontier4.csv.gz"),
            DataFrame,
        )
        opt = JuMPOptimiser(;
            pe = pr,
            slv = slv,
            ret = ArithmeticReturn(; lb = Frontier(; N = 5)),
        )
        res3 = optimise!(
            NearOptimalCentering(;
                r = StandardDeviation(),
                opt = opt,
                alg = ConstrainedNearOptimalCentering(),
            ),
        )
        opt = JuMPOptimiser(;
            pe = pr,
            slv = slv,
            ret = ArithmeticReturn(;
                lb = range(; start = rt_min, stop = rt_max, length = 5),
            ),
        )
        res4 = optimise!(
            NearOptimalCentering(;
                r = StandardDeviation(),
                opt = opt,
                alg = ConstrainedNearOptimalCentering(),
            ),
        )
        @test all(isapprox.(res3.w, res4.w))
        success = isapprox(Matrix(df), hcat(res3.w...); rtol = 1e-4)
        if !success
            find_tol(Matrix(df), hcat(res3.w...))
        end
        @test success
    end
    @testset "Pareto Surface" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        df = CSV.read(
            joinpath(@__DIR__, "./assets/NearOptimalCenteringParetoSurface1.csv.gz"),
            DataFrame,
        )
        r1 = StandardDeviation(;
            settings = RiskMeasureSettings(; scale = 2e2, ub = Frontier(; N = 3)),
        )
        r2 = ConditionalValueatRisk(;
            settings = RiskMeasureSettings(; ub = Frontier(; N = 5)),
        )
        res1 = optimise!(
            NearOptimalCentering(; r = [r1, r2], obj = MaximumReturn(), opt = opt),
        )
        success = isapprox(Matrix(df), hcat(res1.w...); rtol = 5e-5)
        if !success
            find_tol(Matrix(df), hcat(res1.w...))
        end
        @test success

        r1 = factory(StandardDeviation(; settings = RiskMeasureSettings(; scale = 2e2)), pr)
        r2 = ConditionalValueatRisk(; settings = RiskMeasureSettings(;))
        res_min = optimise!(MeanRisk(; r = [r1, r2], opt = opt))
        res_max = optimise!(MeanRisk(; r = [r1, r2], obj = MaximumReturn(), opt = opt))
        rk1_min = expected_risk(r1, res_min.w, pr)
        rk1_max = expected_risk(r1, res_max.w, pr)
        rk2_min = expected_risk(r2, res_min.w, pr)
        rk2_max = expected_risk(r2, res_max.w, pr)
        rt_min = expected_return(ArithmeticReturn(), res_min.w, pr)
        rt_max = expected_return(ArithmeticReturn(), res_max.w, pr)

        df = CSV.read(
            joinpath(@__DIR__, "./assets/NearOptimalCenteringParetoSurface2.csv.gz"),
            DataFrame,
        )
        opt = JuMPOptimiser(;
            pe = pr,
            ret = ArithmeticReturn(;
                lb = range(;
                    start = rt_min,
                    stop = rt_min + 0.0009316063452440891,
                    length = 3,
                ),
            ),
            slv = slv,
        )
        r1 = StandardDeviation(;
            settings = RiskMeasureSettings(;
                scale = 2e2,
                ub = range(;
                    start = rk1_max - 0.006363402327019335,
                    stop = rk1_max,
                    length = 3,
                ),
            ),
        )
        r2 = ConditionalValueatRisk(;
            settings = RiskMeasureSettings(;
                ub = range(;
                    start = rk2_max - 0.014823231590460057,
                    stop = rk2_max,
                    length = 3,
                ),
            ),
        )
        res2 = optimise!(
            NearOptimalCentering(; r = [r1, r2], obj = MaximumUtility(), opt = opt),
        )
        success = isapprox(Matrix(df), hcat(res2.w...); rtol = 5e-5)
        if !success
            find_tol(Matrix(df), hcat(res2.w...))
        end
        @test success

        df = CSV.read(
            joinpath(@__DIR__, "./assets/NearOptimalCenteringParetoSurface3.csv.gz"),
            DataFrame,
        )
        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = MaxScalariser())
        r1 = StandardDeviation(;
            settings = RiskMeasureSettings(; scale = 2e2, ub = Frontier(; N = 3)),
        )
        r2 = ConditionalValueatRisk(;
            settings = RiskMeasureSettings(; ub = Frontier(; N = 5)),
        )
        res3 = optimise!(
            NearOptimalCentering(;
                r = [r1, r2],
                obj = MaximumReturn(),
                opt = opt,
                alg = ConstrainedNearOptimalCentering(),
            ),
        )
        success = isapprox(Matrix(df), hcat(res3.w...); rtol = 5e-5)
        if !success
            find_tol(Matrix(df), hcat(res3.w...))
        end
        @test success
    end
    @testset "Scalarisers" begin
        r1 = NegativeSkewness(;
            alg = SqrtRiskExpr(),
            settings = RiskMeasureSettings(; scale = 50),
        )
        r2 = ConditionalValueatRisk(; settings = RiskMeasureSettings(;))

        res_m1 = optimise!(
            NearOptimalCentering(;
                r = r1,
                obj = MaximumRatio(; rf = rf),
                opt = JuMPOptimiser(; pe = pr, slv = slv),
            ),
        )
        res_m2 = optimise!(
            NearOptimalCentering(;
                r = r2,
                obj = MaximumRatio(; rf = rf),
                opt = JuMPOptimiser(; pe = pr, slv = slv),
            ),
        )

        res1 = optimise!(
            NearOptimalCentering(;
                r = [r1, r2],
                obj = MaximumRatio(; rf = rf),
                opt = JuMPOptimiser(; pe = pr, slv = slv),
            ),
        )
        res2 = optimise!(
            NearOptimalCentering(;
                r = [r1, r2],
                obj = MaximumRatio(; rf = rf),
                opt = JuMPOptimiser(; pe = pr, slv = slv, sce = MaxScalariser()),
            ),
        )
        res3 = optimise!(
            NearOptimalCentering(;
                r = [r1, r2],
                obj = MaximumRatio(; rf = rf),
                opt = JuMPOptimiser(;
                    pe = pr,
                    slv = slv,
                    sce = LogSumExpScalariser(; gamma = 8.5e-4),
                ),
            ),
        )
        res4 = optimise!(
            NearOptimalCentering(;
                r = [r1, r2],
                obj = MaximumRatio(; rf = rf),
                opt = JuMPOptimiser(;
                    pe = pr,
                    slv = slv,
                    sce = LogSumExpScalariser(; gamma = 500),
                ),
            ),
        )
        @test isapprox(res2.w, res_m1.w, rtol = 1e-4)
        @test isapprox(res1.w, res3.w, rtol = 1e-3)
        @test isapprox(res4.w, res_m1.w, rtol = 1e-4)
    end
end
