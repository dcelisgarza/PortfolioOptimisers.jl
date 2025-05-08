@safetestset "NearOptimalCentering Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, StableRNGs, Random, Clarabel,
          StatsBase, LinearAlgebra, Pajarito, HiGHS, JuMP
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
    end
    rf = 4.34 / 100 / 252
    rng = StableRNG(987456321)
    X = randn(rng, 200, 10)
    rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
    @testset "Unconstrainted NearOptimalCentering" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        pr = prior(EmpiricalPriorEstimator(), rd)
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(), MaximumReturn()]
        bins = [1, 5, 10, 20, nothing, 50]
        w_min = optimise!(MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                            obj = MinimumRisk(), opt = opt), rd).w
        w_max = optimise!(MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                            obj = MaximumReturn(), opt = opt), rd).w
        df = CSV.read(joinpath(@__DIR__, "./assets/Unconstrained-NearOptimalCentering.csv"),
                      DataFrame)
        i = 1
        for obj ∈ objs
            for bin ∈ bins
                wt = df[!, i]
                noc1 = NearOptimalCenteringEstimator(; bins = bin,
                                                     r = ConditionalValueatRisk(),
                                                     obj = obj, opt = opt)
                w1 = optimise!(noc1, rd).w
                res1 = isapprox(w1, wt; rtol = 5e-5)
                if !res1
                    println("Iteration $i, compute everything in NOC failed.")
                    find_tol(w1, wt; name1 = :w1, name2 = :wt)
                end
                @test res1

                w_opt = optimise!(MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                                    obj = obj, opt = opt), rd).w
                noc2 = NearOptimalCenteringEstimator(; w_min = w_min, w_max = w_max,
                                                     w_opt = w_opt, bins = bin,
                                                     r = ConditionalValueatRisk(),
                                                     obj = obj, opt = opt)
                w2 = optimise!(noc2, rd).w
                res2 = isapprox(w2, w1)
                if !res1
                    println("Iteration $i, initial vectors provided NOC failed.")
                    find_tol(w2, w1; name1 = :w2, name2 = :w1)
                end
                @test res2
                i += 1
            end
        end
        r = ConditionalValueatRisk()
        mr = MeanRiskEstimator(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w
        ub = expected_risk(r, w1, rd.X)
        lb = expected_returns(ArithmeticReturn(), w1, pr)

        noc1 = NearOptimalCenteringEstimator(; r = r, obj = MaximumRatio(; rf = rf),
                                             opt = opt)
        w2 = optimise!(noc1, rd).w

        noc2 = NearOptimalCenteringEstimator(;
                                             r = ConditionalValueatRisk(;
                                                                        settings = RiskMeasureSettings(;
                                                                                                       ub = ub)),
                                             obj = MaximumReturn(), opt = opt)
        w3 = optimise!(noc2, rd).w
        @test isapprox(w2, w3, rtol = 5e-5)

        opt = JuMPOptimiser(; ret = ArithmeticReturn(; lb = lb), pe = pr, slv = slv)
        noc3 = NearOptimalCenteringEstimator(; r = r, obj = MinimumRisk(), opt = opt)
        w4 = optimise!(noc3, rd).w
        @test isapprox(w2, w4, rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = ConditionalValueatRisk()
        mr = NearOptimalCenteringEstimator(; r = r, obj = MaximumRatio(; rf = rf),
                                           opt = opt)
        w1 = optimise!(mr, rd).w

        r = ConditionalDrawdownatRisk()
        mr = NearOptimalCenteringEstimator(; r = r, obj = MaximumRatio(; rf = rf),
                                           opt = opt)
        w2 = optimise!(mr, rd).w

        r = [ConditionalValueatRisk(), ConditionalDrawdownatRisk()]
        mr = NearOptimalCenteringEstimator(; r = r, obj = MaximumRatio(; rf = rf),
                                           opt = opt)
        w3 = optimise!(mr, rd).w
        @test isapprox(w3,
                       [0.010979049616329416, 0.06713140531624666, 0.010067267106669109,
                        0.02597915950737006, 0.04689117684508539, 0.022555523572993657,
                        0.1348099741676934, 0.33312912809740247, 0.13265356880003143,
                        0.21580374073920855], rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = MaxScalariser())
        r = [ConditionalValueatRisk(), ConditionalDrawdownatRisk()]
        mr = NearOptimalCenteringEstimator(; r = r, obj = MaximumRatio(; rf = rf),
                                           opt = opt)
        w4 = optimise!(mr, rd).w
        @test isapprox(w4,
                       [0.00869185402031569, 0.04623619488036266, 0.007213865019645758,
                        0.022358056550848663, 0.037789152899705504, 0.015153983822462731,
                        0.12982686774260818, 0.32681802094007506, 0.16569729359555802,
                        0.24021470319896132], rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser())
        r = [ConditionalValueatRisk(), ConditionalDrawdownatRisk()]
        mr = NearOptimalCenteringEstimator(; r = r, obj = MaximumRatio(; ohf = 1, rf = rf),
                                           opt = opt)
        w5 = optimise!(mr, rd).w
        @test isapprox(w5,
                       [0.009242527297389894, 0.054648908748791185, 0.008140831454820764,
                        0.02311630047119653, 0.038861917666119045, 0.017052336573436123,
                        0.13494612519443497, 0.33875402175662866, 0.15563480930705087,
                        0.21960221598910773], rtol = 5e-5)
    end
    @testset "Constrainted NearOptimalCentering" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        pr = prior(EmpiricalPriorEstimator(), rd)
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(), MaximumReturn()]
        bins = [1, 5, 10, 20, nothing, 50]
        w_min = optimise!(MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                            obj = MinimumRisk(), opt = opt), rd).w
        w_max = optimise!(MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                            obj = MaximumReturn(), opt = opt), rd).w
        df = CSV.read(joinpath(@__DIR__, "./assets/Unconstrained-NearOptimalCentering.csv"),
                      DataFrame)
        i = 1
        for obj ∈ objs
            for bin ∈ bins
                wt = df[!, i]
                noc1 = NearOptimalCenteringEstimator(;
                                                     alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                                     bins = bin,
                                                     r = ConditionalValueatRisk(),
                                                     obj = obj, opt = opt)
                w1 = optimise!(noc1, rd).w
                res1 = isapprox(w1, wt; rtol = 5e-5)
                if !res1
                    println("Iteration $i, compute everything in NOC failed.")
                    find_tol(w1, wt; name1 = :w1, name2 = :wt)
                end
                @test res1

                w_opt = optimise!(MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                                    obj = obj, opt = opt), rd).w
                noc2 = NearOptimalCenteringEstimator(;
                                                     alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                                     w_min = w_min, w_max = w_max,
                                                     w_opt = w_opt, bins = bin,
                                                     r = ConditionalValueatRisk(),
                                                     obj = obj, opt = opt)
                w2 = optimise!(noc2, rd).w
                res2 = isapprox(w2, w1)
                if !res1
                    println("Iteration $i, initial vectors provided NOC failed.")
                    find_tol(w2, w1; name1 = :w2, name2 = :w1)
                end
                @test res2
                i += 1
            end
        end
        r = ConditionalValueatRisk()
        mr = MeanRiskEstimator(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w
        ub = expected_risk(r, w1, rd.X)
        lb = expected_returns(ArithmeticReturn(), w1, pr)

        noc1 = NearOptimalCenteringEstimator(;
                                             alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                             r = r, obj = MaximumRatio(; rf = rf),
                                             opt = opt)
        w2 = optimise!(noc1, rd).w

        noc2 = NearOptimalCenteringEstimator(;
                                             alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                             r = ConditionalValueatRisk(;
                                                                        settings = RiskMeasureSettings(;
                                                                                                       ub = ub)),
                                             obj = MaximumReturn(), opt = opt)
        w3 = optimise!(noc2, rd).w
        @test expected_risk(r, w3, rd.X) <= ub + sqrt(eps())

        opt = JuMPOptimiser(; ret = ArithmeticReturn(; lb = lb), pe = pr, slv = slv)
        noc3 = NearOptimalCenteringEstimator(;
                                             alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                             r = r, obj = MinimumRisk(), opt = opt)
        w4 = optimise!(noc3, rd).w
        @test expected_returns(ArithmeticReturn(), w4, pr) >= lb - sqrt(eps())
    end
end
