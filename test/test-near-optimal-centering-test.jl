@safetestset "NearOptimalCentering Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, StableRNGs, Random, Clarabel,
          TimeSeries, JuMP, Pajarito, HiGHS
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
        for atol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                  timestamp = :timestamp)
    F = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/factor_prices.csv"));
                  timestamp = :timestamp)
    rd = prices_to_returns(X[(end - 252):end], F[(end - 252):end])
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.75))]
    mip_slv = Solver(; name = :clarabel,
                     solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                        "verbose" => false,
                                                        "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                 MOI.Silent() => true),
                                                        "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                    "verbose" => false,
                                                                                                    "max_step_fraction" => 0.75)),
                     check_sol = (; allow_local = true, allow_almost = true))
    pr = prior(HighOrderPriorEstimator(;
                                       pe = FactorPriorEstimator(;
                                                                 re = DimensionReductionRegression())),
               rd)
    rf = 4.34 / 100 / 252
    T, N = size(pr.X)
    eqw = range(; start = inv(N), stop = inv(N), length = N)
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(; rf = rf), MaximumReturn()]
    bins = [1, 5, 10, 20, nothing, 50]
    @testset "NearOptimalCentering" begin
        w_min = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(),
                                   opt = opt), rd).w
        w_max = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumReturn(),
                                   opt = opt), rd).w
        df = CSV.read(joinpath(@__DIR__, "./assets/Unconstrained-NearOptimalCentering.csv"),
                      DataFrame)
        i = 1
        for obj ∈ objs
            for bin ∈ bins
                wt = df[!, i]
                noc1 = NearOptimalCentering(; bins = bin, r = ConditionalValueatRisk(),
                                            obj = obj, opt = opt)
                w1 = optimise!(noc1, rd).w
                res1 = isapprox(w1, wt; rtol = 1e-6)
                if !res1
                    println("NOC unconstrained failed: iter: $i\nobj: $obj\nbin: $bin.")
                    find_tol(w1, wt; name1 = :w1, name2 = :wt)
                end
                @test res1

                w_opt = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = obj,
                                           opt = opt), rd).w
                noc2 = NearOptimalCentering(; w_min = w_min, w_max = w_max, w_opt = w_opt,
                                            bins = bin, r = ConditionalValueatRisk(),
                                            obj = obj, opt = opt)
                w2 = optimise!(noc2, rd).w
                res2 = isapprox(w2, w1)
                if !res2
                    println("NOC unconstrained initial values failed: iter: $i\nobj: $obj\nbin: $bin.")
                    find_tol(w2, w1; name1 = :w2, name2 = :w1)
                end
                @test res2

                noc3 = NearOptimalCentering(;
                                            alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                            bins = bin, r = ConditionalValueatRisk(),
                                            obj = obj, opt = opt)
                w3 = optimise!(noc3, rd).w
                res3 = isapprox(w3, w1)
                if !res3
                    println("NOC constrained failed: iter: $i\nobj: $obj\nbin: $bin.")
                    find_tol(w3, w1; name1 = :w3, name2 = :wt)
                end
                @test res3

                w_opt = optimise!(MeanRisk(; r = ConditionalValueatRisk(), obj = obj,
                                           opt = opt), rd).w
                noc4 = NearOptimalCentering(;
                                            alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                            w_min = w_min, w_max = w_max, w_opt = w_opt,
                                            bins = bin, r = ConditionalValueatRisk(),
                                            obj = obj, opt = opt)
                w4 = optimise!(noc4, rd).w
                w4 = optimise!(noc4, rd).w
                res2 = isapprox(w4, w2)
                if !res2
                    println("NOC constrained initial values failed: iter: $i\nobj: $obj\nbin: $bin.")
                    find_tol(w4, w2; name1 = :w4, name2 = :w3)
                end
                @test res2
                i += 1
            end
        end
        r = ConditionalValueatRisk()
        mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w
        ub = expected_risk(r, w1, rd.X)
        lb = expected_returns(ArithmeticReturn(), w1, pr)

        noc1 = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w2 = optimise!(noc1, rd).w

        noc2 = NearOptimalCentering(;
                                    r = ConditionalValueatRisk(;
                                                               settings = RiskMeasureSettings(;
                                                                                              ub = ub)),
                                    obj = MaximumReturn(), opt = opt)
        w3 = optimise!(noc2, rd).w
        @test isapprox(w2, w3, rtol = 0.5)

        opt = JuMPOptimiser(; ret = ArithmeticReturn(; lb = lb), pe = pr, slv = slv)
        noc3 = NearOptimalCentering(; r = r, obj = MinimumRisk(), opt = opt)
        w4 = optimise!(noc3, rd).w
        @test isapprox(w2, w4, rtol = 5e-4)

        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = ConditionalValueatRisk()
        mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w

        r = ConditionalDrawdownatRisk()
        mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w2 = optimise!(mr, rd).w

        # r = [ConditionalValueatRisk(), ConditionalDrawdownatRisk()]
        # mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        # w3 = optimise!(mr, rd).w
        # @test isapprox(w3,
        #                [0.010979049616329416, 0.06713140531624666, 0.010067267106669109,
        #                 0.02597915950737006, 0.04689117684508539, 0.022555523572993657,
        #                 0.1348099741676934, 0.33312912809740247, 0.13265356880003143,
        #                 0.21580374073920855], rtol = 5e-5)

        # opt = JuMPOptimiser(; pe = pr, slv = slv, sce = MaxScalariser())
        # r = [ConditionalValueatRisk(), ConditionalDrawdownatRisk()]
        # mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        # w4 = optimise!(mr, rd).w
        # @test isapprox(w4,
        #                [0.00869185402031569, 0.04623619488036266, 0.007213865019645758,
        #                 0.022358056550848663, 0.037789152899705504, 0.015153983822462731,
        #                 0.12982686774260818, 0.32681802094007506, 0.16569729359555802,
        #                 0.24021470319896132], rtol = 5e-5)

        # opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser())
        # r = [ConditionalValueatRisk(), ConditionalDrawdownatRisk()]
        # mr = NearOptimalCentering(; r = r, obj = MaximumRatio(; ohf = 1, rf = rf), opt = opt)
        # w5 = optimise!(mr, rd).w
        # @test isapprox(w5,
        #                [0.009242527297389894, 0.054648908748791185, 0.008140831454820764,
        #                 0.02311630047119653, 0.038861917666119045, 0.017052336573436123,
        #                 0.13494612519443497, 0.33875402175662866, 0.15563480930705087,
        #                 0.21960221598910773], rtol = 5e-5)

        r = ConditionalValueatRisk()
        mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w
        ub = expected_risk(r, w1, rd.X)
        lb = expected_returns(ArithmeticReturn(), w1, pr)

        noc1 = NearOptimalCentering(; alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                    r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w2 = optimise!(noc1, rd).w

        noc2 = NearOptimalCentering(; alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                    r = ConditionalValueatRisk(;
                                                               settings = RiskMeasureSettings(;
                                                                                              ub = ub)),
                                    obj = MaximumReturn(), opt = opt)
        sol = optimise!(noc2, rd)
        @test value(sol.model[:cvar_risk_1]) <= ub + sqrt(eps())

        opt = JuMPOptimiser(; ret = ArithmeticReturn(; lb = lb), pe = pr, slv = slv)
        noc3 = NearOptimalCentering(; alg = ConstrainedNearOptimalCenteringAlgorithm(),
                                    r = r, obj = MinimumRisk(), opt = opt)
        sol = optimise!(noc3, rd)
        @test value(sol.model[:ret]) >= lb - sqrt(eps())
    end
end
