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
                res1 = if i ∈ (1, 4, 5, 7, 13, 14, 15, 19)
                    isapprox(w1, wt; rtol = 1e-4)
                elseif i ∈ (2, 3, 6, 8, 9, 11)
                    isapprox(w1, wt; rtol = 5e-4)
                elseif i ∈ (10, 16, 17, 20)
                    isapprox(w1, wt; rtol = 5e-5)
                elseif i ∈ (12, 18, 21)
                    isapprox(w1, wt; rtol = 5e-6)
                elseif i ∈ (22, 23)
                    isapprox(w1, wt; rtol = 1e-5)
                else
                    isapprox(w1, wt; rtol = 1e-6)
                end
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

        ###############
        ###############
        # r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        # mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        # w3 = optimise!(mr, rd).w
        # @test !isapprox(w1, w3)
        # @test !isapprox(w2, w3)
        # @test !isapprox(sum([w1 w2]; dims = 2), w3)
        # @test isapprox(w3,
        #                [0.0040045132949514655, 0.0068078718659283785, 0.005778120742868111,
        #                 0.01102643618591882, 0.005644348083705127, 0.005944275867574235,
        #                 0.010939162965132743, 0.005549819490698245, 0.01073344107815392,
        #                 0.009934567451467827, 0.034371272920275926, 0.0053822441303191014,
        #                 0.007222957855670791, 0.011615063714488472, 0.006805338428831159,
        #                 0.006896522633479259, 0.008988008734040695, 0.0036262075082924418,
        #                 0.006450178927700632, 0.313400648695782, 0.010245851606501282,
        #                 0.014757808328200814, 0.00833803184904843, 0.004092758407746584,
        #                 0.007910618841243935, 0.2891108968649568, 0.024468650605150813,
        #                 0.14775717344447362, 0.00537023572184974, 0.006826966311175611],
        #                rtol = 1e-5)

        # opt = JuMPOptimiser(; pe = pr, slv = slv, sce = MaxScalariser())
        # r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        # mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        # w4 = optimise!(mr, rd).w
        # @test isapprox(w2, w4, rtol = 1e-4)

        # opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser(; gamma = 1e-3))
        # r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        # mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        # w5 = optimise!(mr, rd).w
        # @test isapprox(w5, w3, rtol = 3e-2)

        # opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser(; gamma = 1e5))
        # r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        # mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        # w6 = optimise!(mr, rd).w
        # @test isapprox(w6, w4, rtol = 1e-3)
        #################
        #################

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
