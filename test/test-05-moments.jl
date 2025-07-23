@safetestset "Moments" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, CovarianceEstimation,
          StableRNGs, StatsBase
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
    rng = StableRNG(123456789)
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    ew = eweights(1:252, inv(252); scale = true)
    fw = fweights(rand(rng, 252))
    rf = 4.34 / 100 / 252
    @testset "Expected Returns" begin
        mes = [ShrunkExpectedReturns(; alg = JamesStein()),
               ShrunkExpectedReturns(; alg = JamesStein(; target = VolatilityWeighted())),
               ShrunkExpectedReturns(; alg = JamesStein(; target = MeanSquareError())),
               ShrunkExpectedReturns(; alg = JamesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BayesStein()),
               ShrunkExpectedReturns(; alg = BayesStein(; target = VolatilityWeighted())),
               ShrunkExpectedReturns(; alg = BayesStein(; target = MeanSquareError())),
               ShrunkExpectedReturns(; alg = BayesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BodnarOkhrinParolya()),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = VolatilityWeighted())),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = MeanSquareError())),
               ShrunkExpectedReturns(; alg = BodnarOkhrinParolya(),
                                     me = SimpleExpectedReturns(; w = ew)),
               EquilibriumExpectedReturns(), ExcessExpectedReturns(; rf = rf)]
        df = CSV.read(joinpath(@__DIR__, "./assets/expected_returns.csv.gz"), DataFrame)
        for (i, me) in pairs(mes)
            mu = mean(me, rd.X)
            success = isapprox(vec(mu), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(mu, df[!, i])
            end
            @test success
        end
    end
    @testset "Covariance Estimators" begin
        ces = [Covariance(; alg = Full()),
               Covariance(; alg = Full(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralWeightedCovariance(;
                                                         ce = SimpleCovariance(;
                                                                               corrected = false),
                                                         w = ew)),
               Covariance(; alg = Full(),
                          ce = GeneralWeightedCovariance(;
                                                         ce = AnalyticalNonlinearShrinkage())),
               Covariance(; alg = Semi()),
               Covariance(; alg = Semi(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralWeightedCovariance(;
                                                         ce = SimpleCovariance(;
                                                                               corrected = false),
                                                         w = ew)),
               Covariance(; alg = Semi(), me = SimpleExpectedReturns(; w = fw),
                          ce = GeneralWeightedCovariance(;
                                                         ce = AnalyticalNonlinearShrinkage(),
                                                         w = fw)), SpearmanCovariance(),
               KendallCovariance(), MutualInfoCovariance(),
               MutualInfoCovariance(; bins = Knuth()),
               MutualInfoCovariance(; bins = FreedmanDiaconis()),
               MutualInfoCovariance(; bins = Scott()), MutualInfoCovariance(; bins = 5),
               MutualInfoCovariance(;
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               DistanceCovariance(), DistanceCovariance(; w = ew), LTDCovariance(),
               GerberCovariance(; alg = Gerber0()),
               GerberCovariance(; alg = Gerber0(),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = NormalisedGerber0()),
               GerberCovariance(;
                                alg = NormalisedGerber0(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew)),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = Gerber1()),
               GerberCovariance(; alg = NormalisedGerber1()),
               GerberCovariance(; alg = Gerber2()),
               GerberCovariance(; alg = NormalisedGerber2()),
               SmythBrobyCovariance(; alg = SmythBroby0()),
               SmythBrobyCovariance(; alg = SmythBroby0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBroby1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby1()),
               SmythBrobyCovariance(; alg = SmythBroby2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby2()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber2())]
        df = CSV.read(joinpath(@__DIR__, "./assets/covariance.csv.gz"), DataFrame)
        for (i, ce) in pairs(ces)
            cei = PortfolioOptimisersCovariance(; ce = ce)
            sigma = cov(cei, rd.X)
            rho = cor(cei, rd.X)
            @test isapprox(cov2cor(sigma), rho)
            success = isapprox(vec(sigma), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(sigma), df[!, i])
            end
            @test success
        end
    end
    res = [StepwiseRegression(; alg = Forward()),
           StepwiseRegression(; alg = Forward(), crit = AIC()),
           StepwiseRegression(; alg = Forward(), crit = AICC()),
           StepwiseRegression(; alg = Forward(), crit = BIC()),
           StepwiseRegression(; alg = Forward(), crit = RSquared()),
           StepwiseRegression(; alg = Forward(), crit = AdjustedRSquared()),
           StepwiseRegression(; alg = Backward()),
           StepwiseRegression(; alg = Backward(), crit = AIC()),
           StepwiseRegression(; alg = Backward(), crit = AICC()),
           StepwiseRegression(; alg = Backward(), crit = BIC()),
           StepwiseRegression(; alg = Backward(), crit = RSquared()),
           StepwiseRegression(; alg = Backward(), crit = AdjustedRSquared()),
           DimensionReductionRegression(), DimensionReductionRegression(; drtgt = PPCA())]
    df = CSV.read(joinpath(@__DIR__, "./assets/Regression.csv.gz"), DataFrame)
    for (i, re) in pairs(res)
        loadings = regression(re, rd.X, rd.F)
        lt = [loadings.b; vec(loadings.M)]
        success = isapprox(lt, df[!, i])
        if !success
            println("Counter: $i")
            find_tol(lt, df[!, i])
        end
        @test success
    end
end
