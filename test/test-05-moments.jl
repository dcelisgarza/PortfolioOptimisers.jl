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
                find_tol(res.w, df[!, i])
            end
            @test success
        end
    end
    @testset "Covariance Estimators" begin end
end
