#=
@safetestset "Regression tests" begin
    using PortfolioOptimisers, DataFrames, Test, CSV, Random, Logging, TimeSeries
    import PortfolioOptimisers: regression_view
    Logging.disable_logging(Logging.Warn)
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
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
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                  timestamp = :timestamp)
    F = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/factor_prices.csv"));
                  timestamp = :timestamp)
    rd = prices_to_returns(X, F)
    (; X, F) = rd
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
    res_t = CSV.read(joinpath(@__DIR__, "./assets/Regression.csv"), DataFrame)
    for (i, re) in enumerate(res)
        loadings = regression(re, X, F)
        lt = [loadings.b; vec(loadings.M)]
        result = isapprox(lt, res_t[!, i])
        if !result
            if i == length(res)
                continue
            end
            println("Test $i fails.\n$(res[i])")
            find_tol(lt, res_t[!, i]; name1 = :loadings, name2 = :loadings_t)
        end
        @test result
    end
    @test res[1] === regression_view(res[1])
    loadings = regression(res[1], X, F)
    lv = regression_view(loadings, [2, 5, 17, 4])
    @test lv.b == view(loadings.b, [2, 5, 17, 4])
    @test lv.M == view(loadings.M, [2, 5, 17, 4], :)
end
=#
