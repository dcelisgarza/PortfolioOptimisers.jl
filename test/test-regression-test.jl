@safetestset "Regression tests" begin
    using PortfolioOptimisers, DataFrames, Test, CSV, Random, StableRNGs
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
    @testset "Regression tests" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        F = X[:, [3, 8, 14, 19, 10]]
        res = [ForwardRegression(), ForwardRegression(AIC()), ForwardRegression(AICC()),
               ForwardRegression(BIC()), ForwardRegression(RSquared()),
               ForwardRegression(AdjustedRSquared()), BackwardRegression(),
               BackwardRegression(AIC()), BackwardRegression(AICC()),
               BackwardRegression(BIC()), BackwardRegression(RSquared()),
               BackwardRegression(AdjustedRSquared()), PCARegression(),
               PCARegression(; target = PPCATarget())]
        res_t = CSV.read(joinpath(@__DIR__, "./assets/Regression.csv"), DataFrame)
        for i ∈ eachindex(res)
            loadings = regression(res[i], X, F)
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
    end
end
