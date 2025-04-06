#=
@safetestset "Philogeny tests" begin
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
    @testset "Philogeny matrix" begin
        using PortfolioOptimisers, StableRNGs, Random, Test, Clustering, CSV, DataFrames
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)

        df = CSV.read(joinpath(@__DIR__, "./assets/Philogeny_Matrix_1.csv"), DataFrame)
        for i ∈ 1:ncol(df)
            A = philogeny_matrix(NetworkEstimator(; n = i), X)
            res = isapprox(vec(A), df[!, i])
            if !res
                println("Iteration $i failed on DBHT_MaximumDistanceSimilarity.")
                find_tol(vec(A), df[!, i]; name1 = :A, name2 = :df)
            end
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/Philogeny_Matrix_2.csv"), DataFrame)
        for i ∈ 1:ncol(df)
            A = philogeny_matrix(NetworkEstimator(; n = i,
                                                  alg = DBHT_MaximumDistanceSimilarity()),
                                 X)
            res = isapprox(vec(A), df[!, i])
            if !res
                println("Iteration $i failed on DBHT_MaximumDistanceSimilarity.")
                find_tol(vec(A), df[!, i]; name1 = :A, name2 = :df)
            end
        end
    end
end
=#