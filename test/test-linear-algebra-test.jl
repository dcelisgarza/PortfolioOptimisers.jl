@safetestset "Linear Algebra" begin
    using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test, CSV, LinearAlgebra,
          DataFrames
    @testset "Operators" begin
        rng = StableRNG(987654321)
        X1 = rand(rng, 10)
        X2 = rand(rng, 10)
        res1 = X1 * transpose(X2)
        res2 = PortfolioOptimisers.:⊗(X1, X2)
        res3 = PortfolioOptimisers.:⊗(transpose(X1), transpose(X2))
        res4 = PortfolioOptimisers.outer_prod(X1, X2)
        res5 = PortfolioOptimisers.outer_prod(transpose(X1), transpose(X2))
        res6 = PortfolioOptimisers.:⊗(X2, X1)
        res7 = PortfolioOptimisers.outer_prod(X2, X1)
        @test isapprox(res1, res2)
        @test isapprox(res1, res3)
        @test isapprox(res1, res4)
        @test isapprox(res1, res5)
        @test !isapprox(res1, res6)
        @test !isapprox(res1, res7)
    end
    @testset "Non Positive Definite Matrices" begin
        rng = StableRNG(123456789)
        X = randn(rng, 10, 20)
        sigma1 = cov(X)
        sigma2 = copy(sigma1)
        sigma3 = copy(sigma1)
        fix_non_positive_definite_matrix!(FNPDM_NearestCorrelationMatrix(), sigma1)
        @test isposdef(sigma1)

        fix_non_positive_definite_matrix!(nothing, sigma2)
        @test !isposdef(sigma2)
        @test isapprox(sigma2, sigma3)
    end
    @testset "Denoise algorithms" begin
        rng = StableRNG(987654321)
        X = randn(rng, 1000, 20)
        T, N = size(X)
        q = T / N
        sigma = cov(X)
        des = [nothing, FixedDenoise(), ShrunkDenoise(), SpectralDenoise()]
        denoise_t = CSV.read(joinpath(@__DIR__, "./assets/Denoise.csv"), DataFrame)
        for i ∈ 1:ncol(denoise_t)
            sigma1 = copy(sigma)
            denoise!(des[i], FNPDM_NearestCorrelationMatrix(), sigma1, q)
            MN = size(sigma1)
            res = isapprox(sigma1, reshape(denoise_t[!, i], MN))
            if !res
                println("Fails on iteration $i")
            end
            @test res
        end
    end
    @testset "Detone" begin
        rng = StableRNG(987654321)
        X = randn(rng, 1000, 20)
        T, N = size(X)
        q = T / N
        sigma = cov(X)
        des = [nothing, Detone(), Detone(; n = 3)]
        detone = CSV.read(joinpath(@__DIR__, "./assets/Detone.csv"), DataFrame)
        for i ∈ 1:ncol(detone)
            sigma1 = copy(sigma)
            detone!(des[i], FNPDM_NearestCorrelationMatrix(), sigma1)
            MN = size(sigma1)
            res = isapprox(sigma1, reshape(detone[!, i], MN))
            if !res
                println("Fails on iteration $i")
            end
            @test res
        end
    end
end
