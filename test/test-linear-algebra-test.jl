@safetestset "Linear Algebra" begin
    using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test, CSV, LinearAlgebra,
          DataFrames, TimeSeries
    @testset "Operators" begin
        rng = StableRNG(987654321)
        X1 = rand(rng, 10)
        X2 = rand(rng, 10)
        res1 = X1 * transpose(X2)
        res2 = PortfolioOptimisers.:⊗(X1, X2)
        res3 = PortfolioOptimisers.:⊗(transpose(X1), transpose(X2))
        res4 = PortfolioOptimisers.:⊗(X1, X2)
        res5 = PortfolioOptimisers.:⊗(transpose(X1), transpose(X2))
        res6 = PortfolioOptimisers.:⊗(X2, X1)
        res7 = PortfolioOptimisers.:⊗(X2, X1)
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
        posdef!(PosdefEstimator(), sigma1)
        @test isposdef(sigma1)

        posdef!(nothing, sigma2)
        @test !isposdef(sigma2)
        @test isapprox(sigma2, sigma3)
    end
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                  timestamp = :timestamp)
    rd = prices_to_returns(X[(end - 252):end])
    X = rd.X
    T, N = size(X)
    q = T / N
    sigma = cov(X)
    @testset "Denoise algorithms" begin
        des = [nothing, Denoise(; alg = FixedDenoise()), Denoise(; alg = ShrunkDenoise()),
               Denoise(; alg = SpectralDenoise())]
        denoise_t = CSV.read(joinpath(@__DIR__, "./assets/Denoise.csv"), DataFrame)
        for i in eachindex(des)
            sigma1 = copy(sigma)
            denoise!(des[i], sigma1, q, PosdefEstimator())
            MN = size(sigma1)
            res = isapprox(sigma1, reshape(denoise_t[!, i], MN))
            if !res
                println("Fails on iteration $i")
            end
            @test res
        end
    end
    @testset "Detone" begin
        des = [nothing, Detone(), Detone(; n = 3)]
        detone_t = CSV.read(joinpath(@__DIR__, "./assets/Detone.csv"), DataFrame)
        for i in eachindex(des)
            sigma1 = copy(sigma)
            detone!(des[i], sigma1, PosdefEstimator())
            MN = size(sigma1)
            res = isapprox(sigma1, reshape(detone_t[!, i], MN))
            if !res
                println("Fails on iteration $i")
            end
            @test res
        end
    end
end
