@safetestset "Linear Algebra" begin
    using PortfolioOptimisers, Test, CSV, LinearAlgebra, DataFrames, TimeSeries, StableRNGs,
          Random
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
        @test isapprox(res1, res6')
        @test isapprox(res1, res7')
    end
    @testset "Non Positive Definite Matrices" begin
        rng = StableRNG(123456789)
        X = randn(rng, 10, 20)
        sigma1 = Statistics.cov(X)
        sigma2 = copy(sigma1)
        sigma3 = copy(sigma1)
        sigma4 = posdef(Posdef(), sigma1)

        posdef!(Posdef(), sigma1)
        @test LinearAlgebra.isposdef(sigma1)
        @test sigma4 == sigma1

        posdef!(nothing, sigma2)
        @test !LinearAlgebra.isposdef(sigma2)
        @test isapprox(sigma2, sigma3)

        @test isnothing(posdef(nothing))
    end
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(EmpiricalPrior(), rd)
    T, N = size(rd.X)
    q = T / N
    @testset "Denoise algorithms" begin
        des = [nothing, Denoise(; alg = FixedDenoise()), Denoise(; alg = ShrunkDenoise()),
               Denoise(; alg = SpectralDenoise())]
        df = CSV.read(joinpath(@__DIR__, "./assets/Denoise.csv.gz"), DataFrame)
        for (i, de) in pairs(des)
            sigma1 = copy(pr.sigma)
            denoise!(de, sigma1, q)
            success = isapprox(vec(sigma1), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(mu, df[!, i])
            end
            @test success
        end
        @test isnothing(denoise(nothing))
        sigma1 = copy(pr.sigma)
        sigma2 = denoise(des[2], sigma1, q)
        denoise!(des[2], sigma1, q)
        @test sigma1 == sigma2
    end
    @testset "Detone" begin
        des = [nothing, Detone(), Detone(; n = 3)]
        df = CSV.read(joinpath(@__DIR__, "./assets/Detone.csv.gz"), DataFrame)
        for (i, de) in pairs(des)
            sigma1 = copy(pr.sigma)
            detone!(des[i], sigma1)
            success = isapprox(vec(sigma1), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(mu, df[!, i])
            end
            @test success
        end
        @test isnothing(detone(nothing))
        sigma1 = copy(pr.sigma)
        sigma2 = detone(des[2], sigma1)
        detone!(des[2], sigma1)
        @test sigma1 == sigma2
    end
    @testset "Matrix processing" begin
        @test isnothing(matrix_processing!(nothing))
        @test isnothing(matrix_processing(nothing))
        @test isnothing(PortfolioOptimisers.matrix_processing_algorithm(nothing))
        sigma1 = copy(pr.sigma)
        sigma2 = copy(pr.sigma)
        sigma3 = copy(pr.sigma)
        sigma4 = matrix_processing(DenoiseDetoneAlgMatrixProcessing(), sigma1, pr.X)
        sigma5 = matrix_processing(nothing, sigma1, pr.X)
        sigma4 == matrix_processing!(DenoiseDetoneAlgMatrixProcessing(), sigma2, pr.X)
        sigma5 == matrix_processing!(nothing, sigma3, pr.X)
    end
end
