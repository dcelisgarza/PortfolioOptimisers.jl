@testset "Linear Algebra" begin
    using PortfolioOptimisers, Test, CSV, LinearAlgebra, DataFrames, TimeSeries, StableRNGs,
          Random
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
        sigma1 = cov(X)
        sigma2 = copy(sigma1)
        sigma3 = copy(sigma1)
        sigma4 = posdef(Posdef(), sigma1)

        posdef!(Posdef(), sigma1)
        @test LinearAlgebra.isposdef(sigma1)
        @test sigma4 == sigma1

        posdef!(nothing, sigma2)
        @test !LinearAlgebra.isposdef(sigma2)
        @test isapprox(sigma2, sigma3)

        @test posdef(nothing, sigma2) === sigma2
    end
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(EmpiricalPrior(), rd)
    T, N = size(rd.X)
    q = T / N
    @testset "Denoise algorithms" begin
        dns = [nothing, Denoise(; alg = FixedDenoise()), Denoise(; alg = ShrunkDenoise()),
               Denoise(; alg = SpectralDenoise())]
        df = CSV.read(joinpath(@__DIR__, "./assets/Denoise.csv.gz"), DataFrame)
        for (i, dn) in pairs(dns)
            sigma1 = copy(pr.sigma)
            denoise!(dn, sigma1, q)
            success = isapprox(vec(sigma1), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(sigma1), df[!, i])
            end
            @test success
        end
        @test denoise(nothing, pr.sigma) === pr.sigma
        sigma1 = copy(pr.sigma)
        sigma2 = denoise(dns[2], sigma1, q)
        denoise!(dns[2], sigma1, q)
        @test sigma1 == sigma2
    end
    @testset "Detone" begin
        dts = [nothing, Detone(), Detone(; n = 3)]
        df = CSV.read(joinpath(@__DIR__, "./assets/Detone.csv.gz"), DataFrame)
        for (i, dt) in pairs(dts)
            sigma1 = copy(pr.sigma)
            detone!(dt, sigma1)
            success = isapprox(vec(sigma1), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(mu, df[!, i])
            end
            @test success
        end
        @test denoise(nothing, pr.sigma) === pr.sigma
        sigma1 = copy(pr.sigma)
        sigma2 = detone(dts[2], sigma1)
        detone!(dts[2], sigma1)
        @test sigma1 == sigma2
        @test detone(nothing, pr.sigma) === pr.sigma
    end
    @testset "Matrix processing" begin
        @test matrix_processing!(nothing, pr.sigma) === pr.sigma
        @test matrix_processing(nothing, pr.sigma) === pr.sigma
        @test PortfolioOptimisers.matrix_processing_algorithm(nothing, pr.sigma) ===
              pr.sigma
        sigma1 = copy(pr.sigma)
        sigma2 = copy(pr.sigma)
        sigma3 = copy(pr.sigma)
        sigma4 = matrix_processing(DenoiseDetoneAlgMatrixProcessing(), sigma1, pr.X)
        sigma5 = matrix_processing(nothing, sigma1, pr.X)
        sigma4 == matrix_processing!(DenoiseDetoneAlgMatrixProcessing(), sigma2, pr.X)
        sigma5 == matrix_processing!(nothing, sigma3, pr.X)
    end
end
