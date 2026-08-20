using PortfolioOptimisers, Test

# Card 7 of the 2026-08-17 maintainability review: the out-of-place verbs are documented
# as an optional part of the extension interface, so an estimator that declares only the
# in-place form must inherit a copying fallback. These estimators declare nothing else.
struct Card7Posdef <: PortfolioOptimisers.AbstractPosdefEstimator end
function PortfolioOptimisers.posdef!(::Card7Posdef, X::AbstractMatrix)
    X .= (X .+ transpose(X)) ./ 2
    return X
end
struct Card7Denoise <: PortfolioOptimisers.AbstractDenoiseEstimator end
function PortfolioOptimisers.denoise!(::Card7Denoise, X::AbstractMatrix, q::Number)
    X .*= 2
    return X
end
struct Card7Detone <: PortfolioOptimisers.AbstractDetoneEstimator end
function PortfolioOptimisers.detone!(::Card7Detone, X::AbstractMatrix)
    X .-= 1
    return X
end

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
        sigma4 = matrix_processing(MatrixProcessing(), sigma1, pr.X)
        sigma5 = matrix_processing(nothing, sigma1, pr.X)
        sigma4 == matrix_processing!(MatrixProcessing(), sigma2, pr.X)
        sigma5 == matrix_processing!(nothing, sigma3, pr.X)
    end
    @testset "Inherited out-of-place fallback" begin
        X = [1.0 2.0; 0.0 1.0]
        Xc = copy(X)
        # Each verb inherits the copying wrapper from its abstract supertype.
        @test posdef(Card7Posdef(), X) == [1.0 1.0; 1.0 1.0]
        @test X == Xc
        @test denoise(Card7Denoise(), X, 2.0) == [2.0 4.0; 0.0 2.0]
        @test X == Xc
        @test detone(Card7Detone(), X) == [0.0 1.0; -1.0 0.0]
        @test X == Xc
        # The field bounds accept a user estimator, so the contract is usable end to end.
        mp = MatrixProcessing(; pdm = Card7Posdef(), dn = Card7Denoise(),
                              dt = Card7Detone())
        @test mp.pdm isa Card7Posdef
        @test mp.dn isa Card7Denoise
        @test mp.dt isa Card7Detone
        @test Denoise(; pdm = Card7Posdef()).pdm isa Card7Posdef
        @test Detone(; pdm = Card7Posdef()).pdm isa Card7Posdef
        sigma = [1.0 0.5; 0.25 1.0]
        sigmac = copy(sigma)
        Xr = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        # Order (:pdm, :dn, :dt, :alg): symmetrise, then double, then subtract one.
        @test matrix_processing(mp, sigma, Xr) == [1.0 -0.25; -0.25 1.0]
        @test sigma == sigmac
    end
end
