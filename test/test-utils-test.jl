@safetestset "Utils tests" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, Dates, Random, StableRNGs, CSV,
          Statistics, LinearAlgebra

    @testset "prices_to_returns" begin
        rng = StableRNG(987654321)
        Px = rand(rng, 101, 20)
        Fx = rand(rng, 101, 5)

        dfx = DataFrame(Px, :auto)
        dfx.all_missing = fill(NaN, nrow(dfx))
        dfx[!, :date] = (today() - Day(100)):Day(1):today()
        Px = TimeArray(dfx; timestamp = :date)

        dfy = DataFrame(Fx, [:fx1, :fx2, :fx3, :fx4, :fx5])
        dfy.all_missing = fill(NaN, nrow(dfy))
        dfy[!, :date] = (today() - Day(100)):Day(1):today()
        Py = TimeArray(dfy; timestamp = :date)

        rd = prices_to_returns(Px, Py; missing_col_percent = 0.1, missing_row_percent = 0.5)
        ts1 = rd.ts
        X1 = rd.X
        F1 = rd.F
        rd = prices_to_returns(Px, Py; missing_col_percent = 0.1, missing_row_percent = Inf)
        ts2 = rd.ts
        X2 = rd.X
        F2 = rd.F

        df = CSV.read(joinpath(@__DIR__, "assets/prices_to_returns_X_F.csv"), DataFrame)
        @test hcat(vcat(X1, X2), vcat(F1, F2)) == Matrix(df)

        rd = prices_to_returns(Px; missing_col_percent = 0.1, missing_row_percent = 0.5)
        ts3 = rd.ts
        X3 = rd.X
        F3 = rd.F
        @test isnothing(F3)

        rd = prices_to_returns(Px; missing_col_percent = 0.1, missing_row_percent = Inf)
        ts4 = rd.ts
        X4 = rd.X
        F4 = rd.F
        @test isnothing(F4)

        df = CSV.read(joinpath(@__DIR__, "assets/prices_to_returns_X.csv"), DataFrame)
        @test vcat(X3, X4) == Matrix(df)

        @test dfy[2:end, :date] == ts1 == ts2 == ts3 == ts4
    end

    @testset "Misc" begin
        rng = StableRNG(987654321)
        X = rand(rng, 2, 5)
        sigma = cov(X)
        @test !isposdef(sigma)
        X1 = copy(sigma)
        X2 = posdef(PosDefEstimator(), X1)
        posdef!(PosDefEstimator(), X1)
        @test isapprox(X1, X2)
        @test !isapprox(X1, sigma; rtol = 1e-8)
        @test isnothing(posdef(nothing))

        X3 = copy(sigma)
        posdef!(PosDefEstimator(), X3)
        X4 = denoise(Denoise(), PosDefEstimator(), X3, 2 / 5)
        denoise!(Denoise(), PosDefEstimator(), X3, 2 / 5)
        @test isapprox(X3, X4)
        @test !isapprox(X3, sigma)
        @test isnothing(denoise(nothing))

        X5 = copy(sigma)
        posdef!(PosDefEstimator(), X5)
        X6 = detone(Detone(), PosDefEstimator(), X5)
        detone!(Detone(), PosDefEstimator(), X5)
        @test isapprox(X5, X6)
        @test !isapprox(X5, sigma)
        @test isnothing(detone(nothing))

        X7 = copy(sigma)
        X8 = matrix_processing(DefaultMatrixProcessing(), X7, X)
        matrix_processing!(DefaultMatrixProcessing(), X7, X)
        @test isapprox(X7, X8)
        @test !isapprox(X7, sigma; rtol = 1e-8)
        @test isnothing(matrix_processing(nothing))

        X9 = copy(sigma)
        X10 = matrix_processing(NonPositiveDefiniteMatrixProcessing(), X9, X)
        matrix_processing!(NonPositiveDefiniteMatrixProcessing(), X9, X)
        @test isapprox(X9, X10)
        @test isapprox(X9, sigma)
        @test isnothing(matrix_processing!(nothing))

        @test isnothing(PortfolioOptimisers.matrix_processing_algorithm(nothing))
    end
end
