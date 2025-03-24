@safetestset "Utils tests" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, Dates, Random, StableRNGs, CSV

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
        @test hcat(vcat(X1, X2), vcat(F1, F2)) == df

        rd = prices_to_returns(Px; missing_col_percent = 0.1, missing_row_percent = 0.5)
        ts3 = rd.ts
        X3 = rd.X
        F3 = rd.F
        @test isempty(F3)

        rd = prices_to_returns(Px; missing_col_percent = 0.1, missing_row_percent = Inf)
        ts4 = rd.ts
        X4 = rd.X
        F4 = rd.F
        @test isempty(F4)

        df = CSV.read(joinpath(@__DIR__, "assets/prices_to_returns_X.csv"), DataFrame)
        @test vcat(X3, X4) == df

        @test dfy[2:end, :date] == ts1[!, 1] == ts2[!, 1] == ts3[!, 1] == ts4[!, 1]
    end
end
