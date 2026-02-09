@safetestset "Cross Validation" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, Clarabel, Dates
    @testset "Walk forward" begin
        rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__,
                                                           "./assets/SP500.csv.gz"));
                                         timestamp = :Date)[(end - 252 * 4):end],
                               TimeArray(CSV.File(joinpath(@__DIR__,
                                                           "./assets/Factors.csv.gz"));
                                         timestamp = :Date)[(end - 252 * 4):end])

        cv = IndexWalkForward(127, 171; reduce_test = false)
        train, test = split(cv, rd)
        N = n_splits(cv, rd)
        @test length(train) == length(test) == N
        @test all(x -> length(x) == 127, train)
        @test all(x -> length(x) == 171, test)
        @test train == UnitRange{Int64}[1:127, 172:298, 343:469, 514:640, 685:811]
        @test test == UnitRange{Int64}[128:298, 299:469, 470:640, 641:811, 812:982]

        cv = IndexWalkForward(137, 111; reduce_test = true, purged_size = 13)
        train, test = split(cv, rd)
        N = n_splits(cv, rd)
        @test length(train) == length(test) == N
        @test all(x -> length(x) == 137, train)
        @test all(x -> length(x) == 111, test[1:(end - 1)])
        @test train ==
              UnitRange{Int64}[1:137, 112:248, 223:359, 334:470, 445:581, 556:692, 667:803,
                               778:914]
        @test test == UnitRange{Int64}[151:261, 262:372, 373:483, 484:594, 595:705, 706:816,
                                       817:927, 928:1008]

        cv = IndexWalkForward(137, 111; reduce_test = true, purged_size = 13,
                              expend_train = true)
        train, test = split(cv, rd)
        N = n_splits(cv, rd)
        @test length(train) == length(test) == N
        for (i, t) in enumerate(train)
            @test length(t) == 137 + (i - 1) * 111
        end
        @test all(x -> length(x) == 111, test[1:(end - 1)])
        @test train ==
              UnitRange{Int64}[1:137, 1:248, 1:359, 1:470, 1:581, 1:692, 1:803, 1:914]
        @test test == UnitRange{Int64}[151:261, 262:372, 373:483, 484:594, 595:705, 706:816,
                                       817:927, 928:1008]

        # cv = DateWalkForward(12, 3; period = Month(1))
        # train, test = split(cv, rd)
        # N = n_splits(cv, rd)

        # cv = DateWalkForward(Day(18), 13; period = Month(1), previous = true)
        # train, test = split(cv, rd)
        # N = n_splits(cv, rd)

    end
end
