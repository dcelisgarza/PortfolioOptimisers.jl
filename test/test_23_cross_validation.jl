# @safetestset "Cross Validation" begin
using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, Clarabel, Dates
# @testset "Walk forward" begin
rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end],
                       TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end])

function test_walk_forward_no_period(train_size::Integer, test_size::Integer,
                                     purged_size::Integer, reduce_test::Bool,
                                     expend_train::Bool,
                                     X::Union{Integer, Matrix, ReturnsResult})
    cv = WalkForward(train_size, test_size; purged_size = purged_size,
                     reduce_test = reduce_test, expend_train = expend_train)
    train, test = split(cv, X)
    T = if isa(X, Integer)
        X
    elseif isa(X, Matrix)
        size(X, 1)
    elseif isa(X, ReturnsResult)
        size(X.X, 1)
    end
    for (i, (train_idx, test_idx)) in enumerate(zip(train, test))
        @test train_idx[1] == (expend_train ? 1 : 1 + (i - 1) * test_size)
        @test train_idx[end] == ((i - 1) * test_size) + train_size
        @test test_idx[1] == ((i - 1) * test_size) + train_size + purged_size
        @test test_idx[end] ==
              min(((i - 1) * test_size) + train_size + test_size + purged_size, T)
    end
end

train_size = [23 * 3, 23 * 5, 27, 27]
test_size = [27, 27, 11 * 5, 31 * 2]
purged_size = [5, 3, 0, 47]
expend_train = [true, false, false, true]
reduce_test = [true, false, false, true]

for (trs, tes, ps, et, rt) in
    zip(train_size, test_size, purged_size, expend_train, reduce_test)
    test_walk_forward_no_period(trs, tes, ps, rt, et, rd)
end

cv = WalkForward(5, 10; period = Day(25),
                 period_offset = Dates.CompoundPeriod(Week(3), Day(2)), previous = false,
                 purged_size = 4, expend_train = true)
train, test = split(cv, rd)

rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                 timestamp = :Date)[(end - 252):end],
                       TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                 timestamp = :Date)[(end - 252):end])
cv = WalkForward(Day(5), 20; period = Day(1), previous = false)
train, test = split(cv, rd)

# end
# end
