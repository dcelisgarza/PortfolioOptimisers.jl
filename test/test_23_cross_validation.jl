# @safetestset "Cross Validation" begin
using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, Clarabel, Dates
# @testset "Walk forward" begin
rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end],
                       TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end])

cv = IndexWalkForward(50, 40; reduce_test = false)
train, test = split(cv, rd)
N = n_splits(cv, rd)

cv = DateWalkForward(12, 3; period = Month(1))
train, test = split(cv, rd)
N = n_splits(cv, rd)

cv = DateWalkForward(Day(18), 13; period = Month(1), previous = true)
train, test = split(cv, rd)
N = n_splits(cv, rd)

# end
# end
