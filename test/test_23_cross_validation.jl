# @safetestset "Cross Validation" begin
using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, Clarabel
# @testset "Walk forward" begin
rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end],
                       TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                 timestamp = :Date)[(end - 252 * 4):end])

cv = WalkForward(23 * 3, 27)
train, test = split(cv, rd)

cv = WalkForward(23, 27 * 3)
train, test = split(cv, rd)
# end
# end
