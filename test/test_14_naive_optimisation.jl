@testset "Naive optimisation tests" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, LinearAlgebra, StableRNGs
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(EmpiricalPrior(), rd)
    N = size(pr.X, 2)

    sets = UniverseSets(; dict = Dict("nx" => rd.nx))
    wb = WeightBoundsEstimator(; lb = "AAPL" => 0.07, ub = nothing)

    res = optimise(InverseVolatility(; pe = pr), rd)
    w = inv.(sqrt.(LinearAlgebra.diag(pr.sigma)))
    w /= sum(w)
    @test isapprox(res.w, w)

    res = optimise(EqualWeighted(), rd)
    @test isapprox(res.w, range(; start = inv(N), stop = inv(N), length = N))

    res = optimise(RandomWeighted(; rng = StableRNG(123456789)), rd)
    @test isapprox(sum(res.w), 1)

    @test optimise(RandomWeighted(; alpha = 5, rng = StableRNG(123456789), seed = 42),
                   rd).w == optimise(RandomWeighted(; alpha = fill(5, size(rd.X, 2)),
                              rng = StableRNG(123456789), seed = 42), rd).w

    res = optimise(RandomWeighted(), rd)
    @test isapprox(sum(res.w), 1)

    res = optimise(InverseVolatility(; wb = wb, sets = sets, pe = pr), rd)
    w = inv.(sqrt.(LinearAlgebra.diag(pr.sigma)))
    @test isapprox(res.w[1], 0.07)

    res = optimise(EqualWeighted(; wb = wb, sets = sets), rd)
    @test isapprox(res.w[1], 0.07)
    @test all(isapprox.(res.w[2:end], (1 - 0.07) / 19))

    res = optimise(RandomWeighted(; wb = wb, sets = sets, rng = StableRNG(123456789)), rd)
    @test isapprox(res.w[1], 0.07)
    @test isapprox(sum(res.w), 1)

    # #885: `ReturnsResult` is always observations × assets, so `dims = 2` must be
    # refused, not answered by reading the observation axis as the asset count.
    res = optimise(EqualWeighted(), rd; dims = 1)
    @test isapprox(res.w, range(; start = inv(N), stop = inv(N), length = N))
    @test_throws PortfolioOptimisers.ConflictingArgumentError optimise(EqualWeighted(), rd;
                                                                       dims = 2)
    @test_throws PortfolioOptimisers.ConflictingArgumentError optimise(RandomWeighted(), rd;
                                                                       dims = 2)
end
