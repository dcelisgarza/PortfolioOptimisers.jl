using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test

rng = StableRNG(123456789)
X = randn(rng, 100, 10)
ew = eweights(1:100, 0.3; scale = true)

ces = [FullCovariance(),
       FullCovariance(; ce = SimpleCovariance(; corrected = false), w = ew),
       SemiCovariance(),
       SemiCovariance(; ce = SimpleCovariance(; corrected = false), w = ew),
       SpearmanCovariance(), KendallCovariance(), MutualInfoCovariance(),
       MutualInfoCovariance(; bins = B_Knuth()),
       MutualInfoCovariance(; bins = B_Freedman()),
       MutualInfoCovariance(; bins = B_Scott()), MutualInfoCovariance(; bins = 5),
       MutualInfoCovariance(; ve = SimpleVariance(; corrected = false), w = ew),
       DistanceCovariance(), DistanceCovariance(; w = ew), LTDCovariance()]

for ce ∈ ces
    cv = cov(ce, X)
    cr = cor(ce, X)
    @test isapprox(cov2cor(cv), cr)
end
