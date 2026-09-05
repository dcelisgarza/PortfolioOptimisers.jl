#=
Check the three dependence-based covariances -- distance, lower tail and rank -- against the
mathematics their docstrings state. Issue #458 of child map 3 (#417) under the map of maps
(#404).

Coverage was ALREADY terminal for all three files, so nothing here exists to reach a line.
Every test below pins a CLAIM the documentation makes, computed a second way. #404's
condition 2 is "checked with real numbers. Not read -- run", and a claim no test holds is a
claim the next reader has to re-derive.

THREE FACTS SHAPE THE PROBES.

1. `StatsBase.corspearman` and `StatsBase.corkendall` are TIE-CORRECTED. The textbook closed
   forms -- Spearman's `1 - 6 sum(d^2) / (T(T^2-1))` and Kendall's `(C-D)/binomial(T,2)` --
   are the NO-TIES case only, and the two disagree measurably on a tied sample. Both rank
   docstrings state the tie-corrected form and name the disagreeing pair of numbers, so both
   sides of that pair are asserted here.

2. `DistanceCovariance` carries NO variance estimator field, so its `cov` is not the generic
   `ve` fallback of `01_Base_Moments.jl`. The diagonal is the DISTANCE standard deviation,
   and `cor` is the matrix rescaled by the square roots of that same diagonal. The other two
   estimators DO take the fallback, and their `cov` diagonal is the ordinary sample variance.

3. `lower_tail_dependence`'s `k > 0` guard is NOT reached by a small positive `alpha`. `ceil`
   of any positive real is at least 1, so `alpha = nextfloat(0.0)` on a non-empty matrix
   still selects one observation. The two cases that do reach it are `alpha == 0` on a direct
   call, and a matrix with no observation. The docstring said otherwise before #458.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs, FLoops

@testset "Distance covariance: the doubly-centred statistic" begin
    rng = StableRNG(987654321)
    X = randn(rng, 40, 4)
    ce = DistanceCovariance(; ex = FLoops.SequentialEx())

    # The reference builds the centring by hand from the Euclidean distance on the line.
    function ref_dcov(v1, v2)
        n = length(v1)
        a = [abs(v1[k] - v1[l]) for k in 1:n, l in 1:n]
        b = [abs(v2[k] - v2[l]) for k in 1:n, l in 1:n]
        A = a .- mean(a; dims = 1) .- mean(a; dims = 2) .+ mean(a)
        B = b .- mean(b; dims = 1) .- mean(b; dims = 2) .+ mean(b)
        return (sqrt(sum(A .* B) / n^2), sqrt(sum(A .* A) / n^2), sqrt(sum(B .* B) / n^2))
    end

    v1, v2 = view(X, :, 1), view(X, :, 2)
    dxy, dxx, dyy = ref_dcov(v1, v2)
    @test PortfolioOptimisers.cov_distance(ce, v1, v2) ≈ dxy
    @test PortfolioOptimisers.cor_distance(ce, v1, v2) ≈ dxy / sqrt(dxx * dyy)

    # A series against itself is exactly 1, and an independent pair is far below a
    # non-linearly dependent one that Pearson cannot see at all.
    @test isone(PortfolioOptimisers.cor_distance(ce, v1, v1))
    rng2 = StableRNG(24680)
    u = randn(rng2, 400)
    indep = randn(rng2, 400)
    nonlin = u .^ 2
    c_indep = PortfolioOptimisers.cor_distance(ce, u, indep)
    c_nonlin = PortfolioOptimisers.cor_distance(ce, u, nonlin)
    @test c_indep < 0.1
    @test c_nonlin > 0.5
    @test abs(cor(u, nonlin)) < 0.2

    # `cov` overrides the generic `ve` fallback: the diagonal is the distance standard
    # deviation, and `cor` is `cov` rescaled by the square roots of that diagonal.
    Sigma = cov(ce, X)
    Rho = cor(ce, X)
    @test diag(Sigma) ≈
          [PortfolioOptimisers.cov_distance(ce, view(X, :, i), view(X, :, i)) for i in 1:4]
    d = sqrt.(diag(Sigma))
    @test Rho ≈ Sigma ./ (d * d')
    @test all(isone, diag(Rho))
    @test !hasproperty(ce, :ve)
end

@testset "Distance covariance: pairwise distances and the asset axis" begin
    # Non-square on purpose: 7 observations, 3 assets, so a transposed index cannot hide.
    Xns = randn(StableRNG(1111), 7, 3)
    ce = DistanceCovariance(; ex = FLoops.SequentialEx())

    @test size(cov(ce, Xns)) == (3, 3)
    @test size(cor(ce, Xns)) == (3, 3)
    @test cov(ce, permutedims(Xns); dims = 2) == cov(ce, Xns)
    @test cor(ce, permutedims(Xns); dims = 2) == cor(ce, Xns)

    # `calc_pairwise_dists` returns two T x T matrices of the metric over the observations.
    v1, v2 = view(Xns, :, 1), view(Xns, :, 2)
    D1, D2 = PortfolioOptimisers.calc_pairwise_dists(ce, v1, v2, nothing)
    @test size(D1) == size(D2) == (7, 7)
    @test D1 ≈ [abs(Xns[k, 1] - Xns[l, 1]) for k in 1:7, l in 1:7]

    # The weighted branch scales each COORDINATE by its weight before the distance is taken.
    w = pweights(range(0.5, 1.5; length = 7))
    Dw1, _ = PortfolioOptimisers.calc_pairwise_dists(ce, v1, v2, w)
    @test Dw1 ≈ [abs(Xns[k, 1] * w[k] - Xns[l, 1] * w[l]) for k in 1:7, l in 1:7]
    @test Dw1 != D1
end

@testset "Distance covariance: the estimator's weights reach the metric" begin
    X = randn(StableRNG(987654321), 40, 4)
    wv = pweights(range(0.5, 1.5; length = 40))
    cew = DistanceCovariance(; w = wv, ex = FLoops.SequentialEx())
    ceu = DistanceCovariance(; ex = FLoops.SequentialEx())

    resolved = PortfolioOptimisers.get_observation_weights(cew.w, X)
    @test cov(cew, X)[1, 2] ==
          PortfolioOptimisers.cov_distance(cew, view(X, :, 1), view(X, :, 2), resolved)
    @test cov(cew, X) != cov(ceu, X)
    @test isnothing(PortfolioOptimisers.get_observation_weights(ceu.w, X))
end

@testset "Lower tail dependence: the joint count at the quantile" begin
    Xt = randn(StableRNG(555), 100, 3)
    T, N = size(Xt)
    alpha = 0.05
    k = ceil(Int, T * alpha)
    @test k == 5

    # The threshold is the k-th ORDER STATISTIC, and the estimate is the joint count over k.
    qs = [sort(Xt[:, i])[k] for i in 1:N]
    ref = [count((Xt[:, i] .<= qs[i]) .& (Xt[:, j] .<= qs[j])) / k for i in 1:N, j in 1:N]
    mv = sqrt(eps(eltype(Xt)))
    code = PortfolioOptimisers.lower_tail_dependence(Xt, alpha)
    @test code ≈ clamp.(ref, mv, one(eltype(Xt)))
    @test all(isone, diag(code))
    @test issymmetric(code)

    # The floor is NOT zero. A pair whose tails never coincide is reported as sqrt(eps).
    @test iszero(ref[1, 3])
    @test code[1, 3] == mv
end

@testset "Lower tail dependence: the k == 0 guard" begin
    Xt = randn(StableRNG(555), 100, 3)

    # No small POSITIVE alpha reaches the guard: `ceil` of a positive real is at least 1.
    @test ceil(Int, 100 * 1e-9) == 1
    @test ceil(Int, 100 * nextfloat(0.0)) == 1
    tiny = PortfolioOptimisers.lower_tail_dependence(Xt, 1 / 100)
    @test all(isone, diag(tiny))
    @test !all(iszero, tiny)

    # The two cases that DO reach it: alpha exactly zero, and a matrix with no observation.
    @test all(iszero, PortfolioOptimisers.lower_tail_dependence(Xt, 0.0))
    empty_res = PortfolioOptimisers.lower_tail_dependence(zeros(0, 3), 0.05)
    @test size(empty_res) == (3, 3)
    @test all(iszero, empty_res)

    # The estimator rules the first case out at construction.
    @test_throws DomainError LowerTailDependenceCovariance(; alpha = 0.0)
    @test_throws DomainError LowerTailDependenceCovariance(; alpha = 1.0)
end

@testset "Rank covariances: the tie correction" begin
    # One tied pair in each series, so the tie-corrected and closed forms MUST disagree.
    Xtie = [1.0 1.0; 2.0 1.0; 2.0 3.0; 4.0 4.0; 5.0 2.0]
    T = size(Xtie, 1)

    rx, ry = tiedrank(Xtie[:, 1]), tiedrank(Xtie[:, 2])
    sp_closed = 1 - 6 * sum((rx .- ry) .^ 2) / (T * (T^2 - 1))
    C = D = 0
    for t in 1:T, s in (t + 1):T
        p = (Xtie[t, 1] - Xtie[s, 1]) * (Xtie[t, 2] - Xtie[s, 2])
        p > 0 && (C += 1)
        p < 0 && (D += 1)
    end
    kd_closed = (C - D) / binomial(T, 2)

    sp = cor(SpearmanCovariance(), Xtie)[1, 2]
    kd = cor(KendallCovariance(), Xtie)[1, 2]

    # The exact pair of numbers both docstrings name.
    @test sp == 0.5526315789473685
    @test sp_closed == 0.575
    @test kd == 0.4444444444444444
    @test kd_closed == 0.4
    @test sp != sp_closed
    @test kd != kd_closed

    # Spearman IS the Pearson correlation of the mid-ranks, and Kendall IS tau_b.
    @test sp == cor(rx, ry)
    n0 = binomial(T, 2)
    nx = ny = binomial(2, 2)   # one tied group of size two in each series
    @test kd == (C - D) / sqrt((n0 - nx) * (n0 - ny))
end

@testset "Rank covariances: without ties the closed forms return" begin
    Xun = [1.0 2.0; 2.0 1.0; 3.0 4.0; 4.0 3.0; 5.0 5.0]
    T = size(Xun, 1)
    ru, su = tiedrank(Xun[:, 1]), tiedrank(Xun[:, 2])
    C = D = 0
    for t in 1:T, s in (t + 1):T
        p = (Xun[t, 1] - Xun[s, 1]) * (Xun[t, 2] - Xun[s, 2])
        p > 0 && (C += 1)
        p < 0 && (D += 1)
    end
    @test cor(SpearmanCovariance(), Xun)[1, 2] ≈
          1 - 6 * sum((ru .- su) .^ 2) / (T * (T^2 - 1))
    @test cor(KendallCovariance(), Xun)[1, 2] ≈ (C - D) / binomial(T, 2)
end

@testset "The generic ve fallback rescales the three that carry one" begin
    Xt = randn(StableRNG(555), 100, 3)
    for est in (SpearmanCovariance(), KendallCovariance(),
                LowerTailDependenceCovariance(; ex = FLoops.SequentialEx()))
        R = cor(est, Xt)
        S = cov(est, Xt)
        v = vec(std(est.ve, Xt))
        @test S ≈ R .* (v * v')
        @test diag(S) ≈ v .^ 2
        @test all(isone, diag(R))
        @test est isa PortfolioOptimisers.AbstractCovarianceEstimator
    end

    # The three correlation matrices are genuinely different statistics.
    @test cor(SpearmanCovariance(), Xt) != cor(KendallCovariance(), Xt)
    @test cor(SpearmanCovariance(), Xt) !=
          cor(LowerTailDependenceCovariance(; ex = FLoops.SequentialEx()), Xt)
end

@testset "The four estimators honour dims" begin
    #=
    `cor` is EXACT under the transpose for all four, but `cov` is not for the three that take
    the generic `ve` fallback: `std(ve, X)` sums a transposed view in a different order, and
    the three matrices differ by 4.4e-16. `DistanceCovariance`, which overrides the fallback,
    is exact on both. So the equality here is `isapprox` and not `==` -- a tightening to `==`
    would be a knife-edge test, not a stronger one.
    =#
    Xt = randn(StableRNG(555), 30, 3)
    for est in (SpearmanCovariance(), KendallCovariance(),
                LowerTailDependenceCovariance(; ex = FLoops.SequentialEx()),
                DistanceCovariance(; ex = FLoops.SequentialEx()))
        @test cor(est, permutedims(Xt); dims = 2) == cor(est, Xt)
        @test cov(est, permutedims(Xt); dims = 2) ≈ cov(est, Xt)
        @test_throws DomainError cor(est, Xt; dims = 3)
        @test_throws DomainError cov(est, Xt; dims = 3)
    end
    @test cov(DistanceCovariance(; ex = FLoops.SequentialEx()), permutedims(Xt);
              dims = 2) == cov(DistanceCovariance(; ex = FLoops.SequentialEx()), Xt)
end
