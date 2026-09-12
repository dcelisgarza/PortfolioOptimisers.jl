#=
The coverage policy: the opt-in that replaces the all-or-nothing Coverage Universe by
available-case estimation, and the per-cell denominators the partial-fit states carry for it.
Decided by issue #866, built by issue #977, under map #861.
=#
struct AdmitEverything <: PortfolioOptimisers.AbstractCoverageAlgorithm end
function PortfolioOptimisers.admits(::AdmitEverything, args...)
    return true
end
function PortfolioOptimisers.fold_inactive!(::AdmitEverything,
                                            state::PortfolioOptimisers.AbstractPartialFitState,
                                            ::AbstractVector{<:Bool})
    return state
end
function coverage_panel(amsk)
    pf = [PortfolioOptimisers.NumericPanelField(; name = "mcap", vals = ones(size(amsk)...),
                                                omsk = trues(size(amsk)...))]
    return AssetPanel(; pf = pf, amsk = amsk, emsk = copy(amsk))
end
# The per-cell oracle of a higher-order available-case fit: each asset is centred on its own
# finite observations, and a cell is the mean of the product over the observations its assets
# share. `clip` is `identity` for a `FullMoment` arm and `x -> min(x, 0)` for a `SemiMoment` one.
function comoment_deviations(X, clip)
    mu = [mean(filter(isfinite, view(X, :, j))) for j in axes(X, 2)]
    return clip.(X .- transpose(mu))
end
function comoment_cell(Y, idx)
    rows = findall(t -> all(i -> isfinite(Y[t, i]), idx), axes(Y, 1))
    return if isempty(rows)
        NaN
    else
        sum(prod(Y[t, i] for i in idx) for t in rows) / length(rows)
    end
end
@testset "CoveragePolicy construction and the algorithm family" begin
    @test CoveragePolicy().min_coverage == 0.0
    @test isa(CoveragePolicy().alg, DecayCoverage)
    @test CoveragePolicy(; min_coverage = 0.25, alg = ResetCoverage()).min_coverage == 0.25
    @test_throws DomainError CoveragePolicy(; min_coverage = -0.1)
    @test_throws DomainError CoveragePolicy(; min_coverage = 1.5)
    @test ExpireCoverage().after == 0
    @test ExpireCoverage(; after = 5).after == 5
    @test_throws DomainError ExpireCoverage(; after = -1)
    @test isa(DecayCoverage(), PortfolioOptimisers.AbstractCoverageAlgorithm)
    @test isa(ResetCoverage(), PortfolioOptimisers.AbstractCoverageAlgorithm)
    @test isa(ExpireCoverage(), PortfolioOptimisers.AbstractCoverageAlgorithm)
    # `admits` is the read-out half of the interface.
    for alg in (DecayCoverage(), ResetCoverage())
        @test PortfolioOptimisers.admits(alg, 0.5, true, 0, 0.25)
        @test !PortfolioOptimisers.admits(alg, 0.2, true, 0, 0.25)
        @test !PortfolioOptimisers.admits(alg, 0.5, false, 0, 0.25)
        # Staleness is not read by either.
        @test PortfolioOptimisers.admits(alg, 0.5, true, 99, 0.25)
    end
    exp3 = ExpireCoverage(; after = 3)
    @test PortfolioOptimisers.admits(exp3, 0.5, false, 3, 0.25)
    @test !PortfolioOptimisers.admits(exp3, 0.5, false, 4, 0.25)
    @test PortfolioOptimisers.admits(exp3, 0.5, true, 99, 0.25)
    @test !PortfolioOptimisers.admits(exp3, 0.1, true, 0, 0.25)
    # `after = 0` is DecayCoverage's admission written the long way, over the triples a fold
    # can produce: an asset that is inactive at an observation has its staleness raised by that
    # observation, so an inactive asset's staleness is never zero.
    exp0 = ExpireCoverage()
    for (share, active, stale) in
        ((0.5, true, 0), (0.5, false, 1), (0.5, false, 7), (0.1, true, 0), (0.1, false, 2))
        @test PortfolioOptimisers.admits(exp0, share, active, stale, 0.25) ==
              PortfolioOptimisers.admits(DecayCoverage(), share, active, stale, 0.25)
    end
end
@testset "A `nothing` policy leaves every moment unchanged" begin
    rng = StableRNG(987654321)
    X = randn(rng, 64, 5)
    for (plain, policed) in
        ((SimpleExpectedReturns(), SimpleExpectedReturns(; cvg = nothing)),)
        @test mean(plain, X) == mean(policed, X)
    end
    @test var(SimpleVariance(), X) == var(SimpleVariance(; cvg = nothing), X)
    @test std(SimpleVariance(), X) == std(SimpleVariance(; cvg = nothing), X)
    @test cov(Covariance(), X) == cov(Covariance(; cvg = nothing), X)
    @test cor(Covariance(), X) == cor(Covariance(; cvg = nothing), X)
    @test cov(Covariance(; alg = SemiMoment()), X) ==
          cov(Covariance(; alg = SemiMoment(), cvg = nothing), X)
    # And a gapped sample is still refused with no policy.
    Xg = copy(X)
    Xg[3, 2] = NaN
    @test_throws PortfolioOptimisers.IsNonFiniteError mean(SimpleExpectedReturns(), Xg)
    @test_throws PortfolioOptimisers.IsNonFiniteError cov(Covariance(), Xg)
end
@testset "The available-case fit is the complete-case fit on a gapless sample" begin
    rng = StableRNG(123456789)
    X = randn(rng, 48, 4)
    cvg = CoveragePolicy()
    @test isapprox(vec(mean(SimpleExpectedReturns(; cvg = cvg), X)),
                   vec(mean(SimpleExpectedReturns(), X)); rtol = 1e-14)
    @test isapprox(vec(var(SimpleVariance(; cvg = cvg), X)), vec(var(SimpleVariance(), X));
                   rtol = 1e-13)
    @test isapprox(cov(Covariance(; cvg = cvg), X), cov(Covariance(), X); rtol = 1e-13)
    @test isapprox(cor(Covariance(; cvg = cvg), X), cor(Covariance(), X); rtol = 1e-13)
    @test isapprox(cov(Covariance(; alg = SemiMoment(), cvg = cvg), X),
                   cov(Covariance(; alg = SemiMoment()), X); rtol = 1e-13)
end
@testset "The available-case fit reads each cell's own observations" begin
    X = [1.0 2.0 NaN; 3.0 4.0 5.0; 5.0 NaN 7.0; 7.0 8.0 9.0]
    cvg = CoveragePolicy()
    mu = vec(mean(SimpleExpectedReturns(; cvg = cvg), X))
    @test mu ≈ [4.0, 14 / 3, 7.0]
    v = vec(var(SimpleVariance(; cvg = cvg), X))
    @test v[1] ≈ var([1.0, 3.0, 5.0, 7.0])
    @test v[2] ≈ var([2.0, 4.0, 8.0])
    @test v[3] ≈ var([5.0, 7.0, 9.0])
    sigma = cov(Covariance(; cvg = cvg), X)
    # Each pair is fitted on the rows the pair shares, and centred on that pair's own means.
    @test sigma[1, 2] ≈ cov([1.0, 3.0, 7.0], [2.0, 4.0, 8.0])
    @test sigma[1, 3] ≈ cov([3.0, 5.0, 7.0], [5.0, 7.0, 9.0])
    @test sigma[2, 3] ≈ cov([4.0, 8.0], [5.0, 9.0])
    @test issymmetric(sigma)
    @test LinearAlgebra.diag(sigma) ≈ v
    # The variance read-out and the covariance diagonal are the same numbers.
    @test LinearAlgebra.diag(sigma) == vec(var(Covariance(; cvg = cvg), X))
end
@testset "A pair with no shared observation is NaN, and the assets around it are not" begin
    X = [1.0 NaN; NaN 2.0; 3.0 NaN; NaN 4.0]
    cvg = CoveragePolicy()
    sigma = cov(Covariance(; cvg = cvg), X)
    @test isfinite(sigma[1, 1])
    @test isfinite(sigma[2, 2])
    @test isnan(sigma[1, 2])
    @test isnan(sigma[2, 1])
end
@testset "Online equals batch to the last bit, over a listing and a delisting" begin
    rng = StableRNG(20260909)
    T, N = 40, 4
    X = randn(rng, T, N)
    amsk = trues(T, N)
    # Asset 2 lists at row 11; asset 3 delists after row 25; asset 4 takes two holidays.
    amsk[1:10, 2] .= false
    amsk[26:end, 3] .= false
    X[1:10, 2] .= NaN
    X[26:end, 3] .= NaN
    X[[7, 19], 4] .= NaN
    cvg = CoveragePolicy()
    for est in (SimpleExpectedReturns(; cvg = cvg), SimpleVariance(; cvg = cvg),
                Covariance(; cvg = cvg))
        folded = est
        for t in axes(X, 1)
            folded = partial_fit!(folded, view(X, t, :); active_mask = view(amsk, t, :))
        end
        block = partial_fit!(est, X; dims = 1, active_mask = amsk)
        @test folded.cache.n == block.cache.n == T
        @test folded.cache.mu == block.cache.mu
        @test folded.cache.cvg.nu == block.cache.cvg.nu
        @test folded.cache.cvg.stale == block.cache.cvg.stale
        @test folded.cache.cvg.active == block.cache.cvg.active
    end
    # And the batch verb is that same fold, so the two answers are identical bit for bit.
    me = SimpleExpectedReturns(; cvg = cvg)
    folded = me
    for t in axes(X, 1)
        folded = partial_fit!(folded, view(X, t, :); active_mask = view(amsk, t, :))
    end
    @test isequal(vec(mean(me, X; active_mask = amsk)), mean(folded))
    ce = Covariance(; cvg = cvg)
    foldedc = ce
    for t in axes(X, 1)
        foldedc = partial_fit!(foldedc, view(X, t, :); active_mask = view(amsk, t, :))
    end
    @test isequal(cov(ce, X; active_mask = amsk), cov(foldedc))
end
@testset "The coverage floor decides which assets reach the answer" begin
    T = 20
    X = randn(StableRNG(11), T, 3)
    # Asset 3 is quoted at eight of the twenty observations, a share of 0.4.
    X[1:12, 3] .= NaN
    below = CoveragePolicy(; min_coverage = 0.45)
    above = CoveragePolicy(; min_coverage = 0.35)
    mu_below = vec(mean(SimpleExpectedReturns(; cvg = below), X))
    mu_above = vec(mean(SimpleExpectedReturns(; cvg = above), X))
    @test isnan(mu_below[3])
    @test isfinite(mu_above[3])
    @test all(isfinite, mu_below[1:2])
    s_below = cov(Covariance(; cvg = below), X)
    s_above = cov(Covariance(; cvg = above), X)
    @test isnan(s_below[3, 3])
    @test all(isnan, s_below[3, :])
    @test all(isnan, s_below[:, 3])
    @test all(isfinite, s_below[1:2, 1:2])
    @test isfinite(s_above[3, 3])
    # The Investable Mask is derived from the frame, with no new door.
    @test (isfinite.(mu_below) .& isfinite.(LinearAlgebra.diag(s_below))) ==
          BitVector([true, true, false])
    @test (isfinite.(mu_above) .& isfinite.(LinearAlgebra.diag(s_above))) ==
          BitVector([true, true, true])
end
@testset "The delisting algorithms differ over one panel" begin
    T, N = 12, 2
    X = Float64[t + 10 * (j - 1) for t in 1:T, j in 1:N]
    amsk = trues(T, N)
    # Asset 2 delists after row 6 and lists again at row 10.
    amsk[7:9, 2] .= false
    X[7:9, 2] .= NaN
    decay = partial_fit!(SimpleVariance(; cvg = CoveragePolicy(; alg = DecayCoverage())), X;
                         active_mask = amsk)
    reset = partial_fit!(SimpleVariance(; cvg = CoveragePolicy(; alg = ResetCoverage())), X;
                         active_mask = amsk)
    # Decay keeps the history it left behind; the reset threw it away at row 7.
    @test decay.cache.cvg.nu == [12, 9]
    @test reset.cache.cvg.nu == [12, 3]
    @test decay.cache.cvg.nu[1] == reset.cache.cvg.nu[1]
    @test decay.cache.mu[2] != reset.cache.mu[2]
    @test reset.cache.mu[2] ≈ mean(X[10:12, 2])
    @test decay.cache.mu[2] ≈ mean(vcat(X[1:6, 2], X[10:12, 2]))
    # And an asset that never lists again leaves the frame under Decay, and stays under Expire.
    amsk2 = trues(T, N)
    amsk2[7:end, 2] .= false
    X2 = copy(X)
    X2[7:end, 2] .= NaN
    d2 = vec(var(SimpleVariance(; cvg = CoveragePolicy(; alg = DecayCoverage())), X2;
                 active_mask = amsk2))
    e2 = vec(var(SimpleVariance(;
                                cvg = CoveragePolicy(; alg = ExpireCoverage(; after = 10))),
                 X2; active_mask = amsk2))
    @test isnan(d2[2])
    @test isfinite(e2[2])
    @test e2[2] ≈ var(X2[1:6, 2])
    # A reset over a per-pair accumulator zeroes the whole row and column of the asset.
    dcov = partial_fit!(Covariance(; cvg = CoveragePolicy(; alg = DecayCoverage())), X;
                        active_mask = amsk)
    rcov = partial_fit!(Covariance(; cvg = CoveragePolicy(; alg = ResetCoverage())), X;
                        active_mask = amsk)
    @test dcov.cache.cvg.nu == [12 9; 9 9]
    @test rcov.cache.cvg.nu == [12 3; 3 3]
    @test rcov.cache.cvg.centre[1, 2] ≈ mean(X[10:12, 1])
    @test rcov.cache.cvg.centre[2, 1] ≈ mean(X[10:12, 2])
    @test dcov.cache.cvg.centre[1, 1] ≈ mean(X[:, 1])
    @test cov(rcov)[2, 2] ≈ var(X[10:12, 2])
    @test cov(dcov)[2, 2] ≈ var(vcat(X[1:6, 2], X[10:12, 2]))
end
@testset "A state under a policy merges, copies and slices" begin
    rng = StableRNG(4242)
    T, N = 24, 3
    X = randn(rng, T, N)
    X[1:5, 2] .= NaN
    X[20:end, 3] .= NaN
    amsk = trues(T, N)
    amsk[1:5, 2] .= false
    amsk[20:end, 3] .= false
    cvg = CoveragePolicy()
    for est in (SimpleExpectedReturns(; cvg = cvg), SimpleVariance(; cvg = cvg),
                Covariance(; cvg = cvg))
        whole = partial_fit!(est, X; active_mask = amsk).cache
        a = partial_fit!(est, view(X, 1:13, :); active_mask = view(amsk, 1:13, :)).cache
        b = partial_fit!(est, view(X, 14:T, :); active_mask = view(amsk, 14:T, :)).cache
        merged = PortfolioOptimisers.merge_states(a, b)
        @test merged.n == whole.n
        @test merged.cvg.nu == whole.cvg.nu
        @test merged.cvg.stale == whole.cvg.stale
        @test merged.cvg.active == whole.cvg.active
        @test isapprox(merged.mu, whole.mu; rtol = 1e-12)
        # The copy shares no array with the original.
        c = copy(whole)
        c.cvg.nu .= 0
        @test !all(iszero, whole.cvg.nu)
        # The slice of the state is the state of the sliced universe.
        i = [1, 3]
        sliced = PortfolioOptimisers.port_opt_view(whole, i)
        @test sliced.mu == whole.mu[i]
        @test sliced.cvg.stale == whole.cvg.stale[i]
        @test sliced.cvg.active == whole.cvg.active[i]
        @test sliced.cvg.nu ==
              (isa(whole.cvg.nu, AbstractMatrix) ? whole.cvg.nu[i, i] : whole.cvg.nu[i])
    end
end
@testset "The Asset Panel seam hands a policed estimator the active mask" begin
    rng = StableRNG(777)
    T, N = 30, 3
    X = randn(rng, T, N)
    X[1:8, 3] .= NaN
    amsk = trues(T, N)
    amsk[1:8, 3] .= false
    pnl = coverage_panel(amsk)
    cvg = CoveragePolicy()
    # With no policy the window reduces to its Coverage Universe: asset 3 is out throughout.
    plain = vec(mean(SimpleExpectedReturns(), X, pnl))
    @test all(isfinite, plain[1:2])
    @test isnan(plain[3])
    # With a policy the estimator sees the mask and admits the asset from its listing on.
    policed = vec(mean(SimpleExpectedReturns(; cvg = cvg), X, pnl))
    @test all(isfinite, policed)
    @test policed[3] ≈ mean(X[9:end, 3])
    @test policed[1] ≈ mean(X[:, 1])
    # And the whole moment seam routes the same way.
    @test all(isfinite, cov(Covariance(; cvg = cvg), X, pnl))
    @test isnan(cov(Covariance(), X, pnl)[3, 3])
    @test all(isfinite, vec(var(SimpleVariance(; cvg = cvg), X, pnl)))
    @test all(isfinite, vec(std(SimpleVariance(; cvg = cvg), X, pnl)))
    @test all(isfinite, vec(std(Covariance(; cvg = cvg), X, pnl)))
    @test all(isfinite, cor(Covariance(; cvg = cvg), X, pnl))
    # The series admits the asset from its listing on rather than nowhere.
    series = PortfolioOptimisers.variance_series(SimpleVariance(; cvg = cvg), X, pnl)
    @test all(isnan, series[1:8, 3])
    @test isfinite(series[end, 3])
    plain_series = PortfolioOptimisers.variance_series(SimpleVariance(), X, pnl)
    @test all(isnan, plain_series[:, 3])
end
@testset "An available-case fit refuses a centre and a weight it cannot reproduce" begin
    X = randn(StableRNG(5), 16, 3)
    cvg = CoveragePolicy()
    @test_throws ArgumentError var(SimpleVariance(; cvg = cvg), X; mean = zeros(1, 3))
    @test_throws ArgumentError cov(Covariance(; cvg = cvg), X; mean = zeros(1, 3))
    @test_throws ArgumentError cov(Covariance(; alg = SemiMoment(), cvg = cvg), X;
                                   mean = zeros(1, 3))
    @test_throws ArgumentError mean(SimpleExpectedReturns(;
                                                          w = StatsBase.pweights(fill(1 /
                                                                                      16,
                                                                                      16)),
                                                          cvg = cvg), X)
    @test_throws DimensionMismatch partial_fit!(SimpleExpectedReturns(; cvg = cvg),
                                                view(X, 1, :); active_mask = trues(2))
    @test_throws DimensionMismatch partial_fit!(Covariance(; cvg = cvg), X;
                                                active_mask = trues(2, 3))
end
@testset "The one-asset arm of an available-case dispersion fit" begin
    x = [1.0, NaN, 3.0, 5.0, NaN, 7.0]
    ve = SimpleVariance(; cvg = CoveragePolicy())
    @test var(ve, x) ≈ var([1.0, 3.0, 5.0, 7.0])
    @test std(ve, x) ≈ std([1.0, 3.0, 5.0, 7.0])
    # The floor is read against the length of the series.
    strict = SimpleVariance(; cvg = CoveragePolicy(; min_coverage = 0.75))
    @test isnan(var(strict, x))
    @test_throws PortfolioOptimisers.IsNonFiniteError var(SimpleVariance(), x)
end
@testset "An available-case fit keeps the element type it was given" begin
    X32 = Float32[1 2 NaN32; 3 4 5; 5 NaN32 7; 7 8 9]
    cvg = CoveragePolicy()
    mu = mean(SimpleExpectedReturns(; cvg = cvg), X32)
    @test eltype(mu) === Float32
    @test mu[1] ≈ 4.0f0
    sigma = cov(Covariance(; cvg = cvg), X32)
    @test eltype(sigma) === Float32
    @test isapprox(sigma[1, 2], cov(Float32[1, 3, 7], Float32[2, 4, 8]); rtol = 1.0f-5)
end
@testset "coverage_valid_block reads a window the way a fold reads its rows" begin
    X = [1.0 NaN; 2.0 4.0; NaN 6.0]
    amsk = trues(3, 2)
    amsk[3, 2] = false
    Xo, msk, mu, active, stale = PortfolioOptimisers.coverage_valid_block(X, amsk; dims = 1)
    @test Xo === X
    @test msk == BitMatrix([true false; true true; false false])
    @test mu ≈ [1.5, 4.0]
    @test active == BitVector([true, false])
    @test stale == [1, 1]
    # No mask means every asset is active, so nothing is ever newly inactive.
    _, msk2, _, active2, _ = PortfolioOptimisers.coverage_valid_block(X, nothing; dims = 1)
    @test msk2 == isfinite.(X)
    @test all(active2)
    @test_throws DimensionMismatch PortfolioOptimisers.coverage_valid_block(X, trues(3, 3);
                                                                            dims = 1)
end
@testset "A user's own coverage algorithm dispatches" begin
    X = [1.0 2.0; 3.0 NaN; 5.0 6.0; 7.0 8.0]
    amsk = trues(4, 2)
    amsk[4, 2] = false
    strict = CoveragePolicy(; min_coverage = 0.9)
    # DecayCoverage refuses asset 2: two of four observations, and inactive at the last.
    @test isnan(vec(mean(SimpleExpectedReturns(; cvg = strict), X; active_mask = amsk))[2])
    # A subtype that admits everything reaches the answer instead.
    @test isfinite(vec(mean(SimpleExpectedReturns(;
                                                  cvg = CoveragePolicy(;
                                                                       alg = AdmitEverything())),
                            X; active_mask = amsk))[2])
end
@testset "The third and fourth order read each cell's own observations" begin
    X = [1.0 2.0 NaN; 3.0 4.0 5.0; 5.0 NaN 7.0; 7.0 8.0 9.0; 2.0 6.0 4.0; 4.0 1.0 8.0]
    cvg = CoveragePolicy()
    N = size(X, 2)
    Y = comoment_deviations(X, identity)
    sk, V = coskewness(Coskewness(; cvg = cvg), X)
    # The repair moves a cokurtosis away from its own fit, so the parity oracle reads the raw
    # one. The repair has its own test, and the third order is never processed.
    kt = cokurtosis(Cokurtosis(; mp = MatrixProcessing(; pdm = nothing), cvg = cvg), X)
    for a in 1:N, i in 1:N, j in 1:N
        @test sk[a, (i - 1) * N + j] ≈ comoment_cell(Y, (a, i, j))
    end
    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
        @test kt[(i - 1) * N + j, (k - 1) * N + l] ≈ comoment_cell(Y, (i, j, k, l))
    end
    # The tensor's per-asset diagonal is the asset's own third and fourth central moment.
    for i in 1:N
        @test sk[i, (i - 1) * N + i] ≈ comoment_cell(Y, (i, i, i))
        @test kt[(i - 1) * N + i, (i - 1) * N + i] ≈ comoment_cell(Y, (i, i, i, i))
    end
    @test size(V) == (N, N)
    @test all(isfinite, V)
end
@testset "The semi arms of the third and fourth order take the policy" begin
    X = [1.0 2.0 NaN; 3.0 4.0 5.0; 5.0 NaN 7.0; 7.0 8.0 9.0; 2.0 6.0 4.0; 4.0 1.0 8.0]
    cvg = CoveragePolicy()
    N = size(X, 2)
    Y = comoment_deviations(X, x -> min(x, zero(x)))
    sk, _ = coskewness(Coskewness(; alg = SemiMoment(), cvg = cvg), X)
    kt = cokurtosis(Cokurtosis(; alg = SemiMoment(), mp = MatrixProcessing(; pdm = nothing),
                               cvg = cvg), X)
    for a in 1:N, i in 1:N, j in 1:N
        @test sk[a, (i - 1) * N + j] ≈ comoment_cell(Y, (a, i, j))
    end
    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
        @test kt[(i - 1) * N + j, (k - 1) * N + l] ≈ comoment_cell(Y, (i, j, k, l))
    end
    # The clip is taken about each asset's own available-case mean, so a centre fitted over the
    # whole window is refused, at both orders and under both moment algorithms.
    for alg in (FullMoment(), SemiMoment())
        @test_throws ArgumentError coskewness(Coskewness(; alg = alg, cvg = cvg), X;
                                              mean = zeros(1, N))
        @test_throws ArgumentError cokurtosis(Cokurtosis(; alg = alg, cvg = cvg), X;
                                              mean = zeros(1, N))
    end
end
@testset "A refused asset is framed across its rows and its pair columns" begin
    X = [1.0 2.0 3.0; 3.0 4.0 5.0; 5.0 6.0 7.0; 7.0 8.0 9.0; 2.0 6.0 NaN; 4.0 1.0 NaN]
    amsk = trues(size(X)...)
    amsk[5:6, 3] .= false
    N = size(X, 2)
    # Asset 3 covers four of six observations and is inactive at the last, so `DecayCoverage`
    # refuses it above that share.
    cvg = CoveragePolicy(; min_coverage = 0.9)
    sk, V = coskewness(Coskewness(; cvg = cvg), X; active_mask = amsk)
    kt = cokurtosis(Cokurtosis(; cvg = cvg), X; active_mask = amsk)
    @test all(isnan, view(sk, 3, :))
    @test all(isfinite, view(sk, 1:2, [(i - 1) * N + j for i in 1:2 for j in 1:2]))
    for i in 1:N, j in 1:N
        touches = i == 3 || j == 3
        @test all(isnan, view(sk, :, (i - 1) * N + j)) == touches
        @test all(isnan, view(kt, (i - 1) * N + j, :)) == touches
        @test all(isnan, view(kt, :, (i - 1) * N + j)) == touches
    end
    # The reduction frames the refused asset too, and repairs the block around it.
    @test all(isnan, view(V, 3, :))
    @test all(isnan, view(V, :, 3))
    @test all(isfinite, view(V, 1:2, 1:2))
end
@testset "An empty cell inside a complete block is refused by name" begin
    # Every pair shares an observation and no observation carries all three assets, so the
    # per-asset diagonals are finite and the triple is empty: the block is the whole tensor.
    X = [1.0 2.0 NaN; 3.0 4.0 NaN; 5.0 NaN 7.0; 7.0 NaN 9.0; NaN 6.0 4.0; NaN 1.0 8.0]
    cvg = CoveragePolicy()
    sk_only = PortfolioOptimisers.coverage_divide([1.0], [1], false, nothing)
    @test isfinite(only(sk_only))
    @test_throws PortfolioOptimisers.IsNonFiniteError coskewness(Coskewness(; cvg = cvg), X)
    # The same law on a covariance frame: an empty pair among fully admitted assets reaches the
    # named refusal rather than LAPACK, which is what a complete diagonal used to short-circuit.
    Xp = [1.0 NaN 3.0; NaN 2.0 5.0; 3.0 NaN 7.0; NaN 4.0 9.0]
    sigma = cov(Covariance(; cvg = cvg), Xp)
    @test all(isfinite, LinearAlgebra.diag(sigma))
    @test isnan(sigma[1, 2])
    @test_throws PortfolioOptimisers.IsNonFiniteError PortfolioOptimisers.matrix_processing_block!(MatrixProcessing(),
                                                                                                   copy(sigma),
                                                                                                   Xp)
    # A matrix with no finite diagonal carries no block, so it is the plain repair's and meets
    # the refusal its caller already met rather than the block rule's.
    @test_throws ArgumentError PortfolioOptimisers.matrix_processing_block!(MatrixProcessing(),
                                                                            fill(NaN, 2, 2),
                                                                            zeros(4, 2))
    @test all(isnan,
              PortfolioOptimisers.matrix_processing_block!(nothing, fill(NaN, 2, 2),
                                                           zeros(4, 2)))
end
@testset "The Investable Mask reads every order the carrier holds" begin
    X = randn(StableRNGs.StableRNG(123), 60, 3) ./ 100
    X[1:30, 3] .= NaN
    amsk = trues(size(X)...)
    amsk[1:30, 3] .= false
    pnl = coverage_panel(amsk)
    cvg = CoveragePolicy()
    lo = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                        ce = Covariance(; cvg = cvg), fill_limit = 1)
    # A policy on the low order alone: the higher orders reduce to the Coverage Universe, the
    # mask narrows back to it, and the fit says so once.
    mixed = @test_logs (:warn,) match_mode = :any prior(HighOrderPriorEstimator(; pe = lo),
                                                        X, nothing, pnl)
    @test PortfolioOptimisers.investable_mask(mixed) == BitVector([1, 1, 0])
    @test all(isnan, view(mixed.sk, 3, :))
    # The matched configuration narrows nothing and says nothing.
    matched = @test_logs prior(HighOrderPriorEstimator(; pe = lo,
                                                       ske = Coskewness(; cvg = cvg),
                                                       kte = Cokurtosis(; cvg = cvg)), X,
                               nothing, pnl)
    @test isnothing(PortfolioOptimisers.investable_mask(matched))
    @test all(isfinite, matched.sk)
    @test all(isfinite, matched.kt)
    # The reduction the optimiser takes at its entry runs on the narrowed universe.
    @test all(isfinite, PortfolioOptimisers.port_opt_view(mixed, [1, 2]).sk)
end
@testset "The online form of the third and fourth order is a buffer refit" begin
    X = randn(StableRNGs.StableRNG(97), 40, 3) ./ 100
    X[1:12, 3] .= NaN
    cvg = CoveragePolicy()
    ske = PortfolioOptimisers.update_online_estimator(Online(Coskewness(; cvg = cvg)))
    for t in axes(X, 1)
        ske = partial_fit!(ske, view(X, t, :))
    end
    sk_o, V_o = coskewness(ske)
    sk_b, V_b = coskewness(Coskewness(; cvg = cvg), X)
    @test isequal(sk_o, sk_b)
    @test isequal(V_o, V_b)
    kte = PortfolioOptimisers.update_online_estimator(Online(Cokurtosis(; cvg = cvg)))
    kte = partial_fit!(kte, X)
    @test isequal(cokurtosis(kte), cokurtosis(Cokurtosis(; cvg = cvg), X))
    # The cap is the memory knob of the carry buffer, so a capped wrapper reads its last rows.
    capped = PortfolioOptimisers.update_online_estimator(Online(Coskewness(; cvg = cvg);
                                                                max_history = 10))
    capped = partial_fit!(capped, X)
    @test isequal(first(coskewness(capped)),
                  first(coskewness(Coskewness(; cvg = cvg), X[(end - 9):end, :])))
end
@testset "A policy prints only where it is set" begin
    @test PortfolioOptimisers.show_fields(Coskewness()) == (:me, :mp, :alg, :w, :cache)
    @test PortfolioOptimisers.show_fields(Cokurtosis()) == (:me, :mp, :alg, :w, :cache)
    @test PortfolioOptimisers.show_fields(Coskewness(; cvg = CoveragePolicy())) ==
          (:me, :mp, :alg, :w, :cvg, :cache)
    @test PortfolioOptimisers.show_fields(Cokurtosis(; cvg = CoveragePolicy())) ==
          (:me, :mp, :alg, :w, :cvg, :cache)
end
