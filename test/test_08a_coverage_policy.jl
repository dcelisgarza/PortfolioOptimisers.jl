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
@testset "A pair with no shared observation is NaN with no peel, and the assets around it are not" begin
    X = [1.0 NaN; NaN 2.0; 3.0 NaN; NaN 4.0]
    cvg = CoveragePolicy(; peel = nothing)
    sigma = cov(Covariance(; cvg = cvg), X)
    @test isfinite(sigma[1, 1])
    @test isfinite(sigma[2, 2])
    @test isnan(sigma[1, 2])
    @test isnan(sigma[2, 1])
    # The default peel removes asset 1, the smallest index of the tie, and keeps asset 2.
    peeled = @test_logs (:warn, r"indices \[1\]") cov(Covariance(; cvg = CoveragePolicy()),
                                                      X)
    @test all(isnan, peeled[1, :])
    @test peeled[2, 2] == sigma[2, 2]
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
    Xo, msk, mu, active, stale = PortfolioOptimisers.coverage_valid_block(X, amsk,
                                                                          DecayCoverage();
                                                                          dims = 1)
    @test Xo === X
    @test msk == BitMatrix([true false; true true; false false])
    @test mu ≈ [1.5, 4.0]
    @test active == BitVector([true, false])
    @test stale == [1, 1]
    # No mask means every asset is active, so nothing is ever newly inactive.
    _, msk2, _, active2, _ = PortfolioOptimisers.coverage_valid_block(X, nothing,
                                                                      DecayCoverage();
                                                                      dims = 1)
    @test msk2 == isfinite.(X)
    @test all(active2)
    @test_throws DimensionMismatch PortfolioOptimisers.coverage_valid_block(X, trues(3, 3),
                                                                            DecayCoverage();
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
    # `peel = nothing` keeps the pair, which the default peel would remove at admission.
    Xp = [1.0 NaN 3.0; NaN 2.0 5.0; 3.0 NaN 7.0; NaN 4.0 9.0]
    sigma = cov(Covariance(; cvg = CoveragePolicy(; peel = nothing)), Xp)
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
@testset "ResetCoverage reaches the two-pass arms" begin
    T, N = 14, 3
    X = Float64[sin(t * j) + 0.1 * t * j for t in 1:T, j in 1:N]
    amsk = trues(T, N)
    # Asset 2 delists at row 6 and lists again at row 9.
    amsk[6:8, 2] .= false
    X[6:8, 2] .= NaN
    dec = CoveragePolicy(; alg = DecayCoverage())
    res = CoveragePolicy(; alg = ResetCoverage())
    # The oracle: the reset is DecayCoverage over a sample whose history before the delisting
    # was never observed.
    Xr = copy(X)
    Xr[1:5, 2] .= NaN
    semi = Covariance(; alg = SemiMoment(), cvg = res)
    @test isequal(cov(semi, X; active_mask = amsk),
                  cov(Covariance(; alg = SemiMoment(), cvg = dec), Xr; active_mask = amsk))
    @test !isequal(cov(semi, X; active_mask = amsk),
                   cov(Covariance(; alg = SemiMoment(), cvg = dec), X; active_mask = amsk))
    @test isequal(first(coskewness(Coskewness(; cvg = res), X; active_mask = amsk)),
                  first(coskewness(Coskewness(; cvg = dec), Xr; active_mask = amsk)))
    @test !isequal(first(coskewness(Coskewness(; cvg = res), X; active_mask = amsk)),
                   first(coskewness(Coskewness(; cvg = dec), X; active_mask = amsk)))
    mp = MatrixProcessing(; pdm = nothing)
    @test isequal(cokurtosis(Cokurtosis(; mp = mp, cvg = res), X; active_mask = amsk),
                  cokurtosis(Cokurtosis(; mp = mp, cvg = dec), Xr; active_mask = amsk))
    # The folding arm meets the same oracle, so the two arms read one universe.
    @test cov(Covariance(; cvg = res), X; active_mask = amsk) ==
          cov(Covariance(; cvg = dec), Xr; active_mask = amsk)
    # The block and the fold agree on the bookkeeping, over two delistings of one asset and a
    # delisting that never relists.
    am2 = trues(T, N)
    am2[3:4, 1] .= false
    am2[9:10, 1] .= false
    am2[12:14, 3] .= false
    X2 = copy(X)
    X2[.!am2] .= NaN
    X2[7, 2] = NaN
    fold = partial_fit!(SimpleVariance(; cvg = res), X2; active_mask = am2).cache
    _, F, mu, active, stale = PortfolioOptimisers.coverage_valid_block(X2, am2,
                                                                       ResetCoverage())
    @test fold.cvg.nu == vec(sum(F; dims = 1)) == [4, 11, 0]
    @test fold.cvg.stale == stale == [0, 0, 3]
    @test fold.cvg.active == active
    @test isapprox(fold.mu, mu; rtol = 1e-14)
    # With no active mask no asset goes inactive, so the reset is DecayCoverage.
    Xh = copy(X)
    Xh[6:8, 2] .= NaN
    @test isequal(cov(Covariance(; cvg = res), Xh), cov(Covariance(; cvg = dec), Xh))
    @test isequal(cov(semi, Xh), cov(Covariance(; alg = SemiMoment(), cvg = dec), Xh))
    # A caller's own algorithm keeps the history in a two-pass arm.
    own = CoveragePolicy(; alg = AdmitEverything())
    @test isequal(first(coskewness(Coskewness(; cvg = own), X; active_mask = amsk)),
                  first(coskewness(Coskewness(; cvg = dec), X; active_mask = amsk)))
end
@testset "The share of a relisted asset and the hold of an expiring one" begin
    # The reset keeps the number of observations folded, so a relisted asset's share is its
    # count since the relisting over the whole window.
    T = 12
    X = Float64[t + 10j for t in 1:T, j in 1:2]
    amsk = trues(T, 2)
    amsk[4:6, 2] .= false
    X[4:6, 2] .= NaN
    strict_reset = CoveragePolicy(; min_coverage = 0.6, alg = ResetCoverage())
    strict_decay = CoveragePolicy(; min_coverage = 0.6, alg = DecayCoverage())
    @test isnan(vec(mean(SimpleExpectedReturns(; cvg = strict_reset), X;
                         active_mask = amsk))[2])
    @test vec(mean(SimpleExpectedReturns(; cvg = strict_decay), X; active_mask = amsk))[2] ≈
          mean(X[[1:3; 7:12], 2])
    # A holiday before a delisting raises the staleness, so it shortens the hold.
    Xe = [1.0 1; 2 2; 3 3; 4 4; 5 NaN; 6 NaN; 7 NaN; 8 NaN]
    ame = trues(8, 2)
    ame[7:8, 2] .= false
    exp3 = CoveragePolicy(; alg = ExpireCoverage(; after = 3))
    @test isnan(vec(mean(SimpleExpectedReturns(; cvg = exp3), Xe; active_mask = ame))[2])
    Xe[5:6, 2] .= [5.0, 6.0]
    @test vec(mean(SimpleExpectedReturns(; cvg = exp3), Xe; active_mask = ame))[2] ≈ 3.5
    # ExpireCoverage keeps the history as DecayCoverage does.
    a = partial_fit!(SimpleVariance(; cvg = exp3), X; active_mask = amsk).cache
    b = partial_fit!(SimpleVariance(; cvg = CoveragePolicy()), X; active_mask = amsk).cache
    @test a.cvg.nu == b.cvg.nu == [12, 9]
    @test a.mu == b.mu
end
@testset "The read-out helpers at their edges" begin
    # A state that folded nothing gives every asset a share of zero.
    c0 = PortfolioOptimisers.coverage_counts_seed(CoveragePolicy(), nothing, 3, Float64,
                                                  false)
    @test isnothing(PortfolioOptimisers.coverage_admission(CoveragePolicy(), c0, 0))
    @test PortfolioOptimisers.coverage_admission(CoveragePolicy(; min_coverage = 0.1), c0,
                                                 0) == falses(3)
    # An exact element type cannot hold the sentinel.
    @test_throws InexactError PortfolioOptimisers.coverage_divide([1//1 2//1], [2 0], false,
                                                                  nothing)
    @test_throws InexactError PortfolioOptimisers.coverage_frame([1 // 2, 2 // 3], [1, 0],
                                                                 nothing)
    @test PortfolioOptimisers.coverage_divide([1//1 2//1], [2 3], false, nothing) ==
          [1//2 2//3]
    @test eltype(PortfolioOptimisers.coverage_divide(Float32[1 2], [2 1], true, nothing)) ===
          Float32
    # The pair columns of a coskewness take the conjunction of the two asset flags.
    cmsk = BitVector([1, 0, 1])
    V = zeros(3, 9)
    PortfolioOptimisers.coverage_refuse_comoment!(V, cmsk, Val(:sk))
    for i in 1:3, j in 1:3
        @test isnan(V[1, (i - 1) * 3 + j]) == !(cmsk[i] && cmsk[j])
    end
    @test all(isnan, V[2, :])
end
@testset "The default floor fills every column that has a variance" begin
    X = randn(StableRNG(3), 20, 3) ./ 100
    cvg = CoveragePolicy()
    pe = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                        ce = Covariance(; cvg = cvg))
    # One observation: a NaN variance, so the Investable Mask drops the asset and nothing is
    # filled.
    X1 = copy(X)
    X1[1:19, 3] .= NaN
    r1 = @test_logs prior(pe, X1)
    @test PortfolioOptimisers.investable_mask(r1) == BitVector([1, 1, 0])
    @test isnan(r1.sigma[3, 3])
    # Two observations: a variance, so the prior fills the column in silence.
    X2 = copy(X)
    X2[1:18, 3] .= NaN
    r2 = @test_logs prior(pe, X2)
    @test isnothing(PortfolioOptimisers.investable_mask(r2))
    @test isfinite(r2.sigma[3, 3])
end

#=
The peel family: the rule that removes the assets of an undetermined pair at admission, so that an
available-case covariance carries no `NaN` inside its finite block. Decided on #1416 (row R86),
built by #1511.

An undetermined pair is a pair of admitted assets that each have a variance but share too few
observations for a covariance. The pairs make a graph, and a set of assets removes every pair
exactly when it is a vertex cover of that graph. `GreedyPeel` is the oracle's rule, max degree
first and the smallest index on a tie. `MinimalPeel` is the minimum cover, and it keeps the
greedy cover whenever that cover is minimum. `peel = nothing` skips the peel, and keeps the `NaN`
for the matrix repair to refuse.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# A graph from a list of edges.
function peel_graph(n, edges)
    U = falses(n, n)
    for (i, j) in edges
        U[i, j] = U[j, i] = true
    end
    return U
end
function peel_is_cover(U, C)
    return all(!U[i, j] || i in C || j in C for i in axes(U, 1), j in axes(U, 2))
end
# The minimum vertex cover by enumeration, and the lexicographically smallest one of that size.
function peel_brute(U)
    n = size(U, 1)
    for k in 0:n
        cs = [findall(b -> (m >> (b - 1)) & 1 == 1, 1:n)
              for m in 0:(2 ^ n - 1) if count_ones(m) == k]
        cs = filter(c -> peel_is_cover(U, c), cs)
        if !isempty(cs)
            return minimum(cs)
        end
    end
end
# The returns of the parity cases. The values come from closed formulas and the gaps from fixed
# windows, so the generator of the stored oracle output and this test build the same matrix, and
# the stored `Input` pins it.
#
# `Counterexample`: five blocks of eight rows. X (asset 2) is valid in block 0 alone, A1 to A3
# (assets 3 to 5) skip block 0 and their own block, and L1 to L3 (assets 6 to 8) are valid in block
# 0 and their own block. The graph is X-(A1, A2, A3) and Ai-Li, and assets 1 and 9 are complete.
# `Windows`: each asset is valid on a window of rows, so a pair is undetermined when its two
# windows share fewer than two rows.
function peel_parity_returns(case::Symbol)
    if case == :Counterexample
        T, N = 40, 9
        blocks = Dict(2 => [0], 3 => [2, 3, 4], 4 => [1, 3, 4], 5 => [1, 2, 4], 6 => [0, 1],
                      7 => [0, 2], 8 => [0, 3])
        keep = [haskey(blocks, j) ? fld(t - 1, 8) in blocks[j] : true
                for t in 1:T, j in 1:N]
    else
        T, N = 30, 10
        windows = [1:30, 1:8, 6:14, 13:22, 20:30, 1:12, 9:20, 18:30, 1:30, 14:15]
        keep = [t in windows[j] for t in 1:T, j in 1:N]
    end
    X = [(sin(0.7 * t + 1.3 * j) * (1 + j / 10) + cos(0.3 * t * j) / 2) / 100
         for t in 1:T, j in 1:N]
    X[.!keep] .= NaN
    return X
end
function peel_covariance(peel; kwargs...)
    return Covariance(; cvg = CoveragePolicy(; peel = peel), kwargs...)
end
# Assets 3 and 4 have disjoint histories, and every other pair shares rows.
function peel_disjoint_returns()
    X = [(sin(0.9 * t + 2.1 * j) + cos(0.2 * t * j) / 3) / 100 for t in 1:40, j in 1:4]
    X[21:end, 3] .= NaN
    X[1:20, 4] .= NaN
    return X
end
struct PeelEveryEnd <: PortfolioOptimisers.AbstractPeel end
function PortfolioOptimisers.peel_assets(::PeelEveryEnd, U::AbstractMatrix{Bool})
    return findall(vec(any(U; dims = 2)))
end
# A rule that peels the first end of the first edge alone, so it can leave an edge.
struct PeelFirstEdge <: PortfolioOptimisers.AbstractPeel end
function PortfolioOptimisers.peel_assets(::PeelFirstEdge, U::AbstractMatrix{Bool})
    i = findfirst(vec(any(U; dims = 2)))
    return isnothing(i) ? Int[] : [i]
end
@testset "Peel family" begin
    @testset "Graph rules" begin
        # The star X-(A1, A2, A3) whose leaves each carry one more edge Ai-Li. Greedy removes X
        # first, then one end of each pendant edge: four assets. Three suffice.
        U = peel_graph(7, ((1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7)))
        @test PortfolioOptimisers.peel_assets(GreedyPeel(), U) == [1, 2, 3, 4]
        @test PortfolioOptimisers.peel_assets(MinimalPeel(), U) == [2, 3, 4]
        @test isempty(PortfolioOptimisers.peel_assets(nothing, U))
        # A path: the greedy cover is minimum, so the minimal rule keeps it.
        P = peel_graph(5, ((1, 2), (2, 3), (3, 4), (4, 5)))
        @test PortfolioOptimisers.peel_assets(GreedyPeel(), P) == [2, 4]
        @test PortfolioOptimisers.peel_assets(MinimalPeel(), P) == [2, 4]
        # No edge, nothing to remove.
        @test isempty(PortfolioOptimisers.peel_assets(MinimalPeel(), falses(3, 3)))
        @test isempty(PortfolioOptimisers.peel_assets(GreedyPeel(), falses(3, 3)))
        # A cycle of five needs three, a cycle of four needs two, and they share no vertex.
        C = peel_graph(9,
                       ((1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (6, 7), (7, 8), (8, 9),
                        (9, 6)))
        @test PortfolioOptimisers.peel_cycle_cover(C, trues(9)) == 5
        @test PortfolioOptimisers.peel_cover_within(C, trues(9), 5)
        @test !PortfolioOptimisers.peel_cover_within(C, trues(9), 4)
        @test !PortfolioOptimisers.peel_cover_within(C, trues(9), -1)
        @test PortfolioOptimisers.peel_max_degree(C, falses(9)) == (0, 0)
        # Against enumeration: the minimal rule is a minimum cover, it is the greedy cover when
        # that cover is minimum, and it is the lexicographically smallest minimum cover otherwise.
        rng = StableRNG(1511)
        larger = 0
        for _ in 1:600
            n = rand(rng, 1:10)
            p = rand(rng) * 0.6
            G = falses(n, n)
            for i in 1:n, j in (i + 1):n
                if rand(rng) < p
                    G[i, j] = G[j, i] = true
                end
            end
            g = PortfolioOptimisers.peel_assets(GreedyPeel(), G)
            m = PortfolioOptimisers.peel_assets(MinimalPeel(), G)
            b = peel_brute(G)
            @test peel_is_cover(G, g)
            if length(g) == length(b)
                @test m == g
            else
                larger += 1
                @test m == b
            end
        end
        # The sample must reach the lexicographic arm, or the last test above checks nothing.
        @test larger > 0
        # The complete graph is the worst case of the cover: every rule keeps one asset, the
        # last (#1771). R7 of #1630 measured 0.13 ms for 32 assets.
        for n in (3, 8, 32)
            K = trues(n, n)
            K[LinearAlgebra.diagind(K)] .= false
            @test PortfolioOptimisers.peel_assets(MinimalPeel(), K) == 1:(n - 1)
            @test PortfolioOptimisers.peel_assets(GreedyPeel(), K) == 1:(n - 1)
        end
    end
    @testset "Policy and interface" begin
        @test CoveragePolicy().peel === MinimalPeel()
        @test CoveragePolicy(; peel = GreedyPeel()).peel === GreedyPeel()
        # `nothing` skips the peel, as it turns other procedures off (#1867).
        @test isnothing(CoveragePolicy(; peel = nothing).peel)
        @test all(T -> T <: PortfolioOptimisers.AbstractPeel, (MinimalPeel, GreedyPeel))
        @test all(s -> Base.isexported(PortfolioOptimisers, s), (:MinimalPeel, :GreedyPeel))
        @test !isdefined(PortfolioOptimisers, :NoPeel)
        @test Base.ispublic(PortfolioOptimisers, :AbstractPeel)
        @test Base.ispublic(PortfolioOptimisers, :peel_assets)
        # A peel of the caller's own reaches the covariance through the interface.
        s = @test_logs (:warn, r"indices \[3, 4\]") cov(peel_covariance(PeelEveryEnd()),
                                                        peel_disjoint_returns())
        @test all(isnan, s[3, :]) && all(isnan, s[4, :]) && all(isfinite, s[1:2, 1:2])
        # A rule whose set leaves an edge leaves a `NaN` inside the block, and the matrix
        # repair and the prior refuse it, as the interface states (#1771). The undetermined
        # pairs make the cycle 3-4-5-6, and the rule peels asset 3 alone.
        Y = hcat(peel_disjoint_returns(), peel_disjoint_returns()[:, 3:4])
        s = @test_logs (:warn, r"indices \[3\]") cov(peel_covariance(PeelFirstEdge()), Y)
        @test all(isnan, s[3, :]) && isnan(s[4, 5]) && isnan(s[5, 6])
        @test all(isfinite, LinearAlgebra.diag(s)[[1, 2, 4, 5, 6]])
        @test_throws PortfolioOptimisers.IsNonFiniteError PortfolioOptimisers.matrix_processing_block!(MatrixProcessing(),
                                                                                                       copy(s),
                                                                                                       Y)
        pe = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = CoveragePolicy()),
                            ce = peel_covariance(PeelFirstEdge()))
        @test_throws PortfolioOptimisers.IsNonFiniteError prior(pe, Y)
    end
    @testset "Covariance peels at admission" begin
        X = peel_disjoint_returns()
        s0 = cov(peel_covariance(nothing), X)
        # Today's available-case answer: a NaN cell between two assets with a variance.
        @test isnan(s0[3, 4]) && isnan(s0[4, 3])
        @test all(isfinite, LinearAlgebra.diag(s0))
        s1 = @test_logs (:warn, r"indices \[3\].*1 undetermined pair") cov(peel_covariance(MinimalPeel()),
                                                                           X)
        @test all(isnan, s1[3, :]) && all(isnan, s1[:, 3])
        keep = [1, 2, 4]
        @test s1[keep, keep] == s0[keep, keep]
        # The peel removes the asset, so the other cells are the fit on the panel without it.
        @test s1[keep, keep] == cov(peel_covariance(MinimalPeel()), X[:, keep])
        @test_throws ArgumentError cov(peel_covariance(MinimalPeel()), X; strict = true)
        @test_throws ArgumentError cor(peel_covariance(GreedyPeel()), X; strict = true)
        r1 = @test_logs (:warn,) cor(peel_covariance(GreedyPeel()), X)
        @test all(isnan, r1[3, :])
        @test r1[keep, keep] ≈ cor(peel_covariance(GreedyPeel()), X[:, keep])
        # The semi-covariance takes its admission from the same rule.
        sm = @test_logs (:warn, r"indices \[3\]") cov(peel_covariance(MinimalPeel();
                                                                      alg = SemiMoment()),
                                                      X)
        @test all(isnan, sm[3, :])
        @test sm[keep, keep] ==
              cov(peel_covariance(MinimalPeel(); alg = SemiMoment()), X[:, keep])
        @test_throws ArgumentError cov(peel_covariance(MinimalPeel(); alg = SemiMoment()),
                                       X; strict = true)
        # The fold reads the state through the same rule, and `strict` reaches it.
        ce = partial_fit!(peel_covariance(MinimalPeel()), X)
        sf = @test_logs (:warn,) cov(ce)
        @test isequal(sf, s1)
        # The fold peels each time it answers, so the peeled set can change between steps
        # (#1771). After rows 1 to 20, assets 3 and 4 share no row, and asset 5 has none. After
        # rows 21 to 40, 3 and 4 share rows, and 1 and 5 share none.
        Z = [(sin(0.9 * t + 2.1 * j) + cos(0.2 * t * j) / 3) / 100 for t in 1:40, j in 1:5]
        Z[21:end, 1] .= NaN
        Z[11:20, 3] .= NaN
        Z[1:10, 4] .= NaN
        Z[1:20, 5] .= NaN
        pc = peel_covariance(MinimalPeel())
        f1 = partial_fit!(pc, Z[1:20, :])
        z1 = @test_logs (:warn, r"indices \[3\]") cov(f1)
        f2 = partial_fit!(f1, Z[21:40, :])
        z2 = @test_logs (:warn, r"indices \[1\]") cov(f2)
        @test findall(isnan, LinearAlgebra.diag(z1)) == [3, 5]
        @test findall(isnan, LinearAlgebra.diag(z2)) == [1]
        @test isequal(z1, @test_logs (:warn,) cov(pc, Z[1:20, :]))
        @test isapprox(z2, (@test_logs (:warn,) cov(pc, Z)); rtol = 1e-12, nans = true)
        @test_throws ArgumentError cov(ce; strict = true)
        @test_throws ArgumentError cor(ce; strict = true)
        # An estimator wrapped in `Online` reads its Sample Buffer through the batch verb, and
        # `strict` reaches that verb too.
        ow = partial_fit!(PortfolioOptimisers.update_online_estimator(Online(peel_covariance(MinimalPeel()))),
                          X)
        @test isequal((@test_logs (:warn,) cov(ow)), s1)
        @test_throws ArgumentError cov(ow; strict = true)
        @test_throws ArgumentError cor(ow; strict = true)
        # The warning names `nothing` as the way to keep the cells, not the removed `NoPeel`
        # (#1867), and an unpeeled matrix still meets the refusal of the matrix repair.
        @test_logs (:warn, r"set `peel = nothing`") cov(peel_covariance(MinimalPeel()), X)
        @test_throws PortfolioOptimisers.IsNonFiniteError PortfolioOptimisers.matrix_processing_block!(MatrixProcessing(),
                                                                                                       copy(s0),
                                                                                                       X)
        # The refusal names the peel, and a coverage floor that rises, never one that falls: a
        # lower floor admits more assets and cannot remove a pair.
        @test_throws "the default `MinimalPeel()`" PortfolioOptimisers.matrix_processing_block!(MatrixProcessing(),
                                                                                                copy(s0),
                                                                                                X)
        @test_throws "raise `min_coverage`" PortfolioOptimisers.matrix_processing_block!(MatrixProcessing(),
                                                                                         copy(s0),
                                                                                         X)
        # A pair that shares exactly one row is undetermined under the Bessel correction alone.
        Y = peel_disjoint_returns()
        Y[20, 4] = 0.003
        @test_logs (:warn, r"indices \[3\]") cov(peel_covariance(MinimalPeel()), Y)
        su = @test_logs cov(peel_covariance(MinimalPeel();
                                            ce = StatsBase.SimpleCovariance(;
                                                                            corrected = false)),
                            Y)
        @test all(isfinite, su)
    end
    @testset "Peel beside an asset that admission refuses" begin
        # Asset 5 holds two rows, under the floor, so admission refuses it before the peel. The
        # peel then removes asset 3 from the mask that admission already cut.
        X = hcat(peel_disjoint_returns(), fill(NaN, 40))
        X[39:40, 5] .= [0.01, -0.02]
        cvg = CoveragePolicy(; min_coverage = 0.1)
        s = @test_logs (:warn, r"indices \[3\]") cov(Covariance(; cvg = cvg), X)
        @test all(isnan, s[3, :]) && all(isnan, s[5, :])
        @test all(isfinite, s[[1, 2, 4], [1, 2, 4]])
        # With no undetermined pair, the policy peels nothing and says nothing.
        @test all(isfinite, @test_logs cov(Covariance(; cvg = CoveragePolicy()), X[:, 1:3]))
    end
    @testset "Prior reads one universe" begin
        X = peel_disjoint_returns()
        function pe(peel; kwargs...)
            return EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = CoveragePolicy()),
                                  ce = Covariance(; cvg = CoveragePolicy(; peel = peel)),
                                  kwargs...)
        end
        pr = @test_logs (:warn, r"indices \[3\]") match_mode = :any prior(pe(MinimalPeel()),
                                                                          X)
        @test PortfolioOptimisers.investable_mask(pr) == [true, true, false, true]
        keep = [1, 2, 4]
        pr0 = prior(pe(MinimalPeel()), X[:, keep])
        @test pr.sigma[keep, keep] == pr0.sigma
        @test pr.mu[keep] == pr0.mu
        @test_throws "peeled the assets at indices [3]" prior(pe(MinimalPeel()), X;
                                                              strict = true)
        @test_throws PortfolioOptimisers.IsNonFiniteError prior(pe(nothing), X)
        # A log-return prior passes `strict` to its covariance too.
        @test_throws "peeled the assets at indices [3]" prior(pe(MinimalPeel();
                                                                 horizon = 1), X;
                                                              strict = true)
        # The folded prior reads the same universe, and `strict` reaches its covariance.
        pf = partial_fit!(pe(MinimalPeel()), X)
        prf = @test_logs (:warn, r"indices \[3\]") match_mode = :any prior(pf)
        @test isequal(prf.sigma, pr.sigma)
        @test_throws "peeled the assets at indices [3]" prior(pf; strict = true)
        @test_throws "peeled the assets at indices [3]" prior(partial_fit!(pe(MinimalPeel();
                                                                              horizon = 1),
                                                                           X);
                                                              strict = true)
    end
    @testset "Parity: the oracle's greedy peel" begin
        # The oracle's covariance setter, with no nearest repair, peels the available-case
        # covariance of each case. `Input` pins the matrix it read, and `GreedyPeel` reaches
        # the matrix it wrote.
        for case in ("Counterexample", "Windows")
            X = peel_parity_returns(Symbol(case))
            s0 = cov(peel_covariance(nothing), X)
            @test parity_compare(s0, parity_load("peel", case, "Input");
                                 name = "peel $(case) Input").ok
            sg = @test_logs (:warn,) cov(peel_covariance(GreedyPeel()), X)
            @test parity_compare(sg, parity_load("peel", case, "Sigma");
                                 name = "peel $(case) Sigma").ok
        end
        # On the counterexample the minimal peel keeps X, and removes A1 to A3 alone.
        sm = @test_logs (:warn, r"indices \[3, 4, 5\]") cov(peel_covariance(MinimalPeel()),
                                                            peel_parity_returns(:Counterexample))
        @test findall(!isfinite, LinearAlgebra.diag(sm)) == [3, 4, 5]
        # On the windows the greedy peel is minimum, so the minimal peel is the oracle's.
        sw = @test_logs (:warn,) cov(peel_covariance(MinimalPeel()),
                                     peel_parity_returns(:Windows))
        @test parity_compare(sw, parity_load("peel", "Windows", "Sigma");
                             name = "peel Windows minimal").ok
    end
end
