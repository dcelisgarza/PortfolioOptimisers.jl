using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, StatsBase, Statistics

# The Coverage Universe of a fit, issue #853, under ADR 0117.
#
# A plain moment estimator refuses a non-finite sample. A prior reduces its returns matrix to
# the Coverage Universe, fits every plain estimator on that clean block, and expands every
# block of its result onto the FULL asset universe with a `NaN` frame outside it.
#
# The oracle is the same estimator on the sample with the non-covered columns removed by
# hand, entry for entry. That is what catches a leak of the gap into the other assets, which
# `GerberIQCovariance` had before this seam existed.

const PO = PortfolioOptimisers

rng = StableRNG(987654321)
T, N = 60, 4
X0 = randn(rng, T, N) ./ 100 .+ 0.0005
F0 = randn(StableRNG(123456789), T, 3) ./ 100
nx = ["a", "b", "c", "d"]

# The four ways an asset leaves the Coverage Universe of one window.
Xlist = copy(X0)                     # a listing: asset 4 lists at observation 31
Xlist[1:30, 4] .= NaN
Xhol = copy(X0)                      # a holiday: asset 2 misses one quote mid-window
Xhol[15, 2] = NaN
Xdel = copy(X0)                      # a delisting: asset 3 stops at observation 50
Xdel[50:end, 3] .= NaN

full_amsk = trues(T, N)
inactive_amsk = trues(T, N)          # an inactive row: asset 1 leaves the universe once
inactive_amsk[20, 1] = false

function make_panel(amsk, emsk = amsk)
    return AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(T, N))],
                      amsk = amsk, emsk = emsk)
end

pnl_full = make_panel(full_amsk)
pnl_inactive = make_panel(inactive_amsk)

@testset "Coverage Universe: finite and active at every row" begin
    # No panel, and a complete window: the `nothing` sentinel, never a mask of every `true`.
    @test isnothing(PO.coverage_mask(X0, nothing))
    @test isnothing(PO.coverage_mask(X0, pnl_full))
    # A static panel carries no mask, so its rule is finiteness alone.
    st = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(N))])
    @test isnothing(PO.coverage_mask(X0, st))
    @test PO.coverage_mask(Xlist, st) == BitVector([1, 1, 1, 0])

    # Each of the four ways out puts exactly its own asset in the frame.
    @test PO.coverage_mask(Xlist, nothing) == BitVector([1, 1, 1, 0])
    @test PO.coverage_mask(Xhol, nothing) == BitVector([1, 0, 1, 1])
    @test PO.coverage_mask(Xdel, nothing) == BitVector([1, 1, 0, 1])
    @test PO.coverage_mask(X0, pnl_inactive) == BitVector([0, 1, 1, 1])

    # One inactive row is enough, and it is enough on its own: the return is finite there.
    @test all(isfinite, X0)

    # The estimation mask is not read. It names the assets that enter a cross-sectional
    # estimate, and a moment is not one.
    emsk = falses(T, N)
    @test isnothing(PO.coverage_mask(X0, make_panel(full_amsk, emsk)))

    # `dims = 2` reads the same window through the transposed orientation.
    @test PO.coverage_mask(transpose(Xlist), nothing; dims = 2) == BitVector([1, 1, 1, 0])

    # No asset in the Coverage Universe has no answer to give.
    Xnone = copy(X0)
    Xnone[1, :] .= NaN
    @test_throws IsEmptyError PO.coverage_mask(Xnone, nothing)
    @test_throws IsEmptyError PO.coverage_mask(X0, make_panel(falses(T, N)))

    # The panel and the returns matrix describe the same assets, and the asset axis is the
    # only axis that must agree.
    @test_throws DimensionMismatch PO.coverage_mask(X0[:, 1:2], pnl_full)

    # A nested prior may drop rows, so the two observation axes can differ. Each scan reads
    # its own rows, and neither pairs a row of one with a row of the other.
    @test isnothing(PO.coverage_mask(X0[2:end, :], pnl_full))
    @test PO.coverage_mask(X0[2:end, :], pnl_inactive) == BitVector([0, 1, 1, 1])
    @test PO.coverage_mask(Xlist[2:end, :], pnl_inactive) == BitVector([0, 1, 1, 0])
end

@testset "The reduction: the complete window allocates nothing" begin
    cmsk, Xc = PO.coverage_reduction(X0, nothing)
    @test isnothing(cmsk)
    @test Xc === X0
    cmsk, Xc = PO.coverage_reduction(Xlist, nothing)
    @test cmsk == BitVector([1, 1, 1, 0])
    @test Xc == Xlist[:, 1:3]
    # The reduction keeps the caller's orientation.
    cmsk, Xc = PO.coverage_reduction(transpose(Xlist), nothing; dims = 2)
    @test Xc == transpose(Xlist)[1:3, :]
end

@testset "The expansion: every shape writes back into a NaN frame" begin
    cmsk = BitVector([1, 1, 1, 0])
    # The `nothing` sentinel returns its argument untouched, at every arity.
    sigma = [1.0 2.0; 2.0 3.0]
    @test PO.expand_moment(sigma, nothing) === sigma
    @test PO.expand_moment(sigma, nothing, 1) === sigma
    @test PO.expand_moment(sigma, nothing, Val(:kt)) === sigma
    @test PO.expand_columns(sigma, nothing) === sigma
    @test PO.expand_rows(sigma, nothing) === sigma
    @test PO.expand_vector([1.0, 2.0], nothing) == [1.0, 2.0]

    # A covariance-like matrix.
    blk = reshape(collect(1.0:9.0), 3, 3)
    frame = PO.expand_moment(blk, cmsk)
    @test size(frame) == (4, 4)
    @test frame[1:3, 1:3] == blk
    @test all(isnan, frame[4, :])
    @test all(isnan, frame[:, 4])

    # A marginal, in both orientations.
    mu = [1.0 2.0 3.0]
    row = PO.expand_moment(mu, cmsk, 1)
    @test size(row) == (1, 4)
    @test row[1, 1:3] == [1.0, 2.0, 3.0]
    @test isnan(row[1, 4])
    col = PO.expand_moment(permutedims(mu), cmsk, 2)
    @test size(col) == (4, 1)
    @test col[1:3, 1] == [1.0, 2.0, 3.0]
    @test isnan(col[4, 1])

    # A marginal that comes back as a vector rather than a row.
    v = PO.expand_moment([1.0, 2.0, 3.0], cmsk, 1)
    @test v[1:3] == [1.0, 2.0, 3.0]
    @test isnan(v[4])

    # The two-matrix slice an estimator with a second per-asset panel needs.
    A = reshape(collect(1.0:8.0), 2, 4)
    @test PO.coverage_reduced_pair(A, A, nothing) == (A, A)
    @test PO.coverage_reduced_pair(A, A, cmsk) == (A[:, 1:3], A[:, 1:3])

    # An idiosyncratic covariance expands in either of the two shapes it takes.
    @test isnothing(PO.expand_idiosyncratic_covariance(nothing, cmsk))
    ev = PO.expand_idiosyncratic_covariance([1.0, 2.0, 3.0], cmsk)
    @test ev[1:3] == [1.0, 2.0, 3.0]
    em = PO.expand_idiosyncratic_covariance(blk, cmsk)
    @test em[1:3, 1:3] == blk

    # The pair index of a co-moment tensor, `a` outermost.
    @test PO.coverage_pair_index(cmsk) == [1, 2, 3, 5, 6, 7, 9, 10, 11]
    @test PO.coverage_pair_index(BitVector([0, 1])) == [4]
end

@testset "The implied volatility seam slices its surface too" begin
    keep = [1, 2, 3]
    iv = fill(0.25, T, N)
    ivpav = [1.1, 1.2, 1.3, 1.4]
    ce = ImpliedVolatility(; alg = ImpliedVolatilityPremium())
    sigma = Statistics.cov(ce, Xlist, pnl_full; iv = iv, ivpa = 1.2)
    @test all(isnan, sigma[4, :])
    @test sigma[keep, keep] ==
          Statistics.cov(ce, Xlist[:, keep]; iv = iv[:, keep], ivpa = 1.2)
    # A vector premium carries one entry per asset, so it takes the slice the surface takes.
    sigma2 = Statistics.cov(ce, Xlist, pnl_full; iv = iv, ivpa = ivpav)
    @test sigma2[keep, keep] ==
          Statistics.cov(ce, Xlist[:, keep]; iv = iv[:, keep], ivpa = ivpav[keep])
    rho = Statistics.cor(ce, Xlist, pnl_full; iv = iv, ivpa = 1.2)
    @test rho[keep, keep] ==
          Statistics.cor(ce, Xlist[:, keep]; iv = iv[:, keep], ivpa = 1.2)
    # The complete window takes the path it took before the seam existed.
    @test Statistics.cov(ce, X0, nothing; iv = iv, ivpa = 1.2) ==
          Statistics.cov(ce, X0; iv = iv, ivpa = 1.2)
    @test_throws IsNonFiniteError Statistics.cov(ce, Xlist; iv = iv, ivpa = 1.2)
    @test_throws IsNonFiniteError Statistics.cor(ce, Xlist; iv = iv, ivpa = 1.2)
    # A scalar premium applies to every asset, so it survives the reduction untouched.
    @test PO.coverage_reduced_ivpa(1.2, nothing) == 1.2
    @test PO.coverage_reduced_ivpa(1.2, BitVector([1, 1, 1, 0])) == 1.2
    @test isnothing(PO.coverage_reduced_ivpa(nothing, BitVector([1, 1, 1, 0])))
    @test PO.coverage_reduced_ivpa(ivpav, BitVector([1, 1, 1, 0])) == ivpav[keep]
end

# Every exported plain estimator, and the verb it answers. The oracle is the same estimator
# on the sample with the non-covered columns removed by hand.
plain_cov_estimators = Any[Covariance(), GerberCovariance(), SmythBrobyCovariance(),
                           DistanceCovariance(), LowerTailDependenceCovariance(),
                           KendallCovariance(), SpearmanCovariance(),
                           MutualInfoCovariance(), GerberIQCovariance(),
                           PortfolioOptimisersCovariance()]
plain_mean_estimators = Any[SimpleExpectedReturns(), MedianExpectedReturns()]

@testset "The oracle: the block equals the estimator on the complete columns" begin
    keep = [1, 2, 3]
    Xk = Xlist[:, keep]
    for ce in plain_cov_estimators
        sigma = Statistics.cov(ce, Xlist, pnl_full)
        @test size(sigma) == (N, N)
        @test all(isnan, sigma[4, :])
        @test all(isnan, sigma[:, 4])
        @test sigma[keep, keep] == Statistics.cov(ce, Xk)
        rho = Statistics.cor(ce, Xlist, pnl_full)
        @test rho[keep, keep] == Statistics.cor(ce, Xk)
        @test all(isnan, rho[4, :])
    end
    for me in plain_mean_estimators
        mu = Statistics.mean(me, Xlist, pnl_full)
        @test size(mu) == (1, N)
        @test mu[1, keep] == vec(Statistics.mean(me, Xk))
        @test isnan(mu[1, 4])
    end
    # The marginal verbs of a covariance estimator take the same seam.
    v = Statistics.var(Covariance(), Xlist, pnl_full)
    @test v[1, keep] == vec(Statistics.var(Covariance(), Xk))
    @test isnan(v[1, 4])
    s = Statistics.std(Covariance(), Xlist, pnl_full)
    @test s[1, keep] == vec(Statistics.std(Covariance(), Xk))
    @test isnan(s[1, 4])
    # A variance estimator answers `var` and `std` on its own.
    sv = Statistics.var(SimpleVariance(), Xlist, pnl_full)
    @test sv[1, keep] == vec(Statistics.var(SimpleVariance(), Xk))
    @test isnan(sv[1, 4])
end

@testset "The oracle: the co-moment tensors" begin
    keep = [1, 2, 3]
    Xk = Xlist[:, keep]
    sk, V = coskewness(Coskewness(), Xlist, pnl_full)
    skk, Vk = coskewness(Coskewness(), Xk)
    @test size(sk) == (N, N^2)
    @test size(V) == (N, N)
    @test sk[keep, PO.coverage_pair_index(BitVector([1, 1, 1, 0]))] == skk
    @test all(isnan, sk[4, :])
    @test V[keep, keep] == Vk
    @test all(isnan, V[4, :])

    kt = cokurtosis(Cokurtosis(), Xlist, pnl_full)
    ktk = cokurtosis(Cokurtosis(), Xk)
    idx = PO.coverage_pair_index(BitVector([1, 1, 1, 0]))
    @test size(kt) == (N^2, N^2)
    @test kt[idx, idx] == ktk
    @test all(isnan, kt[4, :])

    # A `nothing` estimator answers `nothing` through the panel seam too.
    @test coskewness(nothing, Xlist, pnl_full) == (nothing, nothing)
    @test isnothing(cokurtosis(nothing, Xlist, pnl_full))
end

@testset "A complete window takes the path it took before the seam existed" begin
    for ce in plain_cov_estimators
        @test Statistics.cov(ce, X0, nothing) == Statistics.cov(ce, X0)
        @test Statistics.cov(ce, X0, pnl_full) == Statistics.cov(ce, X0)
    end
    for me in plain_mean_estimators
        @test Statistics.mean(me, X0, nothing) == Statistics.mean(me, X0)
    end
end

@testset "A plain moment estimator refuses a non-finite sample" begin
    for ce in plain_cov_estimators
        @test_throws IsNonFiniteError Statistics.cov(ce, Xlist)
        @test_throws IsNonFiniteError Statistics.cor(ce, Xlist)
    end
    for me in plain_mean_estimators
        @test_throws IsNonFiniteError Statistics.mean(me, Xlist)
    end
    @test_throws IsNonFiniteError Statistics.var(SimpleVariance(), Xlist)
    @test_throws IsNonFiniteError Statistics.std(SimpleVariance(), Xlist)
    @test_throws IsNonFiniteError Statistics.var(SimpleVariance(), Xlist[:, 4])
    @test_throws IsNonFiniteError coskewness(Coskewness(), Xlist)
    @test_throws IsNonFiniteError cokurtosis(Cokurtosis(), Xlist)
    # The message names the two ways out, so the refusal tells the caller what to do.
    msg = try
        Statistics.cov(Covariance(), Xlist)
    catch e
        e.msg
    end
    @test occursin("prior", msg)
    @test occursin("mask-aware", msg)
end

@testset "A mask-aware estimator takes the whole window" begin
    ce = RegimeAdjustedExpWeightedVariance()
    # The override reads the panel's own masks rather than the Coverage Universe, so an asset
    # that is inactive at one row still carries a number.
    v = Statistics.var(ce, X0, pnl_inactive)
    @test length(v) == N
    @test isfinite(v[1])
    # No panel, and a static panel, both take the estimator's unmasked path.
    @test Statistics.var(ce, X0, nothing) == Statistics.var(ce, X0)
    @test PO.panel_moment_masks(nothing) == (nothing, nothing)
    @test PO.panel_moment_masks(pnl_full) == (pnl_full.amsk, pnl_full.emsk)
end

@testset "The block repair leaves the frame alone" begin
    keep = [1, 2, 3]
    sigma = Statistics.cov(Covariance(), Xlist, pnl_full)
    frame = copy(sigma)
    mp = MatrixProcessing()
    PO.matrix_processing_block!(mp, sigma, Xlist)
    @test all(isnan, sigma[4, :])
    @test all(isnan, sigma[:, 4])
    blk = copy(frame[keep, keep])
    matrix_processing!(mp, blk, Xlist[:, keep])
    @test sigma[keep, keep] == blk

    # A complete matrix takes the ordinary repair, in place.
    sigma2 = Statistics.cov(Covariance(), X0)
    sigma3 = copy(sigma2)
    PO.matrix_processing_block!(mp, sigma2, X0)
    matrix_processing!(mp, sigma3, X0)
    @test sigma2 == sigma3

    # An off-diagonal `NaN` inside the block is refused, and is never peeled away.
    bad = copy(frame)
    bad[1, 2] = NaN
    bad[2, 1] = NaN
    @test_throws IsNonFiniteError PO.matrix_processing_block!(mp, bad, Xlist)

    # A matrix with no finite diagonal entry has no block to repair, so it goes to the plain
    # repair and meets its refusal. This helper adds no failure face of its own, so a caller
    # whose covariance degenerated for an unrelated reason still meets the error it met
    # before the seam existed.
    none = fill(NaN, N, N)
    @test_throws ArgumentError PO.matrix_processing_block!(mp, none, Xlist)
    @test_throws ArgumentError matrix_processing!(mp, fill(NaN, N, N), Xlist)
end

@testset "The composite forwards the panel and repairs the block" begin
    keep = [1, 2, 3]
    ce = PortfolioOptimisersCovariance()
    sigma = Statistics.cov(ce, Xlist, pnl_full)
    @test all(isnan, sigma[4, :])
    @test sigma[keep, keep] == Statistics.cov(ce, Xlist[:, keep])
    # The bare `matrix_processing!` keeps its whole-matrix refusal.
    @test_throws ArgumentError matrix_processing!(MatrixProcessing(), fill(NaN, 2, 2),
                                                  X0[:, 1:2])
end

@testset "The three priors carry a full-universe result" begin
    rd = ReturnsResult(; nx = nx, X = Xlist, nf = ["f1", "f2", "f3"], F = F0)
    keep = [1, 2, 3]
    cmsk = BitVector([1, 1, 1, 0])

    pr = prior(EmpiricalPrior(), Xlist, nothing, pnl_full)
    @test length(pr.mu) == N
    @test size(pr.sigma) == (N, N)
    @test PO.investable_mask(pr) == cmsk
    prk = prior(EmpiricalPrior(), Xlist[:, keep])
    @test pr.mu[keep] == prk.mu
    @test pr.sigma[keep, keep] == prk.sigma

    # The horizon variant travels the same seam.
    prh = prior(EmpiricalPrior(; horizon = 5), Xlist, nothing, pnl_full)
    @test PO.investable_mask(prh) == cmsk
    @test prh.mu[keep] == prior(EmpiricalPrior(; horizon = 5), Xlist[:, keep]).mu

    # The high order prior expands its two tensors onto the full width.
    hop = prior(HighOrderPriorEstimator(), Xlist, nothing, pnl_full)
    @test size(hop.kt) == (N^2, N^2)
    @test size(hop.sk) == (N, N^2)
    @test size(hop.V) == (N, N)
    @test PO.investable_mask(hop) == cmsk
    hopk = prior(HighOrderPriorEstimator(), Xlist[:, keep])
    idx = PO.coverage_pair_index(cmsk)
    @test hop.kt[idx, idx] == hopk.kt
    @test hop.sk[keep, idx] == hopk.sk

    # The factor prior expands every block it writes, its regression result included.
    fp = prior(FactorPrior(), Xlist, F0, pnl_full)
    @test length(fp.mu) == N
    @test size(fp.sigma) == (N, N)
    @test size(fp.X, 2) == N
    @test size(fp.chol, 2) == N
    @test size(fp.rr.M, 1) == N
    @test PO.investable_mask(fp) == cmsk
    fpk = prior(FactorPrior(), Xlist[:, keep], F0)
    @test fp.mu[keep] == fpk.mu
    @test fp.sigma[keep, keep] == fpk.sigma
    @test fp.rr.M[keep, :] == fpk.rr.M
    @test all(isnan, fp.rr.M[4, :])
end
