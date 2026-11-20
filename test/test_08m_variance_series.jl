#=
A point-in-time variance series reads no observation after its own.

The two-pass cross-sectional regression weights an asset at observation `t` by the inverse of the
idiosyncratic variance estimated from residuals up to `t - 1`. A variance that already carries the
date-`t` squared residual embeds the return being regressed, so an asset with a large date-`t`
shock is down-weighted at date `t` itself, and the weights correlate with the residuals.

`variance_series` is that quantity. Its default refits on an expanding window, so every member of
the covariance surface answers it. `RegimeAdjustedExpWeightedVariance` overrides the default with a
single forward pass, because its estimate is a recursion over one observation. The two must agree
exactly: the fast route reads the cache after observation `t`, and the slow route rebuilds that
same cache from observations `1` to `t`.

The file also holds this estimator's first tests. Two defects it found are fixed in the same
change, and the last two testsets are their regressions. The final testset is issue #743's
regression: `regime_lohi_mult` clamps the multiplier now, so this file also covers that field.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, StableRNGs, LinearAlgebra

const PO = PortfolioOptimisers

# The expanding-window fallback, reached past the estimator's own override.
function slow_series(ce, X; dims::Int = 1, kwargs...)
    return invoke(PO.variance_series, Tuple{PO.AbstractCovarianceEstimator, PO.MatNum}, ce,
                  X; dims = dims, kwargs...)
end

# The expanding window written out by hand, slicing every per-observation keyword with the data.
# The fallback slices the two masks, `iv` and the observation weights of the estimator (#1928), and
# this reference states that slice independently of it.
function masked_slow_series(ce, X; estimation_mask = nothing, active_mask = nothing)
    return permutedims(reduce(hcat,
                              [var(ce, view(X, 1:t, :);
                                   estimation_mask = if isnothing(estimation_mask)
                                       nothing
                                   else
                                       view(estimation_mask, 1:t, :)
                                   end, active_mask = if isnothing(active_mask)
                                       nothing
                                   else
                                       view(active_mask, 1:t, :)
                                   end) for t in axes(X, 1)]))
end

# A variance estimator that reads `iv` and answers the sum of each of its columns, so a test can
# see which rows of `iv` each fit of a series reads.
struct IvEcho <: PO.AbstractVarianceEstimator end
function Statistics.var(::IvEcho, X::PO.MatNum; dims::Int = 1, iv = nothing, kwargs...)
    @assert size(iv) == size(X)
    return sum(iv; dims = 1)
end
# The same answer from an estimator that holds an `ImpliedVolatility`, so the reduce-and-expand
# route narrows its Coverage Universe by the gaps of `iv`.
struct IvSurfaceEcho{T} <: PO.AbstractVarianceEstimator
    ive::T
end
function Statistics.var(::IvSurfaceEcho, X::PO.MatNum; dims::Int = 1, iv = nothing,
                        kwargs...)
    @assert size(iv) == size(X)
    return sum(iv; dims = 1)
end

@testset "the fallback refits on an expanding window" begin
    rng = StableRNG(4321)
    X = randn(rng, 12, 3) .* 0.03

    for ce in (SimpleVariance(), Covariance(), SimpleVariance(; corrected = false))
        val = PO.variance_series(ce, X)
        @test size(val) == size(X)
        for t in axes(X, 1)
            @test isequal(val[t, :], vec(var(ce, view(X, 1:t, :))))
        end
        # `dims = 2` names the same series with the observations on the columns. Each fit runs
        # along `dims`, as the whole-sample `var` does, so the two agree to rounding (#1928).
        @test isapprox(PO.variance_series(ce, permutedims(X); dims = 2), permutedims(val);
                       rtol = 1e-12, nans = true)
    end

    # Row 1 is a fit on one observation, so a corrected estimator has nothing to divide by.
    @test all(isnan, PO.variance_series(SimpleVariance(), X)[1, :])
    @test all(iszero, PO.variance_series(SimpleVariance(; corrected = false), X)[1, :])

    # The guard is the family's, not this verb's.
    @test_throws DomainError PO.variance_series(SimpleVariance(), X; dims = 3)
    @test_throws DomainError PO.variance_series(SimpleVariance(), X; dims = 0)
end

@testset "the recursive override is one forward pass over the same cache" begin
    rng = StableRNG(4322)
    X = randn(rng, 90, 4) .* 0.02
    X[70:end, :] .*= 4.0        # a loud regime, so the regime multiplier is not one

    for ce in
        (RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3),
         RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 4, regime_min_obs = 2,
                                           centring = PreCentred()),
         RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           hac_lags = 2),
         RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           regime_method = PO.LogRegimeAdjusted()),
         RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           regime_method = PO.RootMeanSquaredAdjusted()))
        fast = PO.variance_series(ce, X)
        @test size(fast) == size(X)
        # The recursion and the refit run the same operations in the same order, so the two
        # agree bit for bit rather than to a tolerance.
        @test isequal(fast, slow_series(ce, X))
        # The last row is the whole-sample estimate the family's own verb returns.
        @test isequal(fast[end, :], var(ce, X))
        @test isequal(PO.variance_series(ce, permutedims(X); dims = 2), permutedims(fast))
    end

    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3)
    @test_throws DomainError PO.variance_series(ce, X; dims = 3)
    @test_throws DomainError PO.variance_series(ce, X; dims = 0)
end

@testset "row t reads no observation after t" begin
    rng = StableRNG(4323)
    X = randn(rng, 60, 3) .* 0.02
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3)

    base = PO.variance_series(ce, X)
    for t in (10, 31, 60)
        Xp = copy(X)
        Xp[t, :] .+= 5.0
        pert = PO.variance_series(ce, Xp)
        # Everything before the change is untouched, and the changed observation is felt at once.
        @test isequal(base[1:(t - 1), :], pert[1:(t - 1), :])
        @test !isequal(base[t, :], pert[t, :])
    end

    # The fallback carries the same promise, on an estimator with no recursion.
    sbase = PO.variance_series(SimpleVariance(), X)
    Xp = copy(X)
    Xp[31, :] .+= 5.0
    @test isequal(sbase[1:30, :], PO.variance_series(SimpleVariance(), Xp)[1:30, :])
end

@testset "the masks ride the observation axis" begin
    rng = StableRNG(4324)
    X = randn(rng, 50, 3) .* 0.02
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 4, regime_min_obs = 2)

    emsk = trues(size(X))
    emsk[:, 3] .= false
    amsk = trues(size(X))
    amsk[20:29, 2] .= false

    for kw in ((; estimation_mask = emsk), (; active_mask = amsk),
               (; estimation_mask = emsk, active_mask = amsk))
        fast = PO.variance_series(ce, X; kw...)
        @test isequal(fast, masked_slow_series(ce, X; kw...))
        @test isequal(fast[end, :], var(ce, X; kw...))
    end

    # A mask that does not match the data is refused, as it is for the whole-sample verb.
    @test_throws DimensionMismatch PO.variance_series(ce, X; estimation_mask = trues(4, 4))
    @test_throws DimensionMismatch PO.variance_series(ce, X; active_mask = trues(4, 4))

    # With the observations on the columns, a mask is sliced by column too.
    @test isequal(PO.variance_series(ce, permutedims(X); dims = 2,
                                     estimation_mask = permutedims(emsk),
                                     active_mask = permutedims(amsk)),
                  permutedims(PO.variance_series(ce, X; estimation_mask = emsk,
                                                 active_mask = amsk)))
    # The fallback slices the masks with the data, so the refit reaches the recursion bit for bit.
    @test isequal(slow_series(ce, X; estimation_mask = emsk, active_mask = amsk),
                  PO.variance_series(ce, X; estimation_mask = emsk, active_mask = amsk))
end

@testset "the fallback slices every per-observation input with the data (#1928)" begin
    rng = StableRNG(1928)
    X = randn(rng, 10, 3) .* 0.03
    msk = trues(size(X))
    wv = StatsBase.aweights(collect(range(1.0, 4.0; length = size(X, 1))))
    # Row t under observation weights is the weighted fit on the first t observations, with the
    # weights those observations carry in the full sample.
    function weighted_reference(ce, X, w; rows = t -> 1:t)
        return permutedims(reduce(hcat,
                                  [vec(var(PO.factory(ce, StatsBase.aweights(w[rows(t)])),
                                           X[rows(t), :])) for t in axes(X, 1)]))
    end
    for ce in (SimpleVariance(), Covariance())
        # Two plain estimators that read both masks: the fallback slices them with the data.
        @test isequal(PO.variance_series(ce, X; estimation_mask = msk, active_mask = msk),
                      PO.variance_series(ce, X))
        # With the observations on the columns, each fit runs along `dims = 2`, as the
        # whole-sample `var` does, so the two orientations agree to rounding.
        @test isapprox(PO.variance_series(ce, permutedims(X); dims = 2,
                                          estimation_mask = permutedims(msk),
                                          active_mask = permutedims(msk)),
                       permutedims(PO.variance_series(ce, X)); rtol = 1e-12, nans = true)
        cw = PO.factory(ce, wv)
        ref = weighted_reference(ce, X, wv)
        @test isequal(PO.variance_series(cw, X), ref)
        @test isapprox(PO.variance_series(cw, permutedims(X); dims = 2), permutedims(ref);
                       rtol = 1e-12, nans = true)
        # The panel route refits on the same rows, so it cuts the weights the same way.
        @test isequal(PO.variance_series(cw, X, nothing), ref)
    end
    # The weights of the series are relative: a rescaled vector states the same series.
    sw = PO.factory(SimpleVariance(), wv)
    @test isapprox(PO.variance_series(PO.factory(SimpleVariance(),
                                                 StatsBase.aweights(3 .* wv.values)), X)[2:end,
                                                                                         :],
                   PO.variance_series(sw, X)[2:end, :]; rtol = 1e-12)
    # A windowed wrapper with no weights of its own cuts the weights of its inner estimator to
    # each window, and the weights of the wrapper replace them when it holds some.
    @test isequal(PO.variance_series(WindowedVariance(; ve = sw), X),
                  weighted_reference(SimpleVariance(), X, wv))
    @test isequal(PO.variance_series(WindowedVariance(; ve = sw, window = 3), X),
                  weighted_reference(SimpleVariance(), X, wv; rows = t -> max(1, t - 2):t))
    @test isequal(var(WindowedVariance(; ve = sw, window = 4), X),
                  var(PO.factory(SimpleVariance(), StatsBase.aweights(wv[7:10])),
                      X[7:10, :]))
    @test isequal(var(WindowedVariance(; ve = sw, window = 4,
                                       w = StatsBase.aweights(ones(10))), X),
                  var(PO.factory(SimpleVariance(), StatsBase.aweights(ones(4))),
                      X[7:10, :]))
    # The vector method cuts the weights of the inner estimator to its window too, and a window
    # that keeps every observation leaves the inner estimator as it is.
    @test isequal(var(WindowedVariance(; ve = sw, window = 4), X[:, 1]),
                  var(PO.factory(SimpleVariance(), StatsBase.aweights(wv[7:10])),
                      X[7:10, 1]))
    @test isequal(var(WindowedVariance(; ve = sw), X), var(sw, X))
    # On an Asset Panel, each row of the windowed series fits the inner estimator on the panel
    # of its own rows, with its weights cut to the same rows.
    pnl = PO.AssetPanel(;
                        pf = [PO.NumericPanelField(; name = "mcap", vals = ones(size(X)...),
                                                   omsk = trues(size(X)...))],
                        amsk = trues(size(X)), emsk = trues(size(X)))
    @test isequal(PO.variance_series(WindowedVariance(; ve = sw, window = 3), X, pnl),
                  weighted_reference(SimpleVariance(), X, wv; rows = t -> max(1, t - 2):t))
    # The available-case route refits on the same expanding window as the plain route.
    cp = SimpleVariance(; cvg = CoveragePolicy())
    @test isapprox(PO.variance_series(cp, X, pnl), PO.variance_series(SimpleVariance(), X);
                   rtol = 1e-12, nans = true)
    # An estimator that reads `iv` gets the implied volatility of its own rows. `IvEcho` answers
    # the sum of each column of the `iv` that its fit reads, so row t is the sum of rows 1 to t.
    ivm = reshape(collect(1.0:30.0), 10, 3)
    @test isequal(PO.variance_series(IvEcho(), X; iv = ivm), cumsum(ivm; dims = 1))
    @test isequal(PO.variance_series(WindowedVariance(; ve = IvEcho(), window = 2), X;
                                     iv = ivm),
                  vcat(ivm[1:1, :], ivm[1:(end - 1), :] .+ ivm[2:end, :]))
end

#=
Issues 2225 and 2224. A window of a plain estimator over an Asset Panel that covers no asset made
the windowed series raise an `IsEmptyError` at its first such window, as the residual of an
`EWResidualVolatility` does in its warm-up. Its row is now `NaN`, as the row of a window that keeps
no observation is. And a `PortfolioOptimisersCovariance` reduced each window to its Coverage
Universe whatever its inner estimator was, so over a mask-aware estimator its `var` and its series
dropped every asset with one gap. The inner estimator now reads the window.
=#
@testset "a panel window that covers no asset is a NaN row, and the composite lets its inner estimator read the window" begin
    X = 0.01 .* randn(StableRNG(2225), 40, 3)
    R = copy(X)
    R[1, :] .= NaN
    R[1:6, 2] .= NaN
    amsk = trues(size(X))
    amsk[30:33, 3] .= false
    pnl = PO.AssetPanel(;
                        pf = [PO.NumericPanelField(; name = "mcap", vals = ones(size(X)...),
                                                   omsk = trues(size(X)...))], amsk = amsk,
                        emsk = amsk)
    # The windowed series of a plain estimator, written out by hand: each window is reduced to
    # its Coverage Universe, and a window that covers no asset is a `NaN` row.
    ref = fill(NaN, size(X))
    for t in axes(X, 1)
        rows = max(1, t - 3):t
        c = vec(all(isfinite, R[rows, :]; dims = 1) .& all(amsk[rows, :]; dims = 1))
        if any(c)
            ref[t, c] = vec(Statistics.var(R[rows, c]; dims = 1))
        end
    end
    w4 = PO.variance_series(WindowedVariance(; window = 4), R, pnl)
    @test isapprox(w4, ref; rtol = 1e-12, nans = true)
    @test all(isnan, w4[1:4, :])
    @test all(isfinite, w4[5:end, 1])
    @test isapprox(PO.variance_series(WindowedVariance(; window = 4), permutedims(R), pnl;
                                      dims = 2), permutedims(ref); rtol = 1e-12,
                   nans = true)
    # One fit over a sample that covers no asset still raises.
    @test_throws PO.IsEmptyError var(SimpleVariance(), R, pnl)
    # The probe reads the rows of the innermost fit: through a window with no window of its own,
    # and through a composite inside a window.
    @test isapprox(PO.variance_series(WindowedVariance(;
                                                       ve = WindowedVariance(; window = 4)),
                                      R, pnl), ref; rtol = 1e-12, nans = true)
    poc = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; pdm = nothing))
    @test isapprox(PO.variance_series(WindowedCovariance(; ce = poc, window = 4), R, pnl),
                   ref; rtol = 1e-12, nans = true)
    # The composite over a plain estimator keeps the route of the plain estimator.
    @test isapprox(PO.variance_series(poc, X, pnl),
                   PO.variance_series(Covariance(), X, pnl); rtol = 1e-12, nans = true)
    @test all(isnan, PO.variance_series(poc, R, pnl))
    # A composite over a composite answers for its inner estimator, so over a plain leaf it takes
    # the reduce-and-expand series too, and over a window it refits each row (R85 of #1817).
    pp = PortfolioOptimisersCovariance(; ce = poc, mp = MatrixProcessing(; pdm = nothing))
    @test PO.reduces_each_window(pp)
    @test !PO.reduces_each_window(PortfolioOptimisersCovariance(;
                                                                ce = WindowedCovariance(;
                                                                                        window = 4)))
    @test !PO.reduces_each_window(PortfolioOptimisersCovariance(;
                                                                ce = ExpWeightedCovariance()))
    @test isapprox(PO.variance_series(pp, X, pnl), PO.variance_series(Covariance(), X, pnl);
                   rtol = 1e-12, nans = true)
    @test isapprox(PO.variance_series(pp, R, pnl), PO.variance_series(poc, R, pnl);
                   rtol = 1e-12, nans = true)
    Rn = copy(X)
    Rn[5, 2] = NaN
    @test isapprox(PO.variance_series(poc, Rn, nothing),
                   PO.variance_series(Covariance(), Rn, nothing); rtol = 1e-12, nans = true)
    @test isapprox(PO.variance_series(poc, X), PO.variance_series(Covariance(), X);
                   rtol = 1e-12, nans = true)
    @test !PO.series_folds(poc)
    @test !PO.series_folds(PortfolioOptimisersCovariance(;
                                                         ce = WindowedCovariance(;
                                                                                 ce = ExpWeightedCovariance())))
    # A plain leaf that reads `iv` drops the assets with a gap in the `iv` of its window too.
    ivs = reshape(collect(1.0:120.0), 40, 3)
    ivs[6, 2] = NaN
    ivref = fill(NaN, size(X))
    for t in axes(X, 1)
        rows = max(1, t - 1):t
        c = vec(all(isfinite, ivs[rows, :]; dims = 1) .& all(amsk[rows, :]; dims = 1))
        ivref[t, c] = vec(sum(ivs[rows, c]; dims = 1))
    end
    @test isequal(PO.variance_series(WindowedVariance(;
                                                      ve = IvSurfaceEcho(ImpliedVolatility(;
                                                                                           ppy = 252)),
                                                      window = 2), X, pnl; iv = ivs), ivref)
    # A windowed mask-aware leaf reads the masks of its window, as with the whole panel.
    ev = ExpWeightedVariance(; min_obs = 2)
    @test isapprox(PO.variance_series(WindowedVariance(; ve = ev, window = 10), R, pnl),
                   permutedims(reduce(hcat,
                                      [vec(var(ev, R[max(1, t - 9):t, :],
                                               PO.windowed_panel(pnl, max(1, t - 9):t)))
                                       for t in axes(R, 1)])); rtol = 1e-12, nans = true)
    # The composite over a mask-aware estimator gives the series, the variance and the standard
    # deviation of its inner estimator, with the panel and with no panel. Its fold gives the
    # series of a refit at each row, with the panel and with the mask keywords.
    for ce in (Covariance(; cvg = CoveragePolicy()), ExpWeightedCovariance(; min_obs = 3),
               RegimeAdjustedExpWeightedCovariance(; min_obs = 3))
        pm = PortfolioOptimisersCovariance(; ce = ce,
                                           mp = MatrixProcessing(; pdm = nothing))
        @test PO.series_folds(pm)
        vs = PO.variance_series(pm, R, pnl)
        @test count(isfinite, vs) > 90
        refit = PO.variance_series(pm, nothing, nothing, R, nothing; active_mask = amsk,
                                   estimation_mask = amsk)
        @test isapprox(vs, refit; rtol = 1e-12, nans = true)
        @test isapprox(PO.variance_series(pm, R; active_mask = amsk,
                                          estimation_mask = amsk), refit; rtol = 1e-12,
                       nans = true)
        @test isapprox(PO.variance_series(pm, permutedims(R); dims = 2,
                                          active_mask = permutedims(amsk),
                                          estimation_mask = permutedims(amsk)),
                       permutedims(refit); rtol = 1e-12, nans = true)
        if ce isa RegimeAdjustedExpWeightedCovariance
            continue
        end
        @test isapprox(vs, PO.variance_series(ce, R, pnl); rtol = 1e-12, nans = true)
        @test isapprox(PO.variance_series(pm, permutedims(R), pnl; dims = 2),
                       permutedims(vs); rtol = 1e-12, nans = true)
        @test isapprox(PO.variance_series(pm, R, nothing),
                       PO.variance_series(ce, R, nothing); rtol = 1e-12, nans = true)
        @test isapprox(var(pm, R, pnl), var(ce, R, pnl); rtol = 1e-12, nans = true)
        @test isapprox(std(pm, R, pnl), sqrt.(var(ce, R, pnl)); rtol = 1e-12, nans = true)
        @test isapprox(var(pm, permutedims(R), pnl; dims = 2), permutedims(var(ce, R, pnl));
                       rtol = 1e-12, nans = true)
    end
    # R85 of #1817. A `Covariance` with observation weights cannot fold, because `partial_fit!`
    # refuses its `w`. So the composite over it refits each row, with the panel and with the mask
    # keywords, and it does not raise.
    cw = Covariance(; cvg = CoveragePolicy(),
                    w = StatsBase.aweights(collect(range(0.5, 1.5; length = size(R, 1)))))
    pw = PortfolioOptimisersCovariance(; ce = cw, mp = MatrixProcessing(; pdm = nothing))
    @test !PO.series_folds(pw)
    @test isapprox(PO.variance_series(pw, R, pnl), PO.variance_series(cw, R, pnl);
                   rtol = 1e-12, nans = true)
    @test isapprox(PO.variance_series(pw, R; active_mask = amsk),
                   PO.variance_series(pw, nothing, nothing, R, nothing; active_mask = amsk);
                   rtol = 1e-12, nans = true)
    # The trait names the mask-aware estimators, and a wrapper answers for its inner one.
    pol = CoveragePolicy()
    for (ce, flag) in ((SimpleVariance(), false), (Covariance(), false),
                       (SimpleVariance(; cvg = pol), true), (Covariance(; cvg = pol), true),
                       (ExpWeightedVariance(), true), (ExpWeightedCovariance(), true),
                       (RegimeAdjustedExpWeightedVariance(), true),
                       (RegimeAdjustedExpWeightedCovariance(), true),
                       (WindowedVariance(; ve = ExpWeightedVariance()), true),
                       (WindowedCovariance(), false), (PortfolioOptimisersCovariance(), false),
                       (PortfolioOptimisersCovariance(; ce = ExpWeightedCovariance()), true))
        @test PO.reads_active_mask(ce) == flag
    end
end

#=
Issue 1954. The weighted mean of one observation can differ from it by one unit in the last
place, so a corrected weighted fit on one observation divided a sum of squares of about `1e-36`
by zero, and gave `Inf` for about one asset in four. The unweighted fit gives `NaN`. Row 1 of a
weighted series is a fit on one observation, so the cross-sectional inverse variance of row 2 gave
such an asset a weight of `0`, where the unweighted fit drops it as `NaN`.
=#
@testset "a corrected fit on one weighted observation is NaN, as the unweighted fit is (#1954)" begin
    eps = randn(StableRNG(11), 8, 6) .* 0.02
    wv = StatsBase.aweights(rand(StableRNG(12), 8) .+ 0.1)
    m = trues(size(eps))
    @test all(isnan, PO.variance_series(SimpleVariance(; w = wv), eps)[1, :])
    @test all(isnan, PO.variance_series(Covariance(; w = wv), eps)[1, :])
    @test all(isnan,
              PO.cross_sectional_lagged_inverse_variance(SimpleVariance(; w = wv), eps, m)[2,
                                                                                           :])
    @test isequal(PO.cross_sectional_lagged_inverse_variance(SimpleVariance(; w = wv), eps,
                                                             m)[2, :],
                  PO.cross_sectional_lagged_inverse_variance(SimpleVariance(), eps, m)[2,
                                                                                       :])
    # Over many draws of one observation, every weight type and every verb is `NaN` under the
    # correction, with no `Inf` among them. The unweighted fit is `NaN` too. A frequency weight
    # counts copies, so the weight is at most one: more than one copy has a variance.
    rng = StableRNG(1954)
    for W in (StatsBase.aweights, StatsBase.pweights, StatsBase.fweights), _ in 1:200
        x = randn(rng, 1, 3) .* 0.02
        w1 = W([0.1 + 0.9 * rand(rng)])
        ce = Covariance(; w = w1)
        @test all(isnan, var(SimpleVariance(; w = w1), x))
        @test all(isnan, std(SimpleVariance(; w = w1), x))
        @test isnan(var(SimpleVariance(; w = w1), x[:, 1]))
        @test isnan(std(SimpleVariance(; w = w1), x[:, 1]))
        @test all(isnan, cov(ce, x))
        @test all(isnan, cov(Covariance(; w = w1, alg = SemiMoment()), x))
        @test all(isnan, var(ce, x))
        # A correlation needs two observations, under either correction. The diagonal stays
        # one, as the unweighted fit keeps it.
        for c in (true, false)
            rho = cor(Covariance(; w = w1,
                                 ce = GeneralCovariance(;
                                                        ce = StatsBase.SimpleCovariance(;
                                                                                        corrected = c))),
                      x)
            @test isequal(rho, cor(Covariance(), x))
        end
    end
    @test all(isnan, var(SimpleVariance(), randn(StableRNG(1), 1, 3)))
    # Frequency weights that sum to less than one have a negative divisor, which gave a negative
    # variance.
    @test all(isnan, var(SimpleVariance(; w = StatsBase.fweights([0.4])), [0.01 0.02]))
    # A correlation does not read the divisor, so two observations under such weights have a
    # correlation. The square root of the negative diagonal threw before.
    Xf = [0.01 0.03; -0.02 0.01; 0.02 -0.01]
    wf = StatsBase.fweights([0.2, 0.3, 0.1])
    @test all(isnan, cov(GeneralCovariance(; w = wf), Xf))
    @test cor(GeneralCovariance(; w = wf), Xf) ≈ cor(GeneralCovariance(; w = wf,
                                                                       ce = StatsBase.SimpleCovariance(; corrected = false)), Xf)
    @test PO.weighted_correlation_estimator(StatsBase.SimpleCovariance(), wf) ==
          StatsBase.SimpleCovariance(; corrected = false)
    gce = PO.GerberCovariance()
    @test PO.weighted_correlation_estimator(gce, wf) === gce
    # A fit with a positive divisor does not change: an uncorrected fit, and a corrected fit on
    # one observation that a frequency weight counts twice.
    x1 = [0.01 -0.02 0.03]
    @test all(isfinite,
              var(SimpleVariance(; w = StatsBase.aweights([0.4]), corrected = false), x1))
    @test all(isfinite, var(SimpleVariance(; w = StatsBase.fweights([2.0])), x1))
    X2 = randn(StableRNG(2), 2, 3)
    w2 = StatsBase.aweights([0.3, 0.7])
    @test var(SimpleVariance(; w = w2), X2) == var(X2, w2, 1; corrected = true)
    @test cor(GeneralCovariance(; w = w2), X2) == cor(X2, w2, 1)
    # The guard reads the divisor of `StatsBase` and does not change a valid estimate. A
    # covariance estimator other than `StatsBase.SimpleCovariance` has a divisor that the
    # library does not know, so it keeps its value.
    @test PO.weighted_moment_guard(2.0, StatsBase.aweights([0.3, 0.7]), true) == 2.0
    @test isnan(PO.weighted_moment_guard(2.0f0, StatsBase.pweights([0.0, 0.7]), true))
    @test PO.weighted_moment_guard(2.0f0, StatsBase.pweights([0.0, 0.7]), true) isa Float32
    @test PO.weighted_comoment_guard(Statistics.cov, PO.GerberCovariance(),
                                     [1.0 2.0; 2.0 1.0], StatsBase.aweights([1.0])) ==
          [1.0 2.0; 2.0 1.0]
end

@testset "an observation the pass cannot use leaves the cache where it was" begin
    rng = StableRNG(4327)
    X = randn(rng, 30, 3) .* 0.02
    X[7, :] .= NaN          # no asset is valid, so the observation updates nothing
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 4, regime_min_obs = 2)

    series = PO.variance_series(ce, X)
    @test isequal(series[7, :], series[6, :])
    @test isequal(series, slow_series(ce, X))

    # A floor above every variance the sample reaches leaves every standardised innovation
    # `NaN`, so the regime state never has a reading to smooth.
    quiet = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 4,
                                              regime_min_obs = 2, min_val = 1.0)
    qseries = PO.variance_series(quiet, X)
    @test isequal(qseries, slow_series(quiet, X))
    # The multiplier stays at one, so the series is the unadjusted variance.
    @test isequal(qseries[end, :], var(quiet, X))
end

#=
Defect 1. `regime_adjusted_variance` corrected the exponentially weighted bias by indexing the
correction with a mask and assigning a full-length right side, so `var` threw a `DimensionMismatch`
whenever any asset held no observation at all. An asset with no finite return, and an asset that a
whole-sample `active_mask` switches off, are both that case. The expanding-window fallback meets it
at the first observation of any such asset, so the verb this file adds cannot exist without the fix.
=#
@testset "an asset with no observation is NaN, and does not throw" begin
    rng = StableRNG(4325)
    X = randn(rng, 40, 3) .* 0.02
    X[:, 3] .= NaN
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 4, regime_min_obs = 2)

    val = var(ce, X)
    @test isnan(val[3])
    @test all(isfinite, view(val, 1:2))

    series = PO.variance_series(ce, X)
    @test all(isnan, view(series, :, 3))
    @test isequal(series, slow_series(ce, X))

    # The same case reached through a mask that is false for the whole sample.
    amsk = trues(size(X))
    amsk[:, 2] .= false
    @test isnan(var(ce, randn(StableRNG(1), 40, 3) .* 0.02; active_mask = amsk)[2])
end

#=
Defect 2. `process_observation!` advances `regime_state` and `n_regime_obs` with
`Accessors.@reset`, which rebinds the function's own local because the cache is immutable. The
caller kept its original cache, so `n_regime_obs` never left zero, the multiplier was never
reached, and every regime field on the estimator was inert. The function now returns the cache and
the pass rebinds it.
=#
@testset "the regime state accumulates across observations" begin
    rng = StableRNG(4326)
    X = randn(rng, 120, 3) .* 0.02
    X[90:end, :] .*= 5.0

    for rm in (PO.FirstMomentRegimeAdjusted(), PO.LogRegimeAdjusted(),
               PO.RootMeanSquaredAdjusted())
        live = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                 regime_min_obs = 3, regime_method = rm)
        # A regime threshold the sample cannot reach leaves the multiplier at one, which is the
        # answer the whole family used to give.
        inert = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                  regime_min_obs = 10_000,
                                                  regime_method = rm)
        @test !isequal(var(live, X), var(inert, X))
        @test !isequal(PO.variance_series(live, X), PO.variance_series(inert, X))
    end
end
#=
`regime_method = nothing` is the off switch, added by issue #718 for `EWVolatility`. Before it, the
only way to read the plain exponentially weighted recursion was a `regime_min_obs` the sample
cannot reach, which is a property of the data and not of the estimator. The switch advances no
regime state, so `n_regime_obs` stays at zero and the multiplier stays at one whatever the sample
holds.
=#
@testset "no regime method is the plain exponentially weighted recursion" begin
    rng = StableRNG(718)
    X = randn(rng, 90, 4) .* 0.02
    X[70:end, :] .*= 6.0
    decay, min_obs = exp2(-inv(5.0)), 5

    # The recursion written out by hand: uncentred, bias corrected, no regime factor.
    function plain_series(X, decay, min_obs)
        T, N = size(X)
        V = fill(NaN, T, N)
        for i in 1:N
            s, n = 0.0, 0
            for t in 1:T
                if isfinite(X[t, i])
                    s = decay * s + (1 - decay) * X[t, i]^2
                    n += 1
                end
                if n >= min_obs
                    V[t, i] = s / (1 - decay^n)
                end
            end
        end
        return V
    end

    off = RegimeAdjustedExpWeightedVariance(; decay = decay, min_obs = min_obs,
                                            centring = PreCentred(),
                                            regime_method = nothing)
    expected = plain_series(X, decay, min_obs)
    @test isapprox(PO.variance_series(off, X), expected; rtol = 1e-12, nans = true)
    @test isapprox(vec(var(off, X)), expected[end, :]; rtol = 1e-12, nans = true)

    # The default still applies the multiplier, so the switch changes an answer.
    on = RegimeAdjustedExpWeightedVariance(; decay = decay, min_obs = min_obs,
                                           centring = PreCentred())
    @test !isapprox(PO.variance_series(on, X), expected; rtol = 1e-12, nans = true)

    # The slow route agrees with the fast one, as it does for every other setting.
    @test isapprox(slow_series(off, X), PO.variance_series(off, X); rtol = 1e-12,
                   nans = true)

    # An asset that turns inactive restarts, and the switch does not change that.
    amsk = trues(size(X))
    amsk[40:45, 2] .= false
    Xm = ifelse.(amsk, X, NaN)
    got = PO.variance_series(off, Xm; active_mask = amsk)
    @test all(isnan, view(got, 40:45, 2))
    @test isapprox(got[50, 2], plain_series(view(Xm, 46:50, 2:2), decay, min_obs)[end, 1];
                   rtol = 1e-12)
end
#=
Issue #743. `regime_lohi_mult` validated its own ordering and was stored, but nothing read it: the
documented clamp on the regime multiplier never ran, and the shipped default `(0.7, 1.6)` bounded
nothing. `regime_adjusted_variance` now clamps the multiplier to that range before it is squared,
and the default moved to `nothing`, so a caller who names no bound keeps the answer this library
always gave.
=#
@testset "regime_lohi_mult clamps the multiplier, and nothing leaves it alone" begin
    rng = StableRNG(4326)
    X = randn(rng, 120, 3) .* 0.02
    X[90:end, :] .*= 5.0
    lo, hi = 0.9, 1.1

    for rm in (PO.FirstMomentRegimeAdjusted(), PO.LogRegimeAdjusted(),
               PO.RootMeanSquaredAdjusted())
        # The multiplier alone, backed out of an inert estimator that carries no regime factor.
        inert = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                  regime_min_obs = 10_000,
                                                  regime_method = rm)
        base = var(inert, X)

        unclamped = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                      regime_min_obs = 3,
                                                      regime_method = rm)
        factor = sqrt.(var(unclamped, X) ./ base)

        clamped = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                    regime_min_obs = 3, regime_method = rm,
                                                    regime_lohi_mult = (lo, hi))
        expected = base .* clamp.(factor, lo, hi) .^ 2
        @test isapprox(var(clamped, X), expected; rtol = 1e-12)

        # An explicit `nothing` reproduces the answer the field always gave before this fix.
        off = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                regime_min_obs = 3, regime_method = rm,
                                                regime_lohi_mult = nothing)
        @test isequal(var(unclamped, X), var(off, X))
    end

    # The clamp survives the point-in-time series path too, not only the batch read-out.
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           regime_lohi_mult = (lo, hi))
    @test isequal(PO.variance_series(ce, X)[end, :], var(ce, X))
    @test isequal(slow_series(ce, X), PO.variance_series(ce, X))

    @test_throws DomainError RegimeAdjustedExpWeightedVariance(;
                                                               regime_lohi_mult = (1.1,
                                                                                   0.9))
    @test_throws DomainError RegimeAdjustedExpWeightedVariance(;
                                                               regime_lohi_mult = (0.0,
                                                                                   1.6))
end
#=
Issue #397, the docstring sweep of the family, checked each formula with numbers and found four
defects in the code: a `regime_decay` that no constructor checked, a zero parameter of a regime
method that made the multiplier `NaN` or `Inf`, a log multiplier that read the shape of its law as
its exponent, and a method of the caller's own that failed with a `MethodError` from the fit.
=#
struct MyRegimeMethod397 <: PO.RegimeAdjustedMethod end
PO.regime_multiplier(::MyRegimeMethod397, s::Number) = sqrt(max(s, 0))
function PO.get_regime_state(::MyRegimeMethod397, z2::AbstractVector, ::Number)
    return sum(z2) / length(z2)
end
@testset "The regime methods refuse what they cannot compute (#397)" begin
    # `regime_decay` weighs an exponentially weighted recursion. At one the regime state never
    # leaves its first statistic, and at 1.5 or -0.5 the variance came out as zero (#397).
    @test_throws DomainError RegimeAdjustedExpWeightedVariance(; regime_decay = 1.0)
    @test_throws DomainError RegimeAdjustedExpWeightedVariance(; regime_decay = 1.5)
    @test_throws DomainError RegimeAdjustedExpWeightedVariance(; regime_decay = -0.5)
    # A zero parameter made the multiplier `NaN` or `Inf`: `x = 0` gives the digamma pole in the
    # constant of the log method and a division by zero in the first-moment method, and `y = 0`
    # gives `log(0)` (#397).
    @test_throws DomainError LogRegimeAdjusted(; x = 0.0)
    @test_throws DomainError LogRegimeAdjusted(; y = 0.0)
    @test_throws DomainError FirstMomentRegimeAdjusted(; x = 0.0)

    # The log method's state is a log variance ratio whatever the law of the squares, so a ratio
    # of four is a volatility multiplier of two. The multiplier read `exp(x * s)`, so a law with
    # `x = 2` scaled the variance by the ratio to the fourth power (#397).
    for method in (LogRegimeAdjusted(), LogRegimeAdjusted(; x = 2.0, y = 0.5))
        @test PO.regime_multiplier(method, log(4.0)) ≈ 2.0
    end

    # A method the caller defines runs under the raw statistic, and the two bias corrections,
    # which read the moments of the library methods alone, refuse it at construction rather
    # than with a `MethodError` from the fit (#397).
    rm = MyRegimeMethod397()
    @test_throws ArgumentError RegimeAdjustedExpWeightedVariance(; regime_method = rm)
    @test_throws ArgumentError RegimeAdjustedExpWeightedVariance(; regime_method = rm,
                                                                 debias = LawDebias())
    X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012; 0.008 -0.02; -0.02 0.03]
    kw = (; decay = 0.9, min_obs = 2, regime_min_obs = 1, debias = nothing)
    @test var(RegimeAdjustedExpWeightedVariance(; regime_method = rm, kw...), X) ≈
          var(RegimeAdjustedExpWeightedVariance(; regime_method = RootMeanSquaredAdjusted(),
                                                kw...), X)
end

#=
`std` is the root of `var`, and issue #637 names its absence.

Before this change `RegimeAdjustedExpWeightedVariance` declared `var` alone. `std(ce, X)`
therefore reached the covariance surface's own fallback, which reads `cov(ce, X)`, which reads
`cor(ce, X)`, which reads `cov(ce, X)` back through `StatsBase`'s generic method. The caller met a
`StackOverflowError` rather than a number. The three forms below are the three `var` answers under
their root, so each one closes that cycle for its own arity.

`cov` and `cor` on this type still overflow, because the type still declares neither. Issue #637
carries that half, and `VERB_EXEMPT` in `test/moment_family_setup.jl` still names the type.
=#
@testset "std is the root of var, for each of the three forms" begin
    rng = StableRNG(4327)
    X = randn(rng, 90, 4) .* 0.02
    X[70:end, :] .*= 4.0        # a loud regime, so the regime multiplier is not one

    for ce in
        (RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3),
         RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 4, regime_min_obs = 2,
                                           centring = PreCentred()),
         RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           hac_lags = 2),
         RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           regime_method = PO.LogRegimeAdjusted()),
         RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           regime_method = PO.RootMeanSquaredAdjusted()))
        # The batch form takes the root of the batch variance, bit for bit.
        @test isequal(std(ce, X), sqrt.(var(ce, X)))
        @test isequal(std(ce, permutedims(X); dims = 2), std(ce, X))

        # The state forms take the root of the state variance, bit for bit.
        fitted = partial_fit!(ce, X)
        @test isequal(std(fitted), sqrt.(var(fitted)))
        @test isequal(std(fitted, fitted.cache), std(fitted))
        @test isequal(std(fitted), std(ce, X))
    end

    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3)

    # The guard is the family's, and the keywords reach `var` unchanged.
    @test_throws DomainError std(ce, X; dims = 3)
    estimation_mask = trues(size(X))
    estimation_mask[1:20, 1] .= false
    active_mask = trues(size(X))
    active_mask[1:10, 2] .= false
    @test isequal(std(ce, X; estimation_mask = estimation_mask, active_mask = active_mask),
                  sqrt.(var(ce, X; estimation_mask = estimation_mask,
                            active_mask = active_mask)))

    # An asset with too few observations is `NaN` under the root as it is under the variance.
    short = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 200,
                                              regime_min_obs = 3)
    @test all(isnan, std(short, X))

    # An estimator that was given no observation carries no state, so the read is refused.
    @test_throws ArgumentError std(RegimeAdjustedExpWeightedVariance())
end
#=
A `Float32` panel keeps a `Float32` variance in both regime-adjusted passes. Issue #1972
removed the integer case: integer data convert at the carrier.
=#
@testset "a Float32 panel keeps a Float32 variance" begin
    X32 = Float32[1 2 0; -1 0 1; 1 1 -2; 0 -1 1; 2 1 0; -1 2 1] ./ 100
    ves = (RegimeAdjustedExpWeightedVariance(; decay = 0.5, min_obs = 1,
                                             regime_min_obs = 1),
           RegimeAdjustedExpWeightedVariance(; decay = 0.5, min_obs = 1, regime_min_obs = 1,
                                             hac_lags = 2),
           RegimeAdjustedExpWeightedVariance(; decay = 0.5, min_obs = 1,
                                             centring = PreCentred(), hac_lags = 2,
                                             regime_method = nothing))
    for ce in ves
        @test eltype(PO.variance_series(ce, X32)) == Float32
    end
end
#=
Issue #2221. The variance of a `PortfolioOptimisersCovariance` is the diagonal of its processed
covariance, so the root method runs `mp` on the fit of each row. Row 1 is a fit on one
observation, so its covariance is `NaN`, and `Posdef`, `Denoise` and `Detone` refuse it. The
docstring of the composite states this in a warning, with the configuration that avoids it. This
testset pins each claim of that warning.
=#
# A positive definite repair of the caller that returns a non-finite matrix without change.
struct NonFinitePassPosdef <: PO.AbstractPosdefEstimator end
function PO.posdef!(::NonFinitePassPosdef, X::PO.MatNum)
    return all(isfinite, X) ? PO.posdef!(Posdef(), X) : X
end
@testset "a composite with a step in mp raises on row 1 of its variance series" begin
    X = 0.01 .* randn(StableRNG(78), 40, 3)
    for mp in (MatrixProcessing(), MatrixProcessing(; pdm = nothing, dn = Denoise()),
               MatrixProcessing(; pdm = nothing, dt = Detone()))
        @test_throws ArgumentError PO.variance_series(PortfolioOptimisersCovariance(;
                                                                                    mp = mp),
                                                      X)
    end
    @test_throws ArgumentError PO.variance_series(WindowedCovariance(; window = 10), X)
    # The variance of the composite is the diagonal of its processed covariance, bit for bit.
    ce = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; dn = Denoise()))
    @test vec(var(ce, X)) == LinearAlgebra.diag(cov(ce, X))
    # With no step in `mp`, the series answers, row 1 is `NaN`, and it equals the series of the
    # inner estimator.
    for inner in (Covariance(), ExpWeightedCovariance(; decay = 0.9))
        nopd = PortfolioOptimisersCovariance(; ce = inner,
                                             mp = MatrixProcessing(; pdm = nothing))
        s = PO.variance_series(nopd, X)
        @test all(isnan, s[1, :])
        @test s ≈ PO.variance_series(inner, X) nans = true rtol = 1e-14
        sw = PO.variance_series(WindowedCovariance(; window = 10, ce = nopd), X)
        @test all(isnan, sw[1, :])
        @test sw ≈ PO.variance_series(WindowedCovariance(; window = 10, ce = inner), X) nans = true rtol = 1e-14
    end
    # A `pdm` of the caller that returns a non-finite matrix without change avoids the raise,
    # and it repairs every later row. A `dn` beside it still raises.
    ownpd = PortfolioOptimisersCovariance(;
                                          mp = MatrixProcessing(;
                                                                pdm = NonFinitePassPosdef()))
    s = PO.variance_series(ownpd, X)
    @test all(isnan, s[1, :])
    @test s ≈ PO.variance_series(PortfolioOptimisersCovariance(;
                                                               mp = MatrixProcessing(;
                                                                                     pdm = nothing)),
                                 X) nans = true rtol = 1e-14
    @test_throws ArgumentError PO.variance_series(PortfolioOptimisersCovariance(;
                                                                                mp = MatrixProcessing(;
                                                                                                      pdm = NonFinitePassPosdef(),
                                                                                                      dn = Denoise())),
                                                  X)
end
