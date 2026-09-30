#=
The windowed moment wrappers forward the Asset Panel and the universe masks, and their variance
series rolls with the observation (#1398, map #1375).

Before #1398 a wrapper answered only `(X::MatNum; kwargs...)`. On a panel the call reached the
generic reduce-and-expand, which reduced the WHOLE sample to its Coverage Universe, so an asset
that listed at any row, before or inside the window, came back `NaN`. With an `active_mask`
keyword the wrapper cut the returns and passed the mask whole, which raised a
`DimensionMismatch`, so a windowed exponentially weighted estimator could not sit in the variance
slot of a Cross-Sectional Factor Prior at all.

The oracle keeps the last rows of the returns and of the active mask once, at its first fit. So
its batch fit is the wrapper's batch fit, and the stored cases pin that. In its prior the oracle
fills the variance slot by one `partial_fit` per row, and a window of one row cuts nothing: there
the window has no effect (measured, max difference exactly 0). The maintainer ruled on #1398 that
row `t` of our series is the fit of the wrapper on rows `1` to `t`, which keeps the window that
ends at `t`. That is the library's definition of a variance series, and the stored series pin it
against the oracle's windowed batch fit on each prefix.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

const WM_DECAY = 2.0^(-1 / 20)

function windowed_panel_fixture()
    rng = StableRNG(1398)
    T, N = 120, 4
    X = 0.01 .* randn(rng, T, N)
    amsk = trues(T, N)
    # Asset 3 lists inside the window of the last 40 rows, asset 4 lists before it.
    amsk[1:100, 3] .= false
    amsk[1:30, 4] .= false
    X[.!amsk] .= NaN
    pf = [PortfolioOptimisers.NumericPanelField(; name = "mcap", vals = ones(T, N),
                                                omsk = trues(T, N))]
    return X, amsk, AssetPanel(; pf = pf, amsk = amsk, emsk = copy(amsk))
end

@testset "Windowed wrappers forward the Asset Panel and the masks" begin
    X, amsk, pnl = windowed_panel_fixture()
    T = size(X, 1)
    win = (T - 39):T
    pw = PortfolioOptimisers.asset_panel_view(pnl, win, :, nothing)
    me = ExpWeightedExpectedReturns(; decay = WM_DECAY, min_obs = 1)
    ve = ExpWeightedVariance(; decay = WM_DECAY, min_obs = 1)
    ce = ExpWeightedCovariance(; decay = WM_DECAY, min_obs = 1)

    # Every wrapper and every forwarded generic equals the unwrapped estimator on the rows and
    # the masks that the window keeps, with a mask-aware inner estimator and with a plain one.
    # A plain inner estimator reduces to the Coverage Universe of the window, so asset 4, which
    # is active over the whole window, is in it.
    cases = [(mean, WindowedExpectedReturns(; me = me, window = 40), me),
             (mean, WindowedExpectedReturns(; window = 40), SimpleExpectedReturns()),
             (cov, WindowedCovariance(; ce = ce, window = 40), ce),
             (cor, WindowedCovariance(; ce = ce, window = 40), ce),
             (cov, WindowedCovariance(; window = 40), PortfolioOptimisersCovariance()),
             (var, WindowedVariance(; ve = ve, window = 40), ve),
             (std, WindowedVariance(; ve = ve, window = 40), ve),
             (var, WindowedVariance(; window = 40), SimpleVariance()),
             (coskewness, WindowedCoskewness(; window = 40), Coskewness()),
             (cokurtosis, WindowedCokurtosis(; window = 40), Cokurtosis())]
    for (f, wrapped, inner) in cases
        @test isequal(f(wrapped, X, pnl), f(inner, X[win, :], pw))
    end
    # The fix: before it, both late listings were `NaN` on a panel.
    sigma = cov(WindowedCovariance(; ce = ce, window = 40), X, pnl)
    @test all(isfinite, sigma)
    @test all(isfinite, cov(WindowedCovariance(; window = 40), X, pnl)[4, 1:2])

    # The `active_mask` keyword is cut with the returns. Before the fix it raised a
    # `DimensionMismatch`.
    @test isequal(cov(WindowedCovariance(; ce = ce, window = 40), X; active_mask = amsk),
                  sigma)
    # `dims = 2` cuts the columns of the returns and the rows of the panel.
    @test isequal(cov(WindowedCovariance(; ce = ce, window = 40), permutedims(X), pnl;
                      dims = 2), sigma)
    # An index-vector window, no window, and no panel.
    idx = collect(60:3:120)
    @test isequal(cov(WindowedCovariance(; ce = ce, window = idx), X, pnl),
                  cov(ce, X[idx, :],
                      PortfolioOptimisers.asset_panel_view(pnl, idx, :, nothing)))
    @test isequal(cov(WindowedCovariance(; ce = ce), X, pnl), cov(ce, X, pnl))
    @test isequal(cov(WindowedCovariance(; ce = ce, window = 40), X, nothing),
                  cov(ce, X[win, :], nothing))

    # The preamble cuts only the keywords the caller gave, and passes the others through.
    kw = PortfolioOptimisers.windowed_keywords(win, 1; active_mask = amsk, foo = 1)
    @test keys(kw) == (:active_mask, :foo)
    @test kw.active_mask == amsk[win, :] && kw.foo == 1
    @test PortfolioOptimisers.windowed_keywords(:, 1; estimation_mask = amsk).estimation_mask ===
          amsk
    @test isempty(PortfolioOptimisers.windowed_keywords(win, 1))
end

@testset "A windowed variance series rolls with the observation" begin
    X, amsk, pnl = windowed_panel_fixture()
    T = size(X, 1)
    ve = ExpWeightedVariance(; decay = WM_DECAY, min_obs = 1)
    ce = ExpWeightedCovariance(; decay = WM_DECAY, min_obs = 1)

    # Row `t` is the batch fit of the wrapper on rows `1` to `t`, so it keeps the window that
    # ends at `t`, and the last row is the batch fit on the whole sample.
    for wr in (WindowedVariance(; ve = ve, window = 40),
               WindowedCovariance(; ce = ce, window = 40))
        S = PortfolioOptimisers.variance_series(wr, X; active_mask = amsk,
                                                estimation_mask = amsk)
        @test size(S) == size(X)
        @test all(t -> isequal(S[t, :], vec(var(wr, X[1:t, :]; active_mask = amsk[1:t, :]))),
                  1:T)
        @test isequal(PortfolioOptimisers.variance_series(wr, X, pnl), S)
        @test isequal(PortfolioOptimisers.variance_series(wr, permutedims(X); dims = 2,
                                                          active_mask = permutedims(amsk)),
                      permutedims(S))
    end
    # Before the window is full, row `t` keeps every row up to `t`, so it equals the series of
    # the unwrapped estimator.
    S = PortfolioOptimisers.variance_series(WindowedVariance(; ve = ve, window = 40), X;
                                            active_mask = amsk)
    S0 = PortfolioOptimisers.variance_series(ve, X; active_mask = amsk)
    @test isequal(S[1:40, :], S0[1:40, :])
    @test !isequal(S[41:end, :], S0[41:end, :])
    # An index-vector window keeps its entries up to `t`, and a row before its first entry is
    # `NaN`.
    wi = WindowedVariance(; ve = ve, window = collect(60:3:120))
    Si = PortfolioOptimisers.variance_series(wi, X; active_mask = amsk)
    @test all(isnan, Si[1:59, :])
    @test isequal(Si[end, :], vec(var(wi, X; active_mask = amsk)))
    @test PortfolioOptimisers.windowed_series_rows(nothing, 5) == 1:5
    @test PortfolioOptimisers.windowed_series_rows(3, 5) == 3:5
    @test PortfolioOptimisers.windowed_series_rows(3, 2) == 1:2
    @test isempty(PortfolioOptimisers.windowed_series_rows([4, 6], 3))
    # A plain inner estimator on a panel reduces each fit to its own Coverage Universe.
    Sp = PortfolioOptimisers.variance_series(WindowedVariance(; window = 40), X, pnl)
    @test isequal(Sp[end, :], vec(var(WindowedVariance(; window = 40), X, pnl)))
end

@testset "The windowed moments at parity with the oracle's window" begin
    # `parity_small_panel()` has a late listing, a delisting, a relisting and a holiday. The
    # stored files are the oracle's batch fit with `window_size`, and its windowed batch fit
    # on each prefix `1:t`, which is row `t` of our series. Measured maxrel 6.8e-16 or less.
    fx = parity_small_panel()
    X, pnl = fx.rd.X, fx.rd.pnl
    load(c, k) = parity_load("WindowedMoments", c, k)
    decay = 2.0^(-1 / 10)
    @testset "$(c)" for (c, n, centred) in (("W30HL10", 30, true), ("W12HL10", 12, true),
                                            ("W30HL10Unc", 30, false))
        me = WindowedExpectedReturns(;
                                     me = ExpWeightedExpectedReturns(; decay, min_obs = 5),
                                     window = n)
        vw = WindowedVariance(; ve = ExpWeightedVariance(; decay, min_obs = 5, centred),
                              window = n)
        cw = WindowedCovariance(; ce = ExpWeightedCovariance(; decay, min_obs = 5, centred),
                                window = n)
        @test parity_compare(vec(mean(me, X, pnl)), vec(load(c, "Mu")); name = "$c mu").ok
        @test parity_compare(vec(var(vw, X, pnl)), vec(load(c, "Var")); name = "$c var").ok
        # Every asset shares one history over the last rows, so the covariance matches entry by
        # entry; it compares against its largest entry because of the cancellation of its small
        # off-diagonal entries.
        @test parity_compare(cov(cw, X, pnl), load(c, "Cov"); scale = :array,
                             name = "$c cov").ok
        @test parity_compare(PortfolioOptimisers.variance_series(vw, X, pnl),
                             load(c, "VarSeries"); name = "$c var series").ok
        @test parity_compare(PortfolioOptimisers.variance_series(cw, X, pnl),
                             load(c, "CovSeries"); name = "$c cov series").ok
    end
end

@testset "A windowed variance in the variance slot of the prior" begin
    fx = parity_small_panel()
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    ve = ExpWeightedVariance(; decay = 2.0^(-1 / 10), min_obs = 5, centred = true)
    fit(v, wa = MarketCapWeights()) = prior(CrossSectionalFactorPrior(; factors = factors,
                                                                      minra = 5, wa = wa,
                                                                      ve = v), fx.rd)
    base = fit(ve)
    # A window longer than the sample keeps every row, so the prior is unchanged. Before the
    # fix the wrapper raised a `DimensionMismatch` on the prior's masks.
    @test isequal(fit(WindowedVariance(; ve = ve, window = 1000)).rr.vs, base.rr.vs)
    # A window of 30 rows changes the history after its first 30 rows only.
    pr = fit(WindowedVariance(; ve = ve, window = 30))
    @test isequal(pr.rr.vs[1:30, :], base.rr.vs[1:30, :])
    @test !isequal(pr.rr.vs[31:end, :], base.rr.vs[31:end, :])
    # The second pass of the regression weights reads the same series.
    wa = BlendedInverseVarianceWeights(; lambda = 0.5)
    @test isequal(fit(WindowedVariance(; ve = ve, window = 1000), wa).rr.rw,
                  fit(ve, wa).rr.rw)
end
