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
    # on each prefix `1:t`, which is row `t` of our series. Measured on 2026-10-07, cell by
    # cell: the variances are bit-equal, the means differ by maxrel 2.2e-16, the covariances by
    # 5.6e-16 and the two series by 6.8e-16, one to three ulps of a sum taken in another order.
    # `1e-14` leaves room for a host that orders the sums otherwise.
    fx = parity_small_panel()
    X, pnl = fx.rd.X, fx.rd.pnl
    load(c, k) = parity_load("WindowedMoments", c, k)
    decay = 2.0^(-1 / 10)
    # "Unc" is the oracle's estimated location, which starts at zero and is not divided by its
    # weight: `ZeroStartCentring()` since #1507 (ADR 0190).
    @testset "$(c)" for (c, n, centring) in
                        (("W30HL10", 30, PreCentred()), ("W12HL10", 12, PreCentred()),
                         ("W30HL10Unc", 30, ZeroStartCentring()))
        me = WindowedExpectedReturns(;
                                     me = ExpWeightedExpectedReturns(; decay, min_obs = 5),
                                     window = n)
        vw = WindowedVariance(; ve = ExpWeightedVariance(; decay, min_obs = 5, centring),
                              window = n)
        cw = WindowedCovariance(;
                                ce = ExpWeightedCovariance(; decay, min_obs = 5, centring),
                                window = n)
        @test parity_compare(vec(mean(me, X, pnl)), vec(load(c, "Mu")); rtol = 1e-14,
                             name = "$c mu").ok
        @test parity_compare(vec(var(vw, X, pnl)), vec(load(c, "Var")); rtol = 1e-14,
                             name = "$c var").ok
        # Every asset shares one history over the last rows, so the covariance matches entry by
        # entry. No off-diagonal entry cancels on this fixture, so it compares cell by cell.
        @test parity_compare(cov(cw, X, pnl), load(c, "Cov"); rtol = 1e-14, name = "$c cov").ok
        @test parity_compare(PortfolioOptimisers.variance_series(vw, X, pnl),
                             load(c, "VarSeries"); rtol = 1e-14, name = "$c var series").ok
        @test parity_compare(PortfolioOptimisers.variance_series(cw, X, pnl),
                             load(c, "CovSeries"); rtol = 1e-14, name = "$c cov series").ok
    end
end

@testset "A windowed variance in the variance slot of the prior" begin
    fx = parity_small_panel()
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    ve = ExpWeightedVariance(; decay = 2.0^(-1 / 10), min_obs = 5, centring = PreCentred())
    fit(v, wa = MarketCapWeights()) = prior(CrossSectionalFactorPrior(; lambda = 1,
                                                                      factors = factors,
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

#=
The window rule of a windowed estimator (#1469, ADR 0193, amendment of ADR 0039).

A `RollingWindow`, the default, keeps the last rows of every fit, so on the online seam its host
refits it over the carried rows and the window ends at the last row. A `SeedWindow` keeps the last
rows of the first fit alone: its first fold cuts the block, folds it into the inner estimator, and
spends the window, so every later fold passes every row. That is the oracle's rule for the window
of its EW moments, and the stored files pin it over a stream of batches inside the prior.
=#
const SW_DECAY = 2.0^(-1 / 10)
sw_blocks(e) = [(e[b] + 1):e[b + 1] for b in 1:(length(e) - 1)]
function sw_fold(est, X, r; active_mask = nothing)
    if length(r) == 1
        m = isnothing(active_mask) ? (;) : (; active_mask = vec(active_mask))
        return partial_fit!(est, X[r[1], :]; m...)
    end
    m = isnothing(active_mask) ? (;) : (; active_mask = active_mask)
    return partial_fit!(est, X[r, :]; m...)
end

@testset "The two window rules agree in a batch fit" begin
    X, amsk, pnl = windowed_panel_fixture()
    me = ExpWeightedExpectedReturns(; decay = WM_DECAY, min_obs = 1)
    ce = ExpWeightedCovariance(; decay = WM_DECAY, min_obs = 1)
    ve = ExpWeightedVariance(; decay = WM_DECAY, min_obs = 1)
    @test WindowedExpectedReturns().rule === RollingWindow()
    @test WindowedCovariance().rule === RollingWindow()
    @test WindowedVariance().rule === RollingWindow()
    @test WindowedCoskewness().rule === RollingWindow()
    @test WindowedCokurtosis().rule === RollingWindow()
    cases = [(mean, rule -> WindowedExpectedReturns(; me = me, window = 40, rule = rule)),
             (cov, rule -> WindowedCovariance(; ce = ce, window = 40, rule = rule)),
             (var, rule -> WindowedVariance(; ve = ve, window = 40, rule = rule)),
             (coskewness, rule -> WindowedCoskewness(; window = 40, rule = rule)),
             (cokurtosis, rule -> WindowedCokurtosis(; window = 40, rule = rule))]
    for (f, make) in cases
        @test isequal(f(make(SeedWindow()), X, pnl), f(make(RollingWindow()), X, pnl))
    end
    # A plain co-moment refuses a gapped matrix, so the mask keyword reaches the three
    # mask-aware inner estimators alone.
    for (f, make) in cases[1:3]
        @test isequal(f(make(SeedWindow()), X; active_mask = amsk),
                      f(make(RollingWindow()), X; active_mask = amsk))
    end
end

@testset "A rolling window refits over the carried rows on the online seam" begin
    X = 0.01 .* randn(StableRNG(1469), 90, 4)
    me = WindowedExpectedReturns(; me = ExpWeightedExpectedReturns(; decay = SW_DECAY),
                                 window = 30)
    ce = WindowedCovariance(; ce = ExpWeightedCovariance(; decay = SW_DECAY), window = 30)
    pe = EmpiricalPrior(; me = me, ce = ce)
    # A rolling wrapper does not fold, so the carry of the prior refits it, and every read-out
    # equals the batch fit over every row received.
    @test !PortfolioOptimisers.supports_partial_fit(me)
    for r in sw_blocks([0, 40, 47, 48, 90])
        pe = sw_fold(pe, X, r)
        t = last(r)
        pr = prior(pe)
        @test isequal(pr.mu, vec(mean(me, X[1:t, :])))
        @test isequal(pr.sigma, cov(ce, X[1:t, :]))
        @test pe.me.window == 30
    end
end

@testset "A seed window folds every row after the window of the first fit" begin
    fx = parity_small_panel()
    X, am = fx.rd.X, fx.rd.pnl.amsk
    T, N = size(X)
    mi = ExpWeightedExpectedReturns(; decay = SW_DECAY, min_obs = 5)
    ci = ExpWeightedCovariance(; decay = SW_DECAY, min_obs = 5, centring = PreCentred())
    vi = ExpWeightedVariance(; decay = SW_DECAY, min_obs = 5, centring = PreCentred())
    # The read-out after a stream is the batch fit of the inner estimator over the window of
    # the first block and every later row, the rows `s:t`. The fold and the batch fit run the
    # same recursion, so the two are equal.
    for (edges, n) in
        (([0, 45, 52, 53, 66, T], 30), (vcat([0, 40], 41:T), 12), ([0, 45, T], 100))
        me = WindowedExpectedReturns(; me = mi, window = n, rule = SeedWindow())
        ce = WindowedCovariance(; ce = ci, window = n, rule = SeedWindow())
        vw = WindowedVariance(; ve = vi, window = n, rule = SeedWindow())
        pe = EmpiricalPrior(; me = me, ce = ce)
        s = max(1, edges[2] - n + 1)
        for r in sw_blocks(edges)
            pe = sw_fold(pe, X, r; active_mask = am[r, :])
            vw = sw_fold(vw, X, r; active_mask = am[r, :])
            t = last(r)
            pr = prior(pe)
            @test isequal(pr.mu, vec(mean(mi, X[s:t, :]; active_mask = am[s:t, :])))
            @test isequal(pr.sigma, cov(ci, X[s:t, :]; active_mask = am[s:t, :]))
            @test isequal(vec(var(vw)), vec(var(vi, X[s:t, :]; active_mask = am[s:t, :])))
        end
        # The first fold spends the window and keeps the rule.
        @test isnothing(pe.me.window) && isnothing(pe.ce.window) && isnothing(vw.window)
        @test pe.ce.rule === SeedWindow()
    end
    # `dims = 2` folds the columns, and an index-vector window cuts the first block by index.
    ce = WindowedCovariance(; ce = ci, window = 30, rule = SeedWindow())
    @test isequal(cov(partial_fit!(ce, permutedims(X); dims = 2,
                                   active_mask = permutedims(am))),
                  cov(partial_fit!(ce, X; active_mask = am)))
    idx = [3, 10, 20, 44, 45]
    ci2 = partial_fit!(WindowedCovariance(; ce = ci, window = idx, rule = SeedWindow()),
                       X[1:45, :]; active_mask = am[1:45, :])
    ci2 = partial_fit!(ci2, X[46:T, :]; active_mask = am[46:T, :])
    rows = vcat(idx, 46:T)
    @test isequal(cov(ci2), cov(ci, X[rows, :]; active_mask = am[rows, :]))
end

@testset "A seed window on the higher moments and in a high order prior" begin
    X = 0.01 .* randn(StableRNG(14690), 60, 3)
    ske = WindowedCoskewness(; window = 20, rule = SeedWindow())
    kte = WindowedCokurtosis(; window = 20, rule = SeedWindow())
    me = WindowedExpectedReturns(; window = 20, rule = SeedWindow())
    ce = WindowedCovariance(; ce = Covariance(), window = 20, rule = SeedWindow())
    pe = HighOrderPriorEstimator(; pe = EmpiricalPrior(; me = me, ce = ce), ske = ske,
                                 kte = kte)
    for r in sw_blocks([0, 30, 31, 60])
        pe = sw_fold(pe, X, r)
    end
    pr = prior(pe)
    rows = 11:60
    sk, V = coskewness(Coskewness(), X[rows, :])
    @test isapprox(pr.sk, sk; rtol = 1e-12)
    @test isapprox(pr.V, V; rtol = 1e-12)
    @test isapprox(pr.kt, cokurtosis(Cokurtosis(), X[rows, :]); rtol = 1e-12)
    @test isapprox(pr.pr.mu, vec(mean(SimpleExpectedReturns(), X[rows, :])); rtol = 1e-12)
    @test isapprox(pr.pr.sigma, cov(Covariance(), X[rows, :]); rtol = 1e-12)
    # The one-argument reads forward to the inner estimator.
    sk1, V1 = coskewness(partial_fit!(ske, X))
    sk2, V2 = coskewness(Coskewness(), X[41:60, :])
    @test isapprox(sk1, sk2; rtol = 1e-12) && isapprox(V1, V2; rtol = 1e-12)
    @test isapprox(cokurtosis(partial_fit!(kte, X)), cokurtosis(Cokurtosis(), X[41:60, :]);
                   rtol = 1e-12)
    # Found by #1469: a high order prior read `pe.ske.mp`, which a windowed coskewness does not
    # hold, so its batch fit raised a `FieldError`. It reads the processor of the inner
    # estimator now, which builds `V`.
    hb = prior(HighOrderPriorEstimator(; ske = WindowedCoskewness(; window = 20)), X)
    sk0, V0 = coskewness(Coskewness(), X[41:60, :])
    @test hb.skmp === Coskewness().mp
    @test isapprox(hb.sk, sk0; rtol = 1e-12) && isapprox(hb.V, V0; rtol = 1e-12)
    @test PortfolioOptimisers.coskewness_processor(WindowedCoskewness()) === Coskewness().mp
end

@testset "A refit and a fold that cannot honour it refuse a seed window" begin
    X = 0.01 .* randn(StableRNG(14691), 40, 3)
    seed(ce) = WindowedCovariance(; ce = ce, window = 10, rule = SeedWindow())
    ew = ExpWeightedCovariance(; decay = SW_DECAY)
    # `Online` refits its estimator at each step, and a refit has no first fit to remember.
    err = try
        Online(EmpiricalPrior(; ce = seed(ew)))
    catch e
        e
    end
    @test err isa ArgumentError && occursin("`ce.rule`", err.msg)
    me = WindowedExpectedReturns(; me = ExpWeightedExpectedReturns(), rule = SeedWindow())
    @test_throws ArgumentError Online(EmpiricalPrior(; me = me))
    @test PortfolioOptimisers.seed_window_path(EmpiricalPrior(; me = me)) == "me.rule"
    @test isnothing(PortfolioOptimisers.seed_window_path(EmpiricalPrior()))
    @test Online(EmpiricalPrior(; ce = WindowedCovariance(; ce = ew))) isa Online
    # A fold refuses observation weights and an inner estimator that does not fold.
    wtd = WindowedCovariance(; ce = ew, w = StatsBase.eweights(40, 0.1),
                             rule = SeedWindow())
    @test_throws ArgumentError partial_fit!(wtd, X)
    semi = seed(Covariance(; alg = SemiMoment()))
    @test PortfolioOptimisers.supports_partial_fit(semi)
    @test_throws ArgumentError partial_fit!(EmpiricalPrior(; ce = semi), X)
end

@testset "A seed window at parity with the oracle over a stream of batches" begin
    # The oracle's empirical prior, its EW mean and covariance under a window, folded over
    # a stream of batches, and its EW variance folded beside it. Row `b` of each stored file is
    # the read-out after batch `b`. The covariance compares raw states: the library divides
    # each pair by the weight that the pair holds (ADR 0181, #1420), so the raw state of the
    # oracle is its output times `w * w'`, with `w` the root of the diagonal weights, and the
    # oracle runs without its projection to a positive definite matrix. Under a horizon only
    # the diagonal of `sigma` reads no pair weight. Measured on 2026-10-07, cell by cell over
    # every batch of the six cases: mu maxrel 3.5e-16, variance 3.9e-16, raw covariance
    # 6.0e-16 and horizon diagonal 6.3e-16, one to three ulps, so each takes `1e-14`. The
    # horizon mu is `exp(h mu + h sigma_ii / 2) - 1`: the subtraction of one leaves an error of
    # an ulp of one, maxabs 2.2e-16, which is maxrel 2.1e-12 on a mean near zero. It takes an
    # absolute tolerance of a few ulps of one.
    fx = parity_small_panel()
    X, am = fx.rd.X, fx.rd.pnl.amsk
    T, N = size(X)
    streams = Dict("Blocks" => [0, 45, 52, 53, 66, T], "Rows" => vcat([0, 40], 41:T))
    load(c, k) = parity_load("SeedWindow", c, k)
    # `W30BlocksUnc` pins the oracle's zero-start location, `ZeroStartCentring()` since #1507.
    @testset "$(c)" for (c, n, centring, h, st) in
                        (("W30Blocks", 30, PreCentred(), nothing, "Blocks"),
                         ("W12Blocks", 12, PreCentred(), nothing, "Blocks"),
                         ("W30BlocksUnc", 30, ZeroStartCentring(), nothing, "Blocks"),
                         ("W30Rows", 30, PreCentred(), nothing, "Rows"),
                         ("W100Blocks", 100, PreCentred(), nothing, "Blocks"),
                         ("W30BlocksH5", 30, PreCentred(), 5, "Blocks"))
        mi = ExpWeightedExpectedReturns(; decay = SW_DECAY, min_obs = 5)
        ci = ExpWeightedCovariance(; decay = SW_DECAY, min_obs = 5, centring = centring)
        vi = ExpWeightedVariance(; decay = SW_DECAY, min_obs = 5, centring = centring)
        me = WindowedExpectedReturns(; me = mi, window = n, rule = SeedWindow())
        ce = WindowedCovariance(; ce = ci, window = n, rule = SeedWindow())
        vw = WindowedVariance(; ve = vi, window = n, rule = SeedWindow())
        pe = if isnothing(h)
            EmpiricalPrior(; me = me, ce = ce)
        else
            EmpiricalPrior(; me = me, ce = ce, horizon = h)
        end
        M, C, V = load(c, "MuSeries"), load(c, "CovSeries"), load(c, "VarSeries")
        for (b, r) in enumerate(sw_blocks(streams[st]))
            pe = sw_fold(pe, X, r; active_mask = am[r, :])
            vw = sw_fold(vw, X, r; active_mask = am[r, :])
            pr = prior(pe)
            Ob = C[((b - 1) * N + 1):(b * N), :]
            @test parity_compare(pr.mu, M[b, :]; rtol = 1e-14,
                                 atol = isnothing(h) ? 0.0 : 1e-15, name = "$c mu $b").ok
            @test parity_compare(vec(var(vw)), V[b, :]; rtol = 1e-14, name = "$c var $b").ok
            if isnothing(h)
                s = pe.ce.ce.cache
                f = findall(isfinite, LinearAlgebra.diag(Ob))
                # An empty selection would compare nothing.
                @test !isempty(f)
                w = sqrt.(LinearAlgebra.diag(s.weight))
                @test parity_compare(s.covariance[f, f], (Ob .* (w * transpose(w)))[f, f];
                                     rtol = 1e-14, name = "$c raw cov $b").ok
            else
                @test parity_compare(LinearAlgebra.diag(pr.sigma), LinearAlgebra.diag(Ob);
                                     rtol = 1e-14, name = "$c diag sigma $b").ok
            end
        end
    end
end
