#=
Parity of `factor_attribution` (#1388, map #1375): predicted, realised and rolling, by factor, by
family and by asset, with the standard errors, the observed factors, the time-series route and the
annualisation.

The oracle takes bare arrays. Each case fits a prior here, and the oracle reads the arrays the
attribution reads off its block, untrimmed, so it applies its own exposure lag: the loadings, the
factor covariance and means, the idiosyncratic covariance and the orthogonal mean on the predicted
side; the exposure history, the factor and idiosyncratic returns, the regression weights, the
idiosyncratic variances, the family re-basis and the net portfolio series on the realised side.
`Parity_factor_attribution_<Case>_<Output>.csv.gz` is the oracle output, packed as `fa_pack` packs
ours: `Components` (systematic, idiosyncratic, unattributed, total), `Factors`, `Families` (sorted
by label), `Assets`, `AssetFactorVol` and `AssetFactorMu`, one block of rows per window.

The fixture is `grid_fixture(parity_small_panel())`: asset 2 lists late, 3 delists, 4 relists and
is in the warm-up of its variance, and 5 has a holiday. The base model is a market factor and two
passthrough styles, `minra = 5`, at the oracle's defaults (`GRID_PE`, `GRID_VE`).

Every output is at parity cell by cell, with the oracle's `NaN` pattern, measured 3.5e-13 at most
on the predicted side and 6.2e-14 on the realised side, except the cells below.

  - Deliberate. The predicted side states an unattributed remainder, the gap between `pr.sigma`
    and the model (ADR 0113), where the oracle states none. It is at rounding level here, and the
    four components sum to the total.
  - Changed (#1515, R95). The weight spread divides by `T - ddof`, `T - 1` by default, as the
    exposure spread does: ours is the oracle's times `sqrt(T / (T - 1))`, and `ddof = 0` is at
    parity. A constant weight states a spread of zero, at parity.
  - Changed (#1515, R96, R65). A holding in a non-investable asset warns, and `strict = true`
    refuses it. The default `EntrywiseUnknown()` keeps the finite loadings of the relisted asset,
    so the exposures and the systematic numbers are exact, and gives `NaN` for every number that
    reads its unknown idiosyncratic variance. `ZeroUnknown()` reads the unknown entries as zero and
    is at parity (`PredHeld`), as is its asset axis on every case.
  - Changed (#1515, R97). A held pair with no return at an active cell is a holiday and fills zero
    with no message, so a returns result whose panel holds a holiday is at parity with no warning.
  - Changed (#1515, R98). A static loadings matrix reads every row whatever the lag, as the oracle
    does. `trim = true` cuts it by the lag.
  - Defect found and fixed (#1388). A pair with no idiosyncratic return adds nothing to the net
    series, and the realised decomposition kept its systematic return `w B f`, with the opposite
    amount in the remainder. It is now zero in every component (`attribution_zero_inactive`), at
    parity (`RealBase`, `RealHist`, `RollBase`).
  - Better. A remainder that is constant within the round-off of its terms at every observation is
    that constant, with no volatility and a `NaN` correlation: zero on a cross-sectional fit, the
    intercept on the time-series route. The oracle's volatility is round-off, 1e-18, with a `NaN`
    correlation from an absolute threshold on the volatility, which depends on the units of the
    returns.
  - Better. The standalone volatility, mean and correlation of an asset read the block's own
    entries on the predicted side, and the active pairs of the asset on the realised side. The
    oracle states a zero mean and volatility for an asset without loadings, a volatility without the
    unknown idiosyncratic variance of the relisted asset, and on the realised side the moments of a
    series with a zero at every pair the asset had no return.
  - Better. A standard error that reads an unknown idiosyncratic variance, in the warm-up of the
    variance estimate, is `NaN` (`SeWarmup`). The oracle drops the pair from the Gram matrix, which
    is the covariance of a regression the fit did not run.
  - Better (#1580). A standard error whose sandwich gives a Leverage-One Pair a coefficient that
    is not zero is `NaN` under the default: asset 3 is alone in Utilities until it delists, so the
    market factor and the sibling levels of `SeFam`, `SeFamTwo`, `SeRankDef` and `RollSeFam` read
    its variance, which the fit does not identify. The oracle reads the plug-in variance, and the
    stored cases run under `KindwiseUnknown(; leverage = ZeroUnknown())`, which is at parity.
  - Better. An observed factor is not estimated by the regression, so it leaves the sandwich
    whatever its family label (`SeMacro`). The oracle leaves it only under the label "currency":
    relabelled so (`SeMacroObs`), the oracle equals ours.
  - Better. The oracle's rolling family axis sorts each window by its variance shares and keeps
    the labels of the first window, so a window whose order changes carries the wrong labels
    (`RollSeFam`, windows 4 and 5). Ours sorts every window by label.

The time-series route (`PredTs`, `RealTs`) names no family. The bare-array methods (#1404) take
the family labels from the caller, and `PredTsFam` and `RealTsFam` pin them on that route. A case
whose name carries `Arr` is the bare-array method on the arrays of the block, read against the
stored file of the case without the tag. Without the totals of the prior the model closes the total,
so the predicted remainder is an exact zero where the oracle states none.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))

const FA_UNIT = "factor_attribution"

# Our result, packed in the column order of the stored oracle outputs.
function fa_pack(fa::FactorAttributionResult)
    nz(x) = isnothing(x) ? NaN : x
    col(x, n) = isnothing(x) ? fill(NaN, n) : collect(Float64, x)
    comp(c) = [c.vol c.vol_contrib c.pct_var c.mu_contrib c.corr nz(c.mu_se)]
    f = fa.fbd
    K = length(f.exposure)
    out = Dict{String, Matrix{Float64}}("Components" => vcat(comp(fa.sys), comp(fa.idio),
                                                             comp(fa.unattr), comp(fa.total)),
                                        "Factors" =>
                                            hcat(col(f.exposure, K), col(f.exposure_std, K),
                                                 col(f.vol, K), col(f.corr, K),
                                                 col(f.vol_contrib, K), col(f.pct_var, K),
                                                 col(f.mu, K), col(f.mu_contrib, K),
                                                 col(f.mu_se, K)))
    if !isnothing(fa.fmbd)
        m = fa.fmbd
        n = length(m.exposure)
        out["Families"] = hcat(col(m.exposure, n), col(m.exposure_std, n),
                               col(m.vol_contrib, n), col(m.pct_var, n),
                               col(m.mu_contrib, n), col(m.mu_se, n))
    end
    if !isnothing(fa.abd)
        a = fa.abd
        N = length(a.weight)
        out["Assets"] = hcat(col(a.weight, N), col(a.weight_std, N), col(a.vol, N),
                             col(a.mu, N), col(a.corr, N), col(a.sys_vol_contrib, N),
                             col(a.idio_vol_contrib, N), col(a.vol_contrib, N),
                             col(a.pct_var, N), col(a.sys_mu_contrib, N),
                             col(a.idio_mu_contrib, N), col(a.mu_contrib, N))
        out["AssetFactorVol"] = Matrix{Float64}(fa.afc.vol_contrib)
        out["AssetFactorMu"] = Matrix{Float64}(fa.afc.mu_contrib)
    end
    return out
end
function fa_pack(v::AbstractVector{<:FactorAttributionResult})
    ps = fa_pack.(v)
    return Dict(k => reduce(vcat, [p[k] for p in ps]) for k in keys(ps[1]))
end

# The rows of `sel` in each block of `n` rows.
fa_rows(A, n, sel) = [(b - 1) * n + r for b in 1:(size(A, 1) ÷ n) for r in sel]

# Random weights on the investable assets, and the same with every asset that misses a return at
# some observation set to zero.
function fa_inv_weights(pr; seed::Integer = 1388)
    N = length(pr.mu)
    m = PortfolioOptimisers.investable_mask(pr)
    w = zeros(N)
    act = isnothing(m) ? (1:N) : findall(m)
    w[act] = rand(StableRNG(seed), length(act))
    return w ./ sum(w)
end
function fa_clean_weights(pr, X)
    w = fa_inv_weights(pr)
    w[[any(!isfinite, view(X, :, i)) for i in axes(X, 2)]] .= 0
    return w ./ sum(w)
end

# The moments of each asset's model return over its active pairs, from the block directly.
function fa_active_moments(pr, ret, W, T)
    PO = PortfolioOptimisers
    al = PO.attribution_align(pr.rr, pr, T)
    r = ret[al.rows]
    n, N = size(al.eps)
    A = [dot(PO.attribution_slice(al.B, t)[i, :], al.f[t, :]) + al.eps[t, i]
         for t in 1:n, i in 1:N]
    a(i) = A[al.act[:, i], i]
    y(i) = r[al.act[:, i]]
    return (; vol = [std(a(i)) for i in 1:N], mu = [mean(a(i)) for i in 1:N],
            corr = [std(a(i)) > 0 ? cor(a(i), y(i)) : NaN for i in 1:N],
            full = [all(view(al.act, :, i)) for i in 1:N], n = n)
end

# The time-series route of #783, as `test_21b_factor_attribution.jl` builds it.
function fa_ts_prior(; n_assets::Integer = 8, n_factors::Integer = 3,
                     n_observations::Integer = 90, seed::Integer = 783_001)
    rng = StableRNG(seed)
    F = randn(rng, n_observations, n_factors) ./ 100
    B = randn(rng, n_assets, n_factors)
    X = F * transpose(B) .+ randn(rng, n_observations, n_assets) ./ 200
    rd = ReturnsResult(; nx = ["a$(i)" for i in 1:n_assets], X = X,
                       nf = ["f$(i)" for i in 1:n_factors], F = F)
    return prior(FactorPrior(), rd), rd
end

# The arrays the predicted side reads off a block, and those the realised side reads, positional
# and keyword, as the bare-array methods take them.
function fa_pred_arrays(pr)
    PO = PortfolioOptimisers
    return (pr.rr.M, pr.fpr.sigma, PO.attribution_idiosyncratic_covariance(pr.rr)),
           (; mu_f = pr.fpr.mu, b = pr.rr.b, fam = PO.attribution_families(pr.rr))
end
function fa_real_arrays(pr)
    b = PortfolioOptimisers.attribution_block_arrays(pr.rr, pr)
    return (b.B, b.f, b.eps),
           (; lag = b.lag, rw = b.rw, vs = b.vs, fcb = b.fcb, observed = b.no, fam = b.fam)
end

@testset "Factor attribution at parity with the oracle (#1388)" begin
    PO = PortfolioOptimisers
    rd = grid_fixture(parity_small_panel())
    X = rd.X
    T, N = size(X)
    base = ["market" => ConstantExposure(), "style1" => grid_pass("style1"),
            "style2" => grid_pass("style2")]
    ind = ["market" => ConstantExposure(),
           "industry" => OneHotExposure(; field = "industry", family = "industry"),
           "style1" => grid_pass("style1"), "style2" => grid_pass("style2")]
    ve1 = RegimeAdjustedExpWeightedVariance(; centring = PreCentred(),
                                            debias = RawStatistic(),
                                            regime_lohi_mult = (0.7, 1.6), min_val = 1e-12,
                                            min_obs = 1)
    lz = KindwiseUnknown(; leverage = ZeroUnknown())
    fit(; kw...) = prior(CrossSectionalFactorPrior(; lambda = 1, factors = base, minra = 5,
                                                   pe = GRID_PE, ve = GRID_VE, kw...), rd)
    load(c, o) = parity_load(FA_UNIT, replace(c, "Arr" => ""), o)
    cmp(a, b, name; kw...) = parity_compare(a, b; name = name, kw...).ok
    prb = fit()
    wb = fa_inv_weights(prb)
    wc = fa_clean_weights(prb, X)
    Wh = rand(StableRNG(13881), T, N) .* isfinite.(X)
    Wh[:, 2:5] .= 0
    Wh ./= sum(Wh; dims = 2)
    reth = [dot(Wh[t, :], ifelse.(isfinite.(X[t, :]), X[t, :], 0.0)) for t in 1:T]
    net(w) = PO.attribution_net_returns(w, X, nothing, false)
    # Ours, as a case of `fa_pack` matrices, and the return series the realised side read.
    cases = Dict{String, Any}()
    quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
    quiet() do
        cases["PredBase"] = (fa_pack(factor_attribution(wb, prb; assets = true)), prb)
        cases["PredPpy"] = (fa_pack(factor_attribution(wc, prb; assets = true, ppy = 252)),
                            prb)
        prf = fit(; factors = ind, families = ["industry" => nothing])
        cases["PredFam"] = (fa_pack(factor_attribution(fa_clean_weights(prf, X), prf;
                                                       assets = true)), prf)
        cases["RealBase"] = (fa_pack(factor_attribution(wb, prb, X; assets = true)), prb,
                             wb, net(wb))
        cases["RealPpy"] = (fa_pack(factor_attribution(wc, prb, X; assets = true,
                                                       ppy = 252)), prb, wc, net(wc))
        cases["RealHist"] = (fa_pack(factor_attribution(Wh, prb, reth; assets = true)), prb,
                             Wh, reth)
        cases["RollBase"] = (fa_pack(factor_attribution(wb, prb, X, 30; step = 5,
                                                        assets = true)), prb, wb, net(wb))
        cases["RollHist"] = (fa_pack(factor_attribution(Wh, prb, reth, 30; step = 7,
                                                        assets = true)), prb, Wh, reth)
        cases["SeWarmup"] = (fa_pack(factor_attribution(wc, prb, X; se = true)), prb)
        for (c, p) in (("SeFull", fit(; ve = ve1)),
                       ("SeFam", fit(; ve = ve1, factors = ind, families = ["industry" => nothing])),
                       ("SeFamTwo",
                        fit(; ve = ve1,
                            factors = [ind;
                                       "region" =>
                                           OneHotExposure(; field = "currency", family = "region")],
                            families = ["industry" => nothing, "region" => nothing])),
                       ("SeRankDef", fit(; ve = ve1, factors = ind)),
                       ("SeCurrency", fit(; ve = ve1, factors = [base; "ccy" => CurrencyExposure()])),
                       ("SeMacro",
                        fit(; ve = ve1,
                            factors = [base;
                                       "macro" => ObservedExposure(;
                                                                   xe = grid_pass("macro_beta";
                                                                                  family = "macro"),
                                                                   series = "MACRO", family = "macro")])))
            # The oracle reads the plug-in variance of a Leverage-One Pair, asset 3 alone in
            # Utilities. `lz` does too; the default is a verdict of its own (#1580).
            w = fa_clean_weights(p, X)
            cases[c] = (fa_pack(factor_attribution(w, p, X; se = true, unknown = lz)), p)
            if c == "SeFam"
                cases["RollSeFam"] = (fa_pack(factor_attribution(w, p, X, 40; step = 9,
                                                                 se = true, unknown = lz)),
                                      p)
            end
        end
        cases["SeMacroObs"] = cases["SeMacro"]
    end
    prt, rdt = fa_ts_prior()
    wt = collect(1.0:length(prt.mu))
    wt ./= sum(wt)
    cases["PredTs"] = (fa_pack(factor_attribution(wt, prt; assets = true)), prt)
    cases["RealTs"] = (fa_pack(factor_attribution(wt, prt, rdt.X; assets = true)), prt, wt,
                       PO.attribution_net_returns(wt, rdt.X, nothing, false))
    # The bare-array methods on the arrays of the block (#1404). The predicted side takes no
    # totals, so the model is the total, as the oracle states it.
    wf = fa_clean_weights(cases["PredFam"][2], X)
    quiet() do
        for (c, w, kw) in
            (("PredBase", wb, (;)), ("PredPpy", wc, (; ppy = 252)), ("PredFam", wf, (;)),
             ("PredTs", wt, (;)))
            p = cases[c][2]
            a, o = fa_pred_arrays(p)
            cases[replace(c, "Pred" => "PredArr")] = (fa_pack(factor_attribution(w, a...;
                                                                                 o...,
                                                                                 assets = true,
                                                                                 kw...)), p)
        end
    end
    # The time-series route with families the caller gives.
    fam_ts = ["style", "macro", "style"]
    a, o = fa_pred_arrays(prt)
    cases["PredArrTsFam"] = (fa_pack(factor_attribution(wt, a...; o..., fam = fam_ts,
                                                        assets = true)), prt)
    a, o = fa_real_arrays(prt)
    rett = cases["RealTs"][4]
    cases["RealArrTsFam"] = (fa_pack(factor_attribution(wt, a..., rett; o..., fam = fam_ts,
                                                        assets = true)), prt, wt, rett)

    # The components: the systematic, the idiosyncratic and the total rows cell by cell. A
    # standard error the oracle states where ours is `NaN`, or differs, is a verdict of its own.
    se_better = ("SeWarmup", "SeMacro")
    @testset "The components, $(c)" for c in sort(collect(keys(cases)))
        our = cases[c][1]["Components"]
        orc = load(c, "Components")
        @test size(our) == size(orc)
        cols = c in se_better ? (1:5) : (1:6)
        rows = fa_rows(our, 4, (1, 2, 4))
        # The idiosyncratic mean of a time-series residual is round-off on both sides, so every
        # cell is also read against the scale of the total.
        sc = maximum(abs, filter(isfinite, our[fa_rows(our, 4, (4,)), 1:4]))
        # Measured maxabs 3.0e-13 at most over the cases.
        @test cmp(our[rows, cols], orc[rows, cols], "$(c) components"; atol = 1e-14 * sc)
        rem = fa_rows(our, 4, (3,))
        if startswith(c, "Pred")
            # The oracle states no predicted remainder. Ours is the gap between `pr.sigma` and
            # the model, at rounding level on a plain fit, and it closes the total.
            @test all(isnan, orc[rem, :])
            @test all(abs.(our[rem, 3]) .< 1e-11)
            # Without the totals of the prior the model is the total.
            startswith(c, "PredArr") && @test all(iszero, our[rem, 2:4])
            @test our[1, 2] + our[2, 2] + our[3, 2] ≈ our[4, 2] rtol = 1e-14
            @test our[1, 4] + our[2, 4] + our[3, 4] ≈ our[4, 4] rtol = 1e-14
        else
            # A remainder constant to round-off has no volatility on our side, and a volatility
            # of round-off on the oracle's. Measured maxabs 7.2e-16 at most.
            @test cmp(our[rem, 1:5], orc[rem, 1:5], "$(c) remainder"; atol = 1e-14 * sc)
        end
    end

    @testset "The factor axis, $(c)" for c in sort(collect(keys(cases)))
        our = cases[c][1]["Factors"]
        orc = load(c, "Factors")
        @test size(our) == size(orc)
        cols = c in se_better ? (1:8) : (1:9)
        # The spread of an exposure that does not move is round-off on our side, and zero on the
        # oracle's where the weights and the loadings are both static. Measured maxabs 6.7e-16
        # at most.
        sc = maximum(abs, our[:, 1])
        @test cmp(our[:, 2], orc[:, 2], "$(c) exposure spread"; atol = 1e-14 * sc)
        @test cmp(our[:, setdiff(cols, 2)], orc[:, setdiff(cols, 2)], "$(c) factors")
    end

    @testset "The family axis, $(c)" for c in sort(collect(keys(cases)))
        haskey(cases[c][1], "Families") || continue
        c == "RollSeFam" && continue
        our = cases[c][1]["Families"]
        orc = load(c, "Families")
        cols = c in se_better ? (1:5) : (1:6)
        sc = maximum(abs, our[:, 1])
        # Measured maxabs 1.1e-16 at most.
        @test cmp(our[:, 2], orc[:, 2], "$(c) family spread"; atol = 1e-14 * sc)
        @test cmp(our[:, setdiff(cols, 2)], orc[:, setdiff(cols, 2)], "$(c) families")
    end

    @testset "The asset axis, $(c)" for c in sort(collect(keys(cases)))
        haskey(cases[c][1], "Assets") || continue
        our = cases[c][1]["Assets"]
        orc = load(c, "Assets")
        Na = size(our, 1) ÷ (size(our, 1) ÷ length(cases[c][2].mu))
        @test cmp(cases[c][1]["AssetFactorVol"], load(c, "AssetFactorVol"), "$(c) afc vol")
        @test cmp(cases[c][1]["AssetFactorMu"], load(c, "AssetFactorMu"), "$(c) afc mu")
        # The contributions. The idiosyncratic mean of a time-series residual is round-off on
        # both sides, so it is read against the scale of the whole mean contribution. Measured
        # maxabs 4.2e-17 at most.
        @test cmp(our[:, [1, 6, 7, 8, 9, 10, 12]], orc[:, [1, 6, 7, 8, 9, 10, 12]],
                  "$(c) asset contributions")
        @test cmp(our[:, 11], orc[:, 11], "$(c) idiosyncratic mean";
                  atol = 1e-12 * maximum(abs, our[:, 12]))
        # The weight spread (#782, #1515).
        if startswith(c, "Pred")
            @test all(isnan, our[:, 2]) && all(isnan, orc[:, 2])
        elseif cases[c][3] isa AbstractVector
            @test all(iszero, our[:, 2]) && all(iszero, orc[:, 2])
        else
            n = c == "RollHist" ? 30 : length(cases[c][4]) - 2
            @test cmp(our[:, 2], orc[:, 2] .* sqrt(n / (n - 1)), "$(c) weight spread")
        end
        # The standalone numbers.
        pr = cases[c][2]
        if startswith(c, "Pred")
            imsk = PO.investable_mask(pr)
            inv = isnothing(imsk) ? trues(Na) : imsk
            @test cmp(our[inv, 3:5], orc[inv, 3:5], "$(c) standalone")
            for i in findall(!, inv)
                # The model states the mean of an asset whose loadings it states, and no
                # volatility for an asset whose idiosyncratic variance is unknown.
                if all(isfinite, pr.rr.M[i, :])
                    @test our[i, 4] ≈ orc[i, 4] rtol = 1e-12
                else
                    @test isnan(our[i, 4]) && iszero(orc[i, 4])
                end
                @test isnan(our[i, 3]) && isnan(our[i, 5])
            end
        elseif c in ("RealBase", "RealPpy", "RealHist")
            ppy = c == "RealPpy" ? 252 : 1
            m = fa_active_moments(pr, cases[c][4], cases[c][3], T)
            @test isapprox(our[:, 3], m.vol .* sqrt(ppy); nans = true, rtol = 1e-12)
            @test isapprox(our[:, 4], m.mu .* ppy; nans = true, rtol = 1e-12)
            @test isapprox(our[:, 5], m.corr; nans = true, rtol = 1e-12)
            @test cmp(our[m.full, 3:5], orc[m.full, 3:5], "$(c) standalone, active")
            @test !isapprox(our[.!m.full, 4], orc[.!m.full, 4]; nans = true)
        elseif occursin("Ts", c)
            @test cmp(our[:, 3:5], orc[:, 3:5], "$(c) standalone")
        end
    end

    @testset "Better: a standard error that reads an unknown variance is NaN" begin
        our = cases["SeWarmup"][1]
        @test isnan(our["Components"][1, 6]) && isnan(our["Components"][2, 6])
        @test all(isnan, our["Factors"][:, 9])
        @test all(isfinite, load("SeWarmup", "Factors")[:, 9])
    end

    @testset "Better: an observed factor leaves the sandwich whatever its label" begin
        our = cases["SeMacro"][1]
        # Parity with the oracle once it knows the factor is observed.
        @test cmp(our["Components"][[1, 2], 6], load("SeMacroObs", "Components")[[1, 2], 6],
                  "SeMacroObs error")
        @test cmp(our["Factors"][:, 9], load("SeMacroObs", "Factors")[:, 9],
                  "SeMacroObs factor errors")
        @test cmp(our["Families"][:, 6], load("SeMacroObs", "Families")[:, 6],
                  "SeMacroObs family errors")
        # Under its own label the oracle regresses on the observed factor too.
        @test isnan(our["Factors"][4, 9]) && isfinite(load("SeMacro", "Factors")[4, 9])
        @test !isapprox(our["Components"][1, 6], load("SeMacro", "Components")[1, 6];
                        rtol = 1e-3)
    end

    @testset "Better: a standard error that reads a Leverage-One Pair is NaN (#1580)" begin
        # Asset 3 is alone in Utilities until it delists, so `h1` marks it at rows 1 to 58 of the
        # block. The portfolio holds no asset 3, so its systematic error stays at parity. The zero-
        # sum constraint ties the market factor and the sibling levels to Utilities, and their
        # errors read the variance of the pair: measured ratios 1e-2 to 0.54, against 5.7e-18
        # at most for the systematic error and the styles.
        nanrows = Dict("SeFam" => ([1, 2, 3, 4], [1, 2]),
                       "SeFamTwo" => ([1, 2, 3, 4, 8, 9, 10], [1, 2, 3]),
                       "SeRankDef" => ([1, 2, 3, 4], [1, 2]))
        for (c, (fr, mr)) in nanrows
            p = cases[c][2]
            our = quiet(() -> fa_pack(factor_attribution(fa_clean_weights(p, X), p, X;
                                                         se = true)))
            orc = cases[c][1]
            @test findall(isnan, our["Factors"][:, 9]) == fr
            @test findall(isnan, our["Families"][:, 6]) == mr
            @test all(isfinite, orc["Factors"][:, 9]) &&
                  all(isfinite, orc["Families"][:, 6])
            # Every other cell is the number at parity with the oracle.
            @test isequal(our["Components"], orc["Components"])
            for (k, j) in (("Factors", 9), ("Families", 6))
                m = .!isnan.(our[k][:, j])
                @test isequal(our[k][:, 1:(j - 1)], orc[k][:, 1:(j - 1)])
                @test our[k][m, j] == orc[k][m, j]
            end
        end
        # Every rolling window holds marked rows, so each window loses the same four errors.
        p = cases["RollSeFam"][2]
        our = quiet(() -> fa_pack(factor_attribution(fa_clean_weights(p, X), p, X, 40;
                                                     step = 9, se = true)))
        orc = cases["RollSeFam"][1]
        @test findall(isnan, our["Factors"][:, 9]) ==
              [7 * (b - 1) + k for b in 1:5 for k in 1:4]
        @test isequal(our["Components"], orc["Components"])
        @test isequal(our["Factors"][:, 1:8], orc["Factors"][:, 1:8])
    end

    @testset "Better: each rolling window states the labels of its own families" begin
        our = cases["RollSeFam"][1]["Families"]
        orc = load("RollSeFam", "Families")
        nf = 3
        nw = size(our, 1) ÷ nf
        blk(b) = ((b - 1) * nf + 1):(b * nf)
        # The oracle orders each window by the size of its variance shares, and states the labels
        # of the first window alone. The stored file sorts every window by those labels, so its
        # window `b` is ours in the order `p[b]`, then in the order that sorts the first window.
        p = [sortperm(-abs.(our[blk(b), 4])) for b in 1:nw]
        for b in 1:nw
            r = blk(b)
            # The spread of the market family's constant exposure is round-off on both sides.
            # Measured maxabs 5.6e-16 at most.
            @test cmp(our[r[p[b][invperm(p[1])]], :], orc[r, :], "RollSeFam window $(b)";
                      atol = 1e-14 * maximum(abs, our[r, 1]))
        end
        # A window in another order than the first carries the wrong labels.
        @test any(p[b] != p[1] for b in 2:nw)
    end

    @testset "Changed (#1515): a held non-investable asset is decomposed entry by entry" begin
        M, Fm, muf = prb.rr.M, prb.fpr.sigma, prb.fpr.mu
        # Asset 4 relisted and is in the warm-up of its variance: its loadings are finite, and
        # its idiosyncratic variance is unknown. Asset 3 delisted, and has no loadings.
        @test all(isfinite, M[4, :])
        @test isnan(PO.attribution_idiosyncratic_covariance(prb.rr)[4])
        @test all(isnan, M[3, :])
        # The weights of the stored case `PredHeld` hold both assets.
        wh = copy(wb)
        wh[3] = 0.05
        wh[4] = 0.07
        wh ./= sum(wh)
        w4 = copy(wb)
        w4[4] = 0.07
        w4 ./= sum(w4)
        f4 = @test_logs (:warn, r"Assets \[4\] are not investable") match_mode = :any factor_attribution(w4,
                                                                                                         prb;
                                                                                                         assets = true)
        # The exposures, the systematic variance and the systematic mean are exact.
        g = transpose(ifelse.(iszero.(w4), 0.0, M)) * w4
        @test f4.fbd.exposure ≈ g rtol = 1e-14
        @test f4.sys.vol ≈ sqrt(dot(g, Fm * g)) rtol = 1e-14
        @test f4.sys.mu_contrib ≈ dot(g, muf) rtol = 1e-14
        @test f4.abd.sys_mu_contrib[4] ≈ w4[4] * dot(M[4, :], muf) rtol = 1e-14
        # Every number that reads the unknown variance is unknown.
        @test isnan(f4.total.vol) && isnan(f4.idio.vol) && isnan(f4.unattr.vol_contrib)
        @test isnan(f4.sys.vol_contrib) && isnan(f4.sys.pct_var) && isnan(f4.sys.corr)
        @test all(isnan, f4.fbd.vol_contrib) && all(isnan, f4.fbd.pct_var)
        @test isnan(f4.abd.idio_vol_contrib[4])
        # The mean of the relisted asset is stated, so the means are exact.
        @test isfinite(f4.total.mu_contrib) && isfinite(f4.unattr.mu_contrib)
        # An asset the portfolio does not hold contributes zero.
        z = iszero.(w4)
        @test all(iszero, f4.abd.sys_vol_contrib[z]) && all(iszero, f4.abd.vol_contrib[z])
        @test all(iszero, f4.afc.vol_contrib[z, :])
        # A held asset without loadings makes the exposures that read them unknown.
        fh = quiet(() -> factor_attribution(wh, prb))
        @test all(isnan, fh.fbd.exposure)
        # `ZeroUnknown()` on the arrays of the block is the oracle's predicted attribution.
        a, o = fa_pred_arrays(prb)
        zu = quiet(() -> fa_pack(factor_attribution(wh, a...; o..., assets = true,
                                                    unknown = ZeroUnknown())))
        orc = load("PredHeld", "Components")
        sc = maximum(abs, filter(isfinite, zu["Components"][[4], 1:4]))
        # Measured maxabs 2.2e-16.
        @test cmp(zu["Components"][[1, 2, 4], 1:5], orc[[1, 2, 4], 1:5],
                  "PredHeld components"; atol = 1e-14 * sc)
        @test all(iszero, zu["Components"][3, 2:4])
        for out in ("Factors", "Assets", "AssetFactorVol", "AssetFactorMu")
            @test cmp(zu[out], load("PredHeld", out), "PredHeld $(out)")
        end
        # The prior method reads its anchors, whose unknown entries read the model, so its
        # totals are the oracle's to round-off and its remainder is at rounding level.
        # Measured maxabs 2.1e-13.
        zp = quiet(() -> fa_pack(factor_attribution(wh, prb; assets = true,
                                                    unknown = ZeroUnknown())))
        @test cmp(zp["Components"][[1, 2, 4], 1:5], orc[[1, 2, 4], 1:5], "PredHeld prior";
                  atol = 1e-14 * sc)
        @test all(abs.(zp["Components"][3, 3]) .< 1e-11)
        @test_throws ArgumentError factor_attribution(wh, prb; strict = true)
        @test_throws ArgumentError factor_attribution(wh, prb; strict = true,
                                                      unknown = ZeroUnknown())
        @test_throws ArgumentError factor_attribution(wb, prb, X; strict = true)
    end

    @testset "Changed (#1515): `ZeroUnknown()` puts every standalone moment at parity" begin
        zu = (; assets = true, unknown = ZeroUnknown())
        prf = cases["PredFam"][2]
        zc = quiet() do
            return Dict("PredBase" => factor_attribution(wb, prb; zu...),
                        "PredPpy" => factor_attribution(wc, prb; ppy = 252, zu...),
                        "PredFam" => factor_attribution(wf, prf; zu...),
                        "RealBase" => factor_attribution(wb, prb, X; zu...),
                        "RealPpy" => factor_attribution(wc, prb, X; ppy = 252, zu...),
                        "RealHist" => factor_attribution(Wh, prb, reth; zu...),
                        "RollBase" => factor_attribution(wb, prb, X, 30; step = 5, zu...),
                        "RollHist" =>
                            factor_attribution(Wh, prb, reth, 30; step = 7, zu...))
        end
        for (c, fa) in zc
            our = fa_pack(fa)["Assets"]
            @test cmp(our[:, 3:5], load(c, "Assets")[:, 3:5], "$(c) standalone, zero rule")
            # The weights hold no unknown entry, so the contributions do not read the rule.
            @test isequal(our[:, 6:12], cases[c][1]["Assets"][:, 6:12])
        end
        # The relisted asset 4 states the oracle's volatility and correlation, from its unknown
        # idiosyncratic variance read as zero. The default states neither.
        a0, a1 = cases["PredBase"][1]["Assets"], fa_pack(zc["PredBase"])["Assets"]
        @test isnan(a0[4, 3]) && isnan(a0[4, 5])
        # The loop above compares both cells with the oracle cell by cell.
        @test isfinite(a1[4, 3]) && isfinite(a1[4, 5])
    end

    @testset "Changed (#1515): `ddof = 0` gives the oracle's weight spread" begin
        r0 = quiet(() -> fa_pack(factor_attribution(Wh, prb, reth; assets = true, ddof = 0)))
        @test cmp(r0["Assets"][:, 2], load("RealHist", "Assets")[:, 2], "RealHist ddof 0")
        @test isequal(r0["Assets"][:, [1; 3:12]],
                      cases["RealHist"][1]["Assets"][:, [1; 3:12]])
        w0 = quiet(() -> fa_pack(factor_attribution(Wh, prb, reth, 30; step = 7,
                                                    assets = true, ddof = 0)))
        @test cmp(w0["Assets"][:, 2], load("RollHist", "Assets")[:, 2], "RollHist ddof 0")
    end

    @testset "Changed (#1515): a holiday at a held pair fills zero with no message" begin
        amsk = rd.pnl.amsk
        # Asset 5 has one holiday, an active cell with no return. A book that holds it and no
        # asset outside its active span holds no other gap.
        @test findall(amsk .& .!isfinite.(X)) == [CartesianIndex(45, 5)]
        w5 = copy(wc)
        w5[5] = 0.1
        w5 ./= sum(w5)
        # The returns result carries the active mask, so the holiday is silent, under `strict`
        # too. The series is the zero fill of the bare matrix, the series the oracle reads.
        fr = @test_logs min_level = Logging.Warn factor_attribution(w5, prb, rd;
                                                                    assets = true,
                                                                    strict = true)
        fx = quiet(() -> factor_attribution(w5, prb, X; assets = true))
        @test isequal(fa_pack(fr), fa_pack(fx))
        @test_logs min_level = Logging.Warn factor_attribution(w5, prb, rd, 30;
                                                               strict = true)
        # A bare matrix carries no mask, so no pair is known to be a holiday.
        @test_logs (:warn, r"outside their active span") match_mode = :any factor_attribution(w5,
                                                                                              prb,
                                                                                              X)
        @test_throws ArgumentError factor_attribution(w5, prb, X; strict = true)
        # A held pair outside the active span warns, and refuses under `strict`. Asset 2 lists
        # late.
        @test_logs (:warn, r"Assets \[2\] carry a non-finite return at 20") match_mode = :any factor_attribution(wb,
                                                                                                                 prb,
                                                                                                                 rd)
        @test_throws "outside their active span" PO.attribution_net_returns(wb, X, nothing,
                                                                            true, amsk)
    end

    @testset "A prior method is the bare-array method on the arrays of its block (#1404)" begin
        # The predicted side reads the totals of the prior too.
        for (c, w, kw) in
            (("PredBase", wb, (;)), ("PredPpy", wc, (; ppy = 252)), ("PredFam", wf, (;)),
             ("PredTs", wt, (;)))
            p = cases[c][2]
            a, o = fa_pred_arrays(p)
            our = quiet() do
                return fa_pack(factor_attribution(w, a...; o..., sigma = p.sigma, mu = p.mu,
                                                  assets = true, kw...))
            end
            @test isequal(our, cases[c][1])
        end
        # `kw` is the keywords, then the window of a rolling case.
        function arr_real(W, p, ret, kw...)
            a, o = fa_real_arrays(p)
            return quiet() do
                return fa_pack(factor_attribution(W, a..., ret, kw[2:end]...; o...,
                                                  kw[1]...))
            end
        end
        for (c, kw) in (("RealBase", ((; assets = true),)),
                        ("RealPpy", ((; assets = true, ppy = 252),)),
                        ("RealHist", ((; assets = true),)), ("RealTs", ((; assets = true),)),
                        ("RollBase", ((; assets = true, step = 5), 30)),
                        ("RollHist", ((; assets = true, step = 7), 30)))
            @test isequal(arr_real(cases[c][3], cases[c][2], cases[c][4], kw...),
                          cases[c][1])
        end
        for c in ("SeWarmup", "SeFull", "SeFam", "SeFamTwo", "SeRankDef", "SeCurrency",
                  "SeMacro", "RollSeFam")
            p = cases[c][2]
            w = fa_clean_weights(p, X)
            kw = c == "RollSeFam" ? ((; se = true, step = 9), 40) : ((; se = true),)
            @test isequal(arr_real(w, p, net(w), kw...), cases[c][1])
        end
    end

    @testset "The bare-array methods: shapes, lags and refusals (#1404)" begin
        B, F, d = fa_pred_arrays(prt)[1]
        o = fa_pred_arrays(prt)[2]
        # The idiosyncratic block is a vector of variances or a covariance.
        fv = factor_attribution(wt, B, F, d; o...)
        fm = factor_attribution(wt, B, F, Matrix(Diagonal(d)); o...)
        @test fm.total.vol ≈ fv.total.vol rtol = 1e-15
        @test fm.idio.vol_contrib ≈ fv.idio.vol_contrib rtol = 1e-15
        # Two means of zero are the defaults.
        f0 = factor_attribution(wt, B, F, d)
        @test iszero(f0.total.mu_contrib) && f0.total.vol == fv.total.vol
        @test_throws PO.IsNonFiniteError factor_attribution(wt, B, fill(NaN, size(F)), d)
        # A holding the arrays do not state warns, or raises under `strict`.
        Bn = copy(B)
        Bn[1, :] .= NaN
        @test_logs (:warn,) match_mode = :any factor_attribution(wt, Bn, F, d)
        @test_throws ArgumentError factor_attribution(wt, Bn, F, d; strict = true)
        # A static matrix describes every observation, so its lag defaults to zero, and an
        # exposure history lags the returns by one observation by default.
        (Br, f, eps), _ = fa_real_arrays(prt)
        ret = cases["RealTs"][4]
        r0 = factor_attribution(wt, Br, f, eps, ret)
        @test isequal(fa_pack(r0),
                      fa_pack(factor_attribution(wt, Br, f, eps, ret; lag = 0)))
        # A static matrix reads every row whatever the lag, as the oracle does, and
        # `trim = true` cuts it by the lag (#1515).
        @test isequal(fa_pack(factor_attribution(wt, Br, f, eps, ret; lag = 1)),
                      fa_pack(r0))
        rt = fa_pack(factor_attribution(wt, Br, f, eps, ret; lag = 1, trim = true))
        @test isequal(rt,
                      fa_pack(factor_attribution(wt, Br, f[2:end, :], eps[2:end, :], ret)))
        @test !isequal(rt, fa_pack(r0))
        @test_throws DomainError factor_attribution(wt, Br, f, eps, ret; lag = -1,
                                                    trim = true)
        T = size(f, 1)
        Bh = stack(fill(Br, T); dims = 1)
        @test isequal(fa_pack(factor_attribution(wt, Bh, f, eps, ret)),
                      fa_pack(factor_attribution(wt, Bh, f, eps, ret; lag = 1)))
        @test !isequal(fa_pack(factor_attribution(wt, Bh, f, eps, ret)), fa_pack(r0))
        # `trim` reads a static matrix alone: a history is always cut by its lag.
        @test isequal(fa_pack(factor_attribution(wt, Bh, f, eps, ret; trim = true)),
                      fa_pack(factor_attribution(wt, Bh, f, eps, ret)))
        @test length(factor_attribution(wt, Br, f, eps, ret, 30; step = 10)) ==
              length(30:10:T)
        @test_throws DomainError factor_attribution(wt, Br, f, eps, ret; lag = -1)
        @test_throws DimensionMismatch factor_attribution(wt, Br, f, eps[2:end, :], ret)
        @test_throws DimensionMismatch factor_attribution(wt, Bh[2:end, :, :], f, eps, ret)
    end

    @testset "A Scenario Cap leaves the realised attribution of a re-based fit (#1422)" begin
        # The cap keeps the last rows of the scenarios, of `o_X` and of `fpr.X` alone, and
        # leaves the block whole. The attribution reads the raw-axis factor returns off the
        # block, so it pairs each factor return with the residual of its own row. It read
        # `fpr.X` before, and the capped fit raised a `BoundsError`.
        kw = (; factors = ind, families = ["industry" => nothing])
        prf = fit(; kw...)
        prc = fit(; kw...,
                  pe = EmpiricalPrior(; me = GRID_PE.me, ce = GRID_PE.ce,
                                      max_scenarios = 40))
        @test size(prc.fpr.X, 1) == 40 < size(prc.rr.csr.f, 1)
        @test isequal(prc.rr.fr, prf.rr.fr)
        wf = fa_clean_weights(prf, X)
        quiet() do
            @test isequal(fa_pack(factor_attribution(wf, prc, X; assets = true)),
                          fa_pack(factor_attribution(wf, prf, X; assets = true)))
        end
    end
end
