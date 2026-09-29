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
  - Deliberate (#782). The weight spread divides by `T - 1`, as every other spread does: ours is the
    oracle's times `sqrt(T / (T - 1))`. A constant weight states no spread, `nothing`, where the
    oracle states zeros.
  - Deliberate (#844). A holding in a non-investable asset warns and is zeroed, and `strict = true`
    refuses it. The oracle keeps the loadings of the relisted asset and reads its unknown
    idiosyncratic variance as zero, which understates the portfolio variance.
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
  - Better. An observed factor is not estimated by the regression, so it leaves the sandwich
    whatever its family label (`SeMacro`). The oracle leaves it only under the label "currency":
    relabelled so (`SeMacroObs`), the oracle equals ours.
  - Better. The oracle's rolling family axis sorts each window by its variance shares and keeps
    the labels of the first window, so a window whose order changes carries the wrong labels
    (`RollSeFam`, windows 4 and 5). Ours sorts every window by label.

The time-series route (`PredTs`, `RealTs`) names no family, so the oracle's family axis over
families a caller gives is the bare-array build of #1404.
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
    ve1 = RegimeAdjustedExpWeightedVariance(; centred = true, regime_lohi_mult = (0.7, 1.6),
                                            min_val = 1e-12, min_obs = 1)
    fit(; kw...) = prior(CrossSectionalFactorPrior(; factors = base, minra = 5,
                                                   pe = GRID_PE, ve = GRID_VE, kw...), rd)
    load(c, o) = parity_load(FA_UNIT, c, o)
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
            w = fa_clean_weights(p, X)
            cases[c] = (fa_pack(factor_attribution(w, p, X; se = true)), p)
            if c == "SeFam"
                cases["RollSeFam"] = (fa_pack(factor_attribution(w, p, X, 40; step = 9,
                                                                 se = true)), p)
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
        @test cmp(our[rows, cols], orc[rows, cols], "$(c) components"; atol = 1e-14 * sc)
        rem = fa_rows(our, 4, (3,))
        if startswith(c, "Pred")
            # The oracle states no predicted remainder. Ours is the gap between `pr.sigma` and
            # the model, at rounding level on a plain fit, and it closes the total.
            @test all(isnan, orc[rem, :])
            @test all(abs.(our[rem, 3]) .< 1e-11)
            @test our[1, 2] + our[2, 2] + our[3, 2] ≈ our[4, 2] rtol = 1e-14
            @test our[1, 4] + our[2, 4] + our[3, 4] ≈ our[4, 4] rtol = 1e-14
        else
            # A remainder constant to round-off has no volatility on our side, and a volatility
            # of round-off on the oracle's.
            @test cmp(our[rem, 1:5], orc[rem, 1:5], "$(c) remainder"; atol = 1e-14 * sc)
        end
    end

    @testset "The factor axis, $(c)" for c in sort(collect(keys(cases)))
        our = cases[c][1]["Factors"]
        orc = load(c, "Factors")
        @test size(our) == size(orc)
        cols = c in se_better ? (1:8) : (1:9)
        # The spread of an exposure that does not move is round-off on our side, and zero on the
        # oracle's where the weights and the loadings are both static.
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
        # both sides, so it is read against the scale of the whole mean contribution.
        @test cmp(our[:, [1, 6, 7, 8, 9, 10, 12]], orc[:, [1, 6, 7, 8, 9, 10, 12]],
                  "$(c) asset contributions")
        @test cmp(our[:, 11], orc[:, 11], "$(c) idiosyncratic mean";
                  atol = 1e-12 * maximum(abs, our[:, 12]))
        # The weight spread (#782).
        if startswith(c, "Pred")
            @test all(isnan, our[:, 2]) && all(isnan, orc[:, 2])
        elseif cases[c][3] isa AbstractVector
            @test all(isnan, our[:, 2]) && all(iszero, orc[:, 2])
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
        elseif c == "RealTs" || c == "PredTs"
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
            @test cmp(our[r[p[b][invperm(p[1])]], :], orc[r, :], "RollSeFam window $(b)";
                      atol = 1e-14 * maximum(abs, our[r, 1]))
        end
        # A window in another order than the first carries the wrong labels.
        @test any(p[b] != p[1] for b in 2:nw)
    end

    @testset "Deliberate (#844): a held non-investable asset is zeroed, or refused" begin
        wh = copy(wb)
        wh[3] = 0.05
        wh[4] = 0.07
        wh ./= sum(wh)
        wz = copy(wh)
        wz[3:4] .= 0
        fh = @test_logs (:warn,) match_mode = :any factor_attribution(wh, prb)
        fz = factor_attribution(wz, prb)
        @test fh.sys.vol_contrib ≈ fz.sys.vol_contrib rtol = 1e-14
        @test fh.idio.mu_contrib ≈ fz.idio.mu_contrib rtol = 1e-14
        @test fh.total.vol ≈ fz.total.vol rtol = 1e-14
        @test fh.fbd.exposure ≈ fz.fbd.exposure rtol = 1e-14
        @test_throws ArgumentError factor_attribution(wh, prb; strict = true)
        @test_throws ArgumentError factor_attribution(wb, prb, X; strict = true)
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
