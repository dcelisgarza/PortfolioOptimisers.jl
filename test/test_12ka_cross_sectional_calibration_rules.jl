#=
The rules of the two Calibration Slots of `CrossSectionalFactorPrior`: `PrecisionBlend` and
`SteinShrinkage` compute the Spanned Shrinkage `lambda`, and `ForecastCalibrationSlope` computes
the Orthogonal Forecast Scale `c`.

Each rule is pinned against a hand-coded copy of the rule that the simulation of the map measured.
The copy reads the fitted block of the prior and nothing of the library's own rule code: it splits
each forecast row by its own weighted least squares, and it takes the sample mean and the sample
covariance of the factor returns, which is what `EmpiricalPrior()` states. The measured agreement
is about 1e-16, so the pins hold at `rtol = 1e-12`.

The default of `lambda` is `PrecisionBlend()`. A test that pins a mean states `lambda = 1`.
=#
using Statistics, Distributions, Dates, Random
include(joinpath(@__DIR__, "test06c_setup.jl"))

# The fixture of `test_12k`: a panel with no missing cell, two z-scored style fields and a
# `signal` field that no Factor Exposure reads, so a Return Forecast on it has an orthogonal part.
function cscr_case()
    PO = PortfolioOptimisers
    rd0 = synthetic_asset_panel(; n_assets = 40, n_observations = 120, n_industries = 3,
                                rng = StableRNG(725_900), late_listing_proba = 0.0,
                                delisting_proba = 0.0, missing_ratio = 0.0).rd
    function zscore(A)
        B = similar(A)
        for t in axes(A, 1)
            r = view(A, t, :)
            B[t, :] = (r .- Statistics.mean(r)) ./ Statistics.std(r; corrected = false)
        end
        return B
    end
    mcap = PO.panel_field_values(rd0, "market_cap")
    pf = Any[f for f in rd0.pnl.pf]
    push!(pf, NumericPanelField(; name = "style1", vals = zscore(log.(mcap))))
    push!(pf,
          NumericPanelField(; name = "style2",
                            vals = zscore(PO.panel_field_values(rd0, "book_equity") ./ mcap)))
    push!(pf,
          NumericPanelField(; name = "signal",
                            vals = zscore(randn(StableRNG(739_100), size(rd0.X)...))))
    rd = ReturnsResult(; nx = rd0.nx, X = rd0.X, ts = rd0.ts,
                       pnl = AssetPanel(; pf = identity.(pf), amsk = rd0.pnl.amsk,
                                        emsk = rd0.pnl.emsk))
    pass(field) = CompositeExposure(; descriptors = [Passthrough(; field = field)],
                                    outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "style1" => pass("style1"), "style2" => pass("style2")]
    rfe = FixedWeightedReturnForecast(;
                                      scores = DescriptorScores(;
                                                                descriptors = [Passthrough(;
                                                                                           field = "signal")],
                                                                outlier = nothing,
                                                                scoring = nothing),
                                      scale = 0.02)
    return rd, factors, rfe
end

# The split of one forecast row by weighted least squares over its finite assets.
function cscr_split(alpha, L, w)
    ok = isfinite.(alpha) .& vec(all(isfinite, L; dims = 2))
    Lo = L[ok, :]
    Lw = Lo .* w[ok]
    g = (transpose(Lo) * Lw) \ (transpose(Lw) * alpha[ok])
    return g, alpha - L * g
end

# The split of every row of the forecast history, against the exposures and the weights of the row.
function cscr_history(rr)
    H = rr.rf.hist
    Tb, N = size(H)
    K = size(rr.csr.f, 2)
    G = fill(NaN, K, Tb)
    AP = fill(NaN, Tb, N)
    for t in 1:Tb
        if !(any(isfinite, view(H, t, :)))
            continue
        end
        g, ap = cscr_split(H[t, :], rr.Ms[t, :, 1:K], rr.rw[t, :])
        G[:, t] = g
        AP[t, :] = ap
    end
    return G, AP
end

# The `lambda` rules of the simulation: the precision blend over the history (`pb_raw`) and over
# the latest row (`pb_one`), and the Bayes-Stein and positive-part James-Stein weights.
function cscr_lambda(Fh, Ghist, gT)
    T, K = size(Fh)
    mu = vec(mean(Fh; dims = 1))
    S = cov(Fh)
    P = inv(Symmetric(S))
    m1 = K / T
    q = 0.0
    for t in axes(Ghist, 2)
        dt = Ghist[:, t] - mu
        q += dot(dt, P, dt)
    end
    m2p = max(q / size(Ghist, 2) - m1, 0.0)
    d = mu - gT
    BT = dot(d, P, d)
    ev = eigvals(Symmetric(S))
    ajs = (sum(ev) - 2 * maximum(ev)) / (T * dot(d, d))
    return (; pb_raw = m2p / (m1 + m2p), pb_one = max(0.0, 1 - K / (T * BT)),
            bayes_stein = 1 - (K + 2) / ((K + 2) + T * BT),
            james_stein = 1 - clamp(ajs, 0.0, 1.0))
end

function cscr_shrink(x, se, form, target)
    d = x - target
    if (iszero(d) || !isfinite(se))
        return target
    end
    w = form === :plugin ? d^2 / (d^2 + se^2) : max(zero(x), 1 - se^2 / d^2)
    return target + w * d
end

# The `c` rules of the simulation: the slope of the next row's residual on the orthogonal part,
# under the regression weights of the forecast row, with its HC0 standard error.
function cscr_c(AP, eps, rw)
    Tb, N = size(AP)
    num = 0.0
    den = 0.0
    n = 0
    for t in 1:(Tb - 1)
        hit = false
        for i in 1:N
            a, r, q = AP[t, i], eps[t + 1, i], rw[t, i]
            if !(isfinite(a) && isfinite(r))
                continue
            end
            hit = true
            q = isfinite(q) ? q : 0.0
            num += q * a * r
            den += q * a^2
        end
        n += hit
    end
    ch = den > 0 ? num / den : NaN
    meat = 0.0
    for t in 1:(Tb - 1), i in 1:N
        a, r, q = AP[t, i], eps[t + 1, i], rw[t, i]
        if !(isfinite(a) && isfinite(r))
            continue
        end
        q = isfinite(q) ? q : 0.0
        meat += (q * a * (r - ch * a))^2
    end
    se = sqrt(meat) / den
    chp = isfinite(ch) ? max(ch, 0.0) : 1.0
    return (; ch, se, n, chp, plugin = cscr_shrink(chp, se, :plugin, 1.0),
            pospart = cscr_shrink(chp, se, :pospart, 1.0))
end

@testset "Cross-sectional calibration rules: construction, defaults and the trait" begin
    PO = PortfolioOptimisers
    @test PrecisionBlend <: PO.AbstractSpannedShrinkageCalibrationAlgorithm
    @test SteinShrinkage <: PO.AbstractSpannedShrinkageCalibrationAlgorithm
    @test ForecastCalibrationSlope <: PO.AbstractOrthogonalForecastScaleCalibrationAlgorithm
    @test isa(PrecisionBlend(), PO.Num_SpanShrinkCal)
    @test !isa(ForecastCalibrationSlope(), PO.Num_SpanShrinkCal)
    @test isa(ForecastCalibrationSlope(), PO.Num_OrthFcScaleCal)
    @test isa(PrecisionBlend().err, CurrentForecastError)
    @test isa(SteinShrinkage().alg, BayesStein)
    @test isa(ForecastCalibrationSlope().wu, ThresholdWarmUp)
    @test ThresholdWarmUp().min_obs == 252
    @test isone(ForecastCalibrationSlope().target)
    @test_throws DomainError ThresholdWarmUp(; min_obs = 0)
    @test_throws DomainError ForecastCalibrationSlope(; target = -1)
    @test_throws DomainError ForecastCalibrationSlope(; target = Inf)

    # A rule reads the history exactly when the prior must make it.
    @test !PO.reads_forecast_history(PrecisionBlend())
    @test PO.reads_forecast_history(PrecisionBlend(; err = ForecastHistoryError()))
    @test !PO.reads_forecast_history(SteinShrinkage())
    @test PO.reads_forecast_history(ForecastCalibrationSlope())

    # The default of `lambda` is the rule, and the default of `c` is the number one.
    pe = CrossSectionalFactorPrior(; factors = ["market" => ConstantExposure()])
    @test isa(pe.lambda, PrecisionBlend)
    @test isone(pe.c)

    # A rule that reads the history needs a fitted forecast, and the constructor says so.
    f = ["market" => ConstantExposure()]
    @test_throws ArgumentError CrossSectionalFactorPrior(; factors = f,
                                                         c = ForecastCalibrationSlope())
    @test_throws ArgumentError CrossSectionalFactorPrior(; factors = f,
                                                         lambda = PrecisionBlend(;
                                                                                 err = ForecastHistoryError()))
    # A rule outside the prior has no fit to read.
    pr = PO.LowOrderPrior(; X = randn(StableRNG(1), 10, 2), mu = zeros(2),
                          sigma = Matrix(1.0I, 2, 2))
    @test_throws PO.IsNothingError PrecisionBlend()(:lambda, pr, nothing, nothing,
                                                    CalibrationContext())
    @test_throws PO.IsNothingError ForecastCalibrationSlope()(:c, pr, nothing, nothing,
                                                              CalibrationContext())
end

@testset "Cross-sectional calibration rules: the warm-up forms" begin
    PO = PortfolioOptimisers
    th = ThresholdWarmUp(; min_obs = 5)
    # Below the count the rule takes the target, at or above it the clipped slope.
    @test PO.forecast_scale_warm_up(th, 2.0, 0.1, 4, 1) == 1
    @test PO.forecast_scale_warm_up(th, 2.0, 0.1, 5, 1) == 2.0
    @test PO.forecast_scale_warm_up(th, -0.5, 0.1, 5, 1) == 0.0
    @test PO.forecast_scale_warm_up(th, NaN, 0.1, 50, 1) == 1
    for (wu, form) in ((PlugInWarmUp(), :plugin), (PositivePartWarmUp(), :pospart)),
        (c, se) in ((2.0, 1.0), (3.0, 1.0), (0.4, 0.3), (-0.5, 0.2), (1.2, 0.5))

        @test PO.forecast_scale_warm_up(wu, c, se, 1, 1.0) ≈
              cscr_shrink(max(c, 0.0), se, form, 1.0) rtol = 1e-15
    end
    # The two shrinkage forms take the target where the slope or its error is not finite, and
    # where the clipped slope is the target.
    for wu in (PlugInWarmUp(), PositivePartWarmUp())
        @test PO.forecast_scale_warm_up(wu, NaN, 1.0, 1, 1) == 1
        @test PO.forecast_scale_warm_up(wu, 2.0, Inf, 1, 1) == 1
        @test PO.forecast_scale_warm_up(wu, 1.0, 0.5, 1, 1) == 1
    end
    # Plug-in keeps part of every slope, and positive part drops a slope within one error.
    @test PO.forecast_scale_warm_up(PlugInWarmUp(), 2.0, 1.0, 1, 1.0) == 1.5
    @test PO.forecast_scale_warm_up(PositivePartWarmUp(), 1.5, 1.0, 1, 1.0) == 1.0
    @test PO.forecast_scale_warm_up(PositivePartWarmUp(), 3.0, 1.0, 1, 1.0) == 2.5
    # The HC0 error of the slope through the origin.
    a = [1.0, 2.0, 3.0]
    b = [1.0, 3.0, 2.0]
    q = [1.0, 0.5, 2.0]
    c = PO.forecast_calibration_slope(a, b, q)
    @test PO.forecast_calibration_slope_se(a, b, q, c) ≈
          sqrt(sum((q .* a .* (b .- c .* a)) .^ 2)) / sum(q .* a .^ 2) rtol = 1e-15
    @test isnan(PO.forecast_calibration_slope_se(zeros(2), ones(2), ones(2), NaN))
end

@testset "Cross-sectional calibration rules: each rule equals the rule of the simulation" begin
    PO = PortfolioOptimisers
    rd, factors, rfe = cscr_case()
    # Industries and styles without the market factor give a covariance of full rank, so the
    # formulas of the simulation hold as written, with `inv`.
    ffr = factors[2:end]
    fit(l, c) = prior(CrossSectionalFactorPrior(; factors = ffr, pe = EmpiricalPrior(),
                                                rfe = rfe, lambda = l, c = c), rd)
    p1 = fit(1, 1)
    rr = p1.rr
    @test rank(cov(rr.csr.f)) == size(rr.csr.f, 2)
    G, AP = cscr_history(rr)
    Tb = size(G, 2)
    gT = G[:, Tb]
    # The split of the last row is the split the prior blends.
    @test maximum(abs,
                  gT -
                  PO.cross_sectional_alpha_split(CrossSectionalLinearRegression(), rr.rf.mu,
                                                 rr.L, rr.rw[end, :]).g) < 1e-15
    lib = PO.cross_sectional_split_history((; hist = rr.rf.hist, csfm = rr,
                                            cre = CrossSectionalLinearRegression()))
    @test isapprox(lib.g, G; rtol = 1e-12)
    @test isapprox(lib.ap, AP; rtol = 1e-12)

    cols = [t for t in 1:(Tb - 1) if all(isfinite, G[:, t])]
    @test length(cols) == Tb - 1
    h = cscr_lambda(rr.csr.f, G[:, cols], gT)
    @test isapprox(fit(PrecisionBlend(), 1).rr.lambda, h.pb_one; rtol = 1e-12)
    @test isapprox(fit(PrecisionBlend(; err = ForecastHistoryError()), 1).rr.lambda,
                   h.pb_raw; rtol = 1e-12)
    @test isapprox(fit(SteinShrinkage(), 1).rr.lambda, h.bayes_stein; rtol = 1e-12)
    @test isapprox(fit(SteinShrinkage(; alg = JamesStein()), 1).rr.lambda, h.james_stein;
                   rtol = 1e-12)
    # The blend reads the resolved number: a fit that states it gives the same mean.
    pb = fit(PrecisionBlend(), 1)
    @test pb.mu == fit(pb.rr.lambda, 1).mu
    @test 0 < pb.rr.lambda < 1

    hc = cscr_c(AP, rr.csr.eps, rr.rw)
    @test hc.n == Tb - 1
    @test isapprox(fit(1, ForecastCalibrationSlope(; wu = ThresholdWarmUp(; min_obs = 10))).rr.c,
                   hc.chp; rtol = 1e-12)
    @test isapprox(fit(1, ForecastCalibrationSlope(; wu = PlugInWarmUp())).rr.c, hc.plugin;
                   rtol = 1e-12)
    @test isapprox(fit(1, ForecastCalibrationSlope(; wu = PositivePartWarmUp())).rr.c,
                   hc.pospart; rtol = 1e-12)
    # The warm-up of the default: 119 rows are below one year of daily rows, so `c` is one.
    @test Tb - 1 < 252
    @test fit(1, ForecastCalibrationSlope()).rr.c == 1
    @test fit(1, ForecastCalibrationSlope(; target = 0.5)).rr.c == 0.5
    pc = fit(1, ForecastCalibrationSlope(; wu = ThresholdWarmUp(; min_obs = 10)))
    # The block records the resolved scale, and `b` is that scale times the orthogonal part.
    @test isequal(pc.rr.b, pc.rr.c * rr.b)
end

@testset "Cross-sectional calibration rules: a singular factor covariance" begin
    PO = PortfolioOptimisers
    rd, factors, rfe = cscr_case()
    # The market exposure is the sum of the industry exposures, so the factor returns lie in a
    # subspace and their covariance is singular. The rules read its pseudo-inverse and its rank.
    p1 = prior(CrossSectionalFactorPrior(; factors = factors, pe = EmpiricalPrior(),
                                         rfe = rfe, lambda = 1), rd)
    rr = p1.rr
    F = rr.csr.f
    S = cov(F)
    K = rank(S)
    @test K == size(F, 2) - 1
    mu = vec(mean(F; dims = 1))
    g = PO.cross_sectional_alpha_split(CrossSectionalLinearRegression(), rr.rf.mu, rr.L,
                                       rr.rw[end, :]).g
    d = mu - g
    B = dot(d, pinv(S), d)
    T = size(F, 1)
    pr = prior(CrossSectionalFactorPrior(; factors = factors, pe = EmpiricalPrior(),
                                         rfe = rfe), rd)
    @test isapprox(pr.rr.lambda, max(0, 1 - K / (T * B)); rtol = 1e-10)
    bs = prior(CrossSectionalFactorPrior(; factors = factors, pe = EmpiricalPrior(),
                                         rfe = rfe, lambda = SteinShrinkage()), rd)
    @test isapprox(bs.rr.lambda, 1 - (K + 2) / ((K + 2) + T * B); rtol = 1e-10)

    # With no Return Forecast the default shrinks the factor mean towards zero, under the moments
    # of the default factor prior.
    q1 = prior(CrossSectionalFactorPrior(; factors = factors, lambda = 1), rd)
    Ke = size(q1.rr.csr.f, 2)
    m = q1.fpr.mu[1:Ke]
    Sd = q1.fpr.sigma[1:Ke, 1:Ke]
    Td = PO.effective_sample_size(q1.fpr, q1.fpr.w)
    Bd = dot(m, pinv(Sd), m)
    q0 = prior(CrossSectionalFactorPrior(; factors = factors), rd)
    @test isapprox(q0.rr.lambda, max(0, 1 - rank(Sd) / (Td * Bd)); rtol = 1e-10)
    @test q0.mu ==
          prior(CrossSectionalFactorPrior(; factors = factors, lambda = q0.rr.lambda), rd).mu
    # The Bodnar-Okhrin-Parolya weight divides by a gap that a zero target closes.
    @test_throws DomainError prior(CrossSectionalFactorPrior(; factors = factors,
                                                             lambda = SteinShrinkage(;
                                                                                     alg = BodnarOkhrinParolya())),
                                   rd)
end

@testset "Cross-sectional calibration rules: the history under a family re-basis and its edges" begin
    PO = PortfolioOptimisers
    rd, factors, rfe = cscr_case()
    fams = ["industry" => nothing]
    p1 = prior(CrossSectionalFactorPrior(; factors = factors, families = fams,
                                         pe = EmpiricalPrior(), rfe = rfe, lambda = 1), rd)
    rr = p1.rr
    @test !isnothing(rr.fcb)
    cs = (; hist = rr.rf.hist, csfm = rr, cre = CrossSectionalLinearRegression(),
          g = PO.cross_sectional_alpha_split(CrossSectionalLinearRegression(), rr.rf.mu,
                                             view(rr.L, :, 1:size(rr.csr.f, 2)),
                                             rr.rw[end, :]).g, ap = nothing)
    # The history splits each row on the reduced axis, so its last row is the latest split.
    sh = PO.cross_sectional_split_history(cs)
    @test size(sh.g, 1) == size(rr.csr.f, 2)
    @test isapprox(sh.g[:, end], cs.g; rtol = 1e-12)
    @test isapprox(sh.ap[end, :], rr.rf.mu - view(rr.L, :, 1:size(rr.csr.f, 2)) * cs.g;
                   rtol = 1e-12)
    ph = prior(CrossSectionalFactorPrior(; factors = factors, families = fams,
                                         pe = EmpiricalPrior(), rfe = rfe,
                                         lambda = PrecisionBlend(;
                                                                 err = ForecastHistoryError())),
               rd)
    @test 0 <= ph.rr.lambda <= 1

    # A history with no row before the latest gives the reading of the latest row alone.
    H = copy(rr.rf.hist)
    H[1:(end - 1), :] .= NaN
    cs1 = merge(cs, (; hist = H))
    @test PO.spanned_forecast_sample(ForecastHistoryError(), cs1) ==
          PO.spanned_forecast_sample(CurrentForecastError(), cs1)
    @test all(isnan, PO.cross_sectional_split_history(cs1).ap[1:(end - 1), :])
    # A context with no history is refused by the rule that reads it.
    @test_throws PO.IsNothingError PO.cross_sectional_split_history(merge(cs,
                                                                          (;
                                                                           hist = nothing)))

    # James-Stein divides by `|mu - g|^2`: where `g` is the mean, every weight is the same blend.
    W = Matrix(1.0I, 3, 3)
    ev = ones(3)
    @test PO.stein_spanned_shrinkage(JamesStein(), [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], W, ev,
                                     50) == 1
    # Bodnar-Okhrin-Parolya needs more observations than the rank, and a target that is not a
    # multiple of the mean.
    @test_throws DomainError PO.stein_spanned_shrinkage(BodnarOkhrinParolya(),
                                                        [0.1, 0.2, 0.3], [0.3, 0.1, 0.2], W,
                                                        ev, 3)
    @test_throws DomainError PO.stein_spanned_shrinkage(BodnarOkhrinParolya(),
                                                        [0.1, 0.2, 0.3], zeros(3), W, ev,
                                                        50)
end

@testset "Cross-sectional calibration rules: each Stein intensity equals the library's" begin
    #=
    `ShrunkExpectedReturns` shrinks towards a target that is a multiple of the vector of ones.
    `SteinShrinkage` shrinks towards `g`, so the check sets `g` to the grand mean, where the two
    targets are one vector, and compares the blend.
    =#
    PO = PortfolioOptimisers
    rng = StableRNG(1483)
    for K in (3, 20)
        T = 60
        X = 0.01 .* randn(rng, T, K) .+ 0.001 .* transpose(randn(rng, K))
        mu = vec(mean(X; dims = 1))
        S = cov(X)
        g = fill(mean(mu), K)
        E = eigen(Symmetric(S))
        W = E.vectors ./ transpose(sqrt.(E.values))
        for alg in (BayesStein(), JamesStein())
            lam = PO.stein_spanned_shrinkage(alg, mu, g, W, E.values, T)
            libmu = vec(mean(ShrunkExpectedReturns(; alg = alg), X))
            @test isapprox(lam * mu + (1 - lam) * g, libmu; rtol = 1e-12)
        end
        # The Bodnar-Okhrin-Parolya estimate is `alpha mu + beta g`, and the rule returns the
        # clipped `alpha`.
        libmu = vec(mean(ShrunkExpectedReturns(; alg = BodnarOkhrinParolya()), X))
        ab = hcat(mu, g) \ libmu
        lam = PO.stein_spanned_shrinkage(BodnarOkhrinParolya(), mu, g, W, E.values, T)
        @test isapprox(lam, clamp(ab[1], 0, 1); rtol = 1e-10, atol = 1e-14)
    end
end
