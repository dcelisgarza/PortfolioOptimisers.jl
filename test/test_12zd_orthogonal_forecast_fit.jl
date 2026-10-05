#=
The Orthogonal Forecast Fit of the Cross-Sectional Factor Prior (#1486, map #1375).

A fitted Return Forecast regresses the forward idiosyncratic return, and the cross-sectional fit
makes that return orthogonal to the Factor Exposures under its regression weights. By the
Frisch–Waugh theorem the slope that the member fits on its whole score is attenuated by
`var(s⊥) / var(s)`, so the orthogonal part of its forecast is under-scaled and the spanned part
has no evidence. #1485 measured the under-scale out of sample at 2.05 for the target member and
1.87 for the exponentially weighted member, at equal spanned and orthogonal variance.

The panel of `ofit_panel` is the panel of that measure: static loadings `L = [1 ℓ]`, one score
`s1 = β ℓ + u` with a persistent `u`, and `ε` orthogonal to `L` under the regression weights of
the block. The rules are checked through `cross_sectional_return_forecast`, the one function the
batch fit, the online refit and the carry fold reach the forecast through, and then on each of
those paths of a full prior.

The checks that hold by construction are exact. Under `ScoreNeutralisation` a neutralised linear
member gives a forecast orthogonal to `L` under the regression weights, so `L g` is round-off.
The one statistical check is the ratio `κ⊥ / κ` against the variance ratio
`1 + β² var(ℓ)`. Every asset there has the same idiosyncratic volatility, so the calibration
weights are the regression weights and `C⊥ = C` at each observation. The ratio is then a
ratio of pooled variances alone, and a very long half-life pools every observation.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))

# A rule of the Orthogonal Forecast Scale that reads the Return Forecast history, records it,
# and leaves `c` at one. It is the probe of the history path until a shipped rule reads it.
struct OfitHistoryProbe <:
       PortfolioOptimisers.AbstractOrthogonalForecastScaleCalibrationAlgorithm end
const OFIT_SEEN_HISTORY = Ref{Any}(nothing)
PortfolioOptimisers.reads_forecast_history(::OfitHistoryProbe) = true
function (::OfitHistoryProbe)(key, pr, w, slv, ctx)
    OFIT_SEEN_HISTORY[] = ctx.cs.hist
    return 1.0
end

@testset "The Orthogonal Forecast Fit of the Cross-Sectional Factor Prior (#1486)" begin
    po = PortfolioOptimisers
    function ofit_panel(; N = 300, T = 800, beta = 1.0, seed = 1486, r = nothing,
                        flat = false)
        rng = StableRNG(seed)
        ell = randn(rng, N)
        L = [ones(N) ell]
        r = isnothing(r) ? 0.5 .+ rand(rng, N) : r
        Pp = I - L * ((L' * Diagonal(r) * L) \ (L' * Diagonal(r)))
        sig = flat ? fill(0.02, N) : 0.01 .+ 0.02 .* rand(rng, N)
        U = Matrix{Float64}(undef, T, N)
        u = randn(rng, N)
        for t in 1:T
            if t > 1
                u = 0.99 .* u .+ sqrt(1 - 0.99^2) .* randn(rng, N)
            end
            U[t, :] = u
        end
        E = Matrix{Float64}(undef, T, N)
        E[1, :] = Pp * (sig .* randn(rng, N))
        for t in 2:T
            E[t, :] = Pp * (0.5 / sqrt(252) .* sig .* U[t - 1, :] .+ sig .* randn(rng, N))
        end
        inp = [NumericPanelInput(; name = "s1", vals = beta .* ell' .+ U,
                                 alg = ForwardPanelFill())]
        pnl = asset_panel(inp; amsk = trues(T, N), emsk = trues(T, N))
        rd = ReturnsResult(; nx = ["A$i" for i in 1:N], X = zeros(T, N), pnl = pnl)
        Ms = Array{Float64}(undef, T, N, 2)
        for t in 1:T
            Ms[t, :, :] = L
        end
        csr = CrossSectionalRegression(; f = zeros(T, 2), eps = E, n = fill(N, T))
        csfm = CrossSectionalFactorModel(; M = L, b = zeros(N), csr = csr,
                                         vs = repeat((sig .^ 2)', T, 1), Ms = Ms,
                                         rw = repeat(r', T, 1), nf = ["mkt", "ell"],
                                         fam = ["market", "style"], lag = 1)
        return (; rd = rd, L = L, r = r, csfm = csfm)
    end
    scores(; kw...) = DescriptorScores(; descriptors = [Passthrough(; field = "s1")], kw...)
    cre = CrossSectionalLinearRegression()
    ofit_split(ofit, rfe, p) = po.cross_sectional_return_forecast(ofit, rfe, p.rd, p.csfm,
                                                                  cre, false, nothing)
    # The weighted least squares residual of one cross-section, by hand.
    function wls_residual(y, X, w)
        ok = isfinite.(y) .& vec(all(isfinite, X; dims = 2)) .& (w .> 0)
        W = Diagonal(ifelse.(ok, w, 0.0))
        Xz = ifelse.(isfinite.(X), X, 0.0)
        yz = ifelse.(isfinite.(y), y, 0.0)
        return y - X * ((Xz' * W * Xz) \ (Xz' * W * yz))
    end
    p = ofit_panel()
    members = (; EW = ExpWeightedReturnForecast(; scores = scores()),
               Target = TargetReturnForecast(; scores = scores()))

    @testset "BlockRegressionWeights neutralises under the regression weights of the block" begin
        rng = StableRNG(14861)
        T, N = 6, 12
        Y = randn(rng, T, N)
        Ms = cat(ones(T, N), randn(rng, T, N); dims = 3)
        Ms[2, 3, 2] = NaN
        rw = 0.5 .+ rand(rng, T, N)
        rw[4, 5] = 0.0
        pnl = asset_panel([NumericPanelInput(; name = "y", vals = Y)]; amsk = trues(T, N),
                          emsk = trues(T, N))
        rd = ReturnsResult(; nx = ["A$i" for i in 1:N], X = zeros(T, N), pnl = pnl)
        csfm = CrossSectionalFactorModel(; M = Ms[end, :, :], b = zeros(N), Ms = Ms,
                                         rw = rw, nf = ["mkt", "ell"],
                                         fam = ["market", "style"])
        raw(; kw...) = DescriptorScores(; descriptors = [Passthrough(; field = "y")],
                                        neutralise = ["mkt", "ell"], outlier = nothing,
                                        scoring = nothing, kw...)
        @test raw().nw === EstimationMaskWeights()
        Sb = descriptor_scores(raw(; nw = BlockRegressionWeights()), rd, csfm).S[:, :, 1]
        Sm = descriptor_scores(raw(), rd, csfm).S[:, :, 1]
        Hb = similar(Y)
        Hm = similar(Y)
        for t in 1:T
            Hb[t, :] = wls_residual(Y[t, :], Ms[t, :, :], rw[t, :])
            Hm[t, :] = wls_residual(Y[t, :], Ms[t, :, :], ones(N))
        end
        @test isequal(isnan.(Sb), isnan.(Hb))
        @test parity_compare(Sb, Hb; name = "regression weights", scale = :array).ok
        @test parity_compare(Sm, Hm; name = "estimation mask", scale = :array).ok
        # The scoring step reads the same weights, and the exposures span the constant, so a
        # standardised score stays orthogonal to them under the regression weights.
        Ss = descriptor_scores(raw(; nw = BlockRegressionWeights(),
                                   scoring = CrossSectionalStandardiser(;
                                                                        min_group_size = 2)),
                               rd, csfm).S[:, :, 1]
        for t in 1:T
            ok = isfinite.(Ss[t, :])
            g = Ms[t, ok, :]' * (rw[t, ok] .* Ss[t, ok])
            @test maximum(abs, g) <= 1e-12 * sum(rw[t, ok] .* abs.(Ss[t, ok]))
        end
        # A block with no regression weights is refused only under a Neutralisation.
        bare = CrossSectionalFactorModel(; M = Ms[end, :, :], b = zeros(N), Ms = Ms,
                                         nf = ["mkt", "ell"], fam = ["market", "style"])
        @test_throws po.IsNothingError descriptor_scores(raw(;
                                                             nw = BlockRegressionWeights()),
                                                         rd, bare)
        plain = DescriptorScores(; descriptors = [Passthrough(; field = "y")],
                                 nw = BlockRegressionWeights())
        @test size(descriptor_scores(plain, rd, bare).S) == (T, N, 1)
    end

    @testset "ScoreNeutralisation leaves no spanned part in a fitted member" begin
        for (nm, rfe) in pairs(members)
            sp = ofit_split(ScoreNeutralisation(), rfe, p)
            su = ofit_split(UnadjustedForecast(), rfe, p)
            a = sp.rf.mu
            # Measured 1.4e-16 (EW) and 9.5e-17 (Target): the forecast is orthogonal to `L`
            # under the regression weights, so the split leaves round-off in `L g`.
            @test maximum(abs, p.L * sp.g) <= 1e-12 * maximum(abs, a)
            @test sp.ap ≈ a rtol = 1e-12
            # The member as it stands carries a spanned part, and its orthogonal part holds
            # 1 / 2.217 of the variance of its forecast (measured).
            au = su.rf.mu
            @test sum(p.r .* au .^ 2) / sum(p.r .* su.ap .^ 2) > 1.8
            @test isequal(au, return_forecast(rfe, p.rd, p.csfm).mu)
        end
        ex = po.orthogonal_forecast_member(ScoreNeutralisation(), members.EW, p.csfm)
        @test ex.scores.neutralise == ["mkt", "ell"]
        @test ex.scores.nw === BlockRegressionWeights()
        mine = ExpWeightedReturnForecast(; scores = scores(; neutralise = "ell"))
        @test po.orthogonal_forecast_member(ScoreNeutralisation(), mine, p.csfm).scores.neutralise ==
              ["ell", "mkt"]
        # A member whose forecast the caller states keeps it.
        fixed = FixedWeightedReturnForecast(; scores = scores(), scale = 0.02)
        custom = CustomValueReturnForecast(; mu = fill(0.01, size(p.L, 1)))
        for rfe in (fixed, custom)
            @test po.orthogonal_forecast_member(ScoreNeutralisation(), rfe, p.csfm) === rfe
        end
        @test po.fits_idiosyncratic_target(members.EW)
        @test po.fits_idiosyncratic_target(members.Target)
        @test !po.fits_idiosyncratic_target(fixed)
        # The observed factors are the trailing columns, and the rule leaves them out.
        N = size(p.L, 1)
        obs = CrossSectionalFactorModel(; M = hcat(p.L, ones(N)), b = p.csfm.b,
                                        csr = CrossSectionalRegression(; f = zeros(3, 2),
                                                                       eps = zeros(3, N),
                                                                       n = fill(N, 3)),
                                        fx = zeros(3, 1))
        @test po.estimated_factor_columns(obs) == 1:2
    end

    @testset "OrthogonalPartCalibration scales the orthogonal part by κ⊥ / κ" begin
        rfe = members.Target
        sp = ofit_split(OrthogonalPartCalibration(), rfe, p)
        su = ofit_split(UnadjustedForecast(), rfe, p)
        rf = sp.rf
        # The member still publishes α = κ p, and the prior keeps `g` from it.
        @test isequal(rf.mu, su.rf.mu)
        @test rf.calib === su.rf.calib
        @test isnothing(su.rf.ocalib)
        @test sp.g == su.g
        @test sp.ap ≈ (rf.ocalib / rf.calib) .* su.ap rtol = 1e-12
        # The split of each row is the weighted least squares residual of that row.
        P = randn(StableRNG(14862), 5, size(p.L, 1))
        P[2, :] .= NaN
        Q = po.target_forecast_orthogonal_rows(cre, P, p.csfm, 0)
        @test all(isnan, Q[2, :])
        for t in (1, 3, 4, 5)
            @test Q[t, :] ≈ wls_residual(P[t, :], p.L, p.r) rtol = 1e-12
        end
        # Row t of P is observation t - off of the block, and a row before it has no split.
        @test all(isnan, po.target_forecast_orthogonal_rows(cre, P, p.csfm, 1)[1, :])
        # A calibration that did not run gives a κ⊥ of NaN, and a block with no exposure
        # history or no regression weights has no split.
        fwd = zeros(Float32, 2, 2)
        k = po.target_forecast_orthogonal_coefficient(cre, nothing, fwd, nothing, fwd, rfe,
                                                      p.csfm, 0)
        @test isa(k, Float32) && isnan(k)
        N = size(p.L, 1)
        for kw in
            ((; rw = ones(2, N)), (; Ms = permutedims(cat(p.L, p.L; dims = 3), (3, 1, 2))))
            bare = CrossSectionalFactorModel(; M = p.L, b = zeros(N), kw...)
            @test_throws po.IsNothingError po.target_forecast_orthogonal_rows(cre,
                                                                              P[1:2, :],
                                                                              bare, 0)
        end
        @test_throws DimensionMismatch po.neutralisation_base_weights(BlockRegressionWeights(),
                                                                      ones(2, N), p.csfm)
        # κ⊥ / κ is the ratio of the pooled variances, 1 + β² var(ℓ). Measured 1.018 of it on
        # this seed, and 1.007 to 1.024 over four seeds: the persistent `u` and its projection
        # on `L` move the sample ratio.
        q = ofit_panel(; r = ones(300), flat = true)
        long = TargetReturnForecast(; scores = scores(), half_life = 1e6)
        f = ofit_split(OrthogonalPartCalibration(), long, q).rf
        @test f.ocalib / f.calib ≈ 1 + var(q.L[:, 2]; corrected = false) rtol = 0.05
        # A ratio that is not finite is a warm-up, and gives the zero split.
        warm = TargetReturnForecastResult(; mu = [1.0, 2.0], calib = NaN, ocalib = NaN)
        z = po.orthogonal_forecast_rescale(OrthogonalPartCalibration(), warm, [1.0],
                                           [1.0, 2.0])
        @test iszero(z.g) && iszero(z.ap)
        @test po.orthogonal_forecast_rescale(OrthogonalPartCalibration(), warm, [1.0],
                                             [0.0, 0.0]).g == [1.0]
    end

    @testset "The constructor refuses OrthogonalPartCalibration without a calibrated slope" begin
        factors = ["mkt" => ConstantExposure()]
        @test CrossSectionalFactorPrior(; factors = factors).ofit === ScoreNeutralisation()
        ok = CrossSectionalFactorPrior(; factors = factors, rfe = members.Target,
                                       ofit = OrthogonalPartCalibration())
        @test ok.ofit === OrthogonalPartCalibration()
        for rfe in (nothing, members.EW,
                    FixedWeightedReturnForecast(; scores = scores(), scale = 0.02),
                    CustomValueReturnForecast(; mu = [0.1]),
                    TargetReturnForecast(; scores = scores(), calibrate = false))
            err = try
                CrossSectionalFactorPrior(; factors = factors, rfe = rfe,
                                          ofit = OrthogonalPartCalibration())
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("ScoreNeutralisation()", err.msg)
            @test occursin("ForecastCalibrationSlope", err.msg)
        end
        @test !po.calibrates_orthogonal_part(nothing)
        @test po.calibrates_orthogonal_part(members.Target)
    end

    @testset "The rule holds on the batch fit, the history, the refit and the carry fold" begin
        rd = grid_fixture(parity_large_panel())
        edges = (0, 90, 170, 250)
        rows(r, i) = po.port_opt_view(r, i, :)
        cfg = merge(grid_config("FcEW", rd), (; ofit = ScoreNeutralisation()))
        pe = CrossSectionalFactorPrior(; cfg...)
        # The forecast at the latest observation is orthogonal to the loadings of the
        # estimated factors under the latest regression weights, so the split gives `α⊥ = α`.
        function orthogonality(pr)
            rr = pr.rr
            a = rr.rf.mu
            w = rr.rw[end, :]
            Lr = rr.L[:, 1:size(rr.csr.f, 2)]
            ok = isfinite.(a) .& (w .> 0)
            return maximum(abs, Lr[ok, :]' * (w[ok] .* a[ok])) / sum(w[ok] .* abs.(a[ok]))
        end
        pr = prior(pe, rd)
        pu = prior(CrossSectionalFactorPrior(; grid_config("FcEW", rd)...), rd)
        @test orthogonality(pr) <= 1e-12
        @test orthogonality(pu) > 1e-3
        @test parity_compare(pr.rr.b, pr.rr.c * pr.rr.rf.mu; name = "b", scale = :array).ok
        # The history that a rule in `c` reads is the history of the neutralised member: each
        # of its rows is orthogonal to the exposures of its own observation.
        OFIT_SEEN_HISTORY[] = nothing
        prior(CrossSectionalFactorPrior(; cfg..., c = OfitHistoryProbe()), rd)
        H = OFIT_SEEN_HISTORY[]
        rr = pr.rr
        Ke = po.estimated_factor_columns(rr)
        worst = 0.0
        for t in axes(H, 1)
            h = H[t, :]
            w = rr.rw[t, :]
            ok = isfinite.(h) .& (w .> 0)
            if count(ok) > 0
                Lt = rr.Ms[t, ok, Ke]
                worst = max(worst,
                            maximum(abs, Lt' * (w[ok] .* h[ok])) /
                            sum(w[ok] .* abs.(h[ok])))
            end
        end
        @test worst <= 1e-12
        # The carry fold and the refit run the same rule, and each step equals the batch fit
        # over the same rows.
        online(est) = po.update_online_estimator(Online(est))
        for start in (pe, online(pe))
            e = start
            for k in 1:3
                e = partial_fit!(e, rows(rd, (edges[k] + 1):edges[k + 1]))
                ps = prior(e)
                pb = prior(pe, rows(rd, 1:edges[k + 1]))
                @test parity_compare(ps.mu, pb.mu; name = "mu, step $k", scale = :array).ok
                @test orthogonality(ps) <= 1e-12
            end
        end
    end
end
