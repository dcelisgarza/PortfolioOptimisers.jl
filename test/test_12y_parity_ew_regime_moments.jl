#=
Parity of map #1375 for the moments that the Cross-Sectional Factor Prior composes (#1383): the
three exponentially weighted estimators and the two regime-adjusted ones, alone, with the
non-default fields. Every `Parity_*` file this test reads is an output of the oracle, stored
with the harness of #1376. The conventions of the harness are in the comment that #1376 links;
this file does not repeat them.

THE FIXTURE. `parity_small_panel()`, 12 assets and 80 observations, with its late listing,
delisting, relisting, asset outside the estimation mask and holiday. The case "Small" fills the
holiday cell with a return, so it holds every condition of the panel except the holiday. The case
"Holiday" is the panel as it is. The case "Milli" is "Small" in units one thousand times smaller.
The estimators read the active mask, and the regime-adjusted ones the estimation mask too.

THE CASES. One file holds the cases of one unit side by side, in the order of the constants below.
A regime-adjusted case states the oracle's clip `(0.7, 1.6)`, its floor `1e-12` and its centring,
which since #1383 are the defaults of both estimators except the centring. The half-life is 10 and
`min_obs` is 5 unless the name says otherwise: "Default" is the half-life of 40, "HL10p6" a
half-life of 10.6, "RegimeHL3" a regime half-life of 3, "Full" a `min_obs` of 13, "CorrHL5" a
correlation half-life of 5, "Weights" two portfolios of `PortfolioTarget`. `Series` is
`variance_series`, and the oracle's is its variance after a `partial_fit` of each row.

THE MEASURE. Every case below agreed on the `NaN` pattern.

| Unit | Verdict | Largest difference |
| --- | --- | --- |
| `ExpWeightedExpectedReturns`, `ExpWeightedVariance` | Parity, with and without the holiday | 2.3e-16 |
| `ExpWeightedCovariance` | Parity | 1.8e-15 of the largest entry, 1.2e-13 by cell |
| `ExpWeightedCovariance` on a holiday | Deliberate difference, ADR 0181 | parity off the pairs of the holiday asset; those pairs 2.1e-3 |
| `RegimeAdjustedExpWeightedVariance` | Parity, with and without the holiday | 8.9e-16 |
| `RegimeAdjustedExpWeightedCovariance` | Parity | 1.0e-13 by cell, 2.0e-15 of the largest entry |
| its separate `cor_decay` path on a late listing | Better, amendment of 2026-09-29 to ADR 0181 | the covariance over its multiplier squared 4.4e-15; the multiplier 6.0e-4 |
| both regime estimators in small units | Defect found and fixed: the floor | parity at the new default floor |
| the clamp of a warm-up multiplier | Defect found and fixed | parity |
| the separate path on a constant asset | Better: the variance 0, where the report floored it at `min_val` | `test_08y` |

The Mahalanobis target with `min_obs = 5` on 12 assets reads a singular block (#1415), so its
cases here use `min_obs = 13`.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

const P1383 = PortfolioOptimisers
p1383_dk(hl) = 2.0^(-1.0 / hl)
# The oracle's clip, floor and centring. The first two are the defaults since #1383.
const P1383_ORC = (; regime_lohi_mult = (0.7, 1.6), min_val = 1e-12, centred = true)

function p1383_rv(; hl = 10, min_obs = 5, rhl = hl / 2, rmin = floor(Int, rhl), kw...)
    return RegimeAdjustedExpWeightedVariance(; decay = p1383_dk(hl), min_obs = min_obs,
                                             regime_decay = p1383_dk(rhl),
                                             regime_min_obs = rmin, P1383_ORC..., kw...)
end
function p1383_rc(; hl = 10, min_obs = 5, rhl = hl / 2, rmin = floor(Int, rhl), kw...)
    return RegimeAdjustedExpWeightedCovariance(; decay = p1383_dk(hl), min_obs = min_obs,
                                               regime_decay = p1383_dk(rhl),
                                               regime_min_obs = rmin, P1383_ORC..., kw...)
end

const P1383_MU = ["MuDefault" => () -> ExpWeightedExpectedReturns(; decay = p1383_dk(40)),
                  "MuHL10" =>
                      () -> ExpWeightedExpectedReturns(; decay = p1383_dk(10), min_obs = 5),
                  "MuHL10p6" => () -> ExpWeightedExpectedReturns(; decay = p1383_dk(10.6))]
const P1383_VAR = ["VarDefault" =>
                       () -> ExpWeightedVariance(; decay = p1383_dk(40), centred = true),
                   "VarCentred" =>
                       () -> ExpWeightedVariance(; decay = p1383_dk(10), min_obs = 5,
                                                 centred = true),
                   "VarUncentred" =>
                       () -> ExpWeightedVariance(; decay = p1383_dk(10), min_obs = 5,
                                                 centred = false)]
const P1383_COV = ["CovDefault" =>
                       () -> ExpWeightedCovariance(; decay = p1383_dk(40), centred = true),
                   "CovCentred" =>
                       () -> ExpWeightedCovariance(; decay = p1383_dk(10), min_obs = 5,
                                                   centred = true),
                   "CovUncentred" =>
                       () -> ExpWeightedCovariance(; decay = p1383_dk(10), min_obs = 5,
                                                   centred = false)]
const P1383_RV = ["RVDefault" => () -> p1383_rv(; hl = 40, min_obs = 40),
                  "RVFirst" => () -> p1383_rv(),
                  "RVLog" => () -> p1383_rv(; regime_method = P1383.LogRegimeAdjusted()),
                  "RVRms" =>
                      () -> p1383_rv(; regime_method = P1383.RootMeanSquaredAdjusted()),
                  "RVHac2" => () -> p1383_rv(; hac_lags = 2),
                  "RVRegimeHL3" => () -> p1383_rv(; rhl = 3, rmin = 4),
                  "RVNoClip" => () -> p1383_rv(; regime_lohi_mult = nothing),
                  "RVClipAboveOne" => () -> p1383_rv(; regime_lohi_mult = (1.1, 2.0)),
                  "RVUncentred" => () -> p1383_rv(; centred = false),
                  "RVMinObs3" => () -> p1383_rv(; min_obs = 3)]
# Two portfolios: equal weights, and weights proportional to the index of the asset.
const P1383_W = vcat(fill(1 / 12, 1, 12), permutedims(collect(1.0:12) ./ 78))
const P1383_RV_SERIES = ["RVDefault", "RVLog", "RVHac2", "RVClipAboveOne", "RVUncentred"]
const P1383_RC = ["RCDefault" => () -> p1383_rc(; hl = 40, min_obs = 40),
                  "RCDiagonalFirst" =>
                      () -> p1383_rc(; regime_target = P1383.DiagonalTarget()),
                  "RCDiagonalLog" =>
                      () -> p1383_rc(; regime_target = P1383.DiagonalTarget(),
                                     regime_method = P1383.LogRegimeAdjusted()),
                  "RCDiagonalRms" =>
                      () -> p1383_rc(; regime_target = P1383.DiagonalTarget(),
                                     regime_method = P1383.RootMeanSquaredAdjusted()),
                  "RCPortfolioFirst" => () -> p1383_rc(),
                  "RCPortfolioLog" =>
                      () -> p1383_rc(; regime_method = P1383.LogRegimeAdjusted()),
                  "RCPortfolioRms" =>
                      () -> p1383_rc(; regime_method = P1383.RootMeanSquaredAdjusted()),
                  "RCMahalanobisFirstFull" =>
                      () -> p1383_rc(; regime_target = P1383.MahalanobisTarget(),
                                     min_obs = 13, regime_lohi_mult = nothing),
                  "RCMahalanobisLogFull" =>
                      () -> p1383_rc(; regime_target = P1383.MahalanobisTarget(),
                                     regime_method = P1383.LogRegimeAdjusted(),
                                     min_obs = 13, regime_lohi_mult = nothing),
                  "RCMahalanobisRmsFull" =>
                      () -> p1383_rc(; regime_target = P1383.MahalanobisTarget(),
                                     regime_method = P1383.RootMeanSquaredAdjusted(),
                                     min_obs = 13, regime_lohi_mult = nothing),
                  "RCCorrHL5" => () -> p1383_rc(; cor_decay = p1383_dk(5)),
                  "RCHac2" => () -> p1383_rc(; hac_lags = 2),
                  "RCWeights" =>
                      () -> p1383_rc(; regime_target = P1383.PortfolioTarget(; w = P1383_W)),
                  "RCNoClip" => () -> p1383_rc(; regime_lohi_mult = nothing),
                  "RCClipAboveOne" => () -> p1383_rc(; regime_lohi_mult = (1.1, 2.0)),
                  "RCUncentred" => () -> p1383_rc(; centred = false),
                  "RCRegimeHL3" => () -> p1383_rc(; rhl = 3, rmin = 4)]
const P1383_RC_SERIES = ["RCDefault", "RCDiagonalLog", "RCMahalanobisFirstFull",
                         "RCCorrHL5", "RCClipAboveOne"]

# The multiplier the report applies, read from the state after the fit.
function p1383_mult(ce, X, am, em)
    c = partial_fit!(ce, X; active_mask = am, estimation_mask = em).cache
    if c.n_regime_obs < ce.regime_min_obs
        return one(eltype(X))
    end
    f = P1383.regime_multiplier(ce.regime_method, c.regime_state)
    return isnothing(ce.regime_lohi_mult) ? f : clamp(f, ce.regime_lohi_mult...)
end
# Column `k` of a file that holds its cases side by side, each `n` columns wide.
p1383_col(A, k, n = 1) = A[:, ((k - 1) * n + 1):(k * n)]

@testset "Parity: the exponentially weighted and the regime-adjusted moments (#1383)" begin
    fx = parity_small_panel()
    am = fx.amsk
    em = fx.emsk
    Xh = ifelse.(am, fx.rd.X, NaN)
    X = copy(Xh)
    X[fx.at.holiday...] = 0.0031
    N = size(X, 2)

    @testset "The exponentially weighted moments" begin
        for (case, Xc) in (("Small", X), ("Holiday", Xh))
            Omu = parity_load("ExpWeightedExpectedReturns", case, "Mu")
            Ovar = parity_load("ExpWeightedVariance", case, "Var")
            for (k, (name, make)) in enumerate(P1383_MU)
                r = parity_compare(vec(mean(make(), Xc; active_mask = am)), Omu[:, k];
                                   name = "$case $name")
                @test r.ok
            end
            for (k, (name, make)) in enumerate(P1383_VAR)
                r = parity_compare(vec(var(make(), Xc; active_mask = am)), Ovar[:, k];
                                   name = "$case $name")
                @test r.ok
            end
        end
        # A covariance compares against its largest entry: an off-diagonal cell of two unrelated
        # assets is a cancellation, and carries a relative round-off near 1e-13.
        Ocov = parity_load("ExpWeightedCovariance", "Small", "Cov")
        for (k, (name, make)) in enumerate(P1383_COV)
            r = parity_compare(cov(make(), X; active_mask = am), p1383_col(Ocov, k, N);
                               scale = :array, name = name)
            @test r.ok
        end
    end

    @testset "A holiday holds the correlation, where the oracle holds the covariance (ADR 0181)" begin
        # ADR 0181 changes the pairs of the asset on the holiday and no other entry. So the
        # block without asset 5 agrees with the oracle, and the pairs of asset 5 do not.
        Ocov = parity_load("ExpWeightedCovariance", "Holiday", "Cov")
        k5 = setdiff(1:N, fx.at.holiday[2])
        for (k, (name, make)) in enumerate(P1383_COV)
            S = cov(make(), Xh; active_mask = am)
            O = p1383_col(Ocov, k, N)
            @test parity_compare(S[k5, k5], O[k5, k5]; scale = :array, name = name).ok
            @test !parity_compare(S[:, 5], O[:, 5]; rtol = 1e-4, scale = :array).ok
            f = findall(isfinite, LinearAlgebra.diag(S))
            @test minimum(LinearAlgebra.eigvals(LinearAlgebra.Symmetric(S[f, f]))) >=
                  -1e-15 * maximum(abs, S[f, f])
        end
    end

    @testset "The regime-adjusted variance, its multiplier and its series" begin
        for (case, Xc) in (("Small", X), ("Holiday", Xh))
            Ov = parity_load("RegimeAdjustedExpWeightedVariance", case, "Var")
            Om = parity_load("RegimeAdjustedExpWeightedVariance", case, "Mult")
            for (k, (name, make)) in enumerate(P1383_RV)
                v = vec(var(make(), Xc; active_mask = am, estimation_mask = em))
                @test parity_compare(v, Ov[:, k]; name = "$case $name").ok
                @test parity_compare([p1383_mult(make(), Xc, am, em)], Om[:, k];
                                     name = "$case $name multiplier").ok
            end
        end
        Os = parity_load("RegimeAdjustedExpWeightedVariance", "Small", "Series")
        for (k, name) in enumerate(P1383_RV_SERIES)
            make = Dict(P1383_RV)[name]
            s = P1383.variance_series(make(), X; active_mask = am, estimation_mask = em)
            @test parity_compare(s, p1383_col(Os, k, N); scale = :array,
                                 name = "$name series").ok
        end
    end

    @testset "The regime-adjusted covariance, its multiplier and its series" begin
        Oc = parity_load("RegimeAdjustedExpWeightedCovariance", "Small", "Cov")
        Om = parity_load("RegimeAdjustedExpWeightedCovariance", "Small", "Mult")
        for (k, (name, make)) in enumerate(P1383_RC)
            S = cov(make(), X; active_mask = am, estimation_mask = em)
            m = p1383_mult(make(), X, am, em)
            if name == "RCCorrHL5"
                # Better, amendment of 2026-09-29 to ADR 0181. The report is at parity once each
                # side's covariance is divided by its own multiplier squared. The multiplier is
                # not: the regime statistic here standardises by the correlation of each pair
                # over its own observations, and the oracle's statistic by the correlation that
                # a late listing shrinks towards zero. Measured 0.76054 against 0.76100.
                @test parity_compare(S ./ m^2, p1383_col(Oc, k, N) ./ Om[1, k]^2;
                                     scale = :array, name = "$name report").ok
                @test isapprox(m, 0.7605405673104944; rtol = 1e-12)
                @test !isapprox(m, Om[1, k]; rtol = 1e-4)
                continue
            end
            @test parity_compare(S, p1383_col(Oc, k, N); scale = :array, name = name).ok
            @test parity_compare([m], Om[:, k]; name = "$name multiplier").ok
        end
        Os = parity_load("RegimeAdjustedExpWeightedCovariance", "Small", "Series")
        for (k, name) in enumerate(P1383_RC_SERIES)
            make = Dict(P1383_RC)[name]
            s = P1383.variance_series(make(), X; active_mask = am, estimation_mask = em)
            name == "RCCorrHL5" && continue # the multiplier of each row, as above
            @test parity_compare(s, p1383_col(Os, k, N); scale = :array,
                                 name = "$name series").ok
        end
    end

    @testset "The separate path divides each pair by the weight it holds (ADR 0181, 2026-09-29)" begin
        # Without a holiday, the pair of assets i and j holds the weight 1 - λc^min(n_i, n_j),
        # and its diagonal the weights 1 - λc^n_i and 1 - λc^n_j. The normalisation of the state
        # alone divided by the root of the two diagonal weights, so it shrank each correlation of
        # a late listing by sqrt((1 - λc^n_j) / (1 - λc^n_i)). On this fixture asset 4 lists
        # again with 40 observations against 80, and asset 2 lists late with 60.
        lc = p1383_dk(5)
        ce = p1383_rc(; cor_decay = lc, regime_method = nothing)
        c = partial_fit!(ce, X; active_mask = am, estimation_mask = em).cache
        n = c.obs_count
        R = P1383.regime_adjusted_correlation(cov(ce, X; active_mask = am,
                                                  estimation_mask = em))
        for (i, j) in ((1, 4), (2, 4), (1, 2), (5, 4))
            Q = c.cor_state
            old = Q[i, j] / sqrt(Q[i, i] * Q[j, j])
            w = (1 - lc^min(n[i], n[j])) / sqrt((1 - lc^n[i]) * (1 - lc^n[j]))
            @test isapprox(R[i, j], old / w; rtol = 1e-12)
        end
        @test n[[1, 2, 4]] == [80, 60, 40]
        # One history on every asset gives one scalar weight, and the old normalisation.
        ce1 = p1383_rc(; cor_decay = lc, regime_method = nothing)
        Xs = X[:, [1, 5, 6, 7]]
        c1 = partial_fit!(ce1, Xs).cache
        Q1 = c1.cor_state
        R1 = P1383.regime_adjusted_correlation(cov(ce1, Xs))
        @test isapprox(R1, Q1 ./ sqrt.(LinearAlgebra.diag(Q1) * LinearAlgebra.diag(Q1)');
                       rtol = 1e-12)
    end

    @testset "The defaults: the oracle's clip and floor, and the identity of scale" begin
        # #1383, decided by the maintainer: both estimators clip the multiplier to (0.7, 1.6)
        # and floor at 1e-12 by default, as the oracle does. The centring stays the library's.
        for E in (RegimeAdjustedExpWeightedVariance, RegimeAdjustedExpWeightedCovariance)
            @test E().regime_lohi_mult == (0.7, 1.6)
            @test E().min_val == 1e-12
            @test E().centred == false
        end
        # The multiplier is a ratio of volatilities, so a change of units must not move it. The
        # floor of sqrt(eps) excluded every variance below 1.5e-8, and on returns a thousand
        # times smaller the multiplier fell from 0.869 to 1.0 and each variance moved by 32%.
        c = 1e-3
        Om = parity_load("RegimeAdjustedExpWeightedVariance", "Milli", "Mult")
        Ov = parity_load("RegimeAdjustedExpWeightedVariance", "Milli", "Var")
        for (k, (ce, make)) in enumerate((("RVFirst",
                                           () -> RegimeAdjustedExpWeightedVariance(; decay = p1383_dk(10),
                                                                                   min_obs = 5,
                                                                                   regime_decay = p1383_dk(5),
                                                                                   regime_min_obs = 5,
                                                                                   centred = true)),
                                          ("RVLog",
                                           () -> RegimeAdjustedExpWeightedVariance(; decay = p1383_dk(10),
                                                                                   min_obs = 5,
                                                                                   regime_decay = p1383_dk(5),
                                                                                   regime_min_obs = 5,
                                                                                   regime_method = P1383.LogRegimeAdjusted(),
                                                                                   centred = true))))
            m1 = p1383_mult(make(), X, am, em)
            mc = p1383_mult(make(), X .* c, am, em)
            @test isapprox(mc, m1; rtol = 1e-12)
            @test parity_compare([mc], Om[:, k]; name = "$ce milli multiplier").ok
            v = vec(var(make(), X .* c; active_mask = am, estimation_mask = em))
            @test parity_compare(v, Ov[:, k]; name = "$ce milli").ok
        end
        Oc = parity_load("RegimeAdjustedExpWeightedCovariance", "Milli", "Cov")
        for (k, t) in
            enumerate(((P1383.DiagonalTarget(), P1383.FirstMomentRegimeAdjusted()),
                       (P1383.PortfolioTarget(), P1383.LogRegimeAdjusted())))
            ce = RegimeAdjustedExpWeightedCovariance(; decay = p1383_dk(10), min_obs = 5,
                                                     regime_decay = p1383_dk(5),
                                                     regime_min_obs = 5,
                                                     regime_target = t[1],
                                                     regime_method = t[2], centred = true)
            S = cov(ce, X .* c; active_mask = am, estimation_mask = em)
            @test parity_compare(S, p1383_col(Oc, k, N); scale = :array, name = "milli $k").ok
        end
    end

    @testset "The clamp bounds an estimate, and the warm-up multiplier of one holds" begin
        # A clamp above one used to lift the warm-up multiplier, and the multiplier of
        # `regime_method = nothing`, which the library states is the plain recursion.
        for (E, f) in ((RegimeAdjustedExpWeightedVariance, var),
                       (RegimeAdjustedExpWeightedCovariance, cov))
            plain = f(E(; decay = p1383_dk(10), min_obs = 5, regime_method = nothing,
                        regime_lohi_mult = nothing), X; active_mask = am)
            for lohi in ((1.1, 2.0), (0.2, 0.5))
                off = f(E(; decay = p1383_dk(10), min_obs = 5, regime_method = nothing,
                          regime_lohi_mult = lohi), X; active_mask = am)
                @test isequal(off, plain)
                # Before `regime_min_obs` comparisons the multiplier is one.
                warm = f(E(; decay = p1383_dk(10), min_obs = 5, regime_min_obs = 1000,
                           regime_lohi_mult = lohi), X; active_mask = am)
                @test isequal(warm, plain)
            end
        end
    end
end
