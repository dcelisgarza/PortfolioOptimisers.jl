#=
Parity of map #1375 for the moments that the Cross-Sectional Factor Prior composes (#1383): the
three exponentially weighted estimators and the two regime-adjusted ones, alone, with the
non-default fields. Every `Parity_*` file this test reads is an output of the oracle, stored
with the harness of #1376. The conventions of the harness are in the comment that #1376 links;
this file does not repeat them.

THE FIXTURE. `parity_small_panel()`, 12 assets and 80 observations, with its late listing,
delisting, relisting, asset outside the estimation mask and holiday. The case "Small" fills the
holiday cell with a return, so it holds every condition of the panel except the holiday. The case
"Holiday" is the panel as it is. The case "Full" is "Small" without assets 2, 3 and 4, so every
asset shares one history (#1420). The case "Milli" is "Small" in units one thousand times smaller.
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
| `ExpWeightedCovariance` on "Full" | Parity | 3.3e-16 by cell |
| `ExpWeightedCovariance` on a late listing or a holiday | Better, amendment of 2026-09-29 to ADR 0181 (#1420) | the raw state is the oracle's to 1e-16; each pair is divided by its own weight, where the oracle's congruence shrinks it |
| `RegimeAdjustedExpWeightedVariance` | Parity, with and without the holiday | 8.9e-16 |
| `RegimeAdjustedExpWeightedCovariance` on "Full" | Parity, report, multiplier and series | 3.3e-16 by cell |
| the same on a late listing | Better, amendment of 2026-09-29 to ADR 0181 | the raw state is the oracle's to 1.9e-16; the separate path's report is the oracle's; the multiplier moves by up to 1.9% |
| both regime estimators in small units | Defect found and fixed: the floor | parity at the new default floor |
| the clamp of a warm-up multiplier | Defect found and fixed | parity |
| the separate path on a constant asset | Better: the variance 0, where the report floored it at `min_val` | `test_08y` |

The Mahalanobis target with `min_obs = 5` on 12 assets reads a singular block, so its cases here
use `min_obs = 13`. Every case takes the oracle's raw statistic, `debias = false`. The default
statistic skips an estimate with too few observations and divides the rest by the bias of the
estimate (#1415 for the Mahalanobis target, #1428 for the others, Better, `test_08y`).
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

const P1383 = PortfolioOptimisers
p1383_dk(hl) = 2.0^(-1.0 / hl)
# The oracle's clip, floor, centring and raw statistic. The first two are the defaults since
# #1383. The default statistic divides by the bias of the estimate it reads (#1415, #1428).
const P1383_ORC = (; regime_lohi_mult = (0.7, 1.6), min_val = 1e-12, centred = true,
                   debias = false)

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
                  # The oracle floors each HAC square at zero, which makes the variance 16 % too
                  # large on returns with no autocorrelation; `hac_floor = true` keeps its rule
                  # (#1433).
                  "RVHac2" => () -> p1383_rv(; hac_lags = 2, hac_floor = true),
                  "RVRegimeHL3" => () -> p1383_rv(; rhl = 3, rmin = 4),
                  "RVNoClip" => () -> p1383_rv(; regime_lohi_mult = nothing),
                  "RVClipAboveOne" => () -> p1383_rv(; regime_lohi_mult = (1.1, 2.0)),
                  "RVUncentred" => () -> p1383_rv(; centred = false),
                  "RVMinObs3" => () -> p1383_rv(; min_obs = 3)]
# Two portfolios: equal weights, and weights proportional to the index of the asset.
p1383_w(n) = vcat(fill(1 / n, 1, n), permutedims(collect(1.0:n) ./ (n * (n + 1) / 2)))
const P1383_RV_SERIES = ["RVDefault", "RVLog", "RVHac2", "RVClipAboveOne", "RVUncentred"]
const P1383_RC = ["RCDefault" => (n) -> p1383_rc(; hl = 40, min_obs = 40),
                  "RCDiagonalFirst" =>
                      (n) -> p1383_rc(; regime_target = P1383.DiagonalTarget()),
                  "RCDiagonalLog" =>
                      (n) -> p1383_rc(; regime_target = P1383.DiagonalTarget(),
                                      regime_method = P1383.LogRegimeAdjusted()),
                  "RCDiagonalRms" =>
                      (n) -> p1383_rc(; regime_target = P1383.DiagonalTarget(),
                                      regime_method = P1383.RootMeanSquaredAdjusted()),
                  "RCPortfolioFirst" => (n) -> p1383_rc(),
                  "RCPortfolioLog" =>
                      (n) -> p1383_rc(; regime_method = P1383.LogRegimeAdjusted()),
                  "RCPortfolioRms" =>
                      (n) -> p1383_rc(; regime_method = P1383.RootMeanSquaredAdjusted()),
                  "RCMahalanobisFirstFull" =>
                      (n) -> p1383_rc(; regime_target = P1383.MahalanobisTarget(),
                                      min_obs = 13, regime_lohi_mult = nothing),
                  "RCMahalanobisLogFull" =>
                      (n) -> p1383_rc(; regime_target = P1383.MahalanobisTarget(),
                                      regime_method = P1383.LogRegimeAdjusted(),
                                      min_obs = 13, regime_lohi_mult = nothing),
                  "RCMahalanobisRmsFull" =>
                      (n) -> p1383_rc(; regime_target = P1383.MahalanobisTarget(),
                                      regime_method = P1383.RootMeanSquaredAdjusted(),
                                      min_obs = 13, regime_lohi_mult = nothing),
                  "RCCorrHL5" => (n) -> p1383_rc(; cor_decay = p1383_dk(5)),
                  "RCHac2" => (n) -> p1383_rc(; hac_lags = 2),
                  "RCWeights" =>
                      (n) -> p1383_rc(;
                                      regime_target = P1383.PortfolioTarget(;
                                                                            w = p1383_w(n))),
                  "RCNoClip" => (n) -> p1383_rc(; regime_lohi_mult = nothing),
                  "RCClipAboveOne" => (n) -> p1383_rc(; regime_lohi_mult = (1.1, 2.0)),
                  "RCUncentred" => (n) -> p1383_rc(; centred = false),
                  "RCRegimeHL3" => (n) -> p1383_rc(; rhl = 3, rmin = 4)]
const P1383_RC_SERIES = ["RCDefault", "RCDiagonalLog", "RCMahalanobisFirstFull",
                         "RCCorrHL5", "RCClipAboveOne"]
# The library's multipliers on "Small", in the order of `P1383_RC` (Better, see below).
const P1383_SMALL_MULT = [0.7562660923140582, 0.8388726265938486, 0.8211425463273536,
                          0.8970151341030335, 0.7728297271335702, 0.7, 0.8437780453120634,
                          1.1643502876725171, 1.1697258124619465, 1.1611765971360257,
                          0.7605405673104944, 0.9212423935074233, 0.7208906150115325,
                          0.7728297271335702, 1.1, 0.7867814999193841, 0.7]

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
    # "Full": the assets that share one history, 1 and 5 to 12.
    keep = [1; 5:12]
    XF = X[:, keep]
    aF = am[:, keep]
    eF = em[:, keep]
    NF = length(keep)

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
        # assets is a cancellation, and carries a relative round-off near 1e-13. On "Full"
        # every asset shares one history, and the report is at parity.
        Ocov = parity_load("ExpWeightedCovariance", "Full", "Cov")
        for (k, (name, make)) in enumerate(P1383_COV)
            r = parity_compare(cov(make(), XF; active_mask = aF), p1383_col(Ocov, k, NF);
                               scale = :array, name = "Full $name")
            @test r.ok
        end
    end

    @testset "A late listing and a holiday: the oracle's raw state, each pair over its weight (ADR 0181)" begin
        # Better, amendment of 2026-09-29 to ADR 0181 (#1420). The step is the oracle's, so the
        # raw state is its raw state. The oracle divides it by the per-asset congruence, and the
        # report divides each pair by the weight it holds, so the raw state of the oracle is its
        # output times `w * w'`, with `w` the root of the diagonal weights. The congruence
        # shrinks each correlation of a late listing or of a holiday towards zero.
        for (case, Xc, out) in (("Small", X, "Cov"), ("Holiday", Xh, "CovRaw"))
            O = parity_load("ExpWeightedCovariance", case, out)
            for (k, (name, make)) in enumerate(P1383_COV)
                st = partial_fit!(make(), Xc; active_mask = am).cache
                Ok = p1383_col(O, k, N)
                f = findall(isfinite, LinearAlgebra.diag(Ok))
                w = sqrt.(LinearAlgebra.diag(st.weight))
                @test parity_compare(st.covariance[f, f], (Ok .* (w * transpose(w)))[f, f];
                                     scale = :array, name = "$case $name raw").ok
                S = cov(make(), Xc; active_mask = am)
                @test !parity_compare(S[f, f], Ok[f, f]; rtol = 1e-4, scale = :array).ok
                @test minimum(LinearAlgebra.eigvals(LinearAlgebra.Symmetric(S[f, f]))) >=
                      -1e-15 * maximum(abs, S[f, f])
            end
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
        # "Full": every asset shares one history, so the report, the multiplier and the series
        # are at parity.
        Oc = parity_load("RegimeAdjustedExpWeightedCovariance", "Full", "Cov")
        Om = parity_load("RegimeAdjustedExpWeightedCovariance", "Full", "Mult")
        for (k, (name, make)) in enumerate(P1383_RC)
            S = cov(make(NF), XF; active_mask = aF, estimation_mask = eF)
            @test parity_compare(S, p1383_col(Oc, k, NF); scale = :array,
                                 name = "Full $name").ok
            @test parity_compare([p1383_mult(make(NF), XF, aF, eF)], Om[:, k];
                                 name = "Full $name multiplier").ok
        end
        Os = parity_load("RegimeAdjustedExpWeightedCovariance", "Full", "Series")
        for (k, name) in enumerate(P1383_RC_SERIES)
            s = P1383.variance_series(Dict(P1383_RC)[name](NF), XF; active_mask = aF,
                                      estimation_mask = eF)
            @test parity_compare(s, p1383_col(Os, k, NF); scale = :array,
                                 name = "Full $name series").ok
        end
        # "Small": assets 2 and 4 list late. Better, amendment of 2026-09-29 to ADR 0181. On the
        # path with one decay the raw state is the oracle's, its output over its multiplier
        # squared times `w * w'`. On the separate path the report is the oracle's, each side over
        # its multiplier squared, because the oracle divides that path by the pair count too.
        # The multiplier differs: the regime statistic reads the block with each pair over its
        # own weight, where the oracle's reads the congruence that shrinks a late listing. It
        # moves by 0 on the diagonal target, which reads the variances alone, and by up to 1.9%
        # on the Mahalanobis target.
        Oc = parity_load("RegimeAdjustedExpWeightedCovariance", "Small", "Cov")
        Om = parity_load("RegimeAdjustedExpWeightedCovariance", "Small", "Mult")
        for (k, (name, make)) in enumerate(P1383_RC)
            m = p1383_mult(make(N), X, am, em)
            O = p1383_col(Oc, k, N)
            if P1383.has_separate_cor_decay(make(N))
                S = cov(make(N), X; active_mask = am, estimation_mask = em)
                @test parity_compare(S ./ m^2, O ./ Om[1, k]^2; scale = :array,
                                     name = "Small $name report").ok
            else
                st = partial_fit!(make(N), X; active_mask = am, estimation_mask = em).cache
                f = findall(isfinite, LinearAlgebra.diag(O))
                w = sqrt.(LinearAlgebra.diag(st.weight))
                @test parity_compare(st.covariance[f, f],
                                     (O ./ Om[1, k]^2 .* (w * transpose(w)))[f, f];
                                     scale = :array, name = "Small $name raw").ok
            end
            @test isapprox(m, P1383_SMALL_MULT[k]; rtol = 1e-12)
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
                                                                                   centred = true,
                                                                                   debias = false)),
                                          ("RVLog",
                                           () -> RegimeAdjustedExpWeightedVariance(; decay = p1383_dk(10),
                                                                                   min_obs = 5,
                                                                                   regime_decay = p1383_dk(5),
                                                                                   regime_min_obs = 5,
                                                                                   regime_method = P1383.LogRegimeAdjusted(),
                                                                                   centred = true,
                                                                                   debias = false))))
            m1 = p1383_mult(make(), X, am, em)
            mc = p1383_mult(make(), X .* c, am, em)
            @test isapprox(mc, m1; rtol = 1e-12)
            @test parity_compare([mc], Om[:, k]; name = "$ce milli multiplier").ok
            v = vec(var(make(), X .* c; active_mask = am, estimation_mask = em))
            @test parity_compare(v, Ov[:, k]; name = "$ce milli").ok
        end
        # The covariance holds the same identity, `cov(cX) = c^2 cov(X)`. At the old floor the
        # multiplier of the diagonal target fell from 0.839 to the clip 0.7.
        for (k, t) in
            enumerate(((P1383.DiagonalTarget(), P1383.FirstMomentRegimeAdjusted()),
                       (P1383.PortfolioTarget(), P1383.LogRegimeAdjusted())))
            ce = RegimeAdjustedExpWeightedCovariance(; decay = p1383_dk(10), min_obs = 5,
                                                     regime_decay = p1383_dk(5),
                                                     regime_min_obs = 5,
                                                     regime_target = t[1],
                                                     regime_method = t[2], centred = true)
            S1 = cov(ce, X; active_mask = am, estimation_mask = em)
            Sc = cov(ce, X .* c; active_mask = am, estimation_mask = em)
            @test parity_compare(Sc, c^2 .* S1; scale = :array, name = "milli $k").ok
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
