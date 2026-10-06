#=
Parity of the full runs of the Cross-Sectional Factor Prior (#1391, map #1375): fit, optimise,
walk forward, attribute, the same run through a Pipeline, a hierarchical optimiser, and observed
factors through the meta-optimisers.

Every stored case under `test/assets/Parity_FullRun_<Case>_<Output>.csv.gz` is an oracle output:

  - `Both` and `UbBoth`: `parity_large_panel()`, a market factor and two passthrough styles at the
    oracle's defaults (as in `test_10d`), `MeanRisk` with the variance, `MaximumUtility(l = 4)`,
    the orthogonal mean set at `q = 0.1` and the orthogonal covariance set at `kappa = 1`, long
    only and fully invested; `UbBoth` bounds every weight at 0.04. `FullW` is the fit on every
    observation, `FoldW` the weights of each fold of a walk-forward of 120 training and 40 test
    observations, `OosRet` the out-of-sample returns. `Pred*`, `Real*` and `Wf*` are the
    predicted and realised attribution of `FullW` on the full factor model, and the realised
    attribution of the walk-forward on the same model, packed as `test_21c` packs them.
  - `Hrp`: the 36 assets of `parity_large_panel()` that are active at every observation with a
    finite return, because the oracle's HRP refuses a `NaN` return. HRP at its defaults on the
    same prior: `W`, `FoldW`, the ordered leaves, the linkage, and `OurOrderW`, the oracle's
    bisection on the leaf order of the library for the full fit and each fold.
  - `Ccy`: `ccy_fixture()`, two passthrough styles, the market factor and the Currency Factors,
    `minra = 3`, at the oracle's defaults, under `MeanRisk` at its defaults. `FullW`, `FoldW` and
    `OosRet` of a walk-forward of 60 and 20, and `SubW`, the oracle's `MeanRisk` on each
    sub-universe that `Subsets` states: the four clusters of a `NestedClustered`, then the four
    subsets of a `SubsetResampling`.

Every optimisation compares the objective that each side's weights reach, and the weights at a
looser tolerance, because the objective is flat near the optimum (#1390).

The verdicts:

  - Parity: the full run, the walk-forward, the Pipeline route, every stage of HRP, and the
    walk-forward with Currency Factors.
  - Parity, with the differences that `test_21c` records: the predicted and realised attribution.
  - Better (#1493): `factor_attribution` of a walk-forward on a prior fitted on the full sample
    matches each fold to the block by its row key. The oracle drops the first out-of-sample
    observation, whose exposure the full model holds; the library keeps it, and equals the
    oracle on the rows the oracle keeps.
  - Better (#1494): the library's leaf order is the exact optimal leaf ordering. The oracle's
    "optimal" leaf ordering misses the least adjacent sum on its own tree and its own distance:
    16.065, where an order of the same tree sums to 15.888. With the library's order, the
    oracle's bisection gives the library's weights.
  - Better: the oracle refuses a `NestedClustersOptimization` over this prior, because its panel
    cannot be sliced by asset, and it has no subset-resampling optimiser. Each cluster and each
    subset of the library equals the oracle's `MeanRisk` on that sub-universe.
=#
using JuMP, Clarabel, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "parity_harness.jl"))

# The largest absolute difference of two arrays, the number each tolerance below states.
maxabs(a, b) = maximum(abs, a .- b)

# The components, the factors and the families of an attribution, as `fa_pack` of `test_21c`
# packs them, without the asset axes.
function fullrun_pack(fa::FactorAttributionResult)
    nz(x) = isnothing(x) ? NaN : x
    col(x, n) = isnothing(x) ? fill(NaN, n) : collect(Float64, x)
    comp(c) = [c.vol c.vol_contrib c.pct_var c.mu_contrib c.corr nz(c.mu_se)]
    f = fa.fbd
    K = length(f.exposure)
    m = fa.fmbd
    n = length(m.exposure)
    return Dict("Components" =>
                    vcat(comp(fa.sys), comp(fa.idio), comp(fa.unattr), comp(fa.total)),
                "Factors" =>
                    hcat(col(f.exposure, K), col(f.exposure_std, K), col(f.vol, K),
                         col(f.corr, K), col(f.vol_contrib, K), col(f.pct_var, K),
                         col(f.mu, K), col(f.mu_contrib, K), col(f.mu_se, K)),
                "Families" =>
                    hcat(col(m.exposure, n), col(m.exposure_std, n), col(m.vol_contrib, n),
                         col(m.pct_var, n), col(m.mu_contrib, n), col(m.mu_se, n)))
end

@testset "The full runs of the cross-sectional prior, at parity (#1391)" begin
    PO = PortfolioOptimisers
    load(c, o) = parity_load("FullRun", c, o)
    loadv(c, o) = vec(load(c, o))
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                        ce = RegimeAdjustedExpWeightedCovariance(; centred = true,
                                                                 debias = false,
                                                                 regime_lohi_mult = (0.7,
                                                                                     1.6)))
    ve = RegimeAdjustedExpWeightedVariance(; centred = true, debias = false,
                                           regime_lohi_mult = (0.7, 1.6), min_val = 0.0)
    est() = CrossSectionalFactorPrior(; lambda = 1, factors = factors, pe = pe, ve = ve)
    slv = Solver(; name = :clarabel_1391, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                 "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                                 "max_iter" => 500))
    um = OrthogonalUncertaintySet(; q = 0.1)
    uc = OrthogonalUncertaintySet(; kappa = 1.0)
    function strat(ub)
        wb = isnothing(ub) ? WeightBounds() : WeightBounds(; lb = 0.0, ub = ub)
        return MeanRisk(; r = UncertaintySetVariance(; ucs = uc),
                        obj = MaximumUtility(; l = 4.0),
                        opt = JuMPOptimiser(; pe = est(), slv = slv, wb = wb,
                                            ret = ArithmeticReturn(; ucs = um)))
    end
    # The objective of `MaximumUtility(l = 4)` under both sets, on the investable view.
    function objective(pr, w)
        i = findall(PO.investable_mask(pr))
        prr = PO.port_opt_view(pr, i)
        wi = w[i]
        ms = mu_ucs(um, prr)
        ret = dot(prr.mu, wi) - ms.kappa * norm(transpose(ms.L) * wi)
        return ret - 4 * PO.ucs_variance(sigma_ucs(uc, prr), prr.sigma, wi)
    end
    rd = parity_large_panel().rd
    pr = prior(est(), rd)
    wf = IndexWalkForward(120, 40)
    trains = [(40 * (k - 1) + 1):(40 * (k - 1) + 120) for k in 1:3]
    runs = Dict(c => (; full = optimise(strat(ub), rd),
                      cv = cross_val_predict(strat(ub), rd, wf))
                for (c, ub) in (("Both", nothing), ("UbBoth", 0.04)))

    @testset "A full run fits the prior inside the optimiser, $(c)" for (c, col) in
                                                                        (("Both", 9),
                                                                         ("UbBoth", 16))
        full = runs[c].full
        @test isa(full.retcode, PO.OptimisationSuccess)
        # The oracle fits the prior inside its optimiser as well, so the weights of #1390,
        # which pass the fitted prior, are the same oracle output. Measured 2.5e-11 and 1.1e-11.
        @test maxabs(full.w, loadv(c, "FullW")) <= 1e-10
        @test maxabs(full.w,
                     parity_load("OrthogonalMeanRisk", "PitLarge", "Weights")[:, col]) <=
              1e-10
        # The prior of the run is the investable view of the standalone fit, bit for bit.
        v = PO.port_opt_view(pr, findall(PO.investable_mask(pr)))
        @test isequal(full.pr.mu, v.mu) && isequal(full.pr.sigma, v.sigma)
    end

    @testset "A walk-forward refits the prior on each training window, $(c)" for c in
                                                                                 ("Both",
                                                                                  "UbBoth")
        cv = runs[c].cv
        Wo = load(c, "FoldW")
        @test length(cv.pred) == size(Wo, 2) == 3
        for (k, p) in enumerate(cv.pred)
            prk = prior(est(), PO.port_opt_view(rd, trains[k], :))
            ik = findall(PO.investable_mask(prk))
            @test isequal(p.res.pr.mu, PO.port_opt_view(prk, ik).mu)
            fj = objective(prk, p.res.w)
            fo = objective(prk, Wo[:, k])
            # Measured relative difference of the objective 2.8e-10 at most, and of the
            # weights 2.1e-7 at most (fold 2 of `Both`, where the objective is flat).
            @test abs(fj - fo) <= 1e-9 * abs(fo)
            @test maxabs(p.res.w, Wo[:, k]) <= 5e-7
        end
        # A `NaN` return of a held asset adds zero on both sides. Measured 1.9e-8.
        @test maxabs(cv.mrd.X, loadv(c, "OosRet")) <= 1e-7
    end

    @testset "The attribution of the full run" begin
        # The attribution of the oracle's weights isolates the attribution from the optimiser.
        # The oracle annualises at 252 and states the standard errors by default.
        wo = loadv("Both", "FullW")
        fap = fullrun_pack(factor_attribution(wo, pr; ppy = 252))
        far = fullrun_pack(factor_attribution(wo, pr, rd.X; ppy = 252, se = true))
        cmp(a, b, n) = parity_compare(a, b; rtol = 1e-11, name = n).ok
        # Measured 8.2e-13. The unattributed remainder of the predicted side, the gap between
        # `pr.sigma` and the model, is round-off here, where the oracle states none (ADR 0113,
        # `test_21c`).
        C = load("Both", "PredComponents")
        @test cmp(fap["Components"][[1, 2, 4], :], C[[1, 2, 4], :], "pred components")
        @test all(isnan, C[3, 2:4]) && all(x -> abs(x) < 1e-11, fap["Components"][3, 2:4])
        @test cmp(fap["Factors"], load("Both", "PredFactors"), "pred factors")
        @test cmp(fap["Families"], load("Both", "PredFamilies"), "pred families")
        # Measured 2.9e-13. The standard errors are `NaN`: asset 4 relists and is in the
        # warm-up of its variance at rows the regression reads, so the sandwich of those rows
        # is unknown (`SeWarmup` of `test_21c`), where the oracle drops the pair.
        for (k, se) in (("Components", 6), ("Factors", 9), ("Families", 6))
            o = load("Both", "Real$(k)")
            keep = setdiff(axes(o, 2), se)
            @test cmp(far[k][:, keep], o[:, keep], "real $(k)")
            @test all(isnan, far[k][:, se])
        end
    end

    @testset "The attribution of the walk-forward on the full factor model" begin
        cv = runs["Both"].cv
        # #1493: the block records the data rows of its fit, and each fold its test rows, so
        # the attribution matches them by position.
        @test pr.rr.idx == 2:250
        @test [p.idx for p in cv.pred] == [121:160, 161:200, 201:240]
        # The bare-array route (#1404) states the same answer. Block row `j` is data row
        # `j + 1`, and the exposures of a row are those of the row before.
        blk = PO.attribution_block_arrays(pr.rr, pr)
        function wf_attr(W, ret, rows)
            br = rows .- 1
            return factor_attribution(W, blk.B[br .- blk.lag, :, :], blk.f[br, :],
                                      blk.eps[br, :], ret; lag = 0, rw = blk.rw[br, :],
                                      vs = blk.vs[br, :], fam = blk.fam, se = true,
                                      ppy = 252)
        end
        Wo = load("Both", "FoldW")
        Wh = reduce(vcat, [repeat(transpose(Wo[:, k]), 40) for k in 1:3])
        ret = loadv("Both", "OosRet")
        # The oracle restricts its model to the out-of-sample rows and applies the exposure
        # lag inside the restriction, so it drops row 121, whose exposure the full model holds.
        # On its rows 122 to 240 the library route is at parity: measured 1.7e-14, but for the
        # remainder, an exact zero where the oracle states round-off (`test_21c`), and the
        # standard errors of `SeWarmup`.
        fa = fullrun_pack(wf_attr(Wh[2:end, :], ret[2:end], 122:240))
        for (k, se, rem) in (("Components", 6, 3), ("Factors", 9, 0), ("Families", 6, 0))
            o = load("Both", "Wf$(k)")
            rows = setdiff(axes(o, 1), rem)
            cols = setdiff(axes(o, 2), se)
            @test parity_compare(fa[k][rows, cols], o[rows, cols]; rtol = 1e-12,
                                 name = "wf $(k)").ok
        end
        @test all(iszero, fa["Components"][3, 1:3]) &&
              all(x -> abs(x) < 1e-15, load("Both", "WfComponents")[3, 1:3])
        # The library route keeps row 121: its total is the volatility of all 120 rows.
        fa121 = wf_attr(Wh, ret, 121:240)
        @test fa121.total.vol ≈ std(ret) * sqrt(252) rtol = 1e-14
        @test !isapprox(fa121.total.vol, load("Both", "WfComponents")[4, 1]; rtol = 1e-3)
        # The weights of the library's own folds, through the same route, agree to the
        # tolerance of the weights.
        W, r = PO.attribution_prediction_history(cv)
        @test maxabs(W, Wh) <= 5e-7 && maxabs(r, ret) <= 1e-7
        # The one call on the walk-forward equals the bare route on its own series, row 121
        # included. Measured: a difference of exactly zero.
        fcv = fullrun_pack(factor_attribution(cv, pr; ppy = 252, se = true))
        fref = fullrun_pack(wf_attr(W, r, 121:240))
        for k in ("Components", "Factors", "Families")
            @test isequal(fcv[k], fref[k])
        end
    end

    @testset "The Pipeline route gives the weights of the direct route, $(c)" for (c, ub) in
                                                                                  (("Both",
                                                                                    nothing),
                                                                                   ("UbBoth",
                                                                                    0.04))
        # The panel rides inside `rd`, the `:returns` slot (#662), and a fold slices it before
        # the Pipeline sees the data (#666). One uncertainty step for each axis.
        wb = isnothing(ub) ? WeightBounds() : WeightBounds(; lb = 0.0, ub = ub)
        pipe = Pipeline(;
                        steps = ("prior" => est(),
                                 "umu" => PipelineStep(; est = um, writes = :uncertainty,
                                                       target = :mu),
                                 "usig" => PipelineStep(; est = uc, writes = :uncertainty,
                                                        target = :sigma),
                                 "opt" => MeanRisk(; r = UncertaintySetVariance(),
                                                   obj = MaximumUtility(; l = 4.0),
                                                   opt = JuMPOptimiser(; slv = slv, wb = wb,
                                                                       ret = ArithmeticReturn()))))
        # Equal to the direct route bit for bit, so at parity with the oracle as the direct
        # route is.
        @test fit(pipe, rd).w == runs[c].full.w
        pcv = cross_val_predict(pipe, rd, wf)
        @test all(k -> pcv.pred[k].res.w == runs[c].cv.pred[k].res.w, 1:3)
        @test pcv.mrd.X == runs[c].cv.mrd.X
    end

    @testset "HierarchicalRiskParity under the prior" begin
        fx = parity_large_panel()
        keep = findall(j -> all(fx.amsk[:, j]) && all(isfinite, fx.rd.X[:, j]),
                       axes(fx.rd.X, 2))
        @test setdiff(axes(fx.rd.X, 2), keep) == 2:5
        rdh = PO.port_opt_view(fx.rd, :, keep)
        hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = est()))
        res = optimise(hrp, rdh)
        hcv = cross_val_predict(hrp, rdh, wf)
        # Every stage but the leaf ordering is at parity: with the library's leaf order the
        # oracle's bisection gives the library's weights. Measured 3.0e-13 relative at most,
        # on the full fit and on each fold.
        Wou = load("Hrp", "OurOrderW")
        for (k, w) in enumerate((res.w, (p.res.w for p in hcv.pred)...))
            @test parity_compare(w, Wou[:, k]; rtol = 1e-12, name = "hrp $(k)").ok
        end
        # The tree is the oracle's: the same merge heights (measured 4.4e-16) and the same
        # clusters. The oracle sorts its merges by height.
        L = load("Hrp", "Linkage")
        @test maxabs(sort(res.clr.res.heights), sort(L[:, 3])) <= 1e-14
        # Better (#1494): the library's order is the exact optimal leaf ordering, 15.888. The
        # oracle's "optimal" order of the same tree sums to 16.065, so the orders differ, and
        # so do the weights. The library's heuristic order summed to 15.921 before.
        D = res.clr.D
        adj(o) = sum(D[o[i], o[i + 1]] for i in 1:(length(o) - 1))
        lo = Int.(loadv("Hrp", "Leaves")) .+ 1
        @test res.clr.res.order != lo
        @test adj(res.clr.res.order) < adj(lo) - 0.1
        @test maxabs(res.w, loadv("Hrp", "W")) > 1e-3
    end

    @testset "Observed factors through a walk-forward and the meta-optimisers" begin
        fx = ccy_fixture()
        pec = CrossSectionalFactorPrior(; lambda = 1,
                                        factors = ["style1" => ccy_pass("style1", "style"),
                                                   "style2" => ccy_pass("style2", "style"),
                                                   "market" => ConstantExposure(),
                                                   "currency" => CurrencyExposure()],
                                        minra = 3, pe = PARITY_PE, ve = PARITY_VE)
        inner = MeanRisk(; opt = JuMPOptimiser(; pe = pec, slv = slv))
        # The variance that each side's weights reach, on the library's prior of the fit.
        function relvar(res, wo)
            local sg = res.pr.sigma
            return (dot(res.w, sg, res.w) - dot(wo, sg, wo)) / dot(wo, sg, wo)
        end
        full = optimise(inner, fx.rd)
        # Measured 1.9e-13 (variance) and 7.3e-8 (weights).
        @test abs(relvar(full, loadv("Ccy", "FullW"))) < 1e-12
        @test maxabs(full.w, loadv("Ccy", "FullW")) <= 1e-6
        cv = cross_val_predict(inner, fx.rd, IndexWalkForward(60, 20))
        Wf = load("Ccy", "FoldW")
        for (k, p) in enumerate(cv.pred)
            # Measured 1.1e-13 (variance) and 1.1e-7 (weights).
            @test abs(relvar(p.res, Wf[:, k])) < 1e-12
            @test maxabs(p.res.w, Wf[:, k]) <= 1e-6
        end
        # Measured 7.1e-9.
        @test maxabs(cv.mrd.X, loadv("Ccy", "OosRet")) <= 5e-8

        # Better: the oracle refuses a NestedClustersOptimization over this prior and has no
        # subset-resampling optimiser. Each cluster and each subset fits the prior on its own
        # assets, and equals the oracle's MeanRisk on a panel of those assets alone.
        nco = optimise(NestedClustered(; pe = pec, opti = inner, opto = EqualWeighted()),
                       fx.rd)
        sr = optimise(SubsetResampling(; subset_size = 20, n_subsets = 4,
                                       rng = StableRNG(9), pe = pec, opt = inner), fx.rd)
        S = load("Ccy", "Subsets")
        cls = [findall(==(i), assignments(nco.clr)) for i in 1:(nco.clr.k)]
        @test cls == [findall(==(1), S[:, k]) for k in 1:4]
        @test all(k -> sort(sr.idx[:, k]) == findall(==(1), S[:, 4 + k]), 1:4)
        Ws = load("Ccy", "SubW")
        for (k, r) in enumerate(vcat(nco.resi, sr.ress))
            wo = Ws[findall(==(1), S[:, k]), k]
            # Measured 1.3e-11 (variance) and 9.5e-7 (weights) at most.
            @test abs(relvar(r, wo)) < 5e-11
            @test maxabs(r.w, wo) <= 2e-6
        end
    end
end
