#=
Parity of the Cross-Sectional Factor Prior on a point-in-time panel, output by output (#1384,
map #1375).

Every stored case under `test/assets/Parity_CrossSectionalFactorPrior_<Case>_<Output>.csv.gz` is
the oracle fit of a market factor and the two passthrough styles, on the panels of
`parity_harness.jl`, which list late, delist, list again, skip a holiday and keep an asset outside
the estimation universe. The factor covariance and the idiosyncratic variance are stated
explicitly, so a change of our own defaults does not move these cases. They take the oracle's
clip and its centring. The idiosyncratic variance turns its floor off (`min_val = 0.0`), where
both defaults floor at `1e-12`; that moves only a variance in `(0, 1e-12]`:

  - `PitSmall`: `parity_small_panel()`, `minra = 5`. `PitLarge`: `parity_large_panel()` at the
    default `minra`. The outputs are `mu`, `sigma`, the factor moments, the factor returns, the
    scenarios, the exposure history, the regression and benchmark weights, the idiosyncratic
    variance history, the idiosyncratic block and the square root of the covariance on the
    investable assets.
  - `PitSmallOverlay`: `PitSmall` with `th = 0.1`. `PitLargeOverlayRaw`: `PitLarge` with
    `th = 0.1`, and the idiosyncratic block before its positive definite repair.
  - `Split<Case>`: the split of a forecast against the latest loadings and weights of `PitSmall`.
  - `EntropyPoolingLarge`: the scenario weights of `PitLarge` under an entropy-pooling factor
    prior with two views.
  - `Rows41`, `Idio61`: the first 41 rows of the small panel, and its first 61 rows with an
    idiosyncratic variance that warms up over 60 observations, each at the boundary of a refusal.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# A factor prior that multiplies the scenario weights of the prior it wraps by `s` (#1509).
struct ScaledWeightsPrior{P, T} <: PortfolioOptimisers.AbstractLowOrderPriorEstimator_A
    pe::P
    s::T
end
function PortfolioOptimisers.prior(pe::ScaledWeightsPrior, X::PortfolioOptimisers.MatNum,
                                   args...; kwargs...)
    pr = prior(pe.pe, X, args...; kwargs...)
    return PortfolioOptimisers.forward_prior(pr; w = pweights(pe.s .* pr.w.values),
                                             ens = nothing, kld = nothing)
end

@testset "The prior on a point-in-time panel, at parity (#1384)" begin
    PO = PortfolioOptimisers
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                        ce = RegimeAdjustedExpWeightedCovariance(; centring = PreCentred(),
                                                                 debias = RawStatistic(),
                                                                 regime_lohi_mult = (0.7,
                                                                                     1.6)))
    ve = RegimeAdjustedExpWeightedVariance(; centring = PreCentred(),
                                           debias = RawStatistic(),
                                           regime_lohi_mult = (0.7, 1.6), min_val = 0.0)
    est(; kw...) = CrossSectionalFactorPrior(; lambda = 1, factors = factors, pe = pe,
                                             ve = ve, kw...)
    load(c, o) = parity_load("CrossSectionalFactorPrior", c, o)
    loadv(c, o) = vec(load(c, o))
    flat(Ms) = reshape(permutedims(Ms, (1, 3, 2)), size(Ms, 1), :)
    fxs = parity_small_panel()
    fxl = parity_large_panel()
    prs = prior(est(; minra = 5), fxs.rd)
    prl = prior(est(), fxl.rd)
    # The asset that lists again is in the warm-up of its variance at the latest observation of
    # the small panel, and the delisted asset is inactive, so neither is investable there.
    i4 = fxs.at.relist[1]
    rs = setdiff(axes(fxs.rd.X, 2), [fxs.at.delist[1], i4])

    @testset "Every output of the default fit, $(c)" for (c, pr) in (("PitSmall", prs),
                                                                     ("PitLarge", prl))
        rr = pr.rr
        K = size(rr.L, 2)
        imsk = PO.investable_mask(pr)
        i = isnothing(imsk) ? collect(axes(pr.mu, 1)) : findall(imsk)
        # Every entry, with the `NaN` pattern of each side. Measured maxrel 1.9e-15 and
        # 7.0e-15.
        @test parity_compare(pr.mu, loadv(c, "Mu"); name = "$(c) mu").ok
        # `sigma` compares against its largest entry: an off-diagonal entry of two assets near
        # zero correlation is a cancellation in `L F L'` (#1376). Measured maxscaled 7.9e-16
        # and 5.3e-16, and maxrel 1.6e-14 and 1.7e-14 cell by cell. They were 3.0e-13 and
        # 3.4e-13, and 9.8e-12 and 1.3e-11, when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576).
        @test parity_compare(pr.sigma, load(c, "Sigma"); scale = :array,
                             name = "$(c) sigma").ok
        # Measured maxrel 1.5e-15 at most.
        @test parity_compare(pr.fpr.mu, loadv(c, "FactorMu"); name = "$(c) fmu").ok
        @test parity_compare(pr.fpr.sigma, load(c, "FactorCov"); name = "$(c) fcov").ok
        @test parity_compare(rr.esigma, loadv(c, "IdioCov"); name = "$(c) esigma").ok
        # The square root on the investable assets, the form an optimiser reads: the loadings
        # times the Cholesky factor of the factor covariance, and the idiosyncratic
        # volatilities. Measured maxrel 4.5e-15 and 7.3e-16.
        @test parity_compare(transpose(pr.chol[1:K, i]), load(c, "InvSqrtSystematic");
                             name = "$(c) sqrt systematic").ok
        @test parity_compare(LinearAlgebra.diag(pr.chol[(K + 1):end, i]),
                             loadv(c, "InvSqrtDiagonal"); name = "$(c) sqrt diagonal").ok
        @test isapprox(transpose(pr.chol[:, i]) * pr.chol[:, i], pr.sigma[i, i];
                       rtol = 1e-12)
    end

    @testset "The histories of the block, every row" begin
        rr = prs.rr
        c = "PitSmall"
        # The exposures and the benchmark weights are bit-equal.
        @test parity_compare(flat(rr.Ms), load(c, "Exposures"); rtol = 0.0, name = "Ms").ok
        @test parity_compare(rr.bw, load(c, "BenchmarkWeights"); rtol = 0.0, name = "bw").ok
        # Measured maxrel 2.2e-16, 1.5e-15 and 2.9e-14.
        @test parity_compare(rr.rw, load(c, "RegressionWeights"); name = "rw").ok
        @test parity_compare(rr.vs, load(c, "IdioVariances"); name = "vs").ok
        @test parity_compare(prs.fpr.X, load(c, "FactorReturns"); name = "f").ok
        # A scenario is the sum of a systematic and an idiosyncratic return, so a small one
        # comes from a cancellation, and the comparison is against the largest entry. Measured
        # maxscaled 6.3e-16 on this host and on CI, where maxrel cell by cell moves with the
        # host: 8.2e-13 here, 1.6e-12 on CI. The column of an asset that is not investable is
        # `NaN` on both sides.
        @test parity_compare(prs.X, load(c, "Scenarios"); rtol = 1e-14, scale = :array,
                             name = "X").ok
        @test isnothing(prs.w) && isnothing(prs.fpr.w)
    end

    @testset "An asset in its warm-up: every entry the model determines" begin
        # The asset lists again inside the history, and its idiosyncratic variance has not
        # warmed up at the latest observation, so neither side states its variance. Its
        # latest exposures are finite, so the model determines its mean, and under a diagonal
        # idiosyncratic block, which sets the idiosyncratic covariance of two assets to zero,
        # its covariance with every other asset. Both sides state exactly those entries
        # (#1384). The Investable Mask needs a finite variance, so it still leaves the asset
        # out, on both sides.
        om = loadv("PitSmall", "Mu")
        os = load("PitSmall", "Sigma")
        @test isfinite(prs.mu[i4]) && isnan(prs.sigma[i4, i4])
        @test all(isfinite, prs.sigma[i4, rs])
        # Measured maxrel 3.1e-16 and 2.2e-15 cell by cell.
        @test parity_compare([prs.mu[i4]], [om[i4]]; name = "mu4").ok
        @test parity_compare(prs.sigma[i4, rs], os[i4, rs]; name = "sigma4").ok
        @test findall(PO.investable_mask(prs)) == rs
        # Its systematic root, the loadings times the Cholesky factor of the factor
        # covariance, is finite, and the rows of the idiosyncratic root are `NaN`, because
        # they read its variance. The oracle's systematic root of the full universe agrees.
        # Measured maxrel 4.5e-16.
        K = size(prs.rr.L, 2)
        @test parity_compare(transpose(prs.chol[1:K, :]),
                             load("PitSmall", "SqrtSystematic"); name = "systematic root").ok
        @test all(isnan, prs.chol[(K + 1):end, i4])
        # Under a positive threshold each covariance of the asset reads its variance through
        # its idiosyncratic correlation, so both sides write `NaN` over its row, and still
        # state its mean.
        po = prior(est(; minra = 5, th = 0.1), fxs.rd)
        @test isfinite(po.mu[i4]) && all(isnan, po.sigma[i4, :])
        @test all(isnan, load("PitSmallOverlay", "Sigma")[i4, :])
    end

    @testset "The overlay reads the residuals with no fill: defect fixed" begin
        # The correlation of the idiosyncratic returns used to read the residuals the
        # scenarios read, which write the mean of the other assets of an observation into
        # every active cell with no finite value: the warm-up of a late listing, a holiday.
        # That value is built from the other assets, so it correlates the asset with each of
        # them. On the small panel it kept 72 correlations above the threshold where the
        # oracle keeps 40. A gap is now a gap for the correlation, as it is for the oracle.
        pr = prior(est(; minra = 5, th = 0.1), fxs.rd)
        c = "PitSmallOverlay"
        K = size(pr.rr.L, 2)
        @test parity_compare(pr.mu, loadv(c, "Mu"); name = "overlay mu").ok
        # `sigma` carries the cancellation of the default fit: measured maxscaled 9.9e-16, and
        # maxrel 9.4e-15 cell by cell. They were 3.0e-13 and 9.8e-12 when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576). Measured maxrel
        # 1.3e-15 (`esigma`) and 5.3e-15 (root).
        @test parity_compare(pr.sigma, load(c, "Sigma"); scale = :array,
                             name = "overlay sigma").ok
        @test parity_compare(pr.rr.esigma, load(c, "IdioCov"); name = "overlay esigma").ok
        @test parity_compare(transpose(pr.chol[(K + 1):end, rs]), load(c, "InvSqrtIdio");
                             name = "overlay sqrt").ok
        @test count(!iszero, pr.rr.esigma[rs, rs]) - length(rs) == 40
        eps = pr.rr.csr.eps
        amr = fxs.amsk[(end - size(eps, 1) + 1):end, :]
        Sf = PO.cross_sectional_standardised_residuals(eps, pr.rr.vs, amr)
        Df = PO.cross_sectional_idiosyncratic_covariance(0.1,
                                                         ExpWeightedCovariance(;
                                                                               centring = PreCentred()),
                                                         nothing, Sf, pr.rr.vs[end, :], amr)
        @test count(!iszero, Df[rs, rs]) - length(rs) == 72

        # On the large panel the block before its repair agrees on every pair whose two assets
        # share one history. The raw state of the correlation is the oracle's (#1420, ADR 0181
        # amendment of 2026-09-29), and each pair is divided by the weight it holds, where the
        # oracle divides by the per-asset congruence. So on the pairs of a late listing, a
        # relisting or a holiday, ours is the oracle's times `sqrt(W_ii W_jj) / W_ij`, up to
        # 1.124 here: measured 2.4e-15 on the 1303 pairs that both keep, and 4 pairs cross the
        # threshold. The thresholded block is not positive definite, so the repair binds. Ours
        # clips the eigenvalues by default under the field `pdm`, as the oracle does (#1604),
        # and `test_07c` measures the clip on this block at parity (#1412). `pdm = Posdef()`
        # takes the nearest correlation by Newton, and it gives the block from before #1604 bit
        # for bit. Measured: the clip moves `sigma` by 6.7e-7 from that block.
        pl = prior(est(; th = 0.1), fxl.rd)
        eps = pl.rr.csr.eps
        amr = fxl.amsk[(end - size(eps, 1) + 1):end, :]
        Z = PO.cross_sectional_standardised_residuals(eps, pl.rr.vs, amr; filled = false)
        B = PO.cross_sectional_idiosyncratic_covariance(0.1,
                                                        ExpWeightedCovariance(;
                                                                              centring = PreCentred()),
                                                        nothing, Z, pl.rr.vs[end, :], amr)
        E = load("PitLargeOverlayRaw", "IdioCov")
        st = partial_fit!(ExpWeightedCovariance(; centring = PreCentred()), Z;
                          active_mask = amr).cache
        w = sqrt.(LinearAlgebra.diag(st.weight))
        ratio = (w * transpose(w)) ./ st.weight
        same = [isapprox(ratio[i, j], 1; rtol = 1e-12) for i in axes(B, 1), j in axes(B, 2)]
        @test parity_compare(ifelse.(same, B, 0.0), ifelse.(same, E, 0.0);
                             name = "raw block, one history").ok
        kept = (B .!= 0) .& (E .!= 0) .& isfinite.(B) .& isfinite.(E)
        @test all(isapprox.(B[kept], (E .* ratio)[kept]; rtol = 1e-12))
        @test count(kept .& .!same) > 0
        il = findall(isfinite, pl.mu)
        Br = B[il, il]
        @test !LinearAlgebra.isposdef(LinearAlgebra.Symmetric(Br))
        Bn = copy(Br)
        posdef!(Posdef(; alg = ClippedNearestCorrelation()), Br)
        @test Br == pl.rr.esigma[il, il]
        posdef!(Posdef(), Bn)
        @test Bn == prior(est(; th = 0.1, pdm = Posdef()), fxl.rd).rr.esigma[il, il]
    end

    @testset "The scenario weights and a Scenario Cap" begin
        # The oracle keeps the weights of the scenarios it keeps, and divides them by their
        # sum. Every factor prior of the library carries one scenario for each fitted
        # observation or fewer, so every weight is kept. The prior divides the weights by
        # their sum, as the oracle does (#1509). Measured maxrel 3.0e-16.
        sets = UniverseSets(; dict = Dict("nx" => ["market", "style1", "style2"]))
        views = LinearConstraintEstimator(; val = ["style1 == 0.002", "style2 == -0.001"])
        pr = prior(est(;
                       pe = EntropyPoolingPrior(; pe = pe, sets = sets, mu_views = views)),
                   fxl.rd)
        @test pr.w === pr.fpr.w
        @test length(pr.w) == size(pr.X, 1)
        @test parity_compare(collect(pr.w), loadv("EntropyPoolingLarge", "Weights");
                             name = "ep w").ok
        # Defect fixed: a factor prior with a Scenario Cap carries its last rows alone, and
        # the factor returns and the original returns kept every row, so the result refused
        # the pair. They now keep the rows the scenarios keep, and the moments do not move.
        # This is the route of the oracle's history cap that the docstring of
        # `CrossSectionalFactorPrior` names (#1408).
        cap = EmpiricalPrior(; me = pe.me, ce = pe.ce, max_scenarios = 30)
        pc = prior(est(; minra = 5, pe = cap), fxs.rd)
        @test size(pc.X, 1) == size(pc.fpr.X, 1) == size(pc.o_X, 1) == 30
        @test isequal(pc.X, prs.X[(end - 29):end, :])
        @test isequal(pc.fpr.X, prs.fpr.X[(end - 29):end, :])
        @test isequal(pc.o_X, prs.o_X[(end - 29):end, :])
        @test isequal(pc.mu, prs.mu) && isequal(pc.sigma, prs.sigma)
        @test pc.ens == size(prs.X, 1)
    end

    @testset "The scenario weights are divided by their sum (#1509)" begin
        # A scenario weight vector states a probability measure only when it sums to one, and
        # a reader that forms sum(w .* L) with no division needs that measure. The mean, the
        # covariance, the scenarios and the calibration are ratios in w, so the division
        # moves none of them. Measured: w differs from the plain fit by 1.0e-17.
        sets = UniverseSets(; dict = Dict("nx" => ["market", "style1", "style2"]))
        views = LinearConstraintEstimator(; val = ["style1 == 0.002", "style2 == -0.001"])
        ep = EntropyPoolingPrior(; pe = pe, sets = sets, mu_views = views)
        # The default rule of lambda reads the weights through their effective count.
        pb = PrecisionBlend()
        p1 = prior(est(; pe = ep, lambda = pb), fxl.rd)
        p2 = prior(est(; pe = ScaledWeightsPrior(ep, 2), lambda = pb), fxl.rd)
        @test isapprox(sum(p2.w), 1; rtol = 1e-14)
        @test p2.w isa StatsBase.ProbabilityWeights
        @test p2.w === p2.fpr.w
        @test parity_compare(collect(p2.w), collect(p1.w); name = "w").ok
        # Bit-equal under the default flags. Under `--check-bounds=yes`, the flag of CI, a sum
        # takes another order: measured maxrel 1.8e-15 (`mu`) and 1.5e-16 (`lambda`).
        @test parity_compare(p2.mu, p1.mu; rtol = 1e-14, name = "scaled mu").ok
        @test parity_compare(p2.sigma, p1.sigma; rtol = 1e-14, name = "scaled sigma").ok
        @test parity_compare(p2.X, p1.X; rtol = 1e-14, name = "scaled X").ok
        @test parity_compare(p2.fpr.X, p1.fpr.X; rtol = 1e-14, name = "scaled f").ok
        @test parity_compare([p2.rr.lambda], [p1.rr.lambda]; rtol = 1e-14,
                             name = "scaled lambda").ok
        @test 0 < p1.rr.lambda < 1
        # No weights means unweighted (ADR 0043).
        @test isnothing(prl.w)
        # A sum of zero states no measure, and the prior refuses it.
        @test_throws DomainError prior(est(; pe = ScaledWeightsPrior(ep, 0)), fxl.rd)
        sw = PO.cross_sectional_scenario_weights
        @test_throws DomainError sw(pweights([-1.0, 0.5]))
        # StatsBase refuses a weight vector whose sum is not finite when it builds it.
        @test_throws ArgumentError aweights([floatmax(), floatmax()])
        # The kind of the weights is kept where scaling leaves its corrected variance as it
        # was. A count would divide its corrected variance by sum(w) - 1 = 0, so FrequencyWeights
        # become AnalyticWeights, as does a kind the function cannot rebuild (#1523, #1687).
        @test sw(aweights([2.0, 1.0, 1.0])) isa StatsBase.AnalyticWeights
        @test sw(StatsBase.weights([2.0, 1.0, 1.0])) isa StatsBase.Weights
        @test sw(fweights([2, 1, 1])) == aweights([0.5, 0.25, 0.25])
        @test sw(fweights([2, 1, 1])) isa StatsBase.AnalyticWeights
        @test sw(uweights(4)) == aweights(fill(0.25, 4))
        @test sw(uweights(4)) isa StatsBase.AnalyticWeights
        @test sw(pweights(1:4)) == [0.1, 0.2, 0.3, 0.4]
        @test isnothing(sw(nothing))
    end

    @testset "The split of a forecast; a tiny forecast is Better" begin
        # The split of a forecast against the latest loadings and weights of `PitSmall`. The
        # oracle treats a forecast within 1e-8 of zero at every asset as zero, and splits it
        # into two zero parts. The split is linear, so it has no threshold of its own: a
        # forecast of 1e-9 splits into parts of 1e-9, and the identity `L g + ap = alpha`,
        # which gives `mu = alpha` at `lambda = 0` and `c = 1`, holds for it. The oracle
        # breaks the identity there, and its threshold is a number in the unit of the
        # returns, so the same forecast in basis points would split. Ours tests for an exact
        # zero.
        L = prs.rr.L
        w = prs.rr.rw[end, :]
        base = randn(StableRNG(1384), size(L, 1))
        miss = 1e-3 .* base
        miss[5] = NaN
        cre = CrossSectionalLinearRegression()
        # Measured maxrel 1.4e-15 and 2.2e-15.
        @testset "$(k)" for (k, a) in
                            (("Zero", zeros(size(L, 1))), ("Normal", 1e-3 .* base),
                             ("Missing", miss))
            sp = PO.cross_sectional_alpha_split(cre, a, L, w)
            @test parity_compare(sp.g, loadv("Split$(k)", "G"); name = "$(k) g").ok
            @test parity_compare(sp.ap, loadv("Split$(k)", "Ap"); name = "$(k) ap").ok
        end
        a = 1e-9 .* base
        sp = PO.cross_sectional_alpha_split(cre, a, L, w)
        f = findall(i -> all(isfinite, view(L, i, :)), axes(L, 1))
        @test all(iszero, loadv("SplitTiny", "G")) && all(iszero, loadv("SplitTiny", "Ap"))
        @test !all(iszero, sp.g)
        @test isapprox((L * sp.g + sp.ap)[f], a[f]; rtol = 1e-12)
        # A normal forecast scaled down by 1e6 splits into the parts scaled down by 1e6.
        spn = PO.cross_sectional_alpha_split(cre, 1e-3 .* base, L, w)
        @test isapprox(sp.g, 1e-6 .* spn.g; rtol = 1e-12)
    end

    @testset "An Empty Factor at the defaults on a point-in-time panel" begin
        # The industry "Utilities" holds one asset alone, so on the sub-universe without it
        # the factor of that level is empty (#1372). The fit equals the fit whose industry
        # field states the other three levels alone.
        rv = PO.port_opt_view(fxs.rd, setdiff(axes(fxs.rd.X, 2), fxs.at.empty_level[2]))
        k = findfirst(x -> x.name == "industry", rv.pnl.pf)
        f = rv.pnl.pf[k]
        @test f.levels[end] == fxs.at.empty_level[1] && maximum(f.codes) < length(f.levels)
        f3 = CategoricalPanelField(; name = f.name, levels = f.levels[1:(end - 1)],
                                   codes = Matrix(f.codes), omsk = f.omsk, pmsk = f.pmsk)
        pf = [j == k ? f3 : x for (j, x) in enumerate(rv.pnl.pf)]
        rv3 = ReturnsResult(; nx = rv.nx, X = rv.X, ne = rv.ne, E = rv.E,
                            pnl = AssetPanel(pf, rv.pnl.amsk, rv.pnl.emsk))
        fi = ["industry" => OneHotExposure(; field = "industry", family = "industry"),
              "style1" => mpass("style1"), "style2" => mpass("style2")]
        e(; kw...) = CrossSectionalFactorPrior(; lambda = 1, factors = fi, pe = pe, ve = ve,
                                               minra = 5, kw...)
        pr = prior(e(), rv)
        p3 = prior(e(), rv3)
        j = findfirst(==("industry=Utilities"), pr.rr.nf)
        live = setdiff(eachindex(pr.rr.nf), j)
        @test all(iszero, pr.fpr.sigma[j, :]) && iszero(pr.fpr.mu[j])
        @test pr.fpr.X[:, live] == p3.fpr.X
        @test pr.fpr.mu[live] == p3.fpr.mu
        @test pr.fpr.sigma[live, live] == p3.fpr.sigma
        i = findall(isfinite, p3.mu)
        @test findall(isfinite, pr.mu) == i
        # Measured maxrel 1.7e-16, and a covariance bit-equal.
        @test parity_compare(pr.mu, p3.mu; name = "empty mu").ok
        @test parity_compare(pr.sigma[i, i], p3.sigma[i, i]; name = "empty sigma").ok
        ii = findall(PO.investable_mask(pr))
        @test isapprox(transpose(pr.chol[:, ii]) * pr.chol[:, ii], pr.sigma[ii, ii];
                       rtol = 1e-12)
    end

    @testset "The refusals agree with the oracle at their boundaries" begin
        # Each pair sits on one side of a boundary. The oracle refuses the same four cases
        # and fits the same three: the factor covariance needs 40 observations, the
        # idiosyncratic variance of `Idio*` needs 60, the coverage refuses one asset above the
        # fewest eligible assets of an observation, and a Descriptor that is never finite
        # leaves the whole history cold.
        rows(T) = PO.port_opt_view(fxs.rd, 1:T, :)
        ve60 = RegimeAdjustedExpWeightedVariance(; centring = PreCentred(),
                                                 debias = RawStatistic(),
                                                 regime_lohi_mult = (0.7, 1.6),
                                                 min_val = 0.0, min_obs = 60)
        @test_throws PO.IsNonFiniteError prior(est(; minra = 5), rows(40))
        @test_throws PO.IsEmptyError prior(est(; minra = 5, ve = ve60), rows(50))
        nmin = minimum(prs.rr.csr.n)
        @test nmin == 10
        @test_throws ArgumentError prior(est(; minra = nmin + 1), fxs.rd)
        @test isequal(prior(est(; minra = nmin), fxs.rd).mu, prs.mu)
        cold = deepcopy(fxs.rd)
        cold.pnl.pf[findfirst(x -> x.name == "style1", cold.pnl.pf)].vals .= NaN
        @test_throws ArgumentError prior(est(; minra = 5), cold)
        # The fits on the other side of the boundary, with the `NaN` pattern: both sides state
        # a mean for each active asset in the warm-up of its variance. Measured maxrel
        # 2.3e-15 and 1.1e-14.
        @testset "$(c)" for (c, rd, kw) in (("Rows41", rows(41), (;)),
                                            ("Idio61", rows(61), (; ve = ve60)))
            pr = prior(est(; minra = 5, kw...), rd)
            @test parity_compare(pr.mu, loadv(c, "Mu"); name = "$(c) mu").ok
        end
    end
end

#=
The recovery numbers of the oracle's test suite, class by class, on the library's own draws
(#1391, map #1375).

Each testset reproduces one class of the oracle's statistical recovery tests: the same model of
the returns, the same sizes and the same configuration of the prior, drawn here with a
`StableRNG` of the class's seed. Each asserts the oracle's own thresholds on the recovered
quantities, and the exact identities that the class states. The classes are named as the oracle
numbers them. The covariance identities that every class repeats, `sigma = M F M' + D` and the
square root, are unit identities that `test_12k` and `test_12w` already pin, so they are not
repeated here.

The configuration maps one to one: a passthrough factor is `CompositeExposure` with no outlier
step and no scoring; the benchmark power is `bp`; the regression power is `MarketCapWeights(; p)`;
a sample factor prior is `EmpiricalPrior()`, and the default one is `PARITY_PE`, with
`PARITY_VE` as the idiosyncratic variance; the constrained family is `families`, the
neutralisation `neutralise`, the idiosyncratic correlation threshold `th`, the inverse-variance
weights `BlendedInverseVarianceWeights`, and the alpha a `CustomValueReturnForecast` under
`lambda = 0` with the oracle's Orthogonal Forecast Fit.
=#
using Statistics, LinearAlgebra
include(joinpath(@__DIR__, "parity_grid.jl"))

# A `ReturnsResult` of the returns `R` and its Panel Fields, all active and all in the estimation
# universe unless stated.
function rec_rd(R; fields = Pair{String, Matrix{Float64}}[],
                cats = Pair{String, Matrix{String}}[], mcap = ones(size(R)),
                amsk = trues(size(R)), emsk = trues(size(R)))
    pf = [NumericPanelInput(; name = "market_cap", vals = mcap);
          [NumericPanelInput(; name = k, vals = v) for (k, v) in fields];
          [CategoricalPanelInput(; name = k, vals = v) for (k, v) in cats]]
    return ReturnsResult(; nx = ["a$i" for i in axes(R, 2)], X = R,
                         pnl = asset_panel(pf; amsk = amsk, emsk = emsk))
end
bcast(v, T) = repeat(permutedims(v), T)
pass(f, fam = "style") = grid_pass(f; family = fam)
function rec_fit(rd; kw...)
    return prior(CrossSectionalFactorPrior(; lambda = 1, pe = PARITY_PE, ve = PARITY_VE,
                                           kw...), rd)
end
# The factor returns, the idiosyncratic returns and the true series on their rows.
fhat(pr) = pr.rr.csr.f
ehat(pr) = pr.rr.csr.eps
tail(x, n) = x[(end - n + 1):end, :]
eqw() = MarketCapWeights(; p = 0.0)
mse(a, b) = mean(abs2, a .- b)
# Three industries of ten assets, by block, with market caps 3, 2 and 1.
blocks(N) = [div(i - 1, 10) + 1 for i in 1:N]
function industry_fields(T, ind)
    return ["mkt_exp" => ones(T, length(ind));
            ["ind_$k" => bcast(Float64.(ind .== k), T) for k in 1:3]]
end
function industry_factors()
    return ["market" => pass("mkt_exp", "market");
            ["ind_$k" => pass("ind_$k", "industry") for k in 1:3]]
end

@testset "The recovery numbers of the oracle's test suite (#1391)" begin
    PO = PortfolioOptimisers

    @testset "1. A single factor with constant betas" begin
        rng = StableRNG(42)
        T, N = 2000, 50
        beta = 0.5 .+ rand(rng, N)
        se = 0.005 .+ 0.015 .* rand(rng, N)
        f = 0.01 .* randn(rng, T)
        R = beta' .* f .+ randn(rng, T, N) .* permutedims(se)
        pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(beta, T)]);
                     factors = ["beta" => pass("beta", "market")], pe = EmpiricalPrior(),
                     bp = 0.0, wa = eqw())
        fa = f[2:end]
        # Measured 0.985, 1.005, 0.34 and 0.047.
        @test cor(fhat(pr)[:, 1], fa) > 0.98
        @test isapprox(pr.fpr.sigma[1, 1], 0.01^2; rtol = 0.15)
        @test all(isapprox.(pr.rr.vs[end, :], se .^ 2; rtol = 0.4))
        @test maximum(i -> abs(cor(ehat(pr)[:, i], fa)), 1:N) < 0.08
        @test size(fhat(pr)) == (T - 1, 1) && size(ehat(pr)) == (T - 1, N)
        @test isapprox(pr.rr.M[:, 1], beta; rtol = 1e-10)
    end

    @testset "2. An intercept factor equals the benchmark return" begin
        rng = StableRNG(123)
        T, N = 500, 30
        fm = 0.008 .* randn(rng, T)
        R = fm .+ 0.01 .* randn(rng, T, N)
        mcap = exp.(10 .+ randn(rng, N))
        pr = rec_fit(rec_rd(R; fields = ["intercept" => ones(T, N)], mcap = bcast(mcap, T));
                     factors = ["intercept" => pass("intercept", "market")], bp = 1.0,
                     wa = MarketCapWeights(; p = 1.0))
        w = mcap / sum(mcap)
        @test isapprox(fhat(pr)[:, 1], R[2:end, :] * w; rtol = 1e-10)
        # The factor return is the cap-weighted return exactly, so its correlation with the
        # market is a property of the draw of the caps: `0.008 / sqrt(0.008^2 + 0.01^2 sum
        # w^2)` in expectation, 0.943 here, where the oracle's draw reaches its threshold 0.95.
        # Measured 0.944.
        ex = 0.008 / sqrt(0.008^2 + 0.01^2 * sum(abs2, w))
        @test abs(cor(fhat(pr)[:, 1], fm[2:end]) - ex) < 0.01
        @test all(isapprox.(pr.rr.M[:, 1], 1; rtol = 1e-10))
    end

    @testset "3. An exponentially weighted market beta" begin
        rng = StableRNG(77)
        T, N = 1500, 50
        beta = 0.5 .+ rand(rng, N)
        f = 0.01 .* randn(rng, T)
        R = beta' .* f .+ 0.005 .* randn(rng, T, N)
        ewb = CompositeExposure(;
                                descriptors = [EWMarketBeta(; half_life = 30, min_obs = 60)],
                                outlier = nothing, scoring = nothing, family = "market")
        pr = rec_fit(rec_rd(R); factors = ["beta" => ewb], bp = 0.0, wa = eqw())
        n = size(fhat(pr), 1)
        fa = f[(end - n + 1):end]
        # The first beta is at observation 60, and the exposure lag drops one more.
        @test n == T - 60
        @test cor(fhat(pr)[:, 1], fa) > 0.99
        @test cor(pr.rr.M[:, 1], beta) > 0.98
        @test maximum(i -> abs(cor(ehat(pr)[:, i], fa)), 1:N) < 0.05
    end

    @testset "4. Two uncorrelated factors" begin
        rng = StableRNG(99)
        T, N = 2000, 50
        b1 = 0.5 .+ rand(rng, N)
        b2 = 2 .* rand(rng, N) .- 1
        f1 = 0.01 .* randn(rng, T)
        f2 = 0.005 .* randn(rng, T)
        R = b1' .* f1 .+ b2' .* f2 .+ 0.005 .* randn(rng, T, N)
        pr = rec_fit(rec_rd(R; fields = ["beta1" => bcast(b1, T), "beta2" => bcast(b2, T)]);
                     factors = ["beta1" => pass("beta1"), "beta2" => pass("beta2")],
                     pe = EmpiricalPrior(), bp = 0.0, wa = eqw())
        F = pr.fpr.sigma
        g = fhat(pr)
        @test cor(g[:, 1], f1[2:end]) > 0.99
        @test cor(g[:, 2], f2[2:end]) > 0.96
        @test abs(cor(g[:, 1], f2[2:end])) < 0.05 && abs(cor(g[:, 2], f1[2:end])) < 0.05
        @test abs(F[1, 2] / sqrt(F[1, 1] * F[2, 2])) < 0.05
        @test isapprox(F[1, 1], 0.01^2; rtol = 0.05) &&
              isapprox(F[2, 2], 0.005^2; rtol = 0.1)
        @test isapprox(pr.rr.M, [b1 b2]; rtol = 1e-10)
    end

    @testset "5. Two correlated factors" begin
        rng = StableRNG(2024)
        T, N = 2000, 50
        b1 = 0.5 .+ rand(rng, N)
        b2 = 2 .* rand(rng, N) .- 1
        C = [0.01^2 0.6*0.01*0.008; 0.6*0.01*0.008 0.008^2]
        fs = randn(rng, T, 2) * cholesky(C).U
        R = b1' .* fs[:, 1] .+ b2' .* fs[:, 2] .+ 0.005 .* randn(rng, T, N)
        pr = rec_fit(rec_rd(R; fields = ["beta1" => bcast(b1, T), "beta2" => bcast(b2, T)]);
                     factors = ["beta1" => pass("beta1"), "beta2" => pass("beta2")],
                     pe = EmpiricalPrior(), bp = 0.0, wa = eqw())
        F = pr.fpr.sigma
        @test cor(fhat(pr)[:, 1], fs[2:end, 1]) > 0.99
        @test cor(fhat(pr)[:, 2], fs[2:end, 2]) > 0.98
        # The sample variance of this draw of the first factor is 5.3 % above `C[1, 1]`, so
        # the oracle's tolerance against `C` is a property of its own draw. Against the sample
        # covariance of the true factors, which is what a fit can recover, measured 2.1 %.
        @test all(isapprox.(F, cov(fs[2:end, :]); rtol = 0.05))
        @test isapprox(F[1, 2] / sqrt(F[1, 1] * F[2, 2]), 0.6; atol = 0.03)
    end

    @testset "6. An exposure lag of three with drifting betas" begin
        rng = StableRNG(314)
        T, N, L = 500, 30, 3
        bs = 0.3 .+ 0.4 .* rand(rng, N)
        be = 1.3 .+ 0.4 .* rand(rng, N)
        f = 0.01 .* randn(rng, T)
        eps = 0.005 .* randn(rng, T, N)
        betas = bs' .+ (be .- bs)' .* range(0, 1; length = T)
        R = betas .* f .+ eps
        pr = rec_fit(rec_rd(R; fields = ["beta" => betas]);
                     factors = ["beta" => pass("beta", "market")], lag = L, bp = 0.0,
                     wa = eqw())
        g = fhat(pr)[:, 1]
        Ral = R[(L + 1):end, :]
        @test length(g) == T - L
        @test isapprox(pr.rr.M[:, 1], betas[end, :]; rtol = 1e-10)
        @test !isapprox(pr.rr.M[:, 1], betas[end - L, :]; rtol = 1e-3)
        @test isapprox(ehat(pr), Ral .- betas[1:(end - L), :] .* g; rtol = 1e-10)
        @test mse(Ral, betas[(L + 1):end, :] .* g) > mean(abs2, ehat(pr))
        @test cor(g, f[(L + 1):end]) > 0.99
    end

    @testset "7. The estimation mask" begin
        rng = StableRNG(555)
        T, N, ne = 500, 30, 6
        betas = [fill(10.0, ne); 0.5 .+ rand(rng, N - ne)]
        f = 0.01 .* randn(rng, T)
        R = betas' .* f .+ 0.005 .* randn(rng, T, N)
        emsk = trues(T, N)
        emsk[:, 1:ne] .= false
        pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(betas, T)], emsk = emsk);
                     factors = ["beta" => pass("beta", "market")], bp = 0.0, wa = eqw(),
                     minra = 5)
        g = fhat(pr)[:, 1]
        n = length(g)
        b = betas[(ne + 1):end]
        @test isapprox(g, tail(R, n)[:, (ne + 1):end] * b / dot(b, b); rtol = 1e-10)
        @test !isapprox(g, tail(R, n) * betas / dot(betas, betas); rtol = 1e-3)
        @test isapprox(pr.rr.M[:, 1], betas; rtol = 1e-10)
        @test isapprox(ehat(pr), tail(R, n) .- betas' .* g; rtol = 1e-10)
        @test cor(g, f[(end - n + 1):end]) > 0.99
    end

    @testset "8. The sparse idiosyncratic correlation overlay" begin
        rng = StableRNG(777)
        T, N = 2000, 30
        betas = 0.5 .+ rand(rng, N)
        f = 0.01 .* randn(rng, T)
        eps = 0.005 .* randn(rng, T, N)
        z = 0.005 .* randn(rng, T, 2)
        eps[:, 1] = z[:, 1]
        eps[:, 2] = 0.5 .* z[:, 1] .+ sqrt(0.75) .* z[:, 2]
        R = betas' .* f .+ eps
        pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(betas, T)]);
                     factors = ["beta" => pass("beta", "market")], bp = 0.0, wa = eqw(),
                     th = 0.1)
        D = pr.rr.esigma
        off = [abs(D[i, j]) for i in 1:N, j in 1:N if i != j && !(minmax(i, j) == (1, 2))]
        @test D[1, 2] > 0
        @test maximum(off) < abs(D[1, 2])
        @test cor(fhat(pr)[:, 1], f[2:end]) > 0.99
    end

    # The industry draws of classes 9, 9b and 10b to 12: three blocks of ten assets with market
    # caps 3, 2 and 1, so the benchmark weights of the industries are 1/2, 1/3 and 1/6.
    T, N = 500, 30
    ind = blocks(N)
    mcap = Float64.([3, 2, 1][ind])
    wind = [30, 20, 10] / 60
    wb = mcap / sum(mcap)
    capw = MarketCapWeights(; p = 1.0)
    fam = ["industry" => nothing]

    @testset "9 and 9b. Basket-neutral industries" begin
        rng = StableRNG(888)
        fm = 0.01 .* randn(rng, T)
        fi = 0.01 .* randn(rng, T, 3)
        R = fm .+ fi[:, ind] .+ 0.005 .* randn(rng, T, N)
        pr = rec_fit(rec_rd(R; fields = industry_fields(T, ind), mcap = bcast(mcap, T));
                     factors = industry_factors(), families = fam, bp = 1.0, wa = capw)
        # The full basis: the raw factor returns, the factor covariance and the loadings.
        G = pr.rr.fr
        F = pr.fpr.sigma
        n = size(G, 1)
        @test pr.rr.nf == ["market", "ind_1", "ind_2", "ind_3"] && size(G, 2) == 4
        @test maximum(abs, G[:, 2:4] * wind) < 1e-12
        @test cor(G[:, 1], tail(fm .+ fi * wind, n)[:, 1]) > 0.99
        @test isapprox(G[:, 1], tail(R, n) * wb; rtol = 1e-10)
        @test rank(F) == 3
        ev = eigen(Symmetric(F))
        v = ev.vectors[:, argmin(abs.(ev.values))]
        e = normalize([0; wind])
        @test isapprox(abs(dot(v, e)), 1; atol = 1e-8)
        @test isposdef(Symmetric(pr.sigma))
        # 9b: the fit regresses on the reduced basis and states the full one.
        @test !isnothing(pr.rr.fcb) && size(pr.rr.L, 2) == 3 && size(pr.rr.M, 2) == 4
    end

    @testset "10. Factor neutralisation" begin
        rng = StableRNG(999)
        s0 = randn(rng, N)
        nz = randn(rng, N)
        fm, fs, fmo = (0.01 .* randn(rng, T) for _ in 1:3)
        R0 = 0.005 .* randn(rng, T, N)
        z(x) = (x .- mean(x)) ./ std(x; corrected = false)
        sz = z(s0)
        mz = z(0.7 .* s0 .+ sqrt(0.51) .* nz)
        R = fm .+ sz' .* fs .+ mz' .* fmo .+ R0
        rd = rec_rd(R;
                    fields = ["mkt_exp" => ones(T, N), "size" => bcast(sz, T),
                              "momentum" => bcast(mz, T)])
        facs = ["market" => pass("mkt_exp", "market"), "size" => pass("size"),
                "momentum" => pass("momentum")]
        p0 = rec_fit(rd; factors = facs, bp = 0.0, wa = eqw())
        p1 = rec_fit(rd; factors = facs, bp = 0.0, wa = eqw(),
                     neutralise = ["momentum" => ["size"]])
        E0, E1 = p0.rr.Ms, p1.rr.Ms
        @test abs(cor(E0[end, :, 2], E0[end, :, 3])) > 0.65
        cv(a, b) = mean((a .- mean(a)) .* (b .- mean(b)))
        @test all(t -> abs(cv(E1[t, :, 2], E1[t, :, 3])) < 1e-10, axes(E1, 1))
        @test all(t -> abs(mean(E1[t, :, 3])) < 1e-10 && abs(std(E1[t, :, 3]) - 1) < 1e-10,
                  axes(E1, 1))
        @test isapprox(E1[:, :, 2], E0[:, :, 2]; rtol = 1e-10)
        n = size(fhat(p0), 1)
        @test cor(fhat(p0)[:, 2], fs[(end - n + 1):end]) > 0.98
        @test cor(fhat(p0)[:, 3], fmo[(end - n + 1):end]) > 0.98
    end

    # The style characteristic of classes 10b and 10c leans on the first industry.
    function style_draw(seed)
        rng = StableRNG(seed)
        fm = 0.01 .* randn(rng, T)
        fi = 0.01 .* randn(rng, T, 3)
        fs = 0.01 .* randn(rng, T)
        sc = randn(rng, N) .+ 1.5 .* (ind .== 1)
        R = fm .+ fi[:, ind] .+ sc' .* fs .+ 0.005 .* randn(rng, T, N)
        return R, sc
    end
    # The benchmark-weighted cross products of the style loading with each industry.
    wdot(M, k) = sum(wb .* M[:, 5] .* M[:, k])

    @testset "10b. Neutralisation beside basket-neutral industries" begin
        R, sc = style_draw(777)
        rd = rec_rd(R; fields = [industry_fields(T, ind); "style_char" => bcast(sc, T)],
                    mcap = bcast(mcap, T))
        facs = [industry_factors(); "style" => pass("style_char")]
        raw = rec_fit(rd; factors = facs, families = fam, bp = 1.0, wa = capw)
        neu = rec_fit(rd; factors = facs, families = fam, bp = 1.0, wa = capw,
                      neutralise = ["style" => ["industry"]])
        @test abs(wdot(raw.rr.M, 2)) > 0.05
        @test all(k -> abs(wdot(neu.rr.M, k)) < 1e-8, 2:4)
        @test abs(sum(wb .* neu.rr.M[:, 5])) < 1e-8
        @test isposdef(Symmetric(neu.sigma))
    end

    @testset "10c. Demeaning within industries against neutralisation" begin
        R, sc = style_draw(888)
        rd = rec_rd(R; fields = [industry_fields(T, ind); "style_char" => bcast(sc, T)],
                    cats = ["industry_group" => bcast(string.(ind), T)],
                    mcap = bcast(mcap, T))
        grouped = CompositeExposure(; descriptors = [Passthrough(; field = "style_char")],
                                    outlier = nothing, group = "industry_group",
                                    family = "style")
        plain = CompositeExposure(; descriptors = [Passthrough(; field = "style_char")],
                                  outlier = nothing, family = "style")
        kw = (; families = fam, bp = 1.0, wa = capw)
        neut = ["style" => ["industry"]]
        A = rec_fit(rd; factors = [industry_factors(); "style" => grouped], kw...)
        B = rec_fit(rd; factors = [industry_factors(); "style" => plain], neutralise = neut,
                    kw...)
        C = rec_fit(rd; factors = [industry_factors(); "style" => grouped],
                    neutralise = neut, kw...)
        @test all(p -> all(k -> abs(wdot(p.rr.M, k)) < 1e-8, 2:4), (A, B, C))
        # Neutralisation is a no-op after the demeaning, and the two routes differ in the
        # spread within each industry.
        @test maximum(abs, A.rr.M[:, 5] - C.rr.M[:, 5]) < 1e-10
        @test cor(A.rr.M[:, 5], B.rr.M[:, 5]) < 0.9999
    end

    @testset "11. Inverse idiosyncratic variance regression weights" begin
        rng = StableRNG(777)
        betas = 0.5 .+ rand(rng, N)
        f = 0.01 .* randn(rng, T)
        eps = [0.002 .* randn(rng, T, 15) 0.02 .* randn(rng, T, 15)]
        mc = exp.(10 .+ randn(rng, N))
        R = betas' .* f .+ eps
        rd = rec_rd(R; fields = ["beta" => bcast(betas, T)], mcap = bcast(mc, T))
        # A half-life of ten observations.
        ivs = ExpWeightedVariance(; decay = 2.0^(-1 / 10), min_obs = 1)
        fit(; kw...) = rec_fit(rd; factors = ["beta" => pass("beta", "market")], bp = 0.0,
                               kw...)
        eq = fit(; wa = eqw())
        iv = fit(; wa = BlendedInverseVarianceWeights(; p = 0.0, lambda = 1.0), ve = ivs)
        bl = fit(; wa = BlendedInverseVarianceWeights(; p = 0.5, lambda = 0.5), ve = ivs)
        sq = fit(; wa = MarketCapWeights(; p = 0.5))
        n = size(fhat(eq), 1)
        fa = f[(end - n + 1):end]
        rw = iv.rr.rw
        @test mean(rw[:, 1:15]) > mean(rw[:, 16:30])
        @test mse(fhat(iv)[:, 1], fa) < mse(fhat(eq)[:, 1], fa)
        @test cor(fhat(iv)[:, 1], fa) > 0.95
        @test all(isapprox.(eq.rr.rw, 1; rtol = 1e-10))
        @test isapprox(sq.rr.rw[1, :], sqrt.(mc); rtol = 1e-10)
        @test mse(fhat(bl)[:, 1], fa) < mse(fhat(sq)[:, 1], fa)
        function ratio(W)
            rows = [r ./ sum(r) for r in eachrow(W) if sum(r) > 0]
            return mean(r -> mean(r[1:15]), rows) / mean(r -> mean(r[16:30]), rows)
        end
        @test ratio(sq.rr.rw) < ratio(bl.rr.rw) < ratio(iv.rr.rw)
    end

    @testset "12. The intercept beside industries and a scored style" begin
        rng = StableRNG(999)
        fm = 0.01 .* randn(rng, T)
        fi = 0.01 .* randn(rng, T, 3)
        fs = 0.01 .* randn(rng, T)
        sc = 0.5 .+ rand(rng, N)
        R = fm .+ fi[:, ind] .+ sc' .* fs .+ 0.005 .* randn(rng, T, N)
        rd = rec_rd(R; fields = [industry_fields(T, ind); "style_char" => bcast(sc, T)],
                    mcap = bcast(mcap, T))
        facs = [industry_factors(); "style" => grid_scored("style_char")]
        same = rec_fit(rd; factors = facs, families = fam, bp = 1.0, wa = capw)
        diff = rec_fit(rd; factors = facs, families = fam, bp = 1.0,
                       wa = MarketCapWeights(; p = 0.5))
        @test length(same.rr.nf) == 5
        for p in (same, diff)
            G = p.rr.fr
            n = size(G, 1)
            # The caps are constant within each industry, so both regression weights give
            # the cap-weighted return.
            @test isapprox(G[:, 1], tail(R, n) * wb; rtol = 1e-10)
            @test maximum(abs, G[:, 2:4] * wind) < 1e-12
            @test isposdef(Symmetric(p.sigma))
        end
        n = size(same.rr.fr, 1)
        @test cor(same.rr.fr[:, 5], fs[(end - n + 1):end]) > 0.90
    end

    @testset "13. The regression diagnostics of a true and a noise factor" begin
        rng = StableRNG(1234)
        N13 = 50
        betas = 0.5 .+ rand(rng, N13)
        f = 0.01 .* randn(rng, T)
        R = betas' .* f .+ 0.005 .* randn(rng, T, N13)
        noise = randn(rng, T, N13)
        rd = rec_rd(R; fields = ["beta" => bcast(betas, T), "noise" => noise])
        full = rec_fit(rd; factors = ["signal" => pass("beta"), "noise" => pass("noise")],
                       bp = 0.0, wa = eqw()).rr
        truef = rec_fit(rd; factors = ["signal" => pass("beta")], bp = 0.0, wa = eqw()).rr
        r2f, r2t = cs_regression_r2(full), cs_regression_r2(truef)
        t = cs_regression_t_stats(full).X
        rate = cs_regression_t_stat_exceedance_rate(full; threshold = 2).X
        @test mean(r2t) > 0.1
        @test minimum(r2f .- r2t) >= -1e-10
        @test size(t, 2) == 2
        @test median(abs.(t[:, 1])) > 3 && median(abs.(t[:, 2])) < 2
        @test rate[1] > 0.8 && rate[2] < 0.2
        @test mean(cs_regression_aic(truef)) < mean(cs_regression_aic(full))
        @test mean(cs_regression_bic(truef)) < mean(cs_regression_bic(full))
        a2 = cs_regression_adjusted_r2(full)
        @test length(a2) == length(r2f)
        @test all(i -> !(isfinite(a2[i]) && isfinite(r2f[i])) || a2[i] <= r2f[i] + 1e-12,
                  eachindex(a2))
    end

    @testset "14. Time-varying market caps" begin
        rng = StableRNG(314)
        tg = range(0, 1; length = T)
        base = hcat((3 .- 2 .* tg) .* ones(1, 10), fill(2.0, T, 10),
                    (1 .+ 2 .* tg) .* ones(1, 10))
        scale = 0.8 .+ 0.4 .* rand(rng, N)
        fm = 0.01 .* randn(rng, T)
        fi = 0.005 .* randn(rng, T, 3)
        R = fm .+ 0.005 .* randn(rng, T, N) .+ fi[:, ind]
        mc = base .* permutedims(scale)
        pr = rec_fit(rec_rd(R; fields = industry_fields(T, ind), mcap = mc);
                     factors = industry_factors(), families = fam, bp = 1.0, wa = capw)
        G = pr.rr.fr
        n = size(G, 1)
        # The benchmark and the regression weights read the caps of the exposure date.
        lagcap = mc[(T - n):(T - 1), :]
        @test isapprox(G[:, 1],
                       vec(sum(lagcap ./ sum(lagcap; dims = 2) .* tail(R, n); dims = 2));
                       rtol = 1e-10, atol = 1e-12)
        @test isapprox(pr.rr.rw, lagcap; rtol = 1e-10, atol = 1e-12)
        bs = hcat((sum(lagcap[:, ind .== k]; dims = 2) for k in 1:3)...) ./
             sum(lagcap; dims = 2)
        @test maximum(abs, sum(bs .* G[:, 2:4]; dims = 2)) < 1e-10
        @test !isapprox(mean(G[1:50, 1]), mean(G[(end - 49):end, 1]); atol = 1e-6)
        @test all(isapprox.(pr.rr.M[:, 1], 1; atol = 1e-10))
        @test count(<(1e-12), abs.(eigvals(Symmetric(pr.fpr.sigma)))) == 1
        @test isposdef(Symmetric(pr.sigma))
    end

    @testset "15. Listings and delistings" begin
        rng = StableRNG(4242)
        N15, nf, t0 = 50, 40, 250
        betas = 0.5 .+ rand(rng, N15)
        f = 0.01 .* randn(rng, T)
        Rf = betas' .* f .+ 0.005 .* randn(rng, T, N15)
        R = copy(Rf)
        amsk = trues(T, N15)
        amsk[1:t0, (nf + 1):end] .= false
        R[.!amsk] .= NaN
        # The oracle states `NaN` caps and betas before the listing. A panel consumer ignores
        # an inactive cell (#1411), and the builder refuses a blank without a fill policy, so
        # the inactive cells here hold the values of the active ones.
        facs = ["f1" => pass("beta")]
        pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(betas, T)], amsk = amsk,
                            emsk = amsk); factors = facs, bp = 0.0, wa = eqw())
        ctrl = rec_fit(rec_rd(Rf[:, 1:nf]; fields = ["beta" => bcast(betas[1:nf], T)]);
                       factors = facs, bp = 0.0, wa = eqw())
        g = fhat(pr)[:, 1]
        @test size(pr.mu) == (N15,) && size(pr.sigma) == (N15, N15)
        @test all(isfinite, pr.rr.M) && length(g) == T - 1
        @test isapprox(g[1:(t0 - 1)], fhat(ctrl)[1:(t0 - 1), 1]; rtol = 1e-10)
        @test cor(g, f[2:end]) > 0.95
        @test isposdef(Symmetric(pr.sigma)) && all(isfinite, pr.rr.esigma)
        @test isapprox(pr.rr.M[(nf + 1):end, 1], betas[(nf + 1):end]; rtol = 1e-10)
    end

    @testset "17. Neutralisation of a family against a family" begin
        T17 = 200
        rng = StableRNG(1234)
        lab = [mod(i - 1, 3) + 1 for i in 1:N]
        size0 = randn(rng, N)
        mom0 = randn(rng, N)
        for k in 1:3
            size0[lab .== k] .+= 2 * (k - 1) + 0.5 * randn(rng)
            mom0[lab .== k] .+= -1.5 * (k - 1) + 0.5 * randn(rng)
        end
        mc = 1 .+ 99 .* rand(rng, T17, N)
        fm = 0.01 .* randn(rng, T17)
        fi = 0.01 .* randn(rng, T17, 3)
        fs = 0.01 .* randn(rng, T17)
        fmo = 0.01 .* randn(rng, T17)
        R = fm .+ fi[:, lab] .+ size0' .* fs .+ mom0' .* fmo .+ 0.005 .* randn(rng, T17, N)
        rd = rec_rd(R;
                    fields = [industry_fields(T17, lab); "size" => bcast(size0, T17);
                              "momentum" => bcast(mom0, T17)], mcap = mc)
        facs = [industry_factors(); "size" => pass("size"); "momentum" => pass("momentum")]
        kw = (; factors = facs, families = fam, bp = 1.0, wa = MarketCapWeights(; p = 0.5))
        neu = rec_fit(rd; neutralise = ["style" => ["industry"]], kw...)
        ctl = rec_fit(rd; kw...)
        E, W = neu.rr.Ms, neu.rr.bw
        @test all(abs(sum(W[t, :] .* E[t, :, s] .* E[t, :, j])) < 1e-8
                  for t in axes(E, 1), s in 5:6, j in 2:4)
        @test all(s -> maximum(abs, E[:, :, s] - ctl.rr.Ms[:, :, s]) > 0.01, 5:6)
    end

    @testset "14a and 14b. A spanned and an orthogonal alpha" begin
        N14 = 40
        function alpha_fit(seed, orth, c)
            rng = StableRNG(seed)
            betas = 0.5 .+ rand(rng, N14)
            f = 0.01 .* randn(rng, T)
            R = betas' .* f .+ 0.005 .* randn(rng, T, N14)
            delta = zeros(N14)
            if orth
                raw = 0.001 .* randn(rng, N14)
                Bm = [ones(N14) betas]
                delta = raw - Bm * (Bm \ raw)
                delta .*= 0.001 / (std(delta; corrected = false) + 1e-12)
            end
            alpha = 0.002 .* betas .+ delta
            pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(betas, T)]);
                         factors = ["f1" => pass("beta")], bp = 0.0, wa = eqw(),
                         rfe = CustomValueReturnForecast(; mu = alpha), lambda = 0, c = c,
                         ofit = UnadjustedForecast())
            return pr, betas, alpha, delta
        end
        pr, betas = alpha_fit(77, false, 1.0)
        span = pr.rr.M * pr.fpr.mu
        @test maximum(abs, pr.mu - 0.002 .* betas) < 1e-10
        @test maximum(abs, pr.mu - span) < 1e-10
        @test maximum(abs, span - 0.002 .* betas) < 1e-10
        full, _, alpha, delta = alpha_fit(88, true, 1.0)
        zero, = alpha_fit(88, true, 0.0)
        @test maximum(abs, full.mu - alpha) < 1e-10
        @test maximum(abs, zero.mu - zero.rr.M * zero.fpr.mu) < 1e-10
        @test maximum(abs, full.mu - zero.mu) > 1e-5
        @test maximum(abs, full.mu - full.rr.M * full.fpr.mu - delta) < 1e-10
    end
end
