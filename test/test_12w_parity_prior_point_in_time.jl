#=
Parity of the Cross-Sectional Factor Prior on a point-in-time panel, output by output (#1384,
map #1375).

Every stored case under `test/assets/Parity_CrossSectionalFactorPrior_<Case>_<Output>.csv.gz` is
the oracle fit of a market factor and the two passthrough styles, on the panels of
`parity_harness.jl`, which list late, delist, list again, skip a holiday and keep an asset outside
the estimation universe. The factor covariance and the idiosyncratic variance are stated
explicitly at the oracle's defaults, so a change of our own defaults (#1383) does not move these
cases:

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

@testset "The prior on a point-in-time panel, at parity (#1384)" begin
    PO = PortfolioOptimisers
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                        ce = RegimeAdjustedExpWeightedCovariance(; centred = true,
                                                                 regime_lohi_mult = (0.7,
                                                                                     1.6)))
    ve = RegimeAdjustedExpWeightedVariance(; centred = true, regime_lohi_mult = (0.7, 1.6),
                                           min_val = 0.0)
    est(; kw...) = CrossSectionalFactorPrior(; factors = factors, pe = pe, ve = ve, kw...)
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

    @testset "Every output of the default fit, $(c)" for (c, pr, i) in
                                                         (("PitSmall", prs, rs),
                                                          ("PitLarge", prl,
                                                           findall(isfinite, prl.mu)))
        rr = pr.rr
        K = size(rr.L, 2)
        @test findall(isfinite, pr.mu) == i
        # Measured maxrel 1.9e-15 and 7.0e-15.
        @test parity_compare(pr.mu[i], loadv(c, "Mu")[i]; name = "$(c) mu").ok
        # A covariance compares against its largest entry (#1376). Measured maxscaled 3.0e-13
        # and 3.4e-13, and maxrel 1.4e-11 cell by cell.
        @test parity_compare(pr.sigma[i, i], load(c, "Sigma")[i, i]; scale = :array,
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
        # comes from a cancellation. Measured maxscaled 6.3e-16, and maxrel 8.2e-13 cell by
        # cell (1.5e-11 on the large panel). The column of an asset that is not investable is
        # `NaN` on both sides.
        @test parity_compare(prs.X, load(c, "Scenarios"); scale = :array, name = "X").ok
        @test isnothing(prs.w) && isnothing(prs.fpr.w)
    end

    @testset "An asset in its warm-up: a deliberate difference" begin
        # The asset lists again inside the history, and its idiosyncratic variance has not
        # warmed up at the latest observation, so neither side states its variance. The oracle
        # still states its mean and its covariances with the other assets beside a `NaN`
        # variance. A distribution with a mean and no variance is not one an optimiser can
        # read, and the oracle drops the asset by its own finite-moment rule. Its covariances
        # are also not a stable answer: under a positive threshold they are `NaN` on the
        # oracle's side too, below. The prior states no moment of an asset outside the
        # investable set, so it writes `NaN` over its mean and its row and column, and the
        # systematic part the oracle states stays one line away on the block.
        om = loadv("PitSmall", "Mu")
        os = load("PitSmall", "Sigma")
        @test isnan(prs.mu[i4]) && all(isnan, prs.sigma[i4, :])
        @test isfinite(om[i4]) && isnan(os[i4, i4])
        M = prs.rr.M
        # Measured maxrel 3.1e-16 and 7.9e-16.
        @test parity_compare([(M * prs.fpr.mu + prs.rr.b)[i4]], [om[i4]]; name = "mu4").ok
        @test parity_compare((M * prs.fpr.sigma * transpose(M))[i4, rs], os[i4, rs];
                             scale = :array, name = "sigma4").ok
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
        @test parity_compare(pr.mu[rs], loadv(c, "Mu")[rs]; name = "overlay mu").ok
        # Measured maxscaled 3.0e-13, and maxrel 1.5e-15 and 5.3e-15.
        @test parity_compare(pr.sigma[rs, rs], load(c, "Sigma")[rs, rs]; scale = :array,
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
                                                                               centred = true),
                                                         nothing, Sf, pr.rr.vs[end, :], amr)
        @test count(!iszero, Df[rs, rs]) - length(rs) == 72

        # On the large panel the block before its repair agrees on every pair but those of
        # the asset with a holiday. There the oracle keeps the covariance of each pair on the
        # holiday, and ours keeps the correlation, which keeps the state positive
        # semidefinite (ADR 0181). The thresholded block is not positive definite, so the
        # repair binds: ours takes the nearest correlation by Newton, and the oracle clips
        # the eigenvalues, which #1412 builds as an algorithm of `Posdef` (ADR 0186).
        pl = prior(est(; th = 0.1), fxl.rd)
        eps = pl.rr.csr.eps
        amr = fxl.amsk[(end - size(eps, 1) + 1):end, :]
        Z = PO.cross_sectional_standardised_residuals(eps, pl.rr.vs, amr; filled = false)
        B = PO.cross_sectional_idiosyncratic_covariance(0.1,
                                                        ExpWeightedCovariance(;
                                                                              centred = true),
                                                        nothing, Z, pl.rr.vs[end, :], amr)
        E = load("PitLargeOverlayRaw", "IdioCov")
        h = fxl.at.holiday[2]
        k = setdiff(axes(B, 1), h)
        # Measured maxrel 2.2e-15 off the holiday, and 2.3e-3 on its pairs.
        @test parity_compare(B[k, k], E[k, k]; name = "raw block").ok
        @test !parity_compare(B[h, k], E[h, k]; name = "raw holiday").ok
        il = findall(isfinite, pl.mu)
        Br = B[il, il]
        @test !LinearAlgebra.isposdef(LinearAlgebra.Symmetric(Br))
        posdef!(Posdef(), Br)
        @test Br == pl.rr.esigma[il, il]
    end

    @testset "The scenario weights and a Scenario Cap" begin
        # The oracle keeps the weights of the scenarios it keeps, and divides them by their
        # sum. Every factor prior of the library carries one scenario for each fitted
        # observation or fewer, so every weight is kept, and the weights of an entropy-pooling
        # prior already sum to one. Measured maxrel 3.0e-16.
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
        cap = EmpiricalPrior(; me = pe.me, ce = pe.ce, max_scenarios = 30)
        pc = prior(est(; minra = 5, pe = cap), fxs.rd)
        @test size(pc.X, 1) == size(pc.fpr.X, 1) == size(pc.o_X, 1) == 30
        @test isequal(pc.X, prs.X[(end - 29):end, :])
        @test isequal(pc.fpr.X, prs.fpr.X[(end - 29):end, :])
        @test isequal(pc.o_X, prs.o_X[(end - 29):end, :])
        @test isequal(pc.mu, prs.mu) && isequal(pc.sigma, prs.sigma)
        @test pc.ens == size(prs.X, 1)
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
                                   codes = Matrix(f.codes), omsk = f.omsk)
        pf = [j == k ? f3 : x for (j, x) in enumerate(rv.pnl.pf)]
        rv3 = ReturnsResult(; nx = rv.nx, X = rv.X, ne = rv.ne, E = rv.E,
                            pnl = AssetPanel(pf, rv.pnl.amsk, rv.pnl.emsk))
        fi = ["industry" => OneHotExposure(; field = "industry", family = "industry"),
              "style1" => mpass("style1"), "style2" => mpass("style2")]
        e(; kw...) = CrossSectionalFactorPrior(; factors = fi, pe = pe, ve = ve, minra = 5,
                                               kw...)
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
        @test isapprox(transpose(pr.chol[:, i]) * pr.chol[:, i], pr.sigma[i, i];
                       rtol = 1e-12)
    end

    @testset "The refusals agree with the oracle at their boundaries" begin
        # Each pair sits on one side of a boundary. The oracle refuses the same four cases
        # and fits the same three: the factor covariance needs 40 observations, the
        # idiosyncratic variance of `Idio*` needs 60, the coverage refuses one asset above the
        # fewest eligible assets of an observation, and a Descriptor that is never finite
        # leaves the whole history cold.
        rows(T) = PO.port_opt_view(fxs.rd, 1:T, :)
        ve60 = RegimeAdjustedExpWeightedVariance(; centred = true,
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
        # The fits on the other side of the boundary. Measured maxrel 2.3e-15 and 1.1e-14. The
        # oracle states a mean for each active asset in its warm-up, see above.
        @testset "$(c)" for (c, rd, kw) in (("Rows41", rows(41), (;)),
                                            ("Idio61", rows(61), (; ve = ve60)))
            pr = prior(est(; minra = 5, kw...), rd)
            i = findall(isfinite, pr.mu)
            o = loadv(c, "Mu")
            @test all(isfinite, o[i])
            @test parity_compare(pr.mu[i], o[i]; name = "$(c) mu").ok
        end
    end
end
