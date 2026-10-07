#=
Parity of the uncertainty sets from a fitted prior, and of the weights of the optimisations that
read them (#1390, map #1375).

Every stored case under `test/assets/Parity_<Unit>_<Case>_<Output>.csv.gz` is an oracle output:

  - `OrthogonalUncertaintySet`, `PitSmall<Metric>` and `PitLarge<Metric>`: both orthogonal sets
    fitted on the Cross-Sectional Factor Prior of `test_12w` on the two panels of
    `parity_harness.jl`, reduced to the Investable Mask, under each of the four metrics.
    `FamTwo<Metric>`: the same on the `FamTwo` prior of `parity_grid.jl`, whose two constrained
    families reduce the loadings. A singular vector and a QR factor carry an arbitrary sign, so
    the geometry is stored as `L * L'` and the basis as `Q * Q'`.
  - `OrthogonalUncertaintySet`, `PitSmallView` and `PitLargeView`: both sets refitted on a
    sub-universe of the investable assets.
  - `OrthogonalMeanRisk`: the weights of `MeanRisk` with the variance risk measure, long only and
    fully invested, one column per case of `MR_CASES`, at a solver gap of `1e-12` on both sides.
  - `NormBallUncertaintySet`: the norm-ball sets of the Normal and the bootstrap estimators on a
    plain return matrix. The bootstrap reads the resamples that the library draws.
  - `NormBallMeanRisk`: the weights of `MeanRisk` with a stated norm ball on each axis, one column
    per case of `NB_CASES`, so that only the JuMP terms that consume the sets are compared.

The optimisations compare the objective that each side's weights reach, from the closed forms
below, and the weights themselves. The objective is flat near the optimum, so the weights agree
to a looser tolerance than the objective does.
=#
using JuMP, Clarabel, Distributions, Statistics
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))

@testset "The uncertainty sets from a fitted prior, at parity (#1390)" begin
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
    load(u, c, o) = parity_load(u, c, o)
    loadv(u, c, o) = vec(load(u, c, o))
    prs = prior(est(; minra = 5), parity_small_panel().rd)
    prl = prior(est(), parity_large_panel().rd)
    rdg = grid_fixture(parity_large_panel())
    prf = prior(grid_prior("FamTwo", rdg), rdg)
    metrics = ["InvIdio" => InverseIdiosyncraticVarianceMetric(),
               "Regression" => RegressionWeightMetric(),
               "Benchmark" => BenchmarkWeightMetric(), "Identity" => IdentityMetric()]
    slv = Solver(; name = :clarabel_1390, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                 "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                                 "max_iter" => 500))

    @testset "Both orthogonal sets on the prior of a real fit, $(c)" for (c, pr) in
                                                                         (("PitSmall", prs),
                                                                          ("PitLarge", prl),
                                                                          ("FamTwo", prf))
        i = findall(PO.investable_mask(pr))
        o = setdiff(axes(pr.mu, 1), i)
        for (mn, m) in metrics
            case = "$(c)$(mn)"
            radius = loadv("OrthogonalUncertaintySet", case, "Radius")
            for (out, s, k) in (("MuLLt", IdentityScaling(), 1),
                                ("MuIdioLLt", IdiosyncraticVarianceScaling(), 2))
                mu = mu_ucs(OrthogonalUncertaintySet(; metric = m, scaling = s), pr)
                L = mu.L[i, :]
                # Measured maxscaled 2.2e-14 on the panels and 5.2e-14 on `FamTwo`.
                @test parity_compare(L * transpose(L),
                                     load("OrthogonalUncertaintySet", case, out);
                                     scale = :array, name = "$(case) $(out)").ok
                # The radius reads the rank of the subspace on both sides. Measured 3.8e-16.
                @test parity_compare([mu.kappa], [radius[k]]; name = "$(case) radius").ok
                @test all(iszero, mu.L[o, :])
            end
            sg = sigma_ucs(OrthogonalUncertaintySet(; metric = m), pr)
            Q = sg.Q[i, :]
            # Measured maxscaled 1.1e-15, and maxrel 1.0e-15 on `C`.
            @test parity_compare(Q * transpose(Q),
                                 load("OrthogonalUncertaintySet", case, "CovQQt");
                                 scale = :array, name = "$(case) QQt").ok
            @test parity_compare(sg.C[i], loadv("OrthogonalUncertaintySet", case, "CovC");
                                 name = "$(case) C").ok
            @test all(iszero, sg.Q[o, :]) && all(iszero, sg.C[o])
        end
    end

    @testset "The standalone fit answers the full universe, and a view recovers the reduced fit" begin
        # The oracle refuses a set fitted on the full prior, whose assets outside the
        # Investable Mask carry `NaN` weights. The library reduces the prior, fits, and writes
        # the sets back over the full universe with zero rows outside the mask (#1062), so a
        # set fitted standalone can be handed to an optimiser on the full universe. Better:
        # the reduced block equals the oracle's set, and the full block adds zeros only.
        for pr in (prs, prl)
            i = findall(PO.investable_mask(pr))
            ue = OrthogonalUncertaintySet(; scaling = IdiosyncraticVarianceScaling())
            m, s = ucs(ue, pr)
            mr, sr = ucs(ue, PO.port_opt_view(pr, i))
            mv = PO.port_opt_view(m, i)
            sv = PO.port_opt_view(s, i)
            @test isequal(m.val, pr.mu) && isequal(s.val, pr.sigma)
            @test isapprox(mv.L * transpose(mv.L), mr.L * transpose(mr.L); rtol = 1e-14)
            @test isapprox(sv.Q * transpose(sv.Q), sr.Q * transpose(sr.Q); atol = 1e-14)
            @test sv.C == sr.C && mv.kappa == mr.kappa && sv.kappa == sr.kappa
        end
    end

    @testset "The views on a sub-universe, $(c)" for (c, pr, jj) in
                                                     (("PitSmallView", prs, 1:6),
                                                      ("PitLargeView", prl, 1:2:39))
        j = findall(PO.investable_mask(pr))[jj]
        ue = OrthogonalUncertaintySet()
        m, s = ucs(ue, pr)
        mr, sr = ucs(ue, PO.port_opt_view(pr, j))
        sv = PO.port_opt_view(s, j)
        QQt = load("OrthogonalUncertaintySet", c, "CovQQt")
        LLt = load("OrthogonalUncertaintySet", c, "MuLLt")
        # Both sets refitted on the sub-universe equal the oracle's refit. Measured maxscaled
        # 7.8e-16 (Q Q') and 1.2e-15 (L L').
        @test parity_compare(sr.Q * transpose(sr.Q), QQt; scale = :array,
                             name = "$(c) refit QQt").ok
        @test parity_compare(collect(sr.C), loadv("OrthogonalUncertaintySet", c, "CovC");
                             name = "$(c) refit C").ok
        @test parity_compare(mr.L * transpose(mr.L), LLt; scale = :array,
                             name = "$(c) refit LLt").ok
        # The view of a fitted set is the projection of the set, not the refit (ADR 0189,
        # #1424): a cluster portfolio pays what it pays as a portfolio of the full universe.
        # The mean view slices the rows of `L`, and the covariance view keeps the sliced rows
        # of `Q` and completes them with `R`. The refit is a different set: it spares the span
        # of the cluster's own loadings, which the full set does not spare.
        mv = PO.port_opt_view(m, j)
        @test mv.L == m.L[j, :]
        @test !isapprox(mv.L * transpose(mv.L), LLt; rtol = 1e-2)
        @test sv.C == s.C[j] && sv.Q == s.Q[j, :]
        M = Diagonal(s.C) * (I - s.Q * transpose(s.Q)) * Diagonal(s.C)
        Mjj = M[j, j]
        wj = fill(inv(length(j)), length(j))
        # Each set names its own centre, so the penalty is the worst case less the nominal
        # variance at that centre.
        penalty(set) = PO.ucs_variance(set, set.val, wj) - dot(wj, set.val, wj)
        @test isapprox(penalty(sv), s.kappa * dot(wj, Mjj, wj); rtol = 1e-12)
        # Per unit of radius, the refit charges the equal-weight portfolio less than the
        # projection: 6.9 % and 14.9 % of it on the two panels (#1424).
        @test penalty(sr) / sr.kappa < 0.5 * penalty(sv) / sv.kappa
    end

    # The objective each side's weights reach, from the closed forms. The ratio is over the
    # square root of the variance, the degree of the compact term and the nominal variance.
    function objective(pr, obj, um, uc, w)
        i = findall(PO.investable_mask(pr))
        prr = PO.port_opt_view(pr, i)
        wi = w[i]
        risk = if isnothing(uc)
            dot(wi, prr.sigma, wi)
        else
            PO.ucs_variance(sigma_ucs(uc, prr), prr.sigma, wi)
        end
        ret = dot(prr.mu, wi)
        if !isnothing(um)
            ms = mu_ucs(um, prr)
            ret -= ms.kappa * norm(transpose(ms.L) * wi)
        end
        isa(obj, MinimumRisk) && return -risk
        isa(obj, MaximumUtility) && return ret - 4 * risk
        return ret / sqrt(risk)
    end
    uem(q) = OrthogonalUncertaintySet(; q = q)
    uec(k) = OrthogonalUncertaintySet(; kappa = k)
    util = MaximumUtility(; l = 4.0)
    # The oracle's confidence level 0.9 is `q = 0.1`, and its radius is `kappa`. The `Ub`
    # cases bound every weight, which moves the optimum off the span that the sets spare, so
    # their penalties are positive and `kappa = 10` differs from `kappa = 1`.
    MR_CASES = ["Nominal" => (MinimumRisk(), nothing, nothing, false),
                "NominalUtility" => (util, nothing, nothing, false),
                "NominalRatio" => (MaximumRatio(), nothing, nothing, false),
                "MinRiskCov" => (MinimumRisk(), nothing, uec(1.0), false),
                "MinRiskCov10" => (MinimumRisk(), nothing, uec(10.0), false),
                "UtilityMu" => (util, uem(0.1), nothing, false),
                "UtilityMuIdio" => (util,
                                    OrthogonalUncertaintySet(; q = 0.1,
                                                             scaling = IdiosyncraticVarianceScaling()),
                                    nothing, false),
                "UtilityCov" => (util, nothing, uec(1.0), false),
                "UtilityBoth" => (util, uem(0.1), uec(1.0), false),
                "RatioCov" => (MaximumRatio(), nothing, uec(1.0), false),
                "RatioMu" => (MaximumRatio(), uem(0.5), nothing, false),
                "RatioBoth" => (MaximumRatio(), uem(0.5), uec(1.0), false),
                "UbMinRiskCov" => (MinimumRisk(), nothing, uec(1.0), true),
                "UbMinRiskCov10" => (MinimumRisk(), nothing, uec(10.0), true),
                "UbUtilityMu" => (util, uem(0.1), nothing, true),
                "UbUtilityBoth" => (util, uem(0.1), uec(1.0), true),
                "UbRatioCov" => (MaximumRatio(), nothing, uec(1.0), true),
                "UbRatioBoth" => (MaximumRatio(), uem(0.5), uec(1.0), true)]

    @testset "MeanRisk with the orthogonal sets, $(c)" for (c, pr, ub) in
                                                           (("PitSmall", prs, 0.15),
                                                            ("PitLarge", prl, 0.04))
        W = load("OrthogonalMeanRisk", c, "Weights")
        for (k, (name, (obj, um, uc, bounded))) in enumerate(MR_CASES)
            ret = isnothing(um) ? ArithmeticReturn() : ArithmeticReturn(; ucs = um)
            r = isnothing(uc) ? Variance() : UncertaintySetVariance(; ucs = uc)
            wb = bounded ? WeightBounds(; lb = 0.0, ub = ub) : WeightBounds()
            res = optimise(MeanRisk(; r = r, obj = obj,
                                    opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret,
                                                        wb = wb)))
            @test isa(res.retcode, PO.OptimisationSuccess)
            wo = W[:, k]
            if all(isnan, wo)
                # `UbRatioBoth` on the large panel: no portfolio inside the bounds has a
                # positive worst-case return (the best is -0.034), so no tangency portfolio
                # exists. The oracle's solver fails. The homogenised scale of the library sits
                # on its floor, which is the signal ADR 0126 states. Deliberate difference.
                k_ = PO.get_k(res.model)
                @test isapprox(JuMP.value(k_), JuMP.lower_bound(k_); rtol = 1e-6)
                continue
            end
            fj = objective(pr, obj, um, uc, res.w)
            fo = objective(pr, obj, um, uc, wo)
            # Measured relative difference of the objective 4.5e-9 at most, and of the
            # weights 8.0e-6 at most, where the objective is flat.
            @test abs(fj - fo) <= 5e-8 * abs(fo)
            @test isapprox(res.w, wo; atol = 5e-5)
        end
    end

    @testset "The same set, estimated or pre-built, gives the same weights" begin
        for pr in (prs, prl), name in
                              ("RatioCov", "RatioBoth", "UtilityBoth", "MinRiskCov")

            obj, um, uc, _ = Dict(MR_CASES)[name]
            opt(ret, r) = MeanRisk(; r = r, obj = obj,
                                   opt = JuMPOptimiser(; pe = pr, slv = slv, ret = ret))
            ret = isnothing(um) ? ArithmeticReturn() : ArithmeticReturn(; ucs = um)
            retb = if isnothing(um)
                ArithmeticReturn()
            else
                ArithmeticReturn(; ucs = mu_ucs(um, pr))
            end
            r = UncertaintySetVariance(; ucs = uc)
            rb = UncertaintySetVariance(; ucs = sigma_ucs(uc, pr))
            res = optimise(opt(ret, r))
            resb = optimise(opt(retb, rb))
            # Measured 2.3e-6 at most. The scale of a ratio stays O(1) (#924).
            @test isapprox(res.w, resb.w; atol = 1e-5)
            if isa(obj, MaximumRatio)
                @test JuMP.value(PO.get_k(res.model)) > 0.1
            end
        end
    end

    # A plain return matrix for the Normal and the bootstrap estimators.
    rngx = StableRNG(1390)
    A = randn(rngx, 5, 5) * 0.01
    X = randn(rngx, 120, 5) * transpose(A) .+ 0.001 .* transpose(1:5)
    T, N = size(X)
    nb(d) = NormBallUncertaintySetAlgorithm(; diagonal = d)

    @testset "The norm-ball member of the Normal and bootstrap estimators" begin
        function mk(kind, alg)
            return if kind == "Empirical"
                NormalUncertaintySet(; alg = alg)
            else
                ARCHUncertaintySet(; alg = alg, n_sim = 400, block_size = 3,
                                   rng = StableRNG(11))
            end
        end
        for (kind, ue) in (("Empirical", d -> mk("Empirical", nb(d))),
                           ("Bootstrap", d -> mk("Bootstrap", nb(d))))
            radius = loadv("NormBallUncertaintySet", kind, "Radius")
            for (k, (d, dn)) in enumerate(((true, "Diag"), (false, "Full")))
                m, s = ucs(ue(d), X)
                # Measured maxscaled 2.4e-15 on the mean.
                @test parity_compare(m.L * transpose(m.L),
                                     load("NormBallUncertaintySet", kind, "Mu$(dn)LLt");
                                     scale = :array, name = "$(kind) mu $(dn)").ok
                @test parity_compare([m.kappa], [radius[k]]; name = "$(kind) mu radius").ok
                if kind == "Empirical" && d
                    # Better. The diagonal of the covariance set is the variance of each entry
                    # of the sample covariance, `(Σ_ii Σ_jj + Σ_ij²) / T`, which is the
                    # diagonal of the full shape `(I + K)(Σ ⊗ Σ) / T` of both sides. The
                    # oracle drops `Σ_ij²` on every pair `i ≠ j`, so its diagonal set
                    # contradicts its own full set, by up to 30 % here.
                    S = cov(X)
                    want = [S[a, a] * S[b, b] + S[a, b]^2 for b in 1:N for a in 1:N] ./ T
                    full = load("NormBallUncertaintySet", kind, "CovFullLLt")
                    @test isapprox(diag(s.L * transpose(s.L)), want; rtol = 1e-14)
                    @test isapprox(diag(s.L * transpose(s.L)), diag(full); rtol = 1e-13)
                    @test !isapprox(s.L * transpose(s.L),
                                    load("NormBallUncertaintySet", kind, "CovDiagLLt");
                                    rtol = 1e-2)
                    # `ShapeOfDiagonal()` is the oracle's rule (#1522): it sets `Σ_ij = 0`
                    # before it builds the shape, so each entry is `(1 + δ_ij) Σ_ii Σ_jj / T`.
                    # Measured maxscaled 1.5e-16 against the oracle's diagonal set.
                    want_o = [(1 + (a == b)) * S[a, a] * S[b, b] for b in 1:N for a in 1:N] ./
                             T
                    m_o, s_o = ucs(NormalUncertaintySet(; alg = nb(true),
                                                        dc = ShapeOfDiagonal()), X)
                    @test parity_compare(s_o.L * transpose(s_o.L),
                                         load("NormBallUncertaintySet", kind, "CovDiagLLt");
                                         scale = :array,
                                         name = "$(kind) cov Diag, ShapeOfDiagonal").ok
                    @test isapprox(diag(s_o.L * transpose(s_o.L)), want_o; rtol = 1e-14)
                    @test parity_compare([s_o.kappa], [radius[2 + k]];
                                         name = "$(kind) cov radius, ShapeOfDiagonal").ok
                    # The mean axis reads no construction.
                    @test m_o.L == m.L && m_o.kappa == m.kappa
                    # The rules agree on each variance `(i, i)` and differ on each pair.
                    ii = 1:(N + 1):(N ^ 2)
                    @test isapprox(diag(s_o.L)[ii], diag(s.L)[ii]; rtol = 1e-14)
                else
                    # Measured maxscaled 4.5e-13 on the Normal full shape, whose Cholesky
                    # factor reads a matrix of rank N(N + 1) / 2 after its repair, and 3.7e-15
                    # on the bootstrap.
                    @test parity_compare(s.L * transpose(s.L),
                                         load("NormBallUncertaintySet", kind,
                                              "Cov$(dn)LLt"); scale = :array,
                                         name = "$(kind) cov $(dn)").ok
                end
                if !d
                    # Better. The error of a symmetric covariance lives in the
                    # N(N + 1) / 2 = 15 dimensions of the symmetric matrices: the bootstrap
                    # deviations have rank 15, and so has the Normal shape (I + K)(Σ ⊗ Σ) / T
                    # before its repair. The Mahalanobis distance of the set is chi-squared
                    # with 15 degrees of freedom. The oracle reads N² = 25, a radius of 6.14
                    # against 5.00, which covers 0.999 of the errors at a stated 0.95 (#1425,
                    # ADR 0188). `ambient = true` gives the oracle's radius on both routes.
                    if kind == "Bootstrap"
                        @test rank(s.L) == N * (N + 1) ÷ 2
                    end
                    @test isapprox(s.kappa, sqrt(quantile(Chisq(N * (N + 1) ÷ 2), 0.95));
                                   rtol = 1e-14)
                    @test radius[2 + k] > s.kappa
                    amb = NormBallUncertaintySetAlgorithm(;
                                                          method = ChiSqKUncertaintyAlgorithm(;
                                                                                              ambient = true),
                                                          diagonal = false)
                    _, sa = ucs(mk(kind, amb), X)
                    @test sa.L == s.L
                    @test parity_compare([sa.kappa], [radius[2 + k]];
                                         name = "$(kind) cov radius, ambient").ok
                else
                    @test parity_compare([s.kappa], [radius[2 + k]];
                                         name = "$(kind) cov radius").ok
                end
            end
        end
    end

    @testset "The construction of the diagonal covariance shape reaches every route (#1522)" begin
        od = diag(load("NormBallUncertaintySet", "Empirical", "CovDiagLLt"))
        S = cov(X)
        mine = [S[a, a] * S[b, b] + S[a, b]^2 for b in 1:N for a in 1:N] ./ T
        ue(alg, dc) = NormalUncertaintySet(; alg = alg, dc = dc, seed = 3, n_sim = 500)
        shape(s) = isa(s, EllipsoidalUncertaintySet) ? s.sigma : s.L * transpose(s.L)
        for method in (ChiSqKUncertaintyAlgorithm(), NormalKUncertaintyAlgorithm()),
            mk in
            ((m, d) -> EllipsoidalUncertaintySetAlgorithm(; method = m, diagonal = d),
             (m, d) -> NormBallUncertaintySetAlgorithm(; method = m, diagonal = d))

            _, so = ucs(ue(mk(method, true), ShapeOfDiagonal()), X)
            _, sd = ucs(ue(mk(method, true), DiagonalOfShape()), X)
            sg = sigma_ucs(ue(mk(method, true), ShapeOfDiagonal()), X)
            # The ellipsoid and the norm ball read one diagonal shape under each member.
            @test isapprox(diag(shape(so)), od; rtol = 1e-13)
            @test isapprox(diag(shape(sg)), od; rtol = 1e-13)
            @test isapprox(diag(shape(sd)), mine; rtol = 1e-13)
            # `diagonal = false` does not read the field.
            _, fo = ucs(ue(mk(method, false), ShapeOfDiagonal()), X)
            _, fd = ucs(ue(mk(method, false), DiagonalOfShape()), X)
            @test shape(fo) == shape(fd)
        end
        # The box reads no shape, so the field changes no bound.
        _, bo = ucs(NormalUncertaintySet(; dc = ShapeOfDiagonal(), seed = 1), X)
        _, bd = ucs(NormalUncertaintySet(; seed = 1), X)
        @test bo.lb == bd.lb && bo.ub == bd.ub
        # The construction keeps the number type of the data.
        sb = sigma_ucs(NormalUncertaintySet(; alg = nb(true), dc = ShapeOfDiagonal(),
                                            n_sim = 50), big.(X))
        @test eltype(sb.L) == BigFloat
        @test isapprox(Float64.(diag(sb.L * transpose(sb.L))), od; rtol = 1e-13)
    end

    @testset "The norm-ball terms of MeanRisk" begin
        pr5 = prior(EmpiricalPrior(), X)
        mF = mu_ucs(NormalUncertaintySet(; alg = nb(false)), X)
        sD = sigma_ucs(NormalUncertaintySet(; alg = nb(true)), X)
        nbm(p) = NormBallUncertaintySet(; kappa = mF.kappa, L = Matrix(mF.L), p = p,
                                        class = MuUncertaintySetClass())
        nbc(p) = NormBallUncertaintySet(; kappa = sD.kappa, L = Matrix(sD.L), p = p,
                                        class = SigmaUncertaintySetClass())
        # The worst-case variance that the lifted programme of both sides reaches at a fixed
        # portfolio, which the closed form `ucs_variance` bounds from above.
        function lifted(uc, w)
            n = length(w)
            model = JuMP.Model(Clarabel.Optimizer)
            JuMP.set_silent(model)
            for key in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
                JuMP.set_optimizer_attribute(model, key, 1e-12)
            end
            JuMP.@variable(model, Wm[1:n, 1:n], Symmetric)
            JuMP.@variable(model, E[1:n, 1:n], Symmetric)
            JuMP.@constraint(model, [Wm w; transpose(w) 1] in JuMP.PSDCone())
            JuMP.@constraint(model, E in JuMP.PSDCone())
            JuMP.@variable(model, t)
            x = transpose(uc.L) * vec(Wm + E)
            q = PO.dual_norm_order(uc.p)
            cone = if q == 2
                JuMP.SecondOrderCone()
            elseif isone(q)
                JuMP.MOI.NormOneCone(1 + length(x))
            else
                JuMP.MOI.NormInfinityCone(1 + length(x))
            end
            JuMP.@constraint(model, [t; x] in cone)
            JuMP.@objective(model, Min, dot(pr5.sigma, Wm + E) + uc.kappa * t)
            JuMP.optimize!(model)
            return JuMP.objective_value(model)
        end
        # Under `MaximumRatio` the lifted variance has degree one in the homogenised
        # variables, so both sides maximise the excess return over the variance, not over
        # its square root (#1321).
        function nbobjective(obj, um, uc, w)
            risk = isnothing(uc) ? dot(w, pr5.sigma, w) : lifted(uc, w)
            ret = dot(pr5.mu, w)
            if !isnothing(um)
                ret -= um.kappa * norm(transpose(um.L) * w, PO.dual_norm_order(um.p))
            end
            isa(obj, MinimumRisk) && return -risk
            isa(obj, MaximumUtility) && return ret - 4 * risk
            return ret / risk
        end
        NB_CASES = [(util, nbm(2), nothing), (util, nbm(1), nothing),
                    (util, nbm(3), nothing), (util, nbm(Inf), nothing),
                    (MinimumRisk(), nothing, nbc(2)), (MinimumRisk(), nothing, nbc(1)),
                    (MinimumRisk(), nothing, nbc(Inf)), (util, nothing, nbc(2)),
                    (util, nbm(2), nbc(2)), (MaximumRatio(), nothing, nbc(2))]
        W = load("NormBallMeanRisk", "Empirical", "Weights")
        slvs = Solver(; name = :clarabel_1390_sdp, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                      "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10,
                                      "max_iter" => 500))
        for (k, (obj, um, uc)) in enumerate(NB_CASES)
            ret = isnothing(um) ? ArithmeticReturn() : ArithmeticReturn(; ucs = um)
            r = isnothing(uc) ? Variance() : UncertaintySetVariance(; ucs = uc)
            # Case 9 holds a set on each axis. Both builders registered their epigraph under
            # one name, and the model refused to build (fixed in #1390).
            res = optimise(MeanRisk(; r = r, obj = obj,
                                    opt = JuMPOptimiser(; pe = pr5, slv = slvs, ret = ret)))
            @test isa(res.retcode, PO.OptimisationSuccess)
            fj = nbobjective(obj, um, uc, res.w)
            fo = nbobjective(obj, um, uc, W[:, k])
            # Measured 1.3e-8 at most on the mean terms, and 3.7e-7 on the lifted variance,
            # the accuracy of the semidefinite evaluation itself. The weights: 6.8e-5 at
            # most, at `p = Inf`, where the objective is flat.
            @test abs(fj - fo) <= 1e-6 * abs(fo)
            @test isapprox(res.w, W[:, k]; atol = 1e-4)
        end
    end

    @testset "The views of the norm ball and its converter from the ellipsoid" begin
        j = [1, 3, 4]
        wj = rand(StableRNG(5), 3)
        w = zeros(N)
        w[j] = wj
        mF, _ = ucs(NormalUncertaintySet(; alg = nb(false)), X)
        _, sD = ucs(NormalUncertaintySet(; alg = nb(true)), X)
        # A view is the projection of the set: the penalty of a portfolio held inside the
        # sub-universe equals the penalty of the same portfolio padded with zeros.
        @test norm(transpose(PO.port_opt_view(mF, j).L) * wj) == norm(transpose(mF.L) * w)
        @test norm(transpose(PO.port_opt_view(sD, j).L) * vec(wj * transpose(wj))) ==
              norm(transpose(sD.L) * vec(w * transpose(w)))
        # The converter from the ellipsoid gives the oracle's full empirical set.
        me, se = ucs(NormalUncertaintySet(;
                                          alg = EllipsoidalUncertaintySetAlgorithm(;
                                                                                   diagonal = false)),
                     X)
        cm = NormBallUncertaintySet(me)
        cs = NormBallUncertaintySet(se)
        # Measured maxscaled 2.4e-15 (mu) and 4.5e-13 (cov), the values of the full empirical
        # shapes above: the converter carries the factor of the ellipsoid. The cov tolerance
        # stays at 1e-12 for the reason of the Normal full shape, its repaired Cholesky factor.
        @test parity_compare(cm.L * transpose(cm.L),
                             load("NormBallUncertaintySet", "Empirical", "MuFullLLt");
                             rtol = 1e-13, scale = :array, name = "converted mu").ok
        @test parity_compare(cs.L * transpose(cs.L),
                             load("NormBallUncertaintySet", "Empirical", "CovFullLLt");
                             scale = :array, name = "converted cov").ok
        @test PO.ucs_variance(cs, cov(X), w) == PO.ucs_variance(se, cov(X), w)
    end

    @testset "An invalid confidence level refuses" begin
        # The oracle refuses a confidence level of `True`, 1.5, 0.0 and "0.9"; the library
        # states `q = 1 - confidence`. The oracle refuses a boolean radius too (#1516). A
        # `Bool` is an `Integer` in Julia, so the range check alone takes `kappa = true` as the
        # radius 1. Every slot of that radius refuses a `Bool` by name: the estimator, the
        # compact set, the norm ball, and the ellipsoid, whose `k` the converter carries into
        # the norm ball's `kappa`.
        for q in (true, -0.5, 1.0, 0.0, NaN)
            @test_throws DomainError OrthogonalUncertaintySet(; q = q)
        end
        @test_throws TypeError OrthogonalUncertaintySet(; q = "0.1")
        for k in (-1.0, Inf, NaN)
            @test_throws DomainError OrthogonalUncertaintySet(; kappa = k)
        end
        C = [1.0, 1.0]
        Q = reshape([1.0, 0.0], 2, 1)
        L = [1.0 0.0; 0.0 1.0]
        S = [1.0 0.2; 0.2 1.0]
        for b in (true, false)
            @test_throws "kappa is a radius" OrthogonalUncertaintySet(; kappa = b)
            @test_throws ArgumentError CompactCovarianceUncertaintySet(; kappa = b, C = C,
                                                                       Q = Q)
            @test_throws ArgumentError NormBallUncertaintySet(; kappa = b, L = L,
                                                              class = MuUncertaintySetClass())
            @test_throws ArgumentError EllipsoidalUncertaintySet(S, b,
                                                                 MuUncertaintySetClass())
        end
        # Every number in range still constructs, the degenerate radius 0 among them.
        for k in (0, 1, 0.0, 2.5, 1 // 2)
            @test OrthogonalUncertaintySet(; kappa = k).kappa === k
            @test CompactCovarianceUncertaintySet(; kappa = k, C = C, Q = Q).kappa === k
            @test NormBallUncertaintySet(; kappa = k, L = L,
                                         class = MuUncertaintySetClass()).kappa === k
        end
        @test EllipsoidalUncertaintySet(S, 1, MuUncertaintySetClass()).k === 1
    end

    @testset "The two radius rules meet their stated bounds on a real prior" begin
        # Ours only (#928, #1334, ADR 0127).
        for pr in (prs, prl)
            i = findall(PO.investable_mask(pr))
            prr = PO.port_opt_view(pr, i)
            rr = prr.rr
            d = PO.idiosyncratic_variances(rr)
            nu = rr.edof
            m = isnothing(rr.ediv) ? nu : rr.ediv
            rho = max.(m ./ quantile.(Chisq.(nu), 0.05) .- 1, 0)
            for metric in (InverseIdiosyncraticVarianceMetric(), IdentityMetric(),
                           RegressionWeightMetric())
                ue = OrthogonalUncertaintySet(; kappa = ResidualInflation(),
                                              metric = metric)
                s = sigma_ucs(ue, pr)
                sr = sigma_ucs(ue, prr)
                @test s.kappa == sr.kappa
                P = I - sr.Q * transpose(sr.Q)
                scale = sqrt.(rho .* d) ./ sr.C
                # The penalty covers the inflation of the idiosyncratic variance on every
                # direction of the orthogonal subspace, and is tight on the top one.
                rngb = StableRNG(9)
                for _ in 1:200
                    v = P * randn(rngb, length(i))
                    @test s.kappa * sum(abs2, v) >= sum(abs2, scale .* v)
                end
                v = P * svd(scale .* P).V[:, 1]
                @test isapprox(s.kappa * sum(abs2, v), sum(abs2, scale .* v); rtol = 1e-12)
                if isa(metric, InverseIdiosyncraticVarianceMetric)
                    @test s.kappa <= maximum(rho)
                end
            end
            for f in (0.1, 0.5)
                s = sigma_ucs(OrthogonalUncertaintySet(; kappa = VarianceFraction(; f = f)),
                              prr)
                w0 = fill(inv(length(i)), length(i))
                Cw = s.C .* w0
                pen = s.kappa * sum(abs2, Cw - s.Q * (transpose(s.Q) * Cw))
                @test isapprox(pen, f * dot(w0, prr.sigma, w0); rtol = 1e-12)
            end
        end
    end
end
