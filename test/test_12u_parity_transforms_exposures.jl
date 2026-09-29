#=
Parity of map #1375 for the five cross-sectional transforms, the Exposure Estimators and the
Neutralisation (#1381). Every `Parity_*` file this test reads is an output of the oracle, stored
with the harness of #1376, and each testset states how its case was made.

Every output is at parity. The measure found no defect in the library, and three differences:

  - the rank of a tie. The transforms rank a tie by its midrank on both sides. The rank statistics
    of the diagnostics rank it by its midrank by default (#1332), where the oracle ranks it by the
    order of the assets, so the oracle's rank coefficient changes when the assets are listed in
    another order. `ties = :ordinal` keeps the oracle's rule one keyword away. Better.
  - a benchmark weight that is missing on an inactive or unobserved cell reads as zero. The oracle
    refuses the composite and the derived member on a panel whose weight field keeps its default
    policy, which writes NaN outside the active mask. Its prior writes the field under its zero
    policy instead, and ours equals the oracle under that policy. Better.
  - the constant exposure is `NaN` on an inactive cell, where the oracle writes one. No fit reads
    such a cell, so no fitted quantity moves (#721). Deliberate difference.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# The small fixture of the harness, rebuilt with a `benchmark_weights` Panel Field and with blanks
# in an active stretch. The benchmark weight is the market capitalisation, zero on asset 8 over
# observations 20 to 25, so an active finite cell sits outside the estimation set. `style2` is
# blank on asset 1 at observations 30 and 31 and on asset 7 at observation 50, and
# `book_equity` is blank on asset 1 at observation 30, where two of the three Descriptors of the
# composite are then missing.
function parity_exposure_case(; fx = parity_small_panel())
    pnl = fx.rd.pnl
    amsk = pnl.amsk
    raw(name) = parity_field_rows(panel_field(pnl, name),
                                  amsk .& something(panel_field(pnl, name).omsk, amsk))
    names = ["market_cap", "adj_close", "adj_shares_outstanding", "adj_volume",
             "short_interest", "book_equity", "total_assets", "net_income_ttm", "sales_ttm",
             "style1", "style2"]
    vals = Dict(n => raw(n) for n in names)
    vals["style2"][30:31, 1] .= NaN
    vals["style2"][50, 7] = NaN
    vals["book_equity"][30, 1] = NaN
    bw = copy(vals["market_cap"])
    bw[20:25, 8] .= 0.0
    vals["benchmark_weights"] = bw
    T = size(amsk, 1)
    ccys = ["EUR", "JPY", "USD"]
    pf = [[NumericPanelInput(; name = n, vals = vals[n],
                             alg = ForwardPanelFill(; val = 0.0))
           for n in [names; "benchmark_weights"]];
          CategoricalPanelInput(; name = "industry", vals = repeat(permutedims(fx.ind), T));
          CategoricalPanelInput(; name = "currency",
                                vals = repeat(permutedims(ccys[fx.code]), T))]
    rd = ReturnsResult(; nx = fx.rd.nx, X = fx.rd.X, ne = fx.rd.ne, E = fx.rd.E,
                       pnl = asset_panel(pf; amsk = amsk, emsk = pnl.emsk))
    return (; rd, amsk, fx)
end

# The matrix of the transform cases: `style1` of the small fixture, `NaN` outside the active mask,
# with the tie of assets 9, 10 and 11 at observation 37, a second tie of assets 1 and 2 at
# observation 70, and blanks on asset 5 at observations 44 to 46. The weights are the benchmark
# weights of `parity_exposure_case`, zero wherever they are not finite. The groups are the
# industry codes, missing on asset 2 at observations 40 to 45.
function parity_transform_case(; c = parity_exposure_case())
    rd = c.rd
    X = parity_field_rows(panel_field(rd.pnl, "style1"), c.amsk)
    X[70, 2] = X[70, 1]
    X[44:46, 5] .= NaN
    w = parity_field_rows(panel_field(rd.pnl, "benchmark_weights"), c.amsk)
    w[.!isfinite.(w)] .= 0.0
    G = cross_sectional_groups(rd.pnl, "industry")
    G[40:45, 2] .= PortfolioOptimisers.CS_MISSING_GROUP
    return (; X, w, G)
end

# The five transforms, in the order of the stored columns. The scoring members take
# `min_group_size = 3`, so the groups of three or four assets stand, and the one-asset group
# "Utilities" and the cells with a missing group fall back to the whole cross-section.
function parity_transforms()
    return ["Winsoriser" => CrossSectionalWinsoriser(),
            "TanhShrinker" => CrossSectionalTanhShrinker(),
            "Standardiser" => CrossSectionalStandardiser(; min_group_size = 3),
            "GaussianRank" => CrossSectionalGaussianRank(; min_group_size = 3),
            "PercentileRank" => CrossSectionalPercentileRank(; min_group_size = 3)]
end
function parity_transform_settings(t)
    return ["Plain" => (nothing, nothing), "W" => (t.w, nothing),
            "Groups" => (nothing, t.G), "WGroups" => (t.w, t.G)]
end

# The members of the exposure cases. Each composite reads a ratio with a blank, a Panel Field
# with blanks, and a logarithm, with weights 0.5, 0.3 and 0.2 and a coverage of 0.6, so a cell
# where `style2` alone is missing stands and the cell where two Descriptors are missing is NaN.
function parity_exposures()
    ds = [BookToPrice(), Passthrough(; field = "style2"), LogMarketCap()]
    c1 = CompositeExposure(; descriptors = ds, weights = [0.5, 0.3, 0.2],
                           min_coverage = 0.6)
    c2 = CompositeExposure(; descriptors = ds, weights = [0.5, 0.3, 0.2],
                           min_coverage = 0.6, outlier = CrossSectionalTanhShrinker(),
                           scoring = CrossSectionalGaussianRank(; min_group_size = 3),
                           group = "industry")
    d1 = DerivedExposure(; source = "c1", f = x -> x .^ 2)
    d2 = DerivedExposure(; source = "c1", f = x -> x .^ 2,
                         outlier = CrossSectionalWinsoriser(),
                         scoring = CrossSectionalStandardiser(; min_group_size = 3),
                         group = "industry")
    return (; c1, c2, d1, d2)
end

function parity_passthrough(f)
    return CompositeExposure(; descriptors = [Passthrough(; field = f)], outlier = nothing,
                             scoring = nothing, family = "style")
end

# The factor prior of the fits below clips its regime multiplier to (0.7, 1.6), as the oracle
# does by default; #1383 owns that default.
function parity_neutralise_prior(; neutralise = nothing,
                                 factors = ["market" => ConstantExposure(),
                                            "size" => CompositeExposure(;
                                                                        descriptors = [LogMarketCap()],
                                                                        outlier = nothing,
                                                                        family = "style"),
                                            "style1" => parity_passthrough("style1"),
                                            "style2" => parity_passthrough("style2")])
    pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                        ce = RegimeAdjustedExpWeightedCovariance(; centred = true,
                                                                 regime_lohi_mult = (0.7,
                                                                                     1.6)))
    return CrossSectionalFactorPrior(; factors = factors, neutralise = neutralise,
                                     minra = 5, pe = pe)
end

@testset "Parity: the cross-sectional transforms, the Factor Exposures and the Neutralisation (#1381)" begin
    c = parity_exposure_case()
    t = parity_transform_case(; c = c)

    # `parity_transform_case()`, each transform of `parity_transforms()` under each setting of
    # `parity_transform_settings`, one column per pair, `vec` of the 80 × 12 output. Measured on
    # the small panel, cell by cell: maxrel 2.1e-16 (winsoriser), 9.4e-16 (tanh shrinker),
    # 7.1e-16 (standardiser), 0.0 (percentile rank). The 40 × 250 panel agrees the same way,
    # live and not stored.
    @testset "The five transforms, with and without weights and groups" begin
        O = parity_load("cross_sectional_transform", "Small", "Transforms")
        k = 0
        for (m, ct) in parity_transforms(), (s, (w, g)) in parity_transform_settings(t)
            k += 1
            Y = cross_sectional_transform(ct, t.X; w = w, groups = g)
            # The Gaussian rank of the asset in the middle of an odd cross-section is zero, and
            # the recentring leaves a residue near 1e-17 on each side, with either sign, so a
            # cell comparison of it is meaningless. Against the largest entry the rank agrees to
            # a measured 4.7e-16, and to 1.2e-14 cell by cell on the cells above 1e-12.
            sc = m == "GaussianRank" ? :array : :cell
            @test parity_compare(vec(Y), O[:, k]; scale = sc, name = "$m $s").ok
        end
        @test k == size(O, 2) == 20
    end

    # The transforms rank a tie by its midrank, on both sides. A rank statistic of the
    # diagnostics does too by default (#1332). The oracle ranks by the order of the assets
    # there, measured on its own code: the forecast [1, 1, 2, 2] against the target [1, 2, 3, 4]
    # scores 1.0, and against the same pairs listed in the order [2, 1, 4, 3] it scores 0.6.
    # The midrank gives 4 / √20 in both orders, which is Spearman's coefficient with tied
    # ranks, and `ties = :ordinal` gives the oracle's two values.
    @testset "A tie takes its midrank" begin
        PO = PortfolioOptimisers
        P = cross_sectional_transform(CrossSectionalPercentileRank(), t.X)
        @test P[37, 9] == P[37, 10] == P[37, 11]
        @test P[70, 1] == P[70, 2]
        a = [1.0, 1.0, 2.0, 2.0]
        b = [1.0, 2.0, 3.0, 4.0]
        p = [2, 1, 4, 3]
        @test PO.cs_spearman_correlation(a, b) ≈ 4 / sqrt(20) rtol = 1e-15
        @test PO.cs_spearman_correlation(a[p], b[p]) ≈ 4 / sqrt(20) rtol = 1e-15
        @test PO.cs_spearman_correlation(a, b; ties = :ordinal) == 1.0
        @test PO.cs_spearman_correlation(a[p], b[p]; ties = :ordinal) ≈ 0.6 rtol = 1e-15
        @test isnan(PO.cs_spearman_correlation(fill(1.0, 5), collect(1.0:5.0)))
    end

    # `parity_exposure_case()`, the members of `parity_exposures()`, one column per member, `vec`
    # of the 80 × 12 output: `c1`, `c2`, `d1` and `d2`, each derived member from the `c1` of our
    # side, then the four levels of `OneHotExposure(; field = "industry")`. The oracle ran on the
    # panel with its benchmark weights zero outside the active mask, which is the form its prior
    # writes. Measured maxrel 9.5e-14, 2.0e-13, 2.5e-16 and 5.6e-15, and 0.0 on the one-hot
    # block. The 40 × 250 panel agrees the same way, live and not stored.
    @testset "The composite, the derived and the one-hot members" begin
        O = parity_load("factor_exposure", "Small", "Exposures")
        xs = parity_exposures()
        L1 = factor_exposure(xs.c1, c.rd)
        @test parity_compare(vec(L1), O[:, 1]; name = "composite").ok
        @test parity_compare(vec(factor_exposure(xs.c2, c.rd)), O[:, 2];
                             name = "composite, grouped").ok
        @test parity_compare(vec(factor_exposure(xs.d1, c.rd, L1)), O[:, 3];
                             name = "derived").ok
        @test parity_compare(vec(factor_exposure(xs.d2, c.rd, L1)), O[:, 4];
                             name = "derived, grouped").ok
        B = factor_exposure(OneHotExposure(; field = "industry", family = "industry"), c.rd)
        @test size(B, 3) == 4
        for l in 1:4
            @test parity_compare(vec(B[:, :, l]), O[:, 4 + l]; name = "one-hot $l").ok
        end
        # The weight field holds NaN outside the active mask, and our read makes it zero.
        @test any(isnan, PortfolioOptimisers.panel_field_values(c.rd, "benchmark_weights"))
        # The coverage binds where two of the three Descriptors are missing.
        @test isnan(L1[30, 1]) && isfinite(L1[31, 1]) && isfinite(L1[50, 7])
    end

    # The oracle writes one on every cell, and ours writes NaN on an inactive one (#721). The
    # fits below read the constant on a panel with inactive assets, at parity.
    @testset "The constant exposure is one on every active cell" begin
        K = factor_exposure(ConstantExposure(), c.rd)
        @test all(isone, K[c.amsk])
        @test all(isnan, K[.!c.amsk])
    end

    # `parity_small_panel()`, the prior of `parity_neutralise_prior` with the Neutralisation
    # `"style1" => ["size"], "style2" => ["size", "style1"]`. The outputs are the loadings of the
    # latest observation, the factor returns, `mu` and `sigma`. The Neutralisation moves the
    # loadings of the two styles by up to 0.83. Measured maxrel 1.2e-15 (loadings), 1.6e-13
    # (factor returns) and 2.2e-15 (`mu`).
    @testset "The Neutralisation inside a fit" begin
        fx = parity_small_panel()
        pr = prior(parity_neutralise_prior(;
                                           neutralise = ["style1" => ["size"],
                                                         "style2" => ["size", "style1"]]),
                   fx.rd)
        load(o) = parity_load("CrossSectionalFactorPrior", "Neutralised", o)
        @test pr.rr.nf == ["market", "size", "style1", "style2"]
        @test parity_compare(pr.rr.M, load("Loadings"); name = "loadings").ok
        @test parity_compare(pr.rr.csr.f, load("FactorReturns"); name = "factor returns").ok
        # The relisted asset is in its warm-up at the latest observation, so the prior states
        # no moment for it (#1377).
        i4 = fx.at.relist[1]
        rest = setdiff(axes(fx.rd.X, 2), i4)
        @test parity_compare(pr.mu[rest], vec(load("Mu"))[rest]; name = "mu").ok
        # A covariance compares against its largest entry (#1376). Measured maxscaled 3.8e-13,
        # and maxrel 6.5e-11 cell by cell on its near-zero off-diagonal entries.
        @test parity_compare(pr.sigma[rest, rest], load("Sigma")[rest, rest];
                             scale = :array, name = "sigma").ok
        # The Neutralisation left the other factors alone.
        pp = prior(parity_neutralise_prior(), fx.rd)
        @test isequal(pr.rr.M[:, 1:2], pp.rr.M[:, 1:2])
        @test maximum(abs, filter(isfinite, pr.rr.M[:, 3:4] .- pp.rr.M[:, 3:4])) > 0.5
    end

    # The oracle's API has no observed factor of this form, so this check is internal;
    # `test_12x` runs the oracle's observed-factor path on a continuous loading (`Macro`). An observed
    # member states its return, the column `series` of the Exogenous Series, and the regression
    # explains the returns net of it, `X_t - Z(t - lag) r_t`. So the estimated factors equal those
    # of the fit without the observed member on those net returns, which this testset builds by
    # hand.
    @testset "An observed factor takes the return it states" begin
        fx = parity_small_panel()
        rd = fx.rd
        xe = parity_passthrough("style2")
        obs = ObservedExposure(; xe = xe, series = "MACRO", family = "macro")
        base = ["market" => ConstantExposure(), "style1" => parity_passthrough("style1")]
        po = prior(parity_neutralise_prior(; factors = [base; "macro" => obs]), rd)
        r = rd.E[:, findfirst(isequal("MACRO"), rd.ne)]
        n = size(po.fpr.X, 1)
        @test po.rr.nf == ["market", "style1", "macro"]
        @test po.fpr.X[:, 3] == r[(end - n + 1):end]
        # The exposure of the wrapped member: the passthrough of `style2`, NaN outside the active
        # mask. A composite outside a prior reads a `benchmark_weights` field that the prior writes,
        # so the Descriptor gives it here.
        Z = descriptor(Passthrough(; field = "style2"), rd)
        Xn = copy(rd.X)
        Xn[2:end, :] .-= Z[1:(end - 1), :] .* r[2:end]
        rdn = ReturnsResult(; nx = rd.nx, X = Xn, pnl = rd.pnl)
        pn = prior(parity_neutralise_prior(; factors = base), rdn)
        # Measured bit-equal, the factor returns and the residuals.
        @test parity_compare(po.rr.csr.f, pn.rr.csr.f; rtol = 0.0, name = "observed").ok
        @test parity_compare(po.rr.csr.eps, pn.rr.csr.eps; rtol = 0.0, name = "eps").ok
    end
end
