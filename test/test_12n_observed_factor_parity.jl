#=
Currency factors at parity with a stored oracle, and inside the meta-optimisers and the folds
(#1369).

The oracle under `test/assets/CrossSectionalFactorPriorCurrency{Plain,Family}*.csv.gz` holds the
factor returns, the factor mean, the factor covariance, `mu` and the loadings of the latest
observation. Each case was made through the harness of #1376 (#1381): `parity_write` of
`ccy_fixture().rd`, and the oracle fit of three industry factors and two style factors, each a
passthrough of its Panel Field, the Currency Factors of the `currency` field, and a plain
`EmpiricalPrior` as the nested factor prior. `Family` adds the constrained industry family, and
every other parameter is the default. The fixture draws local returns, adds the Currency Excess
Return of the currency of each asset to form the base-currency returns, and states three
currencies, so the fit must recover the local model through the derived local returns.

The `*Default*` files hold the default fit on the same fixture (#1373, #1374): the factor mean,
the factor covariance, `mu`, `sigma` and the idiosyncratic variance of the latest observation.
Its factor prior decays on the half-life of the idiosyncratic variance, and both of them measure
a second moment about zero.

The last testsets fit the fixture on the assets of one industry, which leaves the other
industry factors empty (#1372).

The `CrossSectionalFactorPriorMacro*` files hold the fit of a prior with an estimated
macro-sensitivity factor (#1365): an `EWMacroSensitivity` that reads the column "FX" of the
Exogenous Series. The `Plain*` files hold the fit with `EmpiricalPrior` and no currency. The
`CurrencyDefault*` files hold the default fit beside three Currency Factors, where the
Descriptors read the returns net of the currencies.

The `Parity_CrossSectionalFactorPrior_CurrencyForecast_*` files hold the default fit of the
currency fixture with a Return Forecast over one `EWMomentum` (#1394): the forecast, the factor
mean and `mu`. The oracle forecasts the local returns, and so does the prior.
=#
using Statistics, Clarabel
include(joinpath(@__DIR__, "parity_harness.jl"))

function ccy_asset(name)
    return Matrix(CSV.read(joinpath(@__DIR__, "assets",
                                    "CrossSectionalFactorPriorCurrency$(name).csv.gz"),
                           DataFrame))
end

@testset "Currency factors at parity, and inside the meta-optimisers" begin
    PO = PortfolioOptimisers
    fx = ccy_fixture()
    factors = ["ind1" => ccy_pass("ind1", "industry"),
               "ind2" => ccy_pass("ind2", "industry"),
               "ind3" => ccy_pass("ind3", "industry"),
               "style1" => ccy_pass("style1", "style"),
               "style2" => ccy_pass("style2", "style"), "currency" => CurrencyExposure()]

    @testset "The fit matches the stored oracle, $(nm)" for (nm, fam) in
                                                            (("Plain", nothing),
                                                             ("Family",
                                                              ["industry" => nothing]))
        pr = prior(CrossSectionalFactorPrior(; lambda = 1, factors = factors,
                                             families = fam, pe = EmpiricalPrior()), fx.rd)
        F = ccy_asset("$(nm)FactorReturns")
        @test size(pr.fpr.X) == size(F) == (119, 8)
        # Measured cell by cell, Plain and Family: maxrel 9.5e-13 and 3.2e-13 (factor
        # returns), 1.7e-14 and 1.7e-14 (factor mean), 6.9e-15 and 3.4e-15 (factor
        # covariance), 5.4e-15 and 1.9e-13 (`mu`).
        @test parity_compare(pr.fpr.X, F; name = "$(nm) factor returns").ok
        # The Currency Factors carry the observed returns, and no regression touched them.
        @test pr.fpr.X[:, 6:8] == fx.R[2:end, :]
        @test parity_compare(pr.fpr.mu, vec(ccy_asset("$(nm)FactorMu"));
                             name = "$(nm) factor mean").ok
        @test parity_compare(pr.fpr.sigma, ccy_asset("$(nm)FactorCov");
                             name = "$(nm) factor covariance").ok
        @test parity_compare(pr.mu, vec(ccy_asset("$(nm)Mu")); name = "$(nm) mu").ok
        # The loadings of the currency columns are the Currency Exposure. Measured bit-equal,
        # the other columns too.
        @test parity_compare(pr.rr.M, ccy_asset("$(nm)Loadings"); rtol = 0.0,
                             name = "$(nm) loadings").ok
    end

    @testset "The default fit matches the stored oracle, $(nm)" for (nm, fam) in
                                                                    (("Plain", nothing),
                                                                     ("Family",
                                                                      ["industry" =>
                                                                           nothing]))
        # The defaults, with the oracle's raw regime statistic (#1428).
        pr = prior(CrossSectionalFactorPrior(; lambda = 1, factors = factors,
                                             families = fam, pe = PARITY_PE,
                                             ve = PARITY_VE), fx.rd)
        @test pr.fpr.mu ≈ vec(ccy_asset("$(nm)DefaultFactorMu")) rtol = 1e-12
        @test pr.fpr.sigma ≈ ccy_asset("$(nm)DefaultFactorCov") rtol = 1e-12
        @test pr.mu ≈ vec(ccy_asset("$(nm)DefaultMu")) rtol = 1e-12
        # The idiosyncratic variance is a second moment about zero, because the model sets
        # the mean of the idiosyncratic return to zero (#1373).
        @test pr.rr.vs[end, :] ≈ vec(ccy_asset("$(nm)DefaultIdioVar")) rtol = 1e-12
        @test pr.sigma ≈ ccy_asset("$(nm)DefaultSigma") rtol = 1e-12
    end

    @testset "A Return Forecast beside Currency Factors forecasts the local returns (#1394)" begin
        rfe = FixedWeightedReturnForecast(;
                                          scores = DescriptorScores(;
                                                                    descriptors = [EWMomentum(;
                                                                                              half_life = 8.0,
                                                                                              skip = 2)],
                                                                    outlier = nothing,
                                                                    scoring = nothing),
                                          scale = 0.02)
        pr = prior(CrossSectionalFactorPrior(; factors = factors, rfe = rfe, lambda = 0.5,
                                             c = 1.0), fx.rd)
        load(o) = vec(parity_load("CrossSectionalFactorPrior", "CurrencyForecast", o))
        @test parity_compare(pr.rr.rf.mu, load("Alpha"); name = "alpha").ok
        @test parity_compare(pr.fpr.mu, load("FactorMu"); name = "factor mu").ok
        @test parity_compare(pr.mu, load("Mu"); name = "mu").ok
        # The forecast of the base-currency returns differs, so the prior does not read `X`.
        @test !(pr.rr.rf.mu ≈ return_forecast(rfe, fx.rd, pr.rr).mu)
    end

    # Every cluster and every subset estimates its own factors. The factors here are
    # continuous styles and a market intercept, and the testsets of #1372 at the end of the
    # file fit industry factors, which a cluster can leave empty.
    pe = CrossSectionalFactorPrior(; lambda = 1,
                                   factors = ["style1" => ccy_pass("style1", "style"),
                                              "style2" => ccy_pass("style2", "style"),
                                              "market" => ConstantExposure(),
                                              "currency" => CurrencyExposure()], minra = 3)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    inner = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv))

    @testset "Every cluster of a NestedClustered fits its own currency factors" begin
        res = optimise(NestedClustered(; pe = pe, opti = inner, opto = EqualWeighted()),
                       fx.rd)
        @test sum(res.w) ≈ 1
        @test all(r -> r.retcode isa OptimisationSuccess, res.resi)
        for r in res.resi
            @test r.pr.rr.nf[(end - 2):end] ==
                  ["currency=EUR", "currency=JPY", "currency=USD"]
            @test r.pr.rr.fx == fx.R[2:end, :]
        end
    end

    @testset "Every subset of a SubsetResampling fits its own currency factors" begin
        res = optimise(SubsetResampling(; subset_size = 20, n_subsets = 4,
                                        rng = StableRNG(9), pe = pe, opt = inner), fx.rd)
        @test sum(res.w) ≈ 1
        @test res.pr.rr.fx == fx.R[2:end, :]
    end

    @testset "Every fold of a walk-forward reads the Exogenous Series of its own rows" begin
        cv = cross_val_predict(inner, fx.rd, IndexWalkForward(60, 20))
        @test length(cv.pred) == 3
        for (k, p) in enumerate(cv.pred)
            tr = (20 * (k - 1) + 1):(20 * (k - 1) + 60)
            @test p.res.pr.rr.fx == fx.R[tr[2:end], :]
        end
        @test cv.mrd.E == fx.R[61:120, :]
        @test cv.mrd.ne == ["EUR", "JPY", "USD"]
    end

    @testset "A cross-sectional prior of a meta-optimiser answers on its fitted rows" begin
        # The outer problem reads one row of net returns per scenario of the prior, and a
        # cross-sectional prior answers on the rows it fitted, the last ones of the data.
        pr = prior(pe, fx.rd)
        rdo = PO.outer_prior_rows(fx.rd, pr)
        @test size(rdo.X, 1) == size(pr.X, 1) == 119
        @test rdo.X == fx.rd.X[2:end, :]
        @test rdo.E == fx.R[2:end, :]
        @test PO.outer_prior_rows(fx.rd, prior(EmpiricalPrior(), fx.rd)) === fx.rd
        big = LowOrderPrior(; X = randn(StableRNG(1), 200, 40), mu = zeros(40),
                            sigma = Matrix(1.0I, 40, 40))
        @test_throws DimensionMismatch PO.outer_prior_rows(fx.rd, big)
    end

    # A sub-universe that holds the assets of industry 2 alone leaves the factors of
    # industries 1 and 3 empty (#1372): no asset of positive weight carries a nonzero
    # exposure to them. A cluster or a subset of a meta-optimiser is such a sub-universe.
    fac3 = ["ind1" => ccy_pass("ind1", "industry"), "ind2" => ccy_pass("ind2", "industry"),
            "ind3" => ccy_pass("ind3", "industry"), "style1" => ccy_pass("style1", "style")]
    i2 = findall(==(2), fx.ind)
    rd2 = PO.port_opt_view(fx.rd, i2)

    cres = (("PseudoInverseFallback", CrossSectionalLinearRegression()),
            ("RankDeficiencyRefusal",
             CrossSectionalLinearRegression(; alg = RankDeficiencyRefusal())))
    @testset "An Empty Factor has no return, mean or variance, $(nm)" for (nm, cre) in cres
        pr = prior(CrossSectionalFactorPrior(; lambda = 1, factors = fac3, cre = cre,
                                             minra = 5), rd2)
        # The empty factors keep their place on every axis of the factor model.
        @test pr.rr.nf == ["ind1", "ind2", "ind3", "style1"]
        @test all(iszero, pr.fpr.X[:, [1, 3]])
        @test all(iszero, pr.fpr.mu[[1, 3]])
        @test all(iszero, pr.fpr.sigma[[1, 3], :]) && all(iszero, pr.fpr.sigma[:, [1, 3]])
        @test all(isfinite, pr.mu) && all(isfinite, pr.sigma) && all(isfinite, pr.chol)
        # An empty factor adds nothing to the fit, so the prior equals the one fitted on the
        # other factors alone.
        prl = prior(CrossSectionalFactorPrior(; lambda = 1, factors = fac3[[2, 4]],
                                              cre = cre, minra = 5), rd2)
        @test pr.fpr.X[:, [2, 4]] == prl.fpr.X
        @test pr.fpr.mu[[2, 4]] == prl.fpr.mu
        @test pr.fpr.sigma[[2, 4], [2, 4]] == prl.fpr.sigma
        @test pr.mu ≈ prl.mu rtol = 1e-12
        @test pr.sigma ≈ prl.sigma rtol = 1e-12
        @test transpose(pr.chol) * pr.chol ≈ pr.sigma rtol = 1e-10
    end

    @testset "An Empty Factor inside a constrained Factor Family" begin
        pr = prior(CrossSectionalFactorPrior(; lambda = 1, factors = fac3,
                                             families = ["industry" => nothing], minra = 5),
                   rd2)
        @test length(pr.rr.nf) == 4
        @test all(isfinite, pr.mu) && all(isfinite, pr.sigma)
        @test all(isfinite, pr.fpr.sigma)
    end

    @testset "An exposure at a pair of zero weight does not make a factor live" begin
        Z = zeros(3, 4, 2)
        Z[:, :, 1] .= 1.0
        Z[:, 4, 2] .= 5.0
        X = [0.01 0.02 0.03 0.04; -0.01 0.0 0.01 0.02; 0.02 0.01 0.0 -0.01]
        W = [1.0 1.0 1.0 0.0; 1.0 1.0 1.0 0.0; 1.0 1.0 1.0 0.0]
        (; csr, lv) = PO.cross_sectional_live_regression(CrossSectionalLinearRegression(),
                                                         Z, X, W)
        @test lv == [true, false]
        @test csr.f[:, 1] ≈ vec(sum(X[:, 1:3]; dims = 2)) ./ 3 rtol = 1e-12
        @test all(iszero, csr.f[:, 2])
        @test csr.n == [3, 3, 3]
        W[2, 4] = 1.0
        @test PO.cross_sectional_live_regression(CrossSectionalLinearRegression(), Z, X,
                                                 W).lv == [true, true]
        @test_throws ArgumentError PO.cross_sectional_live_regression(CrossSectionalLinearRegression(),
                                                                      zeros(3, 4, 2), X, W)
    end

    @testset "A factor prior of the live factors, placed on the whole factor axis" begin
        f = [0.01 0.0 0.02; -0.02 0.0 0.01; 0.03 0.0 -0.01; 0.0 0.0 0.02]
        fm = PO.cross_sectional_factor_moments(EmpiricalPrior(), MatrixProcessing(), f,
                                               BitVector([true, false, true]))
        ref = prior(EmpiricalPrior(), f[:, [1, 3]])
        @test fm.mu == [ref.mu[1], 0.0, ref.mu[2]]
        @test fm.sigma[[1, 3], [1, 3]] == ref.sigma
        @test all(iszero, fm.sigma[2, :]) && all(iszero, fm.sigma[:, 2])
        @test fm.X[:, [1, 3]] == ref.X && all(iszero, fm.X[:, 2])
        @test_throws DimensionMismatch PO.cross_sectional_factor_moments(EmpiricalPrior(),
                                                                         MatrixProcessing(),
                                                                         f, trues(2))
    end

    @testset "A NestedClustered over the industry factors fits every cluster (#1372)" begin
        pei = CrossSectionalFactorPrior(; lambda = 1, factors = fac3, minra = 3)
        innr = MeanRisk(; opt = JuMPOptimiser(; pe = pei, slv = slv))
        res = optimise(NestedClustered(; pe = pei, opti = innr, opto = EqualWeighted()),
                       fx.rd)
        @test sum(res.w) ≈ 1
        @test all(r -> r.retcode isa OptimisationSuccess, res.resi)
    end
end

# Three industries, two styles, and one macro series each asset loads on with its own
# sensitivity (#1365). The macro series is the column "FX" of the Exogenous Series. Under
# `ccy = true` each asset also holds one of three currencies, and `X` is in the base currency.
function mac_fixture(; T = 160, N = 40, seed = 1365, ccy = false)
    rng = StableRNG(seed)
    ind = [mod(i - 1, 3) + 1 for i in 1:N]
    I3 = zeros(T, N, 3)
    for i in 1:N
        I3[:, i, ind[i]] .= 1.0
    end
    z(A) = (A .- mean(A; dims = 2)) ./ std(A; dims = 2, corrected = false)
    s1 = z(randn(rng, T, N) .+ randn(rng, 1, N) .* 3)
    s2 = z(randn(rng, T, N) .+ randn(rng, 1, N) .* 3)
    mcap = exp.(randn(rng, 1, N) .* 0.8 .+ 0.05 .* cumsum(randn(rng, T, N); dims = 1))
    fi = 0.01 .* randn(rng, T, 3)
    fs = 0.004 .* randn(rng, T, 2)
    fx = 0.006 .* randn(rng, T)
    bx = randn(rng, N)
    X = zeros(T, N)
    for t in 2:T, i in 1:N
        X[t, i] = fi[t, ind[i]] +
                  s1[t - 1, i] * fs[t, 1] +
                  s2[t - 1, i] * fs[t, 2] +
                  bx[i] * fx[t] +
                  0.01 * randn(rng)
    end
    X[1, :] .= 0.01 .* randn(rng, N)
    codes = [mod(i, 3) + 1 for i in 1:N]
    R = 0.005 .* randn(StableRNG(seed + 1), T, 3)
    loc = copy(X)
    if ccy
        for t in 1:T, i in 1:N
            X[t, i] += R[t, codes[i]]
        end
    end
    lv = ["EUR", "JPY", "USD"]
    pf = [NumericPanelInput(; name = "market_cap", vals = mcap),
          NumericPanelInput(; name = "style1", vals = s1),
          NumericPanelInput(; name = "style2", vals = s2),
          [NumericPanelInput(; name = "ind$k", vals = I3[:, :, k]) for k in 1:3]...,
          NumericPanelInput(; name = "local", vals = loc),
          CategoricalPanelInput(; name = "currency",
                                vals = repeat(permutedims(lv[codes]), T))]
    pnl = asset_panel(pf; amsk = trues(T, N), emsk = trues(T, N))
    ne = ccy ? [lv; "FX"] : ["FX"]
    E = ccy ? hcat(R, fx) : reshape(fx, :, 1)
    return ReturnsResult(; nx = ["a$i" for i in 0:(N - 1)], X = X, ne = ne, E = E,
                         pnl = pnl)
end
function mac_factors(; ccy = false)
    c = ccy ? ["currency" => CurrencyExposure()] : Pair{String}[]
    return vcat(c,
                ["ind1" => ccy_pass("ind1", "industry"),
                 "ind2" => ccy_pass("ind2", "industry"),
                 "ind3" => ccy_pass("ind3", "industry"),
                 "style1" => ccy_pass("style1", "style"),
                 "style2" => ccy_pass("style2", "style"),
                 "macro" => CompositeExposure(;
                                              descriptors = [EWMacroSensitivity(; series = "FX",
                                                                                half_life = 10)],
                                              outlier = nothing, scoring = nothing,
                                              family = "macro")])
end
function mac_asset(name)
    return Matrix(CSV.read(joinpath(@__DIR__, "assets",
                                    "CrossSectionalFactorPriorMacro$(name).csv.gz"),
                           DataFrame))
end

@testset "A macro-sensitivity factor at parity with a stored oracle (#1365)" begin
    @testset "An estimated macro factor" begin
        rd = mac_fixture()
        pr = prior(CrossSectionalFactorPrior(; lambda = 1, factors = mac_factors(),
                                             pe = EmpiricalPrior(), ve = PARITY_VE), rd)
        @test pr.rr.nf == ["ind1", "ind2", "ind3", "style1", "style2", "macro"]
        F = mac_asset("PlainFactorReturns")
        @test size(pr.fpr.X) == size(F) == (150, 6)
        @test maximum(abs, pr.fpr.X - F) < 1e-14
        @test pr.fpr.sigma ≈ mac_asset("PlainFactorCov") rtol = 1e-12
        @test pr.rr.M ≈ mac_asset("PlainLoadings") rtol = 1e-12
        @test pr.mu ≈ vec(mac_asset("PlainMu")) rtol = 1e-12
        @test pr.sigma ≈ mac_asset("PlainSigma") rtol = 1e-12
    end
    @testset "Beside Currency Factors, the Descriptors read the net returns" begin
        # The estimated members read the returns the regression explains, so the macro
        # sensitivity measures the local move of an asset and not the currency it holds.
        rd = mac_fixture(; ccy = true)
        # The defaults, with the oracle's raw regime statistic (#1428).
        pr = prior(CrossSectionalFactorPrior(; lambda = 1,
                                             factors = mac_factors(; ccy = true),
                                             pe = PARITY_PE, ve = PARITY_VE), rd)
        @test pr.rr.nf[6:end] == ["macro", "currency=EUR", "currency=JPY", "currency=USD"]
        @test pr.fpr.mu ≈ vec(mac_asset("CurrencyDefaultFactorMu")) rtol = 1e-12
        @test pr.fpr.sigma ≈ mac_asset("CurrencyDefaultFactorCov") rtol = 1e-12
        @test pr.rr.M ≈ mac_asset("CurrencyDefaultLoadings") rtol = 1e-12
        @test pr.mu ≈ vec(mac_asset("CurrencyDefaultMu")) rtol = 1e-12
        @test pr.sigma ≈ mac_asset("CurrencyDefaultSigma") rtol = 1e-12
        # A Panel Field of the same local returns under `lx` gives the same fit, because the
        # derived net returns take the exposure of the same observation on the first rows.
        pl = prior(CrossSectionalFactorPrior(; lambda = 1,
                                             factors = mac_factors(; ccy = true),
                                             pe = PARITY_PE, ve = PARITY_VE, lx = "local"),
                   rd)
        @test pl.mu ≈ pr.mu rtol = 1e-12
        @test pl.sigma ≈ pr.sigma rtol = 1e-12
        # The macro loadings measured on the base returns differ, so the fit reads the net
        # returns and not `X`.
        de = EWMacroSensitivity(; series = "FX", half_life = 10)
        @test !(pr.rr.M[:, 6] ≈ descriptor(de, rd)[end, :])
    end
    @testset "An observed macro factor reads the returns net of the Currency Factors (#1395)" begin
        # The oracle has no observed macro factor, so the check is internal: the exposure of
        # the observed factor is the Descriptor of a direct call on the local returns, which
        # are the returns net of the Currency Factors in this fixture.
        rd = mac_fixture(; ccy = true)
        de = EWMacroSensitivity(; series = "FX", half_life = 10)
        xe = CompositeExposure(; descriptors = [de], outlier = nothing, scoring = nothing,
                               family = "fx")
        fx = "fx" => ObservedExposure(; xe = xe, series = "FX")
        @test PortfolioOptimisers.observed_reads_returns(last(fx))
        @test !PortfolioOptimisers.observed_reads_returns(CurrencyExposure())
        rdl = ReturnsResult(; nx = rd.nx,
                            X = PortfolioOptimisers.panel_field_values(rd, "local"),
                            ne = rd.ne, E = rd.E, pnl = rd.pnl)
        ml = descriptor(de, rdl)[end, :]
        fs = mac_factors(; ccy = true)[1:6]
        # The two stages keep the order of the factor list, whichever stage a member is in.
        for (f, nf) in
            ((vcat(fs, [fx]), ["currency=EUR", "currency=JPY", "currency=USD", "fx"]),
             (vcat([fx], fs), ["fx", "currency=EUR", "currency=JPY", "currency=USD"]))
            pr = prior(CrossSectionalFactorPrior(; lambda = 1, factors = f), rd)
            @test pr.rr.nf[6:end] == nf
            k = findfirst(==("fx"), pr.rr.nf)
            @test pr.rr.M[:, k] ≈ ml rtol = 1e-12
            @test !(pr.rr.M[:, k] ≈ descriptor(de, rd)[end, :])
            @test pr.rr.fx[:, k - 5] == rd.E[(end - size(pr.rr.fx, 1) + 1):end, 4]
        end
    end
end
