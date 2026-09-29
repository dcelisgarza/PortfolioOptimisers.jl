#=
Parity of the Cross-Sectional Factor Prior away from its defaults (#1385, map #1375): one keyword
at a time, and the combinations a caller uses. Every `Parity_*` file this test reads is an output of
the oracle, fitted on the exchange `parity_write(dir, grid_fixture(fx).rd; filled = true)` writes.

Every case sets the factor prior and the idiosyncratic variance explicitly, with the regime clip
(0.7, 1.6) and no variance floor, so a change of the defaults of the prior (#1383) moves no stored
case. The base model is a market factor and the two passthrough styles with `minra = 5`, and each
case changes what its name says.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# The fixture of the harness, with three more numeric Panel Fields: `macro_beta`, the loading of
# each asset on the macro series; `local_returns`, the local return of each asset; and `custom_mu`,
# a custom forecast. The base-currency form sets the USD Currency Excess Return to zero, as it is
# for an investor whose base currency is USD, and states the base returns again from the local ones.
function grid_fixture(fx; base_usd::Bool = false)
    rd = fx.rd
    T, N = size(rd.X)
    rng = StableRNG(1385)
    amsk = fx.amsk
    nanout(A) = ifelse.(amsk, A, NaN)
    beta = nanout(repeat(randn(rng, 1, N), T))
    cmu = nanout(0.001 .* randn(rng, T, N))
    R = copy(fx.R)
    if base_usd
        R[:, 3] .= 0.0
    end
    X = nanout(fx.loc .+ R[:, fx.code])
    h = fx.at.holiday
    X[h...] = NaN
    loc = nanout(fx.loc)
    loc[h...] = NaN
    extra = asset_panel([NumericPanelInput(; name = "macro_beta", vals = beta,
                                           alg = ForwardPanelFill(; val = 0.0)),
                         NumericPanelInput(; name = "local_returns", vals = loc,
                                           alg = ForwardPanelFill(; val = 0.0)),
                         NumericPanelInput(; name = "custom_mu", vals = cmu,
                                           alg = ForwardPanelFill(; val = 0.0))];
                        amsk = amsk, emsk = fx.emsk)
    pnl = AssetPanel(; pf = [rd.pnl.pf; extra.pf], amsk = rd.pnl.amsk, emsk = rd.pnl.emsk)
    E = copy(rd.E)
    E[:, 1:3] = R
    return ReturnsResult(; nx = rd.nx, X = base_usd ? X : rd.X, ne = rd.ne, E = E,
                         pnl = pnl)
end

function grid_pass(f; family = "style")
    return CompositeExposure(; descriptors = [Passthrough(; field = f)], outlier = nothing,
                             scoring = nothing, family = family)
end
function grid_scored(f)
    return CompositeExposure(; descriptors = [Passthrough(; field = f)], family = "style")
end

# The factor prior and the idiosyncratic variance at the oracle's defaults: the regime multiplier
# clips to (0.7, 1.6), and the regime statistic reads an asset only when its variance exceeds
# 1e-12. That threshold is `min_val`; `min_val = 0.0` differs only for a variance in (0, 1e-12],
# and the one-member industry of the fixture puts a round-off variance there (`FamOne`).
const GRID_PE = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                               ce = RegimeAdjustedExpWeightedCovariance(; centred = true,
                                                                        regime_lohi_mult = (0.7,
                                                                                            1.6)))
const GRID_VE = RegimeAdjustedExpWeightedVariance(; centred = true,
                                                  regime_lohi_mult = (0.7, 1.6),
                                                  min_val = 1e-12)

function grid_config(name::AbstractString, rd::ReturnsResult)
    base = ["market" => ConstantExposure(), "style1" => grid_pass("style1"),
            "style2" => grid_pass("style2")]
    ind = ["market" => ConstantExposure(),
           "industry" => OneHotExposure(; field = "industry", family = "industry"),
           "style1" => grid_pass("style1"), "style2" => grid_pass("style2")]
    ds = DescriptorScores(;
                          descriptors = [Passthrough(; field = "net_income_ttm"),
                                         Passthrough(; field = "sales_ttm")])
    fc = (; lambda = 0.4, c = 0.6)
    kw = (; factors = base, minra = 5, pe = GRID_PE, ve = GRID_VE)
    cfg = if name == "Base"
        kw
    elseif name == "Lag2"
        (; kw..., lag = 2)
    elseif name == "Lag3"
        (; kw..., lag = 3)
    elseif name == "ScoredBp1"
        (; kw...,
         factors = ["market" => ConstantExposure(), "style1" => grid_scored("style1"),
                    "style2" => grid_scored("style2")])
    elseif name == "ScoredBp05"
        (; grid_config("ScoredBp1", rd)..., bp = 0.5)
    elseif name == "ScoredBp0"
        (; grid_config("ScoredBp1", rd)..., bp = 0.0)
    elseif name == "MinraNone"
        (; kw..., minra = nothing)
    elseif name == "Minra10"
        (; kw..., minra = 10)
    elseif name == "Minra11"
        (; kw..., minra = 11)
    elseif name == "Target"
        (; kw..., cre = CrossSectionalTargetRegression(; tgt = LinearModel()))
    elseif name == "Blend"
        (; kw..., wa = BlendedInverseVarianceWeights(; lambda = 0.5, ratio = 3.0))
    elseif name == "Neutralised"
        (; kw...,
         factors = vcat(["market" => ConstantExposure(),
                         "size" => CompositeExposure(; descriptors = [LogMarketCap()],
                                                     outlier = nothing, family = "style")],
                        base[2:3]),
         neutralise = ["style1" => ["size"], "style2" => ["size", "style1"]])
    elseif name == "FamOne"
        (; kw..., factors = ind, families = ["industry" => nothing])
    elseif name == "FamStated"
        (; kw..., factors = ind, families = ["industry" => "industry=Banks"])
    elseif name == "FamTwo"
        (; kw...,
         factors = [ind;
                    "region" => OneHotExposure(; field = "currency", family = "region")],
         families = ["industry" => nothing, "region" => nothing])
    elseif name == "NeutFam"
        (; kw..., factors = ind, families = ["industry" => nothing],
         neutralise = ["style" => ["industry"]])
    elseif name == "FcFixed"
        (; kw..., fc..., rfe = FixedWeightedReturnForecast(; scores = ds, scale = 0.02))
    elseif name == "FcFixedSharpe"
        (; kw..., fc...,
         rfe = FixedWeightedReturnForecast(; scores = ds, scale = 0.2,
                                           unit = IdiosyncraticSharpeUnit()))
    elseif name == "FcEW"
        (; kw..., fc..., rfe = ExpWeightedReturnForecast(; scores = ds, half_life = 10.0))
    elseif name == "FcTarget"
        # The oracle calibrates its target member on the out-of-fold predictions of a 5-fold
        # split even when it states no split, and fits no intercept here; `cv` states the split.
        (; kw..., fc...,
         rfe = TargetReturnForecast(; scores = ds, half_life = 10.0, cv = KFold(; n = 5)))
    elseif name == "FcCustom"
        (; kw..., fc...,
         rfe = CustomValueReturnForecast(;
                                         mu = PortfolioOptimisers.panel_field_values(rd,
                                                                                     "custom_mu")[end,
                                                                                                  :]))
    elseif name == "FcFamily"
        (; kw..., fc..., factors = ind, families = ["industry" => nothing],
         rfe = FixedWeightedReturnForecast(; scores = ds, scale = 0.02))
    elseif name == "Currency"
        (; kw..., factors = [base; "ccy" => CurrencyExposure()])
    elseif name == "CurrencyLx"
        # The local returns stated as a Panel Field, rather than derived from the base returns.
        (; kw..., factors = [base; "ccy" => CurrencyExposure()], lx = "local_returns")
    elseif name == "CurrencyBase"
        # The oracle has no base currency, and it refuses the fit when the excess return of
        # the base currency, which is zero, is a factor. A factor whose return is zero adds zero
        # to `mu` and to `sigma`, so the model without it is the same model: `base` states it,
        # and the stored case is the oracle's own fit with the base column dropped. Better.
        (; kw..., factors = [base; "ccy" => CurrencyExposure(; base = "USD")])
    elseif name == "Macro"
        # The oracle's observed-factor path takes a one-hot member alone. The stored case runs
        # that path on the continuous loading `macro_beta` and the series `MACRO`, with the
        # returns net of the macro factor, which is the model `ObservedExposure` states.
        (; kw...,
         factors = [base;
                    "macro" =>
                        ObservedExposure(; xe = grid_pass("macro_beta"; family = "macro"),
                                         series = "MACRO", family = "macro")])
    else
        throw(ArgumentError(name))
    end
    return cfg
end
function grid_prior(name::AbstractString, rd::ReturnsResult)
    return CrossSectionalFactorPrior(; grid_config(name, rd)...)
end

# The cases that change the structure of the fit store every output. A Return Forecast changes
# the mean alone, so its cases store `mu`, the factor mean and the forecast, and the test checks
# that every other output equals the fit without the forecast, bit for bit.
const GRID_STRUCT = ["Base", "Lag2", "Lag3", "ScoredBp1", "ScoredBp05", "ScoredBp0",
                     "Target", "Blend", "Neutralised", "FamOne", "FamStated", "FamTwo",
                     "NeutFam", "Currency", "CurrencyLx", "Macro", "CurrencyBase"]
const GRID_FORECAST = ["FcFixed" => "Base", "FcFixedSharpe" => "Base", "FcEW" => "Base",
                       "FcTarget" => "Base", "FcCustom" => "Base", "FcFamily" => "FamOne"]
const GRID_FAMILY = ["FamOne", "FamStated", "FamTwo", "NeutFam", "FcFamily"]

# Compare one fit with the stored outputs of its case. The large panel stores its factor returns
# and `sigma` for nine cases alone, to keep the stored files small; the other large cases were
# measured to the same tolerances.
function grid_check(pr, fix::AbstractString, c::AbstractString, rest)
    load(o) = parity_load("CrossSectionalFactorPrior", "Grid$(fix)$(c)", o)
    has(o) = isfile(parity_asset("CrossSectionalFactorPrior", "Grid$(fix)$(c)", o))
    nm = "$(fix) $(c)"
    # Measured maxrel 5.5e-13 at most, on the large constrained families.
    @test parity_compare(pr.mu[rest], vec(load("Mu"))[rest]; name = "$(nm) mu").ok
    @test parity_compare(pr.fpr.mu, vec(load("FactorMu")); name = "$(nm) factor mu").ok
    if has("Alpha")
        # Measured maxrel 2.3e-13 at most.
        @test parity_compare(pr.rr.rf.mu, vec(load("Alpha")); name = "$(nm) forecast").ok
    end
    if !(has("FactorCov"))
        return nothing
    end
    fam = c in GRID_FAMILY
    # A family case has near-zero off-diagonal factor covariances, which compare against the
    # largest entry, as `sigma` does. Measured maxrel 7.0e-14 cell by cell elsewhere, and
    # maxscaled 6.2e-13 on the families, where the cell-by-cell maxrel is 3.4e-12.
    @test parity_compare(pr.fpr.sigma, load("FactorCov"); scale = fam ? :array : :cell,
                         name = "$(nm) factor cov").ok
    # Measured maxrel 4.8e-14 at most, on the neutralised loadings.
    @test parity_compare(pr.rr.M, load("Loadings"); name = "$(nm) loadings").ok
    # The one member of "Utilities" has a residual of round-off size under a constrained
    # industry family, so its variance is a round-off zero (1e-34) on both sides, and a family
    # case compares against the largest entry. Measured maxrel 3.1e-15 cell by cell elsewhere,
    # and maxscaled 4.4e-15 on the families.
    if has("IdioVariances")
        @test parity_compare(pr.rr.vs, load("IdioVariances"); scale = fam ? :array : :cell,
                             name = "$(nm) vs").ok
    else
        @test parity_compare(pr.rr.vs[end, :], vec(load("IdioVarLatest"));
                             scale = fam ? :array : :cell, name = "$(nm) vs").ok
    end
    # Measured maxrel 2.2e-16.
    if has("BenchmarkWeights")
        @test parity_compare(pr.rr.bw, load("BenchmarkWeights"); name = "$(nm) bw").ok
    elseif has("BenchmarkWeightsLatest")
        @test parity_compare(pr.rr.bw[end, :], vec(load("BenchmarkWeightsLatest"));
                             name = "$(nm) bw").ok
    end
    if !(has("Sigma"))
        return nothing
    end
    # A covariance compares against its largest entry, because its small off-diagonal
    # entries come from a cancellation (#1376). Measured maxscaled 6.9e-13.
    @test parity_compare(pr.sigma[rest, rest], load("Sigma")[rest, rest]; scale = :array,
                         name = "$(nm) sigma").ok
    # A factor return is a difference of weighted sums, and one near zero carries the
    # cancellation, so the factor returns compare against their largest entry too. Under a
    # constrained industry family, "Utilities" is empty after observation 60 of the small
    # panel, so its factor return is not identified there, and both sides give a round-off
    # zero. Measured maxscaled 7.7e-15, and maxrel 3.7e-12 cell by cell outside that column.
    @test parity_compare(pr.fpr.X, load("FactorReturns"); scale = :array,
                         name = "$(nm) factor returns").ok
    return nothing
end

@testset "Parity: the prior away from its defaults (#1385)" begin
    @testset "$(fix)" for (fix, fx) in (("Small", parity_small_panel()),
                                        ("Large", parity_large_panel()))
        rd = grid_fixture(fx)
        rdu = grid_fixture(fx; base_usd = true)
        N = size(rd.X, 2)
        # Asset 4 of the small panel is in its warm-up at the latest observation. `test_12p`
        # states why its `mu` and its covariances are a deliberate difference (#1384).
        rest = fix == "Small" ? setdiff(1:N, fx.at.relist[1]) : (1:N)
        fits = Dict{String, Any}()
        @testset "$(c)" for c in GRID_STRUCT
            rdc = c == "CurrencyBase" ? rdu : rd
            pe = grid_prior(c, rdc)
            pr = prior(pe, rdc)
            fits[c] = pr
            grid_check(pr, fix, c, rest)
            # The factor axis the prior declares before the fit is the axis it fits.
            ax = cross_sectional_factor_axis(pe, rdc)
            @test ax.nf == pr.rr.nf
            @test ax.fam == pr.rr.fam
            sets = cross_sectional_factor_sets(pe, rdc)
            @test sets.dict[PortfolioOptimisers.factor_axis_key(sets, pr.rr)] == pr.rr.nf
            @test all(f -> sort(sets.dict[f]) == sort(pr.rr.nf[pr.rr.fam .== f]),
                      unique(pr.rr.fam))
        end
        @testset "$(c)" for (c, b) in GRID_FORECAST
            pr = prior(grid_prior(c, rd), rd)
            grid_check(pr, fix, c, rest)
            # The forecast enters the mean alone.
            p0 = fits[b]
            @test isequal(pr.rr.M, p0.rr.M) && isequal(pr.fpr.sigma, p0.fpr.sigma)
            @test isequal(pr.fpr.X, p0.fpr.X) && isequal(pr.rr.vs, p0.rr.vs)
            @test isequal(pr.sigma, p0.sigma)
        end
        @testset "The smallest eligible asset count" begin
            # The fewest eligible assets at an observation of the small panel is 10, so
            # `minra = 10` does not bind, and neither does the default `max(2K, 30)` on the
            # large panel. The fit equals `Base` there.
            pr = prior(grid_prior("Minra10", rd), rd)
            @test isequal(pr.mu, fits["Base"].mu) && isequal(pr.sigma, fits["Base"].sigma)
            if fix == "Large"
                pr = prior(grid_prior("MinraNone", rd), rd)
                @test isequal(pr.mu, fits["Base"].mu) &&
                      isequal(pr.fpr.X, fits["Base"].fpr.X)
            else
                # Where it binds, both sides refuse the fit and count the same observations:
                # 79 under the default 30, and 19 under 11, the fewest being 10 at the first.
                for (c, n) in (("MinraNone", 79), ("Minra11", 19))
                    e = try
                        prior(grid_prior(c, rd), rd)
                        nothing
                    catch err
                        err
                    end
                    @test e isa ArgumentError
                    @test occursin("$(n) observation(s) carry fewer", e.msg)
                    @test occursin("the fewest being 10 at observation 1", e.msg)
                end
            end
        end
        @testset "The name of the benchmark-weight field" begin
            # `bw` names the Panel Field the prior writes, so another name changes no value.
            # The oracle has no such keyword.
            sc(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                      family = "style", bw = "bench_w")
            cfg = grid_config("ScoredBp05", rd)
            pr = prior(CrossSectionalFactorPrior(; cfg..., bw = "bench_w",
                                                 factors = ["market" => ConstantExposure(),
                                                            "style1" => sc("style1"),
                                                            "style2" => sc("style2")]), rd)
            p0 = fits["ScoredBp05"]
            @test isequal(pr.mu, p0.mu) && isequal(pr.sigma, p0.sigma)
            @test isequal(pr.rr.bw, p0.rr.bw) && isequal(pr.rr.M, p0.rr.M)
            # A member that reads another field is refused.
            @test_throws ArgumentError prior(CrossSectionalFactorPrior(; cfg...,
                                                                       bw = "bench_w"), rd)
        end
        @testset "The Factor Family Basis of $(c)" for c in
                                                       ("FamOne", "FamStated", "FamTwo")
            PO = PortfolioOptimisers
            pr = fits[c]
            fcb = pr.rr.fcb
            load(o) = parity_load("CrossSectionalFactorPrior", "Grid$(fix)$(c)", o)
            has(o) = isfile(parity_asset("CrossSectionalFactorPrior", "Grid$(fix)$(c)", o))
            # The factor return of row `t` was fitted on the exposures of row `t - 1`, so it is
            # written in the basis of that row, and the round trip through the reduced axis
            # is exact on those rows. Read against the basis of its own row, it is not: the
            # dropped member moves by 0.22 of the largest factor return on the small panel.
            T = size(pr.fpr.X, 1)
            fs = PO.factor_basis_slice(fcb, 1:(T - 1))
            g = PO.reduce_factor_returns(fs, pr.fpr.X[2:T, :])
            @test PO.expand_factor_returns(fs, g) == pr.fpr.X[2:T, :]
            has("BasisRatios") || continue
            # The same dropped member, automatic or stated, and the same ratios. Measured
            # maxrel 5.5e-16.
            @test PO.dropped_factor_indices(fcb) == Int.(vec(load("BasisDropped"))) .+ 1
            @test parity_compare(fcb.ratios, load("BasisRatios"); name = "$(c) ratios").ok
            has("BasisReducedLoadings") || continue
            # Every transform on the fitted outputs. The expansions read the ratios of the
            # second and the third observation, so a stale row would show. Measured maxrel
            # 6.2e-13 at most, on the covariance of two families.
            mu = PO.reduce_factor_mu(fcb, pr.fpr.mu)
            S = PO.reduce_factor_covariance(fcb, pr.fpr.sigma)
            @test parity_compare(PO.reduce_loadings(fcb, pr.rr.M),
                                 load("BasisReducedLoadings")).ok
            @test parity_compare(mu, vec(load("BasisReducedMu"))).ok
            @test parity_compare(PO.expand_factor_mu(fcb, mu, 2),
                                 vec(load("BasisExpandedMu1"))).ok
            @test parity_compare(S, load("BasisReducedCov"); scale = :array).ok
            @test parity_compare(PO.expand_factor_covariance(fcb, S, 3),
                                 load("BasisExpandedCov2"); scale = :array).ok
            @test parity_compare(PO.project_factor_coordinates(fcb,
                                                               vec(load("BasisProjectInput"))),
                                 vec(load("BasisProject"))).ok
        end
    end
end
