#=
Observed factors on `CrossSectionalFactorPrior` (#1368, ADR 0184). An observed member declares
factors whose return the caller observes, read from the Exogenous Series by name. The regression
runs on the returns net of them, and they then follow the estimated factors on every axis of the
factor model. `CurrencyExposure` gives one Currency Factor per currency level, and
`ObservedExposure` gives one factor from a member it wraps.

The fixture holds two currencies. The base-currency return of an asset is its local return plus
the Currency Excess Return of its currency, so the derived local return of the fit is the local
return the fixture drew, and the estimated factor is recovered from it.
=#
using Dates, Statistics

function obs_fixture(; T = 80, N = 12, seed = 1_368, amsk = trues(T, N), R = nothing,
                     Eo = nothing, warm = 0)
    # The warm-up is a run of leading observations with no active asset.
    amsk = copy(amsk)
    amsk[1:warm, :] .= false
    rng = StableRNG(seed)
    beta = collect(range(0.6, 1.4; length = N))
    fl = 0.01 .* randn(rng, T)
    loc = fl .* transpose(beta) .+ 0.002 .* randn(rng, T, N)
    Rc = isnothing(R) ? hcat(0.003 .* randn(rng, T), 0.002 .* randn(rng, T)) : R
    lv = ["EUR", "USD"]
    code = [i <= N ÷ 2 ? 2 : 1 for i in 1:N]
    X = loc .+ Rc[:, code]
    B = repeat(transpose(beta), T)
    mcap = ones(T, N)
    X[.!amsk] .= NaN
    pnl = asset_panel([NumericPanelInput(; name = "beta", vals = B),
                       NumericPanelInput(; name = "market_cap", vals = mcap),
                       CategoricalPanelInput(; name = "currency",
                                             vals = repeat(permutedims(lv[code]), T))];
                      amsk = amsk, emsk = copy(amsk))
    rd = ReturnsResult(; nx = ["a$i" for i in 1:N], X = X, ne = lv,
                       E = isnothing(Eo) ? Rc : Eo, pnl = pnl)
    return (; rd, loc, fl, R = Rc, code, beta)
end
function obs_pass(field, family)
    return CompositeExposure(; descriptors = [Passthrough(; field = field)],
                             outlier = nothing, scoring = nothing, family = family)
end
function obs_prior(; kwargs...)
    return CrossSectionalFactorPrior(; lambda = 1,
                                     factors = ["beta" => obs_pass("beta", "market"),
                                                "currency" => CurrencyExposure()], bp = 0,
                                     wa = MarketCapWeights(; p = 0), minra = 3, kwargs...)
end

@testset "Observed factors" begin
    PO = PortfolioOptimisers

    @testset "Currency factors are stored as direct factors" begin
        fx = obs_fixture()
        pr = prior(obs_prior(), fx.rd)
        rr = pr.rr
        @test rr.nf == ["beta", "currency=EUR", "currency=USD"]
        @test rr.fam == ["market", "currency", "currency"]
        # The observed returns are the Currency Excess Returns of the fitted rows, unchanged.
        @test rr.fx == fx.R[2:end, :]
        @test pr.fpr.X[:, 2:3] == fx.R[2:end, :]
        @test cross_sectional_factor_returns(rr) == pr.fpr.X
        # The regression ran on the local returns, so the observed factors did not enter it.
        @test size(rr.csr.f, 2) == 1
        @test cor(rr.csr.f[:, 1], fx.fl[2:end]) > 0.99
        # The combined model reproduces the base-currency returns exactly.
        X = fx.rd.X
        for t in 2:size(rr.Ms, 1)
            @test X[t + 1, :] ≈ rr.Ms[t - 1, :, :] * pr.fpr.X[t, :] + rr.csr.eps[t, :]
        end
        @test pr.o_X == X[2:end, :]
        # The axis a caller declares before the fit is the axis of the block.
        @test cross_sectional_factor_axis(obs_prior(), fx.rd).nf == rr.nf
        us = cross_sectional_factor_sets(obs_prior(), fx.rd)
        @test us.dict["currency"] == ["currency=EUR", "currency=USD"]
    end

    @testset "The moments are the factor model the block states" begin
        pr = prior(obs_prior(), obs_fixture().rd)
        rr = pr.rr
        @test pr.mu ≈ rr.M * pr.fpr.mu + rr.b
        @test pr.sigma ≈ rr.M * pr.fpr.sigma * transpose(rr.M) + Diagonal(rr.esigma)
        # The shrinkage reaches the estimated factor alone.
        p0 = prior(obs_prior(; lambda = 0.0), obs_fixture().rd)
        @test p0.mu ≈ rr.M[:, 2:3] * pr.fpr.mu[2:3]
    end

    @testset "A constrained family keeps its zero-sum condition, and the basis passes the currencies through" begin
        rng = StableRNG(3)
        T, N = 100, 12
        ind = [mod(i, 2) + 1 for i in 1:N]
        I1 = repeat(transpose(Float64.(ind .== 1)), T)
        I2 = repeat(transpose(Float64.(ind .== 2)), T)
        mcap = repeat(transpose([ind[i] == 1 ? 2.0 : 1.0 for i in 1:N]), T)
        fi = 0.006 .* randn(rng, T, 2)
        loc = 0.01 .* randn(rng, T) .+ I1 .* fi[:, 1] .+ I2 .* fi[:, 2] .+
              0.001 .* randn(rng, T, N)
        R = hcat(0.003 .* randn(rng, T), 0.002 .* randn(rng, T))
        code = [mod(i - 1, 2) + 1 for i in 1:N]
        X = loc .+ R[:, code]
        pnl = asset_panel([NumericPanelInput(; name = "ind_1", vals = I1),
                           NumericPanelInput(; name = "ind_2", vals = I2),
                           NumericPanelInput(; name = "market_cap", vals = mcap),
                           CategoricalPanelInput(; name = "currency",
                                                 vals = repeat(permutedims(["EUR", "USD"][code]),
                                                               T))]; amsk = trues(T, N),
                          emsk = trues(T, N))
        rd = ReturnsResult(; nx = ["a$i" for i in 1:N], X = X, ne = ["EUR", "USD"], E = R,
                           pnl = pnl)
        pe = CrossSectionalFactorPrior(; lambda = 1,
                                       factors = ["market" => ConstantExposure(),
                                                  "ind_1" => obs_pass("ind_1", "industry"),
                                                  "ind_2" => obs_pass("ind_2", "industry"),
                                                  "currency" => CurrencyExposure()],
                                       families = ["industry" => nothing], minra = 4,
                                       bp = 1, wa = MarketCapWeights(; p = 1))
        pr = prior(pe, rd)
        rr = pr.rr
        @test rr.nf == ["market", "ind_1", "ind_2", "currency=EUR", "currency=USD"]
        @test rr.fcb.K == length(rr.nf)
        @test PO.reduce_factor_names(rr.fcb, rr.nf) ==
              ["market", "ind_1", "currency=EUR", "currency=USD"] ||
              PO.reduce_factor_names(rr.fcb, rr.nf) ==
              ["market", "ind_2", "currency=EUR", "currency=USD"]
        @test pr.fpr.X[:, 4:5] ≈ R[2:end, :]
        # The benchmark weights hold 2/3 of the weight in industry 1, so its return and the
        # return of industry 2 sum to zero under 2/3 and 1/3.
        @test pr.fpr.X[:, 2:3] * [2 / 3, 1 / 3] ≈ zeros(T - 1) atol = 1e-12
        @test pr.sigma ≈
              getfield(rr, :L) *
              PO.reduce_factor_covariance(rr.fcb, pr.fpr.sigma) *
              transpose(getfield(rr, :L)) + Diagonal(rr.esigma) rtol = 1e-10
    end

    @testset "The regression diagnostics and the attribution know the observed factors" begin
        fx = obs_fixture()
        pr = prior(obs_prior(), fx.rd)
        @test PO.cs_diagnostic_factor_names(pr.rr) == ["beta"]
        @test size(cs_regression_t_stats(pr.rr).X, 2) == 1
        @test all(isfinite, cs_regression_r2(pr.rr))
        w = fill(1 / 12, 12)
        # A standard error reads the idiosyncratic variance of every pair of the regression, so
        # the attribution fits a variance estimate with no warm-up (#1388).
        pa = prior(obs_prior(;
                             ve = RegimeAdjustedExpWeightedVariance(;
                                                                    centring = PreCentred(),
                                                                    min_obs = 1)), fx.rd)
        fa = factor_attribution(w, pa, fx.rd.X; se = true)
        @test isnan(fa.fbd.mu_se[2]) && isnan(fa.fbd.mu_se[3])
        @test isfinite(fa.fbd.mu_se[1])
    end

    @testset "The currency exposures are NaN on an inactive asset" begin
        amsk = trues(80, 12)
        amsk[1:10, 1] .= false
        fx = obs_fixture(; amsk = amsk)
        rr = prior(obs_prior(), fx.rd).rr
        @test all(isnan, rr.Ms[1:9, 1, 2:3])
        @test all(isfinite, rr.Ms[10:end, 1, 2:3])
        @test all(isfinite, rr.Ms[:, 2:end, 2:3])
    end

    @testset "A gap in the rows the warm-up and the lag consume is accepted" begin
        for warm in (0, 5)
            clean = obs_fixture(; warm = warm)
            Eg = copy(clean.R)
            Eg[1:(warm + 1), :] .= NaN
            gap = obs_fixture(; warm = warm, Eo = Eg)
            pc = prior(obs_prior(), clean.rd)
            pg = prior(obs_prior(), gap.rd)
            @test pg.rr.fx == clean.R[(warm + 2):end, :]
            @test pg.mu ≈ pc.mu
            @test pg.sigma ≈ pc.sigma
        end
    end

    @testset "The refusals" begin
        fx = obs_fixture()
        rd0 = ReturnsResult(; nx = fx.rd.nx, X = fx.rd.X, pnl = fx.rd.pnl)
        @test_throws PO.IsNothingError prior(obs_prior(), rd0)
        rd1 = ReturnsResult(; nx = fx.rd.nx, X = fx.rd.X, ne = ["EUR"], E = fx.R[:, 1:1],
                            pnl = fx.rd.pnl)
        @test_throws ArgumentError prior(obs_prior(), rd1)
        # A column no currency names is ignored.
        rd2 = ReturnsResult(; nx = fx.rd.nx, X = fx.rd.X, ne = ["JPY", "USD", "EUR"],
                            E = hcat(fx.R[:, 2], fx.R[:, 2], fx.R[:, 1]), pnl = fx.rd.pnl)
        @test prior(obs_prior(), rd2).rr.fx == fx.R[2:end, :]
        # A non-finite return on a fitted row names the series and the observation.
        for v in (NaN, Inf)
            Eb = copy(fx.R)
            Eb[6, 1] = v
            err = try
                prior(obs_prior(), obs_fixture(; Eo = Eb).rd)
                nothing
            catch e
                e
            end
            @test err isa PO.IsNonFiniteError
            @test occursin("[\"EUR\"]", err.msg) && occursin("observation 6", err.msg)
        end
        # A family holds estimated factors or observed ones.
        @test_throws ArgumentError CrossSectionalFactorPrior(; lambda = 1,
                                                             factors = ["beta" =>
                                                                            obs_pass("beta",
                                                                                     "currency"),
                                                                        "currency" =>
                                                                            CurrencyExposure()])
        @test_throws ArgumentError CrossSectionalFactorPrior(; lambda = 1,
                                                             factors = ["beta" =>
                                                                            obs_pass("beta",
                                                                                     "market"),
                                                                        "currency" =>
                                                                            CurrencyExposure()],
                                                             families = ["currency" =>
                                                                             nothing])
        # A label no estimated member claims is free, "currency" included.
        @test CrossSectionalFactorPrior(; lambda = 1,
                                        factors = ["beta" => obs_pass("beta", "currency")]) isa
              CrossSectionalFactorPrior
        @test_throws ArgumentError CrossSectionalFactorPrior(; lambda = 1,
                                                             factors = ["beta" =>
                                                                            obs_pass("beta",
                                                                                     "market")],
                                                             lx = "local")
        @test_throws ArgumentError ObservedExposure(; xe = CurrencyExposure(), series = "x")
        # An empty name is refused with a message about the Exogenous Series (#1365).
        err = try
            ObservedExposure(; xe = ConstantExposure(), series = "")
        catch e
            e
        end
        @test err isa PO.IsEmptyError
        @test occursin("Exogenous Series", err.msg)
        @test_throws ArgumentError ObservedExposure(;
                                                    xe = DerivedExposure(; source = "beta",
                                                                         f = abs,
                                                                         family = "x"),
                                                    series = "x")
        # An observed member that wraps a member of many factors is refused at the fit.
        pe = CrossSectionalFactorPrior(; lambda = 1,
                                       factors = ["beta" => obs_pass("beta", "market"),
                                                  "ccy" => ObservedExposure(;
                                                                            xe = OneHotExposure(;
                                                                                                field = "currency",
                                                                                                family = "c"),
                                                                            series = "EUR")],
                                       minra = 3)
        @test_throws DimensionMismatch prior(pe, fx.rd)
    end

    @testset "The base currency takes no factor" begin
        # The base currency earns no Currency Excess Return, so its return is zero.
        fz = obs_fixture(; R = hcat(0.003 .* randn(StableRNG(11), 80), zeros(80)))
        rd = ReturnsResult(; nx = fz.rd.nx, X = fz.rd.X, ne = ["EUR"], E = fz.R[:, 1:1],
                           pnl = fz.rd.pnl)
        fb = ["beta" => obs_pass("beta", "market"),
              "currency" => CurrencyExposure(; base = "USD")]
        pr = prior(obs_prior(; factors = fb), rd)
        @test pr.rr.nf == ["beta", "currency=EUR"]
        usd = findall(==(2), fz.code)
        @test all(iszero, pr.rr.Ms[:, usd, 2])
        @test all(isfinite, pr.sigma)
        @test cross_sectional_factor_axis(obs_prior(; factors = fb), rd).nf == pr.rr.nf
        # A factor of the base currency has no variance. The factor covariance keeps its zero
        # row and column (#1429), so the factor adds no risk, and the asset covariance is the
        # one of the fit that `base` drops the factor from.
        rdz = ReturnsResult(; nx = fz.rd.nx, X = fz.rd.X, ne = ["EUR", "USD"], E = fz.R,
                            pnl = fz.rd.pnl)
        prz = prior(obs_prior(), rdz)
        @test prz.rr.nf == ["beta", "currency=EUR", "currency=USD"]
        @test all(iszero, prz.fpr.sigma[3, :])
        @test all(iszero, prz.fpr.sigma[:, 3])
        @test all(isfinite, prz.sigma)
        @test isapprox(prz.sigma, pr.sigma; rtol = 1e-12)
        @test_throws ArgumentError prior(obs_prior(;
                                                   factors = ["beta" => obs_pass("beta",
                                                                                 "market"),
                                                              "currency" =>
                                                                  CurrencyExposure(;
                                                                                   base = "GBP")]),
                                         rd)
    end

    @testset "A named Panel Field of local returns replaces the derived ones" begin
        fx = obs_fixture()
        pf = Any[f for f in fx.rd.pnl.pf]
        push!(pf, NumericPanelField(; name = "local", vals = fx.loc))
        pnl = AssetPanel(; pf = identity.(pf), amsk = fx.rd.pnl.amsk, emsk = fx.rd.pnl.emsk)
        rd = ReturnsResult(; nx = fx.rd.nx, X = fx.rd.X, ne = fx.rd.ne, E = fx.R, pnl = pnl)
        pn = prior(obs_prior(; lx = "local"), rd)
        pd = prior(obs_prior(), rd)
        @test pn.rr.csr.f ≈ pd.rr.csr.f
        @test pn.mu ≈ pd.mu
        @test pn.sigma ≈ pd.sigma
    end

    @testset "An observed market factor reads its return from E" begin
        fx = obs_fixture()
        m = 0.004 .* randn(StableRNG(7), 80)
        X = fx.rd.X .+ m
        rd = ReturnsResult(; nx = fx.rd.nx, X = X, ne = ["EUR", "USD", "SPX"],
                           E = hcat(fx.R, m), pnl = fx.rd.pnl)
        pe = obs_prior(;
                       factors = ["beta" => obs_pass("beta", "style"),
                                  "currency" => CurrencyExposure(),
                                  "mkt" => ObservedExposure(; xe = ConstantExposure(),
                                                            series = "SPX")])
        pr = prior(pe, rd)
        @test pr.rr.nf == ["beta", "currency=EUR", "currency=USD", "mkt"]
        @test pr.rr.fam == ["style", "currency", "currency", "market"]
        @test pr.fpr.X[:, 4] == m[2:end]
        @test cor(pr.rr.csr.f[:, 1], fx.fl[2:end]) > 0.99
    end

    @testset "A wrapping prior hands the Exogenous Series on" begin
        fx = obs_fixture()
        hp = prior(HighOrderPriorEstimator(; pe = obs_prior()), fx.rd)
        @test hp.rr.fx == fx.R[2:end, :]
        # A prior that reads no Exogenous Series ignores it.
        @test prior(EmpiricalPrior(), fx.rd).mu ≈ vec(mean(fx.rd.X; dims = 1))
    end
end
