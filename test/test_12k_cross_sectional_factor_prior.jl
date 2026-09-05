#=
The Cross-Sectional Factor Prior (issue #725, map #643).

WHAT THIS FILE GATES. The estimator builds a point-in-time factor model from an Asset Panel and
lifts it onto the assets. Four identities are exact and are asserted at machine precision, because
each one is a construction rather than an estimate:

 1. `Ms[t - lag] f_t + eps_t == X_t` on every eligible pair. The realised factor returns expand
    with the LAGGED family ratios exactly so that this holds.
 2. `L F L' + D == sigma` on the investable block, and `chol' chol == sigma` beside it.
 3. `mu == M fpr.mu + b` on the investable assets.
 4. A constrained Factor Family drives the benchmark-weighted sum of its factor returns to zero,
    again under the LAGGED ratios.

The recovery testset is statistical, not exact: the synthetic panel's Panel Fields are noisy
functions of the true loadings, so a fitted exposure correlates with the truth rather than equalling
it. The one exception is the industry block, which a one-hot exposure recovers exactly.

TWO DEPARTURES FROM THE REFERENCE IMPLEMENTATION, both recorded in the resolution comment of #725.

  - The benchmark mask and the eligibility mask both drop a pair whose market capitalisation is not
    finite. The reference implementation lets such a pair carry a `NaN` weight. The library refuses
    a non-finite capitalisation on an eligible pair, and `exposure_benchmark_weights` already zeroes
    a non-finite weight, so dropping the pair is what the library's own convention asks for.
  - The idiosyncratic correlation overlay writes a zero at a standardised residual that is still not
    finite, which happens only where the asset is inactive. The reference implementation's default
    covariance estimator skips such a pair; the library has no exponentially weighted covariance
    with a verb (issue #637), so the default here is the library's own and it admits no `NaN`.
=#
using Statistics, Distributions, Dates, Random
include(joinpath(@__DIR__, "test06c_setup.jl"))

# The four factors every fast testset fits: a market intercept, the one-hot industry block and two
# style composites. Seven factors over sixty assets keeps `minra` at thirty and the fit quick.
function csfp_factors()
    return ["market" => ConstantExposure(),
            "industry" => OneHotExposure(; field = "industry", family = "industry"),
            "size" => CompositeExposure(; descriptors = [LogMarketCap()], family = "style"),
            "value" => CompositeExposure(; descriptors = [BookToPrice()], family = "style")]
end

function csfp_panel(; n_assets::Integer = 60, n_observations::Integer = 300,
                    n_industries::Integer = 4, seed::Integer = 725_001, kwargs...)
    return synthetic_asset_panel(; n_assets = n_assets, n_observations = n_observations,
                                 n_industries = n_industries, rng = StableRNG(seed),
                                 kwargs...)
end

# `Ms[t - lag] f_t + eps_t` against `X_t`, over every pair whose arithmetic is finite. The block's
# exposure history is contemporaneous with the returns, so the identity is checkable from `lag + 1`.
function csfp_reconciliation(pr, rd, lag)
    rr = pr.rr
    Ms, f, eps = rr.Ms, pr.fpr.X, rr.csr.eps
    T, N = size(eps)
    X = view(rd.X, (size(rd.X, 1) - T + 1):size(rd.X, 1), :)
    d = 0.0
    n = 0
    for t in (lag + 1):T, i in 1:N
        v = LinearAlgebra.dot(view(Ms, t - lag, i, :), view(f, t, :)) + eps[t, i]
        if isfinite(v) && isfinite(X[t, i])
            d = max(d, abs(v - X[t, i]))
            n += 1
        end
    end
    return d, n
end

# The benchmark-weighted sum of a family's factor returns, relative to the sum of the magnitudes of
# its terms. The exposures are read `lag` observations back, because the fit's coefficients are
# coordinates in the basis of that observation.
function csfp_zero_sum(rr, f, ind, lag)
    bw, Ms = rr.bw, rr.Ms
    z = 0.0
    for u in axes(f, 1)
        t = u - lag
        if t < 1
            continue
        end
        s = 0.0
        tot = 0.0
        for k in ind
            c = 0.0
            for a in axes(bw, 2)
                e = Ms[t, a, k]
                if isfinite(e)
                    c += bw[t, a] * e
                end
            end
            s += c * f[u, k]
            tot += abs(c * f[u, k])
        end
        z = max(z, abs(s) / max(tot, eps()))
    end
    return z
end

# The largest benchmark-weighted inner product between a factor of `a` and a factor of `b`.
function csfp_orthogonality(rr, a, b)
    bw, Ms = rr.bw, rr.Ms
    o = 0.0
    for t in axes(Ms, 1), p in a, q in b
        s = 0.0
        for j in axes(bw, 2)
            x = Ms[t, j, p]
            y = Ms[t, j, q]
            if isfinite(x) && isfinite(y)
                s += bw[t, j] * x * y
            end
        end
        o = max(o, abs(s))
    end
    return o
end

function csfp_investable(pr)
    msk = PortfolioOptimisers.investable_mask(pr)
    return isnothing(msk) ? collect(eachindex(pr.mu)) : findall(msk)
end

@testset "The estimator, its defaults and its refusals" begin
    PO = PortfolioOptimisers
    pe = CrossSectionalFactorPrior(; factors = csfp_factors())
    @testset "Every default is the reference implementation's own" begin
        @test isa(pe, PO.AbstractLowOrderPriorEstimator_A)
        @test isa(pe.cre, CrossSectionalLinearRegression)
        @test isa(pe.wa, MarketCapWeights)
        @test pe.wa.p == 0.5
        @test isa(pe.pe, EmpiricalPrior)
        @test isa(pe.ve, RegimeAdjustedExpWeightedVariance)
        @test iszero(pe.th)
        @test isone(pe.bp)
        @test pe.mcap == "market_cap"
        @test pe.bw == "benchmark_weights"
        @test isone(pe.lag)
        @test isnothing(pe.minra)
        @test isnothing(pe.neutralise)
        @test isnothing(pe.families)
    end
    @testset "The three list fields collect Pairs and a dictionary alike" begin
        @test [first(p) for p in pe.factors] == ["market", "industry", "size", "value"]
        d = CrossSectionalFactorPrior(; factors = csfp_factors(),
                                      neutralise = Dict("style" => "industry"),
                                      families = ["industry" => nothing])
        @test [first(p) for p in d.neutralise] == ["style"]
        @test [first(p) for p in d.families] == ["industry"]
        @test_throws ArgumentError CrossSectionalFactorPrior(;
                                                             factors = ["a" =>
                                                                            ConstantExposure(),
                                                                        "a" =>
                                                                            ConstantExposure()])
        @test_throws PO.IsEmptyError CrossSectionalFactorPrior(;
                                                               factors = Pair{String, Any}[])
    end
    @testset "The scalar guards" begin
        f = csfp_factors()
        @test_throws DomainError CrossSectionalFactorPrior(; factors = f, th = 1.5)
        @test_throws DomainError CrossSectionalFactorPrior(; factors = f, bp = -1.0)
        @test_throws DomainError CrossSectionalFactorPrior(; factors = f, lag = 0)
        @test_throws DomainError CrossSectionalFactorPrior(; factors = f, minra = 0)
        @test_throws PO.IsEmptyError CrossSectionalFactorPrior(; factors = f, mcap = "")
    end
    @testset "A bare matrix and a missing panel are refused by name" begin
        rd = csfp_panel(; n_assets = 10, n_observations = 20, n_industries = 2).rd
        @test_throws ArgumentError prior(pe, rd.X)
        @test_throws PO.IsNothingError prior(pe, ReturnsResult(; nx = rd.nx, X = rd.X))
        # The residual declaration has no shape that can express this estimator's block.
        @test_throws ArgumentError PO.factor_residual_config(pe)
    end
end

@testset "The exposure history and its dependency order" begin
    PO = PortfolioOptimisers
    rd = csfp_panel(; n_assets = 12, n_observations = 40, n_industries = 3).rd
    mcap = PO.panel_field_values(rd, "market_cap")
    msk = isfinite.(rd.X) .& rd.pnl.emsk
    PO.cross_sectional_cap_finite!(msk, mcap)
    W = PO.cross_sectional_cap_weights(1.0, mcap, msk)
    rdb = PO.cross_sectional_benchmark_carrier(rd, "benchmark_weights", W)
    @testset "The carrier gains the benchmark weights, and replaces its own" begin
        @test PO.panel_field(rdb.pnl, "benchmark_weights").vals == W
        again = PO.cross_sectional_benchmark_carrier(rdb, "benchmark_weights", 2 .* W)
        @test PO.panel_field(again.pnl, "benchmark_weights").vals == 2 .* W
        @test length(again.pnl.pf) == length(rdb.pnl.pf)
    end
    @testset "A derived member is computed after its source, wherever it is written" begin
        f = ["size2" => DerivedExposure(; source = "size", f = x -> abs2.(x)),
             "size" =>
                 CompositeExposure(; descriptors = [LogMarketCap()], family = "style"),
             "market" => ConstantExposure()]
        ord, src = PO.cross_sectional_exposure_order(f)
        @test ord == [2, 3, 1]
        @test src == [2, 0, 0]
        hist = PO.cross_sectional_exposure_history(f, rdb)
        # The factor axis keeps the order the caller wrote, whatever order the fit took.
        @test hist.nf == ["size2", "size", "market"]
        @test hist.fam == ["style", "style", "market"]
        @test size(hist.Ms) == (size(rd.X, 1), size(rd.X, 2), 3)
        @test isequal(hist.Ms[:, :, 1], factor_exposure(last(f[1]), rdb, hist.Ms[:, :, 2]))
    end
    @testset "An unknown source, a self-reference and a cycle are refused" begin
        @test_throws ArgumentError PO.cross_sectional_exposure_order(["a" =>
                                                                          DerivedExposure(;
                                                                                          source = "b",
                                                                                          f = identity)])
        @test_throws ArgumentError PO.cross_sectional_exposure_order(["a" =>
                                                                          DerivedExposure(;
                                                                                          source = "a",
                                                                                          f = identity)])
        @test_throws ArgumentError PO.cross_sectional_exposure_order(["a" =>
                                                                          DerivedExposure(;
                                                                                          source = "b",
                                                                                          f = identity),
                                                                      "b" =>
                                                                          DerivedExposure(;
                                                                                          source = "a",
                                                                                          f = identity)])
    end
    @testset "A one-hot source is refused, because a derived member reads one exposure" begin
        f = ["industry" => OneHotExposure(; field = "industry", family = "industry"),
             "d" => DerivedExposure(; source = "industry", f = identity)]
        @test_throws ArgumentError PO.cross_sectional_exposure_history(f, rdb)
    end
    @testset "A history that is all warm-up, and a blank market capitalisation" begin
        # Every exposure is NaN, so no observation carries a usable asset and the whole
        # history is warm-up.
        Ms = fill(NaN, size(rd.X, 1), size(rd.X, 2), 2)
        @test_throws ArgumentError PO.cross_sectional_warmup(rd.X, Ms, rd.pnl.emsk)
        # The first observation is warm, so the count of leading cold observations is zero.
        Ms[1, 1, :] .= 0.0
        @test iszero(PO.cross_sectional_warmup(rd.X, Ms, rd.pnl.emsk))
        # A pair with no market capitalisation leaves the two weight masks.
        msk = trues(2, 2)
        PO.cross_sectional_cap_finite!(msk, [1.0 NaN; 2.0 3.0])
        @test msk == [true false; true true]
        PO.cross_sectional_cap_finite!(msk, nothing)
        @test msk == [true false; true true]
    end
    @testset "A static Asset Panel states no point-in-time universe" begin
        pnl = AssetPanel(; pf = [NumericPanelField(; name = "a", vals = [1.0, 2.0])])
        @test_throws ArgumentError PO.cross_sectional_panel_masks(pnl)
        @test PO.cross_sectional_panel_masks(rd.pnl) == (rd.pnl.amsk, rd.pnl.emsk)
    end
end

@testset "The fit, and the four identities it makes exact" begin
    PO = PortfolioOptimisers
    res = csfp_panel()
    rd = res.rd
    pr = prior(CrossSectionalFactorPrior(; factors = csfp_factors()), rd)
    rr = pr.rr
    i = csfp_investable(pr)
    @testset "The result is a LowOrderPrior over the full asset universe" begin
        @test isa(pr, LowOrderPrior)
        @test length(pr.mu) == size(rd.X, 2)
        @test size(pr.sigma) == (size(rd.X, 2), size(rd.X, 2))
        @test size(pr.X) == size(pr.o_X)
        @test size(pr.X, 1) == size(rr.csr.f, 1)
        @test size(pr.chol, 2) == length(pr.mu)
        @test isa(rr, CrossSectionalFactorModel)
        @test isa(pr.fpr, LowOrderPrior)
        @test !isnothing(pr.pnl)
        @test size(pr.pnl.amsk) == size(pr.X)
    end
    @testset "A non-investable asset carries NaN, and the mask is derived from it" begin
        @test length(i) < length(pr.mu)
        @test all(isfinite, view(pr.mu, i))
        @test all(isfinite, view(pr.sigma, i, i))
        j = setdiff(eachindex(pr.mu), i)
        @test all(isnan, view(pr.mu, j))
        @test all(isnan, [pr.sigma[k, k] for k in j])
        @test findall(PO.investable_mask(pr)) == i
    end
    @testset "The exposures lagged by one reproduce the returns exactly" begin
        d, n = csfp_reconciliation(pr, rd, 1)
        @test n > 10_000
        @test d < 1e-14
    end
    @testset "The covariance and its square root agree with the factor model" begin
        L = rr.M[i, :]
        S = L * pr.fpr.sigma * transpose(L) + LinearAlgebra.diagm(rr.esigma[i])
        @test isapprox(S, pr.sigma[i, i]; atol = 1e-14)
        C = pr.chol[:, i]
        @test isapprox(transpose(C) * C, pr.sigma[i, i]; atol = 1e-14)
        @test size(pr.chol, 1) == size(rr.M, 2) + length(i)
    end
    @testset "mu is the loadings through the factor mean, plus the orthogonal part" begin
        @test view(pr.mu, i) == view(rr.M * pr.fpr.mu + rr.b, i)
        # The Return Forecast enters `b` after this ticket, so it is zero here.
        @test iszero(rr.b)
    end
    @testset "The block carries the thirteen facts of the fit" begin
        @test size(rr.Ms) == (size(rr.csr.f, 1), size(rd.X, 2), length(rr.nf))
        # `isequal`, not `==`: an inactive asset carries a NaN Factor Exposure.
        @test isequal(rr.Ms[end, :, :], rr.M)
        @test size(rr.vs) == size(rr.rw) == size(rr.bw) == size(rr.csr.eps)
        @test rr.nf ==
              ["market", "industry=Real Estate", "industry=Software", "industry=Banks",
               "industry=Energy", "size", "value"]
        @test rr.fam ==
              ["market", "industry", "industry", "industry", "industry", "style", "style"]
        @test isone(rr.lag)
        @test isnothing(getfield(rr, :L))
        @test isnothing(rr.fcb)
        @test !PO.has_family_rebasis(rr)
        @test isa(rr.esigma, AbstractVector)
        @test rr.esigma == rr.vs[end, :]
        # The pre-fit axis verb and the fitted block answer the same axis.
        ax = PO.cross_sectional_factor_axis(csfp_factors(), rd)
        @test ax.nf == rr.nf
        @test ax.fam == rr.fam
    end
    @testset "The scenarios carry the latest factor and idiosyncratic risk" begin
        S = PO.cross_sectional_standardised_residuals(rr.csr.eps, rr.vs, pr.pnl.amsk)
        @test isequal(pr.X, pr.fpr.X * transpose(rr.M) .+ S .* transpose(sqrt.(rr.esigma)))
        am = view(pr.pnl.amsk, :, i)
        Xi = pr.X[:, i]
        @test all(isfinite, Xi[am])
        # An asset that was not listed at an observation carries NaN in that scenario. The
        # reference implementation leaves the same NaN there, for the same reason: the
        # observation has no idiosyncratic return to standardise.
        @test all(isnan, Xi[.!am])
    end
end

@testset "A constrained Factor Family re-bases the fit" begin
    PO = PortfolioOptimisers
    rd = csfp_panel().rd
    pe = CrossSectionalFactorPrior(; factors = csfp_factors(),
                                   families = ["industry" => nothing])
    pr = prior(pe, rd)
    rr = pr.rr
    i = csfp_investable(pr)
    ind = findall(isequal("industry"), rr.fam)
    @testset "The block states the re-basis, and L sits beside the basis" begin
        @test !isnothing(getfield(rr, :L))
        @test !isnothing(rr.fcb)
        @test PO.has_family_rebasis(rr)
        @test isa(rr.fcb, FactorFamilyBasis)
        @test rr.fcb.fnm == ["industry"]
        @test size(getfield(rr, :L), 2) == PO.reduced_factor_count(rr.fcb)
        @test size(rr.M, 2) == rr.fcb.K
    end
    @testset "The benchmark-weighted family return is zero at every observation" begin
        @test length(ind) == 4
        @test csfp_zero_sum(rr, pr.fpr.X, ind, 1) < 1e-12
    end
    @testset "The re-basis leaves the return reconciliation exact" begin
        d, n = csfp_reconciliation(pr, rd, 1)
        @test n > 10_000
        @test d < 1e-14
    end
    @testset "The moments are built through the reduced basis, never the expanded one" begin
        L = getfield(rr, :L)[i, :]
        F = PO.reduce_factor_covariance(rr.fcb, pr.fpr.sigma)
        S = L * F * transpose(L) + LinearAlgebra.diagm(rr.esigma[i])
        @test isapprox(S, pr.sigma[i, i]; atol = 1e-14)
        # The expanded factor covariance is singular by construction, so it cannot be the
        # one the square root came from.
        @test LinearAlgebra.rank(pr.fpr.sigma) == PO.reduced_factor_count(rr.fcb)
        # The two sides take different routes through the re-basis, so they agree to
        # floating point rather than bit for bit as they do on the raw axis.
        @test isapprox(view(pr.mu, i), view(rr.M * pr.fpr.mu + rr.b, i); rtol = 1e-12)
    end
end

@testset "Neutralisation removes the benchmark-weighted overlap" begin
    PO = PortfolioOptimisers
    rd = csfp_panel().rd
    ctl = prior(CrossSectionalFactorPrior(; factors = csfp_factors()), rd)
    pr = prior(CrossSectionalFactorPrior(; factors = csfp_factors(),
                                         neutralise = ["style" => "industry"]), rd)
    sty = findall(isequal("style"), pr.rr.fam)
    ind = findall(isequal("industry"), pr.rr.fam)
    @test length(sty) == 2
    @test csfp_orthogonality(ctl.rr, sty, ind) > 1e3
    @test csfp_orthogonality(pr.rr, sty, ind) < 1e-8
    # The market and the industry blocks are untouched: only the key's columns are rewritten.
    @test isequal(pr.rr.Ms[:, :, ind], ctl.rr.Ms[:, :, ind])
    @test isequal(pr.rr.Ms[:, :, 1], ctl.rr.Ms[:, :, 1])
end

@testset "The weight policy, the overlay and the lag" begin
    PO = PortfolioOptimisers
    rd = csfp_panel().rd
    @testset "The blended policy takes a second pass, and its weights sum to one" begin
        pr = prior(CrossSectionalFactorPrior(; factors = csfp_factors(),
                                             wa = BlendedInverseVarianceWeights(;
                                                                                lambda = 0.5)),
                   rd)
        @test PO.needs_second_pass(BlendedInverseVarianceWeights(; lambda = 0.5))
        @test all(x -> isapprox(x, 1.0), sum(pr.rr.rw; dims = 2))
        @test all(x -> x >= 0, pr.rr.rw)
        d, n = csfp_reconciliation(pr, rd, 1)
        @test d < 1e-14
    end
    @testset "A positive threshold turns esigma from a vector into a matrix" begin
        pr = prior(CrossSectionalFactorPrior(; factors = csfp_factors(), th = 0.2), rd)
        es = pr.rr.esigma
        i = csfp_investable(pr)
        @test isa(es, AbstractMatrix)
        @test size(es) == (length(pr.mu), length(pr.mu))
        @test isapprox(es[i, i], transpose(es[i, i]))
        @test isapprox([es[k, k] for k in i], pr.rr.vs[end, i])
        @test LinearAlgebra.isposdef(es[i, i])
        L = pr.rr.M[i, :]
        S = L * pr.fpr.sigma * transpose(L) + es[i, i]
        @test isapprox(S, pr.sigma[i, i]; atol = 1e-14)
        C = pr.chol[:, i]
        @test isapprox(transpose(C) * C, pr.sigma[i, i]; atol = 1e-14)
    end
    @testset "A power of zero reads no market capitalisation" begin
        @test !PO.cross_sectional_needs_market_cap(0.0, MarketCapWeights(; p = 0.0))
        @test PO.cross_sectional_needs_market_cap(1.0, MarketCapWeights(; p = 0.0))
        @test PO.cross_sectional_needs_market_cap(0.0, MarketCapWeights(; p = 0.5))
        pr = prior(CrossSectionalFactorPrior(; factors = csfp_factors(), bp = 0.0,
                                             wa = MarketCapWeights(; p = 0.0)), rd)
        @test all(x -> x >= 0, pr.rr.bw)
        d, n = csfp_reconciliation(pr, rd, 1)
        @test d < 1e-14
    end
    @testset "A longer lag shortens the fit and stays exact" begin
        pr = prior(CrossSectionalFactorPrior(; factors = csfp_factors(), lag = 5), rd)
        @test pr.rr.lag == 5
        @test size(pr.X, 1) ==
              size(prior(CrossSectionalFactorPrior(; factors = csfp_factors()), rd).X, 1) -
              4
        d, n = csfp_reconciliation(pr, rd, 5)
        @test d < 1e-14
    end
    @testset "Too short a history, and too thin a cross-section, are refused" begin
        short = csfp_panel(; n_assets = 40, n_observations = 8, n_industries = 2).rd
        @test_throws ArgumentError prior(CrossSectionalFactorPrior(;
                                                                   factors = csfp_factors(),
                                                                   lag = 20), short)
        @test_throws ArgumentError prior(CrossSectionalFactorPrior(;
                                                                   factors = csfp_factors(),
                                                                   minra = 10_000), rd)
    end
end

@testset "The fit recovers the model the synthetic panel was drawn from" begin
    PO = PortfolioOptimisers
    res = csfp_panel(; n_assets = 150, n_observations = 500, seed = 725_002)
    rd, truth = res.rd, res.truth
    st(d) = CompositeExposure(; descriptors = [d], family = "style")
    factors = ["market" =>
                   CompositeExposure(; descriptors = [EWMarketBeta()], outlier = nothing,
                                     scoring = nothing, family = "market"),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "size" => st(LogMarketCap()), "value" => st(BookToPrice()),
               "earnings_yield" => st(EarningsToPrice()),
               "profitability" => st(GrossProfitability()),
               "growth" => st(SalesGrowthRate(; lag = 60)),
               "investment" =>
                   st(PanelFieldRatio(; num = "capex_ttm", den = "total_assets")),
               "leverage" => st(MarketLeverage()),
               "dividend_yield" => st(DividendToPrice()),
               "liquidity" => st(EWShareTurnover()), "volatility" => st(EWVolatility())]
    # The industry block sums to one for every asset, so it is near-collinear with a market
    # factor whose exposure is a beta close to one. The zero-sum constraint is what
    # identifies the members of such a family, so the recovery is measured under it.
    pr = prior(CrossSectionalFactorPrior(; factors = factors,
                                         families = ["industry" => nothing]), rd)
    rr = pr.rr
    @test rr.nf == truth.nf
    @test rr.fam == truth.fgrp
    act = findall(view(rd.pnl.amsk, size(rd.pnl.amsk, 1), :))
    @testset "The industry loadings are recovered exactly" begin
        ind = findall(isequal("industry"), rr.fam)
        @test rr.M[act, ind] == truth.B[act, ind]
    end
    @testset "Every other loading tracks the trait it was drawn from" begin
        for k in eachindex(rr.nf)
            a = view(rr.M, act, k)
            b = view(truth.B, act, k)
            ok = findall(isfinite, a)
            @test Statistics.cor(a[ok], b[ok]) > 0.6
        end
    end
    @testset "The systematic return of every pair is recovered" begin
        # The basis-invariant statement. A factor return alone is identified only up to the
        # basis the family constraint chose, but `Ms[t - lag] f_t` is the quantity the model
        # asserts about the asset, and it is comparable with the generator's own.
        T = size(rr.csr.eps, 1)
        tf = view(truth.f, (size(truth.f, 1) - T + 1):size(truth.f, 1), :)
        a = Float64[]
        b = Float64[]
        for t in 2:T, i in axes(rr.csr.eps, 2)
            u = LinearAlgebra.dot(view(rr.Ms, t - 1, i, :), view(pr.fpr.X, t, :))
            v = LinearAlgebra.dot(view(truth.B, i, :), view(tf, t, :))
            if isfinite(u) && isfinite(v)
                push!(a, u)
                push!(b, v)
            end
        end
        @test length(a) > 40_000
        @test Statistics.cor(a, b) > 0.9
    end
    @testset "Every factor return tracks the series that generated it" begin
        T = size(rr.csr.f, 1)
        tf = view(truth.f, (size(truth.f, 1) - T + 1):size(truth.f, 1), :)
        c = [Statistics.cor(view(pr.fpr.X, :, k), view(tf, :, k)) for k in eachindex(rr.nf)]
        # A constrained member's realised return is a contrast against its own family, so it
        # cannot equal the generator's independent series even in the noiseless case.
        @test all(x -> x > 0.2, c)
        @test Statistics.median(c) > 0.55
        @test c[1] > 0.7
        @test c[findfirst(isequal("size"), rr.nf)] > 0.7
    end
    @testset "The idiosyncratic variances recover their level, not only their order" begin
        ev = rr.esigma
        ok = findall(isfinite, ev)
        @test length(ok) > 100
        @test Statistics.cor(ev[ok], truth.ivar[ok]) > 0.85
        @test Statistics.cor(log.(ev[ok]), log.(truth.ivar[ok])) > 0.8
        @test 0.8 < Statistics.median(ev[ok] ./ truth.ivar[ok]) < 1.5
    end
end
