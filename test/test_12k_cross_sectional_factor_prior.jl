#=
The Cross-Sectional Factor Prior (issues #725 and #739, map #643).

WHAT THIS FILE GATES. The estimator builds a point-in-time factor model from an Asset Panel and
lifts it onto the assets. Four identities are exact and are asserted at machine precision, because
each one is a construction rather than an estimate:

 1. `Ms[t - lag] f_t + eps_t == X_t` on every eligible pair. The realised factor returns expand
    with the LAGGED family ratios exactly so that this holds.
 2. `L F L' + D == sigma` on the investable block, and `chol' chol == sigma` beside it.
 3. `mu == M fpr.mu + b` on the investable assets. `b` is zero until a Return Forecast
    Estimator fills it, and the identity holds either way.
 4. A constrained Factor Family drives the benchmark-weighted sum of its factor returns to zero,
    again under the LAGGED ratios.

The recovery testset is statistical, not exact: the synthetic panel's Panel Fields are noisy
functions of the true loadings, so a fitted exposure correlates with the truth rather than equalling
it. The one exception is the industry block, which a one-hot exposure recovers exactly.

THE FOUR STORED CASES ARE THE REFERENCE IMPLEMENTATION'S OWN OUTPUT.
`assets/CrossSectionalFactorPriorFactorReturns.csv.gz` and
`assets/CrossSectionalFactorPriorFamilyFactorReturns.csv.gz` hold the factor returns the reference
implementation's own prior produced, driven on the panel `csfp_reference_case` rebuilds, with the
same four factors, the same lag, the same two capitalisation powers and the same factor prior.
`assets/CrossSectionalFactorPriorForecastMu.csv.gz` and
`assets/CrossSectionalFactorPriorFamilyForecastMu.csv.gz` hold the expected returns the same prior
produced on the same panel, with its fixed-weighted forecast over `signal`, a shrinkage of one and a
confidence of one. The forecast itself agrees BIT FOR BIT, the factor mean to 1.1e-18 and the
expected returns to 1.6e-17, in both the plain and the constrained-family case. The
whole fit was diffed the same way before the cases were stored, over 40 assets and 120 observations:
the loadings and the benchmark weights agree BIT FOR BIT, the regression weights to 5.7e-14, the
factor returns to 1.3e-16, the idiosyncratic returns to 5.9e-16 and the factor mean to 1.1e-18, in
both the plain and the constrained-family case. The two libraries solve the same weighted least
squares by different routes -- this one factorises the weighted design, the reference implementation
solves the normal equations -- so machine precision is the agreement to expect.

The panel of those testsets is fully active and carries no blank cell, and its two style exposures
are standardised in the test rather than by a Descriptor. Both choices are deliberate: they take
every departure below out of the picture, so the stored cases measure the fit alone. Issue #721
already diffed the Factor Exposures themselves.

ONE DEPARTURE FROM THE REFERENCE IMPLEMENTATION, recorded in the resolution comment of #725. Two
others are gone. The first, recorded in that of #739, went with issue #835, which built ADR 0112, so
the Return Forecast Estimator now reads the WHOLE carrier and answers on the block's rows. The
second went with issue #925, below.

  - The benchmark mask and the eligibility mask both drop a pair whose market capitalisation is not
    finite. The reference implementation lets such a pair carry a `NaN` weight. The library refuses
    a non-finite capitalisation on an eligible pair, and `exposure_benchmark_weights` already zeroes
    a non-finite weight, so dropping the pair is what the library's own convention asks for.

THE IDIOSYNCRATIC OVERLAY'S FILL VALUE IS KEYED ON THE ESTIMATOR (issue #925). The overlay used to
write a zero at every standardised residual that was still not finite, which happens only where the
asset is inactive, and it wrote it unconditionally: no estimator in the `ce` slot ever saw the gap.
It now asks `ce` what a gapped cell is worth to it with `gap_fill_value`. A plain moment estimator
takes the fallback zero, because it refuses a gapped sample outright; a gap-aware one answers `NaN`,
which leaves the gap where it is and is handed the panel's active mask beside it. The slot's default
is now `ExpWeightedCovariance(; centred = true)`, which is what the reference implementation's own
default is, so the overlay reaches its answer to machine precision. `gap_fill_value` recurses
through a composite that forwards the sample untouched, and `test_08z` gates the trait itself.
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

# The fit keeps the tail of the observation axis: the window left after the Descriptors' warm-up
# and the exposure lag. `pr.X` has one row per surviving observation, so the rows the fit used are
# recoverable from the carrier and that count alone.
fit_rows(rd, pr) = (size(rd.X, 1) - size(pr.X, 1) + 1):size(rd.X, 1)

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
        # Issue #925. `EWCovariance(assume_centered=True, nearest=False)` is the reference
        # implementation's own default, and `centred` is the whole residual against it.
        @test isa(pe.ce, ExpWeightedCovariance)
        @test pe.ce.centred
        @test isnan(PO.gap_fill_value(pe.ce))
        @test iszero(pe.th)
        @test isone(pe.bp)
        @test pe.mcap == "market_cap"
        @test pe.bw == "benchmark_weights"
        @test isone(pe.lag)
        @test isnothing(pe.minra)
        @test isnothing(pe.rfe)
        @test isone(pe.lambda)
        @test isone(pe.c)
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
        @test_throws DomainError CrossSectionalFactorPrior(; factors = f, lambda = 1.5)
        @test_throws DomainError CrossSectionalFactorPrior(; factors = f, c = -0.1)
        @test_throws PO.IsEmptyError CrossSectionalFactorPrior(; factors = f, mcap = "")
    end
    @testset "A matrix with no panel is refused by name" begin
        rd = csfp_panel(; n_assets = 10, n_observations = 20, n_industries = 2).rd
        @test_throws PO.IsNothingError prior(pe, rd.X)
        @test_throws PO.IsNothingError prior(pe, rd.X, nothing, nothing)
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
        #=
        `#803` made the Asset Panel the one carrier of feature data, so no prior result
        carries one. The scenario rows are the tail of the observation axis -- the window
        left after the Descriptor warm-up and the exposure lag -- so a consumer recovers the
        masks that produced them from `rd` and the row count, which is what the diagnostics
        below do.
        =#
        @test !hasproperty(pr, :pnl)
        @test size(view(rd.pnl.amsk, fit_rows(rd, pr), :)) == size(pr.X)
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
        # `rtol`, not `==`: the fit and this line both form `M * fpr.mu`, and the reduction
        # order of that product is the BLAS thread count of the machine, so the two answers
        # differ in the last bit on a CI runner.
        @test isapprox(view(pr.mu, i), view(rr.M * pr.fpr.mu + rr.b, i); rtol = 1e-14)
        # The Return Forecast enters `b` after this ticket, so it is zero here.
        @test iszero(rr.b)
    end
    @testset "The block carries the fourteen facts of the fit" begin
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
        # The prior states no Return Forecast Estimator here, so the block carries none.
        @test isnothing(rr.rf)
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
        amsk = view(rd.pnl.amsk, fit_rows(rd, pr), :)
        S = PO.cross_sectional_standardised_residuals(rr.csr.eps, rr.vs, amsk)
        @test isequal(pr.X, pr.fpr.X * transpose(rr.M) .+ S .* transpose(sqrt.(rr.esigma)))
        am = view(amsk, :, i)
        Xi = pr.X[:, i]
        @test all(isfinite, Xi[am])
        # An asset that was not listed at an observation carries NaN in that scenario. The
        # reference implementation leaves the same NaN there, for the same reason: the
        # observation has no idiosyncratic return to standardise.
        @test all(isnan, Xi[.!am])
    end
end

#=
Issue #840. A wrapping prior holds no carrier: it reaches the estimator it nests through the
returns-matrix method. The Asset Panel travels there as the third positional argument, so an
estimator that is fitted on a panel composes like any other one.

The panel of this testset is fully active. A wrapping prior processes the moments it is handed,
and the fit states `NaN` at an asset it holds no moment for, which such processing refuses.

Neither wrapper here reweights observations. The testset below covers the three that do.
=#
@testset "A wrapping prior composes the fit, because the panel travels" begin
    PO = PortfolioOptimisers
    rd = csfp_panel(; n_assets = 20, n_observations = 60, n_industries = 3, seed = 782_001,
                    late_listing_proba = 0.0, delisting_proba = 0.0, missing_ratio = 0.0).rd
    pe = CrossSectionalFactorPrior(; factors = csfp_factors(), minra = 5)
    pr = prior(pe, rd)
    @testset "The returns-matrix method fits from the panel alone" begin
        @test all(isfinite, pr.mu)
        prm = prior(pe, rd.X, nothing, rd.pnl)
        @test prm.mu == pr.mu
        @test prm.sigma == pr.sigma
        @test prm.X == pr.X
        @test isa(prm.rr, CrossSectionalFactorModel)
    end
    @testset "A Black-Litterman prior replaces the moments and forwards the block" begin
        sets = UniverseSets(; dict = Dict("nx" => rd.nx))
        views = LinearConstraintEstimator(; val = "$(rd.nx[1]) == 0.002")
        bl = prior(BlackLittermanPrior(; pe = pe, sets = sets, views = views), rd)
        @test size(bl.X) == size(pr.X)
        @test all(isfinite, bl.mu)
        @test bl.mu != pr.mu
        @test isa(bl.rr, CrossSectionalFactorModel)
        @test bl.rr.M == pr.rr.M
        @test bl.fpr.mu == pr.fpr.mu
    end
    @testset "A high order prior wraps the same fit" begin
        ho = prior(HighOrderPriorEstimator(; pe = pe), rd)
        @test isa(ho, HighOrderPrior)
        @test ho.pr.mu == pr.mu
        @test isa(ho.pr.rr, CrossSectionalFactorModel)
    end
end

#=
Issue #849, and ADR 0116. A prior that reweights observations works on the observation axis its
nested prior ANSWERED. This fit drops the observations the Descriptors warm up over and the
observations the exposure lag consumes, so it answers on fewer rows than it is given. The three
reweighting priors fit the nested prior first and read their prior probabilities on its rows, so
all three compose this estimator.

The mean view is enforced on the matrix the constraint was built against, which is the first,
unweighted fit. The moments of the result come from the refit, so `pr.mu` is not the reweighted
scenario mean.
=#
@testset "A prior that reweights observations composes the fit" begin
    rd = csfp_panel(; n_assets = 20, n_observations = 60, n_industries = 3, seed = 782_001,
                    late_listing_proba = 0.0, delisting_proba = 0.0, missing_ratio = 0.0).rd
    pe = CrossSectionalFactorPrior(; factors = csfp_factors(), minra = 5)
    pr = prior(pe, rd)
    sets = UniverseSets(; dict = Dict("nx" => rd.nx))
    views = LinearConstraintEstimator(; val = "$(rd.nx[1]) == 0.002")
    T = size(pr.X, 1)
    @test T < size(rd.X, 1)
    @testset "Both entropy pooling priors answer on the rows the fit kept" begin
        for wr in (EntropyPoolingPrior(; pe = pe, sets = sets, mu_views = views),
                   MeucciEntropyPoolingPrior(; pe = pe, sets = sets, mu_views = views))
            pw = prior(wr, rd)
            @test size(pw.X) == size(pr.X)
            @test length(pw.w) == T
            @test isapprox(sum(pw.w), 1; rtol = 1e-6)
            @test isapprox(LinearAlgebra.dot(pw.w, pr.X[:, 1]), 0.002; rtol = 1e-5)
            @test isa(pw.rr, CrossSectionalFactorModel)
            @test all(isfinite, pw.mu)
        end
    end
    @testset "An opinion pool makes the scenarios in pe1 and pools over them" begin
        op = OpinionPoolingPrior(; pe1 = pe,
                                 pes = [EntropyPoolingPrior(; sets = sets,
                                                            mu_views = views)])
        pw = prior(op, rd)
        @test size(pw.X) == size(pr.X)
        @test length(pw.w) == T
        @test isapprox(pw.mu[1], 0.002; rtol = 1e-5)
    end
    @testset "An opinion that answers on a shorter axis is refused by name" begin
        op = OpinionPoolingPrior(;
                                 pes = [EntropyPoolingPrior(; pe = pe, sets = sets,
                                                            mu_views = views)])
        @test_throws DimensionMismatch prior(op, rd)
    end
    @testset "A caller's prior probabilities are stated on the axis the fit answered" begin
        w = StatsBase.pweights(fill(inv(T), T))
        @test length(prior(EntropyPoolingPrior(; pe = pe, sets = sets, mu_views = views,
                                               w = w), rd).w) == T
        Ti = size(rd.X, 1)
        wbad = StatsBase.pweights(fill(inv(Ti), Ti))
        @test_throws DimensionMismatch prior(EntropyPoolingPrior(; pe = pe, sets = sets,
                                                                 mu_views = views,
                                                                 w = wbad), rd)
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
    @testset "The overlay asks the estimator what a gapped cell is worth" begin
        # Issue #925. The overlay used to fill before it estimated, so no estimator in the slot
        # ever saw the gap. It now asks. The two answers are the two routes, and this testset
        # pins each against the estimator call it is supposed to make: nothing downstream of
        # that call differs between them.
        T, N, th = 80, 4, 0.2
        rng = StableRNG(925_001)
        g = randn(rng, T)
        # A shared driver, so the pairwise correlations clear the threshold and the testset is
        # not measuring a block of structural zeroes.
        S = 0.7 .* g .+ 0.7 .* randn(rng, T, N)
        amsk = trues(T, N)
        amsk[51:T, 4] .= false                    # asset 4 delists
        amsk[1:20, 3] .= false                    # asset 3 lists late
        S[51:T, 4] .= NaN
        S[1:20, 3] .= NaN
        ev = fill(0.04, N)
        se = sqrt.(ev)
        Z = map(x -> isfinite(x) ? x : 0.0, S)
        # The tail of the overlay, which is the same on both routes. `pdm` is `nothing` below,
        # so the identity is exact rather than up to a positive definite repair.
        function overlay_tail(C)
            R = StatsBase.cov2cor(Matrix(C), sqrt.(LinearAlgebra.diag(C)))
            for k in CartesianIndices(R)
                if k[1] != k[2] && !(abs(R[k]) > th)
                    R[k] = zero(eltype(R))
                end
            end
            for i in axes(R, 1)
                R[i, i] = one(eltype(R))
            end
            return R .* se .* transpose(se)
        end

        # A threshold of zero answers the variances and never reads `ce`, so it never asks.
        @test PO.cross_sectional_idiosyncratic_covariance(0.0,
                                                          PortfolioOptimisersCovariance(),
                                                          nothing, S, ev, amsk) === ev

        # A plain moment estimator refuses a gapped sample, so the overlay fills for it, and
        # the fill is the fallback zero and nothing else.
        plain = PortfolioOptimisersCovariance()
        @test iszero(PO.gap_fill_value(plain))
        @test_throws PO.IsNonFiniteError Statistics.cov(plain, S; dims = 1)
        Dp = PO.cross_sectional_idiosyncratic_covariance(th, plain, nothing, S, ev, amsk)
        @test isapprox(Dp, overlay_tail(Statistics.cov(plain, Z; dims = 1)))
        @test all(isfinite, Dp)

        # The gap-aware default is handed the gap and the mask instead, so its answer is the
        # masked estimator's own.
        ce = ExpWeightedCovariance(; centred = true)
        @test isnan(PO.gap_fill_value(ce))
        Dg = PO.cross_sectional_idiosyncratic_covariance(th, ce, nothing, S, ev, amsk)
        @test isapprox(Dg,
                       overlay_tail(Statistics.cov(ce, S; dims = 1, active_mask = amsk)))
        # The fill would have moved it: the mask freezes a delisted asset's block, and the
        # fill keeps decaying it, so the two routes part on the pairs the gap touches.
        Df = overlay_tail(Statistics.cov(ce, Z; dims = 1))
        @test !isapprox(Dg, Df)
        @test iszero(Dg[1, 4]) && !iszero(Df[1, 4])
        # Nothing non-finite survives the tail, which is what lets the warm-up `NaN` through.
        @test all(isfinite, Dg)

        # A composite that forwards the sample untouched keeps the gap, so the overlay takes
        # the same route through it as it takes through the estimator it wraps.
        for w in (PortfolioOptimisersCovariance(; ce = ce), ProcessedCovariance(; ce = ce),
                  CorrelationCovariance(; ce = ce))
            @test isnan(PO.gap_fill_value(w))
        end
        @test isapprox(PO.cross_sectional_idiosyncratic_covariance(th,
                                                                   PortfolioOptimisersCovariance(;
                                                                                                 ce = ce,
                                                                                                 mp = MatrixProcessing(;
                                                                                                                       pdm = nothing)),
                                                                   nothing, S, ev, amsk),
                       Dg)
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

# The panel every stored case is driven on, and the four factors of that fit. A fully active
# panel with no blank cell: every departure the two libraries have over an inactive or an
# unobserved cell is then out of the picture, and the stored cases measure the fit alone.
#
# The two style fields are standardised here rather than by a Descriptor, so a passthrough
# exposure is bit-identical on both sides and the diff isolates the fit. `signal` is a field no
# Factor Exposure reads, so a Return Forecast built on it carries a part the factors do not
# span, and the stored `mu` measures the split rather than a projection onto the design.
function csfp_reference_case()
    PO = PortfolioOptimisers
    rd0 = csfp_panel(; n_assets = 40, n_observations = 120, n_industries = 3,
                     seed = 725_900, late_listing_proba = 0.0, delisting_proba = 0.0,
                     missing_ratio = 0.0).rd
    function csfp_zscore(A)
        B = similar(A)
        for t in axes(A, 1)
            r = view(A, t, :)
            B[t, :] = (r .- Statistics.mean(r)) ./ Statistics.std(r; corrected = false)
        end
        return B
    end
    mcap = PO.panel_field_values(rd0, "market_cap")
    pf = Any[f for f in rd0.pnl.pf]
    push!(pf, NumericPanelField(; name = "style1", vals = csfp_zscore(log.(mcap))))
    push!(pf,
          NumericPanelField(; name = "style2",
                            vals = csfp_zscore(PO.panel_field_values(rd0, "book_equity") ./
                                               mcap)))
    push!(pf,
          NumericPanelField(; name = "signal",
                            vals = csfp_zscore(randn(StableRNG(739_100), size(rd0.X)...))))
    rd = ReturnsResult(; nx = rd0.nx, X = rd0.X, ts = rd0.ts,
                       pnl = AssetPanel(; pf = identity.(pf), amsk = rd0.pnl.amsk,
                                        emsk = rd0.pnl.emsk))
    csfp_pass(field) = CompositeExposure(; descriptors = [Passthrough(; field = field)],
                                         outlier = nothing, scoring = nothing,
                                         family = "style")
    factors = ["market" => ConstantExposure(),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "style1" => csfp_pass("style1"), "style2" => csfp_pass("style2")]
    return rd, factors
end

@testset "The fit matches the reference implementation's own output" begin
    rd, factors = csfp_reference_case()
    @test all(rd.pnl.amsk)
    for (nm, fams) in (("", nothing), ("Family", ["industry" => nothing]))
        pr = prior(CrossSectionalFactorPrior(; factors = factors, families = fams), rd)
        E = Matrix(CSV.read(joinpath(@__DIR__,
                                     "assets/CrossSectionalFactorPrior$(nm)FactorReturns.csv.gz"),
                            DataFrame))
        @test size(pr.fpr.X) == size(E)
        @test all(isfinite, E)
        # The two libraries solve the same weighted least squares by different routes: this
        # one factorises the weighted design, and the reference implementation solves the
        # normal equations. The absolute agreement is therefore machine precision, and a
        # near-zero factor return makes the relative figure larger than that.
        @test maximum(abs, pr.fpr.X - E) < 1e-14
        # The stored factor returns pin the residuals too, because the reconciliation is
        # exact: `Ms[t - 1] f_t + eps_t == X_t`.
        d, n = csfp_reconciliation(pr, rd, 1)
        @test n > 4000
        @test d < 1e-14
    end
end

@testset "The Return Forecast splits into a spanned part and an orthogonal part" begin
    PO = PortfolioOptimisers
    rd = csfp_panel().rd
    # The market factor and the one-hot industry block sum to the same column, so the split's
    # coefficients are not unique under the four factors the other testsets fit. These three
    # are independent, so `g` is the vector the forecast was built from.
    factors = ["market" => ConstantExposure(),
               "size" =>
                   CompositeExposure(; descriptors = [LogMarketCap()], family = "style"),
               "value" =>
                   CompositeExposure(; descriptors = [BookToPrice()], family = "style")]
    pr0 = prior(CrossSectionalFactorPrior(; factors = factors), rd)
    M = pr0.rr.M
    i = csfp_investable(pr0)
    N = length(pr0.mu)
    @testset "A spanned forecast is the factor mean itself at lambda = 0" begin
        g0 = [0.001, 0.002, -0.003]
        for c in (0.0, 0.5, 1.0)
            pr = prior(CrossSectionalFactorPrior(; factors = factors, lambda = 0.0, c = c,
                                                 rfe = CustomValueReturnForecast(;
                                                                                 mu = M *
                                                                                      g0)),
                       rd)
            @test isapprox(pr.fpr.mu, g0; atol = 1e-15)
            @test maximum(abs, view(pr.rr.b, i)) < 1e-15
            @test isapprox(view(pr.mu, i), view(M * g0, i); atol = 1e-15)
            @test isa(pr.rr.rf, CustomValueReturnForecastResult)
        end
    end
    @testset "An orthogonal forecast leaves the factor mean alone and scales into b" begin
        # A forecast the latest exposures do not span, built by projecting a ramp out of the
        # column space of `M` under the latest regression weights.
        w = pr0.rr.rw[end, :]
        fin = findall(k -> w[k] > zero(eltype(w)) && all(isfinite, view(M, k, :)), 1:N)
        y = zeros(N)
        y[fin] = collect(range(-0.01, 0.01; length = length(fin)))
        A = M[fin, :]
        W = LinearAlgebra.Diagonal(w[fin])
        ap = copy(y)
        ap[fin] = view(y, fin) -
                  A * ((transpose(A) * W * A) \ (transpose(A) * W * view(y, fin)))
        pr = prior(CrossSectionalFactorPrior(; factors = factors, lambda = 1.0, c = 0.5,
                                             rfe = CustomValueReturnForecast(; mu = ap)),
                   rd)
        @test isapprox(pr.fpr.mu, pr0.fpr.mu; atol = 1e-15)
        @test isapprox(view(pr.rr.b, i), 0.5 * view(ap, i); atol = 1e-15)
    end
    @testset "A forecast of zeros, of NaNs, and no estimator at all" begin
        # Each of the three states the same thing: nothing to span and nothing to add.
        for mu in (zeros(N), fill(NaN, N))
            pr = prior(CrossSectionalFactorPrior(; factors = factors, lambda = 0.0, c = 1.0,
                                                 rfe = CustomValueReturnForecast(; mu = mu)),
                       rd)
            @test iszero(pr.fpr.mu)
            @test iszero(pr.rr.b)
            @test iszero(view(pr.mu, i))
        end
        # With no estimator the spanned part is zero, so `lambda` shrinks the factor mean.
        for lambda in (0.0, 0.25, 1.0)
            pr = prior(CrossSectionalFactorPrior(; factors = factors, lambda = lambda), rd)
            @test isapprox(pr.fpr.mu, lambda * pr0.fpr.mu; atol = 1e-18)
            @test iszero(pr.rr.b)
            @test isnothing(pr.rr.rf)
        end
    end
end

@testset "The Return Forecast split matches the reference implementation's own output" begin
    rd, factors = csfp_reference_case()
    # One passthrough Descriptor over `signal`, scored by neither transform, so the forecast
    # is bit-identical on both sides and the two stored cases measure the split alone.
    rfe = FixedWeightedReturnForecast(;
                                      scores = DescriptorScores(;
                                                                descriptors = [Passthrough(;
                                                                                           field = "signal")],
                                                                outlier = nothing,
                                                                scoring = nothing),
                                      scale = 0.02)
    for (nm, fams) in (("", nothing), ("Family", ["industry" => nothing]))
        pr = prior(CrossSectionalFactorPrior(; factors = factors, families = fams,
                                             rfe = rfe, lambda = 1.0, c = 1.0), rd)
        E = vec(Matrix(CSV.read(joinpath(@__DIR__,
                                         "assets/CrossSectionalFactorPrior$(nm)ForecastMu.csv.gz"),
                                DataFrame)))
        @test length(E) == length(pr.mu)
        @test all(isfinite, E)
        # The reference implementation's own expected returns, over the same universe. The
        # split is the last step of the fit, so the stored vector pins the whole chain.
        @test maximum(abs, pr.mu - E) < 1e-14
        # The forecast the split consumed travels on the block, and its `mu` is the last
        # observation of its own history.
        @test pr.rr.rf.mu == pr.rr.rf.hist[end, :]
        @test !iszero(pr.rr.b)
        # `mu` is the loadings through the factor mean plus the orthogonal part, as it is
        # without a forecast.
        @test isapprox(pr.mu, pr.rr.M * pr.fpr.mu + pr.rr.b; atol = 1e-15)
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

@testset "A window too short for the warm-ups it must cover is refused by name (#956)" begin
    PO = PortfolioOptimisers
    # A Descriptor's warm-up and the factor prior's own warm-up are cumulative, and a
    # cross-validation fold hands the estimator its own rows alone, so a rolling train
    # window never grows to absorb either. Every band of a window too short for the sum
    # must refuse by name: the fit cannot state a correct answer, so it must say so.
    factors = csfp_factors()
    @testset "A covariance of one observation is not a number" begin
        # These Factor Exposures read levels, so they warm up over nothing and the whole
        # window reaches the regression. `lag` takes one observation, and what is left is
        # the factor-return history the factor prior reads.
        rd = csfp_panel(; n_observations = 3).rd
        pe = CrossSectionalFactorPrior(; factors = factors, lag = 1)
        # One observation left. The message names the factor prior's covariance, so the
        # caller is not left to read a LAPACK failure out of a factorisation.
        rd2 = PO.port_opt_view(rd, [1, 2], :)
        e = try
            prior(pe, rd2)
        catch err
            err
        end
        @test isa(e, ArgumentError)
        @test occursin("covariance of one observation is not a number", e.msg)
        # The floor is exactly two, and no wider: at three observations the fit clears it
        # and stops at the NEXT warm-up in the chain, the idiosyncratic variance
        # estimator's, which refuses under its own name. Every band is named.
        @test_throws PO.IsEmptyError prior(pe, rd)
    end
    @testset "A factor prior that warms up over the factor returns" begin
        # The reference implementation's own case: a factor prior whose covariance
        # estimator carries a warm-up longer than the factor-return history left to it.
        # It answers `NaN` rather than raising, so only a check on its answer catches it.
        # These Factor Exposures warm up over nothing, so the whole window reaches the
        # factor prior and the refusal below is its warm-up alone, not the Descriptors'.
        rd = csfp_panel(; n_observations = 300).rd
        pe = CrossSectionalFactorPrior(; factors = factors, lag = 1,
                                       pe = EmpiricalPrior(;
                                                           ce = ExpWeightedCovariance(;
                                                                                      min_obs = 500)))
        @test_throws PO.IsNonFiniteError prior(pe, rd)
        # The same fit stands once the factor prior's warm-up fits inside the history, so
        # the refusal reads the warm-up and not merely the estimator.
        pe2 = CrossSectionalFactorPrior(; factors = factors, lag = 1,
                                        pe = EmpiricalPrior(;
                                                            ce = ExpWeightedCovariance(;
                                                                                       min_obs = 40)))
        @test isa(prior(pe2, rd), LowOrderPrior)
    end
    @testset "The refusal reads the moments it was handed" begin
        # The verb itself, over bare arrays. A finite pair passes; a gap in either the
        # mean or the covariance is named and counted.
        @test isnothing(PO.assert_cross_sectional_factor_moments([1.0, 2.0],
                                                                 [1.0 0.0; 0.0 1.0], 5))
        @test_throws PO.IsNonFiniteError PO.assert_cross_sectional_factor_moments([1.0,
                                                                                   NaN],
                                                                                  [1.0 0.0;
                                                                                   0.0 1.0],
                                                                                  5)
        @test_throws PO.IsNonFiniteError PO.assert_cross_sectional_factor_moments([1.0,
                                                                                   2.0],
                                                                                  [1.0 NaN;
                                                                                   NaN 1.0],
                                                                                  1)
    end
    @testset "A fold's own train window takes the same refusal" begin
        # The scenario the issue names: the fold, not a hand-cut slice. The scheme's own
        # split supplies the train rows, and the prior refuses on them by name.
        rd = csfp_panel(; n_observations = 40).rd
        pe = CrossSectionalFactorPrior(; factors = factors, lag = 1)
        cv = IndexWalkForward(2, 5)
        (; train_idx) = PO.split(cv, rd)
        @test length(first(train_idx)) == 2
        @test_throws ArgumentError prior(pe, PO.port_opt_view(rd, first(train_idx), :))
    end
end

@testset "The factor covariance takes its own matrix processing estimator" begin
    PO = PortfolioOptimisers
    # The asset covariance and the factor covariance are different matrices, so each takes
    # its own estimator: `mp` processes the asset block, `f_mp` the factor one. The factor
    # block is the one `cross_sectional_lift` factorises for the low-rank square root, and
    # it is estimated over a factor axis a constrained Family has already reduced, so it can
    # come back positive SEMI-definite -- the fixture below sits at a condition number of
    # about 1e16, and whether its smallest eigenvalue lands above or below zero depends on
    # the LAPACK build. Unprocessed, that is a `PosDefException` out of the Cholesky on one
    # machine and a fit on another.
    rd = csfp_panel(; n_observations = 300).rd
    factors = csfp_factors()
    base = CrossSectionalFactorPrior(; factors = factors, lag = 1)
    # The default is the same estimator the asset block gets, and it is a no-op on a matrix
    # that is already positive definite, so an ordinary fit is untouched by its presence.
    @test isa(base.f_mp, MatrixProcessing)
    dt = CrossSectionalFactorPrior(; factors = factors, lag = 1,
                                   f_mp = MatrixProcessing(; dt = Detone()))
    pa = prior(base, rd)
    pb = prior(dt, rd)
    @test isa(pa, LowOrderPrior)
    @test isa(pb, LowOrderPrior)
    # The estimator is read, and it is read on the FACTOR block: a detoned `f_mp` moves the
    # factor distribution the fit forwards.
    @test !isapprox(pa.fpr.sigma, pb.fpr.sigma; nans = true)
    # The asset block moves with it, because the lift is handed the PROCESSED factor
    # covariance rather than the raw one. That is what puts the repair before the Cholesky.
    @test !isapprox(pa.sigma, pb.sigma; nans = true)
    # `mp` still owns the asset block alone: it is the estimator the idiosyncratic
    # covariance and the lifted asset covariance are processed under.
    @test isa(base.mp, MatrixProcessing)
end
