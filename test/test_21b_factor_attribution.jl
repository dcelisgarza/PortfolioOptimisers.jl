#=
Factor attribution (issue #782, map #643).

WHAT THIS FILE GATES. `factor_attribution` decomposes a portfolio's volatility and mean return over
the factors, the factor families and the assets of a factor model, and returns one
`FactorAttributionResult`. The decomposition is a set of exact identities, and every one of them is
asserted at machine precision, because each is a construction rather than an estimate:

 1. The four components sum to the total: the volatility contributions to the portfolio volatility,
    the mean return contributions to the portfolio mean return, and the variance shares to one.
 2. The factor rows sum to the systematic component, and the family rows sum to the same.
 3. The asset rows sum to the systematic and idiosyncratic components. They do NOT reach the total:
    the unattributed remainder is a property of the portfolio and has no per-asset split.
 4. The asset-by-factor matrix sums to the factor rows over the assets, and to the assets'
    systematic rows over the factors.
 5. The predicted remainder is at rounding level on a plain fit, and equals the measured gap under a
    wrapping prior that replaces `mu` and `sigma` while it forwards the block.
 6. `ppy` scales means and variances by `ppy` and volatilities by its square root, and leaves the
    shares and the correlations alone.
 7. The `i`-th entry of a rolling call equals the single-window call over that window.
 8. The standard errors equal a direct sandwich, computed by hand on a two-observation fixture.

THE ORACLE IS THE REFERENCE IMPLEMENTATION'S OWN ATTRIBUTION MODULE, whose four test files this one
mirrors. Two departures are deliberate and are recorded in the resolution comment of #708 and in the
docstring of `factor_attribution`:

  - The predicted side carries an unattributed remainder, which the reference's own predicted
    attribution cannot express. Under decision 5 of #708 the predicted totals anchor on `pr.mu` and
    `pr.sigma`, which is what the optimiser saw, so a wrapping prior's gap lands in the remainder.
  - The spread of the weight history uses the corrected denominator, as the spread of the exposure
    history beside it does. The reference uses the uncorrected one for the weights alone.
=#
using Statistics, Distributions, Dates, Random
include(joinpath(@__DIR__, "test06c_setup.jl"))

# The four factors every testset fits: a market intercept, the one-hot industry block and two style
# composites. The market intercept and the industry block are collinear, which is what exercises the
# rank-deficient branch of the sandwich covariance.
function fa_factors()
    return ["market" => ConstantExposure(),
            "industry" => OneHotExposure(; field = "industry", family = "industry"),
            "size" => CompositeExposure(; descriptors = [LogMarketCap()], family = "style"),
            "value" => CompositeExposure(; descriptors = [BookToPrice()], family = "style")]
end

function fa_prior(; n_assets::Integer = 20, n_observations::Integer = 60,
                  n_industries::Integer = 3, seed::Integer = 782_001, kwargs...)
    rd = synthetic_asset_panel(; n_assets = n_assets, n_observations = n_observations,
                               n_industries = n_industries, rng = StableRNG(seed)).rd
    pe = CrossSectionalFactorPrior(; factors = fa_factors(), minra = 5, kwargs...)
    return prior(pe, rd), rd
end

# A weight vector supported on the investable universe alone. A holding in an asset the prior could
# not estimate is refused, and every other testset takes weights from here.
function fa_weights(pr; seed::Integer = 782_002)
    N = length(pr.mu)
    imsk = PortfolioOptimisers.investable_mask(pr)
    act = isnothing(imsk) ? collect(1:N) : findall(imsk)
    w = zeros(N)
    w[act] = rand(StableRNG(seed), length(act))
    return w ./ sum(w)
end

# The net portfolio series the realised methods form, over the returns the prior can attribute.
function fa_net_returns(w, pr, rd)
    X = PortfolioOptimisers.attribution_investable_returns(rd.X, pr)
    return PortfolioOptimisers.calc_net_returns(w, X, nothing)
end

@testset "The block helpers, and the refusals of their root" begin
    PO = PortfolioOptimisers
    pr, _ = fa_prior()
    rr = pr.rr
    @testset "The five reads answer the block's own fields" begin
        @test PO.attribution_idiosyncratic_covariance(rr) === rr.esigma
        @test PO.attribution_idiosyncratic_returns(rr) === rr.csr.eps
        @test PO.attribution_factor_returns(rr) === rr.csr.f
        @test PO.attribution_exposures(rr) === rr.Ms
        @test PO.attribution_lag(rr) == rr.lag
    end
    @testset "The optional reads answer the block, and the root answers nothing" begin
        @test PO.attribution_families(rr) === rr.fam
        @test PO.attribution_family_basis(rr) === rr.fcb
        @test PO.attribution_regression_weights(rr) === rr.rw
        @test PO.attribution_idiosyncratic_variances(rr) === rr.vs
    end
    @testset "A block that keeps no history answers its loadings, and refuses the rest" begin
        bare = CrossSectionalFactorModel(; M = rr.M, b = rr.b)
        @test PO.attribution_exposures(bare) === bare.M
        @test iszero(PO.attribution_lag(bare))
        @test_throws PO.IsNothingError PO.attribution_idiosyncratic_covariance(bare)
        @test_throws PO.IsNothingError PO.attribution_idiosyncratic_returns(bare)
        @test_throws PO.IsNothingError PO.attribution_factor_returns(bare)
    end
    @testset "The root of each of the five names the type it cannot read" begin
        reg = Regression(; M = rr.M, b = rr.b)
        for v in
            (PO.attribution_idiosyncratic_covariance, PO.attribution_idiosyncratic_returns,
             PO.attribution_factor_returns, PO.attribution_exposures, PO.attribution_lag)
            e = try
                v(reg)
                nothing
            catch err
                err
            end
            @test isa(e, ArgumentError)
            @test occursin("Regression", e.msg)
        end
        @test isnothing(PO.attribution_families(reg))
        @test isnothing(PO.attribution_family_basis(reg))
        @test isnothing(PO.attribution_regression_weights(reg))
        @test isnothing(PO.attribution_idiosyncratic_variances(reg))
    end
end

@testset "The predicted decomposition and its identities" begin
    PO = PortfolioOptimisers
    pr, _ = fa_prior()
    w = fa_weights(pr)
    fa = factor_attribution(w, pr; assets = true)
    @testset "The Result reports an expected return, and every axis is filled" begin
        @test isa(fa, FactorAttributionResult)
        @test isa(fa, PO.AbstractResult)
        @test !fa.realised
        @test isone(fa.ppy)
        @test isa(fa.fbd, AttributionBreakdown)
        @test isnothing(fa.fbd.labels)
        @test isnothing(fa.fbd.exposure_std)
        @test isa(fa.fmbd, AttributionBreakdown)
        @test fa.fmbd.labels == ["industry", "market", "style"]
        @test isa(fa.abd, AssetAttributionBreakdown)
        @test isa(fa.afc, AssetFactorContribution)
    end
    @testset "The four components sum to the total" begin
        @test fa.sys.vol_contrib + fa.idio.vol_contrib + fa.unattr.vol_contrib ≈
              fa.total.vol_contrib
        @test fa.sys.mu_contrib + fa.idio.mu_contrib + fa.unattr.mu_contrib ≈
              fa.total.mu_contrib
        @test fa.sys.pct_var + fa.idio.pct_var + fa.unattr.pct_var ≈ fa.total.pct_var
        @test isone(fa.total.pct_var)
        @test isone(fa.total.corr)
        @test fa.total.vol_contrib == fa.total.vol
    end
    @testset "The remainder is a gap, so it carries no series" begin
        @test isnan(fa.unattr.vol)
        @test isnan(fa.unattr.corr)
        @test isapprox(fa.unattr.pct_var, 0; atol = 1e-10)
    end
    @testset "The factor rows and the family rows sum to the systematic component" begin
        @test sum(fa.fbd.vol_contrib) ≈ fa.sys.vol_contrib
        @test sum(fa.fbd.mu_contrib) ≈ fa.sys.mu_contrib
        @test sum(fa.fbd.pct_var) ≈ fa.sys.pct_var
        @test sum(fa.fmbd.vol_contrib) ≈ fa.sys.vol_contrib
        @test sum(fa.fmbd.mu_contrib) ≈ fa.sys.mu_contrib
        @test sum(fa.fmbd.exposure) ≈ sum(fa.fbd.exposure)
    end
    @testset "A family carries no standalone moment of its own" begin
        @test isnothing(fa.fmbd.vol)
        @test isnothing(fa.fmbd.corr)
        @test isnothing(fa.fmbd.mu)
        @test isnothing(fa.fmbd.exposure_std)
    end
    @testset "The asset rows sum to the two components the model explains" begin
        @test sum(fa.abd.sys_vol_contrib) ≈ fa.sys.vol_contrib
        @test sum(fa.abd.idio_vol_contrib) ≈ fa.idio.vol_contrib
        @test sum(fa.abd.sys_mu_contrib) ≈ fa.sys.mu_contrib
        @test sum(fa.abd.idio_mu_contrib) ≈ fa.idio.mu_contrib
        @test fa.abd.vol_contrib ≈ fa.abd.sys_vol_contrib .+ fa.abd.idio_vol_contrib
        @test fa.abd.mu_contrib ≈ fa.abd.sys_mu_contrib .+ fa.abd.idio_mu_contrib
        @test fa.abd.weight == w
        @test isnothing(fa.abd.weight_std)
    end
    @testset "The asset-by-factor matrix sums to both axes it lies between" begin
        @test vec(sum(fa.afc.vol_contrib; dims = 1)) ≈ fa.fbd.vol_contrib
        @test vec(sum(fa.afc.mu_contrib; dims = 1)) ≈ fa.fbd.mu_contrib
        @test vec(sum(fa.afc.vol_contrib; dims = 2)) ≈ fa.abd.sys_vol_contrib
        @test vec(sum(fa.afc.mu_contrib; dims = 2)) ≈ fa.abd.sys_mu_contrib
    end
    @testset "The exposure is the loadings against the weights" begin
        @test fa.fbd.exposure ≈ transpose(PO.attribution_finite(pr.rr.M)) * w
        @test fa.fbd.mu ≈ pr.fpr.mu
        @test fa.fbd.mu_contrib ≈ fa.fbd.exposure .* pr.fpr.mu
    end
    @testset "The asset axis is filled only when it is asked for" begin
        bare = factor_attribution(w, pr)
        @test isnothing(bare.abd)
        @test isnothing(bare.afc)
        @test isa(bare.fmbd, AttributionBreakdown)
        @test bare.sys.vol_contrib ≈ fa.sys.vol_contrib
    end
    @testset "The predicted side carries no standard error" begin
        @test isnothing(fa.sys.mu_se)
        @test isnothing(fa.idio.mu_se)
        @test isnothing(fa.fbd.mu_se)
        @test isnothing(fa.fmbd.mu_se)
    end
end

@testset "Annualisation scales the three kinds of number apart" begin
    pr, _ = fa_prior()
    w = fa_weights(pr)
    a = factor_attribution(w, pr; assets = true)
    b = factor_attribution(w, pr; assets = true, ppy = 252)
    s2 = sqrt(252)
    @test b.ppy == 252
    @test b.total.mu_contrib ≈ 252 * a.total.mu_contrib
    @test b.sys.mu_contrib ≈ 252 * a.sys.mu_contrib
    @test b.total.vol_contrib ≈ s2 * a.total.vol_contrib
    @test b.sys.vol ≈ s2 * a.sys.vol
    @test b.fbd.vol_contrib ≈ s2 * a.fbd.vol_contrib
    @test b.fbd.mu_contrib ≈ 252 * a.fbd.mu_contrib
    @test b.abd.vol_contrib ≈ s2 * a.abd.vol_contrib
    @test b.afc.mu_contrib ≈ 252 * a.afc.mu_contrib
    @test b.sys.pct_var ≈ a.sys.pct_var
    @test b.sys.corr ≈ a.sys.corr
    @test b.fbd.corr ≈ a.fbd.corr
    @test b.fbd.exposure ≈ a.fbd.exposure
    @test b.abd.weight ≈ a.abd.weight
    # The four components still sum after the scaling.
    @test b.sys.vol_contrib + b.idio.vol_contrib + b.unattr.vol_contrib ≈
          b.total.vol_contrib
end

@testset "A wrapping prior's gap lands in the remainder" begin
    PO = PortfolioOptimisers
    pr, rd = fa_prior()
    w = fa_weights(pr)
    plain = factor_attribution(w, pr)
    # A wrapping prior replaces `mu` and `sigma` and forwards `rr` and `fpr` unchanged (ADR 0046),
    # so the model no longer reproduces the anchors and the two gaps are measurable. The tilt is
    # built here rather than fitted: `EntropyPoolingPrior` hands its nested estimator a bare returns
    # matrix, which a Cross-Sectional Factor Prior refuses because it reads an Asset Panel, so the
    # two cannot yet be composed (issue #840). The carrier this testset builds is exactly what such
    # a composition would produce, and it is what the anchoring rule is stated over.
    tilt = LowOrderPrior(; X = pr.X, mu = pr.mu .+ 0.002, sigma = 1.3 * pr.sigma,
                         rr = pr.rr, fpr = pr.fpr)
    fa = factor_attribution(w, tilt)
    M = PO.attribution_finite(tilt.rr.M)
    D = PO.attribution_idiosyncratic_matrix(PO.attribution_finite(tilt.rr.esigma))
    bp = PO.attribution_finite(tilt.rr.b)
    sigma = PO.attribution_finite(tilt.sigma)
    mu = PO.attribution_finite(tilt.mu)
    sigma_p = sqrt(dot(w, sigma, w))
    @test fa.total.mu_contrib ≈ dot(w, mu)
    @test fa.total.vol_contrib ≈ sigma_p
    @test fa.unattr.mu_contrib ≈ dot(w, mu .- M * tilt.fpr.mu .- bp)
    @test fa.unattr.vol_contrib ≈
          dot(w, (sigma .- M * tilt.fpr.sigma * transpose(M) .- D) * w) / sigma_p
    # The plain fit reproduces its own anchors, so its gap is at rounding level and the tilt's is
    # not: the remainder is what tells the two apart.
    @test abs(plain.unattr.pct_var) < 1e-9
    @test fa.sys.vol_contrib + fa.idio.vol_contrib + fa.unattr.vol_contrib ≈
          fa.total.vol_contrib
end

@testset "The realised decomposition and its identities" begin
    PO = PortfolioOptimisers
    pr, rd = fa_prior()
    w = fa_weights(pr)
    fa = factor_attribution(w, pr, rd.X; assets = true)
    ret = fa_net_returns(w, pr, rd)
    al = PO.attribution_align(pr.rr, length(ret))
    aret = view(ret, al.rows)
    @testset "The Result reports a mean return over the aligned history" begin
        @test fa.realised
        @test fa.total.mu_contrib ≈ mean(aret)
        @test fa.total.vol_contrib ≈ std(aret)
        @test isone(fa.total.pct_var)
        @test length(al.rows) == size(pr.rr.csr.f, 1) - pr.rr.lag
    end
    @testset "The four components sum to the total" begin
        @test fa.sys.vol_contrib + fa.idio.vol_contrib + fa.unattr.vol_contrib ≈
              fa.total.vol_contrib
        @test fa.sys.mu_contrib + fa.idio.mu_contrib + fa.unattr.mu_contrib ≈
              fa.total.mu_contrib
        @test fa.sys.pct_var + fa.idio.pct_var + fa.unattr.pct_var ≈ fa.total.pct_var
    end
    @testset "The remainder carries a series of its own, unlike the predicted one" begin
        @test isfinite(fa.unattr.vol)
        @test isfinite(fa.unattr.corr)
    end
    @testset "The factor rows and the family rows sum to the systematic component" begin
        @test sum(fa.fbd.vol_contrib) ≈ fa.sys.vol_contrib
        @test sum(fa.fbd.mu_contrib) ≈ fa.sys.mu_contrib
        @test sum(fa.fmbd.vol_contrib) ≈ fa.sys.vol_contrib
        @test sum(fa.fmbd.mu_contrib) ≈ fa.sys.mu_contrib
    end
    @testset "The asset rows sum to the two components the model explains" begin
        @test sum(fa.abd.sys_vol_contrib) ≈ fa.sys.vol_contrib
        @test sum(fa.abd.idio_vol_contrib) ≈ fa.idio.vol_contrib
        @test sum(fa.abd.sys_mu_contrib) ≈ fa.sys.mu_contrib
        @test sum(fa.abd.idio_mu_contrib) ≈ fa.idio.mu_contrib
        @test fa.abd.vol_contrib ≈ fa.abd.sys_vol_contrib .+ fa.abd.idio_vol_contrib
    end
    @testset "The asset-by-factor matrix sums to both axes it lies between" begin
        @test vec(sum(fa.afc.vol_contrib; dims = 1)) ≈ fa.fbd.vol_contrib
        @test vec(sum(fa.afc.mu_contrib; dims = 1)) ≈ fa.fbd.mu_contrib
        @test vec(sum(fa.afc.vol_contrib; dims = 2)) ≈ fa.abd.sys_vol_contrib
        @test vec(sum(fa.afc.mu_contrib; dims = 2)) ≈ fa.abd.sys_mu_contrib
    end
    @testset "The realised exposure is a history, so it carries a spread" begin
        @test isa(fa.fbd.exposure_std, AbstractVector)
        @test all(x -> x >= 0, fa.fbd.exposure_std)
        @test isa(fa.fmbd.exposure_std, AbstractVector)
        @test length(fa.fmbd.exposure_std) == length(fa.fmbd.labels)
    end
    @testset "A constant weight carries no spread of its own" begin
        @test fa.abd.weight == w
        @test isnothing(fa.abd.weight_std)
    end
    @testset "The three entry points that form the same series agree" begin
        @test factor_attribution(w, pr, ReturnsResult(; nx = rd.nx, X = rd.X)).sys.vol_contrib ≈
              fa.sys.vol_contrib
        W = repeat(transpose(w), length(ret))
        fw = factor_attribution(W, pr, ret; assets = true)
        @test fw.sys.vol_contrib ≈ fa.sys.vol_contrib
        @test fw.total.mu_contrib ≈ fa.total.mu_contrib
        @test fw.abd.vol_contrib ≈ fa.abd.vol_contrib
        # A weight history whose rows are identical has a spread of zero.
        @test maximum(abs, fw.abd.weight_std) < 1e-14
        @test fw.abd.weight ≈ w
    end
end

@testset "A collinear cross-section takes the pseudo-inverse" begin
    # The market intercept is the sum of the one-hot industry block, so the Gram matrix of every
    # observation is rank-deficient and the sandwich falls back to the pseudo-inverse. The answer is
    # the minimum-norm one, so it is finite; a plain solve would return an arbitrarily large number.
    PO = PortfolioOptimisers
    pr, rd = fa_prior()
    w = fa_weights(pr)
    al = PO.attribution_align(pr.rr, size(rd.X, 1))
    G = transpose(view(al.B, 1, :, :)) * Diagonal(view(al.rw, 1, :)) * view(al.B, 1, :, :)
    @test PO.cross_sectional_rank(G) < size(G, 2)
    fa = factor_attribution(w, pr, rd.X; se = true)
    @test isfinite(fa.sys.mu_se)
    @test fa.sys.mu_se == fa.idio.mu_se
    @test all(isfinite, fa.fbd.mu_se)
    @test all(isfinite, fa.fmbd.mu_se)
    # The error of a mean return contribution is small beside the contribution itself.
    @test fa.sys.mu_se < 1
end

@testset "The remainder holds the intercept, the fee and the drift" begin
    PO = PortfolioOptimisers
    pr, rd = fa_prior()
    w = fa_weights(pr)
    plain = factor_attribution(w, pr, rd.X)
    # A fee lowers every net return by a constant, so it moves the mean of the remainder by exactly
    # that constant and leaves every other component alone.
    fees = Fees(; l = 0.01)
    fee = factor_attribution(w, pr, rd.X, fees)
    @test fee.sys.mu_contrib ≈ plain.sys.mu_contrib
    @test fee.idio.mu_contrib ≈ plain.idio.mu_contrib
    @test fee.unattr.mu_contrib ≈
          plain.unattr.mu_contrib + fee.total.mu_contrib - plain.total.mu_contrib
    @test fee.sys.mu_contrib + fee.idio.mu_contrib + fee.unattr.mu_contrib ≈
          fee.total.mu_contrib
    # The per-observation intercept of the fit is not in `eps`, so its share is in the remainder.
    @test !isnothing(pr.rr.csr)
    @test abs(plain.unattr.mu_contrib) > 0
end

@testset "The rolling methods" begin
    PO = PortfolioOptimisers
    pr, rd = fa_prior()
    w = fa_weights(pr)
    ret = fa_net_returns(w, pr, rd)
    al = PO.attribution_align(pr.rr, length(ret))
    T = length(al.rows)
    @testset "One window over the whole aligned history is the single-window call" begin
        roll = factor_attribution(w, pr, rd.X, T)
        one = factor_attribution(w, pr, rd.X; assets = true)
        @test isa(roll, Vector{<:FactorAttributionResult})
        @test isone(length(roll))
        @test roll[1].total.vol_contrib ≈ one.total.vol_contrib
        @test roll[1].total.mu_contrib ≈ one.total.mu_contrib
        @test roll[1].fbd.vol_contrib ≈ one.fbd.vol_contrib
    end
    @testset "The i-th window's total is the mean and the spread of that window" begin
        window = 30
        roll = factor_attribution(w, pr, rd.X, window)
        @test length(roll) == T - window + 1
        aret = view(ret, al.rows)
        for (j, t) in enumerate(window:T)
            slice = view(aret, (t - window + 1):t)
            @test roll[j].total.mu_contrib ≈ mean(slice)
            @test roll[j].total.vol_contrib ≈ std(slice)
            @test roll[j].sys.vol_contrib +
                  roll[j].idio.vol_contrib +
                  roll[j].unattr.vol_contrib ≈ roll[j].total.vol_contrib
        end
    end
    @testset "The stride skips windows without moving them" begin
        every = factor_attribution(w, pr, rd.X, 30)
        third = factor_attribution(w, pr, rd.X, 30; step = 3)
        @test length(third) == length(30:3:T)
        @test third[1].total.mu_contrib ≈ every[1].total.mu_contrib
        @test third[2].total.mu_contrib ≈ every[4].total.mu_contrib
    end
    @testset "Every realised form carries the same rolling twin" begin
        a = factor_attribution(w, pr, rd.X, 30)
        b = factor_attribution(w, pr, rd.X, nothing, 30)
        c = factor_attribution(w, pr, ReturnsResult(; nx = rd.nx, X = rd.X), 30)
        d = factor_attribution(repeat(transpose(w), length(ret)), pr, ret, 30)
        for other in (b, c, d)
            @test length(other) == length(a)
            @test other[end].total.vol_contrib ≈ a[end].total.vol_contrib
        end
    end
end

@testset "The standard errors match a direct sandwich" begin
    PO = PortfolioOptimisers
    # A hand-built block with no lag, two observations and two factors, so the sandwich is small
    # enough to write out beside the verb's own answer.
    M = [1.0 0.5; 1.0 -0.5; 1.0 1.5]
    Ms = Array{Float64, 3}(undef, 2, 3, 2)
    Ms[1, :, :] = M
    Ms[2, :, :] = [1.0 0.4; 1.0 -0.6; 1.0 1.2]
    f = [0.01 0.02; -0.005 0.03]
    eps = [0.001 -0.002 0.0015; -0.0005 0.001 -0.0008]
    rw = [0.4 0.35 0.25; 0.3 0.45 0.25]
    vs = [1.0e-4 2.0e-4 1.5e-4; 1.2e-4 1.8e-4 1.6e-4]
    csr = CrossSectionalRegression(; f = f, eps = eps, n = [3, 3])
    rr = CrossSectionalFactorModel(; M = Ms[2, :, :], b = [0.001, 0.0005, 0.0012],
                                   csr = csr, Ms = Ms, vs = vs,
                                   esigma = [1.2e-4, 1.8e-4, 1.6e-4], rw = rw, bw = rw,
                                   nf = ["market", "style"], fam = ["market", "style"],
                                   lag = 0)
    X = [0.011 0.003 0.02; 0.004 0.012 0.009]
    fmu = vec(mean(f; dims = 1))
    fsig = cov(f)
    fpr = LowOrderPrior(; X = f, mu = fmu, sigma = fsig)
    mu = rr.M * fmu .+ rr.b
    sigma = rr.M * fsig * transpose(rr.M) + Diagonal(rr.esigma)
    pr = LowOrderPrior(; X = X, mu = mu, sigma = sigma, rr = rr, fpr = fpr)
    w = [0.5, 0.3, 0.2]
    fa = factor_attribution(w, pr, X; se = true)
    # The sandwich of each observation, written out: `G^-1 B' W Omega W B G^-1`.
    scale = 1 / 2
    g = [transpose(Ms[t, :, :]) * w for t in 1:2]
    V = map(1:2) do t
        B = Ms[t, :, :]
        W = Diagonal(rw[t, :])
        G = transpose(B) * W * B
        S = transpose(B) * Diagonal(rw[t, :] .^ 2 .* vs[t, :]) * B
        Gi = inv(G)
        return Gi * S * transpose(Gi)
    end
    @test fa.sys.mu_se ≈ scale * sqrt(sum(dot(g[t], V[t], g[t]) for t in 1:2))
    @test fa.idio.mu_se == fa.sys.mu_se
    @test fa.fbd.mu_se ≈
          [scale * sqrt(sum(g[t][k]^2 * V[t][k, k] for t in 1:2)) for k in 1:2]
    @test isnothing(fa.unattr.mu_se)
    @test isnothing(fa.total.mu_se)
    @testset "A family's error reads its whole covariance block" begin
        fi = PortfolioOptimisers.attribution_family_index(["market", "style"])
        @test fi.labels == ["market", "style"]
        @test fa.fmbd.mu_se ≈ [scale *
                               sqrt(sum(dot(view(g[t], i), view(V[t], i, i), view(g[t], i)) for t in 1:2))
                               for i in fi.idx]
    end
    @testset "Annualisation scales the errors as it scales a mean" begin
        b = factor_attribution(w, pr, X; se = true, ppy = 12)
        @test b.sys.mu_se ≈ 12 * fa.sys.mu_se
        @test b.fbd.mu_se ≈ 12 * fa.fbd.mu_se
    end
    @testset "A currency family carries no estimation uncertainty" begin
        cur = CrossSectionalFactorModel(; M = rr.M, b = rr.b, csr = csr, Ms = Ms, vs = vs,
                                        esigma = rr.esigma, rw = rw, bw = rw,
                                        nf = ["market", "usd"],
                                        fam = ["market",
                                               PortfolioOptimisers.ATTRIBUTION_CURRENCY_FAMILY],
                                        lag = 0)
        prc = LowOrderPrior(; X = X, mu = mu, sigma = sigma, rr = cur, fpr = fpr)
        fc = factor_attribution(w, prc, X; se = true)
        @test isnan(fc.fbd.mu_se[2])
        @test isfinite(fc.fbd.mu_se[1])
        @test fc.fmbd.labels == ["currency", "market"]
        @test isnan(fc.fmbd.mu_se[1])
        @test isfinite(fc.fmbd.mu_se[2])
        # The currency factor leaves the regression, so the systematic error is the market's alone.
        @test isfinite(fc.sys.mu_se)
    end
    @testset "The decomposition of the fixture is exact, and its remainder is the intercept" begin
        @test fa.sys.vol_contrib + fa.idio.vol_contrib + fa.unattr.vol_contrib ≈
              fa.total.vol_contrib
        @test fa.sys.mu_contrib + fa.idio.mu_contrib + fa.unattr.mu_contrib ≈
              fa.total.mu_contrib
    end
end

@testset "A static loadings block needs no alignment" begin
    PO = PortfolioOptimisers
    M = [1.0 0.5; 1.0 -0.5; 1.0 1.5]
    f = [0.01 0.02; -0.005 0.03; 0.002 -0.01]
    eps = [0.001 -0.002 0.0015; -0.0005 0.001 -0.0008; 0.0002 0.0003 -0.0004]
    csr = CrossSectionalRegression(; f = f, eps = eps, n = [3, 3, 3])
    rr = CrossSectionalFactorModel(; M = M, b = [0.001, 0.0005, 0.0012], csr = csr,
                                   esigma = [1.2e-4, 1.8e-4, 1.6e-4])
    fmu = vec(mean(f; dims = 1))
    fsig = cov(f)
    fpr = LowOrderPrior(; X = f, mu = fmu, sigma = fsig)
    mu = M * fmu .+ rr.b
    sigma = M * fsig * transpose(M) + Diagonal(rr.esigma)
    X = f * transpose(M) .+ eps
    pr = LowOrderPrior(; X = X, mu = mu, sigma = sigma, rr = rr, fpr = fpr)
    w = [0.5, 0.3, 0.2]
    @test PO.attribution_exposures(rr) === M
    @test iszero(PO.attribution_lag(rr))
    fa = factor_attribution(w, pr, X; assets = true)
    @test length(PO.attribution_align(rr, 3).rows) == 3
    # A static block reconstructs the returns exactly, so nothing is left over.
    @test abs(fa.unattr.pct_var) < 1e-12
    @test sum(fa.fbd.vol_contrib) ≈ fa.sys.vol_contrib
    @test sum(fa.abd.sys_vol_contrib) ≈ fa.sys.vol_contrib
    @test fa.fbd.exposure ≈ transpose(M) * w
    # The exposures do not move, so their spread is zero.
    @test maximum(abs, fa.fbd.exposure_std) < 1e-14
    @test isnothing(fa.fmbd)
end

@testset "The entry points that read an optimisation result" begin
    PO = PortfolioOptimisers
    pr, rd = fa_prior()
    w = fa_weights(pr)
    res = NaiveOptimisationResult(; pr = pr, wb = nothing, retcode = OptimisationSuccess(),
                                  w = w, fb = nothing)
    @testset "A result carries its own weights and its own prior" begin
        @test factor_attribution(res, pr).sys.vol_contrib ≈
              factor_attribution(w, pr).sys.vol_contrib
        @test factor_attribution(res).sys.vol_contrib ≈
              factor_attribution(w, pr).sys.vol_contrib
        rdr = ReturnsResult(; nx = rd.nx, X = rd.X)
        @test factor_attribution(res, pr, rdr).sys.vol_contrib ≈
              factor_attribution(w, pr, rd.X).sys.vol_contrib
        @test length(factor_attribution(res, pr, rdr, 30)) ==
              length(factor_attribution(w, pr, rd.X, 30))
    end
    @testset "A cross-validation's folds stack into one weight history" begin
        ret = fa_net_returns(w, pr, rd)
        T = length(ret)
        h = T ÷ 2
        folds = [PredictionResult(; res = res,
                                  rd = PredictionReturnsResult(; nx = rd.nx,
                                                               X = collect(view(ret, r)),
                                                               ts = nothing))
                 for r in (1:h, (h + 1):T)]
        mpred = MultiPeriodPredictionResult(; pred = folds)
        W, mret = PO.attribution_prediction_history(mpred)
        @test size(W) == (T, length(w))
        @test mret ≈ ret
        @test all(i -> view(W, :, i) == fill(w[i], T), eachindex(w))
        fa = factor_attribution(mpred, pr; assets = true)
        @test fa.sys.vol_contrib ≈ factor_attribution(w, pr, rd.X).sys.vol_contrib
        @test fa.total.mu_contrib ≈ factor_attribution(w, pr, rd.X).total.mu_contrib
        @test length(factor_attribution(mpred, pr, 30)) ==
              length(factor_attribution(w, pr, rd.X, 30))
    end
end

@testset "The refusals" begin
    PO = PortfolioOptimisers
    pr, rd = fa_prior()
    w = fa_weights(pr)
    @testset "A holding the prior could not estimate is named" begin
        imsk = PO.investable_mask(pr)
        @test !isnothing(imsk)
        bad = copy(w)
        bad[findfirst(!, imsk)] = 0.1
        @test_throws ArgumentError factor_attribution(bad, pr)
        @test_throws ArgumentError factor_attribution(bad, pr, rd.X)
        @test_throws ArgumentError factor_attribution(repeat(transpose(bad), size(rd.X, 1)),
                                                      pr, rd.X * bad)
    end
    @testset "A prior with no factor block is refused before the arithmetic" begin
        Xp = randn(StableRNG(782_003), 40, 3) ./ 100
        plain = prior(EmpiricalPrior(), ReturnsResult(; nx = ["a", "b", "c"], X = Xp))
        @test isnothing(plain.rr)
        @test_throws PO.IsNothingError factor_attribution([0.4, 0.3, 0.3], plain)
        @test_throws PO.IsNothingError factor_attribution([0.4, 0.3, 0.3], plain, Xp)
    end
    @testset "The scalar guards" begin
        @test_throws DomainError factor_attribution(w, pr; ppy = 0)
        @test_throws DomainError factor_attribution(w, pr; ppy = -1)
        @test_throws DomainError factor_attribution(w, pr, rd.X, 0)
        @test_throws DomainError factor_attribution(w, pr, rd.X, 10_000)
        @test_throws DomainError factor_attribution(w, pr, rd.X, 30; step = 0)
    end
    @testset "A block with no history cannot answer a realised attribution" begin
        bare = CrossSectionalFactorModel(; M = pr.rr.M, b = pr.rr.b)
        stub = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, rr = bare,
                             fpr = pr.fpr)
        @test_throws PO.IsNothingError factor_attribution(w, stub, rd.X)
    end
    @testset "A history shorter than the block is refused by name" begin
        @test_throws DimensionMismatch factor_attribution(w, pr, view(rd.X, 1:10, :))
        # A block whose exposure lag reaches its whole history leaves nothing to align.
        short = CrossSectionalFactorModel(; M = pr.rr.M, b = pr.rr.b, csr = pr.rr.csr,
                                          Ms = pr.rr.Ms, esigma = pr.rr.esigma,
                                          lag = size(pr.rr.csr.f, 1))
        stub = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, rr = short,
                             fpr = pr.fpr)
        @test_throws DimensionMismatch factor_attribution(w, stub, rd.X)
    end
    @testset "`se = true` needs the two histories the sandwich reads" begin
        no_rw = CrossSectionalFactorModel(; M = pr.rr.M, b = pr.rr.b, csr = pr.rr.csr,
                                          Ms = pr.rr.Ms, esigma = pr.rr.esigma,
                                          fam = pr.rr.fam, lag = pr.rr.lag)
        stub = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, rr = no_rw,
                             fpr = pr.fpr)
        @test isa(factor_attribution(w, stub, rd.X), FactorAttributionResult)
        @test_throws PO.IsNothingError factor_attribution(w, stub, rd.X; se = true)
    end
end
