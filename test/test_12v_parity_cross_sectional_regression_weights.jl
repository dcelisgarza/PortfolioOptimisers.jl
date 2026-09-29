#=
Parity of the cross-sectional regression and its weight policies (#1382, map #1375).

Each stored case is the output of the oracle on the inputs that the fixtures below build, through
the CSV exchange of `parity_harness.jl`. The regression is measured alone on four designs: a dense
one, one that is rank deficient on every observation, one that is exactly collinear, and one that
is nearly singular. The weight policies are measured alone on a residual matrix with the masks of
a panel, and inside the Cross-Sectional Factor Prior on `parity_small_panel()`.

Two verdicts are Better. On an exactly collinear design the oracle solves the normal equations, and
its solve does not fail on a normal matrix that is singular in exact arithmetic, so its factor
returns carry a component along the null direction that round-off chooses. We give the
minimum-norm solution. On a nearly singular design the normal equations square the condition
number: against the exact least-squares answer, ours is within 8.8e-11 and the oracle's is within
7.6e-4. A zero idiosyncratic variance is Better too: the oracle refuses its infinite inverse, and
ours equals the oracle's own answer at a variance that is tiny but not zero. A missing market
capitalisation is Better as well: the oracle refuses the whole fit, and the prior gives the pair a
weight of zero, which keeps each cross-section a weighted least squares over the pairs whose
weights are known.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# A dense design: a market column of ones and `K - 1` Gaussian styles. About one pair in ten
# carries a zero weight, and half of those carry a missing return or a missing exposure, which a
# pair of zero weight may carry. `Z2` is a second exposure tensor on another asset axis.
function parity_cs_dense(; T::Integer = 40, N::Integer = 30, K::Integer = 4,
                         seed::Integer = 1382)
    rng = StableRNG(seed)
    Z = randn(rng, T, N, K)
    Z[:, :, 1] .= 1.0
    f = 0.01 .* randn(rng, T, K)
    X = [LinearAlgebra.dot(view(Z, t, i, :), view(f, t, :)) + 0.02 * randn(rng)
         for t in 1:T, i in 1:N]
    mcap = exp.(randn(rng, T, N) .+ 3.0)
    W = sqrt.(mcap)
    W[rand(rng, T, N) .< 0.1] .= 0.0
    for t in 1:T, i in 1:N
        if iszero(W[t, i])
            u = rand(rng)
            if u < 0.25
                X[t, i] = NaN
            elseif u < 0.5
                Z[t, i, 2] = NaN
            end
        end
    end
    Z2 = randn(rng, T, 7, K)
    Z2[:, :, 1] .= 1.0
    return (; Z, X, W, Z2)
end
# A design that is rank deficient on every observation: a market column beside a one-hot block of
# three industries, whose columns sum to the market column, and one style. Observation 3 carries no
# positive weight, and observation 5 carries three, fewer than the five factors.
function parity_cs_deficient(; T::Integer = 12, N::Integer = 20, seed::Integer = 1383)
    rng = StableRNG(seed)
    ind = [mod1(i, 3) for i in 1:N]
    Z = zeros(T, N, 5)
    Z[:, :, 1] .= 1.0
    for i in 1:N
        Z[:, i, 1 + ind[i]] .= 1.0
    end
    Z[:, :, 5] = randn(rng, T, N)
    X = 0.01 .* randn(rng, T) .+ 0.005 .* randn(rng, T, N) .+ 0.003 .* Z[:, :, 5]
    W = exp.(0.5 .* randn(rng, T, N))
    W[3, :] .= 0.0
    W[5, 4:end] .= 0.0
    return (; Z, X, W)
end
# The same collinear design with a positive weight on every pair, so no observation is empty and
# no observation has fewer assets than factors.
function parity_cs_collinear()
    q = parity_cs_deficient(; seed = 1386)
    return (; Z = q.Z, X = q.X, W = exp.(0.5 .* randn(StableRNG(1387), size(q.X))))
end
# A design that is nearly singular: the third column is the second one plus `delta` times noise.
function parity_cs_near_singular(; T::Integer = 6, N::Integer = 30, delta::Real = 1e-6,
                                 seed::Integer = 1384)
    rng = StableRNG(seed)
    Z = zeros(T, N, 3)
    Z[:, :, 1] .= 1.0
    Z[:, :, 2] = randn(rng, T, N)
    Z[:, :, 3] = Z[:, :, 2] .+ delta .* randn(rng, T, N)
    X = 0.01 .* randn(rng, T) .+ 0.02 .* Z[:, :, 2] .+ 0.005 .* randn(rng, T, N)
    W = exp.(0.5 .* randn(rng, T, N))
    return (; Z, X, W)
end
# The exact weighted least-squares factor returns of each observation, in 256-bit arithmetic.
function parity_cs_exact(Z, X, W)
    return setprecision(BigFloat, 256) do
        F = zeros(BigFloat, size(X, 1), size(Z, 3))
        for t in axes(X, 1)
            idx = findall(>(0), view(W, t, :))
            w = big.(W[t, idx])
            A = big.(Z[t, idx, :])
            y = big.(X[t, idx])
            F[t, :] = (transpose(A) * (w .* A)) \ (transpose(A) * (w .* y))
        end
        return Float64.(F)
    end
end
# First-pass residuals with a volatility for each asset, and the masks of a panel: asset 3 lists at
# observation 11, asset 4 is inactive on 30:40, asset 6 is outside the estimation universe on 1:20,
# and about one active pair in twenty is not eligible. A residual is `NaN` where the asset is
# inactive. Under `zero = true`, asset 2 has a zero residual on every observation, so its
# idiosyncratic variance is zero.
function parity_cs_weights_fixture(; T::Integer = 60, N::Integer = 25, seed::Integer = 1385,
                                   zero::Bool = false)
    rng = StableRNG(seed)
    vol = range(0.005, 0.05; length = N)
    eps = randn(rng, T, N) .* transpose(vol)
    amsk = trues(T, N)
    amsk[1:10, 3] .= false
    amsk[30:40, 4] .= false
    emsk = copy(amsk)
    emsk[1:20, 6] .= false
    msk = emsk .& (rand(rng, T, N) .>= 0.05)
    eps[.!amsk] .= NaN
    if zero
        eps[:, 2] .= 0.0
    end
    mcap = exp.(randn(rng, T, N) .+ 3.0)
    return (; eps, mcap, msk, emsk, amsk)
end
# The zero-variance fixture with a residual of asset 2 that is tiny but not zero.
function parity_cs_weights_tiny()
    fx = parity_cs_weights_fixture(; zero = true)
    eps = copy(fx.eps)
    eps[:, 2] = 1e-100 .* randn(StableRNG(1388), size(eps, 1))
    return merge(fx, (; eps))
end
# The weight cases: tag => (p, lambda, ratio, wins). A `lambda` of zero is `MarketCapWeights`.
const PARITY_CS_WEIGHT_CASES = ["Cap00" => (0.0, 0.0, 20.0, (0.025, 0.975)),
                                "Cap05" => (0.5, 0.0, 20.0, (0.025, 0.975)),
                                "Cap10" => (1.0, 0.0, 20.0, (0.025, 0.975)),
                                "Cap20" => (2.0, 0.0, 20.0, (0.025, 0.975)),
                                "Blend05" => (0.5, 0.5, 20.0, (0.025, 0.975)),
                                "Blend025Ratio3" => (0.5, 0.25, 3.0, (0.025, 0.975)),
                                "Blend10Ratio3Wins10" => (0.5, 1.0, 3.0, (0.1, 0.9)),
                                "Blend075Cap10Wins05" => (1.0, 0.75, 20.0, (0.05, 0.95)),
                                "Blend05Cap00" => (0.0, 0.5, 20.0, (0.025, 0.975))]
# The variance of the weights clips its regime multiplier to (0.7, 1.6), as the oracle does by
# default. That default is #1383's to measure, and this file measures the weights.
function parity_cs_weights(fx, (p, lambda, ratio, wins);
                           ve = RegimeAdjustedExpWeightedVariance(; centred = true,
                                                                  regime_lohi_mult = (0.7,
                                                                                      1.6)))
    alg = if iszero(lambda)
        MarketCapWeights(; p = p)
    else
        BlendedInverseVarianceWeights(; p = p, lambda = lambda, ratio = ratio, wins = wins)
    end
    W0 = PortfolioOptimisers.cs_weights_initial(alg, iszero(p) ? nothing : fx.mcap, fx.msk)
    W1 = if PortfolioOptimisers.needs_second_pass(alg)
        PortfolioOptimisers.cs_weights_refine(alg, W0, fx.eps, ve, fx.msk;
                                              estimation_mask = fx.emsk,
                                              active_mask = fx.amsk)
    else
        W0
    end
    return (; alg, W0, W1)
end
# The writers of the exchange that the oracle side reads. Each factor is one file.
function parity_cs_write(dir::AbstractString, Z, X, W; Z2 = nothing)
    mkpath(dir)
    for k in axes(Z, 3)
        parity_write_rows(joinpath(dir, "Z_$(k).csv"), Z[:, :, k])
    end
    parity_write_rows(joinpath(dir, "X.csv"), X)
    parity_write_rows(joinpath(dir, "W.csv"), W)
    if !isnothing(Z2)
        for k in axes(Z2, 3)
            parity_write_rows(joinpath(dir, "Z2_$(k).csv"), Z2[:, :, k])
        end
    end
    return dir
end
function parity_cs_weights_write(dir::AbstractString, fx, cases)
    mkpath(dir)
    parity_write_rows(joinpath(dir, "eps.csv"), fx.eps)
    parity_write_rows(joinpath(dir, "mcap.csv"), fx.mcap)
    parity_write_rows(joinpath(dir, "mask.csv"), Int.(fx.msk))
    parity_write_rows(joinpath(dir, "emask.csv"), Int.(fx.emsk))
    parity_write_rows(joinpath(dir, "amask.csv"), Int.(fx.amsk))
    open(joinpath(dir, "cases.csv"), "w") do io
        for (tag, (p, lambda, ratio, wins)) in cases
            println(io, join((tag, p, lambda, ratio, wins[1], wins[2]), ','))
        end
    end
    return dir
end

@testset "Parity: the cross-sectional regression and its weights (#1382)" begin
    lin(; kwargs...) = CrossSectionalLinearRegression(; kwargs...)
    tgt(; kwargs...) = CrossSectionalTargetRegression(; kwargs...)
    loadl(c, o) = parity_load("CrossSectionalLinearRegression", c, o)
    loadt(c, o) = parity_load("CrossSectionalTargetRegression", c, o)

    @testset "A dense design, at parity" begin
        # `Dense` fits `parity_cs_dense()` with the default linear member. `DenseIntercept` drops
        # the market column and fits an intercept instead, with the linear member and with the
        # default `LinearModel` target. The oracle's external-regressor member wraps its own
        # least-squares regressor, with and without an intercept.
        d = parity_cs_dense()
        csr = cross_sectional_regression(lin(), d.Z, d.X, d.W)
        # Measured maxrel 7.9e-14.
        @test parity_compare(csr.f, loadl("Dense", "Coef"); name = "Dense f").ok
        # Measured maxrel 3.2e-13, on a small residual. A pair of zero weight whose return or
        # exposure is missing has a missing residual on both sides.
        @test parity_compare(csr.eps, loadl("Dense", "Eps"); name = "Dense eps").ok
        # The oracle's count of valid assets is the count of positive weights.
        @test csr.n == vec(count(>(0), d.W; dims = 2))
        @test isnothing(csr.b)
        # The weighted coefficient of determination of each observation, and its mean. The
        # oracle's per-observation value is its own score of a one-observation model. Measured
        # maxrel 1.8e-15 and 2.3e-16.
        @test parity_compare(cross_sectional_r2(csr, d.Z, d.X, d.W),
                             vec(loadl("Dense", "R2")); name = "Dense r2").ok
        @test parity_compare([mean_cross_sectional_r2(csr, d.Z, d.X, d.W)],
                             vec(loadl("Dense", "Score")); name = "Dense score").ok
        # A prediction on another asset axis. Measured maxrel 1.2e-13.
        @test parity_compare(predict(csr, d.Z2), loadl("Dense", "Predict2");
                             name = "Dense predict").ok

        Zi = d.Z[:, :, 2:end]
        csi = cross_sectional_regression(lin(; intercept = true), Zi, d.X, d.W)
        # Measured maxrel 2.2e-14 and 1.1e-14.
        @test parity_compare(csi.f, loadl("DenseIntercept", "Coef"); name = "Intercept f").ok
        @test parity_compare(csi.b, vec(loadl("DenseIntercept", "Intercept"));
                             name = "Intercept b").ok

        # The target member solves through the default `LinearModel`. Measured maxrel 1.6e-13,
        # 2.4e-14 and 3.4e-15.
        cst = cross_sectional_regression(tgt(), d.Z, d.X, d.W)
        @test parity_compare(cst.f, loadt("Dense", "Coef"); name = "Target f").ok
        csti = cross_sectional_regression(tgt(; intercept = true), Zi, d.X, d.W)
        @test parity_compare(csti.f, loadt("DenseIntercept", "Coef");
                             name = "Target intercept f").ok
        @test parity_compare(csti.b, vec(loadt("DenseIntercept", "Intercept"));
                             name = "Target intercept b").ok
    end

    @testset "A rank-deficient design, at parity" begin
        # `Deficient` fits `parity_cs_deficient()` with the default linear member. Its empty
        # observation makes the oracle's batched solve fail, so the oracle pseudo-inverts every
        # normal matrix, which gives the minimum-norm solution that we give.
        q = parity_cs_deficient()
        csr = cross_sectional_regression(lin(), q.Z, q.X, q.W)
        # Measured maxrel 3.2e-14. The empty observation has zero factor returns on both sides.
        @test parity_compare(csr.f, loadl("Deficient", "Coef"); name = "Deficient f").ok
        @test iszero(csr.f[3, :]) && csr.n[3] == 0 && csr.n[5] == 3
        # Observation 5 fits its three assets exactly, so its residuals are round-off on both
        # sides: maxrel 1.5 on those cells, and maxscaled 2.4e-15 against the largest residual.
        @test parity_compare(csr.eps, loadl("Deficient", "Eps"); scale = :array,
                             name = "Deficient eps").ok
        # The empty observation has no ratio on either side. Measured maxrel 7.0e-16.
        r2 = cross_sectional_r2(csr, q.Z, q.X, q.W)
        @test parity_compare(r2, vec(loadl("Deficient", "R2")); name = "Deficient r2").ok
        @test isnan(r2[3])
        # The oracle's external-regressor member refuses the empty observation, and so does ours.
        @test_throws ArgumentError cross_sectional_regression(tgt(), q.Z, q.X, q.W)
    end

    @testset "An exactly collinear design: Better" begin
        # `Collinear` fits `parity_cs_collinear()` with the default linear member. Every
        # observation has rank 4 over 5 factors and no observation is empty, so the oracle's
        # solve of the normal equations does not fail, and its answer is one point of the
        # solution set that round-off chooses.
        c = parity_cs_collinear()
        csr = cross_sectional_regression(lin(), c.Z, c.X, c.W)
        fo = loadl("Collinear", "Coef")
        # The residuals are unique, and they agree. Measured maxscaled 9.8e-16, and maxrel
        # 3.4e-12 on residuals near zero.
        @test parity_compare(csr.eps, loadl("Collinear", "Eps"); scale = :array,
                             name = "Collinear eps").ok
        # The two answers differ along the null direction alone: the market column less the
        # three industry columns. Measured 1.0e-17.
        v = [1.0, -1.0, -1.0, -1.0, 0.0]
        D = fo .- csr.f
        @test maximum(abs.(D .- (D * v) .* transpose(v) ./ LinearAlgebra.dot(v, v))) < 1e-15
        # Ours is the minimum-norm solution: it has no component along the null direction, and
        # its norm is below the oracle's on every observation. The oracle's is up to 5.9 times
        # larger.
        @test maximum(abs.(csr.f * v)) < 1e-15
        @test all(LinearAlgebra.norm(csr.f[t, :]) < LinearAlgebra.norm(fo[t, :])
                  for t in axes(fo, 1))
    end

    @testset "A nearly singular design: Better" begin
        # `NearSingular` fits `parity_cs_near_singular()` with the default linear member. The
        # condition number of each weighted design is 1.5e6 to 2.7e6. We solve the weighted
        # design by QR, and the oracle solves the normal equations, which squares it.
        s = parity_cs_near_singular()
        exact = parity_cs_exact(s.Z, s.X, s.W)
        relerr(a) = maximum(abs.(a .- exact)) / maximum(abs.(exact))
        csr = cross_sectional_regression(lin(), s.Z, s.X, s.W)
        # Measured 8.8e-11 for ours, and 7.6e-4 for the oracle.
        @test relerr(csr.f) < 1e-9
        @test relerr(loadl("NearSingular", "Coef")) > 1e-4
        # Every member that solves the design itself stays near the exact answer. Measured
        # 8.8e-11, and 2.4e-10 for the pseudo-inverse.
        for alg in (UncheckedSolve(), RankDeficiencyRefusal(), MinimumNormSolve())
            @test relerr(cross_sectional_regression(lin(; alg = alg), s.Z, s.X, s.W).f) <
                  1e-9
        end
    end

    @testset "The members that only we have, against the rank of the design" begin
        d = parity_cs_dense()
        q = parity_cs_deficient()
        c = parity_cs_collinear()
        fit(alg, x) = cross_sectional_regression(lin(; alg = alg), x.Z, x.X, x.W).f
        # A full-rank design: the four members agree. Measured maxrel 2.9e-15.
        f0 = fit(PseudoInverseFallback(), d)
        for alg in (UncheckedSolve(), RankDeficiencyRefusal(), MinimumNormSolve())
            @test parity_compare(fit(alg, d), f0; name = "Dense $(nameof(typeof(alg)))").ok
        end
        # A rank-deficient design: the pseudo-inverse members agree exactly, and `\` returns the
        # minimum-norm solution of a non-square design too. Measured maxrel 1.1e-15.
        for x in (q, c)
            fm = fit(MinimumNormSolve(), x)
            @test fit(PseudoInverseFallback(), x) == fm
            @test parity_compare(fit(UncheckedSolve(), x), fm; name = "Unchecked").ok
        end
        # The refusal names the rank, and the answer of `UncheckedSolve`.
        err = try
            fit(RankDeficiencyRefusal(), c)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("observation 1 has rank 4 over 5 factors and 20 eligible assets",
                       err.msg)
        @test occursin("UncheckedSolve() to take whatever `\\` returns", err.msg)
    end

    @testset "The weight policies alone, at parity" begin
        # Each case of `PARITY_CS_WEIGHT_CASES` on `parity_cs_weights_fixture()`. The oracle's
        # weight step reads the residuals through a stub regressor, and returns the weights of
        # its last pass. A `lambda` of zero takes one pass on both sides.
        wf = parity_cs_weights_fixture()
        for (tag, cfg) in PARITY_CS_WEIGHT_CASES
            (; alg, W0, W1) = parity_cs_weights(wf, cfg)
            @test PortfolioOptimisers.needs_second_pass(alg) == !iszero(cfg[2])
            # The cap weights of a power of zero and of one are the mask and the capitalisation.
            if tag == "Cap00"
                @test W0 == Float64.(wf.msk)
            elseif tag == "Cap10"
                @test W0 == ifelse.(wf.msk, wf.mcap, 0.0)
            elseif tag in ("Cap05", "Cap20")
                # Measured maxrel 2.1e-16 and 0.
                @test parity_compare(W0, parity_load("MarketCapWeights", tag, "W0");
                                     name = "$(tag) W0").ok
            else
                # Measured maxrel 3.6e-16 to 5.3e-16. `ratio = 3` binds, and the levels
                # (0.1, 0.9) and (0.05, 0.95) replace the oracle's fixed winsorisation levels.
                @test parity_compare(W1,
                                     parity_load("BlendedInverseVarianceWeights", tag,
                                                 "W1"); name = "$(tag) W1").ok
            end
        end
    end

    @testset "A zero idiosyncratic variance: Better" begin
        # Asset 2 of `parity_cs_weights_fixture(; zero = true)` has a zero residual, so its
        # inverse variance is infinite on every observation after the warm-up. The oracle
        # refuses the infinity in its winsorisation. The stored case is the oracle's answer on
        # `parity_cs_weights_tiny()`, where the residual is tiny but not zero.
        cfg = ("Blend05" => (0.5, 0.5, 20.0, (0.025, 0.975)))[2]
        stored = parity_load("BlendedInverseVarianceWeights", "Blend05Tiny", "W1")
        # The floor of the variance is off, so the inverse is infinite, as in the oracle.
        ve = RegimeAdjustedExpWeightedVariance(; centred = true,
                                               regime_lohi_mult = (0.7, 1.6), min_val = 0.0)
        wz = parity_cs_weights_fixture(; zero = true)
        IV = PortfolioOptimisers.cross_sectional_lagged_inverse_variance(ve, wz.eps, wz.msk;
                                                                         estimation_mask = wz.emsk,
                                                                         active_mask = wz.amsk)
        @test count(isinf, view(IV, :, 2)) == 20
        W1 = parity_cs_weights(wz, cfg; ve = ve).W1
        @test all(isfinite, W1)
        # Each row sums to one, and our answer at zero is the oracle's limit. Measured maxrel
        # 4.3e-16.
        @test all(x -> isapprox(x, 1; rtol = 1e-12),
                  sum(view(W1, 2:size(W1, 1), :); dims = 2))
        @test parity_compare(W1, stored; name = "zero variance").ok
        # Our answer on the tiny residual is at parity too. Measured maxrel 4.3e-16.
        @test parity_compare(parity_cs_weights(parity_cs_weights_tiny(), cfg; ve = ve).W1,
                             stored; name = "tiny variance").ok
    end

    @testset "The weights inside the prior, at parity" begin
        # Each stored case is the fit of `parity_small_panel()` with a market factor, the two
        # passthrough styles and `minra = 5`, under the weight policy the case names. The factor
        # covariance and the idiosyncratic variance clip their regime multiplier to (0.7, 1.6),
        # and the variance has no floor, as the oracle's defaults state. Both are set here, so
        # the defaults of the prior, which #1383 measures, do not move these cases. The outputs
        # are the factor returns, the regression weights, the idiosyncratic variance history,
        # `mu` and `sigma`. The case of `lambda = 0.5` alone is `MasksTwoPass` of `test_12p`.
        fx = parity_small_panel()
        mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                     outlier = nothing, scoring = nothing, family = "style")
        factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
                   "style2" => mpass("style2")]
        pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                            ce = RegimeAdjustedExpWeightedCovariance(; centred = true,
                                                                     regime_lohi_mult = (0.7,
                                                                                         1.6)))
        ve = RegimeAdjustedExpWeightedVariance(; centred = true,
                                               regime_lohi_mult = (0.7, 1.6), min_val = 0.0)
        # Asset 4 is in its warm-up at the latest observation; `test_12p` states why its `mu`
        # and its covariances are a deliberate difference.
        rest = setdiff(axes(fx.rd.X, 2), fx.at.relist[1])
        load(c, o) = parity_load("CrossSectionalFactorPrior", c, o)
        @testset "$(c)" for (c, wa) in ("WeightsBlend03Ratio3" =>
                                            BlendedInverseVarianceWeights(; lambda = 0.3, ratio = 3.0),
                                        "WeightsBlend10Ratio5Wins10" =>
                                            BlendedInverseVarianceWeights(; lambda = 1.0, ratio = 5.0,
                                                                          wins = (0.1, 0.9)),
                                        "WeightsBlend05Cap10" =>
                                            BlendedInverseVarianceWeights(; p = 1.0, lambda = 0.5),
                                        "WeightsCap10" => MarketCapWeights(; p = 1.0))
            pr = prior(CrossSectionalFactorPrior(; factors = factors, minra = 5, wa = wa,
                                                 pe = pe, ve = ve), fx.rd)
            # Measured maxrel 4.5e-14, 1.2e-15, 1.9e-15 and 3.9e-14 at most.
            @test parity_compare(pr.rr.csr.f, load(c, "FactorReturns"); name = "$(c) f").ok
            @test parity_compare(pr.rr.rw, load(c, "RegressionWeights"); name = "$(c) rw").ok
            @test parity_compare(pr.rr.vs, load(c, "IdioVariances"); name = "$(c) vs").ok
            @test parity_compare(pr.mu[rest], vec(load(c, "Mu"))[rest]; name = "$(c) mu").ok
            # A covariance compares against its largest entry, because its small off-diagonal
            # entries come from a cancellation (#1376). Measured maxscaled 3.3e-13, and maxrel
            # 4.6e-11 cell by cell.
            @test parity_compare(pr.sigma[rest, rest], load(c, "Sigma")[rest, rest];
                                 scale = :array, name = "$(c) sigma").ok
        end
    end

    @testset "A missing market capitalisation: Better (#725)" begin
        # The oracle refuses a panel whose market capitalisation is not finite on an active
        # pair, and names the field, so one missing value stops the fit of every observation.
        # The regression weight `cap^p` sets the efficiency of the fit and not its model: any
        # non-negative weights of full rank give an unbiased weighted least squares. A pair whose
        # weight is unknown therefore takes the weight zero, and the fit is the weighted least
        # squares over the pairs whose weights are known. The prior drops the pair from both
        # masks, which #725 records, and it imputes nothing: an imputation is the fill policy of
        # the Panel Field, which runs before the prior reads it.
        fx = parity_small_panel()
        only(filter(f -> f.name == "market_cap", fx.rd.pnl.pf)).vals[46, 8] = NaN
        pr = prior(CrossSectionalFactorPrior(; factors = ["market" => ConstantExposure()],
                                             minra = 5,
                                             wa = BlendedInverseVarianceWeights(;
                                                                                lambda = 0.5)),
                   fx.rd)
        # The regression weights start at observation 2, after the lag of one.
        @test iszero(pr.rr.rw[47 - 1, 8])
        @test count(iszero, pr.rr.rw[46, :]) < size(pr.rr.rw, 2)
        # The other weights of that observation are normalised again, so they sum to one.
        @test isapprox(sum(pr.rr.rw[46, :]), 1; rtol = 1e-12)
        @test all(isfinite, pr.rr.csr.f)
        # The weight policy alone refuses the capitalisation of an eligible pair, and names it.
        @test_throws DomainError PortfolioOptimisers.cs_weights_initial(MarketCapWeights(),
                                                                        [NaN 1.0],
                                                                        [true true])
    end
end
