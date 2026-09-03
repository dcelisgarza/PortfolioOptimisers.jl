#=
Check `src/08_Moments/38_CrossSectionalRegression.jl` against the mathematics its docstrings
state, and against the reference implementation the map of issue #643 ports. Issue #679.

THREE FACTS SHAPE THE PROBES.

1. THE WEIGHT IS THE ONE CONTRACT. A pair with a zero cross-sectional weight leaves the fit
   and may carry a missing return or a missing exposure. A pair with a positive weight may
   not. `The eligibility contract` drives both directions, because a mask read off the data
   rather than off the weights would silently admit a pair the caller meant to exclude.

2. THE FOUR SOLVE MEMBERS AGREE ON A FULL-RANK DESIGN AND SPLIT ON A SQUARE SINGULAR ONE.
   Julia's `\` answers a deficient design two ways, and MEASUREMENT, not the shape of the
   API, says which. A NON-SQUARE design reaches a column-pivoted `QR` whose solve completes
   the orthogonal factorisation, so it returns the MINIMUM-NORM answer and agrees with
   `pinv`; issue #648's ground truth 3 called that a basic solution, and it is not one. A
   SQUARE singular design reaches an `LU` factorisation instead and throws
   `SingularException`. That square case is where `UncheckedSolve` really parts from the
   other three, and `A rank-deficient design` pins both halves.

3. THE TWO MEMBERS SOLVE ONE PROBLEM TWO WAYS. `CrossSectionalLinearRegression` runs the
   closed form on the weighted design, and `CrossSectionalTargetRegression` hands the
   unweighted pair to an external target with the weights as observation weights. They must
   land on the same coefficients, and `The two members agree` computes both. They differ in
   one place only: an observation with no eligible asset, which the closed form answers with
   zero factor returns and the target member refuses.

The reference fit at the top of the file is written out per observation with `pinv`, so no
testset compares the file against itself.
=#

# The weighted least squares of every observation, written out one observation at a time.
# `pinv` is the minimum-norm solution, which is what `PseudoInverseFallback` returns on a
# deficient design and what the closed form returns on a full-rank one.
function reference_cross_sectional_fit(Z, X, W; intercept = false)
    T, N, K = size(Z)
    f = zeros(T, K)
    b = zeros(T)
    n = zeros(Int, T)
    for t in 1:T
        idx = findall(W[t, :] .> 0)
        n[t] = length(idx)
        if isempty(idx)
            continue
        end
        w = W[t, idx]
        A = Z[t, idx, :]
        y = X[t, idx]
        ybar = 0.0
        xbar = zeros(K)
        if intercept
            sw = sum(w)
            ybar = dot(w, y) / sw
            xbar = (A' * w) ./ sw
            y = y .- ybar
            A = A .- xbar'
        end
        sq = sqrt.(w)
        coef = pinv(A .* sq) * (y .* sq)
        f[t, :] = coef
        b[t] = ybar - dot(coef, xbar)
    end
    return f, b, n
end

# A panel whose returns are an exact factor model plus a small noise, with a blanked pair, a
# zero-weight pair and a missing return, so the eligibility contract is exercised throughout.
function cross_sectional_panel(; T = 6, N = 25, K = 4, seed = 987)
    rng = StableRNG(seed)
    Z = randn(rng, T, N, K)
    beta = randn(rng, T, K)
    X = permutedims(reduce(hcat, Z[t, :, :] * beta[t, :] for t in 1:T)) .+
        0.02 * randn(rng, T, N)
    W = 1.0 .+ rand(rng, T, N)
    W[1, 1:3] .= 0.0
    Z[1, 1, 1] = NaN
    X[1, 2] = NaN
    return Z, X, W, beta
end

@testset "Cross-sectional regression" begin
    Z, X, W, beta = cross_sectional_panel()
    # Rows 2 onwards carry no missing entry, so a probe that needs a clean panel takes them.
    Zc, Xc, Wc = Z[2:end, :, :], X[2:end, :], W[2:end, :]

    @testset "The closed form recovers the factor returns" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        f, _, n = reference_cross_sectional_fit(Z, X, W)
        @test csr.f ≈ f
        @test csr.n == n
        @test isnothing(csr.b)
        # The noise is 2% of a unit-variance factor, so the fitted returns sit near the ones
        # the panel was built from.
        @test csr.f≈beta rtol=1e-2
        # The first observation lost exactly the three assets its zero weights name. The
        # missing exposure and the missing return sit on two of those three, so they cost
        # the fit nothing beyond the weights that already excluded them.
        @test csr.n[1] == size(X, 2) - 3
        @test all(==(size(X, 2)), csr.n[2:end])
    end

    @testset "The intercept is recovered from the weighted centroid" begin
        cre = CrossSectionalLinearRegression(; intercept = true)
        csr = cross_sectional_regression(cre, Z, X, W)
        f, b, _ = reference_cross_sectional_fit(Z, X, W; intercept = true)
        @test csr.f ≈ f
        @test csr.b ≈ b
        # A design that already carries a constant column makes the intercept redundant, so
        # the fitted values match the ones the no-intercept fit produces on that design.
        Zi = cat(Zc, ones(size(Zc, 1), size(Zc, 2), 1); dims = 3)
        with_int = cross_sectional_regression(cre, Zi, Xc, Wc)
        no_int = cross_sectional_regression(CrossSectionalLinearRegression(), Zi, Xc, Wc)
        @test predict(with_int, Zi) ≈ predict(no_int, Zi)
    end

    @testset "The residuals and the systematic part partition the returns" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        Xh = predict(csr, Z)
        @test isequal(csr.eps, X - Xh)
        # A missing return leaves a missing residual, which is the honest answer for a pair
        # the fit never saw.
        @test isnan(csr.eps[1, 2])
        @test isnan(csr.eps[1, 1])
        @test all(isfinite, csr.eps[2:end, :])
        # An intercept shifts every fitted value of its observation by the same amount.
        ci = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                        Zc, Xc, Wc)
        shift = predict(ci, Zc) -
                PortfolioOptimisers.cross_sectional_systematic(ci.f, nothing, Zc)
        for t in axes(shift, 1)
            @test all(x -> isapprox(x, ci.b[t]), shift[t, :])
        end
    end

    @testset "The eligibility contract" begin
        # A positive weight on a pair with a missing exposure is refused, and so is one on a
        # pair with a missing return.
        Wz = copy(W)
        Wz[1, 1] = 1.0
        @test_throws PortfolioOptimisers.IsNonFiniteError cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                                     Z, X,
                                                                                     Wz)
        Wy = copy(W)
        Wy[1, 2] = 1.0
        @test_throws PortfolioOptimisers.IsNonFiniteError cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                                     Z, X,
                                                                                     Wy)
        # A zero weight admits both, and the pair leaves the fit rather than poisoning it.
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        @test all(isfinite, csr.f)
        # A negative weight and a non-finite weight are refused before the mask is built.
        Wn = copy(W)
        Wn[1, 1] = -1.0
        @test_throws DomainError cross_sectional_regression(CrossSectionalLinearRegression(),
                                                            Z, X, Wn)
        Wf = copy(W)
        Wf[1, 1] = NaN
        @test_throws PortfolioOptimisers.IsNonFiniteError cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                                     Z, X,
                                                                                     Wf)
        # The three arrays must agree on the observation and the asset axes.
        @test_throws DimensionMismatch cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                  Z, X, W[:, 1:(end - 1)])
        @test_throws DimensionMismatch cross_sectional_regression(CrossSectionalLinearRegression(),
                                                                  Z, X[:, 1:(end - 1)],
                                                                  W[:, 1:(end - 1)])
        # None of the three may be empty.
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_design_mask(zeros(0,
                                                                                                            0,
                                                                                                            0),
                                                                                                      X,
                                                                                                      W)
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_design_mask(Z,
                                                                                                      zeros(0,
                                                                                                            0),
                                                                                                      W)
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.cross_sectional_design_mask(Z,
                                                                                                      X,
                                                                                                      zeros(0,
                                                                                                            0))
    end

    @testset "The four solve members agree on a full-rank design" begin
        base = cross_sectional_regression(CrossSectionalLinearRegression(), Zc, Xc, Wc)
        for alg in (PseudoInverseFallback(), RankDeficiencyRefusal(), UncheckedSolve(),
                    MinimumNormSolve())
            csr = cross_sectional_regression(CrossSectionalLinearRegression(; alg = alg),
                                             Zc, Xc, Wc)
            @test csr.f ≈ base.f
        end
    end

    @testset "A rank-deficient design" begin
        Zd = copy(Zc)
        # The second factor is twice the first, so the numerical rank falls by one and the
        # two coefficients are no longer separately identified.
        Zd[:, :, 2] = 2 .* Zd[:, :, 1]
        @test PortfolioOptimisers.cross_sectional_rank(Zd[1, :, :]) == size(Zd, 3) - 1
        @test PortfolioOptimisers.cross_sectional_rank(Zc[1, :, :]) == size(Zc, 3)
        # A design with no row has rank zero and reads no pivot.
        @test PortfolioOptimisers.cross_sectional_rank(zeros(0, 3)) == 0
        fallback = cross_sectional_regression(CrossSectionalLinearRegression(), Zd, Xc, Wc)
        minnorm = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                            alg = MinimumNormSolve()),
                                             Zd, Xc, Wc)
        unchecked = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                              alg = UncheckedSolve()),
                                               Zd, Xc, Wc)
        # The fallback pseudo-inverts, so it lands on the minimum-norm answer. The design is
        # not square, so `\` reaches a column-pivoted QR whose solve completes the
        # orthogonal factorisation and lands there too. All three agree.
        @test fallback.f ≈ minnorm.f
        @test unchecked.f ≈ minnorm.f
        @test predict(unchecked, Zd) ≈ predict(minnorm, Zd)
        # The two part on a SQUARE design, which `\` sends to an LU factorisation instead.
        # An exactly singular one throws, and the pseudo-inverse answers it.
        Zs = Zd[:, 1:size(Zd, 3), :]
        Xs = Xc[:, 1:size(Zd, 3)]
        Ws = Wc[:, 1:size(Zd, 3)]
        @test_throws LinearAlgebra.SingularException cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                                                               alg = UncheckedSolve()),
                                                                                Zs, Xs, Ws)
        square = cross_sectional_regression(CrossSectionalLinearRegression(), Zs, Xs, Ws)
        @test all(isfinite, square.f)
        @test square.f ≈ cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                    alg = MinimumNormSolve()),
                                     Zs, Xs, Ws).f
        # The refusal names the observation and the rank it measured.
        cre = CrossSectionalLinearRegression(; alg = RankDeficiencyRefusal())
        @test_throws ArgumentError cross_sectional_regression(cre, Zd, Xc, Wc)
        msg = try
            cross_sectional_regression(cre, Zd, Xc, Wc)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("observation 1", msg)
        @test occursin("rank $(size(Zd, 3) - 1)", msg)
        @test occursin("$(size(Zd, 2)) eligible assets", msg)
    end

    @testset "The two members agree" begin
        for intercept in (false, true)
            closed = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                               intercept = intercept),
                                                Zc, Xc, Wc)
            target = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                               intercept = intercept),
                                                Zc, Xc, Wc)
            @test closed.f ≈ target.f
            @test closed.n == target.n
            if intercept
                @test closed.b ≈ target.b
            else
                @test isnothing(closed.b)
                @test isnothing(target.b)
            end
        end
        # A generalised linear model of the Normal family is the same least squares, so the
        # target field really is the seam it claims to be.
        glm = cross_sectional_regression(CrossSectionalTargetRegression(;
                                                                        tgt = GeneralisedLinearModel()),
                                         Zc, Xc, Wc)
        @test glm.f ≈
              cross_sectional_regression(CrossSectionalLinearRegression(), Zc, Xc, Wc).f
    end

    @testset "An observation with no eligible asset" begin
        We = copy(W)
        We[2, :] .= 0.0
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, We)
        @test csr.n[2] == 0
        @test all(iszero, csr.f[2, :])
        # The intercept of an empty cross-section is zero, because it has no centroid.
        ci = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                        Z, X, We)
        @test iszero(ci.b[2])
        @test all(iszero, ci.f[2, :])
        # An external target has no cross-section to fit, so it refuses and names the
        # observation.
        @test_throws ArgumentError cross_sectional_regression(CrossSectionalTargetRegression(),
                                                              Z, X, We)
        msg = try
            cross_sectional_regression(CrossSectionalTargetRegression(), Z, X, We)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("observation 2", msg)
        # The refusal of `RankDeficiencyRefusal` also covers the empty cross-section, whose
        # rank is zero.
        @test_throws ArgumentError cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                                             alg = RankDeficiencyRefusal()),
                                                              Z, X, We)
    end

    @testset "The coefficient of determination" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, W)
        r2 = cross_sectional_r2(csr, Z, X, W)
        @test length(r2) == size(X, 1)
        @test all(x -> 0 <= x <= 1, r2)
        @test mean_cross_sectional_r2(csr, Z, X, W) ≈ sum(r2) / length(r2)
        # An exact fit scores one. Four assets and four factors leave no residual.
        Ze = Zc[:, 1:size(Zc, 3), :]
        Xe = Xc[:, 1:size(Zc, 3)]
        We = Wc[:, 1:size(Zc, 3)]
        exact = cross_sectional_regression(CrossSectionalLinearRegression(), Ze, Xe, We)
        @test cross_sectional_r2(exact, Ze, Xe, We) ≈ ones(size(Ze, 1))
        # An observation with no eligible asset has no ratio, and the mean skips it.
        Wn = copy(W)
        Wn[2, :] .= 0.0
        csn = cross_sectional_regression(CrossSectionalLinearRegression(), Z, X, Wn)
        r2n = cross_sectional_r2(csn, Z, X, Wn)
        @test isnan(r2n[2])
        @test mean_cross_sectional_r2(csn, Z, X, Wn) ≈
              sum(filter(isfinite, r2n)) / count(isfinite, r2n)
        # A cross-section whose returns are all equal has a zero total sum of squares, so
        # its ratio is undefined too, and a panel of such observations has no mean at all.
        Zf = ones(2, 3, 1)
        Xf = ones(2, 3)
        Wf = ones(2, 3)
        csf = cross_sectional_regression(CrossSectionalLinearRegression(), Zf, Xf, Wf)
        @test all(isnan, cross_sectional_r2(csf, Zf, Xf, Wf))
        @test isnan(mean_cross_sectional_r2(csf, Zf, Xf, Wf))
    end

    @testset "The systematic part on a tensor the fit did not see" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Zc, Xc, Wc)
        # The asset axis may differ, and the observation and factor axes may not.
        @test size(predict(csr, Zc[:, 1:3, :])) == (size(Zc, 1), 3)
        @test_throws DimensionMismatch predict(csr, Zc[1:2, :, :])
        @test_throws DimensionMismatch predict(csr, Zc[:, :, 1:2])
    end

    @testset "The result and its view" begin
        csr = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                         Zc, Xc, Wc)
        v = PortfolioOptimisers.port_opt_view(csr, [1, 3, 5])
        # The asset axis of a cross-sectional result is the second one, so only `eps` moves.
        @test size(v.eps) == (size(Zc, 1), 3)
        @test v.eps == csr.eps[:, [1, 3, 5]]
        @test v.f === csr.f
        @test v.n === csr.n
        @test v.b === csr.b
        # The passthrough returns the result it was handed.
        @test cross_sectional_regression(csr, Zc, Xc, Wc) === csr
        @test cross_sectional_regression(csr) === csr
    end

    @testset "The result refuses a malformed construction" begin
        f = [1.0 2.0; 3.0 4.0]
        eps = [0.1 0.2 0.3; 0.4 0.5 0.6]
        n = [3, 3]
        @test CrossSectionalRegression(; f = f, eps = eps, n = n) isa
              CrossSectionalRegression
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalRegression(;
                                                                               f = zeros(0,
                                                                                         0),
                                                                               eps = eps,
                                                                               n = n)
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalRegression(; f = f,
                                                                               eps = zeros(0,
                                                                                           0),
                                                                               n = n)
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalRegression(; f = f,
                                                                               eps = eps,
                                                                               n = Int[])
        @test_throws DimensionMismatch CrossSectionalRegression(; f = f, eps = eps,
                                                                n = [3, 3, 3])
        @test_throws DomainError CrossSectionalRegression(; f = f, eps = eps, n = [3, -1])
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalRegression(; f = f,
                                                                               eps = eps,
                                                                               n = n,
                                                                               b = Float64[])
        @test_throws DimensionMismatch CrossSectionalRegression(; f = f, eps = eps, n = n,
                                                                b = [1.0, 2.0, 3.0])
    end

    @testset "The type tree" begin
        # Both roots are umbrellas. An estimator child names the geometry it fits, and
        # a result child names what a member carries.
        @test StepwiseRegression <:
              PortfolioOptimisers.AbstractTimeSeriesRegressionEstimator
        @test DimensionReductionRegression <:
              PortfolioOptimisers.AbstractTimeSeriesRegressionEstimator
        @test Regression <: PortfolioOptimisers.AbstractLoadingsRegressionResult
        @test CrossSectionalLinearRegression <:
              PortfolioOptimisers.AbstractCrossSectionalRegressionEstimator
        @test CrossSectionalTargetRegression <:
              PortfolioOptimisers.AbstractCrossSectionalRegressionEstimator
        @test CrossSectionalRegression <:
              PortfolioOptimisers.AbstractCrossSectionalRegressionResult
        @test PortfolioOptimisers.AbstractTimeSeriesRegressionEstimator <:
              PortfolioOptimisers.AbstractRegressionEstimator
        @test PortfolioOptimisers.AbstractCrossSectionalRegressionResult <:
              PortfolioOptimisers.AbstractRegressionResult
        @test PortfolioOptimisers.AbstractLoadingsRegressionResult <:
              PortfolioOptimisers.AbstractRegressionResult
        # The loadings criterion partitions the two result children, so neither is a
        # subtype of the other.
        @test !(PortfolioOptimisers.AbstractCrossSectionalRegressionResult <:
                PortfolioOptimisers.AbstractLoadingsRegressionResult)
        @test !(PortfolioOptimisers.AbstractLoadingsRegressionResult <:
                PortfolioOptimisers.AbstractCrossSectionalRegressionResult)
        # The loadings root replaced `AbstractTimeSeriesRegressionResult`, so a site that
        # kept the old name fails to resolve rather than binding a stale root.
        @test !isdefined(PortfolioOptimisers, :AbstractTimeSeriesRegressionResult)
        # `RegE_Reg` pairs the loadings result root with the time-series estimator root,
        # so no cross-sectional type reaches a consumer that reads the loadings matrix.
        @test StepwiseRegression <: PortfolioOptimisers.RegE_Reg
        @test Regression <: PortfolioOptimisers.RegE_Reg
        @test !(CrossSectionalLinearRegression <: PortfolioOptimisers.RegE_Reg)
        @test !(CrossSectionalRegression <: PortfolioOptimisers.RegE_Reg)
        # The four solve members share one algorithm root.
        for alg in (PseudoInverseFallback(), RankDeficiencyRefusal(), UncheckedSolve(),
                    MinimumNormSolve())
            @test isa(alg, PortfolioOptimisers.AbstractCrossSectionalSolveAlgorithm)
            @test isa(alg, PortfolioOptimisers.AbstractRegressionAlgorithm)
        end
        # None of the new abstract types is exported.
        for T in (:AbstractTimeSeriesRegressionEstimator,
                  :AbstractCrossSectionalRegressionEstimator, :AbstractLoadingsRegressionResult,
                  :AbstractCrossSectionalRegressionResult,
                  :AbstractCrossSectionalSolveAlgorithm)
            @test !Base.isexported(PortfolioOptimisers, T)
        end
    end

    @testset "An integer panel promotes to a float result" begin
        Zi = ones(Int, 2, 3, 1)
        Zi[:, 2, 1] .= 2
        Zi[:, 3, 1] .= 3
        Xi = [1 2 3; 2 4 6]
        Wi = ones(Int, 2, 3)
        csr = cross_sectional_regression(CrossSectionalLinearRegression(), Zi, Xi, Wi)
        @test eltype(csr.f) == Float64
        @test csr.f ≈ [1.0; 2.0;;]
    end
end
