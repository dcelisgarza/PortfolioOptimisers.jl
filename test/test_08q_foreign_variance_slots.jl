#=
A slot that reads the variances of a matrix alone takes a covariance estimator of another package
(#2282). The library calls it through `library_covariance_estimator`, so the fit with
`StatsBase.SimpleCovariance(; corrected = true)` is the fit with `SimpleVariance()`.

`ForeignCovariance` has the one method that the API of `StatsBase` asks for, with its two keywords,
no keyword of the library, no `var` and no `std`. A slot that reads the variance of one vector keeps its bound.

The slots of `CrossSectionalFactorPrior` are in test_12k, and the refusal of `EWResidualVolatility`
is in test_08s.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, StableRNGs, LinearAlgebra

const FVS = PortfolioOptimisers

struct ForeignCovariance <: StatsBase.CovarianceEstimator end
function Statistics.cov(::ForeignCovariance, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing)
    return cov(X; dims = dims)
end

@testset "A variance slot takes a covariance estimator of another package (#2282)" begin
    T, N, K = 120, 4, 3
    rng = StableRNG(2282)
    F = randn(rng, T, K) ./ 100
    X = F * randn(rng, K, N) .+ randn(rng, T, N) ./ 200
    sv = SimpleVariance()
    foreign = (StatsBase.SimpleCovariance(; corrected = true), ForeignCovariance())
    @testset "$(nameof(typeof(fc)))" for fc in foreign
        @testset "The covariance fallback" begin
            for mk in
                (ve -> KendallCovariance(; ve = ve), ve -> SpearmanCovariance(; ve = ve),
                 ve -> MutualInfoCovariance(; ve = ve),
                 ve -> LowerTailDependenceCovariance(; ve = ve))
                @test mk(fc).ve === fc
                @test isapprox(cov(mk(fc), X), cov(mk(sv), X); rtol = 1e-12)
                @test isapprox(cov(mk(fc), permutedims(X); dims = 2), cov(mk(sv), X);
                               rtol = 1e-12)
                # The slots of a fit reach the estimator through the composite, with its
                # keywords and with a panel.
                @test isapprox(cov(PortfolioOptimisersCovariance(; ce = mk(fc)), X,
                                   nothing),
                               cov(PortfolioOptimisersCovariance(; ce = mk(sv)), X,
                                   nothing); rtol = 1e-12)
            end
            # The weights of the fit do not reach an estimator that the library does not own.
            @test factory(KendallCovariance(; ve = fc), aweights(ones(T))).ve === fc
        end
        @testset "The regressions" begin
            for choice in (BatchChoice(), PinnedChoice())
                rf = regression(DimensionReductionRegression(; ve = fc, choice = choice), X,
                                F)
                rs = regression(DimensionReductionRegression(; ve = sv, choice = choice), X,
                                F)
                @test isapprox(rf.M, rs.M; rtol = 1e-12)
                @test isapprox(rf.b, rs.b; rtol = 1e-10)
            end
            proj = Matrix(1.0 * I, K, 2)
            @test isapprox(regression(DimensionReductionRegression(; ve = fc, proj = proj),
                                      X, F).M,
                           regression(DimensionReductionRegression(; ve = sv, proj = proj),
                                      X, F).M; rtol = 1e-12)
            # The intercept is the one of the fit under weights of `retgt`, because the
            # factors are centred under the same weights.
            w = pweights(range(0.5, 1.5; length = T))
            re = DimensionReductionRegression(; ve = fc,
                                              retgt = LinearModel(;
                                                                  kwargs = (; weights = w)))
            x1, Vp, mu, sigma = FVS.prep_dim_red_reg(re, F)
            @test isapprox(mu, vec(mean(F, w; dims = 1)))
            y = X[:, 1]
            b = FVS._regression(re, y, mu, sigma, x1, Vp).beta
            m = FVS.StatsAPI.fit(re.retgt, x1, y)
            @test isapprox(b[1] .+ F * b[2:end], FVS.StatsAPI.predict(m); rtol = 1e-10)
            D = [ones(T) F]
            coefs(ve) = FVS.StatsAPI.coef(FVS.StatsAPI.fit(ElasticNet(; ve = ve, ic = 1), D,
                                                           y))
            @test isapprox(coefs(fc), coefs(sv); rtol = 1e-10)
            @test ElasticNet(; ve = fc, w = w).ve === fc
        end
        @testset "ImpliedVolatilityRegression" begin
            @test isapprox(FVS.realised_vol(fc, X, 20), FVS.realised_vol(sv, X, 20);
                           rtol = 1e-12)
            iv = abs.(randn(StableRNG(7), T, N)) .+ 0.1
            ivc(ve) = cov(ImpliedVolatility(;
                                            alg = ImpliedVolatilityRegression(; ve = ve,
                                                                              ws = 10),
                                            ppy = 252), X; iv = iv)
            @test isapprox(ivc(fc), ivc(sv); rtol = 1e-10)
        end
        @testset "The factor priors" begin
            views = BlackLittermanViews(; P = [1.0 0.0 0.0], Q = [0.001])
            for mk in (ve -> FactorPrior(; ve = ve),
                       ve -> FactorBlackLittermanPrior(; ve = ve, views = views))
                pf = prior(mk(fc), X, F)
                ps = prior(mk(sv), X, F)
                @test isapprox(pf.sigma, ps.sigma; rtol = 1e-12)
                @test isapprox(pf.rr.esigma, ps.rr.esigma; rtol = 1e-12)
            end
        end
        @testset "The variance series" begin
            Xn = copy(X)
            Xn[4, 2] = NaN
            vs = FVS.variance_series(fc, X)
            @test isapprox(vs[2:end, :], FVS.variance_series(sv, X)[2:end, :]; rtol = 1e-12)
            @test isapprox(FVS.variance_series(fc, Xn, nothing)[2:end, :],
                           FVS.variance_series(sv, Xn, nothing)[2:end, :]; rtol = 1e-12,
                           nans = true)
            # A plain estimator refuses a gap in the matrix form, and this one does too.
            @test_throws IsNonFiniteError FVS.variance_series(fc, Xn)
            @test FVS.reads_active_mask(fc) == FVS.series_folds(fc) == false
            @test FVS.reduces_each_window(fc)
            @test isapprox(FVS.cross_sectional_lagged_inverse_variance(fc, X, trues(T, N))[3:end,
                                                                                           :],
                           FVS.cross_sectional_lagged_inverse_variance(sv, X, trues(T, N))[3:end,
                                                                                           :];
                           rtol = 1e-10)
            # A composite over it with no step in `mp`.
            poc = PortfolioOptimisersCovariance(; ce = fc,
                                                mp = MatrixProcessing(; pdm = nothing))
            @test isapprox(FVS.variance_series(poc, X)[2:end, :], vs[2:end, :];
                           rtol = 1e-12)
            @test isapprox(FVS.variance_series(poc, Xn, nothing)[2:end, :],
                           FVS.variance_series(sv, Xn, nothing)[2:end, :]; rtol = 1e-12,
                           nans = true)
        end
        @testset "WindowedCovariance" begin
            wf = WindowedCovariance(; ce = fc, window = 20)
            wv = WindowedVariance(; window = 20)
            Xw = X[(T - 19):T, :]
            @test isapprox(cov(wf, X), cov(Xw); rtol = 1e-12)
            @test isapprox(cor(wf, X), cor(Xw); rtol = 1e-12)
            @test isapprox(cov(wf, X, nothing), cov(Xw); rtol = 1e-12)
            @test isapprox(std(wf, X), std(wv, X); rtol = 1e-12)
            @test isapprox(FVS.variance_series(wf, X)[2:end, :],
                           FVS.variance_series(wv, X)[2:end, :]; rtol = 1e-12)
        end
    end
    @testset "The count of the variance" begin
        sc = StatsBase.SimpleCovariance(; corrected = true)
        a, b = FVS.variance_count(sc, X), FVS.variance_count(sv, X)
        @test (a.n, a.m) == (b.n, b.m)
        @test isnothing(FVS.variance_count(ForeignCovariance(), X))
        wc = FVS.variance_count(WindowedCovariance(; ce = sc, window = 20), X)
        @test wc.n == fill(20, N)
        pc = FVS.variance_count(PortfolioOptimisersCovariance(; ce = sc), X)
        @test pc.m == b.m
    end
    @testset "A slot that reads one vector keeps its bound" begin
        sc = StatsBase.SimpleCovariance(; corrected = true)
        @test_throws TypeError FVS.SecondMoment(; ve = sc)
        @test_throws TypeError FVS.HighOrderMoment(; ve = sc)
        @test_throws TypeError FVS.ThirdCentralMoment(; ve = sc)
        @test_throws TypeError WindowedVariance(; ve = sc)
    end
    @testset "DimensionReductionRegression takes a variance estimator with no mean" begin
        # `ExpWeightedVariance` has no field `me`, and the fit read `ve.me`.
        rr = regression(DimensionReductionRegression(; ve = ExpWeightedVariance()), X, F)
        ref = regression(DimensionReductionRegression(), X, F)
        # Every component is kept, so the scale of the factors does not change the fit.
        @test isapprox(rr.M, ref.M; rtol = 1e-8)
    end
end
