#=
The robust covariance estimators of #2025: the minimum covariance determinant
(`MCDCovariance`, with the searches `DetMCD` and `FastMCD`) and Tyler's M-estimator
(`TylerCovariance`), with the location adapter `RobustExpectedReturns`.

The R oracles of the issue (`robustbase::covMcd` and `ICSNP::tyler.shape` / `HR.Mest`) are
not on the test machine, so these tests check the properties that define each estimator:
the resistance to outliers, the agreement with the sample covariance on clean data, the
fixed points of the C-step and of the Tyler equations, the closed forms of the factors, the
determinism of the searches and the refusals.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs, Random

const POR = PortfolioOptimisers
const StatsAPI = PortfolioOptimisers.StatsAPI

function robust_test_sample(rng, n, p; nbad = 0, shift = 8)
    A = randn(rng, p, p)
    L = cholesky(A * A' + I).L
    X = Matrix((L * randn(rng, p, n))')
    clean = copy(X)
    bad = sort(randperm(rng, n)[1:nbad])
    X[bad, :] .= shift .+ randn(rng, nbad, p)
    return X, clean, bad
end
relerr(A, B) = norm(A - B) / norm(B)

@testset "Robust covariance estimators (#2025)" begin
    @testset "Qn scale" begin
        rng = StableRNG(2025)
        function brute_qn(x)
            n = length(x)
            d = sort([abs(x[i] - x[j]) for i in 1:n for j in (i + 1):n])
            h = div(n, 2) + 1
            return d[div(h * (h - 1), 2)] * 2.21914 * POR.qn_finite_factor(n)
        end
        for _ in 1:300
            n = rand(rng, 2:70)
            x = rand(rng, Bool) ? randn(rng, n) : round.(randn(rng, n); digits = 1)
            @test POR.qn_scale(x) ≈ brute_qn(x)
        end
        @test iszero(POR.qn_scale([1.0]))
        # More than half of the values are equal: the quantile rule gives a positive scale.
        z = vcat(zeros(30), randn(rng, 10))
        @test iszero(POR.qn_scale(z))
        @test POR.robust_column_scale(z) > 0
        @test POR.robust_column_scale(zeros(10)) == 1
        @test POR.weighted_high_median([3.0, 1.0, 2.0], [1, 1, 5]) == 2.0
    end
    @testset "Factors" begin
        # Croux-Haesbroeck at p = 2, a = 1/2 has a closed form: q = 2 log 2 and
        # F_{chi2_4}(q) = 1 - e^{-q/2} (1 + q/2).
        q = 2 * log(2)
        @test POR.mcd_consistency(2, 0.5) ≈ 0.5 / (1 - exp(-q / 2) * (1 + q / 2))
        @test POR.mcd_consistency(4, 1.0) == 1
        @test POR.mcd_subset_size(0.5, 100, 5) == 53
        @test POR.mcd_subset_size(1, 100, 5) == 100
        for p in (1, 2, 5), kind in (:raw, :rew)
            f = POR.mcd_correction(MCDSmallSampleCorrection(), p, 60, 0.5, kind)
            @test f > 1
            @test POR.mcd_correction(MCDSmallSampleCorrection(), p, 60, 0.7, kind) > 1
            @test POR.mcd_correction(MCDSmallSampleCorrection(), p, 60, 1.0, kind) ≈ 1
            @test POR.mcd_correction(MCDSmallSampleCorrection(), p, 10^6, 0.5, kind) ≈ 1 atol = 1e-2
        end
        @test POR.mcd_correction(nothing, 3, 60, 0.5, :raw) == 1
    end
    @testset "Outliers" begin
        rng = StableRNG(1)
        X, clean, bad = robust_test_sample(rng, 400, 5; nbad = 20)
        S = cov(clean[setdiff(1:400, bad), :])
        @test relerr(cov(X), S) > 0.5
        for ce in (MCDCovariance(), MCDCovariance(; alg = FastMCD(; seed = 3)),
                   MCDCovariance(; rw = nothing), TylerCovariance())
            r = StatsAPI.fit(ce, X)
            @test relerr(r.sigma, S) < min(0.3, relerr(cov(X), S) / 3)
            @test bad ⊆ findall(r.outlier)
            @test cov(ce, X) == r.sigma
            @test cor(ce, X) ≈ cov2cor(r.sigma)
            @test vec(var(ce, X)) ≈ diag(r.sigma)
        end
        r = StatsAPI.fit(MCDCovariance(), X)
        @test all(iszero, r.w[bad])
        @test sum(r.w) > 0.9 * 380
        raw = StatsAPI.fit(MCDCovariance(; rw = nothing), X)
        @test sum(raw.w) == POR.mcd_subset_size(0.5, 400, 5)
        rt = StatsAPI.fit(TylerCovariance(), X)
        @test rt.w ≈ 5 ./ rt.d2
        @test all(rt.w[bad] .< 0.2)
    end
    @testset "Clean data" begin
        X, = robust_test_sample(StableRNG(5), 600, 4)
        for ce in (MCDCovariance(), TylerCovariance(), MCDCovariance(; ssc = nothing))
            @test relerr(cov(ce, X), cov(X)) < 0.1
            @test norm(mean(RobustExpectedReturns(; ce = ce), X) - mean(X; dims = 1)) < 0.15
        end
    end
    @testset "Searches" begin
        X, = robust_test_sample(StableRNG(7), 150, 3; nbad = 20, shift = 5)
        Z, = POR.mcd_standardise(X, nothing)
        h = POR.mcd_subset_size(0.5, 150, 3)
        idx = POR.mcd_search(DetMCD(), Z, h, false)
        # The best subset of the search is a fixed point of the C-step.
        @test first(POR.mcd_csteps(Z, idx, h, false, 1)) == idx
        # A C-step never raises the determinant.
        start = collect(1:h)
        _, o0 = POR.mcd_csteps(Z, start, h, false, 0)
        _, o1 = POR.mcd_csteps(Z, start, h, false, 1)
        _, o9 = POR.mcd_csteps(Z, start, h, false, 200)
        @test o9 <= o1 <= o0
        # A seed makes FastMCD reproducible, and no seed draws from the generator.
        fm = MCDCovariance(; alg = FastMCD(; n_starts = 50, seed = 11))
        @test cov(fm, X) == cov(fm, X)
        free = MCDCovariance(; alg = FastMCD(; n_starts = 50, rng = StableRNG(3)))
        @test isapprox(cov(free, X), cov(fm, X); rtol = 0.2)
        # A start that is singular grows until it is not: 40 % of the rows are one point.
        Y = randn(StableRNG(9), 100, 2)
        Y[1:40, :] .= 0.3
        @test all(isfinite, cov(MCDCovariance(; alg = FastMCD(; seed = 1)), Y))
        # The exact search of one variable, with a free and a fixed centre.
        x = X[:, 1:1]
        r1 = StatsAPI.fit(MCDCovariance(; rw = nothing, ssc = nothing), x)
        hx = POR.mcd_subset_size(0.5, 150, 1)
        sx = sort(vec(x))
        best = minimum(var(sx[k:(k + hx - 1)]) for k in 1:(150 - hx + 1))
        @test r1.sigma[1] ≈ best * POR.mcd_consistency(1, hx / 150)
        rf = StatsAPI.fit(MCDCovariance(; me = MedianExpectedReturns(), rw = nothing), x)
        @test rf.mu == vec(median(x; dims = 1))
        # alpha = 1 is the classical estimate.
        ra = StatsAPI.fit(MCDCovariance(; alpha = 1, rw = nothing, ssc = nothing), X)
        @test ra.sigma ≈ cov(X)
        @test ra.mu ≈ vec(mean(X; dims = 1))
        # Data on a hyperplane have no robust scatter.
        H = randn(StableRNG(4), 60, 3)
        H[:, 3] .= H[:, 1] .+ H[:, 2]
        @test_throws DomainError cov(MCDCovariance(), H)
    end
    @testset "Tyler" begin
        X, = robust_test_sample(StableRNG(5), 300, 4)
        X .+= [0.1 -0.2 0.3 0.0]
        mu, V = POR.tyler_fixed_point(X, nothing, nothing)
        D = X .- mu'
        d2 = POR.squared_distances(X, mu, V)
        Vfp = 4 * (D' * (D ./ d2)) / 300
        @test isapprox(Vfp, V; rtol = 1e-6)
        @test norm(vec(sum(D ./ sqrt.(d2); dims = 1))) < 1e-5
        # Integer weights are the rows repeated by their weights.
        Y = X[1:60, :]
        k = rand(StableRNG(2), 1:3, 60)
        Yr = reduce(vcat, [repeat(Y[i:i, :], k[i]) for i in 1:60])
        @test cov(TylerCovariance(; w = fweights(k)), Y) ≈ cov(TylerCovariance(), Yr)
        # Unit weights hold no vector of values, and they are the unweighted fit.
        @test cov(TylerCovariance(; w = uweights(60)), Y) ≈ cov(TylerCovariance(), Y)
        # A fixed centre, a variance estimator, and an observation at the centre.
        m0 = vec(median(X; dims = 1))
        @test StatsAPI.fit(TylerCovariance(; me = MedianExpectedReturns()), X).mu == m0
        @test StatsAPI.fit(TylerCovariance(), X; mean = m0').mu == m0
        tv = cov(TylerCovariance(; ve = SimpleVariance()), X)
        @test sqrt.(diag(tv)) ≈ vec(std(X; dims = 1))
        Xc = vcat(X, reshape(StatsAPI.fit(TylerCovariance(), X).mu, 1, :))
        @test all(isfinite, cov(TylerCovariance(), Xc))
        Xm = vcat(X, repeat(reshape(m0, 1, :), 3))
        rm = StatsAPI.fit(TylerCovariance(; me = MedianExpectedReturns()), Xm)
        @test count(isinf, rm.w) == 3
        @test POR.observation_median([3.0, 1.0, 2.0], nothing) == 2.0
        # Half of the rows on a line through the centre: the shape does not exist, and the
        # iteration warns.
        Yb = vcat(hcat(randn(StableRNG(1), 10), zeros(10)), randn(StableRNG(2), 10, 2))
        @test_logs (:warn, r"did not converge") POR.tyler_fixed_point(Yb, nothing, zeros(2))
    end
    @testset "Element type, orientation and composition" begin
        X, = robust_test_sample(StableRNG(8), 200, 3)
        X32 = Float32.(X)
        for ce in (MCDCovariance(), TylerCovariance(),
                   MCDCovariance(; me = SimpleExpectedReturns()))
            @test eltype(cov(ce, X32)) == Float32
            @test cov(ce, Matrix(X'); dims = 2) ≈ cov(ce, X)
        end
        @test eltype(StatsAPI.fit(TylerCovariance(), X32).w) == Float32
        @test size(mean(RobustExpectedReturns(), X; dims = 1)) == (1, 3)
        @test size(mean(RobustExpectedReturns(), Matrix(X'); dims = 2)) == (3, 1)
        pr = prior(EmpiricalPrior(; me = RobustExpectedReturns(), ce = MCDCovariance()), X)
        @test pr.mu ≈ vec(mean(RobustExpectedReturns(), X))
        @test pr.sigma ≈ cov(MCDCovariance(), X)
        for ce in (MCDCovariance(), TylerCovariance())
            on = partial_fit!(POR.update_online_estimator(Online(ce)), X[1:120, :])
            on = partial_fit!(on, X[121:end, :])
            @test cov(on) ≈ cov(ce, X)
            @test !POR.supports_partial_fit(ce)
        end
        # The weights of a factory call reach `me` of MCD and `w` of Tyler.
        @test POR.show_fields(MCDCovariance()) == (:me, :alg, :alpha, :ssc, :rw, :q)
        @test POR.show_fields(TylerCovariance()) == (:me, :ve, :w, :q)
        w = eweights(1:200, 0.05)
        @test POR.factory(MCDCovariance(; me = MedianExpectedReturns()), w).me.w === w
        @test POR.factory(TylerCovariance(), w).w === w
        @test POR.factory(RobustExpectedReturns(; ce = TylerCovariance()), w).ce.w === w
        v = POR.port_opt_view(TylerCovariance(; me = MedianExpectedReturns()), 1:2)
        @test v isa TylerCovariance
    end
    @testset "Refusals" begin
        rng = StableRNG(3)
        @test_throws DomainError cov(MCDCovariance(), randn(rng, 6, 8))
        @test_throws DomainError cov(TylerCovariance(), randn(rng, 8, 8))
        w = fweights(vcat(ones(8), zeros(20)))
        @test_throws DomainError cov(TylerCovariance(; w = w), randn(rng, 28, 8))
        @test_throws DomainError MCDCovariance(; alpha = 0.4)
        @test_throws DomainError MCDCovariance(; q = 1.0)
        @test_throws DomainError MCDReweighting(; q = 0.0)
        @test_throws DomainError TylerCovariance(; q = 1.5)
        @test_throws DomainError FastMCD(; n_starts = 0)
        @test_throws DomainError FastMCD(; n_starts = 5, n_best = 10)
        @test_throws DimensionMismatch RobustCovarianceResult(; mu = [0.0],
                                                              sigma = ones(2, 2),
                                                              d2 = [1.0], w = [1.0],
                                                              outlier = BitVector([0]))
        @test_throws DimensionMismatch RobustCovarianceResult(; mu = [0.0],
                                                              sigma = ones(1, 1),
                                                              d2 = [1.0, 2.0], w = [1.0],
                                                              outlier = BitVector([0]))
    end
end
