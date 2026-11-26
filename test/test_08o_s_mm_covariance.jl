#=
The robust covariance estimators of #2255 with a high efficiency: the S estimator
(`SCovariance`, with the bisquare rho `BisquareRho` and the search `DetS`) and the MM estimator
(`MMCovariance`, an S estimate and the M-step `BisquareReweighting`). The M-step also follows an
`MCDCovariance` as a soft reweighting.

The reference values come from R 4.5.3 with `rrcov` 1.7-7: `CovSest` with `method = "sdet"` and
`CovMMest` from that start, both with a tight tolerance. The drivers are in the parity folder
`smm2255`. The other tests check the resistance to outliers, the efficiency on clean data, the
fixed centre, the determinism of the search, the tuning constants and the refusals.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs, Random

const POS = PortfolioOptimisers
const StatsAPI = PortfolioOptimisers.StatsAPI

srel(A, B) = norm(A - B) / norm(B)
# The sample that R read. The outliers are the first rows.
function s_parity_sample(seed, n, p, nbad)
    rng = StableRNG(seed)
    A = randn(rng, p, p)
    X = Matrix((cholesky(A * A' + I).L * randn(rng, p, n))')
    X[1:nbad, :] .= 8 .+ 0.5 .* randn(rng, nbad, p)
    return X
end
function s_outlier_sample(rng, n, p, nbad, shift)
    A = randn(rng, p, p)
    X = Matrix((cholesky(A * A' + I).L * randn(rng, p, n))')
    clean = copy(X)
    bad = sort(randperm(rng, n)[1:nbad])
    X[bad, :] .= shift .+ randn(rng, nbad, p)
    return X, clean, bad
end

@testset "S and MM covariance estimators (#2255)" begin
    X = s_parity_sample(2255, 100, 3, 5)
    @testset "Parity with R" begin
        s = StatsAPI.fit(SCovariance(), X)
        @test s.mu ≈ [0.37055574224751581, -0.13610232568127398, 0.40299993224787495] rtol = 1e-7
        @test s.sigma ≈ [4.01738346083710773 0.17649232054976494 0.77746979221516122
                         0.17649232054976494 1.74135904146847875 -0.79936564345125105
                         0.77746979221516110 -0.79936564345125127 2.96285192237424200] rtol = 1e-7
        m = StatsAPI.fit(MMCovariance(), X)
        @test m.mu ≈ [0.36418906407044999, -0.11020145574040709, 0.27854590508127985] rtol = 1e-6
        @test m.sigma ≈ [4.172412753852932532 -0.024876612646107462 1.464235161901686455
                         -0.024876612646107462 1.413674116754670695 -0.543058524168229306
                         1.464235161901686455 -0.543058524168229306 3.563656346515132967] rtol = 1e-6
        # `.csolve.bw.S(0.5, p)$cc` and the `c1` of `CovMMest`, both on the distance.
        ref = Dict(1 => (1.5476449798823020, 5.4902492392321349),
                   2 => (2.6608033928547403, 5.8103156171575172),
                   3 => (3.45288165132049185, 6.096266441052812),
                   5 => (4.6520233425976869, 6.5955803873081562),
                   10 => (6.7758211773815198, 7.5877243818531888))
        for (p, (c0, c1)) in ref
            @test sqrt(POS.bisquare_constant(0.5, p)) ≈ c0 rtol = 1e-7
            @test sqrt(POS.mm_constant(BisquareReweighting(), p)) ≈ c1 rtol = 1e-7
        end
        @test POS.bisquare_efficiency(ShapeEfficiency(), 3,
                                      POS.mm_constant(BisquareReweighting(), 3)) ≈ 0.95
    end
    @testset "Outliers" begin
        Y, clean, bad = s_outlier_sample(StableRNG(2255), 400, 4, 20, 20)
        good = setdiff(1:400, bad)
        @test srel(cov(Y), cov(clean)) > 10
        for ce in (SCovariance(; q = 0.999), MMCovariance(; q = 0.999),
                   MCDCovariance(; rw = BisquareReweighting(), q = 0.999))
            r = StatsAPI.fit(ce, Y)
            @test srel(r.sigma, cov(clean)) < 0.15
            @test findall(r.outlier) == bad
            @test all(iszero, r.w[bad])
            @test all(>(0), r.w[good])
            @test all(<=(1), r.w)
        end
    end
    @testset "Efficiency on clean data" begin
        Xc = randn(StableRNG(7), 3000, 3) * [1.0 0.3 0.0; 0.0 1.0 0.5; 0.0 0.0 1.0]
        es = srel(cov(SCovariance(), Xc), cov(Xc))
        em = srel(cov(MMCovariance(), Xc), cov(Xc))
        @test em < 0.01
        @test em < es / 2
    end
    @testset "Weights and distances" begin
        s = StatsAPI.fit(SCovariance(), X)
        c2 = POS.bisquare_constant(0.5, 3)
        @test s.d2 ≈ POS.squared_distances(X, s.mu, s.sigma)
        @test s.w ≈ ifelse.(s.d2 .< c2, (1 .- s.d2 ./ c2) .^ 2, 0.0)
        @test isnothing(s.intensity)
        @test s.outlier == (s.d2 .> quantile(POS.Distributions.Chisq(3), 0.975))
        # The scale of an S estimate is its M-scale: the mean rho is the breakdown point.
        @test mean(t -> POS.srho(BisquareRho(), t, c2), s.d2) ≈ 0.5
        @test POS.srho(BisquareRho(), 2 * c2, c2) == 1
        @test POS.srho_weight(BisquareRho(), 2 * c2, c2) == 0
        m = StatsAPI.fit(MMCovariance(), X)
        # The M-step keeps the scale of the S estimate.
        @test det(m.sigma) ≈ det(s.sigma)
    end
    @testset "Determinism and orientation" begin
        for ce in (SCovariance(), MMCovariance())
            @test cov(ce, X) == StatsAPI.fit(ce, X).sigma
            @test StatsAPI.fit(ce, permutedims(X); dims = 2).sigma ≈
                  StatsAPI.fit(ce, X).sigma
            @test cor(ce, X) ≈ POS.cov2cor_nan!(cov(ce, X))
        end
    end
    @testset "Fixed centre" begin
        med = vec(median(X; dims = 1))
        sm = SCovariance(; me = MedianExpectedReturns())
        @test StatsAPI.fit(sm, X).mu ≈ med
        @test StatsAPI.fit(MMCovariance(; s = sm), X).mu ≈ med
        @test StatsAPI.fit(MMCovariance(), X; mean = med).mu ≈ med
        @test StatsAPI.fit(SCovariance(), X; mean = med).sigma ≈ StatsAPI.fit(sm, X).sigma
    end
    @testset "Element type, scale, composition and online" begin
        # Each branch of the rho function takes the type of `t / c2`, and no closure of the
        # M-scale captures its loop variable.
        @test (@inferred POS.srho(BisquareRho(), 1.0f0, 2.0)) isa Float64
        @test (@inferred POS.srho(BisquareRho(), 3.0f0, 2.0)) isa Float64
        @test (@inferred POS.srho_weight(BisquareRho(), 3.0f0, 2.0)) isa Float64
        @test (@inferred POS.m_scale(BisquareRho(), Float32[1, 2, 3], 2.0, 0.5, 1.0f0)) isa
              Float64
        X32 = Float32.(X)
        for ce in (SCovariance(), MMCovariance())
            @test (@inferred StatsAPI.fit(ce, X)).sigma isa Matrix{Float64}
            # The steps stop at the tolerance of the type of the data.
            @test cov(ce, X32) ≈ cov(ce, X) rtol = 10 * sqrt(eps(Float32))
            # The stop rule of the location reads the metric of the scatter, so it does not
            # change with the units of the data. The S fit stops within its tolerance.
            @test cov(ce, 1e-2 .* X) ≈ 1e-4 .* cov(ce, X) rtol = 1e-7
            # One asset.
            r = StatsAPI.fit(ce, X[:, 1:1])
            @test size(r.sigma) == (1, 1)
            @test r.sigma[1, 1] > 0
            on = partial_fit!(POS.update_online_estimator(Online(ce)), X[1:60, :])
            on = partial_fit!(on, X[61:end, :])
            @test cov(on) ≈ cov(ce, X)
            @test !POS.supports_partial_fit(ce)
        end
        @test cov(MMCovariance(), 1e-2 .* X) ≈ 1e-4 .* cov(MMCovariance(), X) rtol = 1e-12
        w = eweights(1:100, 0.05)
        sm = SCovariance(; me = MedianExpectedReturns())
        @test POS.factory(sm, w).me.w === w
        @test POS.factory(MMCovariance(; s = sm), w).s.me.w === w
        @test POS.port_opt_view(MMCovariance(; s = sm), 1:2) isa MMCovariance
        # The `q` of the MM estimator defaults to the `q` of its S estimate.
        @test MMCovariance(; s = SCovariance(; q = 0.999)).q == 0.999
        @test MMCovariance(; s = SCovariance(; q = 0.999), q = 0.99).q == 0.99
    end
    @testset "MCD with the bisquare M-step" begin
        ce = MCDCovariance(; rw = BisquareReweighting())
        raw = StatsAPI.fit(MCDCovariance(; rw = nothing), X)
        r = StatsAPI.fit(ce, X)
        ref = POS.bisquare_mstep(BisquareReweighting(), X, raw.mu, raw.sigma, false, 0.975,
                                 POS.S_MAX_ITER)
        @test r.sigma ≈ ref.sigma
        @test det(r.sigma) ≈ det(raw.sigma)
        med = vec(median(X; dims = 1))
        @test StatsAPI.fit(ce, X; mean = med).mu ≈ med
    end
    @testset "Warnings and edge rules" begin
        k, b = POS.s_tuning(BisquareRho(), 0.5, 100, 3)
        @test k ≈ POS.bisquare_constant(0.5, 3)
        @test b == 0.5
        @test_logs (:warn, r"did not converge") POS.s_search(DetS(), BisquareRho(), X,
                                                             nothing, 52, k, b, 1)
        s = StatsAPI.fit(SCovariance(), X)
        @test_logs (:warn, r"did not converge") POS.bisquare_mstep(BisquareReweighting(), X,
                                                                   s.mu, s.sigma, false,
                                                                   0.975, 1)
        _, _, _, done = POS.s_isteps(BisquareRho(), X, vec(mean(X; dims = 1)), cov(X),
                                     false, k, b, 1)
        @test !done
        # A low efficiency moves the lower bound of the search of the constant.
        for eff in (0.01, 0.001)
            c2 = POS.mm_constant(BisquareReweighting(; eff = eff), 3)
            @test POS.bisquare_efficiency(ShapeEfficiency(), 3, c2) ≈ eff
        end
        @test POS.chisq_truncated_moment(3, Inf, [0.0, 1.0]) ≈ 3
        @test POS.s_refine(DetS(), BisquareRho(), X, [(s.mu, s.sigma)], false, k, b) ==
              [(s.mu, s.sigma)]
        @test POS.show_fields(SCovariance()) == (:me, :rho, :bdp, :alg, :q)
        @test POS.show_fields(MMCovariance()) == (:s, :rw, :q)
    end
    @testset "The other options (#2365)" begin
        # `CovSest(method = "rocke")`: its iterations stop about 1e-4 short of the minimum.
        r = StatsAPI.fit(SCovariance(; rho = RockeRho()), X)
        @test r.mu ≈ [0.389349062364076626, -0.068150707505010899, 0.448542613726419015] rtol = 1e-3
        @test r.sigma ≈ [3.61373552146385135 0.29932628507077286 0.30935797332677606
                         0.29932628507077286 1.73620520856611682 -0.76947690307871042
                         0.30935797332677611 -0.76947690307871042 2.35555360483499232] rtol = 1e-3
        @test median(r.d2) ≈ quantile(POS.Distributions.Chisq(3), 0.5)
        @test all(w -> 0 <= w <= 1, r.w)
        g, b = POS.s_tuning(RockeRho(), 0.5, 100, 3)
        @test g == 1
        @test b ≈ 0.485
        g8, _ = POS.s_tuning(RockeRho(), 0.5, 300, 8)
        @test g8 ≈ quantile(POS.Distributions.Chisq(8), 0.9) / 8 - 1
        @test POS.srho(RockeRho(), 1.0, g8) == 0.5
        @test POS.srho(RockeRho(), 1 - 2 * g8, g8) == 0
        @test POS.srho(RockeRho(), 1 + 2 * g8, g8) == 1
        @test POS.srho_weight(RockeRho(), 1.0, g8) == 1
        @test POS.srho_weight(RockeRho(), 1 + 2 * g8, g8) == 0
        # `CovMMest(eff.shape = FALSE)` from the start `sdet`.
        lo = BisquareReweighting(; target = LocationEfficiency())
        m = StatsAPI.fit(MMCovariance(; rw = lo), X)
        @test sqrt(POS.mm_constant(lo, 3)) ≈ 5.4902491720703317 rtol = 1e-7
        @test sqrt(POS.mm_constant(lo, 1)) ≈ 4.685 rtol = 1e-4
        @test m.mu ≈ [0.36552643973540061, -0.11309528305381154, 0.29215635967462883] rtol = 1e-6
        @test m.sigma ≈ [4.1586999642984121550 -0.0089014103776141138 1.4263024934877999161
                         -0.0089014103776141138 1.4290203470558620058 -0.5580078051460625188
                         1.4263024934877999161 -0.5580078051460625188 3.5309686315429109627] rtol = 1e-6
        # Each search reaches the minimum that `DetS` finds.
        s = StatsAPI.fit(SCovariance(), X)
        for alg in
            (FastS(; seed = 1), RobustStartS(), RobustStartS(; ce = TylerCovariance()))
            @test StatsAPI.fit(SCovariance(; alg = alg), X).sigma ≈ s.sigma rtol = 1e-6
        end
        @test cov(SCovariance(; alg = FastS(; seed = 7)), X) ==
              cov(SCovariance(; alg = FastS(; seed = 7)), X)
        med = vec(median(X; dims = 1))
        @test StatsAPI.fit(SCovariance(; alg = RobustStartS()), X; mean = med).mu ≈ med
        @test StatsAPI.fit(SCovariance(; alg = FastS(; seed = 1)), X; mean = med).sigma ≈
              StatsAPI.fit(SCovariance(), X; mean = med).sigma rtol = 1e-6
        # A draw of p + 1 rows of data with ties can be singular, so the start grows.
        Xt = round.(randn(StableRNG(3), 60, 2))
        rng = StableRNG(4)
        sizes = map(1:50) do _
            mu, S = POS.fasts_random_start(rng, Xt, nothing)
            !isnothing(POS.scatter_cholesky(S))
        end
        @test all(sizes)
        runs = POS.s_refine(FastS(; n_starts = 4, n_best = 2), BisquareRho(), X,
                            [(s.mu, s.sigma), (vec(mean(X; dims = 1)), cov(X)),
                             (s.mu, 2 .* s.sigma), (zeros(3), Matrix(1.0I, 3, 3))], false,
                            POS.s_tuning(BisquareRho(), 0.5, 100, 3)...)
        @test length(runs) == 2
        # The start estimator of `RobustStartS` takes the views and the weights.
        T = [2.0 0.1 0.0; 0.1 1.0 0.0; 0.0 0.0 0.5]
        ce = SCovariance(;
                         alg = RobustStartS(;
                                            ce = MCDCovariance(;
                                                               me = SimpleExpectedReturns(),
                                                               shr = MCDShrinkage(;
                                                                                  tgt = T),
                                                               ssc = nothing)))
        @test size(POS.port_opt_view(ce, [1, 3]).alg.ce.shr.tgt) == (2, 2)
        w = POS.StatsBase.Weights(fill(1.0, 100))
        @test POS.factory(ce, w).alg.ce.me.w === w
        @test_throws DomainError RockeRho(; arp = 1)
        @test_throws DomainError FastS(; n_starts = 0)
        @test_throws DomainError FastS(; n_starts = 3, n_best = 5)
    end
    @testset "Refusals" begin
        @test_throws DomainError SCovariance(; bdp = 0.6)
        @test_throws DomainError SCovariance(; bdp = 0)
        @test_throws DomainError SCovariance(; q = 1)
        @test_throws DomainError MMCovariance(; q = 0)
        # An `Online` wrapper reads the `cache` of the MM estimator, never the one of `s`.
        @test_throws ArgumentError MMCovariance(;
                                                s = POS.update_online_estimator(Online(SCovariance())))
        @test_throws DomainError BisquareReweighting(; eff = 1)
        @test_throws ArgumentError MCDCovariance(; shr = MCDShrinkage(), ssc = nothing,
                                                 rw = BisquareReweighting())
        # The start subsets of h = (n + p + 1) ÷ 2 rows need h > p.
        @test_throws DomainError StatsAPI.fit(SCovariance(), randn(StableRNG(1), 3, 3))
        @test_throws DomainError StatsAPI.fit(MMCovariance(), randn(StableRNG(1), 3, 3))
        Z = randn(StableRNG(2), 40, 3)
        Z[:, 2] .= 1
        @test_throws DomainError StatsAPI.fit(SCovariance(), Z)
        @test_throws DomainError POS.unit_determinant!(zeros(2, 2))
        Y = copy(X)
        Y[1, 1] = NaN
        @test_throws POS.IsNonFiniteError StatsAPI.fit(SCovariance(), Y)
    end
end
