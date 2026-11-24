#=
The shrinkage of the robust covariance estimators of #2256: the minimum regularised covariance
determinant (`MCDShrinkage`) and the shrunk Tyler shape (`TylerShrinkage`), with the rules of
the shrinkage intensity.

The tests check the equations that define each estimator against the fit, each rule against the
formula of its source computed in another way, the limits of the intensity, the resistance to
outliers with fewer observations than assets, and the refusals.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs, Random

const POR = PortfolioOptimisers
const StatsAPI = PortfolioOptimisers.StatsAPI

# One factor and idiosyncratic noise, with assets of different volatility.
function shrinkage_test_sample(rng, n, p, nbad; shift = 8)
    B = 0.8 .+ 0.3 .* randn(rng, p)
    S = B * B' * 0.3 + Diagonal(0.5 .+ rand(rng, p))
    sd = exp.(0.4 .* randn(rng, p))
    S = Matrix(Symmetric(Diagonal(sd) * S * Diagonal(sd)))
    X = Matrix((cholesky(S).L * randn(rng, p, n))')
    bad = sort(randperm(rng, n)[1:nbad])
    X[bad, :] .+= shift .* sd' .* sign.(randn(rng, 1, p))
    return X, S, bad
end
relerr(A, B) = norm(A - B) / norm(B)
sqdist(X, mu, S) = [(x - mu)' * (S \ (x - mu)) for x in eachrow(X)]
signs(W) = W ./ sqrt.(sum(abs2, W; dims = 2))

@testset "Robust shrinkage (#2256)" begin
    @testset "Fewer observations than assets" begin
        for seed in 1:2
            X, S, bad = shrinkage_test_sample(StableRNG(seed), 50, 70, 5)
            @test_throws "MCDShrinkage" cov(MCDCovariance(), X)
            @test_throws "TylerShrinkage" cov(TylerCovariance(), X)
            for intensity in (ConditionNumberIntensity(), SphericityIntensity(), 0.3),
                tgt in (DiagonalTarget(), ConstantCorrelationTarget()),
                rw in (nothing, MCDReweighting())

                ce = MCDCovariance(; rw = rw, shr = MCDShrinkage(; intensity, tgt))
                r = StatsAPI.fit(ce, X)
                @test isposdef(r.sigma)
                @test issymmetric(r.sigma)
                @test bad ⊆ findall(r.outlier)
                @test count(r.outlier) <= length(bad) + 8
                @test all(iszero, r.w[bad])
                @test relerr(r.sigma, S) < relerr(cov(X), S) / 5
                @test cov(ce, X) == r.sigma
                @test 0 < r.intensity < 1
            end
            # The default reweights, and the reweighting makes the error smaller.
            raw = MCDCovariance(; rw = nothing, shr = MCDShrinkage())
            full = MCDCovariance(; shr = MCDShrinkage())
            @test relerr(cov(full, X), S) < relerr(cov(raw, X), S)
            fm = MCDCovariance(; alg = FastMCD(; seed = 1, n_starts = 60),
                               shr = MCDShrinkage())
            @test bad ⊆ findall(StatsAPI.fit(fm, X).outlier)
            # One outlier row: Tyler's shape does not resist a cluster of them.
            Y, = shrinkage_test_sample(StableRNG(seed), 50, 70, 0)
            Y[7, :] .+= 10 .* vec(std(Y; dims = 1)) .* sign.(randn(StableRNG(seed), 70))
            for alg in (TraceNormalisedShrinkage(), PenalisedShrinkage()),
                intensity in
                (OracleIntensity(), AsymptoticIntensity(), SphericityIntensity(),
                 ConditionNumberIntensity(), 0.6)

                ce = TylerCovariance(; shr = TylerShrinkage(; alg, intensity))
                r = StatsAPI.fit(ce, Y)
                @test isposdef(r.sigma)
                @test issymmetric(r.sigma)
                @test r.outlier[7]
                @test count(r.outlier) <= 8
                @test cov(ce, Y) == r.sigma
                @test 0 < r.intensity <= 1
                @test r.w[7] < median(r.w)
            end
        end
    end
    @testset "The equations of the shrunk MCD" begin
        X, = shrinkage_test_sample(StableRNG(4), 80, 30, 8)
        s = map(x -> POR.robust_column_scale(x .- median(x)), eachcol(X))
        for (tgt, T) in ((DiagonalTarget(), Matrix(Diagonal(s .^ 2))),
                         (IdentityTarget(), Matrix(1.0I, 30, 30)),
                         (ScaledIdentityTarget(), Matrix(mean(s .^ 2) * I, 30, 30)))
            ce = MCDCovariance(; alpha = 0.75, rw = nothing,
                               shr = MCDShrinkage(; tgt, intensity = 0.2))
            r = StatsAPI.fit(ce, X)
            idx = findall(isone, r.w)
            @test length(idx) == 60
            c = POR.mcd_consistency(30, 60 / 80)
            @test r.mu ≈ vec(mean(X[idx, :]; dims = 1))
            @test r.sigma ≈ 0.2 * T + 0.8 * c * cov(X[idx, :])
            # The subset is a fixed point of the shrunk concentration step.
            d2 = sqdist(X, r.mu, r.sigma)
            @test sort(partialsortperm(d2, 1:60)) == idx
            # A distance outside the subset is the distance of the fit, and one inside is the
            # distance from the fit that leaves the observation out.
            out = setdiff(1:80, idx)
            @test r.d2[out] ≈ d2[out]
            i = idx[5]
            rest = setdiff(idx, i)
            S = cov(X[rest, :]) * (58 / 59)
            @test r.d2[i] ≈ only(sqdist(X[i:i, :], vec(mean(X[rest, :]; dims = 1)),
                                        0.2 * T + 0.8 * c * S))
        end
        # The equicorrelation target reads the Kendall correlations.
        K = sinpi.(corkendall(X) ./ 2)
        cbar = (sum(K) - 30) / (30 * 29)
        T = s .* (cbar .+ (1 - cbar) * I(30)) .* s'
        smp = POR.mcd_sample(MCDShrinkage(; tgt = ConstantCorrelationTarget()), X, nothing)
        @test smp.tgt ≈ T
        # In the coordinates of the search, the target is the identity matrix.
        @test POR.shrinkage_target(T, T) === T
        L = smp.Z \ (first(POR.mcd_standardise(X, nothing)))
        @test L * L' ≈ Diagonal(inv.(s)) * T * Diagonal(inv.(s))
        # A fixed target gives the rule that builds it, and a view slices it.
        fixed = MCDShrinkage(; tgt = T, intensity = 0.2)
        rule = MCDShrinkage(; tgt = ConstantCorrelationTarget(), intensity = 0.2)
        @test cov(MCDCovariance(; shr = fixed), X) ≈ cov(MCDCovariance(; shr = rule), X)
        v = POR.port_opt_view(MCDCovariance(; shr = fixed), 1:4)
        @test v.shr.tgt == T[1:4, 1:4]
        @test POR.port_opt_view(rule, 1:4) === rule
        # A given centre fixes the location, and the divisor is the size of the subset.
        m0 = vec(median(X; dims = 1))
        for alg in (DetMCD(), FastMCD(; seed = 2, n_starts = 5, n_best = 2)),
            intensity in (0.2, SphericityIntensity(), ConditionNumberIntensity())

            ce = MCDCovariance(; alg, me = MedianExpectedReturns(), rw = nothing,
                               shr = MCDShrinkage(; intensity))
            r = StatsAPI.fit(ce, X)
            @test r.mu == m0
            idx = findall(isone, r.w)
            D = X[idx, :] .- m0'
            c = POR.mcd_consistency(30, length(idx) / 80)
            @test r.sigma ≈
                  r.intensity * Diagonal(s .^ 2) + (1 - r.intensity) * c * D'D / length(idx)
        end
        # Every observation, and one variable: no search.
        full = MCDCovariance(; alpha = 1, rw = nothing,
                             shr = MCDShrinkage(; intensity = 0.2))
        @test cov(full, X) ≈ 0.2 * Diagonal(s .^ 2) + 0.8 * cov(X)
        one_asset = MCDCovariance(; rw = nothing, shr = MCDShrinkage(; intensity = 0.2))
        r1 = StatsAPI.fit(one_asset, X[:, 1:1])
        sx = sort(X[:, 1])
        best = minimum(var(sx[k:(k + 40)]) for k in 1:40)
        @test r1.sigma[1] ≈ 0.2 * s[1]^2 + 0.8 * best * POR.mcd_consistency(1, 41 / 80)
        # A full shrinkage gives the target.
        @test cov(MCDCovariance(; shr = MCDShrinkage(; intensity = 1)), X) ≈
              Diagonal(s .^ 2)
    end
    @testset "Parity with R" begin
        # R 4.5.3, `rrcov` 1.7-7, `CovMrcd(X, h = 21, rho = 0.2)` and
        # `CovMrcd(X, h = 30, maxcond = 3)`. The `rho` of the second comes from `uniroot`, whose
        # tolerance is 1e-4.
        rng = StableRNG(31)
        A = randn(rng, 4, 4)
        X = Matrix((cholesky(A * A' + I).L * randn(rng, 4, 40))')
        X[1:5, :] .= 8 .+ 0.5 .* randn(rng, 5, 4)
        r = StatsAPI.fit(MCDCovariance(; rw = nothing,
                                       shr = MCDShrinkage(; intensity = 0.2)), X)
        @test findall(isone, r.w) ==
              [8, 9, 10, 12, 14, 15, 16, 17, 21, 22, 23, 27, 29, 31, 32, 33, 34, 35, 37, 38,
               39]
        @test r.mu ≈ [-0.33338082753357945, -0.47027683541339932, 0.56778163390352920,
                      0.47786265858703270]
        @test r.sigma ≈
              [3.5500840474784909 -2.8160314717250192 2.1994560623879384 -1.9355857577468434
               -2.8160314717250192 10.4815945061872728 -5.3366864937287000 3.1272921716744997
               2.1994560623879384 -5.3366864937287000 8.3714382582355533 -1.4406273009269495
               -1.9355857577468432 3.1272921716744997 -1.4406273009269492 5.4982509016698886]
        rule = ConditionNumberIntensity(; max_cond = 3)
        r = StatsAPI.fit(MCDCovariance(; alpha = 0.75, rw = nothing,
                                       shr = MCDShrinkage(; intensity = rule)), X)
        @test isapprox(r.intensity, 0.35337745260028769; atol = 1e-4)
        @test r.mu ≈ [-0.063857872746176936, -0.813832938088058233, 0.612016875529001636,
                      0.126419103613180095]
        @test isapprox(r.sigma,
                       [3.8139090487273597 -2.0062676011708649 1.3236865189080627 -1.2591896664615763
                        -2.0062676011708649 10.4996461570338138 -3.4288638283055062 0.7964069413710438
                        1.3236865189080627 -3.4288638283055057 8.1387548943498551 -0.4510623185940757
                        -1.2591896664615765 0.7964069413710438 -0.4510623185940757 6.1823144494176931];
                       rtol = 1e-4)
    end
    @testset "The search of the shrunk MCD" begin
        X, = shrinkage_test_sample(StableRNG(9), 50, 70, 5)
        smp = POR.mcd_sample(MCDShrinkage(), X, nothing)
        Z = smp.Z
        h, c = 26, POR.mcd_consistency(70, 26 / 50)
        @test POR.mcd_subset_size(MCDShrinkage(), 0.5, 50, 70) == h
        @test POR.mcd_subset_size(MCDShrinkage(), 0.75, 50, 70) == 38
        @test POR.mcd_subset_size(MCDShrinkage(), 0.5, 51, 70) == 26
        # The ceiling is exact: 0.55 * 100 is 55.00000000000001.
        @test all(k -> POR.mcd_subset_size(MCDShrinkage(), k / 100, 100, 70) == k, 51:100)
        # A shrunk concentration step never raises the determinant. With fewer observations than
        # assets, the subset of a start is often its own fixed point.
        Q = first(POR.mcd_standardise(first(shrinkage_test_sample(StableRNG(2), 120, 10,
                                                                  12)), nothing))
        reg = POR.subset_shrinkage(0.2, POR.mcd_consistency(10, 61 / 120), 10)
        start = collect(1:61)
        o0 = last(POR.mcd_csteps(Q, start, 61, false, reg, 0))
        o1 = last(POR.mcd_csteps(Q, start, 61, false, reg, 1))
        o9 = last(POR.mcd_csteps(Q, start, 61, false, reg, 200))
        @test o9 <= o1 < o0
        # The starts do not depend on the basis that the linear algebra library takes for the null
        # space of a singular correlation estimate, and a regular estimate keeps every direction.
        Ps = POR.detmcd_directions(Z)
        @test size(Ps[1], 2) < 70
        @test size(Ps[4], 2) == 50
        @test all(P -> P' * P ≈ I, Ps)
        Y = Z * Ps[1]
        @test POR.outside_directions(Z, Y, Ps[1]) ≈
              vec(sum(abs2, Z * nullspace(Matrix(Ps[1]')); dims = 2))
        @test iszero(POR.outside_directions(Q, Q, Matrix(1.0I, 10, 10)))
        @test size(POR.regular_directions([1.0 2.0; 2.0 1.0])) == (2, 2)
        @test size(POR.regular_directions([1.0 1.0; 1.0 1.0])) == (2, 1)
        o = randperm(StableRNG(1), 70)
        @test map(sort, POR.mcd_starts(DetMCD(), Z[:, o], h, false, 0.1)) ==
              map(sort, POR.mcd_starts(DetMCD(), Z, h, false, 0.1))
        # The pooled intensity: the largest one of the starts that stay.
        starts = [collect(i:(i + h - 1)) for i in 1:6]
        rho, kept = POR.pooled_intensity(0.05, Z, starts, c, false)
        @test rho == 0.05 && length(kept) == 6
        rho, kept = POR.pooled_intensity(0.4, Z, starts, c, false)
        @test rho == 0.4 && length(kept) == 6
        own = [POR.mcd_intensity(smp.intensity, Z, idx, c, false) for idx in starts]
        rho, kept = POR.pooled_intensity(smp.intensity, Z, starts, c, false)
        @test kept == starts[own .<= max(0.1, median(own))]
        @test rho == maximum(own[own .<= max(0.1, median(own))])
        loose = ConditionNumberIntensity(; max_cond = 1e6)
        rho, kept = POR.pooled_intensity(loose, Z, starts, c, false)
        @test rho < 0.1 && length(kept) == 6
        # The closed form of the condition number.
        for idx in starts[1:2], max_cond in (50, 10.0)
            r = POR.mcd_intensity(ConditionNumberIntensity(; max_cond), Z, idx, c, false)
            @test cond(r * I + (1 - r) * c * cov(Z[idx, :])) ≈ max_cond
        end
        # A well-conditioned subset gets no shrinkage, and the estimate is then the MCD.
        Y = randn(StableRNG(3), 400, 3)
        @test iszero(POR.mcd_intensity(ConditionNumberIntensity(), Y, 1:200, 1.0, false))
        free = StatsAPI.fit(MCDCovariance(; rw = nothing, shr = MCDShrinkage()), Y)
        @test iszero(free.intensity)
        idx = findall(isone, free.w)
        @test free.sigma ≈ POR.mcd_consistency(3, 201 / 400) * cov(Y[idx, :])
        # The reweighted fit meets the bound too.
        r = StatsAPI.fit(MCDCovariance(; shr = MCDShrinkage()), X)
        sd = map(x -> POR.robust_column_scale(x .- median(x)), eachcol(X))
        @test cond(Diagonal(inv.(sd)) * r.sigma * Diagonal(inv.(sd))) <= 50 * (1 + 1e-8)
        # The sphericity at the normal model.
        idx = starts[1]
        D = Z .- median(Z; dims = 1)
        G = signs(D)' * signs(D) / 50
        g = clamp(50 / 49 * (70 * tr(G * G) - 70 / 50), 1, 70)
        b = ((g + 70) / 25)
        @test POR.mcd_intensity(SphericityIntensity(), Z, idx, c, false) ≈ b / (g - 1 + b)
    end
    @testset "The equations of the shrunk Tyler shape" begin
        X, = shrinkage_test_sample(StableRNG(5), 40, 60, 0)
        n, p = size(X)
        m0 = vec(median(X; dims = 1))
        W = (X .- m0') ./ POR.quantile_scale.(eachcol(X .- m0'), nothing)'
        V = POR.tyler_shape(POR.ShapeShrinkage(TraceNormalisedShrinkage(), 0.3), W, nothing)
        d2 = sqdist(W, zeros(p), V)
        B = 0.7 * p / n * W' * (W ./ d2) + 0.3 * I
        @test isapprox(p * B / tr(B), V; rtol = 1e-6)
        U = POR.tyler_shape(POR.ShapeShrinkage(PenalisedShrinkage(), 0.6), W, nothing)
        d2 = sqdist(W, zeros(p), U)
        @test isapprox(0.4 * p / n * W' * (W ./ d2) + 0.6 * I, U; rtol = 1e-6)
        @test isapprox(tr(inv(U)), p; rtol = 1e-6)
        # The two forms give one set of shapes.
        U = POR.tyler_shape(POR.ShapeShrinkage(PenalisedShrinkage(), 1 - 0.7 * p / tr(B)),
                            W, nothing)
        @test isapprox(p * U / tr(U), V; rtol = 1e-6)
        # The estimate: the shape in the coordinates of the target, and the trace of the robust
        # variances.
        for alg in (TraceNormalisedShrinkage(), PenalisedShrinkage())
            r = StatsAPI.fit(TylerCovariance(; me = MedianExpectedReturns(),
                                             shr = TylerShrinkage(; alg, intensity = 0.6)),
                             X)
            v = POR.quantile_scale.(eachcol(X .- m0'), nothing) .^ 2
            V = Diagonal(sqrt.(v)) *
                POR.tyler_shape(POR.ShapeShrinkage(alg, 0.6), W, nothing) *
                Diagonal(sqrt.(v))
            @test r.mu == m0
            @test r.sigma ≈ V * sum(v) / tr(V)
            @test tr(r.sigma) ≈ sum(v)
            @test r.intensity == 0.6
        end
        # The joint location is the spatial median in the metric of the shape.
        r = StatsAPI.fit(TylerCovariance(; shr = TylerShrinkage()), X)
        d = sqrt.(sqdist(X, r.mu, r.sigma))
        @test norm(sum((X .- r.mu') ./ d; dims = 1)) < 1e-4
        # A target that is not diagonal, and a fixed target, whose scale has no effect.
        v = POR.quantile_scale.(eachcol(X .- m0'), nothing) .^ 2
        R = cov2cor(signs(W)' * signs(W))
        cbar = (sum(R) - p) / (p * (p - 1))
        T = sqrt.(v) .* (cbar .+ (1 - cbar) * I(p)) .* sqrt.(v)'
        est(tgt) = TylerCovariance(; shr = TylerShrinkage(; tgt, intensity = 0.5))
        @test cov(est(ConstantCorrelationTarget()), X) ≈ cov(est(T), X)
        @test isapprox(cov(est(3 * T), X), cov(est(T), X); rtol = 1e-6)
        @test isapprox(cov(est(IdentityTarget()), X), cov(est(ScaledIdentityTarget()), X);
                       rtol = 1e-6)
        @test POR.port_opt_view(est(T), 1:3).shr.tgt == T[1:3, 1:3]
        # A given variance estimator, and a full shrinkage.
        sv = cov(TylerCovariance(; ve = SimpleVariance(), shr = TylerShrinkage()), X)
        @test sqrt.(diag(sv)) ≈ vec(std(X; dims = 1))
        one_rho = TylerCovariance(; shr = TylerShrinkage(; intensity = 1))
        @test cov(one_rho, X) ≈ Diagonal(v)
        # More observations than assets and an intensity of zero: the shape of Tyler.
        Y, = shrinkage_test_sample(StableRNG(6), 200, 4, 0)
        for alg in (TraceNormalisedShrinkage(), PenalisedShrinkage())
            z = TylerCovariance(; shr = TylerShrinkage(; alg, intensity = 0))
            @test isapprox(cor(z, Y), cor(TylerCovariance(), Y); rtol = 1e-6)
        end
        # The scale of an asset that does not trade on most days.
        @test POR.quantile_scale([0.0, 0.0, 0.0, 1.0, -2.0], nothing) > 0
    end
    @testset "The rules of the Tyler intensity" begin
        X, = shrinkage_test_sample(StableRNG(8), 40, 60, 0)
        n, p = size(X)
        W = (X .- median(X; dims = 1)) ./ std(X; dims = 1)
        Rt = p / n * signs(W)' * signs(W)
        t = tr(Rt * Rt)
        tn, pn = TraceNormalisedShrinkage(), PenalisedShrinkage()
        ti(rule, alg) = POR.tyler_intensity(rule, alg, W, nothing)
        @test ti(0.25, tn) == 0.25
        # The oracle of each form.
        @test ti(OracleIntensity(), tn) ≈
              (p^2 + (1 - 2 / p) * t) / ((p^2 - n * p - 2n) + (n + 1 + 2 * (n - 1) / p) * t)
        oracle(V, n, p) = (a = p - 2 + p * tr(V);
                           a / (a + n * (p + 2) * (tr(inv(V)^2) / p - 1)))
        pilot = POR.tyler_shape(POR.ShapeShrinkage(pn, 1 - n / (2p)), W, nothing)
        @test isapprox(ti(OracleIntensity(), pn), oracle(pilot, n, p); rtol = 1e-5)
        Y = randn(StableRNG(1), 100, 5)
        V = POR.tyler_shape(nothing, Y, nothing)
        V *= tr(inv(V)) / 5
        @test POR.tyler_intensity(OracleIntensity(), pn, Y, nothing) ≈ oracle(V, 100, 5)
        # The asymptotic rule reaches its equation within the width of the bisection.
        r = p / n / (t / p - 1)
        rho = ti(AsymptoticIntensity(), tn)
        function gap(rho)
            a = mean(sqdist(W, zeros(p),
                            POR.tyler_shape(POR.ShapeShrinkage(tn, rho), W, nothing)) ./
                     vec(sum(abs2, W; dims = 2)))
            return rho * a / (1 - rho + rho * a) - r
        end
        @test gap(rho) >= 0 > gap(rho - 2^-10)
        rho = ti(AsymptoticIntensity(), pn)
        weight(rho) = rho * p / tr(POR.tyler_shape(POR.ShapeShrinkage(pn, rho), W, nothing))
        step = (n / p) / 2^10
        @test weight(rho) >= r > weight(rho - step)
        # The sphericity.
        g = clamp(n / (n - 1) * (t / p - p / n), 1, p)
        beta = n * (g - 1) / (n * (g - 1) * (1 - 1 / n) + (p - 1) / (p + 2) * (2g + p))
        @test ti(SphericityIntensity(), tn) ≈ 1 - beta
        rho = ti(SphericityIntensity(), pn)
        @test weight(rho) >= 1 - beta > weight(rho - step)
        # The condition number.
        for (alg, st) in ((tn, 2^-10), (pn, step))
            rho = ti(ConditionNumberIntensity(; max_cond = 20), alg)
            @test cond(POR.tyler_shape(POR.ShapeShrinkage(alg, rho), W, nothing)) <=
                  20 <
                  cond(POR.tyler_shape(POR.ShapeShrinkage(alg, rho - st), W, nothing))
        end
        @test iszero(POR.tyler_intensity(ConditionNumberIntensity(), tn, Y, nothing))
        # Data that are spherical: the target is the answer.
        E = Matrix(1.0I, 6, 6)
        E = vcat(E, -E)
        for alg in (tn, pn)
            @test isone(POR.tyler_intensity(AsymptoticIntensity(), alg, E, nothing))
            @test isone(POR.tyler_intensity(SphericityIntensity(), alg, E, nothing))
        end
        # The normal sample: the oracle is near to the intensity of the smallest error.
        errs(rho) = mean(begin
                             Xs, S = shrinkage_test_sample(StableRNG(k), 40, 40, 0)
                             C = cov(TylerCovariance(;
                                                     shr = TylerShrinkage(;
                                                                          intensity = rho)),
                                     Xs)
                             relerr(C / tr(C), S / tr(S))
                         end
                         for k in 1:3)
        grid = minimum(errs, (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8))
        for rule in (OracleIntensity(), AsymptoticIntensity(), SphericityIntensity())
            @test errs(rule) < 1.15 * grid
        end
    end
    @testset "Observation weights" begin
        X, = shrinkage_test_sample(StableRNG(2), 30, 45, 0)
        k = rand(StableRNG(2), 1:3, 30)
        Xr = reduce(vcat, [repeat(X[i:i, :], k[i]) for i in 1:30])
        # Frequency weights repeat the observations.
        for intensity in
            (0.5, OracleIntensity(), SphericityIntensity(), AsymptoticIntensity())
            shr = TylerShrinkage(; intensity)
            a = StatsAPI.fit(TylerCovariance(; shr, w = fweights(k)), X)
            b = StatsAPI.fit(TylerCovariance(; shr), Xr)
            @test isapprox(a.sigma, b.sigma; rtol = 1e-6)
            @test a.intensity ≈ b.intensity
        end
        ew = eweights(1:30, 0.05)
        @test isposdef(cov(TylerCovariance(; shr = TylerShrinkage(), w = ew), X))
        # The bound of the penalised form reads the largest share of the weight.
        @test POR.tyler_intensity_bound(X, nothing) ≈ 1 - 30 / 45
        @test POR.tyler_intensity_bound(X, ew) ≈ 1 - sum(ew) / (45 * maximum(ew))
        @test POR.tyler_intensity_bound(randn(StableRNG(1), 20, 4), nothing) == -4
    end
    @testset "Outlier flags and composition" begin
        # The cut-off with no shrinkage is the chi-squared quantile.
        @test POR.robust_outliers(nothing, [1.0, 8.0], 2, 0.975) == [false, true]
        d2 = vcat(fill(5.0, 3), collect(4.0:0.1:6.0), 60.0)
        @test findall(POR.robust_outliers(0.1, d2, 2, 0.975)) == [length(d2)]
        # A fit with no shrinkage keeps its distances.
        @test POR.leave_one_out!([1.0, 2.0], [1], 1.5, false) == [1.0, 2.0]
        @test isnothing(POR.mcd_intensity(nothing, d2', [1], 1.0, false))
        @test isnothing(POR.subset_shrinkage(nothing, 1.0, 2))
        S = [2.0 0.5; 0.5 1.0]
        @test POR.shrink_scatter!(copy(S), nothing) == S
        @test POR.shrink_scatter!(copy(S), 3.0) == 3 * S
        X, = shrinkage_test_sample(StableRNG(1), 40, 50, 0)
        X32 = Float32.(X)
        for ce in (MCDCovariance(; shr = MCDShrinkage()),
                   MCDCovariance(; shr = MCDShrinkage(; intensity = 0.2f0)),
                   MCDCovariance(;
                                 shr = MCDShrinkage(; tgt = ConstantCorrelationTarget(),
                                                    intensity = SphericityIntensity())),
                   TylerCovariance(; shr = TylerShrinkage()),
                   TylerCovariance(;
                                   shr = TylerShrinkage(; alg = PenalisedShrinkage(),
                                                        intensity = 0.8f0)))
            @test eltype(cov(ce, X32)) == Float32
            @test cov(ce, Matrix(X'); dims = 2) ≈ cov(ce, X)
            pr = prior(EmpiricalPrior(; me = RobustExpectedReturns(; ce = ce), ce = ce), X)
            @test pr.sigma ≈ cov(ce, X)
            @test pr.mu ≈ StatsAPI.fit(ce, X).mu
            on = partial_fit!(POR.update_online_estimator(Online(ce)), X[1:20, :])
            on = partial_fit!(on, X[21:end, :])
            @test cov(on) ≈ cov(ce, X)
            @test POR.factory(ce, eweights(1:40, 0.05)).shr === ce.shr
        end
        @test POR.port_opt_view(MCDCovariance(), 1:2).shr === nothing
        tyler_rule = TylerShrinkage()
        @test POR.port_opt_view(tyler_rule, 1:2) === tyler_rule
    end
    @testset "A scatter inside the bound of the condition number" begin
        # The closed form of the intensity has the denominator `max_cond - 1 + a`, and a scatter
        # inside the bound has a negative `a`. The denominator is then zero or negative.
        Y = randn(StableRNG(3), 400, 3)
        for s in (0.9, 1.0, 1.05, 1.2, 10.0)
            @test iszero(POR.mcd_intensity(ConditionNumberIntensity(), s * Y, 1:400, 1.0,
                                           false))
        end
        for seed in 1:40
            r = StatsAPI.fit(MCDCovariance(; shr = MCDShrinkage()),
                             randn(StableRNG(seed), 100, 2))
            @test iszero(r.intensity)
            @test isposdef(r.sigma)
        end
        wide = StatsAPI.fit(MCDCovariance(; shr = MCDShrinkage(; tgt = IdentityTarget())),
                            10 * Y)
        @test iszero(wide.intensity)
        @test relerr(wide.sigma, cov(10 * Y)) < 0.2
    end
    @testset "One asset" begin
        # A window of one asset, such as a cluster of one. Every rule gives an estimate.
        x = 10 * randn(StableRNG(3), 400, 1)
        for intensity in (ConditionNumberIntensity(), SphericityIntensity(), 0.3)
            r = StatsAPI.fit(MCDCovariance(; shr = MCDShrinkage(; intensity)), x)
            @test 0 <= r.intensity <= 1
            @test 70 < only(r.sigma) < 130
        end
        for alg in (TraceNormalisedShrinkage(), PenalisedShrinkage()),
            intensity in (OracleIntensity(), AsymptoticIntensity(), SphericityIntensity(),
                          ConditionNumberIntensity(), 0.3)

            r = StatsAPI.fit(TylerCovariance(; shr = TylerShrinkage(; alg, intensity)), x)
            @test 0 <= r.intensity <= 1
            @test only(r.sigma) ≈ POR.quantile_scale(vec(x) .- median(x), nothing)^2
        end
        @test isone(POR.tyler_intensity(OracleIntensity(), PenalisedShrinkage(), x,
                                        nothing))
    end
    @testset "An asset that does not trade" begin
        X = randn(StableRNG(1), 30, 40)
        X[1:20, 3] .= 0
        # Frequency weights repeat the observations, and a zero weight removes one.
        k = rand(StableRNG(2), 0:3, 30)
        Xr = reduce(vcat, [repeat(X[i:i, :], k[i]) for i in 1:30])
        a = cov(TylerCovariance(; shr = TylerShrinkage(), w = fweights(k)), X)
        b = cov(TylerCovariance(; shr = TylerShrinkage()), Xr)
        @test isapprox(a, b; rtol = 1e-6)
        @test POR.quantile_scale(X[:, 3], fweights(k)) ≈
              POR.quantile_scale(Xr[:, 3], nothing)
        # An asset with no robust scale has no variance in the pilot, so a target that reads its
        # variance is refused. The estimate does not invent a variance for it.
        X *= 0.01
        X[:, 3] .= 0.001
        @test iszero(POR.quantile_scale(zeros(10), nothing))
        @test isone(POR.robust_column_scale(zeros(10)))
        for tgt in (DiagonalTarget(), ConstantCorrelationTarget())
            @test_throws "no robust variance" cov(TylerCovariance(;
                                                                  shr = TylerShrinkage(;
                                                                                       tgt)),
                                                  X)
        end
        S = cov(TylerCovariance(; shr = TylerShrinkage(; tgt = ScaledIdentityTarget())), X)
        @test isposdef(S)
        @test 0 < S[3, 3] < 1.5 * 0.01^2
        Y = randn(StableRNG(1), 400, 3)
        Y[2:400, 3] .= 0
        @test_throws "no robust variance" cov(MCDCovariance(; shr = MCDShrinkage()), Y)
        near = MCDCovariance(;
                             shr = MCDShrinkage(; tgt = Matrix(1.0I, 3, 3),
                                                intensity = 0.1))
        @test isposdef(cov(near, Y))
    end
    @testset "The starts of FastMCD about a fixed centre" begin
        X, = shrinkage_test_sample(StableRNG(2), 50, 70, 5)
        smp = POR.mcd_sample(MCDShrinkage(), X, vec(median(X; dims = 1)))
        fm = FastMCD(; seed = 1, n_starts = 20)
        @test length(unique(map(sort, POR.mcd_starts(fm, smp.Z, 26, true, 0.2)))) > 10
    end
    @testset "Refusals" begin
        rng = StableRNG(3)
        X = randn(rng, 20, 30)
        @test_throws DomainError MCDShrinkage(; intensity = 1.2)
        @test_throws DomainError TylerShrinkage(; intensity = -0.1)
        @test_throws DomainError ConditionNumberIntensity(; max_cond = 1)
        @test_throws TypeError MCDShrinkage(; intensity = OracleIntensity())
        @test_throws DimensionMismatch MCDShrinkage(; tgt = ones(2, 3))
        @test_throws DomainError TylerShrinkage(; tgt = [1.0 2.0; 2.0 1.0])
        @test_throws DimensionMismatch cov(MCDCovariance(;
                                                         shr = MCDShrinkage(;
                                                                            tgt = [1.0 0.0;
                                                                                   0.0 1.0])),
                                           X)
        @test_throws "ssc = nothing" MCDCovariance(; shr = MCDShrinkage(),
                                                   ssc = MCDSmallSampleCorrection())
        @test isnothing(MCDCovariance(; shr = MCDShrinkage()).ssc)
        @test_throws DomainError cov(MCDCovariance(; shr = MCDShrinkage()), X[1:1, :])
        # No shrinkage on fewer observations than assets.
        zero_mcd = MCDCovariance(; shr = MCDShrinkage(; intensity = 0))
        @test_throws DomainError cov(zero_mcd, X)
        for alg in (TraceNormalisedShrinkage(), PenalisedShrinkage())
            ce = TylerCovariance(; shr = TylerShrinkage(; alg, intensity = 0))
            @test_throws DomainError cov(ce, X)
            # With as many observations as assets, the shape with no shrinkage is not unique.
            @test_throws DomainError cov(ce, randn(StableRNG(1), 20, 20))
        end
        @test_throws "two observations" cov(TylerCovariance(; shr = TylerShrinkage()),
                                            X[1:1, :])
        @test_throws "two observations" cov(TylerCovariance(; shr = TylerShrinkage(),
                                                            w = fweights([3;
                                                                          zeros(Int, 19)])),
                                            X)
        low = TylerCovariance(;
                              shr = TylerShrinkage(; alg = PenalisedShrinkage(),
                                                   intensity = 0.2))
        @test_throws "TraceNormalisedShrinkage" cov(low, X)
        # A target rule can build a matrix that is not positive definite.
        @test_throws "not positive definite" POR.robust_target(ConstantCorrelationTarget(),
                                                               [1.0 -0.9 -0.9;
                                                                -0.9 1.0 -0.9;
                                                                -0.9 -0.9 1.0])
    end
end
