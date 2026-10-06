#=
The recovery numbers of the oracle's test suite, class by class, on the library's own draws
(#1391, map #1375).

Each testset reproduces one class of the oracle's statistical recovery tests: the same model of
the returns, the same sizes and the same configuration of the prior, drawn here with a
`StableRNG` of the class's seed. Each asserts the oracle's own thresholds on the recovered
quantities, and the exact identities that the class states. The classes are named as the oracle
numbers them. The covariance identities that every class repeats, `sigma = M F M' + D` and the
square root, are unit identities that `test_12k` and `test_12w` already pin, so they are not
repeated here.

The configuration maps one to one: a passthrough factor is `CompositeExposure` with no outlier
step and no scoring; the benchmark power is `bp`; the regression power is `MarketCapWeights(; p)`;
a sample factor prior is `EmpiricalPrior()`, and the default one is `PARITY_PE`, with
`PARITY_VE` as the idiosyncratic variance; the constrained family is `families`, the
neutralisation `neutralise`, the idiosyncratic correlation threshold `th`, the inverse-variance
weights `BlendedInverseVarianceWeights`, and the alpha a `CustomValueReturnForecast` under
`lambda = 0` with the oracle's Orthogonal Forecast Fit.
=#
using Statistics, LinearAlgebra
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))

# A `ReturnsResult` of the returns `R` and its Panel Fields, all active and all in the estimation
# universe unless stated.
function rec_rd(R; fields = Pair{String, Matrix{Float64}}[],
                cats = Pair{String, Matrix{String}}[], mcap = ones(size(R)),
                amsk = trues(size(R)), emsk = trues(size(R)))
    pf = [NumericPanelInput(; name = "market_cap", vals = mcap);
          [NumericPanelInput(; name = k, vals = v) for (k, v) in fields];
          [CategoricalPanelInput(; name = k, vals = v) for (k, v) in cats]]
    return ReturnsResult(; nx = ["a$i" for i in axes(R, 2)], X = R,
                         pnl = asset_panel(pf; amsk = amsk, emsk = emsk))
end
bcast(v, T) = repeat(permutedims(v), T)
pass(f, fam = "style") = grid_pass(f; family = fam)
function rec_fit(rd; kw...)
    return prior(CrossSectionalFactorPrior(; lambda = 1, pe = PARITY_PE, ve = PARITY_VE,
                                           kw...), rd)
end
# The factor returns, the idiosyncratic returns and the true series on their rows.
fhat(pr) = pr.rr.csr.f
ehat(pr) = pr.rr.csr.eps
tail(x, n) = x[(end - n + 1):end, :]
eqw() = MarketCapWeights(; p = 0.0)
mse(a, b) = mean(abs2, a .- b)
# Three industries of ten assets, by block, with market caps 3, 2 and 1.
blocks(N) = [div(i - 1, 10) + 1 for i in 1:N]
function industry_fields(T, ind)
    return ["mkt_exp" => ones(T, length(ind));
            ["ind_$k" => bcast(Float64.(ind .== k), T) for k in 1:3]]
end
function industry_factors()
    return ["market" => pass("mkt_exp", "market");
            ["ind_$k" => pass("ind_$k", "industry") for k in 1:3]]
end

@testset "The recovery numbers of the oracle's test suite (#1391)" begin
    PO = PortfolioOptimisers

    @testset "1. A single factor with constant betas" begin
        rng = StableRNG(42)
        T, N = 2000, 50
        beta = 0.5 .+ rand(rng, N)
        se = 0.005 .+ 0.015 .* rand(rng, N)
        f = 0.01 .* randn(rng, T)
        R = beta' .* f .+ randn(rng, T, N) .* permutedims(se)
        pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(beta, T)]);
                     factors = ["beta" => pass("beta", "market")], pe = EmpiricalPrior(),
                     bp = 0.0, wa = eqw())
        fa = f[2:end]
        # Measured 0.985, 1.005, 0.34 and 0.047.
        @test cor(fhat(pr)[:, 1], fa) > 0.98
        @test isapprox(pr.fpr.sigma[1, 1], 0.01^2; rtol = 0.15)
        @test all(isapprox.(pr.rr.vs[end, :], se .^ 2; rtol = 0.4))
        @test maximum(i -> abs(cor(ehat(pr)[:, i], fa)), 1:N) < 0.08
        @test size(fhat(pr)) == (T - 1, 1) && size(ehat(pr)) == (T - 1, N)
        @test isapprox(pr.rr.M[:, 1], beta; rtol = 1e-10)
    end

    @testset "2. An intercept factor equals the benchmark return" begin
        rng = StableRNG(123)
        T, N = 500, 30
        fm = 0.008 .* randn(rng, T)
        R = fm .+ 0.01 .* randn(rng, T, N)
        mcap = exp.(10 .+ randn(rng, N))
        pr = rec_fit(rec_rd(R; fields = ["intercept" => ones(T, N)], mcap = bcast(mcap, T));
                     factors = ["intercept" => pass("intercept", "market")], bp = 1.0,
                     wa = MarketCapWeights(; p = 1.0))
        w = mcap / sum(mcap)
        @test isapprox(fhat(pr)[:, 1], R[2:end, :] * w; rtol = 1e-10)
        # The factor return is the cap-weighted return exactly, so its correlation with the
        # market is a property of the draw of the caps: `0.008 / sqrt(0.008^2 + 0.01^2 sum
        # w^2)` in expectation, 0.943 here, where the oracle's draw reaches its threshold 0.95.
        # Measured 0.944.
        ex = 0.008 / sqrt(0.008^2 + 0.01^2 * sum(abs2, w))
        @test abs(cor(fhat(pr)[:, 1], fm[2:end]) - ex) < 0.01
        @test all(isapprox.(pr.rr.M[:, 1], 1; rtol = 1e-10))
    end

    @testset "3. An exponentially weighted market beta" begin
        rng = StableRNG(77)
        T, N = 1500, 50
        beta = 0.5 .+ rand(rng, N)
        f = 0.01 .* randn(rng, T)
        R = beta' .* f .+ 0.005 .* randn(rng, T, N)
        ewb = CompositeExposure(;
                                descriptors = [EWMarketBeta(; half_life = 30, min_obs = 60)],
                                outlier = nothing, scoring = nothing, family = "market")
        pr = rec_fit(rec_rd(R); factors = ["beta" => ewb], bp = 0.0, wa = eqw())
        n = size(fhat(pr), 1)
        fa = f[(end - n + 1):end]
        # The first beta is at observation 60, and the exposure lag drops one more.
        @test n == T - 60
        @test cor(fhat(pr)[:, 1], fa) > 0.99
        @test cor(pr.rr.M[:, 1], beta) > 0.98
        @test maximum(i -> abs(cor(ehat(pr)[:, i], fa)), 1:N) < 0.05
    end

    @testset "4. Two uncorrelated factors" begin
        rng = StableRNG(99)
        T, N = 2000, 50
        b1 = 0.5 .+ rand(rng, N)
        b2 = 2 .* rand(rng, N) .- 1
        f1 = 0.01 .* randn(rng, T)
        f2 = 0.005 .* randn(rng, T)
        R = b1' .* f1 .+ b2' .* f2 .+ 0.005 .* randn(rng, T, N)
        pr = rec_fit(rec_rd(R; fields = ["beta1" => bcast(b1, T), "beta2" => bcast(b2, T)]);
                     factors = ["beta1" => pass("beta1"), "beta2" => pass("beta2")],
                     pe = EmpiricalPrior(), bp = 0.0, wa = eqw())
        F = pr.fpr.sigma
        g = fhat(pr)
        @test cor(g[:, 1], f1[2:end]) > 0.99
        @test cor(g[:, 2], f2[2:end]) > 0.96
        @test abs(cor(g[:, 1], f2[2:end])) < 0.05 && abs(cor(g[:, 2], f1[2:end])) < 0.05
        @test abs(F[1, 2] / sqrt(F[1, 1] * F[2, 2])) < 0.05
        @test isapprox(F[1, 1], 0.01^2; rtol = 0.05) &&
              isapprox(F[2, 2], 0.005^2; rtol = 0.1)
        @test isapprox(pr.rr.M, [b1 b2]; rtol = 1e-10)
    end

    @testset "5. Two correlated factors" begin
        rng = StableRNG(2024)
        T, N = 2000, 50
        b1 = 0.5 .+ rand(rng, N)
        b2 = 2 .* rand(rng, N) .- 1
        C = [0.01^2 0.6*0.01*0.008; 0.6*0.01*0.008 0.008^2]
        fs = randn(rng, T, 2) * cholesky(C).U
        R = b1' .* fs[:, 1] .+ b2' .* fs[:, 2] .+ 0.005 .* randn(rng, T, N)
        pr = rec_fit(rec_rd(R; fields = ["beta1" => bcast(b1, T), "beta2" => bcast(b2, T)]);
                     factors = ["beta1" => pass("beta1"), "beta2" => pass("beta2")],
                     pe = EmpiricalPrior(), bp = 0.0, wa = eqw())
        F = pr.fpr.sigma
        @test cor(fhat(pr)[:, 1], fs[2:end, 1]) > 0.99
        @test cor(fhat(pr)[:, 2], fs[2:end, 2]) > 0.98
        # The sample variance of this draw of the first factor is 5.3 % above `C[1, 1]`, so
        # the oracle's tolerance against `C` is a property of its own draw. Against the sample
        # covariance of the true factors, which is what a fit can recover, measured 2.1 %.
        @test all(isapprox.(F, cov(fs[2:end, :]); rtol = 0.05))
        @test isapprox(F[1, 2] / sqrt(F[1, 1] * F[2, 2]), 0.6; atol = 0.03)
    end

    @testset "6. An exposure lag of three with drifting betas" begin
        rng = StableRNG(314)
        T, N, L = 500, 30, 3
        bs = 0.3 .+ 0.4 .* rand(rng, N)
        be = 1.3 .+ 0.4 .* rand(rng, N)
        f = 0.01 .* randn(rng, T)
        eps = 0.005 .* randn(rng, T, N)
        betas = bs' .+ (be .- bs)' .* range(0, 1; length = T)
        R = betas .* f .+ eps
        pr = rec_fit(rec_rd(R; fields = ["beta" => betas]);
                     factors = ["beta" => pass("beta", "market")], lag = L, bp = 0.0,
                     wa = eqw())
        g = fhat(pr)[:, 1]
        Ral = R[(L + 1):end, :]
        @test length(g) == T - L
        @test isapprox(pr.rr.M[:, 1], betas[end, :]; rtol = 1e-10)
        @test !isapprox(pr.rr.M[:, 1], betas[end - L, :]; rtol = 1e-3)
        @test isapprox(ehat(pr), Ral .- betas[1:(end - L), :] .* g; rtol = 1e-10)
        @test mse(Ral, betas[(L + 1):end, :] .* g) > mean(abs2, ehat(pr))
        @test cor(g, f[(L + 1):end]) > 0.99
    end

    @testset "7. The estimation mask" begin
        rng = StableRNG(555)
        T, N, ne = 500, 30, 6
        betas = [fill(10.0, ne); 0.5 .+ rand(rng, N - ne)]
        f = 0.01 .* randn(rng, T)
        R = betas' .* f .+ 0.005 .* randn(rng, T, N)
        emsk = trues(T, N)
        emsk[:, 1:ne] .= false
        pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(betas, T)], emsk = emsk);
                     factors = ["beta" => pass("beta", "market")], bp = 0.0, wa = eqw(),
                     minra = 5)
        g = fhat(pr)[:, 1]
        n = length(g)
        b = betas[(ne + 1):end]
        @test isapprox(g, tail(R, n)[:, (ne + 1):end] * b / dot(b, b); rtol = 1e-10)
        @test !isapprox(g, tail(R, n) * betas / dot(betas, betas); rtol = 1e-3)
        @test isapprox(pr.rr.M[:, 1], betas; rtol = 1e-10)
        @test isapprox(ehat(pr), tail(R, n) .- betas' .* g; rtol = 1e-10)
        @test cor(g, f[(end - n + 1):end]) > 0.99
    end

    @testset "8. The sparse idiosyncratic correlation overlay" begin
        rng = StableRNG(777)
        T, N = 2000, 30
        betas = 0.5 .+ rand(rng, N)
        f = 0.01 .* randn(rng, T)
        eps = 0.005 .* randn(rng, T, N)
        z = 0.005 .* randn(rng, T, 2)
        eps[:, 1] = z[:, 1]
        eps[:, 2] = 0.5 .* z[:, 1] .+ sqrt(0.75) .* z[:, 2]
        R = betas' .* f .+ eps
        pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(betas, T)]);
                     factors = ["beta" => pass("beta", "market")], bp = 0.0, wa = eqw(),
                     th = 0.1)
        D = pr.rr.esigma
        off = [abs(D[i, j]) for i in 1:N, j in 1:N if i != j && !(minmax(i, j) == (1, 2))]
        @test D[1, 2] > 0
        @test maximum(off) < abs(D[1, 2])
        @test cor(fhat(pr)[:, 1], f[2:end]) > 0.99
    end

    # The industry draws of classes 9, 9b and 10b to 12: three blocks of ten assets with market
    # caps 3, 2 and 1, so the benchmark weights of the industries are 1/2, 1/3 and 1/6.
    T, N = 500, 30
    ind = blocks(N)
    mcap = Float64.([3, 2, 1][ind])
    wind = [30, 20, 10] / 60
    wb = mcap / sum(mcap)
    capw = MarketCapWeights(; p = 1.0)
    fam = ["industry" => nothing]

    @testset "9 and 9b. Basket-neutral industries" begin
        rng = StableRNG(888)
        fm = 0.01 .* randn(rng, T)
        fi = 0.01 .* randn(rng, T, 3)
        R = fm .+ fi[:, ind] .+ 0.005 .* randn(rng, T, N)
        pr = rec_fit(rec_rd(R; fields = industry_fields(T, ind), mcap = bcast(mcap, T));
                     factors = industry_factors(), families = fam, bp = 1.0, wa = capw)
        # The full basis: the raw factor returns, the factor covariance and the loadings.
        G = pr.rr.fr
        F = pr.fpr.sigma
        n = size(G, 1)
        @test pr.rr.nf == ["market", "ind_1", "ind_2", "ind_3"] && size(G, 2) == 4
        @test maximum(abs, G[:, 2:4] * wind) < 1e-12
        @test cor(G[:, 1], tail(fm .+ fi * wind, n)[:, 1]) > 0.99
        @test isapprox(G[:, 1], tail(R, n) * wb; rtol = 1e-10)
        @test rank(F) == 3
        ev = eigen(Symmetric(F))
        v = ev.vectors[:, argmin(abs.(ev.values))]
        e = normalize([0; wind])
        @test isapprox(abs(dot(v, e)), 1; atol = 1e-8)
        @test isposdef(Symmetric(pr.sigma))
        # 9b: the fit regresses on the reduced basis and states the full one.
        @test !isnothing(pr.rr.fcb) && size(pr.rr.L, 2) == 3 && size(pr.rr.M, 2) == 4
    end

    @testset "10. Factor neutralisation" begin
        rng = StableRNG(999)
        s0 = randn(rng, N)
        nz = randn(rng, N)
        fm, fs, fmo = (0.01 .* randn(rng, T) for _ in 1:3)
        R0 = 0.005 .* randn(rng, T, N)
        z(x) = (x .- mean(x)) ./ std(x; corrected = false)
        sz = z(s0)
        mz = z(0.7 .* s0 .+ sqrt(0.51) .* nz)
        R = fm .+ sz' .* fs .+ mz' .* fmo .+ R0
        rd = rec_rd(R;
                    fields = ["mkt_exp" => ones(T, N), "size" => bcast(sz, T),
                              "momentum" => bcast(mz, T)])
        facs = ["market" => pass("mkt_exp", "market"), "size" => pass("size"),
                "momentum" => pass("momentum")]
        p0 = rec_fit(rd; factors = facs, bp = 0.0, wa = eqw())
        p1 = rec_fit(rd; factors = facs, bp = 0.0, wa = eqw(),
                     neutralise = ["momentum" => ["size"]])
        E0, E1 = p0.rr.Ms, p1.rr.Ms
        @test abs(cor(E0[end, :, 2], E0[end, :, 3])) > 0.65
        cv(a, b) = mean((a .- mean(a)) .* (b .- mean(b)))
        @test all(t -> abs(cv(E1[t, :, 2], E1[t, :, 3])) < 1e-10, axes(E1, 1))
        @test all(t -> abs(mean(E1[t, :, 3])) < 1e-10 && abs(std(E1[t, :, 3]) - 1) < 1e-10,
                  axes(E1, 1))
        @test isapprox(E1[:, :, 2], E0[:, :, 2]; rtol = 1e-10)
        n = size(fhat(p0), 1)
        @test cor(fhat(p0)[:, 2], fs[(end - n + 1):end]) > 0.98
        @test cor(fhat(p0)[:, 3], fmo[(end - n + 1):end]) > 0.98
    end

    # The style characteristic of classes 10b and 10c leans on the first industry.
    function style_draw(seed)
        rng = StableRNG(seed)
        fm = 0.01 .* randn(rng, T)
        fi = 0.01 .* randn(rng, T, 3)
        fs = 0.01 .* randn(rng, T)
        sc = randn(rng, N) .+ 1.5 .* (ind .== 1)
        R = fm .+ fi[:, ind] .+ sc' .* fs .+ 0.005 .* randn(rng, T, N)
        return R, sc
    end
    # The benchmark-weighted cross products of the style loading with each industry.
    wdot(M, k) = sum(wb .* M[:, 5] .* M[:, k])

    @testset "10b. Neutralisation beside basket-neutral industries" begin
        R, sc = style_draw(777)
        rd = rec_rd(R; fields = [industry_fields(T, ind); "style_char" => bcast(sc, T)],
                    mcap = bcast(mcap, T))
        facs = [industry_factors(); "style" => pass("style_char")]
        raw = rec_fit(rd; factors = facs, families = fam, bp = 1.0, wa = capw)
        neu = rec_fit(rd; factors = facs, families = fam, bp = 1.0, wa = capw,
                      neutralise = ["style" => ["industry"]])
        @test abs(wdot(raw.rr.M, 2)) > 0.05
        @test all(k -> abs(wdot(neu.rr.M, k)) < 1e-8, 2:4)
        @test abs(sum(wb .* neu.rr.M[:, 5])) < 1e-8
        @test isposdef(Symmetric(neu.sigma))
    end

    @testset "10c. Demeaning within industries against neutralisation" begin
        R, sc = style_draw(888)
        rd = rec_rd(R; fields = [industry_fields(T, ind); "style_char" => bcast(sc, T)],
                    cats = ["industry_group" => bcast(string.(ind), T)],
                    mcap = bcast(mcap, T))
        grouped = CompositeExposure(; descriptors = [Passthrough(; field = "style_char")],
                                    outlier = nothing, group = "industry_group",
                                    family = "style")
        plain = CompositeExposure(; descriptors = [Passthrough(; field = "style_char")],
                                  outlier = nothing, family = "style")
        kw = (; families = fam, bp = 1.0, wa = capw)
        neut = ["style" => ["industry"]]
        A = rec_fit(rd; factors = [industry_factors(); "style" => grouped], kw...)
        B = rec_fit(rd; factors = [industry_factors(); "style" => plain], neutralise = neut,
                    kw...)
        C = rec_fit(rd; factors = [industry_factors(); "style" => grouped],
                    neutralise = neut, kw...)
        @test all(p -> all(k -> abs(wdot(p.rr.M, k)) < 1e-8, 2:4), (A, B, C))
        # Neutralisation is a no-op after the demeaning, and the two routes differ in the
        # spread within each industry.
        @test maximum(abs, A.rr.M[:, 5] - C.rr.M[:, 5]) < 1e-10
        @test cor(A.rr.M[:, 5], B.rr.M[:, 5]) < 0.9999
    end

    @testset "11. Inverse idiosyncratic variance regression weights" begin
        rng = StableRNG(777)
        betas = 0.5 .+ rand(rng, N)
        f = 0.01 .* randn(rng, T)
        eps = [0.002 .* randn(rng, T, 15) 0.02 .* randn(rng, T, 15)]
        mc = exp.(10 .+ randn(rng, N))
        R = betas' .* f .+ eps
        rd = rec_rd(R; fields = ["beta" => bcast(betas, T)], mcap = bcast(mc, T))
        # A half-life of ten observations.
        ivs = ExpWeightedVariance(; decay = 2.0^(-1 / 10), min_obs = 1)
        fit(; kw...) = rec_fit(rd; factors = ["beta" => pass("beta", "market")], bp = 0.0,
                               kw...)
        eq = fit(; wa = eqw())
        iv = fit(; wa = BlendedInverseVarianceWeights(; p = 0.0, lambda = 1.0), ve = ivs)
        bl = fit(; wa = BlendedInverseVarianceWeights(; p = 0.5, lambda = 0.5), ve = ivs)
        sq = fit(; wa = MarketCapWeights(; p = 0.5))
        n = size(fhat(eq), 1)
        fa = f[(end - n + 1):end]
        rw = iv.rr.rw
        @test mean(rw[:, 1:15]) > mean(rw[:, 16:30])
        @test mse(fhat(iv)[:, 1], fa) < mse(fhat(eq)[:, 1], fa)
        @test cor(fhat(iv)[:, 1], fa) > 0.95
        @test all(isapprox.(eq.rr.rw, 1; rtol = 1e-10))
        @test isapprox(sq.rr.rw[1, :], sqrt.(mc); rtol = 1e-10)
        @test mse(fhat(bl)[:, 1], fa) < mse(fhat(sq)[:, 1], fa)
        function ratio(W)
            rows = [r ./ sum(r) for r in eachrow(W) if sum(r) > 0]
            return mean(r -> mean(r[1:15]), rows) / mean(r -> mean(r[16:30]), rows)
        end
        @test ratio(sq.rr.rw) < ratio(bl.rr.rw) < ratio(iv.rr.rw)
    end

    @testset "12. The intercept beside industries and a scored style" begin
        rng = StableRNG(999)
        fm = 0.01 .* randn(rng, T)
        fi = 0.01 .* randn(rng, T, 3)
        fs = 0.01 .* randn(rng, T)
        sc = 0.5 .+ rand(rng, N)
        R = fm .+ fi[:, ind] .+ sc' .* fs .+ 0.005 .* randn(rng, T, N)
        rd = rec_rd(R; fields = [industry_fields(T, ind); "style_char" => bcast(sc, T)],
                    mcap = bcast(mcap, T))
        facs = [industry_factors(); "style" => grid_scored("style_char")]
        same = rec_fit(rd; factors = facs, families = fam, bp = 1.0, wa = capw)
        diff = rec_fit(rd; factors = facs, families = fam, bp = 1.0,
                       wa = MarketCapWeights(; p = 0.5))
        @test length(same.rr.nf) == 5
        for p in (same, diff)
            G = p.rr.fr
            n = size(G, 1)
            # The caps are constant within each industry, so both regression weights give
            # the cap-weighted return.
            @test isapprox(G[:, 1], tail(R, n) * wb; rtol = 1e-10)
            @test maximum(abs, G[:, 2:4] * wind) < 1e-12
            @test isposdef(Symmetric(p.sigma))
        end
        n = size(same.rr.fr, 1)
        @test cor(same.rr.fr[:, 5], fs[(end - n + 1):end]) > 0.90
    end

    @testset "13. The regression diagnostics of a true and a noise factor" begin
        rng = StableRNG(1234)
        N13 = 50
        betas = 0.5 .+ rand(rng, N13)
        f = 0.01 .* randn(rng, T)
        R = betas' .* f .+ 0.005 .* randn(rng, T, N13)
        noise = randn(rng, T, N13)
        rd = rec_rd(R; fields = ["beta" => bcast(betas, T), "noise" => noise])
        full = rec_fit(rd; factors = ["signal" => pass("beta"), "noise" => pass("noise")],
                       bp = 0.0, wa = eqw()).rr
        truef = rec_fit(rd; factors = ["signal" => pass("beta")], bp = 0.0, wa = eqw()).rr
        r2f, r2t = cs_regression_r2(full), cs_regression_r2(truef)
        t = cs_regression_t_stats(full).X
        rate = cs_regression_t_stat_exceedance_rate(full; threshold = 2).X
        @test mean(r2t) > 0.1
        @test minimum(r2f .- r2t) >= -1e-10
        @test size(t, 2) == 2
        @test median(abs.(t[:, 1])) > 3 && median(abs.(t[:, 2])) < 2
        @test rate[1] > 0.8 && rate[2] < 0.2
        @test mean(cs_regression_aic(truef)) < mean(cs_regression_aic(full))
        @test mean(cs_regression_bic(truef)) < mean(cs_regression_bic(full))
        a2 = cs_regression_adjusted_r2(full)
        @test length(a2) == length(r2f)
        @test all(i -> !(isfinite(a2[i]) && isfinite(r2f[i])) || a2[i] <= r2f[i] + 1e-12,
                  eachindex(a2))
    end

    @testset "14. Time-varying market caps" begin
        rng = StableRNG(314)
        tg = range(0, 1; length = T)
        base = hcat((3 .- 2 .* tg) .* ones(1, 10), fill(2.0, T, 10),
                    (1 .+ 2 .* tg) .* ones(1, 10))
        scale = 0.8 .+ 0.4 .* rand(rng, N)
        fm = 0.01 .* randn(rng, T)
        fi = 0.005 .* randn(rng, T, 3)
        R = fm .+ 0.005 .* randn(rng, T, N) .+ fi[:, ind]
        mc = base .* permutedims(scale)
        pr = rec_fit(rec_rd(R; fields = industry_fields(T, ind), mcap = mc);
                     factors = industry_factors(), families = fam, bp = 1.0, wa = capw)
        G = pr.rr.fr
        n = size(G, 1)
        # The benchmark and the regression weights read the caps of the exposure date.
        lagcap = mc[(T - n):(T - 1), :]
        @test isapprox(G[:, 1],
                       vec(sum(lagcap ./ sum(lagcap; dims = 2) .* tail(R, n); dims = 2));
                       rtol = 1e-10, atol = 1e-12)
        @test isapprox(pr.rr.rw, lagcap; rtol = 1e-10, atol = 1e-12)
        bs = hcat((sum(lagcap[:, ind .== k]; dims = 2) for k in 1:3)...) ./
             sum(lagcap; dims = 2)
        @test maximum(abs, sum(bs .* G[:, 2:4]; dims = 2)) < 1e-10
        @test !isapprox(mean(G[1:50, 1]), mean(G[(end - 49):end, 1]); atol = 1e-6)
        @test all(isapprox.(pr.rr.M[:, 1], 1; atol = 1e-10))
        @test count(<(1e-12), abs.(eigvals(Symmetric(pr.fpr.sigma)))) == 1
        @test isposdef(Symmetric(pr.sigma))
    end

    @testset "15. Listings and delistings" begin
        rng = StableRNG(4242)
        N15, nf, t0 = 50, 40, 250
        betas = 0.5 .+ rand(rng, N15)
        f = 0.01 .* randn(rng, T)
        Rf = betas' .* f .+ 0.005 .* randn(rng, T, N15)
        R = copy(Rf)
        amsk = trues(T, N15)
        amsk[1:t0, (nf + 1):end] .= false
        R[.!amsk] .= NaN
        # The oracle states `NaN` caps and betas before the listing. A panel consumer ignores
        # an inactive cell (#1411), and the builder refuses a blank without a fill policy, so
        # the inactive cells here hold the values of the active ones.
        facs = ["f1" => pass("beta")]
        pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(betas, T)], amsk = amsk,
                            emsk = amsk); factors = facs, bp = 0.0, wa = eqw())
        ctrl = rec_fit(rec_rd(Rf[:, 1:nf]; fields = ["beta" => bcast(betas[1:nf], T)]);
                       factors = facs, bp = 0.0, wa = eqw())
        g = fhat(pr)[:, 1]
        @test size(pr.mu) == (N15,) && size(pr.sigma) == (N15, N15)
        @test all(isfinite, pr.rr.M) && length(g) == T - 1
        @test isapprox(g[1:(t0 - 1)], fhat(ctrl)[1:(t0 - 1), 1]; rtol = 1e-10)
        @test cor(g, f[2:end]) > 0.95
        @test isposdef(Symmetric(pr.sigma)) && all(isfinite, pr.rr.esigma)
        @test isapprox(pr.rr.M[(nf + 1):end, 1], betas[(nf + 1):end]; rtol = 1e-10)
    end

    @testset "17. Neutralisation of a family against a family" begin
        T17 = 200
        rng = StableRNG(1234)
        lab = [mod(i - 1, 3) + 1 for i in 1:N]
        size0 = randn(rng, N)
        mom0 = randn(rng, N)
        for k in 1:3
            size0[lab .== k] .+= 2 * (k - 1) + 0.5 * randn(rng)
            mom0[lab .== k] .+= -1.5 * (k - 1) + 0.5 * randn(rng)
        end
        mc = 1 .+ 99 .* rand(rng, T17, N)
        fm = 0.01 .* randn(rng, T17)
        fi = 0.01 .* randn(rng, T17, 3)
        fs = 0.01 .* randn(rng, T17)
        fmo = 0.01 .* randn(rng, T17)
        R = fm .+ fi[:, lab] .+ size0' .* fs .+ mom0' .* fmo .+ 0.005 .* randn(rng, T17, N)
        rd = rec_rd(R;
                    fields = [industry_fields(T17, lab); "size" => bcast(size0, T17);
                              "momentum" => bcast(mom0, T17)], mcap = mc)
        facs = [industry_factors(); "size" => pass("size"); "momentum" => pass("momentum")]
        kw = (; factors = facs, families = fam, bp = 1.0, wa = MarketCapWeights(; p = 0.5))
        neu = rec_fit(rd; neutralise = ["style" => ["industry"]], kw...)
        ctl = rec_fit(rd; kw...)
        E, W = neu.rr.Ms, neu.rr.bw
        @test all(abs(sum(W[t, :] .* E[t, :, s] .* E[t, :, j])) < 1e-8
                  for t in axes(E, 1), s in 5:6, j in 2:4)
        @test all(s -> maximum(abs, E[:, :, s] - ctl.rr.Ms[:, :, s]) > 0.01, 5:6)
    end

    @testset "14a and 14b. A spanned and an orthogonal alpha" begin
        N14 = 40
        function alpha_fit(seed, orth, c)
            rng = StableRNG(seed)
            betas = 0.5 .+ rand(rng, N14)
            f = 0.01 .* randn(rng, T)
            R = betas' .* f .+ 0.005 .* randn(rng, T, N14)
            delta = zeros(N14)
            if orth
                raw = 0.001 .* randn(rng, N14)
                Bm = [ones(N14) betas]
                delta = raw - Bm * (Bm \ raw)
                delta .*= 0.001 / (std(delta; corrected = false) + 1e-12)
            end
            alpha = 0.002 .* betas .+ delta
            pr = rec_fit(rec_rd(R; fields = ["beta" => bcast(betas, T)]);
                         factors = ["f1" => pass("beta")], bp = 0.0, wa = eqw(),
                         rfe = CustomValueReturnForecast(; mu = alpha), lambda = 0, c = c,
                         ofit = UnadjustedForecast())
            return pr, betas, alpha, delta
        end
        pr, betas = alpha_fit(77, false, 1.0)
        span = pr.rr.M * pr.fpr.mu
        @test maximum(abs, pr.mu - 0.002 .* betas) < 1e-10
        @test maximum(abs, pr.mu - span) < 1e-10
        @test maximum(abs, span - 0.002 .* betas) < 1e-10
        full, _, alpha, delta = alpha_fit(88, true, 1.0)
        zero, = alpha_fit(88, true, 0.0)
        @test maximum(abs, full.mu - alpha) < 1e-10
        @test maximum(abs, zero.mu - zero.rr.M * zero.fpr.mu) < 1e-10
        @test maximum(abs, full.mu - zero.mu) > 1e-5
        @test maximum(abs, full.mu - full.rr.M * full.fpr.mu - delta) < 1e-10
    end
end
