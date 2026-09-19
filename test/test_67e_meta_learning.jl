#=
The meta-learning rows of the fourth set, as ADR 0165 rules them (issue #1183 on map #1148):
the mixture's Gradient Point and Start Allocation over the experts, the `Ader` and `Sword`
constructors, and the Switching Weighting with `SwitchingPortfolio`.

Every literal below states its provenance: the one-row blend-point step is a hand computation
of the Euclidean step on the shared gradient `-x / ⟨w, x⟩` at `x = [1.2, 0.8]` from
`[0.5, 0.5]`; the longer paths are hand recursions of the papers' updates — Ader's Algorithms
3 and 4 with the surrogate loss, Singer's equations 4 to 6 on the price-adjusted holding — at
`1e-13`; the own-point mixture is the #1161 mixture bit for bit, checked as every expert's
held allocation against its own head; the `γ` degeneracies are equalities of paths.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates
@testset "Online portfolio selection: the gradient point, Ader, Sword and the switching weighting" begin
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection

    rng = StableRNG(11)
    T, N = 30, 3
    R = 0.03 .* randn(rng, T, N)
    X = 1 .+ R
    nx = ["A", "B", "C"]
    ts = Date(2021, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = R, ts = ts)
    rows(r, i) = po.port_opt_view(r, i, :)
    resolve(set, n) = po.resolve_allocation_set(set, n, false, Float64)
    simplex = resolve(BoundedAllocationSet(), N)

    # The library's path after t rows is the allocation held during period t + 1.
    function libpath(alg; kw...)
        opt = OPS(; alg = alg, kw...)
        W = zeros(T, N)
        W[1, :] .= 1 / N
        for t in 1:(T - 1)
            W[t + 1, :] .= optimise(opt, rows(rd, 1:t)).w
        end
        return W
    end
    maxerr(a, b) = maximum(abs.(a .- b))
    # The Euclidean projection onto the simplex, by hand.
    function simplex_projection(q)
        u = sort(q; rev = true)
        cs = cumsum(u)
        rho = findlast(j -> u[j] + (1 - cs[j]) / j > 0, eachindex(u))
        theta = (1 - cs[rho]) / rho
        return max.(q .+ theta, 0)
    end

    @testset "The Gradient Point and the start over the experts" begin
        @test isa(ExpertMixture(; experts = [BuyAndHold()]).grad, OwnPoint)
        @test isnothing(ExpertMixture(; experts = [BuyAndHold()]).p0)
        mix = ExpertMixture(; experts = [GradientProjection(), BuyAndHold()],
                            alg = ExponentiatedGradient(), grad = BlendPoint(),
                            p0 = [2.0, 1.0])
        # A start over the experts is projected once at the seed in the weighting's
        # geometry: the entropic one normalises.
        st = po.rule_state_seed(mix, fill(1 / N, N))
        @test st.p == [2 / 3, 1 / 3]
        # The Euclidean geometry of the buy-and-hold weighting takes the nearest point.
        mixe = ExpertMixture(; experts = [GradientProjection(), BuyAndHold()],
                             p0 = [0.9, 0.3])
        @test po.rule_state_seed(mixe, fill(1 / N, N)).p ≈ [0.8, 0.2]
        # Onto a bounded Expert Set, the bound binds.
        mixc = ExpertMixture(; experts = [GradientProjection(), BuyAndHold()],
                             eset = BoundedAllocationSet(; wb = WeightBounds(0.3, 0.7)),
                             p0 = [1.0, 0.0])
        @test po.rule_state_seed(mixc, fill(1 / N, N)).p ≈ [0.7, 0.3]
        @test_throws DimensionMismatch ExpertMixture(; experts = [BuyAndHold()],
                                                     p0 = [0.5, 0.5])
        @test_throws DomainError ExpertMixture(; experts = [BuyAndHold()], p0 = [Inf])
        # The view carries the point and the start unchanged.
        v = po.port_opt_view(mix, [1, 2])
        @test isa(v.grad, BlendPoint) && v.p0 == [2.0, 1.0]

        # One row by hand: two Euclidean experts at η ∈ {0.1, 0.4} from [0.5, 0.5] on
        # x = [1.2, 0.8]. The blend is [0.5, 0.5], so the shared gradient is
        # -x / ⟨w, x⟩ = [-1.2, -0.8]; each expert steps to [0.5 + 1.2 η, 0.5 + 0.8 η] and
        # the simplex projection subtracts η from both: [0.5 + 0.2 η, 0.5 - 0.2 η]. Both
        # experts returned 1, so the weighting stays uniform and the blend is their mean.
        rd2 = ReturnsResult(; nx = ["A", "B"], X = [0.2 -0.2; -0.1 0.1; 0.05 0.0],
                            ts = Date(2021, 1, 1) .+ Day.(0:2))
        two = ExpertMixture(;
                            experts = [GradientProjection(; eta = 0.1),
                                       GradientProjection(; eta = 0.4)],
                            alg = ExponentiatedGradient(; eta = 0.2), grad = BlendPoint())
        o = po.partial_fit!(OPS(; alg = two, w0 = [0.5, 0.5]), rows(rd2, 1:1))
        @test o.cache.st.h[1] ≈ [0.52, 0.48] atol = 1e-15
        @test o.cache.st.h[2] ≈ [0.58, 0.42] atol = 1e-15
        @test o.cache.st.p == [0.5, 0.5]
        @test o.cache.w ≈ [0.55, 0.45] atol = 1e-15
        # Under the own point the same row moves each expert on its own gradient, which
        # here is the same vector, so the first row agrees; the second row is where the
        # points part, and the own-point expert equals its own head bit for bit.
        own = ExpertMixture(; experts = two.experts, alg = two.alg)
        oo = po.partial_fit!(OPS(; alg = own, w0 = [0.5, 0.5]), rows(rd2, 1:1))
        @test oo.cache.st.h[1] == o.cache.st.h[1] && oo.cache.st.h[2] == o.cache.st.h[2]
        oo2 = po.partial_fit!(oo, rows(rd2, 2:2))
        o2 = po.partial_fit!(o, rows(rd2, 2:2))
        @test oo2.cache.st.h[1] != o2.cache.st.h[1]
        for (k, e) in enumerate(two.experts)
            @test oo2.cache.st.h[k] ==
                  optimise(OPS(; alg = e, w0 = [0.5, 0.5]), rows(rd2, 1:2)).w
        end

        # The blend-point mixture by hand over the rows: the shared gradient at the played
        # blend, every expert's Euclidean step from its own iterate, the exponentially
        # weighted forecaster on the expert returns.
        function blendpath(etas, eps)
            K = length(etas)
            h = [fill(1 / N, N) for _ in 1:K]
            p = fill(1 / K, K)
            W = zeros(T, N)
            W[1, :] .= 1 / N
            for t in 1:(T - 1)
                x = X[t, :]
                w = W[t, :]
                r = [dot(hk, x) for hk in h]
                g = -x ./ dot(w, x)
                h = [simplex_projection(h[k] .- etas[k] .* g) for k in 1:K]
                p = p .* exp.(eps .* r ./ dot(p, r))
                p ./= sum(p)
                W[t + 1, :] .= sum(p[k] .* h[k] for k in 1:K)
            end
            return W
        end
        etas = [0.05, 0.1, 0.2, 0.4]
        grid = ExpertMixture(; experts = [GradientProjection(; eta = e) for e in etas],
                             alg = ExponentiatedGradient(; eta = 0.3), grad = BlendPoint())
        @test maxerr(libpath(grid), blendpath(etas, 0.3)) < 1e-13
        # The sibling first-order rules read the point too: under the own point each equals
        # its own head to the ulp, and under the blend point the paths part after the
        # first row, where the blend leaves the expert's iterate.
        for e in (OptimisticStep(; alg = GradientProjection(; eta = 0.2)),
                  AdaptiveSubgradient())
            mo = ExpertMixture(; experts = [e, GradientProjection(; eta = 0.4)])
            mb = ExpertMixture(; experts = [e, GradientProjection(; eta = 0.4)],
                               grad = BlendPoint())
            oo = po.partial_fit!(OPS(; alg = mo), rows(rd, 1:6))
            ob = po.partial_fit!(OPS(; alg = mb), rows(rd, 1:6))
            @test isapprox(oo.cache.st.h[1], optimise(OPS(; alg = e), rows(rd, 1:6)).w;
                           atol = 1e-15)
            @test maxerr(oo.cache.st.h[1], ob.cache.st.h[1]) > 1e-6
            @test isapprox(sum(ob.cache.st.h[1]), 1; atol = 1e-12)
        end
        # A rule with no gradient ignores the point: the buy-and-hold expert of a
        # blend-point mixture holds its own path.
        mixb = ExpertMixture(; experts = [BuyAndHold(), GradientProjection()],
                             grad = BlendPoint())
        ob = po.partial_fit!(OPS(; alg = mixb), rows(rd, 1:9))
        @test ob.cache.st.h[1] == optimise(OPS(; alg = BuyAndHold()), rows(rd, 1:9)).w
        # The seven-argument update's generic method drops the point, and a `nothing` point
        # is the six-argument update.
        @test isnothing(po.gradient_point(OwnPoint(), [0.5, 0.5]))
        @test po.gradient_point(BlendPoint(), [0.5, 0.5]) == [0.5, 0.5]
        st0, w0 = po.online_update!(GradientProjection(),
                                    po.rule_state_seed(GradientProjection(),
                                                       fill(1 / N, N)), fill(1 / N, N),
                                    X[1, :], nothing, simplex, nothing)
        @test w0 == optimise(OPS(; alg = GradientProjection()), rows(rd, 1:1)).w
        st1, w1 = po.online_update!(BuyAndHold(), nothing, fill(1 / N, N), X[1, :], nothing,
                                    simplex, [0.2, 0.3, 0.5])
        st2, w2 = po.online_update!(BuyAndHold(), nothing, fill(1 / N, N), X[1, :], nothing,
                                    simplex)
        @test w1 == w2 && isnothing(st1)
    end

    @testset "Ader and Sword" begin
        ader = Ader(; eta_min = 0.05, K = 4, eps = 0.3)
        @test [e.eta for e in ader.experts] == [0.05, 0.1, 0.2, 0.4]
        @test all(e -> isa(e.proj, EuclideanProjection), ader.experts)
        @test isa(ader.alg, MirrorDescent) &&
              ader.alg.eta == 0.3 &&
              isa(ader.alg.proj, EntropicProjection)
        @test isa(ader.grad, BlendPoint)
        # The paper's start: C / (i (i + 1)) with C = 1 + 1/K sums to one.
        C = 1 + 1 / 4
        @test ader.p0 ≈ [C / (i * (i + 1)) for i in 1:4]
        @test sum(ader.p0) ≈ 1
        sword = Sword(; eta_min = 0.05, K = 4, eps = 0.3)
        @test [e.eta for e in sword.experts] == [0.05, 0.1, 0.2, 0.4]
        @test isnothing(sword.p0) && isa(sword.grad, BlendPoint)
        @test_throws DomainError Ader(; eta_min = 0, K = 4, eps = 0.3)
        @test_throws DomainError Ader(; eta_min = 0.05, K = 0, eps = 0.3)
        @test_throws DomainError Sword(; eta_min = 0.05, K = 4, eps = 0)
        # The grid's type is the rate's.
        @test eltype([e.eta for e in Ader(; eta_min = 0.05f0, K = 3, eps = 0.3).experts]) ==
              Float32

        # Ader's Algorithms 3 and 4 by hand: the surrogate loss ℓ_t(w) = ⟨∇f_t(w_t), w - w_t⟩,
        # the experts' step on ∇f_t(w_t), the meta weights ∝ w_i exp(-α ℓ_t(h_i)) from the
        # paper's start.
        function aderpath(eta_min, K, alpha)
            etas = [eta_min * 2^(i - 1) for i in 1:K]
            h = [fill(1 / N, N) for _ in 1:K]
            p = [(1 + 1 / K) / (i * (i + 1)) for i in 1:K]
            W = zeros(T, N)
            W[1, :] .= 1 / N
            for t in 1:(T - 1)
                x = X[t, :]
                w = W[t, :]
                g = -x ./ dot(w, x)
                ell = [dot(g, hk .- w) for hk in h]
                h = [simplex_projection(h[k] .- etas[k] .* g) for k in 1:K]
                p = p .* exp.(-alpha .* ell)
                p ./= sum(p)
                W[t + 1, :] .= sum(p[k] .* h[k] for k in 1:K)
            end
            return W
        end
        @test maxerr(libpath(ader), aderpath(0.05, 4, 0.3)) < 1e-13
        # Sword's small-loss form differs from Ader by the uniform start alone.
        function swordpath(eta_min, K, eps)
            etas = [eta_min * 2^(i - 1) for i in 1:K]
            h = [fill(1 / N, N) for _ in 1:K]
            p = fill(1 / K, K)
            W = zeros(T, N)
            W[1, :] .= 1 / N
            for t in 1:(T - 1)
                x = X[t, :]
                w = W[t, :]
                g = -x ./ dot(w, x)
                ell = [dot(g, hk) for hk in h]
                h = [simplex_projection(h[k] .- etas[k] .* g) for k in 1:K]
                p = p .* exp.(-eps .* ell)
                p ./= sum(p)
                W[t + 1, :] .= sum(p[k] .* h[k] for k in 1:K)
            end
            return W
        end
        @test maxerr(libpath(sword), swordpath(0.05, 4, 0.3)) < 1e-13
        # The batch pass and the fold agree at every block size.
        for opt in (OPS(; alg = ader), OPS(; alg = sword))
            o = po.partial_fit!(opt, rows(rd, 1:10))
            o = po.partial_fit!(o, rows(rd, 11:17))
            o = po.partial_fit!(o, rows(rd, 18:18))
            @test optimise(o).w == optimise(opt, rows(rd, 1:18)).w
        end
        # On a Risk Loss the experts read the loss's rows and the blend point alike.
        ar = Ader(; eta_min = 0.05, K = 3, eps = 0.3, obj = RiskLoss(; window = 5))
        @test po.rows_needed(ar) == 5
        orr = po.partial_fit!(OPS(; alg = ar), rows(rd, 1:12))
        @test isapprox(sum(orr.cache.w), 1; atol = 1e-12) && all(>=(0), orr.cache.w)
        # Under a cap on the Expert Set the start is projected onto it and the weights stay
        # within it.
        ac = Ader(; eta_min = 0.05, K = 4, eps = 0.3,
                  eset = BoundedAllocationSet(; wb = WeightBounds(0.1, 0.5)))
        oc = po.partial_fit!(OPS(; alg = ac), rows(rd, 1:8))
        @test all(0.1 - 1e-12 .<= oc.cache.st.p .<= 0.5 + 1e-12)
        @test po.rule_state_seed(ac, fill(1 / N, N)).p[1] ≈ 0.5
    end

    @testset "The switching weighting and the switching portfolio" begin
        sw = SwitchingWeighting()
        @test sw.gamma == 1 / 3 && isa(sw.proj, EuclideanProjection)
        @test_throws DomainError SwitchingWeighting(; gamma = -0.1)
        @test_throws DomainError SwitchingWeighting(; gamma = 1.1)
        @test po.rows_needed(sw) == 0 && isnothing(po.rule_state_seed(sw, fill(1 / N, N)))
        @test_throws DomainError SwitchingPortfolio(; N = 0)
        sp = SwitchingPortfolio(; N = N, gamma = 0.25)
        @test length(sp.experts) == N && sp.alg.gamma == 0.25
        @test [e.w for e in sp.experts] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Singer's equations 4 to 6 on the price-adjusted holding, by hand.
        function singerpath(gamma)
            W = zeros(T, N)
            W[1, :] .= 1 / N
            for t in 1:(T - 1)
                w = W[t, :]
                x = X[t, :]
                q = w .* x ./ dot(w, x)
                W[t + 1, :] .= (1 - gamma) .* q .+ gamma / (N - 1) .* (sum(q) .- q)
            end
            return W
        end
        @test maxerr(libpath(sp), singerpath(0.25)) < 1e-13
        # On the head over the assets the weighting computes the same numbers, because the
        # blend of the unit experts is the weight vector itself.
        @test maxerr(libpath(SwitchingWeighting(; gamma = 0.25)), singerpath(0.25)) < 1e-13
        # The degeneracies: γ = 0 is buy-and-hold over the experts, γ = (N - 1) / N the
        # uniform constant rebalanced portfolio.
        @test maxerr(libpath(SwitchingPortfolio(; N = N, gamma = 0)),
                     libpath(BuyAndHold())) < 1e-14
        @test maxerr(libpath(SwitchingPortfolio(; N = N, gamma = (N - 1) / N)),
                     libpath(ConstantRebalancedPortfolio())) < 1e-14
        # Over one expert there is nothing to switch to.
        one_set = resolve(BoundedAllocationSet(), 1)
        @test po.online_update!(sw, nothing, [1.0], [1.1], nothing, one_set)[2] == [1.0]
        # A cap on the Expert Set clips the share after the mix.
        spc = SwitchingPortfolio(; N = N,
                                 eset = BoundedAllocationSet(; wb = WeightBounds(0.2, 0.5)))
        oc = po.partial_fit!(OPS(; alg = spc), rows(rd, 1:6))
        @test all(0.2 - 1e-12 .<= oc.cache.st.p .<= 0.5 + 1e-12) &&
              isapprox(sum(oc.cache.st.p), 1; atol = 1e-12)
        # The batch pass and the fold agree at every block size.
        opt = OPS(; alg = sp)
        o = po.partial_fit!(opt, rows(rd, 1:10))
        o = po.partial_fit!(o, rows(rd, 11:17))
        o = po.partial_fit!(o, rows(rd, 18:18))
        @test optimise(o).w == optimise(opt, rows(rd, 1:18)).w
        # A view onto a subset of the assets drops no expert, so the unit expert of an
        # excluded asset has nothing to hold on the view and its constant rebalanced
        # portfolio refuses the renormalised zero slice.
        @test_throws DomainError po.port_opt_view(OPS(; alg = sp), [1, 3])
    end
end
