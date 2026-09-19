#=
The second algorithm set of the online portfolio selection family (issue #1177 on map #1148):
the confidence-weighted mean reversion, the anti-correlation rule, the expectation-maximisation
update in its Soft-Bayes online form, and the three weightings — the aggregating algorithm,
the top-k selection and the weak aggregating algorithm — with the constructor over
exponentiated-gradient experts.

Parity is measured against each paper's update written out by hand on the fixture, step for
step, and every rule takes the batch–online identity through the head at block sizes 10/7/1.
=#

# A rate that falls with the step, a probe for the online form's fixed-share pull: a
# Learning-Rate Schedule on the slot, and nothing else in the rule changes.
struct DecayRate <: PortfolioOptimisers.AbstractLearningRateSchedule
    eta0::Float64
end
function PortfolioOptimisers.learning_rate(r::DecayRate, t::Integer, ::Any)
    return r.eta0 / sqrt(t)
end

@testset "Online portfolio selection: the second set" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection

    rng = StableRNG(11)
    T, N = 40, 4
    R = 0.02 .* randn(rng, T, N)
    X = 1 .+ R
    nx = ["A", "B", "C", "D"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = R, ts = ts)
    rows(r, i) = po.port_opt_view(r, i, :)
    simplex = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
    cap = po.resolve_allocation_set(BoundedAllocationSet(; wb = WeightBounds(0, 0.4)), N,
                                    false, Float64)

    # The library's path: W[t + 1, :] is the allocation after rows 1:t.
    function libpath(alg; kw...)
        opt = OPS(; alg = alg, kw...)
        W = zeros(T + 1, N)
        W[1, :] .= 1 / N
        for t in 1:T
            W[t + 1, :] .= optimise(opt, rows(rd, 1:t)).w
        end
        return W
    end
    # One raw call of the update on the bare simplex from a state seeded at w.
    function step(alg, w, x, rows = nothing)
        st = po.rule_state_seed(alg, w)
        return po.online_update!(alg, st, w, x, rows, simplex)
    end
    w0 = fill(1 / N, N)

    @testset "Confidence weighted mean reversion" begin
        # The paper's Algorithm 1 in its 2013 linear form, written out on full matrices.
        function cwmr_paper(X, phi, eps, var::Bool)
            T, N = size(X)
            mu = fill(1 / N, N)
            S = Matrix(I, N, N) / N^2
            W = zeros(T + 1, N)
            W[1, :] .= mu
            for t in 1:T
                x = X[t, :]
                M = dot(mu, x)
                V = dot(x, S * x)
                Wt = dot(x, S * ones(N))
                xbar = dot(ones(N), S * x) / dot(ones(N), S * ones(N))
                if var
                    a = 2phi * V^2 - 2phi * xbar * V * Wt
                    b = 2phi * eps * V - 2phi * V * M + V - xbar * Wt
                    c = eps - M - phi * V
                else
                    a = (V - xbar * Wt + (phi^2 * V) / 2)^2 - (phi^4 * V^2) / 4
                    b = 2 * (eps - M) * (V - xbar * Wt + (phi^2 * V) / 2)
                    c = (eps - M)^2 - phi^2 * V
                end
                d = b^2 - 4a * c
                lam = d < 0 ? 0.0 : max(0.0, (-b + sqrt(d)) / (2a), (-b - sqrt(d)) / (2a))
                mu = po.project_simplex(mu - lam * S * (x - xbar * ones(N)))
                if var
                    S = inv(inv(S) + 2lam * phi * diagm(x .^ 2))
                else
                    u = (-lam * phi * V + sqrt(lam^2 * phi^2 * V^2 + 4V)) / 2
                    S = inv(inv(S) + lam * (phi / u) * diagm(x .^ 2))
                end
                S = S ./ (N^2 * tr(S))
                W[t + 1, :] .= mu
            end
            return W
        end
        for (f, var) in ((VarianceUpdate(), true), (StandardDeviationUpdate(), false))
            Wl = libpath(ConfidenceWeightedMeanReversion(; formulation = f))
            @test isapprox(Wl, cwmr_paper(X, 2.0, 0.5, var); atol = 1e-10)
            @test all(isapprox.(sum(Wl; dims = 2), 1; atol = 1e-12))
        end
        # The two formulations differ from each other and both are passive when the
        # constraint already holds: a threshold above every return leaves the allocation
        # where it was and only rescales the belief, whose seed trace 1 / N falls to the
        # paper's 1 / N² at the first step and stays there.
        @test !isapprox(libpath(ConfidenceWeightedMeanReversion()),
                        libpath(ConfidenceWeightedMeanReversion(;
                                                                formulation = StandardDeviationUpdate()));
                        atol = 1e-6)
        st = po.rule_state_seed(ConfidenceWeightedMeanReversion(), w0)
        @test st.sigma == fill(1 / N^2, N)
        stp, wp = step(ConfidenceWeightedMeanReversion(; eps = 2), w0, X[1, :])
        @test isapprox(wp, w0; atol = 1e-15)
        @test stp.sigma == fill(1 / N^3, N) && stp.n == 1
        # The belief's trace is rescaled to 1 / N² after every step, and the step moves
        # the mean against the price relative.
        sta, wa = step(ConfidenceWeightedMeanReversion(), w0, X[1, :])
        @test isapprox(sum(sta.sigma), 1 / N^2; atol = 1e-15)
        @test !isapprox(wa, w0; atol = 1e-6)
        @test wa[argmax(X[1, :])] < w0[argmax(X[1, :])]
        # The quadratic's root: no real root, a line, both roots negative.
        @test po.nonneg_quadratic_root(1.0, 0.0, 1.0) == 0
        @test po.nonneg_quadratic_root(0.0, 2.0, -1.0) == 0.5
        @test po.nonneg_quadratic_root(0.0, 0.0, 1.0) == 0
        @test po.nonneg_quadratic_root(1.0, 3.0, 2.0) == 0
        @test po.nonneg_quadratic_root(1.0, -3.0, 2.0) == 2
        # The carrier: a view slices the diagonal, a copy aliases nothing, a merge refuses.
        v = po.port_opt_view(sta, [1, 3])
        @test v.sigma == sta.sigma[[1, 3]] && v.n == 1
        c = copy(sta)
        @test c.sigma == sta.sigma && c.sigma !== sta.sigma
        @test_throws ArgumentError po.merge_states(sta, c)
        @test_throws DomainError ConfidenceWeightedMeanReversion(; eps = -1)
        @test_throws DomainError ConfidenceWeightedMeanReversion(; phi = -1)
        @test_throws TypeError ConfidenceWeightedMeanReversion(;
                                                               proj = EntropicProjection())
        @test po.rows_needed(ConfidenceWeightedMeanReversion()) == 0
        # Under a cap the mean's projection is the Euclidean root.
        stc, wc = po.online_update!(ConfidenceWeightedMeanReversion(),
                                    po.rule_state_seed(ConfidenceWeightedMeanReversion(),
                                                       w0), w0, [1.5, 1.0, 0.8, 0.9],
                                    nothing, cap)
        @test all(wc .<= 0.4 + 1e-12) && isapprox(sum(wc), 1; atol = 1e-12)
    end

    @testset "Anti-correlation" begin
        # The paper's transfer written out with loops, on the last 2w rows of log relatives.
        function anticor_paper(w, rows, wn)
            T = size(rows, 1)
            L = log.(1 .+ rows[(T - 2wn + 1):T, :])
            L1, L2 = L[1:wn, :], L[(wn + 1):(2wn), :]
            N = length(w)
            mu1, mu2 = vec(mean(L1; dims = 1)), vec(mean(L2; dims = 1))
            s1, s2 = vec(std(L1; dims = 1)), vec(std(L2; dims = 1))
            cor = zeros(N, N)
            for i in 1:N, j in 1:N
                cv = sum((L1[:, i] .- mu1[i]) .* (L2[:, j] .- mu2[j])) / (wn - 1)
                cor[i, j] = (s1[i] == 0 || s2[j] == 0) ? 0.0 : cv / (s1[i] * s2[j])
            end
            claim = zeros(N, N)
            for i in 1:N, j in 1:N
                if i != j && mu2[i] >= mu2[j] && cor[i, j] > 0
                    claim[i, j] = cor[i, j] +
                                  (cor[i, i] < 0 ? abs(cor[i, i]) : 0.0) +
                                  (cor[j, j] < 0 ? abs(cor[j, j]) : 0.0)
                end
            end
            transfer = zeros(N, N)
            for i in 1:N
                s = sum(claim[i, :])
                if s > 0
                    transfer[i, :] .= w[i] .* claim[i, :] ./ s
                end
            end
            return [w[i] - sum(transfer[i, :]) + sum(transfer[:, i]) for i in 1:N]
        end
        for wn in (2, 3, 5)
            alg = AntiCorrelation(; window = wn)
            @test po.rows_needed(alg) == 2wn
            W = libpath(alg)
            w = copy(w0)
            for t in 1:T
                q = t < 2wn ? w : anticor_paper(w, R[1:t, :], wn)
                @test isapprox(W[t + 1, :], q; atol = 1e-12)
                @test isapprox(sum(W[t + 1, :]), 1; atol = 1e-12)
                @test all(W[t + 1, :] .>= -1e-15)
                w = W[t + 1, :]
            end
            # The rule holds until 2w rows are held, and moves at the first full window.
            @test W[2wn, :] == w0
            @test !isapprox(W[2wn + 1, :], w0; atol = 1e-6)
        end
        # A hand window: asset 1's first-window column tracks asset 2's second-window
        # column, and asset 1 grew at least as much over the latest window, so wealth
        # moves from 1 to 2; a constant column has no correlation and takes part in no
        # claim.
        L = zeros(6, 3)
        L[1:3, 1] .= [0.01, 0.03, 0.02]
        L[4:6, 2] .= [0.01, 0.03, 0.02]
        L[4:6, 1] .= [0.05, 0.04, 0.06]
        cor, mu2 = po.lagged_window_correlation(L[1:3, :], L[4:6, :])
        @test isapprox(cor[1, 2], 1; atol = 1e-12)
        @test cor[:, 3] == zeros(3) && cor[3, :] == zeros(3)
        @test mu2 == vec(mean(L[4:6, :]; dims = 1))
        claim = po.anticorrelation_claims(cor, mu2)
        @test claim[1, 2] > 0 && claim[2, 1] == 0 && all(claim[:, 3] .== 0)
        q = po.wealth_transfer(fill(1 / 3, 3), claim)
        @test q[2] > 1 / 3 && q[1] < 1 / 3 && q[3] == 1 / 3
        @test isapprox(sum(q), 1; atol = 1e-15)
        # Through the update on returns whose log relatives are that window.
        rows3 = expm1.(L)
        _, w3 = step(AntiCorrelation(; window = 3), fill(1 / 3, 3), 1 .+ rows3[end, :],
                     rows3)
        @test isapprox(w3, q; atol = 1e-12)
        @test isnothing(po.rule_state_seed(AntiCorrelation(), w0))
        @test_throws DomainError AntiCorrelation(; window = 1)
        @test_throws TypeError AntiCorrelation(; proj = EntropicProjection())
    end

    @testset "Expectation maximisation, Soft-Bayes" begin
        alg = ExpectationMaximisation(; eta = 0.3)
        x = X[1, :]
        st, w1 = step(alg, w0, x)
        g = x ./ dot(w0, x)
        @test w1 == w0 .* (1 - 0.3 .+ 0.3 .* g)
        @test isapprox(sum(w1), 1; atol = 1e-15)
        @test st.n == 1 && st.w1 == w0
        # The convex combination of holding and Cover's posterior.
        @test isapprox(w1, 0.7 .* w0 .+ 0.3 .* w0 .* x ./ dot(w0, x); atol = 1e-15)
        # It is not the exponentiated gradient, and no weight grows by more than 1 + eta.
        _, we = step(ExponentiatedGradient(; eta = 0.3), w0, x)
        @test !isapprox(w1, we; atol = 1e-6)
        @test all(w1 ./ w0 .<= 1 + 0.3 + 1e-15)
        # The whole path is the plain recursion.
        W = libpath(alg)
        w = copy(w0)
        for t in 1:T
            w = w .* (1 - 0.3 .+ 0.3 .* X[t, :] ./ dot(w, X[t, :]))
            @test isapprox(W[t + 1, :], w; atol = 1e-14)
        end
        # A rate slot answers the same rate at every step, so the pull vanishes; a rate
        # that falls pulls towards the Start Allocation by the online form's fixed share.
        @test po.learning_rate(0.3, 7, nothing) == 0.3
        dalg = ExpectationMaximisation(; eta = DecayRate(0.3))
        wd = copy(w0)
        stt = po.rule_state_seed(dalg, w0)
        for t in 1:5
            stt, wn = po.online_update!(dalg, stt, wd, X[t, :], nothing, simplex)
            eta_t, eta_n = 0.3 / sqrt(t), 0.3 / sqrt(t + 1)
            plain = wd .* (1 - eta_t .+ eta_t .* X[t, :] ./ dot(wd, X[t, :]))
            @test isapprox(wn, plain .* (eta_n / eta_t) .+ (1 - eta_n / eta_t) .* w0;
                           atol = 1e-14)
            @test isapprox(sum(wn), 1; atol = 1e-14)
            wd = wn
        end
        @test_throws DomainError ExpectationMaximisation(; eta = 0)
        @test_throws DomainError ExpectationMaximisation(; eta = 1)
        @test_throws TypeError ExpectationMaximisation(; proj = EntropicProjection())
        # The carrier: a view renormalises the prior, a copy aliases nothing.
        v = po.port_opt_view(st, [1, 2])
        @test v.w1 == [0.5, 0.5] && v.n == 1
        c = copy(st)
        @test c.w1 == st.w1 && c.w1 !== st.w1
        @test_throws ArgumentError po.merge_states(st, c)
        # As a weighting over experts.
        mix = ExpertMixture(; experts = [BuyAndHold(), ExponentiatedGradient()],
                            alg = ExpectationMaximisation(; eta = 0.5))
        r = optimise(OPS(; alg = mix), rd)
        @test isapprox(sum(r.w), 1; atol = 1e-12)
    end

    @testset "Aggregating algorithm" begin
        x = X[1, :]
        _, w1 = step(AggregatingAlgorithm(; eta = 0.5), w0, x)
        @test isapprox(w1, w0 .* x .^ 0.5 ./ sum(w0 .* x .^ 0.5); atol = 1e-15)
        # At eta = 1 it is buy-and-hold, on the head and on the mixture.
        @test isapprox(libpath(AggregatingAlgorithm()), libpath(BuyAndHold()); atol = 1e-14)
        up = UniversalPortfolio(; N = N, n_experts = 30, seed = 5)
        upa = UniversalPortfolio(; N = N, n_experts = 30, seed = 5,
                                 alg = AggregatingAlgorithm())
        @test isapprox(libpath(up), libpath(upa); atol = 1e-14)
        # Below one it discounts the evidence: the weights sit between uniform and
        # buy-and-hold's.
        W = libpath(AggregatingAlgorithm(; eta = 0.5))
        Wb = libpath(BuyAndHold())
        @test all(abs.(W[end, :] .- 1 / N) .< abs.(Wb[end, :] .- 1 / N) .+ 1e-12)
        @test isnothing(po.rule_state_seed(AggregatingAlgorithm(), w0))
        @test po.rows_needed(AggregatingAlgorithm()) == 0
        @test_throws DomainError AggregatingAlgorithm(; eta = 0)
        @test_throws TypeError AggregatingAlgorithm(; proj = EuclideanProjection())
    end

    @testset "Top-k selection" begin
        alg = TopK(; k = 2)
        W = libpath(alg)
        G = zeros(N)
        for t in 1:T
            G .+= log.(X[t, :])
            top = sortperm(G; rev = true)[1:2]
            q = zeros(N)
            q[top] .= 0.5
            @test W[t + 1, :] == q
        end
        # Equal wealth breaks ties by index.
        st = po.rule_state_seed(alg, w0)
        _, w1 = po.online_update!(alg, st, w0, ones(N), nothing, simplex)
        @test w1 == [0.5, 0.5, 0.0, 0.0]
        @test st.G == zeros(N) && st.p0 == w0
        # At k = 1 the mixture holds its best expert's next allocation.
        experts = [BuyAndHold(), ExponentiatedGradient(; eta = 0.5), NewtonStep()]
        mix = ExpertMixture(; experts = experts, alg = TopK(; k = 1))
        o = po.partial_fit!(OPS(; alg = mix), rows(rd, 1:20))
        Se = [prod(dot(h, x)
                   for (h, x) in zip(eachrow(libpath(e)[1:20, :]), eachrow(X[1:20, :])))
              for e in experts]
        best = argmax(Se)
        @test optimise(o).w == libpath(experts[best])[21, :]
        @test_throws DomainError TopK(; k = 0)
        @test_throws DomainError optimise(OPS(; alg = TopK(; k = N + 1)), rd)
        @test_throws TypeError TopK(; proj = EntropicProjection())
        # The carrier: a view slices the wealth and renormalises the prior.
        st2, _ = po.online_update!(alg, po.rule_state_seed(alg, w0), w0, X[1, :], nothing,
                                   simplex)
        v = po.port_opt_view(st2, [2, 4])
        @test v.G == st2.G[[2, 4]] && v.p0 == [0.5, 0.5] && v.n == 1
        c = copy(st2)
        @test c.G == st2.G && c.G !== st2.G && c.p0 !== st2.p0
        @test_throws ArgumentError po.merge_states(st2, c)
        # Under a cap the selection is projected in the Euclidean geometry.
        _, wc = po.online_update!(TopK(; k = 1), po.rule_state_seed(TopK(; k = 1), w0), w0,
                                  X[1, :], nothing, cap)
        @test maximum(wc) <= 0.4 + 1e-12 && isapprox(sum(wc), 1; atol = 1e-12)
    end

    @testset "Weak aggregating algorithm and the exponentiated-gradient constructor" begin
        etas = [0.05, 0.1, 0.2]
        mix = AggregatingExponentialGradient(; etas = etas)
        @test isa(mix, ExpertMixture) && isa(mix.alg, WeakAggregatingAlgorithm)
        @test [e.eta for e in mix.experts] == etas
        @test length(AggregatingExponentialGradient().experts) == 20
        # The papers' recursion by hand: experts' paths, cumulative log wealth, the weight
        # p0 exp(G_t / sqrt(t + 1)), the linear blend of the experts' next allocations.
        E = [libpath(ExponentiatedGradient(; eta = eta)) for eta in etas]
        W = libpath(mix)
        G = zeros(length(etas))
        @test W[1, :] == w0
        for t in 1:T
            for (k, e) in enumerate(E)
                G[k] += log(dot(e[t, :], X[t, :]))
            end
            p = exp.(G ./ sqrt(t + 1)) ./ 3
            p ./= sum(p)
            @test isapprox(W[t + 1, :], sum(p[k] .* E[k][t + 1, :] for k in 1:3);
                           atol = 1e-13)
        end
        # The decay distinguishes it from the wealth weighting and from the top expert:
        # after t rows the log-odds between two experts are their wealth gap over
        # sqrt(t + 1), not the gap itself.
        o = po.partial_fit!(OPS(; alg = mix), rows(rd, 1:T))
        p = o.cache.st.p
        @test isapprox(log(p[1] / p[3]), (G[1] - G[3]) / sqrt(T + 1); atol = 1e-12)
        ob = po.partial_fit!(OPS(; alg = ExpertMixture(; experts = mix.experts)),
                             rows(rd, 1:T))
        @test isapprox(log(ob.cache.st.p[1] / ob.cache.st.p[3]), G[1] - G[3]; atol = 1e-12)
        # The prior is the mixture's p.
        mixp = ExpertMixture(; experts = mix.experts, alg = WeakAggregatingAlgorithm(),
                             p = [0.5, 0.25, 0.25])
        op = po.partial_fit!(OPS(; alg = mixp), rows(rd, 1:T))
        pp = op.cache.st.p
        @test isapprox(log(pp[1] / pp[3]), log(2) + (G[1] - G[3]) / sqrt(T + 1);
                       atol = 1e-12)
        @test op.cache.st.pst.p0 == [0.5, 0.25, 0.25]
        # On the head it weights the assets by their wealth at the shrinking rate.
        Wh = libpath(WeakAggregatingAlgorithm())
        Gh = zeros(N)
        for t in 1:T
            Gh .+= log.(X[t, :])
            q = w0 .* exp.(Gh ./ sqrt(t + 1))
            @test isapprox(Wh[t + 1, :], q ./ sum(q); atol = 1e-14)
        end
        @test po.rows_needed(WeakAggregatingAlgorithm()) == 0
        @test_throws TypeError WeakAggregatingAlgorithm(; proj = EuclideanProjection())
        @test_throws Exception AggregatingExponentialGradient(; etas = Float64[])
        @test_throws Exception AggregatingExponentialGradient(; etas = [0.1, -0.1])
    end

    @testset "The batch-online identity is exact at every block size" begin
        for alg in (ConfidenceWeightedMeanReversion(),
                    ConfidenceWeightedMeanReversion(; formulation = StandardDeviationUpdate()),
                    AntiCorrelation(; window = 3), ExpectationMaximisation(),
                    AggregatingAlgorithm(; eta = 0.5), TopK(; k = 2), WeakAggregatingAlgorithm(),
                    AggregatingExponentialGradient(; etas = [0.05, 0.1]),
                    ExpertMixture(; experts = [BuyAndHold(), NewtonStep()], alg = TopK(; k = 1)),
                    ExpertMixture(; experts = [BuyAndHold(), NewtonStep()],
                                  alg = AggregatingAlgorithm(; eta = 0.5)),
                    ExpertMixture(; experts = [BuyAndHold(), NewtonStep()],
                                  alg = ExpectationMaximisation()))
            opt = OPS(; alg = alg)
            o = po.partial_fit!(opt, rows(rd, 1:10))
            o = po.partial_fit!(o, rows(rd, 11:17))
            o = po.partial_fit!(o, rows(rd, 18:18))
            a = optimise(o)
            b = optimise(opt, rows(rd, 1:18))
            @test a.w == b.w
            @test isa(a.retcode, OptimisationSuccess)
            @test po.observation_count(o) == 18
            # A view of the stepped head slices every carrier of the tree.
            v = po.port_opt_view(o, [1, 3])
            @test length(optimise(v).w) == 2
        end
    end
end
