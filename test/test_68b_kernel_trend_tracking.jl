#=
The kernel-based trend pattern tracking of the forecast-reading arm (#1191): the elastic-net
path solved at its middle point, the three-state statistic that folds with a memory, the
memory on the price-level state, and the kernel-scaled unnormalised tracking step. Parity is
measured against a hand-written recursion, the optimality conditions of the elastic net, and
the worked step the paper prints for five assets at its defaults.
=#
@testset "Kernel trend pattern tracking: the path, the statistic with a memory, the step" begin
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
    maxerr(a, b) = maximum(abs.(a .- b))
    # Levels from a matrix of relatives, the last at one.
    function levels(Y)
        P = ones(size(Y, 1) + 1, size(Y, 2))
        for t in size(Y, 1):-1:1
            P[t, :] .= P[t + 1, :] ./ Y[t, :]
        end
        return P
    end

    @testset "The elastic-net path at its middle point" begin
        path = ElasticNetPath()
        @test path.theta == 0.99 &&
              path.ratio == 1e-3 &&
              path.iters == 1000 &&
              path.tol == 1e-10
        # Orthonormal columns have the closed form `S(Pᵀy, λϑ) / (1 + λ(1 − ϑ))`.
        Q = Matrix(qr(randn(rng, 8, 3)).Q)[:, 1:3]
        y = Q * [0.9, -0.3, 0.05] .+ 0.01 .* randn(rng, 8)
        g = Q' * y
        lmax = maximum(abs, g) / path.theta
        lam = lmax * sqrt(path.ratio)
        closed = sign.(g) .* max.(abs.(g) .- lam * path.theta, 0) ./
                 (1 + lam * (1 - path.theta))
        z = po.elastic_net_path(path, Q, y)
        @test isapprox(z, closed; atol = 1e-9)
        # On collinear columns the optimality conditions hold at the solution: the residual
        # correlation of an active coefficient is the penalty's subgradient, and an inactive
        # one's is inside it.
        P = 1 .+ 0.02 .* randn(rng, 4, 5)
        y = P[:, end] .* (1 .+ 0.03 .* randn(rng, 4))
        lmax = maximum(abs, P' * y) / path.theta
        lam = lmax * sqrt(path.ratio)
        z = po.elastic_net_path(path, P, y)
        corr = P' * (y .- P * z)
        for k in 1:5
            if iszero(z[k])
                @test abs(corr[k]) <= lam * path.theta + 1e-8
            else
                @test isapprox(corr[k],
                               lam * path.theta * sign(z[k]) +
                               lam * (1 - path.theta) * z[k]; atol = 1e-8)
            end
        end
        # The middle point is sparse where the strength is felt; a zero response is zero;
        # and a pattern that is not the optimum's is refused by the polish.
        @test count(!iszero, z) < 5
        @test po.elastic_net_path(path, P, zeros(4)) == zeros(5)
        @test isnothing(po.elastic_net_polish(P, y, zeros(5), lam, path.theta))
        @test isnothing(po.elastic_net_polish(P, y, -z, lam, path.theta))
        @test po.elastic_net_polish(P, y, z, lam, path.theta) ≈ z
        # Without the polish the sweeps alone reach the same point, slowly.
        slow = ElasticNetPath(; iters = 200_000, tol = 1e-16)
        @test isapprox(po.elastic_net_path(slow, P, y), z; atol = 1e-8)
        @test_throws DomainError ElasticNetPath(; theta = 0)
        @test_throws DomainError ElasticNetPath(; theta = 1.1)
        @test_throws DomainError ElasticNetPath(; ratio = 1)
        @test_throws DomainError ElasticNetPath(; iters = 0)
        @test_throws DomainError ElasticNetPath(; tol = 0)
    end

    @testset "The trend-reverting fraction" begin
        # One asset, alternating levels: every level from the third is a turning point.
        @test po.trend_reverting_fraction(reshape([1.0, 2, 1, 2], :, 1)) == 1
        # Monotone levels have none; below three levels the fraction is zero.
        @test po.trend_reverting_fraction(reshape([1.0, 2, 3, 4], :, 1)) == 0
        @test po.trend_reverting_fraction(ones(2, 3)) == 0
        # Two assets, one alternating and one monotone: half the cells.
        @test po.trend_reverting_fraction([1.0 1; 2 2; 1 3; 2 4]) == 0.5
    end

    @testset "The statistic: the hand recursion, the fold, the batch and the memory" begin
        w, nu = 5, 0.5
        alg = KernelTrendPattern(; window = w, nu = nu)
        @test alg.window == 5 && alg.nu == 0.5 && alg.path == ElasticNetPath()
        @test po.folds(alg) && isnothing(po.window_rows(alg)) && po.memory_rows(alg) == 2w
        @test po.rows_needed(alg) == 1
        # The paper's three states, row by row, on the levels the memory reaches; the
        # first prediction is the current price, and every later one is the previous row's
        # brought into the current level's units.
        phat = ones(N)
        hand = zeros(T, N)
        for t in 1:T
            H = X[max(1, t - 2w + 1):t, :]
            P = levels(H)
            L = size(P, 1)
            Pw = P[max(1, L - w + 1):L, :]
            ptilde = vec(maximum(Pw; dims = 1))
            y = nu .* ptilde .+ (1 - nu) .* (t == 1 ? phat : phat ./ X[t, :])
            Pt = permutedims(Pw)
            yhat = max.(Pt * po.elastic_net_path(alg.path, Pt, y), 0)
            lam = po.trend_reverting_fraction(P)
            c = min.(lam ./ (2 .* X[t, :]), 1)
            phat = c .* ptilde .+ (1 .- c) .* yhat
            hand[t, :] .= phat
        end
        me = PriceLevelExpectedReturns(; alg = alg)
        mf = me
        for t in 1:T
            mf = po.partial_fit!(mf, R[t, :])
            @test isapprox(1 .+ vec(mean(mf)), hand[t, :]; atol = 1e-12)
            @test size(mf.cache.hist, 1) == min(t, 2w)
        end
        @test mf.cache.n == T
        @test isapprox(mf.cache.hist, X[(T - 2w + 1):T, :]; atol = 1e-15)
        # The batch over the returns is the same recursion, to rounding.
        @test maxerr(vec(mean(mf)), vec(mean(me, R))) < 1e-12
        @test maxerr(vec(mean(po.partial_fit!(me, R))), vec(mean(me, R))) < 1e-15
        @test po.supports_partial_fit(me) && po.rows_needed(me) == 1
        # The cold start: after one row two levels exist, the fraction is zero, and the
        # prediction is the regression of the peak-and-price mix on the two levels.
        one_row = po.partial_fit!(me, R[1, :])
        @test one_row.cache.n == 1 && size(one_row.cache.hist) == (1, N)
        P1 = levels(X[1:1, :])
        y1 = nu .* vec(maximum(P1; dims = 1)) .+ (1 - nu)
        Pt1 = permutedims(P1)
        @test isapprox(1 .+ vec(mean(one_row)),
                       max.(Pt1 * po.elastic_net_path(alg.path, Pt1, y1), 0); atol = 1e-12)
        # A flat panel is shrunk alike in every asset: the middle of the path takes
        # `sqrt(ratio)` off the regression of ones on ones, the ridge share a little more,
        # and the recursion compounds it to the fixed point `p̂ = s (ν + (1 − ν) p̂)`, so the
        # centred forecast is zero.
        flat = vec(mean(me, zeros(12, N)))
        shrink = 1 - sqrt(alg.path.ratio)
        @test isapprox(flat, fill(shrink * nu / (1 - shrink * (1 - nu)) - 1, N);
                       atol = 5e-4)
        @test maxerr(flat, fill(flat[1], N)) < 1e-12
        # The state with a memory pays the partial-fit interface: copy aliases nothing, a
        # view slices the memory's columns, and the merge is refused.
        st = mf.cache
        c = copy(st)
        @test c.hist == st.hist && c.hist !== st.hist && c.stat !== st.stat
        v = po.port_opt_view(st, [2, 4])
        @test v.hist == st.hist[:, [2, 4]] && v.stat == st.stat[[2, 4]]
        @test po.port_opt_view(mf, [1, 3]).cache.hist == st.hist[:, [1, 3]]
        @test_throws ArgumentError po.merge_states(st, c)
        # A statistic without a memory carries `nothing` there, and its fold is unchanged.
        ema = po.partial_fit!(PriceLevelExpectedReturns(; alg = ExponentialMovingAverage()),
                              R[1:3, :])
        @test isnothing(ema.cache.hist) && po.memory_rows(ExponentialMovingAverage()) == 0
        @test isnothing(copy(ema.cache).hist) &&
              isnothing(po.port_opt_view(ema.cache, [1]).hist)
        @test isnothing(po.push_memory(ExponentialMovingAverage(), nothing, X[1, :]))
        @test_throws DomainError KernelTrendPattern(; window = 1)
        @test_throws DomainError KernelTrendPattern(; nu = -0.1)
        @test_throws DomainError KernelTrendPattern(; nu = 1.5)
    end

    @testset "The step: the paper's five-asset example, the hold, the ball" begin
        # Table 3 of the paper: the forecast, the one-hot allocation, the kernel and the
        # next allocation at the paper's defaults. The printed forecast is rounded to four
        # digits, which at `eta = 1000` moves the projected answer at the third.
        xh = [0.8677, 0.8697, 0.8223, 0.8215, 0.8235]
        w = [0.0, 1.0, 0.0, 0.0, 0.0]
        dev = xh .- mean(xh)
        K = exp.(-abs.((w .- mean(w)) .- dev) .^ (1 / 6))
        @test isapprox(K, [0.4580, 0.3838, 0.4713, 0.4715, 0.4709]; atol = 1e-4)
        set5 = po.resolve_allocation_set(BoundedAllocationSet(), 5, false, Float64)
        alg = KernelTrendTracking(; me = CustomValueExpectedReturns(; val = xh .- 1))
        @test alg.eta == 1000 && alg.q == 6
        _, w2 = po.online_update!(alg, nothing, w, ones(5), nothing, set5)
        @test isapprox(w2, [0.6111, 0.3889, 0, 0, 0]; atol = 5e-3)
        @test w2 ≈ po.project_simplex(w .+ 1000 .* K .* dev)
        # The step is not normalised: doubling the centred forecast doubles the raw step.
        alg2 = KernelTrendTracking(; me = CustomValueExpectedReturns(; val = 2 .* dev),
                                   eta = 0.1)
        alg1 = KernelTrendTracking(; me = CustomValueExpectedReturns(; val = dev),
                                   eta = 0.1)
        K1 = exp.(-abs.((w .- mean(w)) .- dev) .^ (1 / 6))
        K2 = exp.(-abs.((w .- mean(w)) .- 2 .* dev) .^ (1 / 6))
        _, s1 = po.online_update!(alg1, nothing, w, ones(5), nothing, set5)
        _, s2 = po.online_update!(alg2, nothing, w, ones(5), nothing, set5)
        @test s1 ≈ po.project_simplex(w .+ 0.1 .* K1 .* dev)
        @test s2 ≈ po.project_simplex(w .+ 0.2 .* K2 .* dev)
        @test count(>(0), s1) > 1
        # A flat forecast holds, and so does the statistic's on a flat panel.
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        u = fill(1 / N, N)
        flat = KernelTrendTracking(; me = CustomValueExpectedReturns(; val = zeros(N)))
        _, wh = po.online_update!(flat, nothing, u, X[5, :], nothing, set)
        @test wh == u
        flat_rd = ReturnsResult(; nx = nx, X = zeros(12, N))
        @test optimise(OPS(; alg = KernelTrendPatternTracking()), flat_rd).w ≈ u
        # At the default the answer through the statistic is one-hot on real data.
        opt = OPS(; alg = KernelTrendPatternTracking())
        wo = optimise(opt, rows(rd, 1:18)).w
        @test count(>(1e-12), wo) == 1 && sum(wo) ≈ 1
        # Validation and the forecaster slot.
        @test_throws DomainError KernelTrendTracking(; eta = 0)
        @test_throws DomainError KernelTrendTracking(; q = 0)
        @test_throws ArgumentError KernelTrendTracking(;
                                                       me = Online(SimpleExpectedReturns()))
        @test po.rows_needed(KernelTrendTracking()) == 1
        @test po.rows_needed(KernelTrendTracking(; me = PriceLevelExpectedReturns())) == 4
        @test po.port_opt_view(KernelTrendTracking(; eta = 3, q = 4), [1, 2]).q == 4
        # The paper's constructor addresses every parameter.
        k = KernelTrendPatternTracking(; window = 3, nu = 0.4, theta = 0.9, q = 5,
                                       eta = 900)
        @test k.me.alg.window == 3 &&
              k.me.alg.nu == 0.4 &&
              k.me.alg.path.theta == 0.9 &&
              k.q == 5 &&
              k.eta == 900
        @test isa(KernelTrendPatternTracking().me.alg, KernelTrendPattern)
    end

    @testset "The head's verbs" begin
        for alg in (KernelTrendPatternTracking(),
                    KernelTrendTracking(; me = SimpleExpectedReturns(), eta = 0.5))
            opt = OPS(; alg = alg)
            @test po.rows_needed(opt) == 1
            o = po.partial_fit!(opt, rows(rd, 1:10))
            @test o.cache.X.n == 1 && isa(o.cache.st, po.ForecasterState)
            o = po.partial_fit!(o, rows(rd, 11:17))
            o = po.partial_fit!(o, rows(rd, 18:18))
            a = optimise(o)
            b = optimise(opt, rows(rd, 1:18))
            @test a.w == b.w
            @test sum(a.w) ≈ 1 && all(a.w .>= -1e-12)
            @test isnothing(a.pr) && isa(a.retcode, OptimisationSuccess)
            # The folded forecast equals the batch forecast over the same rows.
            folded = 1 .+ vec(mean(o.cache.st.me))
            batch = 1 .+ vec(mean(alg.me, R[1:18, :]))
            @test maxerr(folded, batch) < 1e-12
            # A bounded set is honoured in the rule's Euclidean geometry.
            capped = OPS(; alg = alg,
                         set = BoundedAllocationSet(;
                                                    wb = WeightBounds(; lb = 0, ub = 0.5)))
            wc = optimise(capped, rows(rd, 1:18)).w
            @test all(wc .<= 0.5 + 1e-12) && sum(wc) ≈ 1
            # A view of the head slices the rule, its carrier and the carrier's memory.
            v = po.port_opt_view(o, [1, 2, 4])
            @test length(optimise(v).w) == 3
            c = copy(o.cache)
            @test c.st.me.cache !== o.cache.st.me.cache
            # A walk-forward at test_size = 1 through the online arm.
            cv = OnlineIndexWalkForward(5, 1)
            pred = cross_val_predict(opt, rows(rd, 1:12), cv)
            @test isapprox(pred.pred[1].res.w, optimise(opt, rows(rd, 1:5)).w; atol = 1e-14)
        end
        h = po.partial_fit!(OPS(; alg = KernelTrendPatternTracking()), rows(rd, 1:12))
        @test size(h.cache.st.me.cache.hist) == (10, N)
    end

    @testset "Show" begin
        @test occursin("KernelTrendTracking",
                       sprint(show, MIME("text/plain"), KernelTrendPatternTracking()))
        @test occursin("ElasticNetPath",
                       sprint(show, MIME("text/plain"), KernelTrendPattern()))
    end
end
