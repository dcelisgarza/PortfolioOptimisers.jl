#=
The first-order rule of the online portfolio selection family, as ADR 0165 rules it (issue
#1180 on map #1148): `MirrorDescent` over the four scalar-root geometries, the uniform mix,
the Tsallis and log-barrier projections, the Learning-Rate Schedules and the momentum
Gradient Transforms.

Every literal below states its provenance: the exponentiated-gradient identity is with the
`#1161` fixture through the constructor; the `alpha = 0.2` step is Helmbold's Theorem 4.2 on
one row; the Euclidean, momentum and scheduled paths are hand recursions of the stated
formulas; the barrier roots are checked against an independent bisection at `1e-12`; the
doubling trick's first stage length is Corollary 4.3's `ceil(2 N^2 log N)`. The Risk Loss
(issue #1182) is checked against a hand recursion on the window's sample covariance. The
windowed best rate (issue #1187) is checked against a hand recursion of its paper's
Algorithm 2 at the paper's defaults.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates, Clarabel,
      JuMP
@testset "Online portfolio selection: mirror descent, its geometries and schedules" begin
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection

    rng = StableRNG(7)
    T, N = 40, 4
    R = 0.02 .* randn(rng, T, N)
    X = 1 .+ R
    nx = ["A", "B", "C", "D"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
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
    # A hand recursion over the rows: `f(w, x, t)` answers the next allocation.
    function handpath(f)
        W = zeros(T, N)
        W[1, :] .= 1 / N
        for t in 1:(T - 1)
            W[t + 1, :] .= f(W[t, :], X[t, :], t)
        end
        return W
    end
    maxerr(a, b) = maximum(abs.(a .- b))

    @testset "Construction and the refusals" begin
        md = MirrorDescent()
        @test md.eta == 0.05 && isa(md.proj, EntropicProjection) && md.alpha == 0
        @test isa(md.obj, LogWealth) && isnothing(md.grad)
        # `grad = nothing` applies no transform: no state, and the gradient itself (#1867).
        @test isnothing(po.gradient_state_seed(nothing, [0.5, 0.5]))
        g0 = [0.25, -0.5]
        @test po.transform_gradient!(nothing, nothing, g0) === g0
        @test isa(ExponentiatedGradient(), MirrorDescent)
        @test isa(GradientProjection().proj, EuclideanProjection)
        @test_throws DomainError MirrorDescent(; eta = 0)
        @test_throws DomainError MirrorDescent(; eta = -0.1)
        @test_throws DomainError MirrorDescent(; alpha = 1)
        @test_throws DomainError MirrorDescent(; alpha = -0.1)
        @test_throws TypeError MirrorDescent(;
                                             proj = GramProjection(;
                                                                   slv = Solver(;
                                                                                solver = nothing)))
        # The constructors fill `proj`; a keyword for it no longer exists.
        @test_throws MethodError ExponentiatedGradient(; proj = EuclideanProjection())
        @test_throws DomainError TsallisProjection(; alpha = 0)
        @test_throws DomainError TsallisProjection(; alpha = 1)
        @test_throws DomainError InverseSquareRootRate(; c = 0)
        @test_throws DomainError DoublingTrickRate(; N = 1)
        @test_throws UndefKeywordError DoublingTrickRate()
        @test_throws DomainError SelfConfidentRate(; eta_max = 0)
        @test_throws DomainError GradientMomentum(; gamma1 = 1)
        @test_throws DomainError RootMeanSquareGradient(; gamma2 = -0.1)
        @test_throws DomainError RootMeanSquareGradient(; eps = 0)
        @test_throws DomainError AdaptiveMomentGradient(; gamma1 = 1)
        @test_throws DomainError AdaptiveMomentGradient(; gamma2 = 1)
        @test_throws DomainError AdaptiveMomentGradient(; eps = 0)
        @test isa(EGE().grad, GradientMomentum) && EGE(; gamma1 = 0.5).grad.gamma1 == 0.5
        @test isa(EGR().grad, RootMeanSquareGradient) && EGR().grad.gamma2 == 0
        @test isa(EGA().grad, AdaptiveMomentGradient) && EGA(; eps = 1e-6).grad.eps == 1e-6
        # A schedule on `eta` needs no positivity check of its own.
        @test isa(MirrorDescent(; eta = InverseSquareRootRate()).eta, InverseSquareRootRate)
        # A negative lower bound is refused under both barriers where the rule and the
        # set meet, as under the entropic map.
        for proj in (TsallisProjection(), LogBarrierProjection())
            @test_throws DomainError OPS(; alg = MirrorDescent(; proj = proj),
                                         set = BoundedAllocationSet(;
                                                                    wb = WeightBounds(;
                                                                                      lb = -0.1,
                                                                                      ub = 1)))
        end
        @test_nowarn OPS(; alg = GradientProjection(),
                         set = BoundedAllocationSet(;
                                                    wb = WeightBounds(; lb = -0.1, ub = 1)))
    end

    @testset "The exponentiated gradient is the entropic step at alpha = 0" begin
        # The constructor and the explicit rule are one object, so their paths agree at
        # atol = 0; the #1161 parity with the prototype holds in test_67.
        a = libpath(ExponentiatedGradient())
        b = libpath(MirrorDescent(; eta = 0.05, proj = EntropicProjection()))
        @test a == b
        c = handpath((w, x, t) -> (q = w .* exp.(0.05 .* x ./ dot(w, x)); q ./ sum(q)))
        @test maxerr(a, c) < 1e-14
        # At alpha = 0 the carrier is the played allocation.
        o = po.partial_fit!(OPS(; alg = ExponentiatedGradient()), rows(rd, 1:7))
        @test o.cache.st.u == o.cache.w
        @test o.cache.st.n == 7
        @test isnothing(o.cache.st.s) && isnothing(o.cache.st.gs)
    end

    @testset "The uniform mix: Helmbold's Theorem 4.2 on one row" begin
        # x = [1.2, 0.8], w = [0.5, 0.5], eta = 0.05, alpha = 0.2: the paper's mix is over
        # the relatives normalised to a period maximum of one, [1, 2/3], so the update reads
        # (1 - 0.1) [1, 2/3] + 0.1 = [1, 0.7] up to the scale the step is invariant to —
        # [1.2, 0.84] in the relatives' own units — at the unmixed iterate, stores
        # [0.5044, 0.4956] and plays [0.5035, 0.4965]. The mix of the raw relatives with
        # an unscaled floor, [1.18, 0.82], is a different step ([0.5045, 0.4955]).
        alg = ExponentiatedGradient(; alpha = 0.2)
        w = [0.5, 0.5]
        x = [1.2, 0.8]
        st = po.rule_state_seed(alg, w)
        st, wn = po.online_update!(alg, st, w, x, nothing,
                                   resolve(BoundedAllocationSet(), 2))
        xm = 0.9 .* (x ./ 1.2) .+ 0.1
        @test po.mixed_relatives(x, 0.2) == 0.9 .* x .+ 0.1 * 1.2
        @test po.mixed_relatives(x, 0) === x
        q = w .* exp.(0.05 .* xm ./ dot(w, xm))
        u = q ./ sum(q)
        @test isapprox(st.u, u; atol = 1e-15)
        @test round.(st.u; digits = 4) == [0.5044, 0.4956]
        @test isapprox(wn, 0.8 .* u .+ 0.1; atol = 1e-15)
        @test round.(wn; digits = 4) == [0.5035, 0.4965]
        # The raw mix is not what the rule takes.
        xr = 0.9 .* x .+ 0.1
        qr = w .* exp.(0.05 .* xr ./ dot(w, xr))
        @test maxerr(st.u, qr ./ sum(qr)) > 1e-5
        # The carrier holds the unmixed iterate over the run, and the head plays the mix.
        o = po.partial_fit!(OPS(; alg = alg), rows(rd, 1:5))
        @test isapprox(o.cache.w, 0.8 .* o.cache.st.u .+ 0.2 / N; atol = 1e-15)
        # The mix is admitted under every geometry, and every entry is at least alpha / N.
        for proj in (EuclideanProjection(), TsallisProjection(), LogBarrierProjection())
            W = libpath(MirrorDescent(; proj = proj, alpha = 0.2))
            @test all(W .>= 0.2 / N - 1e-15)
            @test all(isapprox.(sum(W; dims = 2), 1; atol = 1e-12))
        end
    end

    @testset "The Euclidean map is online gradient descent" begin
        a = libpath(GradientProjection(; eta = 0.3))
        b = handpath((w, x, t) -> po.project_simplex(w .+ 0.3 .* x ./ dot(w, x)))
        @test maxerr(a, b) < 1e-14
        @test_throws MethodError GradientProjection(; proj = EntropicProjection())
    end

    @testset "The barrier roots against an independent bisection" begin
        # An independent bisection on the budget multiplier, the clip written out.
        function root(phi, b, lb, ub)
            f(mu) = sum(clamp.(phi.(b .+ mu), lb, ub))
            lo, hi = -maximum(b), 1.0
            while f(hi) > 1
                hi *= 2
            end
            for _ in 1:400
                mid = (lo + hi) / 2
                (f(mid) >= 1 ? (lo = mid) : (hi = mid))
            end
            return clamp.(phi.(b .+ (lo + hi) / 2), lb, ub)
        end
        q = [0.5, 0.3, 0.4, 0.1]
        wh = fill(0.25, 4)
        capped = resolve(BoundedAllocationSet(; wb = WeightBounds(; lb = 0.05, ub = 0.4)),
                         4)
        for (set, lb, ub) in
            ((simplex, zeros(4), ones(4)), (capped, fill(0.05, 4), fill(0.4, 4)))
            lbp = po.project(LogBarrierProjection(), set, q, wh)
            @test isapprox(lbp, root(m -> m > 0 ? 1 / m : Inf, 1 ./ q, lb, ub);
                           atol = 1e-12)
            @test isapprox(sum(lbp), 1; atol = 1e-12)
            for a in (0.3, 0.5, 0.8)
                tp = po.project(TsallisProjection(; alpha = a), set, q, wh)
                @test isapprox(tp,
                               root(m -> m > 0 ? m^(1 / (a - 1)) : Inf, q .^ (a - 1), lb,
                                    ub); atol = 1e-12)
                @test isapprox(sum(tp), 1; atol = 1e-12)
            end
        end
        # On the simplex the log-barrier root keeps the order of the raw step and never
        # zeroes a positive entry; a zero stays zero.
        lbp = po.project(LogBarrierProjection(), simplex, q, wh)
        @test issorted(lbp[sortperm(q)]) && all(lbp .> 0)
        z = po.project(LogBarrierProjection(), simplex, [0.5, 0.0, 0.3, 0.2], wh)
        @test z[2] == 0 && isapprox(sum(z), 1; atol = 1e-12)
        zt = po.project(TsallisProjection(), simplex, [0.5, 0.0, 0.3, 0.2], wh)
        @test zt[2] == 0 && isapprox(sum(zt), 1; atol = 1e-12)
        # The refusals: a negative or non-finite raw entry, a negative floor, a raw step
        # of zeros, and zeros that pin more mass than the caps leave.
        for proj in (TsallisProjection(), LogBarrierProjection())
            @test_throws DomainError po.project(proj, simplex, [0.5, -0.1, 0.3, 0.3], wh)
            @test_throws DomainError po.project(proj, simplex, [0.5, Inf, 0.3, 0.3], wh)
            @test_throws DomainError po.project(proj, simplex, zeros(4), wh)
            @test_throws DomainError po.project(proj, capped, [1.0, 0.0, 0.0, 0.0], wh)
            neg = resolve(BoundedAllocationSet(; wb = WeightBounds(; lb = -0.1, ub = 1)), 4)
            @test_throws DomainError po.project(proj, neg, q, wh)
        end
        # A set whose floors sum to one is that point.
        pinned = resolve(BoundedAllocationSet(; wb = WeightBounds(; lb = 0.25, ub = 1)), 4)
        @test po.project(LogBarrierProjection(), pinned, q, wh) == fill(0.25, 4)
        # The mirror step in each geometry, and its domain.
        u = [0.5, 0.5]
        s = [-0.1, 0.2]
        @test po.mirror_step(EuclideanProjection(), u, s) == u .- s
        @test po.mirror_step(EntropicProjection(), u, s) == u .* exp.(-s)
        @test isapprox(po.mirror_step(LogBarrierProjection(), u, s), 1 ./ (2 .+ s))
        @test isapprox(po.mirror_step(TsallisProjection(; alpha = 0.5), u, s),
                       (u .^ (-0.5) .+ 0.5 .* s) .^ (-2))
        # The log-barrier step exists while eta * w_i x_i / <w, x> < 1: at eta = 1.5 an
        # asset that carries 10/11 of the period's wealth leaves the domain.
        x = [10.0, 1.0]
        @test_throws DomainError po.mirror_step(LogBarrierProjection(), u,
                                                -1.5 .* x ./ dot(u, x))
        @test_nowarn po.mirror_step(LogBarrierProjection(), u, -0.9 .* x ./ dot(u, x))
        @test_throws DomainError po.online_update!(MirrorDescent(; eta = 1.5,
                                                                 proj = LogBarrierProjection()),
                                                   po.rule_state_seed(MirrorDescent(), u),
                                                   u, x, nothing,
                                                   resolve(BoundedAllocationSet(), 2))
    end

    @testset "The barrier rules run, and a zero start stays zero" begin
        for proj in (TsallisProjection(), LogBarrierProjection())
            W = libpath(MirrorDescent(; proj = proj))
            @test all(isapprox.(sum(W; dims = 2), 1; atol = 1e-12))
            @test all(W .> 0)
            # A hand recursion of the stated update: the raw mirror step, then the root.
            H = handpath((w, x, t) -> po.project(proj, simplex,
                                                 po.mirror_step(proj, w,
                                                                -0.05 .* x ./ dot(w, x)), w))
            @test maxerr(W, H) < 1e-13
            z = optimise(OPS(; alg = MirrorDescent(; proj = proj),
                             w0 = [0.5, 0.5, 0.0, 0.0]), rows(rd, 1:6)).w
            @test z[3] == 0 && z[4] == 0 && isapprox(sum(z), 1; atol = 1e-12)
        end
    end

    @testset "The batch-online identity at block sizes 10, 7 and 1" begin
        for alg in (MirrorDescent(; proj = TsallisProjection(), alpha = 0.1),
                    MirrorDescent(; proj = LogBarrierProjection(), eta = SelfConfidentRate()),
                    GradientProjection(; eta = InverseSquareRootRate(; c = 0.2)),
                    ExponentiatedGradient(; eta = DoublingTrickRate(; N = 2)), EGE(), EGA())
            opt = OPS(; alg = alg)
            o = po.partial_fit!(opt, rows(rd, 1:10))
            o = po.partial_fit!(o, rows(rd, 11:17))
            o = po.partial_fit!(o, rows(rd, 18:18))
            a = optimise(o)
            b = optimise(opt, rows(rd, 1:18))
            @test a.w == b.w
            @test o.cache.st.n == 18
        end
    end

    @testset "The schedules" begin
        # The inverse square root reads the count alone: eta_t = c / sqrt(t).
        a = libpath(ExponentiatedGradient(; eta = InverseSquareRootRate(; c = 0.2)))
        b = handpath((w, x, t) -> (q = w .* exp.(0.2 / sqrt(t) .* x ./ dot(w, x));
                                   q ./ sum(q)))
        @test maxerr(a, b) < 1e-14
        @test po.learning_rate(InverseSquareRootRate(; c = 0.2), 4, nothing) == 0.1
        @test po.learning_rate(0.05, 4, nothing) == 0.05
        @test !po.restart(0.05, 4) && !po.restart(InverseSquareRootRate(), 4)
        @test isnothing(po.schedule_state_seed(0.05, ones(2)))
        @test po.mixing_share(InverseSquareRootRate(), 3, 0.2) == 0.2

        # The doubling trick on N = 2: stage 0 is ceil(2 * 4 * log 2) = 6 periods, so the
        # update of row 6 answers the start for period 7, and the stage's share and rate
        # are Theorem 4.2's at T_0 = 6.
        sched = DoublingTrickRate(; N = 2)
        @test po.doubling_stage(sched, 1) == (6, 6)
        @test po.doubling_stage(sched, 6) == (6, 6)
        @test po.doubling_stage(sched, 7) == (6, 12)
        @test po.doubling_stage(sched, 13) == (12, 24)
        @test po.restart(sched, 6) && po.restart(sched, 12) && po.restart(sched, 24)
        @test !po.restart(sched, 5) && !po.restart(sched, 7)
        alpha0 = (4 * log(2) / 48)^(1 / 4)
        @test po.mixing_share(sched, 3, 0.4) == alpha0
        @test po.learning_rate(sched, 3, nothing) == sqrt(8 * alpha0^2 * log(2) / (4 * 6))
        rd2 = ReturnsResult(; nx = nx[1:2], X = R[:, 1:2], ts = ts)
        opt = OPS(; alg = ExponentiatedGradient(; eta = sched, alpha = 0.4))
        # The rule's own alpha yields to the stage's.
        o5 = po.partial_fit!(opt, rows(rd2, 1:5))
        @test isapprox(o5.cache.w, (1 - alpha0) .* o5.cache.st.u .+ alpha0 / 2;
                       atol = 1e-15)
        @test o5.cache.w != [0.5, 0.5]
        o6 = po.partial_fit!(opt, rows(rd2, 1:6))
        @test o6.cache.w == [0.5, 0.5]
        @test o6.cache.st.u == [0.5, 0.5] && o6.cache.st.n == 6
        # A given start is what the restart returns to, projected.
        os = po.partial_fit!(OPS(; alg = ExponentiatedGradient(; eta = sched),
                                 w0 = [0.7, 0.3]), rows(rd2, 1:6))
        @test os.cache.w == [0.7, 0.3]
        # The restart carries the momentum's averages back to zero.
        oe = po.partial_fit!(OPS(; alg = EGE(; eta = sched)), rows(rd2, 1:6))
        @test oe.cache.st.gs == zeros(2)

        # The self-confident rate reads the running excess the carrier keeps.
        sc = SelfConfidentRate(; eta_max = 0.8)
        opt = OPS(; alg = ExponentiatedGradient(; eta = sc))
        o = po.partial_fit!(opt, rows(rd, 1:9))
        Wl = libpath(ExponentiatedGradient(; eta = sc))
        C = sum(maximum(X[t, :] ./ dot(Wl[t, :], X[t, :])) - 1 for t in 1:9)
        @test isapprox(o.cache.st.s[1], C; atol = 1e-13)
        @test po.learning_rate(sc, 10, o.cache.st) == min(sqrt(2 * log(N) / C), 0.8)
        # Before the first row the excess is zero and the cap is the rate.
        seed = po.rule_state_seed(ExponentiatedGradient(; eta = sc), fill(0.25, N))
        @test seed.s == [0.0, log(N)]
        @test po.learning_rate(sc, 1, seed) == 0.8
        # The path is the hand recursion under the running rate, read at the played
        # allocation.
        c = 0.0
        H = handpath((w, x, t) -> begin
                         eta = c > 0 ? min(sqrt(2 * log(N) / c), 0.8) : 0.8
                         q = w .* exp.(eta .* x ./ dot(w, x))
                         c += maximum(x ./ dot(w, x)) - 1
                         q ./ sum(q)
                     end)
        @test maxerr(Wl, H) < 1e-13
        # As a mixture's weighting the schedule counts the weighting's own updates.
        mix = ExpertMixture(; experts = [BuyAndHold(), GradientProjection()],
                            alg = ExponentiatedGradient(; eta = sc))
        om = po.partial_fit!(OPS(; alg = mix), rows(rd, 1:5))
        @test om.cache.st.pst.n == 5 && om.cache.st.pst.s[1] > 0

        # The slot on `ExpectationMaximisation`: the online form reads eta_t before the row
        # and eta_{t+1} after it, on the carrier the row wrote, so the pull follows the
        # self-confident rate's fall, capped at √(t / (t + 1)) under this rate; at a constant
        # rate it is the plain step.
        em = ExpectationMaximisation(; eta = SelfConfidentRate(; eta_max = 0.5))
        w = fill(0.25, N)
        st = po.rule_state_seed(em, w)
        @test st.s == [0.0, log(N)]
        c = 0.0
        for t in 1:6
            x = X[t, :]
            eta_t = c > 0 ? min(sqrt(2 * log(N) / c), 0.5) : 0.5
            cn = c + maximum(x ./ dot(w, x)) - 1
            eta_n = min(sqrt(2 * log(N) / cn), 0.5)
            plain = w .* (1 - eta_t .+ eta_t .* x ./ dot(w, x))
            ratio = min(eta_n / eta_t, sqrt(t / (t + 1)))
            st, wn = po.online_update!(em, st, w, x, nothing, simplex)
            @test isapprox(wn, plain .* ratio .+ (1 - ratio) .* fill(0.25, N); atol = 1e-14)
            @test isapprox(st.s[1], cn; atol = 1e-13)
            c, w = cn, wn
        end
        @test_throws DomainError ExpectationMaximisation(; eta = 1)
        @test isa(ExpectationMaximisation(; eta = DoublingTrickRate(; N = 2)).eta,
                  DoublingTrickRate)
        # The doubling trick restarts the expectation-maximisation step too, count kept.
        oem = po.partial_fit!(OPS(; alg = ExpectationMaximisation(; eta = sched)),
                              rows(rd2, 1:6))
        @test oem.cache.w == [0.5, 0.5] && oem.cache.st.n == 6
        for alg in (ExpectationMaximisation(; eta = SelfConfidentRate()),
                    ExpectationMaximisation(; eta = InverseSquareRootRate(; c = 0.3)))
            opt = OPS(; alg = alg)
            o = po.partial_fit!(po.partial_fit!(opt, rows(rd, 1:10)), rows(rd, 11:18))
            @test optimise(o).w == optimise(opt, rows(rd, 1:18)).w
        end
    end

    @testset "The windowed best rate of the moving-window adaptive paper" begin
        # Construction and the refusals.
        @test_throws po.IsEmptyError WindowedBestRate(; etas = Float64[])
        @test_throws DomainError WindowedBestRate(; etas = [0.0, 0.1])
        @test_throws DomainError WindowedBestRate(; etas = [0.1, Inf])
        @test_throws DomainError WindowedBestRate(; window = 0)
        wbr = WindowedBestRate()
        @test wbr.etas == 0.001:0.001:0.2 && length(wbr.etas) == 200 && wbr.window == 30
        @test isnothing(WindowedBestRate(; window = nothing).window)
        @test isa(MAEG(), MirrorDescent) &&
              MAEG().eta == wbr &&
              isa(MAEG().proj, EntropicProjection)
        @test MAEG(; window = 7).eta.window == 7 && MAEG(; alpha = 0.1).alpha == 0.1
        @test isnothing(AEG().eta.window) && AEG().eta.etas == wbr.etas
        # The trait: the rate is chosen from the period's row, and every other schedule
        # and a number read the past alone.
        @test po.reads_period_row(wbr)
        @test !po.reads_period_row(0.05) && !po.reads_period_row(SelfConfidentRate())
        @test !po.reads_period_row(InverseSquareRootRate()) &&
              !po.reads_period_row(DoublingTrickRate(; N = 2))
        @test_throws ArgumentError ExpectationMaximisation(; eta = wbr)
        @test_throws ArgumentError ExpectationMaximisation(; eta = AEG().eta)

        # Algorithm 2 of the paper as a hand recursion over the rows: every expert is the
        # exponentiated gradient at its own rate from the uniform allocation, the window's
        # cumulative return of each is the product of its period returns over the last `w`
        # periods (the whole prefix while fewer have passed), and b_{t+1} is the investor's
        # own exponentiated-gradient step at the best expert's rate, after x_t is received.
        function maegpath(H, w)
            K = length(H)
            B = fill(1 / N, N, K)
            S = zeros(T, K)
            W = zeros(T, N)
            W[1, :] .= 1 / N
            for t in 1:(T - 1)
                x = X[t, :]
                for k in 1:K
                    S[t, k] = dot(B[:, k], x)
                end
                lo = isnothing(w) ? 1 : max(1, t - w + 1)
                wealth = [prod(S[lo:t, k]) for k in 1:K]
                eta = H[argmax(wealth)]
                b = W[t, :]
                q = b .* exp.(eta .* x ./ dot(b, x))
                W[t + 1, :] .= q ./ sum(q)
                for k in 1:K
                    bk = B[:, k]
                    qk = bk .* exp.(H[k] .* x ./ dot(bk, x))
                    B[:, k] .= qk ./ sum(qk)
                end
            end
            return W
        end
        # Parity at the paper's defaults, and under the whole history.
        @test maxerr(libpath(MAEG()), maegpath(0.001:0.001:0.2, 30)) < 1e-13
        @test maxerr(libpath(AEG()), maegpath(0.001:0.001:0.2, nothing)) < 1e-13
        # A short window on a coarse grid, where the chosen rate visibly switches.
        H = [0.01, 0.05, 0.2]
        Wl = libpath(MAEG(; etas = H, window = 3))
        @test maxerr(Wl, maegpath(H, 3)) < 1e-14
        @test maxerr(libpath(MAEG(; etas = H, window = 3)),
                     libpath(ExponentiatedGradient(; eta = 0.05))) > 1e-6
        # The schedule is a rate, not a rule: at one rate it is the constant step.
        @test maxerr(libpath(MAEG(; etas = [0.05], window = 3)),
                     libpath(ExponentiatedGradient(; eta = 0.05))) < 1e-15

        # The carrier: the experts are on the simplex, the ring holds the last `window` log
        # returns of each expert, and the rate read after the fold is the one just used.
        sched = WindowedBestRate(; etas = H, window = 3)
        opt = OPS(; alg = ExponentiatedGradient(; eta = sched))
        o = po.partial_fit!(opt, rows(rd, 1:7))
        s = o.cache.st.s
        @test s.n == 7 && size(s.B) == (N, 3) && size(s.L) == (3, 3)
        @test all(isapprox.(vec(sum(s.B; dims = 1)), 1; atol = 1e-14))
        Bh = fill(1 / N, N, 3)
        Lh = zeros(7, 3)
        for t in 1:7, k in 1:3
            x = X[t, :]
            Lh[t, k] = log(dot(Bh[:, k], x))
            qk = Bh[:, k] .* exp.(H[k] .* x ./ dot(Bh[:, k], x))
            Bh[:, k] .= qk ./ sum(qk)
        end
        @test isapprox(vec(sum(s.L; dims = 1)), vec(sum(Lh[5:7, :]; dims = 1));
                       atol = 1e-14)
        @test isapprox(s.B, Bh; atol = 1e-14)
        @test po.learning_rate(sched, 8, o.cache.st) ==
              H[argmax(vec(sum(Lh[5:7, :]; dims = 1)))]
        # The whole history keeps one running sum per expert.
        oa = po.partial_fit!(OPS(; alg = AEG(; etas = H)), rows(rd, 1:7))
        @test size(oa.cache.st.s.L) == (1, 3)
        @test isapprox(vec(oa.cache.st.s.L), vec(sum(Lh; dims = 1)); atol = 1e-14)
        # Before the first row the seed is the start allocation at every rate, and the
        # rate of an unwritten ring is the first of the set.
        seed = po.rule_state_seed(ExponentiatedGradient(; eta = sched), fill(0.25, N))
        @test seed.s.n == 0 && seed.s.B == fill(0.25, N, 3) && seed.s.L == zeros(3, 3)
        @test po.learning_rate(sched, 1, seed) == H[1]
        # The copy is deep, and the view slices and renormalises every expert.
        c = copy(o.cache.st)
        @test c.s.B == s.B && c.s.B !== s.B && c.s.L == s.L && c.s.L !== s.L
        v = po.port_opt_view(o.cache.st, [1, 3])
        @test size(v.s.B) == (2, 3) && v.s.n == 7 && v.s.L == s.L
        @test isapprox(v.s.B, s.B[[1, 3], :] ./ sum(s.B[[1, 3], :]; dims = 1); atol = 1e-15)
        @test v.s.B == stack(po.renormalised_view(s.B[:, k], [1, 3]) for k in 1:3)
        # The fold is the batch, and the read-out under a view is the viewed recursion.
        o2 = po.partial_fit!(po.partial_fit!(opt, rows(rd, 1:10)), rows(rd, 11:18))
        @test optimise(o2).w == optimise(opt, rows(rd, 1:18)).w

        # Under a cap the rule's own step is projected while the experts stay the paper's
        # unconstrained runs; the chosen rate is theirs.
        cap = WeightBounds(; lb = 0, ub = 0.3)
        oc = po.partial_fit!(OPS(; alg = MAEG(; etas = H, window = 3),
                                 set = BoundedAllocationSet(; wb = cap)), rows(rd, 1:12))
        @test all(oc.cache.w .<= 0.3 + 1e-12) && isapprox(sum(oc.cache.w), 1; atol = 1e-12)
        @test isapprox(oc.cache.st.s.B,
                       po.partial_fit!(OPS(; alg = MAEG(; etas = H, window = 3)),
                                       rows(rd, 1:12)).cache.st.s.B; atol = 1e-14)
        # The optimistic step and a mixture's weighting carry the schedule too.
        oo = po.partial_fit!(OPS(;
                                 alg = ExponentiatedGradient(;
                                                             eta = WindowedBestRate(;
                                                                                    etas = H,
                                                                                    window = 3),
                                                             predictor = LastGradient())),
                             rows(rd, 1:6))
        @test oo.cache.st.s.n == 6 && isapprox(sum(oo.cache.w), 1; atol = 1e-12)
        vo = po.port_opt_view(oo.cache.st, [2, 4])
        @test size(vo.s.B) == (2, 3)
        mix = ExpertMixture(; experts = [BuyAndHold(), GradientProjection()],
                            alg = ExponentiatedGradient(; eta = sched))
        om = po.partial_fit!(OPS(; alg = mix), rows(rd, 1:5))
        @test om.cache.st.pst.n == 5 &&
              om.cache.st.pst.s.n == 5 &&
              size(om.cache.st.pst.s.B) == (2, 3)
        # The other schedules are untouched by the write point: the self-confident rate
        # still reads the excess through the previous row.
        sc = SelfConfidentRate(; eta_max = 0.8)
        osc = po.partial_fit!(OPS(; alg = ExponentiatedGradient(; eta = sc)), rows(rd, 1:9))
        Wsc = libpath(ExponentiatedGradient(; eta = sc))
        @test isapprox(osc.cache.st.s[1],
                       sum(maximum(X[t, :] ./ dot(Wsc[t, :], X[t, :])) - 1 for t in 1:9);
                       atol = 1e-13)
    end

    @testset "The momentum transforms of the EGM paper" begin
        # EGE: v <- g1 v + (1 - g1) g from zero, the step on v.
        v = zeros(N)
        H = handpath((w, x, t) -> begin
                         g = x ./ dot(w, x)
                         v .= 0.9 .* v .+ 0.1 .* g
                         q = w .* exp.(0.05 .* v)
                         q ./ sum(q)
                     end)
        @test maxerr(libpath(EGE(; gamma1 = 0.9)), H) < 1e-14
        # EGA: both averages, the step on v / (sqrt(m) + eps), no bias correction.
        v = zeros(N)
        m = zeros(N)
        H = handpath((w, x, t) -> begin
                         g = x ./ dot(w, x)
                         v .= 0.9 .* v .+ 0.1 .* g
                         m .= 0.5 .* m .+ 0.5 .* g .^ 2
                         q = w .* exp.(0.05 .* v ./ (sqrt.(m) .+ 1e-8))
                         q ./ sum(q)
                     end)
        @test maxerr(libpath(EGA(; gamma1 = 0.9, gamma2 = 0.5)), H) < 1e-14
        # EGR at the paper's gamma2 = 0: the transformed gradient is the sign up to eps,
        # the same at every asset, so the rule holds its start for the whole run.
        @test maxerr(libpath(EGR()), fill(1 / N, T, N)) < 1e-7
        # With a memory the rescaling moves.
        @test maxerr(libpath(EGR(; gamma2 = 0.9)), fill(1 / N, T, N)) > 1e-3
        # The carriers: a vector, a vector, a pair; sliced and copied by the state's view.
        for (alg, shape) in ((EGE(), Vector), (EGR(), Vector), (EGA(), Tuple))
            o = po.partial_fit!(OPS(; alg = alg), rows(rd, 1:3))
            @test isa(o.cache.st.gs, shape)
            v = po.port_opt_view(o.cache.st, [1, 3])
            @test length(v.u) == 2 && isapprox(sum(v.u), 1; atol = 1e-15)
            @test (shape === Tuple ? length(v.gs[1]) : length(v.gs)) == 2
            c = copy(o.cache.st)
            @test c.u == o.cache.st.u && c.u !== o.cache.st.u
            @test (shape === Tuple ? c.gs[1] !== o.cache.st.gs[1] : c.gs !== o.cache.st.gs)
        end
        @test_throws ArgumentError po.merge_states(po.rule_state_seed(EGE(), ones(2) ./ 2),
                                                   po.rule_state_seed(EGE(), ones(2) ./ 2))
    end

    @testset "The barriers on a programme set" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                     "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12),
                     check_sol = (; allow_local = true, allow_almost = true))
        q = [0.5, 0.3, 0.4]
        wh = [0.3, 0.3, 0.4]
        # Under a slack linear constraint the programme equals the closed form, in both
        # divergences, on the simplex and under a cap.
        pset = resolve(ProgrammeAllocationSet(; slv = slv,
                                              sets = UniverseSets(;
                                                                  dict = Dict("nx" =>
                                                                                  ["A", "B",
                                                                                   "C"])),
                                              lcse = LinearConstraintEstimator(;
                                                                               val = :(A +
                                                                                       B +
                                                                                       C <=
                                                                                       2))),
                       3)
        cset = resolve(ProgrammeAllocationSet(; slv = slv,
                                              wb = WeightBounds(; lb = 0.05, ub = 0.4)), 3)
        for proj in (LogBarrierProjection(), TsallisProjection(; alpha = 0.5),
                     TsallisProjection(; alpha = 0.8))
            @test isapprox(po.project(proj, pset, q, wh),
                           po.project(proj, resolve(BoundedAllocationSet(), 3), q, wh);
                           atol = 1e-6)
            @test isapprox(po.project(proj, cset, q, wh),
                           po.project(proj,
                                      resolve(BoundedAllocationSet(;
                                                                   wb = WeightBounds(;
                                                                                     lb = 0.05,
                                                                                     ub = 0.4)),
                                              3), q, wh); atol = 1e-6)
            # A zero raw entry is pinned at zero in the programme too.
            z = po.project(proj, pset, [0.5, 0.0, 0.5], wh)
            @test abs(z[2]) < 1e-9 && isapprox(sum(z), 1; atol = 1e-8)
            @test_throws DomainError po.project(proj, pset, [0.5, -0.1, 0.6], wh)
            @test_throws DomainError po.project(proj, pset, zeros(3), wh)
            @test_throws ArgumentError po.projection_solver(proj,
                                                            resolve(BoundedAllocationSet(),
                                                                    3))
            nset = resolve(ProgrammeAllocationSet(; slv = slv,
                                                  wb = WeightBounds(; lb = -0.1, ub = 1)),
                           3)
            @test_throws DomainError po.project(proj, nset, q, wh)
        end
        # A run of the barrier rule on a programme set with a turnover ceiling.
        tset = ProgrammeAllocationSet(; slv = slv,
                                      tn = Turnover(; w = zeros(4), val = 0.02))
        r = optimise(OPS(; alg = MirrorDescent(; proj = LogBarrierProjection(), eta = 2),
                         set = tset), rows(rd, 1:5))
        @test isapprox(sum(r.w), 1; atol = 1e-8) && all(r.w .> 0)
    end

    @testset "The Risk Loss: a risk measure over the head's rows as the objective" begin
        # The one deliberate widening of `obj`: every earlier fixture in this file runs on
        # `LogWealth()`, so their agreement above is the proof it left them unchanged.
        rl = RiskLoss()
        @test isa(rl.r, Variance) && rl.window == 20 && isa(rl.sca, SumScalariser)
        @test isa(rl.pe, EmpiricalPrior)
        @test isa(rl, po.AbstractOnlineObjective) &&
              isa(LogWealth(), po.AbstractOnlineObjective)
        @test_throws DomainError RiskLoss(; window = 1)
        # The loss reads rows that change at every step, so a fixed returns series is
        # refused by name, and a weights benchmark, which reads the rows, passes (#1656).
        err = try
            RiskLoss(;
                     r = [Variance(),
                          TrackingRiskMeasure(; tr = ReturnsTracking(; w = zeros(5)))])
        catch e
            e
        end
        @test isa(err, ArgumentError) && occursin("at `r[2].tr`", err.msg)
        @test isa(RiskLoss(;
                           r = TrackingRiskMeasure(;
                                                   tr = WeightsTracking(; w = [0.5, 0.5]))),
                  RiskLoss)
        @test_throws TypeError MirrorDescent(; obj = Variance())
        @test po.rows_needed(MirrorDescent()) == 0
        @test po.rows_needed(MirrorDescent(; obj = RiskLoss(; window = 7))) == 7
        @test po.rows_needed(OPS(;
                                 alg = GradientProjection(; obj = RiskLoss(; window = 7)))) ==
              7
        # A first-order rule on a Risk Loss reads rows, so it is refused as a mixture's
        # weighting, which is applied to the expert-return vector.
        @test_throws ArgumentError ExpertMixture(; experts = [BuyAndHold(), BuyAndHold()],
                                                 alg = GradientProjection(;
                                                                          obj = RiskLoss()))

        # One Euclidean step on `Variance()` over a 20-row window after 20 rows: the
        # buffer holds rows 2:21 at the update of row 21, so the step is
        # `Proj_Δ(w − η · 2 Σ w)` with `Σ` the sample covariance of those rows.
        alg = GradientProjection(; eta = 0.3, obj = RiskLoss(; r = Variance(), window = 20))
        o = po.partial_fit!(OPS(; alg = alg), rows(rd, 1:20))
        u20 = copy(o.cache.st.u)
        o = po.partial_fit!(o, rows(rd, 21:21))
        Σ = cov(R[2:21, :]; corrected = true)
        @test o.cache.st.u == po.project_simplex(u20 .- 0.3 .* (2 .* Σ * u20))
        @test o.cache.n == 21 && o.cache.X.max_history == 20 && o.cache.X.X.n == 20
        # One step of a small rate on a convex quadratic lowers it, unless at its minimum.
        @test dot(o.cache.st.u, Σ, o.cache.st.u) < dot(u20, Σ, u20)
        # The whole path, by hand: the window grows to 20 and then rolls; the first step
        # reads no covariance and is the identity on the iterate.
        function riskpath(proj, step)
            W = zeros(T, N)
            W[1, :] .= 1 / N
            for t in 1:(T - 1)
                if t < 2
                    W[t + 1, :] .= W[t, :]
                else
                    S = cov(R[max(1, t - 19):t, :]; corrected = true)
                    W[t + 1, :] .= step(W[t, :], 2 .* S * W[t, :])
                end
            end
            return W
        end
        a = libpath(alg)
        b = riskpath(EuclideanProjection(), (w, g) -> po.project_simplex(w .- 0.3 .* g))
        @test maxerr(a, b) < 1e-14
        @test a[2, :] == a[1, :]
        # The same loss under the entropic map: the multiplicative step on the gradient.
        c = libpath(ExponentiatedGradient(; eta = 5,
                                          obj = RiskLoss(; r = Variance(), window = 20)))
        d = riskpath(EntropicProjection(), (w, g) -> (q = w .* exp.(-5 .* g); q ./ sum(q)))
        @test maxerr(c, d) < 1e-14

        # A vector under a scale is one step on the mean–variance utility.
        mv = [MeanReturn(; settings = HierarchicalRiskMeasureSettings(; scale = -1.0)),
              Variance(; settings = RiskMeasureSettings(; scale = 0.5))]
        e = libpath(GradientProjection(; eta = 0.3, obj = RiskLoss(; r = mv, window = 20)))
        f = riskpath(EuclideanProjection(),
                     (w, g) -> po.project_simplex(w .- 0.3 .* (0.5 .* g)))
        # `riskpath` hands `2Σw`; the utility's gradient is `-μ + 0.5 · 2Σw`, so add the
        # mean by hand.
        W = zeros(T, N)
        W[1, :] .= 1 / N
        for t in 1:(T - 1)
            if t < 2
                W[t + 1, :] .= W[t, :]
            else
                S = cov(R[max(1, t - 19):t, :]; corrected = true)
                mu = vec(mean(R[max(1, t - 19):t, :]; dims = 1))
                W[t + 1, :] .= po.project_simplex(W[t, :] .-
                                                  0.3 .* (-mu .+ 0.5 .* (2 .* S * W[t, :])))
            end
        end
        @test maxerr(e, W) < 1e-14
        @test maxerr(e, f) > 1e-6

        # A measure with no closed form runs through the finite difference, and a stated
        # covariance on the measure is kept over the window's.
        g = libpath(GradientProjection(; eta = 0.3,
                                       obj = RiskLoss(; r = ConditionalValueatRisk(),
                                                      window = 10)))
        @test all(isapprox.(sum(g; dims = 2), 1; atol = 1e-12)) && all(g .>= 0)
        M = Matrix(1.0I, N, N)
        h = libpath(GradientProjection(; eta = 0.3,
                                       obj = RiskLoss(; r = Variance(; sigma = M),
                                                      window = 5)))
        # `2 M w = 2 w`, and the projection of `w − 0.6 w = 0.4 w` back onto the simplex
        # is `w` itself, so the rule never moves.
        @test maxerr(h, fill(1 / N, T, N)) < 1e-14

        # The batch-online identity holds with a rows buffer: the Causal Pass and the
        # block steps agree.
        full = optimise(OPS(; alg = alg), rd).w
        o = po.partial_fit!(OPS(; alg = alg), rows(rd, 1:13))
        o = po.partial_fit!(o, rows(rd, 14:40))
        @test maxerr(full, o.cache.w) < 1e-14
        @test maxerr(full, optimise(o).w) < 1e-14

        # Issue #1661: the loss reads the last `window` rows of what the head holds, and
        # not every row. The gradient over 20 rows is the one over the last 5, and not
        # the one of a prior fitted on all 20.
        u = fill(1 / N, N)
        rl5 = RiskLoss(; window = 5)
        g20 = po.loss_gradient(rl5, u, X[21, :], rows(rd, 1:20))
        @test g20 == po.loss_gradient(rl5, u, X[21, :], rows(rd, 16:20))
        @test isapprox(g20, 2 .* cov(R[16:20, :]) * u; atol = 1e-15)
        @test maxerr(g20, 2 .* cov(R[1:20, :]) * u) > 1e-5
        # A tracking error on the set needs every row, so the head holds them all. The
        # set does not bind, and the path is still the hand recursion on 5 rows. The set
        # fits no covariance on one row, so the first step is a Held Step at the drifted
        # allocation, and the recursion starts there.
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                     "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12),
                     check_sol = (; allow_local = true, allow_almost = true))
        tw = WeightsTracking(; w = fill(1 / N, N))
        trset = ProgrammeAllocationSet(; slv = slv,
                                       tr = TrackingError(; tr = tw, err = 1.0))
        alg5 = GradientProjection(; eta = 50, obj = rl5)
        @test isnothing(po.rows_needed(OPS(; alg = alg5, set = trset)))
        @test po.rows_needed(alg5) == 5
        function heldpath(win)
            W = zeros(T, N)
            W[1, :] .= 1 / N
            W[2, :] .= W[1, :] .* X[1, :] ./ dot(W[1, :], X[1, :])
            for t in 2:(T - 1)
                S = cov(R[max(1, t - win + 1):t, :]; corrected = true)
                W[t + 1, :] .= po.project_simplex(W[t, :] .- 50 .* (2 .* S * W[t, :]))
            end
            return W
        end
        @test maxerr(libpath(alg5; set = trset), heldpath(5)) < 1e-6
        @test maxerr(heldpath(5), heldpath(T)) > 1e-2

        # Issue #1660: a benchmark or a turnover reference of the loss that is not fixed
        # takes the Price-Adjusted Allocation of the step, and a fixed one stays.
        wb = [0.4, 0.3, 0.2, 0.1]
        trl(w; fixed) = RiskLoss(;
                                 r = TrackingRiskMeasure(;
                                                         tr = WeightsTracking(; w = w,
                                                                              fixed = fixed)),
                                 window = 10)
        wq = fill(1 / N, N)
        @test po.factory(trl(wq; fixed = false), wb).r.tr.w == wb
        @test po.factory(trl(wq; fixed = true), wb).r.tr.w == wq
        @test po.factory(RiskLoss(; r = TurnoverRiskMeasure(; w = wq)), wb).r.w == wb
        @test po.factory(LogWealth(), wb) === LogWealth()
        # On two rows the first step reads one row, so its gradient is zero and every rule
        # holds the uniform start. The second step trades from `wh`, the uniform start after
        # row 2, so a benchmark that is not fixed equals a fixed benchmark at `wh`. Each
        # rule reads the resolved loss, and the optimistic step's predictor reads it too.
        wh2 = X[2, :] ./ sum(X[2, :])
        for mk in (o -> GradientProjection(; eta = 50, obj = o),
                   o -> AdaptiveSubgradient(; obj = o),
                   o -> GradientProjection(; eta = 50, obj = o,
                                           predictor = LastGradient(; at_played = false)))
            walk(o) = optimise(OPS(; alg = mk(o)), rows(rd, 1:2)).w
            w_book = walk(trl(wq; fixed = false))
            @test maxerr(w_book, walk(trl(wh2; fixed = true))) < 1e-14
            @test maxerr(w_book, walk(trl(wq; fixed = true))) > 1e-3
        end
        # The same two rows on a panel whose fourth asset is inactive (#1772). The book keeps
        # the asset at its weight of the uniform start and reads its gap as a relative of
        # one, so `wh` has four legs, and the loss views the benchmark on the three priced
        # ones. Measured: 0.0 against the benchmark at `wh`, 0.33, 0.52 and 0.67 against the
        # caller's. The adaptive subgradient steps there since #1777, because a zero entry of
        # its gradient mass is the limit of a positive one.
        Rn2 = copy(R[1:2, :])
        am2 = trues(2, N)
        am2[:, 4] .= false
        Rn2[.!am2] .= NaN
        rdn2 = ReturnsResult(; nx = nx, X = Rn2, ts = ts[1:2],
                             pnl = AssetPanel(; amsk = am2, emsk = copy(am2)))
        xg = [X[2, 1:3]; 1]
        whm = xg ./ sum(xg)
        for mk in (o -> GradientProjection(; eta = 50, obj = o),
                   o -> AdaptiveSubgradient(; obj = o),
                   o -> GradientProjection(; eta = 50, obj = o,
                                           predictor = LastGradient(; at_played = false)))
            walk(o) = optimise(OPS(; alg = mk(o)), rdn2).w
            w_book = walk(trl(wq; fixed = false))
            @test iszero(w_book[4])
            @test maxerr(w_book, walk(trl(whm; fixed = true))) < 1e-14
            @test maxerr(w_book, walk(trl(wq; fixed = true))) > 0.1
        end
    end

    @testset "A Risk Loss tracks the benchmark of its window (#1657)" begin
        # A vector benchmark that is the series of a fixed portfolio equals that portfolio
        # as a fixed `WeightsTracking`. A matrix benchmark binds after the view at the
        # Investable Mask. The online fold equals the batch fit for both.
        wb = [0.1, 0.2, 0.3, 0.4]
        rdv = ReturnsResult(; nx = nx, X = R, nb = ["BM"], B = R * wb, ts = ts)
        rdm = ReturnsResult(; nx = nx, X = R, nb = nx, B = 0.5 .* R, ts = ts)
        gp(tr) = OPS(;
                     alg = GradientProjection(; eta = 50,
                                              obj = RiskLoss(;
                                                             r = TrackingRiskMeasure(;
                                                                                     tr = tr),
                                                             window = 5)))
        wd = optimise(gp(DataBenchmarkTracking()), rdv).w
        @test isapprox(wd, optimise(gp(WeightsTracking(; w = wb, fixed = true)), rdv).w;
                       atol = 1e-8)
        @test !isapprox(wd, fill(1 / N, N); atol = 1e-2)
        for r in (rdv, rdm)
            o = foldl((o, t) -> po.partial_fit!(o, rows(r, t:t)), 1:T;
                      init = gp(DataBenchmarkTracking()))
            @test optimise(o).w == optimise(gp(DataBenchmarkTracking()), r).w
        end
    end

    @testset "The numbers the docstrings state, on the fixtures they name" begin
        # `mixed_relatives`: the scaled floor against the raw mix, at the default rate and
        # alpha = 0.2, over 60 periods of `1 .+ 0.02 .* randn(StableRNG(7), 60, 4)`.
        X60 = 1 .+ 0.02 .* randn(StableRNG(7), 60, 4)
        function mixpath(Xm, mixf)
            n = size(Xm, 2)
            u = fill(1 / n, n)
            P = zeros(size(Xm, 1) + 1, n)
            P[1, :] .= u
            for t in axes(Xm, 1)
                xm = mixf(Xm[t, :])
                q = u .* exp.(0.05 .* xm ./ dot(u, xm))
                u = q ./ sum(q)
                P[t + 1, :] .= 0.8 .* u .+ 0.2 / n
            end
            return P
        end
        scaled(x) = po.mixed_relatives(x, 0.2)
        raw(x) = (1 - 0.2 / length(x)) .* x .+ 0.2 / length(x)
        @test round(maxerr(mixpath(X60, scaled), mixpath(X60, raw)); sigdigits = 2) ==
              2.6e-6
        @test round(maxerr(mixpath([1.5 1.0 1.0 1.0], scaled),
                           mixpath([1.5 1.0 1.0 1.0], raw)); sigdigits = 2) == 7.0e-5
        # The library plays the scaled mix.
        rd60 = ReturnsResult(; nx = nx, X = X60 .- 1, ts = Date(2020, 1, 1) .+ Day.(0:59))
        o60 = po.partial_fit!(OPS(; alg = ExponentiatedGradient(; alpha = 0.2)),
                              rows(rd60, 1:59))
        @test maxerr(o60.cache.w, mixpath(X60, scaled)[60, :]) < 1e-14

        # `RiskLoss`: 1000 periods of `randn(StableRNG(7), 1000, 4) .* [0.01 0.02 0.03
        # 0.04]`, a window of 20, the uniform allocation.
        RV = randn(StableRNG(7), 1000, 4) .* [0.01 0.02 0.03 0.04]
        u4 = fill(0.25, 4)
        ratio = [norm((1 .+ RV[t + 1, :]) ./ dot(u4, 1 .+ RV[t + 1, :])) /
                 norm(2 .* cov(RV[(t - 19):t, :]) * u4) for t in 20:999]
        @test round.(extrema(ratio); sigdigits = 2) == (1100.0, 6400.0)
        rdV = ReturnsResult(; nx = nx, X = RV, ts = Date(2020, 1, 1) .+ Day.(0:999))
        iv = 1 ./ [0.01, 0.02, 0.03, 0.04] .^ 2
        wmv = iv ./ sum(iv)
        @test round(maxerr(wmv, u4); digits = 2) == 0.45
        w05 = optimise(OPS(; alg = ExponentiatedGradient(; obj = RiskLoss())), rdV).w
        @test round(maxerr(w05, u4); digits = 3) == 0.005
        w100 = optimise(OPS(; alg = ExponentiatedGradient(; eta = 100, obj = RiskLoss())),
                        rdV).w
        @test maxerr(w100, wmv) < 0.06

        # `loss_gradient` on a panel whose window does not cover D: the gradient is
        # 2 Σ u on the three priced assets, u sliced and not renormalised, and zero at D.
        Rn = copy(R[1:10, :])
        am = trues(10, N)
        am[:, 4] .= false
        Rn[.!am] .= NaN
        rdn = ReturnsResult(; nx = nx, X = Rn, ts = ts[1:10],
                            pnl = AssetPanel(; amsk = am, emsk = copy(am)))
        u = [0.3, 0.3, 0.2, 0.2]
        g = po.loss_gradient(RiskLoss(; window = 10), u, X[11, :], rdn)
        @test g[4] == 0
        @test isapprox(g[1:3], 2 .* cov(R[1:10, 1:3]) * u[1:3]; atol = 1e-15)
        # Issue #1660: the measure is viewed on the mask with the prior, so a benchmark of
        # four legs is cut to the three priced ones and is not renormalised.
        tr4 = TrackingRiskMeasure(; tr = WeightsTracking(; w = [0.4, 0.3, 0.2, 0.1]))
        tr3 = TrackingRiskMeasure(; tr = WeightsTracking(; w = [0.4, 0.3, 0.2]))
        gt = po.loss_gradient(RiskLoss(; r = tr4, window = 10), u, X[11, :], rdn)
        @test gt[4] == 0
        @test isapprox(gt[1:3],
                       po.loss_gradient(RiskLoss(; r = tr3, window = 10), u[1:3],
                                        X[11, 1:3],
                                        ReturnsResult(; nx = nx[1:3], X = R[1:10, 1:3]));
                       atol = 1e-12)

        # `reprojection`: a bounded set returns an allocation inside its bounds as the
        # same object, and every other set projects it.
        wh = fill(0.25, 4)
        qin = [0.1, 0.4, 0.3, 0.2]
        @test po.reprojection(EntropicProjection(), simplex, qin, wh) === qin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                     "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12),
                     check_sol = (; allow_local = true, allow_almost = true))
        cap = WeightBounds(; lb = 0, ub = [0.1, 1, 1, 1])
        pset = resolve(ProgrammeAllocationSet(; slv = slv, wb = cap), 4)
        q = [0.2, 0.3, 0.3, 0.2]
        rp = po.reprojection(EuclideanProjection(), pset, q, wh)
        @test rp == po.project(EuclideanProjection(), pset, q, wh)
        @test isapprox(rp,
                       po.project(EuclideanProjection(),
                                  resolve(BoundedAllocationSet(; wb = cap), 4), q, wh);
                       atol = 1e-8)
        # A run with a positive alpha on that set plays inside the cap.
        wp = optimise(OPS(; alg = GradientProjection(; alpha = 0.2),
                          set = ProgrammeAllocationSet(; slv = slv, wb = cap)),
                      rows(rd, 1:6)).w
        @test wp[1] <= 0.1 + 1e-8 && isapprox(sum(wp), 1; atol = 1e-8)
    end
end

#=
The meta-learning rows of the fourth set, as ADR 0165 rules them (issue #1183 on map #1148):
the mixture's Gradient Point and Start Allocation over the experts, the `Ader` and `Sword`
constructors, and the Switching Weighting with `SwitchingPortfolio`; and the start of an expert,
projected onto the Allocation Set as the head's `w0` is (ADR 0162, issue #1268).

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
        # The uniform start meets a bound it violates at the seed too, so the first
        # period's mix honours the Expert Set: a floor of 0.6 on the first expert.
        mixu = ExpertMixture(; experts = [GradientProjection(), BuyAndHold()],
                             eset = BoundedAllocationSet(;
                                                         wb = WeightBounds([0.6, 0.0], 1)))
        @test po.rule_state_seed(mixu, fill(1 / N, N)).p ≈ [0.6, 0.4]
        ou = po.partial_fit!(OPS(; alg = mixu), rows(rd, 1:1))
        @test ou.cache.st.p[1] >= 0.6 - 1e-12
        # The bare default is untouched: exactly uniform.
        @test po.rule_state_seed(ExpertMixture(; experts = mixu.experts), fill(1 / N, N)).p ==
              [0.5, 0.5]
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
        for e in (GradientProjection(; eta = 0.2, predictor = LastGradient()),
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
        # excluded asset has nothing to hold on the view: its constant rebalanced portfolio
        # is the uniform allocation over the kept assets, and the view runs.
        v = po.port_opt_view(OPS(; alg = sp), [1, 3])
        @test v.alg.experts[2].w == [0.5, 0.5] && v.alg.experts[1].w == [1.0, 0.0]
        @test length(optimise(v, po.port_opt_view(rd, 1:5, [1, 3])).w) == 2
    end

    @testset "An expert's start meets the Allocation Set (#1268)" begin
        # The one-hot experts of a switching portfolio under a cap of 0.5. The Euclidean
        # projection of e_1 onto the capped simplex is [0.5, 0.25, 0.25], by hand.
        onehot = [ConstantRebalancedPortfolio(; w = [1, 0, 0]),
                  ConstantRebalancedPortfolio(; w = [0, 1, 0]),
                  ConstantRebalancedPortfolio(; w = [0, 0, 1])]
        cap = BoundedAllocationSet(; wb = WeightBounds(0, 0.5))
        capr = resolve(cap, N)
        u = fill(1 / N, N)
        mix = ExpertMixture(; experts = onehot, alg = ExponentiatedGradient())
        st = po.rule_state_seed(mix, u, capr)
        @test st.h[1] ≈ [0.5, 0.25, 0.25] && st.h[3] ≈ [0.25, 0.25, 0.5]
        @test all(h -> all(<=(0.5 + 1e-12), h) && isapprox(sum(h), 1; atol = 1e-12), st.h)
        # Without a set nothing is projected, and the default method drops the set.
        @test po.rule_state_seed(mix, u).h[1] == [1, 0, 0]
        @test po.rule_state_seed(mix, u, nothing).h[1] == [1, 0, 0]
        @test isnothing(po.rule_state_seed(BuyAndHold(), u, capr))
        # The oracle: a mixture whose experts already hold the projected allocations. The
        # head seeds on its set, so the expert weighting reads the same returns from the
        # first row, and the two mixtures agree row for row.
        feasible = [ConstantRebalancedPortfolio(; w = h) for h in st.h]
        for alg in (ExponentiatedGradient(), BuyAndHold())
            a = po.partial_fit!(OPS(; alg = ExpertMixture(; experts = onehot, alg = alg),
                                    set = cap), rows(rd, 1:1))
            b = po.partial_fit!(OPS(; alg = ExpertMixture(; experts = feasible, alg = alg),
                                    set = cap), rows(rd, 1:1))
            @test a.cache.st.p ≈ b.cache.st.p
            @test a.cache.st.p ≉
                  po.partial_fit!(OPS(; alg = ExpertMixture(; experts = onehot, alg = alg)),
                                  rows(rd, 1:1)).cache.st.p
            wa = optimise(OPS(; alg = ExpertMixture(; experts = onehot, alg = alg),
                              set = cap), rows(rd, 1:8)).w
            wb = optimise(OPS(; alg = ExpertMixture(; experts = feasible, alg = alg),
                              set = cap), rows(rd, 1:8)).w
            @test wa ≈ wb
        end
        # FollowTheLeadingHistory seeds its first expert on the set, and a newcomer of a
        # constant rebalanced base starts at the projection of its own allocation.
        ftl = FollowTheLeadingHistory(; alg = ConstantRebalancedPortfolio(; w = [1, 0, 0]))
        @test po.rule_state_seed(ftl, u, capr).h[1] ≈ [0.5, 0.25, 0.25]
        @test po.rule_state_seed(ftl, u).h[1] == [1, 0, 0]
        of = po.partial_fit!(OPS(; alg = ftl, set = cap), rows(rd, 1:3))
        @test all(h -> h ≈ [0.5, 0.25, 0.25], of.cache.st.h)
    end

    @testset "The mathematics of the meta-learning docstrings (#1200)" begin
        # Singer's equation 6, (1 - γK/(K - 1)) q + γ/(K - 1), is the step on the simplex.
        for (g, K) in ((0.25, 3), (1 / 3, 5), (0.9, 4))
            wk = rand(StableRNG(K), K)
            wk ./= sum(wk)
            xk = 1 .+ 0.05 .* randn(StableRNG(K + 1), K)
            q = wk .* xk ./ dot(wk, xk)
            eq6 = (1 - g * K / (K - 1)) .* q .+ g / (K - 1)
            wn = po.online_update!(SwitchingWeighting(; gamma = g), nothing, wk, xk,
                                   nothing, resolve(BoundedAllocationSet(), K))[2]
            @test maxerr(wn, eq6) < 1e-15
        end
        # At γ = (K - 1) / K every entry is 1 / K.
        @test po.online_update!(SwitchingWeighting(; gamma = 2 / 3), nothing,
                                [0.7, 0.2, 0.1], X[1, :], nothing, simplex)[2] ≈
              fill(1 / 3, 3)

        # The switching portfolio is the Bayesian mixture over the paths of the hidden
        # process: the prior of a path is (1/N) ∏ (1 - γ or γ/(N - 1)), and the wealth of
        # the mixture is the prior-weighted sum of the wealths of all N^Tp paths.
        g, Tp = 0.25, 6
        Wsp = libpath(SwitchingPortfolio(; N = N, gamma = g))
        wealth = prod(dot(Wsp[t, :], X[t, :]) for t in 1:Tp)
        function path_sum(len)
            mass = zeros(N)
            total = 0.0
            for path in Iterators.product(ntuple(_ -> 1:N, len)...)
                prior = 1 / N
                for t in 2:len
                    prior *= path[t] == path[t - 1] ? 1 - g : g / (N - 1)
                end
                v = prior * prod(X[t, path[t]] for t in 1:min(len, Tp))
                total += v
                mass[path[end]] += v
            end
            return total, mass
        end
        total, _ = path_sum(Tp)
        @test isapprox(total, wealth; rtol = 1e-13)
        # The weight on a unit expert is the posterior probability that the process holds
        # that asset in the next period.
        total1, mass = path_sum(Tp + 1)
        @test maxerr(mass ./ total1, Wsp[Tp + 1, :]) < 1e-13

        # The smallest bound on the Euclidean norm of the log-wealth gradient over the
        # simplex is ‖x‖₂ / min x, at the vertex of the smallest price relative. The bound
        # 1 / min x does not hold: at the uniform allocation over [1, 1, 1, 0.6] the norm
        # is 2.04, above 1 / 0.6.
        grad_norm(w, x) = norm(x) / dot(w, x)
        for s in 1:200
            x = 1 .+ 0.1 .* randn(StableRNG(s), 4)
            w = rand(StableRNG(s + 1000), 4)
            w ./= sum(w)
            @test grad_norm(w, x) <= norm(x) / minimum(x) * (1 + 1e-15)
            vertex = [i == argmin(x) ? 1.0 : 0.0 for i in 1:4]
            @test grad_norm(vertex, x) ≈ norm(x) / minimum(x)
        end
        xg = [1, 1, 1, 0.6]
        @test round(grad_norm(fill(0.25, 4), xg); digits = 2) == 2.04 &&
              grad_norm(fill(0.25, 4), xg) > 1 / 0.6

        # Under any objective the weighting weighs the experts by their log wealth: on a
        # Risk Loss the next weight is the exponentiated-gradient step over the experts'
        # returns, p ∝ p ⊙ exp(ε r / ⟨p, r⟩), and not a step on the risk loss.
        for obj in (LogWealth(), RiskLoss(; window = 5))
            eps = 0.3
            o = po.partial_fit!(OPS(;
                                    alg = Ader(; eta_min = 0.05, K = 3, eps = eps,
                                               obj = obj)), rows(rd, 1:12))
            h = deepcopy(o.cache.st.h)
            p = copy(o.cache.st.p)
            r = [dot(hk, X[13, :]) for hk in h]
            pn = p .* exp.(eps .* r ./ dot(p, r))
            pn ./= sum(pn)
            o = po.partial_fit!(o, rows(rd, 13:13))
            @test maxerr(o.cache.st.p, pn) < 1e-14
        end

        # Every rate of the grid takes the numeric type of eta_min.
        @test [e.eta for e in po.rate_grid_experts(1 // 20, 3, LogWealth())] ==
              [1 // 20, 1 // 10, 1 // 5]
        @test [e.eta for e in po.rate_grid_experts(1, 3, LogWealth())] == [1, 2, 4]
    end
end
