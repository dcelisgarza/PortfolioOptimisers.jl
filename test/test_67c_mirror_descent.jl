#=
The first-order rule of the online portfolio selection family, as ADR 0165 rules it (issue
#1180 on map #1148): `MirrorDescent` over the four scalar-root geometries, the uniform mix,
the Tsallis and log-barrier projections, the Learning-Rate Schedules and the momentum
Gradient Transforms.

Every literal below states its provenance: the exponentiated-gradient identity is with the
`#1161` fixture through the constructor; the `alpha = 0.2` step is ADR 0165's worked
example; the Euclidean, momentum and scheduled paths are hand recursions of the stated
formulas; the barrier roots are checked against an independent bisection at `1e-12`; the
doubling trick's first stage length is Corollary 4.3's `ceil(2 N^2 log N)`. The Risk Loss
(issue #1182) is checked against a hand recursion on the window's sample covariance.
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
        @test isa(md.obj, LogWealth) && isa(md.grad, PlainGradient)
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

    @testset "The uniform mix: ADR 0165's worked example" begin
        # x = [1.2, 0.8], w = [0.5, 0.5], eta = 0.05, alpha = 0.2: the update reads the
        # mixed relatives [1.18, 0.82] at the unmixed iterate, stores [0.5045, 0.4955] and
        # plays [0.5036, 0.4964].
        alg = ExponentiatedGradient(; alpha = 0.2)
        w = [0.5, 0.5]
        x = [1.2, 0.8]
        st = po.rule_state_seed(alg, w)
        st, wn = po.online_update!(alg, st, w, x, nothing,
                                   resolve(BoundedAllocationSet(), 2))
        xm = 0.9 .* x .+ 0.1
        q = w .* exp.(0.05 .* xm ./ dot(w, xm))
        u = q ./ sum(q)
        @test isapprox(st.u, u; atol = 1e-15)
        @test round.(st.u; digits = 4) == [0.5045, 0.4955]
        @test isapprox(wn, 0.8 .* u .+ 0.1; atol = 1e-15)
        @test round.(wn; digits = 4) == [0.5036, 0.4964]
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
        # self-confident rate's fall; at a constant rate it is the plain step.
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
            st, wn = po.online_update!(em, st, w, x, nothing, simplex)
            @test isapprox(wn,
                           plain .* (eta_n / eta_t) .+ (1 - eta_n / eta_t) .* fill(0.25, N);
                           atol = 1e-14)
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
                                              lcs = LinearConstraintEstimator(;
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
        tset = ProgrammeAllocationSet(; slv = slv, tn = 0.02)
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
        @test o.cache.n == 21 && o.cache.X.max_history == 20 && o.cache.X.n == 20
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
    end
end
