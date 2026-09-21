#=
The adaptive and optimistic rows of the online portfolio selection family's fourth set, as
ADR 0165 rules them (issue #1181 on map #1148): `AdaptiveSubgradient` under
`DiagonalProjection`, and `OptimisticStep` over `MirrorDescent` with its three Gradient
Predictors and the hint-residual rate.

Every literal below states its provenance: the AdaGrad steps are the composite mirror descent
of Duchi, Hazan and Singer (2011, Algorithm 1) written out by hand on a two-asset row — the
raw step `w + η x / (⟨w, x⟩ s)` and the projection in the norm `diag(s)`, whose budget root
is `θ = (Σq − 1) / Σ(1 / s_i)` where no bound binds; the optimistic steps are the two
half-steps of Rakhlin and Sridharan (2013, §2.2) on the Euclidean and entropic maps; the
weighted roots are checked against an independent quadratic programme at `1e-8`. Under a
schedule the second half-step is taken at the next period's rate, as the paper's Corollary 2
forms it.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Dates, Clarabel, JuMP
@testset "Online portfolio selection: the adaptive subgradient and the optimistic step" begin
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
    resolve(set, n) = po.resolve_allocation_set(set, n, false, Float64)
    simplex2 = resolve(BoundedAllocationSet(), 2)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                 "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12),
                 check_sol = (; allow_local = true, allow_almost = true))

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
    # A Gradient Predictor written to the interface for the identity test: a zero hint.
    struct ZeroHint <: po.AbstractGradientPredictor end
    function po.predict_gradient!(::ZeroHint, ::Nothing, ::po.AbstractOnlineObjective,
                                  g::AbstractVector, ::AbstractVector, ::AbstractVector,
                                  ::Any, ::Integer)
        return nothing, zero(g)
    end
    # The weighted projection onto a bounded set as a quadratic programme, independent of
    # the scalar root.
    function qp_projection(q, h, lb, ub)
        model = JuMP.Model(Clarabel.Optimizer)
        JuMP.set_silent(model)
        JuMP.set_attribute(model, "tol_gap_abs", 1e-12)
        JuMP.set_attribute(model, "tol_gap_rel", 1e-12)
        JuMP.set_attribute(model, "tol_feas", 1e-12)
        JuMP.@variable(model, lb[i] <= w[i = 1:length(q)] <= ub[i])
        JuMP.@constraint(model, sum(w) == 1)
        JuMP.@objective(model, Min, sum(h[i] * (w[i] - q[i])^2 for i in 1:length(q)))
        JuMP.optimize!(model)
        return JuMP.value.(w)
    end

    @testset "Construction and the refusals" begin
        ag = AdaptiveSubgradient()
        @test ag.eta == 1 / sqrt(2) && ag.delta == 0 && isa(ag.proj, DiagonalProjection)
        @test isnothing(ag.proj.h) && isa(ag.obj, LogWealth)
        @test_throws DomainError AdaptiveSubgradient(; eta = 0)
        @test_throws DomainError AdaptiveSubgradient(; delta = -0.1)
        @test_throws TypeError AdaptiveSubgradient(; proj = EuclideanProjection())
        # The objective slot is the first-order rule's: a Risk Loss reads the head's rows.
        @test po.rows_needed(AdaptiveSubgradient(; obj = RiskLoss(; window = 5))) == 5
        @test po.rows_needed(OptimisticStep(;
                                            alg = GradientProjection(;
                                                                     obj = RiskLoss(;
                                                                                    window = 6)))) ==
              6
        @test_throws DomainError DiagonalProjection(; h = [1.0, 0.0])
        @test_throws ArgumentError po.project(DiagonalProjection(), simplex2, [0.6, 0.4],
                                              [0.5, 0.5])
        @test po.projection_geometry(ag) == EuclideanProjection()
        os = OptimisticStep()
        @test isa(os.alg, MirrorDescent) && isa(os.predictor, LastGradient)
        @test os.predictor.at_played
        @test !LastGradient(; at_played = false).at_played
        @test_throws ArgumentError OptimisticStep(; alg = EGE())
        @test_throws ArgumentError OptimisticStep(;
                                                  alg = MirrorDescent(;
                                                                      grad = RootMeanSquareGradient()))
        # The slot's bound refuses an `Online` wrapper before the name check does.
        @test_throws TypeError ForecastGradient(;
                                                me = Online(;
                                                            est = SimpleExpectedReturns()))
        @test_throws DomainError HintResidualRate(; rmax = 0)
        @test po.projection_geometry(os) == EntropicProjection()
        @test po.rows_needed(os) == 0
        @test po.rows_needed(OptimisticStep(; predictor = MeanGradient())) == 0
        fg = OptimisticStep(;
                            predictor = ForecastGradient(;
                                                         me = PriceLevelExpectedReturns(;
                                                                                        alg = MovingAverage(;
                                                                                                            window = 3))))
        @test po.rows_needed(fg) == 2
        # The hint-residual rate on a bare rule is refused at its first read, by name.
        @test_throws ArgumentError optimise(OPS(;
                                                alg = MirrorDescent(;
                                                                    eta = HintResidualRate())),
                                            rows(rd, 1:2))
    end

    @testset "The weighted root against a quadratic programme" begin
        rng2 = StableRNG(3)
        for _ in 1:5
            q = randn(rng2, 5)
            h = 0.2 .+ rand(rng2, 5)
            for (lb, ub) in ((zeros(5), ones(5)), (fill(0.05, 5), fill(0.5, 5)),
                             (fill(-0.2, 5), fill(0.6, 5)))
                wb = WeightBounds(; lb = lb, ub = ub)
                w = po.bounded_quadratic_projection(q, wb, h)
                @test isapprox(w, qp_projection(q, h, lb, ub); atol = 1e-7)
                @test isapprox(sum(w), 1; atol = 1e-12)
                # The unit-weight root is the Euclidean arm, and on the simplex the sort.
                e = po.bounded_quadratic_projection(q, wb)
                @test isapprox(e, qp_projection(q, ones(5), lb, ub); atol = 1e-7)
                set = resolve(BoundedAllocationSet(; wb = wb), 5)
                @test po.project(EuclideanProjection(), set, q, q) ≈ e
                @test po.project(DiagonalProjection(; h = h), set, q, q) == w
            end
            @test isapprox(po.bounded_quadratic_projection(q,
                                                           WeightBounds(; lb = zeros(5),
                                                                        ub = ones(5))),
                           po.project_simplex(q); atol = 1e-12)
        end
    end

    @testset "AdaGrad by hand: the first step is not a no-op" begin
        # Duchi, Hazan and Singer (2011, Algorithm 1): on the uniform start with
        # x₁ = [1.2, 0.8], g₁ = −[1.2, 0.8], s₁ = [1.2, 0.8] and the raw step is the uniform
        # shift [0.5 + η, 0.5 + η]; the projection in the norm diag(s₁) has the budget root
        # θ = 2η / (1/1.2 + 1/0.8) = 0.96η, so w₂ = [0.5 + 0.2η, 0.5 − 0.2η]: the assets with
        # the smaller gradient absorb more of the shift back, and the step moves weight to
        # the asset that rose. At η = 1/√2 that is [0.641421, 0.358579].
        eta = 1 / sqrt(2)
        alg = AdaptiveSubgradient()
        w = [0.5, 0.5]
        x1 = [1.2, 0.8]
        st = po.rule_state_seed(alg, w)
        @test st.s == [0.0, 0.0] && st.n == 0
        st, w2 = po.online_update!(alg, st, w, x1, nothing, simplex2)
        @test isapprox(w2, [0.5 + 0.2 * eta, 0.5 - 0.2 * eta]; atol = 1e-12)
        @test round.(w2; digits = 6) == [0.641421, 0.358579]
        @test st.s == [1.2, 0.8] && st.n == 1
        # The second step at x₂ = [1.1, 0.9] from w₂, by the same recursion.
        x2 = [1.1, 0.9]
        g2 = -x2 ./ dot(w2, x2)
        s2 = sqrt.([1.2, 0.8] .^ 2 .+ g2 .^ 2)
        q = w2 .- eta .* g2 ./ s2
        theta = (sum(q) - 1) / sum(inv, s2)
        st, w3 = po.online_update!(alg, st, w2, x2, nothing, simplex2)
        @test isapprox(w3, q .- theta ./ s2; atol = 1e-12)
        @test round.(w3; digits = 4) == [0.6907, 0.3093]
        @test st.s ≈ s2
        # A zero gradient is a zero step, no mass accrues, and the projection is Euclidean.
        stz = po.rule_state_seed(alg, w)
        stz, wz = po.online_update!(AdaptiveSubgradient(; obj = RiskLoss(; window = 3)),
                                    stz, w, x1, zeros(1, 2), simplex2)
        @test wz == w && stz.s == [0.0, 0.0] && stz.n == 1
        # Equal price relatives are the one case the first step holds.
        st0 = po.rule_state_seed(alg, w)
        _, wh = po.online_update!(alg, st0, w, [1.05, 1.05], nothing, simplex2)
        @test isapprox(wh, w; atol = 1e-15)
        # `delta` enters the diagonal, and a large one makes the step Euclidean up to scale.
        stD = po.rule_state_seed(AdaptiveSubgradient(; delta = 1e6), w)
        _, wD = po.online_update!(AdaptiveSubgradient(; delta = 1e6), stD, w, x1, nothing,
                                  simplex2)
        @test isapprox(wD, w; atol = 1e-6)
    end

    @testset "AdaGrad over the fixture, its carrier and its views" begin
        W = libpath(AdaptiveSubgradient())
        @test all(isapprox.(sum(W; dims = 2), 1; atol = 1e-12))
        @test all(W .>= -1e-15)
        # A hand recursion over the rows agrees with the head's path.
        Wh = zeros(T, N)
        Wh[1, :] .= 1 / N
        s = zeros(N)
        eta = 1 / sqrt(2)
        for t in 1:(T - 1)
            g = -X[t, :] ./ dot(Wh[t, :], X[t, :])
            s = sqrt.(s .^ 2 .+ g .^ 2)
            Wh[t + 1, :] .= po.bounded_quadratic_projection(Wh[t, :] .- eta .* g ./ s,
                                                            WeightBounds(; lb = zeros(N),
                                                                         ub = ones(N)), s)
        end
        @test maxerr(W, Wh) < 1e-12
        o = po.partial_fit!(OPS(; alg = AdaptiveSubgradient()), rows(rd, 1:7))
        @test isa(o.cache.st, po.AdaptiveSubgradientState)
        @test o.cache.st.n == 7 && length(o.cache.st.s) == N
        @test all(o.cache.st.s .> 0)
        v = po.port_opt_view(o, [1, 3])
        @test v.cache.st.s == o.cache.st.s[[1, 3]]
        @test isapprox(sum(v.cache.w), 1; atol = 1e-12)
        c = copy(o.cache.st)
        c.s .= 0
        @test all(o.cache.st.s .> 0)
        @test_throws ArgumentError po.merge_states(o.cache.st, o.cache.st)
        # Under a cap the diagonal root honours it, and under a programme set the weighted
        # programme agrees with the root.
        cap = BoundedAllocationSet(; wb = WeightBounds(; lb = 0.05, ub = 0.4))
        Wc = libpath(AdaptiveSubgradient(); set = cap)
        @test all(Wc[2:end, :] .>= 0.05 - 1e-12) && all(Wc[2:end, :] .<= 0.4 + 1e-12)
        pset = resolve(ProgrammeAllocationSet(; slv = slv,
                                              wb = WeightBounds(; lb = 0.05, ub = 0.4)), N)
        h = [1.5, 0.7, 2.0, 1.1]
        q = [0.6, -0.1, 0.4, 0.3]
        @test isapprox(po.project(DiagonalProjection(; h = h), pset, q, q),
                       po.project(DiagonalProjection(; h = h), resolve(cap, N), q, q);
                       atol = 1e-6)
        r = optimise(OPS(; alg = AdaptiveSubgradient(),
                         set = ProgrammeAllocationSet(; slv = slv, tn = 0.02)),
                     rows(rd, 1:5))
        @test isapprox(sum(r.w), 1; atol = 1e-8)
    end

    @testset "The optimistic step by hand: two half-steps" begin
        # Rakhlin and Sridharan (2013, §2.2) on the Euclidean map at η = 0.1 with the
        # last-gradient hint: from v₁ = w₁ = [0.5, 0.5] and x₁ = [1.2, 0.8], g₁ = −[1.2, 0.8],
        # v₂ = Proj([0.62, 0.58]) = [0.52, 0.48] and w₂ = Proj(v₂ + 0.1 [1.2, 0.8]) =
        # Proj([0.64, 0.56]) = [0.54, 0.46].
        alg = OptimisticStep(; alg = GradientProjection(; eta = 0.1))
        w = [0.5, 0.5]
        x = [1.2, 0.8]
        st = po.rule_state_seed(alg, w)
        @test st.v == w && st.u == w && st.w0 == w && st.n == 0
        @test st.res == [0.0, 0.0] && st.m == [0.0, 0.0] && isnothing(st.ps)
        st, w2 = po.online_update!(alg, st, w, x, nothing, simplex2)
        @test isapprox(st.v, [0.52, 0.48]; atol = 1e-12)
        @test isapprox(w2, [0.54, 0.46]; atol = 1e-12)
        @test st.u == w2 && st.n == 1
        @test st.m == -x
        # The residual of the first period is ‖g₁ − 0‖², and the pair shifts.
        @test st.res == [1.2^2 + 0.8^2, 0.0]
        # The entropic case is two normalised exponential steps.
        alge = OptimisticStep(; alg = ExponentiatedGradient(; eta = 0.1))
        ste = po.rule_state_seed(alge, w)
        ste, w2e = po.online_update!(alge, ste, w, x, nothing, simplex2)
        v = w .* exp.(0.1 .* x)
        v ./= sum(v)
        u = v .* exp.(0.1 .* x)
        u ./= sum(u)
        @test isapprox(ste.v, v; atol = 1e-15)
        @test isapprox(w2e, u; atol = 1e-15)
        # The entropic residual is in the maximum norm.
        @test ste.res == [1.2^2, 0.0]
        # The second period reads the gradient at the played w₂, not at v₂, and steps
        # both halves from v₂.
        x2 = [0.9, 1.1]
        st2, w3 = po.online_update!(alg, st, w2, x2, nothing, simplex2)
        g2 = -x2 ./ dot(w2, x2)
        v3 = po.project_simplex(st.v .- 0.1 .* g2)
        @test isapprox(st2.v, v3; atol = 1e-12)
        @test isapprox(w3, po.project_simplex(v3 .- 0.1 .* g2); atol = 1e-12)
        @test isapprox(st2.res, [st.res[1] + sum(abs2, g2 .+ x), st.res[1]]; atol = 1e-12)
        # The Mirror-Prox form re-evaluates the period's loss at the new secondary iterate.
        algp = OptimisticStep(; alg = GradientProjection(; eta = 0.1),
                              predictor = LastGradient(; at_played = false))
        stp = po.rule_state_seed(algp, w)
        stp, w2p = po.online_update!(algp, stp, w, x, nothing, simplex2)
        @test isapprox(stp.v, [0.52, 0.48]; atol = 1e-12)
        mp = -x ./ dot([0.52, 0.48], x)
        @test isapprox(w2p, po.project_simplex([0.52, 0.48] .- 0.1 .* mp); atol = 1e-12)
        @test stp.m ≈ mp
        # The mean-gradient hint is the running mean, one after the first row.
        algm = OptimisticStep(; alg = GradientProjection(; eta = 0.1),
                              predictor = MeanGradient())
        stm = po.rule_state_seed(algm, w)
        @test stm.ps == [0.0, 0.0]
        stm, w2m = po.online_update!(algm, stm, w, x, nothing, simplex2)
        @test w2m ≈ w2 && stm.ps == -x
        stm, w3m = po.online_update!(algm, stm, w2m, x2, nothing, simplex2)
        @test isapprox(stm.ps, (-x .+ g2) ./ 2; atol = 1e-15)
        @test isapprox(w3m, po.project_simplex(v3 .- 0.1 .* stm.ps); atol = 1e-12)
    end

    @testset "A zero hint is the wrapped rule, and the mix is the wrapped rule's" begin
        # A predictor written to the interface: a zero hint, under which the second
        # half-step is the identity and the optimistic step is the wrapped rule exactly,
        # under every geometry.
        for md in (GradientProjection(; eta = 0.1), ExponentiatedGradient(; eta = 0.1),
                   MirrorDescent(; proj = TsallisProjection(), alpha = 0.1),
                   MirrorDescent(; proj = LogBarrierProjection(), eta = InverseSquareRootRate()))
            a = libpath(OptimisticStep(; alg = md, predictor = ZeroHint()))
            b = libpath(md)
            @test maxerr(a, b) < 1e-14
        end
        # A forecast hint moves the path off the plain step and keeps it on the simplex.
        me = PriceLevelExpectedReturns(; alg = MovingAverage(; window = 3))
        a = libpath(OptimisticStep(; alg = GradientProjection(; eta = 0.1),
                                   predictor = ForecastGradient(; me = me)))
        b = libpath(GradientProjection(; eta = 0.1))
        @test maxerr(a, b) > 1e-6
        @test all(isapprox.(sum(a; dims = 2), 1; atol = 1e-12))
        # The forecast hint is the gradient the forecast would give at the secondary point.
        alg = OptimisticStep(; alg = GradientProjection(; eta = 0.1),
                             predictor = ForecastGradient(; me = SimpleExpectedReturns()))
        st = po.rule_state_seed(alg, fill(1 / N, N))
        @test isa(st.ps, po.ForecasterState)
        for t in 1:3
            st, _ = po.online_update!(alg, st, st.u, X[t, :], nothing,
                                      resolve(BoundedAllocationSet(), N))
        end
        xhat = 1 .+ vec(sum(R[1:3, :]; dims = 1)) ./ 3
        @test isapprox(st.m, -xhat ./ dot(st.v, xhat); atol = 1e-12)
        # The uniform mix reads mixed relatives at the unmixed played iterate — the paper's,
        # over the relatives normalised to a period maximum of one — and plays the mix from
        # the unmixed w_{t+1}.
        algα = OptimisticStep(; alg = ExponentiatedGradient(; eta = 0.1, alpha = 0.2))
        w = [0.5, 0.5]
        x = [1.2, 0.8]
        stα = po.rule_state_seed(algα, w)
        stα, wα = po.online_update!(algα, stα, w, x, nothing, simplex2)
        xm = 0.9 .* (x ./ 1.2) .+ 0.1
        v = w .* exp.(0.1 .* xm ./ dot(w, xm))
        v ./= sum(v)
        u = v .* exp.(0.1 .* xm ./ dot(w, xm))
        u ./= sum(u)
        @test isapprox(stα.u, u; atol = 1e-15)
        @test isapprox(wα, 0.8 .* u .+ 0.1; atol = 1e-15)
        o = po.partial_fit!(OPS(; alg = algα), rows(rd, 1:5))
        @test isapprox(o.cache.w, 0.8 .* o.cache.st.u .+ 0.2 / N; atol = 1e-15)
    end

    @testset "The hint-residual rate and the schedules of the wrapped rule" begin
        # Corollary 2: η_t = rmax · min(1 / (√S_{t−1} + √S_{t−2}), 1), the cap before any
        # residual accrues.
        sched = HintResidualRate(; rmax = 0.5)
        alg = OptimisticStep(; alg = GradientProjection(; eta = sched))
        w = [0.5, 0.5]
        st = po.rule_state_seed(alg, w)
        @test po.learning_rate(sched, 1, st) == 0.5
        st, w2 = po.online_update!(alg, st, w, [1.2, 0.8], nothing, simplex2)
        # The first half-step ran at the cap, and the second at the rate of period 2, which
        # reads the residual of period 1, ‖g₁ − 0‖² = 2.08: the paper forms w₂ with η₂.
        eta2 = 0.5 * min(1 / sqrt(1.2^2 + 0.8^2), 1)
        @test isapprox(w2,
                       po.project_simplex(po.project_simplex(w .+ 0.5 .* [1.2, 0.8]) .+
                                          eta2 .* [1.2, 0.8]); atol = 1e-12)
        @test maxerr(w2,
                     po.project_simplex(po.project_simplex(w .+ 0.5 .* [1.2, 0.8]) .+
                                        0.5 .* [1.2, 0.8])) > 1e-3
        @test po.learning_rate(sched, 2, st) == eta2
        st, _ = po.online_update!(alg, st, w2, [0.9, 1.1], nothing, simplex2)
        @test po.learning_rate(sched, 3, st) ==
              0.5 * min(1 / (sqrt(st.res[1]) + sqrt(st.res[2])), 1)
        @test st.res[2] == 1.2^2 + 0.8^2
        # Over the fixture the rate falls and the path stays on the simplex.
        W = libpath(OptimisticStep(;
                                   alg = ExponentiatedGradient(; eta = HintResidualRate())))
        @test all(isapprox.(sum(W; dims = 2), 1; atol = 1e-12)) && all(W .> 0)
        # Under any schedule the second half-step reads the next period's rate: with
        # c / sqrt(t) the secondary iterate steps at c / sqrt(t) and the played one at
        # c / sqrt(t + 1), by hand over the fixture.
        Wi = libpath(OptimisticStep(;
                                    alg = GradientProjection(;
                                                             eta = InverseSquareRootRate(;
                                                                                         c = 0.2))))
        Wh = zeros(T, N)
        Wh[1, :] .= 1 / N
        v = fill(1 / N, N)
        for t in 1:(T - 1)
            g = -X[t, :] ./ dot(Wh[t, :], X[t, :])
            v = po.project_simplex(v .- 0.2 / sqrt(t) .* g)
            Wh[t + 1, :] .= po.project_simplex(v .- 0.2 / sqrt(t + 1) .* g)
        end
        @test maxerr(Wi, Wh) < 1e-14
        # The played mix is projected once more where the set excludes the uniform
        # allocation, as the wrapped rule's is.
        lbset = BoundedAllocationSet(;
                                     wb = WeightBounds(; lb = [0.3, 0.0, 0.0, 0.0], ub = 1))
        Wb = libpath(OptimisticStep(; alg = ExponentiatedGradient(; alpha = 0.5));
                     set = lbset)
        @test all(Wb[2:end, 1] .>= 0.3 - 1e-12)
        @test all(isapprox.(sum(Wb; dims = 2), 1; atol = 1e-12))
        # The wrapped rule's other schedules count the wrapper's periods: the doubling
        # trick restarts both iterates at the Start Allocation and clears the residuals.
        dt = DoublingTrickRate(; N = 2)
        first_stage = po.doubling_stage(dt, 1)[2]
        algd = OptimisticStep(; alg = ExponentiatedGradient(; eta = dt))
        o = po.partial_fit!(OPS(; alg = algd), rows(rd, 1:first_stage))
        @test o.cache.st.n == first_stage
        @test o.cache.st.v == o.cache.st.w0 && o.cache.st.u == o.cache.st.w0
        @test o.cache.st.res == [0.0, 0.0] && all(iszero, o.cache.st.m)
        o = po.partial_fit!(OPS(; alg = algd), rows(rd, 1:(first_stage - 1)))
        @test o.cache.st.v != o.cache.st.w0
        # The self-confident rate reads the wrapper's carrier.
        Wsc = libpath(OptimisticStep(;
                                     alg = ExponentiatedGradient(;
                                                                 eta = SelfConfidentRate())))
        @test all(isapprox.(sum(Wsc; dims = 2), 1; atol = 1e-12))
        osc = po.partial_fit!(OPS(;
                                  alg = OptimisticStep(;
                                                       alg = ExponentiatedGradient(;
                                                                                   eta = SelfConfidentRate()))),
                              rows(rd, 1:6))
        @test osc.cache.st.s[1] > 0 && osc.cache.st.s[2] == log(N)
    end

    @testset "The batch-online identity at block sizes 10, 7 and 1" begin
        me = PriceLevelExpectedReturns(; alg = MovingAverage(; window = 3))
        for alg in (AdaptiveSubgradient(), AdaptiveSubgradient(; delta = 0.5, eta = 0.3),
                    OptimisticStep(), OptimisticStep(; alg = GradientProjection(; eta = 0.2)),
                    OptimisticStep(; predictor = MeanGradient()),
                    OptimisticStep(; predictor = LastGradient(; at_played = false)),
                    OptimisticStep(; alg = GradientProjection(; eta = HintResidualRate()),
                                   predictor = ForecastGradient(; me = me)),
                    OptimisticStep(;
                                   alg = MirrorDescent(; proj = LogBarrierProjection(), alpha = 0.1)))
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

    @testset "The carrier's views and copies, and the wrapper as an expert" begin
        # A windowed forecaster is refit from the head's rows, so its carrier is
        # `nothing`; a folding one rides on the carrier and is viewed with it.
        me = PriceLevelExpectedReturns(; alg = MovingAverage(; window = 3))
        alg = OptimisticStep(; alg = GradientProjection(; eta = 0.1),
                             predictor = ForecastGradient(; me = me))
        o = po.partial_fit!(OPS(; alg = alg), rows(rd, 1:8))
        st = o.cache.st
        @test isnothing(st.ps)
        v = po.port_opt_view(o, [2, 4])
        @test v.cache.st.m == st.m[[2, 4]]
        @test isapprox(sum(v.cache.st.v), 1; atol = 1e-12)
        @test isapprox(sum(v.cache.st.u), 1; atol = 1e-12)
        @test v.cache.st.res == st.res
        of = po.partial_fit!(OPS(;
                                 alg = OptimisticStep(;
                                                      predictor = ForecastGradient(;
                                                                                   me = SimpleExpectedReturns()))),
                             rows(rd, 1:8))
        @test isa(of.cache.st.ps, po.ForecasterState)
        vf = po.port_opt_view(of, [2, 4])
        @test isa(vf.cache.st.ps, po.ForecasterState)
        @test length(vf.cache.st.ps.me.cache.mu) == 2
        c = copy(st)
        c.m .= 0
        c.res .= 0
        @test any(!iszero, st.m) && st.res[1] > 0
        @test_throws ArgumentError po.merge_states(st, st)
        # A mean-gradient carrier is sliced and copied as a vector.
        om = po.partial_fit!(OPS(; alg = OptimisticStep(; predictor = MeanGradient())),
                             rows(rd, 1:8))
        vm = po.port_opt_view(om, [1, 2])
        @test vm.cache.st.ps == om.cache.st.ps[1:2]
        cm = copy(om.cache.st)
        @test cm.ps == om.cache.st.ps && !(cm.ps === om.cache.st.ps)
        # The two rules are experts of a mixture and weightings of it.
        mix = ExpertMixture(; experts = [AdaptiveSubgradient(), OptimisticStep()])
        r = optimise(OPS(; alg = mix), rows(rd, 1:6))
        @test isapprox(sum(r.w), 1; atol = 1e-12)
        mixw = ExpertMixture(; experts = [ConstantRebalancedPortfolio(), BuyAndHold()],
                             alg = OptimisticStep())
        r = optimise(OPS(; alg = mixw), rows(rd, 1:6))
        @test isapprox(sum(r.w), 1; atol = 1e-12)
        # Both rules step on a Risk Loss over the head's rows, and stay on the simplex.
        for a in (AdaptiveSubgradient(; obj = RiskLoss(; window = 8)),
                  OptimisticStep(; alg = GradientProjection(; obj = RiskLoss(; window = 8))),
                  OptimisticStep(; alg = GradientProjection(; obj = RiskLoss(; window = 8)),
                                 predictor = LastGradient(; at_played = false)))
            W = libpath(a)
            @test all(isapprox.(sum(W; dims = 2), 1; atol = 1e-12)) && all(W .>= -1e-15)
            @test maxerr(W, libpath(GradientProjection())) > 1e-6
        end
        # The rules run through the walk-forward at test_size = 1.
        cv = IndexWalkForward(4, 1)
        for a in (AdaptiveSubgradient(), OptimisticStep())
            pr = cross_val_predict(OPS(; alg = a), rd, cv)
            @test isa(pr, MultiPeriodPredictionResult)
            @test length(pr.pred) == T - 4
        end
    end
end
