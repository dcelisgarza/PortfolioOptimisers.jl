#=
The adaptive and optimistic rows of the online portfolio selection family's fourth set, as
ADR 0165 rules them (issue #1181 on map #1148): `AdaptiveSubgradient` under
`DiagonalProjection`, and the optimistic step of `MirrorDescent` under its three Gradient
Predictors and the hint-residual rate (the `predictor` field since #1864).

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
                                  ::AbstractVector, ::Any, ::Integer)
        return nothing, zero(g)
    end
    # A hint with equal entries, the hint of a flat forecast: the projection onto the budget
    # undoes its uniform shift.
    struct FlatHint <: po.AbstractGradientPredictor end
    function po.predict_gradient!(::FlatHint, ::Nothing, ::po.AbstractOnlineObjective,
                                  g::AbstractVector, ::AbstractVector, ::AbstractVector,
                                  ::AbstractVector, ::Any, ::Integer)
        return nothing, fill(-1.3, length(g))
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
        @test ag.eta == 1 / sqrt(2) && ag.delta == 0 && isa(ag.obj, LogWealth)
        @test_throws DomainError AdaptiveSubgradient(; eta = 0)
        @test_throws DomainError AdaptiveSubgradient(; delta = -0.1)
        # The rule builds its geometry at the step (ADR 0168): no `proj` slot to fill.
        @test_throws MethodError AdaptiveSubgradient(; proj = EuclideanProjection())
        # The objective slot is the first-order rule's: a Risk Loss reads the head's rows.
        @test po.rows_needed(AdaptiveSubgradient(; obj = RiskLoss(; window = 5))) == 5
        @test po.rows_needed(GradientProjection(; obj = RiskLoss(; window = 6),
                                                predictor = LastGradient())) == 6
        # A zero entry is a seminorm, which the geometry takes at its limit (#1777).
        @test po.DiagonalProjection([1.0, 0.0]).h == [1.0, 0.0]
        @test_throws DomainError po.DiagonalProjection([1.0, -0.1])
        @test_throws MethodError po.DiagonalProjection()
        @test po.projection_geometry(ag) == EuclideanProjection()
        os = MirrorDescent(; predictor = LastGradient())
        @test isa(os.predictor, LastGradient) && isnothing(MirrorDescent().predictor)
        @test os.predictor.at_played
        @test !LastGradient(; at_played = false).at_played
        # The optimistic step reads two gradients a period, so a Gradient Transform with a
        # state is refused with a predictor, and kept without one.
        @test_throws ArgumentError MirrorDescent(; grad = GradientMomentum(),
                                                 predictor = LastGradient())
        @test_throws ArgumentError ExponentiatedGradient(; grad = RootMeanSquareGradient(),
                                                         predictor = MeanGradient())
        @test isnothing(EGE().predictor)
        @test isa(po.rule_state_seed(MirrorDescent(), [0.5, 0.5]), po.MirrorDescentState)
        @test isa(po.rule_state_seed(os, [0.5, 0.5]), po.OptimisticMirrorDescentState)
        # The view of the rule views its predictor and keeps the rest.
        fv = po.port_opt_view(GradientProjection(; eta = 0.2,
                                                 predictor = ForecastGradient(;
                                                                              me = SimpleExpectedReturns())),
                              [1, 3])
        @test fv.eta == 0.2 &&
              isa(fv.proj, EuclideanProjection) &&
              isa(fv.predictor, ForecastGradient)
        @test isnothing(po.port_opt_view(MirrorDescent(), [1, 3]).predictor)
        # The slot's bound refuses an `Online` wrapper before the name check does.
        @test_throws TypeError ForecastGradient(;
                                                me = Online(;
                                                            est = SimpleExpectedReturns()))
        @test_throws DomainError HintResidualRate(; rmax = 0)
        @test po.projection_geometry(os) == EntropicProjection()
        @test po.rows_needed(os) == 0
        @test po.rows_needed(MirrorDescent(; predictor = MeanGradient())) == 0
        fg = MirrorDescent(;
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
                @test po.project(po.DiagonalProjection(h), set, q, q) == w
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
        # A zero gradient is a zero step and no mass accrues. At delta = 0 the diagonal is
        # zero, and its projection is the Euclidean one.
        stz = po.rule_state_seed(alg, w)
        stz, wz = po.online_update!(AdaptiveSubgradient(; obj = RiskLoss(; window = 3)),
                                    stz, w, x1,
                                    ReturnsResult(; nx = ["A", "B"], X = zeros(1, 2)),
                                    simplex2)
        @test wz == w && stz.s == [0.0, 0.0] && stz.n == 1
        # Issue #1777: at delta = 0 an asset whose every price relative so far is zero has
        # no mass. The pseudo-inverse keeps its raw step at its weight. At η = 0.2 and
        # x₁ = [1.2, 0], g₁ = [-2, 0] and q = [0.7, 0.5]. The seminorm fixes w₁ = 0.7, and
        # the limit gives the rest of the budget, 0.3, to the second asset.
        a2 = AdaptiveSubgradient(; eta = 0.2)
        s0, w0 = po.online_update!(a2, po.rule_state_seed(a2, w), w, [1.2, 0.0], nothing,
                                   simplex2)
        @test isapprox(w0, [0.7, 0.3]; atol = 1e-15) && s0.s == [2.0, 0.0]
        aδ = AdaptiveSubgradient(; eta = 0.2, delta = 1e-12)
        _, wδ = po.online_update!(aδ, po.rule_state_seed(aδ, w), w, [1.2, 0.0], nothing,
                                  simplex2)
        @test maxerr(w0, wδ) < 1e-10
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
        @test isapprox(po.project(po.DiagonalProjection(h), pset, q, q),
                       po.project(po.DiagonalProjection(h), resolve(cap, N), q, q);
                       atol = 1e-6)
        r = optimise(OPS(; alg = AdaptiveSubgradient(),
                         set = ProgrammeAllocationSet(; slv = slv,
                                                      tn = Turnover(; w = zeros(4),
                                                                    val = 0.02))),
                     rows(rd, 1:5))
        @test isapprox(sum(r.w), 1; atol = 1e-8)
    end

    @testset "A zero entry of the diagonal is the limit of a positive one (#1777)" begin
        # The seminorm projection is the limit of the projection in `h + δ` as `δ → 0⁺`. Each
        # raw step below reaches one branch on the simplex and on the box [-0.2, 0.6]: the
        # free assets absorb the budget, sit at their caps, or sit at their floors.
        h = [1e-3, 2e-3, 5e-4, 0.0, 0.0]
        cases = ([0.1, 0.2, 0.3, 0.25, 0.1], [0.9, -0.5, 0.3, 0.25, 0.1],
                 [-0.3, -0.2, 0.1, 0.4, 0.1], [0.6, 0.5, 0.3, 0.25, 0.1])
        box = WeightBounds(; lb = fill(-0.2, 5), ub = fill(0.6, 5))
        simplex = WeightBounds(; lb = zeros(5), ub = ones(5))
        for q in cases, wb in (simplex, box)
            w = po.bounded_quadratic_projection(q, wb, h)
            @test isapprox(sum(w), 1; atol = 1e-14)
            @test maxerr(w, po.bounded_quadratic_projection(q, wb, h .+ 1e-12)) < 1e-8
            # Every minimiser of the seminorm shares the weights where `h` is positive. A
            # scale of `h` does not move the projection, and the programme is accurate only
            # on weights near one.
            @test isapprox(w[1:3], qp_projection(q, 1e3 .* h .+ 1e-7, wb.lb, wb.ub)[1:3];
                           atol = 1e-6)
        end
        # The branches by hand on the simplex. q = [0.1, 0.2, 0.3, 0.25, 0.1]: the priced
        # assets keep their step, and the free ones take the rest, 0.4, at an equal shift.
        @test isapprox(po.bounded_quadratic_projection(cases[1], simplex, h),
                       [0.1, 0.2, 0.3, 0.275, 0.125]; atol = 1e-15)
        # q = [0.6, 0.5, 0.3, 0.25, 0.1]: the priced assets overrun the budget, so the free
        # ones sit at zero.
        @test po.bounded_quadratic_projection(cases[4], simplex, h)[4:5] == [0.0, 0.0]
        # With no positive entry the limit is the Euclidean projection.
        @test isapprox(po.bounded_quadratic_projection(cases[2], box, zeros(5)),
                       po.bounded_quadratic_projection(cases[2], box); atol = 1e-15)
        # A negative or `NaN` entry is neither priced nor free, so the helper, which a user
        # geometry calls with no `DiagonalProjection` in front, refuses it.
        for hb in ([-1.0, 1.0, 1.0, 0.0, 0.0], [NaN, 1.0, 1.0, 0.0, 0.0])
            @test_throws DomainError po.bounded_quadratic_projection(cases[1], simplex, hb)
        end
        set = resolve(BoundedAllocationSet(; wb = box), 5)
        @test po.project(po.DiagonalProjection(zeros(5)), set, cases[2], cases[2]) ==
              po.project(EuclideanProjection(), set, cases[2], cases[2])
        # The programme arm solves the same limit in two stages.
        for q in cases, wb in (simplex, box)
            pset = resolve(ProgrammeAllocationSet(; slv = slv, wb = wb), 5)
            @test isapprox(po.project(po.DiagonalProjection(h), pset, q, q),
                           po.bounded_quadratic_projection(q, wb, h); atol = 1e-5)
        end
        # Under an L1 penalty the minimisers form a face. With q = [0.9, -0.5, 0.3, 0.25,
        # 0.1] on the box, the budget multiplier is -0.01, so the priced assets keep their
        # clipped step [0.6, 0, 0.3]. The free assets need w ≥ 0 and a sum of 0.1, and the
        # point of that face nearest [0.25, 0.1] is [0.1, 0].
        pl1 = resolve(ProgrammeAllocationSet(; slv = slv, wb = box, l1 = 0.01), 5)
        @test isapprox(po.project(po.DiagonalProjection(h), pl1, cases[2], cases[2]),
                       [0.6, 0.0, 0.3, 0.1, 0.0]; atol = 1e-6)
        # A strictly convex penalty has one minimiser, which the second stage keeps.
        pl2 = resolve(ProgrammeAllocationSet(; slv = slv, wb = box,
                                             l2 = po.L2Regularisation(; val = 0.01)), 5)
        wl2 = po.project(po.DiagonalProjection(h), pl2, cases[2], cases[2])
        @test isapprox(wl2[4], wl2[5]; atol = 1e-5)
        # A second stage that fails keeps the minimiser of the first (#1778). The factory
        # gives Clarabel to the first solve and refuses the second. The L2 penalty has one
        # minimiser, so the first stage finds the limit too. A Held Step would return `q`.
        calls = Ref(0)
        refuse2 = () -> if (calls[] += 1) == 1
            Clarabel.Optimizer()
        else
            error("the second stage is refused")
        end
        refused(f) = resolve(ProgrammeAllocationSet(;
                                                    slv = Solver(; name = :refused,
                                                                 solver = f,
                                                                 settings = slv.settings,
                                                                 check_sol = slv.check_sol),
                                                    wb = box,
                                                    l2 = po.L2Regularisation(; val = 0.01)),
                             5)
        wf = @test_logs (:warn,) match_mode=:any po.project(po.DiagonalProjection(h),
                                                            refused(refuse2), cases[2],
                                                            cases[2])
        @test calls[] == 2
        @test isapprox(wf, wl2; atol = 1e-5)
        # A first stage that fails holds the step.
        refuse1 = () -> error("the first stage is refused")
        wh = @test_logs (:warn,) match_mode=:any po.project(po.DiagonalProjection(h),
                                                            refused(refuse1), cases[2],
                                                            cases[2])
        @test wh == cases[2]

        # The issue's panel: the window of the Risk Loss does not price D, so its gradient is
        # zero there. On two rows the first step reads one row and holds the uniform start.
        # The second has g = 2 Σ u on A, B, C, with g_C < 0 and the others positive, so
        # q = u - η sign(g) on them and q_D = 1/4. C keeps 1/4 + η, and D takes the rest.
        R7 = 0.02 .* randn(StableRNG(7), 40, 4)
        Rn = copy(R7[1:3, :])
        am = trues(3, 4)
        am[:, 4] .= false
        Rn[.!am] .= NaN
        rdn = ReturnsResult(; nx = nx, X = Rn, ts = ts[1:3],
                            pnl = AssetPanel(; amsk = am, emsk = copy(am)))
        risk = RiskLoss(; window = 10)
        o2 = po.partial_fit!(OPS(; alg = AdaptiveSubgradient(; obj = risk)), rows(rdn, 1:2))
        eta = 1 / sqrt(2)
        @test isapprox(o2.cache.w, [0.0, 0.0, 0.25 + eta, 0.75 - eta]; atol = 1e-15)
        @test o2.cache.st.s[4] == 0
        lim(d) = optimise(OPS(; alg = AdaptiveSubgradient(; delta = d, obj = risk)), rdn).w
        @test maxerr(lim(0), lim(1e-12)) < 1e-9
        # A programme set that fits a prior reduces the geometry to its Investable Mask, the
        # diagonal and the Gram matrix alike.
        pce = ProgrammeAllocationSet(; slv = slv,
                                     r = Variance(;
                                                  settings = RiskMeasureSettings(;
                                                                                 ub = 1.0)))
        for d in (0, 1e-3)
            wp = optimise(OPS(; alg = AdaptiveSubgradient(; delta = d, obj = risk),
                              set = pce), rdn).w
            @test isapprox(sum(wp), 1; atol = 1e-8) && wp[4] == 0
        end
        Rg = copy(R7[1:6, :])
        amg = trues(6, 4)
        amg[:, 4] .= false
        Rg[.!amg] .= NaN
        rdg = ReturnsResult(; nx = nx, X = Rg, ts = ts[1:6],
                            pnl = AssetPanel(; amsk = amg, emsk = copy(amg)))
        wg = optimise(OPS(; alg = NewtonStep(; proj = GramProjection(; slv = slv)),
                          set = pce), rdg).w
        @test isapprox(sum(wg), 1; atol = 1e-8) && wg[4] == 0
        # The Gram view: at w_Z = 0 the full norm and the reduced one differ by a constant.
        B = randn(StableRNG(3), 5, 5)
        A = B' * B + I
        qg = randn(StableRNG(4), 5)
        imk = BitVector([1, 1, 0, 1, 0])
        gp, qr = po.investable_geometry_view(imk, GramProjection(; slv = slv, A = A), qg)
        @test gp.A == A[imk, imk]
        function gap(wp)
            wf = zeros(5)
            wf[imk] .= wp
            return dot(wf - qg, A * (wf - qg)) - dot(wp - qr, gp.A * (wp - qr))
        end
        gaps = [gap(randn(StableRNG(k), 3)) for k in 1:4]
        @test maximum(gaps) - minimum(gaps) < 1e-12
        # A Gram geometry with no matrix is the identity: the view keeps it (#1912).
        g0 = GramProjection(; slv = slv)
        gp0, qr0 = po.investable_geometry_view(imk, g0, qg)
        @test gp0 === g0 && qr0 == qg[imk]
        dp, qd = po.investable_geometry_view(imk, po.DiagonalProjection(collect(1.0:5)), qg)
        @test dp.h == [1.0, 2.0, 4.0] && qd == qg[imk]
    end

    @testset "The optimistic step by hand: two half-steps" begin
        # Rakhlin and Sridharan (2013, §2.2) on the Euclidean map at η = 0.1 with the
        # last-gradient hint: from v₁ = w₁ = [0.5, 0.5] and x₁ = [1.2, 0.8], g₁ = −[1.2, 0.8],
        # v₂ = Proj([0.62, 0.58]) = [0.52, 0.48] and w₂ = Proj(v₂ + 0.1 [1.2, 0.8]) =
        # Proj([0.64, 0.56]) = [0.54, 0.46].
        alg = GradientProjection(; eta = 0.1, predictor = LastGradient())
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
        alge = ExponentiatedGradient(; eta = 0.1, predictor = LastGradient())
        ste = po.rule_state_seed(alge, w)
        ste, w2e = po.online_update!(alge, ste, w, x, nothing, simplex2)
        v = w .* exp.(0.1 .* x)
        v ./= sum(v)
        u = v .* exp.(0.1 .* x)
        u ./= sum(u)
        @test isapprox(ste.v, v; atol = 1e-15)
        @test isapprox(w2e, u; atol = 1e-15)
        # The entropic residual is in the maximum norm, and the Tsallis and log-barrier
        # residuals are too. A MirrorDescent takes no other geometry, and none has a method.
        @test ste.res == [1.2^2, 0.0]
        @test po.hint_residual(TsallisProjection(), [1.0, -2.0]) == 4.0
        @test po.hint_residual(LogBarrierProjection(), [1.0, -2.0]) == 4.0
        @test_throws MethodError po.hint_residual(po.DiagonalProjection([1.0, 1.0]),
                                                  [1.0, -2.0])
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
        algp = GradientProjection(; eta = 0.1,
                                  predictor = LastGradient(; at_played = false))
        stp = po.rule_state_seed(algp, w)
        stp, w2p = po.online_update!(algp, stp, w, x, nothing, simplex2)
        @test isapprox(stp.v, [0.52, 0.48]; atol = 1e-12)
        mp = -x ./ dot([0.52, 0.48], x)
        @test isapprox(w2p, po.project_simplex([0.52, 0.48] .- 0.1 .* mp); atol = 1e-12)
        @test stp.m ≈ mp
        # The two hints differ from the first period on: the Mirror-Prox hint is read at
        # v₂ = [0.52, 0.48], not at the played w₁ = [0.5, 0.5].
        @test maxerr(stp.m, st.m) > 5e-3
        @test round.(w2p; digits = 5) == [0.53984, 0.46016]
        # The mean-gradient hint is the running mean, one after the first row.
        algm = GradientProjection(; eta = 0.1, predictor = MeanGradient())
        stm = po.rule_state_seed(algm, w)
        @test stm.ps == [0.0, 0.0]
        stm, w2m = po.online_update!(algm, stm, w, x, nothing, simplex2)
        @test w2m ≈ w2 && stm.ps == -x
        stm, w3m = po.online_update!(algm, stm, w2m, x2, nothing, simplex2)
        @test isapprox(stm.ps, (-x .+ g2) ./ 2; atol = 1e-15)
        @test isapprox(w3m, po.project_simplex(v3 .- 0.1 .* stm.ps); atol = 1e-12)
    end

    @testset "A zero hint is the plain step, and the mix is the plain step's" begin
        # A predictor written to the interface: a zero hint, under which the second
        # half-step is the identity and the optimistic step is the plain step exactly,
        # under every geometry.
        for kw in ((; proj = EuclideanProjection(), eta = 0.1),
                   (; proj = EntropicProjection(), eta = 0.1),
                   (; proj = TsallisProjection(), alpha = 0.1),
                   (; proj = LogBarrierProjection(), eta = InverseSquareRootRate()))
            a = libpath(MirrorDescent(; kw..., predictor = ZeroHint()))
            b = libpath(MirrorDescent(; kw...))
            @test maxerr(a, b) < 1e-14
            @test maxerr(libpath(MirrorDescent(; kw..., predictor = FlatHint())), b) < 1e-14
        end
        # A forecast hint moves the path off the plain step and keeps it on the simplex.
        me = PriceLevelExpectedReturns(; alg = MovingAverage(; window = 3))
        a = libpath(GradientProjection(; eta = 0.1,
                                       predictor = ForecastGradient(; me = me)))
        b = libpath(GradientProjection(; eta = 0.1))
        @test maxerr(a, b) > 1e-6
        @test all(isapprox.(sum(a; dims = 2), 1; atol = 1e-12))
        # The forecast hint is the gradient the forecast would give at the secondary point.
        alg = GradientProjection(; eta = 0.1,
                                 predictor = ForecastGradient(;
                                                              me = SimpleExpectedReturns()))
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
        algα = ExponentiatedGradient(; eta = 0.1, alpha = 0.2, predictor = LastGradient())
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

    @testset "Under the mix a forecaster folds the traded relative, and a re-evaluated loss reads the mixed one" begin
        # Issue #1210: the predictor is handed both relatives. A folding forecaster walks the
        # path that traded, so its forecast is the statistic of the raw rows, not of the
        # mixed ones compressed toward one; the Mirror-Prox hint re-evaluates the loss on the
        # mixed relative the gradient was read on.
        set = resolve(BoundedAllocationSet(), N)
        md = (; eta = 0.1, alpha = 0.2)
        alg = ExponentiatedGradient(; md...,
                                    predictor = ForecastGradient(;
                                                                 me = SimpleExpectedReturns()))
        st = po.rule_state_seed(alg, fill(1 / N, N))
        @test isa(st.ps, po.ForecasterState)
        for t in 1:3
            st, _ = po.online_update!(alg, st, st.u, X[t, :], nothing, set)
        end
        xhat = 1 .+ vec(sum(R[1:3, :]; dims = 1)) ./ 3
        @test isapprox(st.m, -xhat ./ dot(st.v, xhat); atol = 1e-12)
        # The mixed path is a different statistic: the hint is scale-free, so only the mix's
        # uniform shift moves it, by about 1e-3 on this fixture, six orders above the pin.
        Xm = mapreduce(t -> po.mixed_relatives(X[t, :], 0.2)', vcat, 1:3)
        xhatm = vec(sum(Xm; dims = 1)) ./ 3
        @test maxerr(st.m, -xhatm ./ dot(st.v, xhatm)) > 1e-4
        # A folding price-level statistic sees the same raw path.
        algf = ExponentiatedGradient(; md...,
                                     predictor = ForecastGradient(;
                                                                  me = PriceLevelExpectedReturns(;
                                                                                                 alg = ExponentialMovingAverage(;
                                                                                                                                alpha = 0.5))))
        stf = po.rule_state_seed(algf, fill(1 / N, N))
        @test isa(stf.ps, po.ForecasterState)
        for t in 1:3
            stf, _ = po.online_update!(algf, stf, stf.u, X[t, :], nothing, set)
        end
        stg = po.forecaster_seed(algf.predictor.me)
        xhatf = nothing
        for t in 1:3
            stg, xhatf = po.forecast_relative(algf.predictor.me, stg, X[t, :], nothing)
        end
        @test isapprox(stf.m, -xhatf ./ dot(stf.v, xhatf); atol = 1e-12)
        # The Mirror-Prox hint reads the mixed relative at the secondary iterate: the loss
        # the gradient saw, not the traded one.
        algl = ExponentiatedGradient(; md..., predictor = LastGradient(; at_played = false))
        w = fill(1 / N, N)
        stl = po.rule_state_seed(algl, w)
        stl, _ = po.online_update!(algl, stl, w, X[1, :], nothing, set)
        xm = po.mixed_relatives(X[1, :], 0.2)
        @test isapprox(stl.m, -xm ./ dot(stl.v, xm); atol = 1e-14)
        @test maxerr(stl.m, -X[1, :] ./ dot(stl.v, X[1, :])) > 1e-4
    end

    @testset "The hint-residual rate and the schedules of the optimistic step" begin
        # Corollary 2: η_t = rmax · min(1 / (√S_{t−1} + √S_{t−2}), 1), the cap before any
        # residual accrues.
        sched = HintResidualRate(; rmax = 0.5)
        alg = GradientProjection(; eta = sched, predictor = LastGradient())
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
        W = libpath(ExponentiatedGradient(; eta = HintResidualRate(),
                                          predictor = LastGradient()))
        @test all(isapprox.(sum(W; dims = 2), 1; atol = 1e-12)) && all(W .> 0)
        # Under any schedule the second half-step reads the next period's rate: with
        # c / sqrt(t) the secondary iterate steps at c / sqrt(t) and the played one at
        # c / sqrt(t + 1), by hand over the fixture.
        Wi = libpath(GradientProjection(; eta = InverseSquareRootRate(; c = 0.2),
                                        predictor = LastGradient()))
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
        # allocation, as the plain step's is.
        lbset = BoundedAllocationSet(;
                                     wb = WeightBounds(; lb = [0.3, 0.0, 0.0, 0.0], ub = 1))
        Wb = libpath(ExponentiatedGradient(; alpha = 0.5, predictor = LastGradient());
                     set = lbset)
        @test all(Wb[2:end, 1] .>= 0.3 - 1e-12)
        @test all(isapprox.(sum(Wb; dims = 2), 1; atol = 1e-12))
        # The other schedules count the periods of the optimistic step: the doubling
        # trick restarts both iterates at the Start Allocation and clears the residuals.
        dt = DoublingTrickRate(; N = 2)
        first_stage = po.doubling_stage(dt, 1)[2]
        algd = ExponentiatedGradient(; eta = dt, predictor = LastGradient())
        o = po.partial_fit!(OPS(; alg = algd), rows(rd, 1:first_stage))
        @test o.cache.st.n == first_stage
        @test o.cache.st.v == o.cache.st.w0 && o.cache.st.u == o.cache.st.w0
        @test o.cache.st.res == [0.0, 0.0] && all(iszero, o.cache.st.m)
        o = po.partial_fit!(OPS(; alg = algd), rows(rd, 1:(first_stage - 1)))
        @test o.cache.st.v != o.cache.st.w0
        # The self-confident rate reads the carrier of the optimistic step.
        Wsc = libpath(ExponentiatedGradient(; eta = SelfConfidentRate(),
                                            predictor = LastGradient()))
        @test all(isapprox.(sum(Wsc; dims = 2), 1; atol = 1e-12))
        osc = po.partial_fit!(OPS(;
                                  alg = ExponentiatedGradient(; eta = SelfConfidentRate(),
                                                              predictor = LastGradient())),
                              rows(rd, 1:6))
        @test osc.cache.st.s[1] > 0 && osc.cache.st.s[2] == log(N)
        # Under the mean-gradient hint the residual sum is at most ‖g₁‖² plus twice the
        # spread of the gradients about their mean: 4.066 against 4.113 on the fixture.
        algv = GradientProjection(; eta = 0.1, predictor = MeanGradient())
        stv = po.rule_state_seed(algv, fill(1 / N, N))
        G = zeros(T - 1, N)
        for t in 1:(T - 1)
            G[t, :] .= -X[t, :] ./ dot(stv.u, X[t, :])
            stv, _ = po.online_update!(algv, stv, stv.u, X[t, :], nothing,
                                       resolve(BoundedAllocationSet(), N))
        end
        spread = sum(abs2, G .- sum(G; dims = 1) ./ (T - 1))
        @test stv.res[1] <= sum(abs2, G[1, :]) + 2 * spread
        @test round(stv.res[1]; digits = 3) == 4.066
    end

    @testset "The batch-online identity at block sizes 10, 7 and 1" begin
        me = PriceLevelExpectedReturns(; alg = MovingAverage(; window = 3))
        for alg in (AdaptiveSubgradient(), AdaptiveSubgradient(; delta = 0.5, eta = 0.3),
                    MirrorDescent(; predictor = LastGradient()),
                    GradientProjection(; eta = 0.2, predictor = LastGradient()),
                    MirrorDescent(; predictor = MeanGradient()),
                    MirrorDescent(; predictor = LastGradient(; at_played = false)),
                    GradientProjection(; eta = HintResidualRate(),
                                       predictor = ForecastGradient(; me = me)),
                    MirrorDescent(; proj = LogBarrierProjection(), alpha = 0.1,
                                  predictor = LastGradient()))
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

    @testset "The carrier's views and copies, and the optimistic step as an expert" begin
        # A windowed forecaster is refit from the head's rows, so its carrier is
        # `nothing`; a folding one rides on the carrier and is viewed with it.
        me = PriceLevelExpectedReturns(; alg = MovingAverage(; window = 3))
        alg = GradientProjection(; eta = 0.1, predictor = ForecastGradient(; me = me))
        o = po.partial_fit!(OPS(; alg = alg), rows(rd, 1:8))
        st = o.cache.st
        @test isnothing(st.ps)
        v = po.port_opt_view(o, [2, 4])
        @test v.cache.st.m == st.m[[2, 4]]
        @test isapprox(sum(v.cache.st.v), 1; atol = 1e-12)
        @test isapprox(sum(v.cache.st.u), 1; atol = 1e-12)
        @test v.cache.st.res == st.res
        of = po.partial_fit!(OPS(;
                                 alg = MirrorDescent(;
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
        om = po.partial_fit!(OPS(; alg = MirrorDescent(; predictor = MeanGradient())),
                             rows(rd, 1:8))
        vm = po.port_opt_view(om, [1, 2])
        @test vm.cache.st.ps == om.cache.st.ps[1:2]
        cm = copy(om.cache.st)
        @test cm.ps == om.cache.st.ps && !(cm.ps === om.cache.st.ps)
        # The two rules are experts of a mixture and weightings of it.
        mix = ExpertMixture(;
                            experts = [AdaptiveSubgradient(),
                                       MirrorDescent(; predictor = LastGradient())])
        r = optimise(OPS(; alg = mix), rows(rd, 1:6))
        @test isapprox(sum(r.w), 1; atol = 1e-12)
        mixw = ExpertMixture(; experts = [ConstantRebalancedPortfolio(), BuyAndHold()],
                             alg = MirrorDescent(; predictor = LastGradient()))
        r = optimise(OPS(; alg = mixw), rows(rd, 1:6))
        @test isapprox(sum(r.w), 1; atol = 1e-12)
        # Both rules step on a Risk Loss over the head's rows, and stay on the simplex.
        for a in (AdaptiveSubgradient(; obj = RiskLoss(; window = 8)),
                  GradientProjection(; obj = RiskLoss(; window = 8), predictor = LastGradient()),
                  GradientProjection(; obj = RiskLoss(; window = 8),
                                     predictor = LastGradient(; at_played = false)))
            W = libpath(a)
            @test all(isapprox.(sum(W; dims = 2), 1; atol = 1e-12)) && all(W .>= -1e-15)
            @test maxerr(W, libpath(GradientProjection())) > 1e-6
        end
        # The rules run through the walk-forward at test_size = 1.
        cv = IndexWalkForward(4, 1)
        for a in (AdaptiveSubgradient(), MirrorDescent(; predictor = LastGradient()))
            pr = cross_val_predict(OPS(; alg = a), rd, cv)
            @test isa(pr, MultiPeriodPredictionResult)
            @test length(pr.pred) == T - 4
        end
    end
end

#=
The low-dimension ensemble prior and the online low-dimension ensemble portfolio (issue #1186 on
map #1148): a prior whose mean and covariance come off the same random low-dimensional lagged
regressions, and the `FollowTheLeader` configuration whose programme is the paper's
mean-variance-turnover objective over that prior.

Parity: the moments are checked against the paper's equations 4 to 10 written out by hand on
one full-system subsystem, where the kernel weights are all one, and on two single-asset
subsystems, where they are not; the programme is checked against the paper's own solver, the
relaxed coordinate-wise descent followed by a simplex projection, and the tolerance that
relaxation costs is stated; at `xi = 0`, `gamma = 0` the programme is the one-hot argmax of the
forecast.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, StatsBase, Dates,
      Clarabel
@testset "Online portfolio selection: the low-dimension ensemble" begin
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection
    rows(r, i) = po.port_opt_view(r, i, :)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                 "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                                 "max_iter" => 500),
                 check_sol = (; allow_local = true, allow_almost = true))
    # The Euclidean projection onto the simplex, for the paper's last step.
    function simplex(v)
        u = sort(v; rev = true)
        css = cumsum(u)
        rho = findlast(k -> u[k] + (1 - css[k]) / k > 0, eachindex(u))
        return max.(v .+ (1 - css[rho]) / rho, 0)
    end

    rng = StableRNG(11)
    T, N = 40, 4
    R = 0.02 .* randn(rng, T, N)
    nx = ["A", "B", "C", "D"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = R, ts = ts)

    @testset "Construction" begin
        pe = LowDimensionEnsemblePrior()
        @test pe.n_subsystems == 300 && pe.subsystem_size == 3 && pe.sigma == 0.025
        @test isa(pe.pdm, Posdef) && isnothing(pe.seed)
        @test_throws DomainError LowDimensionEnsemblePrior(; n_subsystems = 0)
        @test_throws DomainError LowDimensionEnsemblePrior(; subsystem_size = 0)
        @test_throws DomainError LowDimensionEnsemblePrior(; sigma = 0)
        @test_throws DomainError LowDimensionEnsemblePrior(; sigma = Inf)
        @test isnothing(po.factor_residual_config(pe))
        # The floor a tree states reaches the rule that holds it.
        @test po.fit_min_rows(pe) == 3
        @test po.fit_min_rows(EmpiricalPrior()) == 1
        @test po.fit_min_rows(nothing) == 1
        @test po.fit_min_rows(prior(EmpiricalPrior(), R)) == 1
        @test po.fit_min_rows(PriorExpectedReturns(; pe = pe)) == 3
        @test po.forecast_min_rows(PriorExpectedReturns(; pe = pe)) == 3
        @test po.forecast_min_rows(SimpleExpectedReturns()) == 1
        alg = LowDimensionEnsemblePortfolio(; N = N, slv = slv)
        @test po.leader_min_rows(alg.opt) == 3
        @test po.leader_min_rows(MeanRisk(;
                                          opt = JuMPOptimiser(; pe = EmpiricalPrior(),
                                                              slv = slv))) == 2
        @test alg.sel == LastRows(; W = 6)
        @test alg.gamma == 0
        @test isa(alg.opt.obj, MaximumUtility) && alg.opt.obj.l == 0.25
        @test isa(alg.opt.r, Variance) && isa(alg.opt.r.alg, QuadRiskExpr)
        @test isa(alg.opt.opt.pe, LowDimensionEnsemblePrior)
        @test alg.opt.opt.fees.tn.val == 0.002
        @test alg.opt.opt.fees.tn.w == fill(0.25, N)
        @test alg.opt.opt.sc == 1 && alg.opt.opt.so == 1
        algs = LowDimensionEnsemblePortfolio(; N = N, slv = slv, sc = 2, so = 1e-3)
        @test algs.opt.opt.sc == 2 && algs.opt.opt.so == 1e-3
        @test_throws DomainError LowDimensionEnsemblePortfolio(; N = 0, slv = slv)
        @test_throws DomainError LowDimensionEnsemblePortfolio(; N = N, window = 1,
                                                               slv = slv)
        @test_throws DomainError LowDimensionEnsemblePortfolio(; N = N, gamma = -1,
                                                               slv = slv)
        @test_throws DomainError LowDimensionEnsemblePortfolio(; N = N, xi = -1, slv = slv)
    end

    @testset "The prior on one full-system subsystem" begin
        # One subsystem over every asset: the kernel weights are all one, the forecast is
        # the regression's, and the covariance is the regressors' sample covariance pushed
        # through the coefficients (equations 4, 5, 8 and 9).
        W6 = R[1:6, :]
        P = 1 .+ W6
        Y = P[2:end, :]
        Z = P[1:(end - 1), :]
        B = Z \ Y
        pe1 = LowDimensionEnsemblePrior(; n_subsystems = 1, subsystem_size = N,
                                        pdm = nothing)
        pr = prior(pe1, W6)
        @test isa(pr, LowOrderPrior)
        @test pr.X === W6 || pr.X == W6
        @test isapprox(pr.mu, vec(P[end, :]' * B) .- 1; atol = 1e-12)
        @test isapprox(pr.sigma, B' * cov(Y) * B; atol = 1e-12)
        @test issymmetric(pr.sigma)
        # A subsystem wider than the universe is clipped to it.
        pe9 = LowDimensionEnsemblePrior(; n_subsystems = 1, subsystem_size = 9,
                                        pdm = nothing)
        @test isapprox(prior(pe9, W6).sigma, pr.sigma; atol = 1e-12)
        # Orientation.
        @test isapprox(prior(pe1, permutedims(W6); dims = 2).mu, pr.mu; atol = 1e-12)
        # Too few rows for the regressor covariance.
        @test_throws ArgumentError prior(pe1, W6[1:2, :])
        # Factor returns and a panel are ignored.
        @test isapprox(prior(pe1, W6, W6[:, 1:1]).mu, pr.mu; atol = 1e-12)
    end

    @testset "The prior on two single-asset subsystems" begin
        # Two subsystems of one asset each, drawn as the estimator draws them, so the kernel
        # weights differ and equations 6, 7 and 10 are exercised by hand.
        W6 = R[1:6, 1:3]
        P = 1 .+ W6
        Y = P[2:end, :]
        Z = P[1:(end - 1), :]
        z = P[end, :]
        w = 5
        sigma = 0.025
        pe2 = LowDimensionEnsemblePrior(; n_subsystems = 2, subsystem_size = 1,
                                        sigma = sigma, pdm = nothing, seed = 7)
        drng = po.resolve_rng(pe2.rng, 7)
        idxs = [StatsBase.sample(drng, 1:3, 1; replace = false) for _ in 1:2]
        Bs = [Z[:, idx] \ Y for idx in idxs]
        Rl = [vec(sum(abs2, Y .- Z[:, idxs[l]] * Bs[l]; dims = 1)) ./ w for l in 1:2]
        Fl = [vec(Bs[l]' * z[idxs[l]]) for l in 1:2]
        V = [exp.(-Rl[l] ./ sigma^2) for l in 1:2]
        Vs = V[1] .+ V[2]
        xhat = (V[1] .* Fl[1] .+ V[2] .* Fl[2]) ./ Vs
        Sl = [Bs[l]' * cov(Y[:, idxs[l]]) * Bs[l] for l in 1:2]
        num = (V[1] * V[1]') .* Sl[1] .+ (V[2] * V[2]') .* Sl[2]
        den = V[1] * V[1]' .+ V[2] * V[2]'
        pr = prior(pe2, W6)
        @test isapprox(pr.mu, xhat .- 1; atol = 1e-12)
        @test isapprox(pr.sigma, num ./ den; atol = 1e-12)
        # The two weights are not equal, so the hand check read the kernel.
        @test !isapprox(V[1] ./ Vs, fill(0.5, 3); atol = 1e-3)
        # A seed draws the same subsystems at every fit; no seed re-draws.
        @test prior(pe2, W6).mu == pr.mu
        pe_free = LowDimensionEnsemblePrior(; n_subsystems = 2, subsystem_size = 1,
                                            pdm = nothing, rng = StableRNG(3))
        @test prior(pe_free, W6).mu != prior(pe_free, W6).mu
    end

    @testset "The prior against the paper's equations, loop by loop" begin
        # Equations 4 to 10 written out asset by asset and subsystem by subsystem, with the
        # pseudo-inverse for the coefficients and a BigFloat kernel that does not underflow.
        function paper(X, idxs, sigma)
            P = 1 .+ X
            nt, na = size(P)
            w = nt - 1
            nl = length(idxs)
            Rk = zeros(nl, na)
            Fk = zeros(nl, na)
            Sl = Vector{Matrix{Float64}}(undef, nl)
            for l in 1:nl
                A = idxs[l]
                Xl = P[1:(end - 1), A]
                B = zeros(na, length(A))
                for k in 1:na
                    y = P[2:end, k]
                    b = pinv(Xl) * y
                    B[k, :] = b
                    Rk[l, k] = sum(abs2, y .- Xl * b) / w
                    Fk[l, k] = dot(P[end, A], b)
                end
                Zl = P[2:end, A]
                zb = vec(sum(Zl; dims = 1)) ./ w
                S = [sum((Zl[:, p] .- zb[p]) .* (Zl[:, q] .- zb[q])) / (w - 1)
                     for p in eachindex(A), q in eachindex(A)]
                Sl[l] = B * S * B'
            end
            V = exp.(-big.(Rk) ./ big(sigma)^2)
            V ./= sum(V; dims = 1)
            xhat = vec(sum(V .* Fk; dims = 1))
            num = sum(l -> (V[l, :] * V[l, :]') .* Sl[l], 1:nl)
            return Float64.(xhat .- 1), Float64.(num ./ (V' * V)), Float64.(V)
        end
        draw(pe, N, s) = (drng = po.resolve_rng(pe.rng, pe.seed);
                          [StatsBase.sample(drng, 1:N, s; replace = false)
                           for _ in 1:(pe.n_subsystems)])
        R6 = 0.02 .* randn(StableRNG(11), 40, 6)
        # The paper's settings: L = 300, s = 3, sigma = 0.025, w = 5.
        for (seed, rws) in ((1, 1:6), (2, 10:15), (3, 30:35), (4, 1:12))
            pe = LowDimensionEnsemblePrior(; seed = seed, pdm = nothing)
            mu, S, _ = paper(R6[rws, :], draw(pe, 6, 3), 0.025)
            pr = prior(pe, R6[rws, :])
            @test isapprox(pr.mu, mu; atol = 1e-14)
            @test isapprox(pr.sigma, S; atol = 1e-16)
            # The paper's window of five pairs gives an indefinite aggregate here.
            length(rws) == 6 && @test eigmin(Symmetric(pr.sigma)) < 0
        end
        # Errors far above the bandwidth. The product of two assets' Float64 weights
        # underflows to zero on every subsystem for some pairs, and the pair's own shift
        # still gives the paper's ratio there.
        pe = LowDimensionEnsemblePrior(; n_subsystems = 40, subsystem_size = 2, seed = 3,
                                       sigma = 1e-3, pdm = nothing)
        Xb = 5 .* R6[1:6, :]
        mu, S, V = paper(Xb, draw(pe, 6, 2), 1e-3)
        pr = prior(pe, Xb)
        z = iszero.(V' * V)
        @test count(z) == 24
        @test maximum(abs, S[z]) > 0.01
        @test isapprox(pr.mu, mu; atol = 1e-14)
        @test isapprox(pr.sigma, S; atol = 1e-15)
        # Fewer regression pairs than the subsystem has assets: the least-norm coefficients.
        pe = LowDimensionEnsemblePrior(; n_subsystems = 20, subsystem_size = 3, seed = 9,
                                       pdm = nothing)
        mu, S, _ = paper(R6[1:3, :], draw(pe, 6, 3), 0.025)
        pr = prior(pe, R6[1:3, :])
        @test isapprox(pr.mu, mu; atol = 1e-13)
        @test isapprox(pr.sigma, S; atol = 1e-13)
        # The floor reaches the rule through a field that holds a vector of estimators.
        opt = JuMPOptimiser(; slv = slv)
        rv = Variance(; sigma = LowDimensionEnsemblePrior())
        @test po.fit_min_rows(MeanRisk(; r = rv, opt = opt)) == 3
        @test po.fit_min_rows(MeanRisk(; r = [rv, ConditionalValueatRisk()], opt = opt)) ==
              3
        @test po.fit_min_rows([EmpiricalPrior(), LowDimensionEnsemblePrior()]) == 3
        @test po.fit_min_rows([1.0, 2.0]) == 1
    end

    @testset "The repair and the default" begin
        # The aggregate under entry-wise weights need not be positive semidefinite; the
        # default repairs it and `nothing` leaves it.
        W = R[1:6, :]
        pe = LowDimensionEnsemblePrior(; n_subsystems = 40, subsystem_size = 2, seed = 3)
        pr = prior(pe, W)
        @test isposdef(pr.sigma)
        @test issymmetric(pr.sigma)
        pe0 = LowDimensionEnsemblePrior(; n_subsystems = 40, subsystem_size = 2, seed = 3,
                                        pdm = nothing)
        pr0 = prior(pe0, W)
        @test pr0.mu == pr.mu
        @test issymmetric(pr0.sigma)
        # Errors far above the bandwidth: the shift keeps the weights finite and summing to
        # one, where the unshifted kernel underflows every weight.
        Wbig = 5 .* W
        prb = prior(LowDimensionEnsemblePrior(; n_subsystems = 40, subsystem_size = 2,
                                              seed = 3, sigma = 1e-3), Wbig)
        @test all(isfinite, prb.mu) && all(isfinite, prb.sigma)
        # The prior serves an expected-returns slot through the library-wide adapter.
        me = PriorExpectedReturns(; pe = pe)
        @test isapprox(vec(mean(me, W)), pr.mu; atol = 1e-12)
    end

    @testset "The programme against the paper's solver" begin
        # The paper's coordinate-wise descent (equations 15 to 23): relax `b >= 0`, descend
        # on `c = b - w` under a Lagrange multiplier started at `1.1 xi` and updated from
        # the active set, then project onto the simplex. Run to a tighter stop than the
        # paper's `0.01` so the comparison reads the method, not its stopping rule.
        function paper_step(xhat, S, w, gamma, xi; iters = 2000, tol = 1e-12)
            d = length(w)
            z = 2 * gamma * (S * w) .- xhat
            c = fill(1 / d, d) .- w
            lam = 1.1 * xi
            f(c) = gamma * dot(c, S, c) + dot(z, c) + xi * sum(abs, c)
            fo = f(c)
            for _ in 1:iters
                for i in 1:d
                    off = 2 * gamma * (dot(S[i, :], c) - S[i, i] * c[i])
                    if c[i] > 0
                        c[i] = (lam - xi - z[i] - off) / (2 * gamma * S[i, i])
                    elseif c[i] < 0
                        c[i] = (lam + xi - z[i] - off) / (2 * gamma * S[i, i])
                    end
                end
                act = findall(!iszero, c)
                if !isempty(act)
                    inv = [1 / (2 * gamma * S[i, i]) for i in act]
                    offs = [(z[i] + 2 * gamma * (dot(S[i, :], c) - S[i, i] * c[i])) /
                            (2 * gamma * S[i, i]) for i in act]
                    sgn = [c[i] > 0 ? 1 : -1 for i in act]
                    lam = (sum(offs) - xi * dot(sgn, inv)) / sum(inv)
                end
                fn = f(c)
                if abs(fn - fo) <= tol
                    break
                end
                fo = fn
            end
            return simplex(c .+ w)
        end
        gamma, xi = 0.25, 0.002
        pe = LowDimensionEnsemblePrior(; seed = 5)
        alg = LowDimensionEnsemblePortfolio(; N = N, gamma = gamma, xi = xi, slv = slv,
                                            pe = pe)
        W6 = R[10:15, :]
        rd6 = rows(rd, 10:15)
        pr = prior(pe, W6)
        xhat = 1 .+ pr.mu
        S = pr.sigma
        # A reference allocation away from uniform, so the fee binds.
        wref = [0.5, 0.3, 0.1, 0.1]
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        b = po.leader_allocation(alg.opt, rd6, wref, set, rd6)
        @test isapprox(sum(b), 1; atol = 1e-8)
        @test minimum(b) >= -1e-8
        obj(b) = -dot(xhat, b) + gamma * dot(b, S, b) + xi * norm(b .- wref, 1)
        bp = paper_step(xhat, S, wref, gamma, xi)
        # The programme minimises the stated problem; the relax-then-project answer is
        # feasible for it, so it is no better.
        @test obj(b) <= obj(bp) + 1e-9
        # The tolerance the relaxation costs on this window: the two answers agree to
        # 0.02 in the one-norm.
        @test norm(b .- bp, 1) <= 0.02
        # The fee is the paper's penalty against the reference: with it the answer trades
        # less from `wref` than without it.
        alg0 = LowDimensionEnsemblePortfolio(; N = N, gamma = gamma, xi = 0, slv = slv,
                                             pe = pe)
        b0 = po.leader_allocation(alg0.opt, rd6, wref, set, rd6)
        @test norm(b .- wref, 1) <= norm(b0 .- wref, 1) + 1e-8
        # The variance term pulls towards the variance-minimising direction: at a large
        # risk aversion the answer moves away from the argmax.
        algg = LowDimensionEnsemblePortfolio(; N = N, gamma = 500, xi = 0, slv = slv,
                                             pe = pe)
        bg = po.leader_allocation(algg.opt, rd6, wref, set, rd6)
        @test obj(bg) >= obj(b0) - 1e-9
        @test !isapprox(bg, b0; atol = 1e-3)
        # Without fee and variance the programme is the one-hot argmax of the forecast.
        alg00 = LowDimensionEnsemblePortfolio(; N = N, gamma = 0, xi = 0, slv = slv,
                                              pe = pe)
        b00 = po.leader_allocation(alg00.opt, rd6, wref, set, rd6)
        onehot = zeros(N)
        onehot[argmax(xhat)] = 1
        @test isapprox(b00, onehot; atol = 1e-6)
        # The prior reduces to the Coverage Universe of its window and expands both moments
        # with NaN outside it, as every prior does (ADR 0117): an asset with a gap anywhere
        # in the window is left out, and the fit over the rest is the fit on the reduced
        # carrier, draw for draw.
        Rn = copy(R[1:12, :])
        Rn[1:3, 4] .= NaN
        prn = prior(pe, ReturnsResult(; nx = nx, X = Rn))
        @test isnan(prn.mu[4]) && all(isfinite, prn.mu[1:3])
        @test all(isnan, prn.sigma[4, :]) && all(isnan, prn.sigma[:, 4])
        @test all(isfinite, prn.sigma[1:3, 1:3])
        @test po.investable_mask(prn) == [true, true, true, false]
        pr3 = prior(pe, ReturnsResult(; nx = nx[1:3], X = Rn[:, 1:3]))
        @test prn.mu[1:3] == pr3.mu && prn.sigma[1:3, 1:3] == pr3.sigma
        @test size(prn.X) == (12, 4) && isnan(prn.X[1, 4])
    end

    @testset "The head" begin
        pe = LowDimensionEnsemblePrior(; n_subsystems = 60, seed = 5)
        alg = LowDimensionEnsemblePortfolio(; N = N, slv = slv, pe = pe)
        # Below three rows the rule answers the uniform portfolio; from three it solves.
        @test optimise(OPS(; alg = alg), rows(rd, 1:1)).w == fill(0.25, N)
        @test optimise(OPS(; alg = alg), rows(rd, 1:2)).w == fill(0.25, N)
        w3 = optimise(OPS(; alg = alg), rows(rd, 1:3)).w
        @test isapprox(sum(w3), 1; atol = 1e-8) && !isapprox(w3, fill(0.25, N); atol = 1e-3)
        res = optimise(OPS(; alg = alg), rd)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(sum(res.w), 1; atol = 1e-8)
        @test minimum(res.w) >= -1e-8
        # The last update re-solves on the last six rows with the Price-Adjusted Allocation
        # of the previous target as the fee's reference.
        prev = optimise(OPS(; alg = alg), rows(rd, 1:(T - 1))).w
        X = 1 .+ R
        wh = prev .* X[T, :] ./ dot(prev, X[T, :])
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        bhand = po.leader_allocation(alg.opt, rows(rd, (T - 5):T), wh, set, rd)
        @test isapprox(res.w, bhand; atol = 1e-6)
        # The head's set enters the programme.
        wcap = optimise(OPS(; alg = alg,
                            set = ProgrammeAllocationSet(; slv = slv,
                                                         wb = WeightBounds(0, 0.5))),
                        rows(rd, 1:10)).w
        @test maximum(wcap) <= 0.5 + 1e-6
        @test isapprox(sum(wcap), 1; atol = 1e-8)
        # The rule and its prior view onto the investable assets.
        v = po.port_opt_view(alg, [1, 3], R)
        @test isa(v, FollowTheLeader)
        @test length(v.opt.opt.fees.tn.w) == 2
        @test v.opt.opt.pe === alg.opt.opt.pe
    end

    @testset "Docstrings" begin
        docs = [string(@doc(LowDimensionEnsemblePrior)),
                string(@doc(LowDimensionEnsemblePortfolio)), string(@doc(po.fit_min_rows)),
                string(@doc(po.ensemble_programme))]
        @test all(d -> !occursin("No documentation found", d), docs)
        @test occursin("Forecasts the next price relative", docs[1])
    end
end
