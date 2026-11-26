#=
An asset that the panel marks inactive at a row of the online recursion (issue #2346). The step
of every rule leaves the asset out, and the book holds a zero there. The reference of the first
blocks is the same rule over the other assets, a run that never meets the reduction.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates, Clarabel,
      JuMP, Logging

# A forecaster with stated expected returns, and a loss that prices a part of the assets.
struct StatedMean <: PortfolioOptimisers.AbstractExpectedReturnsEstimator
    mu::Vector{Float64}
end
function Statistics.mean(me::StatedMean, ::PortfolioOptimisers.MatNum,
                         ::PortfolioOptimisers.Option{<:AssetPanel}; kwargs...)
    return me.mu
end
PortfolioOptimisers.rows_needed(::StatedMean) = 1
struct StatedLoss <: PortfolioOptimisers.AbstractOnlineObjective
    g::Vector{Float64}
end
function PortfolioOptimisers.loss_gradient(obj::StatedLoss, ::AbstractVector,
                                           ::AbstractVector, ::Any)
    return copy(obj.g)
end
PortfolioOptimisers.rows_needed(::StatedLoss) = 0

@testset "Online portfolio selection: an asset that is inactive at the row" begin
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection
    maxerr(a, b) = maximum(abs, a .- b)
    quiet(f) = with_logger(f, NullLogger())
    resolve(set, n) = po.resolve_allocation_set(set, n, false, Float64)
    rows(r, i) = po.port_opt_view(r, i, :)
    success(o) = isa(optimise(o).retcode, OptimisationSuccess)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                 "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12),
                 check_sol = (; allow_local = true, allow_almost = true))
    function books(alg, r, n; kwargs...)
        o = OPS(; alg = alg, kwargs...)
        out = Vector{Float64}[]
        for t in 1:n
            o = po.partial_fit!(o, rows(r, t:t))
            push!(out, copy(o.cache.w))
        end
        return out, o
    end
    # The fixture of the issue: the eighth asset is inactive at every row.
    T, N = 300, 8
    X = 0.01 .* randn(StableRNG(1), T, N)
    am = trues(T, N)
    am[:, N] .= false
    X[.!am] .= NaN
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    nx = string.("A", 1:N)
    panel(m) = AssetPanel(; amsk = m, emsk = copy(m))
    rd = ReturnsResult(; nx = nx, X = X, ts = ts, pnl = panel(am))
    pd = ReturnsResult(; nx = nx[1:7], X = X[:, 1:7], ts = ts, pnl = panel(am[:, 1:7]))
    rates = [0.05, 5.0]
    mix = [ExponentiatedGradient(), GradientProjection(; eta = 1e4), BuyAndHold()]
    roster = [BuyAndHold(), ConstantRebalancedPortfolio(), NewtonStep(; eta = 0.1),
              NewtonStep(; proj = GramProjection(; slv = slv)),
              PassiveAggressiveMeanReversion(; eps = 0.9, slack = QuadraticSlack()),
              ExponentiatedGradient(), ExponentiatedGradient(; eta = 10.0, alpha = 0.1),
              GradientProjection(; eta = 1e5), MirrorDescent(; proj = TsallisProjection()),
              MirrorDescent(; proj = LogBarrierProjection()), EGE(), EGR(; gamma2 = 0.9),
              EGA(; gamma2 = 0.9), MAEG(; etas = rates, window = 5), AEG(; etas = rates),
              ExponentiatedGradient(; eta = SelfConfidentRate()),
              ExponentiatedGradient(; eta = DoublingTrickRate(; N = 3)),
              GradientProjection(; eta = 1.0, predictor = LastGradient()),
              GradientProjection(; eta = DoublingTrickRate(; N = 3),
                                 predictor = LastGradient(; at_played = false)),
              ExponentiatedGradient(; eta = HintResidualRate(), alpha = 0.1,
                                    predictor = MeanGradient()), AdaptiveSubgradient(),
              ConfidenceWeightedMeanReversion(),
              ConfidenceWeightedMeanReversion(; eps = 0.99,
                                              formulation = StandardDeviationUpdate()),
              AntiCorrelation(; window = 3), ExpectationMaximisation(),
              ExpectationMaximisation(; eta = InverseSquareRootRate(; c = 0.5)),
              ExpectationMaximisation(; eta = DoublingTrickRate(; N = 3)),
              AggregatingAlgorithm(; eta = 2), TopK(; k = 2), WeakAggregatingAlgorithm(),
              SwitchingWeighting(), SwitchingWeighting(; gamma = 0),
              ExpertMixture(; experts = mix, grad = BlendPoint()),
              ExpertMixture(; experts = mix, alg = TopK(; k = 1)),
              Ader(; eta_min = 0.1, K = 3, eps = 0.1), FollowTheLeadingHistory(),
              FollowTheLeader(),
              FollowTheLeader(; sel = KernelMatch(; window = 2, radius = 0.05),
                              gamma = 0.1),
              FollowTheLeader(; sel = CorrelationMatch(; window = 2)),
              FollowTheLeader(; sel = LastRows(; W = 10),
                              opt = MeanRisk(; opt = JuMPOptimiser(; slv = slv))),
              MovingAverageReversion()]

    @testset "The book of a rule is its book over the other assets" begin
        # The call of the issue, at the row where every priced asset fell.
        res = optimise(OPS(; alg = GradientProjection(; eta = 1e5)), rows(rd, 1:242))
        @test isa(res.retcode, OptimisationSuccess)
        @test res.w[N] == 0 && sum(res.w) == 1
        # Measured over the roster: 1e-5 under the Gram programme, the accuracy of its solver,
        # and 3e-11 elsewhere. The two runs of the leader on a `MeanRisk` give its solver the
        # same inputs here, but inputs that differ by a rounding move its books by 1.3e-5, and
        # the CI run of #2358 measured 1.2e-5. So each rule that solves takes the accuracy of
        # its solver.
        for (k, alg) in enumerate(roster)
            @testset "$(k). $(nameof(typeof(alg)))" begin
                n = isa(alg, FollowTheLeader) || isa(alg, NewtonStep) ? 40 : 120
                (w8, o), (w7, _) = quiet() do
                    return books(alg, rd, n), books(alg, pd, n)
                end
                solves = (isa(alg, NewtonStep) && isa(alg.proj, GramProjection)) ||
                         (isa(alg, FollowTheLeader) && isa(alg.opt, MeanRisk))
                tol = solves ? 1e-4 : 1e-9
                @test maximum(t -> maxerr(w8[t][1:7], w7[t]), 1:n) < tol
                @test all(w -> iszero(w[N]), w8)
                @test success(o)
            end
        end
        # The default rule of the family is the exponentiated gradient over the listed names.
        full = optimise(OPS(), rows(pd, 1:5)).w
        @test maxerr(optimise(OPS(), rows(rd, 1:5)).w[1:7], full) < 1e-15
    end

    # Listings and delistings in blocks of ten rows. The first asset is active at every row.
    Tr = 60
    amr = trues(Tr, N)
    rng = StableRNG(9)
    for j in 2:N, b in 1:6
        if rand(rng) < 0.35
            amr[((b - 1) * 10 + 1):(b * 10), j] .= false
        end
    end
    Xr = 0.01 .* randn(StableRNG(4), Tr, N)
    Xr[.!amr] .= NaN
    rdr = ReturnsResult(; nx = nx, X = Xr, ts = ts[1:Tr], pnl = panel(amr))
    Xs = copy(Xr)
    Xs[.!amr] .= 0.5
    rds = ReturnsResult(; nx = nx, X = Xs, ts = ts[1:Tr], pnl = panel(amr))

    @testset "Listings and delistings, and a stale finite return" begin
        cap = BoundedAllocationSet(; wb = WeightBounds(; lb = 0.0, ub = 0.6))
        for alg in roster
            n = isa(alg, FollowTheLeader) ? 35 : Tr
            (wr, o), (ws, _), (wc, oc) = quiet() do
                return books(alg, rdr, n), books(alg, rds, n), books(alg, rdr, n;
                                                                     set = cap)
            end
            @test all(t -> all(iszero, wr[t][.!amr[t, :]]), 1:n)
            @test all(w -> sum(w) ≈ 1, wr) && success(o)
            # The return of an inactive asset is not read. The gap of a window that spans a
            # listing is the one of the rows, so the kernels over the rows read it.
            if !isa(alg, Union{AntiCorrelation, FollowTheLeader})
                @test maximum(t -> maxerr(wr[t], ws[t]), 1:n) < 1e-12
            end
            # The bounds hold on the book of the fund, with no renormalisation after.
            @test all(t -> all(iszero, wc[t][.!amr[t, :]]), 1:n)
            @test maximum(maximum, wc) < 0.6 + 1e-8 && success(oc)
        end
    end

    @testset "A set that reads no rows, on a head that keeps none" begin
        cap = WeightBounds(; lb = 0.0, ub = 0.3)
        plain = ProgrammeAllocationSet(; slv = slv, wb = cap)
        for alg in (BuyAndHold(), GradientProjection(; eta = 5),
                    ExpectationMaximisation(; eta = DoublingTrickRate(; N = 2)),
                    ExpertMixture(; experts = mix[1:2]))
            a = optimise(OPS(; alg = alg, set = plain), rows(rd, 1:12))
            b = optimise(OPS(; alg = alg, set = BoundedAllocationSet(; wb = cap)),
                         rows(rd, 1:12))
            @test isa(a.retcode, OptimisationSuccess) && a.w[N] == 0
            @test maxerr(a.w, b.w) < 1e-4
        end
        # A row that names the inactive asset drops, and the rest of it binds.
        sets = UniverseSets(; dict = Dict("nx" => nx))
        lc = LinearConstraintEstimator(; val = ["A1 + A8 <= 0.1"])
        named = ProgrammeAllocationSet(; slv = slv, lcse = lc, sets = sets)
        imsk = BitVector([trues(7); false])
        setr, Xn, pr = po.programme_investable_reduction(imsk, named, nothing, nothing)
        @test isnothing(Xn) && isnothing(pr)
        @test setr.sets.dict["ni"] == ["A8"] && setr.sets.dict["nx"] == nx[1:7]
        w = optimise(OPS(; alg = GradientProjection(; eta = 5), set = named, strict = true),
                     rows(rd, 1:12))
        @test isa(w.retcode, OptimisationSuccess) && w.w[N] == 0
        @test sum(w.w) ≈ 1
        # A set with no names takes the mask too.
        setp, _, _ = po.programme_investable_reduction(imsk, plain, nothing, nothing)
        @test isnothing(setp.sets)
    end

    @testset "An asset that lists again" begin
        al = trues(30, 4)
        al[11:20, 4] .= false
        Xl = 0.01 .* randn(StableRNG(2), 30, 4)
        Xl[.!al] .= NaN
        rl = ReturnsResult(; nx = nx[1:4], X = Xl, ts = ts[1:30], pnl = panel(al))
        r3 = ReturnsResult(; nx = nx[1:3], X = Xl[:, 1:3], ts = ts[1:30])
        # A rule that multiplies its allocation keeps the weight of the asset on its iterate.
        for alg in (BuyAndHold(), AggregatingAlgorithm(), SwitchingWeighting(; gamma = 0),
                    ExpectationMaximisation(), ExponentiatedGradient())
            w, o = books(alg, rl, 10)
            u4 = o.cache.w[4]
            o = po.partial_fit!(o, rows(rl, 11:20))
            @test o.cache.w[4] == 0 && o.cache.st.u[4] == u4
            @test sum(o.cache.st.u) ≈ 1
            o = po.partial_fit!(o, rows(rl, 21:21))
            @test isapprox(o.cache.w[4], u4; rtol = 0.05)
        end
        # The weight of the buy-and-hold at the listing: its parked share, drifted one row.
        w, o = books(BuyAndHold(), rl, 20)
        u = copy(o.cache.st.u)
        v = u .* (1 .+ Xl[21, :])
        @test books(BuyAndHold(), rl, 21)[1][21] ≈ v ./ sum(v)
        # Over the other assets it does not trade while the asset is out.
        x3 = vec(prod(1 .+ Xl[12:20, 1:3]; dims = 1))
        b = w[11][1:3] .* x3
        @test w[20][1:3] ≈ b ./ sum(b)
        # A constant rebalanced portfolio holds its target, cut to the active assets. A
        # target with no mass on them is the uniform allocation over them.
        crp, _ = books(ConstantRebalancedPortfolio(; w = [0.1, 0.2, 0.3, 0.4]), rl, 21)
        @test crp[10] ≈ [0.1, 0.2, 0.3, 0.4] ≈ crp[21]
        @test crp[11] ≈ [1 / 6, 1 / 3, 1 / 2, 0]
        one4, _ = books(ConstantRebalancedPortfolio(; w = [0, 0, 0, 1]), rl, 11)
        @test one4[10] == [0, 0, 0, 1] && one4[11] ≈ [1 / 3, 1 / 3, 1 / 3, 0]
        # An additive step starts from the zero of the book.
        _, o = books(PassiveAggressiveMeanReversion(), rl, 20)
        xn = 1 .+ Xl[21, :]
        dev = xn .- mean(xn)
        tau = (dot(o.cache.w, xn) - 0.5) / sum(abs2, dev)
        @test books(PassiveAggressiveMeanReversion(), rl, 21)[1][21] ≈
              po.project_simplex(o.cache.w .- tau .* dev)
        # The states keep the entry of the asset.
        s10, s20 = books(ConfidenceWeightedMeanReversion(), rl, 10)[2].cache.st,
                   books(ConfidenceWeightedMeanReversion(), rl, 20)[2].cache.st
        @test s20.sigma[4] == s10.sigma[4] && sum(s20.sigma) ≈ 1 / 4
        s10, s20 = books(NewtonStep(), rl, 10)[2].cache.st,
                   books(NewtonStep(), rl, 20)[2].cache.st
        @test s20.A[4, :] == s10.A[4, :] && s20.b[4] == s10.b[4] && s20.n == 20
        s10, s20 = books(TopK(; k = 4), rl, 10)[2].cache.st,
                   books(TopK(; k = 4), rl, 20)[2].cache.st
        @test s20.G[4] == s10.G[4] && s20.G[1:3] ≈ vec(sum(log1p.(Xl[1:20, 1:3]); dims = 1))
        # Fewer active assets than `k`: the rule holds them all.
        @test count(!iszero, books(TopK(; k = 4), rl, 20)[1][20]) == 3
        s10, s20 = books(MAEG(; etas = rates, window = 5), rl, 10)[2].cache.st.s,
                   books(MAEG(; etas = rates, window = 5), rl, 20)[2].cache.st.s
        @test s20.B[4, :] == s10.B[4, :] && all(≈(1), sum(s20.B; dims = 1))
        # The self-confident rate reads the number of active assets.
        sc = ExponentiatedGradient(; eta = SelfConfidentRate())
        @test books(sc, rl, 20)[2].cache.st.s[2] == log(3)
        @test books(sc, rl, 21)[2].cache.st.s[2] == log(4)
        # A copy of the state shares no array, and a view cuts the iterate.
        _, o = books(BuyAndHold(), rl, 15)
        c = copy(o.cache.st)
        @test c.u == o.cache.st.u && c.u !== o.cache.st.u && c.n == 15
        @test sum(po.port_opt_view(c, [1, 4]).u) ≈ 1
        @test_throws ArgumentError po.merge_states(c, c)
        @test optimise(po.partial_fit!(po.partial_fit(o, rows(rl, 16:16)), rows(rl, 17:30))).w ==
              optimise(po.partial_fit!(o, rows(rl, 16:30))).w
    end

    @testset "The leader holds no departed asset" begin
        # The fourth asset wins at every row of the sample and departs at the last row.
        ad = trues(8, 4)
        ad[8, 4] = false
        Xd = 0.01 .* randn(StableRNG(3), 8, 4)
        Xd[:, 4] .= 0.2
        Xd[.!ad] .= NaN
        rdd = ReturnsResult(; nx = nx[1:4], X = Xd, ts = ts[1:8], pnl = panel(ad))
        rd3 = ReturnsResult(; nx = nx[1:3], X = Xd[:, 1:3], ts = ts[1:8])
        sel = NearestNeighbourMatch(; window = 1, neighbours = FractionOf(0.5))
        mr = MeanRisk(; obj = MaximumReturn(), opt = JuMPOptimiser(; slv = slv))
        for opt in (BestConstantRebalancedPortfolio(), mr)
            alg = FollowTheLeader(; sel = sel, opt = opt)
            a, b = optimise(OPS(; alg = alg), rdd), optimise(OPS(; alg = alg), rd3)
            @test isa(a.retcode, OptimisationSuccess) && a.w[4] == 0
            @test maxerr(a.w[1:3], b.w) < 1e-8
        end
        # A held optimiser that names the departed asset drops the name under `strict`.
        sets = UniverseSets(; dict = Dict("nx" => nx[1:4]))
        lc = LinearConstraintEstimator(; val = ["A1 + A4 >= 0.4"])
        jo = JuMPOptimiser(; slv = slv, lcse = lc, sets = sets, strict = true)
        lead = FollowTheLeader(; sel = sel, opt = MeanRisk(; opt = jo))
        w = optimise(OPS(; alg = lead), rdd)
        @test isa(w.retcode, OptimisationSuccess) && w.w[4] == 0
        # A sample that is too small gives the uniform portfolio over the active assets.
        none = FollowTheLeader(; sel = KernelMatch(; window = 1, radius = 1e-9),
                               gamma = 0.5)
        _, o7 = books(none, rdd, 7)
        u3 = o7.cache.w[1:3] ./ sum(o7.cache.w[1:3])
        @test books(none, rdd, 8)[1][8] ≈ [(1 / 3 .+ u3) ./ 2; 0]
        # Bounds that do not reach the budget on the active assets hold the step.
        low = BoundedAllocationSet(; wb = WeightBounds(; lb = 0.0, ub = 0.3))
        hold = OPS(; alg = FollowTheLeader(; sel = LastRows(; W = 3)), set = low)
        @test_logs (:warn, r"the 3 assets that the step can price") match_mode = :any optimise(hold,
                                                                                               rdd)
    end

    @testset "A row with no active asset" begin
        ao = trues(6, 3)
        ao[4, :] .= false
        Xo = 0.01 .* randn(StableRNG(6), 6, 3)
        Xo[.!ao] .= NaN
        ro = ReturnsResult(; nx = nx[1:3], X = Xo, ts = ts[1:6], pnl = panel(ao))
        for alg in (BuyAndHold(), ExponentiatedGradient(), GradientProjection(; eta = 5))
            w, _ = books(alg, ro, 6)
            # The step reads one at every asset, so it moves nothing.
            @test w[4] ≈ w[3]
            @test isa(optimise(OPS(; alg = alg), rows(ro, 1:4)).retcode,
                      OptimisationFailure)
            @test isa(optimise(OPS(; alg = alg), ro).retcode, OptimisationSuccess)
        end
        # A book that a hold keeps on departed assets alone is a failure, and the fallback
        # answers it.
        ah = trues(3, 3)
        ah[3, 1] = false
        Xh = 0.01 .* randn(StableRNG(8), 3, 3)
        Xh[3, 1] = NaN
        rh = ReturnsResult(; nx = nx[1:3], X = Xh, ts = ts[1:3], pnl = panel(ah))
        set = BoundedAllocationSet(; wb = WeightBounds(; lb = 0.0, ub = [1.0, 0.4, 0.4]))
        held(fb) = quiet(() -> optimise(OPS(; alg = BuyAndHold(), set = set,
                                            w0 = [1.0, 0.0, 0.0], fb = fb), rh))
        @test isa(held(nothing).retcode, OptimisationFailure)
        @test isa(held(EqualWeighted()).retcode, OptimisationSuccess)
        # A residue of a rounding on the active assets is no mass. The CI run of #2358 left
        # about 2e-16 there, and the read-out renormalised it to a book above the caps.
        function residue_head(noise)
            o = OPS(; alg = BuyAndHold(), set = set, w0 = [1.0, 0.0, 0.0])
            o = po.partial_fit!(o, rows(rh, 1:2))
            o.cache.w[2:3] .= noise
            o.cache.st.u[2:3] .= noise
            return quiet(() -> po.partial_fit!(o, rows(rh, 3:3)))
        end
        @test isa(optimise(residue_head([2.06e-16, 2.14e-16])).retcode, OptimisationFailure)
        @test residue_head([2.06e-16, 2.14e-16]).cache.st.u[2:3] == [0.5, 0.5]
        # A real mass on the active assets is not enough when their bounds miss the budget:
        # the step holds the book of the departed asset, and no renormalisation of it lies in
        # the set.
        heldm(fb) = quiet(() -> optimise(OPS(; alg = BuyAndHold(), set = set,
                                             w0 = [0.8, 0.1, 0.1], fb = fb), rh))
        @test isa(heldm(nothing).retcode, OptimisationFailure)
        @test occursin("bounds", heldm(nothing).retcode.res)
        @test heldm(EqualWeighted()).w ≈ [0, 0.5, 0.5]
        # The same book under bounds that reach the budget reads out in proportion.
        free = BoundedAllocationSet(; wb = WeightBounds(; lb = 0.0, ub = [1.0, 0.6, 0.6]))
        rf = quiet(() -> optimise(OPS(; alg = BuyAndHold(), set = free,
                                      w0 = [0.8, 0.1, 0.1]), rh))
        @test isa(rf.retcode, OptimisationSuccess) && rf.w[1] == 0 && sum(rf.w) ≈ 1
    end

    @testset "The mask of the row meets the mask of a reader" begin
        # A forecaster that prices an inactive asset.
        fc = ForecastTracking(; me = StatedMean([0.01, 0.0, 0.02, 0.5]))
        Xm = 0.01 .* randn(StableRNG(7), 3, 4)
        Xm[:, 4] .= NaN
        amm = trues(3, 4)
        amm[:, 4] .= false
        rm = ReturnsResult(; nx = nx[1:4], X = Xm, ts = ts[1:3], pnl = panel(amm))
        w = optimise(OPS(; alg = fc), rm)
        @test isa(w.retcode, OptimisationSuccess) && w.w[4] == 0
        ref = OPS(; alg = ForecastTracking(; me = StatedMean([0.01, 0.0, 0.02])))
        @test maxerr(w.w[1:3],
                     optimise(ref, ReturnsResult(; nx = nx[1:3], X = Xm[:, 1:3])).w) < 1e-14
        # A forecaster that prices no active asset: the step is flat on them.
        x = [1.01, 0.99, 1.0, NaN]
        u = fill(0.25, 4)
        set4 = resolve(BoundedAllocationSet(), 4)
        none = ForecastTracking(; me = StatedMean([NaN, NaN, NaN, 0.5]))
        _, wn = po.online_update!(none, nothing, u, x, rm, set4)
        @test wn ≈ [1 / 3, 1 / 3, 1 / 3, 0]
        # A hint of the same forecaster moves nothing.
        hint = GradientProjection(; eta = 1.0,
                                  predictor = ForecastGradient(;
                                                               me = StatedMean([NaN, NaN,
                                                                                NaN, 0.5])),
                                  obj = StatedLoss([1.0, 0.0, 0.0, 0.0]))
        st, wp = po.online_update!(hint, po.rule_state_seed(hint, u), u, x, rm, set4)
        @test wp == [st.v[1:3] ./ sum(st.v[1:3]); 0] && st.v[4] == 0.25
        # A loss that prices no active asset holds the step on them, and one that prices a
        # part steps on that part.
        part(g) = begin
            alg = GradientProjection(; eta = 0.1, obj = StatedLoss(g))
            return po.online_update!(alg, po.rule_state_seed(alg, u), u, x, nothing, set4)
        end
        @test part([NaN, NaN, NaN, 1.0])[2] ≈ [1 / 3, 1 / 3, 1 / 3, 0]
        @test part([1.0, 0.0, NaN, -9.0])[2] ≈ [0.45, 0.55, 0, 0]
        @test part([1.0, 0.0, NaN, -9.0])[1].u ≈ [0.225, 0.275, 0.25, 0.25]
        # A Risk Loss reads the book of the active assets that its prior prices.
        g = po.loss_gradient(RiskLoss(; r = Variance(; sigma = Matrix(Diagonal(1.0:4)))), u,
                             x,
                             ReturnsResult(; nx = nx[1:4],
                                           X = 0.01 .* randn(StableRNG(5), 6, 4)))
        @test g[1:3] ≈ 2 .* (1:3) ./ 3 && isnan(g[4])
    end

    @testset "The helpers" begin
        r = [0.1, NaN, 0.2]
        @test po.row_relatives(r, nothing) == [1.1, 1.0, 1.2]
        @test isequal(po.row_relatives(r, [true, true, false]), [1.1, 1.0, NaN])
        @test po.row_relatives(r, falses(3)) == [1.1, 1.0, 1.2]
        @test eltype(po.row_relatives(Float32[0.1, 0.2], [true, false])) == Float32
        x = [1.1, NaN]
        @test po.cash_relative(x) == [1.1, 1.0]
        @test po.cash_relative(r[[1, 3]]) == r[[1, 3]]
        @test po.price_adjusted_allocation([0.5, 0.5], x) ≈ [0.55, 0.5] ./ 1.05
        w = [0.25, 0.75]
        @test po.renormalised_view(w, nothing) === w
        # A part whose mass is a rounding has no proportion, at any scale of the vector.
        @test po.below_rounding(4e-16, [1.0, 4e-16])
        @test !po.below_rounding(1e-14, [1.0, 1e-14])
        @test !po.below_rounding(1e-20, [1e-12, 1e-20])
        @test po.below_rounding(0, [1, 0]) &&
              !po.below_rounding(1 // 10^9, [1 // 1, 0 // 1])
        @test po.renormalised_view([1.0, 1e-16, 3e-16], [2, 3]) == [0.5, 0.5]
        @test po.renormalised_view([1.0, 1e-14, 3e-14], [2, 3]) ≈ [0.25, 0.75]
        @test po.renormalised_view([1e-12, 1e-20, 3e-20], [2, 3]) ≈ [0.25, 0.75]
        @test po.parked_iterate([1.0, 1e-16, 3e-16], [0.0, 0.2, 0.8],
                                BitVector([0, 1, 1])) == [0.0, 0.2, 0.8]
        imsk = BitVector([1, 0])
        @test po.investable_belief(w, nothing) === w
        @test po.investable_belief(w, imsk) == [1.0]
        b = copy(w)
        @test po.rate_step!(b, 0.5, x, imsk) == 1.1 && b == w
        @test isequal(po.loss_gradient(LogWealth(), w, x, nothing), [-1.0, NaN])
        @test isequal(po.mixed_relatives([1.2, 1.0, NaN], 0.2),
                      [0.9 * 1.2 + 0.1 * 1.2, 1.02, NaN])
        @test po.investable_gradient([NaN, 1.0], x) == ([0.0, 0.0], imsk)
        st = po.AllocationIterateState(0, w)
        wh = [0.3, 0.7]
        @test po.price_adjusted_iterate(st, w, wh, [1.0, 1.0], nothing) === wh
        @test po.price_adjusted_iterate(st, copy(w), wh, [1.0, 1.0], nothing) == w
        @test po.price_adjusted_iterate(st, w, wh, x, imsk) == [1.0, 0.0]
        # A blend at a mask projects on every set, because a held expert can be outside it.
        set2 = resolve(BoundedAllocationSet(), 2)
        e = EuclideanProjection()
        @test po.blend_projection(e, set2, w, wh, nothing) === w
        @test po.blend_projection(e, set2, w, wh, imsk) == [1.0, 0.0]
    end
end
