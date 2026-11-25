#=
An asset that a reader of the online step cannot price (issue #2339): a Risk Loss whose window
does not cover it, or a forecaster that gives no forecast for it. The step removes the asset,
as a batch optimiser does with an asset that its prior cannot price, and the book holds a zero
there. The reference of most blocks is the same rule over the priced assets alone, a run that
never meets the reduction.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates, Clarabel,
      JuMP, Logging

# A loss that prices the assets of a fixed mask alone, and a hint that prices others.
struct PartialLoss <: PortfolioOptimisers.AbstractOnlineObjective
    g::Vector{Float64}
end
function PortfolioOptimisers.loss_gradient(obj::PartialLoss, ::AbstractVector,
                                           ::AbstractVector, ::Any)
    return copy(obj.g)
end
PortfolioOptimisers.rows_needed(::PartialLoss) = 0
# A set of a user, which holds no bounds.
struct OwnSimplex <: PortfolioOptimisers.AbstractAllocationSet end
PortfolioOptimisers.port_opt_view(set::OwnSimplex, ::Any, args...) = set
function PortfolioOptimisers.project(::EuclideanProjection, ::OwnSimplex, q::AbstractVector,
                                     ::AbstractVector)
    return PortfolioOptimisers.project_simplex(q)
end
struct PartialHint <: PortfolioOptimisers.AbstractGradientPredictor
    m::Vector{Float64}
end
function PortfolioOptimisers.predict_gradient!(pred::PartialHint, ps,
                                               ::PortfolioOptimisers.AbstractOnlineObjective,
                                               ::AbstractVector, ::AbstractVector,
                                               ::AbstractVector, ::AbstractVector, ::Any,
                                               ::Integer)
    return ps, copy(pred.m)
end

@testset "Online portfolio selection: an asset that the step cannot price" begin
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection
    maxerr(a, b) = maximum(abs, a .- b)
    quiet(f) = with_logger(f, NullLogger())
    resolve(set, n) = po.resolve_allocation_set(set, n, false, Float64)
    rows(r, i) = po.port_opt_view(r, i, :)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                 "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12),
                 check_sol = (; allow_local = true, allow_almost = true))
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
    rdn = ReturnsResult(; nx = nx, X = X, ts = ts)
    pd = ReturnsResult(; nx = nx[1:7], X = X[:, 1:7], ts = ts, pnl = panel(am[:, 1:7]))
    risk = RiskLoss(; r = StandardDeviation(), window = 5)

    @testset "A step on a Risk Loss is the step over the priced assets" begin
        # The call of the issue: NaN weights before, and one on the first asset now.
        w = optimise(OPS(; alg = GradientProjection(; eta = 1e5, obj = risk)), rd)
        @test isa(w.retcode, OptimisationSuccess)
        @test w.w == [1.0; zeros(7)]
        # A stated covariance of the order of one, at the default rate. The minimiser over the
        # seven priced assets is the inverse-variance portfolio. The measure is cut to them.
        stated(n) = RiskLoss(; r = Variance(; sigma = Matrix(Diagonal(1.0:n))), window = 5)
        wm = optimise(OPS(; alg = GradientProjection(; obj = stated(N))), rd)
        iv = 1 ./ (1:7)
        @test wm.w[N] == 0 && maxerr(wm.w[1:7], iv ./ sum(iv)) < 1e-8
        algs = [GradientProjection(; eta = 1e5, obj = risk),
                GradientProjection(; eta = 0.05, obj = risk),
                GradientProjection(; eta = 1.0, obj = RiskLoss(; window = 5)),
                ExponentiatedGradient(; eta = 10.0, obj = risk),
                ExponentiatedGradient(; eta = 10.0, alpha = 0.1, obj = risk),
                MirrorDescent(; eta = 1.0, obj = risk, grad = GradientMomentum()),
                MirrorDescent(; eta = 0.1, obj = risk, grad = RootMeanSquareGradient()),
                MirrorDescent(; eta = 0.1, obj = risk,
                              grad = AdaptiveMomentGradient(; gamma2 = 0.9)),
                MirrorDescent(; eta = 10.0, proj = TsallisProjection(), obj = risk),
                MirrorDescent(; eta = 10.0, proj = LogBarrierProjection(), obj = risk),
                MirrorDescent(; eta = InverseSquareRootRate(), obj = risk),
                ExponentiatedGradient(; eta = DoublingTrickRate(; N = 7), obj = risk),
                GradientProjection(; eta = 1.0, obj = risk, predictor = LastGradient()),
                GradientProjection(; eta = 1.0, obj = risk, predictor = MeanGradient()),
                GradientProjection(; eta = 1e5, obj = risk,
                                   predictor = LastGradient(; at_played = false)),
                ExponentiatedGradient(; eta = DoublingTrickRate(; N = 7), obj = risk,
                                      alpha = 0.1, predictor = LastGradient()),
                AdaptiveSubgradient(; obj = risk),
                AdaptiveSubgradient(; delta = 0.1, obj = risk),
                ExpertMixture(;
                              experts = [GradientProjection(; obj = risk),
                                         ExponentiatedGradient(; obj = risk)],
                              grad = BlendPoint())]
        for alg in algs
            ref = optimise(OPS(; alg = alg), pd).w
            for r in (rd, rdn)
                a = quiet(() -> optimise(OPS(; alg = alg), r))
                @test isa(a.retcode, OptimisationSuccess)
                @test a.w[N] == 0
                @test maxerr(a.w[1:7], ref) < 1e-12
            end
            # Every prefix past the two rows that a covariance needs answers, and the book
            # of the head holds no weight there.
            o = OPS(; alg = alg)
            for t in 1:40
                o = po.partial_fit!(o, rows(rd, t:t))
                t < 2 && continue
                @test iszero(o.cache.w[N]) && sum(o.cache.w) ≈ 1
            end
        end
        # A Held Gap is a weight on a missing return, so a book with a zero there has none.
        warned = Test.TestLogger()
        with_logger(() -> optimise(OPS(; alg = GradientProjection(; obj = risk)),
                                   rows(rdn, 1:20)), warned)
        @test length(warned.logs) == 2
    end

    @testset "The iterate keeps the weight of a leg, and the leg comes back" begin
        Xl = 0.01 .* randn(StableRNG(2), 40, 4)
        al = trues(40, 4)
        al[1:20, 4] .= false
        Xl[.!al] .= NaN
        rl = ReturnsResult(; nx = nx[1:4], X = Xl, ts = ts[1:40], pnl = panel(al))
        o = OPS(; alg = ExponentiatedGradient(; eta = 10.0, obj = risk))
        s4 = Float64[]
        for t in 1:40
            o = po.partial_fit!(o, rows(rl, t:t))
            push!(s4, o.cache.w[4])
            @test sum(o.cache.st.u) ≈ 1
            2 <= t < 25 && @test o.cache.st.u[4] == 0.25
        end
        # The window of five rows clears the unlisted span at row 25. A multiplicative step
        # from a zero would hold it for ever.
        @test all(iszero, s4[2:24]) && all(>(0.1), s4[25:end])
        # The adaptive rule keeps no iterate, and its step buys the leg from the zero.
        oa = po.partial_fit!(OPS(; alg = AdaptiveSubgradient(; obj = risk)), rows(rl, 1:24))
        @test oa.cache.w[4] == 0 && oa.cache.st.s[4] == 0
        oa = po.partial_fit!(oa, rows(rl, 25:40))
        @test oa.cache.st.s[4] > 0 && sum(oa.cache.w) ≈ 1
        # An iterate with no mass on the priced legs gives them the whole mass.
        imsk = BitVector([1, 1, 0])
        v = [0.4, 0.6, 0.0]
        @test po.parked_iterate([0.0, 0.0, 1.0], v, imsk) == v
        @test po.parked_iterate([0.25, 0.25, 0.5], v, imsk) ≈ [0.2, 0.3, 0.5]
        @test po.parked_iterate([0.25, 0.25, 0.5], v, nothing) === v
        @test po.investable_book([0.0, 0.0, 1.0], imsk) == [0.5, 0.5, 0.0]
        @test po.investable_book(v, nothing) === v
        # A book on a leg that departs is sold, and the other legs take the whole mass.
        ad = trues(20, 4)
        ad[10:20, 1] .= false
        Xd = 0.01 .* randn(StableRNG(3), 20, 4)
        Xd[.!ad] .= NaN
        rdd = ReturnsResult(; nx = nx[1:4], X = Xd, ts = ts[1:20], pnl = panel(ad))
        od = optimise(OPS(; alg = GradientProjection(; obj = risk), w0 = [1.0, 0, 0, 0]),
                      rdd)
        @test isa(od.retcode, OptimisationSuccess) && od.w[1] == 0 && sum(od.w) ≈ 1
    end

    @testset "The mask of a gradient, of a hint and of a forecast" begin
        @test isnothing(po.investable_mask([1.0, Inf]))
        @test isnothing(po.investable_mask([NaN, NaN]))
        @test po.investable_mask([1.0, NaN]) == BitVector([1, 0])
        @test po.investable_gradient([1.0, NaN]) == ([1.0, 0.0], BitVector([1, 0]))
        g = [1.0, 2.0]
        @test po.investable_gradient(g)[1] === g && isnothing(po.investable_gradient(g)[2])
        a, b = BitVector([1, 1, 0]), BitVector([0, 1, 1])
        @test isnothing(po.investable_intersection(nothing, nothing))
        @test po.investable_intersection(nothing, b) === b
        @test po.investable_intersection(a, nothing) === a
        @test po.investable_intersection(a, b) == BitVector([0, 1, 0])
        @test po.forecast_where_defined([1.1, NaN, Inf])[1] == 1.1
        @test all(isnan, po.forecast_where_defined([1.1, NaN, Inf])[2:3])
        @test po.forecast_where_defined([NaN, NaN]) == [1.0, 1.0]
        @test po.played_allocation([0.5, 0.5, 0.0], 0.2, a) ≈ [0.5, 0.5, 0.0]
        # A loss that prices no asset states no universe, so its raw step is refused.
        set3 = resolve(BoundedAllocationSet(), 3)
        u = fill(1 / 3, 3)
        x = ones(3)
        none = GradientProjection(; obj = PartialLoss(fill(NaN, 3)))
        @test_throws DomainError po.online_update!(none, po.rule_state_seed(none, u), u, x,
                                                   nothing, set3)
        # The second half-step runs on the assets that the loss and the hint both price. The
        # first moves [1, 2] by the gradient [1, 0] at rate 0.1: [0.45, 0.55]. The hint then
        # prices the second and the third, so the book is the second alone.
        opti(m) = GradientProjection(; eta = 0.1, obj = PartialLoss([1.0, 0.0, NaN]),
                                     predictor = PartialHint(m))
        alg = opti([NaN, 1.0, 1.0])
        st, w = po.online_update!(alg, po.rule_state_seed(alg, u), u, x, nothing, set3)
        @test w == [0.0, 1.0, 0.0]
        @test st.v ≈ [0.45 * 2 / 3, 0.55 * 2 / 3, 1 / 3]
        # They share no asset: the hint moves nothing.
        alg = opti([NaN, NaN, 1.0])
        st, w = po.online_update!(alg, po.rule_state_seed(alg, u), u, x, nothing, set3)
        @test w ≈ [0.45, 0.55, 0.0] && st.u == w
        # The hint of a forecaster that prices a part of the universe.
        hint = GradientProjection(; eta = 1.0,
                                  predictor = ForecastGradient(;
                                                               me = PriceLevelExpectedReturns(;
                                                                                              alg = MovingAverage(;
                                                                                                                  window = 3))))
        ho = OPS(; alg = hint)
        for t in 1:10
            ho = po.partial_fit!(ho, rows(rd, t:t))
            t < 3 && continue
            @test ho.cache.w[N] == 0 && sum(ho.cache.w) ≈ 1 && all(isfinite, ho.cache.w)
        end
    end

    @testset "A forecast step is the step over the priced assets" begin
        algs = [MovingAverageReversion(), RobustMedianReversion(),
                ReweightedPriceRelativeTracking(), GaussianWeightingReversion(),
                ForecastReversion(; me = SimpleExpectedReturns()), PeakPriceTracking(),
                ForecastTracking(; me = SimpleExpectedReturns()),
                KernelTrendPatternTracking(), TransactionCostOptimisation(),
                ShortTermSparsePortfolio(), ShortTermSparsePortfolio(; alg = L1Optimum())]
        for alg in algs
            o8, o7 = OPS(; alg = alg), OPS(; alg = alg)
            quiet() do
                for t in 1:60
                    o8 = po.partial_fit!(o8, rows(rd, t:t))
                    o7 = po.partial_fit!(o7, rows(pd, t:t))
                end
            end
            r8, r7 = optimise(o8), optimise(o7)
            @test isa(r8.retcode, OptimisationSuccess)
            @test r8.w[N] == 0 && maxerr(r8.w[1:7], r7.w) < 1e-10
        end
        # The rows at which the whole book was on the asset with no forecast.
        for t in (9, 33)
            @test isa(optimise(OPS(; alg = MovingAverageReversion()), rows(rd, 1:t)).retcode,
                      OptimisationSuccess)
        end
    end

    @testset "The projection at a mask" begin
        imsk = BitVector([1, 0, 1, 1])
        q = [0.9, NaN, 0.2, 0.1]
        wh = [0.2, 0.4, 0.2, 0.2]
        simplex = resolve(BoundedAllocationSet(), 4)
        e = EuclideanProjection()
        @test po.project(e, simplex, [0.4, 0.3, 0.2, 0.1], wh, nothing) ==
              po.project(e, simplex, [0.4, 0.3, 0.2, 0.1], wh)
        wm = po.project(e, simplex, q, wh, imsk)
        @test wm[2] == 0 &&
              wm[[1, 3, 4]] ==
              po.project(e, resolve(BoundedAllocationSet(), 3), q[[1, 3, 4]], wh[[1, 3, 4]])
        # A set of a user takes the same reduction.
        @test po.project(e, OwnSimplex(), q, wh, imsk) == wm
        # The bounds bind on the assets of the mask.
        cap = WeightBounds(; lb = 0.0, ub = [0.5, 1.0, 0.5, 0.5])
        bset = resolve(BoundedAllocationSet(; wb = cap), 4)
        wc = po.project(e, bset, q, wh, imsk)
        @test wc ≈ [0.5, 0.0, 0.3, 0.2]
        # Bounds that do not reach the budget on the mask hold the step.
        low = resolve(BoundedAllocationSet(; wb = WeightBounds(; lb = 0.0, ub = 0.3)), 4)
        out, held = po.with_projection_step(() -> po.project(e, low, q, wh, imsk), nothing,
                                            1)
        @test out == wh && out !== wh && length(held) == 1
        @test occursin("the 3 assets that the step can price", only(held).reason)
        # A mirror-descent walk through such a step warns, and goes on.
        caps = OPS(; alg = GradientProjection(; obj = risk),
                   set = BoundedAllocationSet(; wb = WeightBounds(; lb = 0.0, ub = 0.14)))
        @test_logs (:warn, r"the 7 assets") match_mode = :any optimise(caps, rows(rd, 1:8))
        # A diagonal and a Gram geometry reduce with the raw step.
        h = [1.0, 2.0, 3.0, 4.0]
        @test po.project(po.DiagonalProjection(h), simplex, q, wh, imsk)[[1, 3, 4]] ==
              po.project(po.DiagonalProjection(h[[1, 3, 4]]),
                         resolve(BoundedAllocationSet(), 3), q[[1, 3, 4]], wh[[1, 3, 4]])
        qg = [0.9, 0.0, 0.2, 0.1]
        gr = po.project(GramProjection(; slv = slv, A = Matrix(Diagonal(h))), simplex, qg,
                        wh, imsk)
        @test gr[2] == 0 &&
              isapprox(gr, po.project(po.DiagonalProjection(h), simplex, q, wh, imsk);
                       atol = 1e-6)
        # A solve that does not answer holds the step over the full universe.
        bad = Solver(; name = :bad, solver = () -> error("refused"))
        hb, hs = quiet() do
            po.with_projection_step(() -> po.project(GramProjection(; slv = bad,
                                                                    A = Matrix(Diagonal(h))),
                                                     simplex, qg, wh, imsk), nothing, 1)
        end
        @test hb == wh && length(hs) == 1
        # The reprojection of a mix reads the bounds on the mask alone: a floor of 0.1 does
        # not bind on the second asset.
        floor = resolve(BoundedAllocationSet(; wb = WeightBounds(; lb = 0.1, ub = 1.0)), 4)
        qm = [0.4, 0.0, 0.3, 0.3]
        @test po.reprojection(e, floor, qm, wh, imsk) ≈ qm
        @test po.reprojection(e, floor, qm, wh) != qm
    end

    @testset "A programme set intersects its mask with the one of the rule" begin
        cap = WeightBounds(; lb = 0.0, ub = 0.3)
        plain = ProgrammeAllocationSet(; slv = slv, wb = cap)
        bounded = BoundedAllocationSet(; wb = cap)
        for alg in (GradientProjection(; eta = 50, obj = risk),
                    ExponentiatedGradient(; eta = 50, alpha = 0.1, obj = risk),
                    MovingAverageReversion())
            a = optimise(OPS(; alg = alg, set = plain), rows(rd, 1:20))
            b = optimise(OPS(; alg = alg, set = bounded), rows(rd, 1:20))
            @test isa(a.retcode, OptimisationSuccess) && a.w[N] == 0
            @test maxerr(a.w, b.w) < 1e-5
        end
        # A set that fits a prior: its rows start at row 1, so its plain prior leaves out an
        # asset that the window of the loss prices.
        Xf = copy(X[1:20, 1:4])
        af = trues(20, 4)
        af[1:3, 2] .= false
        af[:, 4] .= false
        Xf[.!af] .= NaN
        rf = ReturnsResult(; nx = nx[1:4], X = Xf, ts = ts[1:20], pnl = panel(af))
        fitted = ProgrammeAllocationSet(; slv = slv,
                                        r = Variance(;
                                                     settings = RiskMeasureSettings(;
                                                                                    ub = 1.0)))
        wf = optimise(OPS(; alg = GradientProjection(; eta = 50, obj = risk), set = fitted,
                          strict = true), rf)
        @test isa(wf.retcode, OptimisationSuccess)
        @test iszero(wf.w[2]) && iszero(wf.w[4]) && sum(wf.w) ≈ 1
        # The two masks share no asset.
        set4 = resolve(fitted, 4)
        wh = fill(0.25, 4)
        only2 = BitVector([0, 1, 0, 0])
        @test_throws IsEmptyError po.resolve_projection_entry(set4, wh, rf, only2)
        # One row fits no prior, so the projection holds.
        out, held = po.with_projection_step(() -> po.project(EuclideanProjection(), set4,
                                                             wh, wh,
                                                             BitVector([1, 1, 0, 0])),
                                            rows(rf, 4:4), 1)
        @test out == wh && length(held) == 1
        # A mask is read on rows, so a set with no rows refuses one.
        @test_throws IsNothingError po.resolve_projection_entry(resolve(plain, 4), wh,
                                                                nothing, only2)
    end
end
