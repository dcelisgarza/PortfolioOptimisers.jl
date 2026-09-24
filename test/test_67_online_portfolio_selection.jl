#=
The online portfolio selection head, its family type, the Rule State and the first algorithm
set, as ADRs 0155-0160 and 0162 rule them (issue #1161 on map #1148).

Parity is measured against the library's own prototype,
`research/prototypes/09_online_portfolio_selection.jl`, step for step: the library's answer
after rows `1:t-1` equals the prototype's `W[t, :]`, the row it holds during period `t`, and
the Causal Pass takes one more update than the prototype's driver, the Next-Period Allocation.
The prototype's module is named as the head is, so it is loaded into its own module here and
the head is reached through the package.
=#
module OPSPrototype
include(joinpath(@__DIR__, "..", "research", "prototypes",
                 "09_online_portfolio_selection.jl"))
end

@testset "Online portfolio selection: head, state, rules and the first set" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates
    po = PortfolioOptimisers
    PT = OPSPrototype.OnlinePortfolioSelection
    OPS = po.OnlinePortfolioSelection

    rng = StableRNG(7)
    T, N = 40, 4
    R = 0.02 .* randn(rng, T, N)
    X = 1 .+ R
    nx = ["A", "B", "C", "D"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = R, ts = ts)
    rows(r, i) = po.port_opt_view(r, i, :)

    # The library's path after t rows is the prototype's W[t + 1, :].
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

    @testset "Parity with the prototype, step for step" begin
        @test maxerr(libpath(ConstantRebalancedPortfolio()), PT.uniform_crp(X).weights) == 0
        @test maxerr(libpath(ExponentiatedGradient()),
                     PT.exponentiated_gradient(X).weights) < 1e-14
        @test maxerr(libpath(NewtonStep()), PT.online_newton_step(X).weights) < 1e-13
        # The uniform mix is applied to the raw Newton point before the projection, so a
        # bound of zero or a negative lower bound is honoured; the prototype mixes after,
        # so the parity is with the stated recursion, not the prototype.
        let A = Matrix{Float64}(I, N, N), b = zeros(N), W = zeros(T, N)
            W[1, :] .= 1 / N
            for t in 1:(T - 1)
                g = X[t, :] ./ dot(W[t, :], X[t, :])
                A .+= g * g'
                b .+= 2 .* g
                q = 0.125 .* (Symmetric(A) \ b)
                W[t + 1, :] .= po.project_simplex(0.9 .* q .+ 0.1 / N)
            end
            @test maxerr(libpath(NewtonStep(; eta = 0.1)), W) < 1e-13
        end
        @test maxerr(libpath(PassiveAggressiveMeanReversion()), PT.pamr(X).weights) < 1e-14
        @test maxerr(libpath(PassiveAggressiveMeanReversion(; slack = LinearSlack())),
                     PT.pamr(X; variant = :pamr1).weights) < 1e-14
        @test maxerr(libpath(PassiveAggressiveMeanReversion(; slack = QuadraticSlack())),
                     PT.pamr(X; variant = :pamr2).weights) < 1e-14
        # OLMAR and RMR read `1 .+ r` through the rows buffer, the levels reconstructed
        # with the last one at one; the prototype carries levels from one before the
        # first row. The window truncates over the first rows in both.
        @test maxerr(libpath(MovingAverageReversion(; window = 5)),
                     PT.olmar(X; window = 5).weights) < 1e-12
        @test maxerr(libpath(MovingAverageReversion(; window = 3, eps = 3)),
                     PT.olmar(X; window = 3, epsilon = 3).weights) < 1e-12
        @test maxerr(libpath(RobustMedianReversion(; window = 5, eps = 10)),
                     PT.rmr(X; window = 5, epsilon = 10).weights) < 1e-8
        # The universal portfolio is the mixture over the same sampled experts.
        rng2 = StableRNG(11)
        E = -log.(rand(rng2, Float64, 50, N))
        B = E ./ sum(E; dims = 2)
        mix = ExpertMixture(;
                            experts = [ConstantRebalancedPortfolio(; w = B[k, :])
                                       for k in 1:50])
        @test maxerr(libpath(mix),
                     PT.universal_portfolio(X; n_experts = 50, rng = StableRNG(11)).weights) <
              1e-14
        # The Causal Pass takes the one more update the prototype's driver stops short of.
        w_next = optimise(OPS(; alg = ExponentiatedGradient()), rd).w
        p = PT.exponentiated_gradient(X)
        w_last = p.weights[T, :]
        x_last = X[T, :]
        @test isapprox(w_next,
                       (w_last .* exp.(0.05 .* x_last ./ dot(w_last, x_last))) ./
                       sum(w_last .* exp.(0.05 .* x_last ./ dot(w_last, x_last)));
                       atol = 1e-14)
        @test !isapprox(w_next, w_last; atol = 1e-6)
        # The prototype's corrections stand: the reversion family is a total bet on a
        # reverting market, and the momentum rule bets the other way.
        Rrev = zeros(60, 2)
        Rrev[1:2:end, 1] .= 0.10
        Rrev[2:2:end, 1] .= -0.09
        Rrev[1:2:end, 2] .= -0.09
        Rrev[2:2:end, 2] .= 0.10
        rdrev = ReturnsResult(; nx = ["A", "B"], X = Rrev)
        wealth(alg, r) = begin
            opt = OPS(; alg = alg)
            s = 1.0
            w = fill(0.5, 2)
            for t in axes(r.X, 1)
                s *= dot(w, 1 .+ r.X[t, :])
                w = optimise(opt, rows(r, 1:t)).w
            end
            s
        end
        @test wealth(PassiveAggressiveMeanReversion(), rdrev) >
              wealth(ExponentiatedGradient(), rdrev)
    end

    @testset "The batch-online identity is exact at every block size" begin
        for alg in (BuyAndHold(), ExponentiatedGradient(), NewtonStep(),
                    PassiveAggressiveMeanReversion(), MovingAverageReversion(; window = 4),
                    UniversalPortfolio(; N = N, n_experts = 20, seed = 3),
                    ConfidenceWeightedMeanReversion(), AntiCorrelation(; window = 3),
                    ExpectationMaximisation(), AggregatingAlgorithm(), TopK(; k = 2),
                    WeakAggregatingAlgorithm(),
                    AggregatingExponentialGradient(; etas = [0.05, 0.1]))
            opt = OPS(; alg = alg)
            o = po.partial_fit!(opt, rows(rd, 1:10))
            o = po.partial_fit!(o, rows(rd, 11:17))
            o = po.partial_fit!(o, rows(rd, 18:18))
            a = optimise(o)
            b = optimise(opt, rows(rd, 1:18))
            @test a.w == b.w
            @test isnothing(a.pr) && isnothing(b.pr)
            @test isa(a.retcode, OptimisationSuccess)
            @test po.observation_count(o) == 18
            @test po.held_timestamps(o) == ts[1:18]
            @test isnothing(po.held_timestamps(opt))
            # A batch call on a stepped head runs from w0 over rd and reads no state.
            @test optimise(o, rows(rd, 1:5)).w == optimise(opt, rows(rd, 1:5)).w
            @test optimise(o).w == a.w
        end
        # A walk-forward at test_size = 1 holds the same path as the prototype, one row on.
        cv = OnlineIndexWalkForward(5, 1)
        pred = cross_val_predict(OPS(; alg = PassiveAggressiveMeanReversion()), rd, cv)
        p = PT.pamr(X)
        @test isapprox(pred.pred[1].res.w, p.weights[6, :]; atol = 1e-14)
        # A block of k rows is k updates and one read-out: the fold's target is the
        # Next-Period Allocation as of the block's end.
        cv5 = OnlineIndexWalkForward(5, 5)
        pred5 = cross_val_predict(OPS(; alg = PassiveAggressiveMeanReversion()), rd, cv5)
        @test isapprox(pred5.pred[1].res.w, p.weights[6, :]; atol = 1e-14)
        cvb = IndexWalkForward(5, 5; expand_train = true)
        predb = cross_val_predict(OPS(; alg = PassiveAggressiveMeanReversion()), rd, cvb)
        @test isapprox(predb.pred[1].res.w, p.weights[6, :]; atol = 1e-14)
        @test isapprox(predb.mrd.X, pred5.mrd.X; atol = 1e-14)
        # The uncapped refit that `Online(head)` stands for is the online arm bit for bit,
        # and the capped one restarts the recursion from `w0` inside each window.
        @test all(a.res.w == b.res.w for (a, b) in zip(predb.pred, pred5.pred))
        prr = cross_val_predict(OPS(; alg = ExponentiatedGradient()), rd,
                                IndexWalkForward(8, 3))
        @test all(prr.pred[i].res.w == optimise(OPS(; alg = ExponentiatedGradient()),
                       rows(rd, (1 + 3 * (i - 1)):(8 + 3 * (i - 1)))).w
                  for i in eachindex(prr.pred))
    end

    @testset "The Start Allocation" begin
        # Absent, uniform over the pinned universe; given, projected once and held for
        # one period by a rule that does not read it.
        w0 = [0.7, 0.1, 0.1, 0.1]
        crp = OPS(; alg = ConstantRebalancedPortfolio(), w0 = w0)
        @test optimise(crp, rows(rd, 1:1)).w == fill(0.25, 4)
        eg = OPS(; alg = ExponentiatedGradient(), w0 = w0)
        x1 = X[1, :]
        q = w0 .* exp.(0.05 .* x1 ./ dot(w0, x1))
        @test isapprox(optimise(eg, rows(rd, 1:1)).w, q ./ sum(q); atol = 1e-14)
        # A start outside the set is projected, never refused.
        out = OPS(; alg = BuyAndHold(), w0 = [2.0, 0.5, -0.2, 0.1])
        r1 = optimise(out, rows(rd, 1:1)).w
        pw0 = po.project_simplex([2.0, 0.5, -0.2, 0.1])
        @test isapprox(r1, pw0 .* x1 ./ dot(pw0, x1); atol = 1e-14)
        # A rule whose next allocation is not a step from `w` replaces the start at the
        # first update. A constant rebalanced portfolio and a mixture of them never read it
        # again, but the Newton step's gradient reads `w`, and an expert that steps from
        # `w` starts at `w0`, so both carry it forward.
        exps = [ConstantRebalancedPortfolio(; w = [0.4, 0.2, 0.2, 0.2]),
                ConstantRebalancedPortfolio()]
        for (alg, carries) in ((ConstantRebalancedPortfolio(), false),
                               (ExpertMixture(; experts = exps), false), (NewtonStep(), true),
                               (ExpertMixture(; experts = [ExponentiatedGradient(), BuyAndHold()]), true))
            for t in 1:3
                a = optimise(OPS(; alg = alg, w0 = w0), rows(rd, 1:t)).w
                b = optimise(OPS(; alg = alg), rows(rd, 1:t)).w
                @test (a != b) == carries
            end
        end
        # A zero under the entropic geometry stays zero.
        egz = OPS(; alg = ExponentiatedGradient(), w0 = [0.5, 0.5, 0.0, 0.0])
        @test optimise(egz, rows(rd, 1:3)).w[3:4] == [0.0, 0.0]
        # A negative entry under the entropic geometry is refused at the projection.
        @test_throws DomainError optimise(OPS(; alg = ExponentiatedGradient(),
                                              w0 = [0.6, 0.6, -0.2, 0.0]), rows(rd, 1:1))
        @test_throws DimensionMismatch optimise(OPS(; alg = BuyAndHold(), w0 = [0.5, 0.5]),
                                                rows(rd, 1:1))
        @test_throws DomainError OPS(; alg = BuyAndHold(), w0 = [0.5, NaN])
        # A view slices and renormalises w0 with the names.
        v = po.port_opt_view(OPS(; alg = BuyAndHold(), w0 = w0), [1, 2], R)
        @test isapprox(v.w0, [0.875, 0.125]; atol = 1e-14)
    end

    @testset "Non-finite cells, the Held Gap, the uniform start and relisting" begin
        # Asset D is unlisted for rows 1:10, asset B delists from row 26.
        Rg = copy(R)
        amsk = trues(T, N)
        amsk[1:10, 4] .= false
        amsk[26:T, 2] .= false
        Rg[.!amsk] .= NaN
        pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk))
        rdg = ReturnsResult(; nx = nx, X = Rg, ts = ts, pnl = pnl)
        opt = OPS(; alg = ExponentiatedGradient())
        # An inactive cell is filled silently: no warning over the unlisted span.
        res = @test_nowarn optimise(opt, rows(rdg, 1:5))
        # The recursion keeps 1/N on the unlisted leg; the read-out slices it away and
        # renormalises, and the Result expands with a zero.
        o = po.partial_fit!(opt, rows(rdg, 1:5))
        @test 0.2 < o.cache.w[4] < 0.3
        @test res.w[4] == 0
        @test res.imsk == amsk[5, :]
        @test isapprox(sum(res.w), 1; atol = 1e-14)
        @test isapprox(res.w[1:3], o.cache.w[1:3] ./ sum(o.cache.w[1:3]); atol = 1e-14)
        # At row 11 D relists at the recursion's own weight, never at zero.
        res11 = optimise(opt, rows(rdg, 1:11))
        @test res11.w[4] > 0.2
        # The parked weight over the unlisted span enters the gross return that a rule
        # reads. Buy-and-hold scales each leg by its own relative, so its fund is the
        # recursion over the listed names alone; the exponentiated-gradient step divides by
        # the gross return, so its fund is not.
        full = optimise(OPS(; alg = ExponentiatedGradient()),
                        ReturnsResult(; nx = nx[1:3], X = R[1:5, 1:3])).w
        @test !isapprox(res.w[1:3], full; atol = 1e-6)
        @test isapprox(res.w[1:3] .- 1 / 3, full .- 1 / 3; rtol = 1e-2)
        bah = optimise(OPS(; alg = BuyAndHold()), rows(rdg, 1:5)).w
        bah3 = optimise(OPS(; alg = BuyAndHold()),
                        ReturnsResult(; nx = nx[1:3], X = R[1:5, 1:3])).w
        @test isapprox(bah[1:3], bah3; atol = 1e-14) && bah[4] == 0
        # An active cell with a non-finite return is a Held Gap: a warning, and a refusal
        # under `strict`.
        Rh = copy(R)
        Rh[3, 2] = NaN
        rdh = ReturnsResult(; nx = nx, X = Rh, ts = ts)
        @test_logs (:warn, r"non-finite return") optimise(opt, rows(rdh, 1:5))
        @test_throws ArgumentError optimise(OPS(; alg = ExponentiatedGradient(),
                                                strict = true), rows(rdh, 1:5))
        # The message names the row's index in the whole fold, across blocks.
        o12 = po.partial_fit!(opt, rows(rdh, 1:2))
        @test_logs (:warn, r"the first at observation 3\.") po.partial_fit!(o12,
                                                                            rows(rdh, 3:5))
        # A gap at an asset to which the recursion gives no weight is silent. The
        # entropic step keeps the zero exact.
        @test_nowarn optimise(OPS(; alg = ExponentiatedGradient(),
                                  w0 = [0.5, 0.0, 0.25, 0.25], strict = true),
                              rows(rdh, 1:5))
        # The step reads x = 1 at the gap: the same answer as a zero return.
        Rz = copy(R)
        Rz[3, 2] = 0.0
        @test optimise(OPS(; alg = ExponentiatedGradient(), strict = true),
                       rows(ReturnsResult(; nx = nx, X = Rz, ts = ts), 1:5)).w ==
              @test_logs (:warn, r"non-finite return") optimise(opt, rows(rdh, 1:5)).w
        @test po.price_relative(NaN) == 1.0 && po.price_relative(0.5) == 1.5
        @test po.price_relative(-Inf) == 1.0
        # The diagnostic writes nothing: the row keeps its gap.
        rgap = [0.1, NaN, 0.2]
        @test_logs (:warn, r"non-finite return") po.report_row_gaps(rgap, [0.5, 0.5, 0.0],
                                                                    nothing, nx[1:3], false,
                                                                    7)
        @test isnan(rgap[2])
        @test_nowarn po.report_row_gaps(rgap, [0.5, 0.0, 0.5], nothing, nx[1:3], false, 7)
        @test_nowarn po.report_row_gaps(rgap, [0.5, 0.5, 0.0], [true, false, true], nx[1:3],
                                        true, 7)
        @test_throws ArgumentError po.report_row_gaps(rgap, [0.5, 0.5, 0.0], nothing,
                                                      nx[1:3], true, 7)
        # The buffer keeps the gap and the row's active mask, and the carrier the rule reads
        # is the buffer under the pinned names with the masks as a time-varying panel.
        ob = po.partial_fit!(OPS(; alg = MovingAverageReversion(; window = 4)),
                             rows(rdg, 9:12))
        B = ob.cache.X
        @test B.n == 3 && size(B.A) == size(B.X)
        @test isnan(po.sample_buffer(B)[1, 4]) && !B.A[B.off + 1, 4]
        @test all(isfinite, po.sample_buffer(B)[2:3, :]) &&
              all(B.A[(B.off + 2):(B.off + 3), :])
        crd = po.rows_carrier(B, ob.cache.nx)
        @test isa(crd, ReturnsResult) && crd.nx == nx && size(crd.X) == (3, 4)
        @test crd.pnl.amsk == amsk[10:12, :] && crd.pnl.emsk == amsk[10:12, :]
        @test isnothing(po.rows_carrier(nothing, nx)) &&
              isnothing(po.rows_carrier(nothing, nothing))
        @test_throws po.IsNothingError po.rows_carrier(B, nothing)
        # The fold arm reads the current row under the last mask row of the carrier.
        @test po.last_active_mask(crd.pnl) == amsk[12, :]
        @test isnothing(po.last_active_mask(nothing))
        # The fold arm hands the row verbatim: a moment forecaster folds the gap as a
        # gap, and a folding price-level statistic reads the mask itself, so it resets the
        # gapped asset and answers `NaN` for it until the asset quotes again.
        rg = [0.1, NaN, -0.2]
        pl = po.partial_fit!(PriceLevelExpectedReturns(; alg = ExponentialMovingAverage()),
                             rg)
        @test pl.cache.nu == [1, 0, 1] && isnan(vec(mean(pl))[2])
        @test all(isfinite, vec(mean(po.partial_fit!(pl, [0.0, 0.05, 0.0]))))
        @test isa(optimise(OPS(; alg = KernelTrendPatternTracking()), rows(rdg, 1:30)).retcode,
                  OptimisationSuccess)
        @test isnothing(po.last_active_mask(AssetPanel(;
                                                       pf = [NumericPanelField(; name = "a",
                                                                               vals = ones(4))])))
        # On a static panel the buffer records no mask and the carrier carries no panel.
        os = po.partial_fit!(OPS(; alg = MovingAverageReversion(; window = 4)),
                             rows(rd, 1:4))
        @test isnothing(os.cache.X.A) && isnothing(po.buffer_panel(os.cache.X))
        @test isnothing(po.rows_carrier(os.cache.X, nx).pnl)
        # A windowed statistic over a relisted asset reads the rows it has, not filled
        # zeros: at row 15 a plain moving average over D is undefined, so the step holds
        # that leg, and the answer is finite.
        mar = MovingAverageReversion(; window = 15)
        om = po.partial_fit!(OPS(; alg = mar), rows(rdg, 1:15))
        crd15 = po.rows_carrier(om.cache.X, nx)
        muD = vec(mean(PriceLevelExpectedReturns(; alg = MovingAverage(; window = 15)),
                       crd15.X, crd15.pnl; dims = 1))
        @test isnan(muD[4]) && all(isfinite, muD[1:3])
        mres = optimise(OPS(; alg = mar), rows(rdg, 1:15))
        @test isa(mres.retcode, OptimisationSuccess) && all(isfinite, mres.w)
        # A delisted asset at the last row leaves the read-out; a window with every
        # asset dead answers a failure a fallback walks on from.
        res30 = optimise(opt, rows(rdg, 1:30))
        @test res30.w[2] == 0 && res30.imsk == amsk[30, :]
        amsk0 = trues(3, N)
        amsk0[3, :] .= false
        R0 = copy(R[1:3, :])
        R0[3, :] .= NaN
        rd0 = ReturnsResult(; nx = nx, X = R0,
                            pnl = AssetPanel(; amsk = amsk0, emsk = amsk0))
        dead = optimise(OPS(; alg = BuyAndHold()), rd0)
        @test isa(dead.retcode, OptimisationFailure)
        held = OPS(; alg = BuyAndHold(), fb = PreviousWeights(; w = fill(0.25, 4)))
        hs = po.partial_fit!(held, rd0)
        @test optimise(hs).w == fill(0.25, 4)
        # The identity holds under the time-varying panel too.
        og = po.partial_fit!(po.partial_fit!(opt, rows(rdg, 1:12)), rows(rdg, 13:28))
        @test optimise(og).w == optimise(opt, rows(rdg, 1:28)).w
        # The fold loop's Held Gap idiom is reused: `iv`, a panel with fields and
        # `emsk != amsk` are refused by name.
        emsk = copy(amsk)
        emsk[2, 1] = false
        @test_throws ArgumentError optimise(opt,
                                            ReturnsResult(; nx = nx, X = Rg,
                                                          pnl = AssetPanel(; amsk = amsk,
                                                                           emsk = emsk)))
    end

    @testset "The state: view, copy, merge, pin and the row buffer" begin
        opt = OPS(; alg = MovingAverageReversion(; window = 4))
        o = po.partial_fit!(opt, rows(rd, 1:9))
        st = o.cache
        @test st.X.n == 3 && st.X.max_history == 3
        @test st.nx == nx && st.n == 9 && isnothing(st.amsk)
        @test isnothing(po.partial_fit!(OPS(; alg = BuyAndHold()), rows(rd, 1:3)).cache.X)
        @test isnothing(po.partial_fit!(OPS(; alg = NewtonStep()), rows(rd, 1:3)).cache.X) ==
              true
        # A rule tree that reads every row keeps them uncapped; a forecaster that folds is
        # carried on the Rule State and the head holds the current row alone for it, the row
        # verbatim that the fold reads (ADR 0158, ADR 0170).
        pref = OPS(; alg = ForecastReversion(; me = MedianExpectedReturns()))
        @test isnothing(po.partial_fit!(pref, rows(rd, 1:9)).cache.X.max_history)
        @test po.partial_fit!(pref, rows(rd, 1:9)).cache.X.n == 9
        pfold = po.partial_fit!(OPS(;
                                    alg = ForecastReversion(; me = SimpleExpectedReturns())),
                                rows(rd, 1:9)).cache
        @test pfold.X.n == 1 && pfold.X.max_history == 1
        @test pfold.st isa po.ForecasterState && pfold.st.me.cache.n == 9
        @test pfold.st.me.cache.mu ≈ vec(mean(R[1:9, :]; dims = 1))
        # A copy aliases no array.
        c = copy(st)
        @test c.w == st.w && c.w !== st.w && c.X.X !== st.X.X && c.ts !== st.ts
        ns = po.partial_fit!(OPS(; alg = NewtonStep()), rows(rd, 1:3)).cache
        nc = copy(ns)
        @test nc.st.A == ns.st.A && nc.st.A !== ns.st.A
        # A view slices every per-asset axis and renormalises the allocation.
        v = po.port_opt_view(st, [1, 3], R)
        @test isapprox(sum(v.w), 1; atol = 1e-14)
        @test v.w == st.w[[1, 3]] ./ sum(st.w[[1, 3]])
        @test size(po.sample_buffer(v.X), 2) == 2 && v.nx == ["A", "C"] && v.ts == st.ts
        vn = po.port_opt_view(ns, [2, 4], R)
        @test vn.st.A == ns.st.A[[2, 4], [2, 4]] && vn.st.b == ns.st.b[[2, 4]]
        mixs = po.partial_fit!(OPS(;
                                   alg = UniversalPortfolio(; N = N, n_experts = 5,
                                                            seed = 1)), rows(rd, 1:3)).cache
        vm = po.port_opt_view(mixs, [1, 2], R)
        @test length(vm.st.h) == 5 && all(h -> isapprox(sum(h), 1; atol = 1e-14), vm.st.h)
        @test vm.st.p == mixs.st.p
        # The head's view reaches the rule and the set.
        vo = po.port_opt_view(OPS(;
                                  alg = ConstantRebalancedPortfolio(;
                                                                    w = [0.4, 0.3, 0.2,
                                                                         0.1])), [1, 2], R)
        @test vo.alg.w == [0.4, 0.3] ./ 0.7
        # A stepped head viewed by the loop's multiple-randomised path is the same
        # recursion over the selected assets.
        vs = po.port_opt_view(o, [1, 3], R)
        @test vs.cache.nx == ["A", "C"]
        @test optimise(vs).w == po.renormalised_view(optimise(o).w, [1, 3])
        # Merge refuses on the head's state and on every carrier.
        @test_throws ArgumentError po.merge_states(st, copy(st))
        @test_throws ArgumentError po.merge_states(ns.st, copy(ns.st))
        @test_throws ArgumentError po.merge_states(mixs.st, copy(mixs.st))
        # The pin: a reordered universe is refused by name, and so is a carrier of the
        # wrong width or one that drops the timestamps.
        @test_throws ArgumentError po.partial_fit!(o,
                                                   ReturnsResult(; nx = reverse(nx),
                                                                 X = R[10:10, :],
                                                                 ts = ts[10:10]))
        @test_throws DimensionMismatch po.partial_fit!(o,
                                                       ReturnsResult(; nx = nx[1:3],
                                                                     X = R[10:10, 1:3],
                                                                     ts = ts[10:10]))
        @test_throws ArgumentError po.partial_fit!(o,
                                                   ReturnsResult(; nx = nx,
                                                                 X = R[10:10, :]))
        # The refusals of the online seams.
        @test_throws ArgumentError optimise(opt)
        @test_throws ArgumentError po.online_readout(o)
        @test_throws ArgumentError po.update_online_estimator(Online(opt; max_history = 5))
        cv = OnlineIndexWalkForward(5, 1)
        @test_throws ArgumentError cross_val_predict(o, rd, cv)
        # `Resume` re-enters from the timestamps the state holds.
        pr = cross_val_predict(opt, rows(rd, 1:20), cv)
        pr2 = cross_val_predict(Resume(pr), rd, cv)
        one = cross_val_predict(opt, rd, cv)
        @test length(pr.pred) + length(pr2.pred) == length(one.pred)
        @test pr2.pred[end].res.w == one.pred[end].res.w
        @test po.held_timestamps(pr2.opt) == po.held_timestamps(one.opt)
    end

    @testset "The rules' own contracts" begin
        @test_throws DomainError ExponentiatedGradient(; eta = 0)
        @test_throws DomainError NewtonStep(; eta = 1)
        @test_throws DomainError NewtonStep(; delta = 0)
        @test_throws DomainError PassiveAggressiveMeanReversion(; eps = -1)
        @test_throws DomainError LinearSlack(; C = 0)
        @test_throws DomainError ForecastReversion(; eps = 0)
        @test_throws DomainError MovingAverage(; window = 1)
        @test_throws DomainError SpatialMedian(; iters = 0)
        @test_throws DomainError ConstantRebalancedPortfolio(; w = [0.5, -0.5])
        # Every Allocation Set has a budget of one, so a target off it would scale the
        # expert's first-period return in a mixture.
        @test_throws DomainError ConstantRebalancedPortfolio(; w = [2, 0])
        @test_throws DomainError ConstantRebalancedPortfolio(; w = [0.5, 0.4])
        @test ConstantRebalancedPortfolio(; w = [1, 0]).w == [1, 0]
        @test po.port_opt_view(ConstantRebalancedPortfolio(; w = [0.2, 0.3, 0.5]), [1, 3]).w ≈
              [2 / 7, 5 / 7]
        @test_throws Exception ExpertMixture(;
                                             experts = AbstractOnlinePortfolioSelectionAlgorithm[])
        @test_throws ArgumentError ExpertMixture(; experts = [BuyAndHold()],
                                                 alg = MovingAverageReversion())
        @test_throws DimensionMismatch ExpertMixture(;
                                                     experts = [BuyAndHold(), NewtonStep()],
                                                     p0 = [1.0])
        @test_throws DomainError ExpertMixture(; experts = [BuyAndHold(), NewtonStep()],
                                               p0 = [0.7, NaN])
        # A start over the experts outside the Expert Set is projected onto it at the seed,
        # in the weighting's geometry, as the head's `w0` is: never refused.
        mp = ExpertMixture(; experts = [BuyAndHold(), NewtonStep()], p0 = [0.7, 0.7])
        @test po.rule_state_seed(mp, fill(0.25, 4)).p == [0.5, 0.5]
        @test_throws DimensionMismatch UniversalPortfolio(; N = 3, alpha = [1.0, 1.0])
        # The geometry slot is a bound: a combination no theorem covers fails at
        # construction.
        # `ExponentiatedGradient` is a constructor of `MirrorDescent` that fills the slot.
        @test_throws MethodError ExponentiatedGradient(; proj = EuclideanProjection())
        @test_throws TypeError NewtonStep(; proj = EntropicProjection())
        # Under a bound other than the simplex, each geometry is a scalar root: the
        # example of ADR 0159, a raw exponentiated-gradient step under a cap of 0.4.
        w3 = fill(1 / 3, 3)
        x3 = [1.5, 1.0, 0.8]
        q3 = w3 .* exp.(x3 ./ dot(w3, x3))
        cap = po.resolve_allocation_set(BoundedAllocationSet(; wb = WeightBounds(0, 0.4)),
                                        3, false, Float64)
        pe = po.project(EntropicProjection(), cap, q3, w3)
        # The Euclidean root is not scale-free, so it takes the normalised step the ADR
        # states; the entropic one is, and takes the raw one.
        pu = po.project(EuclideanProjection(), cap, q3 ./ sum(q3), w3)
        @test isapprox(pe, [0.400, 0.327, 0.273]; atol = 1e-3)
        @test isapprox(pu, [0.400, 0.324, 0.276]; atol = 1e-3)
        @test isapprox(sum(pe), 1; atol = 1e-12) && isapprox(sum(pu), 1; atol = 1e-12)
        # The uncapped ratio survives the entropic root, the uncapped difference the
        # Euclidean one.
        @test isapprox(pe[2] / pe[3], q3[2] / q3[3]; atol = 1e-10)
        @test isapprox(pu[2] - pu[3], (q3[2] - q3[3]) / sum(q3); atol = 1e-10)
        # Under the simplex bounds the roots reduce to the sort and to normalisation.
        s3 = po.resolve_allocation_set(BoundedAllocationSet(), 3, false, Float64)
        @test po.project(EuclideanProjection(), s3, [2.0, -1.0, 0.5], w3) ==
              po.project_simplex([2.0, -1.0, 0.5])
        capped = OPS(; alg = ExponentiatedGradient(),
                     set = BoundedAllocationSet(; wb = WeightBounds(0, 0.4)))
        wc = optimise(capped, rd).w
        @test all(<=(0.4 + 1e-12), wc) && isapprox(sum(wc), 1; atol = 1e-12)
        # Outside a fold loop, a schedule on `set` steps inside its `default`, or inside
        # the default Allocation Set when it has none.
        cap4 = BoundedAllocationSet(; wb = WeightBounds(0, 0.4))
        free4 = BoundedAllocationSet()
        withd = po.partial_fit!(OPS(; alg = ExponentiatedGradient(),
                                    set = TimeDependent(; val = [free4, free4],
                                                        default = cap4)), rd)
        @test withd.cache.w == po.partial_fit!(capped, rd).cache.w
        nod = po.partial_fit!(OPS(; alg = ExponentiatedGradient(),
                                  set = TimeDependent(; val = [cap4, cap4])), rd)
        @test nod.cache.w ==
              po.partial_fit!(OPS(; alg = ExponentiatedGradient()), rd).cache.w
        lower = OPS(; alg = PassiveAggressiveMeanReversion(),
                    set = BoundedAllocationSet(; wb = WeightBounds(0.1, 1)))
        @test all(>=(0.1 - 1e-12), optimise(lower, rd).w)
        # A negative lower bound is admitted under the Euclidean geometry and refused under
        # the entropic one; an infeasible bound is refused by name.
        neg = po.resolve_allocation_set(BoundedAllocationSet(; wb = WeightBounds(-0.2, 1)),
                                        3, false, Float64)
        @test minimum(po.project(EuclideanProjection(), neg, [1.5, -0.6, 0.1], w3)) < 0
        @test_throws DomainError po.project(EntropicProjection(), neg, q3, w3)
        bad = po.resolve_allocation_set(BoundedAllocationSet(; wb = WeightBounds(0, 0.2)),
                                        3, false, Float64)
        @test_throws ArgumentError po.project(EuclideanProjection(), bad, q3, w3)
        # The Expert Set: a scalar cap on `eset` binds the trust vector, and the blend
        # stays on the head's set.
        exps = [ConstantRebalancedPortfolio(; w = [0.9, 0.05, 0.05, 0.0]),
                ConstantRebalancedPortfolio(; w = [0.05, 0.9, 0.05, 0.0]),
                ConstantRebalancedPortfolio(; w = [0.05, 0.05, 0.9, 0.0])]
        Rtrend = zeros(20, 4)
        Rtrend[:, 1] .= 0.05
        rdt = ReturnsResult(; nx = nx, X = Rtrend)
        free = po.partial_fit!(OPS(; alg = ExpertMixture(; experts = exps)), rdt).cache
        held = po.partial_fit!(OPS(;
                                   alg = ExpertMixture(; experts = exps,
                                                       eset = BoundedAllocationSet(;
                                                                                   wb = WeightBounds(0,
                                                                                                     0.5)))),
                               rdt).cache
        @test free.st.p[1] > 0.5
        @test all(<=(0.5 + 1e-12), held.st.p) && isapprox(sum(held.st.p), 1; atol = 1e-12)
        @test held.w[1] < free.w[1]
        @test isapprox(sum(held.w), 1; atol = 1e-12)
        @test_throws ArgumentError ExpertMixture(; experts = exps,
                                                 eset = BoundedAllocationSet(;
                                                                             wb = WeightBoundsEstimator(),
                                                                             sets = UniverseSets(;
                                                                                                 dict = Dict("nx" =>
                                                                                                                 nx))))
        @test_throws IsNothingError BoundedAllocationSet(; wb = WeightBoundsEstimator())
        # The two projections.
        @test po.project_simplex([0.5, 0.5]) == [0.5, 0.5]
        @test po.project_simplex([2.0, -1.0, 0.5]) == [1.0, 0.0, 0.0]
        @test isapprox(sum(po.project_simplex(randn(StableRNG(1), 7))), 1; atol = 1e-14)
        set = po.resolve_allocation_set(BoundedAllocationSet(), 3, false, Float64)
        @test po.project(EntropicProjection(), set, [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]) ==
              [0.25, 0.5, 0.25]
        @test_throws DomainError po.project(EntropicProjection(), set, [1.0, -2.0, 1.0],
                                            [1.0, 1.0, 1.0])
        @test_throws DomainError po.project(EntropicProjection(), set, [0.0, 0.0, 0.0],
                                            [1.0, 1.0, 1.0])
        # Buy and hold trades nothing: the update is the Price-Adjusted Allocation.
        w = [0.5, 0.3, 0.2]
        x = [1.1, 0.9, 1.0]
        @test po.price_adjusted_allocation(w, x) == w .* x ./ dot(w, x)
        st, wn = po.online_update!(BuyAndHold(), nothing, w, x, nothing, set)
        @test isnothing(st) &&
              isapprox(wn, po.price_adjusted_allocation(w, x); atol = 1e-15)
        # A mixture's weighting under the Newton step carries a K × K Gram.
        mix = ExpertMixture(;
                            experts = [BuyAndHold(), ExponentiatedGradient(), NewtonStep()],
                            alg = NewtonStep())
        ms = po.partial_fit!(OPS(; alg = mix), rows(rd, 1:4)).cache
        @test size(ms.st.pst.A) == (3, 3) && isapprox(sum(ms.st.p), 1; atol = 1e-14)
        @test isapprox(sum(ms.w), 1; atol = 1e-14)
        # The mixture's regret against its best expert is bounded by log K under the
        # wealth weighting.
        up = UniversalPortfolio(; N = N, n_experts = 30, seed = 2)
        S_mix = 1.0
        w = fill(0.25, N)
        opt = OPS(; alg = up)
        for t in 1:T
            S_mix *= dot(w, X[t, :])
            w = optimise(opt, rows(rd, 1:t)).w
        end
        S_best = maximum(prod(X * e.w) for e in up.experts)
        @test log(S_best) - log(S_mix) <= log(30) + 1e-12
        # The forecaster: the levels reconstructed with the last one at one, and the
        # ledger's three-asset example.
        Xe = [0.90 1.05 1.00; 1.05 1.00 0.98]
        mu = vec(mean(PriceLevelExpectedReturns(; alg = MovingAverage(; window = 3)),
                      Xe .- 1; dims = 1))
        @test isapprox(1 .+ mu, [1.0035, 0.9841, 1.0136]; atol = 5e-5)
        @test size(mean(PriceLevelExpectedReturns(), transpose(Xe .- 1); dims = 2)) ==
              (3, 1)
        @test po.rows_needed(MovingAverageReversion(; window = 5)) == 4
        @test po.rows_needed(RobustMedianReversion(; window = 3)) == 2
        # A folding forecaster reads the current row verbatim, so the head holds one for it.
        @test po.rows_needed(SimpleExpectedReturns()) == 1
        @test po.rows_needed(ExponentialMovingAverageReversion()) == 1
        @test isnothing(po.rows_needed(MedianExpectedReturns()))
        @test po.rows_needed(ExpertMixture(;
                                           experts = [BuyAndHold(),
                                                      MovingAverageReversion(; window = 7)])) ==
              6
        @test po.rows_needed(OPS(; alg = NewtonStep())) == 0
        # The spatial median is the coordinatewise median of a symmetric cloud, and the
        # modified iteration stays defined on a data point.
        P = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        @test isapprox(po.spatial_median(P, 100, 1e-10), [0.0, 0.0]; atol = 1e-8)
        P2 = [0.0 0.0; 1.0 0.0; 0.0 1.0]
        y = po.spatial_median(P2, 200, 1e-12)
        @test all(isfinite, y) &&
              sum(norm(P2[i, :] .- y) for i in 1:3) <=
              sum(norm(P2[i, :]) for i in 1:3) + 1e-12
        @test_throws IsEmptyError mean(PriceLevelExpectedReturns(), zeros(0, 3))
        # Every row on the iterate is the median; a data point with balanced directions
        # holds the iterate.
        @test po.spatial_median(ones(4, 2), 10, 1e-8) == [1.0, 1.0]
        P3 = [0.0 0.0; 1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        @test po.spatial_median(P3, 10, 1e-8) == [0.0, 0.0]
        # A forecast that is the same at every asset moves nothing.
        flat = ReturnsResult(; nx = ["A", "B"], X = [0.01 0.01; 0.02 0.02; -0.01 -0.01])
        @test optimise(OPS(; alg = MovingAverageReversion(; window = 3)), flat).w ==
              [0.5, 0.5]
        # A vector concentration draws the experts from the asymmetric prior.
        upv = UniversalPortfolio(; N = 3, n_experts = 4, alpha = [1.0, 2.0, 3.0], seed = 1)
        @test length(upv.experts) == 4 && all(e -> length(e.w) == 3, upv.experts)
        # The state pins a static panel only.
        amsk2 = trues(2, 2)
        amsk2[1, 1] = false
        @test_throws ArgumentError po.OnlinePortfolioSelectionState(; w = [0.5, 0.5],
                                                                    pnl = AssetPanel(;
                                                                                     amsk = amsk2,
                                                                                     emsk = amsk2))
        # A rule with a carrier of no view is honest.
        @test_throws ArgumentError po.rule_state_view(1.0, [1])
    end

    @testset "Fees: the head, the four naive heads, and the loop's door" begin
        fees = Fees(; l = 0.01)
        tnfees = Fees(; tn = Turnover(; w = zeros(N), val = 0.02), l = 0.001)
        cv = OnlineIndexWalkForward(5, 1)
        cvd = OnlineIndexWalkForward(5, 1; pws = DriftedWeights())
        cvb = IndexWalkForward(5, 3)
        # The Result carries the fee, the fold charges it, and a fee-free head is unchanged.
        for (opt, optf) in ((OPS(; alg = ExponentiatedGradient()),
                             OPS(; alg = ExponentiatedGradient(), fees = fees)),
                            (EqualWeighted(), EqualWeighted(; fees = fees)),
                            (InverseVolatility(), InverseVolatility(; fees = fees)),
                            (RandomWeighted(; seed = 1), RandomWeighted(; seed = 1, fees = fees)))
            r = optimise(opt, rd)
            rf = optimise(optf, rd)
            @test isnothing(r.fees) && isa(rf.fees, Fees) && rf.fees.l == 0.01
            @test r.w == rf.w
            @test all(<(0), predict(rf, rd).rd.X .- predict(r, rd).rd.X)
            pf = cross_val_predict(optf, rd, cvb)
            p0 = cross_val_predict(opt, rd, cvb)
            @test all(<(0), pf.mrd.X .- p0.mrd.X)
        end
        # `PreviousWeights` holds a resolved fee, and `factory` reaches it beside `w`.
        pw = PreviousWeights(; fees = tnfees)
        pwf = po.factory(pw, [0.4, 0.3, 0.2, 0.1])
        @test pwf.w == [0.4, 0.3, 0.2, 0.1] && pwf.fees.tn.w == [0.4, 0.3, 0.2, 0.1]
        rpw = optimise(PreviousWeights(; w = fill(0.25, N), fees = fees), rd)
        @test rpw.fees.l == 0.01
        @test all(<(0),
                  predict(rpw, rd).rd.X .-
                  predict(optimise(PreviousWeights(; w = fill(0.25, N)), rd), rd).rd.X)
        held = cross_val_predict(PreviousWeights(; w = fill(0.25, N), fees = fees), rd, cvb)
        @test all(<(0),
                  held.mrd.X .-
                  cross_val_predict(PreviousWeights(; w = fill(0.25, N)), rd, cvb).mrd.X)
        # A `FeesEstimator` needs sets on the heads that resolve one.
        @test_throws IsNothingError EqualWeighted(; fees = FeesEstimator(; l = 0.01))
        @test_throws IsNothingError OPS(; alg = BuyAndHold(),
                                        fees = FeesEstimator(; l = 0.01))
        # The head's `factory` threads the previous weights into the fee and nowhere else.
        opt = OPS(; alg = ExponentiatedGradient(), fees = tnfees, w0 = fill(0.25, N))
        of = po.factory(opt, [0.4, 0.3, 0.2, 0.1])
        @test of.fees.tn.w == [0.4, 0.3, 0.2, 0.1] && of.w0 == opt.w0 && of.alg === opt.alg
        # A turnover fee on the family needs a Previous-Weights Source at the loop's
        # door; without a `tn` term nothing is checked, and the batch loop is untouched.
        @test_throws ArgumentError cross_val_predict(opt, rd, cv)
        @test isa(cross_val_predict(opt, rd, cvd), MultiPeriodPredictionResult)
        @test isa(cross_val_predict(OPS(; alg = ExponentiatedGradient(), fees = fees), rd,
                                    cv), MultiPeriodPredictionResult)
        @test isa(cross_val_predict(opt, rd, cvb), MultiPeriodPredictionResult)
        @test isa(cross_val_predict(EqualWeighted(; fees = tnfees), rd, cv),
                  MultiPeriodPredictionResult)
        # A per-fold schedule on `fees` is read through every entry it can be seen to
        # hold and its default: a scheduled turnover fee is refused at the door as a
        # static one is, and a schedule with no turnover term passes.
        K = length(cross_val_predict(OPS(; alg = BuyAndHold()), rd, cv).pred)
        @test po.fees_carry_turnover(TimeDependent(; val = [fees, tnfees], default = fees))
        @test po.fees_carry_turnover(TimeDependent(; val = fill(fees, K), default = tnfees))
        @test !po.fees_carry_turnover(TimeDependent(; val = fill(fees, K),
                                                    default = nothing))
        # A callable's output does not exist before the fold does, so it is not read.
        @test !po.fees_carry_turnover(TimeDependent(; val = ctx -> tnfees, default = fees))
        sched = OPS(; alg = BuyAndHold(),
                    fees = TimeDependent(; val = fill(tnfees, K), default = fees))
        @test_throws ArgumentError cross_val_predict(sched, rd, cv)
        @test isa(cross_val_predict(sched, rd, cvd), MultiPeriodPredictionResult)
        @test isa(cross_val_predict(OPS(; alg = BuyAndHold(),
                                        fees = TimeDependent(; val = fill(fees, K),
                                                             default = nothing)), rd, cv),
                  MultiPeriodPredictionResult)
        # Under the source the turnover fee is charged against the drifted book.
        pd = cross_val_predict(opt, rd, cvd)
        p0 = cross_val_predict(OPS(; alg = ExponentiatedGradient(), w0 = fill(0.25, N)), rd,
                               cvd)
        @test all(<(0), pd.mrd.X .- p0.mrd.X)
        # A static turnover fee makes the head need the previous weights, so the loop
        # threads the drifted book into every fold's fee — never the constructor's `w`.
        @test po.needs_previous_weights(opt)
        @test all(k -> pd.pred[k].res.fees.tn.w == pd.pred[k - 1].hw.w, 2:length(pd.pred))
        # Buy and hold's target is the drifted book, so a pure turnover fee charges nothing
        # after the first fold, whose fee is the entry trade from the constructor's `w`.
        tnonly = Fees(; tn = Turnover(; w = zeros(N), val = 0.02))
        bah = cross_val_predict(OPS(; alg = BuyAndHold(), fees = tnonly), rd, cvd)
        bah0 = cross_val_predict(OPS(; alg = BuyAndHold()), rd, cvd)
        @test isapprox(bah.mrd.X[1], bah0.mrd.X[1] - 0.02; atol = 1e-14)
        @test isapprox(bah.mrd.X[2:end], bah0.mrd.X[2:end]; atol = 1e-14)
        # The batch loop threads the previous target into a static fee on the other naive
        # heads too.
        for est in (EqualWeighted(; fees = tnfees), InverseVolatility(; fees = tnfees),
                    RandomWeighted(; seed = 1, fees = tnfees))
            @test po.needs_previous_weights(est)
            pb = cross_val_predict(est, rd, cvb)
            @test all(k -> pb.pred[k].res.fees.tn.w == pb.pred[k - 1].res.w,
                      2:length(pb.pred))
        end
        # `needs_previous_weights` is the naive family's derived rule.
        @test !po.needs_previous_weights(OPS(; alg = BuyAndHold()))
        @test po.needs_previous_weights(OPS(; alg = BuyAndHold(), fb = PreviousWeights()))
        # A fee is problem definition and may be scheduled per fold on the naive heads.
        td = EqualWeighted(;
                           fees = TimeDependent(; val = [fees, nothing], default = nothing))
        @test isnothing(optimise(td, rd).fees)
    end

    @testset "Show, docs and the search seam" begin
        opt = po.partial_fit!(OPS(; alg = BuyAndHold()), rows(rd, 1:3))
        s = sprint(show, MIME("text/plain"), opt)
        @test occursin("BuyAndHold", s) && !occursin("cache", s)
        # The search addresses a rule's knob by path.
        cv = OnlineIndexWalkForward(5, 1)
        gs = GridSearchCrossValidation(["alg.me.alg.window" => [3, 5]]; cv = cv,
                                       r = MeanReturn(; flag = true))
        res = search_cross_validation(OPS(; alg = MovingAverageReversion()), gs, rd)
        @test res.opt.alg.me.alg.window in (3, 5)
        @test size(res.test_scores, 2) == 2
    end

    @testset "The mathematics of the selection-rule docstrings (#1174)" begin
        # Each literal is a hand recursion of the paper's update on this file's fixture: the
        # Li-Hoi survey for buy-and-hold and the constant rebalanced portfolio, Figure 1 of
        # Agarwal et al. (2006) for the Newton step, Algorithm 1 of Li et al. (2012) for the
        # passive-aggressive step, eq. 34 and Theorem 1 of Cover and Ordentlich (1996) for the
        # mixtures.
        function simplex_projection(q)
            u = sort(q; rev = true)
            cs = cumsum(u)
            rho = findlast(j -> u[j] + (1 - cs[j]) / j > 0, eachindex(u))
            theta = (1 - cs[rho]) / rho
            return max.(q .+ theta, 0)
        end
        padj(w, x) = w .* x ./ dot(w, x)
        simplex = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)

        # Buy-and-hold is the Price-Adjusted Allocation, and the weight of asset i is the
        # share w_1i ∏ x_si of the wealth.
        W = libpath(BuyAndHold())
        H = zeros(T, N)
        H[1, :] .= 1 / N
        for t in 1:(T - 1)
            H[t + 1, :] .= padj(H[t, :], X[t, :])
        end
        @test maxerr(W, H) < 1e-15
        g = vec(prod(X[1:(T - 1), :]; dims = 1))
        @test maxerr(W[T, :], g ./ sum(g)) < 1e-15

        # The constant rebalanced portfolio holds b from the second period, after a given
        # Start Allocation, and its wealth over periods 2 to t is S_t(b) / S_1(b).
        b = [0.1, 0.2, 0.3, 0.4]
        Wc = libpath(ConstantRebalancedPortfolio(; w = b))
        @test maxerr(Wc[2:end, :], repeat(b', T - 1)) < 1e-15
        @test po.partial_fit!(OPS(; alg = ConstantRebalancedPortfolio(; w = b),
                                  w0 = [0.7, 0.1, 0.1, 0.1]), rows(rd, 1:1)).cache.w ≈ b
        S(v, t) = prod(dot(v, X[s, :]) for s in 1:t)
        @test prod(dot(Wc[s, :], X[s, :]) for s in 2:T) ≈ S(b, T) / S(b, 1)
        @test_throws IsEmptyError ConstantRebalancedPortfolio(; w = Float64[])
        @test_throws DomainError ConstantRebalancedPortfolio(; w = [NaN, 1.0])
        # A view whose slice sums to zero is uniform over the selected assets.
        @test po.port_opt_view(ConstantRebalancedPortfolio(; w = [0.0, 0.0, 1.0]), 1:2).w ==
              [0.5, 0.5]

        # The Newton step is Figure 1 of the paper under the Euclidean norm, with the
        # uniform mix before the projection.
        for (beta, delta, eta) in ((1, 0.125, 0), (0.5, 0.3, 0), (1, 0.125, 0.2))
            Wn = libpath(NewtonStep(; beta = beta, delta = delta, eta = eta))
            Hn = zeros(T, N)
            Hn[1, :] .= 1 / N
            A = Matrix(1.0I, N, N)
            bt = zeros(N)
            for t in 1:(T - 1)
                gt = X[t, :] ./ dot(Hn[t, :], X[t, :])
                A .+= gt * gt'
                bt .+= (1 + 1 / beta) .* gt
                q = (1 - eta) .* delta .* (A \ bt) .+ eta / N
                Hn[t + 1, :] .= simplex_projection(q)
            end
            @test maxerr(Wn, Hn) < 1e-13
        end

        # The passive-aggressive step, Algorithm 1 of the paper, under the three step-length
        # rules, at thresholds below, at and above one.
        steps = ((NoSlack(), (l, d) -> l / d),
                 (LinearSlack(; C = 5), (l, d) -> min(5, l / d)),
                 (QuadraticSlack(; C = 5), (l, d) -> l / (d + 1 / 10)))
        for (slack, f) in steps, eps in (0.5, 1.0, 1.01)
            Wp = libpath(PassiveAggressiveMeanReversion(; eps = eps, slack = slack))
            Hp = zeros(T, N)
            Hp[1, :] .= 1 / N
            for t in 1:(T - 1)
                d = X[t, :] .- mean(X[t, :])
                l = max(0, dot(Hp[t, :], X[t, :]) - eps)
                Hp[t + 1, :] .= simplex_projection(Hp[t, :] .- f(l, sum(abs2, d)) .* d)
            end
            @test maxerr(Wp, Hp) < 1e-13
        end
        # Under NoSlack the raw step keeps the budget and earns exactly eps on the period.
        w4, x4 = [0.1, 0.2, 0.3, 0.4], [0.9, 1.0, 1.1, 1.2]
        d4 = x4 .- mean(x4)
        q4 = w4 .-
             po.passive_aggressive_step(NoSlack(), dot(w4, x4) - 0.5, sum(abs2, d4)) .* d4
        @test sum(q4) ≈ 1 && dot(q4, x4) ≈ 0.5
        # NoSlack is the limit of the other two rules as C grows.
        for s in (LinearSlack(; C = 1e12), QuadraticSlack(; C = 1e12))
            @test po.passive_aggressive_step(s, 0.3, 0.02) ≈
                  po.passive_aggressive_step(NoSlack(), 0.3, 0.02)
        end
        # Every asset moved alike: the step is zero.
        @test po.online_update!(PassiveAggressiveMeanReversion(; eps = 0.0), nothing, w4,
                                fill(1.3, N), nothing, simplex)[2] ≈ w4

        # Under buy-and-hold the mixture's wealth is the p1-weighted average of the experts'
        # wealths, so the best expert beats it by at most -log p1 of that expert.
        targets = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], fill(0.25, 4),
                   [0.0, 0.0, 0.5, 0.5]]
        crps = [ConstantRebalancedPortfolio(; w = v) for v in targets]
        Sk = [S(v, T - 1) for v in targets]
        for p1 in (fill(0.25, 4), [0.4, 0.3, 0.2, 0.1])
            Wm = libpath(ExpertMixture(; experts = crps, p0 = p1))
            Wm[1, :] .= sum(p1 .* targets)
            Smix = prod(dot(Wm[t, :], X[t, :]) for t in 1:(T - 1))
            @test Smix ≈ dot(p1, Sk)
            @test log(maximum(Sk)) - log(Smix) <= -log(p1[argmax(Sk)])
        end
        @test_throws IsEmptyError ExpertMixture(; experts = [BuyAndHold()], p0 = Float64[])
        @test_throws DomainError ExpertMixture(; experts = [BuyAndHold()], p0 = [NaN])

        # The universal portfolio is the wealth-weighted average of its sampled targets, and
        # the best sampled expert beats it by at most log K.
        up = UniversalPortfolio(; N = N, n_experts = 50, seed = 3)
        B = reduce(hcat, [e.w for e in up.experts])
        St = [S(B[:, k], 25) for k in 1:50]
        @test maxerr(optimise(OPS(; alg = up), rows(rd, 1:25)).w, B * St ./ sum(St)) < 1e-15
        Wu = libpath(up)
        Wu[1, :] .= vec(mean(B; dims = 2))
        Su = prod(dot(Wu[t, :], X[t, :]) for t in 1:(T - 1))
        SK = [S(B[:, k], T - 1) for k in 1:50]
        @test Su ≈ mean(SK)
        @test log(maximum(SK)) - log(Su) <= log(50)
        @test_throws IsEmptyError UniversalPortfolio(; N = 2, alpha = Float64[])
        @test_throws DomainError UniversalPortfolio(; N = 2, alpha = -1)
        # Theorem 1 of Cover and Ordentlich on the exact integral over two assets: the best
        # constant rebalanced portfolio beats the uniform-prior universal portfolio by at
        # most (N - 1) log(T + 1), here 0.99 against log 31 on an alternating market.
        Xa = [isodd(t) ? [2.0, 0.5] : [0.5, 2.0] for t in 1:30]
        S2(p) = prod(p * x[1] + (1 - p) * x[2] for x in Xa)
        grid = range(0, 1; length = 20_001)
        vals = S2.(grid)
        Sup = sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(grid)
        gap = log(maximum(vals)) - log(Sup)
        @test round(gap; digits = 2) == 0.99 && gap <= log(31)
    end
end
