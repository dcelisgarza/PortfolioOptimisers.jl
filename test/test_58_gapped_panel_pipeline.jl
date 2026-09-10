#=
A gapped point-in-time panel driven through a `Pipeline` and through a walk-forward fold,
issue #958 of map #955.

Map #667 built the whole downstream contract for a gapped panel -- the Coverage Universe, the
Investable Mask, the Held Gap, the reduce-and-expand identity -- and closed without ever
exercising the `Pipeline` path. `test_57_point_in_time_universe.jl` is that map's closing
verification, and it names no `Pipeline`. This file asks the question the other one skipped:
does the pipeline-and-fold path carry a gapped panel end to end?

It records two answers, and the second is the one that costs.

 1. **At the returns level it already does.** A `Pipeline` ending in a JuMP optimiser and one
    ending in a hierarchical optimiser both fit a gapped panel, a walk-forward over either
    runs every fold, a delisting inside a test window is reported as a Held Gap, and
    `assert_universe_aligned` never fires -- because reduce-and-expand keeps the result on the
    full universe, so the train and test universes are the same names either way.

 2. **At the price level it now does too, and ADR 0133 is why.** When this file was written
    `prices_to_returns` read a `NaN` price as `missing` and deleted every observation row
    holding one, so a panel with three gapped assets lost 60% of its history without a
    warning; and because that deletion was window-local, a walk-forward's train and test
    windows disagreed on the universe and `assert_universe_aligned` refused the fold by name.
    Issue #985 deleted that path. The conversion carries the gap unconditionally, so the
    whole clock and the whole universe survive every window and the fold runs with no
    fill and no panel, agreeing weight for weight with the returns level.

    The unbounded fill is still measured here, because it is the record of what the released
    remedy did and so the argument for issue #989's removal of `Imputer`. A caller reaches it
    now by declaring an all-listed calendar on the carrier -- ADR 0129's route, and
    bit-identical to the estimator that went -- and `PriceGapFill` reproduces it: it makes the
    fold run by inventing prices for assets that were not listed. The declaration is also the
    active mask, so asking for the unbounded fill *replaces* the honest panel a caller hands
    in beside it, and the defence that used to keep the invention out of the weights is gone
    with it: a delisted asset takes a quarter of the book with a panel and without one alike.
    Under the carrier's own Listing Span, which is what `price_ingestion` states, the same
    step invents nothing and both dead names hold exactly zero. The bound is the whole
    difference, and both halves are asserted below.

The panel-wide-versus-per-window question the map settled at charting is verified here rather
than re-decided: deriving the active mask once over the whole panel gives every fold the same
Coverage Universe, and so the same weights, as deriving it inside each training window.
=#
using Clarabel, Statistics, Dates

# Two identical programmes drift apart at the shipped solver defaults, and the drift is the
# solver's rather than the reduction's, so every parity comparison here runs one tightened
# Clarabel.
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             check_sol = (; allow_local = true, allow_almost = true),
             settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                             "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                             "tol_infeas_abs" => 1e-12, "tol_infeas_rel" => 1e-12))

# The fixture: all three span cases at once, hand-built exactly as map #667's own tests build
# theirs. `a` and `b` are priced throughout, so no window's universe is ever empty.
T58, N58 = 120, 5
nx58 = ["a", "b", "c", "d", "e"]
X58 = randn(StableRNG(958), T58, N58) ./ 100 .+ 0.0005
X58[1:30, 3] .= NaN            # c lists at observation 31: a LEADING run of gaps
X58[91:120, 4] .= NaN          # d delists at observation 91: a TRAILING run of gaps
X58[55:65, 5] .= NaN           # e is suspended and lists again: an INTERIOR gap

# The span rule the map settled: a leading run and a trailing run are inactive, an interior
# one is a Held Gap on an asset that is still listed. The estimation mask is the active mask
# intersected with finiteness.
amsk58 = trues(T58, N58)
amsk58[1:30, 3] .= false
amsk58[91:120, 4] .= false
emsk58 = amsk58 .& isfinite.(X58)

function panel58(a, e)
    return AssetPanel(;
                      pf = [NumericPanelField(; name = "mcap",
                                              vals = ones(size(a, 1), size(a, 2)))],
                      amsk = a, emsk = e)
end

rd58 = ReturnsResult(; nx = nx58, X = X58, pnl = panel58(amsk58, emsk58))

mr58 = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
hrp58 = HierarchicalRiskParity(; opt = HierarchicalOptimiser())
pipe_jump58 = Pipeline(; steps = (EmpiricalPrior(), mr58))
pipe_hier58 = Pipeline(; steps = (EmpiricalPrior(), hrp58))

# `IndexWalkForward(60, 20)` gives three rolling folds. Fold 1's training window ends at
# observation 60, which is inside e's suspension -- the case the map asked for by name.
iwf58 = IndexWalkForward(60, 20)
cv58 = split(iwf58, rd58)

@testset "A gapped panel through a Pipeline and a fold: the returns level" begin
    # The folds are the ones the file's prose describes.
    @test [(first(t), last(t)) for t in cv58.train_idx] == [(1, 60), (21, 80), (41, 100)]
    @test [(first(t), last(t)) for t in cv58.test_idx] == [(61, 80), (81, 100), (101, 120)]
    # Fold 1's training window ends inside e's interior gap.
    @test !isfinite(X58[60, 5])

    @testset "a Pipeline fits the whole panel, JuMP and hierarchical" begin
        for pipe in (pipe_jump58, pipe_hier58)
            res = fit(pipe, rd58)
            w = res.ctx.opt.w
            @test length(w) == N58
            @test all(isfinite, w)
            @test isapprox(sum(w), 1; rtol = 1e-8)
            # Over the whole panel only `a` and `b` are covered at every row, so every other
            # asset holds exactly zero rather than a small number.
            @test iszero(w[3]) && iszero(w[4]) && iszero(w[5])
            @test !iszero(w[1]) && !iszero(w[2])
        end
    end

    @testset "a walk-forward runs every fold" begin
        for est in (mr58, hrp58, pipe_jump58, pipe_hier58)
            p = cross_val_predict(est, rd58, iwf58)
            @test length(p.pred) == length(cv58.train_idx)
            for f in p.pred
                @test all(isfinite, f.rd.X)
                @test length(f.res.w) == N58
            end
        end
    end

    @testset "the Coverage Universe of each fold, and the weights outside it" begin
        # An asset is covered when its return is finite at every row of the window AND the
        # panel says it is active at every row of it.
        covered = [BitVector([1, 1, 0, 1, 0]),   # train 1:60  -- c unlisted, e suspended
                   BitVector([1, 1, 0, 1, 0]),   # train 21:80 -- c unlisted, e suspended
                   BitVector([1, 1, 1, 0, 0])]   # train 41:100 -- d delists, e suspended
        for (k, tr) in enumerate(cv58.train_idx)
            cm = PortfolioOptimisers.coverage_mask(view(X58, tr, :),
                                                   PortfolioOptimisers.port_opt_view(rd58.pnl,
                                                                                     tr,
                                                                                     1:N58))
            @test cm == covered[k]
            w = optimise(mr58,
                         ReturnsResult(; nx = nx58, X = X58[tr, :],
                                       pnl = panel58(amsk58[tr, :], emsk58[tr, :]))).w
            # Outside the Coverage Universe the weight is exactly zero, and inside it is not.
            @test all(iszero, view(w, .!cm))
            @test all(!iszero, view(w, cm))
        end
    end

    @testset "a delisting inside a test window is a Held Gap" begin
        # Fold 2 trains on 21:80, where d is live, and tests on 81:100, where d delists at
        # observation 91. The fold holds d and d has no return, so the pair is a Held Gap.
        tr, te = cv58.train_idx[2], cv58.test_idx[2]
        res = optimise(mr58,
                       ReturnsResult(; nx = nx58, X = X58[tr, :],
                                     pnl = panel58(amsk58[tr, :], emsk58[tr, :])))
        @test !iszero(res.w[4])
        @test @test_logs (:warn,) match_mode=:any predict(res, rd58, te) isa Any
        # The pair contributes zero rather than poisoning the series.
        pred = with_logger(NullLogger()) do
            return predict(res, rd58, te)
        end
        @test all(isfinite, pred.rd.X)
        # Under `strict` the same fold refuses by name instead of warning.
        @test_throws ArgumentError predict(res, rd58, te; strict = true)

        # Folds 1 and 3 hold nothing that goes missing in their test window, so they are
        # silent: fold 1 holds no e, and fold 3 holds no d.
        for k in (1, 3)
            trk, tek = cv58.train_idx[k], cv58.test_idx[k]
            resk = optimise(mr58,
                            ReturnsResult(; nx = nx58, X = X58[trk, :],
                                          pnl = panel58(amsk58[trk, :], emsk58[trk, :])))
            @test predict(resk, rd58, tek; strict = true) isa Any
        end
    end

    @testset "assert_universe_aligned never fires at the returns level" begin
        # Reduce-and-expand returns the fitted result on the FULL universe, so the pipeline's
        # training universe is every name and the test window's is the same. The check the
        # fit/apply contract exists for has nothing to catch here.
        for (tr, te) in zip(cv58.train_idx, cv58.test_idx)
            res = fit(pipe_jump58, PortfolioOptimisers.pipeline_data_view(rd58, tr))
            @test res.ctx.returns.nx == nx58
            @test PortfolioOptimisers.assert_universe_aligned(res,
                                                              PortfolioOptimisers.pipeline_data_view(rd58,
                                                                                                     te)) ===
                  nothing
        end
    end
end

# The span rule applied inside one window, which is what a per-window derivation would do: a
# gap that runs to either edge of the window reads as that window's listing or delisting.
function span_amsk58(Xw)
    Tw, Nw = size(Xw)
    a = trues(Tw, Nw)
    for j in 1:Nw
        col = view(Xw, :, j)
        f = findfirst(isfinite, col)
        if isnothing(f)
            a[:, j] .= false
        else
            a[1:(f - 1), j] .= false
            a[(findlast(isfinite, col) + 1):Tw, j] .= false
        end
    end
    return a
end

@testset "The active mask is derived once over the panel, and that leaks nothing" begin
    # The rule reproduces the hand-built mask when it reads the whole panel.
    @test span_amsk58(X58) == amsk58

    for (k, tr) in enumerate(cv58.train_idx)
        Xtr = X58[tr, :]
        apw, epw = amsk58[tr, :], emsk58[tr, :]
        aw = span_amsk58(Xtr)
        ew = aw .& isfinite.(Xtr)
        rd_pw = ReturnsResult(; nx = nx58, X = Xtr, pnl = panel58(apw, epw))
        rd_win = ReturnsResult(; nx = nx58, X = Xtr, pnl = panel58(aw, ew))
        # Fold 1's window ends inside e's suspension, so the two masks genuinely disagree
        # there: panel-wide says e is still listed, per-window reads a delisting.
        @test (aw != apw) == (k == 1)
        # They nonetheless agree on the Coverage Universe, because window-local finiteness
        # already excludes the column either way -- which is why the fits are identical.
        @test PortfolioOptimisers.coverage_mask(Xtr, rd_pw.pnl) ==
              PortfolioOptimisers.coverage_mask(Xtr, rd_win.pnl)
        @test optimise(mr58, rd_pw).w == optimise(mr58, rd_win).w
        @test optimise(hrp58, rd_pw).w == optimise(hrp58, rd_win).w
        # And the fit does not read the mask at all once finiteness has spoken: the same
        # window with no panel reaches the same weights.
        @test optimise(mr58, ReturnsResult(; nx = nx58, X = Xtr)).w ==
              optimise(mr58, rd_pw).w
    end

    # The two derivations differ where the map said they would: a fold's TEST window. Fold 1
    # tests on 61:80, which opens inside e's suspension. Panel-wide reports a Held Gap on a
    # listed asset; per-window would call the column dead and drop it silently.
    te = cv58.test_idx[1]
    @test count(view(amsk58[te, :], :, 5)) == length(te)
    @test count(view(span_amsk58(X58[te, :]), :, 5)) == length(te) - 5
    # The other two test windows agree, because neither opens or closes inside a gap.
    for k in (2, 3)
        tek = cv58.test_idx[k]
        @test span_amsk58(X58[tek, :]) == amsk58[tek, :]
    end
end

# The price level. The same three span cases, cumulated per column on that column's own
# finite returns so that a gap never poisons the rest of the column, then punched back in.
P58 = Matrix{Float64}(undef, T58 + 1, N58)
for j in 1:N58
    P58[1, j] = 100.0
    for t in 1:T58
        P58[t + 1, j] = P58[t, j] * (1 + (isfinite(X58[t, j]) ? X58[t, j] : 0.0))
    end
end
P58[1:31, 3] .= NaN
P58[92:end, 4] .= NaN
P58[56:66, 5] .= NaN
Pta58 = TimeArray(Date(2020, 1, 1) .+ Day.(0:T58), P58, nx58)

amskP58 = trues(T58 + 1, N58)
amskP58[1:31, 3] .= false
amskP58[92:end, 4] .= false
pnlP58 = panel58(amskP58, amskP58 .& isfinite.(P58))

pipe_price58 = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), mr58))
# The unbounded fill, said the way ADR 0130 leaves open: a caller's declared calendar saying
# every asset is listed at every observation, which bounds the fill by nothing at all. The
# reduction is the median of the observed training prices, which is what the removed estimator
# defaulted to, so the numbers below are the ones it produced.
pipe_fix58 = Pipeline(;
                      steps = (MissingDataFilter(), PriceGapFill(; fill = MedianValue()),
                               PricesToReturns(), EmpiricalPrior(), mr58))
all_listed58 = trues(T58 + 1, N58)

@testset "The price level carries a gapped panel" begin
    @testset "prices_to_returns deletes nothing, in every window" begin
        # ADR 0133. The conversion computes a return and nothing else: no observation row and
        # no asset column is deleted, so the whole clock and the whole universe survive.
        r = prices_to_returns(Pta58)
        @test r.nx == nx58
        @test size(r.X) == (T58, N58)
        @test !all(isfinite, r.X)

        # There is nothing window-local left to disagree about: every window keeps every row
        # and every name, whatever the gaps inside it are doing.
        for w in (1:61, 61:81, 21:81, 81:101, 41:101, 101:121)
            rw = prices_to_returns(TimeArray(TimeSeries.timestamp(Pta58)[w], P58[w, :],
                                             nx58))
            @test rw.nx == nx58
            @test size(rw.X, 1) == length(w) - 1
        end
    end

    @testset "the walk-forward runs, with a panel and without one" begin
        # The universe no longer moves between the train and the test window, so the fit/apply
        # contract at `src/23_Pipeline/03_Pipeline.jl:794` has nothing to catch.
        for pr in (PricesResult(; X = Pta58, pnl = pnlP58), PricesResult(; X = Pta58))
            @test length(cross_val_predict(pipe_price58, pr, iwf58).pred) == 3
        end
    end

    @testset "the unbounded fill runs, and the declaration that buys it costs the panel" begin
        # An all-listed calendar is what the released remedy amounted to: it bounds the fill
        # by nothing, so every gap is filled and every fold runs.
        pr_p = PricesResult(; X = Pta58, pnl = pnlP58, span = all_listed58)
        pr_n = PricesResult(; X = Pta58, span = all_listed58)
        pp = cross_val_predict(pipe_fix58, pr_p, iwf58)
        pn = cross_val_predict(pipe_fix58, pr_n, iwf58)
        @test length(pp.pred) == length(pn.pred) == 3

        # It fills them by inventing a price. `c` is unlisted until observation 31 and gets
        # its median training price, so it carries a return over a span in which it did not
        # trade.
        res = fit(pipe_fix58, pr_p)
        @test size(res.ctx.returns.X) == (T58, N58)
        @test all(isfinite, res.ctx.returns.X)
        jc = findfirst(==("c"), res.ctx.returns.nx)
        @test all(iszero, view(res.ctx.returns.X, 1:30, jc))

        # Issue #989, and the reason the estimator that used to do this had to go. The only
        # way to ask for an unbounded fill is to declare that every asset is listed at every
        # observation -- and that declaration IS the active mask (ADR 0129, ADR 0132), so it
        # replaces the honest panel the caller also handed in. The defence the caller was
        # relying on is the very thing the request threw away: the two runs are identical.
        @test all(res.ctx.returns.pnl.amsk)
        for k in 1:3
            @test pp.pred[k].res.w == pn.pred[k].res.w
        end
        # So the fabricated prices are indistinguishable from real ones and the optimiser
        # buys them, panel or no panel -- a delisted asset takes a quarter of the book.
        @test !iszero(pp.pred[1].res.w[3])
        @test !iszero(pp.pred[2].res.w[3])
        @test pp.pred[3].res.w[4] > 0.2
        @test pn.pred[3].res.w[4] > 0.2
    end

    @testset "the filled listing boundary is a fabricated return, issue #964" begin
        # The fill is the asset's MEDIAN training price, which is not the price it listed at,
        # so the first real observation shows up as a jump. That jump lands on `c`'s first
        # ACTIVE row, where the panel says the asset is live and the return is finite -- so
        # neither mask filters it, and a window opening there admits `c` carrying it.
        res = fit(pipe_fix58, PricesResult(; X = Pta58, pnl = pnlP58, span = all_listed58))
        rr = res.ctx.returns
        jc = findfirst(==("c"), rr.nx)
        # The all-listed declaration is the active mask, so `c` reads as live from the first
        # observation and no mask filters anything.
        @test findfirst(view(rr.pnl.amsk, :, jc)) == 1
        t0 = 31                                # c's first genuinely priced observation
        @test rr.pnl.emsk[t0, jc]
        @test rr.X[t0, jc] < -0.1              # c listed at ~100.8, filled at ~115.9

        w = t0:(t0 + 59)
        rdw = ReturnsResult(; nx = rr.nx, X = rr.X[w, :],
                            pnl = PortfolioOptimisers.port_opt_view(rr.pnl, w, 1:N58))
        got = optimise(mr58, rdw).w
        # The oracle drops the fabricated return, which is all that separates the two.
        Xo = copy(rr.X[w, :])
        Xo[1, jc] = NaN
        ref = optimise(mr58,
                       ReturnsResult(; nx = rr.nx, X = Xo,
                                     pnl = PortfolioOptimisers.port_opt_view(rr.pnl, w,
                                                                             1:N58))).w
        @test !iszero(got[jc])
        @test iszero(ref[jc])

        # Issue #989. The fabrication is the *calendar's*, not the step's: bound the same
        # step by the Listing Span the price panel itself states, and `c`'s leading run is
        # outside its listing, so nothing is written there and no boundary jump exists.
        # Only `e`'s suspension, a Held Gap inside a listing, is filled.
        bounded = fit(Pipeline(;
                               steps = (MissingDataFilter(),
                                        PriceGapFill(; fill = MedianValue()),
                                        PricesToReturns(), EmpiricalPrior(), mr58)),
                      PricesResult(; X = Pta58, pnl = pnlP58, span = listing_span(P58)))
        rb = bounded.ctx.returns
        @test all(!isfinite, view(rb.X, 1:31, jc))
        je = findfirst(==("e"), rb.nx)
        @test all(isfinite, view(rb.X, 55:66, je))
        jd = findfirst(==("d"), rb.nx)
        @test all(!isfinite, view(rb.X, 91:T58, jd))
    end
end

@testset "A carried gap needs no fill and no panel" begin
    # The gap reaches the returns instead of deleting the observation that holds it, and
    # every consumer downstream already handles one. This is the map's destination at the
    # price level, and after ADR 0133 it is the only path.
    pipe_carry58 = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), mr58))

    @testset "nothing is deleted, and the gap stays where it is" begin
        got = prices_to_returns(PricesResult(; X = Pta58, pnl = pnlP58))
        # The whole clock and the whole universe survive.
        @test got.nx == nx58
        @test size(got.X) == (T58, N58)
        # A run of `k` gapped prices makes exactly the `k + 1` returns that read one of them,
        # and the two ungapped assets are untouched.
        @test all(isfinite, view(got.X, :, 1))
        @test all(isfinite, view(got.X, :, 2))
        @test findall(!isfinite, view(got.X, :, 3)) == collect(1:31)
        @test findall(!isfinite, view(got.X, :, 4)) == collect(91:T58)
        @test findall(!isfinite, view(got.X, :, 5)) == collect(55:66)
        # The panel is carried and sliced to the returns.
        @test size(got.pnl.amsk) == size(got.pnl.emsk) == size(got.X)
    end

    @testset "the walk-forward runs, and the panel changes nothing" begin
        # `assert_universe_aligned` has nothing to catch: every window keeps every asset.
        pp = cross_val_predict(pipe_carry58, PricesResult(; X = Pta58, pnl = pnlP58), iwf58)
        pn = cross_val_predict(pipe_carry58, PricesResult(; X = Pta58), iwf58)
        @test length(pp.pred) == length(pn.pred) == 3
        for k in 1:3
            # Carrying the gap removes the look-ahead at the source, so the mask has nothing
            # left to defend against and the two runs agree exactly. Under an unbounded fill
            # they do not: there a delisted asset takes 0.247 of the book without a panel.
            @test pp.pred[k].res.w == pn.pred[k].res.w
        end
        # An asset the fold cannot estimate holds exactly zero, with or without the panel.
        @test iszero(pp.pred[1].res.w[3]) && iszero(pn.pred[1].res.w[3])
        @test iszero(pp.pred[3].res.w[4]) && iszero(pn.pred[3].res.w[4])
    end

    @testset "the carrying pipeline is the plain one" begin
        # There is no flag, so `pipe_carry58` and `pipe_price58` are the same programme, and
        # a gapless table is unaffected by any of it.
        @test :nan_to_missing ∉ fieldnames(PricesToReturns)
        pc = cross_val_predict(pipe_carry58, PricesResult(; X = Pta58), iwf58)
        pp = cross_val_predict(pipe_price58, PricesResult(; X = Pta58), iwf58)
        for k in 1:3
            @test pc.pred[k].res.w == pp.pred[k].res.w
        end
    end
end
