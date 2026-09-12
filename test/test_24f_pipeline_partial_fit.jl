#=
The Pipeline takes the online step, issue #1022, against the decision of #872 (ADR 0142).

A Pipeline is a host: `partial_fit!(pipe, data)` walks the steps in order and folds each
block into the row owner — the prior step, else the optimiser step — and `fit(pipe)` with no
data reads the fitted `PipelineResult` out. The steps before the owner fall into three
classes. A row-local step (`PricesToReturns`, `PriceGapFill` with a carried price,
`MissingDataFilter` at `row_thr = 1`) folds and emits the transformed rows; a universe-only
step (a selector, the column filter) folds nothing and is refitted at the read-out over the
owner's rows, its universe applied as a view of the owner's state; a window-valued step is
refused at warm-up by name, and `Online(pipe)` is the declared refit that admits it.

Seven identities are the contract, ADR 0142's list: the hand-stepped fit against the batch
fit step for step; each row-local form alone against its batch fit, exactly; a re-selection
between two steps; the map's closing test through the Pipeline, over a gapped panel with a
listing and a delisting, through a JuMP optimiser, a hierarchical one and a schedule after a
prior step; the capped refit against the rolling batch walk-forward; every refusal by
message; and the search door.

The fixture is synthetic: the identities are structural, and a solver run over six assets is
what keeps the JuMP families cheap.
=#
@testset "Pipeline partial fit: a Pipeline is a host of the online step" begin
    using Test, PortfolioOptimisers, Clarabel, StableRNGs, Statistics, Dates, TimeSeries,
          FLoops
    po = PortfolioOptimisers
    rng = StableRNG(20260912)
    T, N = 160, 6
    P = 100 .* exp.(cumsum(randn(rng, T, N) ./ 100 .+ 0.0003; dims = 1))
    # A listing at row 31, a delisting after row 130, and three gaps inside the span.
    P[1:30, 3] .= NaN
    P[131:end, 5] .= NaN
    P[70, 1] = NaN
    P[95:97, 4] .= NaN
    P[62, 2] = NaN
    ts = Date(2024, 1, 1) .+ Day.(0:(T - 1))
    nx = ["A$i" for i in 1:N]
    pr = price_ingestion(PriceIngestion(), TimeArray(ts, P, nx))
    # The returns twin: the filled panel converted, so its estimation mask is its active
    # mask and a prior folds it exactly as the optimiser's step does.
    rd = apply_preprocessing(PricesToReturns(),
                             apply_preprocessing(fit_preprocessing(PriceGapFill(), pr), pr))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = false, allow_almost = false),
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))
    mr = MeanRisk(; opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv))
    hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = EmpiricalPrior()))
    w, t, p = 60, 20, 3
    batch_cv = IndexWalkForward(w, t; purged_size = p, expand_train = true)
    online_cv = IndexWalkForward(w, t; purged_size = p, ff = OnlineStep())
    rows(d, i) = po.pipeline_data_view(d, i)
    weight_tol(pipe) = isa(pipe.steps[end], po.JuMPOptimisationEstimator) ? 1e-5 : 1e-10
    same_carrier(a, b) = isequal(a.nx, b.nx) &&
                         isequal(a.ts, b.ts) &&
                         isequal(Matrix(a.X), Matrix(b.X)) &&
                         isequal(a.pnl.amsk, b.pnl.amsk) &&
                         isequal(a.pnl.emsk, b.pnl.emsk)
    function stepped(pipe, data, blocks)
        pipe = po.update_online_estimator(pipe)
        for b in blocks
            pipe = po.partial_fit!(pipe, rows(data, b))
        end
        return pipe
    end
    blocks = [1:60, 61:61, 62:75, 76:80]

    @testset "Stepped 1:t then fit(pipe) equals fit(pipe, data[1:t]), step for step" begin
        prices_pipe = Pipeline(;
                               steps = (PriceGapFill(), MissingDataFilter(; col_thr = 0.05),
                                        PricesToReturns(), EmpiricalPrior(), mr))
        returns_pipe = Pipeline(;
                                steps = (ScoreSelector(; score = MeanReturn(),
                                                       rule = RankRule(; best = 4)),
                                         EmpiricalPrior(), hrp))
        # A prior-less head owns the rows itself, and an optimiser owns them when no prior
        # step precedes it.
        head_pipe = Pipeline(; steps = (PricesToReturns(), EqualWeighted()))
        opt_pipe = Pipeline(; steps = (PriceGapFill(), PricesToReturns(), mr))
        for (pipe, data) in
            ((prices_pipe, pr), (returns_pipe, rd), (head_pipe, pr), (opt_pipe, pr))
            spipe = stepped(pipe, data, blocks)
            o = fit(spipe)
            b = fit(pipe, rows(data, 1:80))
            @test isapprox(o.w, b.w; atol = weight_tol(pipe))
            @test o.ctx.returns.nx == b.ctx.returns.nx
            @test isequal(Matrix(o.ctx.returns.X), Matrix(b.ctx.returns.X))
            for (ro, rb) in zip(o.results, b.results)
                if isa(rb, PriceGapFillResult)
                    @test ro.nx == rb.nx && ro.v == rb.v && ro.fill == rb.fill
                elseif isa(rb, Union{<:MissingDataFilterResult, <:AssetSelectorResult})
                    @test ro.nx == rb.nx
                elseif isa(rb, PricesToReturns)
                    @test isequal(ro, rb)
                elseif isa(rb, po.AbstractPriorResult)
                    @test isequal(ro.X, rb.X)
                    @test isapprox(ro.mu, rb.mu; atol = 1e-15)
                    @test isapprox(ro.sigma, rb.sigma; atol = 1e-15)
                end
            end
            # The read-out is pure: a second call answers the same, and a further step
            # moves it.
            @test isapprox(fit(spipe).w, o.w; atol = weight_tol(pipe))
            spipe2 = po.partial_fit!(spipe, rows(data, 81:90))
            @test isapprox(fit(spipe2).w, fit(pipe, rows(data, 1:90)).w;
                           atol = weight_tol(pipe))
        end
        # The public door reads a hand-stepped pipeline out and predicts on a window.
        spipe = stepped(prices_pipe, pr, blocks)
        pred = po.fit_and_predict(spipe, pr; test_idx = 84:100)
        @test isapprox(pred.res.w, fit(prices_pipe, rows(pr, 1:80)).w; atol = 1e-5)
        # The state and the Fold Context render only where set.
        @test !occursin("cache", sprint(show, MIME"text/plain"(), prices_pipe))
        @test occursin("cache", sprint(show, MIME"text/plain"(), spipe))
        # And a Pipeline with no owner has nothing to fold into.
        @test_throws ArgumentError po.partial_fit!(Pipeline(; steps = (PricesToReturns(),)),
                                                   rows(pr, 1:10))
        @test_throws ArgumentError fit(prices_pipe)
    end

    @testset "Each row-local form alone against its batch fit, exactly" begin
        uneven = [1:10, 11:11, 12:21, 22:22, 23:30, 31:70, 71:100, 101:160]
        function online_rows(est, data, blocks)
            parts = []
            for b in blocks
                est, out = po.partial_fit_transform(est, rows(data, b))
                push!(parts, out)
            end
            return est, reduce(po.vcat_carrier_rows, parts)
        end
        for ptr in
            (PricesToReturns(), PricesToReturns(; gap_return_alg = CatchUpGapReturn()),
             PricesToReturns(; padding = true, gap_return_alg = CatchUpGapReturn()),
             PricesToReturns(; ret_method = :log, gap_return_alg = CatchUpGapReturn()))
            est, online = online_rows(ptr, pr, uneven)
            @test same_carrier(online, apply_preprocessing(ptr, pr))
            @test isequal(fit_preprocessing(est), ptr)
            # `partial_fit!` is the first half of the transform.
            c1 = po.partial_fit!(ptr, rows(pr, 1:10)).cache
            c2 = po.partial_fit_transform(ptr, rows(pr, 1:10))[1].cache
            @test isequal(values(c1.tail.X), values(c2.tail.X)) &&
                  isequal(c1.anchor, c2.anchor)
        end
        est, online = online_rows(PriceGapFill(), pr, uneven)
        bres = fit_preprocessing(PriceGapFill(), pr)
        @test isequal(values(online.X), values(apply_preprocessing(bres, pr).X))
        @test fit_preprocessing(est).nx == bres.nx
        @test fit_preprocessing(est).v == bres.v
        for col_thr in (0.0, 0.05, 0.5)
            est, online = online_rows(MissingDataFilter(; col_thr = col_thr), pr, uneven)
            @test isequal(values(online.X), values(pr.X))
            @test fit_preprocessing(est).nx ==
                  fit_preprocessing(MissingDataFilter(; col_thr = col_thr), pr).nx
        end
        # A state merges by block: two halves fold to the whole.
        a = po.partial_fit!(PricesToReturns(; gap_return_alg = CatchUpGapReturn()),
                            rows(pr, 1:70))
        b = po.partial_fit!(PricesToReturns(; gap_return_alg = CatchUpGapReturn()),
                            rows(pr, 71:160))
        m = po.merge_states(a.cache, b.cache)
        @test isequal(m.anchor, po.partial_fit!(a, rows(pr, 71:160)).cache.anchor)
        @test isequal(po.merge_states(po.partial_fit!(PriceGapFill(), rows(pr, 1:70)).cache,
                                      po.partial_fit!(PriceGapFill(), rows(pr, 71:160)).cache).v,
                      po.partial_fit!(PriceGapFill(), pr).cache.v)
        # The two window-valued configurations and a caller's algorithm answer `false`.
        @test !po.supports_partial_fit(PriceGapFill(; fill = MeanValue()))
        @test !po.supports_partial_fit(MissingDataFilter(; row_thr = 0.5))
        @test po.supports_partial_fit(PriceGapFill()) &&
              po.supports_partial_fit(MissingDataFilter()) &&
              po.supports_partial_fit(PricesToReturns())
        struct SpreadGapReturn <: po.AbstractGapReturnAlgorithm end
        @test !po.supports_partial_fit(PricesToReturns(;
                                                       gap_return_alg = SpreadGapReturn()))
        @test_throws ArgumentError po.partial_fit!(PriceGapFill(; fill = MeanValue()),
                                                   rows(pr, 1:10))
        @test_throws ArgumentError po.partial_fit!(MissingDataFilter(; row_thr = 0.5),
                                                   rows(pr, 1:10))
        # A block that changes the universe is refused by name.
        @test_throws ArgumentError po.partial_fit!(po.partial_fit!(PriceGapFill(),
                                                                   rows(pr, 1:10)),
                                                   po.port_opt_view(pr, 11:20, 1:3))
    end

    @testset "A carrier with factors and a benchmark converts online as batch, catch-up included" begin
        F = TimeArray(ts, 50 .* exp.(cumsum(randn(StableRNG(2), T, 2) ./ 100; dims = 1)),
                      ["F1", "F2"])
        Fv = values(F)
        Fv[40:42, 1] .= NaN
        F = TimeArray(ts, Fv, ["F1", "F2"])
        Bv = 80 .* exp.(cumsum(randn(StableRNG(3), T, 1) ./ 100; dims = 1))
        Bv[77, 1] = NaN
        B1 = TimeArray(ts, Bv, ["B"])
        BN = TimeArray(ts, P .* 0.5, ["B$i" for i in 1:N])
        uneven = [1:39, 40:41, 42:76, 77:77, 78:160]
        for B in (B1, BN)
            prfb = price_ingestion(PriceIngestion(), TimeArray(ts, P, nx); F = F, B = B)
            ptr = PricesToReturns(; gap_return_alg = CatchUpGapReturn())
            est = ptr
            parts = []
            for b in uneven
                est, out = po.partial_fit_transform(est, rows(prfb, b))
                push!(parts, out)
            end
            online = reduce(po.vcat_carrier_rows, parts)
            batch = apply_preprocessing(ptr, prfb)
            @test same_carrier(online, batch)
            @test isequal(Matrix(online.F), Matrix(batch.F))
            @test isequal(online.nf, batch.nf) && isequal(online.nb, batch.nb)
            @test isequal(if isa(online.B, AbstractVector)
                              collect(online.B)
                          else
                              Matrix(online.B)
                          end,
                          isa(batch.B, AbstractVector) ? collect(batch.B) : Matrix(batch.B))
            @test isequal(fit_preprocessing(est), ptr)
        end
        # A plain conversion's state merges too, and a static panel is pinned and kept.
        a = po.partial_fit!(PricesToReturns(), rows(pr, 1:70)).cache
        b = po.partial_fit!(PricesToReturns(), rows(pr, 71:160)).cache
        @test isnothing(po.merge_states(a, b).anchor)
        static = AssetPanel(; pf = [NumericPanelField(; name = "size", vals = ones(N))])
        @test po.vcat_panel_rows(static, static) === static
        @test_throws ArgumentError po.vcat_panel_rows(static, nothing)
        @test isnothing(po.vcat_panel_rows(nothing, nothing))
        rds = ReturnsResult(; nx = nx, X = rd.X[1:10, :], pnl = static)
        @test po.vcat_carrier_rows(rds, rds).pnl === static
        @test_throws ArgumentError po.vcat_carrier_rows(rds, rows(rd, 1:10))
        @test_throws ArgumentError po.vcat_carrier_rows(rd,
                                                        ReturnsResult(; nx = nx,
                                                                      X = rd.X[1:10, :]))
    end

    @testset "A re-selection between two steps: the state is viewed, never re-sliced" begin
        # Four assets. Over 1:40 the means rank A > B > C > D, so the selector keeps {A, B};
        # row 41 lifts C above B, so over 1:41 it keeps {A, C}. D carries every gap, so the
        # column filter keeps it at 40 rows and drops it at 41.
        T4 = 41
        R = fill(0.001, T4, 4)
        R[:, 1] .= 0.004
        R[:, 2] .= 0.003
        R[:, 3] .= 0.002
        R[41, 3] = 0.1
        R .+= randn(StableRNG(4), T4, 4) ./ 10_000
        P4 = 100 .* cumprod(1 .+ R; dims = 1)
        P4[38:41, 4] .= NaN
        ts4 = Date(2024, 1, 1) .+ Day.(0:(T4 - 1))
        pr4 = price_ingestion(PriceIngestion(), TimeArray(ts4, P4, ["A", "B", "C", "D"]))
        sel = ScoreSelector(; score = MeanReturn(), rule = RankRule(; best = 2))
        pipe4 = Pipeline(;
                         steps = (PriceGapFill(), MissingDataFilter(; col_thr = 0.09),
                                  PricesToReturns(), sel, EmpiricalPrior(),
                                  InverseVolatility()))
        spipe = stepped(pipe4, pr4, [1:40])
        o40 = fit(spipe)
        b40 = fit(pipe4, rows(pr4, 1:40))
        @test o40.results[2].nx == b40.results[2].nx == [:A, :B, :C, :D]
        @test o40.results[4].nx == b40.results[4].nx == ["A", "B"]
        @test isapprox(o40.w, b40.w; atol = 1e-10)
        spipe = po.partial_fit!(spipe, rows(pr4, 41:41))
        o41 = fit(spipe)
        b41 = fit(pipe4, rows(pr4, 1:41))
        @test o41.results[2].nx == b41.results[2].nx == [:A, :B, :C]
        @test o41.results[4].nx == b41.results[4].nx == ["A", "C"]
        @test isapprox(o41.w, b41.w; atol = 1e-10)
        @test o41.ctx.returns.nx == b41.ctx.returns.nx == ["A", "C"]
        @test isequal(o41.ctx.prior.X, b41.ctx.prior.X)
        # The owner's state holds the input universe at warm-up width throughout, and the
        # Pipeline's Fold Context pins the same names.
        pe = spipe.steps[5]
        @test length(spipe.cache.nx) == 4
        @test size(po.sample_buffer(po.prior_returns_buffer(pe)), 2) == 4
        @test length(po.returns_result(spipe.cache, po.prior_returns_buffer(pe)).nx) == 4
    end

    @testset "The closing identity through the Pipeline: online equals batch, fold for fold" begin
        prep = (PriceGapFill(), PricesToReturns())
        sched = TimeDependent([iseven(i) ? mr : hrp for i in 1:n_splits(online_cv, pr)])
        pipes = (Pipeline(; steps = (prep..., EmpiricalPrior(), mr)),
                 Pipeline(; steps = (prep..., EmpiricalPrior(), hrp)),
                 Pipeline(; steps = (prep..., EmpiricalPrior(), sched)),
                 Pipeline(; steps = (prep..., mr)),
                 Pipeline(;
                          steps = (prep...,
                                   ScoreSelector(; score = MeanReturn(),
                                                 rule = RankRule(; best = 5)),
                                   EmpiricalPrior(), hrp)))
        for pipe in pipes
            b = cross_val_predict(pipe, pr, batch_cv)
            o = cross_val_predict(pipe, pr, online_cv)
            @test length(o.pred) == length(b.pred) == n_splits(online_cv, pr)
            for (pb, po_) in zip(b.pred, o.pred)
                @test isapprox(po_.res.w, pb.res.w; atol = 1e-5)
                @test po_.res.imsk == pb.res.imsk
                @test isapprox(po_.rd.X, pb.rd.X; atol = 1e-5)
            end
        end
        # A derived-slot step before the owner and wrapped steps run at the read-out as
        # batch runs them: a clustering step, a prior in a `PipelineStep`, and a callable
        # writing the constraints slot.
        wpipe = Pipeline(;
                         steps = (prep..., ClustersEstimator(),
                                  PipelineStep(; est = EmpiricalPrior(),
                                               reads = (:returns,), writes = :prior),
                                  PipelineStep(;
                                               est = ctx -> WeightBounds(;
                                                                         lb = zeros(length(ctx.returns.nx)),
                                                                         ub = fill(0.6,
                                                                                   length(ctx.returns.nx))),
                                               reads = (:returns,), writes = :constraints),
                                  hrp))
        # A clustering step reads every column, so it runs over a panel with no gap.
        prc = price_ingestion(PriceIngestion(),
                              TimeArray(ts,
                                        100 .*
                                        exp.(cumsum(randn(StableRNG(5), T, N) ./ 100;
                                                    dims = 1)), nx))
        b = cross_val_predict(wpipe, prc, batch_cv)
        o = cross_val_predict(wpipe, prc, online_cv)
        @test all(isapprox(po_.res.w, pb.res.w; atol = 1e-10)
                  for (pb, po_) in zip(b.pred, o.pred))
        @test all(maximum(po_.res.w) <= 0.6 + 1e-10 for po_ in o.pred)
        # The returns-level pipeline reaches the same identity, and so does the date form.
        rpipe = Pipeline(; steps = (EmpiricalPrior(), mr))
        b = cross_val_predict(rpipe, rd, batch_cv)
        o = cross_val_predict(rpipe, rd, online_cv)
        @test all(isapprox(po_.res.w, pb.res.w; atol = 1e-5)
                  for (pb, po_) in zip(b.pred, o.pred))
        bd = DateWalkForward(w, t; period = Day(1), purged_size = p, expand_train = true)
        od = DateWalkForward(w, t; period = Day(1), purged_size = p, ff = OnlineStep())
        b = cross_val_predict(pipes[2], pr, bd)
        o = cross_val_predict(pipes[2], pr, od)
        @test all(isapprox(po_.res.w, pb.res.w; atol = 1e-10)
                  for (pb, po_) in zip(b.pred, o.pred))
        # The fold's callback receives no window, and the read-out rebuilds the batch
        # fold's training window exactly.
        cvr = split(online_cv, pr)
        n = length(cvr.train_idx)
        seen = Int[]
        po.fold_loop(pipes[1], n, FLoops.SequentialEx(); rd = pr, train_idx = cvr.train_idx,
                     test_idx = cvr.test_idx, cv = online_cv) do fold
            push!(seen, fold.i)
            @test isnothing(fold.train)
            train = fit(pipes[1], rows(pr, cvr.train_idx[fold.i])).ctx.returns
            @test same_carrier(fit(fold.est).ctx.returns, train)
            return po.fit_and_predict(fold.est, fold.rd; test_idx = fold.test)
        end
        @test seen == 1:n
        # The multiple-randomised form views the data and folds the pipeline afresh per
        # path; the returns twin holds four assets in its Coverage Universe.
        mrb = MultipleRandomised(batch_cv; subset_size = 3, seed = 7)
        mro = MultipleRandomised(online_cv; subset_size = 3, seed = 7)
        rpipe_h = Pipeline(; steps = (EmpiricalPrior(), hrp))
        b = cross_val_predict(rpipe_h, rd, mrb)
        o = cross_val_predict(rpipe_h, rd, mro)
        @test length(b.pred) == length(o.pred) == 2
        for (pb, po_) in zip(b.pred, o.pred)
            @test all(isapprox(y.res.w, x.res.w; atol = 1e-10)
                      for (x, y) in zip(pb.pred, po_.pred))
        end
    end

    @testset "Online(pipe; max_history = w) is the rolling batch walk-forward; Online(pipe) is the host route" begin
        # A statistic fill is window-valued, which the host route refuses and the refit
        # route admits: every fold is a batch fit over the buffer.
        pipe = Pipeline(;
                        steps = (PriceGapFill(; fill = MeanValue()), PricesToReturns(),
                                 EmpiricalPrior(), hrp))
        rolling = IndexWalkForward(w + p, t; purged_size = p)
        capped = IndexWalkForward(w + p, t; purged_size = p, ff = OnlineStep())
        b = cross_val_predict(pipe, pr, rolling)
        o = cross_val_predict(po.Online(pipe; max_history = w), pr, capped)
        @test all(isapprox(po_.res.w, pb.res.w; atol = 1e-10)
                  for (pb, po_) in zip(b.pred, o.pred))
        b = cross_val_predict(pipe, pr, batch_cv)
        o = cross_val_predict(po.Online(pipe), pr, online_cv)
        @test all(isapprox(po_.res.w, pb.res.w; atol = 1e-10)
                  for (pb, po_) in zip(b.pred, o.pred))
        # The refit route equals the host route where both run.
        hpipe = Pipeline(;
                         steps = (PriceGapFill(), PricesToReturns(), EmpiricalPrior(), hrp))
        h = cross_val_predict(hpipe, pr, online_cv)
        o = cross_val_predict(po.Online(hpipe), pr, online_cv)
        @test all(isapprox(po_.res.w, ph.res.w; atol = 1e-10)
                  for (ph, po_) in zip(h.pred, o.pred))
        # And the returns level, and the multiple-randomised door.
        rpipe = Pipeline(; steps = (EmpiricalPrior(), hrp))
        b = cross_val_predict(rpipe, rd, rolling)
        o = cross_val_predict(po.Online(rpipe; max_history = w), rd, capped)
        @test all(isapprox(po_.res.w, pb.res.w; atol = 1e-10)
                  for (pb, po_) in zip(b.pred, o.pred))
        mro = MultipleRandomised(online_cv; subset_size = 3, seed = 7)
        b = cross_val_predict(rpipe, rd,
                              MultipleRandomised(batch_cv; subset_size = 3, seed = 7))
        o = cross_val_predict(po.Online(rpipe), rd, mro)
        for (pb, po_) in zip(b.pred, o.pred)
            @test all(isapprox(y.res.w, x.res.w; atol = 1e-10)
                      for (x, y) in zip(pb.pred, po_.pred))
        end
        # The buffer by hand: append, cap, read out.
        opipe = po.update_online_estimator(po.Online(pipe; max_history = 30))
        @test isa(opipe.cache, po.PipelineBufferState)
        @test_throws ArgumentError fit(opipe)
        opipe = po.partial_fit!(po.partial_fit!(opipe, rows(pr, 1:20)), rows(pr, 21:50))
        @test po.carrier_rows(opipe.cache.data) == 30
        @test isapprox(fit(opipe).w, fit(pipe, rows(pr, 21:50)).w; atol = 1e-10)
        half = po.partial_fit!(po.PipelineBufferState(), rows(pr, 1:20))
        @test isequal(values(po.merge_states(half,
                                             po.partial_fit!(po.PipelineBufferState(),
                                                             rows(pr, 21:50))).data.X),
                      values(rows(pr, 1:50).X))
        @test po.merge_states(half, po.PipelineBufferState()) === half
        @test_throws ArgumentError po.merge_states(half,
                                                   po.PipelineBufferState(;
                                                                          max_history = 3))
    end

    @testset "Every refusal, by message" begin
        function refusal(f)
            err = try
                f()
                nothing
            catch e
                e
            end
            @test isa(err, ArgumentError)
            return isnothing(err) ? "" : err.msg
        end
        prep = (PriceGapFill(), PricesToReturns())
        # A schedule that owns the rows.
        n = n_splits(online_cv, pr)
        sched = TimeDependent([isodd(i) ? mr : hrp for i in 1:n])
        msg = refusal(() -> cross_val_predict(Pipeline(; steps = (prep..., sched)), pr,
                                              online_cv))
        @test occursin("TimeDependent", msg) && occursin("prior step", msg)
        # With a prior step before it, the schedule composes (pinned above); a schedule on
        # the owner optimiser's prior does not.
        msg = refusal(() -> cross_val_predict(Pipeline(;
                                                       steps = (prep...,
                                                                MeanRisk(;
                                                                         opt = JuMPOptimiser(;
                                                                                             pe = TimeDependent(fill(EmpiricalPrior(),
                                                                                                                     n)),
                                                                                             slv = slv)))),
                                              pr, online_cv))
        @test occursin("TimeDependent", msg) && occursin("pe", msg)
        # A callable data-slot step.
        cal = PipelineStep(; est = ctx -> ctx.returns, reads = (:returns,),
                           writes = :returns)
        msg = refusal(() -> cross_val_predict(Pipeline(;
                                                       steps = (prep..., cal,
                                                                EmpiricalPrior(), hrp)), pr,
                                              online_cv))
        @test occursin("partial_fit_transform", msg) && occursin("Online(pipe)", msg)
        # The two window-valued configurations.
        for step in
            (PriceGapFill(; fill = MeanValue()), MissingDataFilter(; row_thr = 0.5))
            msg = refusal(() -> cross_val_predict(Pipeline(;
                                                           steps = (step, PricesToReturns(),
                                                                    EmpiricalPrior(), hrp)),
                                                  pr, online_cv))
            @test occursin("no online form", msg) && occursin("Online(pipe)", msg)
        end
        # A nested pipeline writing a data slot, and a caller's estimator without a form.
        nested = Pipeline(; steps = (PriceGapFill(), PricesToReturns()))
        msg = refusal(() -> po.assert_online_entry(Pipeline(;
                                                            steps = (nested,
                                                                     EmpiricalPrior(), hrp))))
        @test occursin("nested", msg)
        struct NoFormFilter <: po.AbstractReturnsPreprocessingEstimator end
        msg = refusal(() -> po.assert_online_entry(Pipeline(;
                                                            steps = (prep...,
                                                                     NoFormFilter(),
                                                                     EmpiricalPrior(), hrp))))
        @test occursin("NoFormFilter", msg) && occursin("no online form", msg)
        @test_throws ArgumentError po.partial_fit_transform(NoFormFilter(), rd)
        # An Online below an Online(pipe).
        msg = refusal(() -> cross_val_predict(po.Online(Pipeline(;
                                                                 steps = (prep...,
                                                                          po.Online(EmpiricalPrior()),
                                                                          hrp))), pr,
                                              online_cv))
        @test occursin("Online(pipe)", msg) && occursin("prior", msg)
        msg = refusal(() -> cross_val_predict(po.Online(Pipeline(;
                                                                 steps = (prep...,
                                                                          MeanRisk(;
                                                                                   opt = JuMPOptimiser(;
                                                                                                       pe = po.Online(EmpiricalPrior()),
                                                                                                       slv = slv))))),
                                              pr, online_cv))
        @test occursin("opt.opt.pe", msg)
        # An Online(pipe) under a batch scheme, and an Online step at a fold-less fit.
        hpipe = Pipeline(; steps = (prep..., EmpiricalPrior(), hrp))
        msg = refusal(() -> cross_val_predict(po.Online(hpipe), pr, batch_cv))
        @test occursin("no Fold Fit", msg)
        msg = refusal(() -> fit(Pipeline(;
                                         steps = (prep..., po.Online(EmpiricalPrior()),
                                                  hrp)), pr))
        @test occursin("warm-up", msg)
        # A state at entry, at the pipeline and at a step, naming the field.
        spipe = stepped(hpipe, pr, [1:60])
        msg = refusal(() -> cross_val_predict(spipe, pr, online_cv))
        @test occursin("`cache`", msg)
        msg = refusal(() -> cross_val_predict(Pipeline(;
                                                       steps = (prep..., spipe.steps[3],
                                                                hrp)), pr, online_cv))
        @test occursin("prior.cache", msg)
        msg = refusal(() -> cross_val_predict(po.Online(Pipeline(;
                                                                 steps = (prep...,
                                                                          spipe.steps[3],
                                                                          hrp))), pr,
                                              online_cv))
        @test occursin("prior.cache", msg)
        # A prices pipeline whose rows never become returns.
        msg = refusal(() -> po.partial_fit!(Pipeline(; steps = (PriceGapFill(), hrp)),
                                            rows(pr, 1:10)))
        @test occursin("PricesToReturns", msg)
        # A holdout and a finite allocation keep their refusals.
        @test_throws ArgumentError cross_val_predict(Pipeline(;
                                                              steps = (TrainTestSplit(;
                                                                                      test_size = 0.2),
                                                                       EmpiricalPrior(),
                                                                       hrp)), rd, online_cv)
    end

    @testset "The search door picks the batch candidate" begin
        pipe = Pipeline(;
                        steps = (PriceGapFill(), PricesToReturns(), EmpiricalPrior(),
                                 MeanRisk(;
                                          opt = JuMPOptimiser(; pe = EmpiricalPrior(),
                                                              slv = slv, l1 = 0.0005))))
        grid = ["opt.opt.l1" => [0.0005, 0.005]]
        r = ConditionalValueatRisk()
        b = search_cross_validation(pipe,
                                    GridSearchCrossValidation(grid; cv = batch_cv, r = r),
                                    pr)
        o = search_cross_validation(pipe,
                                    GridSearchCrossValidation(grid; cv = online_cv, r = r),
                                    pr)
        @test size(o.test_scores) == size(b.test_scores) == (n_splits(online_cv, pr), 2)
        @test isapprox(o.test_scores, b.test_scores; atol = 1e-6)
        @test o.idx == b.idx
        rs = search_cross_validation(pipe,
                                     RandomisedSearchCrossValidation(grid; n_iter = 2,
                                                                     seed = 3,
                                                                     cv = online_cv, r = r),
                                     pr)
        @test isa(rs, SearchCrossValidationResult)
        # A warm pipeline is refused once, before the grid.
        spipe = stepped(pipe, pr, [1:60])
        @test_throws ArgumentError search_cross_validation(spipe,
                                                           GridSearchCrossValidation(grid;
                                                                                     cv = online_cv,
                                                                                     r = r),
                                                           pr)
    end
end
