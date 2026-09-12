#=
A Result resumes an online run, issue #1025, against the decision of #1018 (ADR 0144).

An online walk-forward's `MultiPeriodPredictionResult` carries the threaded estimator in
`opt`, folded through the last training end, and `Resume(res)` hands the Result back to the
fold loop over the full history extended: the folds the Result holds are skipped, the ordinary
delta is folded into a copy of the state, and the loop continues from the fold after them.
The oracle is the one-shot run, `vcat(old.pred, new.pred)` equals
`cross_val_predict(mr, rd_{T+k}, cv).pred` fold for fold, pinned here for a JuMP head with a
turnover term, a hierarchical head, a prior-less head, a capped prior, the date form, a
Weight Drift on the scheme, a chain of resumes and a Pipeline host. Around the identity sit
the re-entry refusals, the value semantics of a Result, `vcat`, and the stdlib round trip.

The fixture is synthetic, because the identities are structural and a solver run over six
assets is what keeps the JuMP families cheap.
=#
@testset "Resume: a Result re-enters the fold loop from the folds it holds" begin
    using Test, PortfolioOptimisers, Clarabel, StableRNGs, Statistics, Dates, LinearAlgebra,
          TimeSeries, Serialization, FLoops
    po = PortfolioOptimisers
    rng = StableRNG(20260912)
    T, N = 200, 6
    X = randn(rng, T, N) ./ 100 .+ 0.0003
    nx = ["A$i" for i in 1:N]
    ts = Date(2024, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = X, ts = ts)
    # A listing at row 31 and a delisting at row 161, expressed by the panel's masks.
    amsk = trues(T, N)
    amsk[1:30, 3] .= false
    amsk[161:end, 5] .= false
    Xg = copy(X)
    Xg[.!amsk] .= NaN
    rdg = ReturnsResult(; nx = nx, X = Xg, ts = ts,
                        pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk)))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = false, allow_almost = false),
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))
    jopt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)
    mr = MeanRisk(; opt = jopt)
    tn = MeanRisk(;
                  opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                      tn = Turnover(; w = fill(inv(N), N), val = 0.5)))
    hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = EmpiricalPrior()))
    ew = EqualWeighted()
    w, t, p = 60, 20, 3
    rows(r, i) = po.port_opt_view(r, i, :)
    weights(res) = [pr.res.w for pr in res.pred]
    weight_tol(opt) = isa(opt, po.JuMPOptimisationEstimator) ? 1e-5 : 1e-10
    online_cv = IndexWalkForward(w, t; purged_size = p, ff = OnlineStep())
    batch_cv = IndexWalkForward(w, t; purged_size = p, expand_train = true)
    # The old history ends at a fold boundary, so its last fold is full.
    T0 = 140
    rd_T = rows(rd, 1:T0)
    rdg_T = rows(rdg, 1:T0)
    function same_weights(stack, one, opt)
        @test length(stack.pred) == length(one.pred)
        for (a, b) in zip(stack.pred, one.pred)
            @test isapprox(a.res.w, b.res.w; atol = weight_tol(opt))
            @test a.rd.ts == b.rd.ts
            @test a.res.imsk == b.res.imsk
        end
        return nothing
    end
    msg(f) =
        try
            f()
            ""
        catch e
            sprint(showerror, e)
        end

    @testset "The Result carries the threaded estimator, and a batch run carries none" begin
        o = cross_val_predict(mr, rd_T, online_cv)
        b = cross_val_predict(mr, rd_T, batch_cv)
        @test isnothing(b.opt)
        @test isa(o.opt, MeanRisk)
        # `opt` is folded through the last training end, not the end of the data.
        cvr = split(online_cv, rd_T)
        @test po.returns_result(o.opt).X == rows(rd_T, 1:last(cvr.train_idx[end])).X
        @test po.held_timestamps(o.opt) == ts[1:last(cvr.train_idx[end])]
        # The loop returns the estimator beside the predictions, and `nothing` from the
        # batch arms.
        preds, est = po.fold_loop(mr, length(cvr.train_idx), FLoops.SequentialEx();
                                  rd = rd_T, train_idx = cvr.train_idx,
                                  test_idx = cvr.test_idx, cv = online_cv) do fold
            return po.fit_and_predict(fold.est, fold.rd; test_idx = fold.test)
        end
        @test length(preds) == length(cvr.train_idx)
        @test isa(est, MeanRisk)
        preds, est = po.fold_loop(mr, length(cvr.train_idx), FLoops.SequentialEx();
                                  rd = rd_T, train_idx = cvr.train_idx,
                                  test_idx = cvr.test_idx, cv = batch_cv) do fold
            return po.fit_and_predict(fold.est, fold.rd; train_idx = fold.train,
                                      test_idx = fold.test)
        end
        @test isnothing(est)
    end

    @testset "The identity: a resume equals the one-shot run, fold for fold" begin
        for (opt, r, r_T) in
            ((mr, rdg, rdg_T), (tn, rd, rd_T), (hrp, rdg, rdg_T), (ew, rdg, rdg_T))
            old = cross_val_predict(opt, r_T, online_cv)
            new = cross_val_predict(Resume(old), r, online_cv)
            one = cross_val_predict(opt, r, online_cv)
            @test length(old.pred) + length(new.pred) == length(one.pred)
            same_weights(vcat(old, new), one, opt)
            # The new Result holds the new folds only, and carries the estimator again,
            # folded through the same last training end as the one-shot run's.
            @test new.pred[1].rd.ts[1] == one.pred[length(old.pred) + 1].rd.ts[1]
            @test isequal(po.returns_result(new.opt).X, po.returns_result(one.opt).X)
            @test po.held_timestamps(new.opt) == po.held_timestamps(one.opt)
        end
        # The turnover term reads the previous fold's weights across the seam: the first
        # resumed fold's turnover target is the old run's last weights.
        old = cross_val_predict(tn, rd_T, online_cv)
        new = cross_val_predict(Resume(old), rd, online_cv)
        @test new.pred[1].res.jr.pa.tn.w == old.pred[end].res.w
        # And the emitted message says the run resumed.
        n = n_splits(online_cv, rd)
        @test_logs (:info, po.cv_resume_info(length(old.pred), n)) cross_val_predict(Resume(old),
                                                                                     rd,
                                                                                     online_cv)
    end

    @testset "A capped run, whose buffer holds no total" begin
        capped = MeanRisk(;
                          opt = JuMPOptimiser(;
                                              pe = po.Online(EmpiricalPrior();
                                                             max_history = w), slv = slv))
        old = cross_val_predict(capped, rd_T, online_cv)
        @test length(po.held_timestamps(old.opt)) == w
        new = cross_val_predict(Resume(old), rd, online_cv)
        one = cross_val_predict(capped, rd, online_cv)
        same_weights(vcat(old, new), one, capped)
        # The held span is the cap, and the resumed state holds the cap again.
        @test length(po.held_timestamps(new.opt)) == w
        @test po.held_timestamps(new.opt) == po.held_timestamps(one.opt)
    end

    @testset "The date form anchors on the whole ts" begin
        od = DateWalkForward(w, t; period = Day(1), purged_size = p, ff = OnlineStep())
        for opt in (mr, hrp)
            old = cross_val_predict(opt, rdg_T, od)
            new = cross_val_predict(Resume(old), rdg, od)
            same_weights(vcat(old, new), cross_val_predict(opt, rdg, od), opt)
        end
    end

    @testset "A Weight Drift threads the held weights across the seam" begin
        sfd = SelfFinancingDrift()
        online_d = IndexWalkForward(w, t; purged_size = p, ff = OnlineStep(), wd = sfd,
                                    pws = DriftedWeights())
        old = cross_val_predict(tn, rd_T, online_d)
        new = cross_val_predict(Resume(old), rd, online_d)
        one = cross_val_predict(tn, rd, online_d)
        same_weights(vcat(old, new), one, tn)
        @test all(isapprox(a.hw.w, b.hw.w; atol = 1e-5)
                  for (a, b) in zip(vcat(old, new).pred, one.pred))
        @test new.pred[1].res.jr.pa.tn.w == old.pred[end].hw.w
    end

    @testset "A Result is a value: resumed twice, and a resume resumed" begin
        old = cross_val_predict(tn, rd_T, online_cv)
        held = deepcopy(po.returns_result(old.opt).X)
        a = cross_val_predict(Resume(old), rd, online_cv)
        b = cross_val_predict(Resume(old), rd, online_cv)
        @test weights(a) == weights(b)
        @test po.returns_result(old.opt).X == held
        # A chain: the old run, a resume to an intermediate history, and a resume of that.
        T1 = 160
        mid = cross_val_predict(Resume(old), rows(rd, 1:T1), online_cv)
        @test length(mid.pred) == 1
        # `mid` holds one fold, and the chain still skips every fold before it: the folds to
        # skip are read off the state's last held timestamp, not off `length(mid.pred)`.
        last_ = cross_val_predict(Resume(mid), rd, online_cv)
        one = cross_val_predict(tn, rd, online_cv)
        same_weights(vcat(vcat(old, mid), last_), one, tn)
        @test length(last_.pred) == length(one.pred) - length(old.pred) - 1
        # A terminal view of the leftover rows never spoils the continuation.
        term = cross_val_predict(Resume(old), rows(rd, 1:150),
                                 IndexWalkForward(w, t; purged_size = p, ff = OnlineStep(),
                                                  reduce_test = true))
        @test length(term.pred[end].rd.ts) == 10
        same_weights(vcat(old, cross_val_predict(Resume(old), rd, online_cv)), one, tn)
    end

    @testset "The re-entry refusals, each by name" begin
        old = cross_val_predict(mr, rd_T, online_cv)
        # A batch Result, and a population.
        @test_throws ArgumentError Resume(cross_val_predict(mr, rd_T, batch_cv))
        @test occursin("batch run",
                       msg(() -> Resume(cross_val_predict(mr, rd_T, batch_cv))))
        mo = MultipleRandomised(online_cv; subset_size = 4, n_subsets = 2, seed = 7)
        @test_throws ArgumentError Resume(cross_val_predict(mr, rd_T, mo))
        # A scheme that is not a walk-forward, one with no Fold Fit, and a population
        # scheme.
        @test_throws ArgumentError cross_val_predict(Resume(old), rd, KFold())
        @test_throws ArgumentError cross_val_predict(Resume(old), rd, batch_cv)
        @test occursin("no Fold Fit",
                       msg(() -> cross_val_predict(Resume(old), rd, batch_cv)))
        @test_throws ArgumentError cross_val_predict(Resume(old), rd, mo)
        @test_throws ArgumentError cross_val_predict(Resume(old), rd,
                                                     CombinatorialCrossValidation())
        # The search.
        gs = GridSearchCrossValidation(["opt.l1" => [0.0002, 0.002]]; cv = online_cv,
                                       r = ConditionalValueatRisk())
        @test_throws ArgumentError search_cross_validation(Resume(old), gs, rd)
        # A carrier that adds no fold.
        @test_throws ArgumentError cross_val_predict(Resume(old), rd_T, online_cv)
        @test occursin("adds no fold",
                       msg(() -> cross_val_predict(Resume(old), rows(rd, 1:150), online_cv)))
        # Timestamps are required, on the carrier and on the state.
        no_ts = ReturnsResult(; nx = nx, X = X)
        @test_throws ArgumentError cross_val_predict(Resume(old), no_ts, online_cv)
        old_no_ts = cross_val_predict(mr, rows(no_ts, 1:T0), online_cv)
        @test isnothing(po.held_timestamps(old_no_ts.opt))
        @test_throws ArgumentError cross_val_predict(Resume(old_no_ts), rd, online_cv)
        @test occursin("synthetic calendar",
                       msg(() -> cross_val_predict(Resume(old_no_ts), rd, online_cv)))
        # An index-only caller attaches a synthetic calendar, and the check is exact over the
        # held span.
        idx = ReturnsResult(; nx = nx, X = X, ts = Date(1) .+ Day.(0:(T - 1)))
        old_idx = cross_val_predict(mr, rows(idx, 1:T0), online_cv)
        new_idx = cross_val_predict(Resume(old_idx), idx, online_cv)
        same_weights(vcat(old_idx, new_idx), cross_val_predict(mr, idx, online_cv), mr)
        # A prefix shifted by one row, a row dropped inside the held span, and a shifted
        # calendar are all refused.
        shifted = ReturnsResult(; nx = nx, X = X, ts = ts .+ Day(1))
        @test_throws ArgumentError cross_val_predict(Resume(old), shifted, online_cv)
        dropped = ReturnsResult(; nx = nx, X = X[[1:99; 101:T], :], ts = ts[[1:99; 101:T]])
        @test_throws ArgumentError cross_val_predict(Resume(old), dropped, online_cv)
        @test occursin("no fold of the scheme",
                       msg(() -> cross_val_predict(Resume(old), dropped, online_cv)))
        # A row inside the held span that is not the row the state folded: the training
        # end still names a fold, and the prefix check catches it.
        altered = copy(ts)
        altered[100] = Date(2000, 1, 1)
        altered_rd = ReturnsResult(; nx = nx, X = X, ts = altered)
        @test_throws ArgumentError cross_val_predict(Resume(old), altered_rd, online_cv)
        @test occursin("do not carry them",
                       msg(() -> cross_val_predict(Resume(old), altered_rd, online_cv)))
        # A changed scheme moves the last training end.
        @test_throws ArgumentError cross_val_predict(Resume(old), rd,
                                                     IndexWalkForward(w, t; purged_size = 5,
                                                                      ff = OnlineStep()))
        # A partial last fold is terminal, and the message names the two-resume workaround.
        reduced = IndexWalkForward(w, t; purged_size = p, ff = OnlineStep(),
                                   reduce_test = true)
        part = cross_val_predict(mr, rows(rd, 1:150), reduced)
        @test length(part.pred[end].rd.ts) == 10
        @test_throws ArgumentError cross_val_predict(Resume(part), rd, reduced)
        @test occursin("terminal", msg(() -> cross_val_predict(Resume(part), rd, reduced)))
        # The pinned context runs on every delta step: a renamed asset is refused.
        renamed = ReturnsResult(; nx = ["B$i" for i in 1:N], X = X, ts = ts)
        @test_throws ArgumentError cross_val_predict(Resume(old), renamed, online_cv)
        # An estimator that keeps no context has no timestamps to align.
        pw = cross_val_predict(PreviousWeights(; w = fill(inv(N), N)), rd_T, online_cv)
        @test isnothing(po.held_timestamps(pw.opt))
        @test_throws ArgumentError cross_val_predict(Resume(pw), rd, online_cv)
    end

    @testset "vcat stacks a run and its resume, and refuses a pair that does not abut" begin
        old = cross_val_predict(mr, rd_T, online_cv)
        new = cross_val_predict(Resume(old), rd, online_cv)
        stack = vcat(old, new)
        @test stack.pred == vcat(old.pred, new.pred)
        @test stack.mrd.ts == vcat(old.mrd.ts, new.mrd.ts)
        @test stack.mrd.X == vcat(old.mrd.X, new.mrd.X)
        @test stack.opt === new.opt
        @test isnothing(stack.id)
        @test_throws ArgumentError vcat(new, old)
        @test_throws ArgumentError vcat(old, old)
        no_ts = cross_val_predict(mr, rows(ReturnsResult(; nx = nx, X = X), 1:T0),
                                  online_cv)
        @test_throws ArgumentError vcat(no_ts, no_ts)
        # A stacked Result scores as the one-shot Result does.
        one = cross_val_predict(mr, rd, online_cv)
        @test isapprox(expected_risk(ConditionalValueatRisk(), stack),
                       expected_risk(ConditionalValueatRisk(), one); atol = 1e-8)
    end

    @testset "A Result round-trips through the stdlib Serialization and resumes" begin
        for opt in (tn, hrp)
            old = cross_val_predict(opt, rdg_T, online_cv)
            io = IOBuffer()
            serialize(io, old)
            seekstart(io)
            back = deserialize(io)
            @test weights(back) == weights(old)
            @test isequal(po.returns_result(back.opt).X, po.returns_result(old.opt).X)
            @test po.held_timestamps(back.opt) == po.held_timestamps(old.opt)
            a = cross_val_predict(Resume(back), rdg, online_cv)
            b = cross_val_predict(Resume(old), rdg, online_cv)
            @test weights(a) == weights(b)
        end
    end

    @testset "A Pipeline host resumes through the state its row owner keeps" begin
        # A prior owner leaves the pipeline its own context; an optimisation owner keeps
        # its own; `Online(pipe)` buffers the carrier itself.
        pipes = (Pipeline(; steps = (EmpiricalPrior(), hrp)), Pipeline(; steps = (mr,)),
                 Pipeline(;
                          steps = (ScoreSelector(; score = MeanReturn(),
                                                 rule = RankRule(; best = 4)),
                                   EmpiricalPrior(), mr)))
        for pipe in pipes
            old = cross_val_predict(pipe, rd_T, online_cv)
            @test isa(old.opt, Pipeline)
            new = cross_val_predict(Resume(old), rd, online_cv)
            one = cross_val_predict(pipe, rd, online_cv)
            same_weights(vcat(old, new), one, pipe.steps[end])
        end
        wrapped = po.Online(pipes[1])
        old = cross_val_predict(wrapped, rd_T, online_cv)
        @test isa(old.opt.cache, po.PipelineBufferState)
        new = cross_val_predict(Resume(old), rd, online_cv)
        same_weights(vcat(old, new), cross_val_predict(wrapped, rd, online_cv), hrp)
        # The pipeline doors refuse what the optimiser doors refuse.
        @test_throws ArgumentError cross_val_predict(Resume(old), rd, batch_cv)
        @test_throws ArgumentError cross_val_predict(Resume(old), rd_T, online_cv)
        gs = GridSearchCrossValidation(["opt.l1" => [0.0002, 0.002]]; cv = online_cv,
                                       r = ConditionalValueatRisk())
        @test_throws ArgumentError search_cross_validation(Resume(old), gs, rd)
        # A price-level pipeline aligns by the prices' timestamps: the returns' are the
        # prices' from the second row on, so the held span equals its rows of the price
        # carrier under the host route and under the refit route alike.
        P = 100 .* exp.(cumsum(X; dims = 1))
        pr = price_ingestion(PriceIngestion(), TimeSeries.TimeArray(ts, P, nx))
        pr_T = po.pipeline_data_view(pr, 1:T0)
        ppipe = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), mr))
        for est in (ppipe, po.Online(ppipe))
            old = cross_val_predict(est, pr_T, online_cv)
            new = cross_val_predict(Resume(old), pr, online_cv)
            same_weights(vcat(old, new), cross_val_predict(est, pr, online_cv), mr)
        end
    end

    @testset "A schedule on a stateless field reads the combined enumeration's i and n" begin
        # A callable schedule is copied through untouched and resolves per fold against
        # the extended history's fold count, so the resumed fold reads the same context
        # the one-shot fold reads.
        # The bound reads `i` alone: the old run's folds were resolved under the old
        # enumeration's `n`, and the identity is over the folds, not over `n`.
        seen = Tuple{Int, Int}[]
        sched = TimeDependent(ctx -> begin
                                  push!(seen, (ctx.i, ctx.n))
                                  return WeightBounds(; lb = 0.0, ub = 0.3 + 0.02 * ctx.i)
                              end)
        ts_mr = MeanRisk(;
                         opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                             wb = sched))
        old = cross_val_predict(ts_mr, rd_T, online_cv)
        @test po.is_time_dependent(Resume(old))
        empty!(seen)
        new = cross_val_predict(Resume(old), rd, online_cv)
        n = n_splits(online_cv, rd)
        @test seen == [(i, n) for i in (length(old.pred) + 1):n]
        same_weights(vcat(old, new), cross_val_predict(ts_mr, rd, online_cv), ts_mr)
        @test all(maximum(p.res.w) <= 0.3 + 0.02 * (length(old.pred) + k) + 1e-8
                  for (k, p) in enumerate(new.pred))
    end

    @testset "The keyword constructor, the asset view at the door, and a cold estimator" begin
        old = cross_val_predict(mr, rd_T, online_cv)
        @test Resume(; res = old).res === old
        # The carrier is viewed by `cols` as the one-shot door views it, and the Result's
        # estimator, threaded over the old run's view already, is not.
        cols = 1:4
        old_c = cross_val_predict(mr, rd_T, online_cv; cols = cols)
        new_c = cross_val_predict(Resume(old_c), rd, online_cv; cols = cols)
        same_weights(vcat(old_c, new_c), cross_val_predict(mr, rd, online_cv; cols = cols),
                     mr)
        @test all(length(p.res.w) == 4 for p in new_c.pred)
        # A cold estimator holds no context and so no timestamps, and a value that is not
        # an estimator is copied as itself.
        @test isnothing(po.held_timestamps(mr))
        @test isnothing(po.held_timestamps(hrp))
        @test po.copy_states(nothing) === nothing
        @test po.copy_states(mr) === mr
    end
end
