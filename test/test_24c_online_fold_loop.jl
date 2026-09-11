#=
The fold loop takes the online step, issue #969, against the decision of #870 (ADR 0140).

A walk-forward declares its Fold Fit in `ff`, and `OnlineStep()` sends `fold_loop` down a
third arm: warm up once on the first training window, fold each fold's new rows into one
threaded estimator, and read it out where a refit would have run. Two identities are the
contract. The expanding one — the map's closing test's first half — says the online run
reaches the weights of `expand_train = true` fold for fold, over a panel with a listing and a
delisting, through a JuMP optimiser and a hierarchical one. The capped one says a rolling
batch scheme equals the online scheme with the prior's buffer capped at the window. Around
them sit the derivation of `expand_train`, the purge, the refusals, the ordering, and the
public read-out entry.

The fixture is synthetic, because the identities are structural and a solver run over eight
assets is what keeps the JuMP families cheap.
=#
@testset "Online fold loop: the walk-forward declares its Fold Fit" begin
    using Test, PortfolioOptimisers, Clarabel, StableRNGs, Statistics, Dates, LinearAlgebra,
          FLoops
    po = PortfolioOptimisers
    rng = StableRNG(20260911)
    T, N = 160, 8
    X = randn(rng, T, N) ./ 100 .+ 0.0003
    nx = ["A$i" for i in 1:N]
    ts = Date(2024, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = X, ts = ts)
    # A listing at row 31 and a delisting at row 131, expressed by the panel's masks.
    amsk = trues(T, N)
    amsk[1:30, 3] .= false
    amsk[131:end, 5] .= false
    Xg = copy(X)
    Xg[.!amsk] .= NaN
    rdg = ReturnsResult(; nx = nx, X = Xg, ts = ts,
                        pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk)))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = false, allow_almost = false),
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))
    jopt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)
    hopt = HierarchicalOptimiser(; pe = EmpiricalPrior())
    mr = MeanRisk(; opt = jopt)
    hrp = HierarchicalRiskParity(; opt = hopt)
    w, t, p = 60, 20, 3
    rows(r, i) = po.port_opt_view(r, i, :)
    weights(res) = [pr.res.w for pr in res.pred]
    same_fields(a, b) = all(f -> isequal(getfield(a, f), getfield(b, f)),
                            (:nx, :X, :nf, :F, :nb, :B, :ts, :iv, :ivpa))
    function weight_tol(opt)
        solved = isa(opt, po.JuMPOptimisationEstimator) || isa(opt, NestedClustered)
        return isa(opt, RiskBudgeting) ? 5e-5 : (solved ? 1e-5 : 1e-10)
    end
    batch_cv = IndexWalkForward(w, t; purged_size = p, expand_train = true)
    online_cv = IndexWalkForward(w, t; purged_size = p, ff = OnlineStep())

    @testset "The expanding identity, over a panel with a listing and a delisting" begin
        fams = (mr, RiskBudgeting(; opt = jopt), hrp,
                HierarchicalEqualRiskContribution(; opt = hopt), InverseVolatility(),
                NestedClustered(; opti = mr, opto = mr))
        for r in (rd, rdg), opt in fams
            b = cross_val_predict(opt, r, batch_cv)
            o = cross_val_predict(opt, r, online_cv)
            @test length(o.pred) == length(b.pred) == n_splits(online_cv, r)
            for (pb, po_) in zip(b.pred, o.pred)
                @test isapprox(po_.res.w, pb.res.w; atol = weight_tol(opt))
                @test po_.res.imsk == pb.res.imsk
                # The test window is read the same way: the fold's returns series agrees to
                # the weights' own tolerance.
                @test isapprox(po_.rd.X, pb.rd.X; atol = weight_tol(opt))
            end
        end
        # The carrier the read-out rebuilds is exactly the batch fold's training window, and
        # the fold hands its callback no window of its own.
        cvr = split(online_cv, rdg)
        n = length(cvr.train_idx)
        seen = Int[]
        po.fold_loop(mr, n, FLoops.SequentialEx(); rd = rdg, train_idx = cvr.train_idx,
                     test_idx = cvr.test_idx, cv = online_cv) do fold
            push!(seen, fold.i)
            @test isnothing(fold.train)
            @test fold.test == cvr.test_idx[fold.i]
            @test same_fields(po.returns_result(fold.est), rows(rdg, cvr.train_idx[fold.i]))
            return po.fit_and_predict(fold.est, fold.rd; test_idx = fold.test)
        end
        @test seen == 1:n
    end

    @testset "The capped identity: a rolling scheme equals an online one with a capped prior" begin
        rolling = IndexWalkForward(w + p, t; purged_size = p)
        stepped = IndexWalkForward(w + p, t; purged_size = p, ff = OnlineStep())
        @test all(length.(split(rolling, rd).train_idx) .== w)
        for (opt, capped) in ((mr,
                               MeanRisk(;
                                        opt = JuMPOptimiser(;
                                                            pe = po.Online(EmpiricalPrior(); max_history = w), slv = slv))),
                              (InverseVolatility(),
                               InverseVolatility(; pe = po.Online(EmpiricalPrior(); max_history = w))))
            b = cross_val_predict(opt, rd, rolling)
            o = cross_val_predict(capped, rd, stepped)
            for (pb, po_) in zip(b.pred, o.pred)
                # A capped buffer reads out over exactly the rolling window's rows, so the
                # two fits see the same sample and the weights agree to the solver's own
                # precision.
                @test isapprox(po_.res.w, pb.res.w; atol = weight_tol(opt))
            end
        end
    end

    @testset "The date form" begin
        bd = DateWalkForward(w, t; period = Day(1), purged_size = p, expand_train = true)
        od = DateWalkForward(w, t; period = Day(1), purged_size = p, ff = OnlineStep())
        @test split(od, rd).train_idx == split(bd, rd).train_idx
        for opt in (mr, hrp)
            b = cross_val_predict(opt, rdg, bd)
            o = cross_val_predict(opt, rdg, od)
            for (pb, po_) in zip(b.pred, o.pred)
                @test isapprox(po_.res.w, pb.res.w; atol = weight_tol(opt))
            end
        end
    end

    @testset "The purge: a row is folded when the window reaches it, never dropped" begin
        cvr = split(online_cv, rd)
        n = length(cvr.train_idx)
        folded = Vector{Int}(undef, n)
        po.fold_loop(InverseVolatility(), n, FLoops.SequentialEx(); rd = rd,
                     train_idx = cvr.train_idx, test_idx = cvr.test_idx, cv = online_cv
                     ) do fold
            i = fold.i
            rdp = po.returns_result(fold.est)
            folded[i] = size(rdp.X, 1)
            # Fold `i` has folded `1 : test_start_i - p`, and nothing beyond.
            @test rdp.X == X[1:(first(cvr.test_idx[i]) - 1 - p), :]
            @test folded[i] == last(cvr.train_idx[i])
            return po.fit_and_predict(fold.est, fold.rd; test_idx = fold.test)
        end
        # The purged rows of fold `i` are folded by fold `i + 1`.
        for i in 1:(n - 1)
            @test first(cvr.test_idx[i]) - 1 <= folded[i + 1]
        end
    end

    @testset "`expand_train` derives from the Fold Fit" begin
        @test IndexWalkForward(252, 21).expand_train == false
        @test isnothing(IndexWalkForward(252, 21).ff)
        @test IndexWalkForward(252, 21; ff = OnlineStep()).expand_train == true
        @test IndexWalkForward(252, 21; expand_train = true).expand_train == true
        @test IndexWalkForward(252, 21; expand_train = true, ff = OnlineStep()).expand_train
        @test_throws ArgumentError IndexWalkForward(252, 21; expand_train = false,
                                                    ff = OnlineStep())
        @test DateWalkForward(252, 21).expand_train == false
        @test DateWalkForward(252, 21; ff = OnlineStep()).expand_train == true
        @test_throws ArgumentError DateWalkForward(252, 21; expand_train = false,
                                                   ff = OnlineStep())
        # The refusal names the composition that gives a rolling window computed online.
        err = try
            IndexWalkForward(252, 21; purged_size = 2, expand_train = false,
                             ff = OnlineStep())
            nothing
        catch e
            e
        end
        @test isa(err, ArgumentError) && occursin("max_history = 250", err.msg)
        # The switch is read by its verb, and every other scheme answers `nothing`.
        @test po.fold_fit(online_cv) == OnlineStep()
        @test isnothing(po.fold_fit(batch_cv))
        @test isnothing(po.fold_fit(KFold()))
        @test isnothing(po.fold_fit(split(online_cv, rd)))
        @test po.fold_fit(MultipleRandomised(online_cv; seed = 1)) == OnlineStep()
        @test isnothing(po.fold_fit(nothing))
    end

    @testset "A state at entry is refused, before any solve, and the error names the field" begin
        function refusal(opt, r = rd)
            try
                cross_val_predict(opt, r, online_cv)
                nothing
            catch e
                e
            end
        end
        # A stepped head: the context sits on the bundle.
        err = refusal(partial_fit!(mr, rows(rd, 1:5)))
        @test isa(err, ArgumentError) && occursin("`opt.cache`", err.msg)
        # A stepped host: the context sits on the host itself.
        err = refusal(partial_fit!(InverseVolatility(), rows(rd, 1:5)))
        @test isa(err, ArgumentError) && occursin("`cache`", err.msg)
        # A state three levels down, on a moment estimator the prior holds.
        me = partial_fit!(SimpleExpectedReturns(), X[1:5, :])
        err = refusal(MeanRisk(;
                               opt = JuMPOptimiser(; pe = EmpiricalPrior(; me = me),
                                                   slv = slv)))
        @test isa(err, ArgumentError) && occursin("`opt.pe.me.cache`", err.msg)
        # The walk itself, on the tree and on what is not one.
        @test isnothing(po.online_entry_state(mr))
        @test po.online_entry_state(partial_fit!(mr, rows(rd, 1:5))) == "opt.cache"
        @test isnothing(po.online_entry_state(nothing))
        @test isnothing(po.online_entry_state(TimeDependent([me]; default = me)))
        # The batch arms read a stepped estimator as configuration, as they always did.
        @test isapprox(weights(cross_val_predict(partial_fit!(mr, rows(rd, 1:5)), rd,
                                                 batch_cv)),
                       weights(cross_val_predict(mr, rd, batch_cv)); atol = 1e-5)
    end

    @testset "A schedule reaches stateless fields only" begin
        cvr = split(online_cv, rd)
        n = length(cvr.train_idx)
        # On the prior: refused at warm-up. The schedule is as long as the enumeration, so
        # the fold count admits it and the refusal is the online arm's own. (A JuMP or
        # hierarchical head's `opt` cannot hold a schedule at all: its bound refuses one at
        # construction.)
        sched_pe = MeanRisk(;
                            opt = JuMPOptimiser(;
                                                pe = TimeDependent(fill(EmpiricalPrior(),
                                                                        n);
                                                                   default = EmpiricalPrior()),
                                                slv = slv))
        err = try
            cross_val_predict(sched_pe, rd, online_cv)
            nothing
        catch e
            e
        end
        @test isa(err, ArgumentError) && occursin("`JuMPOptimiser.pe`", err.msg)
        @test_throws TypeError MeanRisk(; opt = TimeDependent([jopt]; default = jopt))
        # A schedule of optimisers at the root.
        sched_root = TimeDependent(fill(mr, n); default = mr)
        err = try
            cross_val_predict(sched_root, rd, online_cv)
            nothing
        catch e
            e
        end
        @test isa(err, ArgumentError) && occursin("stateless fields only", err.msg)
        # Both schedules run in batch, where a fold resolves them before it fits.
        @test length(cross_val_predict(sched_pe, rd, batch_cv).pred) == n
        @test length(cross_val_predict(sched_root, rd, batch_cv).pred) == n
        # The step's own refusals say the same thing.
        for bad in (sched_pe, sched_root)
            err = try
                partial_fit!(bad, rows(rd, 1:1))
                nothing
            catch e
                e
            end
            @test isa(err, ArgumentError) && occursin("stateless fields only", err.msg)
        end
        # A schedule on `wb` composes, and the state threads unchanged through the swap:
        # the online run equals the expanding batch run under the same schedule.
        wbs = [WeightBounds(; lb = 0.0, ub = 0.3 + 0.05 * (i - 1)) for i in 1:n]
        tdmr = MeanRisk(;
                        opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                            wb = TimeDependent(wbs)))
        b = cross_val_predict(tdmr, rd, batch_cv)
        o = cross_val_predict(tdmr, rd, online_cv)
        for (i, (pb, po_)) in enumerate(zip(b.pred, o.pred))
            @test isapprox(po_.res.w, pb.res.w; atol = 1e-5)
            @test maximum(po_.res.w) <= wbs[i].ub + 1e-6
        end
        # And the callback sees the schedule resolved on its copy while the threaded
        # estimator keeps its state.
        po.fold_loop(tdmr, n, FLoops.SequentialEx(); rd = rd, train_idx = cvr.train_idx,
                     test_idx = cvr.test_idx, cv = online_cv) do fold
            @test fold.est.opt.wb == wbs[fold.i]
            @test !isnothing(fold.est.opt.cache)
            return po.fit_and_predict(fold.est, fold.rd; test_idx = fold.test)
        end
    end

    @testset "The warm-up resolves every wrapper" begin
        cvr = split(online_cv, rd)
        n = length(cvr.train_idx)
        wrapped = MeanRisk(;
                           opt = JuMPOptimiser(; pe = po.Online(EmpiricalPrior()),
                                               slv = slv))
        po.fold_loop(wrapped, n, FLoops.SequentialEx(); rd = rd, train_idx = cvr.train_idx,
                     test_idx = cvr.test_idx, cv = online_cv) do fold
            @test !isa(fold.est.opt.pe, po.Online)
            @test isa(fold.est.opt.pe, EmpiricalPrior)
            @test isa(po.partial_fit_cache(fold.est.opt.pe), po.SampleBufferState)
            @test isnothing(po.online_entry_state(wrapped))
            return po.fit_and_predict(fold.est, fold.rd; test_idx = fold.test)
        end
        # A wrapped prior refits from its buffer and reaches the expanding batch weights.
        @test isapprox(weights(cross_val_predict(wrapped, rd, online_cv)),
                       weights(cross_val_predict(mr, rd, batch_cv)); atol = 1e-5)
    end

    @testset "Ordering: the online arm runs in order and says so" begin
        # A `MeanRisk` needs no previous weights, so its batch run takes the parallel arm
        # and says nothing; the online run emits `cv_online_info` and never the sequential
        # message.
        @test_logs cross_val_predict(mr, rd, batch_cv)
        @test_logs (:info, po.cv_online_info()) cross_val_predict(mr, rd, online_cv)
        # An optimiser that needs the previous weights still takes the online arm first.
        tn = MeanRisk(;
                      opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                          tn = Turnover(; w = fill(inv(N), N), val = 0.5)))
        @test_logs (:info, po.cv_sequential_info()) cross_val_predict(tn, rd, batch_cv)
        @test_logs (:info, po.cv_online_info()) cross_val_predict(tn, rd, online_cv)
        @test isapprox(weights(cross_val_predict(tn, rd, online_cv)),
                       weights(cross_val_predict(tn, rd, batch_cv)); atol = 1e-5)
        # Under a multiple-randomised scheme every path threads its own sliced estimator,
        # and reaches the batch path's weights.
        mb = MultipleRandomised(batch_cv; subset_size = 5, n_subsets = 2, seed = 7)
        mo = MultipleRandomised(online_cv; subset_size = 5, n_subsets = 2, seed = 7)
        @test split(mo, rd).asset_idx == split(mb, rd).asset_idx
        b = cross_val_predict(mr, rdg, mb)
        o = @test_logs (:info, po.cv_online_info()) (:info, po.cv_online_info()) cross_val_predict(mr,
                                                                                                   rdg,
                                                                                                   mo)
        for (pathb, patho) in zip(b.pred, o.pred)
            @test length(patho.pred) == length(pathb.pred)
            for (pb, po_) in zip(pathb.pred, patho.pred)
                @test isapprox(po_.res.w, pb.res.w; atol = 1e-5)
                @test po_.rd.nx == pb.rd.nx
            end
        end
    end

    @testset "The read-out entry: a hand-stepped estimator equals the cold one" begin
        cvr = split(online_cv, rd)
        for i in (1, 3)
            train, test = cvr.train_idx[i], cvr.test_idx[i]
            stepped = partial_fit!(po.update_online_estimator(mr), rows(rd, train))
            cold = po.fit_and_predict(mr, rd; train_idx = train, test_idx = test)
            warm = po.fit_and_predict(stepped, rd; test_idx = test)
            @test isapprox(warm.res.w, cold.res.w; atol = 1e-5)
            @test isapprox(warm.rd.X, cold.rd.X; atol = 1e-5)
            # And with an asset view: the stepped estimator is sliced as the cold one is.
            cold = po.fit_and_predict(mr, rd; train_idx = train, test_idx = test,
                                      cols = 2:6)
            warm = po.fit_and_predict(stepped, rd; test_idx = test, cols = 2:6)
            @test isapprox(warm.res.w, cold.res.w; atol = 1e-5)
            @test warm.rd.nx == cold.rd.nx == nx[2:6]
        end
        # A cold estimator with no window has nothing to read out.
        @test_throws ArgumentError po.fit_and_predict(mr, rd; test_idx = cvr.test_idx[1])
    end

    @testset "The search and the Pipeline refuse a Fold Fit by name, until their own tickets" begin
        pgrid = ["opt.l1" => range(; start = 0.0005, stop = 0.001, length = 2)]
        @test_throws ArgumentError search_cross_validation(MeanRisk(;
                                                                    opt = JuMPOptimiser(;
                                                                                        pe = EmpiricalPrior(),
                                                                                        slv = slv,
                                                                                        l1 = 0.0005)),
                                                           GridSearchCrossValidation(pgrid;
                                                                                     cv = online_cv,
                                                                                     r = Variance()),
                                                           rd)
        pipe = Pipeline(; steps = (EmpiricalPrior(), EqualWeighted()))
        @test_throws ArgumentError cross_val_predict(pipe, rd, online_cv)
        @test_throws ArgumentError cross_val_predict(pipe, rd,
                                                     MultipleRandomised(online_cv;
                                                                        subset_size = 5,
                                                                        seed = 1))
        err = try
            cross_val_predict(pipe, rd, online_cv)
            nothing
        catch e
            e
        end
        @test isa(err, ArgumentError) && occursin("#872", err.msg)
        # And the arm itself refuses a pipeline by name, should a route ever reach it.
        @test_throws ArgumentError po.assert_online_entry(pipe)
        # The batch twin runs at every one of those doors.
        @test length(cross_val_predict(pipe, rd, batch_cv).pred) == n_splits(batch_cv, rd)
    end
end
