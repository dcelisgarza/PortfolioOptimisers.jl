#=
The search scores every candidate through the one fold loop, issue #1020, against the
decision of #871 (ADR 0141).

`search_cross_validation` builds candidate `i` through the grid's lenses and scores it through
`fit_and_predict(opt_i, rd, gscv.cv; ex = SequentialEx())`, one row per fold, whatever the
scheme's fit. The map's closing test's second half is the identity: a grid search over an
online walk-forward reads the matrix of the same search over the batch expanding walk-forward
and picks the same column, through a JuMP optimiser and a hierarchical one, and the randomised
search under the same seed inherits it. Around it sit what the route changed in batch (a
walk-forward search now threads previous weights and resolves schedules), what it kept (a
search with none of those is bit for bit the released one), the door's refusals, the failed
candidate, the multiple-randomised layout, the Pipeline's search, and the executor.

The fixture is the online fold loop's, because the identities are structural.
=#
@testset "The search scores every candidate through the one fold loop" begin
    using Test, PortfolioOptimisers, Clarabel, StableRNGs, Statistics, Dates, LinearAlgebra,
          FLoops, Accessors, Distributions, StatsBase
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
    batch_cv = IndexWalkForward(w, t; purged_size = p, expand_train = true)
    online_cv = OnlineIndexWalkForward(w, t; purged_size = p)
    r = ConditionalValueatRisk()
    # The identity's grids tune the weight bounds, which bind and so move the winner well
    # clear of the tolerance; the plain grids elsewhere tune the L1 penalty.
    jgrid = ["opt.l1" => [0.0002, 0.0005, 0.002]]
    wbgrid = ["opt.wb" =>
                  [WeightBounds(; lb = 0.0, ub = 1.0), WeightBounds(; lb = 0.0, ub = 0.2),
                   WeightBounds(; lb = 0.1, ub = 1.0)]]
    gs(cv, grid; kwargs...) = GridSearchCrossValidation(grid; cv = cv, r = r, kwargs...)
    function weight_tol(opt)
        solved = isa(opt, po.JuMPOptimisationEstimator) || isa(opt, NestedClustered)
        return isa(opt, RiskBudgeting) ? 5e-5 : (solved ? 1e-5 : 1e-10)
    end

    @testset "1. The search identity, the map's closing test's second half" begin
        for r_ in (rd, rdg), (opt, grid) in ((mr, wbgrid), (hrp, wbgrid))
            b = search_cross_validation(opt, gs(batch_cv, grid; train_score = true), r_)
            o = search_cross_validation(opt, gs(online_cv, grid; train_score = true), r_)
            @test size(o.test_scores) ==
                  size(b.test_scores) ==
                  (n_splits(batch_cv, r_), length(grid[1][2]))
            @test isapprox(o.test_scores, b.test_scores; atol = weight_tol(opt))
            @test isapprox(o.train_scores, b.train_scores; atol = weight_tol(opt))
            @test o.idx == b.idx
            @test o.val_grid[o.idx] == b.val_grid[b.idx]
            # The winner is not a knife-edge: its column beats the runner-up by orders of
            # magnitude more than the two matrices disagree by.
            means = vec(mean(b.test_scores; dims = 1))
            sorted = sort(means; rev = true)
            @test sorted[1] - sorted[2] >
                  100 * maximum(abs.(o.test_scores .- b.test_scores))
        end
        # The randomised search under the same seed draws the same grid, and inherits it.
        rgrid = ["opt.l1" => [0.0002, 0.0005, 0.002, 0.005]]
        rs(cv) = RandomisedSearchCrossValidation(rgrid; cv = cv, r = r, n_iter = 3,
                                                 seed = 3)
        b = search_cross_validation(mr, rs(batch_cv), rdg)
        o = search_cross_validation(mr, rs(online_cv), rdg)
        @test o.val_grid == b.val_grid
        @test isapprox(o.test_scores, b.test_scores; atol = 1e-5)
        @test o.idx == b.idx
    end

    @testset "2. A released search is unchanged" begin
        # A batch search with no `pws`, no previous-weights term and no schedule is bit for
        # bit the hand loop of the per-fold refit, under a `KFold` and a walk-forward.
        l1 = po.parse_lens("opt.l1")
        for cv in (KFold(; n = 4), batch_cv, IndexWalkForward(w, t; purged_size = p))
            cvr = split(cv, rd)
            res = search_cross_validation(mr, gs(cv, jgrid; train_score = true), rd)
            for (i, val) in enumerate(jgrid[1][2]), j in eachindex(cvr.train_idx)
                pred = po.fit_and_predict(Accessors.set(mr, l1, val), rd;
                                          train_idx = cvr.train_idx[j],
                                          test_idx = cvr.test_idx[j])
                @test res.test_scores[j, i] == -expected_risk(r, pred)
                @test res.train_scores[j, i] == -expected_risk(r, pred.res)
            end
        end
    end

    @testset "3. The batch search now threads previous weights" begin
        # A walk-forward search with `pws` set over a `Turnover` term reaches the matrix
        # of `fit_and_predict(opt, rd, cv)` scored per fold, and differs from the hand loop
        # of an independent refit per fold, which threads nothing.
        tn = MeanRisk(;
                      opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                          tn = Turnover(; w = fill(inv(N), N), val = 0.5)))
        cv = IndexWalkForward(w, t; purged_size = p, expand_train = true,
                              wd = SelfFinancingDrift(), pws = DriftedWeights())
        res = search_cross_validation(tn, gs(cv, jgrid), rd)
        l1 = po.parse_lens("opt.l1")
        cvr = split(cv, rd)
        hwd = po.held_weights_drift(SelfFinancingDrift(), DriftedWeights())
        for (i, val) in enumerate(jgrid[1][2])
            opti = Accessors.set(tn, l1, val)
            loop = po.fit_and_predict(opti, rd, cv)
            @test res.test_scores[:, i] == [-expected_risk(r, pr) for pr in loop.pred]
            # The loop threads fold `j - 1`'s held weights into fold `j`'s term.
            @test all(j -> loop.pred[j].res.jr.pa.tn.w == loop.pred[j - 1].hw.w,
                      2:length(loop.pred))
            hand = [-expected_risk(r,
                                   po.fit_and_predict(opti, rd;
                                                      train_idx = cvr.train_idx[j],
                                                      test_idx = cvr.test_idx[j],
                                                      wd = SelfFinancingDrift(), hwd = hwd))
                    for j in eachindex(cvr.train_idx)]
            @test res.test_scores[1, i] == hand[1]
            @test !isapprox(res.test_scores[2:end, i], hand[2:end]; atol = 1e-12)
        end
        # And the online search reads the same threaded matrix.
        o = search_cross_validation(tn,
                                    gs(OnlineIndexWalkForward(w, t; purged_size = p,
                                                              wd = SelfFinancingDrift(),
                                                              pws = DriftedWeights()),
                                       jgrid), rd)
        @test isapprox(o.test_scores, res.test_scores; atol = 1e-5)
        @test o.idx == res.idx
    end

    @testset "4. The batch search now resolves a schedule" begin
        n = n_splits(batch_cv, rd)
        wbs = [WeightBounds(; lb = 0.0, ub = 0.3 + 0.05 * (i - 1)) for i in 1:n]
        tdmr = MeanRisk(;
                        opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                            wb = TimeDependent(wbs)))
        b = search_cross_validation(tdmr, gs(batch_cv, jgrid), rd)
        l1 = po.parse_lens("opt.l1")
        for (i, val) in enumerate(jgrid[1][2])
            loop = po.fit_and_predict(Accessors.set(tdmr, l1, val), rd, batch_cv)
            @test b.test_scores[:, i] == [-expected_risk(r, pr) for pr in loop.pred]
            @test all(j -> maximum(loop.pred[j].res.w) <= wbs[j].ub + 1e-6, 1:n)
        end
        o = search_cross_validation(tdmr, gs(online_cv, jgrid), rd)
        @test isapprox(o.test_scores, b.test_scores; atol = 1e-5)
        # A schedule on `pe` is refused once, before the grid, with the online arm's own
        # message, and no solve ran; the batch search resolves it per fold.
        sched_pe = MeanRisk(;
                            opt = JuMPOptimiser(;
                                                pe = TimeDependent(fill(EmpiricalPrior(),
                                                                        n);
                                                                   default = EmpiricalPrior()),
                                                slv = slv))
        err = try
            search_cross_validation(sched_pe, gs(online_cv, jgrid), rd)
            nothing
        catch e
            e
        end
        @test isa(err, ArgumentError) && occursin("`JuMPOptimiser.pe`", err.msg)
        @test size(search_cross_validation(sched_pe, gs(batch_cv, jgrid), rd).test_scores) ==
              (n, 3)
    end

    @testset "5. A warm estimator is refused once, before the grid, whatever `ff` is" begin
        warm = partial_fit!(mr, rows(rd, 1:5))
        deep = MeanRisk(;
                        opt = JuMPOptimiser(;
                                            pe = EmpiricalPrior(;
                                                                me = partial_fit!(SimpleExpectedReturns(),
                                                                                  X[1:5, :])),
                                            slv = slv))
        for cv in (batch_cv, online_cv, KFold(; n = 4)),
            (opt, path) in ((warm, "`opt.cache`"), (deep, "`opt.pe.me.cache`"))

            err = try
                search_cross_validation(opt, gs(cv, jgrid), rd)
                nothing
            catch e
                e
            end
            @test isa(err, ArgumentError) && occursin(path, err.msg)
            @test occursin(if po.folds_are_stepped(cv)
                               "online arm"
                           else
                               "search_cross_validation"
                           end, err.msg)
        end
        @test_throws ArgumentError po.assert_search_entry(warm, batch_cv)
        @test isnothing(po.assert_search_entry(mr, batch_cv))
        @test isnothing(po.assert_search_entry(mr, online_cv))
    end

    @testset "6. A failed step drops the candidate and every step runs" begin
        bad = WeightBounds(; lb = fill(0.5, N), ub = ones(N))
        ok = WeightBounds(; lb = zeros(N), ub = ones(N))
        n = n_splits(online_cv, rd)
        # Candidate 2 fails at fold 2 alone; candidate 3 is infeasible throughout.
        grid = ["opt.wb" => [ok, TimeDependent([i == 2 ? bad : ok for i in 1:n]), bad]]
        base = MeanRisk(;
                        opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                            sets = UniverseSets(; dict = Dict("nx" => nx))))
        for cv in (online_cv, batch_cv)
            res = search_cross_validation(base, gs(cv, grid), rd)
            @test all(isfinite, res.test_scores[:, 1])
            @test isnan(res.test_scores[2, 2])
            @test all(isfinite, res.test_scores[[1; 3:n], 2])
            @test all(isnan, res.test_scores[:, 3])
            @test res.idx == 1
            @test res.val_grid[res.idx] == (ok,)
            @test po.finite_candidate_order(HighestMeanScore(), res.test_scores, 3) ==
                  ([1], [mean(res.test_scores[:, 1])])
            @test res.rank_idx == [1]
        end
    end

    @testset "7. `MultipleRandomised` over an online walk-forward" begin
        mb = MultipleRandomised(batch_cv; subset_size = 5, n_subsets = 2, seed = 7)
        mo = MultipleRandomised(online_cv; subset_size = 5, n_subsets = 2, seed = 7)
        b = search_cross_validation(mr, gs(mb, jgrid), rdg)
        o = search_cross_validation(mr, gs(mo, jgrid), rdg)
        cvr = split(mb, rdg)
        @test size(b.test_scores, 1) == length(cvr.train_idx)
        @test isapprox(o.test_scores, b.test_scores; atol = 1e-5)
        @test o.idx == b.idx
        # Rows are in `split`'s order: row `j` is fold `j` of the enumeration, scored on
        # that fold's own asset subset, and each path is threaded within itself.
        l1 = po.parse_lens("opt.l1")
        for (i, val) in enumerate(jgrid[1][2]), j in eachindex(cvr.train_idx)
            pred = po.fit_and_predict(Accessors.set(mr, l1, val), rdg;
                                      train_idx = cvr.train_idx[j],
                                      test_idx = cvr.test_idx[j], cols = cvr.asset_idx[j])
            @test b.test_scores[j, i] == -expected_risk(r, pred)
        end
        @test po.score_rows(cvr) == [findall(==(k), cvr.path_ids) for k in 1:2]
        # The layout survives a path whose predictions come back reordered.
        @test length(cvr.train_idx) == 10
        perm = [4, 5, 1, 2, 3, 6, 7, 8, 9, 10]
        shuffled = po.MultipleRandomisedResult(; train_idx = cvr.train_idx[perm],
                                               test_idx = cvr.test_idx[perm],
                                               asset_idx = cvr.asset_idx[perm],
                                               path_ids = cvr.path_ids)
        @test po.score_rows(shuffled) == [[3, 4, 5, 1, 2], [6, 7, 8, 9, 10]]
        # An unseeded scheme is pinned once, so every candidate scores the same folds and
        # the matrix lines up with a split of the pinned scheme.
        unseeded = MultipleRandomised(batch_cv; subset_size = 5, n_subsets = 2,
                                      rng = StableRNG(11))
        pinned = po.pin_draw(unseeded)
        @test !isnothing(pinned.seed)
        @test po.pin_draw(pinned) === pinned
        @test po.pin_draw(batch_cv) === batch_cv
        res = search_cross_validation(mr,
                                      gs(MultipleRandomised(batch_cv; subset_size = 5,
                                                            n_subsets = 2,
                                                            rng = StableRNG(11)), jgrid),
                                      rd)
        cvp = split(pinned, rd)
        for (i, val) in enumerate(jgrid[1][2]), j in eachindex(cvp.train_idx)
            pred = po.fit_and_predict(Accessors.set(mr, l1, val), rd;
                                      train_idx = cvp.train_idx[j],
                                      test_idx = cvp.test_idx[j], cols = cvp.asset_idx[j])
            @test res.test_scores[j, i] == -expected_risk(r, pred)
        end
    end

    @testset "8. The Pipeline's search takes the same route in batch and online" begin
        pipe = Pipeline(; steps = (EmpiricalPrior(), mr))
        grid = ["steps[2].opt.l1" => [0.0002, 0.0005]]
        res = search_cross_validation(pipe, gs(batch_cv, grid; train_score = true), rd)
        l1 = po.parse_lens("steps[2].opt.l1")
        for (i, val) in enumerate(grid[1][2])
            loop = cross_val_predict(Accessors.set(pipe, l1, val), rd, batch_cv)
            @test res.test_scores[:, i] == [-expected_risk(r, pr) for pr in loop.pred]
            @test res.train_scores[:, i] == [-expected_risk(r, pr.res) for pr in loop.pred]
        end
        # It agrees with the optimiser's search over the same head.
        direct = search_cross_validation(mr, gs(batch_cv, ["opt.l1" => grid[1][2]]), rd)
        @test isapprox(res.test_scores, direct.test_scores; atol = 1e-8)
        @test res.idx == direct.idx
        # Under a multiple-randomised scheme too.
        mb = MultipleRandomised(batch_cv; subset_size = 5, n_subsets = 2, seed = 7)
        pres = search_cross_validation(pipe, gs(mb, grid), rd)
        dres = search_cross_validation(mr, gs(mb, ["opt.l1" => grid[1][2]]), rd)
        @test isapprox(pres.test_scores, dres.test_scores; atol = 1e-8)
        # The door takes an Online Scheme since #1022, and picks the batch candidate; test_24f pins
        # the Pipeline's identities.
        ores = search_cross_validation(pipe, gs(online_cv, grid), rd)
        @test isapprox(ores.test_scores, res.test_scores; atol = 1e-6)
        @test ores.idx == res.idx
        # A failed candidate loses the Pipeline's search too (ADR 0120).
        bad = WeightBounds(; lb = fill(0.5, N), ub = ones(N))
        ok = WeightBounds(; lb = zeros(N), ub = ones(N))
        sets = UniverseSets(; dict = Dict("nx" => nx))
        pbase = Pipeline(;
                         steps = (EmpiricalPrior(),
                                  MeanRisk(;
                                           opt = JuMPOptimiser(; pe = EmpiricalPrior(),
                                                               slv = slv, sets = sets))))
        fres = search_cross_validation(pbase,
                                       gs(batch_cv, ["steps[2].opt.wb" => [bad, ok]]), rd)
        @test all(isnan, fres.test_scores[:, 1])
        @test all(isfinite, fres.test_scores[:, 2])
        @test fres.idx == 2
    end

    @testset "9. Executor: a search over an Online Scheme with a threaded executor" begin
        seq = search_cross_validation(mr, gs(online_cv, jgrid; ex = FLoops.SequentialEx()),
                                      rdg)
        thr = search_cross_validation(mr, gs(online_cv, jgrid; ex = FLoops.ThreadedEx()),
                                      rdg)
        @test thr.test_scores == seq.test_scores
        @test thr.idx == seq.idx
    end

    @testset "10. The grid, an empty value vector, and the combinatorial paths" begin
        # The grid is the product of the value vectors, with the first key fastest, and a
        # vector of sets concatenates the grids of its sets.
        lg, vg = po.lens_val_grid(["opt.l1" => [1, 2, 3], "opt.l2" => [10, 20]])
        @test vg == [(1, 10), (2, 10), (3, 10), (1, 20), (2, 20), (3, 20)]
        @test length(lg) == 6 && all(==(lg[1]), lg)
        sets = Union{Vector{Pair{String, Vector{Int}}}, Dict{String, Vector{Int}}}[["opt.l1" =>
                                                                                        [1,
                                                                                         2]],
                                                                                   Dict("opt.l2" =>
                                                                                            [10,
                                                                                             20,
                                                                                             30])]
        @test po.lens_val_grid(sets)[2] == [(1,), (2,), (10,), (20,), (30,)]
        # An empty value vector makes the product empty. The grid refuses it by name
        # before any candidate is fitted; the search used to report that every candidate
        # failed a fold.
        for p in (["opt.l1" => Float64[], "opt.l2" => [0.1]], Dict("opt.l1" => Float64[]))
            err = try
                po.lens_val_grid(p)
                nothing
            catch e
                e
            end
            @test isa(err, IsEmptyError) && occursin("`opt.l1`", sprint(showerror, err))
        end
        err = try
            search_cross_validation(mr, gs(KFold(; n = 3), ["opt.l1" => Float64[]]), rd)
            nothing
        catch e
            e
        end
        @test isa(err, IsEmptyError)
        # The combinatorial method scores each path by the risk of its pooled series, and
        # scores each fold of a path over the fold's own training window, in path order.
        ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        cvr = split(ccv, rd)
        res = search_cross_validation(mr,
                                      gs(ccv, jgrid; train_score = true,
                                         ex = FLoops.SequentialEx()), rd)
        l1 = po.parse_lens("opt.l1")
        for (i, val) in enumerate(jgrid[1][2])
            pp = po.fit_and_predict(Accessors.set(mr, l1, val), rd, ccv)
            @test res.test_scores[:, i] == [-expected_risk(r, path) for path in pp.pred]
            for (q, path) in enumerate(pp.pred)
                ks = [I[2] for I in findall(==(q), cvr.path_ids)]
                @test all(k -> path.pred[k].res.pr.X == view(rd.X, cvr.train_idx[ks[k]], :),
                          eachindex(path.pred))
                @test res.train_scores[q][:, i] ==
                      [-expected_risk(r, fp.res) for fp in path.pred]
            end
        end
        # A candidate that fails a path never wins.
        bad = WeightBounds(; lb = fill(0.5, N), ub = ones(N))
        ok = WeightBounds(; lb = zeros(N), ub = ones(N))
        base = MeanRisk(;
                        opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                            sets = UniverseSets(; dict = Dict("nx" => nx))))
        fres = search_cross_validation(base, gs(ccv, ["opt.wb" => [bad, ok]]), rd)
        @test all(isnan, fres.test_scores[:, 1])
        @test all(isfinite, fres.test_scores[:, 2])
        @test fres.idx == 2
    end

    @testset "11. The randomised search draws `n_iter` joint trials" begin
        # Each trial is one point of the space, a joint value of every key, so the count of
        # candidates is `n_iter`, not `n_iter` per key multiplied over the keys.
        g3 = ["opt.a" => [1, 2, 3, 4], "opt.b" => [10, 20, 30, 40], "opt.c" => [5, 6, 7, 8]]
        trials = po.make_p_grid(g3, 3, StableRNG(1))
        lg, vg = po.lens_val_grid(trials)
        @test length(trials) == length(vg) == 3
        full = po.lens_val_grid(g3)[2]
        @test allunique(vg) && all(in(full), vg)
        # A space of vectors alone gives distinct points in the order of the grid, so
        # `n_iter` at the size of the grid or more gives the grid itself.
        @test issorted([findfirst(==(v), full) for v in vg])
        @test po.lens_val_grid(po.make_p_grid(g3, 64, StableRNG(1)))[2] == full
        @test po.lens_val_grid(po.make_p_grid(g3, 100, StableRNG(1)))[2] == full
        sets = Union{Vector{Pair{String, Vector{Int}}}, Dict{String, Vector{Int}}}[["opt.a" =>
                                                                                        [1,
                                                                                         2,
                                                                                         3],
                                                                                    "opt.b" =>
                                                                                        [10,
                                                                                         20]],
                                                                                   Dict("opt.c" =>
                                                                                            [7,
                                                                                             8])]
        @test po.lens_val_grid(po.make_p_grid(sets, 8, StableRNG(2)))[2] ==
              po.lens_val_grid(sets)[2]
        # Every point of a space of 12 comes in 3 of 12 draws, a quarter of the time.
        sets12 = [["opt.a" => [1, 2, 3], "opt.b" => [10, 20]], ["opt.c" => collect(1:6)]]
        R = 4000
        rng12 = StableRNG(3)
        cnt = countmap(reduce(vcat,
                              [po.lens_val_grid(po.make_p_grid(sets12, 3, rng12))[2]
                               for _ in 1:R]))
        @test length(cnt) == 12
        @test all(c -> isapprox(c / R, 0.25; atol = 0.03), values(cnt))
        # A space past `typemax(Int)`: 32 keys of 5 values give 5^32 points.
        big_ps = ["k$j" => collect(1:5) .+ 10j for j in 1:32]
        bt = po.make_p_grid(big_ps, 4, StableRNG(4))
        @test length(bt) == 4 && allunique(bt)
        @test all(t -> all(j -> t[j][2][1] in big_ps[j][2], 1:32), bt)
        bidx = po.sample_grid_positions(StableRNG(5), big(5)^32, 6)
        @test eltype(bidx) == BigInt && allunique(bidx) && issorted(bidx)
        @test all(i -> 1 <= i <= big(5)^32, bidx)
        @test po.sample_grid_positions(StableRNG(5), 12, 12) == 1:12
        # A distribution makes every trial an independent draw: a set chosen with equal
        # probability, and one value of each of its keys.
        mix = Union{Vector{Pair{String, Any}}, Dict{String, Vector{Int}}}[Pair{String, Any}["opt.a" => [1,
                                                                                                        2,
                                                                                                        3],
                                                                                            "opt.x" => Normal(2,
                                                                                                              3)],
                                                                          Dict("opt.c" =>
                                                                                   [7, 8])]
        mt = po.make_p_grid(mix, 6000, StableRNG(6))
        @test length(mt) == 6000
        s1 = filter(t -> length(t) == 2, mt)
        s2 = filter(t -> length(t) == 1, mt)
        @test length(s1) + length(s2) == 6000
        @test all(t -> t[1][1] == "opt.c" && t[1][2][1] in (7, 8), s2)
        @test isapprox(length(s1) / 6000, 0.5; atol = 0.03)
        @test isapprox(mean(t[2][2][1] for t in s1), 2; atol = 0.2)
        @test all(t -> t[1][2][1] in 1:3, s1)
        mv = po.make_p_grid(Pair{String, Any}["opt.m" => MvNormal(zeros(2), I(2))], 2,
                            StableRNG(7))
        @test all(t -> t[1][2][1] isa AbstractVector && length(t[1][2][1]) == 2, mv)
        # An empty value vector is refused by name, and the trial count meets the cap of
        # the grid before any draw.
        for p in (Pair{String, Any}["opt.a" => Float64[], "opt.x" => Normal()],
                  ["opt.a" => Float64[], "opt.b" => [0.1]])
            err = try
                po.make_p_grid(p, 2, StableRNG(1))
            catch e
                e
            end
            @test err isa IsEmptyError
            @test occursin("`opt.a`", sprint(showerror, err))
        end
        @test_throws DomainError po.make_p_grid(["opt.x" => Normal()], 100_001,
                                                StableRNG(1))
        @test_throws DomainError po.make_p_grid(["opt.a" => 1:1000, "opt.b" => 1:1000],
                                                100_001, StableRNG(1))
        # The search scores the trials as the grid search scores its grid.
        cgrid = ["opt.wb" => wbgrid[1][2],
                 "opt.pe.ce.ce.ce.ce" => [SimpleCovariance(; corrected = true),
                                          SimpleCovariance(; corrected = false)]]
        kcv = KFold(; n = 4)
        gres = search_cross_validation(hrp, gs(kcv, cgrid; ex = SequentialEx()), rd)
        rs6 = RandomisedSearchCrossValidation(cgrid; cv = kcv, r = r, ex = SequentialEx(),
                                              n_iter = 6, seed = 1)
        rres = search_cross_validation(hrp, rs6, rd)
        @test rres.test_scores == gres.test_scores
        @test rres.val_grid == gres.val_grid && rres.idx == gres.idx
        rs3 = RandomisedSearchCrossValidation(cgrid; cv = kcv, r = r, ex = SequentialEx(),
                                              n_iter = 3, seed = 1)
        r3 = search_cross_validation(hrp, rs3, rd)
        cols = [findfirst(==(v), gres.val_grid) for v in r3.val_grid]
        @test size(r3.test_scores) == (4, 3)
        @test r3.test_scores == gres.test_scores[:, cols]
    end

    #=
    Issue #1937: the search keeps its `n_best` best candidates in rank order, as
    configurations built from the estimator that it was given, through the per-candidate
    value `candidate_scores` of the scorer.
    =#
    @testset "12. The search keeps its n best candidates in rank order" begin
        @eval struct OnlyIndexScore <: PortfolioOptimisers.CrossValidationSearchScorer end
        @eval (::OnlyIndexScore)(X::AbstractMatrix) = argmax(vec(minimum(X; dims = 1)))
        hms = HighestMeanScore()
        kcv = KFold(; n = 4)
        cgrid = ["opt.wb" => wbgrid[1][2],
                 "opt.pe.ce.ce.ce.ce" => [SimpleCovariance(; corrected = true),
                                          SimpleCovariance(; corrected = false)]]
        means(res) = vec(mean(res.test_scores; dims = 1))

        # n_best = 1 is the released Result: the index of the index form, and one kept
        # candidate, which is `opt` itself.
        r1 = search_cross_validation(hrp, gs(kcv, cgrid; ex = SequentialEx()), rd)
        @test r1.idx == hms(r1.test_scores)
        @test r1.rank_idx == [r1.idx] && r1.ranked[1] === r1.opt
        @test r1.rank_scores == [means(r1)[r1.idx]]
        @test isnothing(r1.path_ids)

        # n_best = k keeps the first k of the stable descending order of the column means.
        r4 = search_cross_validation(hrp, gs(kcv, cgrid; n_best = 4, ex = SequentialEx()),
                                     rd)
        @test r4.test_scores == r1.test_scores && r4.opt === r4.ranked[1]
        @test r4.rank_idx == sortperm(means(r4); rev = true)[1:4]
        @test r4.rank_scores == means(r4)[r4.rank_idx]
        @test r4.idx == r1.idx && length(r4.ranked) == 4
        for (cand, k) in zip(r4.ranked, r4.rank_idx)
            @test (cand.opt.wb, cand.opt.pe.ce.ce.ce.ce) == r4.val_grid[k]
        end
        # n_best larger than the grid keeps the whole grid, with no warning.
        r9 = search_cross_validation(hrp, gs(kcv, cgrid; n_best = 9, ex = SequentialEx()),
                                     rd)
        @test r9.rank_idx == sortperm(means(r9); rev = true)

        # The gap: a joined grid with different keys. Each kept candidate is built from the
        # base estimator, so a covariance candidate keeps the base weight bounds, where a
        # candidate built from `opt` carries the bounds of the winner.
        sets = [Pair{String, Vector}["opt.wb" => wbgrid[1][2]],
                Pair{String, Vector}["opt.pe.ce.ce.ce.ce" => [SimpleCovariance(;
                                                                               corrected = false)]]]
        rj = search_cross_validation(hrp, gs(kcv, sets; n_best = 4, ex = SequentialEx()),
                                     rd)
        @test length(rj.ranked) == 4
        for (cand, k) in zip(rj.ranked, rj.rank_idx)
            ref = po.search_candidate(hrp, rj.lens_grid[k], rj.val_grid[k])
            @test cand.opt.wb == ref.opt.wb &&
                  cand.opt.pe.ce.ce.ce.ce == ref.opt.pe.ce.ce.ce.ce
        end
        j = findfirst(==(4), rj.rank_idx)
        @test rj.ranked[j].opt.wb == hrp.opt.wb
        # Built from a winner of the first set, candidate 4 would carry its bounds.
        for k in 2:3
            win = po.search_candidate(hrp, rj.lens_grid[k], rj.val_grid[k])
            from_opt = po.search_candidate(win, rj.lens_grid[4], rj.val_grid[4])
            @test from_opt.opt.wb != rj.ranked[j].opt.wb
        end

        # A tie keeps grid order, in the order and in the index form.
        S = [0.5 0.8 0.8 0.25
             0.7 0.6 0.6 0.25]
        o, v = po.finite_candidate_order(hms, S, 4)
        @test o == [2, 3, 1, 4] && v == vec(mean(S; dims = 1))[o]
        @test hms(S) == 2
        # A failed candidate never enters the order, and the cap is the count of finite
        # columns.
        Sf = [NaN 0.5 0.8
              0.2 0.5 0.8]
        @test po.finite_candidate_order(hms, Sf, 3) == ([3, 2], [0.8, 0.5])
        # The index form of HighestMeanScore is argmax of its values.
        X = [0.5 0.75 0.25; 1.0 0.5 0.25]
        @test po.candidate_scores(hms, X) == vec(mean(X; dims = 1))
        @test argmax(po.candidate_scores(hms, X)) == hms(X)

        # Refusals: a count below one, and a function scorer with more than one candidate,
        # at construction. A function scorer keeps working at n_best = 1.
        fsc = X -> argmax(vec(minimum(X; dims = 1)))
        @test_throws DomainError GridSearchCrossValidation(cgrid; n_best = 0)
        @test_throws DomainError RandomisedSearchCrossValidation(cgrid; n_best = 0)
        for est in (GridSearchCrossValidation, RandomisedSearchCrossValidation)
            err = try
                est(cgrid; scorer = fsc, n_best = 2)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError && occursin("candidate_scores", err.msg)
        end
        rf = search_cross_validation(hrp, gs(kcv, cgrid; scorer = fsc, ex = SequentialEx()),
                                     rd)
        @test rf.idx == fsc(rf.test_scores) && rf.rank_idx == [rf.idx]
        @test isnothing(rf.rank_scores)
        # A subtype with the index form alone works at n_best = 1 and is refused at 2.
        @test isnothing(po.candidate_scores(OnlyIndexScore(), X))
        ro = search_cross_validation(hrp,
                                     gs(kcv, cgrid; scorer = OnlyIndexScore(),
                                        ex = SequentialEx()), rd)
        @test ro.idx == rf.idx && isnothing(ro.rank_scores)
        err = try
            po.finite_candidate_order(OnlyIndexScore(), S, 2)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError &&
              occursin("OnlyIndexScore", err.msg) &&
              occursin("candidate_scores", err.msg)
        # A subtype's method on any matrix type is more specific than the fallback, so it is
        # never ambiguous with it.
        @eval struct WorstFoldScore <: PortfolioOptimisers.CrossValidationSearchScorer end
        @eval function PortfolioOptimisers.candidate_scores(::WorstFoldScore,
                                                            X::AbstractMatrix)
            return vec(minimum(X; dims = 1))
        end
        @test po.finite_candidate_order(WorstFoldScore(), S, 2) == ([2, 3], [0.6, 0.6])
        # R31 of #1817: the values must line up with the columns, and NaN has no place in an
        # order. A short vector picked a wrong candidate at n_best = 1 with no error, a long
        # one raised a BoundsError, and a NaN gave an order that depends on the sort. An
        # infinite value keeps its place.
        @eval struct ShortScore <: PortfolioOptimisers.CrossValidationSearchScorer end
        @eval PortfolioOptimisers.candidate_scores(::ShortScore, X::AbstractMatrix) = [1.0]
        @eval struct LongScore <: PortfolioOptimisers.CrossValidationSearchScorer end
        @eval function PortfolioOptimisers.candidate_scores(::LongScore, X::AbstractMatrix)
            return fill(1.0, size(X, 2) + 1)
        end
        @eval struct NaNScore <: PortfolioOptimisers.CrossValidationSearchScorer end
        @eval function PortfolioOptimisers.candidate_scores(::NaNScore, X::AbstractMatrix)
            return [1.0, NaN, 0.5, 0.25]
        end
        @eval struct InfScore <: PortfolioOptimisers.CrossValidationSearchScorer end
        @eval function PortfolioOptimisers.candidate_scores(::InfScore, X::AbstractMatrix)
            return [1.0, Inf, -Inf, 0.5]
        end
        for sc in (ShortScore(), LongScore()), nb in (1, 2)
            @test_throws DimensionMismatch po.finite_candidate_order(sc, S, nb)
        end
        err = try
            po.finite_candidate_order(NaNScore(), S, 2)
            nothing
        catch e
            e
        end
        @test err isa IsNonFiniteError && occursin("NaNScore", err.msg)
        @test po.finite_candidate_order(InfScore(), S, 4) ==
              ([2, 1, 4, 3], [Inf, 1.0, 0.5, -Inf])

        # The Result refuses kept fields that do not agree.
        @test_throws DimensionMismatch SearchCrossValidationResult(; opt = hrp,
                                                                   test_scores = S,
                                                                   train_scores = nothing,
                                                                   lens_grid = r1.lens_grid,
                                                                   val_grid = r1.val_grid,
                                                                   idx = 2, rank_idx = [3])

        # path_ids: the path of each fold under MultipleRandomised, 1:M under a
        # combinatorial scheme, and the combinatorial arm ranks by the same rule.
        mcv = MultipleRandomised(IndexWalkForward(60, 30); subset_size = 5, n_subsets = 2,
                                 seed = 7)
        rm = search_cross_validation(hrp, gs(mcv, cgrid; n_best = 2, ex = SequentialEx()),
                                     rd)
        @test rm.path_ids == split(mcv, rd).path_ids
        @test rm.rank_idx == sortperm(means(rm); rev = true)[1:2]
        ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        rc = search_cross_validation(hrp, gs(ccv, cgrid; n_best = 3), rd)
        @test rc.path_ids == 1:maximum(split(ccv, rd).path_ids)
        @test rc.rank_idx == sortperm(means(rc); rev = true)[1:3]
        @test rc.rank_scores == means(rc)[rc.rank_idx]

        # The randomised search forwards n_best.
        rr = search_cross_validation(hrp,
                                     RandomisedSearchCrossValidation(cgrid; cv = kcv, r = r,
                                                                     ex = SequentialEx(),
                                                                     n_iter = 6, seed = 1,
                                                                     n_best = 3), rd)
        @test rr.rank_idx == r4.rank_idx[1:3] && rr.rank_scores == r4.rank_scores[1:3]
    end

    @testset "A result's own prior scores its weights at the result's mask" begin
        # A fit over a point-in-time window reduces to the Coverage Universe and expands
        # its weights back, so `expected_risk(r, res)` views the weights at the mask
        # before they meet the result's own prior; a caller's prior is taken as given.
        cvr = split(batch_cv, rdg)
        for opt in (mr, hrp, InverseVolatility())
            res = optimise(opt, rows(rdg, cvr.train_idx[1]))
            @test count(res.imsk) == N - 1
            @test length(res.w) == N
            @test expected_risk(r, res) ==
                  expected_risk(r, res.w[res.imsk], res.pr, po.extract_fees(res, nothing))
            @test expected_risk(Variance(), res) ==
                  expected_risk(Variance(), res.w[res.imsk], res.pr)
            @test isapprox(expected_risk(r, res,
                                         prior(EmpiricalPrior(),
                                               rows(rdg, cvr.train_idx[1]))),
                           expected_risk(r, res); atol = 1e-12)
        end
        full = optimise(mr, rows(rd, cvr.train_idx[1]))
        @test expected_risk(r, full) == expected_risk(r, full.w, full.pr)
    end
end
#=
A search tunes a field of a prior with forwarded properties, issue #1497.

`search_candidate` builds each candidate with `Accessors.set`, which rebuilds every struct on
the lens path with `ConstructionBase.setproperties`. The default of that function refuses a
type whose `propertynames` differ from its `fieldnames`, and `@forward_properties` adds
forwarded names to nine priors, so each candidate failed before any fit. The macro now
generates the method. Each candidate must score as the candidate built by hand with the
keyword constructor, under the grid and the randomised search, directly, through a nested
prior, and through a Pipeline.
=#
@testset "A search tunes a field of a prior with forwarded properties" begin
    using Test, PortfolioOptimisers, StableRNGs, Accessors
    po = PortfolioOptimisers
    rng = StableRNG(1495)
    F = 0.01 .* randn(rng, 250, 3)
    X = F * (0.5 .+ rand(rng, 3, 10)) .+ 0.005 .* randn(rng, 250, 10)
    rd = ReturnsResult(; nx = ["a$i" for i in 1:10], X = X, nf = ["f1", "f2", "f3"], F = F)
    cv = IndexWalkForward(120, 40)
    cvr = split(cv, rd)
    r = ConditionalValueatRisk()
    res_vals = [StepwiseRegression(), DimensionReductionRegression()]
    pe_vals = [EmpiricalPrior(), EmpiricalPrior(; ce = PortfolioOptimisersCovariance())]
    function scores_of(opt)
        return [-expected_risk(r,
                               po.fit_and_predict(opt, rd; train_idx = cvr.train_idx[j],
                                                  test_idx = cvr.test_idx[j]))
                for j in eachindex(cvr.train_idx)]
    end
    cases = (("pe.re", InverseVolatility(; pe = FactorPrior()), res_vals,
              v -> InverseVolatility(; pe = FactorPrior(; re = v))),
             ("pe.pe", InverseVolatility(; pe = FactorPrior()), pe_vals,
              v -> InverseVolatility(; pe = FactorPrior(; pe = v))),
             # Two decorated priors on one lens path.
             ("pe.pe.re",
              InverseVolatility(; pe = HighOrderPriorEstimator(; pe = FactorPrior())),
              res_vals,
              v -> InverseVolatility(;
                                     pe = HighOrderPriorEstimator(;
                                                                  pe = FactorPrior(;
                                                                                   re = v)))))
    for (lens, opt, vals, by_hand) in cases
        res = search_cross_validation(opt,
                                      GridSearchCrossValidation([lens => vals]; cv = cv,
                                                                r = r), rd)
        for (i, v) in enumerate(vals)
            @test res.test_scores[:, i] == scores_of(by_hand(v))
        end
        rres = search_cross_validation(opt,
                                       RandomisedSearchCrossValidation([lens => vals];
                                                                       cv = cv, r = r,
                                                                       n_iter = 2,
                                                                       seed = 7), rd)
        for (i, v) in enumerate(rres.val_grid)
            @test rres.test_scores[:, i] == scores_of(by_hand(v[1]))
        end
    end
    # A Pipeline reaches the prior through its step name.
    pipe = Pipeline(; steps = ("prior" => FactorPrior(), "opt" => InverseVolatility()))
    pres = search_cross_validation(pipe,
                                   GridSearchCrossValidation(["prior.re" => res_vals];
                                                             cv = cv, r = r), rd)
    for (i, v) in enumerate(res_vals)
        hand = Pipeline(;
                        steps = ("prior" => FactorPrior(; re = v),
                                 "opt" => InverseVolatility()))
        @test pres.test_scores[:, i] == scores_of(hand)
    end
    # A forwarded name has no storage of its own, so a lens onto it is refused before any
    # candidate is scored, and the message names the path that holds the value.
    err = try
        search_cross_validation(InverseVolatility(; pe = FactorPrior()),
                                GridSearchCrossValidation(["pe.ce" => [Covariance()]];
                                                          cv = cv, r = r), rd)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("forwards from `pe.ce`", sprint(showerror, err))
    @test occursin("Set `pe.ce` of the `FactorPrior` instead", sprint(showerror, err))
end
