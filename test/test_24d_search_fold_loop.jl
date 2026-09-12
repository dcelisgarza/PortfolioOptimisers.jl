#=
The search scores every candidate through the one fold loop, issue #1020, against the
decision of #871 (ADR 0141).

`search_cross_validation` builds candidate `i` through the grid's lenses and scores it through
`fit_and_predict(opt_i, rd, gscv.cv; ex = SequentialEx())`, one row per fold, whatever the
scheme's Fold Fit. The map's closing test's second half is the identity: a grid search over an
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
          FLoops, Accessors
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
    online_cv = IndexWalkForward(w, t; purged_size = p, ff = OnlineStep())
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
                                    gs(IndexWalkForward(w, t; purged_size = p,
                                                        ff = OnlineStep(),
                                                        wd = SelfFinancingDrift(),
                                                        pws = DriftedWeights()), jgrid), rd)
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
            @test occursin(if isnothing(po.fold_fit(cv))
                               "search_cross_validation"
                           else
                               "online arm"
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
            @test po.finite_candidate_index(HighestMeanScore(), res.test_scores) == 1
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
        # The door takes a Fold Fit since #1022, and picks the batch candidate; test_24f pins
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

    @testset "9. Executor: a search over `OnlineStep` with a threaded executor" begin
        seq = search_cross_validation(mr, gs(online_cv, jgrid; ex = FLoops.SequentialEx()),
                                      rdg)
        thr = search_cross_validation(mr, gs(online_cv, jgrid; ex = FLoops.ThreadedEx()),
                                      rdg)
        @test thr.test_scores == seq.test_scores
        @test thr.idx == seq.idx
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
