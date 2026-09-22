#=
The online portfolio selection family against the cross-validation and tuning machinery.

The per-set files prove each rule's own step, and `test_67f` proves the family's promises on
the walk-forward it was built for. This file proves what the rest of the machinery does with
the head: every splitting scheme, both searches, and both kinds of time dependence. The head
is the library's only optimiser whose read-out is its own recursion (ADR 0155) and whose
Result carries no carrier (ADR 0158), so each seam is asserted rather than assumed.

Eight groups:

 1. every scheme the library ships reaches the head, batch and stepped alike;
 2. `MultipleRandomised` crosses an asset subset with the stepped walk-forward, and the
    state is viewed by asset rather than refitted;
 3. both searches tune the head, over a rule knob, over the allocation set, and over the
    rule itself;
 4. a `TimeDependent` schedule reaches `fees`, `fb` and the Allocation Set, and is refused
    by type on the two fields that carry the recursion;
 5. a learning-rate schedule varies the rate per period, which is the family's own time
    dependence (ADR 0165);
 6. the downstream readers — `Resume`, `performance_summary` and `log_wealth_regret` —
    consume a stepped run;
 7. a search scores the head in sample, over every arm that writes a train score (#1240);
 8. the head refuses `Online(head)` and refuses to own a Pipeline's rows, and the routes
    those two messages name all work (#1241, #1242).
=#

@testset "Online portfolio selection: cross-validation and tuning" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection

    rng = StableRNG(7)
    T, N = 60, 4
    R = 0.02 .* randn(rng, T, N)
    nx = ["A", "B", "C", "D"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = R, ts = ts)
    rows(r, i) = po.port_opt_view(r, i, :)
    eg() = OPS(; alg = ExponentiatedGradient())

    @testset "1. Every scheme reaches the head" begin
        # The batch schemes refit each fold by the Causal Pass over the fold's training
        # window, so every scheme the library enumerates reaches the head unchanged.
        for cv in (KFold(; n = 3), IndexWalkForward(20, 5),
                   IndexWalkForward(20, 5; expand_train = true), DateWalkForward(Week(2), 1),
                   HindsightSplit())
            @test isa(cross_val_predict(eg(), rd, cv), MultiPeriodPredictionResult)
        end
        # A combinatorial split recombines its test groups into paths.
        @test isa(cross_val_predict(eg(), rd,
                                    CombinatorialCrossValidation(; n_folds = 5,
                                                                 n_test_folds = 2)),
                  PopulationPredictionResult)
        # The Online Schemes take the stepped arm instead: one estimator, threaded.
        for cv in (OnlineIndexWalkForward(20, 5), OnlineDateWalkForward(Week(2), 1),
                   OnlineHindsightSplit())
            @test isa(cross_val_predict(eg(), rd, cv), MultiPeriodPredictionResult)
        end
        # The identity ADR 0155 states: the stepped arm reaches the expanding batch arm
        # fold for fold, because a purged walk-forward folds every row its training window
        # reaches and drops none.
        on = cross_val_predict(eg(), rd, OnlineIndexWalkForward(20, 5))
        ba = cross_val_predict(eg(), rd, IndexWalkForward(20, 5; expand_train = true))
        @test length(on.pred) == length(ba.pred)
        @test all(k -> isapprox(on.pred[k].res.w, ba.pred[k].res.w; atol = 1e-12),
                  eachindex(on.pred))
    end

    @testset "2. A randomised asset path crosses the stepped walk-forward" begin
        # A path is one asset subset crossed with the walk-forward's folds, and the online
        # arm takes the path's view once, at the warm-up, so the threaded state is sliced
        # by asset rather than refitted per fold.
        cv = MultipleRandomised(OnlineIndexWalkForward(20, 5); subset_size = 3,
                                n_subsets = 2, seed = 1)
        res = cross_val_predict(eg(), rd, cv)
        @test isa(res, PopulationPredictionResult)
        @test length(res.pred) == 2
        # Each path's folds hold three names, the subset the path drew.
        @test all(p -> all(f -> length(f.res.w) == 3, p.pred), res.pred)
        # The batch form of the same scheme reaches the same shape.
        resb = cross_val_predict(eg(), rd,
                                 MultipleRandomised(IndexWalkForward(20, 5);
                                                    subset_size = 3, n_subsets = 2,
                                                    seed = 1))
        @test isa(resb, PopulationPredictionResult) && length(resb.pred) == 2
    end

    @testset "3. Both searches tune the head" begin
        cv = OnlineIndexWalkForward(20, 5)
        r = MeanReturn(; flag = true)
        # A rule's own knob, addressed by path through the head.
        gs = GridSearchCrossValidation(["alg.eta" => [0.05, 0.2]]; cv = cv, r = r)
        res = search_cross_validation(eg(), gs, rd)
        @test res.opt.alg.eta in (0.05, 0.2)
        @test size(res.test_scores, 2) == 2
        # The allocation set is problem definition, and is tuned the same way.
        sets = [BoundedAllocationSet(),
                BoundedAllocationSet(; wb = WeightBounds(; lb = 0.05, ub = 0.5))]
        gset = GridSearchCrossValidation(["set" => sets]; cv = cv, r = r)
        rset = search_cross_validation(eg(), gset, rd)
        @test isa(rset.opt.set, BoundedAllocationSet)
        # The rule itself is one field, so a search chooses between two rules.
        galg = GridSearchCrossValidation(["alg" => [ExponentiatedGradient(), NewtonStep()]];
                                         cv = cv, r = r)
        ralg = search_cross_validation(eg(), galg, rd)
        @test isa(ralg.opt.alg, Union{<:MirrorDescent, <:NewtonStep})
        # The randomised search samples the same grid.
        rs = RandomisedSearchCrossValidation(["alg.eta" => [0.05, 0.1, 0.2]]; cv = cv,
                                             r = r, n_iter = 2, seed = 1)
        rres = search_cross_validation(eg(), rs, rd)
        @test size(rres.test_scores, 2) == 2
        # A search scores over a batch scheme and over a randomised asset path too.
        @test isa(search_cross_validation(eg(),
                                          GridSearchCrossValidation(["alg.eta" =>
                                                                         [0.05, 0.2]];
                                                                    cv = IndexWalkForward(20,
                                                                                          5),
                                                                    r = r), rd),
                  SearchCrossValidationResult)
        @test isa(search_cross_validation(eg(),
                                          GridSearchCrossValidation(["alg.eta" =>
                                                                         [0.05, 0.2]];
                                                                    cv = MultipleRandomised(OnlineIndexWalkForward(20,
                                                                                                                   5);
                                                                                            subset_size = 3,
                                                                                            n_subsets = 2,
                                                                                            seed = 1),
                                                                    r = r), rd),
                  SearchCrossValidationResult)
        # The search door refuses a warm estimator by name: a search tunes the
        # configuration alone, so a state it was handed would reach no fold.
        warm = po.partial_fit!(eg(), rows(rd, 1:10))
        @test_throws ArgumentError search_cross_validation(warm, gs, rd)
    end

    @testset "4. A per-fold schedule reaches the problem definition alone" begin
        cv = OnlineIndexWalkForward(20, 5)
        K = length(cross_val_predict(OPS(; alg = BuyAndHold()), rd, cv).pred)
        fees = Fees(; l = 0.01)
        # `fees` and `fb` are problem definition: neither carries the recursion, so a
        # schedule on either resolves per fold and composes with the threaded state.
        for f in (TimeDependent(; val = fill(fees, K), default = nothing),
                  TimeDependent(; val = ctx -> fees, default = nothing))
            @test isa(cross_val_predict(OPS(; alg = BuyAndHold(), fees = f), rd, cv),
                      MultiPeriodPredictionResult)
        end
        @test isa(cross_val_predict(OPS(; alg = BuyAndHold(),
                                        fb = TimeDependent(; val = fill(EqualWeighted(), K),
                                                           default = nothing)), rd, cv),
                  MultiPeriodPredictionResult)
        # A scheduled fee is charged: the scheduled run loses to the free one every fold.
        pf = cross_val_predict(OPS(; alg = BuyAndHold(),
                                   fees = TimeDependent(; val = fill(fees, K),
                                                        default = nothing)), rd, cv)
        p0 = cross_val_predict(OPS(; alg = BuyAndHold()), rd, cv)
        @test all(<(0), pf.mrd.X .- p0.mrd.X)
        # The Allocation Set is problem definition too, and it is the one scheduled field
        # the recursion's *step* reads: every row's raw step is projected onto it. So the
        # fold loop resolves it one fold ahead of the read-out, before the fold's rows are
        # folded, and fold `i` steps inside entry `i` (#1243).
        ubs = collect(range(0.65, 0.26, K))
        sets = [BoundedAllocationSet(; wb = WeightBounds(; lb = 0.0, ub = u)) for u in ubs]
        caps = TimeDependent(sets; default = BoundedAllocationSet())
        w0 = [0.7, 0.1, 0.1, 0.1]
        capped = cross_val_predict(OPS(; alg = BuyAndHold(), set = caps, w0 = w0), rd, cv)
        @test isa(capped, MultiPeriodPredictionResult)
        # Fold `i`'s allocation honours entry `i`'s cap, and the cap tightens fold by fold.
        @test all(i -> maximum(capped.pred[i].res.w) <= ubs[i] + 1e-8, 1:K)
        # The cap is read by the step and not by the read-out alone: the same run at the
        # loosest entry alone leaves every fold above the tight folds' caps, so a run that
        # folded its rows under entry 1 could not answer what this one answers.
        loose = cross_val_predict(OPS(; alg = BuyAndHold(), set = first(sets), w0 = w0), rd,
                                  cv)
        @test all(i -> maximum(loose.pred[i].res.w) > ubs[i] + 1e-8, 4:K)
        # A callable entry is resolved the same way, and its rows cap is `nothing` —
        # uncapped — because its entries do not exist before the fold does.
        cb = TimeDependent(; val = ctx -> sets[ctx.i], default = BoundedAllocationSet())
        @test isnothing(po.rows_needed(cb))
        cbres = cross_val_predict(OPS(; alg = BuyAndHold(), set = cb, w0 = w0), rd, cv)
        @test isapprox(cbres.pred[end].res.w, capped.pred[end].res.w; atol = 1e-12)
        # A vector schedule's rows cap is the maximum over its entries, so a buffer sized
        # at the seed still holds what a later entry reads.
        @test iszero(po.rows_needed(caps))
        # Outside a fold loop the schedule is inert: the fold-less solve runs the
        # schedule's own `default`, which is the simplex, so no cap binds.
        @test maximum(optimise(OPS(; alg = BuyAndHold(), set = caps, w0 = w0), rd).w) >
              ubs[end]
        # The batch arm takes the same schedule, as it takes every other one.
        @test isa(cross_val_predict(OPS(; alg = BuyAndHold(), set = caps, w0 = w0), rd,
                                    IndexWalkForward(20, 5)), MultiPeriodPredictionResult)
        # `Resume` re-enters a scheduled run: the Result carries the head with its schedule
        # unresolved, so the resumed folds resolve from the schedule and not from the entry
        # the last run stopped in. The schedule is the callable one, because a vector states
        # one entry per fold and the short run enumerates fewer folds than the full one.
        short = cross_val_predict(OPS(; alg = BuyAndHold(), set = cb, w0 = w0),
                                  rows(rd, 1:45), cv)
        @test isa(short.opt.set, TimeDependent)
        @test isapprox(cross_val_predict(Resume(short), rd, cv).pred[end].res.w,
                       cbres.pred[end].res.w; atol = 1e-12)
        # A vector schedule states one entry per fold, and the loop counts them.
        @test_throws DimensionMismatch cross_val_predict(OPS(; alg = BuyAndHold(),
                                                             set = TimeDependent(sets[1:2];
                                                                                 default = BoundedAllocationSet())),
                                                         rd, cv)
        # Each entry is held to the set refusals, at construction, one entry at a time.
        @test_throws DomainError OPS(; alg = ExponentiatedGradient(),
                                     set = TimeDependent([BoundedAllocationSet(),
                                                          BoundedAllocationSet(;
                                                                               wb = WeightBounds(;
                                                                                                 lb = -0.1,
                                                                                                 ub = 1.0))];
                                                         default = BoundedAllocationSet()))
        # The two fields that carry the recursion refuse a schedule by their type bound,
        # which is the enforcement the design rules ask for: a schedule replaces the
        # field's value every fold, and the state is threaded through both (ADR 0155,
        # ADR 0162).
        @test_throws TypeError OPS(;
                                   alg = TimeDependent(;
                                                       val = [ExponentiatedGradient(),
                                                              NewtonStep()],
                                                       default = ExponentiatedGradient()))
        @test_throws TypeError OPS(; alg = ExponentiatedGradient(),
                                   w0 = TimeDependent(;
                                                      val = [fill(inv(N), N),
                                                             fill(inv(N), N)],
                                                      default = fill(inv(N), N)))
        # A schedule of whole heads at the root is refused by name under an Online Scheme,
        # because the loop threads one estimator and a schedule is a different one per
        # fold. The batch arm refits every fold, so it takes the same schedule.
        sched = TimeDependent(; val = [OPS(; alg = BuyAndHold()) for _ in 1:K],
                              default = OPS(; alg = BuyAndHold()))
        @test_throws ArgumentError cross_val_predict(sched, rd, cv)
        @test isa(cross_val_predict(sched, rd, IndexWalkForward(20, 5)),
                  MultiPeriodPredictionResult)
    end

    @testset "5. The family's own time dependence is per period" begin
        # A `TimeDependent` varies a field per fold. A learning-rate schedule varies the
        # rate per row inside the recursion, which is where a rate that decays with the
        # horizon belongs, and it needs no fold at all.
        cv = OnlineIndexWalkForward(20, 5)
        sched = OPS(; alg = ExponentiatedGradient(; eta = InverseSquareRootRate()))
        fixed = OPS(; alg = ExponentiatedGradient(; eta = 0.05))
        @test isa(cross_val_predict(sched, rd, cv), MultiPeriodPredictionResult)
        # The two paths differ, so the schedule is read rather than ignored.
        @test optimise(sched, rd).w != optimise(fixed, rd).w
        # A fold-less solve reads it too: the rate is the rule's, not the loop's.
        @test isapprox(sum(optimise(sched, rd).w), 1; atol = 1e-12)
        # A search tunes the schedule's own knob by path.
        gs = GridSearchCrossValidation(["alg.eta.c" => [0.5, 1.0]]; cv = cv,
                                       r = MeanReturn(; flag = true))
        @test search_cross_validation(sched, gs, rd).opt.alg.eta.c in (0.5, 1.0)
    end

    @testset "6. The downstream readers consume a stepped run" begin
        cv = OnlineIndexWalkForward(20, 5)
        # `Resume` re-enters the recursion where the first run stopped, so the resumed run
        # and the one-shot run agree on the folds they share.
        pr = cross_val_predict(eg(), rows(rd, 1:45), cv)
        pr2 = cross_val_predict(Resume(pr), rd, cv)
        one = cross_val_predict(eg(), rd, cv)
        @test isapprox(pr2.pred[end].res.w, one.pred[end].res.w; atol = 1e-12)
        # The summary and the regret verb both read a stepped run's predictions.
        @test isa(performance_summary(one), PerformanceSummaryResult)
        bah = cross_val_predict(OPS(; alg = BuyAndHold()), rd, cv)
        reg = log_wealth_regret(one, bah)
        @test isa(reg, LogWealthRegretResult)
        @test length(reg.difference) == length(one.mrd.X)
    end

    @testset "7. A search scores the head in sample (#1240)" begin
        # The head is the library's only optimiser whose Result carries no carrier (ADR
        # 0155 rules `pr = nothing` in both arms), so the train score had no returns to
        # read and every `train_score = true` search raised a `MethodError`. The search now
        # hands the fold its own training window, which is the same data every other head
        # keeps on `res.pr`.
        r = MeanReturn(; flag = true)
        for cv in (IndexWalkForward(20, 5), KFold(; n = 3),
                   IndexWalkForward(20, 5; expand_train = true), OnlineIndexWalkForward(20, 5))
            res = search_cross_validation(eg(),
                                          GridSearchCrossValidation(["alg.eta" =>
                                                                         [0.05, 0.2]];
                                                                    cv = cv, r = r,
                                                                    train_score = true), rd)
            @test isa(res.train_scores, Matrix)
            @test size(res.train_scores) == size(res.test_scores)
            @test all(isfinite, res.train_scores)
        end

        # A randomised asset path draws a subset per fold, so the window is taken at the
        # fold's own columns as well as its own rows.
        mr = search_cross_validation(eg(),
                                     GridSearchCrossValidation(["alg.eta" => [0.05, 0.2]];
                                                               cv = MultipleRandomised(IndexWalkForward(20,
                                                                                                        5);
                                                                                       subset_size = 3,
                                                                                       n_subsets = 2,
                                                                                       seed = 1),
                                                               r = r, train_score = true),
                                     rd)
        @test all(isfinite, mr.train_scores)

        # The combinatorial arm keeps one matrix per recombined path, and its folds are
        # read off the columns of a matrix-shaped `path_ids`.
        cb = search_cross_validation(eg(),
                                     GridSearchCrossValidation(["alg.eta" => [0.05, 0.2]];
                                                               cv = CombinatorialCrossValidation(;
                                                                                                 n_folds = 5,
                                                                                                 n_test_folds = 2),
                                                               r = r, train_score = true),
                                     rd)
        @test isa(cb.train_scores, Vector{<:Matrix})
        @test all(m -> all(isfinite, m), cb.train_scores)

        # The randomised search forwards to the grid search, so it scores in sample too.
        rs = search_cross_validation(eg(),
                                     RandomisedSearchCrossValidation(["alg.eta" =>
                                                                          [0.05, 0.1, 0.2]];
                                                                     cv = IndexWalkForward(20,
                                                                                           5),
                                                                     r = r, n_iter = 2,
                                                                     seed = 1,
                                                                     train_score = true),
                                     rd)
        @test all(isfinite, rs.train_scores)

        # The window handed in is the fold's own, not the whole panel: the score of the
        # first fold equals the risk of that fold's weights over rows 1:20 alone.
        cv = IndexWalkForward(20, 5)
        sp = split(cv, rd)
        pd = cross_val_predict(eg(), rd, cv)
        one = search_cross_validation(eg(),
                                      GridSearchCrossValidation(["alg.eta" => [0.05]];
                                                                cv = cv, r = r,
                                                                train_score = true), rd)
        @test one.train_scores[1, 1] ==
              expected_risk(r, pd.pred[1].res, view(rd.X, sp.train_idx[1], :))

        # A value-level caller who hands no carrier in is refused by name rather than by a
        # `MethodError`, and the same call with a carrier answers.
        res1 = optimise(eg(), rd)
        @test isnothing(res1.pr)
        @test_throws IsNothingError expected_risk(r, res1)
        @test_throws IsNothingError expected_risk(ConditionalValueatRisk(), res1)
        @test isfinite(expected_risk(r, res1, rd.X))
    end

    @testset "8. The two refusals the head owes, and the routes they name (#1241, #1242)" begin
        # `Online(est)` declares a refit from a buffer. For a head both of its settings are
        # a batch walk-forward that runs today, so the wrapper adds no answer and is
        # refused at the construction door (ADR 0155). Every spelling reaches it.
        @test_throws ArgumentError Online(eg())
        @test_throws ArgumentError Online(eg(); max_history = 20)
        @test_throws ArgumentError Online(; est = eg())
        # The refusal names the two schemes that give the two answers.
        msg = try
            Online(eg())
        catch e
            sprint(showerror, e)
        end
        @test occursin("IndexWalkForward", msg)
        @test occursin("OnlineIndexWalkForward", msg)
        # The seed the refusal used to live on is gone: the head now falls to the generic
        # seed, which no caller can reach, because no `Online` can hold a head.
        @test isa(po.online_state_seed(eg(), nothing), po.SampleBufferState)

        # The uncapped refit is the expanding walk-forward, and its allocation is the
        # online arm's exactly: both take the same single-row updates from `w0`.
        wp(res) = reduce(vcat, transpose(p.w) for p in res.res)
        wo = wp(cross_val_predict(eg(), rd, OnlineIndexWalkForward(20, 1)))
        we = wp(cross_val_predict(eg(), rd, IndexWalkForward(20, 1; expand_train = true)))
        @test wo == we
        # The capped refit is the rolling walk-forward, which is the recursion restarted
        # from `w0` inside each window, and it differs from the online arm.
        wc = wp(cross_val_predict(eg(), rd, IndexWalkForward(20, 1)))
        @test size(wc) == size(wo)
        @test wc != wo

        # A head does not own a Pipeline's rows: its read-out is the recursion, so it
        # rebuilds no carrier for the universe steps to refit over. The head folded every
        # row and then died at the read-out; it is now refused at both doors.
        sel = ScoreSelector(; score = SCM(), rule = RankRule(; best = 3))
        bare() = Pipeline(; steps = [:opt => eg()])
        with_sel() = Pipeline(; steps = [:sel => sel, :opt => eg()])
        for p in (bare, with_sel)
            @test_throws ArgumentError po.assert_online_entry(p())
            @test_throws ArgumentError partial_fit!(p(), rows(rd, 1:1))
            @test_throws ArgumentError cross_val_predict(p(), rd,
                                                         OnlineIndexWalkForward(20, 5))
        end

        # A carrier would not mend it either: the read-out expresses a selection as a view
        # of the owner's state, and a view of a recursion is not the recursion over those
        # columns. The allocation is a path, the projection couples the columns, and the
        # wealth factor reads every one of them. Two rules show the gap at two sizes.
        function view_gap(alg, idx)
            h = OPS(; alg = alg)
            for t in 1:size(rd.X, 1)
                h = partial_fit!(h, rows(rd, t:t))
            end
            wv = optimise(po.port_opt_view(h, idx, rd.X)).w
            wcol = optimise(OPS(; alg = alg), po.port_opt_view(rd, :, idx)).w
            return maximum(abs, wv - wcol)
        end
        @test view_gap(AntiCorrelation(), [2, 4]) > 0.4
        @test view_gap(AdaptiveSubgradient(), [1, 3]) > 1e-2

        # And the routes the messages name all work. A prior step in front makes the prior
        # the row owner; `Online(pipe)` refits every step and equals the batch fit.
        @test isa(cross_val_predict(Pipeline(;
                                             steps = [:pe => EmpiricalPrior(),
                                                      :opt => eg()]), rd,
                                    OnlineIndexWalkForward(20, 5)),
                  PortfolioOptimisers.AbstractPredictionResult)
        op = po.update_online_estimator(Online(with_sel()))
        for t in 1:size(rd.X, 1)
            op = partial_fit!(op, rows(rd, t:t))
        end
        @test fit(op).ctx.opt.w == fit(with_sel(), rd).ctx.opt.w
    end
end
