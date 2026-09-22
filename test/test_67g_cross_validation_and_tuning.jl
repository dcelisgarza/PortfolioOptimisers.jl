#=
The online portfolio selection family against the cross-validation and tuning machinery.

The per-set files prove each rule's own step, and `test_67f` proves the family's promises on
the walk-forward it was built for. This file proves what the rest of the machinery does with
the head: every splitting scheme, both searches, and both kinds of time dependence. The head
is the library's only optimiser whose read-out is its own recursion (ADR 0155) and whose
Result carries no carrier (ADR 0158), so each seam is asserted rather than assumed.

Six groups:

 1. every scheme the library ships reaches the head, batch and stepped alike;
 2. `MultipleRandomised` crosses an asset subset with the stepped walk-forward, and the
    state is viewed by asset rather than refitted;
 3. both searches tune the head, over a rule knob, over the allocation set, and over the
    rule itself;
 4. a `TimeDependent` schedule reaches `fees` and `fb`, and is refused by type on the four
    fields that carry the recursion;
 5. a learning-rate schedule varies the rate per period, which is the family's own time
    dependence (ADR 0165);
 6. the downstream readers — `Resume`, `performance_summary` and `log_wealth_regret` —
    consume a stepped run.
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
        # The four fields that carry the recursion refuse a schedule by their type bound,
        # which is the enforcement the design rules ask for: a schedule replaces the
        # field's value every fold, and the state is threaded through these four.
        @test_throws TypeError OPS(;
                                   alg = TimeDependent(;
                                                       val = [ExponentiatedGradient(),
                                                              NewtonStep()],
                                                       default = ExponentiatedGradient()))
        @test_throws TypeError OPS(; alg = ExponentiatedGradient(),
                                   set = TimeDependent(;
                                                       val = [BoundedAllocationSet(),
                                                              BoundedAllocationSet()],
                                                       default = BoundedAllocationSet()))
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
end
