@testset "Fee liquidation carriers" begin
    using PortfolioOptimisers, Test

    # ADR 0121 gives `Fees` and `FeesEstimator` two liquidation carriers, `lq` and `flq`.
    # They price the positions an optimisation is forced to sell when an asset leaves the
    # Investable Mask, so they live on the **complement** of that mask while the five older
    # per-asset fields live on the mask itself.
    #
    # This file pins the two types against each other. The estimator must carry the same two
    # fields as its result counterpart, resolve them by the same route `tn` takes, and reach
    # a `Fees` a caller could have written by hand.

    @testset "The estimator carries the same two fields as its result" begin
        fn_r = fieldnames(Fees)
        fn_e = fieldnames(FeesEstimator)
        for f in (:lq, :flq)
            @test f in fn_r
            @test f in fn_e
        end
        # Both types order the carriers after the five per-asset fields and before the
        # clock, so a reader of one reads the other.
        @test findfirst(==(:lq), fn_r) == findfirst(==(:fs), fn_r) + 1
        @test findfirst(==(:flq), fn_r) == findfirst(==(:lq), fn_r) + 1
        @test findfirst(==(:lq), fn_e) == findfirst(==(:fs), fn_e) + 1
        @test findfirst(==(:flq), fn_e) == findfirst(==(:lq), fn_e) + 1
        # The result keeps its clock and its `isapprox` keywords last on both types.
        @test fn_r[(end - 1):end] == (:fa, :kwargs)
        @test fn_e[(end - 1):end] == (:fa, :kwargs)
    end

    @testset "Both default to nothing, so a caller that states none is unchanged" begin
        @test isnothing(Fees().lq)
        @test isnothing(Fees().flq)
        @test isnothing(FeesEstimator().lq)
        @test isnothing(FeesEstimator().flq)
        # The keyword constructor forwards positionally, so the two orders must agree.
        @test Fees(; tn = Turnover(; w = [0.1, 0.2], val = 0.01), l = 0.002).lq === nothing
    end

    @testset "The estimator resolves its carriers exactly as it resolves `tn`" begin
        nx = ["a", "b", "c", "d"]
        sets = UniverseSets(; dict = Dict("nx" => nx))
        w = [0.25, 0.25, 0.25, 0.25]

        fest = FeesEstimator(;
                             tn = TurnoverEstimator(; w = w, val = Dict("a" => 0.001),
                                                    dval = 0.002),
                             lq = TurnoverEstimator(; w = w, val = Dict("c" => 0.010),
                                                    dval = 0.003),
                             flq = TurnoverEstimator(; w = w, val = Dict("c" => 5.0),
                                                     dval = 1.0))
        fees = fees_constraints(fest, sets)

        @test isa(fees, Fees)
        # A name-keyed rate with a default resolves to a full-length vector, the same route
        # `tn` takes, so the carriers are `Turnover` results after the door.
        @test isa(fees.lq, Turnover)
        @test isa(fees.flq, Turnover)
        @test fees.lq.val == [0.003, 0.003, 0.010, 0.003]
        @test fees.flq.val == [1.0, 1.0, 5.0, 1.0]
        @test fees.lq.w == w
        @test fees.flq.w == w
        # And it reaches exactly the `Fees` a caller could have written by hand.
        byhand = Fees(; tn = Turnover(; w = w, val = [0.001, 0.002, 0.002, 0.002]),
                      lq = Turnover(; w = w, val = [0.003, 0.003, 0.010, 0.003]),
                      flq = Turnover(; w = w, val = [1.0, 1.0, 5.0, 1.0]))
        @test fees.lq.val == byhand.lq.val
        @test fees.flq.val == byhand.flq.val
        @test fees.tn.val == byhand.tn.val

        # A carrier the caller leaves out stays out.
        @test isnothing(fees_constraints(FeesEstimator(; l = 0.001), sets).lq)
        @test isnothing(fees_constraints(FeesEstimator(; l = 0.001), sets).flq)
    end

    @testset "Either carrier alone asks for the previous weights" begin
        w = [0.25, 0.25, 0.25, 0.25]
        # A carrier prices a trade against a reference vector, so it needs the fold's
        # previous weights exactly as `tn` does, and `needs_previous_weights` reads all
        # three carriers rather than `tn` alone.
        @test PortfolioOptimisers.needs_previous_weights(Fees(;
                                                              lq = Turnover(; w = w,
                                                                            val = 0.01)))
        @test PortfolioOptimisers.needs_previous_weights(Fees(;
                                                              flq = Turnover(; w = w,
                                                                             val = 5.0)))
        @test !PortfolioOptimisers.needs_previous_weights(Fees(; l = 0.001))
        # A `fixed` carrier pins its own reference weights, so it asks for none.
        @test !PortfolioOptimisers.needs_previous_weights(Fees(;
                                                               lq = Turnover(; w = w,
                                                                             val = 0.01,
                                                                             fixed = true)))
    end

    @testset "`factory` threads the previous weights into both carriers" begin
        w = [0.25, 0.25, 0.25, 0.25]
        pw = [0.4, 0.3, 0.2, 0.1]
        # Both carriers are `@fprop`, so the fold's previous weights reach them beside
        # `tn.w`.
        fees = Fees(; tn = Turnover(; w = w, val = 0.001),
                    lq = Turnover(; w = w, val = 0.010), flq = Turnover(; w = w, val = 5.0))
        f = factory(fees, pw)
        @test f.tn.w == pw
        @test f.lq.w == pw
        @test f.flq.w == pw
    end

    @testset "The view splits the two axes at the mask and its complement" begin
        # This is what replaces a dedicated door verb. `port_opt_view` is handed the
        # **unreduced** returns matrix, so it derives the complement from `size(X, 2)` and
        # the indices it was given. Asset 3 of four leaves the universe.
        X = rand(7, 4)
        i = [1, 2, 4]
        w = [0.1, 0.2, 0.3, 0.4]
        fees = Fees(; tn = Turnover(; w = w, val = [0.001, 0.002, 0.010, 0.003]),
                    l = [0.01, 0.02, 0.03, 0.04], s = [0.05, 0.06, 0.07, 0.08],
                    fl = [1.0, 2.0, 3.0, 4.0], fs = [5.0, 6.0, 7.0, 8.0],
                    lq = Turnover(; w = w, val = [0.001, 0.002, 0.010, 0.003]),
                    flq = Turnover(; w = w, val = [9.0, 10.0, 11.0, 12.0]))
        v = PortfolioOptimisers.port_opt_view(fees, i, X)

        # The five per-asset fields keep the assets the portfolio holds.
        @test collect(v.l) == [0.01, 0.02, 0.04]
        @test collect(v.s) == [0.05, 0.06, 0.08]
        @test collect(v.fl) == [1.0, 2.0, 4.0]
        @test collect(v.fs) == [5.0, 6.0, 8.0]
        @test collect(v.tn.w) == [0.1, 0.2, 0.4]
        @test collect(v.tn.val) == [0.001, 0.002, 0.003]

        # The two carriers keep the asset that left, and nothing else.
        @test collect(v.lq.w) == [0.3]
        @test collect(v.lq.val) == [0.010]
        @test collect(v.flq.w) == [0.3]
        @test collect(v.flq.val) == [11.0]

        # The clock and the tolerance ride through untouched.
        @test v.kwargs == fees.kwargs
        @test v.fa === fees.fa

        # Every asset investable means nothing exited, so both carriers go. A `Turnover`
        # refuses an empty `w`, so `nothing` is the answer the type asks for as well as the
        # answer the rule asks for.
        vall = PortfolioOptimisers.port_opt_view(fees, [1, 2, 3, 4], X)
        @test isnothing(vall.lq)
        @test isnothing(vall.flq)
        @test collect(vall.l) == fees.l

        # The estimator takes the same two axes.
        fest = FeesEstimator(; tn = TurnoverEstimator(; w = w, val = 0.001),
                             l = [0.01, 0.02, 0.03, 0.04],
                             lq = TurnoverEstimator(; w = w, val = 0.010),
                             flq = TurnoverEstimator(; w = w, val = 5.0))
        ve = PortfolioOptimisers.port_opt_view(fest, i, X)
        @test collect(ve.l) == [0.01, 0.02, 0.04]
        @test collect(ve.tn.w) == [0.1, 0.2, 0.4]
        @test collect(ve.lq.w) == [0.3]
        @test collect(ve.flq.w) == [0.3]
        @test ve.dl === fest.dl
        @test ve.kwargs == fest.kwargs
    end

    @testset "A caller that hands no matrix leaves the carriers alone" begin
        # Without the unreduced matrix there is no complement to derive, so the fallback
        # slices the five per-asset fields and passes the carriers through. They already sit
        # on their own axis, so that is the correct answer rather than a degraded one.
        w = [0.1, 0.2, 0.3, 0.4]
        fees = Fees(; l = [0.01, 0.02, 0.03, 0.04],
                    lq = Turnover(; w = w, val = [0.001, 0.002, 0.010, 0.003]))
        v = PortfolioOptimisers.port_opt_view(fees, [1, 2, 4])
        @test collect(v.l) == [0.01, 0.02, 0.04]
        @test collect(v.lq.w) == w
    end

    @testset "An Investable Mask drives the split, end to end" begin
        using StableRNGs, LinearAlgebra, Clarabel

        # The point of the two carriers, in the setting they exist for. Asset 3 delists
        # part way through the window, so its column carries `NaN`. Nothing is poked into
        # the prior by hand: the prior fits the full universe, cannot estimate that asset,
        # and marks it, so the mask **derives itself** from the data. `investable_reduction`
        # then takes the door, and the five per-asset fields must come out on the mask with
        # the two carriers on its complement — no verb of their own, and no mask stored.
        rng = StableRNG(987654321)
        T, N = 200, 5
        Xf = randn(rng, T, N) ./ 100 .+ 0.0005
        k = 3
        keep = [1, 2, 4, 5]
        Xf[120:end, k] .= NaN          # the delisting
        nx = ["a", "b", "c", "d", "e"]
        rd = ReturnsResult(; nx = nx, X = Xf)
        slv = Solver(; name = :cl, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("verbose" => false, "max_step_fraction" => 0.75))

        prn = prior(EmpiricalPrior(), rd)
        imsk = PortfolioOptimisers.investable_mask(prn)
        @test imsk == BitVector([1, 1, 0, 1, 1])

        w = [0.2, 0.2, 0.2, 0.2, 0.2]
        lval = [0.01, 0.02, 0.03, 0.04, 0.05]
        qval = [0.001, 0.002, 0.010, 0.003, 0.004]
        fval = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Stated over the FULL universe, because the caller does not know which asset will
        # leave. The door is what selects.
        fees = Fees(; tn = Turnover(; w = w, val = qval), l = lval,
                    lq = Turnover(; w = w, val = qval), flq = Turnover(; w = w, val = fval))
        opt = JuMPOptimiser(; pe = prn, fees = fees, slv = slv)
        _, _, opt_v, _ = PortfolioOptimisers.investable_reduction(prn, opt, rd)

        # The holding fees keep the four investable assets.
        @test collect(opt_v.fees.l) == lval[keep]
        @test collect(opt_v.fees.tn.w) == w[keep]
        @test collect(opt_v.fees.tn.val) == qval[keep]

        # The liquidation carriers keep the one asset that left, and its rate is the rate
        # the caller stated for it.
        @test collect(opt_v.fees.lq.w) == [w[k]]
        @test collect(opt_v.fees.lq.val) == [qval[k]]
        @test collect(opt_v.fees.flq.w) == [w[k]]
        @test collect(opt_v.fees.flq.val) == [fval[k]]

        # The two axes partition the universe: nothing is counted twice, nothing is lost.
        @test length(opt_v.fees.l) + length(opt_v.fees.lq.w) == N

        # The natural setting: solve the problem. The optimisation reduces at its entry,
        # solves on the four surviving assets, and expands the weights back to the full
        # universe with a zero at the asset that delisted. The fees it actually used are on
        # its processed attributes, and they carry the two axes.
        res = optimise(MeanRisk(;
                                opt = JuMPOptimiser(; pe = prn, fees = fees,
                                                    wb = WeightBounds(; lb = 0, ub = 1),
                                                    bgt = 1, slv = slv)), rd)
        @test length(res.w) == N
        @test iszero(res.w[k])
        @test isapprox(sum(res.w), 1; atol = 1e-8)
        @test res.jr.pa.imsk == imsk
        used = res.jr.pa.fees
        @test length(used.l) == length(keep)
        @test collect(used.l) == lval[keep]
        @test collect(used.lq.w) == [w[k]]
        @test collect(used.lq.val) == [qval[k]]
        @test collect(used.flq.val) == [fval[k]]

        # A window with no delisting derives no mask, so the door is a no-op and the fees
        # are the ones the caller stated, carriers included.
        rdf = ReturnsResult(; nx = nx, X = randn(StableRNG(11), T, N) ./ 100 .+ 0.0005)
        prf = prior(EmpiricalPrior(), rdf)
        @test isnothing(PortfolioOptimisers.investable_mask(prf))
        opt_full = JuMPOptimiser(; pe = prf, fees = fees, slv = slv)
        _, _, opt_fv, _ = PortfolioOptimisers.investable_reduction(prf, opt_full, rdf)
        @test opt_fv.fees.lq === fees.lq
        @test opt_fv.fees.flq === fees.flq

        # Parity: the same mask, driven through the estimator, reaches the same two axes.
        sets = UniverseSets(; dict = Dict("nx" => nx))
        fest = FeesEstimator(; tn = TurnoverEstimator(; w = w, val = Dict(zip(nx, qval))),
                             l = Dict(zip(nx, lval)),
                             lq = TurnoverEstimator(; w = w, val = Dict(zip(nx, qval))),
                             flq = TurnoverEstimator(; w = w, val = Dict(zip(nx, fval))))
        opte = JuMPOptimiser(; pe = prn, fees = fest, sets = sets, slv = slv)
        _, _, opte_v, _ = PortfolioOptimisers.investable_reduction(prn, opte, rd)

        # The estimator's carriers land on the same complement as the result's.
        @test collect(opte_v.fees.lq.w) == collect(opt_v.fees.lq.w)
        @test collect(opte_v.fees.flq.w) == collect(opt_v.fees.flq.w)
        @test collect(opte_v.fees.tn.w) == collect(opt_v.fees.tn.w)
        # **The one thing the view cannot fix: the resolution order.**
        #
        # A `Fees` is already resolved, so the view is the whole story for it, and every
        # assertion above passes. A `FeesEstimator` is not. Today every family resolves it
        # *after* the door, against the reduced `sets`, and by then two things have gone
        # wrong for a name-keyed carrier: the departed asset's name is no longer in the
        # universe, so its rate is dropped with a warning under `strict = false`; and the
        # carrier's `w` already sits on the complement while `sets` sits on the mask, so the
        # two lengths cannot agree.
        #
        # The fix is not a new verb. It is to resolve `fees_constraints` on the **full**
        # `sets` *before* `investable_reduction`, and let this view split the resolved
        # `Fees`. That reorders the fourteen fit sites and changes what `strict` refuses, so
        # it is issue #911 rather than a change made here.
        #
        # **This assertion is a tripwire.** It pins today's behaviour, so it fails when #911
        # lands. Whoever fixes the order should delete it and keep the positive assertions
        # below, which already state the answer the fixed order must reach.
        reduced_sets = UniverseSets(; dict = Dict("nx" => nx[keep]))
        @test_throws DimensionMismatch fees_constraints(opte_v.fees, reduced_sets)

        # Resolving on the FULL sets first, then viewing, is the order that works, and it
        # reaches exactly the two axes the hand-written `Fees` reached.
        resolved_first = fees_constraints(fest, sets)
        split_after = PortfolioOptimisers.port_opt_view(resolved_first, keep, prn.X)
        @test collect(split_after.l) == lval[keep]
        @test collect(split_after.lq.w) == [w[k]]
        @test collect(split_after.lq.val) == [qval[k]]
        @test collect(split_after.flq.val) == [fval[k]]
        @test collect(split_after.lq.w) == collect(opt_v.fees.lq.w)
        @test collect(split_after.lq.val) == collect(opt_v.fees.lq.val)
    end

    @testset "Both axes are charged, and the row sums reproduce the series" begin
        atol = 1e-14
        # The invariant the two axes exist to preserve. Every fixture here sets **both**
        # carriers, because a fixture that left `flq` unset would pass whether or not the
        # one-off liquidation is charged at all — which is exactly how an unwired carrier
        # slipped through before.
        #
        # A liquidated asset earns no return, so its column of the charge matrix holds the
        # charge alone. The per period part lands on every observation and the one-off part
        # at the index the clock names, which is the same two steps the investable axis
        # takes. That is what keeps one clock across both axes.
        X3 = [0.010 -0.020 0.030
              -0.015 0.025 0.012
              0.020 0.010 -0.008
              -0.005 -0.030 0.018
              0.008 0.014 0.006]
        w3 = [0.4, -0.3, 0.9]
        mkf = fa -> Fees(; tn = Turnover(; w = fill(0.25, 3), val = [0.001, 0.002, 0.003]),
                         fl = [1.0, 0.0, 2.0], lq = Turnover(; w = [0.25], val = [0.010]),
                         flq = Turnover(; w = [0.25], val = [5.0]), fa = fa)
        tot = p -> sum(p[1]) + sum(p[2])

        for fa in (nothing, FirstObservationFees(), AmortisedFees())
            fees = mkf(fa)

            # The split sums to the scalar, on both halves of the clock. Before the
            # carriers were charged, the periodic gap here was the exit itself.
            @test isapprox(tot(PortfolioOptimisers.calc_asset_periodic_fees(w3, fees)),
                           PortfolioOptimisers.calc_periodic_fees(w3, fees); atol = atol)
            @test isapprox(tot(PortfolioOptimisers.calc_asset_one_off_fees(w3, fees)),
                           PortfolioOptimisers.calc_one_off_fees(w3, fees); atol = atol)

            # And the two matrices together reproduce the portfolio series, under every
            # clock — including the one that lands the fixed charge on one observation.
            A, C = calc_net_asset_returns(w3, X3, fees)
            @test size(A) == (5, 3)
            @test size(C) == (5, 1)
            @test isapprox(vec(sum(A; dims = 2)) .+ vec(sum(C; dims = 2)),
                           calc_net_returns(w3, X3, fees); atol = atol)
        end

        # The clock actually moves the charge, so the loop above is not vacuous: a spreading
        # clock puts the fixed exit on every observation, the default puts it on the first.
        Cfirst = calc_net_asset_returns(w3, X3, mkf(FirstObservationFees()))[2]
        Cspread = calc_net_asset_returns(w3, X3, mkf(AmortisedFees()))[2]
        @test Cfirst[1, 1] != Cfirst[2, 1]
        @test isapprox(Cspread[1, 1], Cspread[2, 1]; atol = atol)
        # Either way the whole fixed exit is paid exactly once over the series.
        @test isapprox(sum(Cfirst), sum(Cspread); atol = atol)
        # `5.0` fixed, plus `0.010 * 0.25` per period over five observations.
        @test isapprox(-sum(Cfirst), 5.0 + 5 * 0.010 * 0.25; atol = atol)

        # With no carrier the charge matrix is empty, so a caller destructures the same
        # shape whether or not an asset left.
        A0, C0 = calc_net_asset_returns(w3, X3,
                                        Fees(;
                                             tn = Turnover(; w = fill(0.25, 3),
                                                           val = [0.001, 0.002, 0.003])))
        @test size(C0) == (5, 0)
        @test isapprox(vec(sum(A0; dims = 2)),
                       calc_net_returns(w3, X3,
                                        Fees(;
                                             tn = Turnover(; w = fill(0.25, 3),
                                                           val = [0.001, 0.002, 0.003])));
                       atol = atol)
    end

    @testset "The amortisation override carries both carriers" begin
        # `override_fee_amortisation` rebuilds a `Fees` field by field, so it is the one
        # site that silently drops a field the type gains.
        w = [0.25, 0.25]
        fees = Fees(; l = 0.001, lq = Turnover(; w = w, val = 0.010),
                    flq = Turnover(; w = w, val = 5.0))
        o = PortfolioOptimisers.override_fee_amortisation(fees, AmortisedFees())
        @test o.lq === fees.lq
        @test o.flq === fees.flq
        @test isa(o.fa, AmortisedFees)
    end
end
