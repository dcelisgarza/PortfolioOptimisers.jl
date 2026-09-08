@testset "Reference fee parity" begin
    using PortfolioOptimisers, Test, LinearAlgebra

    # Every expected number in this file was measured by RUNNING the reference
    # implementation, not by reading it. The fixture below is stated as literals on both
    # sides, so the two languages parse identical `Float64` bits and no data file is
    # exchanged.
    #
    # The reference's cost model, which this file pins:
    #
    #   total_cost = sum(transaction_costs .* abs.(previous_weights - weights)) + liquidation
    #   total_fee  = sum(management_fees .* weights)
    #   returns    = X * weights .- total_cost .- total_fee
    #
    # The whole cost is subtracted from **every** observation. That is the same clock this
    # library puts `l`, `s` and `tn` on, so the two agree term for term:
    #
    #   | the reference       | this library                        |
    #   | ------------------- | ----------------------------------- |
    #   | `transaction_costs` | `tn`, a `Turnover` carrier          |
    #   | `management_fees`   | `l` on a long book, `s` on a short  |
    #   | `liquidation_cost`  | the `lq` carrier of ADR 0121        |
    #
    # The reference has no fixed fee, so `fl`, `fs` and the `flq` carrier of ADR 0121 have
    # no counterpart there and are pinned by the hand oracles of `test_11_fees_and_returns`.
    #
    # Two conventions differ deliberately, and neither is exercised here:
    #
    #  1. The reference charges its proportional fee as `management_fees .* weights` over
    #     every asset, so a short position earns a **credit**. This library splits the term
    #     into `l` over `w .>= 0` and `s` over `w .< 0`, and negates the short half so it is
    #     a positive charge. The two agree exactly on a long-only book, which is what the
    #     proportional testset uses.
    #  2. The reference stores `compounded` as an attribute of its portfolio object. Here it
    #     is the positional `compound` argument of `cumulative_returns` and `drawdowns`.
    #     Same switch, same default, different carrier.
    #
    # Tolerances. Every value below was measured **bit-for-bit equal** to the reference
    # except the three noted at their assertions, which are sums that cancel to within a few
    # ulp of zero and so carry an absolute error near `1e-16` with no meaningful relative
    # one. `atol` guards those, and guards the row sums against a BLAS that reassociates.
    #
    # **Every weight vector here is stated, never solved.** That is what buys the tolerance
    # above: the file compares fee arithmetic on identical inputs, so no solver enters it.
    # A test that first optimises cannot hold this tolerance, because the reference reaches
    # its weights through a different build of the solver. Extend this file that way only
    # with a tolerance sized to the weights, and note that `max_step_fraction = 0.75` brings
    # this library's solver settings close to the reference's defaults.
    atol = 1e-15

    X = [0.010 -0.020 0.005 0.030
         -0.015 0.025 -0.010 0.012
         0.020 0.010 0.015 -0.008
         -0.005 -0.030 0.020 0.018
         0.008 0.014 -0.025 0.006]
    # Asset names, in this order, are "a", "b", "c", "d" on the reference side.
    tc = [0.001, 0.002, 0.010, 0.003]
    mf = [0.0005, 0.0004, 0.0003, 0.0002]
    prev = fill(0.25, 4)

    @testset "The turnover charge on a long-short book" begin
        w = [0.4, -0.3, 0.5, 0.4]
        fees = Fees(; tn = Turnover(; w = prev, val = tc))
        # The reference reported `total_cost = 0.0042`.
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, fees), 0.0042; atol = atol)
        # `tn` is a per period rate, so the charge lands on every observation, which is what
        # the reference does with its `total_cost`.
        @test isapprox(PortfolioOptimisers.calc_one_off_fees(w, fees), 0.0; atol = atol)
        @test isapprox(calc_net_returns(w, X, fees),
                       [0.020300000000000002, -0.0179, 0.0050999999999999995, 0.02,
                        -0.015299999999999998]; atol = atol)
        # The per asset split sums to the series the scalar verb charges. It is one matrix
        # on the caller's universe, and this book has no forced exit, so every column of it
        # is an investable one.
        A = calc_net_asset_returns(w, X, fees)
        @test size(A) == size(X)
        @test isapprox(vec(sum(A; dims = 2)), calc_net_returns(w, X, fees); atol = atol)
    end

    @testset "The proportional charge on a long-only book" begin
        w = [0.4, 0.1, 0.2, 0.3]
        fees = Fees(; tn = Turnover(; w = prev, val = tc), l = mf)
        # The reference reported `total_cost = 0.0011` and `total_fee = 0.00036`.
        @test isapprox(calc_fees(w, Turnover(; w = prev, val = tc)), 0.0010999999999999998;
                       atol = atol)
        @test isapprox(calc_fees(w, mf, .>=), 0.00036; atol = atol)
        # The book is long, so `s` is unused and the two conventions coincide.
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, fees),
                       0.0010999999999999998 + 0.00036; atol = atol)
        # The last entry is the one that cancels: the reference reported
        # `-6.000000000000097e-5` where this library reports `-6.0000000000001025e-5`.
        @test isapprox(calc_net_returns(w, X, fees),
                       [0.01054, -0.0033599999999999997, 0.00814, 0.0029399999999999995,
                        -6.000000000000097e-5]; atol = atol)
    end

    @testset "The cumulative summaries under both settings of `compound`" begin
        w = [0.4, 0.1, 0.2, 0.3]
        fees = Fees(; tn = Turnover(; w = prev, val = tc), l = mf)
        r = calc_net_returns(w, X, fees)
        # `compound = false` is the reference's `compounded=False`, and `true` its `True`.
        # The compounded pair matched bit-for-bit; the simple pair differs in the last ulp
        # of its final entry, which is a sum that cancels.
        @test isapprox(cumulative_returns(r),
                       [0.01054, 0.007180000000000001, 0.01532, 0.01826,
                        0.018199999999999997]; atol = atol)
        @test cumulative_returns(r, true) ==
              [1.01054, 1.0071445855999999, 1.0153427425267838, 1.0183278501898125,
               1.0182667505188012]
        @test isapprox(drawdowns(r), [0.0, -0.00336, 0.0, 0.0, -6.0000000000001025e-5];
                       atol = atol)
        @test drawdowns(r, true) ==
              [0.0, -0.0033600000000001407, 0.0, 0.0, -5.999999999994898e-5]
    end

    @testset "A forced liquidation is charged through a reduced `Fees`" begin
        # ADR 0121 charges an asset that leaves the Investable Mask through a `Turnover`
        # carrier on the complement of the mask. This testset walks the path a caller
        # actually takes: a reduced `Fees` carrying `lq`, read by the ordinary fee verbs.
        # The reference meets the same charge the same way — it holds a named
        # `previous_weights` that includes an asset absent from `X`, and reports the cost
        # through the portfolio's `total_cost`.
        #
        # Asset "c" leaves a four-asset universe holding a quarter of the book in each.
        # `w` is the surviving book. The liquidation term does not read it, because the
        # exiting asset sits on the other axis, and that independence is itself contract.
        w = [0.4, -0.3, 0.9]

        # A per asset rate. The reference reported turnover `0.25` and cost `0.0025`.
        lq = Turnover(; w = [0.25], val = [0.010])
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, Fees(; lq = lq)), 0.0025;
                       atol = atol)
        @test isapprox(sum(abs, lq.w), 0.25; atol = atol)
        # `lq` is a rate per period, so it never lands in the one-off half.
        @test isapprox(PortfolioOptimisers.calc_one_off_fees(w, Fees(; lq = lq)), 0.0;
                       atol = atol)

        # One scalar rate over every exit. The reference reported cost `0.001`.
        sc = Fees(; lq = Turnover(; w = [0.25], val = 0.004))
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, sc), 0.001; atol = atol)

        # Two exits of opposite sign: "b" short at `-0.4` and "c" long at `0.25`. The
        # reference reported turnover `0.65` and cost `0.0033`, so the charge reads the
        # absolute previous weight and does not credit the short.
        lq2 = Turnover(; w = [-0.4, 0.25], val = [0.002, 0.010])
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, Fees(; lq = lq2)), 0.0033;
                       atol = atol)
        @test isapprox(sum(abs, lq2.w), 0.65; atol = atol)

        # A position already at zero is not traded, so it is charged nothing.
        z = Fees(; lq = Turnover(; w = [0.0, 0.25], val = [0.002, 0.010]))
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, z), 0.0025; atol = atol)

        # The arithmetic underneath is the established `Turnover` method priced at a zero
        # target, which is why the charge took no verb of its own.
        @test isapprox(calc_fees(zeros(2), lq2), 0.0033; atol = atol)
        @test isapprox(calc_asset_fees(zeros(2), lq2), [0.0008, 0.0025]; atol = atol)
        @test isapprox(sum(calc_asset_fees(zeros(2), lq2)), calc_fees(zeros(2), lq2);
                       atol = atol)
    end

    @testset "A reduced fit charges the exit on every observation" begin
        # The whole of ADR 0121's clock decision, walked end to end through the ordinary
        # verbs. A fit that reduced to the three investable assets still owes the exit, and
        # the charge rides on every observation beside the reduced turnover rather than one
        # time.
        #
        # This is the reference's own reduced portfolio: it was handed `X` for "a", "b" and
        # "d" with a named `previous_weights` still naming "c", and it reported
        # `total_cost = 0.0057` — the reduced turnover `0.0032` plus the exit `0.0025` —
        # subtracted from each of the five observations.
        Xred = X[:, [1, 2, 4]]
        wred = [0.4, -0.3, 0.9]
        fees = Fees(; tn = Turnover(; w = fill(0.25, 3), val = [0.001, 0.002, 0.003]),
                    lq = Turnover(; w = [0.25], val = [0.010]))

        @test isapprox(PortfolioOptimisers.calc_periodic_fees(wred, fees), 0.0057;
                       atol = atol)
        # The two halves of that total, so a regression names which one moved.
        @test isapprox(calc_fees(wred, fees.tn), 0.0032; atol = atol)
        @test isapprox(PortfolioOptimisers.calc_liquidation_fees(wred, fees.lq), 0.0025;
                       atol = atol)

        # The whole charge lands on every observation, which is the reference's series.
        @test isapprox(calc_net_returns(wred, Xred, fees),
                       [0.031299999999999994, -0.0084, -0.0079, 0.017499999999999998,
                        -0.001299999999999999]; atol = atol)

        # Charging the exit one time instead would leave four of the five observations
        # short by the whole charge, so the two clocks are distinguishable here.
        notn = Fees(; tn = Turnover(; w = fill(0.25, 3), val = [0.001, 0.002, 0.003]))
        @test !isapprox(calc_net_returns(wred, Xred, notn),
                        calc_net_returns(wred, Xred, fees); atol = atol)
        # And the gap between them is exactly the exit, on every observation.
        @test isapprox(calc_net_returns(wred, Xred, notn) .-
                       calc_net_returns(wred, Xred, fees), fill(0.0025, 5); atol = atol)
    end

    @testset "A solved walk-forward" begin
        using Clarabel

        # The testsets above compare arithmetic on stated weights, which is why they hold a
        # `1e-15` tolerance. This one solves, so it cannot: the reference reaches its weights
        # through a different build of the solver. `max_step_fraction = 0.75` brings this
        # library's Clarabel close to the one the reference drives through its modelling
        # layer, and the residual gap measured `8.3e-5` on a weight and `2.9e-5` on a summed
        # return series, so the tolerances below are sized to that and not to the arithmetic.
        w_atol = 1e-3
        s_atol = 2e-4

        # A deterministic panel from an integer LCG, so both languages hold identical bits
        # with no data file between them. Every element matched the reference exactly.
        T, N = 120, 5
        seed = 12345
        vals = Vector{Float64}(undef, T * N)
        for i in 1:(T * N)
            seed = mod(1103515245 * seed + 12345, 2^31)
            vals[i] = (seed / 2^31 - 0.5) * 0.04
        end
        Xcv = permutedims(reshape(vals, N, T))
        rdcv = ReturnsResult(; nx = ["a", "b", "c", "d", "e"], X = Xcv)
        slv = Solver(; name = :cl, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("verbose" => false, "max_step_fraction" => 0.75))

        tccv = [0.001, 0.002, 0.010, 0.003, 0.004]
        mgmt = [0.0005, 0.0004, 0.0003, 0.0002, 0.0006]
        z5 = zeros(5)

        # Three folds of a rolling sixty-observation window and a twenty-observation test,
        # which is the reference's `train_size = 60, test_size = 20`. `expand_train` is
        # `false` by default on both sides, so the window rolls rather than expands.
        #
        # **Reproducing the reference's drift needs both of this library's switches.** The
        # reference carries one bundled flag: with it set, the series drifts *and* the next
        # fold budgets its turnover against the drifted ending weights. Here those are two
        # independent switches, `wd` and `pws`, so a caller can drift the series while still
        # budgeting turnover against the targets. `wd` alone left the turnover cases
        # `3.5e-3` from the reference; `wd` with `pws = DriftedWeights()` brought every case
        # back inside `s_atol`.
        flat = () -> IndexWalkForward(60, 20)
        drift = () -> IndexWalkForward(60, 20; wd = SelfFinancingDrift(),
                                       pws = DriftedWeights())

        # Every proportional term this library carries, against its counterpart in the
        # reference: `l` is the reference's per asset holding fee, and `tn` its transaction
        # cost. `s` has no counterpart, because the reference credits a short holding fee
        # where this library charges it, so the books below are long only. The two
        # liquidation carriers of ADR 0121 are pinned by the two testsets above, which is as
        # far as they can be taken until issue #897 puts them on `Fees`.
        fee_cases = ["nofee" => nothing, "mgmt" => Fees(; l = mgmt),
                     "tn" => Fees(; tn = Turnover(; w = z5, val = tccv)),
                     "both" => Fees(; l = mgmt, tn = Turnover(; w = z5, val = tccv))]

        # Measured from the reference, per case: the summed net return series, then the last
        # cumulative return under `compound = false` and under `compound = true`.
        expected = Dict("nofee_flat" => (-0.007998610740968356, -0.007998610740968363,
                                         0.9912156849257472),
                        "nofee_drift" => (-0.004617355100166587, -0.004617355100166587,
                                          0.9945679659411881),
                        "mgmt_flat" => (-0.03298586309675567, -0.03298586309675567,
                                        0.9667460115983989),
                        "mgmt_drift" => (-0.029604607455953905, -0.029604607455953898,
                                         0.9700169059029946),
                        "tn_flat" => (-0.10175505628719284, -0.10175505628719285,
                                      0.9022776888821106),
                        "tn_drift" =>
                            (-0.1019142888775082, -0.10191428887750799, 0.9021280010875558),
                        "both_flat" => (-0.12674230864298017, -0.12674230864298017,
                                        0.8799688762453504),
                        "both_drift" => (-0.12690154123329553, -0.12690154123329575,
                                         0.8798228692359749))

        # The reference solved the same weights in all four fee cases, because its default
        # objective minimises risk and a proportional cost enters the return expression
        # alone. This library's default objective is the same, so it agrees.
        ref_w0 = [0.23892993656480035, 0.1412161727402494, 0.18064324954439945,
                  0.18706231534627446, 0.2521483258042763]
        ref_w1 = [0.19560053837766128, 0.1632418491720253, 0.19383371557461357,
                  0.1665127842983928, 0.28081111257730695]
        ref_w2 = [0.1441611559065762, 0.21862818311983637, 0.18954337007604785,
                  0.19378709425730659, 0.2538801966402329]

        for (fl, fe) in fee_cases, (dl, cvf) in ["flat" => flat, "drift" => drift]
            mr = MeanRisk(;
                          opt = JuMPOptimiser(; wb = WeightBounds(; lb = 0, ub = 1),
                                              bgt = 1, fees = fe, slv = slv))
            pred = cross_val_predict(mr, rdcv, cvf())
            r = pred.mrd.X
            ret_sum, smp_last, cmp_last = expected[fl * "_" * dl]

            @test length(pred.pred) == 3
            @test isapprox(pred.pred[1].res.w, ref_w0; atol = w_atol)
            @test isapprox(pred.pred[2].res.w, ref_w1; atol = w_atol)
            @test isapprox(pred.pred[3].res.w, ref_w2; atol = w_atol)

            @test isapprox(sum(r), ret_sum; atol = s_atol)
            # `compound = false` is the reference's `compounded=False`, `true` its `True`.
            @test isapprox(cumulative_returns(r)[end], smp_last; atol = s_atol)
            @test isapprox(cumulative_returns(r, true)[end], cmp_last; atol = s_atol)
        end
    end

    @testset "A walk-forward over a delisting, with a liquidation carrier" begin
        using StableRNGs, LinearAlgebra, Clarabel

        # The test that verifies the port end to end. An asset delists inside the last test
        # fold, so the mask derives itself from the data, the fold that loses it charges a
        # forced exit, and the folds that lose nothing charge none.
        #
        # **What can and cannot be compared.** The reference reaches a delisting only
        # through its exponentially weighted moments with `active_mask` routing, because its
        # plain prior refuses a `NaN`. This library's plain prior handles the gap natively.
        # The two therefore fit different moments and solve to different weights, so the
        # series cannot be compared. What can be compared exactly is the **charge**, which
        # is arithmetic on the weights: the first half below feeds this library's fee verbs
        # the reference's own per fold weights and matches its reported cost to rounding.
        # The second half then drives this library's whole pipeline and pins the invariants
        # the reference cannot speak to.

        T3, N3 = 120, 5
        seed3 = 12345
        v3 = Vector{Float64}(undef, T3 * N3)
        for i in 1:(T3 * N3)
            seed3 = mod(1103515245 * seed3 + 12345, 2^31)
            v3[i] = (seed3 / 2^31 - 0.5) * 0.04
        end
        X3 = permutedims(reshape(v3, N3, T3))
        k3 = 3
        X3[81:end, k3] .= NaN          # asset "c" delists inside the last test fold
        nx3 = ["a", "b", "c", "d", "e"]
        rd3 = ReturnsResult(; nx = nx3, X = X3)
        inv3 = [1, 2, 4, 5]
        tc3 = [0.001, 0.002, 0.010, 0.003, 0.004]
        mgmt3 = [0.0005, 0.0004, 0.0003, 0.0002, 0.0006]
        slv3 = Solver(; name = :cl, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false, "max_step_fraction" => 0.75))

        @testset "The parity matrix, over the delisting" begin
            # The matrix the port is verified by: every fee the reference supports, against
            # both weight-drift settings, summarised under both settings of `compound` —
            # all of it over a panel where an asset delists, so every cell charges a forced
            # exit at the fold that loses it.
            #
            # The comparison is driven from the **reference's own per fold weights**. That
            # is not a shortcut, it is the only way the cells are comparable: the reference
            # reaches a delisting solely through its exponentially weighted moments with
            # `active_mask` routing, because its plain prior refuses a `NaN`, while this
            # library's plain prior handles the gap natively. The two therefore fit
            # different moments and solve to different weights. Holding the weights fixed
            # removes that difference and leaves exactly what is being verified: the fee
            # arithmetic, the clock, the drift and the two cumulative conventions.
            #
            # The reference solved the same weights in all six cells, because its default
            # objective minimises risk and a proportional cost does not move that argmin.
            W = [[0.23372395184635317, 0.14070442963485008, 0.18641912894883597,
                  0.17987749947299542, 0.2592749900969653],
                 [0.17137301967228055, 0.1666189142652727, 0.1961303186945051,
                  0.1587711481740035, 0.30710659919393823],
                 [0.16730556399117805, 0.26717172952111706, 0.0, 0.2450269918497811,
                  0.32049571463792387]]
            # Budgeting against the targets threads the previous fold's target; threading
            # the drifted holdings threads what was actually held. The exit is priced
            # against whichever the scheme names, so the two columns differ.
            PW = Dict("flat" => [zeros(5), W[1], W[2]],
                      "drift" => [zeros(5),
                                  [0.2433787797334486, 0.13742235801900893, 0.18262568039164098,
                                   0.1665151048099047, 0.27005807704599694],
                                  [0.17016331410170402, 0.16136630292859214, 0.19165843402333646,
                                   0.13656907313225372, 0.34024287581411367]])
            # The last fold loses asset "c"; the first two keep every asset, because the
            # mask is derived from each fold's own training window.
            iv = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 4, 5]]
            rows = [61:80, 81:100, 101:120]
            @test iszero(W[3][k3])

            # Per fold `total_cost` and `total_fee` as the reference reported them.
            cost = Dict("tn_flat" => [0.003956056559411261, 0.0004659372891764765,
                                      0.002478800265941117],
                        "tn_drift" => [0.003956056559411261, 0.0004368712140818054,
                                       0.0025354153443862817], "mgmt_flat" => zeros(3),
                        "mgmt_drift" => zeros(3))
            cost["both_flat"] = cost["tn_flat"]
            cost["both_drift"] = cost["tn_drift"]
            mfee = [0.0004206099804145457, 0.0004271913603017645, 0.0004318243009567464]
            fee = Dict("tn_flat" => zeros(3), "tn_drift" => zeros(3), "mgmt_flat" => mfee,
                       "mgmt_drift" => mfee, "both_flat" => mfee, "both_drift" => mfee)
            # The last cumulative return of the whole path, simple and compounded.
            smp = Dict("tn_flat" => -0.11758386526314575,
                       "tn_drift" => -0.11615253012097801,
                       "mgmt_flat" => -0.005160495806029792,
                       "mgmt_drift" => -0.0031781805968521953,
                       "both_flat" => -0.1431763780966069,
                       "both_drift" => -0.14174504295443915)
            cmp = Dict("tn_flat" => 0.8881733007938063, "tn_drift" => 0.889433288044936,
                       "mgmt_flat" => 0.994108461599983, "mgmt_drift" => 0.9960704820919892,
                       "both_flat" => 0.8656826445068109,
                       "both_drift" => 0.8669112374385091)

            # The fee a fold charges: the five per asset fields on the assets it keeps, and
            # the liquidation carrier on the ones it lost, priced at the previous weights.
            fold_fee = function (f, pw, use_tn, use_mg)
                keep = iv[f]
                gone = setdiff(1:N3, keep)
                return Fees(; tn = if use_tn
                                Turnover(; w = pw[keep], val = tc3[keep])
                            else
                                nothing
                            end, l = use_mg ? mgmt3[keep] : nothing,
                            lq = if (use_tn && !isempty(gone))
                                Turnover(; w = pw[gone], val = tc3[gone])
                            else
                                nothing
                            end)
            end

            for (fl, use_tn, use_mg) in
                (("tn", true, false), ("mgmt", false, true), ("both", true, true)),
                (dl, wd) in (("flat", nothing), ("drift", SelfFinancingDrift()))

                tag = fl * "_" * dl
                pw = PW[dl]
                r = Float64[]
                for f in 1:3
                    keep = iv[f]
                    fe = fold_fee(f, pw[f], use_tn, use_mg)
                    wf = W[f][keep]

                    # The charge this library computes for the fold, against the two
                    # numbers the reference reported for it.
                    @test isapprox(PortfolioOptimisers.calc_periodic_fees(wf, fe),
                                   cost[tag][f] + fee[tag][f]; atol = atol)

                    # Only the fold that loses an asset owes an exit, and it owes the rate
                    # times the previous weight the scheme threaded.
                    if f == 3 && use_tn
                        @test isapprox(PortfolioOptimisers.calc_liquidation_fees(wf, fe.lq),
                                       tc3[k3] * pw[3][k3]; atol = atol)
                    else
                        @test isnothing(fe.lq)
                    end

                    # The fold's realised series. A fold may hold an asset over a window
                    # where it has no return — fold two holds "c" after it delists — and
                    # the fold zeroes that Held Gap once, which this reconstruction does
                    # through the same verb.
                    Xc = PortfolioOptimisers.filter_held_gaps(wf, X3[rows[f], keep], false)
                    append!(r, if isnothing(wd)
                                calc_net_returns(wf, Xc, fe)
                            else
                                calc_net_returns(wf, Xc, fe, wd)
                            end)
                end

                @test length(r) == 60
                @test all(isfinite, r)
                # Both cumulative conventions, against the reference's own summaries.
                @test isapprox(cumulative_returns(r)[end], smp[tag]; atol = 1e-14)
                @test isapprox(cumulative_returns(r, true)[end], cmp[tag]; atol = 1e-14)
            end
        end

        @testset "This library's pipeline, across drift and compound" begin
            w03 = fill(0.2, N3)
            cases = ["tn" => Fees(; tn = Turnover(; w = w03, val = tc3),
                                  lq = Turnover(; w = w03, val = tc3)),
                     "mgmt" => Fees(; l = mgmt3, lq = Turnover(; w = w03, val = tc3)),
                     "both" => Fees(; tn = Turnover(; w = w03, val = tc3), l = mgmt3,
                                    lq = Turnover(; w = w03, val = tc3))]
            schemes = ["flat" => IndexWalkForward(60, 20),
                       "drift" => IndexWalkForward(60, 20; wd = SelfFinancingDrift(),
                                                   pws = DriftedWeights())]

            for (_, fee) in cases, (sl, cv) in schemes
                mr3 = MeanRisk(;
                               opt = JuMPOptimiser(; wb = WeightBounds(; lb = 0, ub = 1),
                                                   bgt = 1, fees = fee, slv = slv3))
                pred = cross_val_predict(mr3, rd3, cv)
                @test length(pred.pred) == 3

                # The mask derives itself: the first two folds see every asset, the last
                # loses one.
                @test isnothing(pred.pred[1].res.imsk)
                @test isnothing(pred.pred[2].res.imsk)
                @test pred.pred[3].res.imsk == BitVector([1, 1, 0, 1, 1])

                # A fold that loses nothing carries no carrier and owes no exit. This is
                # the defect the end-to-end run found: the carrier is stated on the full
                # universe, and without the strip it was charged in full here.
                for i in 1:2
                    @test isnothing(pred.pred[i].res.fees.lq)
                    @test isapprox(PortfolioOptimisers.calc_liquidation_fees([0.0],
                                                                             pred.pred[i].res.fees.lq),
                                   0.0; atol = atol)
                end

                # The fold that loses the asset expands its weight to zero, and carries the
                # carrier on the complement, holding the previous fold's weight in it.
                exit_res = pred.pred[3].res
                @test length(exit_res.w) == N3
                @test iszero(exit_res.w[k3])
                @test length(exit_res.fees.lq.w) == 1
                held = only(exit_res.fees.lq.w)
                # The charge is the rate times the weight the fold actually threaded, which
                # is the reference's arithmetic on this library's own weights.
                @test isapprox(PortfolioOptimisers.calc_liquidation_fees([0.0],
                                                                         exit_res.fees.lq),
                               tc3[k3] * held; atol = atol)
                # **Which** weight that is, is the `pws` switch, and the exit obeys it like
                # every other turnover term. Budgeting against the targets charges the exit
                # at the previous fold's target; threading the drifted holdings charges it
                # at what was actually held when the asset left, which is a different
                # number.
                if sl == "flat"
                    @test isapprox(held, pred.pred[2].res.w[k3]; atol = atol)
                else
                    @test !isapprox(held, pred.pred[2].res.w[k3]; atol = 1e-6)
                    @test isapprox(held, pred.pred[2].res.w[k3]; atol = 5e-3)
                end

                # The series is finite under both schemes, and the two cumulative
                # conventions agree with their own definitions on it.
                r3 = pred.mrd.X
                @test all(isfinite, r3)
                @test isapprox(cumulative_returns(r3)[end], sum(r3); atol = 1e-12)
                @test isapprox(cumulative_returns(r3, true)[end],
                               prod(one(eltype(r3)) .+ r3); atol = 1e-12)
            end
        end
    end

    @testset "The drawdown peak includes the starting capital, and the reference's does not" begin
        # Found while pinning the walk-forward above, where the cost-heavy cases disagreed
        # on the maximum drawdown by `1.1e-2` while their return series agreed to `7e-6`.
        #
        # A drawdown is the decline from a **historical peak of the equity curve**, and the
        # capital the portfolio starts with is a point on that curve. So the running peak at
        # observation `t` is the maximum over `V₀, V₁, …, Vₜ`, and `V₀` is the starting
        # capital. This library computes exactly that: `relative_drawdown_arr` seeds its
        # running peak with `init = one(eltype(X))`, and its additive twin with a zero, so a
        # series that opens down is already in drawdown at its first observation.
        #
        # The reference seeds its peak at the **first observation** instead, so it reports
        # zero drawdown there whatever the first return, and a portfolio that loses money
        # from the first period and never recovers reports less than its true peak-to-trough
        # loss. **This library is the one that matches the definition**, which is what the
        # third assertion below pins: the by-definition computation, written out here with
        # the starting capital prepended to the curve, reproduces `drawdowns` exactly and
        # differs from the reference.
        #
        # The two agree whenever the series does rise above where it opened, because the
        # starting capital then stops being the peak. That is why the walk-forward testset
        # above asserts returns, which carry no such convention, and asserts no drawdown.

        # The first fold of the reference's own transaction-cost walk-forward. Its cost
        # drags the curve under water on the first observation and it never recovers.
        r = [-0.011788105221992843, -0.0051264818462376655, 0.005458948749114548,
             -0.00755922174545086, 0.0006201259574699089, -0.011913393832684459,
             -0.0059727004038226846, -0.01222042003663268, -0.0010357568565395724,
             -0.01310603346857926, -0.002531933330607626, 0.0009395572155512719,
             -0.0006490986796599128, 0.003105318463142509, -0.009275028041584244,
             -0.007481291914784366, -0.012652588370685482, -0.006639243004377236,
             -0.011326254458105164, 0.005205429707888839]
        cr = cumulative_returns(r)

        # The definition: the peak runs over the curve with the starting capital included.
        by_definition = minimum(cr .- accumulate(max, vcat(zero(eltype(cr)), cr))[2:end])
        @test isapprox(minimum(drawdowns(r)), by_definition; atol = atol)
        @test isapprox(by_definition, -0.10915360082646583; atol = atol)

        # The curve never recovers above its start, so the deepest drawdown is the deepest
        # cumulative loss itself.
        @test isapprox(minimum(drawdowns(r)), minimum(cr); atol = atol)

        # The reference's peak starts at the first observation, so it reports a shallower
        # drawdown: it never counts the opening loss.
        theirs = minimum(cr .- accumulate(max, cr))
        @test isapprox(theirs, -0.09769796250734987; atol = atol)
        @test theirs > minimum(drawdowns(r))
        # It understates by `1.1e-2` here. The shortfall is close to, but not exactly, the
        # opening decline of `-0.0118`, because the curve does edge above its first value at
        # the third observation before falling away, which lifts the reference's peak a
        # little off that first value.
        @test isapprox(minimum(drawdowns(r)) - theirs, -0.01145563831911596; atol = atol)

        # The two coincide once the series rises above where it opened.
        up = [0.01054, -0.00336, 0.00814]
        cup = cumulative_returns(up)
        @test isapprox(drawdowns(up), cup .- accumulate(max, cup); atol = atol)
    end
end
