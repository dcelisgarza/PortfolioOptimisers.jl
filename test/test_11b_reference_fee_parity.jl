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
        # The per asset split sums to the series the scalar verb charges.
        @test isapprox(vec(sum(calc_net_asset_returns(w, X, fees); dims = 2)),
                       calc_net_returns(w, X, fees); atol = atol)
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

    @testset "A forced liquidation is a trade to zero at the previous weight" begin
        # ADR 0121 charges an asset that leaves the Investable Mask through a `Turnover`
        # carrier on the complement of the mask. The charge needs no new verb: a forced exit
        # trades to zero, so the established `Turnover` method priced at a zero target is
        # the reference's arithmetic exactly. This testset pins that equivalence before
        # issue #897 wires the carrier onto `Fees`.
        #
        # Asset "c" leaves a four-asset universe holding a quarter of the book in each.

        # A per asset rate. The reference reported turnover `0.25` and cost `0.0025`.
        lq = Turnover(; w = [0.25], val = [0.010])
        @test isapprox(calc_fees(zeros(1), lq), 0.0025; atol = atol)
        @test isapprox(sum(abs, lq.w), 0.25; atol = atol)

        # One scalar rate over every exit. The reference reported cost `0.001`.
        @test isapprox(calc_fees(zeros(1), Turnover(; w = [0.25], val = 0.004)), 0.001;
                       atol = atol)

        # Two exits of opposite sign: "b" short at `-0.4` and "c" long at `0.25`. The
        # reference reported turnover `0.65` and cost `0.0033`, so the charge reads the
        # absolute previous weight and does not credit the short.
        lq2 = Turnover(; w = [-0.4, 0.25], val = [0.002, 0.010])
        @test isapprox(calc_fees(zeros(2), lq2), 0.0033; atol = atol)
        @test isapprox(sum(abs, lq2.w), 0.65; atol = atol)

        # The per asset split names the asset that caused each charge.
        @test isapprox(calc_asset_fees(zeros(2), lq2), [0.0008, 0.0025]; atol = atol)
        @test isapprox(sum(calc_asset_fees(zeros(2), lq2)), calc_fees(zeros(2), lq2);
                       atol = atol)

        # A position already at zero is not traded, so it is charged nothing.
        @test isapprox(calc_fees(zeros(2),
                                 Turnover(; w = [0.0, 0.25], val = [0.002, 0.010])), 0.0025;
                       atol = atol)
    end

    @testset "A reduced fit charges the exit on every observation" begin
        # The whole of ADR 0121's clock decision, measured against the reference: a fit that
        # reduced to the three investable assets still owes the exit, and the charge rides
        # on every observation beside the reduced turnover rather than one time.
        Xred = X[:, [1, 2, 4]]
        wred = [0.4, -0.3, 0.9]
        red_tn = calc_fees(wred, Turnover(; w = fill(0.25, 3), val = [0.001, 0.002, 0.003]))
        liq = calc_fees(zeros(1), Turnover(; w = [0.25], val = [0.010]))
        # The reference reported `total_cost = 0.0057` for the reduced portfolio, which is
        # the reduced turnover `0.0032` plus the exit `0.0025`.
        @test isapprox(red_tn, 0.0032; atol = atol)
        @test isapprox(liq, 0.0025; atol = atol)
        @test isapprox(red_tn + liq, 0.0057; atol = atol)
        @test isapprox(Xred * wred .- (red_tn + liq),
                       [0.031299999999999994, -0.0084, -0.0079, 0.017499999999999998,
                        -0.001299999999999999]; atol = atol)
        # Charging the exit one time instead would leave four of the five observations
        # short by the whole charge, so the two clocks are distinguishable here.
        @test !isapprox(Xred * wred .- red_tn,
                        [0.031299999999999994, -0.0084, -0.0079, 0.017499999999999998,
                         -0.001299999999999999]; atol = atol)
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
