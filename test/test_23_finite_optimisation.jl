@testset "Finite allocation" begin
    using PortfolioOptimisers, Clarabel, HiGHS, Test, CSV, TimeSeries, DataFrames,
          LinearAlgebra, StatsBase
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
    rd = prices_to_returns(X)
    slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                     settings = Dict("log_to_console" => false),
                     check_sol = (; allow_local = true, allow_almost = true))
    da = DiscreteAllocation(; slv = mip_slv)
    ga = GreedyAllocation(; unit = 0.3, kwargs = (sigdigits = 1,))
    mr = MeanRisk(; obj = MaximumRatio(; rf = 4.2 / 252 / 100),
                  opt = JuMPOptimiser(; sbgt = 1, bgt = 0.5,
                                      wb = WeightBounds(; lb = -1, ub = 1), slv = slv))
    res = optimise(mr, rd)

    res_da = optimise(da,
                      FiniteAllocationInput(; w = res.w, prices = vec(values(X[end])),
                                            cash = 4206.9))
    @test isapprox(sum(res_da.cost), 4206.9 * 0.5, rtol = 5e-3)
    @test isapprox(sum(res.w[res.w .< 0]), -1, rtol = 1e-4)
    @test isapprox(res_da.shares .* vec(values(X[end])), res_da.cost)
    @test isapprox(rmsd(res.w, res_da.w), 0.01186776139758978, rtol = 5e-4)

    res_ga = optimise(ga,
                      FiniteAllocationInput(; w = res.w, prices = vec(values(X[end])),
                                            cash = 4206.9))
    @test isapprox(sum(res_ga.cost), 4206.9 * 0.5, rtol = 4e-2)
    @test isapprox(sum(res.w[res.w .< 0]), -1, rtol = 1e-4)
    @test isapprox(res_ga.shares .* vec(values(X[end])), res_ga.cost)
    @test isapprox(rmsd(res.w, res_ga.w), 0.01048359738507303, rtol = 2e-3)
    @test all(isapprox.(mod.(round.(mod.(abs.(res_ga.shares), 1), sigdigits = 1), ga.unit),
                        0, atol = 1e-10))

    mr = MeanRisk(; obj = MaximumRatio(; rf = 4.2 / 252 / 100),
                  opt = JuMPOptimiser(; sbgt = 1, bgt = 1.2,
                                      wb = WeightBounds(; lb = -1, ub = 1), slv = slv))
    res = optimise(mr, rd)

    res_da = optimise(da,
                      FiniteAllocationInput(; w = res.w, prices = vec(values(X[end])),
                                            cash = 4206.9))
    # Each side's targets sum to its cash, so the book spends the budget down to a leftover
    # that buys no further long share.
    @test isapprox(sum(res_da.cost) + res_da.cash, 4206.9 * 1.2)
    @test 0 <= res_da.cash < minimum(vec(values(X[end]))[res.w .>= 0])
    @test isapprox(sum(res.w[res.w .< 0]), -1, rtol = 1e-3)
    @test isapprox(res_da.shares .* vec(values(X[end])), res_da.cost)
    @test isapprox(rmsd(res.w, res_da.w), 0.011295820717513184, rtol = 5e-4)

    res_ga = optimise(ga,
                      FiniteAllocationInput(; w = res.w, prices = vec(values(X[end])),
                                            cash = 4206.9))
    @test isapprox(sum(res_ga.cost), 4206.9 * 1.2, rtol = 1e-2)
    @test isapprox(sum(res.w[res.w .< 0]), -1, rtol = 1e-3)
    @test isapprox(res_ga.shares .* vec(values(X[end])), res_ga.cost)
    @test isapprox(rmsd(res.w, res_ga.w), 0.01640695936037548, rtol = 5e-5)

    mr = MeanRisk(; obj = MaximumRatio(; rf = 4.2 / 252 / 100),
                  opt = JuMPOptimiser(; bgt = 0.8, slv = slv))
    res = optimise(mr, rd)

    res_da = optimise(da,
                      FiniteAllocationInput(; w = res.w, prices = vec(values(X[end])),
                                            cash = 4206.9))

    @test isapprox(sum(res_da.cost) + res_da.cash, 4206.9 * 0.8)
    @test 0 <= res_da.cash < minimum(vec(values(X[end]))[res.w .> 0])
    @test isapprox(res_da.shares .* vec(values(X[end])), res_da.cost)
    @test isapprox(rmsd(res.w, res_da.w), 0.001331170818965931, rtol = 5e-2)
    res_ga = optimise(ga,
                      FiniteAllocationInput(; w = res.w, prices = vec(values(X[end])),
                                            cash = 4206.9))
    @test isapprox(sum(res_ga.cost), 4206.9 * 0.8, rtol = 1e-2)
    @test isapprox(res_ga.shares .* vec(values(X[end])), res_ga.cost)
    @test isapprox(rmsd(res.w, res_ga.w), 0.0010146002336457861, rtol = 5e-3)
    @test all(isapprox.(mod.(round.(mod.(abs.(res_ga.shares), 1), sigdigits = 1), ga.unit),
                        0, atol = 1e-10))

    # `DiscreteAllocation` defaults `fb` to a `GreedyAllocation`, so every allocation
    # above goes through the generic `optimise` and its fallback chain. Only `fb =
    # nothing` reaches the shortcut method in `02_DiscreteFiniteAllocation.jl`, which
    # calls `_optimise` directly. The shortcut must agree with the generic exactly: for a
    # fallback-less estimator the chain runs `_optimise` once and returns it unwrapped.
    da_nofb = DiscreteAllocation(; slv = mip_slv, fb = nothing)
    fai = FiniteAllocationInput(; w = res.w, prices = vec(values(X[end])), cash = 4206.9)
    @test which(optimise, Tuple{typeof(da_nofb), typeof(fai)}) !=
          which(optimise, Tuple{typeof(da), typeof(fai)})
    res_shortcut = optimise(da_nofb, fai)
    @test isa(res_shortcut, DiscreteAllocationResult)
    @test isnothing(res_shortcut.fb)
    @test isapprox(res_shortcut.shares .* vec(values(X[end])), res_shortcut.cost)

    # The greedy second pass buys `unit` shares at a time, so the affordability test must be
    # on `p[i] * unit`, not on `p[i]`. Testing one share alone overdrew the budget for every
    # `unit > 1` -- at `unit = 10` below it spent 15195.0 of a 10000.0 budget -- and refused
    # affordable buys for every `unit < 1`.
    w_ov = [0.35, 0.25, 0.20, 0.12, 0.08]
    p_ov = [131.7, 47.3, 903.5, 12.9, 268.4]
    cash_ov = 10_000.0
    for u in (0.3, 1, 2, 5, 10)
        r_ov = optimise(GreedyAllocation(; unit = u),
                        FiniteAllocationInput(; w = w_ov, prices = p_ov, cash = cash_ov))
        spent = dot(collect(r_ov.shares), p_ov)
        @test spent <= cash_ov + eps(cash_ov)
        @test r_ov.cash >= 0
        @test isapprox(spent + r_ov.cash, cash_ov)
    end
    # `unit = 1` is the default and is unchanged by the fix: there, one share is one purchase.
    r_unit1 = optimise(GreedyAllocation(),
                       FiniteAllocationInput(; w = w_ov, prices = p_ov, cash = cash_ov))
    @test collect(r_unit1.shares) == [26.0, 52.0, 2.0, 93.0, 3.0]
    @test isapprox(r_unit1.cash, 304.30000000000075)

    # `roundmult` truncates towards zero to a multiple of `prec` and then rounds that
    # product. It is not a round to the nearest multiple: 8.0 is the nearest multiple of 2.
    @test PortfolioOptimisers.roundmult(7.5, 2) == 6.0
    @test PortfolioOptimisers.roundmult(26.58, 1) == 26.0
    @test PortfolioOptimisers.roundmult(7.5, 2) != round(7.5 / 2) * 2
end
# Issue #900: a fee is a cost of the portfolio the allocator actually buys. It is priced
# on `x .* p`, the money in each position, and it is charged inside the allocation rather
# than deducted from the cash beforehand.
@testset "Finite allocation fees" begin
    using PortfolioOptimisers, HiGHS, Test, LinearAlgebra
    PO = PortfolioOptimisers
    mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                     settings = Dict("log_to_console" => false),
                     check_sol = (; allow_local = true, allow_almost = true))
    da = DiscreteAllocation(; slv = mip_slv, fb = nothing)
    ga = GreedyAllocation()

    # `prev_cash` defaults to `cash`, and it must be non-negative.
    fai = FiniteAllocationInput(; w = [0.6, 0.4], prices = [10.0, 20.0], cash = 1000.0)
    @test fai.prev_cash == fai.cash == 1000.0
    @test FiniteAllocationInput(; w = [0.6, 0.4], prices = [10.0, 20.0], cash = 1000.0,
                                prev_cash = 0.0).prev_cash == 0.0
    @test_throws DomainError FiniteAllocationInput(; w = [0.6, 0.4], prices = [10.0, 20.0],
                                                   cash = 1000.0, prev_cash = -1.0)

    # The charge is money, and it is the by-hand table of #898. On `w = [0.5, 0.5]`,
    # `p = [100.0, 200.0]`, `cash = 1e6`, `T = 252`, `l = 0.01` and `tn.val = 0.002`
    # against `w0 = [0, 0]`, a fully invested book owes `252 * 10_000` proportional,
    # `252 * 2_000` turnover and `2 * 5.0` fixed. The deleted price-carrying family
    # charged `rate * dot(w, p)`, which is `1.5` and `0.3` per period.
    fmoney = Fees(; l = 0.01, fl = 5.0, tn = Turnover(; w = [0.0, 0.0], val = 0.002))
    lsf, _ = PO.allocation_side_fees(fmoney, 252, 1e6, [true, true], Float64[])
    @test PO.allocation_fee(lsf, [100.0, 200.0], [5000.0, 2500.0]) ==
          252 * 0.01 * 1e6 + 252 * 0.002 * 1e6 + 10.0
    # A side that states no fee is charged nothing, whatever it holds.
    @test iszero(PO.allocation_fee(nothing, [100.0, 200.0], [5000.0, 2500.0]))

    # The delta of one purchase agrees with the difference of two whole charges.
    sh0 = [3.0, 7.0]
    sh1 = [4.0, 7.0]
    @test isapprox(PO.greedy_fee_delta(lsf, [100.0, 200.0], sh0, 1, 1.0),
                   PO.allocation_fee(lsf, [100.0, 200.0], sh1) -
                   PO.allocation_fee(lsf, [100.0, 200.0], sh0))
    # A position that is opened by the purchase pays the fixed fee one time.
    @test PO.greedy_fee_delta(lsf, [100.0, 200.0], [0.0, 7.0], 1, 1.0) -
          PO.greedy_fee_delta(lsf, [100.0, 200.0], [3.0, 7.0], 1, 1.0) == 5.0

    # A long-only allocation pays the charge of the shares it bought, and the cash it
    # reports is what is left after both the shares and the fee.
    w = [0.6, 0.4]
    p = [10.0, 20.0]
    cash = 1000.0
    T = 12
    fee = Fees(; l = 0.001, fl = 2.0)
    for alloc in (da, ga)
        r = optimise(alloc,
                     FiniteAllocationInput(; w = w, prices = p, cash = cash, horizon = T,
                                           fees = fee))
        shares = collect(r.shares)
        cost = collect(r.cost)
        @test isapprox(r.fees, T * 0.001 * sum(cost) + 2.0 * count(!iszero, shares))
        @test isapprox(sum(cost) + r.cash + r.fees, cash)
        @test r.cash >= 0
        # The same book, priced by the shared verb, is the number the result reports.
        lsf2, _ = PO.allocation_side_fees(fee, T, cash, [true, true], Float64[])
        @test isapprox(r.fees, PO.allocation_fee(lsf2, p, shares))
    end

    # A fee competes with a position, so a book that pays one buys no more than a book
    # that pays none, and the free book spends the whole budget.
    for alloc in (da, ga)
        r_free = optimise(alloc, FiniteAllocationInput(; w = w, prices = p, cash = cash))
        r_fee = optimise(alloc,
                         FiniteAllocationInput(; w = w, prices = p, cash = cash,
                                               horizon = T, fees = fee))
        @test iszero(r_free.fees)
        @test isapprox(sum(collect(r_free.cost)) + r_free.cash, cash)
        @test sum(collect(r_fee.cost)) <= sum(collect(r_free.cost))
    end

    # A fixed fee prices a small position out. The third asset is worth about 30 of the
    # 1000, and a fixed fee of 40 buys too little tracking to be worth holding.
    w3 = [0.5, 0.47, 0.03]
    p3 = [10.0, 20.0, 30.0]
    r_small = optimise(da,
                       FiniteAllocationInput(; w = w3, prices = p3, cash = 1000.0,
                                             horizon = 1, fees = Fees(; fl = 40.0)))
    @test iszero(collect(r_small.shares)[3])
    r_nofee = optimise(da, FiniteAllocationInput(; w = w3, prices = p3, cash = 1000.0))
    @test !iszero(collect(r_nofee.shares)[3])

    # `prev_cash` states the money held before the trade, so it moves the turnover fee.
    # A book that already held the target owes less than one that starts from nothing.
    ftn = Fees(; tn = Turnover(; w = [0.6, 0.4], val = 0.002))
    for alloc in (da, ga)
        r_held = optimise(alloc,
                          FiniteAllocationInput(; w = w, prices = p, cash = cash,
                                                prev_cash = cash, horizon = T, fees = ftn))
        r_fresh = optimise(alloc,
                           FiniteAllocationInput(; w = w, prices = p, cash = cash,
                                                 prev_cash = 0.0, horizon = T, fees = ftn))
        @test r_held.fees < r_fresh.fees
        # Selling out is a trade, so a book that held money and buys nothing still owes.
        # No share of a price of 1e6 is affordable out of 100, and the exit of the 1000
        # held before the trade costs `1 * 0.002 * 1000`, which 100 pays.
        r_none = optimise(alloc,
                          FiniteAllocationInput(; w = w, prices = [1e6, 1e6], cash = 100.0,
                                                prev_cash = cash, horizon = 1, fees = ftn))
        @test all(iszero, collect(r_none.shares))
        @test isapprox(r_none.fees, 0.002 * cash)
        @test isapprox(r_none.cash, 100.0 - 0.002 * cash)
    end

    # A fee larger than the cash makes the budget of the MIP infeasible. The model then
    # holds no finite value, so the book is read as empty and the return code carries the
    # failure rather than the read-back raising on the conversion to `Int`.
    r_broke = optimise(da,
                       FiniteAllocationInput(; w = w, prices = [1e6, 1e6], cash = 1.0,
                                             prev_cash = cash, horizon = T, fees = ftn))
    @test isa(r_broke.retcode, PortfolioOptimisers.OptimisationFailure)
    @test all(iszero, collect(r_broke.shares))
    @test isapprox(r_broke.fees, T * 0.002 * cash)
    # The greedy allocator has no budget constraint to break, so it answers the same book
    # and reports the debt as a negative leftover.
    r_broke_g = optimise(ga,
                         FiniteAllocationInput(; w = w, prices = [1e6, 1e6], cash = 1.0,
                                               prev_cash = cash, horizon = T, fees = ftn))
    @test all(iszero, collect(r_broke_g.shares))
    @test isapprox(r_broke_g.fees, T * 0.002 * cash)
    @test r_broke_g.cash < 0

    # A long-short book charges each side its own rates, and the reported fee is never
    # signed. `l` and `fl` reach the long side, `s` and `fs` the short one.
    wls = [0.7, -0.3]
    fls = Fees(; l = 0.001, s = 0.004, fl = 1.0, fs = 3.0)
    for alloc in (da, ga)
        rls = optimise(alloc,
                       FiniteAllocationInput(; w = wls, prices = p, cash = cash,
                                             horizon = T, fees = fls))
        money = collect(rls.shares) .* p
        @test rls.fees > 0
        @test isapprox(rls.fees,
                       T * (0.001 * money[1] - 0.004 * money[2]) +
                       1.0 * !iszero(money[1]) +
                       3.0 * !iszero(money[2]))
    end
end
# Issue #914: an optimisation that reduced to its Investable Mask hands the allocator a
# **two-axis** `Fees` beside a **full-length** `w`. The input carries the mask, the fee is
# lifted back onto the axis the weights live on, and ADR 0121's two liquidation carriers
# are charged inside the allocation's own model, on the money the forced exit sold.
@testset "Finite allocation over a reduced universe" begin
    using PortfolioOptimisers, HiGHS, Test, LinearAlgebra
    PO = PortfolioOptimisers
    mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                     settings = Dict("log_to_console" => false),
                     check_sol = (; allow_local = true, allow_almost = true))
    da = DiscreteAllocation(; slv = mip_slv, fb = nothing)
    ga = GreedyAllocation()

    @testset "`lift_fees` inverts the two-axis view" begin
        # Four assets, the second of which leaves the universe.
        Xf = ones(3, 4)
        imsk = BitVector([true, false, true, true])
        i = findall(imsk)
        full = Fees(;
                    tn = Turnover(; w = [0.1, 0.2, 0.3, 0.4],
                                  val = [0.001, 0.002, 0.003, 0.004]),
                    l = [0.01, 0.02, 0.03, 0.04], s = [0.05, 0.06, 0.07, 0.08],
                    fl = [1.0, 2.0, 3.0, 4.0], fs = [5.0, 6.0, 7.0, 8.0],
                    lq = Turnover(; w = [0.1, 0.2, 0.3, 0.4],
                                  val = [0.011, 0.012, 0.013, 0.014]),
                    flq = Turnover(; w = [0.1, 0.2, 0.3, 0.4],
                                   val = [9.0, 10.0, 11.0, 12.0]))
        red = PO.investable_fees_view(full, imsk, Xf)
        # The door reduces the five holding fields to the mask and the two carriers to its
        # complement, which is the pair of lengths #914 reported at the allocator's door,
        # and marks the fee with the mask it lifts at (#1067).
        @test length(red.l) == 3
        @test length(red.lq.w) == 1
        @test red.imsk == imsk
        lift = PO.lift_fees(red)
        @test isnothing(lift.imsk)
        # Every per-asset field comes back at the full width, zero where it says nothing.
        @test lift.l == [0.01, 0.0, 0.03, 0.04]
        @test lift.s == [0.05, 0.0, 0.07, 0.08]
        @test lift.fl == [1.0, 0.0, 3.0, 4.0]
        @test lift.fs == [5.0, 0.0, 7.0, 8.0]
        @test lift.tn.w == [0.1, 0.0, 0.3, 0.4]
        @test lift.tn.val == [0.001, 0.0, 0.003, 0.004]
        # The carriers lift at the complement, so they are non-zero exactly where the
        # holding fields are zero.
        @test lift.lq.w == [0.0, 0.2, 0.0, 0.0]
        @test lift.lq.val == [0.0, 0.012, 0.0, 0.0]
        @test lift.flq.w == [0.0, 0.2, 0.0, 0.0]
        @test lift.flq.val == [0.0, 10.0, 0.0, 0.0]
        # A lift charges what the reduced fee charged: the entries it added are zero.
        @test isapprox(PO.calc_liquidation_fees(red.lq), PO.calc_liquidation_fees(lift.lq))
        @test isapprox(PO.calc_fixed_liquidation_fees(red.flq, red.kwargs),
                       PO.calc_fixed_liquidation_fees(lift.flq, lift.kwargs))
        # A fee that states neither carrier owes no forced exit, and neither does a
        # `nothing` fee: the two methods answer a zero in the type the horizon and the
        # cash promote to.
        @test iszero(PO.allocation_liquidation_fee(nothing, 2, 1e4))
        @test iszero(PO.allocation_liquidation_fee(Fees(; l = 0.01), 2, 1e4))
        # An unmarked fee is on the full universe already, so it lifts nothing, and so
        # does a `nothing` fee. A scalar rate carries through a lift.
        @test PO.lift_fees(full) === full
        @test isnothing(PO.lift_fees(nothing))
        @test PO.lift_fees(Fees(; l = 0.01, imsk = imsk)).l == 0.01
        @test isnothing(PO.lift_fees(Fees(; l = 0.01, imsk = imsk)).lq)
    end

    # The reproduction of #914, in shape. `w` is on the full universe with a zero at the
    # asset that left, `l` is on the three investable assets, and the two carriers are on
    # the one that exited.
    w = [0.5, 0.0, 0.3, 0.2]
    prices = [10.0, 20.0, 30.0, 40.0]
    imsk = BitVector([true, false, true, true])
    cash = 1e4
    T = 2
    fees = Fees(; l = [0.001, 0.002, 0.003], lq = Turnover(; w = [0.25], val = [0.01]),
                flq = Turnover(; w = [0.25], val = [5.0]))
    # `prev_cash` defaults to `cash`, so the exit sold `1e4 * 0.25`. `lq` is a rate over
    # `T` periods and `flq` is a currency amount charged one time.
    liq = T * 0.01 * cash * 0.25 + 5.0
    lfull = [0.001, 0.0, 0.002, 0.003]

    @testset "the mask lets a reduced fee meet full-length weights" begin
        fai = FiniteAllocationInput(; w = w, prices = prices, cash = cash, horizon = T,
                                    fees = fees, imsk = imsk)
        @test fai.imsk === imsk
        # Without the mask the fee stays on the shorter axis, and the full-length side
        # selector runs off its end. That is the `BoundsError` of #914.
        @test_throws BoundsError optimise(ga,
                                          FiniteAllocationInput(; w = w, prices = prices,
                                                                cash = cash, horizon = T,
                                                                fees = fees))
        for alloc in (da, ga)
            r = optimise(alloc, fai)
            cost = collect(r.cost)
            # The whole charge: the proportional rate on the money bought, plus the
            # constant forced exit.
            @test isapprox(r.fees, T * LinearAlgebra.dot(lfull, cost) + liq)
            # The money adds up: the shares, the cash left and the fee are the budget.
            @test isapprox(sum(cost) + r.cash + r.fees, cash)
            @test r.cash >= 0
        end
        # The greedy walk never buys an asset whose target weight is zero, so the asset
        # that left the universe holds nothing.
        @test iszero(collect(optimise(ga, fai).cost)[2])
    end

    @testset "the forced exit is charged whatever the book buys" begin
        # The exit is a constant, so a book with no cash to spend still owes it, and a
        # book that pays no other fee owes it alone.
        fai = FiniteAllocationInput(; w = w, prices = prices, cash = cash, horizon = T,
                                    fees = Fees(; lq = Turnover(; w = [0.25], val = [0.01]),
                                                flq = Turnover(; w = [0.25], val = [5.0])),
                                    imsk = imsk)
        for alloc in (da, ga)
            r = optimise(alloc, fai)
            @test isapprox(r.fees, liq)
            @test isapprox(sum(collect(r.cost)) + r.cash + r.fees, cash)
        end
        # A short-only book has an empty long side, and the exit still reaches the report.
        # `lq` is stated by hand here, so no mask puts the zero-weight assets on that side.
        fais = FiniteAllocationInput(; w = [-0.6, -0.4], prices = [10.0, 20.0], cash = cash,
                                     horizon = T,
                                     fees = Fees(;
                                                 lq = Turnover(; w = [0.25], val = [0.01])))
        rs = optimise(ga, fais)
        @test isapprox(rs.fees, T * 0.01 * cash * 0.25)
    end

    @testset "the input reads what a fitted optimisation carries" begin
        # `EqualWeighted` fits no prior estimator, so its result carries the window it saw
        # rather than a prior result, and it carries no fee at all.
        rd = ReturnsResult(; nx = ["a", "b", "c", "d"],
                           X = [0.01 0.02 0.03 0.04
                                -0.01 0.00 0.02 -0.02
                                0.02 0.01 -0.01 0.03])
        res = optimise(EqualWeighted(), rd)
        fai = FiniteAllocationInput(res; prices = prices, cash = cash)
        @test fai.w == res.w
        @test isnothing(fai.fees)
        @test fai.imsk == PO.result_investable_mask(res)
        # The horizon falls back to the observation count of what the result carries.
        @test fai.horizon == size(rd.X, 1) == 3
        # Every stated keyword wins over the result.
        @test FiniteAllocationInput(res; prices = prices, cash = cash, horizon = 7).horizon ==
              7
        @test FiniteAllocationInput(res; prices = prices, cash = cash, w = w).w == w
        faif = FiniteAllocationInput(res; prices = prices, cash = cash, horizon = T,
                                     fees = fees, imsk = imsk)
        # A caller's reduced fee is theirs, marked with the mask they state (#1067).
        @test faif.fees.l === fees.l && faif.fees.lq === fees.lq
        @test faif.fees.imsk == imsk == faif.imsk
        # A marked fee supplies the mask the caller leaves out, and a marked fee beside a
        # different mask is refused, so the fee is the one source of the axes it is on.
        marked = faif.fees
        faim = FiniteAllocationInput(; w = faif.w, prices = prices, cash = cash,
                                     horizon = T, fees = marked)
        @test faim.fees === marked && faim.imsk == imsk
        @test_throws ArgumentError FiniteAllocationInput(; w = w, prices = prices,
                                                         cash = cash, horizon = T,
                                                         fees = marked,
                                                         imsk = BitVector([1, 1, 0, 1]))
        @test PO.mark_fees(nothing, imsk) == (nothing, imsk)
        @test PO.mark_fees(fees, nothing) == (fees, nothing)
        @test PO.mark_fees(marked, imsk) == (marked, imsk)
        # A marked fee allocates as the hand-built input does.
        @test optimise(ga, faim).fees == optimise(ga, faif).fees
        # A result carrying no prior at all derives no horizon, and the input then takes
        # the caller's — and only a fee needs one.
        res0 = PO.NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                          retcode = PO.OptimisationSuccess(),
                                          w = [0.5, 0.5], fb = nothing)
        @test isnothing(PO.allocation_horizon(res0))
        fai0 = FiniteAllocationInput(res0; prices = [10.0, 20.0], cash = 1000.0)
        @test fai0.w == [0.5, 0.5]
        @test isnothing(fai0.horizon)
        @test isnothing(fai0.fees)
        @test isnothing(fai0.imsk)
    end

    @testset "the input refuses a mask that cannot be its own" begin
        @test_throws DimensionMismatch FiniteAllocationInput(; w = w, prices = prices,
                                                             imsk = BitVector([true, true]))
        @test_throws PortfolioOptimisers.IsEmptyError FiniteAllocationInput(; w = w,
                                                                            prices = prices,
                                                                            imsk = falses(4))
    end
end
@testset "A finite allocation records the fallback chain that answered it (#1024)" begin
    using PortfolioOptimisers, Test
    PO = PortfolioOptimisers
    fai = FiniteAllocationInput(; w = [0.6, 0.4], prices = [10.0, 20.0], cash = 1000.0)
    # A solver-less discrete allocation fails, and the greedy default answers: the answer
    # carries the `(estimator, result)` pair of the failure.
    da = DiscreteAllocation(; slv = Solver(; name = :none, solver = nothing))
    res = optimise(da, fai)
    @test isa(res, GreedyAllocationResult)
    @test isa(res.retcode, OptimisationSuccess)
    @test isa(res.fb, PO.FbChain)
    @test length(res.fb) == 1
    @test res.fb[1][1] === da
    @test isa(res.fb[1][2], DiscreteAllocationResult)
    @test isa(res.fb[1][2].retcode, OptimisationFailure)
    @test isnothing(res.fb[1][2].fb)
    # A finite result's `fb` admits a finite fallback estimator, which is what the finite
    # rebuild takes; it was bound to the continuous alias before.
    @test PO.factory(res, GreedyAllocation()).fb == GreedyAllocation()
    @test isnothing(PO.factory(res, nothing).fb)
end
@testset "The docstrings of 03_GreedyFiniteAllocation.jl against numbers" begin
    using PortfolioOptimisers, Test, StableRNGs
    PO = PortfolioOptimisers

    # The two passes as the `# Mathematical definition` of `GreedyAllocation` states them,
    # on one long side. `F` is `allocation_fee`, and each `ΔF` is a difference of two
    # whole fees, so this checks `greedy_fee_delta` as well.
    function documented_greedy(w, p, C, unit, sf; kwargs = (;))
        N = length(w)
        o = sortperm(w; rev = true)
        w = w[o] / sum(w[o])
        p = p[o]
        sfo = PO.permute_side_fees(sf, o)
        F = x -> PO.allocation_fee(sfo, p, x)
        x = zeros(N)
        r = C - F(x)
        for i in 1:N
            y = copy(x)
            y[i] = round(floor(w[i] * C / (p[i] * unit)) * unit; kwargs...)
            c = y[i] * p[i] + F(y) - F(x)
            if c > r
                break
            end
            r -= c
            x = y
        end
        while r > 0
            held = sum(x .* p)
            d = w - (iszero(held) ? zeros(N) : x .* p / held)
            best = 0
            for i in 1:N
                y = copy(x)
                y[i] += unit
                if p[i] * unit + F(y) - F(x) <= r &&
                   d[i] > 0 &&
                   (best == 0 || d[i] > d[best])
                    best = i
                end
            end
            if best == 0
                break
            end
            y = copy(x)
            y[best] += unit
            r -= p[best] * unit + F(y) - F(x)
            x = y
        end
        xo = similar(x)
        xo[o] = x
        return xo, C - sum(x .* p) - F(x)
    end
    rng = StableRNG(907)
    for _ in 1:200
        N = rand(rng, 2:7)
        w = rand(rng, N) .^ 2
        if rand(rng) < 0.3
            w[rand(rng, 1:N)] = 0.0
        end
        w ./= sum(w)
        p = round.(1 .+ 300 * rand(rng, N); digits = 2)
        C = round(100 + 20_000 * rand(rng); digits = 2)
        unit = rand(rng, (1, 2, 5, 0.5))
        kw = unit == 0.5 ? (digits = 1,) : (;)
        T = rand(rng, (1, 3))
        w0 = rand(rng, N)
        w0 ./= sum(w0)
        fee = rand(rng,
                   (nothing, Fees(; l = 0.001),
                    Fees(; l = rand(rng, N) / 500, fl = 2.0,
                         tn = Turnover(; w = w0, val = rand(rng, N) / 200))))
        r = optimise(GreedyAllocation(; unit = unit, kwargs = kw),
                     FiniteAllocationInput(; w = w, prices = p, cash = C, horizon = T,
                                           fees = fee))
        lsf, _ = PO.allocation_side_fees(fee, T, C, trues(N), Float64[])
        x, cash = documented_greedy(w, p, C, unit, lsf; kwargs = kw)
        @test collect(r.shares) ≈ x
        @test isapprox(r.cash, cash; atol = 1e-8)
    end

    # When the first pass buys nothing, the realised weight is zero rather than 0 / 0, so
    # the deficit is the target and an asset with a zero target is never bought. The old
    # code bought one share of the second asset here, from a deficit of NaN.
    r0 = optimise(GreedyAllocation(),
                  FiniteAllocationInput(; w = [1.0, 0.0], prices = [200.0, 1.0],
                                        cash = 150.0))
    @test collect(r0.shares) == [0.0, 0.0]
    @test r0.cash == 150.0
    # A side whose target weights are all zero buys nothing. The long side of this short
    # book holds one zero weight, and the old code renormalised it to NaN.
    rz = optimise(GreedyAllocation(),
                  FiniteAllocationInput(; w = [-1.0, 0.0], prices = [10.0, 20.0],
                                        cash = 100.0))
    @test collect(rz.shares) == [-10.0, 0.0]
    @test collect(rz.w) == [-1.0, 0.0]
    @test rz.cash == 0.0

    # `finite_sub_allocation!` changes none of its arguments.
    ga = GreedyAllocation()
    wv = [0.3, 0.5]
    PO.finite_sub_allocation!(view(wv, 1:2), [10.0, 20.0], 1000.0, 1.0, nothing, ga)
    @test wv == [0.3, 0.5]
    # An empty side still pays the forced exit, out of its cash.
    sf = (T = 1, prop = nothing, fixed = nothing, tn_val = nothing, prev_money = nothing,
          liq = 7.0)
    re = PO.finite_sub_allocation!(Float64[], Float64[], 100.0, 0.0, sf, ga)
    @test isempty(re[1]) && isempty(re[2]) && isempty(re[3])
    @test re[4] == 93.0
    @test re[5] == 7.0

    # `roundmult` rounds to an integer by default, so a `prec` below one can lose the
    # multiple. `RoundDown` loses it too, and `digits` keeps it.
    @test PO.roundmult(1.25, 0.3) == 1.0
    @test PO.roundmult(1.25, 0.3, RoundDown) == 1.0
    @test PO.roundmult(1.25, 0.3; digits = 1) == 1.2

    # The turnover term of `greedy_fee_delta` is negative when the purchase moves the
    # position towards the money it held before the trade, and the delta is still the
    # difference of two whole fees.
    f = Fees(; l = [0.01, 0.02], fl = [5.0, 3.0],
             tn = Turnover(; w = [0.4, 0.6], val = [0.002, 0.004]))
    lsf, _ = PO.allocation_side_fees(f, 12, 1e4, [true, true], Float64[])
    p = [100.0, 200.0]
    sh = [10.0, 0.0]
    @test isapprox(PO.greedy_fee_delta(lsf, p, sh, 1, 2.0),
                   12 * 0.01 * 200 + 12 * 0.002 * (abs(1200 - 4000) - abs(1000 - 4000)))
    @test isapprox(PO.greedy_fee_delta(lsf, p, sh, 1, 2.0),
                   PO.allocation_fee(lsf, p, sh + [2.0, 0.0]) -
                   PO.allocation_fee(lsf, p, sh))
    @test isapprox(PO.greedy_fee_delta(lsf, p, sh, 2, 3.0),
                   PO.allocation_fee(lsf, p, sh + [0.0, 3.0]) -
                   PO.allocation_fee(lsf, p, sh))
    @test iszero(PO.greedy_fee_delta(nothing, p, sh, 1, 2.0))

    # A `GreedyAllocation` with a fallback goes through the generic `optimise`, and the
    # greedy passes answer it.
    fai = FiniteAllocationInput(; w = [0.6, 0.4], prices = p, cash = 1e4)
    gfb = GreedyAllocation(; fb = GreedyAllocation())
    @test which(optimise, Tuple{typeof(gfb), typeof(fai)}) !=
          which(optimise, Tuple{typeof(ga), typeof(fai)})
    rfb = optimise(gfb, fai)
    @test isa(rfb, GreedyAllocationResult)
    @test collect(rfb.shares) == collect(optimise(ga, fai).shares)
end
# Issue #906: the sweep of `02_DiscreteFiniteAllocation.jl`. Each claim of the docstrings is
# checked with numbers: the four error formulations against an enumeration of every
# affordable book, the target money of a side, and the model entries of the fee.
# `JuMP` must be bound before the testset below expands its macros.
using JuMP
@testset "Discrete allocation: the programme, checked against enumeration (#906)" begin
    using PortfolioOptimisers, HiGHS, Clarabel, Pajarito, JuMP, Test, LinearAlgebra
    PO = PortfolioOptimisers
    mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                     settings = Dict("log_to_console" => false),
                     check_sol = (; allow_local = true, allow_almost = true))
    # HiGHS takes no cone, so the two `Squared` formulations need a conic MIP solver.
    conic_slv = Solver(; name = :pajarito,
                       solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                          "verbose" => false,
                                                          "oa_solver" =>
                                                              optimizer_with_attributes(HiGHS.Optimizer,
                                                                                        JuMP.MOI.Silent() =>
                                                                                            true),
                                                          "conic_solver" =>
                                                              optimizer_with_attributes(Clarabel.Optimizer,
                                                                                        "verbose" =>
                                                                                            false)),
                       check_sol = (; allow_local = true, allow_almost = true))
    wfs = (AbsoluteErrorWeightFinaliser(), SquaredAbsoluteErrorWeightFinaliser(),
           RelativeErrorWeightFinaliser(), SquaredRelativeErrorWeightFinaliser())
    da(wf) = DiscreteAllocation(; slv = conic_slv, wf = wf, fb = nothing)

    @testset "validation" begin
        @test_throws PO.IsEmptyError DiscreteAllocation(; slv = Solver[])
        @test_throws DomainError DiscreteAllocation(; slv = mip_slv, sc = 0)
        @test_throws DomainError DiscreteAllocation(; slv = mip_slv, so = -1)
    end

    @testset "each formulation solves the programme its docstring states" begin
        # The objective of the docstring, written out by hand: the error term `e` plus the
        # leftover cash, over every book the cash affords.
        function objective(wf, w, p, C, x)
            we = map(v -> iszero(v) ? eps(eltype(w)) : v, w)
            r = C - dot(x, p)
            return r + if isa(wf, AbsoluteErrorWeightFinaliser)
                norm(w * C - x .* p, 1)
            elseif isa(wf, SquaredAbsoluteErrorWeightFinaliser)
                norm(w * C - x .* p, 2)
            elseif isa(wf, RelativeErrorWeightFinaliser)
                C * norm((x .* p) ./ (we * C) .- 1, 1)
            else
                C * norm((x .* p) ./ (we * C) .- 1, 2)
            end
        end
        function enumerate_best(wf, w, p, C)
            best = (Inf, Int[])
            ub = floor.(Int, C ./ p)
            for i in 0:ub[1], j in 0:ub[2], k in 0:ub[3]
                x = [i, j, k]
                dot(x, p) > C && continue
                o = objective(wf, w, p, C, x)
                o < best[1] && (best = (o, x))
            end
            return best
        end
        # The weights sum to 0.8, so the long side's cash is 0.8 of the cash, and the side's
        # own weights are normalised before they are multiplied by it.
        w = [0.36, 0.28, 0.16]
        p = [13.0, 29.0, 47.0]
        cash = 750.0
        C = 0.8 * cash
        for wf in wfs
            r = optimise(da(wf), FiniteAllocationInput(; w = w, prices = p, cash = cash))
            obj, x = enumerate_best(wf, w / sum(w), p, C)
            @test isa(r.retcode, OptimisationSuccess)
            @test collect(r.shares) == x
            @test isapprox(JuMP.objective_value(r.l_model), obj; rtol = 1e-6)
            # The objective is `e + r`: `u` under an absolute error, `C u` under a relative one.
            m = r.l_model
            e = JuMP.value(m[:u]) * (if isa(wf,
                                            Union{RelativeErrorWeightFinaliser,
                                                  SquaredRelativeErrorWeightFinaliser})
                                         C
                                     else
                                         1
                                     end)
            @test isapprox(JuMP.objective_value(m), e + JuMP.value(m[:r]); rtol = 1e-6)
        end
        # The row that `wf` names is the one in the model.
        rows = [:cabs_err, :csqabs_err, :crel_err, :csqrel_err]
        for (wf, row) in zip(wfs, rows)
            m = optimise(da(wf), FiniteAllocationInput(; w = w, prices = p, cash = cash)).l_model
            @test haskey(m, row)
            @test all(!haskey(m, other) for other in setdiff(rows, [row]))
        end
    end

    @testset "a side's targets sum to its cash" begin
        # A long-only book with a budget of 0.8 spends the whole side cash, 1000 of 1250,
        # and the exact book of the target is reachable.
        fai = FiniteAllocationInput(; w = [0.4, 0.4], prices = [10.0, 100.0], cash = 1250.0)
        for wf in wfs
            r = optimise(da(wf), fai)
            @test collect(r.shares) == [50.0, 5.0]
            @test iszero(r.cash)
            @test collect(r.w) == [0.4, 0.4]
        end
        # A long-short book: each side's targets sum to its own cash.
        rls = optimise(DiscreteAllocation(; slv = mip_slv, fb = nothing),
                       FiniteAllocationInput(; w = [0.9, 0.6, -0.5],
                                             prices = [9.0, 6.0, 5.0], cash = 900.0))
        @test collect(rls.shares) == [90.0, 90.0, -90.0]
        # A long side whose weights are all zero keeps them, and buys nothing.
        r0 = optimise(DiscreteAllocation(; slv = mip_slv, fb = nothing),
                      FiniteAllocationInput(; w = [0.0, -1.0], prices = [10.0, 20.0],
                                            cash = 1000.0))
        @test collect(r0.shares) == [0.0, -50.0]
    end

    @testset "a relative error is priced in money" begin
        # The 50/50 book is 71 and 5 shares with 3 left idle. A relative error with no unit
        # beside the idle cash gives up the tracking, 100 and 3 shares, to spend those 3.
        fai = FiniteAllocationInput(; w = [0.5, 0.5], prices = [7.0, 100.0], cash = 1000.0)
        for wf in wfs
            r = optimise(da(wf), fai)
            @test collect(r.shares) == [71.0, 5.0]
            @test isapprox(r.cash, 3.0)
        end
        # A zero target weight is replaced by `eps` on a copy, so the division is defined and
        # the caller's weights do not change.
        w = [0.5, 0.0, 0.5]
        wc = copy(w)
        for wf in wfs[3:4]
            r = optimise(da(wf),
                         FiniteAllocationInput(; w = w, prices = [10.0, 20.0, 100.0],
                                               cash = 1000.0))
            @test collect(r.shares) == [50.0, 0.0, 5.0]
        end
        @test w == wc
        model = JuMP.Model()
        JuMP.@expression(model, sc, 1)
        JuMP.@variables(model, begin
                            x[1:3] >= 0, Int
                            u
                        end)
        wv = [0.5, 0.0, 0.5]
        err = PO.set_discrete_error!(model, wv, [10.0, 20.0, 100.0], 1000.0,
                                     RelativeErrorWeightFinaliser())
        @test wv == [0.5, 0.0, 0.5]
        @test isequal(err, 1000.0 * u)
    end

    @testset "the fee rows" begin
        w = [0.6, 0.4]
        p = [10.0, 20.0]
        fee = Fees(; tn = Turnover(; w = [0.6, 0.4], val = 0.002), fl = 2.0, l = 0.001)
        r = optimise(DiscreteAllocation(; slv = mip_slv, fb = nothing),
                     FiniteAllocationInput(; w = w, prices = p, cash = 1000.0,
                                           prev_cash = 500.0, horizon = 3, fees = fee))
        m = r.l_model
        for k in
            (:money, :fee_prop, :fee_tn, :fee_fixed, :fee, :t_ftn, :b, :cftn_ub, :cftn_lb,
             :cb_ub, :cb_lb, :cr, :r, :x, :u, :sc, :so, :cabs_err)
            @test haskey(m, k)
        end
        x = JuMP.value.(m[:x])
        money = x .* p
        # `b` is the indicator of a held position.
        @test JuMP.value.(m[:b]) ≈ Float64.(x .> 0.5)
        # `t_ftn` is an epigraph of the money traded, so it never lies below it.
        @test all(JuMP.value.(m[:t_ftn]) .>= abs.(money - 500.0 * w) .- 1e-6)
        # The model fee bounds the fee of the book, and the reported fee is the exact one:
        # 3 * 0.001 * 990 + 3 * 0.002 * 490 + 2 * 2 on this book.
        lsf, _ = PO.allocation_side_fees(fee, 3, 500.0, [true, true], Float64[])
        @test JuMP.value(m[:fee]) >= r.fees - 1e-6
        @test r.fees == PO.allocation_fee(lsf, p, collect(r.shares))
        @test isapprox(r.fees, 3 * 0.001 * 990 + 3 * 0.002 * 490 + 2 * 2)
        # The budget row holds the fee inside the leftover cash.
        @test JuMP.value(m[:r]) >= r.fees - 1e-6
    end

    @testset "a failed side warns by name" begin
        # A solver-less allocation fails on both sides of a long-short book, and each side
        # warns in turn: the short side is solved first.
        fai = FiniteAllocationInput(; w = [0.7, -0.3], prices = [10.0, 20.0], cash = 1000.0)
        da0 = DiscreteAllocation(; slv = Solver(; name = :none, solver = nothing),
                                 fb = nothing)
        r = @test_logs((:warn, r"s_retcode"), (:warn, r"l_retcode"), match_mode = :any,
                       optimise(da0, fai))
        @test isa(r.retcode, OptimisationFailure)
        @test isa(r.s_retcode, OptimisationFailure)
        @test isa(r.l_retcode, OptimisationFailure)
        @test all(iszero, collect(r.shares))
    end
end
@testset "The docstrings of 01_Base_FiniteAllocation.jl against numbers" begin
    using PortfolioOptimisers, Test, StableRNGs
    PO = PortfolioOptimisers
    rng = StableRNG(905)

    # `setup_alloc_optim`: the two side budgets of its `# Mathematical definition`, and the
    # consequence it states.
    w = [0.5, -0.2, 0.0, 0.7]
    lbgt, sbgt, lidx, sidx = PO.setup_alloc_optim(w)
    @test (lbgt, sbgt) == (1.2, 0.2)
    @test lidx == [true, false, true, true]
    @test sidx == .!lidx
    @test sum(w) ≈ lbgt - sbgt
    lbgt, sbgt, lidx, sidx = PO.setup_alloc_optim([0.5, 0.0, 0.5])
    @test isempty(sidx) && all(lidx)
    @test (lbgt, sbgt) == (1.0, 0.0)

    # `allocation_turnover_money`: a position that changes side has a negative previous
    # money, and the short side's money is negated.
    tn = Turnover(; w = [-0.2, 1.2, 0.3, -0.3], val = [0.01, 0.02, 0.03, 0.04])
    lidx = [true, true, false, false]
    lv, lprev = PO.allocation_turnover_money(tn, 100.0, lidx, false)
    sv, sprev = PO.allocation_turnover_money(tn, 100.0, .!lidx, true)
    @test collect(lprev) ≈ [-20.0, 120.0]
    @test collect(sprev) ≈ [-30.0, 30.0]
    @test collect(lv) == [0.01, 0.02]
    @test collect(sv) == [0.03, 0.04]
    @test PO.allocation_turnover_money(nothing, 100.0, lidx, false) === (nothing, nothing)

    # `allocation_liquidation_fee` is `T F_lq + F_flq` of `Fees`, with the rate term in money.
    lq = Turnover(; w = [0.25, -0.4], val = [0.01, 0.02])
    flq = Turnover(; w = [0.25, -0.4, 0.0], val = [5.0, 7.0, 11.0])
    fees = Fees(; lq = lq, flq = flq)
    @test PO.allocation_liquidation_fee(fees, 3, 1e4) ≈
          3 * 1e4 * PO.calc_liquidation_fees(lq) +
          PO.calc_fixed_liquidation_fees(flq, fees.kwargs)
    @test PO.allocation_liquidation_fee(fees, 3, 1e4) ≈ 3 * 1e4 * (0.0025 + 0.008) + 12.0
    fees = Fees(; lq = Turnover(; w = [0.25, -0.4], val = 0.01))
    @test PO.allocation_liquidation_fee(fees, 3, 1e4) ≈ 3 * 1e4 * 0.01 * 0.65
    @test iszero(PO.allocation_liquidation_fee(nothing, 3, 1e4))

    # `allocation_fee` summed over the two sides is `C_prev T F_r(w) + F_o(w)` of `Fees`, on
    # the signed weights `w = m / C_prev` of the book.
    errs = map(1:200) do _
        N = 6
        Cp = 1e4
        T = rand(rng, 1:30)
        p = 1 .+ 99 .* rand(rng, N)
        x = round.(20 .* randn(rng, N))
        x[rand(rng, 1:N)] = 0
        m = x .* p
        fees = Fees(; l = 0.001 .* rand(rng, N), s = 0.002 .* rand(rng, N),
                    fl = 5 .* rand(rng, N), fs = 3 .* rand(rng, N),
                    tn = Turnover(; w = 0.2 .* randn(rng, N), val = 0.003 .* rand(rng, N)),
                    lq = Turnover(; w = [0.1, -0.2], val = [0.01, 0.02]),
                    flq = Turnover(; w = [0.1, -0.2], val = [4.0, 6.0]))
        lidx = x .>= 0
        sidx = .!lidx
        lsf, ssf = PO.allocation_side_fees(fees, T, Cp, lidx, sidx)
        F = PO.allocation_fee(lsf, p[lidx], x[lidx]) +
            PO.allocation_fee(ssf, p[sidx], -x[sidx])
        ref = Cp * T * PO.calc_periodic_fees(m / Cp, fees) +
              PO.calc_one_off_fees(m / Cp, fees)
        abs(F - ref) / abs(ref)
    end
    @test maximum(errs) < 1e-12
    @test PO.allocation_side_fees(nothing, 3, 1e4, lidx, .!lidx) === (nothing, nothing)
    @test iszero(PO.allocation_fee(nothing, [1.0, 2.0], [3.0, 4.0]))
    # The empty book owes the sale of what the side held, and the forced exit.
    fees = Fees(; tn = Turnover(; w = [0.3, 0.7], val = 0.01),
                flq = Turnover(; w = [0.1], val = 4.0))
    lsf, _ = PO.allocation_side_fees(fees, 2, 1e3, [true, true], Float64[])
    @test PO.allocation_fee(lsf, [10.0, 20.0], [0.0, 0.0]) ≈ 2 * 0.01 * 1e3 + 4.0

    # `permute_side_fees` does not change the fee.
    N = 5
    p = 1 .+ 9 .* rand(rng, N)
    x = round.(10 .* rand(rng, N))
    fees = Fees(; l = rand(rng, N) / 100, fl = rand(rng, N),
                tn = Turnover(; w = rand(rng, N) / N, val = 0.01),
                lq = Turnover(; w = [0.1], val = 0.02))
    lsf, _ = PO.allocation_side_fees(fees, 7, 1e3, trues(N), Float64[])
    o = sortperm(rand(rng, N))
    @test PO.allocation_fee(PO.permute_side_fees(lsf, o), view(p, o), view(x, o)) ≈
          PO.allocation_fee(lsf, p, x)
    @test isnothing(PO.permute_side_fees(nothing, o))

    # `factory` keeps every field and replaces the last one, `fb`.
    fai = FiniteAllocationInput(; w = [0.6, 0.4], prices = [10.0, 20.0], cash = 1000.0)
    g = optimise(GreedyAllocation(), fai)
    chain = Tuple{PO.OptimisationEstimator, PO.OptimisationResult}[(GreedyAllocation(), g)]
    gf = PO.factory(g, chain)
    @test gf.fb === chain
    @test all(getfield(gf, i) === getfield(g, i) for i in 1:(fieldcount(typeof(g)) - 1))

    # `FOptE_FOpt`: a precomputed result is a fallback, and `optimise` returns it as it is.
    da = DiscreteAllocation(; slv = Solver(; name = :none, solver = nothing), fb = g)
    rfb = optimise(da, fai)
    @test isa(rfb, GreedyAllocationResult)
    @test collect(rfb.shares) == collect(g.shares)
    @test isa(rfb.fb, PO.FbChain) && rfb.fb[1][1] === da

    # `FiniteAllocationInput`: every raise of its `## Validation`.
    @test_throws PO.IsEmptyError FiniteAllocationInput(; w = Float64[], prices = [1.0])
    @test_throws PO.IsEmptyError FiniteAllocationInput(; w = [1.0], prices = Float64[])
    @test_throws DimensionMismatch FiniteAllocationInput(; w = [1.0], prices = [1.0, 2.0])
    @test_throws DomainError FiniteAllocationInput(; w = [1.0], prices = [1.0], cash = 0.0)
    @test_throws DomainError FiniteAllocationInput(; w = [1.0], prices = [1.0],
                                                   prev_cash = -1.0)
    @test_throws PO.IsNothingError FiniteAllocationInput(; w = [1.0], prices = [1.0],
                                                         fees = Fees(; l = 0.01))
    @test FiniteAllocationInput(; w = [1.0], prices = [1.0], cash = 5.0).prev_cash == 5.0
end
@testset "The collateral algorithms against their docstrings (#1337)" begin
    using PortfolioOptimisers, Test, StableRNGs, HiGHS
    PO = PortfolioOptimisers

    # The two methods of each algorithm against its `# Mathematical definition`. With
    # C = 100 and w = [1.2, -0.5]: b = 0.7, b_L = 1.2, b_S = 0.5, C_L = 120 and C_S = 50.
    w = [1.2, -0.5]
    p = [1.0, 1000.0]
    C = 100.0
    pc = ProceedsCollateral()
    @test pc(w, p, C) ≈ 50.0
    # The whole target sold with no fee gives the long target back.
    @test pc(w, p, C, 50.0, 0.0) ≈ 120.0
    @test pc(w, p, C, 30.0, 2.0) ≈ 70.0 + 30.0 - 2.0
    @test pc(w, p, C, 0.0, 0.0) ≈ 70.0
    # A book with b < 0 whose short side sells nothing buys nothing long.
    @test pc([0.2, -0.5], p, C, 0.0, 0.0) == 0.0
    # A long-only book: no short cash, and the long target.
    @test iszero(pc([0.5, 0.5], p, C))
    @test pc([0.5, 0.5], p, C, 0.0, 0.0) ≈ 100.0

    cc = CashCollateral()
    @test isnothing(cc.amount)
    @test cc(w, p, C) ≈ 50.0
    # K = C binds: min(C (b_L + b_S), K) = min(170, 100).
    @test cc(w, p, C, 0.0, 0.0) ≈ 100.0
    @test cc(w, p, C, 30.0, 2.0) ≈ 68.0
    # A large K: the long target plus the collateral that the short side did not use.
    ck = CashCollateral(; amount = 1000.0)
    @test ck(w, p, C, 0.0, 0.0) ≈ 120.0 + 50.0
    @test ck(w, p, C, 50.0, 0.0) ≈ 120.0
    # A small K caps the short side too, and the long side then gets nothing.
    cs = CashCollateral(; amount = 30.0)
    @test cs(w, p, C) ≈ 30.0
    @test iszero(cs(w, p, C, 30.0, 0.0))
    # A long-only book above a unit budget is capped at K.
    @test cc([0.9, 0.6], p, C, 0.0, 0.0) ≈ 100.0
    @test_throws DomainError CashCollateral(; amount = 0.0)
    @test_throws DomainError CashCollateral(; amount = -1)

    # The input holds the algorithm, and the default is `ProceedsCollateral`.
    fai = FiniteAllocationInput(; w = w, prices = p, cash = C)
    @test fai.ca === ProceedsCollateral()
    @test FiniteAllocationInput(; w = w, prices = p, cash = C, ca = cc).ca === cc

    # The rows of #1337. The short side cannot sell one share at a price of 1000, so it
    # leaves its whole cash. Under the default the long side spends `C b`, a continuous rule,
    # and under `CashCollateral()` it never spends more than the cash.
    mip = Solver(; name = :highs1337, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    rows = [[1.3, -0.3], [0.5, -0.5], [1.2, -0.5], [0.9, -0.8], [1.0999, -0.1], [1.1, -0.1]]
    proceeds = [100.0, 0.0, 70.0, 10.0, 99.0, 100.0]
    for alg in (GreedyAllocation(), DiscreteAllocation(; slv = mip, fb = nothing))
        alloc(w, ca) = collect(optimise(alg,
                                        FiniteAllocationInput(; w = w, prices = p, cash = C,
                                                              ca = ca)).shares)
        for (w, n) in zip(rows, proceeds)
            @test alloc(w, ProceedsCollateral()) == [n, 0.0]
            @test alloc(w, CashCollateral()) == [100.0, 0.0]
        end
    end

    # The cash identity of each algorithm, on books whose short side sells and pays a fee.
    # `sum(cost)` is the long money less the short money, and `fees` holds both sides' fees,
    # so the reported cash is the long cash less the long money and the long fee. A long
    # cash floored at zero buys nothing and pays no fee here, so its cash is zero.
    rng = StableRNG(1337)
    fees = Fees(; l = 0.002, s = 0.003, fl = 0.5, fs = 0.7)
    for _ in 1:20
        w = randn(rng, 6) / 3
        p = 5.0 .+ 50.0 * rand(rng, 6)
        C = 1000.0 + 5000.0 * rand(rng)
        for ga in (GreedyAllocation(), DiscreteAllocation(; slv = mip, fb = nothing))
            fai = FiniteAllocationInput(; w = w, prices = p, cash = C, horizon = 3,
                                        fees = fees)
            r = optimise(ga, fai)
            cost = collect(r.cost)
            @test r.cash ≈ max(0.0, C * sum(w) - sum(cost) - r.fees) atol = 1e-8
            K = 0.9 * C
            fai = FiniteAllocationInput(; w = w, prices = p, cash = C, horizon = 3,
                                        fees = fees, ca = CashCollateral(; amount = K))
            r = optimise(ga, fai)
            cost = collect(r.cost)
            @test r.cash ≈ max(0.0, min(C * sum(abs, w), K) - sum(abs, cost) - r.fees) atol = 1e-8
            @test sum(abs, cost) + r.fees <= min(C * sum(abs, w), K) + 1e-8
        end
    end

    # A fallback chain reads one rule, because the input holds it. A solver-less head fails,
    # and its greedy fallback answers with the head's collateral algorithm.
    fai = FiniteAllocationInput(; w = [1.2, -0.5], prices = [1.0, 1000.0], cash = 100.0,
                                ca = CashCollateral())
    da0 = DiscreteAllocation(; slv = Solver(; name = :none, solver = nothing))
    r = @test_logs((:warn,), match_mode = :any, optimise(da0, fai))
    @test collect(r.shares) == collect(optimise(GreedyAllocation(), fai).shares)
    @test collect(r.shares) == [100.0, 0.0]
end
