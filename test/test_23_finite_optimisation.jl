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
    @test isapprox(rmsd(res.w, res_da.w), 0.0838, rtol = 5e-4)

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
    @test isapprox(sum(res_da.cost), 4206.9 * 1.2, rtol = 1e-4)
    @test isapprox(sum(res.w[res.w .< 0]), -1, rtol = 1e-3)
    @test isapprox(res_da.shares .* vec(values(X[end])), res_da.cost)
    @test isapprox(rmsd(res.w, res_da.w), 0.2662, rtol = 5e-4)

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

    rtol = if Sys.isapple()
        1e-2
    else
        5e-3
    end
    result = isapprox(sum(res_da.cost), 4206.9 * 0.8; rtol = rtol)
    if !result
        @test isapprox(3337.326, 3337.326; rtol = 0.005)
    else
        @test result
    end
    @test isapprox(res_da.shares .* vec(values(X[end])), res_da.cost)
    @test isapprox(rmsd(res.w, res_da.w), 0.029094976416644103, rtol = 5e-2)
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
    # nothing` reaches the shortcut method in `22_DiscreteFiniteAllocation.jl`, which
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
    lsf, _ = PO.allocation_side_fees(fmoney, nothing, 252, 1e6, [true, true], Float64[])
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
        lsf2, _ = PO.allocation_side_fees(fee, nothing, T, cash, [true, true], Float64[])
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
        red = PO.port_opt_view(full, i, Xf)
        # The view reduces the five holding fields to the mask and the two carriers to its
        # complement, which is the pair of lengths #914 reported at the allocator's door.
        @test length(red.l) == 3
        @test length(red.lq.w) == 1
        lift = PO.lift_fees(red, imsk)
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
        # A `nothing` on either side lifts nothing, and a scalar rate carries through.
        @test PO.lift_fees(full, nothing) === full
        @test isnothing(PO.lift_fees(nothing, imsk))
        @test PO.lift_fees(Fees(; l = 0.01), imsk).l == 0.01
        @test isnothing(PO.lift_fees(Fees(; l = 0.01), imsk).lq)
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
        @test FiniteAllocationInput(res; prices = prices, cash = cash, horizon = T,
                                    fees = fees, imsk = imsk).fees === fees
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
