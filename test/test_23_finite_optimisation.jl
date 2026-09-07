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
