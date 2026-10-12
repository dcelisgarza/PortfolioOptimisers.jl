using JuMP: JuMP
include(joinpath(@__DIR__, "parity_harness.jl"))
@testset "Fees" begin
    using PortfolioOptimisers, Test, DataFrames, TimeSeries, CSV, Clarabel, HiGHS
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
    rd = prices_to_returns(X)
    pr = prior(EmpiricalPrior(), rd)
    rf = 4.2 / 100 / 252
    w = fill(inv(size(pr.X, 2)), size(pr.X, 2))
    slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    mip_slv = Solver(; name = :highs1, solver = HiGHS.Optimizer,
                     settings = Dict("log_to_console" => false),
                     check_sol = (; allow_local = true, allow_almost = true))
    da = DiscreteAllocation(; slv = mip_slv)
    sets = UniverseSets(;
                        dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                                    "group2" => rd.nx[2:2:end]))
    fest = FeesEstimator(; tn = TurnoverEstimator(; w = w, val = Dict("BAC" => 0.001)),
                         l = Dict("group2" => 0.002), s = Dict("group1" => 0.003),
                         fl = Dict("XOM" => 0.005, "WMT" => 0.005, "LLY" => 0.005),
                         fs = Dict("BBY" => 0.007, "CVX" => 0.007, "group3" => 0.011))
    fes = [fees_constraints(fest, sets),
           Fees(; tn = Turnover(; val = 0.001, w = w), l = 0.002, s = 0.003, fl = 0.005,
                fs = 0.007)]
    @test factory(fes[2], 2.5 * w).tn.w == 2.5 * w
    @test factory(fest, 2.5 * w).tn.w == 2.5 * w
    @test factory(FeesEstimator(;
                                tn = TurnoverEstimator(; w = w, val = 0.001, fixed = true)),
                  2.5 * w).tn.w == w
    T, N = size(pr.X)
    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; wb = WeightBounds(; lb = -1, ub = 1),
                                                sbgt = 1, bgt = 1, pe = pr, slv = slv)))
    # Issue #900: the fee of a finite allocation, priced by hand on the money the
    # allocator actually bought. `shares .* p` is that money exactly. The long side takes
    # `l` and `fl`, the short side takes `s` and `fs`, and the turnover charges the money
    # traded against `prev_cash .* tn.w`, whose absolute value is the same on both sides.
    function alloc_fee_by_hand(alloc, wt, p, T, fe, prev_cash)
        dot_scalar = PortfolioOptimisers.dot_scalar
        money = collect(alloc.shares) .* p
        lidx = wt .>= 0
        sidx = .!lidx
        fee = zero(eltype(money))
        if !isnothing(fe.l)
            fee += T * dot_scalar(PortfolioOptimisers.nothing_scalar_array_view(fe.l, lidx),
                                  view(money, lidx))
        end
        if !isnothing(fe.s)
            fee -= T * dot_scalar(PortfolioOptimisers.nothing_scalar_array_view(fe.s, sidx),
                                  view(money, sidx))
        end
        if !isnothing(fe.tn)
            prev = prev_cash * fe.tn.w
            fee += T * dot_scalar(fe.tn.val, abs.(money - prev))
        end
        if !isnothing(fe.fl)
            fee += dot_scalar(PortfolioOptimisers.nothing_scalar_array_view(fe.fl, lidx),
                              .!iszero.(view(money, lidx)))
        end
        if !isnothing(fe.fs)
            fee += dot_scalar(PortfolioOptimisers.nothing_scalar_array_view(fe.fs, sidx),
                              .!iszero.(view(money, sidx)))
        end
        return fee
    end
    @testset "Fees" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/Fees.csv.gz"), DataFrame)
        f1s = [0.02002313426946848, 0.12149580659357644]
        for (i, fe) in pairs(fes)
            res_mip = optimise(da,
                               FiniteAllocationInput(; w = res.w,
                                                     prices = vec(values(X[end])),
                                                     cash = 1000, horizon = T, fees = fe))
            f1 = sum(calc_fees(res.w, T, fe))
            @test isapprox(f1s[i], f1)
            fa2, fo2 = calc_asset_fees(res.w, T, fe)
            f2 = fa2[1] .+ fo2[1]
            @test isapprox(df[!, "$(2*(i-1)+1)"], f2)
            # Issue #900: the allocation charges its fee inside its own model, on the
            # money it actually buys, so the result reports the charge of the realised
            # share counts and not of the target weights.
            fopt1 = alloc_fee_by_hand(res_mip, res.w, vec(values(X[end])), T, fe, 1000)
            @test isapprox(res_mip.fees, fopt1)
            @test res_mip.fees > 0
            # Issue #898: `fa` is `nothing` here, so the per period charge falls on every
            # row and the one-off charge falls on the first row alone.
            sched = fill(PortfolioOptimisers.calc_periodic_fees(res.w, fe), size(pr.X, 1))
            sched[1] += PortfolioOptimisers.calc_one_off_fees(res.w, fe)
            @test all(isapprox(calc_net_returns(res.w, pr.X) .- sched,
                               calc_net_returns(res.w, pr.X, fe)))
            asched = repeat(transpose(PortfolioOptimisers.calc_asset_periodic_fees(res.w,
                                                                                   fes[1])[1]),
                            size(pr.X, 1), 1)
            asched[1, :] .+= PortfolioOptimisers.calc_asset_one_off_fees(res.w, fes[1])[1]
            # The split is one matrix on the caller's own universe. This fee carries no
            # liquidation carrier, so every column is an investable one.
            @test size(calc_net_asset_returns(res.w, pr.X, fes[1])) == size(pr.X)
            @test all(isapprox(calc_net_asset_returns(res.w, pr.X) .- asched,
                               calc_net_asset_returns(res.w, pr.X, fes[1])))
        end
        @test all(iszero, calc_fees(res.w, T, Fees()))
        @test all(iszero,
                  calc_asset_fees(res.w, T, Fees())[1][1] .+
                  calc_asset_fees(res.w, T, Fees())[2][1])
        # An input that states no fee pays none, and the whole cash is spent or left over.
        res_free = optimise(da,
                            FiniteAllocationInput(; w = res.w, prices = vec(values(X[end])),
                                                  cash = 1000))
        @test iszero(res_free.fees)
        res_greedy = optimise(GreedyAllocation(),
                              FiniteAllocationInput(; w = res.w,
                                                    prices = vec(values(X[end])),
                                                    cash = 1000))
        @test iszero(res_greedy.fees)
    end
    @testset "Expected Returns" begin
        r = factory(Variance(), pr, slv)
        # Issue #898: an expected return is a per period number, so the one-off terms
        # enter it spread over the observation count of the fit.
        f = PortfolioOptimisers.calc_periodic_fees(res.w, fes[1]) +
            PortfolioOptimisers.calc_one_off_fees(res.w, fes[1]) / T
        rt = expected_return(res.ret, res.w, pr)
        rtf = expected_return(res.ret, res.w, pr, fes[1])
        rk = expected_risk(r, res.w, pr, fes[1])
        sr = (rt - rf) / rk
        srf = (rt - rf - f) / rk
        sric = sr - N / (T * sr)
        srfic = srf - N / (T * srf)

        @test isapprox(rtf, rt - f)
        @test isapprox(srf, expected_ratio(r, res.ret, res.w, pr, fes[1]; rf = rf))
        @test isapprox(sr, expected_ratio(r, res.ret, res.w, pr; rf = rf))
        @test isapprox(sric, expected_sric(r, res.ret, res.w, pr; rf = rf))
        @test isapprox(srfic, expected_sric(r, res.ret, res.w, pr, fes[1]; rf = rf))
        @test all(isapprox.((rk, rtf, srf),
                            expected_risk_ret_ratio(r, res.ret, res.w, pr, fes[1]; rf = rf)))
        @test all(isapprox.((rk, rt, sric),
                            expected_risk_ret_sric(r, res.ret, res.w, pr; rf = rf)))
        @test all(isapprox.((rk, rtf, srfic),
                            expected_risk_ret_sric(r, res.ret, res.w, pr, fes[1]; rf = rf)))

        @test isapprox(expected_risk(ExpectedReturn(), res.w, pr, fes[1]), rt - f)
        @test isapprox(expected_risk(factory(ExpectedReturnRiskRatio(; rf = rf), pr), res.w,
                                     pr, fes[1]), srf)
    end
    # Issue #545, condition 2: the reference-weight vocabulary of `src/12_Turnover.jl`,
    # checked with numbers rather than read.
    @testset "Turnover reference weights, name resolution and views" begin
        w0 = [0.2, 0.3, 0.5]
        wn = [0.1, 0.4, 0.5]
        tnf = TurnoverEstimator(; w = w0, val = Dict("A" => 0.1, "B" => 0.2), dval = 0.0,
                                fixed = true)
        tnv = TurnoverEstimator(; w = w0, val = Dict("A" => 0.1, "B" => 0.2), dval = 0.0,
                                fixed = false)

        # `fixed` decides which weight vector survives.
        @test factory(tnf, wn) === tnf
        @test factory(tnf, wn).w == w0
        @test factory(tnv, wn).w == wn
        @test factory(Turnover(; w = w0, val = 0.1, fixed = true), wn).w == w0
        @test factory(Turnover(; w = w0, val = 0.1, fixed = false), wn).w == wn

        # `needs_previous_weights` is `!fixed`, and the vector method is `any`, not `all`.
        @test PortfolioOptimisers.needs_previous_weights(tnf) == !tnf.fixed
        @test PortfolioOptimisers.needs_previous_weights(tnv) == !tnv.fixed
        tnmix = PortfolioOptimisers.concrete_typed_array([tnf, tnv])
        @test PortfolioOptimisers.needs_previous_weights(tnmix)
        @test !all(PortfolioOptimisers.needs_previous_weights.(tnmix))
        @test !PortfolioOptimisers.needs_previous_weights([tnf, tnf])

        # Name resolution follows the universe, and `dval` fills what the keys miss.
        tnsets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        tnd = TurnoverEstimator(; w = w0, val = Dict("C" => 0.3, "A" => 0.1), dval = 0.05)
        @test turnover_constraints(tnd, tnsets).val == [0.1, 0.05, 0.3]

        # `dval = nothing` fills with `zero(datatype)`, and `datatype` reaches the fill.
        tnn = TurnoverEstimator(; w = w0, val = Dict("C" => 0.3, "A" => 0.1))
        @test turnover_constraints(tnn, tnsets).val == [0.1, 0.0, 0.3]
        @test eltype(turnover_constraints(tnn, tnsets).val) == Float64
        @test eltype(turnover_constraints(tnn, tnsets; datatype = Float32).val) == Float32

        # An unmatched name raises when `strict` is set, and warns otherwise.
        tnx = TurnoverEstimator(; w = w0, val = Dict("Z" => 0.3))
        @test_throws ArgumentError turnover_constraints(tnx, tnsets; strict = true)
        @test (@test_logs (:warn,) turnover_constraints(tnx, tnsets; strict = false)).val ==
              [0.0, 0.0, 0.0]

        # The vector method maps over the vector and preserves its order.
        tn2 = TurnoverEstimator(; w = wn, val = Dict("B" => 0.15))
        tnvec = turnover_constraints([tnd, tn2], tnsets)
        @test [tni.val for tni in tnvec] == [[0.1, 0.05, 0.3], [0.0, 0.15, 0.0]]
        @test [tni.w for tni in tnvec] == [w0, wn]

        # A `Turnover` passes through `turnover_constraints` unchanged.
        tnres = Turnover(; w = w0, val = [0.1, 0.2, 0.3])
        @test turnover_constraints(tnres, tnsets) === tnres
        @test isnothing(turnover_constraints(nothing, tnsets))

        # The constructor guards, by exception type.
        @test_throws DimensionMismatch Turnover(; w = w0, val = [0.1, 0.2])
        @test_throws DomainError Turnover(; w = w0, val = [0.1, -0.2, 0.3])
        @test_throws DomainError Turnover(; w = w0, val = [0.1, NaN, 0.3])
        @test_throws DomainError Turnover(; w = w0, val = NaN)
        @test_throws PortfolioOptimisers.IsEmptyError Turnover(; w = Float64[], val = 0.1)

        # The view slices `w` and a vector `val` alike, and leaves a scalar `val` alone.
        tvv = Turnover(; w = w0, val = [0.1, 0.2, 0.4], fixed = true)
        tvs = Turnover(; w = w0, val = 0.02)
        vv = PortfolioOptimisers.port_opt_view(tvv, [1, 3])
        vs = PortfolioOptimisers.port_opt_view(tvs, [1, 3])
        hand = Turnover(; w = w0[[1, 3]], val = [0.1, 0.4], fixed = true)
        @test collect(vv.w) == hand.w
        @test collect(vv.val) == hand.val
        @test vv.fixed == hand.fixed
        @test collect(vs.w) == w0[[1, 3]]
        @test vs.val === tvs.val
        ve = PortfolioOptimisers.port_opt_view(tnf, [1, 3])
        @test collect(ve.w) == w0[[1, 3]]
        @test ve.val === tnf.val
    end
    # Issue #546, condition 2: the fee arithmetic of `src/13_Fees.jl`, checked with
    # numbers rather than read.
    @testset "Fee terms, the per-asset identity and name resolution" begin
        wf = [0.6, -0.4, 0.0, 0.25]
        tnf = Turnover(; w = [0.1, 0.2, 0.3, 0.4], val = [0.01, 0.02, 0.03, 0.04])
        fev = Fees(; tn = tnf, l = [0.001, 0.002, 0.003, 0.004],
                   s = [0.005, 0.006, 0.007, 0.008], fl = [1.0, 2.0, 3.0, 4.0],
                   fs = [5.0, 6.0, 7.0, 8.0])
        tns = Turnover(; w = [0.1, 0.2, 0.3, 0.4], val = 0.02)
        fesc = Fees(; tn = tns, l = 0.001, s = 0.005, fl = 1.0, fs = 5.0)

        # The per-asset fee sums to the portfolio fee, up to the order of summation.
        tot2 = p -> sum(p[1]) + sum(p[2])
        @test all(isapprox.(map(tot2, calc_asset_fees(wf, 21, fev)),
                            calc_fees(wf, 21, fev)))
        @test all(isapprox.(map(tot2, calc_asset_fees(wf, 21, fesc)),
                            calc_fees(wf, 21, fesc)))

        # The short proportional term is a positive charge, not a credit.
        @test sum(calc_fees([0.6, -0.4], 21, Fees(; s = 0.01))) == 0.004
        let p = calc_asset_fees([0.6, -0.4], 21, Fees(; s = 0.01))
            @test p[1][1] .+ p[2][1] == [0.0, 0.004]
        end

        # Issue #900: the fee family carries no price. A price reaches the finite
        # allocation alone, which holds the share counts and prices the money it buys.
        ffx = Fees(; fl = 3.0, fs = 7.0)
        @test sum(calc_fees([0.6, -0.4], 21, ffx)) == 10.0
        @test isempty(methods(calc_fees,
                              Tuple{Vector{Float64}, Vector{Float64}, Int, Fees}))
        @test isempty(methods(calc_total_fees,
                              Tuple{Vector{Float64}, Vector{Float64}, Int, Fees}))
        @test isempty(methods(calc_asset_fees,
                              Tuple{Vector{Float64}, Vector{Float64}, Int, Fees}))
        @test isempty(methods(calc_total_asset_fees,
                              Tuple{Vector{Float64}, Vector{Float64}, Int, Fees}))
        @test calc_fixed_fees([0.6, -0.4], 3.0, (; atol = 1e-8), .>=) == 3.0
        @test calc_fixed_fees([0.6, -0.4], 7.0, (; atol = 1e-8), .<) == 7.0

        # `kwargs` decides how near zero counts as zero, and `atol` moves the boundary.
        @test Fees(; fl = 2.0).kwargs == (; atol = 1e-8)
        @test sum(calc_fees([1e-9, 0.5], 21, Fees(; fl = 2.0))) == 2.0
        @test sum(calc_fees([1e-7, 0.5], 21, Fees(; fl = 2.0))) == 4.0
        @test sum(calc_fees([1e-7, 0.5], 21, Fees(; fl = 2.0, kwargs = (; atol = 1e-6)))) ==
              2.0
        @test sum(calc_fees([1e-9, 0.5], 21, Fees(; fl = 2.0, kwargs = (; atol = 1e-10)))) ==
              4.0

        # Issue #546: `calc_asset_fixed_fees` on a vector rate used to raise a
        # `DimensionMismatch` whenever the selected side held a near-zero weight, because
        # it wrote one entry per charged position into one slot per selected position.
        @test calc_asset_fixed_fees([0.6, 0.0, 0.25], [2.0, 3.0, 4.0], (; atol = 1e-8),
                                    .>=) == [2.0, 0.0, 4.0]
        @test sum(calc_asset_fixed_fees([0.6, 0.0, 0.25], [2.0, 3.0, 4.0], (; atol = 1e-8),
                                        .>=)) ==
              calc_fixed_fees([0.6, 0.0, 0.25], [2.0, 3.0, 4.0], (; atol = 1e-8), .>=)
        @test calc_asset_fixed_fees([-0.6, 0.0, -0.25], [2.0, 3.0, 4.0], (; atol = 1e-8),
                                    .<) == [2.0, 0.0, 4.0]

        # The turnover term, against the two expressions computed by hand.
        tnc = Turnover(; w = tns.w, val = fill(0.02, 4))
        @test calc_fees(wf, tns) == tns.val * sum(abs.(wf - tns.w))
        @test calc_fees(wf, tnc) == dot(tnc.val, abs.(wf - tnc.w))
        @test isapprox(calc_fees(wf, tns), calc_fees(wf, tnc))

        # `fixed` is a `factory` flag; no `calc_fees` method reads it.
        tn_fx = Turnover(; w = tns.w, val = 0.02, fixed = true)
        tn_fr = Turnover(; w = tns.w, val = 0.02, fixed = false)
        @test calc_fees(wf, tn_fx) == calc_fees(wf, tn_fr)
        @test calc_asset_fees(wf, tn_fx) == calc_asset_fees(wf, tn_fr)
        @test calc_fees(wf, factory(tn_fx, wf)) == calc_fees(wf, tn_fx)
        @test iszero(calc_fees(wf, factory(tn_fr, wf)))

        # The `Nothing` methods return a typed zero.
        w32 = Float32[0.6, -0.4]
        @test typeof(calc_fees(w32, nothing, .>=)) === Float32
        @test typeof(calc_fees(w32, nothing)) === Float32
        @test typeof(calc_fixed_fees(w32, nothing, (; atol = 1e-8), .>=)) === Float32
        @test eltype(calc_asset_fees(w32, nothing, .>=)) === Float32
        @test eltype(calc_asset_fees(w32, nothing)) === Float32
        @test eltype(calc_asset_fixed_fees(w32, nothing, (; atol = 1e-8), .>=)) === Float32

        # Each default fills its own field and never a neighbour's.
        fsets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        fest2 = FeesEstimator(;
                              tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5],
                                                     val = Dict("C" => 0.3), dval = 0.05),
                              l = Dict("A" => 0.001), dl = 0.01, s = Dict("B" => 0.002),
                              ds = 0.02, fl = Dict("C" => 3.0), dfl = 30.0,
                              fs = Dict("A" => 4.0), dfs = 40.0)
        fr2 = fees_constraints(fest2, fsets)
        @test fr2.l == [0.001, 0.01, 0.01]
        @test fr2.s == [0.02, 0.002, 0.02]
        @test fr2.fl == [30.0, 30.0, 3.0]
        @test fr2.fs == [4.0, 40.0, 40.0]

        # The nested `tn` resolves too, so a `TurnoverEstimator` becomes a `Turnover`.
        @test fr2.tn isa Turnover
        @test fr2.tn.val == [0.05, 0.05, 0.3]
        @test fr2.tn.w == [0.2, 0.3, 0.5]

        # Issue #546: `fees_constraints` used to drop the estimator's `kwargs`, so the
        # `atol` a caller set never reached the fixed-fee boundary.
        festk = FeesEstimator(; fl = Dict("A" => 2.0), dfl = 2.0, kwargs = (; atol = 1e-4))
        @test fees_constraints(festk, fsets).kwargs === festk.kwargs
        @test sum(calc_fees([1e-5, 0.5, 0.5], 21, fees_constraints(festk, fsets))) == 4.0
        @test sum(calc_fees([1e-5, 0.5, 0.5], 21,
                            fees_constraints(FeesEstimator(; fl = Dict("A" => 2.0),
                                                           dfl = 2.0), fsets))) == 6.0

        # An unmatched name raises when `strict` is set, and warns otherwise.
        festx = FeesEstimator(; l = Dict("Z" => 0.001), dl = 0.01)
        @test_throws ArgumentError fees_constraints(festx, fsets; strict = true)
        @test (@test_logs (:warn,) fees_constraints(festx, fsets; strict = false)).l ==
              [0.01, 0.01, 0.01]

        # A `nothing` fee field stays `nothing`; no default invents one.
        femp = fees_constraints(FeesEstimator(), fsets)
        @test isnothing(femp.tn)
        @test isnothing(femp.l)
        @test isnothing(femp.s)
        @test isnothing(femp.fl)
        @test isnothing(femp.fs)

        # A `Fees` passes through `fees_constraints` unchanged.
        @test fees_constraints(fev) === fev
        @test isnothing(fees_constraints(nothing))

        # Only the turnover term needs a previous weight vector.
        @test PortfolioOptimisers.needs_previous_weights(Fees(; tn = tn_fr))
        @test !PortfolioOptimisers.needs_previous_weights(Fees(; tn = tn_fx))
        @test !PortfolioOptimisers.needs_previous_weights(Fees(; l = 0.01, fl = 1.0))
    end
    # Issue #898: `l`, `s` and `tn` are rates per period, and `fl` and `fs` are charged
    # one time for the whole holding period. `fa` names the clock the two fixed terms fall
    # on, and it carries no number: every site that charges a fee hands in the count it
    # charges over.
    @testset "Fee amortisation" begin
        wf = [0.6, -0.4, 0.0, 0.25]
        tnf = Turnover(; w = [0.1, 0.2, 0.3, 0.4], val = 0.02)
        fee0 = Fees(; tn = tnf, l = 0.001, s = 0.002, fl = 0.5, fs = 1.0)
        feeA = Fees(; tn = tnf, l = 0.001, s = 0.002, fl = 0.5, fs = 1.0,
                    fa = AmortisedFees())

        # `fa` defaults to `nothing` on both fee types, and `AmortisedFees` is field-less.
        @test isnothing(Fees().fa)
        @test isnothing(FeesEstimator().fa)
        @test isempty(fieldnames(AmortisedFees))
        @test AmortisedFees() isa PortfolioOptimisers.AbstractFeeAmortisation

        # The two halves. `l`, `s` and `tn` charge every period; `fl` and `fs` charge once.
        periodic = PortfolioOptimisers.calc_periodic_fees(wf, fee0)
        oneoff = PortfolioOptimisers.calc_one_off_fees(wf, fee0)
        @test isapprox(periodic,
                       calc_fees(wf, fee0.l, .>=) - calc_fees(wf, fee0.s, .<) +
                       calc_fees(wf, tnf))
        @test isapprox(oneoff,
                       calc_fixed_fees(wf, fee0.fl, fee0.kwargs, .>=) +
                       calc_fixed_fees(wf, fee0.fs, fee0.kwargs, .<))

        # A `nothing` clock reports the two halves apart, whatever count it is handed.
        @test all(isapprox.(calc_fees(wf, 3, fee0), (periodic, oneoff)))
        @test all(isapprox.(calc_fees(wf, 5, fee0), (periodic, oneoff)))
        # An `AmortisedFees` clock spreads the one-off half over the count it is handed.
        @test all(isapprox.(calc_fees(wf, 3, feeA), (periodic + oneoff / 3, 0.0)))
        @test all(isapprox.(calc_fees(wf, 5, feeA), (periodic + oneoff / 5, 0.0)))

        # The per asset split sums to the portfolio pair, under both clocks and both
        # families, to the order of summation.
        tot3 = p -> sum(p[1]) + sum(p[2])
        @test all(isapprox.(map(tot3, calc_asset_fees(wf, 3, fee0)),
                            calc_fees(wf, 3, fee0)))
        @test all(isapprox.(map(tot3, calc_asset_fees(wf, 3, feeA)),
                            calc_fees(wf, 3, feeA)))

        # The whole holding period: `T` periods of the rates, and the fixed terms one time.
        # The clock does not move that total, only where the cost lands on a series.
        @test isapprox(calc_total_fees(wf, 3, fee0), 3 * periodic + oneoff)
        @test isapprox(calc_total_fees(wf, 3, fee0), calc_total_fees(wf, 3, feeA))
        @test isapprox(tot3(calc_total_asset_fees(wf, 3, fee0)),
                       calc_total_fees(wf, 3, fee0))
        @test isapprox(tot3(calc_total_asset_fees(wf, 3, feeA)),
                       calc_total_fees(wf, 3, feeA))
        @test iszero(calc_total_fees(wf, 3, nothing))
        @test all(iszero, calc_total_asset_fees(wf, 3, nothing))

        # The two clocks charge the same total over a series, and land it differently.
        Xf = [0.01 0.02 -0.01 0.03; 0.03 0.04 0.02 -0.02; -0.01 0.005 0.01 0.04]
        gross = Xf * wf
        net0 = calc_net_returns(wf, Xf, fee0)
        netA = calc_net_returns(wf, Xf, feeA)
        @test isapprox(sum(gross) - sum(net0), calc_total_fees(wf, 3, fee0))
        @test isapprox(sum(gross) - sum(netA), calc_total_fees(wf, 3, feeA))
        # The `nothing` clock puts the whole one-off cost on the first observation.
        @test isapprox(gross[1] - net0[1], periodic + oneoff)
        @test isapprox(gross[2] - net0[2], periodic)
        @test isapprox(gross[3] - net0[3], periodic)
        # The `AmortisedFees` clock puts an equal share on each.
        @test all(isapprox.(gross .- netA, periodic + oneoff / 3))
        # The per asset rows sum to the portfolio series, under both clocks.
        @test isapprox(vec(sum(calc_net_asset_returns(wf, Xf, fee0); dims = 2)), net0)
        @test isapprox(vec(sum(calc_net_asset_returns(wf, Xf, feeA); dims = 2)), netA)

        # `l` and `s` are unmoved by the clock.
        @test feeA.l == fee0.l && feeA.s == fee0.s

        # `fees_constraints` carries the estimator's `fa` to the result, unchanged.
        fsets2 = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        festA = FeesEstimator(; l = Dict("A" => 0.001), fa = AmortisedFees())
        @test fees_constraints(festA, fsets2).fa === festA.fa
    end
    # Issue #902: the family gained a second leaf, so a caller who must *state* the
    # first-observation clock has a word for it. On a `Fees` it is a synonym for `nothing`.
    @testset "FirstObservationFees is the word for the nothing clock" begin
        wf2 = [0.6, -0.4, 0.0, 0.25]
        tnf2 = Turnover(; w = [0.1, 0.2, 0.3, 0.4], val = 0.02)
        fee0 = Fees(; tn = tnf2, l = 0.001, s = 0.002, fl = 0.5, fs = 1.0)
        feeF = Fees(; tn = tnf2, l = 0.001, s = 0.002, fl = 0.5, fs = 1.0,
                    fa = FirstObservationFees())
        feeA = Fees(; tn = tnf2, l = 0.001, s = 0.002, fl = 0.5, fs = 1.0,
                    fa = AmortisedFees())

        @test isempty(fieldnames(FirstObservationFees))
        @test FirstObservationFees() isa PortfolioOptimisers.AbstractFeeAmortisation

        # The two value-level verbs answer the same under the leaf as under `nothing`.
        # Issue #900 deleted the price-carrying family, so only the no-price pair remains.
        @test all(isapprox.(calc_fees(wf2, 3, feeF), calc_fees(wf2, 3, fee0)))
        @test all(isapprox.(calc_asset_fees(wf2, 3, feeF)[1],
                            calc_asset_fees(wf2, 3, fee0)[1]))
        @test all(isapprox.(calc_asset_fees(wf2, 3, feeF)[2],
                            calc_asset_fees(wf2, 3, fee0)[2]))

        # The supertype decides no answer: each leaf carries its own method, so a third
        # clock added to the family would get a `MethodError` rather than silently
        # inheriting the amortised arm. No clock-reading method binds the bare supertype.
        clock_param(m) = m.sig.parameters[end]
        clock_readers = [PortfolioOptimisers.calc_fees, PortfolioOptimisers.calc_asset_fees,
                         PortfolioOptimisers.charge_one_time_fees]
        for verb in clock_readers
            bound = unique(clock_param(m) for m in methods(verb))
            @test !any(t -> t === PortfolioOptimisers.AbstractFeeAmortisation, bound)
            @test any(t -> t === AmortisedFees, bound)
            @test any(t -> t === Union{Nothing, FirstObservationFees}, bound)
        end

        # The model's one-off charge lands on the first observation and touches no other,
        # which is the rule `charge_one_time_fees` documents. Only its dispatch was checked
        # before, so the arm that charges the first observation asserted no number at all.
        mknet = m -> [JuMP.@expression(m, 1 * m[:vj][1]),
                      JuMP.@expression(m, 2 * m[:vj][1]),
                      JuMP.@expression(m, 3 * m[:vj][1])]
        mdl = JuMP.Model()
        JuMP.@variable(mdl, vj[1:1])
        mdl[:vj] = vj
        ot = JuMP.@expression(mdl, 10 * vj[1])
        first_obs = PortfolioOptimisers.charge_one_time_fees(mdl, mknet(mdl), ot, 3,
                                                             nothing)
        @test JuMP.coefficient(first_obs[1], vj[1]) == 1 - 10
        @test JuMP.coefficient(first_obs[2], vj[1]) == 2
        @test JuMP.coefficient(first_obs[3], vj[1]) == 3
        # The named clock is the same arm, and the amortising one spreads it over `T`.
        named = PortfolioOptimisers.charge_one_time_fees(mdl, mknet(mdl), ot, 3,
                                                         FirstObservationFees())
        @test all(JuMP.coefficient(named[i], vj[1]) ==
                  JuMP.coefficient(first_obs[i], vj[1]) for i in 1:3)
        spread = PortfolioOptimisers.charge_one_time_fees(mdl, mknet(mdl), ot, 3,
                                                          AmortisedFees())
        @test all(isapprox(JuMP.coefficient(spread[i], vj[1]), i - 10 / 3) for i in 1:3)
        # Both clocks charge the same total over the horizon.
        @test isapprox(sum(JuMP.coefficient(first_obs[i], vj[1]) for i in 1:3),
                       sum(JuMP.coefficient(spread[i], vj[1]) for i in 1:3))
        # Both arms charge in place, so the caller's own vector is the one that comes back.
        # That is what lets the builder skip a second array the length of the series, and
        # it is safe because `set_net_portfolio_returns!` builds `net` and hands it straight
        # here.
        inplace = mknet(mdl)
        @test PortfolioOptimisers.charge_one_time_fees(mdl, inplace, ot, 3, nothing) ===
              inplace
        inplace = mknet(mdl)
        @test PortfolioOptimisers.charge_one_time_fees(mdl, inplace, ot, 3,
                                                       AmortisedFees()) === inplace

        # The series lands the one-off cost the same way under both spellings.
        Xf2 = [0.01 0.02 -0.01 0.03; 0.03 0.04 0.02 -0.02; -0.01 0.005 0.01 0.04]
        @test calc_net_returns(wf2, Xf2, feeF) == calc_net_returns(wf2, Xf2, fee0)
        @test calc_net_asset_returns(wf2, Xf2, feeF) ==
              calc_net_asset_returns(wf2, Xf2, fee0)

        # `override_fee_amortisation` resolves the scheme's clock against the fee's own.
        @test PortfolioOptimisers.override_fee_amortisation(fee0, nothing) === fee0
        @test isnothing(PortfolioOptimisers.override_fee_amortisation(nothing, nothing))
        @test isnothing(PortfolioOptimisers.override_fee_amortisation(nothing,
                                                                      AmortisedFees()))
        ov = PortfolioOptimisers.override_fee_amortisation(feeA, FirstObservationFees())
        @test isa(ov.fa, FirstObservationFees)
        @test ov.tn === feeA.tn
        @test ov.l == feeA.l
        @test ov.s == feeA.s
        @test ov.fl == feeA.fl
        @test ov.fs == feeA.fs
        @test ov.kwargs == feeA.kwargs
        # The override charges the fee the report states, and leaves the fee it was
        # given untouched.
        @test calc_net_returns(wf2, Xf2, ov) == calc_net_returns(wf2, Xf2, fee0)
        @test isa(feeA.fa, AmortisedFees)
        # Round trip: the other direction rebuilds the amortised clock.
        @test isa(PortfolioOptimisers.override_fee_amortisation(fee0, AmortisedFees()).fa,
                  AmortisedFees)
    end
    # Ticket #765, settled by #898: a `WeightsTracking` benchmark fee needs no fold. The
    # verb that charges it hands in the length of the series it charges, so an
    # `AmortisedFees` clock spreads the two fixed terms over that series and needs nothing
    # stamped onto it.
    @testset "a WeightsTracking fee needs no fold" begin
        Xt = [0.01 0.02 -0.01 0.03; 0.03 0.04 0.02 -0.02; -0.01 0.005 0.01 0.04]
        wbt = [0.3, 0.2, 0.4, 0.1]
        tnb = Turnover(; w = [0.25, 0.25, 0.25, 0.25], val = 0.02)
        fee_n = Fees(; tn = tnb, l = 0.001, fl = 0.5)
        fee_b = Fees(; tn = tnb, l = 0.001, fl = 0.5, fa = AmortisedFees())
        tr_n = WeightsTracking(; fees = fee_n, w = wbt)
        tr_b = WeightsTracking(; fees = fee_b, w = wbt)
        Tt = size(Xt, 1)

        bn = PortfolioOptimisers.tracking_benchmark(tr_n, Xt)
        bb = PortfolioOptimisers.tracking_benchmark(tr_b, Xt)
        oneoff = PortfolioOptimisers.calc_one_off_fees(wbt, fee_n)
        @test oneoff > zero(oneoff)

        # The two clocks charge the same total over the benchmark series.
        @test isapprox(sum(bn), sum(bb))
        # They differ on where the one-off cost lands. The `nothing` clock puts it all on
        # the first observation; the `AmortisedFees` clock puts a third of it on each.
        @test isapprox(bb[1] - bn[1], oneoff * (1 - inv(Tt)))
        @test isapprox(bb[2] - bn[2], -oneoff / Tt)
        @test isapprox(bb[3] - bn[3], -oneoff / Tt)
        # The turnover term is a rate per period, so the clock never reaches it.
        @test all(isapprox.(bb .- bb[1], 0.0; atol = 1e-15)) == false ||
              isapprox(bb[1], bb[2])

        # `factory` advances the benchmark's reference weights and leaves the clock alone.
        wpt = [0.25, 0.25, 0.3, 0.2]
        trf = factory(tr_b, wpt)
        @test trf.w == wpt
        @test trf.fees.tn.w == wbt
        @test isa(trf.fees.fa, AmortisedFees)
    end
    @testset "the JuMP fee bounds the definition from above until the decomposition is pinned" begin
        # The `Fees` docstring: without `xbgt`, `lw` and `sw` only bound the parts of `w`, and
        # the budget fills both sides, so the model charges more than `calc_fees` does. A
        # long-only model has no short side, and the two agree.
        fe = Fees(; l = 0.002, s = 0.003)
        ls = optimise(MeanRisk(;
                               opt = JuMPOptimiser(; wb = WeightBounds(; lb = -1, ub = 1),
                                                   sbgt = 0.5, bgt = 1, fees = fe, pe = pr,
                                                   slv = slv)))
        m = ls.jr.model
        @test JuMP.value(m[:fees]) / JuMP.value(m[:k]) >
              PortfolioOptimisers.calc_periodic_fees(ls.w, fe) + 1e-4
        lo = optimise(MeanRisk(;
                               opt = JuMPOptimiser(; wb = WeightBounds(; lb = 0, ub = 1),
                                                   bgt = 1, fees = fe, pe = pr, slv = slv)))
        m = lo.jr.model
        @test isapprox(JuMP.value(m[:fees]) / JuMP.value(m[:k]),
                       PortfolioOptimisers.calc_periodic_fees(lo.w, fe); rtol = 1e-6)
    end
end

# The net-returns pair of `src/14_NetReturnsDrawdowns.jl`, swept under issue #547.
@testset "Net returns" begin
    using PortfolioOptimisers, Test

    PO = PortfolioOptimisers
    wn = [0.6, -0.4, 0.0, 0.25]
    Xn = [0.01 0.02 -0.01 0.03; 0.03 0.04 0.02 -0.02; -0.01 0.005 0.01 0.04]
    fn = Fees(; l = 0.002, s = 0.003, fl = 0.01, fs = 0.02)

    @testset "the per asset returns sum to the portfolio series" begin
        # `calc_net_returns` subtracts the scalar `calc_fees`; `calc_net_asset_returns`
        # subtracts the per-asset `calc_asset_fees`. The two sides add in a different
        # order, so the identity holds to rounding and not to `==`.
        a = calc_net_returns(wn, Xn, fn)
        b = vec(sum(calc_net_asset_returns(wn, Xn, fn); dims = 2))
        @test a ≈ b
        @test maximum(abs, a - b) < 1e-16

        # and with no fee at all
        @test calc_net_returns(wn, Xn) ≈ vec(sum(calc_net_asset_returns(wn, Xn); dims = 2))
    end

    @testset "the fee is charged on the clock the fee names" begin
        # Issue #898: `calc_fees` returns the pair `(amortised, one_time)`. The first is
        # charged on every row, and the second on the first row alone, because `fn.fa` is
        # `nothing`. The two together are the number the old scalar returned.
        amortised, one_time = PO.calc_fees(wn, size(Xn, 1), fn)
        @test amortised + one_time == 0.0429
        f = fill(amortised, size(Xn, 1))
        f[1] += one_time
        @test calc_net_returns(wn, Xn, fn) ≈ Xn * wn .- f
        @test all(calc_net_returns(wn, Xn) - calc_net_returns(wn, Xn, fn) .≈ f)

        # the per asset form charges its own vectors on the same clock
        av, ov = PO.calc_asset_fees(wn, size(Xn, 1), fn)
        F = repeat(transpose(av[1]), size(Xn, 1), 1)
        F[1, :] .+= ov[1]
        @test calc_net_asset_returns(wn, Xn, fn) ≈ calc_net_asset_returns(wn, Xn) .- F
    end

    @testset "a nothing fee reaches the args... method" begin
        # It must not charge a zero fee through the `Fees` method.
        m = which(calc_net_returns, (typeof(wn), typeof(Xn), Nothing))
        @test m.file ==
              Symbol(joinpath(dirname(@__DIR__), "src", "14_NetReturnsDrawdowns.jl"))
        @test calc_net_returns(wn, Xn, nothing) == Xn * wn
        @test calc_net_asset_returns(wn, Xn, nothing) == Xn .* transpose(wn)
    end

    @testset "a vector of weight vectors gives one series each" begin
        ws = [wn, [0.25, 0.25, 0.25, 0.25]]
        r = calc_net_returns(ws, Xn)
        @test length(r) == 2
        @test r[1] ≈ Xn * ws[1]
        @test r[2] ≈ Xn * ws[2]
    end
    @testset "the weight path split sums to the drifted series (#769)" begin
        # Decision #755: the weight argument's type is the picker. A vector is one target
        # vector and a matrix is a weight path, so the split reads `X ⊙ U` and its rows
        # still sum to the portfolio series of the same path.
        wdn = SelfFinancingDrift()
        U = PO.weight_path(wdn, wn, Xn)
        @test U[1, :] == wn

        a = calc_net_returns(wn, Xn, fn, wdn)
        b = vec(sum(calc_net_asset_returns(U, Xn, fn); dims = 2))
        @test a ≈ b
        @test maximum(abs, a - b) < 1e-15
        @test calc_net_returns(wn, Xn, nothing, wdn) ≈
              vec(sum(calc_net_asset_returns(U, Xn); dims = 2))

        # The fee is charged from the path's first row, which is the target weights, so
        # the same `N × 1` vector is subtracted from every row here as there.
        av, ov = PO.calc_asset_fees(wn, size(Xn, 1), fn)
        F = repeat(transpose(av[1]), size(Xn, 1), 1)
        F[1, :] .+= ov[1]
        @test calc_net_asset_returns(U, Xn, fn) ≈ calc_net_asset_returns(U, Xn) .- F

        # The constant path is the reader-facing shape of a window that ran no drift, so
        # the `MatNum` methods reproduce the `VecNum` ones on it, exactly.
        Uc = PO.weight_path(nothing, wn, Xn)
        @test calc_net_asset_returns(Uc, Xn, fn) == calc_net_asset_returns(wn, Xn, fn)
        @test calc_net_asset_returns(Uc, Xn) == calc_net_asset_returns(wn, Xn)
        @test calc_net_asset_returns(Uc, Xn, nothing) == Xn .* transpose(wn)

        # A `nothing` fee reaches the `args...` method here too, and charges nothing.
        m = which(calc_net_asset_returns, (typeof(U), typeof(Xn), Nothing))
        @test m.file ==
              Symbol(joinpath(dirname(@__DIR__), "src", "14_NetReturnsDrawdowns.jl"))

        # A path that is not the shape of the window is a caller error the broadcast names.
        @test_throws DimensionMismatch calc_net_asset_returns(view(U, 1:2, :), Xn, fn)
    end
    @testset "the weight path is the picker for the portfolio series (#773)" begin
        # Decision #772: the weight argument's type is the picker at the base verb too, so
        # a scorer that forwards its weights here reads a path with no change of its own.
        # The method states no arithmetic of its own — it sums the split of the same path
        # along the asset axis — so the two cannot drift apart.
        wdn = SelfFinancingDrift()
        U = PO.weight_path(wdn, wn, Xn)

        @test calc_net_returns(U, Xn, fn) ==
              vec(sum(calc_net_asset_returns(U, Xn, fn); dims = 2))
        @test calc_net_returns(U, Xn) == vec(sum(calc_net_asset_returns(U, Xn); dims = 2))

        # It is the series the drift route forms from the same window under the same drift.
        # The two sides add in a different order, so this needs an absolute tolerance.
        a = calc_net_returns(wn, Xn, fn, wdn)
        b = calc_net_returns(U, Xn, fn)
        @test a ≈ b
        @test maximum(abs, a - b) < 1e-15

        # A constant path is the reader-facing shape of a window that ran no drift, so the
        # `MatNum` method reproduces the `VecNum` one on it, again to rounding.
        Uc = PO.weight_path(nothing, wn, Xn)
        c = calc_net_returns(Uc, Xn, fn)
        d = calc_net_returns(wn, Xn, fn)
        @test c ≈ d
        @test maximum(abs, c - d) < 1e-16
        @test calc_net_returns(Uc, Xn) ≈ calc_net_returns(wn, Xn)

        # A `nothing` fee reaches the `args...` method and charges nothing.
        m = which(calc_net_returns, (typeof(U), typeof(Xn), Nothing))
        @test m.file ==
              Symbol(joinpath(dirname(@__DIR__), "src", "14_NetReturnsDrawdowns.jl"))
        @test calc_net_returns(Uc, Xn, nothing) ≈ Xn * wn

        # A row count that is not the window's is a caller error the broadcast names, on
        # either side of the pair.
        @test_throws DimensionMismatch calc_net_returns(view(U, 1:2, :), Xn, fn)
        @test_throws DimensionMismatch calc_net_returns(U, view(Xn, 1:2, :), fn)
    end
end
@testset "Weight drift" begin
    using PortfolioOptimisers, Test, Dates

    PO = PortfolioOptimisers
    wd = SelfFinancingDrift()

    # Fixture 1 of research #749: three assets, three observations. Every number below
    # is a printed output of the oracle.
    R1 = [0.10 -0.04 0.02
          -0.03 0.08 0.01
          0.05 -0.02 -0.01]
    # Fixture 2 of research #749: four assets, six observations, one short weight.
    R2 = [0.012 -0.031 0.004 0.021
          -0.008 0.017 -0.022 0.005
          0.033 0.002 0.011 -0.014
          -0.019 -0.007 0.026 0.009
          0.005 0.028 -0.003 0.018
          0.024 -0.011 0.015 -0.006]

    @testset "the drift reproduces the stored oracle" begin
        # Grilling #758 fixed the tolerance at `rtol = atol = 1e-14`, the oracle's
        # own, at any panel size. It absorbs the summation-order drift a wide panel
        # carries, which a bare equality bound to a small fixture would not. Cell by cell,
        # 12 of the 14 comparisons below are exact, and the worst is maxrel 2.2e-16 (the
        # charged period), so the tolerance stays far above the measure by that decision.
        rtol = atol = 1e-14

        # Long only.
        w = [0.5, 0.3, 0.2]
        @test parity_compare(calc_net_returns(w, R1, nothing, wd),
                             [0.04200000000000004, 0.008234165067178445,
                              0.01750823354718345]; rtol, atol, name = "drift 1").ok
        @test parity_compare(PO.weight_path(wd, w, R1),
                             [0.5 0.3 0.2
                              0.527831094049904 0.27639155470249516 0.19577735124760076
                              0.5078147309105446 0.2960650307449218 0.1961202383445335];
                             rtol, atol, name = "drift 2").ok
        @test parity_compare(PO.held_weights(wd, w, R1),
                             [0.5240306170272836, 0.285151235699135, 0.19081814727358146];
                             rtol, atol, name = "drift 3").ok

        # Long short.
        w = [1.1, -0.4, 0.3]
        @test parity_compare(calc_net_returns(w, R1, nothing, wd),
                             [0.13200000000000012, -0.05650176678445251,
                              0.05981873338077248]; rtol, atol, name = "drift 4").ok
        @test parity_compare(PO.weight_path(wd, w, R1),
                             [1.1 -0.4 0.3
                              1.068904593639576 -0.33922261484098937 0.2703180212014134
                              1.0989288790681997 -0.38830006366802744 0.28937118459982775];
                             rtol, atol, name = "drift 5").ok
        @test parity_compare(PO.held_weights(wd, w, R1),
                             [1.088747808166026, -0.35905579926935327, 0.2703079911033273];
                             rtol, atol, name = "drift 6").ok

        # Partly invested. The cash position earns zero and the recursion still holds.
        w = [0.4, 0.2, 0.1]
        @test parity_compare(calc_net_returns(w, R1, nothing, wd),
                             [0.03400000000000003, 0.003075435203094834,
                              0.015583216028075997]; rtol, atol, name = "drift 7").ok
        @test parity_compare(PO.weight_path(wd, w, R1),
                             [0.4 0.2 0.1
                              0.4255319148936171 0.18568665377176016 0.09864603481624759
                              0.4115004145857036 0.19992672438728087 0.09932702134634297];
                             rtol, atol, name = "drift 8").ok
        @test parity_compare(PO.held_weights(wd, w, R1),
                             [0.4254456242441918, 0.19292184707995289, 0.09682490767960966];
                             rtol, atol, name = "drift 9").ok

        # The charged period. The oracle charges the whole cost and the whole fee on
        # every observation, so `fa` stays `nothing` here.
        w = [0.5, 0.3, 0.2]
        fees = Fees(; l = 0.001, tn = Turnover(; w = [0.4, 0.4, 0.2], val = 0.002))
        @test sum(PO.calc_fees(w, size(R1, 1), fees)) == 0.0014
        @test parity_compare(calc_net_returns(w, R1, fees, wd),
                             [0.04060000000000004, 0.006834165067178446,
                              0.016108233547183447]; rtol, atol, name = "drift 10").ok

        # The four-asset fixture, one period over the whole panel.
        w = [0.55, -0.25, 0.40, 0.15]
        @test parity_compare(calc_net_returns(w, R2, nothing, wd),
                             [0.019099999999999895, -0.01632862329506435,
                              0.019844904856505252, 0.0025246633703204235,
                              -0.0024704157155172046, 0.020955902076400745]; rtol, atol,
                             name = "drift 11").ok
        @test parity_compare(PO.weight_path(wd, w, R2),
                             [0.55 -0.25 0.4 0.15
                              0.5461681876165245 -0.23770974389166913 0.394073201844765 0.1502796585222255
                              0.5507925257828371 -0.245763793948783 0.3918011650345975 0.15353812298651465
                              0.5578972610680699 -0.2414635013265368 0.388403154208739 0.14844275687782565
                              0.5459189515276911 -0.23916943450665185 0.39749808735724307 0.1494015530612434
                              0.550007292945471 -0.24647507457054046 0.3972870572850599 0.15246743897369103];
                             rtol, atol, name = "drift 12").ok
        @test parity_compare(PO.held_weights(wd, w, R2),
                             [0.5516472032050764, -0.23876040899954856, 0.3949694226011338,
                              0.1484419004108052]; rtol, atol, name = "drift 13").ok
        @test parity_compare(calc_net_returns(w, R2, nothing, nothing),
                             [0.0191, -0.0167, 0.019950000000000002, 0.0030499999999999985,
                              -0.00275, 0.021050000000000003]; rtol, atol,
                             name = "drift 14").ok
    end

    @testset "the held weights are what a chain carries forward" begin
        # Fixture 3 of research #749, the executed-turnover oracle. The threading itself
        # is the fold loop's, so this reproduces each period's arithmetic by hand: the
        # previous weights of a period are the held weights of the period before it. The
        # tolerance is the one of #758 above. Measured cell by cell: maxrel 1.1e-16 on the
        # first chain, and every other comparison exact.
        rtol = atol = 1e-14
        targets = [[0.55, -0.25, 0.40, 0.15], [0.30, 0.30, 0.20, 0.20],
                   [0.60, -0.10, 0.35, 0.00]]
        rows = [1:2, 3:4, 5:6]
        prev = zeros(4)
        chain = Float64[]
        turnovers = Float64[]
        endings = Vector{Vector{Float64}}()
        for (t, rg) in zip(targets, rows)
            fees = Fees(; tn = Turnover(; w = prev, val = 0.001))
            append!(chain, calc_net_returns(t, R2[rg, :], fees, wd))
            push!(turnovers, sum(abs, t - prev))
            prev = PO.held_weights(wd, t, R2[rg, :])
            push!(endings, collect(prev))
        end
        @test parity_compare(chain,
                             [0.017749999999999894, -0.01767862329506435,
                              0.008865180638220317, -0.001985705588138785,
                              -0.0018861255335676944, 0.019818651026508604]; rtol, atol,
                             name = "chain 1").ok
        @test parity_compare(turnovers, [1.35, 1.0348193617797028, 1.036125533567566]; rtol,
                             atol, name = "chain 2").ok
        @test parity_compare(endings[1],
                             [0.5507925257828371, -0.245763793948783, 0.3918011650345975,
                              0.15353812298651465]; rtol, atol, name = "chain 3").ok
        @test parity_compare(endings[2],
                             [0.30131820563706624, 0.29585098098528584, 0.20561902757915068,
                              0.19721178579849719]; rtol, atol, name = "chain 4").ok
        @test parity_compare(endings[3],
                             [0.6053723917377186, -0.09967695178090742, 0.3472438694197147,
                              0.0]; rtol, atol, name = "chain 5").ok

        # With both switches off the chain keeps the numbers it has today.
        prev = zeros(4)
        chain = Float64[]
        turnovers = Float64[]
        for (t, rg) in zip(targets, rows)
            fees = Fees(; tn = Turnover(; w = prev, val = 0.001))
            append!(chain, calc_net_returns(t, R2[rg, :], fees, nothing))
            push!(turnovers, sum(abs, t - prev))
            prev = t
        end
        @test parity_compare(chain,
                             [0.017750000000000002, -0.01805, 0.00885,
                              -0.0018500000000000005, -0.0019000000000000002, 0.0197]; rtol,
                             atol, name = "chain 6").ok
        @test isapprox(turnovers, [1.35, 1.05, 1.0499999999999998]; rtol = rtol,
                       atol = atol)
    end

    @testset "the switch off reproduces the constant weight series exactly" begin
        # The only identity of the three that is bit-exact. Research #749 measured it.
        w = [0.55, -0.25, 0.40, 0.15]
        @test calc_net_returns(w, R2, nothing, nothing) == R2 * w
        @test calc_net_returns(w, R2, nothing, nothing) == calc_net_returns(w, R2)
        fees = Fees(; l = 0.002, s = 0.003)
        @test calc_net_returns(w, R2, fees, nothing) == calc_net_returns(w, R2, fees)
        ws = [w, [0.25, 0.25, 0.25, 0.25]]
        @test calc_net_returns(ws, R2, nothing, nothing) == calc_net_returns(ws, R2)
        @test calc_net_returns(ws, R2, fees, nothing) == calc_net_returns(ws, R2, fees)
    end

    @testset "two identities hold to rounding and not exactly" begin
        # Research #749 measured both. They are true of the mathematics and false of the
        # floating point, because the wealth ratio divides where the dot product does not.
        w = [0.5, 0.3, 0.2]
        one_obs = R1[1:1, :]
        @test calc_net_returns(w, one_obs, nothing, wd) != one_obs * w
        @test isapprox(calc_net_returns(w, one_obs, nothing, wd), one_obs * w; atol = 1e-14)
        @test calc_net_returns(w, one_obs, nothing, wd)[1] == 0.04200000000000004

        # A one-observation window has a path of exactly one row, the target weights,
        # and its held weights are the oracle's own one-observation ending weights.
        # Both comparisons of the oracle's outputs here measure exact, at the tolerance
        # of #758.
        @test PO.weight_path(wd, w, one_obs) == transpose(w)
        @test parity_compare(PO.held_weights(wd, w, one_obs),
                             [0.527831094049904, 0.27639155470249516, 0.19577735124760076];
                             rtol = 1e-14, atol = 1e-14, name = "window 1").ok

        single = R1[:, 1:1]
        @test calc_net_returns([1.0], single, nothing, wd) != vec(single)
        @test isapprox(calc_net_returns([1.0], single, nothing, wd), vec(single);
                       atol = 1e-14)
        @test parity_compare(calc_net_returns([1.0], single, nothing, wd),
                             [0.10000000000000009, -0.030000000000000138,
                              0.050000000000000044]; rtol = 1e-14, atol = 1e-14,
                             name = "window 2").ok
    end

    @testset "the weight path and the held weights sum to one with the cash" begin
        # The identity the oracle's own test asserts. The deflated cash is what
        # the weights leave uninvested, and it earns zero.
        w = [0.4, 0.2, 0.1]
        cash = 1 - sum(w)
        P = PO.drift_position_values(wd, w, R1)
        V = PO.drift_wealth(P, w)
        U = PO.weight_path(wd, w, R1)
        prev_wealth = vcat(1.0, V[1:(end - 1)])
        @test all(isapprox.(vec(sum(U; dims = 2)) .+ cash ./ prev_wealth, 1.0;
                            atol = 1e-14))
        @test isapprox(sum(PO.held_weights(wd, w, R1)) + cash / V[end], 1.0; atol = 1e-14)

        # The terminal wealth is the target weights grown by the whole panel, plus cash.
        @test isapprox(V[end], sum(w .* vec(prod(1 .+ R1; dims = 1))) + cash; rtol = 1e-14,
                       atol = 1e-14)
    end

    @testset "a non-positive wealth raises and forms no series" begin
        # Grilling #752 decided the raise. A 2x long book on one asset is ruined at a
        # return of -0.5, and that is the one case that makes a non-finite value.
        w = [2.0, 0.0]
        Xzero = reshape([-0.5, 0.0], 1, 2)
        @test_throws NonPositiveWealthError calc_net_returns(w, Xzero, nothing, wd)
        @test_throws NonPositiveWealthError PO.weight_path(wd, w, Xzero)
        @test_throws NonPositiveWealthError PO.held_weights(wd, w, Xzero)

        # A negative wealth is finite, so nothing downstream would read the failure.
        Xneg = [-0.6 0.0; 0.1 0.0]
        @test all(isfinite, PO.drift_wealth(PO.drift_position_values(wd, w, Xneg), w))
        @test_throws NonPositiveWealthError calc_net_returns(w, Xneg, nothing, wd)

        # A window that turns non-positive at its last observation gives no series at
        # all, because the check runs before any return is formed.
        Xlast = [0.01 0.0; -0.6 0.0]
        @test_throws NonPositiveWealthError calc_net_returns(w, Xlast, nothing, wd)

        # The message states the condition, prints the wealth, and names the observation
        # three ways: by its label, by its panel row, and by its row inside the window.
        msg = try
            calc_net_returns(w, Xneg, nothing, wd, [Date(2020, 1, 6), Date(2020, 1, 7)])
        catch e
            sprint(showerror, e)
        end
        @test occursin("all(>(0), wealth)", msg)
        @test occursin("observation 2020-01-06", msg)
        @test occursin("-0.19999999999999996", msg)
        msg = try
            calc_net_returns(w, Xneg, nothing, wd, [127, 128])
        catch e
            sprint(showerror, e)
        end
        @test occursin("panel row 127", msg)
        msg = try
            calc_net_returns(w, Xlast, nothing, wd)
        catch e
            sprint(showerror, e)
        end
        @test occursin("row 2 of the window", msg)

        # The switch off raises nothing on the same data.
        @test calc_net_returns(w, Xneg, nothing, nothing) == Xneg * w
        @test calc_net_returns(w, Xlast, nothing, nothing) == Xlast * w
        @test calc_net_returns(w, Xzero, nothing, nothing) == Xzero * w
    end

    @testset "a population drops a ruined member and raises when none survives" begin
        # Grilling #752 decided this too. A single weight vector is a population of one,
        # so it raises; a population survives its ruined members.
        X = [0.01 0.02; -0.6 0.03; 0.02 -0.01]
        pop = [[0.5, 0.5], [2.0, 0.0], [0.3, 0.7]]
        out = @test_logs (:warn, r"is not positive") calc_net_returns(pop, X, nothing, wd)
        @test length(out) == 3
        @test all(isnan, out[2])
        @test out[1] == calc_net_returns(pop[1], X, nothing, wd)
        @test out[3] == calc_net_returns(pop[3], X, nothing, wd)
        @test eltype(out) == Vector{Float64}

        # Every member ruined raises, and the message names the member.
        pop = [[2.0, 0.0], [3.0, 0.0]]
        @test_throws NonPositiveWealthError calc_net_returns(pop, X, nothing, wd)
        msg = try
            calc_net_returns(pop, X, nothing, wd)
        catch e
            sprint(showerror, e)
        end
        @test occursin("the wealth of member 1", msg)

        # No ruined member emits no warning, and each series is its own member's.
        pop = [[0.5, 0.5], [0.3, 0.7]]
        out = @test_logs min_level = Logging.Warn calc_net_returns(pop, X, nothing, wd)
        @test out[1] == calc_net_returns(pop[1], X, nothing, wd)
        @test out[2] == calc_net_returns(pop[2], X, nothing, wd)
    end
    @testset "a rebuilt population path matches the stored one, ruined members and all (#769)" begin
        # `rebuild_weight_path`'s own docstring promises a rebuild that is bit-identical to
        # the store. It was not, for a population carrying a ruined member: the store fills
        # such a member with `NaN`, and the rebuild ran the drift again and raised on it.
        X = [0.01 0.02; -0.6 0.03; 0.02 -0.01]
        pop = [[0.5, 0.5], [2.0, 0.0], [0.3, 0.7]]
        stored, ruined = PO.held_weights_result(wd, pop, X, true)
        lazy, _ = PO.held_weights_result(wd, pop, X, false)
        @test ruined == [2]
        @test isnothing(lazy.U)

        Us = PO.weight_path(stored, pop)
        Ul = PO.weight_path(lazy, pop)
        @test length(Ul) == 3
        @test all(isequal(a, b) for (a, b) in zip(Us, Ul))
        @test all(isnan, Ul[2])
        @test Ul[1] == PO.weight_path(wd, pop[1], X)
        @test Ul[3] == PO.weight_path(wd, pop[3], X)
    end
end

@testset "Oracle fee parity" begin
    using PortfolioOptimisers, Test, LinearAlgebra

    # Every expected number in this file was measured by RUNNING the oracle, not by reading
    # it. The fixture below is stated as literals on both sides, so the two languages parse
    # identical `Float64` bits and no data file is exchanged.
    #
    # The oracle's cost model, which this file pins:
    #
    #   total_cost = sum(transaction_costs .* abs.(previous_weights - weights)) + liquidation
    #   total_fee  = sum(management_fees .* weights)
    #   returns    = X * weights .- total_cost .- total_fee
    #
    # The whole cost is subtracted from **every** observation. That is the same clock this
    # library puts `l`, `s` and `tn` on, so the two agree term for term:
    #
    #   | the oracle          | this library                        |
    #   | ------------------- | ----------------------------------- |
    #   | `transaction_costs` | `tn`, a `Turnover` carrier          |
    #   | `management_fees`   | `l` on a long book, `s` on a short  |
    #   | `liquidation_cost`  | the `lq` carrier of ADR 0121        |
    #
    # The oracle has no fixed fee, so `fl`, `fs` and the `flq` carrier of ADR 0121 have
    # no counterpart there and are pinned by the hand oracles of `test_11_fees_and_returns`.
    #
    # Two conventions differ:
    #
    #  1. The oracle charges its proportional fee as `management_fees .* weights` over
    #     every asset, so a short position earns a **credit**. This library splits the term
    #     into `l` over `w .>= 0` and `s` over `w .< 0`, and negates the short half. A
    #     non-negative `s` makes it a charge, and `s = -l` makes it the same credit (#1518).
    #     The testset on a short book pins that case against the oracle.
    #  2. The oracle stores `compounded` as an attribute of its portfolio object. Here it
    #     is the positional `compound` argument of `cumulative_returns` and `drawdowns`.
    #     Same switch, same default, different carrier.
    #
    # Tolerances. Every value below was measured **bit-for-bit equal** to the oracle
    # except the three noted at their assertions, which are sums that cancel to within a few
    # ulp of zero and so carry an absolute error near `1e-16` with no meaningful relative
    # one. `atol` guards those, and guards the row sums against a BLAS that reassociates.
    #
    # **Every weight vector here is stated, never solved.** That is what buys the tolerance
    # above: the file compares fee arithmetic on identical inputs, so no solver enters it.
    # A test that first optimises cannot hold this tolerance, because the oracle reaches
    # its weights through a different build of the solver. Extend this file that way only
    # with a tolerance sized to the weights, and note that `max_step_fraction = 0.75` brings
    # this library's solver settings close to the oracle's defaults.
    # The arithmetic below measured maxabs 3.5e-18 at most against the oracle's literals.
    atol = 1e-15

    X = [0.010 -0.020 0.005 0.030
         -0.015 0.025 -0.010 0.012
         0.020 0.010 0.015 -0.008
         -0.005 -0.030 0.020 0.018
         0.008 0.014 -0.025 0.006]
    # Asset names, in this order, are "a", "b", "c", "d" on the oracle side.
    tc = [0.001, 0.002, 0.010, 0.003]
    mf = [0.0005, 0.0004, 0.0003, 0.0002]
    prev = fill(0.25, 4)

    @testset "The turnover charge on a long-short book" begin
        w = [0.4, -0.3, 0.5, 0.4]
        fees = Fees(; tn = Turnover(; w = prev, val = tc))
        # The oracle reported `total_cost = 0.0042`.
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, fees), 0.0042; atol = atol)
        # `tn` is a per period rate, so the charge lands on every observation, which is what
        # the oracle does with its `total_cost`.
        @test isapprox(PortfolioOptimisers.calc_one_off_fees(w, fees), 0.0; atol = atol)
        @test parity_compare(calc_net_returns(w, X, fees),
                             [0.020300000000000002, -0.0179, 0.0050999999999999995, 0.02,
                              -0.015299999999999998]; rtol = 0.0, atol = atol,
                             name = "long-short net").ok
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
        # The oracle reported `total_cost = 0.0011` and `total_fee = 0.00036`.
        @test isapprox(calc_fees(w, Turnover(; w = prev, val = tc)), 0.0010999999999999998;
                       atol = atol)
        @test isapprox(calc_fees(w, mf, .>=), 0.00036; atol = atol)
        # The book is long, so `s` is unused and the two conventions coincide.
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, fees),
                       0.0010999999999999998 + 0.00036; atol = atol)
        # The last entry is the one that cancels: the oracle reported
        # `-6.000000000000097e-5` where this library reports `-6.0000000000001025e-5`.
        @test parity_compare(calc_net_returns(w, X, fees),
                             [0.01054, -0.0033599999999999997, 0.00814,
                              0.0029399999999999995, -6.000000000000097e-5]; rtol = 0.0,
                             atol = atol, name = "long-short mf net").ok
    end

    @testset "A short book earns the linear fee's credit under `s = -l`" begin
        # #1518, row R101 of #1416. The proportional fee `l max(w, 0) - s min(w, 0)` is
        # convex if and only if `s >= -l`, and `s = -l` is the linear `l w` the oracle
        # charges. Every number below was measured by running the oracle on these literals.
        w = [0.4, -0.3, 0.5, 0.4]
        fees = Fees(; l = mf, s = -mf)
        # The oracle reported `total_fee = 0.00031`.
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, fees), 0.00031;
                       atol = atol)
        @test parity_compare(calc_net_returns(w, X, fees),
                             [0.02419, -0.014009999999999998, 0.00899, 0.023889999999999998,
                              -0.01141]; rtol = 0.0, atol = atol, name = "short fee net").ok
        A = calc_net_asset_returns(w, X, fees)
        @test isapprox(vec(sum(A; dims = 2)), calc_net_returns(w, X, fees); atol = atol)
        # With its transaction cost too, the oracle reported `total_cost = 0.0042`.
        fees_tn = Fees(; tn = Turnover(; w = prev, val = tc), l = mf, s = -mf)
        @test parity_compare(calc_net_returns(w, X, fees_tn),
                             [0.01999, -0.01821, 0.004789999999999999, 0.01969, -0.01561];
                             rtol = 0.0, atol = atol, name = "short fee tn net").ok
        # The book of the ticket: `w = (1.5, -0.5)` and `f = 0.01`. The oracle reported
        # `total_fee = 0.009999999999999998`. The long side pays 0.015 and the short earns
        # 0.005.
        @test isapprox(PortfolioOptimisers.calc_periodic_fees([1.5, -0.5],
                                                              Fees(; l = 0.01, s = -0.01)),
                       0.009999999999999998; atol = atol)
        # `s = l` is the carry reading, which charges both sides.
        @test isapprox(PortfolioOptimisers.calc_periodic_fees([1.5, -0.5],
                                                              Fees(; l = 0.01, s = 0.01)),
                       0.02; atol = atol)
        # Below `-l` the fee is not convex, so the constructor refuses it, entry by entry.
        # A `nothing` `l` reads as zero.
        @test_throws DomainError Fees(; l = 0.01, s = -0.02)
        @test_throws DomainError Fees(; s = -0.01)
        @test_throws DomainError Fees(; l = [0.01, 0.02], s = [-0.01, -0.03])
        @test Fees(; l = [0.01, 0.02], s = -0.01).s == -0.01
        # The estimator resolves `s` by asset name, so it accepts a credit and `Fees` checks
        # the bound after the resolution.
        @test FeesEstimator(; l = 0.01, s = -0.01, ds = -0.01).s == -0.01
        @test_throws DomainError FeesEstimator(; s = Inf)
    end

    @testset "The cumulative summaries under both settings of `compound`" begin
        w = [0.4, 0.1, 0.2, 0.3]
        fees = Fees(; tn = Turnover(; w = prev, val = tc), l = mf)
        r = calc_net_returns(w, X, fees)
        # `compound = false` is the oracle's `compounded=False`, and `true` its `True`.
        # The compounded pair matched bit-for-bit; the simple pair differs in the last ulp
        # of its final entry, which is a sum that cancels.
        @test parity_compare(cumulative_returns(r),
                             [0.01054, 0.007180000000000001, 0.01532, 0.01826,
                              0.018199999999999997]; rtol = 0.0, atol = atol,
                             name = "cumulative").ok
        @test cumulative_returns(r, true) ==
              [1.01054, 1.0071445855999999, 1.0153427425267838, 1.0183278501898125,
               1.0182667505188012]
        @test parity_compare(drawdowns(r),
                             [0.0, -0.00336, 0.0, 0.0, -6.0000000000001025e-5]; rtol = 0.0,
                             atol = atol, name = "drawdowns").ok
        @test drawdowns(r, true) ==
              [0.0, -0.0033600000000001407, 0.0, 0.0, -5.999999999994898e-5]
    end

    @testset "A forced liquidation is charged through a reduced `Fees`" begin
        # ADR 0121 charges an asset that leaves the Investable Mask through a `Turnover`
        # carrier on the complement of the mask. This testset walks the path a caller
        # actually takes: a reduced `Fees` carrying `lq`, read by the ordinary fee verbs.
        # The oracle meets the same charge the same way — it holds a named
        # `previous_weights` that includes an asset absent from `X`, and reports the cost
        # through the portfolio's `total_cost`.
        #
        # Asset "c" leaves a four-asset universe holding a quarter of the book in each.
        # `w` is the surviving book. The liquidation term does not read it, because the
        # exiting asset sits on the other axis, and that independence is itself contract.
        w = [0.4, -0.3, 0.9]

        # A per asset rate. The oracle reported turnover `0.25` and cost `0.0025`.
        lq = Turnover(; w = [0.25], val = [0.010])
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, Fees(; lq = lq)), 0.0025;
                       atol = atol)
        @test isapprox(sum(abs, lq.w), 0.25; atol = atol)
        # `lq` is a rate per period, so it never lands in the one-off half.
        @test isapprox(PortfolioOptimisers.calc_one_off_fees(w, Fees(; lq = lq)), 0.0;
                       atol = atol)

        # One scalar rate over every exit. The oracle reported cost `0.001`.
        sc = Fees(; lq = Turnover(; w = [0.25], val = 0.004))
        @test isapprox(PortfolioOptimisers.calc_periodic_fees(w, sc), 0.001; atol = atol)

        # Two exits of opposite sign: "b" short at `-0.4` and "c" long at `0.25`. The
        # oracle reported turnover `0.65` and cost `0.0033`, so the charge reads the
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
        # This is the oracle's own reduced portfolio: it was handed `X` for "a", "b" and
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
        @test isapprox(PortfolioOptimisers.calc_liquidation_fees(fees.lq), 0.0025;
                       atol = atol)

        # The whole charge lands on every observation, which is the oracle's series.
        @test parity_compare(calc_net_returns(wred, Xred, fees),
                             [0.031299999999999994, -0.0084, -0.0079, 0.017499999999999998,
                              -0.001299999999999999]; rtol = 0.0, atol = atol,
                             name = "liquidation net").ok

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
        # `1e-15` tolerance. This one solves, so it cannot: the oracle reaches its weights
        # through a different build of the solver. `max_step_fraction = 0.75` brings this
        # library's Clarabel close to the one the oracle drives through its modelling
        # layer, and the residual gap measured `1.5e-4` on a weight and `2.9e-5` on a summed
        # return series, so the tolerances below are sized to that and not to the arithmetic.
        w_atol = 1e-3
        s_atol = 2e-4

        # A deterministic panel from an integer LCG, so both languages hold identical bits
        # with no data file between them. Every element matched the oracle exactly.
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
        # which is the oracle's `train_size = 60, test_size = 20`. `expand_train` is
        # `false` by default on both sides, so the window rolls rather than expands.
        #
        # **`wd` alone reproduces the oracle's drift (#1518, row R102 of #1416).** The oracle
        # carries one flag: with it set, the series drifts *and* the next fold budgets its
        # turnover against the drifted ending weights. Here those are two switches, `wd` and
        # `pws`, and an unset `pws` follows `wd`, so `wd` alone gives the same consistent
        # pair. Before #1518 an unset `pws` meant the targets, and `wd` alone left the
        # turnover cases `3.5e-3` from the oracle. `pws = TargetWeights()` keeps that mixed
        # pair one keyword away.
        flat = () -> IndexWalkForward(60, 20)
        drift = () -> IndexWalkForward(60, 20; wd = SelfFinancingDrift())
        mixed = () -> IndexWalkForward(60, 20; wd = SelfFinancingDrift(),
                                       pws = TargetWeights())

        # Every proportional term this library carries, against its counterpart in the
        # oracle: `l` is the oracle's per asset holding fee, and `tn` its transaction
        # cost. `s` has no counterpart, because the oracle credits a short holding fee
        # where this library charges it, so the books below are long only. The two
        # liquidation carriers of ADR 0121 are pinned by the two testsets above, which is as
        # far as they can be taken until issue #897 puts them on `Fees`.
        fee_cases = ["nofee" => nothing, "mgmt" => Fees(; l = mgmt),
                     "tn" => Fees(; tn = Turnover(; w = z5, val = tccv)),
                     "both" => Fees(; l = mgmt, tn = Turnover(; w = z5, val = tccv))]

        # Measured from the oracle, per case: the summed net return series, then the last
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

        # The oracle solved the same weights in all four fee cases, because its default
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
            @test parity_compare(pred.pred[1].res.w, ref_w0; rtol = 0.0, atol = w_atol,
                                 name = "cv w0 $(fl) $(dl)").ok
            @test parity_compare(pred.pred[2].res.w, ref_w1; rtol = 0.0, atol = w_atol,
                                 name = "cv w1 $(fl) $(dl)").ok
            @test parity_compare(pred.pred[3].res.w, ref_w2; rtol = 0.0, atol = w_atol,
                                 name = "cv w2 $(fl) $(dl)").ok

            @test isapprox(sum(r), ret_sum; atol = s_atol)
            # `compound = false` is the oracle's `compounded=False`, `true` its `True`.
            @test isapprox(cumulative_returns(r)[end], smp_last; atol = s_atol)
            @test isapprox(cumulative_returns(r, true)[end], cmp_last; atol = s_atol)
        end

        # The resolution of the two switches, on each timeline scheme's `fold_evaluation`.
        resolve = PortfolioOptimisers.resolve_previous_weights_source
        sfd = SelfFinancingDrift()
        @test isnothing(resolve(nothing, nothing))
        @test resolve(nothing, sfd) isa DriftedWeights
        @test isnothing(resolve(TargetWeights(), sfd))
        @test isnothing(resolve(TargetWeights(), nothing))
        @test resolve(DriftedWeights(), nothing) isa DriftedWeights
        @test PortfolioOptimisers.fold_evaluation(drift()).pws isa DriftedWeights
        @test isnothing(PortfolioOptimisers.fold_evaluation(mixed()).pws)
        @test isnothing(PortfolioOptimisers.fold_evaluation(flat()).pws)
        @test PortfolioOptimisers.fold_evaluation(OnlineIndexWalkForward(60, 20; wd = sfd)).pws isa
              DriftedWeights
        @test PortfolioOptimisers.fold_evaluation(HindsightSplit(; wd = sfd)).pws isa
              DriftedWeights

        # An explicit source overrides the rule. With the drifted source stated, the run is
        # the run of `wd` alone, bit for bit. With the targets stated, a case that reads
        # previous weights keeps the mixed pair, `3.5e-3` from the oracle, and a case that
        # reads none does not move.
        mr_tn = MeanRisk(;
                         opt = JuMPOptimiser(; wb = WeightBounds(; lb = 0, ub = 1), bgt = 1,
                                             fees = Fees(;
                                                         tn = Turnover(; w = z5,
                                                                       val = tccv)),
                                             slv = slv))
        r_drift = cross_val_predict(mr_tn, rdcv, drift()).mrd.X
        r_both = cross_val_predict(mr_tn, rdcv,
                                   IndexWalkForward(60, 20; wd = sfd,
                                                    pws = DriftedWeights())).mrd.X
        r_mixed = cross_val_predict(mr_tn, rdcv, mixed()).mrd.X
        @test r_both == r_drift
        @test abs(sum(r_mixed) - expected["tn_drift"][1]) > 1e-3
        mr_l = MeanRisk(;
                        opt = JuMPOptimiser(; wb = WeightBounds(; lb = 0, ub = 1), bgt = 1,
                                            fees = Fees(; l = mgmt), slv = slv))
        @test cross_val_predict(mr_l, rdcv, mixed()).mrd.X ==
              cross_val_predict(mr_l, rdcv, drift()).mrd.X
    end

    @testset "A walk-forward over a delisting, with a liquidation carrier" begin
        using StableRNGs, LinearAlgebra, Clarabel

        # The test that verifies the port end to end. An asset delists inside the last test
        # fold, so the mask derives itself from the data, the fold that loses it charges a
        # forced exit, and the folds that lose nothing charge none.
        #
        # **What can and cannot be compared.** The oracle reaches a delisting only
        # through its exponentially weighted moments with `active_mask` routing, because its
        # plain prior refuses a `NaN`. This library's plain prior handles the gap natively.
        # The two therefore fit different moments and solve to different weights, so the
        # series cannot be compared. What can be compared exactly is the **charge**, which
        # is arithmetic on the weights: the first half below feeds this library's fee verbs
        # the oracle's own per fold weights and matches its reported cost to rounding.
        # The second half then drives this library's whole pipeline and pins the invariants
        # the oracle cannot speak to.

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
            # The matrix the port is verified by: every fee the oracle supports, against
            # both weight-drift settings, summarised under both settings of `compound` —
            # all of it over a panel where an asset delists, so every cell charges a forced
            # exit at the fold that loses it.
            #
            # The comparison is driven from the **oracle's own per fold weights**. That
            # is not a shortcut, it is the only way the cells are comparable: the oracle
            # reaches a delisting solely through its exponentially weighted moments with
            # `active_mask` routing, because its plain prior refuses a `NaN`, while this
            # library's plain prior handles the gap natively. The two therefore fit
            # different moments and solve to different weights. Holding the weights fixed
            # removes that difference and leaves exactly what is being verified: the fee
            # arithmetic, the clock, the drift and the two cumulative conventions.
            #
            # The oracle solved the same weights in all six cells, because its default
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

            # Per fold `total_cost` and `total_fee` as the oracle reported them.
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
                    # numbers the oracle reported for it.
                    # Measured maxrel 1.7e-16, one rounding of a sum over five assets.
                    @test parity_compare([PortfolioOptimisers.calc_periodic_fees(wf, fe)],
                                         [cost[tag][f] + fee[tag][f]]; rtol = 1e-15,
                                         name = "$(tag) fold $(f) charge").ok

                    # Only the fold that loses an asset owes an exit, and it owes the rate
                    # times the previous weight the scheme threaded.
                    if f == 3 && use_tn
                        @test isapprox(PortfolioOptimisers.calc_liquidation_fees(fe.lq),
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
                # Both cumulative conventions, against the oracle's own summaries.
                # Measured maxrel 1.6e-15 on the simple return, a sum of 60 returns that
                # cancels to a total near 3e-3 (`mgmt_drift`), and 1.3e-16 on the compounded
                # one, a product of 60 factors near one.
                @test parity_compare([cumulative_returns(r)[end]], [smp[tag]]; rtol = 1e-14,
                                     name = "$(tag) simple").ok
                @test parity_compare([cumulative_returns(r, true)[end]], [cmp[tag]];
                                     rtol = 1e-15, name = "$(tag) compounded").ok
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
                    @test isapprox(PortfolioOptimisers.calc_liquidation_fees(pred.pred[i].res.fees.lq),
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
                # is the oracle's arithmetic on this library's own weights.
                @test isapprox(PortfolioOptimisers.calc_liquidation_fees(exit_res.fees.lq),
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

    @testset "The drawdown peak includes the starting capital" begin
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
        # The oracle seeded its peak at the first observation at the tag this file was
        # measured at, so it understated this case by `1.1e-2`. At the tag on disk it seeds
        # the peak at the starting capital too, and it gives `-0.10915360082646582` on the
        # series below (#1416, row R73). The rule is now at parity.

        # The first fold of the oracle's own transaction-cost walk-forward. Its cost drags
        # the curve under water on the first observation and it never recovers.
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

        # The two coincide once the series rises above where it opened.
        up = [0.01054, -0.00336, 0.00814]
        cup = cumulative_returns(up)
        @test isapprox(drawdowns(up), cup .- accumulate(max, cup); atol = atol)
    end
end

#=
Check `calc_turnover` in `src/14_NetReturnsDrawdowns.jl`. Issue #937.

The verb is the library's first value-level turnover. `plot_turnover`
(`ext/PortfolioOptimisersPlotsExt.jl`) computed the same sum inline, which made it the one
place in the extension that computed a statistic rather than drawing one, and
`forecast_portfolio` needed the same number. The inline expression is kept below as the
behavioural reference, so the refactor is pinned to what the plot drew before it.

TWO CONVENTIONS SHAPE THE PROBES.

1. THE FIRST OBSERVATION IS `NaN`, NOT ZERO. A path that opens a book from cash trades its
   whole gross, and one that inherits a book trades nothing; the series cannot tell them
   apart, so it declines to answer. The answer therefore has one entry per observation of
   the path and the caller drops the leading gap, which is what `plot_turnover` does.

2. THE TWO METHODS ARE THE SAME NUMBER ON THE TWO SHAPES. A matrix carries one observation
   per row and a vector of vectors carries one per element, and the library holds a weight
   path in both shapes, so the two are asserted against each other rather than each against
   its own expectation.
=#
@testset "Value-level turnover" begin
    using Test, PortfolioOptimisers, StableRNGs

    PO = PortfolioOptimisers

    # The expression `plot_turnover` carried inline before #937, kept as the reference.
    ref_turnover(w) = [sum(abs, w[t] .- w[t - 1]) for t in 2:length(w)]

    @testset "The sum is the one-way absolute weight change" begin
        w = [0.5 0.5; 0.25 0.75; 0.25 0.75; -0.5 1.5]
        tn = calc_turnover(w)
        @test length(tn) == size(w, 1)
        @test isnan(tn[1])
        @test tn[2] ≈ 0.5
        @test tn[3] == 0
        @test tn[4] ≈ 1.5
    end

    @testset "A single observation carries only the leading gap" begin
        @test isequal(calc_turnover([0.4 0.6]), [NaN])
        @test isequal(calc_turnover([[0.4, 0.6]]), [NaN])
    end

    @testset "The matrix and the vector-of-vectors methods agree" begin
        rng = StableRNG(987654321)
        K, N = 7, 5
        w_series = [randn(rng, N) for _ in 1:K]
        W = permutedims(reduce(hcat, w_series))
        tn_v = calc_turnover(w_series)
        tn_m = calc_turnover(W)
        @test isequal(tn_v, tn_m)
        @test isnan(tn_v[1])
        @test tn_v[2:end] ≈ ref_turnover(w_series)
    end

    @testset "The bound is `<:Number`, so a complex path answers a real turnover" begin
        # `abs` of a complex weight is real, so nothing in the body orders the data and the
        # bound is wider than the `<:Real` the numeric kernels of the library carry.
        w = [1.0 + 0.0im 0.0 + 1.0im; 0.0 + 0.0im 0.0 + 0.0im]
        tn = calc_turnover(w)
        @test eltype(tn) === Float64
        @test tn[2] ≈ 2
    end

    @testset "A non-finite weight poisons its own observation and the next" begin
        # No guard: the verb reports the sum it computed. A `NaN` weight makes the two
        # observations that read it non-finite and leaves every other one alone.
        w = [0.5 0.5; NaN 0.5; 0.25 0.75; 0.25 0.75]
        tn = calc_turnover(w)
        @test isnan(tn[1])
        @test isnan(tn[2])
        @test isnan(tn[3])
        @test tn[4] == 0
    end
end
