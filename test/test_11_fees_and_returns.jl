using JuMP: JuMP
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
    # Issue #545, condition 2: the reference-weight vocabulary of `src/15_Turnover.jl`,
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
    # Issue #546, condition 2: the fee arithmetic of `src/16_Fees.jl`, checked with
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
end

# The net-returns pair of `src/17_NetReturnsDrawdowns.jl`, swept under issue #547.
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
              Symbol(joinpath(dirname(@__DIR__), "src", "17_NetReturnsDrawdowns.jl"))
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
              Symbol(joinpath(dirname(@__DIR__), "src", "17_NetReturnsDrawdowns.jl"))

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
              Symbol(joinpath(dirname(@__DIR__), "src", "17_NetReturnsDrawdowns.jl"))
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
    # is a printed output of the reference implementation.
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

    @testset "the drift reproduces the reference implementation" begin
        # Grilling #758 fixed the tolerance at `rtol = atol = 1e-14`, the reference's
        # own, at any panel size. It absorbs the summation-order drift a wide panel
        # carries, which a bare equality bound to a small fixture would not.
        rtol = atol = 1e-14

        # Long only.
        w = [0.5, 0.3, 0.2]
        @test isapprox(calc_net_returns(w, R1, nothing, wd),
                       [0.04200000000000004, 0.008234165067178445, 0.01750823354718345];
                       rtol = rtol, atol = atol)
        @test isapprox(PO.weight_path(wd, w, R1),
                       [0.5 0.3 0.2
                        0.527831094049904 0.27639155470249516 0.19577735124760076
                        0.5078147309105446 0.2960650307449218 0.1961202383445335];
                       rtol = rtol, atol = atol)
        @test isapprox(PO.held_weights(wd, w, R1),
                       [0.5240306170272836, 0.285151235699135, 0.19081814727358146];
                       rtol = rtol, atol = atol)

        # Long short.
        w = [1.1, -0.4, 0.3]
        @test isapprox(calc_net_returns(w, R1, nothing, wd),
                       [0.13200000000000012, -0.05650176678445251, 0.05981873338077248];
                       rtol = rtol, atol = atol)
        @test isapprox(PO.weight_path(wd, w, R1),
                       [1.1 -0.4 0.3
                        1.068904593639576 -0.33922261484098937 0.2703180212014134
                        1.0989288790681997 -0.38830006366802744 0.28937118459982775];
                       rtol = rtol, atol = atol)
        @test isapprox(PO.held_weights(wd, w, R1),
                       [1.088747808166026, -0.35905579926935327, 0.2703079911033273];
                       rtol = rtol, atol = atol)

        # Partly invested. The cash position earns zero and the recursion still holds.
        w = [0.4, 0.2, 0.1]
        @test isapprox(calc_net_returns(w, R1, nothing, wd),
                       [0.03400000000000003, 0.003075435203094834, 0.015583216028075997];
                       rtol = rtol, atol = atol)
        @test isapprox(PO.weight_path(wd, w, R1),
                       [0.4 0.2 0.1
                        0.4255319148936171 0.18568665377176016 0.09864603481624759
                        0.4115004145857036 0.19992672438728087 0.09932702134634297];
                       rtol = rtol, atol = atol)
        @test isapprox(PO.held_weights(wd, w, R1),
                       [0.4254456242441918, 0.19292184707995289, 0.09682490767960966];
                       rtol = rtol, atol = atol)

        # The charged period. The reference charges the whole cost and the whole fee on
        # every observation, so `fa` stays `nothing` here.
        w = [0.5, 0.3, 0.2]
        fees = Fees(; l = 0.001, tn = Turnover(; w = [0.4, 0.4, 0.2], val = 0.002))
        @test sum(PO.calc_fees(w, size(R1, 1), fees)) == 0.0014
        @test isapprox(calc_net_returns(w, R1, fees, wd),
                       [0.04060000000000004, 0.006834165067178446, 0.016108233547183447];
                       rtol = rtol, atol = atol)

        # The four-asset fixture, one period over the whole panel.
        w = [0.55, -0.25, 0.40, 0.15]
        @test isapprox(calc_net_returns(w, R2, nothing, wd),
                       [0.019099999999999895, -0.01632862329506435, 0.019844904856505252,
                        0.0025246633703204235, -0.0024704157155172046,
                        0.020955902076400745]; rtol = rtol, atol = atol)
        @test isapprox(PO.weight_path(wd, w, R2),
                       [0.55 -0.25 0.4 0.15
                        0.5461681876165245 -0.23770974389166913 0.394073201844765 0.1502796585222255
                        0.5507925257828371 -0.245763793948783 0.3918011650345975 0.15353812298651465
                        0.5578972610680699 -0.2414635013265368 0.388403154208739 0.14844275687782565
                        0.5459189515276911 -0.23916943450665185 0.39749808735724307 0.1494015530612434
                        0.550007292945471 -0.24647507457054046 0.3972870572850599 0.15246743897369103];
                       rtol = rtol, atol = atol)
        @test isapprox(PO.held_weights(wd, w, R2),
                       [0.5516472032050764, -0.23876040899954856, 0.3949694226011338,
                        0.1484419004108052]; rtol = rtol, atol = atol)
        @test isapprox(calc_net_returns(w, R2, nothing, nothing),
                       [0.0191, -0.0167, 0.019950000000000002, 0.0030499999999999985,
                        -0.00275, 0.021050000000000003]; rtol = rtol, atol = atol)
    end

    @testset "the held weights are what a chain carries forward" begin
        # Fixture 3 of research #749, the executed-turnover oracle. The threading itself
        # is the fold loop's, so this reproduces each period's arithmetic by hand: the
        # previous weights of a period are the held weights of the period before it.
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
        @test isapprox(chain,
                       [0.017749999999999894, -0.01767862329506435, 0.008865180638220317,
                        -0.001985705588138785, -0.0018861255335676944,
                        0.019818651026508604]; rtol = rtol, atol = atol)
        @test isapprox(turnovers, [1.35, 1.0348193617797028, 1.036125533567566];
                       rtol = rtol, atol = atol)
        @test isapprox(endings[1],
                       [0.5507925257828371, -0.245763793948783, 0.3918011650345975,
                        0.15353812298651465]; rtol = rtol, atol = atol)
        @test isapprox(endings[2],
                       [0.30131820563706624, 0.29585098098528584, 0.20561902757915068,
                        0.19721178579849719]; rtol = rtol, atol = atol)
        @test isapprox(endings[3],
                       [0.6053723917377186, -0.09967695178090742, 0.3472438694197147, 0.0];
                       rtol = rtol, atol = atol)

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
        @test isapprox(chain,
                       [0.017750000000000002, -0.01805, 0.00885, -0.0018500000000000005,
                        -0.0019000000000000002, 0.0197]; rtol = rtol, atol = atol)
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
        # and its held weights are the reference's own one-observation ending weights.
        @test PO.weight_path(wd, w, one_obs) == transpose(w)
        @test isapprox(PO.held_weights(wd, w, one_obs),
                       [0.527831094049904, 0.27639155470249516, 0.19577735124760076];
                       rtol = 1e-14, atol = 1e-14)

        single = R1[:, 1:1]
        @test calc_net_returns([1.0], single, nothing, wd) != vec(single)
        @test isapprox(calc_net_returns([1.0], single, nothing, wd), vec(single);
                       atol = 1e-14)
        @test isapprox(calc_net_returns([1.0], single, nothing, wd),
                       [0.10000000000000009, -0.030000000000000138, 0.050000000000000044];
                       rtol = 1e-14, atol = 1e-14)
    end

    @testset "the weight path and the held weights sum to one with the cash" begin
        # The identity the reference's own oracle test asserts. The deflated cash is what
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
