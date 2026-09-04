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
    @testset "Fees" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/Fees.csv.gz"), DataFrame)
        f1s = [0.02002313426946848, 0.12149580659357644]
        for (i, fe) in pairs(fes)
            res_mip = optimise(da,
                               FiniteAllocationInput(; w = res.w,
                                                     prices = vec(values(X[end])),
                                                     cash = 1000, horizon = T, fees = fe))
            f1 = calc_fees(res.w, fe)
            @test isapprox(f1s[i], f1)
            f2 = calc_asset_fees(res.w, fe)
            @test isapprox(df[!, "$(2*(i-1)+1)"], f2)
            f3 = calc_asset_fees(res.w, vec(values(X[end])), fe)
            @test isapprox(df[!, "$(2*(i-1)+2)"], f3)
            fopt1 = calc_fees(res.w, vec(values(X[end])), fe) * T
            fopt2 = 1000 - (sum(res_mip.cost) + res_mip.cash)
            result = isapprox(fopt1, fopt2)
            if !result
                fopt2_t = if i == 1
                    67.80797690253598
                elseif i == 2
                    138.48615533295037
                else
                    fopt1
                end
                result = isapprox(fopt2, fopt2_t; rtol = 1e-6)
                if !result
                    println("Counter: $i")
                    println("fopt1: $fopt1")
                    println("fopt2: $fopt2")
                    findtol(fopt1, fopt2)
                end
                @test result
            else
                @test result
            end
            @test all(isapprox(calc_net_returns(res.w, pr.X) .- calc_fees(res.w, fe),
                               calc_net_returns(res.w, pr.X, fe)))
            @test all(isapprox(calc_net_asset_returns(res.w, pr.X) .-
                               transpose(calc_asset_fees(res.w, fes[1])),
                               calc_net_asset_returns(res.w, pr.X, fes[1])))
        end
        @test iszero(calc_fees(res.w, Fees()))
        @test all(iszero, calc_asset_fees(res.w, Fees()))
        @test iszero(calc_fees(res.w, vec(values(X[end])), Fees()))
        @test all(iszero, calc_asset_fees(res.w, vec(values(X[end])), Fees()))
    end
    @testset "Expected Returns" begin
        r = factory(Variance(), pr, slv)
        f = calc_fees(res.w, fes[1])
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

        @test isapprox(expected_risk(ExpectedReturn(), res.w, pr, fes[1]),
                       rt - calc_fees(res.w, fes[1]))
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
        pf = [100.0, 50.0, 20.0, 10.0]
        tnf = Turnover(; w = [0.1, 0.2, 0.3, 0.4], val = [0.01, 0.02, 0.03, 0.04])
        fev = Fees(; tn = tnf, l = [0.001, 0.002, 0.003, 0.004],
                   s = [0.005, 0.006, 0.007, 0.008], fl = [1.0, 2.0, 3.0, 4.0],
                   fs = [5.0, 6.0, 7.0, 8.0])
        tns = Turnover(; w = [0.1, 0.2, 0.3, 0.4], val = 0.02)
        fesc = Fees(; tn = tns, l = 0.001, s = 0.005, fl = 1.0, fs = 5.0)

        # The per-asset fee sums to the portfolio fee, up to the order of summation.
        @test isapprox(sum(calc_asset_fees(wf, pf, fev)), calc_fees(wf, pf, fev))
        @test isapprox(sum(calc_asset_fees(wf, fev)), calc_fees(wf, fev))
        @test isapprox(sum(calc_asset_fees(wf, pf, fesc)), calc_fees(wf, pf, fesc))
        @test isapprox(sum(calc_asset_fees(wf, fesc)), calc_fees(wf, fesc))

        # The short proportional term is a positive charge, not a credit.
        @test calc_fees([0.6, -0.4], Fees(; s = 0.01)) == 0.004
        @test calc_asset_fees([0.6, -0.4], Fees(; s = 0.01)) == [0.0, 0.004]
        @test calc_fees([0.6, -0.4], [100.0, 50.0], Fees(; s = 0.01)) == 0.2

        # The fixed term carries no price: change `p` and read the same number.
        ffx = Fees(; fl = 3.0, fs = 7.0)
        @test calc_fees([0.6, -0.4], [100.0, 50.0], ffx) ==
              calc_fees([0.6, -0.4], [1.0, 1.0], ffx) ==
              calc_fees([0.6, -0.4], ffx) ==
              10.0
        @test calc_fixed_fees([0.6, -0.4], 3.0, (; atol = 1e-8), .>=) == 3.0
        @test calc_fixed_fees([0.6, -0.4], 7.0, (; atol = 1e-8), .<) == 7.0

        # `kwargs` decides how near zero counts as zero, and `atol` moves the boundary.
        @test Fees(; fl = 2.0).kwargs == (; atol = 1e-8)
        @test calc_fees([1e-9, 0.5], Fees(; fl = 2.0)) == 2.0
        @test calc_fees([1e-7, 0.5], Fees(; fl = 2.0)) == 4.0
        @test calc_fees([1e-7, 0.5], Fees(; fl = 2.0, kwargs = (; atol = 1e-6))) == 2.0
        @test calc_fees([1e-9, 0.5], Fees(; fl = 2.0, kwargs = (; atol = 1e-10))) == 4.0

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
        @test calc_fees(wf, pf, tns) == tns.val * dot(abs.(wf - tns.w), pf)
        tnc = Turnover(; w = tns.w, val = fill(0.02, 4))
        @test calc_fees(wf, pf, tnc) == dot(tnc.val, abs.(wf - tnc.w) .* pf)
        @test isapprox(calc_fees(wf, pf, tns), calc_fees(wf, pf, tnc))
        @test calc_fees(wf, tns) == tns.val * sum(abs.(wf - tns.w))
        @test calc_fees(wf, tnc) == dot(tnc.val, abs.(wf - tnc.w))
        @test isapprox(calc_fees(wf, tns), calc_fees(wf, tnc))

        # `fixed` is a `factory` flag; no `calc_fees` method reads it.
        tn_fx = Turnover(; w = tns.w, val = 0.02, fixed = true)
        tn_fr = Turnover(; w = tns.w, val = 0.02, fixed = false)
        @test calc_fees(wf, pf, tn_fx) == calc_fees(wf, pf, tn_fr)
        @test calc_fees(wf, tn_fx) == calc_fees(wf, tn_fr)
        @test calc_asset_fees(wf, pf, tn_fx) == calc_asset_fees(wf, pf, tn_fr)
        @test calc_asset_fees(wf, tn_fx) == calc_asset_fees(wf, tn_fr)
        @test calc_fees(wf, factory(tn_fx, wf)) == calc_fees(wf, tn_fx)
        @test iszero(calc_fees(wf, factory(tn_fr, wf)))

        # The `Nothing` methods return a typed zero, and the priced ones promote `w` and `p`.
        w32 = Float32[0.6, -0.4]
        p64 = [100.0, 50.0]
        @test typeof(calc_fees(w32, p64, nothing, .>=)) === Float64
        @test typeof(calc_fees(w32, p64, nothing)) === Float64
        @test typeof(calc_fees(w32, nothing, .>=)) === Float32
        @test typeof(calc_fees(w32, nothing)) === Float32
        @test typeof(calc_fixed_fees(w32, nothing, (; atol = 1e-8), .>=)) === Float32
        @test eltype(calc_asset_fees(w32, p64, nothing, .>=)) === Float64
        @test eltype(calc_asset_fees(w32, p64, nothing)) === Float64
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
        @test calc_fees([1e-5, 0.5, 0.5], fees_constraints(festk, fsets)) == 4.0
        @test calc_fees([1e-5, 0.5, 0.5],
                        fees_constraints(FeesEstimator(; fl = Dict("A" => 2.0), dfl = 2.0),
                                         fsets)) == 6.0

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
    # Issue #754/#760: fee amortisation spreads the one-off terms, `tn`, `fl` and `fs`,
    # over a holding period, and `l`/`s` are never touched.
    @testset "Fee amortisation" begin
        wf = [0.6, -0.4, 0.0, 0.25]
        pf = [100.0, 50.0, 20.0, 10.0]
        tnf = Turnover(; w = [0.1, 0.2, 0.3, 0.4], val = 0.02)
        fee0 = Fees(; tn = tnf, l = 0.001, s = 0.002, fl = 0.5, fs = 1.0)

        # `fa` defaults to `nothing` on both fee types.
        @test isnothing(Fees().fa)
        @test isnothing(FeesEstimator().fa)

        # `AmortisedFees` validates a stated `horizon`, positive and finite.
        @test isnothing(AmortisedFees().horizon)
        @test AmortisedFees(; horizon = 21).horizon == 21
        @test_throws DomainError AmortisedFees(; horizon = 0)
        @test_throws DomainError AmortisedFees(; horizon = -1)
        @test_throws DomainError AmortisedFees(; horizon = Inf)
        @test_throws DomainError AmortisedFees(; horizon = NaN)

        # `amortise_fees` returns its argument unchanged in three cases: a `nothing` fee, a
        # `nothing` `fa`, and an `fa.horizon` that is already stated.
        @test isnothing(PortfolioOptimisers.amortise_fees(nothing, 3))
        @test PortfolioOptimisers.amortise_fees(fee0, 3) === fee0
        feeH = Fees(; tn = tnf, l = 0.001, s = 0.002, fl = 0.5, fs = 1.0,
                    fa = AmortisedFees(; horizon = 5))
        @test PortfolioOptimisers.amortise_fees(feeH, 3) === feeH

        # A bare `AmortisedFees()` settles its horizon to the fold's length.
        feeA = Fees(; tn = tnf, l = 0.001, s = 0.002, fl = 0.5, fs = 1.0,
                    fa = AmortisedFees())
        famort = PortfolioOptimisers.amortise_fees(feeA, 3)
        @test famort.fa.horizon == 3

        # `fa === nothing` reproduces every current number: the census fixture, one trade
        # charged in full on every one of the fold's three rows.
        @test isapprox(calc_fees(wf, fee0) * 3, 6.09795)

        # A site that holds no fold and no `horizon` charges the whole cost, as today: a
        # bare, unresolved `AmortisedFees()` divides by `1`, exactly like `fa === nothing`.
        @test PortfolioOptimisers.amortisation_divisor(nothing) == 1
        @test PortfolioOptimisers.amortisation_divisor(AmortisedFees()) == 1
        @test PortfolioOptimisers.amortisation_divisor(AmortisedFees(; horizon = 7)) == 7
        @test isapprox(calc_fees(wf, feeA), calc_fees(wf, fee0))
        @test isapprox(calc_asset_fees(wf, feeA), calc_asset_fees(wf, fee0))
        @test isapprox(calc_fees(wf, pf, feeA), calc_fees(wf, pf, fee0))
        @test isapprox(calc_asset_fees(wf, pf, feeA), calc_asset_fees(wf, pf, fee0))

        # `AmortisedFees()` over a 3-row fold charges the one-off part exactly one time:
        # `l` and `s` still charge on every row, `tn`, `fl` and `fs` charge once in total.
        l_term = calc_fees(wf, fee0.l, .>=)
        s_term = -calc_fees(wf, fee0.s, .<)
        fixedlong = PortfolioOptimisers.calc_fixed_fees(wf, fee0.fl, fee0.kwargs, .>=)
        fixedshort = PortfolioOptimisers.calc_fixed_fees(wf, fee0.fs, fee0.kwargs, .<)
        turn = calc_fees(wf, tnf)
        oneoff = fixedlong + fixedshort + turn
        @test isapprox(calc_fees(wf, famort) * 3, 2.0359499999999997)
        @test isapprox(calc_fees(wf, famort), l_term + s_term + oneoff / 3)

        # A stated `horizon` overrides the fold everywhere the fee is read.
        @test isapprox(calc_fees(wf, feeH), l_term + s_term + oneoff / 5)

        # `l` and `s` are unmoved under every state of `fa`.
        @test famort.l == fee0.l && famort.s == fee0.s
        @test feeH.l == fee0.l && feeH.s == fee0.s

        # The per-asset identity holds under every state of `fa`, to rounding.
        @test isapprox(sum(calc_asset_fees(wf, famort)), calc_fees(wf, famort))
        @test isapprox(sum(calc_asset_fees(wf, feeH)), calc_fees(wf, feeH))
        @test isapprox(sum(calc_asset_fees(wf, pf, famort)), calc_fees(wf, pf, famort))
        @test isapprox(sum(calc_asset_fees(wf, pf, feeH)), calc_fees(wf, pf, feeH))

        # The price-carrying family divides the same three terms.
        l_termp = calc_fees(wf, pf, fee0.l, .>=)
        s_termp = -calc_fees(wf, pf, fee0.s, .<)
        fixedlongp = PortfolioOptimisers.calc_fixed_fees(wf, fee0.fl, fee0.kwargs, .>=)
        fixedshortp = PortfolioOptimisers.calc_fixed_fees(wf, fee0.fs, fee0.kwargs, .<)
        turnp = calc_fees(wf, pf, tnf)
        oneoffp = fixedlongp + fixedshortp + turnp
        @test isapprox(calc_fees(wf, pf, famort), l_termp + s_termp + oneoffp / 3)
        @test isapprox(calc_fees(wf, pf, feeH), l_termp + s_termp + oneoffp / 5)

        # `fees_constraints` carries the estimator's `fa` to the result, unchanged.
        fsets2 = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
        festA = FeesEstimator(; l = Dict("A" => 0.001), fa = AmortisedFees(; horizon = 10))
        @test fees_constraints(festA, fsets2).fa === festA.fa
    end
    # Ticket #765: a `WeightsTracking` benchmark fee reads no fold. `amortise_fees` is the
    # one verb that stamps a fold length onto a fee, and `predict` calls it on the
    # portfolio's own fee, never on `tr.fees`.
    @testset "a WeightsTracking fee reads no fold" begin
        Xt = [0.01 0.02 -0.01 0.03; 0.03 0.04 0.02 -0.02; -0.01 0.005 0.01 0.04]
        wbt = [0.3, 0.2, 0.4, 0.1]
        tnb = Turnover(; w = [0.25, 0.25, 0.25, 0.25], val = 0.02)
        fee_n = Fees(; tn = tnb, l = 0.001, fl = 0.5)
        fee_b = Fees(; tn = tnb, l = 0.001, fl = 0.5, fa = AmortisedFees())
        fee_h = Fees(; tn = tnb, l = 0.001, fl = 0.5, fa = AmortisedFees(; horizon = 21))
        tr_n = WeightsTracking(; fees = fee_n, w = wbt)
        tr_b = WeightsTracking(; fees = fee_b, w = wbt)
        tr_h = WeightsTracking(; fees = fee_h, w = wbt)

        # A bare `AmortisedFees()` charges the one-off terms in full, exactly as a
        # `nothing` `fa` does, because no fold length reaches this fee.
        @test PortfolioOptimisers.tracking_benchmark(tr_b, Xt) ==
              PortfolioOptimisers.tracking_benchmark(tr_n, Xt)

        # A stated `horizon` is the only way to divide them, and it divides `fl` and `tn`
        # while it leaves `l` alone.
        bn = PortfolioOptimisers.tracking_benchmark(tr_n, Xt)
        bh = PortfolioOptimisers.tracking_benchmark(tr_h, Xt)
        oneoff = PortfolioOptimisers.calc_fixed_fees(wbt, fee_n.fl, fee_n.kwargs, .>=) +
                 calc_fees(wbt, tnb)
        @test isapprox(bh .- bn, fill(oneoff * (1 - inv(21)), size(Xt, 1)))

        # Both halves of the tracking norm read one clock: the benchmark's fee and the
        # portfolio's fee are charged at the same site, over the same `X`, and a bare
        # `AmortisedFees()` divides neither.
        wpt = [0.25, 0.25, 0.3, 0.2]
        @test TrackingRiskMeasure(; tr = tr_b)(wpt, Xt, fee_b) ==
              TrackingRiskMeasure(; tr = tr_n)(wpt, Xt, fee_n)

        # `factory` advances the benchmark's reference weights and stamps no horizon.
        trf = factory(tr_b, wpt)
        @test trf.w == wpt
        @test trf.fees.tn.w == wbt
        @test isa(trf.fees.fa, AmortisedFees) && isnothing(trf.fees.fa.horizon)
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

    @testset "the fee is charged in every period" begin
        # `calc_fees` returns one number for the whole weight vector, and it is
        # subtracted from every row of `X * w`, so a `T`-row matrix charges it `T` times.
        f = PO.calc_fees(wn, fn)
        @test f == 0.0429
        @test calc_net_returns(wn, Xn, fn) ≈ Xn * wn .- f
        @test all(calc_net_returns(wn, Xn) - calc_net_returns(wn, Xn, fn) .≈ f)

        # the per asset form charges its own vector in every period too
        fa = PO.calc_asset_fees(wn, fn)
        @test calc_net_asset_returns(wn, Xn, fn) ≈
              calc_net_asset_returns(wn, Xn) .- transpose(fa)
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
        @test PO.calc_fees(w, fees) == 0.0014
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
end
