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
