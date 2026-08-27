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
end
