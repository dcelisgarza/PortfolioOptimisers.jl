@safetestset "Fees" begin
    using PortfolioOptimisers, Test, DataFrames, TimeSeries, CSV, Clarabel, HiGHS
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
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
    sets = AssetSets(;
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
                                                sbgt = 1, bgt = 1, pr = pr, slv = slv)))
    @testset "Fees" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/Fees.csv.gz"), DataFrame)
        f1s = [0.02002313426946848, 0.12149580659357644]
        for (i, fe) in pairs(fes)
            res_mip = optimise(da, res.w, vec(values(X[end])), 1000, T, fe)
            f1 = calc_fees(res.w, fe)
            @test isapprox(f1s[i], f1)
            f2 = calc_asset_fees(res.w, fe)
            @test isapprox(df[!, "$(2*(i-1)+1)"], f2)
            f3 = calc_asset_fees(res.w, vec(values(X[end])), fe)
            @test isapprox(df[!, "$(2*(i-1)+2)"], f3)
            @test isapprox(calc_fees(res.w, vec(values(X[end])), fe) * T,
                           1000 - (sum(res_mip.cost) + res_mip.cash))
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
        @test isapprox(sric, expected_sric(r, res.ret, res.w, pr, fes[1]; rf = rf))
        @test all(isapprox.((rk, rtf, srf),
                            expected_risk_ret_ratio(r, res.ret, res.w, pr, fes[1]; rf = rf)))
        @test all(isapprox.((rk, rt, sric),
                            expected_risk_ret_sric(r, res.ret, res.w, pr; rf = rf)))

        @test isapprox(expected_risk(ReturnRiskMeasure(), res.w, pr, fes[1]),
                       rt - calc_fees(res.w, fes[1]))
        @test isapprox(expected_risk(factory(RatioRiskMeasure(; rf = rf), pr), res.w, pr,
                                     fes[1]), srf)
    end
end
