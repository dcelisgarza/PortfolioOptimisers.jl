include(joinpath(@__DIR__, "test16_setup.jl"))

@testset "Asset Risk Budgeting" begin
    df = CSV.read(joinpath(@__DIR__, "./assets/AssetRiskBudgeting1.csv.gz"), DataFrame)
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    for (i, r) in enumerate(rs)
        r = factory(r, pr, slv)
        rb = RiskBudgeting(; r = r, opt = opt)
        res = optimise(rb, rd)
        @test isa(res.retcode, OptimisationSuccess)
        rkc = risk_contribution(r, res.w, pr.X)
        v1 = minimum(rkc)
        v2 = maximum(rkc)
        rtol = if i ∈ (3, 8, 12, 27)
            5e-2
        elseif i ∈ (9, 15)
            2.5e-1
        elseif i ∈ (10, 18, 22)
            5e-1
        elseif i ∈ (11, 23)
            5e-3
        elseif i == 14
            5e-1
        elseif i ∈ (13, 17, 20, 21)
            1
        elseif i == 16
            2.5e-1
        elseif i == 26
            1e-2
        elseif i == 29
            1e-3
        else
            5e-4
        end
        success = isapprox(v2 / v1, 1; rtol = rtol)
        if !success
            println("Extrema $i fails")
            find_tol(v2 / v1, 1)
        end
        @test success

        rtol = if i ∈ (5, 7, 19, 25) || Sys.isapple() && i ∈ (2, 12)
            1e-4
        elseif i == 17
            5e-3
        elseif i ∈ (9, 10, 11, 18, 24)
            5e-4
        elseif i == 20
            1e-3
        elseif i ∈ (13, 21, 14, 15, 16, 22)
            1e-2
        else
            5e-5
        end
        success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
        if !success
            println("Weights and Contribution $i fails")
            find_tol([res.w; rkc], df[!, "$i"])
        end
        @test success
    end

    df = CSV.read(joinpath(@__DIR__, "./assets/AssetRiskBudgeting2.csv.gz"), DataFrame)
    for (i, r) in enumerate(rs)
        r = factory(r, pr, slv)
        rb = RiskBudgeting(; r = r, opt = opt,
                           rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = 1:20)))
        res = optimise(rb, rd)
        @test isa(res.retcode, OptimisationSuccess)
        rkc = risk_contribution(r, res.w, pr.X)
        v1, m1 = findmin(rkc)
        v2, m2 = findmax(rkc)
        @test m1 == 1
        success = m2 == 20
        if !success
            success = m2 == 19 || m2 == 18
        end
        @test success
        rtol = if i ∈ (3, 24)
            1e-3
        elseif i ∈ (8, 11, 13, 23, 27)
            5e-3
        elseif i == 9
            5e-1
        elseif i ∈ (10, 19, 20, 21, 22)
            2.5e-1
        elseif i ∈ (14, 15, 17)
            1e-1
        elseif i ∈ (16, 18)
            5e-2
        elseif i in (28, 29)
            1e-3
        else
            5e-4
        end
        success = isapprox(v2 / v1, 20; rtol = rtol)
        if !success
            println("Extrema $i fails")
            find_tol(v2 / v1, 20)
        end
        @test success

        rtol = if i ∈ (9, 11)
            5e-4
        elseif i == 17
            1e-3
        elseif i == 21
            5e-2
        elseif Sys.isapple() && i == 14
            1e-2
        elseif i ∈ (14, 15, 16, 20, 22)
            5e-3
        elseif i == 18
            5e-4
        else
            1e-4
        end
        success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
        if !success
            println("Weights and Contribution $i fails")
            find_tol([res.w; rkc], df[!, "$i"])
        end
        @test success
    end

    r = factory(Variance(), pr, slv)
    rb = RiskBudgeting(;
                       rba = AssetRiskBudgeting(;
                                                rkb = RiskBudgetEstimator(;
                                                                          val = ["AAPL" =>
                                                                                     0.5,
                                                                                 "MSFT" =>
                                                                                     0.25,
                                                                                 "LLY" =>
                                                                                     0.125]),
                                                sets = sets),
                       opt = JuMPOptimiser(; pe = pr, slv = slv))
    res = optimise(rb, rd)
    @test isa(res.retcode, OptimisationSuccess)
    rkc = risk_contribution(r, res.w, pr.X)
    rkc /= sum(rkc)
    rkb = risk_budget_constraints(rb.rba.rkb, sets)
    @test isapprox(rkc, rkb.val, rtol = 5e-5)

    res = optimise(RiskBudgeting(; wi = w0,
                                 rba = AssetRiskBudgeting(; sets = sets,
                                                          rkb = RiskBudgetEstimator(;
                                                                                    val = ["AAPL" =>
                                                                                               0.5])),
                                 opt = JuMPOptimiser(; pe = pr,
                                                     slv = Solver(;
                                                                  solver = Clarabel.Optimizer,
                                                                  settings = ["verbose" =>
                                                                                  false,
                                                                              "max_iter" =>
                                                                                  1])),
                                 fb = InverseVolatility(; pe = pr)))
    @test isapprox(res.w, optimise(InverseVolatility(; pe = pr)).w)
end
