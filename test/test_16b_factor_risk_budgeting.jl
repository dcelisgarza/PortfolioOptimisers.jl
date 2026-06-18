include(joinpath(@__DIR__, "test16_setup.jl"))

@testset "Factor Risk Budgeting" begin
    df = CSV.read(joinpath(@__DIR__, "./assets/FactorRiskBudgeting1.csv.gz"), DataFrame)
    opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = BudgetRange(; lb = 0, ub = nothing),
                        bgt = 1, wb = WeightBounds(; lb = nothing, ub = nothing))
    rr = regression(StepwiseRegression(), rd)
    for (i, r) in enumerate(rs)
        if i == 25
            continue
        end
        rb = RiskBudgeting(; r = r, opt = opt, rba = FactorRiskBudgeting(; re = rr))
        res = optimise(rb, rd)
        @test isa(res.retcode, OptimisationSuccess)
        rkc = factor_risk_contribution(factory(r, pr, slv), res.w, pr.X; re = res.prb.rr)
        v1 = minimum(rkc[1:5])
        v2 = maximum(rkc[1:5])
        rtol = if i ∈ (1, 2, 10, 17)
            5e-1
        elseif i ∈ (4, 5, 7, 13, 24)
            5e-4
        elseif i == 6
            1e-4
        elseif i == 9
            1
        elseif i ∈ (11, 15, 18, 19, 20, 22)
            1e-1
        elseif i == 14
            2.5e-1
        elseif i == 26
            5e-3
        else
            5e-2
        end
        success = isapprox(v2 / v1, 1; rtol = rtol)
        if !success
            println("Extrema $i fails")
            find_tol(v2 / v1, 1)
        end
        @test success

        rtol = if i ∈ (9, 15, 16, 17, 22, 27)
            5e-3
        elseif i == 14
            5e-2
        elseif i == 21
            5e-3
        elseif i ∈ (18, 19)
            5e-4
        elseif i ∈ (10, 11)
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

    df = CSV.read(joinpath(@__DIR__, "./assets/FactorRiskBudgeting2.csv.gz"), DataFrame)
    rr = regression(StepwiseRegression(; alg = Backward()), rd)
    for (i, r) in enumerate(rs)
        if i == 25
            continue
        end
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        rb = RiskBudgeting(; r = r, opt = opt,
                           rba = FactorRiskBudgeting(; flag = true, re = rr,
                                                     rkb = RiskBudget(; val = 1:5)))
        res = optimise(rb, rd)
        @test isa(res.retcode, OptimisationSuccess)
        rkc = factor_risk_contribution(factory(r, pr, slv), res.w, pr.X; re = res.prb.rr)
        v1, m1 = findmin(rkc[1:5])
        v2, m2 = findmax(rkc[1:5])
        @test m1 == 1
        success = m2 == 5
        if !success
            success = m2 == 4
        end
        @test success
        rtol = if i ∈ (1, 2, 6, 7)
            1e-2
        elseif i == 9
            1
        elseif i ∈ (10, 11, 14, 16)
            2.5e-1
        elseif i ∈ (13, 15, 17, 18, 19, 20, 21, 22)
            5e-1
        elseif i == 26
            1e-3
        elseif i in (28, 29)
            0.25
        else
            5e-2
        end
        success = isapprox(v2 / v1, 5; rtol = rtol)
        if !success
            println("Extrema $i fails")
            find_tol(v2 / v1, 5)
        end
        @test success

        rtol = if i == 22 || Sys.isapple() && i ∈ (18, 20) || Sys.iswindows() && i == 10
            1e-3
        elseif i ∈ (1, 10) || Sys.isapple() && i ∈ (2, 6)
            5e-4
        elseif i ∈ (15, 16, 17, 19)
            5e-3
        elseif i in (13, 14)
            1e-2
        elseif i ∈ (18, 20, 24, 27)
            5e-4
        elseif i == 21
            5e-2
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
                       rba = FactorRiskBudgeting(; re = rr,
                                                 rkb = RiskBudgetEstimator(;
                                                                           val = "MTUM" =>
                                                                               0.5),
                                                 sets = fsets),
                       opt = JuMPOptimiser(; pe = pr, slv = slv,
                                           sbgt = BudgetRange(; lb = 0, ub = nothing),
                                           bgt = 1,
                                           wb = WeightBounds(; lb = nothing, ub = nothing)))
    res = optimise(rb, rd)
    @test isa(res.retcode, OptimisationSuccess)
    rkc = factor_risk_contribution(r, res.w, pr.X; re = res.prb.rr)
    rkc[1:5] /= sum(rkc[1:5])
    rkb = risk_budget_constraints(rb.rba.rkb, fsets)
    @test isapprox(rkc[1:5], rkb.val, rtol = 5e-4)
end

@testset "Factor Risk Budgeting regression estimator/result data contract" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    ## A regression ESTIMATOR must fit the factor model, so it needs the returns data:
    ## omitting `rd` raises a contextual IsNothingError rather than a deep cryptic one.
    rb_est = RiskBudgeting(; r = Variance(), opt = opt,
                           rba = FactorRiskBudgeting(; re = StepwiseRegression()))
    @test_throws IsNothingError optimise(rb_est)
    @test isa(optimise(rb_est, rd).retcode, OptimisationSuccess)
    ## A precomputed regression RESULT carries the factor model, so it needs no data.
    rr = regression(StepwiseRegression(), rd)
    rb_res = RiskBudgeting(; r = Variance(), opt = opt,
                           rba = FactorRiskBudgeting(; re = rr))
    @test isa(optimise(rb_res).retcode, OptimisationSuccess)
end
