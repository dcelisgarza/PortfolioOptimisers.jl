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
            # Host-sensitive: reproduces at 1e-4 on a developer machine, needs 5e-4 on CI.
            5e-4
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
    rr = regression(StepwiseRegression(; alg = BackwardElimination()), rd)
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

        rtol = if i ∈ (12, 18, 22) || Sys.isapple() && i == 20 || Sys.iswindows() && i == 10
            # 12 is host-sensitive: reproduces at 1e-4 on a developer machine, needs 5e-4 on CI.
            #
            # 18 is host-sensitive too, and it is the reason this row is no longer keyed on
            # the platform. It was granted 1e-3 on macOS alone and 5e-4 everywhere else, and
            # it fails at 5e-4 on a Linux developer machine while CI passes at the same
            # commit: the solve is a weight drift, not a failed solve, and the exported tip
            # reproduces the drift to the last digit. So the threshold sits inside the
            # spread of the hosts rather than on one platform's side of it. Issue #518.
            1e-3
        elseif i ∈ (1, 10) || Sys.isapple() && i ∈ (2, 6)
            5e-4
        elseif i ∈ (15, 16, 17, 19)
            5e-3
        elseif i in (13, 14)
            1e-2
        elseif i ∈ (20, 24, 27)
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
                                                 sets = xfsets),
                       opt = JuMPOptimiser(; pe = pr, slv = slv,
                                           sbgt = BudgetRange(; lb = 0, ub = nothing),
                                           bgt = 1,
                                           wb = WeightBounds(; lb = nothing, ub = nothing)))
    res = optimise(rb, rd)
    @test isa(res.retcode, OptimisationSuccess)
    rkc = factor_risk_contribution(r, res.w, pr.X; re = res.prb.rr)
    rkc[1:5] /= sum(rkc[1:5])
    ## The budget is now resolved against the *factor* axis, so the key is `sets.tfkey`.
    rkb = risk_budget_constraints(rb.rba.rkb, xfsets, xfsets.tfkey)
    @test isapprox(rkc[1:5], rkb.val, rtol = 5e-4)
    ## Bit-identity: the migration changed only the *lookup*. `xfsets.dict["nf"]` is the
    ## vector `fsets.dict["nx"]` was, so the same named budget resolves to the same vector
    ## and the model is the same one, weights included.
    @test rkb.val == risk_budget_constraints(rb.rba.rkb, fsets).val
    rb_pre = RiskBudgeting(; rba = FactorRiskBudgeting(; re = rr, rkb = rkb), opt = rb.opt)
    @test optimise(rb_pre, rd).w == res.w
end

@testset "Factor Risk Budgeting declared factor axis" begin
    rr = regression(StepwiseRegression(), rd)
    opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = BudgetRange(; lb = 0, ub = nothing),
                        bgt = 1, wb = WeightBounds(; lb = nothing, ub = nothing))
    rkbe = RiskBudgetEstimator(; val = "MTUM" => 0.5)

    ## A `RiskBudgetEstimator` still requires sets, and the message names the factor axis.
    @test_throws IsNothingError FactorRiskBudgeting(; re = rr, rkb = rkbe)

    ## The pre-migration shape — factor names under the asset key, no factor axis declared —
    ## is now a missing-axis error, reported against `sets.tfkey` rather than a `KeyError`
    ## about an asset universe the user never wrote in.
    @test_throws KeyError optimise(RiskBudgeting(;
                                                 rba = FactorRiskBudgeting(; re = rr,
                                                                           rkb = rkbe,
                                                                           sets = fsets),
                                                 opt = opt), rd)

    ## A declared factor axis of the wrong length is caught against `rr.L`.
    badf = UniverseSets(; dict = Dict("nx" => rd.nx, "nf" => rd.nf[1:2]))
    @test_throws DimensionMismatch optimise(RiskBudgeting(;
                                                          rba = FactorRiskBudgeting(;
                                                                                    re = rr,
                                                                                    rkb = rkbe,
                                                                                    sets = badf),
                                                          opt = opt), rd)

    ## An unread axis is left unvalidated: a `RiskBudget` result resolves no names, so a
    ## factor-less sets alongside one is not an error.
    @test isa(optimise(RiskBudgeting(;
                                     rba = FactorRiskBudgeting(; re = rr,
                                                               rkb = RiskBudget(;
                                                                                val = 1:5),
                                                               sets = fsets), opt = opt),
                       rd).retcode, OptimisationSuccess)

    ## `sets` is `@vprop`: a view slices the asset axis and leaves the factor axis and the
    ## budget alone. This is the inversion the migration forces — before it, the field held
    ## factors only and had to stay *out* of the view.
    i = 1:10
    rba = FactorRiskBudgeting(; re = rr, rkb = rkbe, sets = xfsets)
    rbav = PortfolioOptimisers.port_opt_view(rba, i)
    @test rbav.sets.dict["nx"] == rd.nx[i]
    @test rbav.sets.dict["nf"] == rd.nf
    @test rbav.rkb === rba.rkb
    @test size(rbav.re.M) == (length(i), size(rr.M, 2))
    ## The budget the viewed object generates is the unviewed one: an asset slice cannot
    ## move a factor budget.
    @test risk_budget_constraints(rbav.rkb, rbav.sets, rbav.sets.tfkey).val ==
          risk_budget_constraints(rba.rkb, rba.sets, rba.sets.tfkey).val
end

@testset "Factor Risk Budgeting regression estimator/result data contract" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    ## A regression ESTIMATOR must fit the factor model, so it needs the returns data:
    ## omitting `rd` raises a contextual IsNothingError rather than a deep cryptic one.
    ## `pr` here is an EmpiricalPrior and carries no factor block, so there is no third
    ## carrier to read the loadings from. A factor prior would answer instead of throwing.
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
