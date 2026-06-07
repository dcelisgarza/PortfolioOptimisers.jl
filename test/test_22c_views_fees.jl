include(joinpath(@__DIR__, "test22_setup.jl"))

@testset "Fees" begin
    fees = FeesEstimator(; tn = TurnoverEstimator(; w = w0, val = Dict("PG" => 0.5 / 252)),
                         l = Dict("MRK" => 0.3 / 252), s = "BAC" => 0.2 / 252,
                         fl = ["XOM" => 0.5 / 252], fs = "PFE" => 0.3 / 252)
    opti = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                         wb = WeightBounds(; lb = -1, ub = 1), fees = fees, sets = sets)
    opto = JuMPOptimiser(; slv = slv, sbgt = 1, bgt = 1,
                         wb = WeightBounds(; lb = -1, ub = 1))
    res = optimise(NestedClustered(; cle = clr,
                                   opti = MeanRisk(; r = ConditionalDrawdownatRisk(),
                                                   opt = opti),
                                   opto = MeanRisk(; opt = opto)), rd)

    clusters = assignments(clr)
    idx = findfirst(x -> x == "PG", rd.nx)
    idxc = clusters[idx]
    idx = findall(x -> x == idxc, clusters)
    idx = findfirst(x -> x == "PG", rd.nx[idx])
    @test isapprox(res.resi[idxc].w[idx], 0.05)
    @test isapprox(res.w[findfirst(x -> x == "MRK", rd.nx)], 0)
    @test isapprox(res.w[findfirst(x -> x == "BAC", rd.nx)], 0)
    @test isapprox(res.w[findfirst(x -> x == "PFE", rd.nx)], 0)

    opti = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                         wb = WeightBounds(; lb = -1, ub = 1),
                         fees = fees_constraints(fees, sets), sets = sets)
    @test isapprox(res.w,
                   optimise(NestedClustered(; cle = clr,
                                            opti = MeanRisk(;
                                                            r = ConditionalDrawdownatRisk(),
                                                            opt = opti),
                                            opto = MeanRisk(; opt = opto)), rd).w)
end
@testset "Advanced use" begin
    res = optimise(NestedClustered(; cle = clr,
                                   opti = MeanRisk(; r = ConditionalValueatRisk(),
                                                   opt = JuMPOptimiser(; pe = pr,
                                                                       slv = mip_slv,
                                                                       scard = [2, 1],
                                                                       smtx = concrete_typed_array([AssetSetsMatrixEstimator(;
                                                                                                                             val = "nx_clusters1"),
                                                                                                    asset_sets_matrix(AssetSetsMatrixEstimator(;
                                                                                                                                               val = "nx_clusters2"),
                                                                                                                      sets)]),
                                                                       sets = sets)),
                                   opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv))), rd)

    @test sum(.!iszero.([res.resi[1].w[res.resi[1].smtx[1][i, :]] for i in axes(res.resi[1].smtx[1], 1)])) < 3
    @test sum(.!iszero.([res.resi[1].w[res.resi[1].smtx[2][i, :]] for i in axes(res.resi[1].smtx[2], 1)])) < 2

    @test sum(.!iszero.([res.resi[2].w[res.resi[2].smtx[1][i, :]] for i in axes(res.resi[2].smtx[1], 1)])) < 3
    @test sum(.!iszero.([res.resi[2].w[res.resi[2].smtx[2][i, :]] for i in axes(res.resi[2].smtx[2], 1)])) < 2

    opt = NestedClustered(; cle = clr,
                          opti = MeanRisk(; r = ConditionalValueatRisk(),
                                          opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                                                              lt = ThresholdEstimator(;
                                                                                      val = ["WMT" =>
                                                                                                 0.2,
                                                                                             "group2" =>
                                                                                                 0.48]),
                                                              sets = sets)),
                          opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv)))
    res = optimise(opt, rd)
    clusters = assignments(clr)

    group2 = sets.dict["group2"]
    for i in 1:(clr.k)
        nx = rd.nx[findall(x -> x == i, clusters)]
        idx = findfirst(x -> x == "WMT", nx)
        if !isnothing(idx)
            @test isapprox(res.resi[i].w[idx], 0.2; rtol = 1e-6)
        end
        idx = [findfirst(x -> x == i, nx) for i in group2]
        filter!(!isnothing, idx)
        if !isempty(idx)
            for w in res.resi[i].w[idx]
                if abs(w) > sqrt(20) * sqrt(eps(w))
                    @test w > 0.48 - sqrt(eps(w))
                end
            end
        end
    end
    opt = NestedClustered(; cle = clr,
                          opti = MeanRisk(; r = ConditionalValueatRisk(),
                                          opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                                                              lt = threshold_constraints(ThresholdEstimator(;
                                                                                                            val = ["WMT" =>
                                                                                                                       0.2,
                                                                                                                   "group2" =>
                                                                                                                       0.48]),
                                                                                         sets))),
                          opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv)))
    @test isapprox(res.w, optimise(opt, rd).w)
end

@testset "Efficient frontier" begin
    mr1 = MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv))
    mr2 = MeanRisk(; obj = MaximumRatio(), opt = JuMPOptimiser(; pe = pr, slv = slv))
    mr3 = MeanRisk(;
                   opt = JuMPOptimiser(; ret = ArithmeticReturn(; lb = Frontier(; N = 10)),
                                       slv = slv))
    nco = NestedClustered(; opti = mr1, opto = mr3)
    res = optimise(nco, rd)
    df = CSV.read(joinpath(@__DIR__, "./assets/NestedClusteredEfficientFrontier.csv.gz"),
                  DataFrame)
    success = isapprox(Matrix(df), reduce(hcat, res.w); rtol = 1e-3)
    if !success
        find_tol(Matrix(df), reduce(hcat, res.w))
    end
    @test success

    st = Stacking(; opti = [mr1, mr2], opto = mr3)
    res = optimise(st, rd)
    df = CSV.read(joinpath(@__DIR__, "./assets/StackingEfficientFrontier.csv.gz"),
                  DataFrame)
    success = isapprox(Matrix(df), reduce(hcat, res.w); rtol = 5e-4)
    if !success
        find_tol(Matrix(df), reduce(hcat, res.w))
    end
    @test success
end
@testset "Prior views" begin
    pes = [EmpiricalPrior(), FactorPrior(),
           FactorPrior(; re = DimensionReductionRegression()), HighOrderPriorEstimator(),
           HighOrderFactorPriorEstimator(),
           HighOrderFactorPriorEstimator(;
                                         pe = FactorPrior(;
                                                          re = DimensionReductionRegression())),
           BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1),
                               views = LinearConstraintEstimator(;
                                                                 val = ["AAPL == 0.00002",
                                                                        "BAC == CVX",
                                                                        "WMT == group2",
                                                                        "RRC-group1 == 0.0005"])),
           BayesianBlackLittermanPrior(; pe = FactorPrior(; pe = EmpiricalPrior(;)),
                                       sets = fsets, tau = 1 / size(rd.X, 1),
                                       views = LinearConstraintEstimator(;
                                                                         val = ["MTUM == 0.0001",
                                                                                "QUAL - USMV == -0.0003"])),
           BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1),
                               views_conf = [0.05, 0.2, 0.5, 0.9],
                               views = LinearConstraintEstimator(;
                                                                 val = ["AAPL == 0.00002",
                                                                        "BAC == CVX",
                                                                        "WMT == group2",
                                                                        "RRC-group1 == 0.0005"]))]
    prs = prior.(pes, rd)

    jopto = JuMPOptimiser(; slv = slv)
    jopties = [JuMPOptimiser(; pe = pe, slv = slv, sets = sets) for pe in pes]
    joptirs = [JuMPOptimiser(; pe = pr, slv = slv, sets = sets) for pr in prs]
    for (i, (joptir, joptie)) in enumerate(zip(joptirs, jopties))
        res1 = optimise(NestedClustered(; cle = clr, opti = MeanRisk(; opt = joptir),
                                        opto = MeanRisk(; opt = jopto)), rd)
        res2 = optimise(NestedClustered(; cle = clr, opti = MeanRisk(; opt = joptie),
                                        opto = MeanRisk(; opt = jopto)), rd)

        rtol = if i in (2, 3, 5, 6)
            5e-4
        elseif i == 7
            1e-3
        elseif i == 9
            5e-3
        else
            1e-6
        end
        res = isapprox(res1.w, res2.w; rtol = rtol)
        if !res
            println("Failed iteration: $i")
            find_tol(res1.w, res2.w)
        end
        @test res
    end
end
