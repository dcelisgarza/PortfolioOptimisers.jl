include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Cardinality" begin
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, card = 3)
    mre = MeanRisk(; opt = opt)
    res = optimise(mre)
    w = res.w
    @test count(w .> 1e-10) <= 3

    opt = JuMPOptimiser(; l2 = L2Regularisation(; val = 0.1), pe = pr, slv = mip_slv,
                        card = 3)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mre)
    w = res.w
    @test count(w .> 1e-10) <= 3

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, card = 7)
    mre = MeanRisk(; opt = opt)
    res = optimise(mre)
    w = res.w
    @test count(abs.(w) .> 1e-10) <= 7

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                        gcarde = LinearConstraintEstimator(;
                                                           val = [:(XOM + MRK + WMT <= 2),
                                                                  :(group2 == 5)]),
                        sets = sets)
    mre = MeanRisk(; opt = opt)
    res = optimise(mre)
    w = res.w
    @test rd.nx[.!iszero.(vec(res.gcardr.A_ineq[1, :]))] == ["MRK", "WMT", "XOM"]
    @test rd.nx[.!iszero.(vec(res.gcardr.A_eq[1, :]))] == rd.nx[2:2:end]
    @test count(w[.!iszero.(vec(res.gcardr.A_ineq[1, :]))] .> 1e-10) <= 2
    @test count(w[.!iszero.(vec(res.gcardr.A_eq[1, :]))] .> 1e-10) == 5

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1,
                        gcarde = LinearConstraintEstimator(;
                                                           val = [:(XOM + MRK + WMT <= 2),
                                                                  :(group2 == 3)]),
                        sets = sets)
    mre = MeanRisk(; opt = opt)
    res = optimise(mre)
    w = res.w
    @test rd.nx[.!iszero.(vec(res.gcardr.A_ineq[1, :]))] == ["MRK", "WMT", "XOM"]
    @test rd.nx[.!iszero.(vec(res.gcardr.A_eq[1, :]))] == rd.nx[2:2:end]
    @test count(w[.!iszero.(vec(res.gcardr.A_ineq[1, :]))] .> 1e-10) <= 2
    @test count(w[.!iszero.(vec(res.gcardr.A_eq[1, :]))] .> 1e-10) == 3

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1,
                        smtx = AssetSetsMatrixEstimator(; val = "clusters1"), sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise(mre, rd)
    w = res.w
    @test sum(.!iszero.([sum(w[res.smtx[i, :]]) for i in axes(res.smtx, 1)])) == 1

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 2,
                        slt = Threshold(; val = fill(0.51, 3)),
                        smtx = AssetSetsMatrixEstimator(; val = "clusters1"), sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise(mre, rd)
    w = res.w
    @test sum(.!iszero.([sum(w[res.smtx[i, :]]) for i in axes(res.smtx, 1)])) == 1

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 2,
                        slt = Threshold(; val = fill(0.73, 3)),
                        sst = Threshold(; val = 0.38), wb = WeightBounds(; lb = -1, ub = 1),
                        sbgt = 1, bgt = nothing,
                        smtx = AssetSetsMatrixEstimator(; val = "clusters1"), sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mre, rd)
    w = res.w
    @test count([sum(w[res.smtx[i, :]]) for i in axes(res.smtx, 1)] .> 1e-10) <= 2
    ts = res.smtx * w
    @test isapprox(minimum(ts), -0.38)
    @test isapprox(maximum(ts), 0.73)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 2,
                        smtx = AssetSetsMatrixEstimator(; val = "clusters2"), sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mre, rd)
    w = res.w
    @test sum(.!iszero.([sum(w[res.smtx[i, :]]) for i in axes(res.smtx, 1)])) == 2

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = [1, 1],
                        smtx = [AssetSetsMatrixEstimator(; val = "clusters1"),
                                AssetSetsMatrixEstimator(; val = "clusters2")], sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
    res = optimise(mre, rd)
    w = res.w
    @test sum(.!iszero.([sum(w[res.smtx[1][i, :]]) for i in axes(res.smtx[1], 1)])) == 1
    @test sum(.!iszero.([sum(w[res.smtx[2][i, :]]) for i in axes(res.smtx[2], 1)])) == 1

    i = 1
    dict = Dict{Tuple{Int, Int}, Int}()
    clusters3 = Int[]
    for cs in zip(sets.dict["clusters1"], sets.dict["clusters2"])
        if !haskey(dict, cs)
            dict[cs] = i
            i += 1
        end
        push!(clusters3, dict[cs])
    end
    sets.dict["clusters3"] = clusters3
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1,
                        smtx = AssetSetsMatrixEstimator(; val = "clusters3"), sets = sets)
    mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
    @test isapprox(res.w, optimise(mre, rd).w)

    idx = [sets.dict["nx_industries"] .== i for i in sets.dict["ux_industries"]]
    for (i, ui) in zip(idx, sets.dict["ux_industries"])
        sets.dict[ui] = sets.dict["nx"][i]
    end
    m_idx = hcat(idx...)'

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                gcarde = LinearConstraintEstimator(;
                                                                                   val = :(nx ==
                                                                                           8)))),
                   rd)

    @test count(res.w[.!iszero.(vec(res.gcardr.A_eq[1, :]))] .> 1e-10) == 8

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                gcarde = LinearConstraintEstimator(;
                                                                                   val = :(nx >=
                                                                                           11)))),
                   rd)
    @test count(res.w[.!iszero.(vec(res.gcardr.A_ineq[1, :]))] .> 1e-10) >= 20

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                gcarde = LinearConstraintEstimator(;
                                                                                   val = :(JNJ +
                                                                                           MRK <=
                                                                                           1)))),
                   rd)
    @test count(res.w[.!iszero.(vec(res.gcardr.A_ineq[1, :]))] .> 1e-10) <= 1
    @test count(res.w[rd.nx .== "JNJ" .|| rd.nx .== "MRK"] .> 1e-10) <= 1

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                lt = Threshold(; val = 0.000001),
                                                gcarde = LinearConstraintEstimator(;
                                                                                   val = :(JNJ <=
                                                                                           BAC)))),
                   rd)
    @test count(res.w[.!iszero.(vec(res.gcardr.A_ineq[1, :]))] .> 1e-10) >= 2
    @test count(res.w[rd.nx .== "JNJ" .|| rd.nx .== "BAC"] .> 1e-10) >= 2

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sglt = Threshold(0.015),
                                                sgcarde = LinearConstraintEstimator(;
                                                                                    key = "ux_industries",
                                                                                    val = [:(ux_industries ==
                                                                                             4)]),
                                                sgmtx = AssetSetsMatrixEstimator(;
                                                                                 val = "nx_industries"),
                                                sets = sets)), rd)
    @test res.sgmtx == m_idx
    @test count((m_idx * res.w) .> 1e-10) == 4

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv,
                                                sglt = [ThresholdEstimator(;
                                                                           val = fill(0.015,
                                                                                      7),
                                                                           key = "ux_industries")],
                                                sgcarde = [LinearConstraintEstimator(;
                                                                                     key = "ux_industries",
                                                                                     val = [:(ux_industries >=
                                                                                              4),
                                                                                            :(ux_industries <=
                                                                                              6)])],
                                                sgmtx = [AssetSetsMatrixEstimator(;
                                                                                  val = "nx_industries")],
                                                sets = sets)), rd)
    @test 4 <= count((m_idx * res.w) .> 5e-10) <= 6

    slt = ThresholdEstimator(; val = fill(0.015, 7), key = "ux_industries")
    smtx = AssetSetsMatrixEstimator(; val = "nx_industries")
    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; slv = mip_slv, scard = 5, slt = slt,
                                                smtx = smtx, sglt = slt,
                                                sgcarde = LinearConstraintEstimator(;
                                                                                    key = "ux_industries",
                                                                                    val = [:(ux_industries >=
                                                                                             4),
                                                                                           :(ux_industries <=
                                                                                             6)]),
                                                sgmtx = smtx, sets = sets)), rd)
    @test 4 <= count((m_idx * res.w) .> 5e-10) <= 6

    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; slv = mip_slv,
                                                wb = WeightBounds(; lb = -1, ub = 1),
                                                sbgt = 1, bgt = nothing,
                                                sglt = [ThresholdEstimator(;
                                                                           val = fill(0.53,
                                                                                      7),
                                                                           key = "ux_industries")],
                                                sgst = [ThresholdEstimator(;
                                                                           val = fill(0.32,
                                                                                      7),
                                                                           key = "ux_industries")],
                                                sgcarde = [LinearConstraintEstimator(;
                                                                                     key = "ux_industries",
                                                                                     val = [:(ux_industries >=
                                                                                              4),
                                                                                            :(ux_industries <=
                                                                                              6)])],
                                                sgmtx = [AssetSetsMatrixEstimator(;
                                                                                  val = "nx_industries")],
                                                sets = sets)), rd)
    @test res.sgmtx[1] == m_idx
    ts = res.sgmtx[1] * res.w
    @test 4 <= count(abs.(ts) .> 5e-10) <= 6
    @test all(ts[ts .< 0 .&& abs.(ts) .>= 1e-10] .<= -0.32 + sqrt(eps()))
    @test all(ts[ts .>= 0 .&& abs.(ts) .>= 1e-10] .>= 0.53 - sqrt(eps()))

    smtx = AssetSetsMatrixEstimator(; val = "nx_industries")
    slt = ThresholdEstimator(; val = fill(0.53, 7), key = "ux_industries")
    sst = ThresholdEstimator(; val = fill(0.32, 7), key = "ux_industries")
    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; slv = mip_slv,
                                                wb = WeightBounds(; lb = -1, ub = 1),
                                                sbgt = 1, bgt = nothing, scard = 8,
                                                smtx = smtx, slt = slt, sst = sst,
                                                sglt = slt, sgst = sst,
                                                sgcarde = LinearConstraintEstimator(;
                                                                                    key = "ux_industries",
                                                                                    val = [:(ux_industries >=
                                                                                             4),
                                                                                           :(ux_industries <=
                                                                                             6)]),
                                                sgmtx = smtx, sets = sets)), rd)
    @test res.sgmtx == m_idx
    ts = res.sgmtx * res.w
    @test 4 <= count(abs.(ts) .> 5e-10) <= 6
    @test all(ts[ts .< 0 .&& abs.(ts) .>= 1e-10] .<= -0.32 + sqrt(eps()))
    @test all(ts[ts .>= 0 .&& abs.(ts) .>= 1e-10] .>= 0.53 - sqrt(eps()))

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                gcarde = LinearConstraintEstimator(;
                                                                                   val = :(4 *
                                                                                           MRK +
                                                                                           Consumer_Staples ==
                                                                                           4)))),
                   rd)
    @test (all(res.w[rd.nx .== "MRK"] .<= 1e-10) &&
           all(res.w[sets.dict["nx_industries"] .== "Consumer_Staples"] .> 1e-10) ||
           all(res.w[rd.nx .== "MRK"] .> 1e-10) &&
           all(res.w[sets.dict["nx_industries"] .== "Consumer_Staples"] .<= 1e-10))

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                gcarde = LinearConstraintEstimator(;
                                                                                   val = :(3 *
                                                                                           JNJ +
                                                                                           Energy ==
                                                                                           3)))),
                   rd)
    @test (all(res.w[rd.nx .== "JNJ"] .<= 1e-10) &&
           all(res.w[sets.dict["nx_industries"] .== "Energy"] .> 1e-10) ||
           all(res.w[rd.nx .== "JNJ"] .> 1e-10) &&
           all(res.w[sets.dict["nx_industries"] .== "Energy"] .<= 1e-10))

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                lt = Threshold(0.05),
                                                gcarde = LinearConstraintEstimator(;
                                                                                   val = [:(4 *
                                                                                            BAC +
                                                                                            Consumer_Staples ==
                                                                                            4),
                                                                                          :(4 *
                                                                                            JPM +
                                                                                            Consumer_Staples ==
                                                                                            4)]))),
                   rd)
    @test (all(res.w[sets.dict["nx_industries"] .== "Financials"] .<= 1e-10) &&
           all(res.w[sets.dict["nx_industries"] .== "Consumer_Staples"] .> 1e-10) ||
           all(res.w[sets.dict["nx_industries"] .== "Financials"] .> 1e-10) &&
           all(res.w[sets.dict["nx_industries"] .== "Consumer_Staples"] .<= 1e-10))

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                lt = Threshold(0.05),
                                                gcarde = LinearConstraintEstimator(;
                                                                                   val = :(AMD +
                                                                                           Consumer_Staples >=
                                                                                           5)))),
                   rd)
    @test all(res.w[rd.nx .== "AMD" .|| sets.dict["nx_industries"] .== "Consumer_Staples"] .>=
              0.05)

    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                lt = Threshold(0.05),
                                                gcarde = LinearConstraintEstimator(;
                                                                                   val = [:(AMD +
                                                                                            Consumer_Staples >=
                                                                                            5),
                                                                                          :(AAPL +
                                                                                            Consumer_Staples >=
                                                                                            5),
                                                                                          :(MSFT +
                                                                                            Consumer_Staples >=
                                                                                            5)]))),
                   rd)
    @test all(res.w[sets.dict["nx_industries"] .== "Technology" .|| sets.dict["nx_industries"] .== "Consumer_Staples"] .>=
              0.05)
end

@testset "Cardinality rows: names, sub-group shapes, thresholds and the investable mask" begin
    hslv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                  settings = "log_to_console" => false,
                  check_sol = (; allow_local = true, allow_almost = true))
    nxc = ["a", "b", "c", "d", "e", "f"]
    Xc = 0.01 .* randn(StableRNG(42), 80, 6) .+ 0.001
    rdc = ReturnsResult(; nx = nxc, X = Xc)
    setsc = UniverseSets(;
                         dict = Dict("nx" => nxc,
                                     "nx_grp" => ["g1", "g1", "g2", "g2", "g3", "g3"],
                                     "ux_grp" => ["g1", "g2", "g3"]))
    S3 = [1.0 1 0 0 0 0; 0 0 1 1 0 0; 0 0 0 0 1 1]
    S2 = [1.0 0 1 0 1 0; 0 1 0 1 0 1]
    function solve_c(rdx = rdc; kwargs...)
        opt = JuMPOptimiser(; slv = hslv, sets = setsc, kwargs...)
        return optimise(MeanRisk(; r = ConditionalValueatRisk(), opt = opt), rdx)
    end
    function card_rows(res)
        ks = collect(keys(JuMP.object_dictionary(res.model)))
        return sort!(filter(k -> occursin("card", String(k)), ks))
    end
    held(v) = count(abs.(v) .> 1e-8)
    lc_ineq(A, B) = LinearConstraint(; ineq = PartialLinearConstraint(; A = A, B = B))
    lc_eq(A, B) = LinearConstraint(; eq = PartialLinearConstraint(; A = A, B = B))

    # The asset space names its rows without a prefix, and the counts hold.
    gce = LinearConstraintEstimator(; val = [:(a + b + c <= 1), :(d + e == 1)])
    res = solve_c(; card = 3, gcarde = gce)
    @test card_rows(res) == [:card, :gcard_eq, :gcard_ineq]
    @test held(res.w) <= 3 && held(res.w[1:3]) <= 1 && held(res.w[4:5]) == 1

    # A sub-grouped constraint is written over the sub-groups: one column for each row of
    # `sgmtx`, and any number of rows. With no limit the portfolio holds all three.
    one_of_three = lc_ineq([1.0 1.0 1.0], [1.0])
    @test held(S3 * solve_c().w) == 3
    res = solve_c(; sgcarde = one_of_three, sgmtx = S3)
    @test card_rows(res) == [:sggcard_ineq_1_]
    @test held(S3 * res.w) == 1
    @test_throws DimensionMismatch JuMPOptimiser(; slv = hslv,
                                                 sgcarde = lc_ineq(ones(2, 6), [1.0, 1.0]),
                                                 sgmtx = S2)
    @test_throws DimensionMismatch JuMPOptimiser(; slv = hslv,
                                                 sgcarde = lc_eq([1.0 1.0], [1.0]),
                                                 sgmtx = S3)
    @test_throws DimensionMismatch JuMPOptimiser(; slv = hslv,
                                                 sgcarde = [one_of_three, one_of_three],
                                                 sgmtx = [S3, S2])

    # One matrix and one threshold shared by both kinds of row take one builder, and its
    # vector method gives each matrix its own entries.
    mv = [S3, S2]
    res = solve_c(; scard = [2, 1], smtx = mv, sgmtx = mv,
                  sgcarde = [one_of_three, lc_eq([1.0 1.0], [1.0])])
    @test card_rows(res) == [:scard_1_, :scard_2_, :sgcard_eq_2_, :sgcard_ineq_1_]
    @test held(S3 * res.w) == 1 && held(S2 * res.w) == 1

    # A sub-grouped threshold that is not the object of the sub-group threshold gets its own
    # builder, so the model reads it on a shared matrix as it does on a copy.
    two_of_three = lc_ineq([1.0 1.0 1.0], [2.0])
    for sgm in (S3, copy(S3))
        res = solve_c(; sgcarde = two_of_three, smtx = S3, sgmtx = sgm,
                      slt = Threshold(0.05), sglt = Threshold(0.6))
        @test card_rows(res) == [:sggcard_ineq_1_]
        @test held(S3 * res.w) == 1
    end

    # A vector of thresholds with no cardinality reaches the vector methods.
    over(x, t) = all(v -> abs(v) < 1e-8 || abs(v) >= t - 1e-8, x)
    res = solve_c(; slt = [Threshold(0.3), Threshold(0.3)], smtx = mv)
    @test over(S3 * res.w, 0.3) && over(S2 * res.w, 0.3)
    res = solve_c(; sglt = [Threshold(0.3), Threshold(0.3)], sgmtx = mv)
    @test over(S3 * res.w, 0.3) && over(S2 * res.w, 0.3)

    # An asset that leaves the investable universe removes a column of `sgmtx` and no
    # sub-group, so a sub-grouped constraint, estimated or precomputed, still solves.
    Xn = copy(Xc)
    Xn[5, 6] = NaN
    rdn = ReturnsResult(; nx = nxc, X = Xn)
    lce1 = LinearConstraintEstimator(; key = "ux_grp", val = [:(ux_grp <= 1)])
    for (sgc, sgm) in
        ((lce1, AssetSetsMatrixEstimator(; val = "nx_grp")), (one_of_three, S3))
        res = solve_c(rdn; sgcarde = sgc, sgmtx = sgm)
        @test iszero(res.w[6]) && held(S3 * res.w) == 1
    end

    # Each sub-group method with nothing to add returns before it reads the model.
    m = JuMP.Model()
    wbc = WeightBounds(; lb = 0.0, ub = 1.0)
    for f in (PortfolioOptimisers.set_scardmip_constraints!,
              PortfolioOptimisers.set_sgcardmip_constraints!)
        @test isnothing(f(m, wbc, nothing, nothing, nothing, nothing, nothing))
        @test isnothing(f(m, wbc, nothing, mv, nothing, nothing, nothing))
    end
    @test isnothing(PortfolioOptimisers.set_all_smip_constraints!(m, wbc, nothing, nothing,
                                                                  nothing, nothing, nothing,
                                                                  nothing))
    @test isnothing(PortfolioOptimisers.set_all_smip_constraints!(m, wbc, nothing, nothing,
                                                                  mv, nothing, nothing,
                                                                  nothing))
    @test isempty(JuMP.object_dictionary(m))

    # The two validators, branch by branch.
    af = PortfolioOptimisers.assert_subgroup_mip_fields
    agf = PortfolioOptimisers.assert_subgrouped_mip_fields
    thr = Threshold(0.1)
    td = TimeDependent([1, 2])
    @test isnothing(af(td, nothing, nothing, nothing))
    @test isnothing(af(1, S3, thr, nothing))
    @test_throws DomainError af(0, S3, nothing, nothing)
    @test_throws ArgumentError af(1, [S3], nothing, nothing)
    @test_throws ArgumentError af(1, S3, [thr], nothing)
    @test_throws ArgumentError af(1, S3, nothing, [thr])
    @test isnothing(af([1, 2], mv, [thr, nothing], [nothing, thr]))
    @test_throws ArgumentError af([1], S3, nothing, nothing)
    @test_throws DimensionMismatch af([1, 2], [S3], nothing, nothing)
    @test_throws PortfolioOptimisers.IsEmptyError af([1], [S3], Threshold[], nothing)
    @test_throws DimensionMismatch af([1], [S3], [thr, thr], nothing)
    @test_throws PortfolioOptimisers.IsEmptyError af([1], [S3], nothing, Threshold[])
    @test_throws DimensionMismatch af([1], [S3], nothing, [thr, thr])
    @test isnothing(af(nothing, S3, thr, nothing))
    @test_throws ArgumentError af(nothing, [S3], nothing, thr)
    @test isnothing(af(nothing, mv, [thr, thr], [thr, nothing]))
    @test_throws ArgumentError af(nothing, S3, [thr], nothing)
    @test_throws PortfolioOptimisers.IsEmptyError af(nothing, Matrix{Float64}[], [thr],
                                                     nothing)
    @test_throws PortfolioOptimisers.IsEmptyError af(nothing, [S3], Threshold[], nothing)
    @test_throws DimensionMismatch af(nothing, [S3], [thr, thr], nothing)
    @test_throws PortfolioOptimisers.IsEmptyError af(nothing, [S3], nothing, Threshold[])
    @test_throws DimensionMismatch af(nothing, [S3], nothing, [thr, thr])

    @test isnothing(agf(td, nothing, nothing, nothing))
    @test isnothing(agf(one_of_three, S3, thr, nothing))
    @test_throws DimensionMismatch agf(one_of_three, S2, nothing, nothing)
    @test_throws ArgumentError agf(one_of_three, [S3], nothing, nothing)
    @test_throws ArgumentError agf(one_of_three, S3, [thr], nothing)
    @test_throws ArgumentError agf(one_of_three, S3, nothing, [thr])
    @test isnothing(agf([one_of_three], [S3], [thr], [thr]))
    @test_throws PortfolioOptimisers.IsEmptyError agf(LinearConstraint[], [S3], nothing,
                                                      nothing)
    @test_throws ArgumentError agf([one_of_three], S3, nothing, nothing)
    @test_throws PortfolioOptimisers.IsEmptyError agf([one_of_three], Matrix{Float64}[],
                                                      nothing, nothing)
    @test_throws DimensionMismatch agf([one_of_three], mv, nothing, nothing)
    @test_throws PortfolioOptimisers.IsEmptyError agf([one_of_three], [S3], Threshold[],
                                                      nothing)
    @test_throws DimensionMismatch agf([one_of_three], [S3], [thr, thr], nothing)
    @test_throws PortfolioOptimisers.IsEmptyError agf([one_of_three], [S3], nothing,
                                                      Threshold[])
    @test_throws DimensionMismatch agf([one_of_three], [S3], nothing, [thr, thr])
    @test_throws DimensionMismatch agf([one_of_three], [S2], nothing, nothing)
    @test isnothing(agf(nothing, S3, thr, nothing))
    @test_throws ArgumentError agf(nothing, [S3], nothing, thr)
    @test isnothing(agf(nothing, mv, [thr, thr], [thr, nothing]))
    @test_throws ArgumentError agf(nothing, S3, [thr], nothing)
    @test_throws PortfolioOptimisers.IsEmptyError agf(nothing, Matrix{Float64}[], [thr],
                                                      nothing)
    @test_throws PortfolioOptimisers.IsEmptyError agf(nothing, [S3], Threshold[], nothing)
    @test_throws DimensionMismatch agf(nothing, [S3], [thr, thr], nothing)
    @test_throws PortfolioOptimisers.IsEmptyError agf(nothing, [S3], nothing, Threshold[])
    @test_throws DimensionMismatch agf(nothing, [S3], nothing, [thr, thr])
end

@testset "Phylogeny" begin
    plc = IntegerPhylogenyEstimator(; pl = NetworkEstimator(), B = 1)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1),
                        l2 = L2Regularisation(; val = 0.001))
    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt))
    @test all(JuMP.value.(res.plr.A * res.model[:ib]) .<= res.plr.B)
    idx = [BitVector(res.plr.A[:, i]) for i in axes(res.plr.A, 2)]
    @test all([(count(abs.(getindex(res.w, i)) .> 1e-10) <= 1) for i in idx])
    #=
    Pajarito's outer approximation does not reach the same integer vertex on every host, so
    the weights are not a comparable number. Each of the eight solvers in `mip_slv`, run on
    its own, succeeds, and they stop at four vertices:

      mip1, mip5        assets 2, 3, 7, 12, 18, 20       Sharpe -1.09% vs the reference
      mip2, mip3, mip4  assets 2, 3, 10, 12, 20          Sharpe -1.02%
      mip6, mip7        assets 2, 3, 7, 10, 12, 20       Sharpe +0.002% (the reference)
      mip8              19 assets, tol_feas = 1e-4       Sharpe +2.97%

    The model accepts each of the first three supports: with the binaries fixed and the
    rest solved by Clarabel, the objective is 2.7435e-4, 2.7301e-4 and 2.7227e-4. The
    reference is the best of them, and the others are where the solver stopped.

    So compare the ratio `MaximumRatio` maximises. `rtol = 0.02` is about twice the widest
    spread of the solvers that satisfy the constraint, and it refuses the mip8 answer, whose
    small weights break the phylogeny constraint the check above also reads. See #1276.
    =#
    sharpe = (dot(pr.mu, res.w) - rf) / sqrt(dot(res.w, pr.sigma, res.w))
    @test isapprox(sharpe, 0.1822224153260592; rtol = 0.02)

    plc = IntegerPhylogenyEstimator(; pl = NetworkEstimator(), B = fill(2, size(pr.X, 2)))
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1),
                        l2 = L2Regularisation(; val = 0.0001))
    res = optimise(MeanRisk(; obj = MinimumRisk(), opt = opt))
    @test all(JuMP.value.(res.plr.A * res.model[:ib]) .<= res.plr.B)
    idx = [BitVector(res.plr.A[:, i]) for i in axes(res.plr.A, 2)]
    @test all([(count(abs.(getindex(res.w, i)) .> 1e-10) <= 2) for i in idx])
    success = isapprox(res.w,
                       [-7.681434332864639e-13, 7.405561598382192e-14, 0.014250247570361224,
                        -6.974708088175452e-13, 0.08537332610615428,
                        -1.0150718449032514e-12, 0.03983561006114424, 0.2502122641236305,
                        0.05404071746483757, -1.5666916399782238e-12,
                        -1.538861209090204e-12, 0.2025433723191622, -0.0014220443665369262,
                        -1.5802206832408362e-12, -1.5503975446319445e-12,
                        0.15784094556549422, -4.439084812281355e-13,
                        -1.4843816960768885e-12, 0.1276616300735639, 0.0696639310927601];
                       rtol = 1e-6)

    plc = IntegerPhylogenyEstimator(; pl = ClustersEstimator(), B = [2, 1])
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1),
                        l2 = L2Regularisation(; val = 0.0001))
    res = optimise(MeanRisk(; obj = MinimumRisk(), opt = opt))
    @test all(JuMP.value.(res.plr.A * res.model[:ib]) .<= res.plr.B)
    idx = [BitVector(res.plr.A[i, :]) for i in axes(res.plr.A, 1)]
    @test all([(count(abs.(getindex(res.w, i)) .> 1e-10) <= 2) for i in idx])
    @test isapprox(res.w,
                   [-8.32253597789343e-14, -1.594625540670819e-14, -1.1337620095266581e-13,
                    -7.615656061114373e-14, -1.4550212340578226e-13, -1.079970867058386e-13,
                    0.23699252155911452, -2.1813171374502812e-13, -1.2603823102022852e-13,
                    -1.8998772470476241e-13, -1.8199948157385003e-13, 0.5615378679061623,
                    -8.861295956489424e-14, -1.9188139255250999e-13,
                    -1.8623586726948033e-13, -1.970863146476292e-13, -5.215453906692728e-14,
                    -1.778276197103779e-13, -1.9695027660052278e-13, 0.20146961053707255],
                   rtol = 1e-6)

    plc = SemiDefinitePhylogeny(; A = phylogeny_matrix(clr, pr.X))
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    res = optimise(MeanRisk(; obj = MinimumRisk(), opt = opt))
    @test isapprox(JuMP.value.(res.plr.A .* res.model[:W]), zeros(size(pr.sigma)),
                   atol = 1e-10)

    plc = SemiDefinitePhylogenyEstimator(; pl = ClustersEstimator(), p = 1000)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    @test isapprox(res.w, optimise(MeanRisk(; obj = MinimumRisk(), opt = opt)).w)

    plc = SemiDefinitePhylogeny(; A = phylogeny_matrix(clr, pr.X), p = 10)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    @test isapprox(res.w, optimise(MeanRisk(; obj = MinimumRisk(), opt = opt)).w)

    plc = SemiDefinitePhylogeny(; A = phylogeny_matrix(clr, pr.X))
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    res1 = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                             opt = opt))
    @test isapprox(JuMP.value.(res1.plr.A .* res1.model[:W]), zeros(size(pr.sigma)),
                   atol = 1e-10)
    @test isapprox(res1.w,
                   [-8.941947065896066e-10, -0.04932710440381371, -6.863938562738464e-10,
                    1.3988410703687132e-9, 7.874062368849783e-9, 1.412585165547874e-10,
                    1.7321574723075382e-11, 3.14807397478865e-9, 2.533850859517078e-10,
                    4.2259407336717684e-9, 9.27519848836695e-9, 0.5762987006529464,
                    -8.9766618039303e-10, 3.2999661080490746e-9, 1.890273506358333e-9,
                    2.284039630442464e-9, 4.098035458673607e-9, 2.8259584736181038e-9,
                    1.9502905591238874e-9, 0.4730283635464767], rtol = 1e-6)

    plc = SemiDefinitePhylogeny(; A = phylogeny_matrix(clr, pr.X), p = 5)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    res2 = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                             opt = opt))
    @test isapprox(JuMP.value.(res2.plr.A .* res2.model[:W]), zeros(size(pr.sigma)),
                   atol = 1e-10)
    @test isapprox(res1.w, res2.w; rtol = 0.25)
    @test isapprox(res2.w,
                   [4.538887108402e-12, -6.1572184586398375e-12, 5.890457303938837e-12,
                    1.1064891957189628e-11, 8.16860330669961e-11, 1.0447686807968064e-11,
                    7.515312850909054e-12, 3.4302092558703144e-11, 9.830689623680973e-12,
                    4.196444332597126e-11, 1.595363611128629e-10, 0.44103049364497793,
                    4.015117234808678e-12, 3.734698654521197e-11, 1.9818372625965523e-11,
                    2.3540415362795815e-11, 6.674140406117119e-11, 3.620870179464855e-11,
                    2.9544937962876324e-11, 0.5589695057771867], rtol = 1e-6)

    plc = SemiDefinitePhylogeny(; A = phylogeny_matrix(clr, pr.X))
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    res1 = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumUtility(),
                             opt = opt))
    @test isapprox(JuMP.value.(res1.plr.A .* res1.model[:W]), zeros(size(pr.sigma)),
                   atol = 1e-10)
    @test isapprox(res1.w,
                   [5.795063775028467e-10, 1.6700229186291166e-11, 1.3737504141980956e-9,
                    7.553971775400608e-10, 0.15491240378613524, 8.130769655580166e-10,
                    0.052786428307945235, 0.3010004052356739, 0.12864415218245234,
                    1.6194212944642728e-9, 1.7330018300923791e-9, 0.3044224075653002,
                    1.6587453277888904e-9, 1.1846793161814234e-9, 6.789585538788945e-10,
                    1.2549206102211462e-9, 5.816904347357348e-10, 9.192974758044433e-10,
                    1.3752920313726993e-9, 0.05823418837805521], rtol = 1e-6)

    plc = SemiDefinitePhylogeny(; A = phylogeny_matrix(clr, pr.X), p = 5)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    res2 = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumUtility(),
                             opt = opt))
    @test isapprox(JuMP.value.(res2.plr.A .* res2.model[:W]), zeros(size(pr.sigma)),
                   atol = 1e-10)
    @test isapprox(res2.w,
                   [9.734211379642436e-10, 4.1083003213516983e-10, 1.8417217137810698e-9,
                    1.1175942079694052e-9, 0.22174568685948925, 1.1252994093191541e-9,
                    0.10878815061030933, 0.22584558658264856, 0.16804999751018831,
                    1.3691721404910584e-9, 1.8115167652939963e-9, 0.2755704634873393,
                    2.712875597891968e-9, 1.264257843844942e-9, 6.749859978725407e-10,
                    1.2598949048283668e-9, 8.634272415613415e-10, 7.875561180262464e-10,
                    1.7964324466334559e-9, 9.69410371286415e-8], rtol = 1e-6)
end

@testset "Integer phylogeny rows: positions, vector bounds and widths" begin
    hslv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                  settings = "log_to_console" => false,
                  check_sol = (; allow_local = true, allow_almost = true))
    Xp = 0.01 .* randn(StableRNG(42), 80, 6) .+ 0.001
    rdp = ReturnsResult(; nx = string.('a':'f'), X = Xp)
    function solve_p(slv = hslv; kwargs...)
        opt = JuMPOptimiser(; slv = slv, wb = WeightBounds(; lb = 0, ub = 0.4), kwargs...)
        res = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumReturn(),
                                opt = opt), rdp)
        @test isa(res.retcode, OptimisationSuccess)
        return res
    end
    held_bits(res) = round.(Int, JuMP.value.(res.model[:ib]))

    # Without the rows, assets 1 and 6 are both held.
    w = solve_p().w
    @test w[1] > 1e-3 && w[6] > 1e-3
    # The first entry links assets 1 and 6, and 2 and 4, under a vector bound. The second
    # links 1 and 3. Each entry writes its rows under its own position.
    A1 = zeros(Int, 6, 6)
    A1[1, 6] = A1[6, 1] = 1
    A1[2, 4] = A1[4, 2] = 1
    A2 = zeros(Int, 6, 6)
    A2[1, 3] = A2[3, 1] = 1
    pl1 = IntegerPhylogeny(; A = A1, B = [1, 1, 1, 1])
    pl2 = IntegerPhylogeny(; A = A2, B = 1)
    res = solve_p(; ple = [pl1, pl2])
    ib = held_bits(res)
    @test all(pl1.A * ib .<= pl1.B) && all(pl2.A * ib .<= pl2.B)
    @test min(abs(res.w[1]), abs(res.w[6])) < 1e-8
    @test all(abs.(res.w[ib .== 0]) .< 1e-8)
    @test haskey(res.model, :card_plg_1) && haskey(res.model, :card_plg_2)

    # An entry of another kind adds no row here, and the integer entry keeps its position.
    # The semidefinite entry needs a conic MIP solver.
    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; slv = mip_slv,
                                                wb = WeightBounds(; lb = 0, ub = 0.4),
                                                ple = [SemiDefinitePhylogeny(; A = A1,
                                                                             p = 0.05),
                                                       pl1])), rdp)
    @test isa(res.retcode, OptimisationSuccess)
    @test haskey(res.model, :sdp_plg_1) && haskey(res.model, :card_plg_2)
    @test !haskey(res.model, :card_plg_1)
    @test all(pl1.A * held_bits(res) .<= pl1.B)

    # An entry with one column too few is refused before the solve.
    @test_throws DimensionMismatch solve_p(; ple = IntegerPhylogeny(; A = zeros(Int, 5, 5)))
end

@testset "Tracking" begin
    rdb = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__,
                                                        "./assets/SP500_idx.csv.gz"));
                                      timestamp = :Date)[(end - 252):end])
    wr = vec(rdb.X)
    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 3e-3))
    @test !PortfolioOptimisers.needs_previous_weights(opt)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    @test !PortfolioOptimisers.needs_previous_weights(mre)
    res = optimise(mre)
    @test LinearAlgebra.norm(rd.X * res.w - wr) / sqrt(size(rd.X, 1)) <= 3e-3

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 2.5e-3,
                                           alg = L1Norm()))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise(mre)
    @test LinearAlgebra.norm(rd.X * res.w - wr, 1) / size(rd.X, 1) <= 2.5e-3

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = TrackingError(; tr = WeightsTracking(; w = w0), err = 2e-3))
    @test PortfolioOptimisers.needs_previous_weights(opt)
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    @test PortfolioOptimisers.needs_previous_weights(mre)
    res = optimise(mre)
    @test LinearAlgebra.norm(rd.X * (res.w - w0)) / sqrt(size(rd.X, 1)) <= 2e-3

    # `err` is stated in the units of `alg`, so a `SquaredL2Norm` bound is on the squared
    # error. The model shares the `L2Norm` cone and square-roots the bound to match, which is
    # what `tracking_error_soc_factor` is for.
    sq = SquaredL2Norm()
    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 9e-6,
                                           alg = sq))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise(mre)
    @test PortfolioOptimisers.norm_error(sq, rd.X * res.w, wr, size(rd.X, 1)) <=
          9e-6 * (1 + 1e-6)
    # The square is what separates the two norms, not the cone.
    @test isapprox(PortfolioOptimisers.norm_error(sq, rd.X * res.w, wr, size(rd.X, 1)),
                   PortfolioOptimisers.norm_error(L2Norm(; ddof = sq.ddof), rd.X * res.w,
                                                  wr, size(rd.X, 1))^2)

    #=
    Issue #547 asked for the `SquaredL2Norm` conversion to be confirmed against a solved
    model, because the `TrackingError` docstring states it as a trap: `err` is read in the
    units of `alg`, so the same number is two different bounds. The conversion is the
    square root and it carries no dependence on `T`, so a `SquaredL2Norm` bound of `9e-6`
    and an `L2Norm` bound of `3e-3` are the SAME bound. `tracking_error_soc_factor` writes
    one cone bound for both, and the two models must therefore return the same weights.
    Read the deviation from the weights, not from a model key: `TrackingError` registers
    `:t_tr_`, `:tr_`, `:ctr_soc_` and `:ctr_`, and never `:sq_tracking_risk_`, which
    belongs to `TrackingRiskMeasure`.
    =#
    optsq = JuMPOptimiser(; pe = pr, slv = slv,
                          tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 9e-6,
                                             alg = SquaredL2Norm()))
    optl2 = JuMPOptimiser(; pe = pr, slv = slv,
                          tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 3e-3,
                                             alg = L2Norm()))
    ressq = optimise(MeanRisk(; obj = MinimumRisk(), opt = optsq))
    resl2 = optimise(MeanRisk(; obj = MinimumRisk(), opt = optl2))
    @test PortfolioOptimisers.tracking_error_soc_factor(SquaredL2Norm(), 9e-6,
                                                        size(rd.X, 1)) ==
          PortfolioOptimisers.tracking_error_soc_factor(L2Norm(), 3e-3, size(rd.X, 1))
    @test isapprox(ressq.w, resl2.w, atol = 1e-8)
    dsq = PortfolioOptimisers.norm_error(SquaredL2Norm(), rd.X * ressq.w, wr, size(rd.X, 1))
    dl2 = PortfolioOptimisers.norm_error(L2Norm(), rd.X * resl2.w, wr, size(rd.X, 1))
    @test isapprox(dsq, dl2^2)
    # both bounds bind, so the equality is not the trivial one of a slack constraint
    @test dsq <= 9e-6 * (1 + 1e-6)
    @test dl2 <= 3e-3 * (1 + 1e-6)
    @test dsq / 9e-6 > 0.999
    @test dl2 / 3e-3 > 0.999
    # the model registers the tracking rows under `tr`, not under `tracking_risk`
    ks = keys(JuMP.object_dictionary(ressq.model))
    @test :t_tr_1 in ks
    @test :tr_1 in ks
    @test :ctr_soc_1 in ks
    @test :ctr_1 in ks
    @test !(:sq_tracking_risk_1 in ks)
    @test !(:tracking_risk_1 in ks)
    # `ddof` moves the cone bound, so it moves the realised deviation
    optd0 = JuMPOptimiser(; pe = pr, slv = slv,
                          tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 3e-3,
                                             alg = L2Norm(; ddof = 0)))
    resd0 = optimise(MeanRisk(; obj = MinimumRisk(), opt = optd0))
    @test LinearAlgebra.norm(rd.X * resd0.w - wr) > LinearAlgebra.norm(rd.X * resl2.w - wr)
    @test isapprox(LinearAlgebra.norm(rd.X * resd0.w - wr),
                   PortfolioOptimisers.tracking_error_soc_factor(L2Norm(; ddof = 0), 3e-3,
                                                                 size(rd.X, 1)),
                   rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 4.5e-3,
                                           alg = LpNorm()))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise(mre)
    @test LinearAlgebra.norm(rd.X * res.w - wr, 3) / cbrt(size(rd.X, 1) - 1) <= 4.5e-3

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 2e-2,
                                           alg = LInfNorm()))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise(mre)
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test LinearAlgebra.norm(rd.X * res.w - wr, Inf) <= 2e-2 * (1 + 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = [TrackingError(; tr = WeightsTracking(; w = w0), err = 2e-3,
                                            alg = L1Norm())])
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise(mre)
    @test LinearAlgebra.norm(rd.X * (res.w - w0), 1) / size(rd.X, 1) <= 2e-3

    tr = RiskTrackingError(; err = 0.0, tr = WeightsTracking(; w = w0),
                           alg = DependentVariableTracking())
    opt = JuMPOptimiser(; pe = pr, slv = slv, tr = tr)
    @test PortfolioOptimisers.needs_previous_weights(opt)
    mre = MeanRisk(; r = ConditionalValueatRisk(), opt = opt)
    @test PortfolioOptimisers.needs_previous_weights(mre)
    res = optimise(mre)
    @test isapprox(res.w,
                   optimise(MeanRisk(; r = ConditionalValueatRisk(),
                                     opt = JuMPOptimiser(; pe = pr, slv = slv))).w,
                   rtol = 1e-6)

    tr = RiskTrackingError(; err = 0.5, tr = WeightsTracking(; w = w0),
                           alg = IndependentVariableTracking())
    opt = JuMPOptimiser(; pe = pr, slv = slv, tr = tr)
    mre = MeanRisk(; obj = MaximumRatio(), opt = opt)
    res = optimise(mre)
    @test isapprox(res.w,
                   optimise(MeanRisk(; obj = MaximumRatio(),
                                     opt = JuMPOptimiser(; pe = pr, slv = slv))).w,
                   rtol = 5e-4)

    tr = RiskTrackingError(; err = 0, tr = WeightsTracking(; w = w0),
                           alg = IndependentVariableTracking())
    opt = JuMPOptimiser(; pe = pr, slv = slv, tr = tr)
    mre = MeanRisk(; obj = MaximumRatio(), opt = opt)
    res = optimise(mre)
    @test isapprox(res.w, w0, rtol = 1e-6)
end

@testset "Budget" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumReturn(), opt = opt)
    res = optimise(mr)
    @test isapprox(sum(res.w), 1)
    @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -1, rtol = 1e-6)
    @test isapprox(sum(res.w[res.w .>= zero(eltype(res.w))]), 2, rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 0.15, bgt = 0.5,
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mr)
    @test isapprox(sum(res.w), 0.5)
    @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -0.15, rtol = 1e-4)
    @test isapprox(sum(res.w[res.w .>= zero(eltype(res.w))]), 0.65, rtol = 5e-5)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                        sbgt = BudgetRange(; lb = 0.15, ub = 0.15),
                        bgt = BudgetRange(; lb = 0.3, ub = 0.45),
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mr)
    @test 0.1 <= sum(res.w) <= 0.45
    @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -0.15, rtol = 5e-5)
    @test 0.45 <= sum(res.w[res.w .>= zero(eltype(res.w))]) <= 0.60

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = nothing, bgt = 1.7,
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise(mr)
    @test isapprox(sum(res.w), 1.7)
    @test all(res.w .>= 0)
    @test !haskey(res.model, :sbgt)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1.4, bgt = nothing,
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mr)
    @test isapprox(sum(res.w[res.w .< 0]), -1.4, rtol = 1e-3)
    @test !haskey(res.model, :bgt)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                        sbgt = BudgetRange(; lb = 0.41, ub = 0.63), bgt = 0.87,
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mr)
    @test isapprox(sum(res.w), 0.87)
    @test -0.63 <= sum(res.w[res.w .< zero(eltype(res.w))]) <= -0.41
    @test 0.87 + 0.41 <= sum(res.w[res.w .>= zero(eltype(res.w))]) <= 0.87 + 0.63

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 0.61,
                        bgt = BudgetRange(; lb = 0.4, ub = 0.79),
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumUtility(), opt = opt)
    res = optimise(mr)
    @test 0.4 <= sum(res.w) <= 0.79
    @test isapprox(sum(res.w[res.w .< zero(eltype(res.w))]), -0.61, rtol = 5e-5)
    @test 0.61 + 0.4 <= sum(res.w[res.w .> zero(eltype(res.w))]) <= 0.61 + 0.79

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                        sbgt = BudgetRange(; lb = 0.41, ub = 0.63), bgt = nothing,
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mr)
    @test -0.63 <= sum(res.w[res.w .< zero(eltype(res.w))]) <= -0.41
    @test !haskey(res.model, :bgt)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = nothing,
                        bgt = BudgetRange(; lb = 0.4, ub = 0.79),
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumUtility(), opt = opt)
    res = optimise(mr)
    @test 0.4 <= sum(res.w) <= 0.79
    @test !haskey(res.model, :sbgt)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                        sbgt = BudgetRange(; lb = nothing, ub = 0.23),
                        bgt = BudgetRange(; lb = 0.41, ub = nothing),
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mr)
    @test sum(res.w) >= 0.41
    @test sum(res.w[res.w .< 0]) >= -0.23
    @test sum(res.w[res.w .>= 0]) >= 0.64

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                        sbgt = BudgetRange(; lb = 0.35, ub = nothing),
                        bgt = BudgetRange(; lb = nothing, ub = 0.65),
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mr)
    @test sum(res.w) <= 0.41
    @test sum(res.w[res.w .< 0]) <= -0.35
    @test sum(res.w[res.w .>= 0]) <= 0.76

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = nothing, bgt = nothing,
                        wb = WeightBounds(; lb = -1, ub = 1))
    mr = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mr)
    @test !haskey(res.model, :sbgt)
    @test !haskey(res.model, :lbgt)
end

@testset "Scalar weight bounds (broadcast ⊖ regression)" begin
    # Scalar positive bounds with MaximumRatio previously errored ("Subtraction between an
    # array and a JuMP scalar") because set_weight_constraints! built `w - k*lb` / `w - k*ub`
    # without broadcasting. Vector bounds worked by accident; scalars did not. The constraint
    # now uses `⊖`, so a scalar upper cap must assemble, solve, and bind.
    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; ub = 0.15))
    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt))
    @test isapprox(sum(res.w), 1; rtol = 1e-6)
    @test all(res.w .<= 0.15 + 1e-6)
    @test isapprox(maximum(res.w), 0.15; atol = 1e-4)

    # Estimator form (lb defaulting to nothing) must also assemble and solve. The
    # WeightBoundsEstimator resolves per-name bounds, so it needs `sets`.
    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                        wb = WeightBoundsEstimator(; lb = nothing, ub = 0.15))
    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt))
    @test isapprox(sum(res.w), 1; rtol = 1e-6)
    @test all(res.w .<= 0.15 + 1e-6)

    # Scalar lower and upper bounds together both bind.
    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = 0.01, ub = 0.2))
    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt))
    @test all(res.w .>= 0.01 - 1e-6)
    @test all(res.w .<= 0.2 + 1e-6)
end

@testset "Name resolution does not edit the caller's Universe Sets" begin
    # `name_to_val!` used to call `unique!` on the member list it read out of `sets.dict`, so
    # resolving a group with a repeated member permanently shortened the user's Universe Sets.
    # Universe Sets are configuration and are reused across folds and optimisers, so the
    # resolver must leave them untouched. `resolve_axis_name` now returns a copy.
    dup_sets = UniverseSets(; xkey = "nx",
                            dict = Dict("nx" => ["A", "B", "C"], "Tech" => ["A", "B", "A"]))
    val = PortfolioOptimisers.estimator_to_val("Tech" => 0.5, dup_sets)
    @test val == [0.5, 0.5, 0.0]
    @test dup_sets.dict["Tech"] == ["A", "B", "A"]

    # The dictionary form takes the same path and must also leave the sets alone.
    val = PortfolioOptimisers.estimator_to_val(Dict("Tech" => 0.25, "C" => 0.75), dup_sets)
    @test val == [0.25, 0.25, 0.75]
    @test dup_sets.dict["Tech"] == ["A", "B", "A"]

    # An asset name takes precedence over a group of the same spelling.
    clash_sets = UniverseSets(; xkey = "nx",
                              dict = Dict("nx" => ["A", "B", "C"], "A" => ["B", "C"]))
    val = PortfolioOptimisers.estimator_to_val("A" => 1.0, clash_sets)
    @test val == [1.0, 0.0, 0.0]

    # An unknown name warns when `strict` is false and throws when it is true. Both branches
    # now run through `strict_diagnostic`, so the policy is one function.
    @test_logs (:warn,) PortfolioOptimisers.estimator_to_val("Nope" => 1.0, dup_sets)
    @test_throws ArgumentError PortfolioOptimisers.estimator_to_val("Nope" => 1.0, dup_sets;
                                                                    strict = true)

    # A group whose member is missing from the axis keeps the members it can place, and
    # reports the rest under the same strictness policy.
    partial_sets = UniverseSets(; xkey = "nx",
                                dict = Dict("nx" => ["A", "B", "C"], "Mixed" => ["A", "Z"]))
    val = @test_logs (:warn,) PortfolioOptimisers.estimator_to_val("Mixed" => 0.3,
                                                                   partial_sets)
    @test val == [0.3, 0.0, 0.0]
    @test_throws ArgumentError PortfolioOptimisers.estimator_to_val("Mixed" => 0.3,
                                                                    partial_sets;
                                                                    strict = true)
end

@testset "Row assembly and carriers (issue #513)" begin
    sets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))

    # `merge_partial_linear_constraints` stacks the halves it is given, in input order,
    # skips the absent ones, and refuses rows of differing width.
    p1 = PartialLinearConstraint(; A = [1.0 0.0 0.0; 0.0 1.0 0.0], B = [1.0, 2.0])
    p2 = PartialLinearConstraint(; A = [1.0 1.0 1.0], B = [3.0])
    m = PortfolioOptimisers.merge_partial_linear_constraints([p1, nothing, p2])
    @test size(m.A, 1) == length(m.B) == 3
    @test m.B == [1.0, 2.0, 3.0]
    @test isnothing(PortfolioOptimisers.merge_partial_linear_constraints([nothing, nothing]))
    @test_throws DimensionMismatch PortfolioOptimisers.merge_partial_linear_constraints([p1,
                                                                                         PartialLinearConstraint(;
                                                                                                                 A = [1.0 1.0],
                                                                                                                 B = [0.0])])

    # A one-element vector comes back as the element itself, not as a copy of it, and a
    # bare constraint reaches the same method through the `LinearConstraint` arity.
    lc = LinearConstraint(; ineq = p1)
    @test PortfolioOptimisers.merge_linear_constraints([lc]) === lc
    @test PortfolioOptimisers.merge_linear_constraints(lc) === lc
    @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.merge_linear_constraints(LinearConstraint[])

    # The four computed properties answer `nothing` for the half that is absent.
    @test lc.A_ineq === p1.A
    @test lc.B_ineq === p1.B
    @test isnothing(lc.A_eq)
    @test isnothing(lc.B_eq)
    lce = LinearConstraint(; eq = p2)
    @test isnothing(lce.A_ineq)
    @test isnothing(lce.B_ineq)
    @test lce.A_eq === p2.A
    @test lce.B_eq === p2.B

    # Two or more elements take the merging branch, and the halves concatenate in input
    # order.
    both = PortfolioOptimisers.merge_linear_constraints([lc, lce])
    @test size(both.A_ineq, 1) == 2
    @test size(both.A_eq, 1) == 1
    @test both.B_ineq == [1.0, 2.0]
    @test both.B_eq == [3.0]

    # A precomputed constraint is carried through a sub-selection unchanged: the row is
    # written over the whole universe it was assembled against, so slicing `A` would
    # change what the row asserts. A row wider than the weight vector then fails loudly
    # at model build time; `NestedClustered` refuses a bare precomputed constraint for
    # exactly this reason.
    @test PortfolioOptimisers.port_opt_view(lc, [1, 2]) === lc

    # An already-assembled constraint, `nothing`, and a vector of constraints pass
    # through `linear_constraints` untouched.
    @test linear_constraints(lc, sets) === lc
    @test isnothing(linear_constraints(nothing, sets))
    v = [lc, lce]
    @test linear_constraints(v, nothing) === v

    # A row carrying a name that matched nothing is dropped whole (ADR 0125), so a
    # constraint that keeps no row at all answers `nothing` rather than an empty matrix.
    # One warning, not two: the row goes at the name, so it never reaches the empty-row
    # report that used to say the same thing a second time.
    @test isnothing(@test_logs (:warn,) linear_constraints("Z <= 1", sets))
    @test occursin("row dropped",
                   PortfolioOptimisers.unknown_variable_msg("Z", ["A", "B", "C"], "nx";
                                                            consequence = "row dropped"))
    @test_throws ArgumentError linear_constraints("Z <= 1", sets; strict = true)

    # A missing asset universe suggests only keys no other declared axis speaks for, so a
    # dict carrying a feature axis alone is not answered with the feature key.
    @test_throws KeyError UniverseSets(; dict = Dict("nxx" => ["A"], "nz" => ["z"]))
    @test PortfolioOptimisers.unclaimed_sets_keys(Dict("nxx" => ["A"], "nz" => ["z"]),
                                                  ("ux", "nf", "uf", "nz")) == ["nxx"]

    # `universe_axis` reads the axis off the key, so a factor universe is named as one.
    fsets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "nf" => ["F1", "F2"]))
    @test PortfolioOptimisers.universe_axis(fsets, "nx") == "asset"
    @test PortfolioOptimisers.universe_axis(fsets, "nf") == "factor"
    @test PortfolioOptimisers.universe_axis(fsets, "nf_style") == "factor"
    # The cross-sectional factor axis is a factor axis too, and `"ncf"` does not start with
    # `"nf"`, so the key's own prefix is what names it. Issue #1273.
    @test PortfolioOptimisers.universe_axis(fsets, "ncf") == "factor"
    @test PortfolioOptimisers.universe_axis(fsets, "ncf_style") == "factor"

    # `factor_universe` raises at the point of need, naming the axis it reads and the matrix
    # it reconciles it against. It takes the key positionally, because `UniverseSets`
    # declares two factor axes and a consumer has to say which one it means.
    @test_throws KeyError PortfolioOptimisers.factor_universe(sets, "nf", 2, "a test",
                                                              "rr.M")
    @test_throws DimensionMismatch PortfolioOptimisers.factor_universe(fsets, "nf", 3,
                                                                       "a test", "rr.M")
    @test PortfolioOptimisers.factor_universe(fsets, "nf", 2, "a test", "rr.M") ==
          ["F1", "F2"]

    # `estimator_to_val` answers each shape it is given against the same universe.
    @test PortfolioOptimisers.estimator_to_val(nothing, sets) === nothing
    @test PortfolioOptimisers.estimator_to_val(0.7, sets) == 0.7
    @test PortfolioOptimisers.estimator_to_val([1.0, 2.0, 3.0], sets) == [1.0, 2.0, 3.0]
    @test_throws DimensionMismatch PortfolioOptimisers.estimator_to_val([1.0, 2.0], sets)
    # The matrix method validates `size(val, dims)`, and `dims` is 2 by default, so the
    # universe names the *columns* of the matrix.
    mat = [1.0 2.0 3.0; 4.0 5.0 6.0]
    @test PortfolioOptimisers.estimator_to_val(mat, sets) === mat
    tmat = transpose(mat)
    @test PortfolioOptimisers.estimator_to_val(tmat, sets; dims = 1) === tmat
    @test_throws DimensionMismatch PortfolioOptimisers.estimator_to_val(tmat, sets)
    # `UniformValues` answers a range rather than a vector, and its entries sum to one.
    for N in (3, 7, 10)
        uv = PortfolioOptimisers.estimator_to_val(UniformValues(),
                                                  UniverseSets(;
                                                               dict = Dict("nx" =>
                                                                               string.(1:N))))
        @test uv isa AbstractRange
        @test length(uv) == N
        @test sum(uv) == 1.0
    end
    # Two entries naming overlapping groups let the last write win.
    osets = UniverseSets(;
                         dict = Dict("nx" => ["A", "B", "C"], "g1" => ["A", "B"],
                                     "g2" => ["B", "C"]))
    @test PortfolioOptimisers.estimator_to_val(["g1" => 1.0, "g2" => 2.0], osets) ==
          [1.0, 2.0, 2.0]
    @test PortfolioOptimisers.estimator_to_val(["g2" => 2.0, "g1" => 1.0], osets) ==
          [1.0, 1.0, 2.0]
end
@testset "An estimator value algorithm is a value, not an equation (issue #633)" begin
    #=
    `AbstractEstimatorValueAlgorithm` splits by the shape of the value the algorithm
    returns. `UniformValues` returns a `Num_VecNum`, so it sits under
    `VectorAbstractEstimatorValueAlgorithm`, and a slot that resolves a value admits that
    branch by writing it into its own bound.

    An equation slot admits no branch at all. `LinearConstraintEstimator` used to accept
    one, because `EqnType` carried the family root, and the method that answered the
    resulting estimator returned the numeric value `estimator_to_val` computes. That value
    is not a `LinearConstraint`: it carries neither a comparison operator nor a side, so no
    constraint row can be assembled from it. The return annotation refused it, and the
    estimator raised on every call it could receive. The bound now refuses the estimator
    itself, at the slot the caller wrote.
    =#
    sets = UniverseSets(; xkey = "nx", dict = Dict("nx" => ["A", "B", "C", "D"]))
    @test UniformValues <: PortfolioOptimisers.VectorAbstractEstimatorValueAlgorithm
    @test UniformValues <: PortfolioOptimisers.AbstractEstimatorValueAlgorithm
    @test UniformValues <:
          PortfolioOptimisers.EstValType{<:PortfolioOptimisers.VectorAbstractEstimatorValueAlgorithm}
    @test !(UniformValues <: PortfolioOptimisers.EqnType)
    # The keyword route names the slot that refused the value; the positional route has no
    # method for it.
    @test_throws TypeError LinearConstraintEstimator(; val = UniformValues())
    @test_throws MethodError LinearConstraintEstimator(UniformValues())
    # Equation text is unaffected, and so is a precomputed constraint.
    lce = LinearConstraintEstimator(; val = ["A + B == 1", "A >= 0.1"])
    @test linear_constraints(lce, sets) isa LinearConstraint
    # Every slot that resolves a value keeps the algorithm, and resolves it to the uniform
    # vector over the universe.
    wb = weight_bounds_constraints(WeightBoundsEstimator(; ub = UniformValues()), sets)
    @test isapprox(collect(wb.ub), fill(0.25, 4))
    rkb = risk_budget_constraints(RiskBudgetEstimator(; val = UniformValues()), sets)
    @test isapprox(collect(rkb.val), fill(0.25, 4))
    @test ThresholdEstimator(; val = UniformValues()).val === UniformValues()
    @test TurnoverEstimator(; w = fill(0.25, 4), val = UniformValues()).val ===
          UniformValues()
    @test FeesEstimator(; l = UniformValues()).l === UniformValues()
    # `PortfolioTarget`'s `w` was such a slot. It now names the pair of shapes the
    # regime-adjusted fit can honour, so it refuses the algorithm at the keyword.
    @test_throws TypeError PortfolioOptimisers.PortfolioTarget(; w = UniformValues())
end
