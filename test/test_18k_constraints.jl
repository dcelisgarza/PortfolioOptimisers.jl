include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Cardinality" begin
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, card = 3)
    mre = MeanRisk(; opt = opt)
    res = optimise(mre)
    w = res.w
    @test count(w .> 1e-10) <= 3

    opt = JuMPOptimiser(; l2 = 0.1, pe = pr, slv = mip_slv, card = 3)
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

@testset "Phylogeny" begin
    plc = IntegerPhylogenyEstimator(; pl = NetworkEstimator(), B = 1)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1), l2 = 0.001)
    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt))
    @test all(JuMP.value.(res.plr.A * res.model[:ib]) .<= res.plr.B)
    idx = [BitVector(res.plr.A[:, i]) for i in axes(res.plr.A, 2)]
    @test all([(count(abs.(getindex(res.w, i)) .> 1e-10) <= 1) for i in idx])
    @test (isapprox(res.w,
                    [-5.83349251195602e-14, -0.7549102373927532, -0.24489685036355585,
                     2.5425310153855233e-16, 1.177641184405827e-13, -1.1761565067160182e-14,
                     -3.8671847486726096e-14, 1.2944646679137976e-14,
                     -1.9475087209248426e-14, 0.26951016407394074, 8.367112463199218e-14,
                     0.7303004971078252, -6.194837681595708e-14, 2.309812136692163e-14,
                     -2.5460214715889667e-14, -8.58412864279287e-15, 1.1757019029014802e-13,
                     2.1546906367143115e-14, 5.7299916350654655e-15, 0.9999964265744166];
                    rtol = 1e-6) || isapprox(res.w,
                                             [-4.400162786139801e-14, -0.7068647775696704, -0.20178701626483694,
                                              -1.0233912058364145e-15, 8.336251924927799e-14,
                                              -1.0638713372805883e-14, -0.09129358185248605, 4.326412562038434e-15,
                                              -1.626576456755123e-14, 0.2751905375771391, 5.428666085696465e-14,
                                              0.7247587488302388, -4.6429454132911124e-14, 1.1792066087999556e-14,
                                              -2.2237835848208057e-14, -1.100539171494297e-14, 8.51166817026479e-14,
                                              1.0879558335436499e-14, 8.09956925306327e-16, 0.9999960892795525];
                                             rtol = 1e-6))

    plc = IntegerPhylogenyEstimator(; pl = NetworkEstimator(), B = fill(2, size(pr.X, 2)))
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1), l2 = 0.0001)
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
                        wb = WeightBounds(; lb = -1, ub = 1), l2 = 0.0001)
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

    plc = SemiDefinitePhylogenyEstimator(; pl = clr)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    res = optimise(MeanRisk(; obj = MinimumRisk(), opt = opt))
    @test isapprox(JuMP.value.(res.plr.A .* res.model[:W]), zeros(size(pr.sigma)),
                   atol = 1e-10)

    plc = SemiDefinitePhylogenyEstimator(; pl = ClustersEstimator(), p = 1000)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    @test isapprox(res.w, optimise(MeanRisk(; obj = MinimumRisk(), opt = opt)).w)

    plc = phylogeny_constraints(SemiDefinitePhylogenyEstimator(; pl = clr, p = 10), rd.X)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1, ple = plc,
                        wb = WeightBounds(; lb = -1, ub = 1))
    @test isapprox(res.w, optimise(MeanRisk(; obj = MinimumRisk(), opt = opt)).w)

    plc = SemiDefinitePhylogenyEstimator(; pl = clr)
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

    plc = SemiDefinitePhylogenyEstimator(; pl = clr, p = 5)
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

    plc = SemiDefinitePhylogenyEstimator(; pl = clr)
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

    plc = SemiDefinitePhylogenyEstimator(; pl = clr, p = 5)
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
                                           alg = L1Tracking()))
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

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 4.5e-3,
                                           alg = LpTracking()))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise(mre)
    @test LinearAlgebra.norm(rd.X * res.w - wr, 3) / cbrt(size(rd.X, 1)) <= 4.5e-3

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = TrackingError(; tr = ReturnsTracking(; w = wr), err = 8e-5,
                                           alg = LInfTracking()))
    mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
    res = optimise(mre)
    @test LinearAlgebra.norm(rd.X * res.w - wr, Inf) / size(rd.X, 1) <= 8e-5

    opt = JuMPOptimiser(; pe = pr, slv = slv,
                        tr = [TrackingError(; tr = WeightsTracking(; w = w0), err = 2e-3,
                                            alg = L1Tracking())])
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
