include(joinpath(@__DIR__, "test22_setup.jl"))

@testset "Risk measure views" begin
    ucse = NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(987654321),
                                alg = BoxUncertaintySetAlgorithm())
    ucs = sigma_ucs(ucse, rd.X)
    jopti = JuMPOptimiser(; pe = pr, slv = slv, sets = sets)
    jopto = JuMPOptimiser(; slv = slv,
                          pe = HighOrderPriorEstimator(;
                                                       ske = Coskewness(;
                                                                        mp = MatrixProcessing(;
                                                                                              pdm = nothing))))

    resa = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(; r = Kurtosis(; mu = pr.mu),
                                                    opt = jopti),
                                    opto = MeanRisk(; r = Kurtosis(), opt = jopto)), rd)
    resb = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(; r = Kurtosis(; kt = pr.kt),
                                                    opt = jopti),
                                    opto = MeanRisk(; r = Kurtosis(), opt = jopto)), rd)
    @test resa.w == resb.w

    resa = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(;
                                                    r = UncertaintySetVariance(;
                                                                               ucs = ucse),
                                                    opt = jopti),
                                    opto = MeanRisk(;
                                                    r = UncertaintySetVariance(;
                                                                               ucs = ucse),
                                                    opt = jopto)), rd)
    resb = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(;
                                                    r = UncertaintySetVariance(; ucs = ucs),
                                                    opt = jopti),
                                    opto = MeanRisk(;
                                                    r = UncertaintySetVariance(;
                                                                               ucs = ucse),
                                                    opt = jopto)), rd)
    @test resa.w != resb.w

    resa = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(;
                                                    r = LowOrderMoment(;
                                                                       alg = MeanAbsoluteDeviation()),
                                                    opt = jopti),
                                    opto = MeanRisk(;
                                                    r = LowOrderMoment(;
                                                                       alg = MeanAbsoluteDeviation()),
                                                    opt = jopto)), rd)
    resb = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(;
                                                    r = LowOrderMoment(; mu = pr.mu,
                                                                       alg = MeanAbsoluteDeviation()),
                                                    opt = jopti),
                                    opto = MeanRisk(;
                                                    r = LowOrderMoment(;
                                                                       alg = MeanAbsoluteDeviation()),
                                                    opt = jopto)), rd)
    @test resa.w == resb.w

    resa = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(; r = NegativeSkewness(;), opt = jopti),
                                    opto = MeanRisk(; r = NegativeSkewness(;), opt = jopto)),
                    rd)
    resb = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(;
                                                    r = NegativeSkewness(; sk = pr.sk,
                                                                         V = pr.V),
                                                    opt = jopti),
                                    opto = MeanRisk(; r = NegativeSkewness(), opt = jopto)),
                    rd)
    @test resa.w == resb.w

    res = optimise(NestedClustered(; cle = clr,
                                   opti = MeanRisk(; r = ValueatRisk(),
                                                   opt = JuMPOptimiser(; pe = pr,
                                                                       slv = mip_slv,
                                                                       sets = sets)),
                                   opto = MeanRisk(; r = ValueatRisk(),
                                                   opt = JuMPOptimiser(; slv = mip_slv))),
                   rd)
    @test isapprox(res.w,
                   [0.0, 0.0, 0.05185452300759591, 0.011033389567697087, 0.0, 0.0,
                    0.07030782696268922, 0.21987054724580155, 0.056990122442879244,
                    0.20922287906742792, 0.0, 0.19272555326079038, 0.00357686646553502,
                    0.05051771647808476, 0.0, 0.0, 0.0, 0.0, 0.09783965575644371,
                    0.036060919745055015], rtol = 1e-6)
    res = optimise(NestedClustered(; cle = clr,
                                   opti = MeanRisk(; r = DrawdownatRisk(),
                                                   opt = JuMPOptimiser(; pe = pr,
                                                                       slv = mip_slv,
                                                                       sets = sets)),
                                   opto = MeanRisk(; r = DrawdownatRisk(),
                                                   opt = JuMPOptimiser(; slv = mip_slv))),
                   rd)
    @test isapprox(res.w,
                   [-6.780288742096869e-16, 0.0, 0.0, 0.0, 0.0, 0.0,
                    -2.9844285607531957e-16, 0.28242029275393987, 0.0, 0.029318873321122395,
                    0.13207557497818467, 0.16273751543408388, 0.0, 0.23177818533829975, 0.0,
                    0.0, 0.0, 0.09184252058295696, 0.06982703759141368,
                    -5.778405041902124e-16], rtol = 1e-6)

    resa = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(;
                                                    r = ValueatRisk(;
                                                                    alg = DistributionValueatRisk()),
                                                    opt = JuMPOptimiser(; pe = pr,
                                                                        slv = slv,
                                                                        sets = sets)),
                                    opto = MeanRisk(;
                                                    r = ValueatRisk(;
                                                                    alg = DistributionValueatRisk()),
                                                    opt = JuMPOptimiser(; slv = slv))), rd)
    resb = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(;
                                                    r = ValueatRisk(;
                                                                    alg = DistributionValueatRisk(;
                                                                                                  mu = pr.mu,
                                                                                                  sigma = pr.sigma)),
                                                    opt = JuMPOptimiser(; pe = pr,
                                                                        slv = slv,
                                                                        sets = sets)),
                                    opto = MeanRisk(;
                                                    r = ValueatRisk(;
                                                                    alg = DistributionValueatRisk()),
                                                    opt = JuMPOptimiser(; slv = slv))), rd)
    @test resa.w == resb.w

    res = optimise(NestedClustered(; cle = clr,
                                   opti = MeanRisk(; r = ValueatRiskRange(),
                                                   opt = JuMPOptimiser(; pe = pr,
                                                                       slv = mip_slv,
                                                                       sets = sets)),
                                   opto = MeanRisk(; r = ValueatRiskRange(),
                                                   opt = JuMPOptimiser(; slv = mip_slv))),
                   rd)
    @test isapprox(res.w,
                   [0.0002143701113304258, 0.0, 0.0, 5.5786513449552756e-5, 0.0,
                    0.0004011239690056024, 0.0011731921141047454, 0.0,
                    0.0027613837664847867, 0.19284923843692664, 0.16394453981228352, 0.0,
                    0.0, 0.3005550653698399, 0.0, 0.0, 0.0, 0.0, 0.33517474548548876,
                    0.0028705544210860263], rtol = 1e-6)

    resa = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(;
                                                    r = ValueatRiskRange(;
                                                                         alg = DistributionValueatRisk()),
                                                    opt = JuMPOptimiser(; pe = pr,
                                                                        slv = slv,
                                                                        sets = sets)),
                                    opto = MeanRisk(;
                                                    r = ValueatRiskRange(;
                                                                         alg = DistributionValueatRisk()),
                                                    opt = JuMPOptimiser(; slv = slv))), rd)
    resb = optimise(NestedClustered(; cle = clr,
                                    opti = MeanRisk(;
                                                    r = ValueatRiskRange(;
                                                                         alg = DistributionValueatRisk(;
                                                                                                       mu = pr.mu,
                                                                                                       sigma = pr.sigma)),
                                                    opt = JuMPOptimiser(; pe = pr,
                                                                        slv = slv,
                                                                        sets = sets)),
                                    opto = MeanRisk(;
                                                    r = ValueatRiskRange(;
                                                                         alg = DistributionValueatRisk()),
                                                    opt = JuMPOptimiser(; slv = slv))), rd)
    @test resa.w == resb.w

    res = optimise(NestedClustered(; cle = clr,
                                   opti = MeanRisk(; r = TurnoverRiskMeasure(; w = w0),
                                                   opt = jopti),
                                   opto = MeanRisk(;
                                                   r = TurnoverRiskMeasure(;
                                                                           w = fill(1 / 2,
                                                                                    2)),
                                                   opt = jopto)), rd)
    @test isapprox(res.w,
                   [0.045454545461020436, 0.04545454545389754, 0.04545454545389754,
                    0.04545454545389754, 0.04545454545389754, 0.04545454545389754,
                    0.04545454545389754, 0.055555555478081366, 0.04545454545389754,
                    0.05555555556523971, 0.05555555556523971, 0.05555555556523971,
                    0.04545454545389754, 0.05555555556523971, 0.05555555556523971,
                    0.05555555556523971, 0.04545454545389754, 0.05555555556523971,
                    0.05555555556523971, 0.04545454545389754], rtol = 1e-6)
end
