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
    # The derived big-M constant of #1323 moved these weights. The old ones held a VaR of
    # 0.013824, and these hold 0.013200, because b = 1000 stopped the second cluster short.
    @test isapprox(res.w,
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32515447570275585, 0.0,
                    0.3086086955000851, 0.04387217515833515, 0.1125361601310135, 0.0, 0.0,
                    0.08740615267221835, 0.0, 0.0, 0.0, 0.12242234083559198, 0.0],
                   rtol = 1e-6)
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

    # A MIP range is two blocks of `T` binaries, so this case runs on the short window.
    res = optimise(NestedClustered(; cle = clr_mip,
                                   opti = MeanRisk(; r = ValueatRiskRange(),
                                                   opt = JuMPOptimiser(; pe = pr_mip,
                                                                       slv = mip_slv,
                                                                       sets = sets)),
                                   opto = MeanRisk(; r = ValueatRiskRange(),
                                                   opt = JuMPOptimiser(; slv = mip_slv))),
                   rd_mip)
    @test isapprox(res.w,
                   [0.0, 0.0, 0.0, 0.003306398447882896, 0.020909276394013954, 0.0, 0.0,
                    0.5126835695993831, 0.07670087618420798, 0.0011238748494735982,
                    0.008282356039216072, 0.1721357232487124, 0.0, 0.017523225424448376,
                    0.0, 0.010000303504920214, 0.0, 0.08659447685499204,
                    0.05294777837182523, 0.037792141080924116], rtol = 1e-6)

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
