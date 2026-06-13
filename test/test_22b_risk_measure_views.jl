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
    if Sys.isapple()
        @test isapprox(res.w,
                       [0.0, 0.0, 0.004821476046698469, 0.0010258936044387464, 0.0, 0.0,
                        0.006537279371877492, 0.18088433701440654, 0.005298988291077931,
                        0.3472528756459372, 0.0, 0.2402405305712111, 0.00033257997539444,
                        0.05000349773341354, 0.03242526200848679, 0.0, 0.0, 0.0,
                        0.1278243058653844, 0.0033529738716733854], rtol = 1e-6)
    else
        @test isapprox(res.w,
                       [0.0, 0.0, 0.005602412808720073, 0.0011920580781122523, 0.0, 0.0,
                        0.007596125612252459, 0.2375180560536035, 0.006157267937796995,
                        0.3435374666842433, 0.011940000819969961, 0.20302112082396714,
                        0.0003864481117456734, 0.0, 0.04878243497008711, 0.0, 0.0, 0.0,
                        0.1303705514595793, 0.0038960566399224694], rtol = 1e-6)
    end

    res = optimise(NestedClustered(; cle = clr,
                                   opti = MeanRisk(; r = DrawdownatRisk(),
                                                   opt = JuMPOptimiser(; pe = pr,
                                                                       slv = mip_slv,
                                                                       sets = sets)),
                                   opto = MeanRisk(; r = DrawdownatRisk(),
                                                   opt = JuMPOptimiser(; slv = mip_slv))),
                   rd)
    @test isapprox(res.w,
                   [0.0012234700209226132, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005385255722520739,
                    0.281628194040886, 0.0, 0.029236643246224468, 0.13170514517674964,
                    0.16228108868342814, 0.0, 0.23112812156090545, 0.0, 0.0, 0.0,
                    0.09158493164822426, 0.06963119505448694, 0.0010426849959224247],
                   rtol = 1e-6)

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
                   [0.00021148772894721179, 0.0, 0.0, 5.503641791343221e-5, 0.0,
                    0.0003957305274730832, 0.0011574175816340929, 0.0, 0.002724254691574924,
                    0.19286877092707, 0.16396114472674364, 0.0, 0.0, 0.3005855067078481,
                    0.0, 0.0, 0.0, 0.0, 0.3352086932338209, 0.0028319574569745654],
                   rtol = 1e-6)

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
