include(joinpath(@__DIR__, "test18_setup.jl"))

@testset "Scalarisers" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
    w0_1 = optimise(MeanRisk(; r = r[1], opt = opt), rd).w
    w0_2 = optimise(MeanRisk(; r = r[2], opt = opt), rd).w

    mr = MeanRisk(; r = r, opt = opt)
    w1 = optimise(mr, rd).w
    @test isapprox(w1,
                   [1.7074698994991376e-10, 8.433104973224101e-11, 1.9860067068611146e-9,
                    3.1027970850564853e-10, 0.09898828657505643, 0.0038933979989648256,
                    6.636136386259997e-10, 0.35967323582529487, 0.012150886312492075,
                    0.10067191014130493, 2.964806359249512e-10, 0.14595634518390374,
                    3.891455764455336e-10, 0.14289553677354302, 5.362751594199594e-10,
                    0.03009102863550913, 1.1386315235757199e-10, 1.1094461786914318e-9,
                    0.07216633866035838, 0.033513028233384], rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sca = MaxScalariser())
    mr = MeanRisk(; r = r, opt = opt)
    w2 = optimise(mr, rd).w
    @test isapprox(w2, w0_1, rtol = 5e-4)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sca = LogSumExpScalariser(; gamma = 1e-3))
    mr = MeanRisk(; r = r, opt = opt)
    w3 = optimise(mr, rd).w
    @test isapprox(w3, w1, rtol = 5e-2)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sca = LogSumExpScalariser(; gamma = 1e5))
    mr = MeanRisk(; r = r, opt = opt)
    w4 = optimise(mr, rd).w
    @test isapprox(w4, w2, rtol = 1e-4)
end

@testset "Arithmetic return uncertainty set" begin
    rng = StableRNG(123456789)
    ucs1 = mu_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng,
                                       alg = BoxUncertaintySetAlgorithm()), pr.X)
    ucs2 = mu_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = rng,
                                       alg = EllipsoidalUncertaintySetAlgorithm()), pr.X)
    ucss = [ucs1, ucs2]
    objs = [MinimumRisk(), MaximumRatio(; rf = rf), MaximumReturn()]
    df = CSV.read(joinpath(@__DIR__, "./assets/MeanRiskUncertainty.csv.gz"), DataFrame)
    i = 1
    for ucs in ucss
        for obj in objs
            ret = ArithmeticReturn(; ucs = ucs)
            opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
            mre = MeanRisk(; obj = obj, opt = opt)
            res = optimise(mre)
            @test isa(res.retcode, OptimisationSuccess)
            rtol = 1e-6
            success = isapprox(res.w, df[!, i]; rtol = rtol)
            if !success
                println("Counter: $i")
                find_tol(res.w, df[!, i])
            end
            @test success
            i += 1
        end
    end
end

@testset "Weight bounds" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBoundsEstimator(; lb = ["group1" => -1, "group2" => 0.1],
                                                   ub = Dict("group1" => -0.1,
                                                             "group2" => 1)))
    mr = MeanRisk(; opt = opt)
    pred = fit_predict(mr, rd)
    res1 = pred.res
    @test pred.rd.iv == rd.iv * abs.(pred.res.w)
    @test pred.rd.ivpa == dot(rd.ivpa, abs.(pred.res.w))
    @test isapprox(sum(res1.w), 1)
    @test isapprox(sum(res1.w[res1.w .< zero(eltype(res1.w))]), -1)
    @test isapprox(sum(res1.w[res1.w .>= zero(eltype(res1.w))]), 2)
    @test all(res1.pa.wb.lb[1:2:end] .<= res1.w[1:2:end])
    @test all(abs.(res1.w[1:2:end] .- res1.pa.wb.ub[1:2:end]) .< 5e-10)
    @test all(res1.pa.wb.lb[2:2:end] .<= res1.w[2:2:end] .<= res1.pa.wb.ub[2:2:end])
end

@testset "Buy-in threshold" begin
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                        lt = ThresholdEstimator(; val = ["WMT" => 0.23, "group2" => 0.48]),
                        sets = sets)
    mre = MeanRisk(; opt = opt)
    res = optimise(mre)
    @test res.w[findfirst(x -> x == "WMT", rd.nx)] >= 0.23
    @test res.w[2:2:end][res.w[2:2:end] .> 1e-9][1] >= 0.48

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = Threshold(; val = 0.15), sets = sets)
    mre = MeanRisk(; opt = opt)
    res = optimise(mre)
    res.w[res.w .>= 1e-10] .>= 0.15

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                        lt = Threshold(; val = fill(0.15, size(pr.X, 2))), sets = sets)
    mre = MeanRisk(; opt = opt)
    @test isapprox(res.w, optimise(mre).w)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, st = Threshold(; val = 0.25),
                        lt = Threshold(; val = 0.4), sets = sets)
    mre = MeanRisk(; opt = opt)
    res = optimise(mre)
    @test all(res.w[res.w .> 0][res.w[res.w .>= 0] .>= 1e-10] .>= 0.4)
    if Sys.isapple()
        @test all(res.w[res.w .< 0][res.w[res.w .< 0] .<= -1e-10] .<= -0.25 + sqrt(eps()))
    else
        @test all(res.w[res.w .< 0][res.w[res.w .< 0] .<= -1e-10] .<= -0.25)
    end

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, wb = WeightBounds(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, st = Threshold(; val = 0.25),
                        lt = Threshold(; val = 0.4), sets = sets)
    mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
    res = optimise(mre)
    @test all(res.w[res.w .> 0][res.w[res.w .>= 0] .>= 1e-10] .>= 0.4)
    @test all(res.w[res.w .< 0][res.w[res.w .< 0] .<= -1e-10] .<= -0.25)
end

@testset "Linear weight constraints" begin
    ivpa = rand(StableRNG(123))
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end];
                           B = TimeArray(CSV.File(joinpath(@__DIR__,
                                                           "./assets/SP500_idx.csv.gz"));
                                         timestamp = :Date), iv = iv, ivpa = ivpa)
    pr = prior(HighOrderPriorEstimator(), rd)

    sets.dict["group4"] = ["AMD", "BAC"]
    sets.dict["group5"] = ["PG", "XOM"]
    lcs = LinearConstraintEstimator(;
                                    val = ["AAPL >= 0.2*MRK", "group4 >= 3*group5",
                                           "RRC <=-0.02", "MSFT==0.01"])
    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), lcse = lcs)
    mr = MeanRisk(; obj = MinimumRisk(), opt = opt)
    pred = fit_predict(mr, rd)
    res = pred.res
    @test pred.rd.ivpa == ivpa
    @test res.w[findfirst(x -> x == "AAPL", rd.nx)] >=
          0.2 * res.w[findfirst(x -> x == "MRK", rd.nx)]
    @test sum(res.w[[findfirst(x -> x == a, rd.nx) for a in sets.dict["group4"]]]) >=
          3 * sum(res.w[[findfirst(x -> x == a, rd.nx) for a in sets.dict["group5"]]])
    @test res.w[findfirst(x -> x == "RRC", rd.nx)] <= -0.02
    @test isapprox(res.w[findfirst(x -> x == "MSFT", rd.nx)], 0.01, rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), lcse = res.lcsr)
    @test isapprox(res.w, optimise(MeanRisk(; obj = MinimumRisk(), opt = opt)).w)

    @test isnothing(linear_constraints(LinearConstraintEstimator(; val = ["FOO >= 0.2"]),
                                       sets; strict = false))
end

@testset "Regularisation" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), l1 = 5e-6)
    mr = MeanRisk(; opt = opt)
    res = optimise(mr)
    @test isapprox(res.w,
                   [-0.08259934606632031, -0.01735523094219388, -6.306917873785564e-9,
                    -0.019964241234679967, 0.08433288155125772, 0.04082860469386889,
                    0.0245444126670805, 0.38468746514362956, 0.025975231635603585,
                    0.11323392647659213, -0.0455153966134236, 0.1769386350615763,
                    0.04466740435373562, 0.12535190765089996, -0.01382516940599513,
                    0.021879232823390448, -0.01259642910877665, -3.789411268851941e-9,
                    0.09356110720420865, 0.05585501420587528], rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), l1 = 1)
    mr = MeanRisk(; opt = opt)
    res = optimise(mr)
    @test isapprox(res.w,
                   [4.1633941065577117e-7, 2.6448742701527e-7, 4.477941406289695e-6,
                    4.854555411641452e-7, 0.07424358812530261, 0.00792596837274629,
                    1.352652412857295e-6, 0.36980684793744134, 0.007584348893331609,
                    0.11118617081059556, 3.753498265381272e-7, 0.17467158275776748,
                    9.348289673318389e-7, 0.08974785581907543, 8.391727846180317e-7,
                    0.023971871590369637, 3.3459673922392774e-7, 1.3947538587917836e-6,
                    0.0935238000646315, 0.04732709005036381], rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), l2 = 5e-6)
    mr = MeanRisk(; opt = opt)
    res = optimise(mr)
    @test isapprox(res.w,
                   [-0.11064349447254161, -0.017854196186798618, -0.04397298675384868,
                    -0.03145351033102496, 0.09495263689036014, 0.05053896454111151,
                    0.03699197653220963, 0.38114357883139005, 0.06929867727498179,
                    0.11865327291902569, -0.06633836583807416, 0.19538041855641222,
                    0.07381248663128888, 0.13328230798627264, -0.037943963731425036,
                    0.029972928788686973, -0.017281968260805896, -0.008700415062896462,
                    0.09297968577516197, 0.05718196591051368], rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), l2 = 1e-4)
    mr = MeanRisk(; opt = opt)
    res = optimise(mr)
    @test isapprox(res.w,
                   [-0.03076204518648503, -0.03965358415790933, 0.01741465312222848,
                    -0.014071169950968142, 0.08160008886554988, 0.038967358792986795,
                    0.03853758826706406, 0.17170470580976588, 0.03806574632010554,
                    0.10075437963820232, 0.024679044550725296, 0.1375839532075256,
                    0.024805294188283956, 0.10194818468441112, 0.03329884602273654,
                    0.08767866540531458, -0.0157785635897723, 0.03858247320319518,
                    0.09776343657139772, 0.06688094423564199], rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), l1 = 5e-6, l2 = 5e-6)
    mr = MeanRisk(; opt = opt)
    res = optimise(mr)
    @test isapprox(res.w,
                   [-0.07591829609679655, -0.019207462900668874, -4.164287783441495e-9,
                    -0.019697945739295775, 0.08623890830132815, 0.03970246984743602,
                    0.024849760269582157, 0.3562105498789526, 0.028493593064519384,
                    0.11264585349973763, -0.03577564886530365, 0.1762813917849517,
                    0.03994057834010078, 0.12239469833440599, -0.007813742040459417,
                    0.03415102550994622, -0.01302979483152005, -5.177412500119382e-10,
                    0.09499371625016041, 0.05554035007495228], rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), l1 = 1, l2 = 1e-4)
    mr = MeanRisk(; opt = opt)
    res = optimise(mr)
    @test isapprox(res.w,
                   [2.0232629822283223e-6, 9.760082045106613e-7, 0.0025796058703603294,
                    2.436499442593761e-6, 0.07216922707119579, 0.017185343248064092,
                    0.012997513915337116, 0.17907277860116347, 0.03127342741481768,
                    0.09864631045614426, 0.021928766640300995, 0.15045440958368198,
                    4.894112955006514e-6, 0.09912274998209057, 0.035519395576107456,
                    0.08660290480901828, 1.711128248182528e-6, 0.03477208922064647,
                    0.09904023885895782, 0.05862319774028108], rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1),
                        lp = LpRegularisation(; val = 1e-4))
    mr = MeanRisk(; opt = opt)
    res = optimise(mr)
    @test isapprox(res.w,
                   [-0.05764697001130279, -0.04431070972751474, 0.00747553535021098,
                    -0.023908797442917386, 0.08821875094947536, 0.046977548770644396,
                    0.04908160016925732, 0.15439190880914685, 0.04884651182287862,
                    0.10640175439005223, 0.02159205126254584, 0.13329454362025117,
                    0.042156711769183466, 0.10774681701781148, 0.03202888209996061,
                    0.09501657403907941, -0.022310182501551167, 0.04164013305521783,
                    0.1011939820035937, 0.07211335193158373], rtol = 5e-5)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), linf = 8e-5)
    mr = MeanRisk(; opt = opt)
    res = optimise(mr)
    @test isapprox(res.w,
                   [-0.10012716916989911, -0.037330227223905686, -0.0362005729255792,
                    -0.03437255975979355, 0.1297453778361631, 0.048219568550521216,
                    0.04271321394187886, 0.12974638519965564, 0.0860548071819024,
                    0.1297463758040975, 0.01047545372694966, 0.1297463839813271,
                    0.0643702863005255, 0.12974637704447126, 0.015476757316173454,
                    0.1297463695755112, -0.02667527421008015, 0.020454176815837947,
                    0.1176102092952984, 0.05085406071894451], rtol = 1e-6)
end

@testset "Turnover" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), tn = Turnover(; w = w0))
    @test PortfolioOptimisers.needs_previous_weights(opt)
    mr = MeanRisk(; opt = opt)
    @test PortfolioOptimisers.needs_previous_weights(mr)
    @test isapprox(w0, optimise(mr).w)

    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1),
                        tn = [TurnoverEstimator(; w = w0,
                                                val = ["AAPL" => 0, "MRK" => 0.05],
                                                dval = Inf)])
    @test PortfolioOptimisers.needs_previous_weights(opt)
    mr = MeanRisk(; opt = opt)
    @test PortfolioOptimisers.needs_previous_weights(mr)
    res = optimise(mr)
    @test isapprox(res.w,
                   [0.049999999999993605, -0.04520519208370696, -0.04869401288598486,
                    -0.04327195569636555, 0.11569334228575458, 0.0414229664404061,
                    0.03303019511733334, 0.4429818998364995, 0.07583334507776132,
                    0.10762664262865197, -0.04825139998117526, 0.09999541531046201,
                    -0.0016373970470112344, 0.10510925375967227, -0.023144445601095607,
                    0.043540146321535515, -0.01798434663243727, -0.023893575203165842,
                    0.09199157017254597, 0.04485754818032653], rtol = 1e-6)
end

@testset "Number of effective assets" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, nea = 10)
    res = optimise(MeanRisk(; obj = MinimumRisk(), opt = opt))
    @test round(inv(LinearAlgebra.dot(res.w, res.w))) >= 10

    opt = JuMPOptimiser(; pe = pr, slv = slv, nea = 15)
    res = optimise(MeanRisk(; obj = MaximumUtility(), opt = opt))
    @test round(inv(LinearAlgebra.dot(res.w, res.w))) >= 15
end

@testset "Centrality" begin
    ces = [CentralityConstraint(; A = CentralityEstimator(), B = MinValue(), comp = >=),
           CentralityConstraint(; A = CentralityEstimator(; ct = EigenvectorCentrality()),
                                B = MeanValue(), comp = <=),
           CentralityConstraint(; A = CentralityEstimator(; ct = ClosenessCentrality()),
                                B = MedianValue(), comp = ==),
           CentralityConstraint(; A = CentralityEstimator(; ct = StressCentrality()),
                                B = MaxValue(), comp = ==),
           CentralityConstraint(; A = CentralityEstimator(; ct = RadialityCentrality()),
                                B = 0.63, comp = ==)]

    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1, bgt = 1,
                                                cte = ces[1],
                                                wb = WeightBounds(; lb = -1, ub = 1))))
    @test average_centrality(ces[1].A, res.w, pr) >=
          minimum(centrality_vector(ces[1].A, pr).X)

    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1, bgt = 1,
                                                cte = ces[2],
                                                wb = WeightBounds(; lb = -1, ub = 1))))
    @test average_centrality(ces[2].A, res.w, pr) <= mean(centrality_vector(ces[2].A, pr).X)

    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1, bgt = 1,
                                                cte = ces[3],
                                                wb = WeightBounds(; lb = -1, ub = 1))))
    @test isapprox(average_centrality(ces[3].A, res.w, pr),
                   median(centrality_vector(ces[3].A, pr).X))

    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1, bgt = 1,
                                                cte = ces[4],
                                                wb = WeightBounds(; lb = -1, ub = 1))))
    @test isapprox(average_centrality(ces[4].A, res.w, pr),
                   maximum(centrality_vector(ces[4].A, pr).X))

    res = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1, bgt = 1,
                                                cte = ces[5],
                                                wb = WeightBounds(; lb = -1, ub = 1))))
    @test isapprox(average_centrality(ces[5].A, res.w, pr), 0.63)

    @test isapprox(res.w,
                   optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                     opt = JuMPOptimiser(; pe = pr, slv = slv, sbgt = 1,
                                                         bgt = 1,
                                                         cte = centrality_constraints(ces[5],
                                                                                      pr.X),
                                                         wb = WeightBounds(; lb = -1,
                                                                           ub = 1)))).w)
end

@testset "Fees" begin
    r = ConditionalDrawdownatRisk()
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1))
    w1 = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt)).w

    fees = FeesEstimator(; tn = TurnoverEstimator(; w = w0, val = Dict("XOM" => 0.3 / 252)),
                         l = Dict("JNJ" => 0.1 / 252), s = "BBY" => 0.2 / 252,
                         fl = ["HD" => 0.016 / 252], fs = "PFE" => 0.03 / 252)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), fees = fees, sets = sets)
    @test PortfolioOptimisers.needs_previous_weights(opt)
    mr = MeanRisk(; r = r, obj = MinimumRisk(), opt = opt)
    @test PortfolioOptimisers.needs_previous_weights(mr)
    res = optimise(mr)
    @test isapprox(res.w,
                   [-0.049849956175178456, 0.05299448616838142, -0.22771293882094312,
                    -0.08176557902583757, 0.033689114572132625, -0.005373211789678957, -0.0,
                    0.35475349716097293, 0.2908544599359326, 0.05579460357877945,
                    0.053431396866726384, 0.22107385221763304, 0.04052699357314736,
                    0.46868406888646474, -0.3137232044821237, -0.24996091041399768,
                    -0.05636967458501815, 0.1482804333363026, 0.10768140179418678,
                    0.1569911672021177], rtol = 1e-6)

    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1),
                        fees = fees_constraints(fees, sets))
    @test PortfolioOptimisers.needs_previous_weights(opt)
    mr = MeanRisk(; r = r, obj = MinimumRisk(), opt = opt)
    @test PortfolioOptimisers.needs_previous_weights(mr)
    @test isapprox(res.w, optimise(mr).w)

    fees = FeesEstimator(; tn = TurnoverEstimator(; w = w0, val = Dict("XOM" => 1)),
                         l = Dict("JNJ" => 1), s = "BBY" => 1, fl = ["HD" => 1],
                         fs = "PFE" => 1)
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                        wb = WeightBounds(; lb = -1, ub = 1), fees = fees, sets = sets)
    @test PortfolioOptimisers.needs_previous_weights(opt)
    mr = MeanRisk(; r = r, obj = MinimumRisk(), opt = opt)
    @test PortfolioOptimisers.needs_previous_weights(mr)
    res = optimise(mr)

    @test isapprox(res.w[findfirst(x -> x == "XOM", rd.nx)],
                   w0[findfirst(x -> x == "XOM", rd.nx)])
    @test isapprox(res.w[findfirst(x -> x == "JNJ", rd.nx)], 0)
    @test isapprox(res.w[findfirst(x -> x == "BBY", rd.nx)], 0)
    @test isapprox(res.w[findfirst(x -> x == "HD", rd.nx)], 0)
    @test isapprox(res.w[findfirst(x -> x == "PFE", rd.nx)], 0)
    @test isapprox(res.w,
                   [0.05201790942682152, -0.004972845913212141, -0.5888432775595224, 0.0,
                    0.22730252177448815, -0.02316379567148528, -0.0, 0.0,
                    0.4423716947104361, 0.17060216343485277, -0.030706210074928315,
                    0.2813132780743069, -0.03468309934130539, 0.40833226855001803, 0.0,
                    -0.22737493979523524, -0.061270023409634534, 0.2636697569240946,
                    0.07540459887030522, 0.05], rtol = 1e-6)

    fees = FeesEstimator(; fl = ["JNJ" => 1])
    opt = JuMPOptimiser(; pe = pr, slv = mip_slv, bgt = 1, fees = fees, sets = sets)
    res = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt))
    @test isapprox(res.w[findfirst(x -> x == "JNJ", rd.nx)], 0)
    @test isapprox(res.w,
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.10628880639076618,
                    0.37971339847731767, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0996419548234818,
                    0.21860531676987255, 0.1957505235385615], rtol = 1e-6)
end

@testset "Variance risk contribution" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets)
    lcs = LinearConstraintEstimator(; val = [["$a <= 0.2" for a in rd.nx];
                                             ["c3 <= 0.1"]])
    r = Variance(; rc = lcs)
    obj = MaximumRatio()
    mr = MeanRisk(; obj = obj, opt = opt, r = r)
    res = optimise(mr)
    rkc = risk_contribution(factory(r, pr, slv), res.w, pr)
    rkc = rkc / sum(rkc)
    @test all(rkc .- 0.2 .- 20 * sqrt(eps()) .< 0)
    @test abs(sum(rkc[3:3:end]) - 0.1) < 20 * sqrt(eps())
end

@testset "Weighted risk expressions" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    mip_opt = JuMPOptimiser(; pe = pr, slv = mip_slv)
    rs1 = [LowOrderMoment(; mu = 0),
           LowOrderMoment(; mu = 0, alg = MeanAbsoluteDeviation()),
           LowOrderMoment(; mu = 0, alg = SecondMoment(; alg1 = SemiMoment())),
           LowOrderMoment(; mu = 0, alg = SecondMoment()),
           LowOrderMoment(; mu = 0,
                          alg = SecondMoment(; alg1 = SemiMoment(), alg2 = SOCRiskExpr())),
           LowOrderMoment(; mu = 0, alg = SecondMoment(; alg2 = SOCRiskExpr())),
           LowOrderMoment(; mu = 0,
                          alg = SecondMoment(; alg1 = SemiMoment(), alg2 = RSOCRiskExpr())),
           LowOrderMoment(; mu = 0, alg = SecondMoment(; alg2 = RSOCRiskExpr())),
           LowOrderMoment(; mu = 0,
                          alg = SecondMoment(; alg1 = SemiMoment(), alg2 = QuadRiskExpr())),
           LowOrderMoment(; mu = 0, alg = SecondMoment(; alg2 = QuadRiskExpr())),
           ConditionalValueatRisk(), EntropicValueatRisk(), ConditionalValueatRiskRange(),
           EntropicValueatRiskRange(),
           DistributionallyRobustConditionalValueatRisk(; l = 1e-1, r = 1e-3),
           DistributionallyRobustConditionalValueatRiskRange(; l_a = 1e-1, r_a = 1e-3,
                                                             l_b = 1e-1, r_b = 1e-3),
           ValueatRisk(), ValueatRiskRange(),
           ValueatRisk(; alg = DistributionValueatRisk()),
           ValueatRiskRange(; alg = DistributionValueatRisk()), DrawdownatRisk(),
           LowOrderMoment(; mu = 0, alg = EvenMoment()),
           LowOrderMoment(; mu = 0, alg = EvenMoment(; alg = SemiMoment()))]
    rs2 = [LowOrderMoment(; mu = 0, w = wp),
           LowOrderMoment(; mu = 0, w = wp, alg = MeanAbsoluteDeviation()),
           LowOrderMoment(; mu = 0, w = wp, alg = SecondMoment(; alg1 = SemiMoment())),
           LowOrderMoment(; mu = 0, w = wp, alg = SecondMoment()),
           LowOrderMoment(; mu = 0, w = wp,
                          alg = SecondMoment(; alg1 = SemiMoment(), alg2 = SOCRiskExpr())),
           LowOrderMoment(; mu = 0, w = wp, alg = SecondMoment(; alg2 = SOCRiskExpr())),
           LowOrderMoment(; mu = 0, w = wp,
                          alg = SecondMoment(; alg1 = SemiMoment(), alg2 = RSOCRiskExpr())),
           LowOrderMoment(; mu = 0, w = wp, alg = SecondMoment(; alg2 = RSOCRiskExpr())),
           LowOrderMoment(; mu = 0, w = wp,
                          alg = SecondMoment(; alg1 = SemiMoment(), alg2 = QuadRiskExpr())),
           LowOrderMoment(; mu = 0, w = wp, alg = SecondMoment(; alg2 = QuadRiskExpr())),
           ConditionalValueatRisk(; w = wp), EntropicValueatRisk(; w = wp),
           ConditionalValueatRiskRange(; w = wp), EntropicValueatRiskRange(; w = wp),
           DistributionallyRobustConditionalValueatRisk(; l = 1e-1, r = 1e-3, w = wp),
           DistributionallyRobustConditionalValueatRiskRange(; l_a = 1e-1, r_a = 1e-3,
                                                             l_b = 1e-1, r_b = 1e-3,
                                                             w = wp), ValueatRisk(; w = wp),
           ValueatRiskRange(; w = wp),
           ValueatRisk(; alg = DistributionValueatRisk(), w = wp),
           ValueatRiskRange(; alg = DistributionValueatRisk(), w = wp),
           DrawdownatRisk(; w = wp), LowOrderMoment(; mu = 0, w = wp, alg = EvenMoment()),
           LowOrderMoment(; mu = 0, w = wp, alg = EvenMoment(; alg = SemiMoment()))]
    for (i, (r1, r2)) in enumerate(zip(rs1, rs2))
        if i <= 21
            continue
        end
        res1, res2 = if isa(r1,
                            Union{<:ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk},
                                  <:ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                     <:MIPValueatRisk}, <:DrawdownatRisk})
            optimise(MeanRisk(; r = r1, opt = mip_opt)),
            optimise(MeanRisk(; r = r2, opt = mip_opt))
        else
            optimise(MeanRisk(; r = r1, opt = opt)), optimise(MeanRisk(; r = r2, opt = opt))
        end
        rtol = if i in (3, 5)
            5e-6
        elseif i in (4, 8)
            5e-3
        elseif i in (12, 23)
            1e-4
        elseif i in (6, 14)
            5e-5
        elseif i in (7, 22)
            5e-4
        else
            1e-6
        end
        res = isapprox(res1.w, res2.w; rtol = rtol)
        if !res
            println("Iteration $i failed:")
            find_tol(res1.w, res2.w)
            display([res1.w res2.w res1.w - res2.w])
        end
        @test res
    end
end

@testset "Benchmark tracking" begin
    opt = JuMPOptimiser(; slv = slv, brt = true)
    mr = MeanRisk(; opt = opt)
    res1 = optimise(mr, rd)

    rdb = returns_result_picker(rd, true)
    prb = prior(EmpiricalPrior(), rdb)
    opt = JuMPOptimiser(; pe = prb, slv = slv, brt = true)
    mr = MeanRisk(; opt = opt)
    res2 = optimise(mr, rd)

    @test res1.w == res2.w
    @test isapprox(res1.w,
                   [0.13594491020480565, 0.06664942220565599, 0.03347388625739562,
                    0.018519121388742617, 0.037028053491839685, 0.07413397219776975,
                    0.08586567489340931, 3.171794447542938e-6, 0.08014095369469784,
                    0.03728279594798708, 0.030039697931611924, 0.03946098091476264,
                    0.1844158171814443, 0.052745229082914055, 0.023728773389230906,
                    0.005354628032118466, 0.010563552176813158, 0.04042806167670747,
                    0.02919805117977965, 0.015023246357866301], rtol = 1e-6)
end
