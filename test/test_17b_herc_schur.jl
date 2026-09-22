include(joinpath(@__DIR__, "test17_setup.jl"))

@testset "HierarchicalEqualRiskContribution" begin
    w1 = [0.02771765212089022, 0.009402158178351775, 0.03331519584748935,
          0.017073221259513743, 0.032359261928293595, 0.02887250096235004,
          0.03579588604148309, 0.12380428820248753, 0.03935562840030466,
          0.09707432790168266, 0.05055963165295286, 0.09453256731040742,
          0.028302824480310254, 0.09927818284167722, 0.05153141359201795,
          0.07783839405594621, 0.008855636176616296, 0.0632442983330085,
          0.05261979423155193, 0.02846713648266485]
    res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ri = Variance(),
                                                     ro = Variance(; sigma = pr.sigma),
                                                     opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ri = [Variance()], opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ri = [Variance()], ro = [Variance()],
                                                     opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ri = [Variance()], ro = Variance(),
                                                     opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ri = Variance(), ro = [Variance()],
                                                     opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)

    res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                     opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ri = Variance(),
                                                     ro = Variance(; sigma = pr.sigma),
                                                     opt = opt, ex = FLoops.SequentialEx()))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                     ri = [Variance()], opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                     ri = [Variance()], ro = [Variance()],
                                                     opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                     ri = [Variance()], ro = Variance(),
                                                     opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
    res = optimise(HierarchicalEqualRiskContribution(; ex = FLoops.SequentialEx(),
                                                     ri = Variance(), ro = [Variance()],
                                                     opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w, w1)
end
@testset "HierarchicalEqualRiskContribution scalarisers" begin
    sces = [SumScalariser(), MaxScalariser(), LogSumExpScalariser(; gamma = 1e-3),
            LogSumExpScalariser(; gamma = 1e2), MinScalariser()]
    df = CSV.read(joinpath(@__DIR__, "./assets/HierarchicalEqualRiskContribution2.csv.gz"),
                  DataFrame)
    for (i, sca) in pairs(sces)
        res = optimise(HierarchicalEqualRiskContribution(;
                                                         ri = [ConditionalValueatRisk(),
                                                               Variance(;
                                                                        settings = RiskMeasureSettings(;
                                                                                                       scale = 1e1))],
                                                         opt = opt, scai = sca))
        @test isa(res.retcode, OptimisationSuccess)
        success = isapprox(res.w, df[!, i])
        if !success
            println("Counter parallel: $i")
            find_tol(res.w, df[!, i])
        end
        @test success
        res = optimise(HierarchicalEqualRiskContribution(;
                                                         ri = [ConditionalValueatRisk(),
                                                               Variance(;
                                                                        settings = RiskMeasureSettings(;
                                                                                                       scale = 1e1))],
                                                         opt = opt, scai = sca,
                                                         ex = FLoops.SequentialEx()))
        @test isa(res.retcode, OptimisationSuccess)
        success = isapprox(res.w, df[!, i])
        if !success
            println("Counter serial: $i")
            find_tol(res.w, df[!, i])
        end
        @test success
    end
end
@testset "Weight bounds" begin
    sets = UniverseSets(; dict = Dict("nx" => rd.nx, "group1" => ["AAPL", "MSFT"]))
    eqn = WeightBoundsEstimator(; lb = ["JNJ" => 0.03, "group1" => 0.035],
                                ub = Dict("PEP" => 0.08, "JNJ" => 0.03))
    opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn)
    res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test all(abs.(res.w[[findfirst(x -> x == i, sets.dict[sets.xkey])
                          for i in sets.dict["group1"]]] .- 0.035) .<= 1e-10)
    @test all(res.w[[findfirst(x -> x == i, sets.dict[sets.xkey])
                     for i in sets.dict["group1"]]] .>= 0.035)
    @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.xkey])] - 0.08) < 5e-10

    opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn,
                                wf = JuMPWeightFinaliser(;
                                                         alg = RelativeErrorWeightFinaliser(),
                                                         slv = slv))
    res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w[findfirst(x -> x == "JNJ", sets.dict[sets.xkey])], 0.03)
    @test all(abs.(res.w[[findfirst(x -> x == i, sets.dict[sets.xkey])
                          for i in sets.dict["group1"]]] .- 0.035) .<= 1e-10)
    @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.xkey])] - 0.08) < 5e-10

    opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn,
                                wf = JuMPWeightFinaliser(;
                                                         alg = SquaredRelativeErrorWeightFinaliser(),
                                                         slv = slv))
    res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w[findfirst(x -> x == "JNJ", sets.dict[sets.xkey])], 0.03)
    @test all(abs.(res.w[[findfirst(x -> x == i, sets.dict[sets.xkey])
                          for i in sets.dict["group1"]]] .- 0.035) .<= 1e-10)
    @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.xkey])] - 0.08) < 5e-10

    opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn,
                                wf = JuMPWeightFinaliser(;
                                                         alg = AbsoluteErrorWeightFinaliser(),
                                                         slv = slv))
    res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w[findfirst(x -> x == "JNJ", sets.dict[sets.xkey])], 0.03)
    @test all(res.w[[findfirst(x -> x == i, sets.dict[sets.xkey])
                     for i in sets.dict["group1"]]] .>= 0.035)
    @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.xkey])] - 0.08) < 5e-10

    opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv, sets = sets, wb = eqn,
                                wf = JuMPWeightFinaliser(;
                                                         alg = SquaredAbsoluteErrorWeightFinaliser(),
                                                         slv = slv))
    res = optimise(HierarchicalEqualRiskContribution(; opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(res.w[findfirst(x -> x == "JNJ", sets.dict[sets.xkey])], 0.03)
    @test all(res.w[[findfirst(x -> x == i, sets.dict[sets.xkey])
                     for i in sets.dict["group1"]]] .>= 0.035)
    @test abs(res.w[findfirst(x -> x == "PEP", sets.dict[sets.xkey])] - 0.08) < 5e-10
end
@testset "SchurComplementHierarchicalRiskParity" begin
    r = factory(Variance(), pr)
    hrp = HierarchicalRiskParity(; r = r, opt = opt)
    res0 = optimise(hrp)
    rk0 = expected_risk(r, res0.w, pr)

    sch = SchurComplementHierarchicalRiskParity(;
                                                params = SchurComplementParams(; gamma = 0,
                                                                               alg = NonMonotonicSchurComplement()),
                                                opt = opt)
    res = optimise(sch)
    rk = expected_risk(r, res.w, pr)
    @test isapprox(res0.w, res.w)
    @test isapprox(rk0, rk)

    sch = SchurComplementHierarchicalRiskParity(;
                                                params = SchurComplementParams(; gamma = 1,
                                                                               alg = MonotonicSchurComplement()),
                                                opt = opt)
    res = optimise(sch)
    rk = expected_risk(r, res.w, pr)
    @test rk <= rk0

    sch = SchurComplementHierarchicalRiskParity(;
                                                params = SchurComplementParams(;
                                                                               gamma = 0.675,
                                                                               alg = NonMonotonicSchurComplement()),
                                                opt = opt)
    res = optimise(sch)
    rk = expected_risk(r, res.w, pr)
    @test rk >= rk0

    rk0 = -Inf
    for gamma in range(; start = 0.0, stop = 0.20, length = 10)
        sch = SchurComplementHierarchicalRiskParity(;
                                                    params = SchurComplementParams(;
                                                                                   gamma = 0.675,
                                                                                   alg = NonMonotonicSchurComplement()),
                                                    opt = opt)
        res = optimise(sch)
        rk = expected_risk(r, res.w, pr)
        @test rk >= rk0
        rk0 = rk
    end

    sch = SchurComplementHierarchicalRiskParity(;
                                                params = SchurComplementParams(;
                                                                               gamma = 0.05,
                                                                               alg = NonMonotonicSchurComplement()),
                                                opt = opt)
    res0 = optimise(sch)
    sch = SchurComplementHierarchicalRiskParity(;
                                                params = SchurComplementParams(;
                                                                               r = StandardDeviation(),
                                                                               gamma = 0.1,
                                                                               alg = NonMonotonicSchurComplement()),
                                                opt = opt)
    res1 = optimise(sch)

    sch = SchurComplementHierarchicalRiskParity(;
                                                params = [SchurComplementParams(;
                                                                                gamma = 0.05,
                                                                                alg = NonMonotonicSchurComplement()),
                                                          SchurComplementParams(;
                                                                                r = StandardDeviation(;
                                                                                                      settings = RiskMeasureSettings(;
                                                                                                                                     scale = 2)),
                                                                                gamma = 0.1,
                                                                                alg = NonMonotonicSchurComplement())],
                                                opt = opt)
    res2 = optimise(sch)

    w2 = res0.w + 2 * res1.w
    w2 /= sum(w2)
    @test isapprox(res2.w, w2)
end
@testset "MonotonicSchurComplement returns the weights of the gamma it reports" begin
    # An 8-asset one-factor panel. The search runs over the natural leaf order, so no
    # clustering choice moves the numbers.
    function schur_fixture(seed, ls, is)
        rng = StableRNG(seed)
        F = randn(rng, 200, 1) * 0.01
        B = randn(rng, 8, 1) * ls
        E = randn(rng, 200, 8) * 0.01 * is
        return prior(EmpiricalPrior(), F * B' + E)
    end
    items = [collect(1:8)]
    wb = WeightBounds(; lb = zeros(8), ub = ones(8))
    function nm_weights(pr, gamma)
        p = SchurComplementParams(; gamma = gamma, alg = NonMonotonicSchurComplement(),
                                  flag = false)
        return PortfolioOptimisers.schur_complement_weights(pr, items, wb, p)[1]
    end
    # (seed, loading scale, idiosyncratic scale, maximum gamma, turning point). The
    # turning points are those of an independent implementation of the same search,
    # with the same scan grid. The cases cover: the variance rises from gamma = 0, still
    # falls at the top of the range, and turns inside the range twice.
    for (seed, ls, is, g, gtp) in ((1, 0.5, 0.5, 0.5, 0.0), (2, 0.5, 0.5, 0.5, 0.5),
                                   (2, 0.5, 0.5, 1.0, 0.9554687499999999), (2, 1.0, 0.5, 0.5, 0.3765625))
        pr = schur_fixture(seed, ls, is)
        p = SchurComplementParams(; gamma = g,
                                  alg = MonotonicSchurComplement(;
                                                                 N = ceil(Int, g / 0.1) + 1))
        w, gamma, _ = @test_logs min_level = Logging.Warn PortfolioOptimisers.schur_complement_weights(pr,
                                                                                                       items,
                                                                                                       wb,
                                                                                                       p)
        @test isapprox(gamma, gtp; atol = 1e-12)
        @test w == nm_weights(pr, gamma)
    end

    # No midpoint passes the test: the incumbent and its own weights are the answer.
    obj(x) = (fill(x, 1), x <= 0.2 ? 1 - x : 2.0)
    @test (@test_logs min_level = Logging.Warn PortfolioOptimisers.schur_complement_binary_search(obj,
                                                                                                  0.2,
                                                                                                  0.4,
                                                                                                  0.8,
                                                                                                  [0.5])) ==
          ([0.5], 0.2)
    @test (@test_logs (:warn, r"did not converge") PortfolioOptimisers.schur_complement_binary_search(obj,
                                                                                                      0.2,
                                                                                                      0.4,
                                                                                                      0.8,
                                                                                                      [0.5],
                                                                                                      1e-4,
                                                                                                      1)) ==
          ([0.5], 0.2)
    # The scan needs both ends of the range.
    @test_throws DomainError MonotonicSchurComplement(; N = 1)
end
