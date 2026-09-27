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

    # The variance is not monotonic in gamma. On this panel it falls to a turning point
    # near 0.11, then rises.
    gs = range(; start = 0.0, stop = 0.20, length = 10)
    rks = [expected_risk(r,
                         optimise(SchurComplementHierarchicalRiskParity(;
                                                                        params = SchurComplementParams(;
                                                                                                       gamma = g,
                                                                                                       alg = NonMonotonicSchurComplement()),
                                                                        opt = opt)).w, pr)
           for g in gs]
    k = argmin(rks)
    @test k == 6
    @test issorted(rks[1:k]; rev = true)
    @test issorted(rks[k:end])
    # The monotonic search stops at the turning point, below every value of the scan.
    res = optimise(SchurComplementHierarchicalRiskParity(;
                                                         params = SchurComplementParams(;
                                                                                        gamma = 0.2,
                                                                                        alg = MonotonicSchurComplement()),
                                                         opt = opt))
    @test gs[k - 1] < res.gamma < gs[k + 1]
    @test expected_risk(r, res.w, pr) <= minimum(rks)

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
    @test_throws DomainError MonotonicSchurComplement(; tol = 0)
    @test_throws DomainError MonotonicSchurComplement(; iter = 0)

    # The scan finds no rise, and the variance rises just below gamma, so the turning point
    # lies between the last two values of the scan.
    pr = schur_fixture(2, 0.5, 0.5)
    p = SchurComplementParams(; gamma = 1.0, alg = MonotonicSchurComplement(; N = 10))
    w, gamma, _ = @test_logs min_level = Logging.Warn PortfolioOptimisers.schur_complement_weights(pr,
                                                                                                   items,
                                                                                                   wb,
                                                                                                   p)
    @test 8 / 9 < gamma < 1
    @test isapprox(gamma, 0.9555121527777777; atol = 1e-12)
    @test w == nm_weights(pr, gamma)
    # A maximum gamma of zero runs the allocation at zero.
    w, gamma, _ = PortfolioOptimisers.schur_complement_weights(pr, items, wb,
                                                               SchurComplementParams(;
                                                                                     gamma = 0))
    @test gamma == 0
    @test w == nm_weights(pr, 0)
end
@testset "The docstrings of 03_SchurComplementHierarchicalRiskParity.jl against numbers" begin
    PO = PortfolioOptimisers
    # A six-asset covariance matrix, its leaf order, and the weights of the reference
    # implementation at three values of gamma and after its monotonic search.
    sigma = [7.782724171567788e-05 2.972127065997405e-05 -1.0114668612318741e-05 -2.9250463659699806e-05 -7.148034104575605e-06 1.2420446918541947e-05;
             2.972127065997405e-05 0.00010677258384304971 -2.5656442592669714e-05 2.2799233426940835e-06 -3.1371817657803e-05 -3.333868821011886e-05;
             -1.0114668612318741e-05 -2.5656442592669714e-05 4.5654013279841325e-05 -1.7069599855629958e-06 8.902017870314719e-06 1.321778381265861e-05;
             -2.9250463659699806e-05 2.2799233426940835e-06 -1.7069599855629958e-06 7.032234722364808e-05 -9.86642748864631e-06 -2.54986933025987e-05;
             -7.148034104575605e-06 -3.1371817657803e-05 8.902017870314719e-06 -9.86642748864631e-06 5.6816572868351124e-05 2.213001400466131e-05;
             1.2420446918541947e-05 -3.333868821011886e-05 1.321778381265861e-05 -2.54986933025987e-05 2.213001400466131e-05 7.055260745015985e-05]
    order = [3, 1, 6, 2, 5, 4]
    wref = Dict(0.3 => [0.10134690001730667, 0.12874075939894067, 0.19590381763233805,
                        0.20833918726949668, 0.2473048952267471, 0.1183644404551708],
                0.6 => [0.10583262260735729, 0.1366909319749356, 0.19501296338677082,
                        0.2038912543214287, 0.22896508924731293, 0.12960713846219465],
                0.9 => [0.11109612127205296, 0.14339089231355542, 0.19211073022046368,
                        0.20101184264989935, 0.21070269593829816, 0.14168771760573048])
    wmono = [0.11298612311771603, 0.14537895885681043, 0.1906849560345217,
             0.20039766043954868, 0.20465275006208986, 0.1458995514893133]
    prs = LowOrderPrior(; X = zeros(2, 6), mu = zeros(6), sigma = sigma)
    wb6 = WeightBounds(; lb = zeros(6), ub = ones(6))
    for g in (0.3, 0.6, 0.9)
        p = SchurComplementParams(; gamma = g, alg = NonMonotonicSchurComplement(),
                                  flag = false)
        w, gamma, r = PO.schur_complement_weights(prs, [order], wb6, p)
        @test isapprox(w, wref[g]; atol = 1e-15)
        # The weights of the recursion sum to one before any finaliser.
        @test isapprox(sum(w), 1; atol = 1e-15)
        @test gamma == g
        @test r.sigma === sigma
    end
    # The reference returns the weights of the last midpoint it evaluated, and this search
    # returns the weights of the value it reports. Both find the same value here.
    p = SchurComplementParams(; gamma = 1.0, alg = MonotonicSchurComplement(; N = 11))
    w, gamma, _ = PO.schur_complement_weights(prs, [order], wb6, p)
    @test gamma == 1.0
    @test isapprox(w, wmono; atol = 1e-15)

    # symmetric_step_up_matrix: the identity, the average of the insertions, and the
    # scaled transpose. Every row sums to one.
    @test PO.symmetric_step_up_matrix(4, 4) == LinearAlgebra.I(4)
    for (n1, n2) in ((3, 2), (5, 4), (2, 1), (2, 3), (4, 5), (1, 2))
        M = Matrix(PO.symmetric_step_up_matrix(n1, n2))
        @test size(M) == (n1, n2)
        @test isapprox(M * ones(n2), ones(n1); atol = 1e-15)
        if n1 == n2 + 1
            e = Matrix(1.0 * LinearAlgebra.I(n2))
            E = [vcat(e[1:(k - 1), :], fill(1 / n2, 1, n2), e[k:end, :]) for k in 1:n1]
            @test isapprox(M, sum(E) / n1; atol = 1e-15)
        else
            @test isapprox(M, transpose(PO.symmetric_step_up_matrix(n2, n1)) * n1 / n2;
                           atol = 1e-15)
        end
    end
    @test_throws DomainError PO.symmetric_step_up_matrix(4, 2)

    # schur_augmentation against the formula with explicit inverses.
    l, rgt = order[1:3], order[4:6]
    A, B, C = sigma[l, l], sigma[l, rgt], sigma[rgt, rgt]
    M = PO.symmetric_step_up_matrix(3, 3)
    for g in (0.25, 1.0)
        Am = A - g * B * inv(C) * B'
        R = I - g * B * inv(C) * M'
        S = (R \ Am + (R \ Am)') / 2
        @test isapprox(PO.schur_augmentation(A, B, C, g), S; rtol = 1e-12)
    end
    @test PO.schur_augmentation(A, B, C, 0) === A
    @test PO.schur_augmentation(A[1:1, 1:1], B[1:1, :], C, 0.5) == A[1:1, 1:1]
    @test PO.schur_augmentation(A, B[:, 1:1], C[1:1, 1:1], 0.5) == A

    # naive_portfolio_risk: inverse variance weights for both measures.
    wn = inv.(LinearAlgebra.diag(sigma))
    wn ./= sum(wn)
    @test PO.naive_portfolio_risk(Variance(), sigma) ≈ dot(wn, sigma, wn)
    @test PO.naive_portfolio_risk(StandardDeviation(), sigma) ≈ sqrt(dot(wn, sigma, wn))

    # At gamma = 0 a Variance measure gives the allocation of HierarchicalRiskParity with a
    # Variance measure. A StandardDeviation measure does not, because the naive weights
    # are inverse variance, not inverse volatility.
    rng = StableRNG(123)
    X = randn(rng, 400, 2) * 0.01 * randn(rng, 2, 10) + randn(rng, 400, 10) * 0.005
    rdx = ReturnsResult(; nx = ["A$i" for i in 1:10], X = X)
    for (r, same) in ((Variance(), true), (StandardDeviation(), false))
        wh = optimise(HierarchicalRiskParity(; r = r), rdx).w
        ws = optimise(SchurComplementHierarchicalRiskParity(;
                                                            params = SchurComplementParams(;
                                                                                           r = r,
                                                                                           gamma = 0,
                                                                                           alg = NonMonotonicSchurComplement())),
                      rdx).w
        @test isapprox(wh, ws; atol = 1e-14) == same
    end

    # An integer returns matrix allocates in floating point, and gives the weights of the
    # same matrix in Float64. A Float32 matrix stays in Float32.
    Xi = rand(StableRNG(3), -5:5, 100, 6)
    rdi = ReturnsResult(; nx = ["A$i" for i in 1:6], X = Xi)
    rdf = ReturnsResult(; nx = ["A$i" for i in 1:6], X = float.(Xi))
    for params in (SchurComplementParams(; alg = NonMonotonicSchurComplement()),
                   SchurComplementParams(),
                   [SchurComplementParams(; gamma = 0),
                    SchurComplementParams(; r = StandardDeviation(), gamma = 0.3)])
        sh = SchurComplementHierarchicalRiskParity(; params = params)
        resi = optimise(sh, rdi)
        @test eltype(resi.w) == Float64
        @test resi.w == optimise(sh, rdf).w
    end
    res32 = optimise(SchurComplementHierarchicalRiskParity(),
                     ReturnsResult(; nx = ["A$i" for i in 1:6], X = Float32.(Xi)))
    @test eltype(res32.w) == Float32

    # A vector of bundles blends the portfolios by the scale of each measure. Its result
    # holds one measure and one gamma per bundle, and expected_risk adds the scaled risks
    # at the blended weights.
    ps = [SchurComplementParams(; gamma = 0.2, alg = NonMonotonicSchurComplement()),
          SchurComplementParams(;
                                r = StandardDeviation(;
                                                      settings = RiskMeasureSettings(;
                                                                                     scale = 2)),
                                gamma = 0.1, alg = NonMonotonicSchurComplement())]
    resv = optimise(SchurComplementHierarchicalRiskParity(; params = ps), rdx)
    res1 = optimise(SchurComplementHierarchicalRiskParity(; params = ps[1]), rdx)
    res2 = optimise(SchurComplementHierarchicalRiskParity(; params = ps[2]), rdx)
    @test isapprox(resv.w, (res1.w + 2 * res2.w) / 3; atol = 1e-15)
    @test resv.gamma == [0.2, 0.1]
    @test length(resv.r) == 2
    v = dot(resv.w, resv.pr.sigma, resv.w)
    @test expected_risk(resv.r, resv.w, resv.pr) ≈ v + 2 * sqrt(v)
    @test expected_risk(res1.r, res1.w, res1.pr) ≈ dot(res1.w, res1.pr.sigma, res1.w)

    # flag = false abandons an allocation whose augmented block is not positive definite,
    # and assert_schur_weights names gamma. flag = true rethrows a repair that throws.
    prx = prior(EmpiricalPrior(), rdx)
    wbx = WeightBounds(; lb = zeros(10), ub = ones(10))
    pnf = SchurComplementParams(; gamma = 1, alg = NonMonotonicSchurComplement(),
                                flag = false)
    wf, gf, _ = PO.schur_complement_weights(prx, [collect(1:10)], wbx, pnf)
    @test isnothing(wf)
    @test gf == 1
    @test_throws ArgumentError PO.assert_schur_weights(wf, gf)
    pft = SchurComplementParams(; gamma = 1, alg = NonMonotonicSchurComplement())
    @test_throws ArgumentError PO.schur_complement_weights(prx, [collect(1:10)], wbx, pft)
    @test isnothing(PO.assert_schur_weights(ones(2), 0.5))

    # A bracket already narrower than tol ends the bisection with no warning, even when the
    # derived budget is zero or less.
    obj(x) = (fill(x, 1), (x - 5e-8)^2)
    @test (@test_logs min_level = Logging.Warn PO.schur_complement_binary_search(obj, 0.0,
                                                                                 1e-7,
                                                                                 obj(0.0)[2],
                                                                                 [0.0],
                                                                                 1e-4,
                                                                                 nothing,
                                                                                 true)) ==
          ([0.0], 0.0)

    # port_opt_view of a bundle views the measure and keeps the other fields.
    pv = SchurComplementParams(; r = Variance(; sigma = sigma), gamma = 0.4,
                               alg = NonMonotonicSchurComplement(), flag = false)
    pvv = PO.port_opt_view(pv, [2, 4], zeros(3, 6))
    @test pvv.r.sigma == sigma[[2, 4], [2, 4]]
    @test (pvv.gamma, pvv.alg, pvv.flag) == (0.4, NonMonotonicSchurComplement(), false)
end
@testset "The docstrings of 04_HierarchicalEqualRiskContribution.jl against numbers" begin
    PO = PortfolioOptimisers
    rng = StableRNG(42)
    N = 12
    X = randn(rng, 300, 3) * 0.01 * randn(rng, 3, N) + randn(rng, 300, N) * 0.006
    rdx = ReturnsResult(; nx = ["A$i" for i in 1:N], X = X)
    prx = prior(EmpiricalPrior(), rdx)
    clx = clusterise(ClustersEstimator(), prx)
    optx = HierarchicalOptimiser(; pe = prx, cle = clx)
    S = prx.sigma

    # An independent oracle of the # Mathematical definition: recurse from the root, stop
    # at a cluster, and split each node between its branches by the risks of their
    # clusters. rho_i and rho_o are Variance or StandardDeviation, so their unitary risks
    # and the risk of a sub-portfolio have closed forms.
    function herc_oracle(clr, rku_i, rku_o, rho_o)
        cls = [findall(==(k), assignments(clr)) for k in 1:(clr.k)]
        sets = Set.(cls)
        rkcl = map(cls) do cl
            v = zeros(size(S, 1))
            v[cl] = inv.(rku_o[cl]) / sum(inv.(rku_o[cl]))
            return rho_o(v)
        end
        W = ones(length(cls))
        function split!(node)
            Set(PO.pre_order(node)) in sets && return nothing
            l = Set(PO.pre_order(node.left))
            r = Set(PO.pre_order(node.right))
            lc = findall(s -> issubset(s, l), sets)
            rc = findall(s -> issubset(s, r), sets)
            a = sum(rkcl[rc]) / (sum(rkcl[lc]) + sum(rkcl[rc]))
            W[lc] .*= a
            W[rc] .*= 1 - a
            split!(node.left)
            return split!(node.right)
        end
        split!(PO.to_tree(clr.res)[1])
        w = zeros(size(S, 1))
        for (k, cl) in pairs(cls)
            w[cl] = W[k] * inv.(rku_i[cl]) / sum(inv.(rku_i[cl]))
        end
        return w, W, cls
    end
    var_rk = LinearAlgebra.diag(S)
    sd_rk = sqrt.(var_rk)
    var_r(v) = dot(v, S, v)
    sd_r(v) = sqrt(dot(v, S, v))
    @test clx.k == 3
    for (ri, ro, rki, rko, rho) in ((Variance(), Variance(), var_rk, var_rk, var_r),
                                    (StandardDeviation(), Variance(), sd_rk, var_rk, var_r),
                                    (Variance(), StandardDeviation(), var_rk, sd_rk, sd_r))
        wo, W, cls = herc_oracle(clx, rki, rko, rho)
        # The cluster weights and the weights sum to one before the normalisation.
        @test isapprox(sum(W), 1; atol = 1e-15)
        @test isapprox(sum(wo), 1; atol = 1e-15)
        for ex in (FLoops.SequentialEx(), FLoops.ThreadedEx())
            res = optimise(HierarchicalEqualRiskContribution(; ri = ri, ro = ro, ex = ex,
                                                             opt = optx), rdx)
            @test isapprox(res.w, wo; atol = 1e-15)
            @test isapprox([sum(res.w[cl]) for cl in cls], W; atol = 1e-15)
        end
    end

    # K = 2: the one split is the flat inverse-risk allocation over the two clusters.
    cl2 = PO.Clusters(; res = clx.res, S = clx.S, D = clx.D, k = 2)
    res2 = optimise(HierarchicalEqualRiskContribution(;
                                                      opt = HierarchicalOptimiser(;
                                                                                  pe = prx,
                                                                                  cle = cl2)),
                    rdx)
    cls2 = [findall(==(k), assignments(cl2)) for k in 1:2]
    rk2 = map(cls2) do cl
        v = zeros(N)
        v[cl] = inv.(var_rk[cl]) / sum(inv.(var_rk[cl]))
        return var_r(v)
    end
    @test isapprox([sum(res2.w[cl]) for cl in cls2], inv.(rk2) / sum(inv.(rk2));
                   atol = 1e-15)

    # Two merges at one height across the cut. The cut does the first merge of the sorted
    # heights, so the nodes above it are the second and third merges, and every weight is
    # finite and equal to the closed form.
    Xt = randn(StableRNG(1), 200, 4) .* [0.01 0.02 0.015 0.012]
    rdt = ReturnsResult(; nx = ["A$i" for i in 1:4], X = Xt)
    prt = prior(EmpiricalPrior(), rdt)
    ct0 = clusterise(ClustersEstimator(), prt)
    hc = PO.Clustering.Hclust([-1 -2; -3 -4; 1 2], [1.0, 1.0, 2.0], [1, 2, 3, 4], :single)
    ct = PO.Clusters(; res = hc, S = ct0.S, D = ct0.D, k = 3)
    @test assignments(ct) == [1, 1, 2, 3]
    rest = optimise(HierarchicalEqualRiskContribution(;
                                                      opt = HierarchicalOptimiser(;
                                                                                  pe = prt,
                                                                                  cle = ct)),
                    rdt)
    @test isa(rest.retcode, OptimisationSuccess)
    St = prt.sigma
    function vt(cl)
        v = zeros(4)
        v[cl] = inv.(LinearAlgebra.diag(St)[cl])
        return v / sum(v)
    end
    rkt = [dot(vt(cl), St, vt(cl)) for cl in ([1, 2], [3], [4])]
    a1 = (rkt[2] + rkt[3]) / sum(rkt)
    a2 = rkt[3] / (rkt[2] + rkt[3])
    @test isapprox(rest.w, a1 * vt([1, 2]) + (1 - a1) * (a2 * vt([3]) + (1 - a2) * vt([4]));
                   atol = 1e-15)

    # The shares inside a cluster sum to one under every scalariser, so the cluster keeps
    # the weight that the outer recursion gives it. A sum is the average weighted by the
    # scales, and a maximum or a minimum is the naive weights of the measure it selects.
    rs = factory([ConditionalValueatRisk(),
                  Variance(; settings = RiskMeasureSettings(; scale = 10))], prx)
    rkus = PO.herc_unitary_risks(rs, prx.X, nothing)
    @test rkus == [PO.unitary_expected_risks(r, prx.X, nothing) for r in rs]
    naive(rku, cl) = inv.(rku[cl]) / sum(inv.(rku[cl]))
    cl = findall(==(1), assignments(clx))
    vs = PO.herc_intra_cluster_weights(SumScalariser(), rs, rkus, cl)
    @test isapprox(vs, (naive(rkus[1], cl) + 10 * naive(rkus[2], cl)) / 11; atol = 1e-15)
    kmax = argmax([s * sum(rku[cl]) for (s, rku) in zip((1, 10), rkus)])
    kmin = argmin([s * sum(rku[cl]) for (s, rku) in zip((1, 10), rkus)])
    @test PO.herc_intra_cluster_weights(MaxScalariser(), rs, rkus, cl) ≈
          naive(rkus[kmax], cl)
    @test PO.herc_intra_cluster_weights(MinScalariser(), rs, rkus, cl) ≈
          naive(rkus[kmin], cl)
    for g in (1e-3, 1.0, 1e2)
        u = log.(exp.(g * naive(rkus[1], cl)) + exp.(g * 10 * naive(rkus[2], cl))) / g
        @test isapprox(PO.herc_intra_cluster_weights(LogSumExpScalariser(; gamma = g), rs,
                                                     rkus, cl), u / sum(u); rtol = 1e-12)
    end
    _, W, cls = herc_oracle(clx, var_rk, var_rk, var_r)
    for sca in (SumScalariser(), MaxScalariser(), MinScalariser(),
                LogSumExpScalariser(; gamma = 1e-3), LogSumExpScalariser())
        res = optimise(HierarchicalEqualRiskContribution(;
                                                         ri = [ConditionalValueatRisk(),
                                                               Variance()], ro = Variance(),
                                                         scai = sca, opt = optx), rdx)
        @test isapprox([sum(res.w[cl]) for cl in cls], W; atol = 1e-15)
    end

    # The risk of a cluster under a vector of measures scalarises the scaled risks of each
    # measure's own sub-portfolio. One measure is not scaled.
    p = [s * PO.herc_cluster_risk(SumScalariser(), r, rku, cl, prx.X, nothing)
         for (s, r, rku) in zip((1, 10), rs, rkus)]
    @test PO.herc_cluster_risk(SumScalariser(), rs, rkus, cl, prx.X, nothing) ≈ sum(p)
    @test PO.herc_cluster_risk(MaxScalariser(), rs, rkus, cl, prx.X, nothing) == maximum(p)
    @test PO.herc_cluster_risk(MinScalariser(), rs, rkus, cl, prx.X, nothing) == minimum(p)
    @test PO.herc_cluster_risk(LogSumExpScalariser(; gamma = 2), rs, rkus, cl, prx.X,
                               nothing) ≈ log(sum(exp.(2 * p))) / 2
    v = zeros(N)
    v[cl] = naive(rkus[2], cl)
    @test PO.herc_cluster_risk(SumScalariser(), rs[2], rkus[2], cl, prx.X, nothing) ≈
          var_r(v)

    # The weights do not depend on the order of the leaves.
    for bo in (:r, :barjoseph)
        @test isapprox(optimise(HierarchicalEqualRiskContribution(), rdx; branchorder = bo).w,
                       optimise(HierarchicalEqualRiskContribution(), rdx).w; atol = 1e-15)
    end

    # An integer returns matrix allocates in floating point, and gives the weights of the
    # same matrix in Float64. A Float32 matrix stays in Float32.
    Xi = rand(StableRNG(3), -5:5, 200, 6)
    rdi = ReturnsResult(; nx = ["A$i" for i in 1:6], X = Xi)
    rdf = ReturnsResult(; nx = ["A$i" for i in 1:6], X = float.(Xi))
    for (ri, ro) in
        ((Variance(), Variance()), ([Variance(), StandardDeviation()], Variance()),
         (Variance(), [Variance(), StandardDeviation()]))
        h = HierarchicalEqualRiskContribution(; ri = ri, ro = ro)
        resi = optimise(h, rdi)
        @test eltype(resi.w) == Float64
        @test resi.w == optimise(h, rdf).w
    end
    @test eltype(optimise(HierarchicalEqualRiskContribution(),
                          ReturnsResult(; nx = ["A$i" for i in 1:6], X = Float32.(Xi))).w) ==
          Float32

    # The view reads the prior's X, keeps ri and ro one object, and views each measure.
    r0 = Variance()
    vv = PO.port_opt_view(HierarchicalEqualRiskContribution(; ri = r0, ro = r0, opt = optx),
                          [2, 5, 7], zeros(3, N))
    @test vv.ri === vv.ro
    @test vv.opt.pe.X == prx.X[:, [2, 5, 7]]
    hs = HierarchicalEqualRiskContribution(; ri = Variance(; sigma = S), ro = Variance())
    vs2 = PO.port_opt_view(hs, [2, 5], X)
    @test vs2.ri.sigma == S[[2, 5], [2, 5]]
    @test isnothing(vs2.ro.sigma)

    # The validations throw.
    @test_throws IsEmptyError HierarchicalEqualRiskContribution(; ri = Variance[])
    @test_throws IsEmptyError HierarchicalEqualRiskContribution(; ro = Variance[])
    @test_throws ArgumentError HierarchicalEqualRiskContribution(; ri = NoRisk())
    @test_throws ArgumentError HierarchicalEqualRiskContribution(; ro = [NoRisk()])

    # A turnover measure needs the previous weights, and the default does not.
    @test !PO.needs_previous_weights(HierarchicalEqualRiskContribution())
    @test PO.needs_previous_weights(HierarchicalEqualRiskContribution(;
                                                                      ro = TurnoverRiskMeasure(;
                                                                                               w = fill(1 /
                                                                                                        N,
                                                                                                        N))))

    # non_investable_universe writes the names to the non-investable axis of opt.sets.
    setsx = UniverseSets(; dict = Dict("nx" => rdx.nx))
    hn = PO.non_investable_universe(HierarchicalEqualRiskContribution(;
                                                                      opt = HierarchicalOptimiser(;
                                                                                                  sets = setsx)),
                                    ["B1"])
    @test hn.opt.sets.dict[hn.opt.sets.nikey] == ["B1"]
    @test hn.opt.sets.dict["nx"] == rdx.nx
end
