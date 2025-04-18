@safetestset "Risk Measure Tests" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames, StatsBase,
          Clarabel, LinearAlgebra
    import PortfolioOptimisers: risk_measure_factory, risk_measure_view, ucs_view
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
    end
    @testset "Risk" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        A = rand(rng, 3, 20)
        B = rand(rng, 3)
        ew = eweights(1:1000, 1 / 1000; scale = true)
        w = rand(rng, 20)
        w ./= sum(w)
        pr1 = prior(HighOrderPriorEstimator(), X)

        i = [20, 3, 9]
        wview = view(w, i)
        Xview = view(X, :, i)
        slv = Solver(; name = :Clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false))
        ucs1 = DeltaUncertaintySetEstimator(;)
        ucs2 = NormalUncertaintySetEstimator(;)
        settings = RiskMeasureSettings(; rke = false, scale = -1, ub = 3)
        formulation = RSOC()
        ucs1view = ucs_view(ucs1, ucs2, i)
        ucs2view = ucs_view(nothing, ucs2, i)
        sigma = pr1.sigma * 1.5
        sigmaview = view(sigma, i, i)
        prsigmaview = view(pr1.sigma, i, i)
        rc = LinearConstraint(; A = LinearConstraintSide(; group = :A, name = :B), B = 0)
        rs = [Variance(; settings = settings, formulation = formulation, sigma = sigma,
                       rc = rc), Variance(;)]
        r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
        rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))
        @test r[1].settings === settings
        @test r[1].formulation === formulation
        @test r[1].sigma === sigma
        @test r[1].rc === rc
        @test expected_risk(r[1], w, X) == dot(w, sigma, w)
        @test rv[1].settings === settings
        @test rv[1].formulation === formulation
        @test rv[1].sigma === sigmaview
        @test rv[1].rc === rc
        @test expected_risk(rv[1], wview, Xview) == dot(wview, sigmaview, wview)

        @test r[2].sigma === pr1.sigma
        @test expected_risk(r[2], w, X) == dot(w, pr1.sigma, w)
        @test rv[2].sigma === prsigmaview
        @test expected_risk(rv[2], wview, Xview) == dot(wview, prsigmaview, wview)

        sigma = pr1.sigma * 2
        sigmaview = view(sigma, i, i)
        rs = [StandardDeviation(; settings = settings, sigma = sigma), StandardDeviation(;)]
        r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
        rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))
        @test r[1].settings === settings
        @test r[1].sigma === sigma
        @test expected_risk(r[1], w, X) == sqrt(dot(w, sigma, w))
        @test rv[1].settings === settings
        @test rv[1].sigma === sigmaview
        @test expected_risk(rv[1], wview, Xview) == sqrt(dot(wview, sigmaview, wview))

        @test r[2].sigma === pr1.sigma
        @test expected_risk(r[2], w, X) == sqrt(dot(w, pr1.sigma, w))
        @test rv[2].sigma === prsigmaview
        @test expected_risk(rv[2], wview, Xview) == sqrt(dot(wview, prsigmaview, wview))

        sigma = pr1.sigma * 2.5
        sigmaview = view(sigma, i, i)
        rs = [UncertaintySetVariance(; settings = settings, ucs = ucs1, sigma = sigma),
              UncertaintySetVariance(;)]
        r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
        rv = risk_measure_view(rs, Ref(pr1), Ref(i), Ref(slv), Ref(ucs2))
        @test r[1].settings === settings
        @test r[1].ucs === ucs1
        @test r[1].sigma === sigma
        @test expected_risk(r[1], w, X) == dot(w, sigma, w)
        @test rv[1].settings === settings
        @test rv[1].ucs === ucs1view
        @test rv[1].sigma === sigmaview
        @test expected_risk(rv[1], wview, Xview) == dot(wview, sigmaview, wview)

        @test r[2].sigma === pr1.sigma
        @test r[2].ucs === ucs2
        @test expected_risk(r[2], w, X) == dot(w, pr1.sigma, w)
        @test rv[2].sigma === prsigmaview
        @test rv[2].ucs === ucs2view
        @test expected_risk(rv[2], wview, Xview) == dot(wview, prsigmaview, wview)

        zerovec = fill(0.0, 20)
        formulation = Quad()
        mu = rand(rng, 20)
        rs = [LowOrderMoment(; settings = settings, w = ew, mu = mu),
              LowOrderMoment(; mu = 0), LowOrderMoment(; mu = zerovec),
              LowOrderMoment(; alg = SemiDeviation(; ddof = 2)),
              LowOrderMoment(; alg = SemiVariance(; ddof = 3, formulation = formulation)),
              LowOrderMoment(; alg = MeanAbsoluteDeviation()),
              LowOrderMoment(; alg = MeanAbsoluteDeviation(; w = ew))]
        r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
        @test r[1].settings === settings
        @test r[1].w === ew
        @test r[1].mu === mu
        val = X * w .- dot(w, mu)
        val = val[val .<= zero(eltype(val))]
        @test expected_risk(rs[1], w, X) == -sum(val) / size(X, 1)

        @test r[2].mu == 0
        val = X * w
        val = val[val .<= zero(eltype(val))]
        @test expected_risk(rs[2], w, X) == -sum(val) / size(X, 1)

        @test isa(r[3].alg, FirstLowerMoment)
        @test r[3].mu === zerovec
        val = X * w .- dot(w, zerovec)
        val = val[val .<= zero(eltype(val))]
        @test expected_risk(rs[3], w, X) == -sum(val) / size(X, 1)

        @test r[4].alg.ddof == 2
        @test r[4].mu === pr1.mu
        val = X * w .- dot(w, pr1.mu)
        val = val[val .<= zero(eltype(val))]
        @test expected_risk(rs[4], w, X) ==
              sqrt(dot(val, val) / (size(X, 1) - rs[4].alg.ddof))

        @test r[5].alg.ddof == 3
        @test r[5].alg.formulation === formulation
        @test r[5].mu === pr1.mu
        val = X * w .- dot(w, pr1.mu)
        val = val[val .<= zero(eltype(val))]
        @test expected_risk(rs[5], w, X) == dot(val, val) / (size(X, 1) - rs[5].alg.ddof)

        @test isnothing(r[6].alg.w)
        @test r[6].mu === pr1.mu
        val = X * w .- dot(w, pr1.mu)
        @test expected_risk(rs[6], w, X) == mean(abs.(val))

        @test r[7].alg.w === ew
        @test r[7].mu === pr1.mu
        val = X * w .- dot(w, pr1.mu)
        @test expected_risk(rs[7], w, X) == mean(abs.(val), ew)

        rs = [HighOrderMoment(; settings = settings, w = ew, mu = mu),
              HighOrderMoment(; alg = FourthLowerMoment(), mu = 0),
              HighOrderMoment(; alg = FourthCentralMoment(), mu = zerovec),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = ThirdLowerMoment())),
              HighOrderMoment(;
                              alg = HighOrderDeviation(; alg = FourthLowerMoment(),
                                                       ve = SimpleVariance(; me = nothing,
                                                                           w = ew))),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()))]
        r = risk_measure_factory(rs, Ref(pr1), Ref(slv), Ref(ucs2))
        @test r[1].settings === settings
        @test r[1].w === ew
        @test r[1].mu === mu
        val = X * w .- dot(w, mu)
        val = val[val .<= zero(eltype(val))]
        @test expected_risk(rs[1], w, X) == -sum(val .^ 3) / size(X, 1)

        @test r[2].mu === 0
        val = X * w
        val = val[val .<= zero(eltype(val))]
        @test expected_risk(rs[2], w, X) == sum(val .^ 4) / size(X, 1)

        @test isa(r[3].alg, FourthCentralMoment)
        @test r[3].mu === zerovec
        val = X * w .- dot(w, zerovec)
        @test expected_risk(rs[3], w, X) == sum(val .^ 4) / size(X, 1)

        @test r[4].mu === pr1.mu
        val = X * w .- dot(w, pr1.mu)
        val = val[val .<= zero(eltype(val))]
        s = std(rs[4].alg.ve, val; mean = zero(eltype(val)))
        @test expected_risk(rs[4], w, X) == -sum(val .^ 3) / size(X, 1) / s^3

        # @test r[5].mu === pr1.mu
        # @test r[5].alg.ve.w === ew
        # @test isa(r[5].alg, HighOrderDeviation)
        # @test isa(r[5].alg.alg, FourthLowerMoment)

        # rs = [SquareRootKurtosis(; settings = s, alg = Full()),
        #       SquareRootKurtosis(; settings = s, w = ew),
        #       SquareRootKurtosis(; settings = s, mu = fill(0.0, 20)),
        #       SquareRootKurtosis(; settings = s, kt = pr1.skt),
        #       SquareRootKurtosis(; settings = s, alg = Semi())]
    end
end
