#=
@safetestset "Moments" begin
    using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test, CovarianceEstimation,
          CSV, DataFrames, TimeSeries
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    rng = StableRNG(123456789)
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                  timestamp = :timestamp)
    rd = prices_to_returns(X[(end - 252):end])
    X = rd.X
    ew = eweights(1:252, inv(252); scale = true)
    fw = fweights(rand(rng, 252))
    rf = 4.34 / 100 / 252
    @testset "Expected Returns" begin
        mes = [SimpleExpectedReturns(), SimpleExpectedReturns(; w = ew),
               ShrunkExpectedReturns(; alg = JamesStein()),
               ShrunkExpectedReturns(; alg = JamesStein(; target = VolatilityWeighted())),
               ShrunkExpectedReturns(; alg = JamesStein(; target = MeanSquareError())),
               ShrunkExpectedReturns(; alg = JamesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = JamesStein(; target = VolatilityWeighted()),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = JamesStein(; target = MeanSquareError()),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BayesStein()),
               ShrunkExpectedReturns(; alg = BayesStein(; target = VolatilityWeighted())),
               ShrunkExpectedReturns(; alg = BayesStein(; target = MeanSquareError())),
               ShrunkExpectedReturns(; alg = BayesStein(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BayesStein(; target = VolatilityWeighted()),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BayesStein(; target = MeanSquareError()),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(; alg = BodnarOkhrinParolya()),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = VolatilityWeighted())),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = MeanSquareError())),
               ShrunkExpectedReturns(; alg = BodnarOkhrinParolya(),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = VolatilityWeighted()),
                                     me = SimpleExpectedReturns(; w = ew)),
               ShrunkExpectedReturns(;
                                     alg = BodnarOkhrinParolya(;
                                                               target = MeanSquareError()),
                                     me = SimpleExpectedReturns(; w = ew)),
               EquilibriumExpectedReturns(), EquilibriumExpectedReturns(; l = 2),
               ExcessExpectedReturns(), ExcessExpectedReturns(; rf = rf)]
        ert = CSV.read(joinpath(@__DIR__, "./assets/Expected-Returns.csv"), DataFrame)
        for i in eachindex(mes)
            er = mean(mes[i], X)
            res = isapprox(er, reshape(ert[!, i], size(er)))
            if !res
                println("Test $i fails on:\n$(me[i])\n$(res)\n")
                find_tol(er, reshape(ert[!, i], size(er)); name1 = :er, name2 = :er_t)
            end
            @test res
        end
        @test_throws AssertionError EquilibriumExpectedReturns(w = [])
    end
    @testset "Covariance and Correlation correctness" begin
        ces = [PortfolioOptimisersCovariance(), Covariance(; alg = Full()),
               Covariance(; alg = Full(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralWeightedCovariance(;
                                                         ce = SimpleCovariance(;
                                                                               corrected = false),
                                                         w = ew)),
               Covariance(; alg = Full(),
                          ce = GeneralWeightedCovariance(;
                                                         ce = AnalyticalNonlinearShrinkage())),
               Covariance(; alg = Full(), me = SimpleExpectedReturns(; w = fw),
                          ce = GeneralWeightedCovariance(;
                                                         ce = AnalyticalNonlinearShrinkage(),
                                                         w = fw)),
               Covariance(; alg = Semi()),
               Covariance(; alg = Semi(), me = SimpleExpectedReturns(; w = ew),
                          ce = GeneralWeightedCovariance(;
                                                         ce = SimpleCovariance(;
                                                                               corrected = false),
                                                         w = ew)),
               Covariance(; alg = Semi(),
                          ce = GeneralWeightedCovariance(;
                                                         ce = AnalyticalNonlinearShrinkage())),
               Covariance(; alg = Semi(), me = SimpleExpectedReturns(; w = fw),
                          ce = GeneralWeightedCovariance(;
                                                         ce = AnalyticalNonlinearShrinkage(),
                                                         w = fw)), SpearmanCovariance(),
               KendallCovariance(), MutualInfoCovariance(),
               MutualInfoCovariance(; bins = Knuth()),
               MutualInfoCovariance(; bins = FreedmanDiaconis()),
               MutualInfoCovariance(; bins = Scott()), MutualInfoCovariance(; bins = 5),
               MutualInfoCovariance(;
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               DistanceCovariance(), DistanceCovariance(; w = ew), LTDCovariance(),
               GerberCovariance(; alg = Gerber0()),
               GerberCovariance(; alg = Gerber0(),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = NormalisedGerber0()),
               GerberCovariance(;
                                alg = NormalisedGerber0(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew)),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = Gerber1()),
               GerberCovariance(; alg = Gerber1(),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = NormalisedGerber1()),
               GerberCovariance(;
                                alg = NormalisedGerber1(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew)),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = Gerber2()),
               GerberCovariance(; alg = Gerber2(),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               GerberCovariance(; alg = NormalisedGerber2()),
               GerberCovariance(;
                                alg = NormalisedGerber2(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew)),
                                ve = SimpleVariance(; me = SimpleExpectedReturns(; w = ew),
                                                    corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBroby0()),
               SmythBrobyCovariance(; alg = SmythBroby0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBroby1()),
               SmythBrobyCovariance(; alg = SmythBroby1(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby1(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBroby2()),
               SmythBrobyCovariance(; alg = SmythBroby2(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(; corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby2(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber0(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber1(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber1(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = SmythBrobyGerber2()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber2(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew)),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber2(),
                                    me = SimpleExpectedReturns(; w = ew),
                                    ve = SimpleVariance(;
                                                        me = SimpleExpectedReturns(;
                                                                                   w = ew),
                                                        corrected = false, w = ew))]
        cvrt = CSV.read(joinpath(@__DIR__,
                                 "./assets/Covariance-and-Correlation-correctness.csv"),
                        DataFrame)
        for (i, j) in zip(1:length(ces), 1:2:(2 * length(ces)))
            cv = cov(ces[i], Matrix(transpose(X)); dims = 2)
            cr = cor(ces[i], X)
            MN = size(cv)
            res1 = isapprox(cv, reshape(cvrt[!, j], MN))
            res2 = isapprox(cr, reshape(cvrt[!, j + 1], MN))
            if !res1
                println("Fails on cov iteration $i")
                find_tol(cv, reshape(cvrt[!, j], MN); name1 = :cv, name2 = :cv_t)
            end
            @test res1
            if !res2
                println("Fails on cor iteration $i")
                find_tol(cr, reshape(cvrt[!, j + 1], MN); name1 = :cr, name2 = :cr_t)
            end
            @test res2
            res3 = isapprox(if isa(cv, Matrix)
                                StatsBase.cov2cor(cv)
                            else
                                StatsBase.cov2cor(Matrix(cv))
                            end, cr)
            if !res3
                println("Test cov2cor $i fails on:\n$(ce)\n")
                find_tol(if isa(cv, Matrix)
                             StatsBase.cov2cor(cv)
                         else
                             StatsBase.cov2cor(Matrix(cv))
                         end, cr; name1 = :cv, name2 = :cr)
            end
            @test res3
        end
    end
    @testset "SimpleVariance" begin
        ve1 = SimpleVariance(; corrected = true)
        ve2 = SimpleVariance(; corrected = false, w = ew)
        s1 = std(ve1, X)
        s2 = std(ve2, X)
        v1 = var(ve1, X)
        v2 = var(ve2, X)
        @test isapprox(s1, sqrt.(v1))
        @test isapprox(s2, sqrt.(v2))
        @test !isapprox(s1, s2)
        @test !isapprox(v1, v2)

        @test isapprox(var(ve1, X[:, 1]), var(X[:, 1]; corrected = true))
        @test isapprox(var(ve2, X[:, 1]), var(X[:, 1], ew; corrected = false))
    end
    @testset "Misc tests" begin
        @test iszero(PortfolioOptimisers.intrinsic_mutual_info(rand(1, 1)))
    end
    @testset "Coskewness" begin
        cses = [Coskewness(; alg = Full()), Coskewness(; alg = Semi())]
        sk_t = CSV.read(joinpath(@__DIR__, "./assets/CoskewnessEstimator.csv"), DataFrame)
        for i in eachindex(cses)
            sk, v = coskewness(cses[i], transpose(X); dims = 2)
            MN1 = size(sk)
            MN2 = size(v)
            res1 = isapprox(sk, reshape(sk_t[1:prod(MN1), i], MN1))
            res2 = isapprox(v, reshape(sk_t[(1 + prod(MN1)):end, i], MN2))
            if !res1
                println("Fails on coskewness estimator iteration $i")
                find_tol(sk, reshape(sk_t[1:prod(MN1), i], MN1); name1 = :sk, name2 = :sk_t)
            end
            @test res1
            if !res2
                println("Fails on spectral matrix estimator iteration $i")
                find_tol(v, reshape(v_t[1:prod(MN1), i], MN1); name1 = :v, name2 = :v_t)
            end
            @test res2

            a, b = coskewness(nothing)
            @test isnothing(a)
            @test isnothing(b)
        end
    end
    @testset "Cokurtosis" begin
        kes = [Cokurtosis(; alg = Full()), Cokurtosis(; alg = Semi())]
        kt_t = CSV.read(joinpath(@__DIR__, "./assets/CokurtosisEstimator.csv"), DataFrame)
        for i in eachindex(kes)
            kt = cokurtosis(kes[i], transpose(X); dims = 2)
            MN = size(kt)
            res = isapprox(kt, reshape(kt_t[!, i], MN))
            if !res
                println("Fails on cokurtosis iteration $i")
                find_tol(kt, reshape(kt_t[!, i], MN); name1 = :kt, name2 = :kt_t)
            end
            @test res
        end
        @test isnothing(cokurtosis(nothing))
    end
    @testset "Absolute Distances" begin
        des = [Distance(; alg = SimpleAbsoluteDistance()),
               DistanceDistance(; alg = SimpleAbsoluteDistance()),
               Distance(; alg = LogDistance()), DistanceDistance(; alg = LogDistance())]

        desg = [GeneralDistance(; alg = SimpleAbsoluteDistance()),
                GeneralDistanceDistance(; alg = SimpleAbsoluteDistance()),
                GeneralDistance(; alg = LogDistance()),
                GeneralDistanceDistance(; alg = LogDistance())]

        dist_t = CSV.read(joinpath(@__DIR__, "./assets/Absolute-Distance.csv"), DataFrame)
        ce = PortfolioOptimisersCovariance()
        for i in eachindex(des)
            dist1 = distance(des[i], ce, X)
            dist2 = distance(des[i], cov(ce, X), X)
            dist1g = distance(desg[i], ce, X)
            dist2g = distance(desg[i], cov(ce, X), X)
            MN = size(dist1)
            res1 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on Absolute Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res1

            res2 = isapprox(dist1, dist2)
            if !res2
                println("Fails on Absolute Distance method comparison iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist2)
            end
            @test res2

            res3 = isapprox(dist1, dist1g)
            if !res3
                println("Fails on General Absolute Distance iteration $i")
                find_tol(dist1, dist1g; name1 = :dist1, name2 = :dist1g)
            end
            @test res3

            res4 = isapprox(dist2, dist2g)
            if !res4
                println("Fails on General Absolute Distance method comparison iteration $i")
                find_tol(dist2, dist2g; name1 = :dist2, name2 = :dist2g)
            end
        end
    end
    @testset "Canonical and General Canonical Distance" begin
        ces = [PortfolioOptimisersCovariance(), Covariance(; alg = Full()),
               Covariance(; alg = Semi()), SpearmanCovariance(), KendallCovariance(),
               MutualInfoCovariance(), MutualInfoCovariance(; bins = 5),
               DistanceCovariance(), LTDCovariance(; alpha = 0.15),
               GerberCovariance(; alg = Gerber0()),
               GerberCovariance(; alg = NormalisedGerber0()),
               GerberCovariance(; alg = Gerber1()),
               GerberCovariance(; alg = NormalisedGerber1()),
               GerberCovariance(; alg = Gerber2()),
               GerberCovariance(; alg = NormalisedGerber2()),
               SmythBrobyCovariance(; alg = SmythBroby0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby0()),
               SmythBrobyCovariance(; alg = SmythBroby1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby1()),
               SmythBrobyCovariance(; alg = SmythBroby2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby2()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber2())]
        dist_t = CSV.read(joinpath(@__DIR__, "./assets/Canonical-Distance.csv"), DataFrame)
        de = Distance(; alg = CanonicalDistance())
        deg = GeneralDistance(; alg = CanonicalDistance())
        for i in eachindex(ces)
            dist1 = distance(de, ces[i], transpose(X); dims = 2)
            dist2 = if isa(ces[i], MutualInfoCovariance)
                distance(Distance(;
                                  alg = VariationInfoDistance(; bins = ces[i].bins,
                                                              normalise = ces[i].normalise)),
                         cov(ces[i], X), X)
            elseif isa(ces[i], DistanceCovariance)
                distance(Distance(; alg = CorrelationDistance(;)), cov(ces[i], X), X)
            elseif isa(ces[i], LTDCovariance)
                distance(Distance(; alg = LogDistance()), cov(ces[i], X), X)
            else
                distance(de, cov(ces[i], X), X)
            end
            dist3 = distance(deg, ces[i], transpose(X); dims = 2)
            dist4 = if isa(ces[i], MutualInfoCovariance)
                distance(GeneralDistance(;
                                         alg = VariationInfoDistance(; bins = ces[i].bins,
                                                                     normalise = ces[i].normalise)),
                         cov(ces[i], X), X)
            elseif isa(ces[i], DistanceCovariance)
                distance(GeneralDistance(; alg = CorrelationDistance(;)), cov(ces[i], X), X)
            elseif isa(ces[i], LTDCovariance)
                distance(GeneralDistance(; alg = LogDistance()), cov(ces[i], X), X)
            else
                distance(deg, cov(ces[i], X), X)
            end
            MN = size(dist1)
            res1 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on Canonical Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res1

            res2 = isapprox(dist1, dist2)
            if !res2
                println("Fails on Canonical Distance method comparison iteration $i")
                find_tol(dist1, dist2; name1 = :dist1, name2 = :dist2)
            end
            @test res2

            MN = size(dist3)
            res1 = isapprox(dist3, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on General Canonical Distance iteration $i")
                find_tol(dist3, reshape(dist_t[!, i], MN); name1 = :dist3, name2 = :dist3_t)
            end
            @test res1

            res2 = isapprox(dist3, dist4)
            if !res2
                println("Fails on General Canonical Distance method comparison iteration $i")
                find_tol(dist3, dist4; name1 = :dist3, name2 = :dist4)
            end
            @test res2
        end
    end
    @testset "Canonical and General Canonical Distance Distance" begin
        ces = [PortfolioOptimisersCovariance(), Covariance(; alg = Full()),
               Covariance(; alg = Semi()), SpearmanCovariance(), KendallCovariance(),
               MutualInfoCovariance(), MutualInfoCovariance(; bins = 5),
               DistanceCovariance(), LTDCovariance(; alpha = 0.15),
               GerberCovariance(; alg = Gerber0()),
               GerberCovariance(; alg = NormalisedGerber0()),
               GerberCovariance(; alg = Gerber1()),
               GerberCovariance(; alg = NormalisedGerber1()),
               GerberCovariance(; alg = Gerber2()),
               GerberCovariance(; alg = NormalisedGerber2()),
               SmythBrobyCovariance(; alg = SmythBroby0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby0()),
               SmythBrobyCovariance(; alg = SmythBroby1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby1()),
               SmythBrobyCovariance(; alg = SmythBroby2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBroby2()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber0()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber1()),
               SmythBrobyCovariance(; alg = SmythBrobyGerber2()),
               SmythBrobyCovariance(; alg = NormalisedSmythBrobyGerber2())]
        dist_t = CSV.read(joinpath(@__DIR__, "./assets/Canonical-Distance-Distance.csv"),
                          DataFrame)
        de = DistanceDistance(; alg = CanonicalDistance())
        deg = GeneralDistanceDistance(; alg = CanonicalDistance())
        for i in eachindex(ces)
            dist1 = distance(de, ces[i], transpose(X); dims = 2)
            dist2 = if isa(ces[i], MutualInfoCovariance)
                distance(DistanceDistance(;
                                          alg = VariationInfoDistance(; bins = ces[i].bins,
                                                                      normalise = ces[i].normalise)),
                         cov(ces[i], X), X)
            elseif isa(ces[i], DistanceCovariance)
                distance(DistanceDistance(; alg = CorrelationDistance()), cov(ces[i], X), X)
            elseif isa(ces[i], LTDCovariance)
                distance(DistanceDistance(; alg = LogDistance()), cov(ces[i], X), X)
            else
                distance(de, cov(ces[i], X), X)
            end
            dist3 = distance(deg, ces[i], transpose(X); dims = 2)
            dist4 = if isa(ces[i], MutualInfoCovariance)
                distance(GeneralDistanceDistance(;
                                                 alg = VariationInfoDistance(;
                                                                             bins = ces[i].bins,
                                                                             normalise = ces[i].normalise)),
                         cov(ces[i], X), X)
            elseif isa(ces[i], DistanceCovariance)
                distance(GeneralDistanceDistance(; alg = CorrelationDistance(;)),
                         cov(ces[i], X), X)
            elseif isa(ces[i], LTDCovariance)
                distance(GeneralDistanceDistance(; alg = LogDistance()), cov(ces[i], X), X)
            else
                distance(deg, cov(ces[i], X), X)
            end

            MN = size(dist1)
            res1 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on Canonical Distance Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res1

            res2 = isapprox(dist1, dist2)
            if !res2
                println("Fails on Canonical Distance Distance method comparison iteration $i")
                find_tol(dist1, dist2; name1 = :dist1, name2 = :dist2)
            end
            @test res2

            res3 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res3
                println("Fails on General Canonical Distance Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res3

            res4 = isapprox(dist3, dist4)
            if !res4
                println("Fails on General Canonical Distance Distance method comparison iteration $i")
                find_tol(dist3, dist4; name1 = :dist3, name2 = :dist4)
            end
            @test res4
        end
    end
    @testset "Factories" begin
        rng = StableRNG(123456789)
        ew = eweights(1:10, 0.3; scale = true)
        fw = fweights(rand(rng, 10))

        me1 = ShrunkExpectedReturns(;
                                    me = ExcessExpectedReturns(;
                                                               me = EquilibriumExpectedReturns()))
        me2 = PortfolioOptimisers.factory(me1, fw)
        @test isnothing(me2.me.me.w)
        @test me2.ce.ce.ce.w === fw

        me1 = ShrunkExpectedReturns(;
                                    me = ExcessExpectedReturns(;
                                                               me = SimpleExpectedReturns()))
        me2 = PortfolioOptimisers.factory(me1, fw)
        @test me2.me.me.w === fw
        @test me2.ce.ce.ce.w === fw

        ce1 = PortfolioOptimisersCovariance(;
                                            ce = Covariance(;
                                                            ce = GeneralWeightedCovariance()))
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.ce.ce.w === fw
        @test ce2.ce.me.w == fw

        ce1 = GerberCovariance()
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.ve.w == fw

        ce1 = GerberCovariance(; alg = NormalisedGerber0())
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.alg.me.w == fw
        @test ce2.ve.w == fw

        ce1 = SmythBrobyCovariance()
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.me.w == fw
        @test ce2.ve.w == fw

        ce1 = SmythBrobyCovariance(; alg = NormalisedSmythBroby0())
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.me.w == fw
        @test ce2.ve.w == fw

        ce1 = DistanceCovariance()
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.w == fw

        ce1 = LTDCovariance()
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.ve.w == fw

        ce1 = KendallCovariance()
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.ve.w == fw

        ce1 = SpearmanCovariance()
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.ve.w == fw

        ce1 = MutualInfoCovariance()
        ce2 = PortfolioOptimisers.factory(ce1, fw)
        @test ce2.ve.w == fw

        ske1 = Coskewness()
        ske2 = PortfolioOptimisers.factory(ske1, fw)
        @test ske2.me.w == fw

        kte1 = Cokurtosis()
        kte2 = PortfolioOptimisers.factory(kte1, fw)
        @test kte2.me.w == fw
    end
end
=#
