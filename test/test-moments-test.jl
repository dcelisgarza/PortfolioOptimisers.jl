@safetestset "Moments" begin
    using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test, CovarianceEstimation,
          CSV, DataFrames

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
    @testset "Coskewness" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        cses = [FullCoskewness(), SemiCoskewness()]
        sk_t = CSV.read(joinpath(@__DIR__, "./assets/CoskewnessEstimator.csv"), DataFrame)
        for i ∈ eachindex(cses)
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
        end
    end
    @testset "Cokurtosis" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        df = DataFrame()
        kes = [FullCokurtosis(), SemiCokurtosis()]
        kt_t = CSV.read(joinpath(@__DIR__, "./assets/CokurtosisEstimator.csv"), DataFrame)
        for i ∈ eachindex(kes)
            kt = cokurtosis(kes[i], transpose(X); dims = 2)
            MN = size(kt)
            res = isapprox(kt, reshape(kt_t[!, i], MN))
            if !res
                println("Fails on cokurtosis iteration $i")
                find_tol(kt, reshape(kt_t[!, i], MN); name1 = :kt, name2 = :kt_t)
            end
            @test res
        end
    end
    @testset "Expected Returns" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ew = eweights(1:1000, 0.01; scale = true)
        mes = [SimpleExpectedReturns(), SimpleExpectedReturns(; w = ew),
               JamesSteinExpectedReturns(),
               JamesSteinExpectedReturns(; target = SERT_VolatilityWeighted()),
               JamesSteinExpectedReturns(; target = SERT_MeanSquareError()),
               JamesSteinExpectedReturns(; me = SimpleExpectedReturns(; w = ew)),
               JamesSteinExpectedReturns(; me = SimpleExpectedReturns(; w = ew),
                                         target = SERT_VolatilityWeighted()),
               JamesSteinExpectedReturns(; me = SimpleExpectedReturns(; w = ew),
                                         target = SERT_MeanSquareError()),
               BayesSteinExpectedReturns(),
               BayesSteinExpectedReturns(; target = SERT_VolatilityWeighted()),
               BayesSteinExpectedReturns(; target = SERT_MeanSquareError()),
               BayesSteinExpectedReturns(; me = SimpleExpectedReturns(; w = ew)),
               BayesSteinExpectedReturns(; me = SimpleExpectedReturns(; w = ew),
                                         target = SERT_VolatilityWeighted()),
               BayesSteinExpectedReturns(; me = SimpleExpectedReturns(; w = ew),
                                         target = SERT_MeanSquareError()),
               BodnarOkhrinParolyaExpectedReturns(),
               BodnarOkhrinParolyaExpectedReturns(; target = SERT_VolatilityWeighted()),
               BodnarOkhrinParolyaExpectedReturns(; target = SERT_MeanSquareError()),
               BodnarOkhrinParolyaExpectedReturns(; me = SimpleExpectedReturns(; w = ew)),
               BodnarOkhrinParolyaExpectedReturns(; me = SimpleExpectedReturns(; w = ew),
                                                  target = SERT_VolatilityWeighted()),
               BodnarOkhrinParolyaExpectedReturns(; me = SimpleExpectedReturns(; w = ew),
                                                  target = SERT_MeanSquareError()),
               EquilibriumExpectedReturns(), EquilibriumExpectedReturns(; l = 2),
               ExcessExpectedReturns(), ExcessExpectedReturns(; rf = 0.01)]
        ert = CSV.read(joinpath(@__DIR__, "./assets/Expected-Returns.csv"), DataFrame)
        for i ∈ eachindex(mes)
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
    @testset "Absolute Distances" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        des = [SimpleAbsoluteDistance(), SimpleAbsoluteDistanceDistance(), LogDistance(),
               LogDistanceDistance()]

        dist_t = CSV.read(joinpath(@__DIR__, "./assets/Absolute-Distance.csv"), DataFrame)

        ce = PortfolioOptimisersCovariance()
        for i ∈ 1:ncol(dist_t)
            dist1 = distance(des[i], ce, X)
            MN = size(dist1)
            res1 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on Absolute Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res1

            dist2 = distance(des[i], cov(ce, X), X)
            res2 = isapprox(dist1, dist2)
            if !res2
                println("Fails on Absolute Distance method comparison iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist2)
            end
            @test res2
        end

        des = [GeneralAbsoluteDistance(), GeneralAbsoluteDistanceDistance(),
               GeneralLogDistance(), GeneralLogDistanceDistance()]
        for i ∈ 1:ncol(dist_t)
            dist1 = distance(des[i], ce, X)
            MN = size(dist1)
            res1 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on General Absolute Distance Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res1

            dist2 = distance(des[i], cov(ce, X), X)
            res2 = isapprox(dist1, dist2)
            if !res2
                println("Fails on General Absolute Distance Distance method comparison iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist2)
            end
            @test res2
        end
    end
    @testset "Canonical and General Canonical Distance" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ces = [PortfolioOptimisersCovariance(), FullCovariance(), SemiCovariance(),
               SpearmanCovariance(), KendallCovariance(), MutualInfoCovariance(),
               MutualInfoCovariance(; bins = 5), DistanceCovariance(),
               LTDCovariance(; alpha = 0.15), Gerber0Covariance(),
               Gerber0NormalisedCovariance(), Gerber1Covariance(),
               Gerber1NormalisedCovariance(), Gerber2Covariance(),
               Gerber2NormalisedCovariance(), SmythBroby0Covariance(),
               SmythBroby0NormalisedCovariance(), SmythBroby1Covariance(),
               SmythBroby1NormalisedCovariance(), SmythBroby2Covariance(),
               SmythBroby2NormalisedCovariance(), SmythBrobyGerber0Covariance(),
               SmythBrobyGerber0NormalisedCovariance(), SmythBrobyGerber1Covariance(),
               SmythBrobyGerber1NormalisedCovariance(), SmythBrobyGerber2Covariance(),
               SmythBrobyGerber2NormalisedCovariance()]
        dist_t = CSV.read(joinpath(@__DIR__, "./assets/Canonical-Distance.csv"), DataFrame)

        de = CanonicalDistance()
        for i ∈ 1:ncol(dist_t)
            dist1 = distance(de, ces[i], transpose(X); dims = 2)
            MN = size(dist1)
            res1 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on Canonical Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res1

            dist2 = if isa(ces[i], MutualInfoCovariance)
                distance(VariationInfoDistance(; bins = ces[i].bins,
                                               normalise = ces[i].normalise),
                         cov(ces[i], X), X)
            elseif isa(ces[i], DistanceCovariance)
                distance(CorrelationDistance(;), cov(ces[i], X), X)
            elseif isa(ces[i], LTDCovariance)
                distance(LogDistance(), cov(ces[i], X), X)
            else
                distance(de, cov(ces[i], X), X)
            end
            res2 = isapprox(dist1, dist2)
            if !res2
                println("Fails on Canonical Distance method comparison iteration $i")
                find_tol(dist1, dist2; name1 = :dist1, name2 = :dist2)
            end
            @test res2
        end

        de = GeneralCanonicalDistance()
        for i ∈ 1:ncol(dist_t)
            dist1 = distance(de, ces[i], transpose(X); dims = 2)
            MN = size(dist1)
            res1 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on General Canonical Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res1

            dist2 = if isa(ces[i], MutualInfoCovariance)
                distance(GeneralVariationInfoDistance(; bins = ces[i].bins,
                                                      normalise = ces[i].normalise),
                         cov(ces[i], X), X)
            elseif isa(ces[i], DistanceCovariance)
                distance(GeneralCorrelationDistance(;), cov(ces[i], X), X)
            elseif isa(ces[i], LTDCovariance)
                distance(GeneralLogDistance(), cov(ces[i], X), X)
            else
                distance(de, cov(ces[i], X), X)
            end
            res2 = isapprox(dist1, dist2)
            if !res2
                println("Fails on General Canonical Distance method comparison iteration $i")
                find_tol(dist1, dist2; name1 = :dist1, name2 = :dist2)
            end
            @test res2
        end
    end
    @testset "Canonical and General Canonical Distance Distance" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ces = [PortfolioOptimisersCovariance(), FullCovariance(), SemiCovariance(),
               SpearmanCovariance(), KendallCovariance(), MutualInfoCovariance(),
               MutualInfoCovariance(; bins = 5), DistanceCovariance(),
               LTDCovariance(; alpha = 0.15), Gerber0Covariance(),
               Gerber0NormalisedCovariance(), Gerber1Covariance(),
               Gerber1NormalisedCovariance(), Gerber2Covariance(),
               Gerber2NormalisedCovariance(), SmythBroby0Covariance(),
               SmythBroby0NormalisedCovariance(), SmythBroby1Covariance(),
               SmythBroby1NormalisedCovariance(), SmythBroby2Covariance(),
               SmythBroby2NormalisedCovariance(), SmythBrobyGerber0Covariance(),
               SmythBrobyGerber0NormalisedCovariance(), SmythBrobyGerber1Covariance(),
               SmythBrobyGerber1NormalisedCovariance(), SmythBrobyGerber2Covariance(),
               SmythBrobyGerber2NormalisedCovariance()]
        dist_t = CSV.read(joinpath(@__DIR__, "./assets/Canonical-Distance-Distance.csv"),
                          DataFrame)

        de = CanonicalDistanceDistance()
        for i ∈ 1:ncol(dist_t)
            dist1 = distance(de, ces[i], transpose(X); dims = 2)
            MN = size(dist1)
            res1 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on Canonical Distance Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res1

            dist2 = if isa(ces[i], MutualInfoCovariance)
                distance(VariationInfoDistanceDistance(; bins = ces[i].bins,
                                                       normalise = ces[i].normalise),
                         cov(ces[i], X), X)
            elseif isa(ces[i], DistanceCovariance)
                distance(CorrelationDistanceDistance(;), cov(ces[i], X), X)
            elseif isa(ces[i], LTDCovariance)
                distance(LogDistanceDistance(), cov(ces[i], X), X)
            else
                distance(de, cov(ces[i], X), X)
            end
            res2 = isapprox(dist1, dist2)
            if !res2
                println("Fails on Canonical Distance Distance method comparison iteration $i")
                find_tol(dist1, dist2; name1 = :dist1, name2 = :dist2)
            end
            @test res2
        end

        de = GeneralCanonicalDistanceDistance()
        for i ∈ 1:ncol(dist_t)
            dist1 = distance(de, ces[i], transpose(X); dims = 2)
            MN = size(dist1)
            res1 = isapprox(dist1, reshape(dist_t[!, i], MN))
            if !res1
                println("Fails on General Canonical Distance Distance iteration $i")
                find_tol(dist1, reshape(dist_t[!, i], MN); name1 = :dist1, name2 = :dist1_t)
            end
            @test res1

            dist2 = if isa(ces[i], MutualInfoCovariance)
                distance(GeneralVariationInfoDistanceDistance(; bins = ces[i].bins,
                                                              normalise = ces[i].normalise),
                         cov(ces[i], X), X)
            elseif isa(ces[i], DistanceCovariance)
                distance(GeneralCorrelationDistanceDistance(;), cov(ces[i], X), X)
            elseif isa(ces[i], LTDCovariance)
                distance(GeneralLogDistanceDistance(), cov(ces[i], X), X)
            else
                distance(de, cov(ces[i], X), X)
            end
            res2 = isapprox(dist1, dist2)
            if !res2
                println("Fails on General Canonical Distance Distance method comparison iteration $i")
                find_tol(dist1, dist2; name1 = :dist1, name2 = :dist2)
            end
            @test res2
        end
    end
    @testset "Covariance and Correlation correctness" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        fw = FrequencyWeights(rand(rng, 1000))
        ew = eweights(1:1000, 0.01; scale = true)
        ces = [PortfolioOptimisersCovariance(), FullCovariance(),
               FullCovariance(; me = SimpleExpectedReturns(; w = ew),
                              ce = GeneralWeightedCovariance(;
                                                             ce = SimpleCovariance(;
                                                                                   corrected = false),
                                                             w = ew)),
               FullCovariance(;
                              ce = GeneralWeightedCovariance(;
                                                             ce = AnalyticalNonlinearShrinkage())),
               FullCovariance(; me = SimpleExpectedReturns(; w = fw),
                              ce = GeneralWeightedCovariance(;
                                                             ce = AnalyticalNonlinearShrinkage(),
                                                             w = fw)), SemiCovariance(),
               SemiCovariance(; me = SimpleExpectedReturns(; w = ew),
                              ce = GeneralWeightedCovariance(;
                                                             ce = SimpleCovariance(;
                                                                                   corrected = false),
                                                             w = ew)),
               SemiCovariance(;
                              ce = GeneralWeightedCovariance(;
                                                             ce = AnalyticalNonlinearShrinkage())),
               SemiCovariance(; me = SimpleExpectedReturns(; w = fw),
                              ce = GeneralWeightedCovariance(;
                                                             ce = AnalyticalNonlinearShrinkage(),
                                                             w = fw)), SpearmanCovariance(),
               KendallCovariance(), MutualInfoCovariance(),
               MutualInfoCovariance(; bins = B_Knuth()),
               MutualInfoCovariance(; bins = B_FreedmanDiaconis()),
               MutualInfoCovariance(; bins = B_Scott()), MutualInfoCovariance(; bins = 5),
               MutualInfoCovariance(;
                                    ve = PortfolioOptimisers.SimpleVariance(;
                                                                            corrected = false,
                                                                            w = ew)),
               DistanceCovariance(), DistanceCovariance(; w = ew), LTDCovariance(),
               Gerber0Covariance(),
               Gerber0Covariance(;
                                 ve = PortfolioOptimisers.SimpleVariance(;
                                                                         corrected = false,
                                                                         w = ew)),
               Gerber0NormalisedCovariance(),
               Gerber0NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = PortfolioOptimisers.SimpleVariance(;
                                                                                   corrected = false,
                                                                                   w = ew)),
               Gerber1Covariance(),
               Gerber1Covariance(;
                                 ve = PortfolioOptimisers.SimpleVariance(;
                                                                         corrected = false,
                                                                         w = ew)),
               Gerber1NormalisedCovariance(),
               Gerber1NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = PortfolioOptimisers.SimpleVariance(;
                                                                                   corrected = false,
                                                                                   w = ew)),
               Gerber2Covariance(),
               Gerber2Covariance(;
                                 ve = PortfolioOptimisers.SimpleVariance(;
                                                                         corrected = false,
                                                                         w = ew)),
               Gerber2NormalisedCovariance(),
               Gerber2NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = PortfolioOptimisers.SimpleVariance(;
                                                                                   corrected = false,
                                                                                   w = ew)),
               SmythBroby0Covariance(),
               SmythBroby0Covariance(; me = SimpleExpectedReturns(; w = ew),
                                     ve = PortfolioOptimisers.SimpleVariance(;
                                                                             corrected = false,
                                                                             w = ew)),
               SmythBroby0NormalisedCovariance(),
               SmythBroby0NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                               ve = PortfolioOptimisers.SimpleVariance(;
                                                                                       corrected = false,
                                                                                       w = ew)),
               SmythBroby1Covariance(),
               SmythBroby1Covariance(; me = SimpleExpectedReturns(; w = ew),
                                     ve = PortfolioOptimisers.SimpleVariance(;
                                                                             corrected = false,
                                                                             w = ew)),
               SmythBroby1NormalisedCovariance(),
               SmythBroby1NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                               ve = PortfolioOptimisers.SimpleVariance(;
                                                                                       corrected = false,
                                                                                       w = ew)),
               SmythBroby2Covariance(),
               SmythBroby2Covariance(; me = SimpleExpectedReturns(; w = ew),
                                     ve = PortfolioOptimisers.SimpleVariance(;
                                                                             corrected = false,
                                                                             w = ew)),
               SmythBroby2NormalisedCovariance(),
               SmythBroby2NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                               ve = PortfolioOptimisers.SimpleVariance(;
                                                                                       corrected = false,
                                                                                       w = ew)),
               SmythBrobyGerber0Covariance(),
               SmythBrobyGerber0Covariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = PortfolioOptimisers.SimpleVariance(;
                                                                                   corrected = false,
                                                                                   w = ew)),
               SmythBrobyGerber0NormalisedCovariance(),
               SmythBrobyGerber0NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                                     ve = PortfolioOptimisers.SimpleVariance(;
                                                                                             corrected = false,
                                                                                             w = ew)),
               SmythBrobyGerber1Covariance(),
               SmythBrobyGerber1Covariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = PortfolioOptimisers.SimpleVariance(;
                                                                                   corrected = false,
                                                                                   w = ew)),
               SmythBrobyGerber1NormalisedCovariance(),
               SmythBrobyGerber1NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                                     ve = PortfolioOptimisers.SimpleVariance(;
                                                                                             corrected = false,
                                                                                             w = ew)),
               SmythBrobyGerber2Covariance(),
               SmythBrobyGerber2Covariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = PortfolioOptimisers.SimpleVariance(;
                                                                                   corrected = false,
                                                                                   w = ew)),
               SmythBrobyGerber2NormalisedCovariance(),
               SmythBrobyGerber2NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                                     ve = PortfolioOptimisers.SimpleVariance(;
                                                                                             corrected = false,
                                                                                             w = ew))]
        cvrt = CSV.read(joinpath(@__DIR__,
                                 "./assets/Covariance-and-Correlation-correctness.csv"),
                        DataFrame)
        for (i, j) ∈ zip(1:55, 1:2:ncol(cvrt))
            cv = cov(ces[i], X)
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
        end
    end
    @testset "cov2cor" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        fw = FrequencyWeights(rand(rng, 100))
        ew = eweights(1:100, 0.3; scale = true)

        ces = [PortfolioOptimisersCovariance(), FullCovariance(),
               FullCovariance(;
                              ce = GeneralWeightedCovariance(;
                                                             ce = SimpleCovariance(;
                                                                                   corrected = false),
                                                             w = ew)),
               FullCovariance(;
                              ce = GeneralWeightedCovariance(;
                                                             ce = AnalyticalNonlinearShrinkage())),
               FullCovariance(;
                              ce = GeneralWeightedCovariance(;
                                                             ce = AnalyticalNonlinearShrinkage(),
                                                             w = fw)), SemiCovariance(),
               SemiCovariance(;
                              ce = GeneralWeightedCovariance(;
                                                             ce = SimpleCovariance(;
                                                                                   corrected = false),
                                                             w = ew)),
               SemiCovariance(;
                              ce = GeneralWeightedCovariance(;
                                                             ce = AnalyticalNonlinearShrinkage())),
               SemiCovariance(;
                              ce = GeneralWeightedCovariance(;
                                                             ce = AnalyticalNonlinearShrinkage(),
                                                             w = fw)), SpearmanCovariance(),
               KendallCovariance(), MutualInfoCovariance(),
               MutualInfoCovariance(; bins = B_Knuth()),
               MutualInfoCovariance(; bins = B_FreedmanDiaconis()),
               MutualInfoCovariance(; bins = B_Scott()), MutualInfoCovariance(; bins = 5),
               MutualInfoCovariance(; ve = SimpleVariance(; corrected = false, w = ew)),
               DistanceCovariance(), DistanceCovariance(; w = ew), LTDCovariance(),
               Gerber0Covariance(),
               Gerber0Covariance(; ve = SimpleVariance(; corrected = false, w = ew)),
               Gerber0NormalisedCovariance(),
               Gerber0NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = SimpleVariance(; corrected = false, w = ew)),
               Gerber1Covariance(),
               Gerber1Covariance(; ve = SimpleVariance(; corrected = false, w = ew)),
               Gerber1NormalisedCovariance(),
               Gerber1NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = SimpleVariance(; corrected = false, w = ew)),
               Gerber2Covariance(),
               Gerber2Covariance(; ve = SimpleVariance(; corrected = false, w = ew)),
               Gerber2NormalisedCovariance(),
               Gerber2NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = SimpleVariance(; corrected = false, w = ew)),
               SmythBroby0Covariance(),
               SmythBroby0Covariance(; me = SimpleExpectedReturns(; w = ew),
                                     ve = SimpleVariance(; corrected = false, w = ew)),
               SmythBroby0NormalisedCovariance(),
               SmythBroby0NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                               ve = SimpleVariance(; corrected = false,
                                                                   w = ew)),
               SmythBroby1Covariance(),
               SmythBroby1Covariance(; me = SimpleExpectedReturns(; w = ew),
                                     ve = SimpleVariance(; corrected = false, w = ew)),
               SmythBroby1NormalisedCovariance(),
               SmythBroby1NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                               ve = SimpleVariance(; corrected = false,
                                                                   w = ew)),
               SmythBroby2Covariance(),
               SmythBroby2Covariance(; me = SimpleExpectedReturns(; w = ew),
                                     ve = SimpleVariance(; corrected = false, w = ew)),
               SmythBroby2NormalisedCovariance(),
               SmythBroby2NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                               ve = SimpleVariance(; corrected = false,
                                                                   w = ew)),
               SmythBrobyGerber0Covariance(),
               SmythBrobyGerber0Covariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = SimpleVariance(; corrected = false, w = ew)),
               SmythBrobyGerber0NormalisedCovariance(),
               SmythBrobyGerber0NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                                     ve = SimpleVariance(;
                                                                         corrected = false,
                                                                         w = ew)),
               SmythBrobyGerber1Covariance(),
               SmythBrobyGerber1Covariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = SimpleVariance(; corrected = false, w = ew)),
               SmythBrobyGerber1NormalisedCovariance(),
               SmythBrobyGerber1NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                                     ve = SimpleVariance(;
                                                                         corrected = false,
                                                                         w = ew)),
               SmythBrobyGerber2Covariance(),
               SmythBrobyGerber2Covariance(; me = SimpleExpectedReturns(; w = ew),
                                           ve = SimpleVariance(; corrected = false, w = ew)),
               SmythBrobyGerber2NormalisedCovariance(),
               SmythBrobyGerber2NormalisedCovariance(; me = SimpleExpectedReturns(; w = ew),
                                                     ve = SimpleVariance(;
                                                                         corrected = false,
                                                                         w = ew))]

        for (i, ce) ∈ pairs(ces)
            cv = cov(ce, X)
            cr = cor(ce, X)
            res1 = isapprox(if isa(cv, Matrix)
                                StatsBase.cov2cor(cv)
                            else
                                StatsBase.cov2cor(Matrix(cv))
                            end, cr)
            if !res1
                println("Test $i fails on:\n$(ce)\n")
                find_tol(if isa(cv, Matrix)
                             StatsBase.cov2cor(cv)
                         else
                             StatsBase.cov2cor(Matrix(cv))
                         end, cr; name1 = :cv, name2 = :cr)
            end
            @test res1

            cvt = cov(ce, Matrix(transpose(X)); dims = 2)
            crt = cor(ce, Matrix(transpose(X)); dims = 2)
            res2 = isapprox(if isa(cvt, Matrix)
                                StatsBase.cov2cor(cvt)
                            else
                                StatsBase.cov2cor(Matrix(cvt))
                            end, crt)
            if !res2
                println("Test `dims = 2` $i fails on:\n$(ce)\n")
                find_tol(if isa(cvt, Matrix)
                             StatsBase.cov2cor(cvt)
                         else
                             StatsBase.cov2cor(Matrix(cvt))
                         end, crt; name1 = :cvt, name2 = :crt)
            end
            @test res2

            res3 = isapprox(cv, cvt)
            if !res3
                find_tol(cv, cvt; name1 = :cv, name2 = :cvt)
            end
            @test res3

            res4 = isapprox(cr, crt)
            if !res4
                find_tol(cr, crt; name1 = :cr, name2 = :crt)
            end
            @test res4
        end
    end
    @testset "SimpleVariance" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        fw = FrequencyWeights(rand(rng, 100))
        ew = eweights(1:100, 0.3; scale = true)

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
    end
    @testset "Misc tests" begin
        @test iszero(PortfolioOptimisers.intrinsic_mutual_info(rand(1, 1)))
    end
end
