@safetestset "Moments" begin
    using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test, CovarianceEstimation,
          CSV, DataFrames
    @testset "Absolute Distances" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        des = [SimpleAbsoluteDistance(), SimpleAbsoluteDistanceDistance(), LogDistance(),
               LogDistanceDistance()]

        dist_t = CSV.read(joinpath(@__DIR__, "./assets/Absolute-Distance.csv"), DataFrame)

        ce = FullCovariance()
        for i ∈ 1:ncol(dist_t)
            dist = distance(des[i], ce, X)
            MN = size(dist)
            res = isapprox(dist, reshape(dist_t[!, i], MN))
            if !res
                println("Fails on Absolute Distance iteration $i")
            end
            @test res
        end

        des = [GeneralAbsoluteDistance(), GeneralAbsoluteDistanceDistance(),
               GeneralLogDistance(), GeneralLogDistanceDistance()]
        for i ∈ 1:ncol(dist_t)
            dist = distance(des[i], ce, X)
            MN = size(dist)
            res = isapprox(dist, reshape(dist_t[!, i], MN))
            if !res
                println("Fails on General Absolute Distance iteration $i")
            end
            @test res
        end
    end
    @testset "Canonical and General Canonical Distance" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ces = [FullCovariance(), SemiCovariance(), SpearmanCovariance(),
               KendallCovariance(), MutualInfoCovariance(),
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
            dist = distance(de, ces[i], transpose(X); dims = 2)
            MN = size(dist)
            res = isapprox(dist, reshape(dist_t[!, i], MN))
            if !res
                println("Fails on Canonical Distance iteration $i")
            end
            @test res
        end

        de = GeneralCanonicalDistance()
        for i ∈ 1:ncol(dist_t)
            dist = distance(de, ces[i], transpose(X); dims = 2)
            MN = size(dist)
            res = isapprox(dist, reshape(dist_t[!, i], MN))
            if !res
                println("Fails on General Canonical Distance iteration $i")
            end
            @test res
        end
    end
    @testset "Canonical and General Canonical Distance Distance" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ces = [FullCovariance(), SemiCovariance(), SpearmanCovariance(),
               KendallCovariance(), MutualInfoCovariance(),
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
            dist = distance(de, ces[i], transpose(X); dims = 2)
            MN = size(dist)
            res = isapprox(dist, reshape(dist_t[!, i], MN))
            if !res
                println("Fails on Canonical Distance Distance iteration $i")
            end
            @test res
        end

        de = GeneralCanonicalDistanceDistance()
        for i ∈ 1:ncol(dist_t)
            dist = distance(de, ces[i], transpose(X); dims = 2)
            MN = size(dist)
            res = isapprox(dist, reshape(dist_t[!, i], MN))
            if !res
                println("Fails on General Canonical Distance Distance iteration $i")
            end
            @test res
        end
    end
    @testset "Covariance and Correlation correctness" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        fw = FrequencyWeights(rand(rng, 1000))
        ew = eweights(1:1000, 0.01; scale = true)
        ces = [FullCovariance(),
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
            end
            @test res1
            if !res2
                println("Fails on cor iteration $i")
            end
            @test res2
        end
    end
    @testset "cov2cor" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        fw = FrequencyWeights(rand(rng, 100))
        ew = eweights(1:100, 0.3; scale = true)

        ces = [FullCovariance(),
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
            result = isapprox(if isa(cv, Matrix)
                                  StatsBase.cov2cor(cv)
                              else
                                  StatsBase.cov2cor(Matrix(cv))
                              end, cr)
            if !result
                println("Test $i fails on:\n$(ce)\n")
            end
            @test result

            cvt = cov(ce, Matrix(transpose(X)); dims = 2)
            crt = cor(ce, Matrix(transpose(X)); dims = 2)
            resultt = isapprox(if isa(cvt, Matrix)
                                   StatsBase.cov2cor(cvt)
                               else
                                   StatsBase.cov2cor(Matrix(cvt))
                               end, crt)
            if !resultt
                println("Test `dims = 2` $i fails on:\n$(ce)\n")
            end
            @test resultt

            @test isapprox(cv, cvt)
            @test isapprox(cr, crt)
        end
    end
    @testset "Expected Returns" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        ew = eweights(1:100, 0.3; scale = true)
        mes = [SimpleExpectedReturns(), SimpleExpectedReturns(; w = ew)]
        results = [[-0.11099547379382146 -0.06958849697425303 0.1526187688612024 0.16461490675532198 -0.020956576537108633 -0.06233489179063739 -0.0367477367864812 0.007756747487672335 0.14605965903863052 0.14492727054039556],
                   [-0.26016452896926434 -0.2163901047814199 0.5441847730413261 0.1097939982985088 -0.4522897551948844 -0.0774204397703598 -0.34838218141402566 -0.4397900603694508 0.7072827148709439 0.048936555413477884]]
        for (i, (me, res)) ∈ enumerate(zip(mes, results))
            er = mean(me, X)
            result = isapprox(er, res)
            if !result
                println("Test $i fails on:\n$(me)\n$(res)\n")
            end
            @test result

            er = mean(me, transpose(X); dims = 2)
            result = isapprox(er, transpose(res))
            if !result
                println("Test `dims = 2` $i fails on:\n$(me)\n$(res)\n")
            end
            @test result
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
    end
    @testset "Misc tests" begin
        @test iszero(PortfolioOptimisers.intrinsic_mutual_info(rand(1, 1)))
    end
end
