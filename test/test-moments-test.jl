@safetestset "Moments" begin
    using PortfolioOptimisers, StatsBase, Random, StableRNGs, Test, CovarianceEstimation

    @testset "cov2cor" begin
        rng = StableRNG(123456789)
        X = randn(rng, 100, 10)
        fw = FrequencyWeights(rand(rng, 100))
        ew = eweights(1:100, 0.3; scale = true)

        ces = [FullCovariance(),
               FullCovariance(; ce = SimpleCovariance(; corrected = false), w = ew),
               FullCovariance(; ce = AnalyticalNonlinearShrinkage()),
               FullCovariance(; ce = AnalyticalNonlinearShrinkage(), w = fw),
               SemiCovariance(),
               SemiCovariance(; ce = SimpleCovariance(; corrected = false), w = ew),
               SemiCovariance(; ce = AnalyticalNonlinearShrinkage()),
               SemiCovariance(; ce = AnalyticalNonlinearShrinkage(), w = fw),
               SpearmanCovariance(), KendallCovariance(), MutualInfoCovariance(),
               MutualInfoCovariance(; bins = B_Knuth()),
               MutualInfoCovariance(; bins = B_FreedmanDiaconis()),
               MutualInfoCovariance(; bins = B_Scott()), MutualInfoCovariance(; bins = 5),
               MutualInfoCovariance(; ve = SimpleVariance(; corrected = false), w = ew),
               DistanceCovariance(), DistanceCovariance(; w = ew), LTDCovariance()]

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
    end
end
