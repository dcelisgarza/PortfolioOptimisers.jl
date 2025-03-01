@safetestset "DBHT" begin
    using PortfolioOptimisers, DataFrames, CSV, Random, StableRNGs, Test, StatsBase,
          Statistics

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

    @testset "LoGo" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ce = PortfolioOptimisersCovariance()
        sigma = cov(ce, X)

        logo_t = CSV.read(joinpath(@__DIR__, "./assets/LoGo-MaximumDistanceSimilarity.csv"),
                          DataFrame)

        des = [CanonicalDistance(), CanonicalDistanceDistance(), SimpleDistance(),
               SimpleDistanceDistance(), SimpleAbsoluteDistance(),
               SimpleAbsoluteDistanceDistance(), CorrelationDistance(),
               CorrelationDistanceDistance(), LogDistance(), LogDistanceDistance(),
               VariationInfoDistance(), VariationInfoDistanceDistance()]

        for i ∈ 1:ncol(logo_t)
            sigma1 = copy(sigma)
            LoGo!(PortfolioOptimisers.LoGo(; dist = des[i]),
                  FNPDM_NearestCorrelationMatrix(), sigma1, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo MaxDist similarity iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1
        end

        des = [GeneralCanonicalDistance(), GeneralCanonicalDistanceDistance(),
               GeneralDistance(), GeneralDistanceDistance(), GeneralAbsoluteDistance(),
               GeneralAbsoluteDistanceDistance(), GeneralCorrelationDistance(),
               GeneralCorrelationDistanceDistance(), GeneralLogDistance(),
               GeneralLogDistanceDistance(), GeneralVariationInfoDistance(),
               GeneralVariationInfoDistanceDistance()]

        for i ∈ 1:ncol(logo_t)
            sigma1 = copy(sigma)
            LoGo!(PortfolioOptimisers.LoGo(; dist = des[i]),
                  FNPDM_NearestCorrelationMatrix(), sigma1, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo General distance MaxDist similarity iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1
        end

        logo_t = CSV.read(joinpath(@__DIR__, "./assets/LoGo-ExponentialSimilarity.csv"),
                          DataFrame)

        des = [CanonicalDistance(), CanonicalDistanceDistance(), SimpleDistance(),
               SimpleDistanceDistance(), SimpleAbsoluteDistance(),
               SimpleAbsoluteDistanceDistance(), CorrelationDistance(),
               CorrelationDistanceDistance(), LogDistance(), LogDistanceDistance(),
               VariationInfoDistance(), VariationInfoDistanceDistance()]

        for i ∈ 1:ncol(logo_t)
            sigma1 = copy(sigma)
            LoGo!(PortfolioOptimisers.LoGo(; dist = des[i],
                                           similarity = DBHT_ExponentialSimilarity()),
                  FNPDM_NearestCorrelationMatrix(), sigma1, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo ExpDist similarity iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1
        end

        des = [GeneralCanonicalDistance(), GeneralCanonicalDistanceDistance(),
               GeneralDistance(), GeneralDistanceDistance(), GeneralAbsoluteDistance(),
               GeneralAbsoluteDistanceDistance(), GeneralCorrelationDistance(),
               GeneralCorrelationDistanceDistance(), GeneralLogDistance(),
               GeneralLogDistanceDistance(), GeneralVariationInfoDistance(),
               GeneralVariationInfoDistanceDistance()]

        for i ∈ 1:ncol(logo_t)
            sigma1 = copy(sigma)
            LoGo!(PortfolioOptimisers.LoGo(; dist = des[i],
                                           similarity = DBHT_ExponentialSimilarity()),
                  FNPDM_NearestCorrelationMatrix(), sigma1, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo General distance ExpDist similarity iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1
        end
    end
end
