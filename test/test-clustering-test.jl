@safetestset "Clustering tests" begin
    using PortfolioOptimisers, StableRNGs, Random, Test, Clustering
    @testset "Clustering tests" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        clr = clusterise(ClusteringEstimator(; ce = PortfolioOptimisersCovariance(),
                                             de = CanonicalDistance(),
                                             alg = HierarchicalClustering(),
                                             nch = SecondOrderDifference()), X)
        clr_t = Clustering.Hclust{Float64}([-8 -18; -3 -2; -5 -15; -12 -10; -11 -4; 2 -19;
                                            -17 -14;
                                             -9 -6; 7 4; -16 -1; -20 -13; -7 8; 10 6; 5 3;
                                            13 1;
                                             14 11; 9 16; 15 12; 18 17],
                                           [0.6823375860427897, 0.6890306478220115,
                                            0.6746034164602529, 0.6898380575401942,
                                            0.680816778226377, 0.7013996347000043,
                                            0.6920471854075161, 0.695900421949497,
                                            0.7009530475965285, 0.7061838595244663,
                                            0.6962015583975546, 0.7045566215837087,
                                            0.7212316655970359, 0.7134398333617952,
                                            0.7255262914312883, 0.7250680778417793,
                                            0.7382584957073629, 0.7390041302551158,
                                            0.7560476856461593],
                                           [16, 1, 3, 2, 19, 8, 18, 7, 9, 6, 17, 14, 12, 10,
                                            11, 4, 5, 15, 20, 13], :ward)
        @test clr.clustering.merges == clr_t.merges
        @test isapprox(clr.clustering.heights, clr_t.heights)
        @test clr.clustering.labels == clr_t.labels
        @test clr.clustering.linkage == clr_t.linkage
        clr = clusterise(ClusteringEstimator(; ce = PortfolioOptimisersCovariance(),
                                             de = CanonicalDistance(), alg = DBHT(),
                                             nch = SecondOrderDifference()), X)
        clr_t = Hclust{Float64}([-5 -13; -10 -12; -9 -6; 3 -17; -8 -18; -14 -7; -3 -1;
                                 -11 -4;
                                 -20 -2; 9 -19; -16 -15; 8 5; 2 6; 10 12; 11 1; 14 7; 15 4;
                                 17 13;
                                 16 18],
                                [0.05263157894736842, 0.05555555555555555,
                                 0.058823529411764705, 0.0625, 0.06666666666666667,
                                 0.07142857142857142, 0.07692307692307693,
                                 0.08333333333333333, 0.09090909090909091, 0.1,
                                 0.1111111111111111, 0.125, 0.14285714285714285,
                                 0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.5,
                                 1.0],
                                [20, 2, 19, 11, 4, 8, 18, 3, 1, 16, 15, 5, 13, 9, 6, 17, 10,
                                 12, 14, 7], :DBHT)
        @test clr.clustering.merges == clr_t.merges
        @test isapprox(clr.clustering.heights, clr_t.heights)
        @test clr.clustering.labels == clr_t.labels
        @test clr.clustering.linkage == clr_t.linkage
        @test 2 == PortfolioOptimisers.optimal_number_clusters(SecondOrderDifference(;
                                                                                     max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(SecondOrderDifference(;
                                                                                     max_k = 1),
                                                               clr.clustering, clr.D)
        @test 2 == PortfolioOptimisers.optimal_number_clusters(SecondOrderDifference(;
                                                                                     max_k = 100),
                                                               clr.clustering, clr.D)

        @test 2 == PortfolioOptimisers.optimal_number_clusters(StandardisedSilhouetteScore(;
                                                                                           max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(StandardisedSilhouetteScore(;
                                                                                           max_k = 1),
                                                               clr.clustering, clr.D)
        @test 2 == PortfolioOptimisers.optimal_number_clusters(StandardisedSilhouetteScore(;
                                                                                           max_k = 100),
                                                               clr.clustering, clr.D)

        @test 2 == PortfolioOptimisers.optimal_number_clusters(StandardisedSilhouetteScore(;
                                                                                           max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(StandardisedSilhouetteScore(;
                                                                                           max_k = 1),
                                                               clr.clustering, clr.D)
        @test 2 == PortfolioOptimisers.optimal_number_clusters(StandardisedSilhouetteScore(;
                                                                                           max_k = 100),
                                                               clr.clustering, clr.D)
        @test 5 ==
              PortfolioOptimisers.optimal_number_clusters(PredefinedNumberClusters(; k = 10,
                                                                                   max_k = nothing),
                                                          clr.clustering, clr.D)
        @test 1 ==
              PortfolioOptimisers.optimal_number_clusters(PredefinedNumberClusters(; k = 1,
                                                                                   max_k = 5),
                                                          clr.clustering, clr.D)
        @test 2 ==
              PortfolioOptimisers.optimal_number_clusters(PredefinedNumberClusters(; k = 2,
                                                                                   max_k = 5),
                                                          clr.clustering, clr.D)
        @test 3 ==
              PortfolioOptimisers.optimal_number_clusters(PredefinedNumberClusters(; k = 3,
                                                                                   max_k = 5),
                                                          clr.clustering, clr.D)
        @test 4 ==
              PortfolioOptimisers.optimal_number_clusters(PredefinedNumberClusters(; k = 4,
                                                                                   max_k = 5),
                                                          clr.clustering, clr.D)
        @test 5 ==
              PortfolioOptimisers.optimal_number_clusters(PredefinedNumberClusters(; k = 5,
                                                                                   max_k = 5),
                                                          clr.clustering, clr.D)
        @test 5 ==
              PortfolioOptimisers.optimal_number_clusters(PredefinedNumberClusters(; k = 6,
                                                                                   max_k = 10),
                                                          clr.clustering, clr.D)
    end
end
