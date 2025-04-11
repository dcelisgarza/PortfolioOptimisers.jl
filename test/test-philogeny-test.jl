@safetestset "Philogeny tests" begin
    using PortfolioOptimisers, StableRNGs, Random, Test, Clustering, CSV, DataFrames
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
    @testset "Clustering tests" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        clr = clusterise(ClusteringEstimator(; ce = PortfolioOptimisersCovariance(),
                                             de = Distance(; alg = CanonicalDistance()),
                                             alg = HierarchicalClustering(),
                                             onc = OptimalNumberClusters(;
                                                                         alg = SecondOrderDifference())),
                         X)
        clr2 = clusterise(clr)
        @test clr === clr2
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
                                             de = Distance(; alg = CanonicalDistance()),
                                             alg = DBHT(),
                                             onc = OptimalNumberClusters(;
                                                                         alg = SecondOrderDifference())),
                         X)
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
        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = 1),
                                                               clr.clustering, clr.D)
        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = 100),
                                                               clr.clustering, clr.D)

        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = 1),
                                                               clr.clustering, clr.D)
        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = 100),
                                                               clr.clustering, clr.D)

        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = 1),
                                                               clr.clustering, clr.D)
        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = 100),
                                                               clr.clustering, clr.D)
        @test 5 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = PredefinedNumberClusters(;
                                                                                                                    k = 10),
                                                                                     max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = PredefinedNumberClusters(;
                                                                                                                    k = 1),
                                                                                     max_k = 5),
                                                               clr.clustering, clr.D)
        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = PredefinedNumberClusters(;
                                                                                                                    k = 2),
                                                                                     max_k = 5),
                                                               clr.clustering, clr.D)
        @test 3 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = PredefinedNumberClusters(;
                                                                                                                    k = 3),
                                                                                     max_k = 5),
                                                               clr.clustering, clr.D)
        @test 4 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = PredefinedNumberClusters(;
                                                                                                                    k = 4),
                                                                                     max_k = 5),
                                                               clr.clustering, clr.D)
        @test 5 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = PredefinedNumberClusters(;
                                                                                                                    k = 5),
                                                                                     max_k = 5),
                                                               clr.clustering, clr.D)
        @test 5 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = PredefinedNumberClusters(;
                                                                                                                    k = 6),
                                                                                     max_k = 10),
                                                               clr.clustering, clr.D)

        rng = StableRNG(1)
        X = randn(rng, 10, 1)
        clr = clusterise(ClusteringEstimator(; ce = PortfolioOptimisersCovariance(),
                                             de = DistanceDistance(;
                                                                   alg = CanonicalDistance()),
                                             alg = HierarchicalClustering(),
                                             onc = OptimalNumberClusters(;
                                                                         #  alg = PredefinedNumberClusters(;
                                                                         #                                 k = 9)

                                                                         alg = StandardisedSilhouetteScore())),
                         X)
        @test clr.k == 1

        rng = StableRNG(3)
        X = randn(rng, 1000, 20)
        clr = clusterise(ClusteringEstimator(; ce = PortfolioOptimisersCovariance(),
                                             de = DistanceDistance(;
                                                                   alg = CanonicalDistance()),
                                             alg = DBHT(; sim = MaximumDistanceSimilarity(),
                                                        root = UniqueRoot()),
                                             onc = OptimalNumberClusters(;
                                                                         alg = PredefinedNumberClusters(;
                                                                                                        k = 5))),
                         X)
        @test clr.k == 4
    end
    @testset "Centrality tests" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ces = [BetweennessCentrality(), ClosenessCentrality(), DegreeCentrality(),
               EigenvectorCentrality(), KatzCentrality(), Pagerank(), RadialityCentrality(),
               StressCentrality()]
        df1 = CSV.read(joinpath(@__DIR__, "./assets/Centrality.csv"), DataFrame)
        df2 = CSV.read(joinpath(@__DIR__, "./assets/Average_Centrality.csv"), DataFrame)
        w = fill(inv(20), 20)
        for i ∈ 1:ncol(df1)
            v1 = centrality_vector(CentralityEstimator(; cent = ces[i]), X)
            res = isapprox(v1, df1[!, i])
            if !res
                println("Iteration $i failed on default centrality algorithm.")
                find_tol(v1, df1[!, i]; name1 = :v1, name2 = :df1)
            end
            @test res

            c = average_centrality(CentralityEstimator(; cent = ces[i]), w, X)
            res = isapprox(c, df2[i, 1])
            if !res
                println("Iteration $i failed on default average centrality algorithm.")
                find_tol(c, df2[i, 1]; name1 = :c, name2 = :df2)
            end
            @test res
        end
    end
    @testset "Philogeny matrix" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)

        df = CSV.read(joinpath(@__DIR__, "./assets/Philogeny_Matrix_1.csv"), DataFrame)
        for i ∈ 1:ncol(df)
            A = philogeny_matrix(NetworkEstimator(; n = i), X)
            res = isapprox(vec(A), df[!, i])
            if !res
                println("Iteration $i failed on detault network estimator.")
                find_tol(vec(A), df[!, i]; name1 = :A, name2 = :df)
            end
            @test res
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/Philogeny_Matrix_2.csv"), DataFrame)
        for i ∈ 1:ncol(df)
            A = philogeny_matrix(NetworkEstimator(; n = i,
                                                  alg = MaximumDistanceSimilarity()), X)
            res = isapprox(vec(A), df[!, i])
            if !res
                println("Iteration $i failed on MaximumDistanceSimilarity.")
                find_tol(vec(A), df[!, i]; name1 = :A, name2 = :df)
            end
            @test res
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/Philogeny_Matrix_3.csv"), DataFrame)
        A = philogeny_matrix(ClusteringEstimator(), X)
        @test isapprox(vec(A), df[!, 1])

        w = fill(inv(20), 20)
        @test isapprox(asset_philogeny(NetworkEstimator(), w, X), 0.09500000000000004)
        @test isapprox(asset_philogeny(ClusteringEstimator(), w, X), 0.45000000000000023)
    end
end
