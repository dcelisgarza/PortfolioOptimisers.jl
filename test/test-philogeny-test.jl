@safetestset "Philogeny tests" begin
    using PortfolioOptimisers, Random, Test, Clustering, CSV, DataFrames, TimeSeries
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
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                  timestamp = :timestamp)
    F = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/factor_prices.csv"));
                  timestamp = :timestamp)
    rd = prices_to_returns(X[(end - 252):end], F[(end - 252):end])
    pr = prior(FactorPriorEstimator(; re = DimensionReductionRegression()), rd)
    @testset "Clustering tests" begin
        X = pr.X
        clr = clusterise(ClusteringEstimator(; ce = PortfolioOptimisersCovariance(),
                                             de = Distance(; alg = CanonicalDistance()),
                                             alg = HierarchicalClustering(),
                                             onc = OptimalNumberClusters(;
                                                                         alg = SecondOrderDifference())),
                         X)
        clr2 = clusterise(clr)
        @test clr === clr2
        clr_t = Hclust{Float64}([-29 -3; -16 1; -10 2; 3 -25; -1 4; -28 -24; -23 -14;
                                 -17 -19;
                                 -12 8; -6 9; -7 6; -15 -30; -13 12; 10 -2; 13 -4; -11 -18;
                                 -8 -5;
                                 7 -20; 18 -21; 11 19; -22 -27; -9 20; 17 15; 22 14; 24 5;
                                 21 -26;
                                 16 26; 23 27; 25 28],
                                [0.06660338941801756, 0.09478780041235547,
                                 0.14473759310960513, 0.16015164443931967,
                                 0.19147964929228423, 0.10117104385862949,
                                 0.08307922792534263, 0.0743216109518963,
                                 0.08708634320137355, 0.13522426691008954,
                                 0.19484271202566422, 0.06955846777061862,
                                 0.14946561269230538, 0.24262023590234522,
                                 0.16085255124008674, 0.18153418508925473,
                                 0.22729994563435377, 0.15286122114702627,
                                 0.2532236290809116, 0.27125768172820036,
                                 0.24925670306242456, 0.38581221294513307,
                                 0.28487943672741695, 0.45168418395548565,
                                 0.694417617023943, 0.3125508783481835, 0.4357121874915939,
                                 0.5636426580531525, 1.6414756111174895],
                                [9, 7, 28, 24, 23, 14, 20, 21, 6, 12, 17, 19, 2, 1, 10, 16,
                                 29, 3, 25, 8, 5, 13, 15, 30, 4, 11, 18, 22, 27, 26], :ward)
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
        clr_t = Hclust{Float64}([-29 -1; -16 -3; -10 2; -25 3; 4 1; -17 -6; -20 -23; 7 -9;
                                 -19 -12;
                                 -2 9; -28 -24; -14 11; 12 -7; 10 6; -21 13; 15 8; 14 16;
                                 -13 -8;
                                 -30 -15; 19 18; -11 -18; -27 -22; -26 22; -4 -5; 24 21;
                                 23 25;
                                 20 5; 27 17; 26 28],
                                [0.2, 0.25, 0.3333333333333333, 0.5, 1.0,
                                 0.08333333333333333, 0.09090909090909091, 0.1,
                                 0.1111111111111111, 0.125, 0.14285714285714285,
                                 0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.5,
                                 1.0, 0.3333333333333333, 0.5, 1.0, 0.16666666666666666,
                                 0.2, 0.25, 0.3333333333333333, 0.5, 1.0, 2.0, 3.0, 4.0],
                                [26, 27, 22, 4, 5, 11, 18, 30, 15, 13, 8, 25, 10, 16, 3, 29,
                                 1, 2, 19, 12, 17, 6, 21, 14, 28, 24, 7, 20, 23, 9], :DBHT)
        @test clr.clustering.merges == clr_t.merges
        @test isapprox(clr.clustering.heights, clr_t.heights)
        @test clr.clustering.labels == clr_t.labels
        @test clr.clustering.linkage == clr_t.linkage
        @test 3 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = 1),
                                                               clr.clustering, clr.D)
        @test 3 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = 100),
                                                               clr.clustering, clr.D)

        @test 3 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = 1),
                                                               clr.clustering, clr.D)
        @test 3 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = 100),
                                                               clr.clustering, clr.D)

        @test 3 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = nothing),
                                                               clr.clustering, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = 1),
                                                               clr.clustering, clr.D)
        @test 3 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = StandardisedSilhouetteScore(),
                                                                                     max_k = 100),
                                                               clr.clustering, clr.D)
        @test 4 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
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
        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = PredefinedNumberClusters(;
                                                                                                                    k = 3),
                                                                                     max_k = 2),
                                                               clr.clustering, clr.D)

        clr = clusterise(ClusteringEstimator(; ce = PortfolioOptimisersCovariance(),
                                             de = DistanceDistance(;
                                                                   alg = CanonicalDistance()),
                                             alg = HierarchicalClustering(),
                                             onc = OptimalNumberClusters(;
                                                                         alg = StandardisedSilhouetteScore())),
                         X)
        @test clr.k == 2

        clr = clusterise(ClusteringEstimator(; ce = PortfolioOptimisersCovariance(),
                                             de = DistanceDistance(;
                                                                   alg = CanonicalDistance()),
                                             alg = DBHT(; sim = MaximumDistanceSimilarity(),
                                                        root = UniqueRoot()),
                                             onc = OptimalNumberClusters(;
                                                                         alg = PredefinedNumberClusters(;
                                                                                                        k = 5))),
                         X)
        @test clr.k == 5
    end
    @testset "Centrality tests" begin
        X = pr.X
        ces = [BetweennessCentrality(), ClosenessCentrality(), DegreeCentrality(),
               EigenvectorCentrality(), KatzCentrality(), Pagerank(), RadialityCentrality(),
               StressCentrality()]
        df1 = CSV.read(joinpath(@__DIR__, "./assets/Centrality.csv"), DataFrame)
        df2 = CSV.read(joinpath(@__DIR__, "./assets/Average_Centrality.csv"), DataFrame)
        w = fill(inv(30), 30)
        for (i, ce) in enumerate(ces)
            v1 = centrality_vector(CentralityEstimator(; cent = ce), X)
            res = isapprox(v1, df1[!, i])
            if !res
                println("Iteration $i failed on default centrality algorithm.")
                find_tol(v1, df1[!, i]; name1 = :v1, name2 = :df1)
            end
            @test res

            c = average_centrality(CentralityEstimator(; cent = ce), w, X)
            res = isapprox(c, df2[i, 1])
            if !res
                println("Iteration $i failed on default average centrality algorithm.")
                find_tol(c, df2[i, 1]; name1 = :c, name2 = :df2)
            end
            @test res
        end
    end
    @testset "Philogeny matrix" begin
        X = pr.X
        df = CSV.read(joinpath(@__DIR__, "./assets/Philogeny_Matrix_1.csv"), DataFrame)
        for i in 1:10
            A = philogeny_matrix(NetworkEstimator(; n = i), X)
            res = isapprox(vec(A), df[!, i])
            if !res
                println("Iteration $i failed on detault network estimator.")
                find_tol(vec(A), df[!, i]; name1 = :A, name2 = :df)
            end
            @test res
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/Philogeny_Matrix_2.csv"), DataFrame)
        for i in 1:3
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

        w = fill(inv(30), 30)
        @test isapprox(asset_philogeny(NetworkEstimator(), w, X), 0.06444444444444444)
        @test isapprox(asset_philogeny(ClusteringEstimator(), w, X), 0.32888888888888984)

        A1 = PortfolioOptimisers.calc_adjacency(NetworkEstimator(; alg = KruskalTree()), X)
        A2 = PortfolioOptimisers.calc_adjacency(NetworkEstimator(; alg = BoruvkaTree()), X)
        A3 = PortfolioOptimisers.calc_adjacency(NetworkEstimator(; alg = PrimTree()), X)

        @test A1 == A2
        @test A1 == A3
    end
end
