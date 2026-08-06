@testset "Phylogeny tests" begin
    using PortfolioOptimisers, Test, Clustering, CSV, DataFrames, TimeSeries, StableRNGs,
          StatsBase, SparseArrays, LinearAlgebra, Clarabel
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(EmpiricalPrior(), rd)
    wt = pweights(fill(inv(size(pr.X, 1)), size(pr.X, 1)))
    @testset "Similarity matrix tests" begin
        _, D = cor_and_dist(Distance(), PortfolioOptimisersCovariance(), pr.X)
        Sc = PortfolioOptimisers.distance_to_similarity(ComplementSimilarity(); D = D)
        Sa = PortfolioOptimisers.distance_to_similarity(AngularSimilarity(); D = D)
        # Both are pure elementwise transforms of D.
        @test isapprox(Sc, one(eltype(D)) .- D)
        @test isapprox(Sa, cos.(pi .* D))
        # Symmetry and the unit diagonal survive both, since `pairwise` zeroes the diagonal.
        for S in (Sc, Sa)
            @test issymmetric(S)
            @test isapprox(diag(S), ones(size(S, 1)))
        end
        # `AngularSimilarity` is bounded; `ComplementSimilarity` is only bounded when D is.
        @test all(-one(eltype(Sa)) .<= Sa .<= one(eltype(Sa)))
        @test isapprox(PortfolioOptimisers.distance_to_similarity(ComplementSimilarity();
                                                                  D = [0.0 7.0; 7.0 0.0]),
                       [1.0 -6.0; -6.0 1.0])
        # `AngularSimilarity` inverts a normalised angular distance exactly.
        rho = cor(pr.X)
        Da = acos.(clamp.(rho, -1, 1)) ./ pi
        @test isapprox(PortfolioOptimisers.distance_to_similarity(AngularSimilarity();
                                                                  D = Da), rho)
        # The zero-row convention: D = 1 against a non-zero row, D = 0 between two zero rows.
        @test isapprox(PortfolioOptimisers.distance_to_similarity(AngularSimilarity();
                                                                  D = [0.0 1.0 1.0
                                                                       1.0 0.0 0.0
                                                                       1.0 0.0 0.0]),
                       [1.0 -1.0 -1.0; -1.0 1.0 1.0; -1.0 1.0 1.0])
        # `default_similarity` falls back to the linear complement for every metric.
        for metric in (PortfolioOptimisers.Distances.CosineDist(),
                       PortfolioOptimisers.Distances.Jaccard(),
                       PortfolioOptimisers.Distances.BrayCurtis(),
                       PortfolioOptimisers.Distances.CorrDist(),
                       PortfolioOptimisers.Distances.Euclidean())
            @test PortfolioOptimisers.default_similarity(metric) === ComplementSimilarity()
        end
        # Both new members are on the exported API and in the family.
        @test ComplementSimilarity() isa
              PortfolioOptimisers.AbstractSimilarityMatrixAlgorithm
        @test AngularSimilarity() isa PortfolioOptimisers.AbstractSimilarityMatrixAlgorithm
    end
    @testset "Feature distance tests" begin
        DS = PortfolioOptimisers.Distances
        rng = StableRNG(987654321)
        # The elementwise method is `AngularDist`'s contract; `_pairwise!` is an internal
        # hook overloaded to delegate to Distances' gemm `CosineDist` kernel. Nothing in
        # Distances pins the two together, so this does.
        function elementwise_pairwise(metric, Z; dims = 1)
            n = dims == 1 ? size(Z, 1) : size(Z, 2)
            obs(i) = dims == 1 ? view(Z, i, :) : view(Z, :, i)
            D = zeros(Float64, n, n)
            for i in 1:n, j in 1:n
                D[i, j] = i == j ? 0.0 : metric(obs(i), obs(j))
            end
            return D
        end
        @testset "AngularDist gemm path matches the elementwise method" begin
            for N in (2, 5, 50, 200), K in (1, 3, 30)
                Z = randn(rng, N, K)
                @test isapprox(DS.pairwise(AngularDist(), Z; dims = 1),
                               elementwise_pairwise(AngularDist(), Z); atol = 1e-8)
            end
            # Zero rows are where the gemm kernel and the elementwise method could most
            # easily disagree: the gemm divides by the norm and produces NaN.
            Zz = randn(rng, 8, 4)
            Zz[2, :] .= 0
            Zz[5, :] .= 0
            @test isapprox(DS.pairwise(AngularDist(), Zz; dims = 1),
                           elementwise_pairwise(AngularDist(), Zz); atol = 1e-8)
            @test isapprox(DS.pairwise(AngularDist(), permutedims(Zz); dims = 2),
                           elementwise_pairwise(AngularDist(), permutedims(Zz); dims = 2);
                           atol = 1e-8)
            # The convention itself: 0 between two zero rows, 1 against a non-zero one.
            Da = DS.pairwise(AngularDist(), Zz; dims = 1)
            @test all(iszero, Da[[2, 5], [2, 5]])
            @test all(isone, Da[[2, 5], [1, 3, 4, 6, 7, 8]])
            # A true metric, unlike `CosineDist`.
            @test AngularDist() isa DS.Metric
            @test DS.result_type(AngularDist(), Int, Int) === Float64
        end
        @testset "2-D feature matrices" begin
            Z = [1.0 0.0; 0.0 1.0; 1.0 1.0]
            de = FeatureDistance()
            @test de.metric === AngularDist()
            @test de.alg === LastObservation()
            # `sim` is defaulted from the metric, and `AngularDist` gets the exact inverse.
            @test de.sim === AngularSimilarity()
            @test FeatureDistance(; metric = DS.CosineDist()).sim === ComplementSimilarity()
            D = distance(de, Z)
            @test issymmetric(D)
            @test all(iszero, diag(D))
            @test isapprox(D, [0.0 0.5 0.25; 0.5 0.0 0.25; 0.25 0.25 0.0])
            # `cor_and_dist`'s first slot is the similarity, derived from this very `D`.
            S, D2 = cor_and_dist(de, Z)
            @test D2 == D
            @test isapprox(S, cos.(pi .* D))
            # `AngularSimilarity` recovers the cosine similarity exactly, with no `Z`.
            Zc = randn(rng, 40, 7)
            Sc, _ = cor_and_dist(de, Zc)
            @test isapprox(Sc,
                           one(eltype(Zc)) .- DS.pairwise(DS.CosineDist(), Zc; dims = 1))
            # Every `SemiMetric` yields a similarity; nothing throws.
            Se, De = cor_and_dist(FeatureDistance(; metric = DS.Euclidean()), Zc)
            @test isapprox(Se, one(eltype(De)) .- De)
            # `dims` retargets to `Z`, so a transposed matrix gives the same answer.
            @test isapprox(distance(de, permutedims(Zc); dims = 2), distance(de, Zc))
            # Producer #4's square adjacency is integer-valued and needs no promotion.
            Zi = [1 0 1; 0 1 1; 1 1 0]
            @test eltype(distance(de, Zi)) === Float64
            # Asset views are `SubArray`s and need no `collect`.
            @test isapprox(distance(de, view(Zi, 1:2, :)), distance(de, Zi[1:2, :]))
        end
        @testset "Degenerate 2-D inputs" begin
            de = FeatureDistance()
            # A single feature: every asset is on one ray, so the angular distance is 0.
            # `CorrDist` is NaN against any constant row, hence unusable here (#162).
            Z1 = reshape([1.0, 2.0, 3.0], 3, 1)
            @test all(iszero, distance(de, Z1))
            Dc = distance(FeatureDistance(; metric = DS.CorrDist()), Z1)
            @test all(isnan, Dc[(1:3) .!= (1:3)'])
            # Zero rows: valid input the metric cannot measure. The patch rewrites only the
            # entries the metric left as NaN.
            Zz = [1.0 0.0; 0.0 0.0; 0.0 0.0; 1.0 1.0]
            for metric in (AngularDist(), DS.CosineDist(), DS.Jaccard(), DS.BrayCurtis())
                D = distance(FeatureDistance(; metric = metric), Zz)
                @test !any(isnan, D)
                @test iszero(D[2, 3])
                @test isone(D[2, 1])
                @test isone(D[3, 4])
            end
            # `Euclidean` places a zero row at the origin — a real distance, not a NaN — so
            # the patch must leave it alone.
            De = distance(FeatureDistance(; metric = DS.Euclidean()), Zz)
            @test isapprox(De[2, 1], 1.0)
            @test isapprox(De[2, 4], sqrt(2))
            @test iszero(De[2, 3])
        end
        @testset "Collapse algorithms" begin
            Z3 = abs.(randn(rng, 6, 5, 4))
            algs = (LastObservation(), AggregateFeatures(), AggregateDistances(),
                    StackObservations(), AggregateFeatures(; alg = MedianCollapse()))
            Ds = [distance(FeatureDistance(; alg = alg), Z3) for alg in algs]
            for D in Ds
                @test size(D) == (5, 5)
                @test issymmetric(D)
                @test all(iszero, diag(D))
                @test !any(isnan, D)
            end
            # The four rules genuinely differ; `StackObservations` is not a rename of one
            # of the other three.
            for i in 1:4, j in (i + 1):4
                @test !isapprox(Ds[i], Ds[j])
            end
            # `LastObservation` discards the window.
            @test Ds[1] == distance(FeatureDistance(), Z3[6, :, :])
            # `dims = 2` swaps the two trailing axes, for every rule.
            Z3p = permutedims(Z3, (1, 3, 2))
            for (alg, D) in zip(algs, Ds)
                @test isapprox(distance(FeatureDistance(; alg = alg), Z3p; dims = 2), D)
            end
            # At T == 1 there is nothing to collapse, so all four agree exactly.
            Z1 = Z3[6:6, :, :]
            D1 = [distance(FeatureDistance(; alg = alg), Z1) for alg in algs]
            @test all(D -> D == D1[1], D1)
            @test D1[1] == Ds[1]
            # `AggregateDistances` applies the zero-feature convention per observation, so
            # an asset that is zero at *some* observations differs from the stacked form.
            Zs = zeros(2, 3, 2)
            Zs[1, :, :] = [1.0 0.0; 0.0 0.0; 0.0 1.0]
            Zs[2, :, :] = [1.0 1.0; 1.0 0.0; 1.0 0.0]
            @test !isapprox(distance(FeatureDistance(; alg = StackObservations()), Zs),
                            distance(FeatureDistance(; alg = AggregateDistances()), Zs))
            # A median of distance matrices is not a metric, so it is rejected outright.
            @test_throws ArgumentError AggregateDistances(; alg = MedianCollapse())
            @test AggregateFeatures(; alg = MedianCollapse()).alg === MedianCollapse()
            # The 2-D method never consults `alg`.
            Z2 = Z3[6, :, :]
            for alg in algs
                @test distance(FeatureDistance(; alg = alg), Z2) ==
                      distance(FeatureDistance(), Z2)
            end
        end
        @testset "Observation weights" begin
            Z3 = abs.(randn(rng, 6, 5, 4))
            w = pweights([1.0, 1, 1, 1, 1, 5])
            pairs = [(AggregateFeatures(), AggregateFeatures(; w = w)),
                     (AggregateDistances(), AggregateDistances(; w = w)),
                     (AggregateFeatures(; alg = MedianCollapse()),
                      AggregateFeatures(; w = w, alg = MedianCollapse()))]
            for (alg, walg) in pairs
                unweighted = distance(FeatureDistance(; alg = alg), Z3)
                weighted = distance(FeatureDistance(; alg = walg), Z3)
                @test !isapprox(unweighted, weighted)
                @test issymmetric(weighted)
                @test all(iszero, diag(weighted))
            end
            # `@wprop` on the collapse algorithm plus `@fprop` on `FeatureDistance.alg`
            # closes the chain, so `factory` installs threaded weights.
            fd = factory(FeatureDistance(; alg = AggregateFeatures()), w)
            @test fd.alg.w === w
            @test fd.metric === AngularDist()
            @test fd.sim === AngularSimilarity()
            # The rules with no observation axis to weight ignore threaded weights.
            @test factory(FeatureDistance(), w).alg === LastObservation()
            # `FeatureDistance` must be `@propagatable`, or `ClustersEstimator`'s `@fprop de`
            # has nothing to recurse into.
            @test hasmethod(factory, Tuple{FeatureDistance, Vararg{Any}})
        end
        @testset "Feature matrix validation" begin
            de = FeatureDistance()
            Z = [1.0 2.0; 3.0 4.0]
            @test_throws DomainError distance(de, Z; dims = 3)
            @test_throws PortfolioOptimisers.IsEmptyError distance(de,
                                                                   Matrix{Float64}(undef, 0,
                                                                                   0))
            @test_throws PortfolioOptimisers.IsNonFiniteError distance(de,
                                                                       [1.0 NaN; 3.0 4.0])
            @test_throws PortfolioOptimisers.IsNonFiniteError distance(de,
                                                                       [1.0 Inf; 3.0 4.0])
            # `Jaccard` is the Ruzicka form and returns values up to 2 on signed input,
            # silently. The non-negativity check is mandatory for it, and for the two other
            # non-negative-domain metrics, but must not fire for the default.
            for metric in (DS.Jaccard(), DS.BrayCurtis(), DS.ChiSqDist())
                @test_throws DomainError distance(FeatureDistance(; metric = metric),
                                                  [1.0 -2.0; 3.0 4.0])
            end
            @test !any(isnan, distance(de, [1.0 -2.0; 3.0 4.0]))
            # Structural degeneracy the metric can handle is admitted, not rejected.
            @test size(distance(de, [1.0 1.0; 1.0 1.0; 0.0 0.0])) == (3, 3)
            # The 3-D entry point validates the same way.
            @test_throws DomainError distance(de, abs.(randn(rng, 2, 3, 2)); dims = 3)
        end
        @testset "FeatureDistance is a distance estimator" begin
            # It is a peer of `Distance`/`DistanceDistance`, not one of their algorithms:
            # every consumer types its `de` field as `AbstractDistanceEstimator`.
            @test FeatureDistance() isa PortfolioOptimisers.AbstractDistanceEstimator
            @test LastObservation() isa PortfolioOptimisers.AbstractFeatureCollapseAlgorithm
            @test MeanCollapse() isa PortfolioOptimisers.AbstractCollapseAlgorithm
            @test AngularDist() isa DS.SemiMetric
        end
    end
    @testset "Clustering tests" begin
        clr = clusterise(ClustersEstimator(; ce = PortfolioOptimisersCovariance(),
                                           de = Distance(; alg = CanonicalDistance()),
                                           alg = HClustAlgorithm(),
                                           onc = OptimalNumberClusters(;
                                                                       alg = SecondOrderDifference())),
                         pr.X)
        @test factory(clr) === clr
        clr2 = clusterise(clr)
        @test clr === clr2
        alg = HClustAlgorithm()
        @test factory(alg) === alg
        clr_t = Hclust{Float64}([-1 -13;
                                 -2 1;
                                 -7 -4;
                                 -3 -9;
                                 4 -6;
                                 -14 -10;
                                 6 -16;
                                 2 3;
                                 -8 -12;
                                 -11 9;
                                 10 -15;
                                 -18 11;
                                 -19 7;
                                 -5 -20;
                                 14 -17;
                                 5 8;
                                 13 12;
                                 15 16;
                                 18 17],
                                [0.2992796916890263, 0.3905612031361099, 0.4116454947407609,
                                 0.22529867864314176, 0.48924055164900887,
                                 0.28518628966607656, 0.38008631653392483, 0.58425801808901,
                                 0.4209146698909511, 0.4605479608276049, 0.491265779899127,
                                 0.5080236207888611, 0.6028393326616824,
                                 0.24670816017739164, 0.5174631833197462,
                                 0.6821605083740293, 0.739945604975504, 0.9602133336085281,
                                 1.0707389175241802],
                                [5, 20, 17, 3, 9, 6, 2, 1, 13, 7, 4, 19, 14, 10, 16, 18, 11,
                                 8, 12, 15], :ward)
        @test clr.res.merges == clr_t.merges
        @test isapprox(clr.res.heights, clr_t.heights)
        @test clr.res.labels == clr_t.labels
        @test clr.res.linkage == clr_t.linkage
        clr = clusterise(ClustersEstimator(; ce = PortfolioOptimisersCovariance(),
                                           de = Distance(; alg = CanonicalDistance()),
                                           alg = DBHT(),
                                           onc = OptimalNumberClusters(;
                                                                       alg = SecondOrderDifference())),
                         pr.X)
        S, D = cor_and_dist(Distance(), PortfolioOptimisersCovariance(), pr.X)
        @test isapprox(PortfolioOptimisers.distance_to_similarity(ExponentialSimilarity();
                                                                  D = D),
                       PortfolioOptimisers.distance_to_similarity(GeneralExponentialSimilarity();
                                                                  D = D))
        A1, tri1, separators1, cliques1, cliqueTree1 = PortfolioOptimisers.PMFG_T2s(S, 5)
        @test isapprox(A1,
                       sparse([2, 3, 4, 6, 7, 9, 10, 13, 14, 16, 1, 3, 4, 5, 6, 7, 10, 13,
                               1, 2, 5, 6, 9, 17, 20, 1, 2, 6, 7, 2, 3, 6, 17, 20, 1, 2, 3,
                               4, 5, 9, 20, 1, 2, 4, 13, 10, 11, 12, 14, 15, 16, 18, 1, 3,
                               6, 1, 2, 8, 13, 14, 16, 18, 19, 8, 12, 14, 15, 18, 8, 11, 15,
                               18, 1, 2, 7, 10, 14, 1, 8, 10, 11, 13, 16, 18, 19, 8, 11, 12,
                               1, 8, 10, 14, 3, 5, 20, 8, 10, 11, 12, 14, 19, 10, 14, 18, 3,
                               5, 6, 17],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                               3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
                               6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10,
                               10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13,
                               13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 16,
                               16, 16, 16, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19,
                               20, 20, 20, 20],
                              [0.7309937996487316, 0.5668675596240081, 0.5355007421820464,
                               0.5523884099446082, 0.5953909053658156, 0.5504673846415659,
                               0.5238826236796035, 0.8208633322850427, 0.5498675558344721,
                               0.4626642723017695, 0.7309937996487316, 0.5587684928683124,
                               0.5175969259573149, 0.30828602794747867, 0.5842918137900647,
                               0.5205377943383558, 0.35483032135142517, 0.7218237063084127,
                               0.5668675596240081, 0.5587684928683124, 0.36425765594194476,
                               0.6151498475519698, 0.8984810108033087, 0.3184894251163475,
                               0.3136969914559097, 0.5355007421820464, 0.5175969259573149,
                               0.43711848419991794, 0.6610959733192684, 0.30828602794747867,
                               0.36425765594194476, 0.3843133311924944, 0.5541495975546271,
                               0.878270167403773, 0.5523884099446082, 0.5842918137900647,
                               0.6151498475519698, 0.43711848419991794, 0.3843133311924944,
                               0.6160217057162051, 0.3657792538597795, 0.5953909053658156,
                               0.5205377943383558, 0.6610959733192684, 0.6200169598027441,
                               0.5689305452193616, 0.596978948380081, 0.6456616813411833,
                               0.5915091093521218, 0.5648400479211761, 0.5852102407917118,
                               0.5496443467517238, 0.5504673846415659, 0.8984810108033087,
                               0.6160217057162051, 0.5238826236796035, 0.35483032135142517,
                               0.5689305452193616, 0.45561647944986217, 0.8373375603729933,
                               0.7492161311669502, 0.5402212420937037, 0.4333758907536177,
                               0.596978948380081, 0.5895386196231154, 0.4737520004431321,
                               0.532687366794617, 0.5588993628146661, 0.6456616813411833,
                               0.5895386196231154, 0.5478307357327432, 0.49616738639081326,
                               0.8208633322850427, 0.7218237063084127, 0.6200169598027441,
                               0.45561647944986217, 0.4888202345591516, 0.5498675558344721,
                               0.5915091093521218, 0.8373375603729933, 0.4737520004431321,
                               0.4888202345591516, 0.7360558249705658, 0.5621669114261062,
                               0.4703722973501223, 0.5648400479211761, 0.532687366794617,
                               0.5478307357327432, 0.4626642723017695, 0.5852102407917118,
                               0.7492161311669502, 0.7360558249705658, 0.3184894251163475,
                               0.5541495975546271, 0.5816810478730438, 0.5496443467517238,
                               0.5402212420937037, 0.5588993628146661, 0.49616738639081326,
                               0.5621669114261062, 0.32769175050170984, 0.4333758907536177,
                               0.4703722973501223, 0.32769175050170984, 0.3136969914559097,
                               0.878270167403773, 0.3657792538597795, 0.5816810478730438],
                              20, 20))
        @test isapprox(vec(tri1),
                       [10, 14, 10, 10, 10, 14, 10, 1, 10, 14, 1, 13, 1, 2, 10, 14, 14, 8,
                        8, 18, 8, 11, 1, 2, 1, 2, 1, 6, 10, 14, 2, 6, 6, 3, 3, 5, 14, 1, 14,
                        1, 1, 1, 13, 13, 16, 16, 2, 2, 7, 7, 8, 8, 18, 18, 11, 11, 12, 12,
                        4, 4, 6, 6, 3, 3, 18, 18, 3, 3, 5, 5, 20, 20, 19, 13, 13, 2, 16, 16,
                        2, 7, 8, 8, 3, 7, 4, 4, 18, 11, 11, 12, 15, 12, 15, 15, 6, 6, 9, 5,
                        9, 9, 19, 19, 5, 20, 20, 17, 17, 17])
        @test isapprox(vec(separators1),
                       [10, 10, 10, 1, 1, 10, 14, 8, 8, 1, 1, 1, 10, 2, 6, 3, 14, 1, 14, 13,
                        2, 14, 8, 18, 11, 2, 2, 6, 14, 6, 3, 5, 1, 13, 16, 2, 7, 8, 18, 11,
                        12, 4, 6, 3, 18, 3, 5, 20])
        @test isapprox(vec(cliques1),
                       [10, 10, 10, 10, 1, 1, 10, 14, 8, 8, 1, 1, 1, 10, 2, 6, 3, 14, 14, 1,
                        14, 13, 2, 14, 8, 18, 11, 2, 2, 6, 14, 6, 3, 5, 1, 1, 13, 16, 2, 7,
                        8, 18, 11, 12, 4, 6, 3, 18, 3, 5, 20, 13, 16, 2, 8, 7, 4, 18, 11,
                        12, 15, 6, 3, 9, 19, 5, 20, 17])
        @test isapprox(cliqueTree1,
                       sparse([4, 5, 7, 14, 3, 7, 14, 1, 2, 6, 11, 12, 1, 2, 8, 14, 3, 11,
                               12, 5, 12, 1, 2, 4, 9, 7, 10, 14, 8, 9, 5, 6, 13, 15, 5, 6,
                               11, 16, 12, 15, 16, 1, 2, 4, 7, 8, 12, 13, 17, 13, 15, 16],
                              [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6,
                               6, 7, 7, 7, 7, 8, 8, 8, 9, 10, 11, 11, 11, 11, 12, 12, 12,
                               12, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 17, 17))
        clr_t = Hclust{Float64}([-1 -13;
                                 -7 -4;
                                 -3 -9;
                                 -5 -20;
                                 4 -17;
                                 -2 1;
                                 3 -6;
                                 7 6;
                                 8 2;
                                 5 9;
                                 -14 -10;
                                 -19 11;
                                 -11 -8;
                                 -12 -15;
                                 -18 13;
                                 -16 15;
                                 16 14;
                                 12 17;
                                 10 18],
                                [0.1, 0.1111111111111111, 0.125, 0.14285714285714285,
                                 0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.5,
                                 1.0, 0.125, 0.14285714285714285, 0.16666666666666666, 0.2,
                                 0.25, 0.3333333333333333, 0.5, 1.0, 2.0],
                                [5, 20, 17, 3, 9, 6, 2, 1, 13, 7, 4, 19, 14, 10, 16, 18, 11,
                                 8, 12, 15], :DBHT)
        @test clr.res.merges == clr_t.merges
        @test isapprox(clr.res.heights, clr_t.heights)
        @test clr.res.labels == clr_t.labels
        @test clr.res.linkage == clr_t.linkage

        @test 4 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = nothing),
                                                               clr.res, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = 1),
                                                               clr.res, clr.D)
        @test 4 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = 100),
                                                               clr.res, clr.D)
        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SilhouetteScore(),
                                                                                     max_k = nothing),
                                                               clr.res, clr.D)
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SilhouetteScore(),
                                                                                     max_k = 1),
                                                               clr.res, clr.D)
        @test 2 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SilhouetteScore(),
                                                                                     max_k = 100),
                                                               clr.res, clr.D)
        @test 4 ==
              PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(; alg = 10,
                                                                                max_k = nothing),
                                                          clr.res, clr.D)
        @test 1 ==
              PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(; alg = 1,
                                                                                max_k = 5),
                                                          clr.res, clr.D)
        @test 2 ==
              PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(; alg = 2,
                                                                                max_k = 5),
                                                          clr.res, clr.D)
        @test 2 ==
              PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(; alg = 3,
                                                                                max_k = 2),
                                                          clr.res, clr.D)

        clr = clusterise(ClustersEstimator(; ce = PortfolioOptimisersCovariance(),
                                           de = DistanceDistance(;
                                                                 alg = CanonicalDistance()),
                                           alg = HClustAlgorithm(),
                                           onc = OptimalNumberClusters(;
                                                                       alg = SilhouetteScore())),
                         pr.X)
        @test clr.k == 2

        clr = clusterise(ClustersEstimator(; ce = PortfolioOptimisersCovariance(),
                                           de = DistanceDistance(;
                                                                 alg = CanonicalDistance()),
                                           alg = DBHT(; sim = MaximumDistanceSimilarity(),
                                                      root = UniqueRoot()),
                                           onc = OptimalNumberClusters(; alg = 5)), pr.X)
        @test clr.k == 4

        clr = clusterise(ClustersEstimator(; ce = PortfolioOptimisersCovariance(),
                                           de = DistanceDistance(;
                                                                 alg = CanonicalDistance()),
                                           alg = DBHT(; sim = MaximumDistanceSimilarity(),
                                                      root = EqualRoot()),
                                           onc = OptimalNumberClusters(; alg = 2)), pr.X)
        @test clr.k == 2

        alg = KMeansAlgorithm(; rng = StableRNG(42), seed = 42, kwargs = (; init = :kmcen))
        @test factory(alg) === alg
        wt2 = pweights(fill(inv(size(pr.X, 2)), size(pr.X, 2)))
        alg2 = factory(alg, wt2)
        @test alg2.kwargs[:weights] === wt2
        clr = clusterise(ClustersEstimator(; ce = PortfolioOptimisersCovariance(),
                                           de = Distance(; alg = CanonicalDistance()),
                                           alg = alg,
                                           onc = OptimalNumberClusters(;
                                                                       alg = SilhouetteScore())),
                         pr.X)
        @test clr.res.assignments ==
              [3, 3, 3, 3, 1, 3, 3, 2, 3, 2, 2, 2, 3, 2, 2, 2, 1, 2, 1, 1]
        @test isapprox(clr.res.costs,
                       [0.17766245474431486, 0.19080736561118172, 0.22868094113151827,
                        0.27672765546618017, 0.15862426791009376, 0.24968616703931623,
                        0.2588799728188551, 0.1786262195742978, 0.22722331843868737,
                        0.22990129758696654, 0.2567847522783442, 0.2704494266876818,
                        0.18382368006096605, 0.21331389368901732, 0.274581606177021,
                        0.22265814453145616, 0.22658135714037542, 0.22219845380251435,
                        0.5342043271374468, 0.16001795714648281])
        @test clr.k == 3

        @test 4 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = nothing),
                                                               alg, clr.D)[2]
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = 1),
                                                               alg, clr.D)[2]
        @test 4 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SecondOrderDifference(),
                                                                                     max_k = 100),
                                                               alg, clr.D)[2]
        @test 3 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SilhouetteScore(),
                                                                                     max_k = nothing),
                                                               alg, clr.D)[2]
        @test 1 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SilhouetteScore(),
                                                                                     max_k = 1),
                                                               alg, clr.D)[2]
        @test 3 == PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = SilhouetteScore(),
                                                                                     max_k = 100),
                                                               alg, clr.D)[2]
        # A single cluster has no silhouette, so `W_list[1]` is never computed — it must
        # still be *written*, or `argmax` reads uninitialised memory and returns `k = 1`
        # whenever that garbage beats every real score. That made this whole block flaky,
        # since the garbage is whatever the allocator last left there. Fresh allocations
        # each iteration, so a reintroduction fails here rather than intermittently
        # somewhere downstream.
        onc_sil = OptimalNumberClusters(; alg = SilhouetteScore(), max_k = nothing)
        rng_sil = StableRNG(1234)
        for _ in 1:50
            Dr = PortfolioOptimisers.cor_and_dist(Distance(; alg = CanonicalDistance()),
                                                  PortfolioOptimisersCovariance(),
                                                  randn(rng_sil, 200, 20))[2]
            alg_sil = KMeansAlgorithm(; rng = StableRNG(42), seed = 42,
                                      kwargs = (; init = :kmcen))
            k = PortfolioOptimisers.optimal_number_clusters(onc_sil, alg_sil, Dr)[2]
            # `max_k = nothing` gives `c1 = floor(sqrt(20)) = 4`, and the silhouette of a
            # one-cluster partition is undefined, so `k = 1` is never a legitimate answer.
            @test 2 <= k <= 4
        end
        @test 4 ==
              PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(; alg = 10,
                                                                                max_k = nothing),
                                                          alg, clr.D)[2]
        @test 1 ==
              PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(; alg = 1,
                                                                                max_k = 5),
                                                          alg, clr.D)[2]
        @test 2 ==
              PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(; alg = 2,
                                                                                max_k = 5),
                                                          alg, clr.D)[2]
        @test 2 ==
              PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(; alg = 3,
                                                                                max_k = 2),
                                                          alg, clr.D)[2]

        cle0 = ClustersEstimator()
        @test factory(cle0) === cle0
        cle1 = factory(cle0, wt)
        @test cle1.ce.ce.me.w === wt
        @test cle1.ce.ce.ce.w === wt

        onc = OptimalNumberClusters(; alg = SilhouetteScore())
        @test factory(onc) === onc
        onc2 = factory(onc, wt)
        @test onc2.alg.alg.mv.w === wt
        @test onc2.alg.alg.sv.w === wt

        onc = OptimalNumberClusters(;)
        @test factory(onc) === onc
        onc2 = factory(onc, wt)
        @test onc2.alg.alg.mv.w === wt
        @test onc2.alg.alg.sv.w === wt

        clr = clusterise(NetworkClustersEstimator(; nte = NetworkEstimator(;)), pr.X)
        clr_t = Hclust{Float64}([-2 -1; -7 -4; 1 -13; -20 -5; -9 -3; -6 5; -19 -14; -10 -16;
                                 -15 -8; 9 -12; 10 -11; 7 -18; 3 2; -17 4; 6 13; 12 8;
                                 16 11; 14 15; 18 17],
                                [0.0, 0.0, 0.3045085076488489, 0.0, 0.0, 0.3581666988279946,
                                 0.0, 0.0, 0.0, 0.3882307150253385, 0.42633640634280284,
                                 0.47339491952572044, 0.6532873901085925,
                                 0.3855084964240403, 0.775920065742249, 0.6145535259841394,
                                 0.8716917373925989, 1.033182981547855, 1.1992052386632663],
                                [17, 20, 5, 6, 9, 3, 2, 1, 13, 7, 4, 19, 14, 18, 10, 16, 15,
                                 8, 12, 11], :ward)
        @test clr.res.merges == clr_t.merges
        @test isapprox(clr.res.heights, clr_t.heights)
        @test clr.res.labels == clr_t.labels
        @test clr.res.linkage == clr_t.linkage

        clr = clusterise(NetworkClustersEstimator(;
                                                  nte = NetworkEstimator(;
                                                                         alg = MaximumDistanceSimilarity()),
                                                  alg = DBHT()), pr.X)
        clr_t = Hclust{Float64}([-1 -13; -7 -4; -9 -3; -5 -17; -20 4; -2 -6; -16 1; 3 2;
                                 6 5; 8 7; 10 9; -14 -10; 12 -19; -11 -8; -15 -12; 13 -18;
                                 15 14; 17 16; 11 18],
                                [0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                                 0.14285714285714285, 0.16666666666666666, 0.2, 0.25,
                                 0.3333333333333333, 0.5, 1.0, 0.14285714285714285,
                                 0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.5,
                                 1.0, 2.0],
                                [9, 3, 7, 4, 16, 1, 13, 2, 6, 20, 5, 17, 15, 12, 11, 8, 14,
                                 10, 19, 18], :DBHT)
        @test clr.res.merges == clr_t.merges
        @test isapprox(clr.res.heights, clr_t.heights)
        @test clr.res.labels == clr_t.labels
        @test clr.res.linkage == clr_t.linkage

        clr = clusterise(NetworkClustersEstimator(;
                                                  alg = KMeansAlgorithm(;
                                                                        rng = StableRNG(42),
                                                                        seed = 42,
                                                                        kwargs = (;
                                                                                  init = :kmcen))),
                         pr.X)
        @test clr.res.assignments ==
              [2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2]
        @test isapprox(clr.res.costs,
                       [0.8084243382409362, 0.4823137818592649, 0.4449980885512659,
                        0.5756168699151285, 0.8319611307695158, 0.6650833126385258,
                        0.7305444332229705, 0.6436524455289394, 0.5635425434334671,
                        0.45097786305584187, 0.41692555461393965, 0.44452315606872084,
                        0.5833758245529381, 0.859913739023499, 0.44658151098578713,
                        0.4246925133166215, 0.8043043578502438, 0.3386253443760783,
                        0.6407143305429042, 1.0446276900764264])
    end
    @testset "Centrality tests" begin
        ces = [BetweennessCentrality(), ClosenessCentrality(), DegreeCentrality(),
               EigenvectorCentrality(), KatzCentrality(), Pagerank(), RadialityCentrality(),
               StressCentrality()]
        df1 = CSV.read(joinpath(@__DIR__, "./assets/Centrality1.csv.gz"), DataFrame)
        df2 = CSV.read(joinpath(@__DIR__, "./assets/AverageCentrality1.csv.gz"), DataFrame)
        w = fill(inv(20), 20)
        for (i, ce) in enumerate(ces)
            v1 = centrality_vector(CentralityEstimator(; ct = ce), pr)
            @test v1 === centrality_vector(v1)
            v1 = v1.X
            res = isapprox(v1, df1[!, i])
            if !res
                println("Default centrality iteration: $i")
                find_tol(v1, df1[!, i])
            end
            @test res

            cte = CentralityEstimator(; ct = ce)
            c = average_centrality(cte, w, pr)
            @test isapprox(c, average_centrality(cte.pl, cte.ct, w, pr))
            res = isapprox(c, df2[i, 1])
            if !res
                println("Average default centrality iteration: $i")
                find_tol(c, df2[i, 1])
            end
            @test res
        end
    end
    @testset "Phylogeny matrix" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/PhylogenyMatrix1.csv.gz"), DataFrame)
        for i in 1:8
            A = phylogeny_matrix(NetworkEstimator(; sep = HopCount(; n = i)), pr)
            @test A === phylogeny_matrix(A)
            A = A.X
            res = isapprox(vec(A), df[!, i])
            if !res
                println("Iteration $i failed on detault network estimator.")
                find_tol(vec(A), df[!, i]; name1 = :A, name2 = :df)
            end
            @test res
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/PhylogenyMatrix2.csv.gz"), DataFrame)
        for i in 1:5
            A = phylogeny_matrix(NetworkEstimator(; sep = HopCount(; n = i),
                                                  alg = MaximumDistanceSimilarity()), pr).X
            res = isapprox(vec(A), df[!, i])
            if !res
                println("Iteration $i failed on MaximumDistanceSimilarity.")
                find_tol(vec(A), df[!, i]; name1 = :A, name2 = :df)
            end
            @test res
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/PhylogenyMatrix3.csv.gz"), DataFrame)
        A = phylogeny_matrix(ClustersEstimator(), pr).X
        @test isapprox(vec(A), df[!, 1])

        w = fill(inv(20), 20)
        @test isapprox(asset_phylogeny(NetworkEstimator(), w, pr), 0.09500000000000008)
        @test isapprox(asset_phylogeny(ClustersEstimator(), w, pr), 0.4550000000000004)

        A1 = PortfolioOptimisers.calc_adjacency(NetworkEstimator(; alg = KruskalTree()),
                                                pr.X)
        A2 = PortfolioOptimisers.calc_adjacency(NetworkEstimator(; alg = BoruvkaTree()),
                                                pr.X)
        A3 = PortfolioOptimisers.calc_adjacency(NetworkEstimator(; alg = PrimTree()), pr.X)

        @test A1 == A2
        @test A1 == A3

        df = CSV.read(joinpath(@__DIR__, "./assets/PhylogenyMatrix4.csv.gz"), DataFrame)
        A = phylogeny_matrix(ClustersEstimator(;
                                               alg = KMeansAlgorithm(; rng = StableRNG(420),
                                                                     kwargs = (;
                                                                               init = :kmcen))),
                             pr).X
        @test isapprox(vec(A), df[!, 1])
    end
    #=
    @testset "DBHT Clustering tests" begin
        X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                      timestamp = :timestamp)
        rd = prices_to_returns(X[(end - 252):end])
        X = rd.X
        ce = PortfolioOptimisersCovariance()
        de = Distance(; alg = SimpleDistance())
        rho = cor(ce, X)
        dist = distance(de, rho, X)

        @test isapprox(PortfolioOptimisers.distance_to_similarity(GeneralExponentialSimilarity();
                                                           D = rho),
                       PortfolioOptimisers.distance_to_similarity(ExponentialSimilarity(); D = rho))

        sim = MaximumDistanceSimilarity()
        S = PortfolioOptimisers.distance_to_similarity(sim; S = rho, D = dist)
        root = UniqueRoot()
        T8, Rpm, Adjv, Dpm, Mv, Z1, dbht = PortfolioOptimisers.DBHTs(dist, S;
                                                                     branchorder = :default,
                                                                     root = root)
        Z1_t = reshape([-3.0, -1.0, -7.0, -25.0, -21.0, -2.0, -8.0, -6.0, -28.0, -26.0, -22.0,
                        -14.0, -9.0, -11.0, -30.0, -4.0, -17.0, -10.0, -5.0, 6.0, 17.0, 2.0,
                        18.0, 11.0, 13.0, 22.0, 4.0, 15.0, 27.0, -29.0, 1.0, -16.0, 3.0, -24.0,
                        -19.0, -15.0, -12.0, 8.0, -27.0, 10.0, -20.0, 12.0, -18.0, 14.0, -13.0,
                        -23.0, 16.0, 7.0, 9.0, 20.0, 5.0, 19.0, 23.0, 21.0, 25.0, 26.0, 24.0,
                        28.0, 0.034482758620689655, 0.03571428571428571, 0.037037037037037035,
                        0.038461538461538464, 0.04, 0.041666666666666664, 0.043478260869565216,
                        0.045454545454545456, 0.047619047619047616, 0.05, 0.05263157894736842,
                        0.05555555555555555, 0.058823529411764705, 0.0625, 0.06666666666666667,
                        0.07142857142857142, 0.07692307692307693, 0.08333333333333333,
                        0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                        0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                        0.5, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 3.0, 2.0,
                        3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0, 5.0, 7.0, 5.0, 6.0, 9.0, 10.0, 15.0,
                        18.0, 12.0, 30.0], :, 4)
        @test isapprox(Z1, Z1_t)

        A1, tri1, separators1, cliques1, cliqueTree1 = PortfolioOptimisers.PMFG_T2s(S, 5)
        A1_t = SparseArrays.sparse([3, 24, 29, 3, 6, 19, 1, 2, 6, 7, 10, 14, 15, 16, 17, 19, 21, 24, 29, 5,
                       8, 10, 11, 13, 15, 16, 22, 26, 27, 4, 8, 11, 15, 18, 30, 2, 3, 12, 16,
                       17, 19, 21, 23, 24, 28, 3, 10, 16, 25, 29, 4, 5, 15, 16, 14, 17, 20, 3,
                       4, 7, 13, 15, 16, 25, 4, 5, 15, 18, 30, 6, 17, 19, 28, 4, 10, 15, 22, 3,
                       9, 17, 19, 20, 3, 4, 5, 8, 10, 11, 13, 16, 18, 22, 27, 3, 4, 6, 7, 8, 10,
                       15, 17, 23, 24, 25, 29, 3, 6, 9, 12, 14, 16, 19, 20, 23, 28, 5, 11, 15,
                       30, 2, 3, 6, 12, 14, 17, 20, 9, 14, 17, 19, 3, 6, 24, 4, 13, 15, 26, 27,
                       6, 16, 17, 1, 3, 6, 16, 21, 29, 7, 10, 16, 4, 22, 27, 4, 15, 22, 26, 6,
                       12, 17, 1, 3, 7, 16, 24, 5, 11, 18],
                      [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                       4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
                       7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
                       11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15,
                       15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                       16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19,
                       19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23,
                       23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 28,
                       28, 28, 29, 29, 29, 29, 29, 30, 30, 30],
                      [0.7816641847363256, 0.7316807590015212, 0.7683746613243841,
                       0.7902418858652103, 0.7901906289071934, 0.7950432446999863,
                       0.7816641847363256, 0.7902418858652103, 0.7915181774751375,
                       0.7696847921093792, 0.7933378743892743, 0.7888571183860922,
                       0.7557516290489048, 0.8004821874279777, 0.7800037858385857,
                       0.7775134750269215, 0.7782824257876472, 0.8757969868545106,
                       0.940246788899921, 0.7060748528496048, 0.7368999498709721,
                       0.7737888218120279, 0.6928099997482, 0.7804589095290857,
                       0.8431995287270456, 0.7609187638477687, 0.7563996400073232,
                       0.6697431022982079, 0.7387004451572843, 0.7060748528496048,
                       0.6531408875956941, 0.7235603792607732, 0.7363185959263081,
                       0.7095744015387154, 0.7109498158447688, 0.7901906289071934,
                       0.7915181774751375, 0.825313418545372, 0.7972892531534442,
                       0.8636318852869648, 0.8545064792020292, 0.773100411774844,
                       0.7881479607984379, 0.77232315344912, 0.7883261802417582,
                       0.7696847921093792, 0.7427616167265069, 0.8280873727403872,
                       0.6992674912804311, 0.7426100374529017, 0.7368999498709721,
                       0.6531408875956941, 0.7170659075173489, 0.694208204520547,
                       0.7745741975682464, 0.7456074483471982, 0.7247481694195679,
                       0.7933378743892743, 0.7737888218120279, 0.7427616167265069,
                       0.7606686996464144, 0.7873073555828847, 0.7826619148453403,
                       0.7155234015554408, 0.6928099997482, 0.7235603792607732,
                       0.661643865044952, 0.6802392647412279, 0.6640068485897885,
                       0.825313418545372, 0.8102576225087195, 0.8072731782293077,
                       0.7874319636394267, 0.7804589095290857, 0.7606686996464144,
                       0.7911386173022757, 0.6793609775167029, 0.7888571183860922,
                       0.7745741975682464, 0.8231535542454383, 0.786281888942755,
                       0.8004665082535787, 0.7557516290489048, 0.8431995287270456,
                       0.7363185959263081, 0.7170659075173489, 0.7873073555828847,
                       0.661643865044952, 0.7911386173022757, 0.7511113235075095,
                       0.6313292411074813, 0.7164639522699217, 0.6935176920107173,
                       0.8004821874279777, 0.7609187638477687, 0.7972892531534442,
                       0.8280873727403872, 0.694208204520547, 0.7826619148453403,
                       0.7511113235075095, 0.7745043933535881, 0.7848223314871908,
                       0.7617315985880675, 0.7143000938896955, 0.8039864466804436,
                       0.7800037858385857, 0.8636318852869648, 0.7456074483471982,
                       0.8102576225087195, 0.8231535542454383, 0.7745043933535881,
                       0.836276826496608, 0.7404841297603928, 0.7735740633005248,
                       0.7588707977185343, 0.7095744015387154, 0.6802392647412279,
                       0.6313292411074813, 0.6457088737163896, 0.7950432446999863,
                       0.7775134750269215, 0.8545064792020292, 0.8072731782293077,
                       0.786281888942755, 0.836276826496608, 0.759345751786569,
                       0.7247481694195679, 0.8004665082535787, 0.7404841297603928,
                       0.759345751786569, 0.7782824257876472, 0.773100411774844,
                       0.8310547931419772, 0.7563996400073232, 0.6793609775167029,
                       0.7164639522699217, 0.6704282888502833, 0.6612756054878979,
                       0.7881479607984379, 0.7848223314871908, 0.7735740633005248,
                       0.7316807590015212, 0.8757969868545106, 0.77232315344912,
                       0.7617315985880675, 0.8310547931419772, 0.848708256191095,
                       0.6992674912804311, 0.7155234015554408, 0.7143000938896955,
                       0.6697431022982079, 0.6704282888502833, 0.7404966900836785,
                       0.7387004451572843, 0.6935176920107173, 0.6612756054878979,
                       0.7404966900836785, 0.7883261802417582, 0.7874319636394267,
                       0.7588707977185343, 0.7683746613243841, 0.940246788899921,
                       0.7426100374529017, 0.8039864466804436, 0.848708256191095,
                       0.7109498158447688, 0.6640068485897885, 0.6457088737163896], 30, 30)
        tri1_t = reshape([16, 3, 16, 16, 16, 3, 16, 3, 16, 3, 3, 6, 6, 17, 3, 17, 3, 24, 3, 6,
                          16, 6, 6, 17, 16, 3, 17, 19, 16, 3, 16, 10, 10, 15, 3, 29, 17, 14, 15,
                          4, 16, 15, 16, 7, 15, 4, 15, 4, 4, 22, 15, 4, 15, 5, 5, 11, 3, 7, 3,
                          7, 29, 29, 24, 24, 6, 6, 17, 17, 19, 19, 19, 19, 6, 6, 19, 19, 17, 17,
                          12, 12, 7, 7, 14, 14, 10, 10, 15, 15, 4, 4, 24, 24, 20, 20, 13, 13, 4,
                          4, 10, 10, 8, 8, 22, 22, 27, 27, 5, 5, 11, 11, 18, 18, 15, 29, 17, 29,
                          24, 1, 6, 21, 23, 2, 14, 28, 12, 12, 14, 20, 21, 21, 2, 2, 23, 23, 28,
                          28, 25, 10, 9, 20, 4, 15, 8, 13, 13, 27, 1, 1, 9, 9, 22, 22, 8, 11,
                          25, 25, 5, 5, 27, 26, 26, 26, 18, 11, 18, 30, 30, 30], :, 3)
        separators1_t = reshape([16, 16, 16, 3, 6, 3, 3, 3, 16, 6, 16, 17, 16, 16, 10, 3, 17,
                                 15, 16, 16, 15, 15, 4, 15, 15, 5, 3, 3, 3, 6, 17, 17, 24, 6, 6,
                                 17, 3, 19, 3, 10, 15, 29, 14, 4, 15, 7, 4, 4, 22, 4, 5, 11, 29,
                                 24, 6, 17, 19, 19, 6, 19, 17, 12, 7, 14, 10, 15, 4, 24, 20, 13,
                                 4, 10, 8, 22, 27, 5, 11, 18], :, 3)
        cliques1_t = reshape([16, 16, 16, 16, 3, 6, 3, 3, 3, 16, 6, 16, 17, 16, 16, 10, 3, 17,
                              15, 16, 16, 15, 15, 4, 15, 15, 5, 3, 3, 3, 3, 6, 17, 17, 24, 6, 6,
                              17, 3, 19, 3, 10, 15, 29, 14, 4, 15, 7, 4, 4, 22, 4, 5, 11, 7, 29,
                              24, 6, 17, 19, 19, 6, 19, 17, 12, 7, 14, 10, 15, 4, 24, 20, 13, 4,
                              10, 8, 22, 27, 5, 11, 18, 29, 24, 6, 17, 19, 12, 14, 21, 2, 23,
                              28, 10, 20, 15, 4, 13, 1, 9, 22, 8, 25, 5, 27, 26, 11, 18, 30], :,
                             4)
        cliqueTree1_t = SparseArrays.sparse([3, 4, 14, 17, 21, 1, 4, 8, 12, 14, 1, 2, 5, 9, 10, 12, 14, 17,
                                1, 2, 3, 6, 7, 8, 9, 11, 12, 14, 4, 8, 10, 11, 13, 5, 7, 9, 10,
                                13, 5, 6, 9, 18, 3, 4, 5, 9, 17, 4, 5, 6, 7, 8, 4, 5, 6, 11, 5,
                                6, 10, 2, 3, 4, 15, 6, 7, 1, 2, 3, 4, 12, 16, 20, 21, 14, 19,
                                21, 22, 23, 25, 15, 20, 22, 23, 25, 2, 3, 8, 13, 16, 20, 22, 24,
                                25, 15, 16, 19, 23, 25, 1, 12, 14, 15, 16, 19, 20, 23, 26, 16,
                                19, 20, 22, 25, 23, 16, 19, 20, 22, 23, 27, 25, 26],
                               [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                                4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7,
                                8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12,
                                12, 12, 12, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15,
                                15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 18, 19, 19, 19, 19,
                                19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23,
                                23, 23, 23, 23, 24, 25, 25, 25, 25, 25, 25, 26, 27],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 27, 27)
        @test isapprox(A1, A1_t)
        @test isapprox(tri1, tri1_t)
        @test isapprox(separators1, separators1_t)
        @test isapprox(cliques1, cliques1_t)
        @test isapprox(cliqueTree1, cliqueTree1_t)

        sim = ExponentialSimilarity()
        S = PortfolioOptimisers.distance_to_similarity(sim; S = rho, D = dist)

        root = EqualRoot()
        T8, Rpm, Adjv, Dpm, Mv, Z2, dbht = PortfolioOptimisers.DBHTs(dist, S;
                                                                     branchorder = :optimal,
                                                                     root = root)
        Z2_t = reshape([-3.0, -1.0, -10.0, -25.0, -21.0, -9.0, -8.0, -14.0, -23.0, -26.0, -22.0,
                        -11.0, -30.0, -6.0, -28.0, -2.0, -4.0, -7.0, -5.0, 6.0, 2.0, 17.0, 15.0,
                        4.0, 11.0, 18.0, 13.0, 24.0, 27.0, -29.0, 1.0, -16.0, 3.0, -24.0, -17.0,
                        -15.0, -20.0, 8.0, -27.0, 10.0, -18.0, 12.0, -12.0, 14.0, -19.0, -13.0,
                        9.0, 7.0, 16.0, 5.0, 19.0, 20.0, 21.0, 22.0, 23.0, 25.0, 26.0, 28.0,
                        0.034482758620689655, 0.03571428571428571, 0.037037037037037035,
                        0.038461538461538464, 0.04, 0.041666666666666664, 0.043478260869565216,
                        0.045454545454545456, 0.047619047619047616, 0.05, 0.05263157894736842,
                        0.05555555555555555, 0.058823529411764705, 0.0625, 0.06666666666666667,
                        0.07142857142857142, 0.07692307692307693, 0.08333333333333333,
                        0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                        0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                        0.5, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 3.0, 2.0,
                        3.0, 2.0, 3.0, 2.0, 2.0, 4.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 8.0, 11.0,
                        11.0, 19.0, 30.0], :, 4)
        @test isapprox(Z2, Z2_t)

        A2, tri2, separators2, cliques2, cliqueTree2 = PortfolioOptimisers.PMFG_T2s(S, 5)
        A2_t = SparseArrays.sparse([3, 10, 29, 3, 6, 9, 14, 16, 17, 19, 24, 29, 1, 2, 6, 7, 10, 14, 15, 16,
                       17, 21, 24, 29, 5, 8, 10, 11, 13, 15, 16, 22, 26, 27, 4, 8, 11, 15, 18,
                       30, 2, 3, 12, 17, 19, 28, 3, 14, 16, 20, 23, 4, 5, 15, 16, 2, 14, 17, 1,
                       3, 4, 13, 15, 16, 25, 29, 4, 5, 15, 18, 30, 6, 17, 19, 28, 4, 10, 15, 22,
                       2, 3, 7, 9, 16, 17, 20, 23, 3, 4, 5, 8, 10, 11, 13, 16, 18, 22, 27, 2, 3,
                       4, 7, 8, 10, 14, 15, 23, 25, 29, 2, 3, 6, 9, 12, 14, 19, 28, 5, 11, 15,
                       30, 2, 6, 12, 17, 7, 14, 23, 3, 24, 29, 4, 13, 15, 26, 27, 7, 14, 16, 20,
                       2, 3, 21, 29, 10, 16, 29, 4, 22, 27, 4, 15, 22, 26, 6, 12, 17, 1, 2, 3,
                       10, 16, 21, 24, 25, 5, 11, 18],
                      [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                       4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7,
                       7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11,
                       11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
                       15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16,
                       16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19,
                       19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 24,
                       24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 29, 29,
                       29, 29, 29, 29, 29, 29, 30, 30, 30],
                      [0.626714502196399, 0.5952276453835667, 0.6179951299556474,
                       0.6325515407567773, 0.6325161474870034, 0.5924079972337691,
                       0.6106041919669216, 0.6206866609821369, 0.6286681151371817,
                       0.6358952066692547, 0.6265615221157578, 0.6223013135544568,
                       0.626714502196399, 0.6325515407567773, 0.633434868048951,
                       0.6188380533633001, 0.634701123817698, 0.6315975576372738,
                       0.6100496671262657, 0.639752325945361, 0.6256040973828371,
                       0.6244594043846303, 0.7029816579361001, 0.7831392951842335,
                       0.5814974807568316, 0.5987370295764817, 0.6215017020446391,
                       0.5745045639699345, 0.6259078511190505, 0.673019892021328,
                       0.6132642618250566, 0.6104500103814954, 0.5628850634269977,
                       0.5997906020038952, 0.5814974807568316, 0.5549110427046743,
                       0.5910969347285154, 0.5983980109272683, 0.5833829226498065,
                       0.584128739811752, 0.6325161474870034, 0.633434868048951,
                       0.6583916983140445, 0.6912314675204657, 0.6828800006328515,
                       0.6312329984708543, 0.6188380533633001, 0.612064361367784,
                       0.6605889611569918, 0.598013939116339, 0.6153223437038713,
                       0.5987370295764817, 0.5549110427046743, 0.587478579683404,
                       0.5752305027550456, 0.5924079972337691, 0.6220154968220857,
                       0.6038838355405132, 0.5952276453835667, 0.634701123817698,
                       0.6215017020446391, 0.6131075046565078, 0.6305353063656726,
                       0.6273847253959038, 0.5866285361925857, 0.6269545308834255,
                       0.5745045639699345, 0.5910969347285154, 0.5589563588640523,
                       0.5680908628481256, 0.5600948344560923, 0.6583916983140445,
                       0.6468801243693407, 0.6446765049795941, 0.630620506889349,
                       0.6259078511190505, 0.6131075046565078, 0.6331717620414965,
                       0.5676501604783291, 0.6106041919669216, 0.6315975576372738,
                       0.612064361367784, 0.6220154968220857, 0.6055263583359068,
                       0.6566979239106111, 0.6397410979586964, 0.6309510710647757,
                       0.6100496671262657, 0.673019892021328, 0.5983980109272683,
                       0.587478579683404, 0.6305353063656726, 0.5589563588640523,
                       0.6331717620414965, 0.6072058377708328, 0.5448838422759799,
                       0.5871464329433354, 0.574871671642645, 0.6206866609821369,
                       0.639752325945361, 0.6132642618250566, 0.6605889611569918,
                       0.5752305027550456, 0.6273847253959038, 0.6055263583359068,
                       0.6072058377708328, 0.6288437457362273, 0.5859569051741962,
                       0.6422779153086382, 0.6286681151371817, 0.6256040973828371,
                       0.6912314675204657, 0.6038838355405132, 0.6468801243693407,
                       0.6566979239106111, 0.6672255040868339, 0.6119840350234862,
                       0.5833829226498065, 0.5680908628481256, 0.5448838422759799,
                       0.5514392538382796, 0.6358952066692547, 0.6828800006328515,
                       0.6446765049795941, 0.6672255040868339, 0.598013939116339,
                       0.6397410979586964, 0.6041771689536221, 0.6244594043846303,
                       0.6629674084628745, 0.600460517353584, 0.6104500103814954,
                       0.5676501604783291, 0.5871464329433354, 0.5632208996756527,
                       0.5587794994968943, 0.6153223437038713, 0.6309510710647757,
                       0.6288437457362273, 0.6041771689536221, 0.6265615221157578,
                       0.7029816579361001, 0.6629674084628745, 0.6777597574877918,
                       0.5866285361925857, 0.5859569051741962, 0.576913460418243,
                       0.5628850634269977, 0.5632208996756527, 0.6008471674747321,
                       0.5997906020038952, 0.574871671642645, 0.5587794994968943,
                       0.6008471674747321, 0.6312329984708543, 0.630620506889349,
                       0.6119840350234862, 0.6179951299556474, 0.6223013135544568,
                       0.7831392951842335, 0.6269545308834255, 0.6422779153086382,
                       0.600460517353584, 0.6777597574877918, 0.576913460418243,
                       0.584128739811752, 0.5600948344560923, 0.5514392538382796], 30, 30)
        tri2_t = reshape([3, 16, 3, 3, 3, 16, 3, 2, 3, 14, 3, 2, 2, 17, 17, 6, 3, 16, 3, 16, 3,
                          29, 16, 14, 17, 6, 3, 16, 16, 10, 10, 15, 14, 7, 3, 29, 14, 2, 15, 4,
                          16, 15, 16, 29, 15, 4, 15, 4, 4, 22, 15, 4, 15, 5, 5, 11, 16, 14, 16,
                          14, 2, 2, 29, 29, 2, 2, 17, 17, 6, 6, 19, 19, 14, 14, 29, 29, 24, 24,
                          7, 7, 12, 12, 10, 10, 15, 15, 4, 4, 23, 23, 10, 10, 17, 17, 13, 13, 4,
                          4, 10, 10, 8, 8, 22, 22, 27, 27, 5, 5, 11, 11, 18, 18, 7, 2, 15, 17,
                          24, 29, 21, 24, 6, 9, 6, 19, 19, 28, 12, 12, 7, 23, 1, 25, 21, 21, 23,
                          20, 28, 28, 15, 4, 8, 13, 13, 27, 20, 20, 1, 1, 9, 9, 22, 22, 8, 11,
                          25, 25, 5, 5, 27, 26, 26, 26, 18, 11, 18, 30, 30, 30], :, 3)
        separators2_t = reshape([3, 3, 3, 3, 2, 17, 3, 3, 3, 16, 17, 3, 16, 10, 14, 3, 14, 15,
                                 16, 16, 15, 15, 4, 15, 15, 5, 16, 2, 14, 2, 17, 6, 16, 16, 29,
                                 14, 6, 16, 10, 15, 7, 29, 2, 4, 15, 29, 4, 4, 22, 4, 5, 11, 2,
                                 29, 2, 17, 6, 19, 14, 29, 24, 7, 12, 10, 15, 4, 23, 10, 17, 13,
                                 4, 10, 8, 22, 27, 5, 11, 18], :, 3)
        cliques2_t = reshape([3, 3, 3, 3, 3, 2, 17, 3, 3, 3, 16, 17, 3, 16, 10, 14, 3, 14, 15,
                              16, 16, 15, 15, 4, 15, 15, 5, 16, 16, 2, 14, 2, 17, 6, 16, 16, 29,
                              14, 6, 16, 10, 15, 7, 29, 2, 4, 15, 29, 4, 4, 22, 4, 5, 11, 14, 2,
                              29, 2, 17, 6, 19, 14, 29, 24, 7, 12, 10, 15, 4, 23, 10, 17, 13, 4,
                              10, 8, 22, 27, 5, 11, 18, 2, 29, 24, 17, 6, 19, 12, 7, 10, 21, 23,
                              28, 15, 4, 13, 20, 1, 9, 22, 8, 25, 5, 27, 26, 11, 18, 30], :, 4)
        cliqueTree2_t = SparseArrays.sparse([3, 5, 9, 11, 13, 18, 1, 4, 5, 8, 10, 13, 17, 21, 2, 4, 5, 9, 17,
                                1, 2, 3, 6, 8, 2, 3, 4, 7, 12, 18, 5, 12, 18, 6, 2, 4, 9, 13,
                                16, 1, 2, 3, 8, 10, 14, 3, 9, 17, 1, 8, 6, 7, 1, 2, 8, 9, 15,
                                17, 20, 21, 13, 19, 21, 22, 23, 25, 14, 20, 22, 23, 25, 11, 3,
                                9, 10, 13, 21, 4, 5, 6, 15, 20, 22, 24, 25, 14, 15, 19, 23, 25,
                                9, 13, 14, 17, 15, 19, 20, 23, 26, 15, 19, 20, 22, 25, 23, 15,
                                19, 20, 22, 23, 27, 25, 26],
                               [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4,
                                4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 8, 8, 8, 8, 8, 9, 9, 9,
                                9, 9, 9, 10, 10, 10, 11, 11, 12, 12, 13, 13, 13, 13, 13, 13, 13,
                                13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 17, 17, 17,
                                17, 17, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21,
                                21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 24, 25, 25,
                                25, 25, 25, 25, 26, 27],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1], 27, 27)
        @test isapprox(A2, A2_t)
        @test isapprox(tri2, tri2_t)
        @test isapprox(separators2, separators2_t)
        @test isapprox(cliques2, cliques2_t)
        @test isapprox(cliqueTree2, cliqueTree2_t)

        T8, Rpm, Adjv, Dpm, Mv, Z3, dbht = PortfolioOptimisers.DBHTs(dist, S; branchorder = :r,
                                                                     root = root)
        Z3_t = reshape([-3.0, -1.0, -10.0, -25.0, -21.0, -9.0, -8.0, -14.0, -23.0, -26.0, -22.0,
                        -11.0, -30.0, -6.0, -28.0, -2.0, -4.0, -7.0, -5.0, 6.0, 2.0, 17.0, 15.0,
                        4.0, 11.0, 18.0, 13.0, 24.0, 27.0, -29.0, 1.0, -16.0, 3.0, -24.0, -17.0,
                        -15.0, -20.0, 8.0, -27.0, 10.0, -18.0, 12.0, -12.0, 14.0, -19.0, -13.0,
                        9.0, 7.0, 16.0, 5.0, 19.0, 20.0, 21.0, 22.0, 23.0, 25.0, 26.0, 28.0,
                        0.034482758620689655, 0.03571428571428571, 0.037037037037037035,
                        0.038461538461538464, 0.04, 0.041666666666666664, 0.043478260869565216,
                        0.045454545454545456, 0.047619047619047616, 0.05, 0.05263157894736842,
                        0.05555555555555555, 0.058823529411764705, 0.0625, 0.06666666666666667,
                        0.07142857142857142, 0.07692307692307693, 0.08333333333333333,
                        0.09090909090909091, 0.1, 0.1111111111111111, 0.125,
                        0.14285714285714285, 0.16666666666666666, 0.2, 0.25, 0.3333333333333333,
                        0.5, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 3.0, 2.0,
                        3.0, 2.0, 3.0, 2.0, 2.0, 4.0, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 8.0, 11.0,
                        11.0, 19.0, 30.0], :, 4)
        @test isapprox(Z3, Z3_t)
    end
    @testset "LoGo" begin
        X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                      timestamp = :timestamp)
        rd = prices_to_returns(X[(end - 252):end])
        X = rd.X
        ce = PortfolioOptimisersCovariance()
        sigma = cov(ce, X)

        logo_t = CSV.read(joinpath(@__DIR__, "./assets/LoGo-MaximumDistanceSimilarity.csv"),
                          DataFrame)

        des = [Distance(; alg = CanonicalDistance()),
               DistanceDistance(; alg = CanonicalDistance()),
               Distance(; alg = SimpleDistance()), DistanceDistance(; alg = SimpleDistance()),
               Distance(; alg = SimpleAbsoluteDistance()),
               DistanceDistance(; alg = SimpleAbsoluteDistance()),
               Distance(; alg = CorrelationDistance()),
               DistanceDistance(; alg = CorrelationDistance()), Distance(; alg = LogDistance()),
               DistanceDistance(; alg = LogDistance()),
               Distance(; alg = VariationInfoDistance()),
               DistanceDistance(; alg = VariationInfoDistance())]
        desg = [Distance(;power=1, alg = CanonicalDistance()),
                DistanceDistance(;power=1, alg = CanonicalDistance()),
                Distance(;power=1, alg = SimpleDistance()),
                DistanceDistance(;power=1, alg = SimpleDistance()),
                Distance(;power=1, alg = SimpleAbsoluteDistance()),
                DistanceDistance(;power=1, alg = SimpleAbsoluteDistance()),
                Distance(;power=1, alg = CorrelationDistance()),
                DistanceDistance(;power=1, alg = CorrelationDistance()),
                Distance(;power=1, alg = LogDistance()),
                DistanceDistance(;power=1, alg = LogDistance()),
                Distance(;power=1, alg = VariationInfoDistance()),
                DistanceDistance(;power=1, alg = VariationInfoDistance())]
        for i in eachindex(des)
            sigma1 = copy(sigma)
            sigma2 = copy(sigma)
            PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                      dist = des[i]),
                                                             sigma1, X)
            PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                      dist = desg[i]),
                                                             sigma2, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo MaxDist sim iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1

            res2 = isapprox(sigma2, reshape(logo_t[!, i], MN))
            if !res2
                println("Fails on LoGo MaxDist sim iteration $i")
                find_tol(sigma2, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res2
        end

        logo_t = CSV.read(joinpath(@__DIR__, "./assets/LoGo-ExponentialSimilarity.csv"),
                          DataFrame)

        des = [Distance(; alg = CanonicalDistance()),
               DistanceDistance(; alg = CanonicalDistance()),
               Distance(; alg = SimpleDistance()), DistanceDistance(; alg = SimpleDistance()),
               Distance(; alg = SimpleAbsoluteDistance()),
               DistanceDistance(; alg = SimpleAbsoluteDistance()),
               Distance(; alg = CorrelationDistance()),
               DistanceDistance(; alg = CorrelationDistance()), Distance(; alg = LogDistance()),
               DistanceDistance(; alg = LogDistance()),
               Distance(; alg = VariationInfoDistance()),
               DistanceDistance(; alg = VariationInfoDistance())]
        desg = [Distance(;power=1, alg = CanonicalDistance()),
                DistanceDistance(;power=1, alg = CanonicalDistance()),
                Distance(;power=1, alg = SimpleDistance()),
                DistanceDistance(;power=1, alg = SimpleDistance()),
                Distance(;power=1, alg = SimpleAbsoluteDistance()),
                DistanceDistance(;power=1, alg = SimpleAbsoluteDistance()),
                Distance(;power=1, alg = CorrelationDistance()),
                DistanceDistance(;power=1, alg = CorrelationDistance()),
                Distance(;power=1, alg = LogDistance()),
                DistanceDistance(;power=1, alg = LogDistance()),
                Distance(;power=1, alg = VariationInfoDistance()),
                DistanceDistance(;power=1, alg = VariationInfoDistance())]
        for i in eachindex(des)
            sigma1 = copy(sigma)
            sigma2 = copy(sigma)
            PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                      dist = des[i],
                                                                                      sim = ExponentialSimilarity()),
                                                             sigma1, X)
            PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                      dist = desg[i],
                                                                                      sim = ExponentialSimilarity()),
                                                             sigma2, X)
            MN = size(sigma1)
            res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
            if !res1
                println("Fails on LoGo ExpDist sim iteration $i")
                find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res1

            res2 = isapprox(sigma2, reshape(logo_t[!, i], MN))
            if !res2
                println("Fails on LoGo ExpDist sim iteration $i")
                find_tol(sigma2, reshape(logo_t[!, i], MN); name1 = :sigma, name2 = :logo_t)
            end
            @test res2
        end

        @test isnothing(PortfolioOptimisers.logo!(nothing))
    end
    =#
    @testset "Subsetting optimisers reject a precomputed phylogeny" begin
        #=
        A meta-optimiser hands its subproblems a subset of the universe, and a phylogeny
        cannot follow them there. #184 got there through a phylogeny constraint *estimator*
        holding a precomputed result in `pl` -- configuration on the outside, data one level
        down, invisible to every check aimed at results.

        That shape is now unconstructible rather than guarded: `pl` is bounded by `NwE_ClE`,
        which admits only sources, on all three estimators that have the slot. So the checks
        below assert a *type* error at construction, and the runtime guard is left with one
        job -- a precomputed constraint result -- which is what it was originally written
        for, and which no type bound can take over because `ple` legitimately accepts a
        result outside a meta-optimiser.
        =#
        Xp = rd.X
        plr = phylogeny_matrix(NetworkEstimator(), Xp)
        clr = clusterise(ClustersEstimator(), Xp)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false),
                     check_sol = (; allow_local = true, allow_almost = true))
        mk(ple) = MeanRisk(; opt = JuMPOptimiser(; slv = slv, ple = ple))

        # The root cause: no estimator accepts precomputed structure in `pl` any more.
        # A `Clusters` is rejected for the same reason as a `PhylogenyResult` -- both answer
        # for the universe they were built on, and the slot asks how to build for any.
        # The bound rejects at construction, so there is no runtime check to reach.
        @test_throws TypeError SemiDefinitePhylogenyEstimator(; pl = plr)
        @test_throws TypeError SemiDefinitePhylogenyEstimator(; pl = clr)
        @test_throws TypeError IntegerPhylogenyEstimator(; pl = plr)
        @test_throws TypeError IntegerPhylogenyEstimator(; pl = clr)
        @test_throws TypeError CentralityEstimator(; pl = plr)
        @test_throws TypeError CentralityEstimator(; pl = clr)
        @test isa(CentralityEstimator(; pl = NetworkEstimator()), CentralityEstimator)
        # Sources still construct, by keyword and positionally.
        @test isa(SemiDefinitePhylogenyEstimator(; pl = NetworkEstimator()),
                  SemiDefinitePhylogenyEstimator)
        @test isa(SemiDefinitePhylogenyEstimator(ClustersEstimator(), 0.05),
                  SemiDefinitePhylogenyEstimator)
        @test isa(IntegerPhylogenyEstimator(; pl = ClustersEstimator()),
                  IntegerPhylogenyEstimator)
        #=
        Known and NOT specific to these two types: `@concrete` emits a fully generic
        positional constructor (`ConcreteStructs.jl:146`) that is more specific than
        nothing and shadows the annotated one, so every type bound in the package is
        bypassable positionally. `PhylogenyFeatures(vector_shaped_result, alg)` does the
        same thing and predates this. Pinned here so the limit of the guarantee is on the
        record: the keyword constructor is the enforced API.
        =#
        @test isa(SemiDefinitePhylogenyEstimator(plr, 0.05), SemiDefinitePhylogenyEstimator)

        # What the runtime guard is still for: a precomputed constraint *result* in `ple`.
        # It was already rejected when held directly...
        plres = phylogeny_constraints(SemiDefinitePhylogenyEstimator(), Xp)
        @test_throws ArgumentError PortfolioOptimisers.assert_external_optimiser(mk(plres))
        @test_throws ArgumentError PortfolioOptimisers.assert_internal_optimiser(mk(plres))
        # ...but not when held in a vector: the branch meant to catch that was unreachable,
        # because `||` binds looser than `&&`, so a vector satisfied the first clause and
        # short-circuited to pass -- in exactly the case the branch was written for.
        @test_throws ArgumentError PortfolioOptimisers.assert_external_optimiser(mk([plres]))
        @test_throws ArgumentError PortfolioOptimisers.assert_internal_optimiser(mk([plres]))

        # An estimator that refits per subproblem is the supported form and still passes.
        @test isnothing(PortfolioOptimisers.assert_external_optimiser(mk(SemiDefinitePhylogenyEstimator())))
        @test isnothing(PortfolioOptimisers.assert_external_optimiser(mk(nothing)))
        @test isnothing(PortfolioOptimisers.assert_external_optimiser(mk([SemiDefinitePhylogenyEstimator()])))
    end
    @testset "A zero distance is repaired, not deleted" begin
        #=
        A distance matrix and a weighted graph disagree about `0`: the distance codomain
        means *closest*, `SimpleWeightedGraph` reserves it for *absent* and sparsifies it
        away. Left alone, the constructor deletes exactly the cheapest edge -- the one the
        MST most wants -- and the most related pair in the universe comes out non-adjacent
        with nothing raised. `graph_weight_matrix` moves each zero to the smallest
        representable positive value instead.
        =#
        Dg = PortfolioOptimisers.distance(Distance(), PortfolioOptimisersCovariance(), pr.X)
        tiny = nextfloat(zero(eltype(Dg)))

        # Nothing to repair: `D` comes back untouched and uncopied, so the copy is only
        # paid for when it buys something.
        @test PortfolioOptimisers.graph_weight_matrix(Dg) === Dg

        Dz = copy(Dg)
        Dz[1, 2] = Dz[2, 1] = zero(eltype(Dz))
        Wz = PortfolioOptimisers.graph_weight_matrix(Dz)
        @test Wz !== Dz                       # repaired on a copy...
        @test Dz[1, 2] === zero(eltype(Dz))   # ...leaving the caller's matrix exact,
        @test Wz[1, 2] == tiny                # which `clusterise`'s `P` depends on.
        @test Wz[2, 1] == tiny
        # Only the zeros move.
        @test all(Wz[i, j] == Dz[i, j]
                  for i in axes(Dz, 1), j in axes(Dz, 2) if !iszero(Dz[i, j]))
        # `-0.0` is a zero too.
        Dm = copy(Dg)
        Dm[1, 2] = Dm[2, 1] = -zero(eltype(Dm))
        @test PortfolioOptimisers.graph_weight_matrix(Dm)[1, 2] == tiny

        #=
        Negative and NaN have no nearest representable value and are *unsound* rather than
        merely wrong downstream: Dijkstra returns an answer on a negative edge instead of
        raising, and NaN silently fails every comparison the tree algorithms make. A NaN
        arrives from ordinary bad data -- a zero-variance asset gives a NaN correlation.
        =#
        Dn = copy(Dg)
        Dn[1, 2] = -eps(eltype(Dn))
        @test_throws DomainError PortfolioOptimisers.graph_weight_matrix(Dn)
        Dnan = copy(Dg)
        Dnan[2, 3] = NaN
        @test_throws DomainError PortfolioOptimisers.graph_weight_matrix(Dnan)
        # Inf is allowed: it is the honest LogDistance between uncorrelated assets, the
        # graph accepts it, and a spanning tree simply takes those edges last.
        Di = copy(Dg)
        Di[1, 2] = Di[2, 1] = Inf
        @test PortfolioOptimisers.graph_weight_matrix(Di) === Di
        # The diagonal is exempt -- it is zero by construction.
        @test all(iszero, LinearAlgebra.diag(PortfolioOptimisers.graph_weight_matrix(Dz)))

        #=
        The end-to-end failure, with `clusterise` as the oracle. `SimpleAbsoluteDistance`
        is defined on `abs(rho)`, so an exactly anti-correlated pair -- a long/short leg,
        an inverse ETF -- sits at distance zero and is genuinely maximally related. Before
        the repair the two consumers of one estimator contradicted each other about that
        pair: `clusterise` reads `D` directly and put them in the same cluster, while
        `phylogeny_matrix` reads only the sparsified graph and declared them unrelated --
        so the phylogeny constraints left the one pair they exist to separate unbounded.
        =#
        Xa = randn(StableRNG(11), 300, 12)
        Xa[:, 12] .= -Xa[:, 1]
        ntea = NetworkEstimator(; ce = Covariance(),
                                de = Distance(; alg = SimpleAbsoluteDistance()),
                                alg = KruskalTree())
        Da = PortfolioOptimisers.distance(ntea.de, ntea.ce, Xa)
        @test Da[1, 12] == zero(eltype(Da))          # the pair really is at the floor,
        @test Da[1, 12] == minimum(Da)               # and nothing is closer.
        pma = phylogeny_matrix(ntea, Xa)
        cla = clusterise(NetworkClustersEstimator(; nte = ntea), Xa)
        asg = Clustering.cutree(cla.res; k = cla.k)
        @test size(pma.X) == (12, 12)                # the universe survives...
        @test isone(pma.X[1, 12])                    # ...and the pair is adjacent,
        @test asg[1] == asg[12]                      # which is what `clusterise` says too.
        # The repair is a single-edge swap: 1--12 enters, and the 3--12 edge that stood in
        # for it while 1--12 was deleted is gone. Every other edge is untouched.
        @test iszero(pma.X[3, 12])
        @test count(isone, pma.X) == 2 * (size(Xa, 2) - 1)
    end
    @testset "The weighted adjacency tiers" begin
        #=
        `calc_adjacency` no longer has a body of its own: the structure is built once by
        `calc_weighted_adjacency_graph`, and the other two tiers are one operation each.
        These are regressions of a fact already measured -- the refactor is subtractive, so
        the binary matrix must come out of the new chain unchanged in value *and* in type.
        The oracles below are the two branch bodies `calc_adjacency` used to carry.
        =#
        Gr = PortfolioOptimisers.Graphs
        SWG = PortfolioOptimisers.SimpleWeightedGraphs
        function tree_oracle(nte, X)
            D = PortfolioOptimisers.distance(nte.de, nte.ce, X; dims = 1)
            G = SWG.SimpleWeightedGraph(PortfolioOptimisers.graph_weight_matrix(D))
            tree = PortfolioOptimisers.calc_mst(nte.alg, G)
            return Gr.adjacency_matrix(Gr.SimpleGraph(G[tree]))
        end
        function pmfg_similarity(nte, X)
            S, D = cor_and_dist(nte.de, nte.ce, X; dims = 1)
            return PortfolioOptimisers.distance_to_similarity(nte.alg; S = S, D = D)
        end
        function pmfg_oracle(nte, X)
            Rpm = PortfolioOptimisers.PMFG_T2s(pmfg_similarity(nte, X))[1]
            return Gr.adjacency_matrix(Gr.SimpleGraph(Rpm))
        end
        # `AngularSimilarity` is absent because `PMFG_T2s` rejects its negative entries on
        # ordinary data -- pre-existing, and issue #239's.
        tree_algs = (KruskalTree(), BoruvkaTree(), PrimTree())
        pmfg_algs = (MaximumDistanceSimilarity(), ExponentialSimilarity(),
                     GeneralExponentialSimilarity(), ComplementSimilarity())
        for (algs, oracle) in ((tree_algs, tree_oracle), (pmfg_algs, pmfg_oracle))
            for alg in algs
                nte = NetworkEstimator(; alg = alg)
                A = PortfolioOptimisers.calc_adjacency(nte, pr.X)
                O = oracle(nte, pr.X)
                @test A == O
                @test typeof(A) === typeof(O)
                @test A isa SparseMatrixCSC{Int, Int}
                # Same structure, different values: the weighted tier keeps the sparsity
                # pattern exactly and only stops discarding the numbers. `adjacency_matrix`
                # of a *weighted* graph returns the weights, not 0/1.
                W = PortfolioOptimisers.calc_weighted_adjacency(nte, pr.X)
                @test W.colptr == A.colptr
                @test W.rowval == A.rowval
                @test eltype(W) === eltype(pr.X)
                @test !all(isone, W.nzval)
            end
        end
        # Per-branch polarity, recovered by dispatch on `nte.alg` and carried by no tag: the
        # tree weights are the distances `calc_mst` minimised, strictly positive after the
        # repair; the PMFG weights are the similarities `PMFG_T2s` maximised.
        ntet = NetworkEstimator(; alg = KruskalTree())
        Dt = PortfolioOptimisers.distance(ntet.de, ntet.ce, pr.X)
        Wt = PortfolioOptimisers.calc_weighted_adjacency(ntet, pr.X)
        @test all(>(zero(eltype(Wt))), Wt.nzval)
        @test all(Wt[i, j] == Dt[i, j] for (i, j) in zip(findnz(Wt)[1], findnz(Wt)[2]))
        # The PMFG's weighted adjacency *is* `PMFG_T2s(S)[1]` verbatim -- the graph round
        # trip is structurally the identity, because `PMFG_T2s` emits no stored zero.
        for alg in pmfg_algs
            nte = NetworkEstimator(; alg = alg)
            Rpm = PortfolioOptimisers.PMFG_T2s(pmfg_similarity(nte, pr.X))[1]
            W = PortfolioOptimisers.calc_weighted_adjacency(nte, pr.X)
            @test count(iszero, Rpm.nzval) == 0
            @test W.colptr == Rpm.colptr
            @test W.rowval == Rpm.rowval
            @test W.nzval == Rpm.nzval
        end
        #=
        The one place the removed `SimpleGraph` round trip could have diverged: an
        explicitly stored zero. `adjacency_matrix` is `T.(copy(weights))` and the broadcast
        drops a numerical zero, so a zero-weight edge survives the graph tier but not the
        matrix tier. Injected by hand, since `PMFG_T2s` never produces one -- and both the
        old path and the new one drop the edge, keeping every vertex.
        =#
        ntep = NetworkEstimator(; alg = ComplementSimilarity())
        Rz = copy(PortfolioOptimisers.PMFG_T2s(pmfg_similarity(ntep, pr.X))[1])
        Rz[1, 2] = Rz[2, 1] = zero(eltype(Rz))
        @test count(iszero, Rz.nzval) == 2          # the zeros really are stored,
        old_path = Gr.adjacency_matrix(Gr.SimpleGraph(Rz))
        new_path = Gr.adjacency_matrix(Gr.SimpleGraph(SWG.SimpleWeightedGraph(Rz)))
        @test old_path == new_path
        @test iszero(old_path[1, 2])                # ...and both paths drop the edge,
        @test nnz(old_path) == nnz(Rz) - 2
        @test Gr.nv(SWG.SimpleWeightedGraph(Rz)) == size(pr.X, 2)  # losing no vertex.
    end
end
