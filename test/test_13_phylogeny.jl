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
    @testset "Non-negative similarity interface (#239)" begin
        nonneg = (MaximumDistanceSimilarity(), ExponentialSimilarity(),
                  GeneralExponentialSimilarity(), ComplementSimilarity())
        # Four of the five members are in the interface; `AngularSimilarity` is out, and
        # stays in the wider family it was always in.
        for sim in nonneg
            @test sim isa AbstractNonNegativeSimilarityMatrixAlgorithm
            @test sim isa AbstractSimilarityMatrixAlgorithm
            @test sim isa PortfolioOptimisers.Tree_SimMat
        end
        @test !isa(AngularSimilarity(), AbstractNonNegativeSimilarityMatrixAlgorithm)
        @test AngularSimilarity() isa AbstractSimilarityMatrixAlgorithm
        @test !isa(AngularSimilarity(), PortfolioOptimisers.Tree_SimMat)
        @test AbstractNonNegativeSimilarityMatrixAlgorithm <:
              AbstractSimilarityMatrixAlgorithm
        # Both abstract types are exported, so an extension can subtype the interface
        # without reaching through the module prefix.
        for T in (:AbstractSimilarityMatrixAlgorithm,
                  :AbstractNonNegativeSimilarityMatrixAlgorithm)
            @test T in names(PortfolioOptimisers)
        end
        #=
        Issue #239's own reproduction. It reported `NetworkEstimator`, but the blast radius
        is three estimators: `DBHT` and `LoGo` reach `PMFG_T2s` by their own routes. All
        three now fail at *construction*, naming the caller's keyword, rather than inside
        `PMFG_T2s` naming `W`.
        =#
        @test_throws TypeError NetworkEstimator(; alg = AngularSimilarity())
        @test_throws TypeError DBHT(; sim = AngularSimilarity())
        @test_throws TypeError LoGo(; sim = AngularSimilarity())
        #=
        The *positional* constructors refuse too. That needs saying, because it does not
        come for free: `@concrete` generates its own constructor per struct, and a field
        written bare gets an unbounded type parameter, so the generated method is the only
        one matching a refused argument and it accepts. Declaring the bound on the field
        itself -- `sim <: AbstractNonNegativeSimilarityMatrixAlgorithm` -- puts it on the
        generated type parameter, so there is no method left to match and the refusal is a
        `MethodError`. Every other field on these three structs is still bare.
        =#
        @test_throws MethodError DBHT(AngularSimilarity(), UniqueRoot())
        @test_throws MethodError LoGo(Distance(), AngularSimilarity(), Posdef())
        @test_throws MethodError NetworkEstimator(PortfolioOptimisersCovariance(),
                                                  Distance(), AngularSimilarity(),
                                                  HopCount())
        # The admitted members build by either route.
        @test DBHT(MaximumDistanceSimilarity(), UniqueRoot()) isa DBHT
        @test LoGo(Distance(), ComplementSimilarity(), Posdef()) isa LoGo
        @test NetworkEstimator(PortfolioOptimisersCovariance(), Distance(), KruskalTree(),
                               HopCount()) isa NetworkEstimator
        # Every surviving member still constructs on all three.
        for sim in nonneg
            @test NetworkEstimator(; alg = sim).alg === sim
            @test DBHT(; sim = sim).sim === sim
            @test LoGo(; sim = sim).sim === sim
        end
        # `FeatureDistance.sim` is untouched: its similarity never reaches `PMFG_T2s`,
        # because every PMFG entry point recomputes one from its own algorithm.
        @test FeatureDistance(; sim = AngularSimilarity()).sim === AngularSimilarity()
        @testset "The domain precondition" begin
            de = Distance()
            Dok = [0.0 0.5 0.25; 0.5 0.0 0.75; 0.25 0.75 0.0]
            Dbig = [0.0 7.0 0.25; 7.0 0.0 0.75; 0.25 0.75 0.0]
            Dinf = [0.0 Inf 0.25; Inf 0.0 0.75; 0.25 0.75 0.0]
            # A bounded, finite `D` is in every member's domain.
            for sim in nonneg
                @test isnothing(PortfolioOptimisers.assert_similarity_domain(sim, de, Dok))
            end
            # The two members that declare nothing take the no-op fallback, so an
            # unbounded or infinite `D` is theirs to transform. `exp(-Inf)` is 0 exactly.
            for sim in (ExponentialSimilarity(), GeneralExponentialSimilarity())
                @test isnothing(PortfolioOptimisers.assert_similarity_domain(sim, de, Dbig))
                @test isnothing(PortfolioOptimisers.assert_similarity_domain(sim, de, Dinf))
            end
            # `ComplementSimilarity` declares `D <= 1`, `MaximumDistanceSimilarity`
            # declares finiteness, and `D <= 1` implies finiteness.
            @test_throws DomainError PortfolioOptimisers.assert_similarity_domain(ComplementSimilarity(),
                                                                                  de, Dbig)
            @test_throws DomainError PortfolioOptimisers.assert_similarity_domain(ComplementSimilarity(),
                                                                                  de, Dinf)
            @test isnothing(PortfolioOptimisers.assert_similarity_domain(MaximumDistanceSimilarity(),
                                                                         de, Dbig))
            @test_throws DomainError PortfolioOptimisers.assert_similarity_domain(MaximumDistanceSimilarity(),
                                                                                  de, Dinf)
            # The message names *both* halves -- the distance estimator that produced the
            # offending value, and the similarity that refused it. That is the whole point
            # of the check: `PMFG_T2s` could only ever name `W`.
            msg = try
                PortfolioOptimisers.assert_similarity_domain(ComplementSimilarity(),
                                                             Distance(;
                                                                      alg = LogDistance()),
                                                             Dbig)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("ComplementSimilarity", msg)
            @test occursin("LogDistance", msg)
            @test occursin("7.0", msg)
            msg = try
                PortfolioOptimisers.assert_similarity_domain(MaximumDistanceSimilarity(),
                                                             DistanceDistance(), Dinf)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("MaximumDistanceSimilarity", msg)
            @test occursin("DistanceDistance", msg)
            # The check is *interface-scoped*, not member-wide: `distance_to_similarity`
            # stays a pure transformation with no domain of its own, so the shipped
            # `FeatureDistance` promise that every `SemiMetric` yields a similarity and
            # nothing throws is untouched.
            @test isapprox(PortfolioOptimisers.distance_to_similarity(ComplementSimilarity();
                                                                      D = Dbig),
                           one(eltype(Dbig)) .- Dbig)
        end
        @testset "The precondition at the PMFG entry points" begin
            #=
            The oracle is that every pairing the precondition refuses already throws today,
            only with a worse message. `LogDistance` and `DistanceDistance` both exceed 1;
            `SimpleDistance` does not. `LogDistance` with `ExponentialSimilarity` works and
            must keep working -- a blanket finiteness rule on the branch would break it.
            =#
            for (de, sim, throws) in
                ((Distance(; alg = SimpleDistance()), ComplementSimilarity(), false),
                 (Distance(; alg = LogDistance()), ExponentialSimilarity(), false),
                 (Distance(; alg = LogDistance()), ComplementSimilarity(), true),
                 (DistanceDistance(), ComplementSimilarity(), true))
                nte = NetworkEstimator(; de = de, alg = sim)
                cle = ClustersEstimator(; de = de, alg = DBHT(; sim = sim))
                je = LoGo(; de = de, sim = sim)
                sigma = cov(PortfolioOptimisersCovariance(), pr.X)
                if throws
                    @test_throws DomainError PortfolioOptimisers.calc_adjacency(nte, pr.X)
                    @test_throws DomainError PortfolioOptimisers.calc_distance_weighted_graph(nte,
                                                                                              pr.X)
                    @test_throws DomainError clusterise(cle, pr.X)
                    @test_throws DomainError PortfolioOptimisers.logo!(je, copy(sigma),
                                                                       pr.X)
                    @test_throws DomainError clusterise(NetworkClustersEstimator(;
                                                                                 nte = nte),
                                                        pr.X)
                else
                    @test PortfolioOptimisers.calc_adjacency(nte, pr.X) isa
                          SparseMatrixCSC{Int, Int}
                    @test clusterise(cle, pr.X) isa PortfolioOptimisers.Clusters
                    @test isnothing(PortfolioOptimisers.logo!(je, copy(sigma), pr.X))
                    @test clusterise(NetworkClustersEstimator(; nte = nte), pr.X) isa
                          PortfolioOptimisers.Clusters
                end
            end
            #=
            The one live route to a non-finite distance: #237 established that `Denoise()`
            makes exactly zero correlations, which `LogDistance` correctly maps to `Inf`.
            `MaximumDistanceSimilarity` is the default of both `DBHT` and `LoGo`, so this
            reached `PMFG_T2s` as a matrix of `NaN`s and was reported as a *negative*
            weight. It is now named as what it is, at the configuration that caused it.
            =#
            ce = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; dn = Denoise()))
            de = Distance(; alg = LogDistance())
            # `pr.X` is too correlated to denoise to an exact zero, so this needs noise.
            Xn = randn(StableRNG(987654321), 500, 20)
            @test any(isinf, PortfolioOptimisers.distance(de, ce, Xn))
            nte = NetworkEstimator(; ce = ce, de = de, alg = MaximumDistanceSimilarity())
            @test_throws DomainError PortfolioOptimisers.calc_adjacency(nte, Xn)
            # `ExponentialSimilarity` declares no domain and is right not to: `exp(-Inf)`
            # is `0` exactly, so an infinite distance is a legal zero similarity.
            nte = NetworkEstimator(; ce = ce, de = de, alg = ExponentialSimilarity())
            @test PortfolioOptimisers.calc_adjacency(nte, Xn) isa SparseMatrixCSC{Int, Int}
        end
        @testset "`PMFG_T2s`' backstop" begin
            #=
            Kept, because the interface is open by *declaration*: an extension can subtype
            it and return a negative anyway, and the DBHT failure below is silent. The two
            checks are split because `0 <= NaN` is `false`, so one check reported a `NaN` as
            a negative weight and sent the caller looking for the wrong thing.
            =#
            W = ones(9, 9)
            @test PortfolioOptimisers.PMFG_T2s(W)[1] isa SparseMatrixCSC
            Wn = copy(W)
            Wn[1, 2] = Wn[2, 1] = NaN
            msg = try
                PortfolioOptimisers.PMFG_T2s(Wn)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("isnan", msg)
            @test !occursin(">= 0", msg)
            Wm = copy(W)
            Wm[1, 2] = Wm[2, 1] = -1.0
            msg = try
                PortfolioOptimisers.PMFG_T2s(Wm)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin(">= 0", msg)
            @test occursin("-1.0", msg)
        end
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
    @testset "The radius ball selects; the hop ball is untouched" begin
        NAS = size(pr.X, 2)
        NPAIRS = (NAS * (NAS - 1)) ÷ 2
        algs = (KruskalTree(), BoruvkaTree(), PrimTree(), MaximumDistanceSimilarity(),
                ExponentialSimilarity(), GeneralExponentialSimilarity(),
                ComplementSimilarity())
        # The hop branch moved behind `_phylogeny_matrix`, so the honest regression is its
        # own body reproduced verbatim as an oracle -- values *and* type.
        function hop_oracle(nte, X)
            A = PortfolioOptimisers.calc_adjacency(nte, X; dims = 1)
            P = zeros(Int, size(A))
            for i in 0:(nte.sep.n)
                P .+= A^i
            end
            return clamp!(P, 0, 1) - I
        end
        for alg in algs, n in 1:5
            nte = NetworkEstimator(; alg = alg, sep = HopCount(; n = n))
            A = phylogeny_matrix(nte, pr.X).X
            want = hop_oracle(nte, pr.X)
            @test A == want
            @test typeof(A) === typeof(want)
        end

        # The radius ball is `Int`-valued. #204 decided values do not widen, only selection
        # does, so this is the decision itself and not an implementation detail.
        for alg in (KruskalTree(), MaximumDistanceSimilarity()), dmax in (1.0, 1.5, 2.0)
            nte = NetworkEstimator(; alg = alg, sep = PathLength(; dmax = dmax))
            A = phylogeny_matrix(nte, pr.X).X
            @test isa(A, Matrix{Int})
            @test issymmetric(A)
            @test all(iszero, diag(A))
            # It *is* the thresholded separation matrix, reusing the one traversal rather
            # than a second all-pairs routine of its own.
            d = separation_matrix(nte.sep, nte, pr.X; dims = 1)
            @test A == Int.(d .<= separation_budget(nte.sep, nte, d)) - I
        end

        # A larger budget relates a superset. The radius knob is a dial, not a reshuffle.
        for alg in (KruskalTree(), MaximumDistanceSimilarity())
            prev = nothing
            for dmax in (0.75, 1.0, 1.5, 2.0, 2.5)
                A = phylogeny_matrix(NetworkEstimator(; alg = alg,
                                                      sep = PathLength(; dmax = dmax)),
                                     pr.X).X
                if !isnothing(prev)
                    @test all(prev .<= A)
                end
                prev = A
            end
        end

        # `PathLength()` bare resolves to the observed diameter, so it relates *every*
        # reachable pair -- the opposite end of the dial from `HopCount()`'s default `n = 1`.
        # Documented rather than guarded: it is the honest reading of an unstated budget.
        for alg in (KruskalTree(), MaximumDistanceSimilarity())
            A = phylogeny_matrix(NetworkEstimator(; alg = alg, sep = PathLength()), pr.X).X
            @test count(isone, A) == 2 * NPAIRS
            @test all(iszero, diag(A))
        end

        # The gain is intermediate cardinalities *between* the hop shells, not a different
        # neighbourhood. The hop knob cannot express a count the radius knob reaches.
        npmfg = NetworkEstimator(; alg = MaximumDistanceSimilarity(),
                                 sep = HopCount(; n = 1))
        shells = [count(isone,
                        phylogeny_matrix(NetworkEstimator(;
                                                          alg = MaximumDistanceSimilarity(),
                                                          sep = HopCount(; n = n)), pr.X).X) ÷
                  2 for n in 1:2]
        @test shells == [54, 121]
        between = count(isone,
                        phylogeny_matrix(NetworkEstimator(;
                                                          alg = MaximumDistanceSimilarity(),
                                                          sep = PathLength(; dmax = 0.9768)),
                                         pr.X).X) ÷ 2
        @test shells[1] < between < shells[2]

        # And it does not re-rank: a hop shell is the equal-cardinality prefix of the
        # path-length ordering. Exactly so on the PMFG, near enough on the tree.
        dh = separation_matrix(HopCount(), npmfg, pr.X; dims = 1)
        dp = separation_matrix(PathLength(), npmfg, pr.X; dims = 1)
        pairs = [(i, k) for i in 1:NAS for k in (i + 1):NAS]
        ord = sortperm([dp[i, k] for (i, k) in pairs])
        for n in 1:4
            shell = Set(p for p in pairs if dh[p...] <= n)
            @test shell == Set(pairs[ord[1:length(shell)]])
        end

        # The radius ball is reachable from both constraint families for free -- they call
        # `phylogeny_matrix(plc.pl, X)` and dispatch does the rest. That reachability is the
        # entire reason the radius ball exists.
        for alg in (KruskalTree(), MaximumDistanceSimilarity())
            nte = NetworkEstimator(; alg = alg, sep = PathLength(; dmax = 1.5))
            sdp = phylogeny_constraints(SemiDefinitePhylogenyEstimator(; pl = nte), pr.X)
            ip = phylogeny_constraints(IntegerPhylogenyEstimator(; pl = nte), pr.X)
            @test isa(sdp, SemiDefinitePhylogeny)
            @test isa(ip, IntegerPhylogeny)
            @test eltype(sdp.A) === Int
            @test eltype(ip.A) === Int
            @test sdp.A == phylogeny_matrix(nte, pr.X).X
        end

        # `clusterise` refuses a `PathLength` at *dispatch*. Its `D^i - A^i` is a power sum
        # indexed by hops, and a radius has no analogue of a matrix power -- so the fourth
        # type parameter is narrowed rather than the field access left to fail.
        for alg in (KruskalTree(), MaximumDistanceSimilarity())
            @test_throws MethodError clusterise(NetworkClustersEstimator(;
                                                                         nte = NetworkEstimator(;
                                                                                                alg = alg,
                                                                                                sep = PathLength())),
                                                pr.X)
            # The hop path is unaffected.
            @test isa(clusterise(NetworkClustersEstimator(;
                                                          nte = NetworkEstimator(;
                                                                                 alg = alg,
                                                                                 sep = HopCount(;
                                                                                                n = 2))),
                                 pr.X), PortfolioOptimisers.Clusters)
        end
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
        # `AngularSimilarity` is absent because `NetworkEstimator.alg` now refuses it at
        # construction: it is outside `AbstractNonNegativeSimilarityMatrixAlgorithm`, and
        # `PMFG_T2s` cannot take its negative entries. Issue #239.
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
    @testset "clusterise reads the shared adjacency routine" begin
        #=
        Both `clusterise` methods used to re-derive the structure inline -- the tree one
        rebuilt the graph, the tree and the adjacency, the PMFG one called `PMFG_T2s`
        itself. They now enter `calc_weighted_adjacency` at its two-argument form, which
        takes the *selecting quantity*: the distance on the tree branch, the similarity on
        the PMFG branch. `clusterise` has already paid for that matrix -- it needs it for
        its own power sum -- so the two-argument form is what keeps the fold free.

        The oracles below are the two inline bodies, reproduced verbatim. `P` is the
        quantity that changed hands, so `P` is what is compared, and bit-for-bit rather
        than approximately: the fold is meant to be a substitution, not an improvement.
        =#
        Gr = PortfolioOptimisers.Graphs
        SWG = PortfolioOptimisers.SimpleWeightedGraphs
        function tree_P_oracle(nte, X, n)
            _, D = cor_and_dist(nte.de, nte.ce, X; dims = 1)
            P = zeros(eltype(D), size(D))
            G = SWG.SimpleWeightedGraph(PortfolioOptimisers.graph_weight_matrix(D))
            A = Gr.adjacency_matrix(G[PortfolioOptimisers.calc_mst(nte.alg, G)])
            for i in 0:n
                P .+= D^i - A^i
            end
            P .-= Diagonal(P)
            return P, A
        end
        function pmfg_P_oracle(nte, X, n)
            S, D = cor_and_dist(nte.de, nte.ce, X; dims = 1)
            P = zeros(eltype(D), size(D))
            S = PortfolioOptimisers.distance_to_similarity(nte.alg; S = S, D = D)
            Rpm = PortfolioOptimisers.PMFG_T2s(S)[1]
            for i in 0:n
                P .+= S^i - Rpm^i
            end
            P .-= Diagonal(P)
            return P, Rpm
        end
        # `AngularSimilarity` is absent for the same reason as above: issue #239.
        tree_algs = (KruskalTree(), BoruvkaTree(), PrimTree())
        pmfg_algs = (MaximumDistanceSimilarity(), ExponentialSimilarity(),
                     GeneralExponentialSimilarity(), ComplementSimilarity())
        for (algs, oracle) in ((tree_algs, tree_P_oracle), (pmfg_algs, pmfg_P_oracle))
            for alg in algs, n in 1:4
                nte = NetworkEstimator(; alg = alg, sep = HopCount(; n = n))
                Po, Ao = oracle(nte, pr.X, n)
                clr = clusterise(NetworkClustersEstimator(; nte = nte), pr.X)
                @test clr.P == Symmetric(Po)
                @test typeof(clr.P) === typeof(Symmetric(Po))
                # The substitution's premise, stated separately from its consequence: the
                # matrix the shared routine returns *is* the one the inline body built,
                # values and type, on both branches. `W` is the selecting quantity, which
                # is where the two branches differ.
                Sw, Dw = cor_and_dist(nte.de, nte.ce, pr.X; dims = 1)
                W = if alg isa PortfolioOptimisers.AbstractTreeType
                    Dw
                else
                    PortfolioOptimisers.distance_to_similarity(alg; S = Sw, D = Dw)
                end
                An = PortfolioOptimisers.calc_weighted_adjacency(alg, W)
                @test An == Ao
                @test typeof(An) === typeof(Ao)
                # The middle tier's two entry points differ only in who derives `W`.
                @test An == PortfolioOptimisers.calc_weighted_adjacency(nte, pr.X)
            end
        end
    end
end

using PortfolioOptimisers, Test, SparseArrays, LinearAlgebra

# A network estimator handing the kernels a *weighted* graph chosen by the test. Every
# estimator `calc_adjacency` can build is connected -- a spanning tree or a PMFG -- so this
# is the only way to drive the unreachable branch. It is an `AbstractNetworkEstimator` and
# not a `NetworkEstimator`, which is also what makes it evidence that `phylogeny_matrix`
# splits on the *separation* rather than on the estimator's own type. Defined at top level
# because a `@testset` body becomes a function, which cannot host a struct.
struct FixedDistanceGraph{T} <: PortfolioOptimisers.AbstractNetworkEstimator
    W::Matrix{Float64}
    sep::T
end
function PortfolioOptimisers.calc_distance_weighted_graph(pl::FixedDistanceGraph, X;
                                                          kwargs...)
    return PortfolioOptimisers.SimpleWeightedGraphs.SimpleWeightedGraph(pl.W)
end
function PortfolioOptimisers.calc_adjacency(pl::FixedDistanceGraph, X; kwargs...)
    return SparseArrays.sparse(Int.(pl.W .!= 0))
end

@testset "An unreachable pair is outside every budget" begin
    # Two disjoint edges, 1-2 and 3-4. `floyd_warshall_shortest_paths` reports `Inf` across
    # the components, and the budget comparison rejects it without a repair.
    W = [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
    X4 = zeros(Float64, 3, 4)
    want = [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]

    d = separation_matrix(PathLength(), FixedDistanceGraph(W, PathLength()), X4)
    @test d[1, 3] == Inf
    @test separation_budget(PathLength(), FixedDistanceGraph(W, PathLength()), d) == 1.0

    @test phylogeny_matrix(FixedDistanceGraph(W, PathLength()), X4).X == want
    # The clamp to the observed diameter excludes the sentinel, so even a budget far above
    # the diameter cannot reach across a component.
    @test phylogeny_matrix(FixedDistanceGraph(W, PathLength(; dmax = 100.0)), X4).X == want
    # A hop count agrees on this graph, which is what makes the comparison meaningful: the
    # two separations differ on the budget, not on reachability.
    @test phylogeny_matrix(FixedDistanceGraph(W, HopCount(; n = 3)), X4).X == want
end

# A distance estimator that counts how often the correlation is derived from `X`. It only
# forwards; the counters are the whole point. Defined at top level because a `@testset`
# body becomes a function, which cannot host a struct.
mutable struct CountingDistance{T} <: PortfolioOptimisers.AbstractDistanceEstimator
    const de::T
    n_cor_and_dist::Int
    n_distance::Int
end
CountingDistance(de) = CountingDistance(de, 0, 0)
function PortfolioOptimisers.cor_and_dist(de::CountingDistance, ce, X; kwargs...)
    de.n_cor_and_dist += 1
    return PortfolioOptimisers.cor_and_dist(de.de, ce, X; kwargs...)
end
function PortfolioOptimisers.distance(de::CountingDistance, ce, X; kwargs...)
    de.n_distance += 1
    return PortfolioOptimisers.distance(de.de, ce, X; kwargs...)
end

@testset "clusterise derives the correlation once" begin
    #=
    Why `calc_weighted_adjacency` has a two-argument form at all. `clusterise` holds the
    selecting quantity already -- it needs `D` and `S` for its own power sum and for the
    `Clusters` it returns -- so entering the shared routine at its `(nte, X)` form would
    derive the same correlation a second time. That is not a rounding error: under
    `VariationInfoDistance` the derivation is `98%` of `clusterise`'s runtime, so the
    second one would almost double it. This test is the guard on that, and it fails for
    the naive substitution rather than merely running slower.
    =#
    Xc = randn(StableRNG(987654321), 200, 10)
    for alg in (KruskalTree(), ComplementSimilarity())
        de = CountingDistance(Distance(; alg = CanonicalDistance()))
        nte = NetworkEstimator(; de = de, alg = alg)
        clusterise(NetworkClustersEstimator(; nte = nte), Xc)
        @test de.n_cor_and_dist == 1
        @test de.n_distance == 0
        # The naive substitution, for contrast: it re-enters the derivation on its own.
        PortfolioOptimisers.calc_weighted_adjacency(nte, Xc)
        @test de.n_cor_and_dist + de.n_distance == 2
    end
end

#=
Weighted centrality (#205). `centrality_polarity` declares which quantity an algorithm's
weights must be, and `centrality_graph` supplies it. Nothing in the path raises: an
algorithm that declares no polarity, and a source that carries no weights, run on the plain
graph instead.
=#
using PortfolioOptimisers, Test

# An algorithm that says nothing about itself, for the fallback. Defined at top level
# because a `@testset` body becomes a function, which cannot host a struct.
struct UndeclaredCentrality <: PortfolioOptimisers.AbstractCentralityAlgorithm end

@testset "Weighted centrality" begin
    using PortfolioOptimisers, Test, CSV, DataFrames, TimeSeries, StatsBase, SparseArrays,
          LinearAlgebra
    PO = PortfolioOptimisers
    G = PortfolioOptimisers.Graphs
    SWG = PortfolioOptimisers.SimpleWeightedGraphs

    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    X = prior(EmpiricalPrior(), rd).X
    nte_t = NetworkEstimator()                                       # tree branch
    nte_p = NetworkEstimator(; alg = MaximumDistanceSimilarity())    # similarity branch
    ces = [BetweennessCentrality(), ClosenessCentrality(), DegreeCentrality(),
           EigenvectorCentrality(), KatzCentrality(), Pagerank(), RadialityCentrality(),
           StressCentrality()]

    @testset "The declared polarity" begin
        # The fallback declares nothing, so an algorithm is unweightable until it opts in.
        struct UndeclaredCentrality <: PortfolioOptimisers.AbstractCentralityAlgorithm end
        @test PO.centrality_polarity(UndeclaredCentrality()) === nothing
        # Every shipped member, one assertion each.
        for ct in (BetweennessCentrality(), ClosenessCentrality(), RadialityCentrality(),
                   StressCentrality())
            @test PO.centrality_polarity(ct) === PO.DistancePolarity()
        end
        @test PO.centrality_polarity(EigenvectorCentrality()) === PO.SimilarityPolarity()
        for ct in (DegreeCentrality(), KatzCentrality(), Pagerank())
            @test PO.centrality_polarity(ct) === nothing
        end
    end

    @testset "The routing table" begin
        #=
        Which graph each pair gets. The distance route is available on both branches; the
        similarity route only on the branch a similarity actually selected.
        =#
        want_t = [SWG.SimpleWeightedGraph, SWG.SimpleWeightedGraph, G.SimpleGraph,
                  G.SimpleGraph, G.SimpleGraph, G.SimpleGraph, SWG.SimpleWeightedGraph,
                  SWG.SimpleWeightedGraph]
        want_p = [SWG.SimpleWeightedGraph, SWG.SimpleWeightedGraph, G.SimpleGraph,
                  SWG.SimpleWeightedGraph, G.SimpleGraph, G.SimpleGraph,
                  SWG.SimpleWeightedGraph, SWG.SimpleWeightedGraph]
        for (i, ct) in pairs(ces)
            @test isa(PO.centrality_graph(nte_t, ct, X), want_t[i])
            @test isa(PO.centrality_graph(nte_p, ct, X), want_p[i])
            # A clustering source is weightless whatever the algorithm declares.
            @test isa(PO.centrality_graph(ClustersEstimator(), ct, X), G.SimpleGraph)
        end
    end

    @testset "Weights re-rank, and equal weights do not" begin
        #=
        Both halves of the claim, on a graph constructed so the first one must hold. The
        cycle 1-2-3-4-5-1 has a heavy shortcut on (1,5): unweighted, vertices 1 and 5 are
        adjacent and tie with 3 on closeness; weighted, the shortcut is not worth taking
        and 3 wins outright. `FixedDistanceGraph` supplies the graph directly, because
        every structure `calc_adjacency` builds from data is a spanning tree or a PMFG.
        =#
        W = zeros(5, 5)
        for (i, j, w) in ((1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (1, 5, 10.0))
            W[i, j] = W[j, i] = w
        end
        Xd = zeros(3, 5)
        nte = FixedDistanceGraph(W, HopCount(; n = 1))
        unw = PO.calc_centrality(ClosenessCentrality(),
                                 G.SimpleGraph(phylogeny_matrix(nte, Xd).X))
        wtd = centrality_vector(nte, ClosenessCentrality(), Xd).X
        @test wtd != unw
        # The re-ranking, not merely a rescale: 1 ties with 3 unweighted and loses weighted.
        @test unw[1] == unw[3]
        @test wtd[3] > wtd[1]

        # Equal weights reproduce the unweighted answer exactly, on the same structure.
        We = Float64.(W .!= 0)
        nte_e = FixedDistanceGraph(We, HopCount(; n = 1))
        for ct in (BetweennessCentrality(), ClosenessCentrality(), RadialityCentrality(),
                   StressCentrality())
            @test centrality_vector(nte_e, ct, Xd).X ==
                  PO.calc_centrality(ct, G.SimpleGraph(phylogeny_matrix(nte_e, Xd).X))
        end
    end

    @testset "A tree's shortest paths are weight-independent" begin
        #=
        Exact `==`, because it is a theorem rather than a numerical coincidence: a tree has
        exactly one path between any two vertices, so the shortest-path set does not depend
        on the weights. Betweenness and stress count paths, so both are invariant. Closeness
        and radiality read the path *lengths* and are not -- which is what makes the pair of
        assertions evidence that the weights really did arrive.
        =#
        for alg in (KruskalTree(), BoruvkaTree(), PrimTree())
            nte = NetworkEstimator(; alg = alg)
            unw(ct) = PO.calc_centrality(ct, G.SimpleGraph(phylogeny_matrix(nte, X).X))
            for ct in (BetweennessCentrality(), StressCentrality())
                @test centrality_vector(nte, ct, X).X == unw(ct)
            end
            for ct in (ClosenessCentrality(), RadialityCentrality())
                @test centrality_vector(nte, ct, X).X != unw(ct)
            end
        end
        # It is a fact about the tree, not about the algorithms: on a PMFG, where a pair has
        # many paths, both of them do move.
        for ct in (BetweennessCentrality(), StressCentrality())
            @test centrality_vector(nte_p, ct, X).X !=
                  PO.calc_centrality(ct, G.SimpleGraph(phylogeny_matrix(nte_p, X).X))
        end
    end

    @testset "The similarity branch is re-weighted with D, not with S" begin
        #=
        The PMFG's own weights are the similarities that selected its edges, and a shortest
        path over those seeks the route through the weakest links. `calc_distance_weighted_graph`
        supplies the distances on the same structure instead. Asserting the S-weighted answer
        is *not* what ships: it correlates about 0.95 with the right one, so a correlation
        test would pass the bug.
        =#
        gD = PO.centrality_graph(PO.DistancePolarity(), nte_p, X)
        gS = PO.centrality_graph(PO.SimilarityPolarity(), nte_p, X)
        # Same structure, different weights.
        @test G.adjacency_matrix(G.SimpleGraph(gD)) == G.adjacency_matrix(G.SimpleGraph(gS))
        @test G.weights(gD) != G.weights(gS)
        for ct in (ClosenessCentrality(), RadialityCentrality())
            shipped = centrality_vector(nte_p, ct, X).X
            @test shipped == PO.calc_centrality(ct, gD)
            @test shipped != PO.calc_centrality(ct, gS)
        end
        # Eigenvector is the one algorithm that wants the similarities, and gets them.
        @test PO.centrality_graph(nte_p, EigenvectorCentrality(), X) === nothing ||
              G.weights(PO.centrality_graph(nte_p, EigenvectorCentrality(), X)) ==
              G.weights(gS)
    end

    @testset "The five unweightable cases return the unweighted answer" begin
        #=
        #240: weightedness is a property of the source, not of the request. There is no
        flag, so a caller never asks for weights, and an unweightable pair has not been
        handed a request it cannot serve. Exact `==` against the plain-graph result --
        except for eigenvector, whose `Graphs.eigenvector_centrality` seeds its Arnoldi
        iteration randomly and differs from itself at about `6e-16` between two runs on one
        and the same graph.
        =#
        eig_noise(ct) = isa(ct, EigenvectorCentrality)
        same(ct, a, b) = eig_noise(ct) ? isapprox(a, b) : a == b

        # 1. Weightless sources: a clustering estimator, a precomputed `Clusters`, and a
        #    precomputed `PhylogenyResult`.
        clr = clusterise(ClustersEstimator(), X)
        for pl in (ClustersEstimator(), clr)
            for ct in ces
                @test same(ct, centrality_vector(pl, ct, X).X,
                           PO.calc_centrality(ct, G.SimpleGraph(phylogeny_matrix(pl, X).X)))
            end
        end
        plr = phylogeny_matrix(nte_p, X)
        for ct in ces
            @test same(ct, centrality_vector(plr, ct).X,
                       PO.calc_centrality(ct, G.SimpleGraph(plr.X)))
        end

        # 2--4. Degree, pagerank and Katz declare no polarity, on either branch.
        for nte in (nte_t, nte_p), ct in (DegreeCentrality(), Pagerank(), KatzCentrality())
            @test centrality_vector(nte, ct, X).X ==
                  PO.calc_centrality(ct, G.SimpleGraph(phylogeny_matrix(nte, X).X))
        end
        # Katz needs the route rather than merely the absence of a check: it does not ignore
        # the weights, it fails on them.
        @test_throws InexactError PO.calc_centrality(KatzCentrality(),
                                                     PO.calc_distance_weighted_graph(nte_t,
                                                                                     X))

        # 5. Eigenvector on a tree branch, where no similarity exists.
        for alg in (KruskalTree(), BoruvkaTree(), PrimTree())
            nte = NetworkEstimator(; alg = alg)
            @test isapprox(centrality_vector(nte, EigenvectorCentrality(), X).X,
                           PO.calc_centrality(EigenvectorCentrality(),
                                              G.SimpleGraph(phylogeny_matrix(nte, X).X)))
        end
    end

    @testset "The separation is read on the unweighted route only" begin
        #=
        The weighted routes read the structure, not the separation closure, because a power
        of a weighted matrix sums products of distances rather than counting edges. So `sep`
        is inert there and live on the unweighted route. At the default `n = 1` the two
        agree, since the closure of a graph at one hop is the graph.
        =#
        base_w = centrality_vector(nte_t, ClosenessCentrality(), X).X
        base_u = centrality_vector(nte_t, DegreeCentrality(), X).X
        @test centrality_vector(NetworkEstimator(; sep = HopCount(; n = 1)),
                                DegreeCentrality(), X).X == base_u
        for n in 2:3
            nte = NetworkEstimator(; sep = HopCount(; n = n))
            @test centrality_vector(nte, ClosenessCentrality(), X).X == base_w
            @test centrality_vector(nte, DegreeCentrality(), X).X != base_u
        end
    end

    @testset "A matrix in args is refused" begin
        #=
        `args` was already a half-working weighted channel and is now a second one. The
        working half silently overrode the declared polarity; the other half bound the
        matrix to `betweenness_centrality`'s `vs` and overflowed the stack -- so this check
        also closes a crash. Non-matrix positional arguments are untouched.
        =#
        D = zeros(3, 3)
        for T in (BetweennessCentrality, ClosenessCentrality, StressCentrality)
            @test_throws PortfolioOptimisers.ConflictingArgumentError T(; args = (D,))
            @test_throws PortfolioOptimisers.ConflictingArgumentError T(; args = (1:3, D))
            @test isa(T(; args = (1:3,)), T)
            @test isa(T(), T)
        end
    end
end

#=
The polarity override (#258). `TopologyOnly` in an algorithm's `ov` field withdraws the
declared polarity, so `centrality_polarity` answers `nothing` and `centrality_graph` routes
to the plain graph. It runs one way only -- away from weights -- so every request is served
on every source, and nothing warns and nothing goes inert.
=#
@testset "Polarity override" begin
    using PortfolioOptimisers, Test, CSV, DataFrames, TimeSeries, StatsBase, SparseArrays,
          LinearAlgebra
    PO = PortfolioOptimisers
    G = PortfolioOptimisers.Graphs

    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    X = prior(EmpiricalPrior(), rd).X
    nte_t = NetworkEstimator()                                       # tree branch
    nte_p = NetworkEstimator(; alg = MaximumDistanceSimilarity())    # similarity branch

    # The five that declare a polarity, plain and overridden. Kept as pairs so every test
    # below compares like with like.
    pairs_ov = ((BetweennessCentrality(), BetweennessCentrality(; ov = TopologyOnly())),
                (ClosenessCentrality(), ClosenessCentrality(; ov = TopologyOnly())),
                (StressCentrality(), StressCentrality(; ov = TopologyOnly())),
                (RadialityCentrality(), RadialityCentrality(; ov = TopologyOnly())),
                (EigenvectorCentrality(), EigenvectorCentrality(; ov = TopologyOnly())))
    # `Graphs.eigenvector_centrality` seeds its Arnoldi iteration randomly and differs from
    # itself at about `6e-16` between two runs on one and the same graph.
    eq(ct, a, b) = isa(ct, EigenvectorCentrality) ? isapprox(a, b) : a == b

    @testset "The effective polarity" begin
        #=
        `centrality_polarity` answers the *effective* polarity, not the declared one, and
        each method returns one concrete type. A `Union` return would put a dynamic
        dispatch on the one call site in `src/`.
        =#
        for (plain, ov) in pairs_ov
            @test PO.centrality_polarity(plain) !== nothing
            @test PO.centrality_polarity(ov) === nothing
            @test only(Base.return_types(PO.centrality_polarity, (typeof(ov),))) === Nothing
            @test isconcretetype(only(Base.return_types(PO.centrality_polarity,
                                                        (typeof(plain),))))
        end
        # The override is `nothing` by default, so the declaration stands untouched.
        for (plain, _) in pairs_ov
            @test plain.ov === nothing
        end
    end

    @testset "The three without the field refuse the keyword" begin
        #=
        Capability is type-level, so the refusal costs no check: only the five that declare
        a polarity carry `ov`. Julia's own message reads `got unsupported keyword argument
        "ov"` and lists the supported ones, so the throw is asserted and not the text.
        =#
        @test_throws MethodError DegreeCentrality(; ov = TopologyOnly())
        @test_throws MethodError KatzCentrality(; ov = TopologyOnly())
        @test_throws MethodError Pagerank(; ov = TopologyOnly())
        for T in (DegreeCentrality, KatzCentrality, Pagerank)
            @test !hasfield(T, :ov)
        end
        for (plain, _) in pairs_ov
            @test hasfield(typeof(plain), :ov)
        end
    end

    @testset "The override gives the plain-graph answer" begin
        #=
        The oracle, on both branches. #257 measured the gap as 2 of 8 on a tree and 5 of 8
        on a graph, so a test that only exercises a tree passes vacuously for three of the
        five -- hence both branches, and hence the second assertion, which pins where the
        override actually moves the answer.
        =#
        moves_t = (false, true, false, true, false)   # tree: closeness and radiality
        moves_p = (true, true, true, true, true)      # graph: all five
        for (nte, moves) in ((nte_t, moves_t), (nte_p, moves_p))
            plain_g = PO.centrality_graph(nothing, nte, X)
            for (i, (plain, ov)) in pairs(pairs_ov)
                got = centrality_vector(nte, ov, X).X
                @test eq(ov, got, PO.calc_centrality(ov, plain_g))
                # The routing, not merely the number: an overridden call builds a plain
                # graph rather than a weighted one.
                @test isa(PO.centrality_graph(nte, ov, X), G.SimpleGraph)
                # And it is not a no-op wherever the weights were reaching the answer.
                was = centrality_vector(nte, plain, X).X
                @test eq(ov, was, got) == !moves[i]
            end
        end
    end

    @testset "Trivially satisfied on the weightless sources" begin
        #=
        #253 and #254: the override runs one way, away from weights, so a source that
        carries none already returns the requested answer. Nothing warns and nothing
        throws, and the equality is trivial by construction.
        =#
        clr = clusterise(ClustersEstimator(), X)
        for pl in (ClustersEstimator(), clr)
            for (plain, ov) in pairs_ov
                got = @test_logs centrality_vector(pl, ov, X).X
                @test eq(ov, got,
                         PO.calc_centrality(ov, G.SimpleGraph(phylogeny_matrix(pl, X).X)))
            end
        end
        # A precomputed `PhylogenyResult` builds its own plain graph and never passes the
        # polarity seam at all.
        plr = phylogeny_matrix(nte_p, X)
        for (plain, ov) in pairs_ov
            got = @test_logs centrality_vector(plr, ov).X
            @test eq(ov, got, PO.calc_centrality(ov, G.SimpleGraph(plr.X)))
        end
        # Eigenvector on a tree branch: `SimilarityPolarity` already routes to the plain
        # graph there, because a tree carries no similarity to read.
        for alg in (KruskalTree(), BoruvkaTree(), PrimTree())
            nte = NetworkEstimator(; alg = alg)
            plain, ov = pairs_ov[5]
            @test isapprox(centrality_vector(nte, ov, X).X,
                           centrality_vector(nte, plain, X).X)
        end
    end

    @testset "The separation goes live again" begin
        #=
        `sep` is inert on a weighted route, because a power of a weighted matrix sums
        products of distances rather than counting edges. The override moves the call off
        that route, so the separation closure is read once more.
        =#
        ov = ClosenessCentrality(; ov = TopologyOnly())
        base = centrality_vector(NetworkEstimator(; sep = HopCount(; n = 1)), ov, X).X
        for n in 2:3
            nte = NetworkEstimator(; sep = HopCount(; n = n))
            @test centrality_vector(nte, ov, X).X != base
            # The declared route stays inert, which is what makes the contrast evidence.
            @test centrality_vector(nte, ClosenessCentrality(), X).X ==
                  centrality_vector(nte_t, ClosenessCentrality(), X).X
        end
    end

    @testset "The seams take the override positionally, and only positionally" begin
        #=
        The carrier is the algorithm, and `ct` is positional on every public surface, so
        the override reaches all of them with no signature change. The trap this avoids:
        an undeclared keyword is swallowed in silence at every seam, so a documented `ov =`
        beside an undeclared one would have shipped a request that does nothing.
        =#
        ov = ClosenessCentrality(; ov = TopologyOnly())
        want = centrality_vector(nte_p, ov, X).X
        w = fill(inv(size(X, 2)), size(X, 2))

        # The estimator bundle carries it through, and the bundle identity still holds.
        cte = CentralityEstimator(; pl = nte_p, ct = ov)
        @test centrality_vector(cte, X).X == want
        @test average_centrality(cte, w, X) == average_centrality(cte.pl, cte.ct, w, X)
        @test average_centrality(cte, w, X) ≈ LinearAlgebra.dot(want, w)

        # No seam accepts an `ov` keyword: the keyword form is swallowed and returns the
        # plain answer, so the request must be made by configuring the algorithm.
        @test centrality_vector(nte_p, ClosenessCentrality(), X; ov = TopologyOnly()).X ==
              centrality_vector(nte_p, ClosenessCentrality(), X).X
        @test centrality_vector(nte_p, ClosenessCentrality(), X; ov = TopologyOnly()).X !=
              want

        # The constraint generator reads `cc.A.ct`, so it gets the override for free.
        cte_pl = CentralityEstimator(; pl = nte_p, ct = ClosenessCentrality())
        lc_ov = centrality_constraints(CentralityConstraint(; A = cte, B = 0.5, comp = <=),
                                       X)
        lc_pl = centrality_constraints(CentralityConstraint(; A = cte_pl, B = 0.5,
                                                            comp = <=), X)
        @test vec(lc_ov.ineq.A) == want
        @test vec(lc_ov.ineq.A) != vec(lc_pl.ineq.A)
    end
end
