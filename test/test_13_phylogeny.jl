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
        # Both new members are on the exported API and in the family. The family's
        # abstract types are unexported, so the members reach them by the module prefix.
        @test ComplementSimilarity() isa
              PortfolioOptimisers.AbstractSimilarityMatrixAlgorithm
        @test AngularSimilarity() isa PortfolioOptimisers.AbstractSimilarityMatrixAlgorithm
    end
    @testset "The similarity sweep of #449" begin
        #=
        Each of the five closed forms of `src/09_Distance/04_Similarity.jl` computed by
        hand and compared, rather than read. The testset above already pins the two the
        library shipped first; this one covers all five, the element type every branch
        carries through, the sign that excludes `AngularSimilarity` from the non-negative
        family, and the `default_similarity` method declared in `05_FeatureDistance.jl`.
        =#
        Ds = [0.0 0.25 0.5; 0.25 0.0 1.0; 0.5 1.0 0.0]
        gse = GeneralExponentialSimilarity(; coef = 2, power = 3)
        for (sim, hand) in ((MaximumDistanceSimilarity(), ceil(maximum(Ds)^2) .- Ds .^ 2),
                            (ExponentialSimilarity(), exp.(-Ds)),
                            (GeneralExponentialSimilarity(), exp.(-Ds)), (gse, exp.(-2 .* Ds .^ 3)),
                            (ComplementSimilarity(), 1 .- Ds), (AngularSimilarity(), cos.(pi .* Ds)))
            @test PortfolioOptimisers.distance_to_similarity(sim; D = Ds) == hand
        end
        # The ceiling is the maximum squared and then rounded up, taken once as a scalar.
        @test ceil(maximum(Ds)^2) == 1.0
        #=
        Every branch carries the element type of `D` through. `ComplementSimilarity`
        writes its unit as `one(eltype(D))` and `AngularSimilarity` scales by the
        `Irrational` `pi` for exactly that reason, so a widening here is a defect in the
        branch and not in the test.
        =#
        D32 = Float32[0.0 0.25; 0.25 0.0]
        for sim in (MaximumDistanceSimilarity(), ExponentialSimilarity(),
                    GeneralExponentialSimilarity(), gse, ComplementSimilarity(),
                    AngularSimilarity())
            @test eltype(PortfolioOptimisers.distance_to_similarity(sim; D = D32)) ===
                  Float32
        end
        #=
        The sign of `AngularSimilarity` is why it is excluded from
        `AbstractNonNegativeSimilarityMatrixAlgorithm` permanently rather than pending a
        domain precondition: it crosses zero at `D = 0.5` and is negative above it.
        =#
        Dg = [0.0 0.25 0.75; 0.25 0.0 0.5; 0.75 0.5 0.0]
        Sg = PortfolioOptimisers.distance_to_similarity(AngularSimilarity(); D = Dg)
        @test Sg[1, 2] > 0
        @test isapprox(Sg[2, 3], 0; atol = 1e-15)
        @test Sg[1, 3] < 0
        @test !(AngularSimilarity() isa
                PortfolioOptimisers.AbstractNonNegativeSimilarityMatrixAlgorithm)
        for sim in (MaximumDistanceSimilarity(), ExponentialSimilarity(),
                    GeneralExponentialSimilarity(), ComplementSimilarity())
            @test sim isa PortfolioOptimisers.AbstractNonNegativeSimilarityMatrixAlgorithm
        end
        #=
        `MaximumDistanceSimilarity` is `NaN` at a non-finite entry and `Inf` everywhere
        else, which is the whole reason `assert_similarity_domain` refuses it there.
        =#
        Sinf = PortfolioOptimisers.distance_to_similarity(MaximumDistanceSimilarity();
                                                          D = [0.0 Inf; Inf 0.0])
        @test all(isnan, [Sinf[1, 2], Sinf[2, 1]])
        @test all(isinf, [Sinf[1, 1], Sinf[2, 2]])
        # `ExponentialSimilarity` takes an infinite distance, where `exp(-Inf)` is 0.
        @test PortfolioOptimisers.distance_to_similarity(ExponentialSimilarity();
                                                         D = [0.0 Inf; Inf 0.0]) ==
              [1.0 0.0; 0.0 1.0]
        # The fallback of `assert_similarity_domain` is a no-op on that same matrix.
        @test isnothing(PortfolioOptimisers.assert_similarity_domain(ExponentialSimilarity(),
                                                                     Distance(),
                                                                     [0.0 Inf; Inf 0.0]))
        #=
        `default_similarity` speaks for both of its methods. The `AngularDist` method is
        declared in `src/09_Distance/05_FeatureDistance.jl`, beside the metric; the
        docstring lives in `04_Similarity.jl` and states the branch.
        =#
        @test PortfolioOptimisers.default_similarity(AngularDist()) === AngularSimilarity()
        #=
        The pairing trap both warnings name, to the numbers they quote. `SimpleDistance`
        shares the `[0, 1]` range of an angular distance, so nothing can detect the
        mispairing: `0.706` is a perfectly legal bounded distance.
        =#
        dtrap = sqrt((1 - 0.003) / 2)
        @test isapprox(dtrap, 0.706; atol = 5e-4)
        @test isapprox(1 - dtrap, 0.29; atol = 5e-3)
        @test isapprox(cos(pi * dtrap), -0.603; atol = 5e-4)
        #=
        `AngularSimilarity` against `AngularDist` is the one pairing that is exact: it is
        the algebraic inverse, so the recovered cosine matches the one computed from the
        features themselves.
        =#
        rngs = StableRNG(123)
        Xf = randn(rngs, 200, 8)
        cosf = [dot(Xf[:, i], Xf[:, j]) / (norm(Xf[:, i]) * norm(Xf[:, j]))
                for i in 1:8, j in 1:8]
        Daf = PortfolioOptimisers.Distances.pairwise(AngularDist(), Xf; dims = 2)
        @test isapprox(PortfolioOptimisers.distance_to_similarity(AngularSimilarity();
                                                                  D = Daf), cosf;
                       atol = 1e-14)
    end
    @testset "Non-negative similarity interface (#239)" begin
        nonneg = (MaximumDistanceSimilarity(), ExponentialSimilarity(),
                  GeneralExponentialSimilarity(), ComplementSimilarity())
        # Four of the five members are in the interface; `AngularSimilarity` is out, and
        # stays in the wider family it was always in.
        for sim in nonneg
            @test sim isa PortfolioOptimisers.AbstractNonNegativeSimilarityMatrixAlgorithm
            @test sim isa PortfolioOptimisers.AbstractSimilarityMatrixAlgorithm
            @test sim isa PortfolioOptimisers.Tree_SimMat
        end
        @test !isa(AngularSimilarity(),
                   PortfolioOptimisers.AbstractNonNegativeSimilarityMatrixAlgorithm)
        @test AngularSimilarity() isa PortfolioOptimisers.AbstractSimilarityMatrixAlgorithm
        @test !isa(AngularSimilarity(), PortfolioOptimisers.Tree_SimMat)
        @test PortfolioOptimisers.AbstractNonNegativeSimilarityMatrixAlgorithm <:
              PortfolioOptimisers.AbstractSimilarityMatrixAlgorithm
        # Both abstract types are unexported, per the repository convention. An extension
        # subtypes the interface through the module prefix. `test_43` is the census that
        # holds the whole exported abstract surface to its allow-list.
        for T in (:AbstractSimilarityMatrixAlgorithm,
                  :AbstractNonNegativeSimilarityMatrixAlgorithm)
            @test T ∉ names(PortfolioOptimisers)
            @test isdefined(PortfolioOptimisers, T)
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
        @testset "The coefficient is finite as well as positive" begin
            #=
            `coef = Inf` used to construct. The zero diagonal then gives `exp(-Inf * 0)`,
            which is `NaN`, so a NaN similarity matrix reached `PMFG_T2s` and the DBHT
            clustering instead of a typed error naming the field. `power` is a separate
            case and stays as it was: `power = Inf` yields a finite matrix.
            =#
            Dc = [0.0 0.5 0.25; 0.5 0.0 0.75; 0.25 0.75 0.0]
            for coef in (Inf, -Inf, NaN)
                @test_throws DomainError GeneralExponentialSimilarity(; coef = coef)
            end
            msg = try
                GeneralExponentialSimilarity(; coef = Inf)
                ""
            catch e
                sprint(showerror, e)
            end
            @test occursin("coef", msg)
            # The positivity half of the guard is unchanged.
            for coef in (0, -1)
                @test_throws DomainError GeneralExponentialSimilarity(; coef = coef)
            end
            # A finite coefficient still builds, and an infinite `power` is still admitted.
            sec = GeneralExponentialSimilarity(; coef = 2, power = Inf)
            @test sec isa GeneralExponentialSimilarity
            @test all(isfinite, PortfolioOptimisers.distance_to_similarity(sec; D = Dc))
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
            #=
            `pr.X` is too correlated to denoise to an exact zero, so this needs noise. The
            zero needs *every* eigenvalue under the fitted edge: `ShrunkDenoise` at the
            default `alpha = 0` keeps the signal block, so an exact zero off the diagonal
            asks that no signal block survives. Ticket 475 sharpened the fit, and the draw
            this test used keeps four components under it; `StableRNG(1)` is a draw whose
            whole spectrum still falls under the edge, by a margin of 5.8 %.
            =#
            Xn = randn(StableRNG(1), 500, 20)
            @test any(isinf, PortfolioOptimisers.distance(de, ce, Xn))
            nte = NetworkEstimator(; ce = ce, de = de, alg = MaximumDistanceSimilarity())
            @test_throws DomainError PortfolioOptimisers.calc_adjacency(nte, Xn)
            #=
            `ExponentialSimilarity` declares no domain and is right not to: `exp(-Inf)` is
            `0` exactly, so an infinite distance is a legal similarity *value*. It is not a
            legal edge *weight*. `PMFG_T2s` stores the structure and the weights in one
            matrix, so a zero weight is an absent edge, and the denoised correlation of
            pure noise is the identity -- every off-diagonal similarity is zero and the
            PMFG comes back with `0` of its `54` edges. That empty structure used to be
            returned as an answer. `assert_pmfg_weights` names it instead.
            =#
            nte = NetworkEstimator(; ce = ce, de = de, alg = ExponentialSimilarity())
            @test_throws DomainError PortfolioOptimisers.calc_adjacency(nte, Xn)
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
        @testset "AngularDist against acos(rho)/pi, computed by hand" begin
            # The elementwise method's own contract, checked against the closed form rather
            # than against the other path.
            Zh = [1.0 0.0; 0.0 1.0; 1.0 1.0; -1.0 1.0]
            for i in 1:4, j in 1:4
                a = Zh[i, :]
                b = Zh[j, :]
                rho = dot(a, b) / (norm(a) * norm(b))
                @test isapprox(AngularDist()(a, b), acos(clamp(rho, -1, 1)) / pi;
                               atol = 1e-14)
            end
            #=
            The two paths part on the diagonal, and the gemm path is the correct one there.
            `acos` has an infinite derivative at 1, so the cosine of a vector with itself
            rounding to 0.9999999999999999 becomes a 1e-8 error in the distance. The gemm
            path writes an exact zero instead. This is the measurement the `AngularDist`
            docstring's `# Algorithm` section states.
            =#
            Zd = randn(rng, 8, 5)
            Dg = DS.pairwise(AngularDist(), Zd; dims = 1)
            @test all(iszero, diag(Dg))
            self = [AngularDist()(Zd[i, :], Zd[i, :]) for i in 1:8]
            @test maximum(self) <= 6.707879276254074e-9
            @test any(!iszero, self)
            offg = [Dg[i, j] for i in 1:8, j in 1:8 if i != j]
            offe = [AngularDist()(Zd[i, :], Zd[j, :]) for i in 1:8, j in 1:8 if i != j]
            @test maximum(abs.(offg .- offe)) <= 1e-14
        end
        @testset "The four collapse algorithms against a hand-built answer" begin
            #=
            Three axes of different lengths: 4 observations, 5 assets, 3 features. A cubic
            array hides a permuted axis. `Euclidean` is non-linear in the features, so
            `AggregateFeatures` and `AggregateDistances` cannot agree by accident.
            =#
            Zc = abs.(randn(rng, 4, 5, 3)) .+ 0.1
            metric = DS.Euclidean()
            de(alg) = FeatureDistance(; metric = metric, alg = alg)
            # `LastObservation` takes the last slice of the observation axis.
            @test isapprox(distance(de(LastObservation()), Zc),
                           DS.pairwise(metric, Zc[4, :, :]; dims = 1))
            # `AggregateFeatures` collapses the features, then measures once.
            Zm = [mean(Zc[:, j, k]) for j in 1:5, k in 1:3]
            @test isapprox(distance(de(AggregateFeatures()), Zc),
                           DS.pairwise(metric, Zm; dims = 1))
            Zmed = [median(Zc[:, j, k]) for j in 1:5, k in 1:3]
            @test isapprox(distance(de(AggregateFeatures(; alg = MedianCollapse())), Zc),
                           DS.pairwise(metric, Zmed; dims = 1))
            # `AggregateDistances` measures every observation, then collapses the matrices.
            Dad = mean([DS.pairwise(metric, Zc[t, :, :]; dims = 1) for t in 1:4])
            @test isapprox(distance(de(AggregateDistances()), Zc), Dad)
            # `StackObservations` concatenates the window into one vector per asset.
            Zst = PortfolioOptimisers.stack_observations(Zc, 1)
            @test size(Zst) == (5, 12)
            Zst_hand = Matrix{Float64}(undef, 5, 12)
            for j in 1:5
                Zst_hand[j, :] = vec(permutedims(Zc, (2, 1, 3))[j, :, :])
            end
            @test Zst == Zst_hand
            @test isapprox(distance(de(StackObservations()), Zc),
                           DS.pairwise(metric, Zst_hand; dims = 1))
            # The two aggregations differ in *what* they aggregate, so on a non-linear
            # metric they must give different answers.
            @test !isapprox(distance(de(AggregateFeatures()), Zc), Dad)
            # `dims = 2` swaps the two trailing axes, for every rule and both kernels.
            Zp = permutedims(Zc, (1, 3, 2))
            @test size(PortfolioOptimisers.stack_observations(Zp, 2)) == (5, 12)
            for alg in (LastObservation(), AggregateFeatures(), AggregateDistances(),
                        StackObservations(), AggregateFeatures(; alg = MedianCollapse()))
                @test isapprox(distance(de(alg), Zp; dims = 2), distance(de(alg), Zc))
            end
        end
        @testset "The weighted collapses, and the weighted median interpolates" begin
            Zc = abs.(randn(rng, 4, 5, 3)) .+ 0.1
            metric = DS.Euclidean()
            wc = pweights([1.0, 2.0, 3.0, 4.0])
            # The weighted mean is the convex combination the docstring states.
            Zm = [sum(wc .* Zc[:, j, k]) / sum(wc) for j in 1:5, k in 1:3]
            @test isapprox(distance(FeatureDistance(; metric = metric,
                                                    alg = AggregateFeatures(; w = wc)), Zc),
                           DS.pairwise(metric, Zm; dims = 1))
            # `AggregateDistances` weights the matrices and divides by the weight total.
            Dad = sum(wc[t] .* DS.pairwise(metric, Zc[t, :, :]; dims = 1) for t in 1:4) ./
                  sum(wc)
            @test isapprox(distance(FeatureDistance(; metric = metric,
                                                    alg = AggregateDistances(; w = wc)),
                                    Zc), Dad)
            #=
            `Statistics.median(v, w)` is the StatsBase 0.5-quantile, not an order statistic,
            so a weighted median need not be an element of the window. Check the collapse
            against that function rather than against a value of `v`.
            =#
            Zmed = [median(Zc[:, j, k], wc) for j in 1:5, k in 1:3]
            @test isapprox(distance(FeatureDistance(; metric = metric,
                                                    alg = AggregateFeatures(; w = wc,
                                                                            alg = MedianCollapse())),
                                    Zc), DS.pairwise(metric, Zmed; dims = 1))
            #=
            A fixed window, so the claim does not rest on a draw. Under `[1, 2, 3, 4]` the
            weighted median of `[0, 1, 2, 3]` is `11/6`, which lies strictly between the
            second and the third value and is no element of the window. Under `[4, 3, 2, 1]`
            it lands on `v[2]`, so interpolation is what the quantile *may* do, not what it
            always does.
            =#
            v = [0.0, 1.0, 2.0, 3.0]
            @test median(v, pweights([1.0, 2, 3, 4])) ≈ 11 / 6
            @test median(v, pweights([1.0, 2, 3, 4])) ∉ v
            @test v[2] < median(v, pweights([1.0, 2, 3, 4])) < v[3]
            @test median(v, pweights([4.0, 3, 2, 1])) == v[2]
            # The unweighted median of an even window averages the two central values.
            @test median(v) ≈ 1.5
            @test median(v) ∉ v
        end
        @testset "The estimator is not mutated by a call" begin
            # An in-place broadcast on a weight field has written into an estimator in this
            # library before, so the whole configuration is compared before and after.
            Zc = abs.(randn(rng, 4, 5, 3)) .+ 0.1
            wc = pweights([1.0, 2.0, 3.0, 4.0])
            for alg in (AggregateFeatures(; w = wc), AggregateDistances(; w = wc),
                        AggregateFeatures(; w = wc, alg = MedianCollapse()))
                fde = FeatureDistance(; alg = alg)
                before = deepcopy(fde)
                distance(fde, Zc)
                cor_and_dist(fde, Zc)
                @test collect(fde.alg.w) == collect(before.alg.w)
                @test fde.metric === before.metric
                @test fde.sim === before.sim
                @test fde.alg.alg === before.alg.alg
            end
            # The window itself is read, never written.
            Zk = copy(Zc)
            distance(FeatureDistance(; alg = AggregateDistances()), Zc)
            @test Zc == Zk
        end
        @testset "cor_and_dist carries the distance this call produced" begin
            Zc = abs.(randn(rng, 4, 5, 3)) .+ 0.1
            for Zin in (Zc[4, :, :], Zc)
                for alg in (LastObservation(), AggregateFeatures(), AggregateDistances(),
                            StackObservations())
                    fde = FeatureDistance(; alg = alg)
                    S, D = cor_and_dist(fde, Zin)
                    @test D == distance(fde, Zin)
                    # `AngularSimilarity` is the exact inverse of `AngularDist`.
                    @test isapprox(S, cos.(pi .* D))
                end
            end
        end
        @testset "A median collapse of an integer window" begin
            #=
            The median of an even window is the mean of two order statistics, so a window
            of integers has a fractional median. Preallocating the result at the window's
            element type threw `InexactError` there; the collapse now takes its element
            type from the values it computes.
            =#
            Zi = rand(rng, 1:9, 4, 5, 3)
            wc = pweights([1.0, 2.0, 3.0, 4.0])
            for alg in (AggregateFeatures(; alg = MedianCollapse()),
                        AggregateFeatures(; w = wc, alg = MedianCollapse()))
                D = distance(FeatureDistance(; alg = alg), Zi)
                @test eltype(D) === Float64
                @test issymmetric(D)
                @test !any(isnan, D)
            end
            @test eltype(PortfolioOptimisers.collapse_features(MedianCollapse(), Zi,
                                                               nothing)) === Float64
            # A window whose median stays inside its own type keeps that type.
            @test eltype(PortfolioOptimisers.collapse_features(MedianCollapse(),
                                                               Float32.(Zi), nothing)) ===
                  Float32
            @test eltype(PortfolioOptimisers.collapse_features(MedianCollapse(),
                                                               Rational{Int}.(Zi), nothing)) ===
                  Rational{Int}
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
        # `HClustAlgorithm` above reaches `AbstractClustersAlgorithm`'s own identity method,
        # not the one on `AbstractPhylogenyAlgorithm`. A tree type and a centrality algorithm
        # are the phylogeny algorithms that carry no nearer method, so they are what reaches
        # it. Nothing else in the suite did, which left both of its lines uncovered.
        for palg in (KruskalTree(), BoruvkaTree(), PrimTree(), DegreeCentrality(),
                     BetweennessCentrality())
            @test factory(palg) === palg
            @test factory(palg, wt) === palg
            @test which(factory, Tuple{typeof(palg), Any}).sig ===
                  Tuple{typeof(factory), PortfolioOptimisers.AbstractPhylogenyAlgorithm,
                        Vararg{Any}}
        end
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
        # Observation weights stop at `ce`. `Clustering.kmeans` weights its points, which
        # here are the assets, so a weighted arity must return the algorithm unchanged
        # rather than write an `ObsWeights` into `kwargs` (#393).
        wt2 = pweights(fill(inv(size(pr.X, 2)), size(pr.X, 2)))
        @test factory(alg, wt2) === alg
        @test !haskey(factory(alg, wt2).kwargs, :weights)
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
    # Issue #511. This testset and the `LoGo` testset below sat inside a `#=` … `=#` from
    # commit 2e3a993bf4 until now, and the fixture they read, `assets/asset_prices.csv`,
    # never reached the repository. Both are rebuilt here on the file's own SP500 window.
    #
    # Every `_t` value below was re-recorded from a run of this code. A re-recorded value
    # pins the current answer and cannot tell a right answer from a wrong one, so each one
    # is paired with an assertion that carries its own authority. Those come from the
    # definition of a planar maximal filtered graph in Song, Di Matteo and Aste, *Nested
    # hierarchies in planar graphs*, Discrete Applied Mathematics 159 (2011) 2135-2146: on
    # `N` vertices it has `3N - 6` edges, `2N - 4` triangular faces, `N - 4` non-face
    # 3-cliques and `N - 3` 4-cliques, and the dendrogram built over it has `N - 1` merges
    # that name each of the `N` leaves exactly once.
    @testset "DBHT Clustering tests" begin
        ce = PortfolioOptimisersCovariance()
        de = Distance(; alg = SimpleDistance())
        rho = cor(ce, pr.X)
        dist = distance(de, rho, pr.X)
        N = size(rho, 1)
        # `GeneralExponentialSimilarity` at its defaults is `coef = 1` and `power = 1`, so
        # it computes `exp(-D)`, which is what `ExponentialSimilarity` computes. This is an
        # identity between the two members, not a recorded value.
        @test isapprox(PortfolioOptimisers.distance_to_similarity(GeneralExponentialSimilarity();
                                                                  D = dist),
                       PortfolioOptimisers.distance_to_similarity(ExponentialSimilarity();
                                                                  D = dist))
        # The first configuration: `MaximumDistanceSimilarity` and a `UniqueRoot`.
        S1 = PortfolioOptimisers.distance_to_similarity(MaximumDistanceSimilarity();
                                                        D = dist)
        _, _, _, _, _, Z1, dbht1d = PortfolioOptimisers.DBHTs(dist, S1;
                                                              branchorder = :default,
                                                              root = UniqueRoot())
        A1, tri1, separators1, cliques1, cliqueTree1 = PortfolioOptimisers.PMFG_T2s(S1, 5)
        # The five shapes the planar maximal filtered graph must have.
        @test SparseArrays.nnz(A1) == 2 * (3 * N - 6)
        @test size(tri1) == (2 * N - 4, 3)
        @test size(separators1) == (N - 4, 3)
        @test size(cliques1) == (N - 3, 4)
        @test size(cliqueTree1) == (N - 3, N - 3)
        @test issymmetric(A1)
        # `PMFG_T2s` returns the structure and the weights in one matrix, so the stored
        # entries are the edges and each one carries its own similarity. The edge set is
        # therefore the whole of what has to be recorded: the weights follow from it.
        edges1_t = SparseArrays.sparse([2, 3, 4, 6, 7, 9, 10, 13, 14, 16, 1, 3, 4, 5, 6, 7,
                                        10, 13, 1, 2, 5, 6, 9, 17, 20, 1, 2, 6, 7, 2, 3, 6,
                                        17, 20, 1, 2, 3, 4, 5, 9, 20, 1, 2, 4, 13, 10, 11,
                                        12, 14, 15, 16, 18, 1, 3, 6, 1, 2, 8, 13, 14, 16,
                                        18, 19, 8, 12, 14, 15, 18, 8, 11, 15, 18, 1, 2, 7,
                                        10, 14, 1, 8, 10, 11, 13, 16, 18, 19, 8, 11, 12, 1,
                                        8, 10, 14, 3, 5, 20, 8, 10, 11, 12, 14, 19, 10, 14,
                                        18, 3, 5, 6, 17],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                                        2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5,
                                        6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
                                        8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11,
                                        11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14,
                                        14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16,
                                        16, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19,
                                        20, 20, 20, 20], true, 20, 20)
        @test SparseArrays.sparse(A1 .!= 0) == edges1_t
        @test A1 == S1 .* edges1_t
        tri1_t = reshape([10, 14, 10, 10, 10, 14, 10, 1, 10, 14, 1, 13, 1, 2, 10, 14, 14, 8,
                          8, 18, 8, 11, 1, 2, 1, 2, 1, 6, 10, 14, 2, 6, 6, 3, 3, 5, 14, 1,
                          14, 1, 1, 1, 13, 13, 16, 16, 2, 2, 7, 7, 8, 8, 18, 18, 11, 11, 12,
                          12, 4, 4, 6, 6, 3, 3, 18, 18, 3, 3, 5, 5, 20, 20, 19, 13, 13, 2,
                          16, 16, 2, 7, 8, 8, 3, 7, 4, 4, 18, 11, 11, 12, 15, 12, 15, 15, 6,
                          6, 9, 5, 9, 9, 19, 19, 5, 20, 20, 17, 17, 17], 36, 3)
        separators1_t = reshape([10, 10, 10, 1, 1, 10, 14, 8, 8, 1, 1, 1, 10, 2, 6, 3, 14,
                                 1, 14, 13, 2, 14, 8, 18, 11, 2, 2, 6, 14, 6, 3, 5, 1, 13,
                                 16, 2, 7, 8, 18, 11, 12, 4, 6, 3, 18, 3, 5, 20], 16, 3)
        cliques1_t = reshape([10, 10, 10, 10, 1, 1, 10, 14, 8, 8, 1, 1, 1, 10, 2, 6, 3, 14,
                              14, 1, 14, 13, 2, 14, 8, 18, 11, 2, 2, 6, 14, 6, 3, 5, 1, 1,
                              13, 16, 2, 7, 8, 18, 11, 12, 4, 6, 3, 18, 3, 5, 20, 13, 16, 2,
                              8, 7, 4, 18, 11, 12, 15, 6, 3, 9, 19, 5, 20, 17], 17, 4)
        # `cliqueTree` is not symmetric. It counts how many of the *first three* vertices
        # of each 4-clique every other 4-clique holds, so the count it takes for the pair
        # `(i, j)` is not the count it takes for `(j, i)`.
        cliqueTree1_t = SparseArrays.sparse([4, 5, 7, 14, 3, 7, 14, 1, 2, 6, 11, 12, 1, 2,
                                             8, 14, 3, 11, 12, 5, 12, 1, 2, 4, 9, 7, 10, 14,
                                             8, 9, 5, 6, 13, 15, 5, 6, 11, 16, 12, 15, 16,
                                             1, 2, 4, 7, 8, 12, 13, 17, 13, 15, 16],
                                            [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
                                             5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 10, 11,
                                             11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14,
                                             14, 14, 14, 15, 15, 15, 16, 16, 17], true, 17,
                                            17)
        @test tri1 == tri1_t
        @test separators1 == separators1_t
        @test cliques1 == cliques1_t
        @test SparseArrays.sparse(cliqueTree1 .!= 0) == cliqueTree1_t
        # A merge matrix over `N` leaves: `N - 1` merges, each leaf named once as a
        # negative index, and each earlier merge named once as a positive index.
        @test size(Z1) == (N - 1, 4)
        @test sort(vec(Z1[:, 1:2])[vec(Z1[:, 1:2]) .< 0]) == -collect(N:-1:1)
        @test sort(vec(Z1[:, 1:2])[vec(Z1[:, 1:2]) .> 0]) == collect(1.0:(N - 2))
        Z1_t = reshape([-1.0, -4.0, -3.0, -5.0, -17.0, -2.0, -6.0, 6.0, 2.0, 5.0, -10.0,
                        -19.0, -8.0, -12.0, -18.0, -16.0, 14.0, 12.0, 10.0, -13.0, -7.0,
                        -9.0, -20.0, 4.0, 1.0, 3.0, 7.0, 8.0, 9.0, -14.0, 11.0, -11.0,
                        -15.0, 13.0, 15.0, 16.0, 17.0, 18.0, 0.1, 0.1111111111111111, 0.125,
                        0.14285714285714285, 0.16666666666666666, 0.2, 0.25,
                        0.3333333333333333, 0.5, 1.0, 0.125, 0.14285714285714285,
                        0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.5, 1.0, 2.0,
                        2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 6.0, 8.0, 11.0, 2.0, 3.0, 2.0,
                        2.0, 3.0, 4.0, 6.0, 9.0, 20.0], 19, 4)
        @test isapprox(Z1, Z1_t)
        # The second configuration: `ExponentialSimilarity` and an `EqualRoot`.
        S2 = PortfolioOptimisers.distance_to_similarity(ExponentialSimilarity(); D = dist)
        _, _, _, _, _, Z2, dbht2d = PortfolioOptimisers.DBHTs(dist, S2;
                                                              branchorder = :default,
                                                              root = EqualRoot())
        A2, tri2, separators2, cliques2, cliqueTree2 = PortfolioOptimisers.PMFG_T2s(S2, 5)
        @test SparseArrays.nnz(A2) == 2 * (3 * N - 6)
        @test size(tri2) == (2 * N - 4, 3)
        @test size(separators2) == (N - 4, 3)
        @test size(cliques2) == (N - 3, 4)
        @test size(cliqueTree2) == (N - 3, N - 3)
        @test issymmetric(A2)
        edges2_t = SparseArrays.sparse([2, 3, 4, 6, 7, 9, 10, 13, 14, 18, 19, 1, 3, 4, 5, 6,
                                        7, 13, 18, 1, 2, 5, 6, 9, 17, 20, 1, 2, 6, 7, 2, 3,
                                        6, 17, 20, 1, 2, 3, 4, 5, 9, 20, 1, 2, 4, 13, 10,
                                        11, 12, 14, 15, 16, 1, 3, 6, 1, 8, 14, 16, 18, 19,
                                        8, 12, 14, 15, 8, 11, 14, 15, 16, 1, 2, 7, 14, 18,
                                        1, 8, 10, 11, 12, 13, 16, 18, 19, 8, 11, 12, 8, 10,
                                        12, 14, 18, 3, 5, 20, 1, 2, 10, 13, 14, 16, 1, 10,
                                        14, 3, 5, 6, 17],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                        2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                                        5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8,
                                        8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
                                        12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14,
                                        14, 14, 14, 14, 14, 14, 15, 15, 15, 16, 16, 16, 16,
                                        16, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19,
                                        20, 20, 20, 20], true, 20, 20)
        @test SparseArrays.sparse(A2 .!= 0) == edges2_t
        @test A2 == S2 .* edges2_t
        tri2_t = reshape([1, 14, 1, 1, 1, 14, 14, 18, 1, 13, 14, 10, 1, 13, 1, 2, 1, 2, 1,
                          2, 1, 6, 14, 16, 14, 8, 8, 12, 1, 14, 2, 6, 6, 3, 3, 5, 14, 13,
                          14, 13, 18, 18, 10, 10, 18, 18, 16, 16, 2, 2, 7, 7, 4, 4, 6, 6, 3,
                          3, 8, 8, 12, 12, 11, 11, 10, 10, 3, 3, 5, 5, 20, 20, 13, 18, 19,
                          7, 10, 16, 8, 16, 2, 2, 12, 8, 3, 7, 4, 4, 6, 6, 9, 5, 9, 9, 11,
                          12, 11, 15, 15, 15, 19, 19, 5, 20, 20, 17, 17, 17], 36, 3)
        separators2_t = reshape([1, 14, 1, 14, 1, 1, 1, 1, 1, 14, 14, 8, 1, 2, 6, 3, 14, 18,
                                 13, 10, 13, 2, 2, 2, 6, 16, 8, 12, 14, 6, 3, 5, 18, 10, 18,
                                 16, 2, 7, 4, 6, 3, 8, 12, 11, 10, 3, 5, 20], 16, 3)
        cliques2_t = reshape([1, 1, 14, 1, 14, 1, 1, 1, 1, 1, 14, 14, 8, 1, 2, 6, 3, 14, 14,
                              18, 13, 10, 13, 2, 2, 2, 6, 16, 8, 12, 14, 6, 3, 5, 13, 18,
                              10, 18, 16, 2, 7, 4, 6, 3, 8, 12, 11, 10, 3, 5, 20, 18, 10,
                              16, 2, 8, 7, 4, 6, 3, 9, 12, 11, 15, 19, 5, 20, 17], 17, 4)
        cliqueTree2_t = SparseArrays.sparse([3, 6, 14, 1, 4, 5, 2, 11, 14, 1, 2, 7, 8, 9, 3,
                                             12, 14, 1, 4, 8, 9, 6, 9, 6, 7, 10, 15, 6, 7,
                                             8, 16, 9, 15, 16, 5, 13, 11, 12, 1, 2, 3, 5, 9,
                                             10, 17, 10, 15, 16],
                                            [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5,
                                             5, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9,
                                             10, 10, 10, 11, 11, 12, 13, 14, 14, 14, 14, 15,
                                             15, 15, 16, 16, 17], true, 17, 17)
        @test tri2 == tri2_t
        @test separators2 == separators2_t
        @test cliques2 == cliques2_t
        @test SparseArrays.sparse(cliqueTree2 .!= 0) == cliqueTree2_t
        @test size(Z2) == (N - 1, 4)
        @test sort(vec(Z2[:, 1:2])[vec(Z2[:, 1:2]) .< 0]) == -collect(N:-1:1)
        @test sort(vec(Z2[:, 1:2])[vec(Z2[:, 1:2]) .> 0]) == collect(1.0:(N - 2))
        Z2_t = reshape([-1.0, -2.0, -4.0, -10.0, -19.0, -3.0, -8.0, -5.0, -17.0, -12.0,
                        -6.0, 7.0, 1.0, -16.0, 3.0, 13.0, 14.0, 12.0, 9.0, -18.0, -13.0,
                        -7.0, -14.0, 4.0, -9.0, -11.0, -20.0, 8.0, -15.0, 6.0, 10.0, 2.0,
                        5.0, 11.0, 15.0, 16.0, 17.0, 18.0, 0.05263157894736842,
                        0.05555555555555555, 0.058823529411764705, 0.0625,
                        0.06666666666666667, 0.07142857142857142, 0.07692307692307693,
                        0.08333333333333333, 0.09090909090909091, 0.1, 0.1111111111111111,
                        0.125, 0.14285714285714285, 0.16666666666666666, 0.2, 0.25,
                        0.3333333333333333, 0.5, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0,
                        2.0, 3.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 9.0, 13.0, 17.0, 20.0], 19,
                       4)
        @test isapprox(Z2, Z2_t)
        # `branchorder` is read after `Z` is built, and it rewrites the `Hclust` alone, so
        # all three orderings return the same `Z`. The block this testset replaces pinned
        # `Z` once per `branchorder`, so none of its three assertions could see the keyword
        # it varied. The assertion that can see it is the one on the seventh return value.
        _, _, _, _, _, Z1r, dbht1r = PortfolioOptimisers.DBHTs(dist, S1; branchorder = :r,
                                                               root = UniqueRoot())
        _, _, _, _, _, Z1o, dbht1o = PortfolioOptimisers.DBHTs(dist, S1;
                                                               branchorder = :optimal,
                                                               root = UniqueRoot())
        _, _, _, _, _, Z2r, dbht2r = PortfolioOptimisers.DBHTs(dist, S2; branchorder = :r,
                                                               root = EqualRoot())
        _, _, _, _, _, Z2o, dbht2o = PortfolioOptimisers.DBHTs(dist, S2;
                                                               branchorder = :optimal,
                                                               root = EqualRoot())
        @test Z1r == Z1
        @test Z1o == Z1
        @test Z2r == Z2
        @test Z2o == Z2
        # Every ordering returns a permutation of the leaves under the DBHT linkage.
        for dbht in (dbht1d, dbht1r, dbht1o, dbht2d, dbht2r, dbht2o)
            @test sort(dbht.order) == collect(1:N)
            @test dbht.linkage === :DBHT
        end
        # `:default` writes `Z` out as it stands.
        @test dbht1d.merges == Int.(Z1[:, 1:2])
        @test dbht1d.heights == Z1[:, 3]
        @test dbht2d.merges == Int.(Z2[:, 1:2])
        @test dbht2d.heights == Z2[:, 3]
        # `:r` sorts the merges by height and leaves the leaf order alone. The heights of
        # the first tree do not rise, so it rewrites that one.
        @test !issorted(Z1[:, 3])
        @test issorted(dbht1r.heights)
        @test dbht1r.merges != dbht1d.merges
        @test dbht1r.order == dbht1d.order
        # The heights of the second tree already rise, so `:r` has nothing to sort and
        # returns the `:default` answer entire.
        @test issorted(Z2[:, 3])
        @test dbht2r.merges == dbht2d.merges
        @test dbht2r.heights == dbht2d.heights
        @test dbht2r.order == dbht2d.order
        # `:optimal` reorders the branches to shorten the distance between adjacent
        # leaves. It keeps every height and moves the leaves.
        @test dbht1o.heights == dbht1d.heights
        @test dbht2o.heights == dbht2d.heights
        @test dbht1o.merges != dbht1d.merges
        @test dbht2o.merges != dbht2d.merges
        @test dbht1o.order != dbht1d.order
        @test dbht2o.order != dbht2d.order
        @test dbht1d.order ==
              [17, 5, 20, 4, 7, 2, 1, 13, 6, 3, 9, 19, 10, 14, 12, 15, 16, 18, 8, 11]
        @test dbht2d.order ==
              [17, 5, 20, 8, 11, 12, 15, 16, 19, 10, 14, 1, 18, 2, 13, 4, 7, 6, 3, 9]
        @test dbht1o.order ==
              [5, 20, 17, 3, 9, 6, 2, 1, 13, 7, 4, 19, 14, 10, 16, 18, 11, 8, 12, 15]
        @test dbht2o.order ==
              [5, 20, 17, 4, 7, 3, 9, 6, 2, 13, 1, 18, 19, 14, 10, 16, 15, 12, 8, 11]
        @test dbht1o.merges ==
              reshape([-1, -7, -3, -5, 4, -2, 3, 7, 8, 5, -14, -19, -11, -12, -18, -16, 16,
                       12, 10, -13, -4, -9, -20, -17, 1, -6, 6, 2, 9, -10, 11, -8, -15, 13,
                       15, 14, 17, 18], 19, 2)
        @test dbht2o.merges ==
              reshape([-1, -2, -4, -14, -19, -3, -8, -5, 8, -15, 6, 10, 2, 5, 3, 15, 16, 17,
                       9, -18, -13, -7, -10, 4, -9, -11, -20, -17, -12, -6, 7, 1, -16, 11,
                       13, 14, 12, 18], 19, 2)
    end
    # The reference columns are re-recorded from a run of this code, as the values in the
    # testset above are. The assertion that carries its own authority here is the sparsity
    # of the inverse: LoGo builds `J_LoGo` from the cliques and separators of the planar
    # maximal filtered graph, and inverts it, so the inverse of what it returns must be
    # zero at every pair the graph holds no edge for.
    @testset "LoGo" begin
        ce = PortfolioOptimisersCovariance()
        sigma = cov(ce, pr.X)
        N = size(sigma, 1)
        # `logo!` reads the correlation matrix, not the covariance matrix, so the graph the
        # assertions below rebuild is the graph of `Sc`.
        Sc = StatsBase.cov2cor(sigma, sqrt.(diag(sigma)))
        des = [Distance(; alg = CanonicalDistance()),
               DistanceDistance(; alg = CanonicalDistance()),
               Distance(; alg = SimpleDistance()),
               DistanceDistance(; alg = SimpleDistance()),
               Distance(; alg = SimpleAbsoluteDistance()),
               DistanceDistance(; alg = SimpleAbsoluteDistance()),
               Distance(; alg = CorrelationDistance()),
               DistanceDistance(; alg = CorrelationDistance()),
               Distance(; alg = LogDistance()), DistanceDistance(; alg = LogDistance()),
               Distance(; alg = VariationInfoDistance()),
               DistanceDistance(; alg = VariationInfoDistance())]
        # The same twelve with the power written out. `power = nothing` takes the first
        # power, so the two lists must give the same answer.
        desg = [Distance(; power = 1, alg = CanonicalDistance()),
                DistanceDistance(; power = 1, alg = CanonicalDistance()),
                Distance(; power = 1, alg = SimpleDistance()),
                DistanceDistance(; power = 1, alg = SimpleDistance()),
                Distance(; power = 1, alg = SimpleAbsoluteDistance()),
                DistanceDistance(; power = 1, alg = SimpleAbsoluteDistance()),
                Distance(; power = 1, alg = CorrelationDistance()),
                DistanceDistance(; power = 1, alg = CorrelationDistance()),
                Distance(; power = 1, alg = LogDistance()),
                DistanceDistance(; power = 1, alg = LogDistance()),
                Distance(; power = 1, alg = VariationInfoDistance()),
                DistanceDistance(; power = 1, alg = VariationInfoDistance())]
        offdiag = .!Matrix(LinearAlgebra.I(N))
        for (sim, file) in
            ((MaximumDistanceSimilarity(), "LoGo-MaximumDistanceSimilarity.csv.gz"),
             (ExponentialSimilarity(), "LoGo-ExponentialSimilarity.csv.gz"))
            logo_t = CSV.read(joinpath(@__DIR__, "./assets/", file), DataFrame)
            for i in eachindex(des)
                sigma1 = copy(sigma)
                sigma2 = copy(sigma)
                PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                          de = des[i],
                                                                                          sim = sim),
                                                                 sigma1, pr.X)
                PortfolioOptimisers.matrix_processing_algorithm!(PortfolioOptimisers.LoGo(;
                                                                                          de = desg[i],
                                                                                          sim = sim),
                                                                 sigma2, pr.X)
                MN = size(sigma1)
                res1 = isapprox(sigma1, reshape(logo_t[!, i], MN))
                if !res1
                    println("Fails on LoGo $(nameof(typeof(sim))) iteration $i")
                    find_tol(sigma1, reshape(logo_t[!, i], MN); name1 = :sigma,
                             name2 = :logo_t)
                end
                @test res1
                # The written-out power is the default power.
                @test isapprox(sigma1, sigma2)
                # A covariance matrix comes back, and the repair keeps it one.
                @test issymmetric(sigma1)
                @test isposdef(sigma1)
                # The sparsity of the inverse is the graph, plus the diagonal.
                D = distance(des[i], Sc, pr.X)
                A = PortfolioOptimisers.PMFG_T2s(PortfolioOptimisers.distance_to_similarity(sim;
                                                                                            D = D),
                                                 5)[1]
                J = inv(sigma1)
                off = (A .== 0) .& offdiag
                @test maximum(abs, J[off]) <= 1e-10 * maximum(abs, J)
            end
        end
        @test isnothing(PortfolioOptimisers.logo!(nothing))
    end
    # The three defects that the documentation sweep of the DBHT family, issue #469, found on
    # the `EqualRoot` path: #507, #508 and #509. That path is now
    # `src/11_Phylogeny/09_CliqueHierarchy.jl`. The reference implementation
    # is `DBHTs.m`, MATLAB Central File Exchange submission 46750 by Won-Min Song and Tomaso
    # Aste, and it carries all three. The papers it cites are Song, Di Matteo and Aste,
    # *Nested hierarchies in planar graphs*, Discrete Applied Mathematics 159 (2011)
    # 2135-2146, and Song, Di Matteo and Aste, *Hierarchical information clustering by means
    # of topologically embedded graphs*, PLoS ONE 7 (2012) e31929.
    @testset "The EqualRoot path of #507, #508 and #509" begin
        @testset "AdjCliq scores a candidate against itself alone (#507)" begin
            # The reproduction of #507. No pair of the three cliques shares two vertices, so
            # no pair is adjacent.
            CliqList = [1 2 3; 4 5 6; 1 4 7]
            Adj = PortfolioOptimisers.AdjCliq(zeros(7, 7), CliqList, [1, 2, 3])
            @test Adj == SparseArrays.spzeros(Int, 3, 3)
        end
        @testset "AdjCliq joins the candidates that share an edge" begin
            # Cliques 1 and 2 share the edge (1, 2). Clique 3 shares one vertex with each of
            # them, so it is adjacent to neither.
            CliqList = [1 2 3; 1 2 4; 1 5 6]
            Adj = PortfolioOptimisers.AdjCliq(zeros(6, 6), CliqList, [1, 2, 3])
            @test Adj == SparseArrays.sparse([1, 2], [2, 1], [1, 1], 3, 3)
            @test Adj == transpose(Adj)
            @test all(x -> x == 0 || x == 1, Adj)
        end
        @testset "AdjCliq indexes a column by the clique, not by the candidate" begin
            # `CliqRoot` names cliques 2 and 4, which share the edge (1, 2). Cliques 1 and 3
            # are not root candidates, so their rows and their columns stay empty.
            CliqList = [7 8 9; 1 2 3; 7 8 10; 1 2 4]
            Adj = PortfolioOptimisers.AdjCliq(zeros(10, 10), CliqList, [2, 4])
            @test Adj == SparseArrays.sparse([2, 4], [4, 2], [1, 1], 4, 4)
        end
        @testset "CliqueRoot(::EqualRoot, …) answers on one root candidate (#508)" begin
            # The reproduction of #508. Clique 1 is the root, and cliques 2 and 3 are its
            # children. One candidate has nothing to be joined to, so the answer is the
            # parent-child hierarchy alone.
            H = PortfolioOptimisers.CliqueRoot(EqualRoot(), [1], [0, 1, 1], 3, zeros(5, 5),
                                               [1 2 3; 1 2 4; 1 2 5])
            @test H == SparseArrays.sparse([2, 3, 1, 1], [1, 1, 2, 3], [1, 1, 1, 1], 3, 3)
        end
        @testset "BuildHierarchy stops the loop at a parent tie (#509)" begin
            # The reproduction of #509. Clique 1 holds the vertices {1, 2}. Cliques 2 and 3
            # are both supersets of it and both hold three vertices, so column 1 ties.
            M = SparseArrays.sparse([1 1 1; 1 1 1; 0 1 0; 0 0 1])
            @test PortfolioOptimisers.BuildHierarchy(M) == Int[]
        end
        @testset "BuildHierarchy builds the hierarchy when no parent ties" begin
            # Clique 1 holds {1, 2}, and clique 2 holds {1, 2, 3}, its only superset.
            M = SparseArrays.sparse([1 1; 1 1; 0 1])
            @test PortfolioOptimisers.BuildHierarchy(M) == [2, 0]
        end
    end
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
# structure `calc_adjacency` can build is connected -- a spanning tree or a PMFG -- so this is
# the only way to drive the unreachable branch through a verb that takes an **estimator**:
# `centrality_vector` and `phylogeny_matrix` do, and cannot be handed a graph. The separation
# kernels themselves take the structure, so they are driven directly below. It is an
# `AbstractNetworkEstimator` and not a `NetworkEstimator`, which is also what makes it
# evidence that `phylogeny_matrix` splits on the *separation* rather than on the estimator's
# own type. Defined at top level because a `@testset` body becomes a function, which cannot
# host a struct.
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
    # the components, and `is_related` rejects it without a repair. Every structure a shipped
    # estimator can build is connected -- a spanning tree or a PMFG -- so the disconnected one
    # is handed to the kernels directly, through their graph-taking entry points. No test
    # double subtypes `AbstractNetworkEstimator` to answer an internal generic any more;
    # `NetworkEstimator()` stands in the inert estimator slots.
    PO = PortfolioOptimisers
    SWG = PortfolioOptimisers.SimpleWeightedGraphs
    W = [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
    nte = NetworkEstimator()
    gw = SWG.SimpleWeightedGraph(W)
    want = [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]

    d = separation_matrix(PathLength(), gw)
    @test d[1, 3] == Inf
    @test separation_budget(PathLength(), nte, d) == 1.0
    @test !PO.is_reachable(PathLength(), d[1, 3])
    @test PO.is_reachable(PathLength(), d[1, 2])

    @test PO._phylogeny_matrix(PathLength(), nte, gw) == want
    # The clamp to the observed diameter excludes the sentinel, so even a budget far above
    # the diameter cannot reach across a component.
    @test PO._phylogeny_matrix(PathLength(; dmax = 100.0), nte, gw) == want
    # A hop count agrees on this graph, which is what makes the comparison meaningful: the
    # two separations differ on the budget, not on reachability. It reads the same structure
    # binarised, which is `separation_graph`'s graph-taking entry point.
    gh = PO.separation_graph(HopCount(), gw)
    @test PO._phylogeny_matrix(HopCount(; n = 3), nte, gh) == want
    # And the sentinel a hop count reports is the one `isfinite` alone would let through.
    @test separation_matrix(HopCount(), gh)[1, 3] == typemax(Int)
    @test !PO.is_reachable(HopCount(), typemax(Int))
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

    #=
    And once is once with a budget *rule* too. A rule needs the structure to answer, so a
    rule asked to derive its own would spend this method's saving right back -- the second
    full derivation the two-argument entry point exists to prevent, inside the very function
    that avoided it. `separation_graph` is what closes that: the structure is built once and
    handed to `resolve_separation`.
    =#
    for alg in (KruskalTree(), ComplementSimilarity())
        de = CountingDistance(Distance(; alg = CanonicalDistance()))
        nte = NetworkEstimator(; de = de, alg = alg,
                               sep = HopCount(; n = HopCountQuantile(; q = 0.5)))
        clusterise(NetworkClustersEstimator(; nte = nte), Xc)
        @test de.n_cor_and_dist == 1
        @test de.n_distance == 0
    end
end

@testset "phylogeny_matrix and phylogeny_features build one structure per call" begin
    #=
    The same guard on the other two consumers of a network, over both separations and over
    both halves of the dial: a stated budget builds one structure, and a budget rule builds
    the same one rather than a second. `phylogeny_features` reaches the separation kernels
    through `Proximity`, so it is checked here beside `phylogeny_matrix` rather than by its
    own counting fixture.
    =#
    Xc = randn(StableRNG(123456789), 200, 10)
    seps = (HopCount(; n = 2), HopCount(; n = HopCountQuantile(; q = 0.5)), PathLength(),
            PathLength(; dmax = PathLengthQuantile(; q = 0.5)))
    for alg in (KruskalTree(), ComplementSimilarity()), sep in seps
        de = CountingDistance(Distance(; alg = CanonicalDistance()))
        nte = NetworkEstimator(; de = de, alg = alg, sep = sep)
        phylogeny_matrix(nte, Xc)
        # A tree branch derives `D` through `distance`, a PMFG branch derives both through
        # `cor_and_dist`, so the sum is what the count is about.
        @test de.n_cor_and_dist + de.n_distance == 1
        de2 = CountingDistance(Distance(; alg = CanonicalDistance()))
        nte2 = NetworkEstimator(; de = de2, alg = alg, sep = sep)
        phylogeny_features(Proximity(), nte2, Xc)
        @test de2.n_cor_and_dist + de2.n_distance == 1
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

    @testset "A second weighting channel is refused on the tree family" begin
        #=
        The tree family carries the same two splat fields, and every channel they can reach
        re-weights a graph that is already weighted. Unlike the centrality case the calls
        SUCCEED, so the cost is a silent wrong result rather than a crash: a matrix binds to
        `distmx`, a vector binds to `kruskal_mst`'s `weight_vector`, and `minimize = false`
        yields a MAXIMUM spanning tree that everything downstream still reads as a minimum
        one. None of the three functions declares a positional argument that is not a weight,
        so both shapes are refused outright.
        =#
        D = zeros(3, 3)
        for T in (KruskalTree, BoruvkaTree, PrimTree)
            @test_throws PortfolioOptimisers.ConflictingArgumentError T(; args = (D,))
            @test_throws PortfolioOptimisers.ConflictingArgumentError T(;
                                                                        args = (zeros(3),))
            @test_throws PortfolioOptimisers.ConflictingArgumentError T(; args = (1:3,))
            @test_throws PortfolioOptimisers.ConflictingArgumentError T(; args = (2, D))
            for kw in ((; minimize = false), (; minimize = true))
                @test_throws PortfolioOptimisers.ConflictingArgumentError T(; kwargs = kw)
            end
            @test isa(T(), T)
            @test isa(T(; args = (), kwargs = (;)), T)
        end
    end

    @testset "kwargs needs no guard on the centrality family" begin
        #=
        A keyword binds by NAME, so a matrix in `kwargs` can never reach the positional slot
        the `args` guard was written for. None of the four functions declares a matrix-valued
        keyword and each of them refuses one on its own, so the family fails closed at the
        call without a guard. Constructing is deliberately left alone.
        =#
        D = zeros(6, 6)
        cc = PortfolioOptimisers.calc_centrality
        g = PortfolioOptimisers.Graphs.SimpleGraph(PortfolioOptimisers.Graphs.grid((3, 2)))
        for (T, kw) in
            ((BetweennessCentrality, (; distmx = D)), (ClosenessCentrality, (; distmx = D)),
             (StressCentrality, (; distmx = D)), (DegreeCentrality, (; normalize = D)),
             (BetweennessCentrality, (; normalize = D)))
            ct = T(; kwargs = kw)
            @test isa(ct, T)
            @test_throws Union{MethodError, TypeError} cc(ct, g)
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

# Rules for a separation budget. Defined at top level because a `@testset` body becomes a
# function, which cannot host a struct.
struct ConstantHops{T} <: PortfolioOptimisers.HopCountAlgorithm
    n::T
end
function (r::ConstantHops)(nte, X, g; dims::Int = 1, kwargs...)
    return r.n
end
struct ConstantRadius{T} <: PortfolioOptimisers.PathLengthAlgorithm
    dmax::T
end
function (r::ConstantRadius)(nte, X, g; dims::Int = 1, kwargs...)
    return r.dmax
end
# Records what it was handed, so the argument contract is asserted rather than assumed.
mutable struct RecordingHops <: PortfolioOptimisers.HopCountAlgorithm
    calls::Int
    nte::Any
    size::Any
    dims::Int
    g::Any
end
RecordingHops() = RecordingHops(0, nothing, nothing, 0, nothing)
function (r::RecordingHops)(nte, X, g; dims::Int = 1, kwargs...)
    r.calls += 1
    r.nte = nte
    r.size = size(X)
    r.dims = dims
    r.g = g
    return 2
end

@testset "A separation budget may be a rule" begin
    using PortfolioOptimisers, Test, CSV, DataFrames, TimeSeries, LinearAlgebra
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(EmpiricalPrior(), rd)
    X = pr.X
    nte = NetworkEstimator()
    n_assets = size(X, 2)
    n_pairs = n_assets * (n_assets - 1)
    related(sep, Xs = X) = count(!iszero,
                                 phylogeny_matrix(NetworkEstimator(; sep = sep), Xs).X)

    @testset "A stated budget passes through untouched" begin
        # The fallback on the abstract type is an identity, so an extension inherits the
        # kernel and a stated budget costs nothing.
        for sep in (HopCount(), HopCount(; n = 4), PathLength(), PathLength(; dmax = 1.5))
            @test resolve_separation(sep, nte, X) === sep
        end
    end

    @testset "A rule is stored uncalled and resolved at the point of use" begin
        rec = RecordingHops()
        sep = HopCount(; n = rec)
        # Construction does not run it.
        @test rec.calls == 0
        @test sep.n === rec

        resolved = resolve_separation(sep, nte, X)
        @test rec.calls == 1
        @test resolved == HopCount(; n = 2)
        # The rule is handed the estimator that owns it, the data the network is about to be
        # built from, and the structure already built from them -- which is what lets it read
        # the network without paying for a second one.
        @test rec.nte === nte
        @test rec.size == size(X)
        @test rec.dims == 1
        @test rec.g == PortfolioOptimisers.separation_graph(sep, nte, X)
        # It is the structure the separation measures over, which for a hop count is the
        # binarised one: a weighted graph would make `A^i` sum products of distances.
        @test isa(rec.g, PortfolioOptimisers.Graphs.SimpleGraph)
        @test PortfolioOptimisers.Graphs.adjacency_matrix(rec.g) ==
              PortfolioOptimisers.calc_adjacency(nte, X; dims = 1)
        # And the graph-taking form is the one the wrapper delegates to.
        @test resolve_separation(HopCount(; n = RecordingHops()), nte, X,
                                 PortfolioOptimisers.separation_graph(sep, nte, X)) ==
              HopCount(; n = 2)
    end

    @testset "A rule answers exactly as the value it returns" begin
        # The point of the widening: a rule is a deferred way of writing the same budget,
        # not a second notion of one.
        @test related(HopCount(; n = ConstantHops(3))) == related(HopCount(; n = 3))
        @test related(PathLength(; dmax = ConstantRadius(1.5))) ==
              related(PathLength(; dmax = 1.5))
        # A bare `Function` is admitted in the same field.
        @test related(HopCount(; n = (nte, X, g; kwargs...) -> 3)) ==
              related(HopCount(; n = 3))
        @test related(PathLength(; dmax = (nte, X, g; kwargs...) -> 1.5)) ==
              related(PathLength(; dmax = 1.5))
    end

    @testset "The return value is checked, and then validated" begin
        # A functor's return type is not part of its signature, so the `Integer` obligation
        # is a run-time check. It is not cosmetic: three readers use `0:n` as a
        # matrix-power count, where `0:1.5` drops a power silently.
        @test_throws ArgumentError resolve_separation(HopCount(; n = ConstantHops(1.5)),
                                                      nte, X)
        @test_throws ArgumentError resolve_separation(HopCount(; n = ConstantHops(nothing)),
                                                      nte, X)
        # `nothing` is a stated budget rather than a computed one, so a path length rule
        # may not answer with it.
        @test_throws ArgumentError resolve_separation(PathLength(;
                                                                 dmax = ConstantRadius(nothing)),
                                                      nte, X)
        @test_throws ArgumentError resolve_separation(PathLength(;
                                                                 dmax = ConstantRadius("1")),
                                                      nte, X)
        # Resolution goes back through the ordinary constructor, so a rule's answer meets
        # exactly the validation a stated budget meets.
        @test_throws DomainError resolve_separation(HopCount(; n = ConstantHops(0)), nte, X)
        @test_throws DomainError resolve_separation(PathLength(;
                                                               dmax = ConstantRadius(-1.0)),
                                                    nte, X)
        # Storing those same rules is fine. The check belongs where the value exists.
        @test HopCount(; n = ConstantHops(0)) isa HopCount
        @test PathLength(; dmax = ConstantRadius(-1.0)) isa PathLength
    end

    @testset "separation_budget refuses an unresolved separation" begin
        # It takes the separation matrix rather than the data, deliberately, so it is the
        # one kernel that cannot resolve a rule. Returning the rule would put a function
        # where every caller expects a number.
        d = separation_matrix(HopCount(), nte, X)
        @test_throws ArgumentError separation_budget(HopCount(; n = ConstantHops(2)), nte,
                                                     d)
        @test_throws ArgumentError separation_budget(PathLength(;
                                                                dmax = ConstantRadius(1.5)),
                                                     nte, d)
        # A resolved one answers as before.
        @test separation_budget(resolve_separation(HopCount(; n = ConstantHops(2)), nte, X),
                                nte, d) == 2
    end

    @testset "Every consumer of a network resolves" begin
        rule = HopCount(; n = ConstantHops(3))
        stated = HopCount(; n = 3)

        @test phylogeny_matrix(NetworkEstimator(; sep = rule), X).X ==
              phylogeny_matrix(NetworkEstimator(; sep = stated), X).X

        # Both `clusterise` branches read `n` as a matrix-power count.
        for alg in (KruskalTree(), MaximumDistanceSimilarity())
            r = clusterise(NetworkClustersEstimator(;
                                                    nte = NetworkEstimator(; alg = alg,
                                                                           sep = rule)), X)
            s = clusterise(NetworkClustersEstimator(;
                                                    nte = NetworkEstimator(; alg = alg,
                                                                           sep = stated)),
                           X)
            @test r.k == s.k
            @test PortfolioOptimisers.assignments(r) == PortfolioOptimisers.assignments(s)
        end

        # The graded feature producer reads the budget through `separation_budget`.
        prule = PathLength(; dmax = ConstantRadius(1.5))
        @test PortfolioOptimisers.phylogeny_features(Proximity(),
                                                     NetworkEstimator(; sep = prule), X) ==
              PortfolioOptimisers.phylogeny_features(Proximity(),
                                                     NetworkEstimator(;
                                                                      sep = PathLength(;
                                                                                       dmax = 1.5)),
                                                     X)

        # Centrality reaches the separation through the unweighted route's phylogeny matrix.
        @test centrality_vector(CentralityEstimator(; pl = NetworkEstimator(; sep = rule)),
                                X).X == centrality_vector(CentralityEstimator(;
                                                pl = NetworkEstimator(; sep = stated)),
                            X).X
    end

    @testset "The quantile rules place the budget by cardinality" begin
        # `q` is the share of the reachable off-diagonal pairs the budget relates, so a
        # continuous budget lands on it and a discrete one cannot.
        for q in (0.1, 0.25, 0.5)
            share = related(PathLength(; dmax = PathLengthQuantile(; q = q))) / n_pairs
            @test isapprox(share, q; atol = 0.01)
        end
        # The hop rule rounds to a shell, and three values of `q` collapse onto one budget
        # here. That is the unit, not a defect.
        ns = [resolve_separation(HopCount(; n = HopCountQuantile(; q = q)), nte, X).n
              for q in (0.1, 0.2, 0.25)]
        @test all(isa.(ns, Integer))
        @test allequal(ns)
        # Wider quantiles are never narrower budgets.
        dmaxs = [resolve_separation(PathLength(; dmax = PathLengthQuantile(; q = q)), nte,
                                    X).dmax for q in 0.1:0.1:0.9]
        @test issorted(dmaxs)
        @test_throws DomainError PathLengthQuantile(; q = 1.5)
        @test_throws DomainError HopCountQuantile(; q = -0.1)
    end

    @testset "A quantile rule holds the pair count still across folds" begin
        # The motivating case. A `dmax` tuned on the whole sample is a different constraint
        # strength on every fold; the rule fixes the strength and moves the radius instead.
        folds = [1:63, 64:126, 127:189, 190:252]
        fixed = [related(PathLength(; dmax = 1.0107), X[f, :]) for f in folds]
        ruled = [related(PathLength(; dmax = PathLengthQuantile(; q = 0.25)), X[f, :])
                 for f in folds]
        radii = [resolve_separation(PathLength(; dmax = PathLengthQuantile(; q = 0.25)),
                                    nte, X[f, :]).dmax for f in folds]
        @test !allequal(fixed)
        @test allequal(ruled)
        @test !allequal(radii)
    end

    @testset "The quantile population excludes the diagonal and the sentinel" begin
        # Two disjoint edges. Every reachable off-diagonal separation is one hop, so any
        # quantile of them is `1`. Admitting `typemax(Int)` would blow the budget up, and
        # admitting the zero diagonal would drag it down.
        W = [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
        X4 = zeros(Float64, 3, 4)
        pl = FixedDistanceGraph(W, HopCount())
        d = separation_matrix(HopCount(), pl, X4)
        @test maximum(d) == typemax(Int)
        for q in (0.0, 0.5, 1.0)
            @test PortfolioOptimisers.separation_quantile(HopCount(), d, q) == 1
        end
        dp = separation_matrix(PathLength(), pl, X4)
        @test maximum(dp) == Inf
        @test PortfolioOptimisers.separation_quantile(PathLength(), dp, 1.0) == 1.0
        # A structure with no reachable pair of distinct assets has no population.
        @test_throws ArgumentError PortfolioOptimisers.separation_quantile(HopCount(),
                                                                           zeros(Int, 1, 1),
                                                                           0.5)
    end
end
@testset "A zero similarity is an absent edge, not a weak one" begin
    using PortfolioOptimisers, Test, LinearAlgebra, StatsBase
    #=
    ADR 0049 admits an exactly zero similarity on the PMFG path, on the ground that
    `PMFG_T2s`'s gain argmax is shift-invariant. That holds for the argmax and fails for
    the graph: `PMFG_T2s` ends with `A = W ⊙ ((A + A') .== 1)`, so a zero weight is an
    *absent* edge rather than a weak one, and it inserts every remaining vertex whatever
    the gain, so it declines none. Before `assert_pmfg_weights` the run carried the
    shrunken graph as far as `turn_into_Hclust_merges` and died there with a `BoundsError`
    about a matrix index.

    The fixture is 12 assets over 16 observations, built from a Hadamard matrix so that
    many sample correlations are *exactly* zero. No random number is involved: the zeros
    are structural. `LogDistance` maps them to `Inf`, and `ExponentialSimilarity` maps an
    `Inf` to `exp(-Inf)`, which is `0` exactly.
    =#
    H = [1 1 1 1 1 1 1 1; 1 -1 1 -1 1 -1 1 -1; 1 1 -1 -1 1 1 -1 -1; 1 -1 -1 1 1 -1 -1 1;
         1 1 1 1 -1 -1 -1 -1; 1 -1 1 -1 -1 1 -1 1; 1 1 -1 -1 -1 -1 1 1;
         1 -1 -1 1 -1 1 1 -1]
    Xa = Float64.(vcat(H, H))
    Xz = hcat(Xa[:, 2:8], [Xa[:, 2] .* 0.9 .+ 0.05 .* Xa[:, k] for k in 3:7]...)
    rho = cor(Xz)
    Dz = max.(-log.(abs.(rho)), zero(eltype(rho)))
    Sz = exp.(-Dz)
    N = size(Xz, 2)
    @test count(isinf, Dz) == 80
    @test count(iszero, Sz) == 80

    # The same distances with the infinities finite, so the similarity stays strictly
    # positive. This is the isolation: the zero is the cause, not the `Inf`.
    Dp = copy(Dz)
    Dp[isinf.(Dp)] .= 20.0
    Sp = exp.(-Dp)
    @test all(>(0), Sp)

    #=
    A PMFG on `N` vertices has `3N - 6` edges. The zeros cost this one several of its
    thirty, and the shortfall is what the guard reports.

    Assert the SHORTFALL, never a particular count. `PMFG_T2s` inserts by maximum gain, and
    the gains here are read off correlations whose non-zero entries differ in their last
    bits from one machine to the next, so a near-tie resolves differently and the surviving
    edge count moves with it. CI has produced both 20 and 21 on this same fixture, at jobs
    96256586958 and 96273304920, with no commit between them touching the computation. The
    count is not the decision under test. The decision is that an exactly zero weight is an
    absent edge, so the structure falls short.

    The bound is not vacuous. 80 of the 132 off-diagonal entries are zero, which is 40 of
    the 66 pairs, so at most 26 edges can carry a weight and 30 is unreachable by
    construction.
    =#
    edges_z = count(!iszero, PortfolioOptimisers.PMFG_T2s(Sz)[1]) ÷ 2
    @test edges_z < 3 * N - 6
    @test count(!iszero, PortfolioOptimisers.PMFG_T2s(Sp)[1]) ÷ 2 == 3 * N - 6

    #=
    Every site that consumes the weighted structure refuses it.

    `pdm = nothing` is load-bearing, and this is the second thing the fixture cannot leave
    to the host. `rho` is singular to working precision -- `eigmin(rho)` is about -8.3e-16
    -- so whether `isposdef` succeeds is a coin toss across machines. Where it fails,
    `Posdef`'s default `NearestCorrelationMatrix.Newton` step repairs the matrix and moves
    EVERY entry: measured here, `count(iszero, rho)` goes from 78 to 0. The similarity is
    then strictly positive, the graph is a full PMFG, and these three sites throw nothing.
    That is what CI job 96256586958 recorded -- `clusterise`, `calc_weighted_adjacency_graph`
    and `calc_distance_weighted_graph` all reported "No exception thrown" on the same run
    that produced 20 edges.

    Switching the repair off is what makes the input reach the guard on every host. It also
    states the test's subject exactly: the guard is about a zero weight, not about whether a
    fixture happens to survive a Cholesky.
    =#
    ce0 = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; pdm = nothing))
    nte = NetworkEstimator(; ce = ce0, de = Distance(; alg = LogDistance()),
                           alg = ExponentialSimilarity())
    cle = ClustersEstimator(; ce = ce0, de = Distance(; alg = LogDistance()),
                            alg = DBHT(; sim = ExponentialSimilarity()))
    # The zeros survive the estimator's own correlation, which is the precondition the three
    # refusals below rest on. Assert it, so a regression names its cause.
    @test count(iszero, PortfolioOptimisers.cor(ce0, Xz)) == 80
    @test_throws DomainError PortfolioOptimisers.DBHTs(Dz, Sz)
    @test_throws DomainError clusterise(cle, Xz)
    @test_throws DomainError PortfolioOptimisers.calc_weighted_adjacency_graph(ExponentialSimilarity(),
                                                                               Sz)
    @test_throws DomainError PortfolioOptimisers.calc_weighted_adjacency_graph(nte, Xz)
    @test_throws DomainError PortfolioOptimisers.calc_distance_weighted_graph(nte, Xz)

    # The message names the shortfall, not the symptom.
    raised(f) =
        try
            f()
            nothing
        catch e
            e
        end
    err = raised(() -> PortfolioOptimisers.DBHTs(Dz, Sz))
    @test err isa DomainError
    msg = sprint(showerror, err)
    # `edges => …` carries the same host-dependent count as the assertion above, so read it
    # from the same computation rather than writing a number. `3 * N - 6` is structural.
    @test occursin("edges => $edges_z", msg)
    @test occursin("3 * N - 6 => 30", msg)
    @test occursin("of its edges and the structure is not a PMFG", msg)

    #=
    It also names as much of the configuration as the site holds, which is what
    `assert_similarity_domain` does one step earlier. `DBHTs` called with the matrices
    alone knows neither half, `clusterise` forwards the similarity, and
    `calc_distance_weighted_graph` holds both.
    =#
    @test !occursin("must hold for", msg)
    @test occursin("must hold for ExponentialSimilarity. Got",
                   sprint(showerror, raised(() -> clusterise(cle, Xz))))
    both = sprint(showerror,
                  raised(() -> PortfolioOptimisers.calc_distance_weighted_graph(nte, Xz)))
    #=
    Julia decides at `show` time whether a type name is qualified. It reads visibility
    from `Base.active_module()`, and it applies the one decision to every name in the
    type. So the message reads `Distance{Nothing, LogDistance}` where both names are
    visible, and `PortfolioOptimisers.Distance{Nothing, PortfolioOptimisers.LogDistance}`
    where neither is. Neither spelling is a substring of the other, so do not write one.
    Render the expectation with the same renderer that wrote the message.
    =#
    @test occursin("for ExponentialSimilarity, from", both)
    @test occursin(string(typeof(nte.de)), both)

    # The strictly positive counterpart runs, with the infinite distances left in place.
    @test PortfolioOptimisers.DBHTs(Dp, Sp) isa Tuple
    @test PortfolioOptimisers.DBHTs(Dz, Sp) isa Tuple

    #=
    `logo!` is the fourth `PMFG_T2s` caller and is deliberately not guarded. It reads
    separators and cliques, which `PMFG_T2s` derives from the insertion order rather than
    from `A`, so a zero weight does not shrink them and refusing it would refuse a
    configuration that works.
    =#
    c3z, cqz = PortfolioOptimisers.PMFG_T2s(Sz, 4)[3:4]
    c3p, cqp = PortfolioOptimisers.PMFG_T2s(Sp, 4)[3:4]
    @test size(c3z) == size(c3p) == (N - 4, 3)
    @test size(cqz) == size(cqp) == (N - 3, 4)
end
@testset "An integer phylogeny carries no scale of its own" begin
    using PortfolioOptimisers, Test, StableRNGs
    #=
    `IntegerPhylogeny` and `IntegerPhylogenyEstimator` each carried a `scale` field, a
    big-M holdover from before the model owned the constraint scale. Nothing read it:
    `set_iplg_constraints!` takes `sc = get_constraint_scale(model)` and emits
    `sc * (A * ib ⊖ B) <= 0`, so the field reached no constraint. Measured before its
    removal, `scale = 100_000.0`, `scale = 1.0` and `scale = -7.5` gave bit-identical
    weights on a minimum-conditional-value-at-risk model, and the constructor validated
    nothing there, so the negative value constructed. Issue #395 removed the field.

    The field set is pinned here rather than a single `scale` absence, so that any field
    added to either type is a deliberate edit to this test. The polarity that matters is
    a second, *private* scale returning beside the model-wide one that every other
    constraint family shares.
    =#
    @test fieldnames(IntegerPhylogenyEstimator) == (:pl, :B)
    @test fieldnames(IntegerPhylogeny) == (:A, :B)
    # The keyword is gone from both keyword constructors.
    @test_throws MethodError IntegerPhylogenyEstimator(; pl = NetworkEstimator(), B = 1,
                                                       scale = 100_000.0)
    @test_throws MethodError IntegerPhylogeny(; A = [0.0 1.0; 1.0 0.0], B = 2,
                                              scale = 100_000.0)
    # The positional constructors take the two fields and no third.
    @test isa(IntegerPhylogenyEstimator(NetworkEstimator(), 1), IntegerPhylogenyEstimator)
    @test isa(IntegerPhylogeny([0.0 1.0; 1.0 0.0], 2), IntegerPhylogeny)
    # The refit from estimator to result carries only what the constraint reads.
    X = randn(StableRNG(987654321), 200, 5)
    ip = phylogeny_constraints(IntegerPhylogenyEstimator(; pl = NetworkEstimator(), B = 1),
                               X)
    @test isa(ip, IntegerPhylogeny)
    @test propertynames(ip) == (:A, :B)
    @test ip.B == 1
end
@testset "A two-asset cluster keeps the gap series finite" begin
    using PortfolioOptimisers, Test, StableRNGs, StatsBase
    # `Clustering` is not in `runtests.jl`'s `init_code`, and an in-block `using` does not
    # bind the sandbox module, so reach `cutree` through the package.
    cutree = PortfolioOptimisers.Clustering.cutree
    # A cluster of exactly two assets carries one pairwise distance, and a single value has
    # no corrected standard deviation. `StandardisedValue` divides by one in that case, so
    # the cluster contributes its mean pairwise distance instead of a `NaN`.
    rng = StableRNG(987654321)
    X = randn(rng, 400, 20)
    X[:, 2] = X[:, 1] + 0.02 * randn(rng, 400)
    X[:, 4] = X[:, 3] + 0.02 * randn(rng, 400)
    clr = clusterise(ClustersEstimator(), X)
    res, D = clr.res, clr.D
    # Every cut the statistic scores carries at least one two-asset cluster.
    @test all(2 in counts(cutree(res; k = k)) for k in 2:6)
    # Rebuild the two-difference gap series the estimator maximises.
    function gap_series(alg, res, D)
        c1 = min(floor(Int, sqrt(size(D, 1))) + 2, size(D, 1))
        W = Vector{Float64}(undef, c1)
        W[1] = typemin(Float64)
        for i in 2:c1
            lvl = cutree(res; k = i)
            W[i] = sum(1:maximum(lvl)) do j
                idx = lvl .== j
                sub = D[idx, idx]
                M = size(sub, 1)
                return if M < 2
                    0.0
                else
                    PortfolioOptimisers.vec_to_real_measure(alg,
                                                            [sub[r, c] for c in 1:M
                                                             for r in (c + 1):M])
                end
            end
        end
        return W[1:(end - 2)] + W[3:end] - 2 * W[2:(end - 1)]
    end
    gaps = gap_series(StandardisedValue(), res, D)
    # The first entry is the deliberate `typemin` sentinel. The rest are finite.
    @test all(isfinite, gaps[2:end])
    # The selected count is a real maximiser of that series, not the fallback.
    k = PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                          alg = SecondOrderDifference()),
                                                    res, D)
    @test k == argmax(gaps)
    @test k == clr.k
end

# The claims `src/11_Phylogeny/02_Clusters.jl` makes about the two selection scores, about the
# `linkage` symbol and about the `Clusters` constructor. Defined at top level because a
# `@testset` body becomes a function, and the sweep of #467 reads these numbers.
@testset "The clustering vocabulary states what it does (#467)" begin
    using PortfolioOptimisers, Test, StableRNGs, StatsBase
    @testset "The two dispersion measures answer 6 and 4 on the named sample" begin
        # `SecondOrderDifference`'s docstring names this sample and these two numbers. The
        # source takes the mean of a cluster's pairwise distances; the default standardises
        # it, which is a different statistic and selects a different count.
        X = randn(StableRNG(987654321), 400, 40)
        k(g) = clusterise(ClustersEstimator(;
                                            onc = OptimalNumberClusters(;
                                                                        alg = SecondOrderDifference(;
                                                                                                    alg = g))),
                          X).k
        @test k(MeanValue()) == 6
        @test k(StandardisedValue()) == 4
        # 6 is the ceiling itself, so the two differ inside the admissible range.
        @test floor(Int, sqrt(size(X, 2))) == 6
    end
    @testset "`linkage` is unchecked at construction and raises at use" begin
        # The set of criteria belongs to `Clustering.hclust`, so the constructor takes any
        # `Symbol` and the raise lands inside `clusterise`.
        alg = HClustAlgorithm(; linkage = :nonsense)
        @test alg.linkage === :nonsense
        X = randn(StableRNG(987654321), 200, 10)
        @test_throws ArgumentError clusterise(ClustersEstimator(; alg = alg), X)
        # A criterion the package does accept clusters without a raise.
        @test clusterise(ClustersEstimator(; alg = HClustAlgorithm(; linkage = :average)),
                         X) isa Clusters
    end
    @testset "Every `Clusters` validation arm throws" begin
        res = clusterise(ClustersEstimator(), randn(StableRNG(1), 50, 3)).res
        S = [1.0 0.5; 0.5 1.0]
        D = [0.0 0.5; 0.5 0.0]
        @test_throws PortfolioOptimisers.IsEmptyError Clusters(; res = res, S = zeros(0, 0),
                                                               D = D, k = 1)
        @test_throws PortfolioOptimisers.IsEmptyError Clusters(; res = res, S = S,
                                                               D = zeros(0, 0), k = 1)
        @test_throws DimensionMismatch Clusters(; res = res, S = S, D = ones(3, 3), k = 1)
        @test_throws PortfolioOptimisers.IsEmptyError Clusters(; res = res, S = S, D = D,
                                                               P = zeros(0, 0), k = 1)
        # A `P` of the wrong size is refused against `S`, which `D` already matches.
        @test_throws DimensionMismatch Clusters(; res = res, S = S, D = D, P = ones(3, 3),
                                                k = 1)
        @test_throws DomainError Clusters(; res = res, S = S, D = D, k = 0)
        # `P` is optional, and both shapes build.
        @test isnothing(Clusters(; res = res, S = S, D = D, k = 1).P)
        @test Clusters(; res = res, S = S, D = D, P = D, k = 1).P === D
        @test_throws DomainError OptimalNumberClusters(; max_k = 0)
        @test_throws DomainError OptimalNumberClusters(; alg = 0)
    end
    @testset "`factory` is the identity on a clustering algorithm" begin
        w = pweights(fill(1 / 400, 400))
        alg = HClustAlgorithm(; linkage = :average)
        # An algorithm carries no prior-dependent field, so the same object comes back.
        @test factory(alg, w) === alg
        dbht = DBHT()
        @test factory(dbht, w) === dbht
        # `OptimalNumberClusters` is an estimator, so it rebuilds and propagates `alg`.
        onc = OptimalNumberClusters(; max_k = 5,
                                    alg = SecondOrderDifference(; alg = MeanValue()))
        fonc = factory(onc, w)
        @test fonc.max_k == 5
        @test fonc.alg.alg.w === w
    end
end

# Two traversal strategies whose property is not the node's `id`, for the element type
# `pre_order` collects into (#510). Defined at top level because a `@testset` body becomes a
# function, which cannot host a struct.
struct PreorderTreeByHeight <: PortfolioOptimisers.AbstractPreorderBy end
function PortfolioOptimisers.get_node_property(::PreorderTreeByHeight,
                                               a::PortfolioOptimisers.ClusterNode)
    return a.height
end
struct PreorderTreeByLabel <: PortfolioOptimisers.AbstractPreorderBy end
function PortfolioOptimisers.get_node_property(::PreorderTreeByLabel,
                                               a::PortfolioOptimisers.ClusterNode)
    return string("asset_", a.id)
end
# The claims `src/11_Phylogeny/03_Hierarchical.jl` and
# `src/11_Phylogeny/05_NonHierarchicalClustering.jl` make about the tree, about the search
# that replaces an invalid cut, and about the flat partition. Defined at top level because a
# `@testset` body becomes a function, and the sweep of #468 reads these numbers.
@testset "The dendrogram search and its rejection path (#468)" begin
    using PortfolioOptimisers, Test, Clustering, StableRNGs, StatsBase, LinearAlgebra

    # A balanced binary dendrogram whose merges tie level by level. `Clustering.cutree` and
    # `sortperm` then disagree about which merges a cut removes, and that disagreement is the
    # only way `validate_k_value` can answer `false`.
    function balanced_hclust(m)
        n = 2^m
        merges = Matrix{Int}(undef, n - 1, 2)
        heights = Vector{Float64}(undef, n - 1)
        row = 0
        ids = Int[]
        for i in 1:2:(n - 1)
            row += 1
            merges[row, 1] = -i
            merges[row, 2] = -(i + 1)
            heights[row] = 1.0
            push!(ids, row)
        end
        lvl = 2
        while length(ids) > 1
            nxt = Int[]
            for i in 1:2:(length(ids) - 1)
                row += 1
                merges[row, 1] = ids[i]
                merges[row, 2] = ids[i + 1]
                heights[row] = Float64(lvl)
                push!(nxt, row)
            end
            ids = nxt
            lvl += 1
        end
        return Hclust(merges, heights, collect(1:n), :ward)
    end
    function tall_first(res)
        nds = PortfolioOptimisers.to_tree(res)[2]
        return nds[sortperm([i.height for i in nds]; rev = true)]
    end

    @testset "`to_tree` numbers the leaves first and returns the root last" begin
        res8 = balanced_hclust(3)
        root, nds8 = PortfolioOptimisers.to_tree(res8)
        @test length(nds8) == 2 * 8 - 1
        @test [n.id for n in nds8] == collect(1:15)
        @test root === nds8[end]
        @test root.level == 8
        @test PortfolioOptimisers.pre_order(root) == collect(1:8)
        @test length(PortfolioOptimisers.pre_order(root)) == root.level
        @test PortfolioOptimisers.is_leaf(nds8[1])
        @test !PortfolioOptimisers.is_leaf(root)
        # A node given children takes `left.level + right.level` and ignores the argument.
        @test ClusterNode(9, nds8[1], nds8[2], 1.0, 99).level == 2
        @test PortfolioOptimisers.get_node_property(PreorderTreeByID(), nds8[3]) == 3
    end

    @testset "`pre_order` collects the strategy's own type (#510)" begin
        res8 = balanced_hclust(3)
        root, _ = PortfolioOptimisers.to_tree(res8)
        # The default strategy is unmoved: the walk still returns a `Vector{Int}`.
        @test PortfolioOptimisers.pre_order(root) isa Vector{Int}
        # A `Float64` property is collected as a `Float64`. `to_tree` gives every leaf a
        # height of `0.0`, which a `Vector{Int}` took silently.
        hs = PortfolioOptimisers.pre_order(root, PreorderTreeByHeight())
        @test hs isa Vector{Float64}
        @test hs == zeros(8)
        # A fractional leaf height is kept whole. A `Vector{Int}` raised an `InexactError`
        # here, and truncated a whole-valued height without a word.
        l1 = ClusterNode(1, nothing, nothing, 0.25, 1)
        l2 = ClusterNode(2, nothing, nothing, 0.75, 1)
        top = ClusterNode(3, l1, l2, 1.5, 2)
        @test PortfolioOptimisers.pre_order(top, PreorderTreeByHeight()) == [0.25, 0.75]
        # A `String` property no longer raises. The two visited sets are keyed on the node's
        # `id`, which is what they store, and not on the property.
        ls = PortfolioOptimisers.pre_order(root, PreorderTreeByLabel())
        @test ls isa Vector{String}
        @test ls == string.("asset_", 1:8)
    end

    @testset "Only a tie in the heights makes a cut invalid" begin
        # 160 dendrograms with distinct heights, and not one rejected count among them.
        rejected = 0
        for seed in 1:40, link in (:ward, :single, :complete, :average)
            rng = StableRNG(seed)
            Xr = randn(rng, 60, 16)
            Xr[:, 2] = Xr[:, 1] + 0.01 * randn(rng, 60)
            Xr[:, 4] = Xr[:, 3] + 0.01 * randn(rng, 60)
            _, Dr = cor_and_dist(Distance(), PortfolioOptimisersCovariance(), Xr)
            rr = hclust(Dr; linkage = link, branchorder = :optimal)
            nr = tall_first(rr)
            rejected += count(k -> !PortfolioOptimisers.validate_k_value(rr, nr, k), 1:6)
        end
        @test rejected == 0

        # The tied tree rejects. `k = 1` and `k = N` stay valid.
        res8 = balanced_hclust(3)
        nodes8 = tall_first(res8)
        @test [k for k in 1:8 if PortfolioOptimisers.validate_k_value(res8, nodes8, k)] ==
              [1, 2, 4, 8]

        # A leaf reaches the walk only when a merge sits at height zero: `sortperm` is stable
        # and the leaves come first among the ties.
        res0 = Hclust([-1 -2; -3 -4; 1 2], [0.0, 1.0, 2.0], [1, 2, 3, 4], :ward)
        nds0 = tall_first(res0)
        @test PortfolioOptimisers.is_leaf(nds0[3])
        @test PortfolioOptimisers.validate_k_value(res0, nds0, 4)
    end

    @testset "`valid_k_clusters` walks past a rejected candidate and always ends" begin
        res8 = balanced_hclust(3)
        # The maximiser is rejected, so it is blanked and the next one is taken.
        arr = zeros(8)
        arr[2] = 5.0
        arr[3] = 10.0
        @test PortfolioOptimisers.valid_k_clusters(res8, arr) == 2
        @test arr[3] == typemin(Float64)
        # The mutation reaches no caller: both callers pass a local score array.
        @test PortfolioOptimisers.valid_k_clusters(res8, [0.0, 9.0, 1.0, 2.0]) == 2
        # No finite entry, and the tree rejects `length(arr)`: the answer is `1`.
        @test PortfolioOptimisers.valid_k_clusters(res8, fill(-Inf, 7)) == 1
        # A wholly-`NaN` array takes the same escape. Under the old `all(isinf, arr)` guard
        # it re-offered the same rejected candidate and the loop never ended.
        @test PortfolioOptimisers.valid_k_clusters(res8, fill(NaN, 7)) == 1

        # The index of the score array is the count the tree is cut at, so the second-order
        # difference selects the LEFT end of its triple. This series peaks at `c = 3`, which
        # is the triple `W[3], W[4], W[5]`, and the cut comes back as `3`.
        W = [typemin(Float64), 10.0, 9.0, 3.0, 2.5, 2.4, 2.35]
        gaps = W[1:(end - 2)] + W[3:end] - 2 * W[2:(end - 1)]
        @test argmax(gaps) == 3
        Xr = randn(StableRNG(7), 100, 25)
        _, Dr = cor_and_dist(Distance(), PortfolioOptimisersCovariance(), Xr)
        rr = hclust(Dr; linkage = :ward, branchorder = :optimal)
        @test PortfolioOptimisers.valid_k_clusters(rr, copy(gaps)) == 3
        # The `typemin` seed keeps a single cluster out of the answer without emptying the
        # `all(!isfinite, .)` test.
        @test typemin(Float64) == -Inf
        @test !isfinite(gaps[1])
        @test !all(!isfinite, gaps)
    end

    @testset "The `Integer` method searches, and a tie goes to the side with more room" begin
        # Valid counts on this tree are 1, 2, 4 and 8, and the ceiling is
        # `floor(Int, sqrt(64)) = 8`.
        res64 = balanced_hclust(6)
        D64 = zeros(64, 64)
        function onc_k(k, mk = nothing)
            return PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                     alg = k,
                                                                                     max_k = mk),
                                                               res64, D64)
        end
        @test onc_k(2) == 2       # valid as stated
        @test onc_k(3) == 4       # tie, and 8 - 4 > 2 - 1, so the upper side
        @test onc_k(5) == 4       # du = 3, dl = 1, so the nearer lower side
        @test onc_k(6) == 4       # tie, and 8 - 8 < 4 - 1, so the lower side
        @test onc_k(7) == 8       # du = 1, dl = 3, so the nearer upper side
        @test onc_k(5, 5) == 4    # nothing valid above a ceiling of 5, so the lower side
        @test onc_k(100) == 8     # lowered to the ceiling, which this tree admits
    end

    @testset "The two branches measure different dispersions" begin
        rng = StableRNG(987654321)
        Xg = randn(rng, 200, 20)
        Xg[:, 2] = Xg[:, 1] + 0.02 * randn(rng, 200)
        Xg[:, 4] = Xg[:, 3] + 0.02 * randn(rng, 200)
        cle = ClustersEstimator(; alg = HClustAlgorithm())
        Sg, Dg = cor_and_dist(cle.de, cle.ce, Xg)
        rg = hclust(Dg; linkage = cle.alg.linkage, branchorder = :optimal)
        algk = KMeansAlgorithm(; rng = StableRNG(987654321), seed = 987654321)
        kh = PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                               alg = SecondOrderDifference()),
                                                         rg, Dg)
        pk, kk = PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                   alg = SecondOrderDifference()),
                                                             algk, Dg)
        # The hierarchical dispersion sums within-cluster pairwise distances and rises with
        # the count; the k-means per-point costs fall with it.
        @test kh == 3
        @test kk == 2
        # The hierarchical method answers a bare `k`; the flat one answers `(res, k)`.
        @test kh isa Integer
        @test pk isa Clustering.ClusteringResult

        # A universe too small to carry a second-order difference answers `c1` itself.
        X2 = Xg[:, 1:2]
        _, D2 = cor_and_dist(Distance(), PortfolioOptimisersCovariance(), X2)
        r2 = hclust(D2; linkage = :ward, branchorder = :optimal)
        @test PortfolioOptimisers.optimal_number_clusters(OptimalNumberClusters(;
                                                                                alg = SecondOrderDifference()),
                                                          r2, D2) == 2

        # `resolve_rng` copies and seeds, so a seeded estimator never advances its own
        # generator and two calls partition alike.
        r = StableRNG(42)
        @test PortfolioOptimisers.resolve_rng(r, nothing) === r
        @test PortfolioOptimisers.resolve_rng(r, 7) !== r
        a_seed = KMeansAlgorithm(; seed = 123456789)
        p1 = PortfolioOptimisers.get_k_clusters_from_alg(a_seed, Dg, 3)
        p2 = PortfolioOptimisers.get_k_clusters_from_alg(a_seed, Dg, 3)
        @test p1.assignments == p2.assignments

        # `assignments` answers one label per asset on both result shapes.
        ah = assignments(Clusters(; res = rg, S = Sg, D = Dg, k = 3))
        ak = assignments(Clusters(; res = p1, S = Sg, D = Dg, k = 3))
        @test length(ah) == size(Dg, 1)
        @test length(ak) == size(Dg, 1)
        @test sort(unique(ah)) == 1:3
        @test sort(unique(ak)) == 1:3
    end
end

# The numbers that the DBHT files of `src/11_Phylogeny/` claim, run rather than read. This is
# condition 2 of the sweep of #470, under child map #418 of the map of maps #404. The file is
# a port, so the check that carries weight is a number from a hand-computed answer set beside
# the number the code returns. The sources are `DBHTs.m`, MATLAB Central File Exchange
# submission 46750 by Won-Min Song and Tomaso Aste, and `distance_wei.m` and `breadth.m` of
# the Brain Connectivity Toolbox by Mika Rubinov and Olaf Sporns. Defined at top level
# because a `@testset` body becomes a function, and two of these bodies build a `struct`-free
# fixture the whole file shares.
@testset "The DBHT port answers its own numbers (#470)" begin
    using PortfolioOptimisers, Test, Clustering, CSV, DataFrames, TimeSeries, StatsBase,
          SparseArrays, LinearAlgebra

    PO470 = PortfolioOptimisers
    rd470 = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__,
                                                          "./assets/SP500.csv.gz"));
                                        timestamp = :Date)[(end - 252):end])
    X470 = rd470.X
    ce470 = PortfolioOptimisersCovariance()
    rho470 = PO470.cor(ce470, X470)
    dist470 = PO470.distance(Distance(; alg = SimpleDistance()), rho470, X470)
    sim470 = MaximumDistanceSimilarity()
    S470 = PO470.distance_to_similarity(sim470; S = rho470, D = dist470)

    @testset "A maximal planar graph carries 3N - 6 edges, at two N" begin
        #=
        `PMFG_T2s`' docstring states that the structure is a maximal planar graph, so it
        carries `3N - 6` edges against the `N - 1` of a minimum spanning tree, and that a
        20-asset sample gives `54`. The count is structural: the greedy inserts every
        remaining vertex into a face and each insertion adds exactly three edges, so no
        near-tie in the gains can move it. Assert it at two `N`, so the identity is read
        rather than one number.
        =#
        for n in (20, 12)
            idx = 1:n
            S = PO470.distance_to_similarity(sim470; S = rho470[idx, idx],
                                             D = dist470[idx, idx])
            A = PO470.PMFG_T2s(S)[1]
            @test size(A, 1) == n
            @test count(!iszero, A) ÷ 2 == 3 * n - 6
            @test count(!iszero, A) ÷ 2 > n - 1
        end
        @test count(!iszero, PO470.PMFG_T2s(S470)[1]) ÷ 2 == 54

        # The face list and the non-face 3-clique list carry the counts the algorithm
        # section states: `2N - 4` faces of a triangulation, and one non-face 3-clique per
        # vertex inserted after the tetrahedron.
        A470, tri470, c3470 = PO470.PMFG_T2s(S470)[1:3]
        @test size(tri470) == (2 * 20 - 4, 3)
        @test size(c3470) == (20 - 4, 3)

        #=
        `[tri; clique3]` is claimed to be every 3-clique of the graph. `clique3` the
        function answers the same question independently, by intersecting the neighbourhoods
        of every edge, so the two lists must agree as sets.
        =#
        union470 = sort(collect(unique(eachrow(sort(vcat(tri470, c3470); dims = 2)))))
        @test length(union470) == 52
        @test union470 == sort(collect(eachrow(PO470.clique3(A470)[3])))
    end

    @testset "`nargout` is positional, and the last two outputs are built by arity" begin
        # `cliques` is built when `nargout > 3` and `cliqueTree` when `nargout > 4`. Each is
        # `nothing` otherwise, and the first three outputs are always built.
        r3 = PO470.PMFG_T2s(S470)
        r4 = PO470.PMFG_T2s(S470, 4)
        r5 = PO470.PMFG_T2s(S470, 5)
        @test isnothing(r3[4]) && isnothing(r3[5])
        @test !isnothing(r4[4]) && isnothing(r4[5])
        @test !isnothing(r5[4]) && !isnothing(r5[5])
        # One 4-clique per vertex after the first three, so `N - 3` of them.
        @test size(r4[4]) == (20 - 3, 4)
        @test size(r5[5]) == (20 - 3, 20 - 3)
        # The default arity is `3`, so the default and the explicit `3` agree.
        @test isnothing(PO470.PMFG_T2s(S470, 3)[4])
        for i in 1:3
            @test r3[i] == r4[i] == r5[i]
        end
    end

    @testset "Each of the three validation arms throws" begin
        # `N >= 9` is a `DimensionMismatch`; the other two are a `DomainError`. The `NaN`
        # arm is split from the non-negativity arm because `0 <= NaN` is `false`, so one
        # check would report a `NaN` as a negative weight.
        @test_throws DimensionMismatch PO470.PMFG_T2s(S470[1:8, 1:8])
        Wnan = Matrix(copy(S470))
        Wnan[2, 3] = NaN
        @test_throws DomainError PO470.PMFG_T2s(Wnan)
        Wneg = Matrix(copy(S470))
        Wneg[2, 3] = -0.5
        @test_throws DomainError PO470.PMFG_T2s(Wneg)
        # Nine is the boundary, not a rejected size.
        @test PO470.PMFG_T2s(S470[1:9, 1:9])[1] isa AbstractMatrix
    end

    @testset "`distance_wei` answers the hand-computed shortest path" begin
        #=
        A four-vertex path whose lengths are 1, 2 and 4, with a direct 1-4 edge of length
        10. The shortest 1-4 route is the path, at 7, and it crosses three edges. `B` is
        the edge count of the winning path, and `distance_wei` returned an unrelated
        matrix until #470: it read the column of the `CartesianIndex` that `findmin`
        returns instead of the row, and compared it with `3` where the matrix has two rows.
        =#
        L = SparseArrays.sparse(Float64[0 1 0 10; 1 0 2 0; 0 2 0 4; 10 0 4 0])
        D, B = PO470.distance_wei(L)
        @test D[1, :] == [0.0, 1.0, 3.0, 7.0]
        @test D == transpose(D)
        @test B[1, :] == [0, 1, 2, 3]
        @test all(iszero, LinearAlgebra.diag(D))

        # A star with a leaf: 1 joins 2, 3 and 4 at length 1, and 5 hangs off 2. The
        # shortest 1-5 route crosses two edges, and 1-2, 1-3 and 1-4 cross one each.
        Lg = SparseArrays.sparse(Float64[0 1 1 1 0; 1 0 0 0 1; 1 0 0 0 0; 1 0 0 0 0;
                                         0 1 0 0 0])
        Dg, Bg = PO470.distance_wei(Lg)
        @test Dg[1, :] == [0.0, 1.0, 1.0, 1.0, 2.0]
        @test Bg[1, :] == [0, 1, 1, 1, 2]

        # A vertex no path reaches keeps the `typemax` the matrix was filled with, and its
        # edge count stays zero.
        L2 = SparseArrays.sparse(Float64[0 1 0 10 0; 1 0 2 0 0; 0 2 0 4 0; 10 0 4 0 0;
                                         0 0 0 0 0])
        D2, B2 = PO470.distance_wei(L2)
        @test isinf(D2[1, 5])
        @test all(isinf, D2[1:4, 5])
        @test iszero(D2[5, 5])
        @test iszero(B2[1, 5])
        # The reachable block is the four-vertex answer above, unchanged by the extra
        # vertex.
        @test D2[1:4, 1:4] == D
        @test B2[1:4, 1:4] == B
    end

    @testset "`breadth` walks the graph, and does not keep the source at zero" begin
        #=
        1-2-3-4 with a leaf 5 on vertex 1, and an isolated vertex 6. The BFS distances and
        the BFS tree are read off the picture.

        `distance[source]` is `2`, not `0`. The arm that reads a zero distance fires on the
        source itself the first time one of its own neighbours is expanded, which is the
        behaviour of the MATLAB original. Every caller resets the entry, so the value is
        asserted here rather than corrected in the kernel.
        =#
        CIJ = SparseArrays.sparse(Int[0 1 0 0 1 0; 1 0 1 0 0 0; 0 1 0 1 0 0; 0 0 1 0 0 0;
                                      1 0 0 0 0 0; 0 0 0 0 0 0])
        d, b = PO470.breadth(CIJ, 1)
        @test d[2:6] == [1.0, 2.0, 3.0, 1.0, Inf]
        @test d[1] == 2.0
        @test b == [-1, 1, 2, 3, 1, 0]

        # A self-loop on the source makes the same arm write `1` instead.
        CIJs = copy(CIJ)
        CIJs[1, 1] = 1
        @test PO470.breadth(CIJs, 1)[1][1] == 1.0

        # A vertex no walk reaches keeps `Inf` and a branch of `0`, and starting there
        # reaches nothing at all.
        d6, b6 = PO470.breadth(CIJ, 6)
        @test d6[6] == 0.0
        @test all(isinf, d6[1:5])
        @test b6 == [0, 0, 0, 0, 0, -1]
    end

    @testset "`clique3` finds every triangle, and `FindDisjoint` splits on one" begin
        # `K4` is the smallest triangulation. Its four faces are its four 3-cliques.
        K4 = SparseArrays.sparse(Int[0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0])
        @test PO470.clique3(K4)[3] == [1 2 3; 1 2 4; 1 3 4; 2 3 4]

        #=
        A separating triangle `{1, 2, 3}` with `{4, 5}` on one side and `{6, 7}` on the
        other, and no edge between the two sides. `FindDisjoint` labels the triangle `0`,
        the side that holds the first non-clique vertex `2`, and the far side `1`.
        =#
        Asep = zeros(Int, 7, 7)
        for (i, j) in
            ((1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4), (4, 5), (1, 5), (2, 5), (1, 6),
             (2, 6), (3, 6), (6, 7), (1, 7), (2, 7))
            Asep[i, j] = 1
            Asep[j, i] = 1
        end
        Asep = SparseArrays.sparse(Asep)
        T, IndxNot = PO470.FindDisjoint(Asep, [1, 2, 3])
        @test IndxNot == [4, 5, 6, 7]
        @test T == [0, 0, 0, 2, 2, 1, 1]
        # The label of a side is the side's own, so swapping the roles swaps the labels.
        T2 = PO470.FindDisjoint(Asep, [1, 2, 6])[1]
        @test T2[[1, 2, 6]] == [0, 0, 0]
        @test length(unique(T2)) == 3
    end

    @testset "The two `CliqueRoot` methods are not the same clustering" begin
        #=
        `UniqueRoot` and `EqualRoot` are the one configuration knob of the clustering, and
        they differ where the clique forest has more than one root. `CliqueRoot(::UniqueRoot,
        …)` appends a virtual parent to `Pred` and points every root at it, so
        `BubbleHierarchy` reads a different `Pred` and builds a different bubble matrix.
        `CliqueRoot(::EqualRoot, …)` leaves `Pred` as it found it and joins the roots with
        `AdjCliq` instead.

        The knob is therefore not a no-op: `Mv` differs. It is also not a different answer:
        the graph, the converging bubbles, the discrete clusters and the merge heights all
        agree. It moves the order in which the leaves inside a bubble are merged.
        =#
        for n in (20, 15)
            idx = 1:n
            S = PO470.distance_to_similarity(sim470; S = rho470[idx, idx],
                                             D = dist470[idx, idx])
            D = dist470[idx, idx]
            ru = PO470.DBHTs(D, S; branchorder = :default, root = UniqueRoot())
            re = PO470.DBHTs(D, S; branchorder = :default, root = EqualRoot())
            @test ru[1] == re[1]          # T8, the discrete clusters
            @test ru[2] == re[2]          # Rpm, the weighted graph
            @test ru[3] == re[3]          # Adjv, the converging-bubble membership
            @test ru[4] == re[4]          # Dpm, the shortest paths
            @test ru[7].heights == re[7].heights
            @test ru[5] != re[5]          # Mv, the bubble matrix
        end

        # More than one root is the precondition the difference rests on. Assert it, so a
        # regression names its cause rather than the symptom.
        Rpm = PO470.DBHTs(dist470, S470; branchorder = :default)[2]
        A = Rpm .!= 0
        cl = PO470.clique3(A)[3]
        M = SparseArrays.spzeros(Int, size(Rpm, 1), size(cl, 1))
        for n in axes(cl, 1)
            T = PO470.FindDisjoint(A, cl[n, :])[1]
            i0 = findall(T .== 0)
            i1 = findall(T .== 1)
            i2 = findall(T .== 2)
            is = length(i1) > length(i2) ? vcat(i2, i0) : vcat(i1, i0)
            M[is, n] .= 1
        end
        @test count(iszero, PO470.BuildHierarchy(M)) == 4
    end

    @testset "`turn_into_Hclust_merges` writes the order `Clustering.Hclust` reads" begin
        #=
        The merge order is right when cutting the tree at `k` reproduces the discrete
        cluster membership that `BubbleCluster8s` produced, asset by asset. The assertion is
        equality and not a bijection of the labels, because the two agree label for label.
        =#
        T8, _, _, _, _, Z, hcl = PO470.DBHTs(dist470, S470; branchorder = :default)
        k = maximum(T8)
        @test k == 2
        ct = Clustering.cutree(hcl; k = k)
        @test ct == T8
        # A leaf is a negative index and an earlier merge is a positive one, and the fourth
        # column counts the leaves below each merge. The last merge holds every asset.
        @test size(Z) == (length(T8) - 1, 4)
        @test Z[end, 4] == length(T8)
        @test all(<=(length(T8)), abs.(Z[:, 1:2]))
        @test length(hcl.heights) == length(T8) - 1
    end

    @testset "`DirectHb` sums signed mass, and no shipped route can reach one" begin
        #=
        `PMFG_T2s`' docstring warns that a cancelling row manufactures a separating bubble
        and that the failure is silent. Both halves are true, and the second is why the
        first cannot arrive from a configuration this library ships.

        The reachability first: `PMFG_T2s` refuses a negative weight with a `DomainError`,
        and it returns `A = W ⊙ mask`, so every stored weight of `Rpm` is one of `W`'s own
        non-negative entries. A negative mass cannot enter `DirectHb` through `DBHTs`.
        =#
        Rpm = PO470.DBHTs(dist470, S470; branchorder = :default)[2]
        @test all(>=(0), SparseArrays.nonzeros(Rpm))
        Wneg = Matrix(copy(S470))
        Wneg[2, 3] = -0.5
        Wneg[3, 2] = -0.5
        @test_throws DomainError PO470.PMFG_T2s(Wneg)

        # The mechanism, driven by calling `DirectHb` directly, which takes plain matrices
        # and carries no type bound of its own.
        H1, Hb, Mb, CliqList, Sb = PO470.CliqHierarchyTree2s(Rpm, UniqueRoot())
        Mb = Mb[1:size(CliqList, 1), :]
        Mv = SparseArrays.spzeros(Int, size(Rpm, 1), 0)
        for n in axes(Mb, 2)
            vc = SparseArrays.spzeros(Int, size(Rpm, 1))
            vc[sort!(unique(CliqList[Mb[:, n] .!= 0, :]))] .= 1
            Mv = hcat(Mv, vc)
        end
        Sep = PO470.DirectHb(Rpm, Hb, Mb, Mv, CliqList)[2]
        # A converging bubble is a `1`. On the non-negative weights the sample has two.
        @test count(==(1), Sep) == 2

        # The same structure with the sign of the weights that cross vertex 10 flipped.
        # Nothing raises, and the set of converging bubbles moves.
        Rsigned = Matrix(Rpm)
        for i in axes(Rsigned, 1), j in axes(Rsigned, 2)
            if (i > 10) != (j > 10)
                Rsigned[i, j] = -Rsigned[i, j]
            end
        end
        Sep2 = PO470.DirectHb(SparseArrays.sparse(Rsigned), Hb, Mb, Mv, CliqList)[2]
        @test any(<(0), Rsigned)
        @test Sep2 != Sep
        @test count(==(1), Sep2) == 3
    end

    @testset "`J_LoGo` sparsifies the precision, and the covariance stays dense" begin
        #=
        `logo!` writes `J_LoGo(sigma, separators, cliques) \\ I` into `sigma`, so the sparse
        object is the **precision** and the covariance it ships is dense. Read the precision
        back with an inverse and check that its zeros sit exactly off the clique support.
        =#
        sigma = PO470.cov(ce470, X470)
        je = LoGo()
        s = LinearAlgebra.diag(sigma)
        R = StatsBase.cov2cor(sigma, sqrt.(s))
        Sl = PO470.distance_to_similarity(je.sim; S = R, D = PO470.distance(je.de, R, X470))
        cliques = PO470.PMFG_T2s(Sl, 4)[4]
        support = falses(size(sigma))
        for r in axes(cliques, 1), a in cliques[r, :], b in cliques[r, :]
            support[a, b] = true
        end

        sg = PO470.logo(je, sigma, X470)
        P = inv(sg)
        @test count(!, support) == 272
        @test maximum(abs, P[.!support]) < 1e-8
        @test minimum(abs, P[support]) > 1
        @test count(iszero, sg) == 0
        @test isapprox(sg, transpose(sg))

        # `logo` is `logo!` on a copy, and it leaves its argument alone.
        sg2 = copy(sigma)
        PO470.logo!(je, sg2, X470)
        @test sg2 == sg
        @test sigma == PO470.cov(ce470, X470)
        # The `nothing` method of the seam is the no-op the callers expect.
        @test isnothing(PO470.logo!(nothing))
        @test isnothing(PO470.logo!(nothing, sigma, X470; dims = 1))
    end

    @testset "`LoGo_dist_assert` narrows on both variation-of-information estimators" begin
        #=
        The narrow method is bounded by `DVarInfo_DDVarInfo`, and its docstring states that
        a `Distance` **and** a `DistanceDistance` whose algorithm is a
        `VariationInfoDistance` both reach it. Until #470 the union named the algorithm in
        the second type parameter of `DistanceDistance`, which is `args`, so the
        `DistanceDistance` arm matched nothing and a mismatched `X` died in
        `PMFG_T2s` with a `BoundsError` instead.
        =#
        sigma = PO470.cov(ce470, X470)
        dv = Distance(; alg = VariationInfoDistance())
        ddv = DistanceDistance(; alg = VariationInfoDistance())
        @test dv isa PO470.DVarInfo_DDVarInfo
        @test ddv isa PO470.DVarInfo_DDVarInfo
        @test isnothing(PO470.LoGo_dist_assert(dv, sigma, X470))
        @test isnothing(PO470.LoGo_dist_assert(ddv, sigma, X470))
        @test_throws DimensionMismatch PO470.LoGo_dist_assert(dv, sigma[1:5, 1:5], X470)
        @test_throws DimensionMismatch PO470.LoGo_dist_assert(ddv, sigma[1:5, 1:5], X470)
        @test_throws DimensionMismatch PO470.logo(LoGo(; de = ddv), sigma[1:5, 1:5], X470)

        # An estimator outside the union reads the correlation rather than `X`, so it takes
        # the `args...` fallback and no shape is read.
        dc = Distance(; alg = CanonicalDistance())
        @test !(dc isa PO470.DVarInfo_DDVarInfo)
        @test isnothing(PO470.LoGo_dist_assert(dc, sigma[1:5, 1:5], X470))
        @test isnothing(PO470.LoGo_dist_assert())
        @test isnothing(PO470.LoGo_dist_assert(1, 2, 3, 4))

        # Both variation-of-information estimators run to a finite answer at the right
        # shape, so the narrowing refuses a mismatch rather than the family.
        for d in (dv, ddv)
            sg = PO470.logo(LoGo(; de = d), sigma, X470)
            @test size(sg) == size(sigma)
            @test all(isfinite, sg)
        end
    end
end

# The four branches of the DBHT files of `src/11_Phylogeny/` that the suite left uncovered
# when #470 measured them, and the input that reaches each. Condition 3 of the sweep. Defined at top
# level because a `@testset` body becomes a function.
@testset "The DBHT branches the suite did not reach (#470)" begin
    using PortfolioOptimisers, Test, Clustering, CSV, DataFrames, TimeSeries, StatsBase,
          SparseArrays, LinearAlgebra

    PO470b = PortfolioOptimisers
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    X = rd.X
    rho = PO470b.cor(PortfolioOptimisersCovariance(), X)
    dist = PO470b.distance(Distance(; alg = SimpleDistance()), rho, X)
    sim = MaximumDistanceSimilarity()

    @testset "A clique-free graph takes the empty arms of the hierarchy" begin
        #=
        `CliqueRoot(::EqualRoot, …)` returns a `0 × 0` matrix when `Pred` is empty, and
        `CliqHierarchyTree2s` then takes its own empty arm and returns two `0 × 0` matrices
        rather than calling `BubbleHierarchy`. `Pred` is empty when the graph carries no
        3-clique, so a cycle is the input: it is connected, it is planar, and it holds no
        triangle.

        `UniqueRoot` cannot reach the same arm. It builds an `Nc + 1` square matrix, which
        is `1 × 1` at `Nc = 0` and therefore not empty.
        =#
        Cyc = SparseArrays.sparse(Float64[0 1 0 0 1; 1 0 1 0 0; 0 1 0 1 0; 0 0 1 0 1;
                                          1 0 0 1 0])
        @test isempty(PO470b.clique3(Cyc)[3])
        H, H2, Mb, CliqList, Sb = PO470b.CliqHierarchyTree2s(Cyc, EqualRoot())
        @test size(CliqList, 1) == 0
        @test size(H) == (0, 0)
        @test size(H2) == (0, 0)
        @test size(Mb) == (0, 0)
        @test isempty(H)
        # The method called on its own, with the empty `Pred` the arm is keyed on.
        @test size(PO470b.CliqueRoot(EqualRoot(), Int[], Int[], 0, Cyc,
                                     Matrix{Int}(undef, 0, 3))) == (0, 0)
        # The same graph under the other root is not empty, so the arm is `EqualRoot`'s.
        @test !isempty(PO470b.CliqHierarchyTree2s(Cyc, UniqueRoot())[1])
    end

    @testset "A vertex in two converging bubbles is assigned by edge-weight share" begin
        #=
        `BubbleCluster8s` splits the vertices of the converging bubbles into those in one
        bubble and those in more than one. The second set is `uv`, and each of its vertices
        goes to the bubble whose share of its edge weight is largest, `chi(v, b_alpha)`.

        No prefix of the sample reaches that loop: over `N = 9` to `20` every vertex of a
        converging bubble sits in exactly one. This sixteen-column subset puts two vertices
        in both, and the column list is written out rather than searched, so the fixture is
        deterministic.
        =#
        cols = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 20]
        D = dist[cols, cols]
        S = PO470b.distance_to_similarity(sim; S = rho[cols, cols], D = D)
        Rpm = PO470b.PMFG_T2s(S)[1]
        H1, Hb, Mb, CliqList, Sb = PO470b.CliqHierarchyTree2s(Rpm, UniqueRoot())
        Mb = Mb[1:size(CliqList, 1), :]
        Mv = SparseArrays.spzeros(Int, size(Rpm, 1), 0)
        for n in axes(Mb, 2)
            vc = SparseArrays.spzeros(Int, size(Rpm, 1))
            vc[sort!(unique(CliqList[Mb[:, n] .!= 0, :]))] .= 1
            Mv = hcat(Mv, vc)
        end
        Sep = PO470b.DirectHb(Rpm, Hb, Mb, Mv, CliqList)[2]
        indx = findall(Sep .== 1)
        @test length(indx) == 2
        Bubv = Mv[:, indx]
        @test count(vec(sum(Bubv; dims = 2) .> 1)) == 2

        Apm = copy(Rpm)
        Apm[Apm .!= 0] = D[Apm .!= 0]
        Dpm = PO470b.distance_wei(Apm)[1]
        Adjv, Tc = PO470b.BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)
        # Every vertex lands in exactly one discrete cluster, the shared ones included.
        @test length(Tc) == length(cols)
        @test sort(unique(Tc)) == 1:2
        @test all(>(0), Tc)
        @test size(Adjv) == (length(cols), 2)
        # The full route agrees with the kernel, so the loop is on the shipped path.
        @test PO470b.DBHTs(D, S; branchorder = :default)[1] == Tc
    end

    @testset "The three `branchorder` values each take their own arm" begin
        #=
        `:optimal` and `:barjoseph` both call `orderbranches_barjoseph!`, `:r` calls
        `orderbranches_r!`, and anything else leaves the merge order as
        `HierarchyConstruct4s` wrote it. The reordering permutes the leaves and leaves the
        clustering alone, so the cut is the invariant across all four.
        =#
        S = PO470b.distance_to_similarity(sim; S = rho, D = dist)
        res = Dict(b => PO470b.DBHTs(dist, S; branchorder = b)
                   for b in (:optimal, :barjoseph, :r, :default))
        for b in (:optimal, :barjoseph, :r, :default)
            hcl = res[b][7]
            @test hcl isa Clustering.Hclust
            @test Clustering.cutree(hcl; k = 2) == res[b][1]
            # Every leaf appears once as a negative index, and every merge but the
            # root appears once as a positive one.
            @test sort(-hcl.merges[hcl.merges .< 0]) == collect(1:size(dist, 1))
            @test sort(hcl.merges[hcl.merges .> 0]) == collect(1:(size(dist, 1) - 2))
        end
        @test res[:optimal][7].merges == res[:barjoseph][7].merges
        @test res[:r][7].merges != res[:default][7].merges
        @test res[:optimal][7].heights == res[:default][7].heights
    end
end

#=
Condition 2 of the sweep of the phylogeny family, issue #472. The eight centrality algorithms
now live in `src/11_Phylogeny/14_Centrality.jl`. The file that held them was
coverage terminal before this testset was written -- 319 executable lines, 0 misses -- and
every check below exists because the line it covers was already executed and asserted
nothing about the value it produced.

The graphs are a path, a star and a cycle, which separate the eight centrality algorithms
from each other and whose closed forms a reader can compute by hand. Each expected value is
written as its closed form rather than as a recorded decimal, so the test carries its own
authority and cannot pass a wrong answer.

Defined at top level because a `@testset` body becomes a function.
=#
@testset "The network layer answers its own closed forms (#472)" begin
    using PortfolioOptimisers, Test, CSV, DataFrames, TimeSeries, LinearAlgebra, Statistics

    PO472 = PortfolioOptimisers
    Gr472 = PortfolioOptimisers.Graphs
    cc = PortfolioOptimisers.calc_centrality
    rd472 = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__,
                                                          "./assets/SP500.csv.gz"));
                                        timestamp = :Date)[(end - 252):end])
    Xt = prior(EmpiricalPrior(), rd472).X

    P4 = Gr472.path_graph(4)
    S5 = Gr472.star_graph(5)
    C5 = Gr472.cycle_graph(5)

    @testset "The eight algorithms against their closed forms" begin
        #=
        Degree counts the edges that touch a node and divides by `n - 1`. Closeness is
        `(n - 1) / sum(d)` on a connected graph. Betweenness counts the shortest paths
        through a node over the unordered pairs and divides by `(n - 1)(n - 2) / 2`; stress
        is the same count over the ORDERED pairs and divides by nothing. Radiality reads the
        mean length against the diameter. Eigenvector and Katz are the leading eigenvector
        and the resolvent, both at unit 2-norm. PageRank is the damped walk's stationary
        share, which sums to one.
        =#
        # A path of four. Degrees 1, 2, 2, 1.
        @test cc(DegreeCentrality(), P4) == [1, 2, 2, 1] ./ 3
        @test cc(DegreeCentrality(; kwargs = (; normalize = false)), P4) == [1, 2, 2, 1]
        # Distance sums 6, 4, 4, 6 against `n - 1 == 3`.
        @test cc(ClosenessCentrality(), P4) == 3 ./ [6, 4, 4, 6]
        # Node 2 lies on 1--3 and 1--4; node 3 on 1--4 and 2--4. `(n - 1)(n - 2) / 2 == 3`.
        @test cc(BetweennessCentrality(), P4) == [0, 2, 2, 0] ./ 3
        # The same two pairs, counted in both directions and left undivided.
        @test cc(StressCentrality(), P4) == [0, 4, 4, 0]
        # Diameter 3, mean lengths 6/3, 4/3, 4/3, 6/3 against `(3 + 1 - mean) / 3`.
        @test cc(RadialityCentrality(), P4) == ((3 + 1) .- [6, 4, 4, 6] ./ 3) ./ 3
        # The leading eigenvalue of a path of four is the golden ratio, and its eigenvector
        # is `[1, phi, phi, 1]`.
        phi = (1 + sqrt(5)) / 2
        @test isapprox(cc(EigenvectorCentrality(), P4),
                       [1, phi, phi, 1] ./ norm([1, phi, phi, 1]))
        # The resolvent, solved directly.
        vk4 = (I - 0.3 * Matrix{Float64}(Gr472.adjacency_matrix(P4))) \ ones(4)
        @test isapprox(cc(KatzCentrality(), P4), vk4 ./ norm(vk4))
        # `PR2 = 0.0375 + 0.85 * PR1 + 0.425 * PR2` and `PR1 = 0.0375 + 0.425 * PR2`.
        pr2 = 0.069375 / 0.21375
        # The stopping rule is `sum(abs, x - xlast) < n * epsilon`, so the closed form
        # is met to about `n * epsilon` and not to `epsilon`.
        @test isapprox(cc(Pagerank(), P4), [0.0375 + 0.425pr2, pr2, pr2, 0.0375 + 0.425pr2];
                       atol = 1e-5)

        # A star of five. The centre touches every leaf; a leaf touches the centre alone.
        @test cc(DegreeCentrality(), S5) == [4, 1, 1, 1, 1] ./ 4
        @test cc(DegreeCentrality(; kwargs = (; normalize = false)), S5) == [4, 1, 1, 1, 1]
        # Distance sums 4 and 1 + 2 + 2 + 2 against `n - 1 == 4`.
        @test cc(ClosenessCentrality(), S5) == 4 ./ [4, 7, 7, 7, 7]
        # The centre carries every one of the `binomial(4, 2) == 6` leaf pairs, and
        # `(n - 1)(n - 2) / 2 == 6`.
        @test cc(BetweennessCentrality(), S5) == [1, 0, 0, 0, 0]
        # The same six pairs in both directions.
        @test cc(StressCentrality(), S5) == [12, 0, 0, 0, 0]
        # Diameter 2, mean lengths 4/4 and 7/4 against `(2 + 1 - mean) / 2`.
        @test cc(RadialityCentrality(), S5) == ((2 + 1) .- [4, 7, 7, 7, 7] ./ 4) ./ 2
        # A star's leading eigenvector puts `sqrt(k)` on the centre and one on each leaf.
        @test isapprox(cc(EigenvectorCentrality(), S5),
                       [2, 1, 1, 1, 1] ./ norm([2, 1, 1, 1, 1]))
        # `c = 1 + 4 * alpha * l` and `l = 1 + alpha * c`, so `c = (1 + 4a) / (1 - 4a^2)`.
        cs = (1 + 4 * 0.3) / (1 - 4 * 0.3^2)
        ls = 1 + 0.3 * cs
        @test isapprox(cc(KatzCentrality(), S5),
                       [cs, ls, ls, ls, ls] ./ norm([cs, ls, ls, ls, ls]))
        # `PRc = 0.03 + 3.4 * PRl` and `PRc + 4 * PRl = 1`, so `PRl = 0.97 / 7.4`.
        prl = 0.97 / 7.4
        @test isapprox(cc(Pagerank(), S5), [1 - 4prl, prl, prl, prl, prl]; atol = 1e-5)

        # A cycle of five. It is vertex-transitive, so every algorithm is constant on it.
        @test cc(DegreeCentrality(), C5) == fill(2 / 4, 5)
        @test cc(DegreeCentrality(; kwargs = (; normalize = false)), C5) == fill(2.0, 5)
        # Every distance sum is `0 + 1 + 1 + 2 + 2 == 6` against `n - 1 == 4`.
        @test cc(ClosenessCentrality(), C5) == fill(4 / 6, 5)
        # One pair of the four remaining vertices routes through each node.
        @test cc(BetweennessCentrality(), C5) == fill(1 / 6, 5)
        @test cc(StressCentrality(), C5) == fill(2, 5)
        @test cc(RadialityCentrality(), C5) == fill(((2 + 1) - 6 / 4) / 2, 5)
        # `A * 1 == 2 * 1`, so both the eigenvector and the resolvent are constant.
        @test isapprox(cc(EigenvectorCentrality(), C5), fill(1 / sqrt(5), 5))
        @test isapprox(cc(KatzCentrality(), C5), fill(1 / sqrt(5), 5))
        @test isapprox(cc(Pagerank(), C5), fill(1 / 5, 5); atol = 1e-5)
    end

    @testset "`DegreeCentrality`'s divisor, and its three `kind` values" begin
        #=
        The wrapper's name does not say that `Graphs._degree_centrality` divides by
        `n - 1`. It does, and `normalize = false` recovers the count. `kind` selects the
        total, the in- or the out-degree, which are one number on an undirected graph and
        three on a directed one -- so the field is kept because `Graphs.jl` takes it, not
        because it selects anything the library builds.
        =#
        for g in (P4, S5, C5)
            n = Gr472.nv(g)
            raw = cc(DegreeCentrality(; kwargs = (; normalize = false)), g)
            @test cc(DegreeCentrality(), g) == raw ./ (n - 1)
            @test raw == vec(sum(Gr472.adjacency_matrix(g); dims = 2))
            # The three coincide, exactly, on every undirected structure.
            @test allequal(cc(DegreeCentrality(; kind = k), g) for k in 0:2)
        end
        # And separate on a directed one, so the field is not inert in `Graphs.jl`.
        dg = Gr472.SimpleDiGraph(4)
        for (u, v) in ((1, 2), (2, 3), (3, 4), (1, 4))
            Gr472.add_edge!(dg, u, v)
        end
        dirs = [cc(DegreeCentrality(; kind = k, kwargs = (; normalize = false)), dg)
                for k in 0:2]
        @test dirs[1] == [2, 2, 2, 2]        # total
        @test dirs[2] == [0, 1, 1, 2]        # in
        @test dirs[3] == [2, 1, 1, 0]        # out
        @test dirs[1] == dirs[2] + dirs[3]
        @test !allequal(dirs)
        @test_throws DomainError DegreeCentrality(; kind = 3)
        @test_throws DomainError DegreeCentrality(; kind = -1)
    end

    @testset "A forwarded argument reaches `Graphs.jl`" begin
        #=
        Four algorithms splat `args` and `kwargs` into the routine they select. The guards
        of `assert_centrality_args` refuse a matrix in `args`, which is asserted elsewhere;
        what is asserted here is that what survives the guard arrives and changes the
        answer.
        =#
        # `normalize = false` replaces betweenness' `1 / ((n - 1)(n - 2))` by `1 / 2`, so
        # the two answers differ by exactly `(n - 1)(n - 2) / 2`.
        n4 = Gr472.nv(P4)
        b_raw = cc(BetweennessCentrality(; kwargs = (; normalize = false)), P4)
        @test b_raw == cc(BetweennessCentrality(), P4) .* ((n4 - 1) * (n4 - 2) / 2)
        # Closeness' default carries the reachable-share factor, which is one on a
        # connected graph and bites on a disconnected one.
        two = Gr472.SimpleGraph(4)
        Gr472.add_edge!(two, 1, 2)
        Gr472.add_edge!(two, 3, 4)
        @test cc(ClosenessCentrality(; kwargs = (; normalize = false)), two) == ones(4)
        @test cc(ClosenessCentrality(), two) == fill(1 / 3, 4)
        # A positional argument is a source list, so restricting it restricts the sum.
        @test cc(StressCentrality(; args = ([1],)), P4) == [0, 2, 1, 0]
        @test cc(BetweennessCentrality(; args = ([1],)), P4) == [0, 4, 2, 0] ./ 3
    end

    @testset "`EigenvectorCentrality` discards `ct`, and `ov` still reaches the polarity" begin
        #=
        Its `calc_centrality` method is `Graphs.eigenvector_centrality(g)` with the
        algorithm dropped, so the override cannot act there. It acts one frame earlier, in
        `centrality_polarity`, which is what decides the graph.
        =#
        for g in (P4, S5, C5)
            @test isapprox(cc(EigenvectorCentrality(), g),
                           cc(EigenvectorCentrality(; ov = TopologyOnly()), g))
        end
        @test centrality_polarity(EigenvectorCentrality()) === SimilarityPolarity()
        @test centrality_polarity(EigenvectorCentrality(; ov = TopologyOnly())) === nothing
    end

    @testset "The three trees select one tree" begin
        #=
        Kruskal, Boruvka and Prim minimise the same total over the same graph, so on a
        graph with distinct edge weights they must return the same edge set. They do not
        return the same edge TYPE: `Graphs.prim_mst` answers with a `SimpleEdge` and the
        other two with a `SimpleWeightedEdge`, so a caller reading a weight off an edge
        must read it off the graph instead.
        =#
        SWG472 = PortfolioOptimisers.SimpleWeightedGraphs
        W = [0.0 1.0 4.0 3.0
             1.0 0.0 2.0 5.0
             4.0 2.0 0.0 6.0
             3.0 5.0 6.0 0.0]
        gw = SWG472.SimpleWeightedGraph(PortfolioOptimisers.graph_weight_matrix(W))
        nvw = size(W, 1)
        keyed(t) = Set((min(Gr472.src(e), Gr472.dst(e)), max(Gr472.src(e), Gr472.dst(e)))
                       for e in t)
        total(t) = sum(W[Gr472.src(e), Gr472.dst(e)] for e in t)
        ts = [PortfolioOptimisers.calc_mst(a, gw)
              for a in (KruskalTree(), BoruvkaTree(), PrimTree())]
        # The minimum spanning tree of this matrix is 1--2, 2--3, 1--4, of weight 6.
        @test all(t -> length(t) == nvw - 1, ts)
        @test all(t -> keyed(t) == Set([(1, 2), (2, 3), (1, 4)]), ts)
        @test all(t -> total(t) == 6.0, ts)
        @test allequal(keyed(t) for t in ts)
        # Prim's edge type is the odd one out.
        @test eltype(ts[3]) === Gr472.SimpleGraphs.SimpleEdge{Int}
        @test eltype(ts[1]) === eltype(ts[2]) !== eltype(ts[3])
    end

    @testset "The hop separation is the graph distance, and the ball is a power sum" begin
        #=
        `separation_matrix(::HopCount, g)` is `Graphs.gdistances` per vertex, which on a
        path is `abs(i - j)` and needs no routine to check. `_phylogeny_matrix`'s power sum
        is the other half of the same statement: the `n`-ball of that distance is the
        clamped sum of the first `n` powers of the adjacency matrix.
        =#
        p6 = Gr472.path_graph(6)
        d6 = separation_matrix(HopCount(), p6)
        @test d6 == [abs(i - j) for i in 1:6, j in 1:6]
        A6 = Matrix{Int}(Gr472.adjacency_matrix(p6))
        for n in 1:3
            # The power sum, computed here rather than read from the source.
            P = zeros(Int, 6, 6)
            for i in 0:n
                P .+= A6^i
            end
            @test Int.(d6 .<= n) - Matrix(I, 6, 6) == clamp!(P, 0, 1) - Matrix(I, 6, 6)
        end
        # An unreachable pair takes the sentinel rather than a repaired number.
        @test separation_matrix(HopCount(), Gr472.SimpleGraph(2))[1, 2] == typemax(Int)
    end

    @testset "The quantile population is the off-diagonal, both triangles" begin
        #=
        `separation_quantile` collects `d[i, j]` for every `i != j`, so a symmetric matrix
        enters each unordered pair twice. Doubling every entry leaves the empirical
        distribution alone, so the answer equals the upper triangle's quantile -- and it is
        NOT the whole matrix's, which the zero diagonal drags down.
        =#
        d4 = separation_matrix(HopCount(), P4)
        offd = [d4[i, j] for j in 1:4 for i in 1:4 if i != j]
        upper = [d4[i, j] for j in 1:4 for i in 1:4 if i < j]
        @test sort(offd) == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
        @test sort(upper) == [1, 1, 1, 2, 2, 3]
        for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            got = PortfolioOptimisers.separation_quantile(HopCount(), d4, q)
            @test got == Statistics.quantile(offd, q)
            @test got == Statistics.quantile(upper, q)
        end
        # The diagonal would move it. `q = 0.25` is `1` without it and `0.75` with it.
        @test PortfolioOptimisers.separation_quantile(HopCount(), d4, 0.25) == 1
        @test Statistics.quantile(vec(d4), 0.25) == 0.75
        # The rules round the same population: a hop count to a shell, a path length not.
        @test HopCountQuantile(; q = 0.5)(NetworkEstimator(), zeros(0, 0), P4) ==
              round(Int, Statistics.quantile(offd, 0.5))
        @test PathLengthQuantile(; q = 0.5)(NetworkEstimator(), zeros(0, 0), P4) ==
              Statistics.quantile(offd, 0.5)
    end

    @testset "The observed diameter is a ceiling, and a stated budget builds nothing" begin
        #=
        `separation_budget(::PathLength, …)` returns `min(dmax, delta)`, so a budget above
        the observed diameter is lowered to it rather than kept. And the resolved case of
        `resolve_separation` builds no structure: a stated budget must not pay for one to
        be told that it is already a value, which a `0 x 0` sample proves -- building one
        would raise long before the budget was read.
        =#
        nte = NetworkEstimator()
        gp = PortfolioOptimisers.calc_distance_weighted_graph(nte, Xt)
        d = separation_matrix(PathLength(), gp)
        diam = maximum(d)
        @test isapprox(diam, 3.4743422191984883)
        @test separation_budget(PathLength(), nte, d) == diam
        @test separation_budget(PathLength(; dmax = diam / 2), nte, d) == diam / 2
        for over in (diam + 1, 1.0e6)
            @test separation_budget(PathLength(; dmax = over), nte, d) == diam
        end
        # No structure for a stated budget, on either member.
        empty = zeros(0, 0)
        stated_h = HopCount(; n = 3)
        stated_p = PathLength(; dmax = 1.5)
        @test resolve_separation(stated_h, nte, empty) === stated_h
        @test resolve_separation(stated_p, nte, empty) === stated_p
        # The same call on a rule does build one, and therefore raises on the same sample.
        @test_throws Exception resolve_separation(HopCount(;
                                                           n = (nte, X, g; kwargs...) -> 2),
                                                  nte, empty)
    end

    @testset "The measured claims of the file's own docstrings" begin
        #=
        Five docstrings illustrate a claim with a number measured over "a 20-asset minimum
        spanning tree". The tree is this one -- the file's own SP500 window -- and the
        numbers are these. They were re-measured by the sweep of #472, because the values
        the library-wide sweep recorded belonged to a structure no fixture in the
        repository builds.
        =#
        nte = NetworkEstimator(; alg = KruskalTree())
        gT = PortfolioOptimisers.centrality_graph(nothing, nte, Xt)
        A = Gr472.adjacency_matrix(gT)
        @test Gr472.nv(gT) == 20
        @test Gr472.ne(gT) == 19

        # `DegreeCentrality`: the divisor is `n - 1`, and the first six degrees are these.
        dn = vec(sum(A; dims = 2))
        sc = cc(DegreeCentrality(), gT)
        @test dn[1:6] == [3, 2, 1, 1, 2, 3]
        @test isapprox(round.(sc[1:6]; digits = 4),
                       [0.1579, 0.1053, 0.0526, 0.0526, 0.1053, 0.1579])
        @test isapprox(sc, dn ./ 19)
        @test isapprox(maximum(abs.(dn .- sc)), 4.7368421052631575)

        # `KatzCentrality`: the bound, and what falls outside it.
        lmax = maximum(abs.(eigvals(Matrix{Float64}(A))))
        @test isapprox(lmax, 2.57344493899609)
        @test isapprox(inv(lmax), 0.388584183343788)
        @test 0.3 < inv(lmax) < 0.5
        inside = cc(KatzCentrality(; alpha = 0.3), gT)
        @test all(>(0), inside)
        @test isapprox(minimum(inside), 0.10751767636172335)
        @test isapprox(maximum(inside), 0.4642121115623047)
        # Above the bound the solve still answers, and the answer is not a centrality. It
        # is a silent wrong number by design: the constructor cannot see the graph, so the
        # bound is the caller's to respect. See the type's docstring.
        outside = cc(KatzCentrality(; alpha = 0.5), gT)
        @test count(<(0), outside) == 11
        @test isapprox(minimum(outside), -0.5562688954034373)
        @test isapprox(maximum(outside), 0.19623930476732365)
        # A denser structure over the same assets lowers the bound below the default.
        gpm = PortfolioOptimisers.centrality_graph(nothing,
                                                   NetworkEstimator(;
                                                                    alg = MaximumDistanceSimilarity()),
                                                   Xt)
        lpm = maximum(abs.(eigvals(Matrix{Float64}(Gr472.adjacency_matrix(gpm)))))
        @test isapprox(lpm, 6.174911353215694)
        @test isapprox(inv(lpm), 0.16194564468998124)
        @test inv(lpm) < 0.3

        #=
        `EigenvectorCentrality`: the two measured claims of its docstring, over the same
        triangulated maximally filtered graph. The type declares `SimilarityPolarity`, so
        the default scores the graph WEIGHTED by the similarities that selected its edges,
        and `TopologyOnly` withdraws the declaration and scores the plain `gpm` above. The
        docstring pair that stood before the P4 review of #625 -- a maximum absolute
        `0.009892049284000948` and a correlation of `0.9985` -- reproduces on neither
        route, and on no window or similarity of this fixture. The eigensolver moves by
        about `6e-16` between calls on one graph, so each claim is pinned to the digits it
        holds, and the residual is pinned as a bound.
        =#
        ntepm = NetworkEstimator(; alg = MaximumDistanceSimilarity())
        gpmw = PortfolioOptimisers.centrality_graph(ntepm, EigenvectorCentrality(), Xt)
        Aw = Matrix{Float64}(Gr472.adjacency_matrix(gpmw))
        @test Gr472.nv(gpmw) == 20
        @test Gr472.ne(gpmw) == 3 * 20 - 6
        # The weighted route carries the similarities, so its matrix is not binary.
        @test !all(a -> a in (0.0, 1.0), Aw)
        lw = maximum(abs.(eigvals(Aw)))
        @test isapprox(lw, 4.844612909369407)
        qw = cc(EigenvectorCentrality(), gpmw)
        # The formula is `EC = A q / lambda_max`, whose value is `q` itself.
        @test maximum(abs.(Aw * qw / lw - qw)) < 1e-15
        @test isapprox(norm(qw), 1)
        @test round(minimum(qw); digits = 5) == 0.07199
        @test round(maximum(qw); digits = 5) == 0.40756
        # `TopologyOnly` routes to the plain graph, which is `gpm` and its own `lambda_max`.
        gpmu = PortfolioOptimisers.centrality_graph(ntepm,
                                                    EigenvectorCentrality(;
                                                                          ov = TopologyOnly()),
                                                    Xt)
        @test Gr472.adjacency_matrix(gpmu) == Gr472.adjacency_matrix(gpm)
        qu = cc(EigenvectorCentrality(), gpmu)
        @test round(maximum(abs.(qw - qu)); digits = 5) == 0.02561
        @test round(cor(qw, qu); digits = 5) == 0.99361
        @test round(median(qw); digits = 4) == 0.1969
        @test round(median(qu); digits = 4) == 0.1969

        # `_phylogeny_matrix`: the shell-by-shell form and the power sum agree, over these
        # related-pair counts.
        for (n, want) in zip(1:4, (19, 48, 84, 115))
            Pn = phylogeny_matrix(NetworkEstimator(; alg = KruskalTree(),
                                                   sep = HopCount(; n = n)), Xt).X
            @test count(isone, Pn) ÷ 2 == want
            # `B_{1,l}`, accumulated shell by shell, is the same selection.
            shells = zeros(Int, size(A))
            for k in 1:n
                shells .+= clamp!(Matrix{Int}(A)^k + Matrix(I, 20, 20), 0, 1) -
                           Matrix(I, 20, 20)
            end
            @test maximum(abs.(Pn - (clamp!(shells, 0, 1)))) == 0
        end

        # `average_centrality` and `asset_phylogeny`: the code equals the formula exactly at
        # equal weights.
        w = fill(inv(20), 20)
        @test average_centrality(nte, DegreeCentrality(), w, Xt) == dot(sc, w)
        plr = phylogeny_matrix(NetworkEstimator(; sep = HopCount(; n = 2)), Xt)
        aw = abs.(w * transpose(w))
        @test asset_phylogeny(plr, w) == dot(plr.X, aw) / sum(aw)
    end

    @testset "`Pagerank` converges inside `n`, and raises when it does not" begin
        #=
        The default `n = 100` is enough on both branches of this library's structures, and
        the scores are a probability vector. It is not enough everywhere: a small `n` makes
        `Graphs.pagerank` RAISE rather than return an unconverged vector, which is the
        whole reason the knob is a field.
        =#
        # The sweeps each branch needs at the shipped `alpha` and `epsilon`. The tree
        # takes 57 of the default 100 and the filtered graph 14, so the default holds
        # on both and neither has much room: at `n = 50` the tree branch RAISES.
        for (alg, want) in ((KruskalTree(), 57), (MaximumDistanceSimilarity(), 14))
            g = PortfolioOptimisers.centrality_graph(nothing, NetworkEstimator(; alg = alg),
                                                     Xt)
            p = cc(Pagerank(), g)
            @test all(>=(0), p)
            @test isapprox(sum(p), 1)
            @test want < 100
            @test isapprox(cc(Pagerank(; n = want), g), p)
            @test_throws ErrorException cc(Pagerank(; n = want - 1), g)
        end
        gT = PortfolioOptimisers.centrality_graph(nothing,
                                                  NetworkEstimator(; alg = KruskalTree()),
                                                  Xt)
        @test_throws ErrorException cc(Pagerank(; n = 1, epsilon = 1e-12), gT)
        @test_throws DomainError Pagerank(; n = 0)
        @test_throws DomainError Pagerank(; epsilon = 0.0)
        @test_throws Exception Pagerank(; alpha = 1.5)
    end
end
@testset "The centrality constraint row does not depend on the orientation of `X`" begin
    using PortfolioOptimisers, Test, StableRNGs
    #=
    `centrality_constraints` accumulated each row flat and reshaped the accumulator with
    `size(X, 2)`, which is the asset count only when `dims = 1`. Under `dims = 2` on a
    non-square `X` the reshape RAISED rather than giving a wrong row: measured on an
    8x250 matrix, `DimensionMismatch: array size 8 must be divisible by the product of
    the new dimensions (250, Colon())`. Issue #516 found it. The count now follows
    `dims`, and `assert_dims` refuses any other value, which the body did not check at
    all before.
    =#
    Xc = randn(StableRNG(123456), 250, 8)
    ccc = CentralityConstraint()
    lc1 = centrality_constraints(ccc, Xc)
    lc2 = centrality_constraints(ccc, permutedims(Xc); dims = 2)
    @test size(lc1.ineq.A) == (1, 8)
    @test lc1.ineq.A == lc2.ineq.A
    @test lc1.ineq.B == lc2.ineq.B
    @test_throws DomainError centrality_constraints(ccc, Xc; dims = 3)
end
@testset "A zero centrality vector drops its constraint and reports the drop" begin
    using PortfolioOptimisers, Test, StableRNGs
    #=
    Two assets give a one-edge tree, and no vertex lies on a shortest path between two
    others, so `BetweennessCentrality` and `StressCentrality` both give the zero vector.
    The row such a constraint would build, `0' w <= B`, holds for every `w`, so
    `centrality_constraints` drops it. Issue #516 measured the drop, and issue #527 chose
    the diagnostic: the drop routes through `strict_diagnostic`, so `strict = true` raises
    and the default warns. The drop happens either way. The NAME messages are NOT reused --
    a zero centrality vector is a fact about the graph and not a typo -- which is what
    `zero_centrality_msg` exists for. The message carries the algorithm and the LENGTH of
    the vector, never its entries.
    =#
    X2 = randn(StableRNG(9), 200, 2)
    for ct in (BetweennessCentrality(), StressCentrality())
        @test all(iszero, centrality_vector(CentralityEstimator(; ct = ct), X2).X)
    end
    zc = CentralityConstraint(; A = CentralityEstimator(; ct = BetweennessCentrality()))
    dc = CentralityConstraint(; A = CentralityEstimator(; ct = DegreeCentrality()))
    @test isnothing(@test_logs (:warn, r"BetweennessCentrality") centrality_constraints(zc,
                                                                                        X2))
    lcm = @test_logs (:warn, r"BetweennessCentrality") centrality_constraints([zc, dc], X2)
    @test size(lcm.ineq.A) == (1, 2)
    @test isnothing(lcm.eq)
    @test_throws ArgumentError centrality_constraints(zc, X2; strict = true)
    @test_throws ArgumentError centrality_constraints([zc, dc], X2; strict = true)
    msg = PortfolioOptimisers.zero_centrality_msg(BetweennessCentrality(), 2)
    @test occursin("BetweennessCentrality", msg)
    @test occursin("all 2 entries", msg)
    @test occursin("empty",
                   PortfolioOptimisers.zero_centrality_msg(BetweennessCentrality(), 0))
    @test_throws PortfolioOptimisers.IsEmptyError centrality_constraints(CentralityConstraint[],
                                                                         X2)
end
@testset "The centrality threshold is read off the unscaled vector" begin
    using PortfolioOptimisers, Test, StableRNGs
    #=
    `B = d * vec_to_real_measure(cc.B, A)` runs BEFORE `A .*= d`, so the measure never
    sees the sign-flipped row. Deriving it from the flipped row would negate it twice and
    turn a `MinValue()` into a `MaxValue()` with the wrong sign. On this eight-asset
    degree vector the smallest entry is 1/7 and the largest 3/7, so `>=` must give exactly
    the negation of what `<=` gives.
    =#
    Xc = randn(StableRNG(123456), 250, 8)
    cv = centrality_vector(CentralityEstimator(), Xc).X
    @test isapprox(minimum(cv), 1 / 7)
    @test isapprox(maximum(cv), 3 / 7)
    for (Bm, want) in ((MinValue(), minimum(cv)), (MaxValue(), maximum(cv)), (0.2, 0.2))
        le = centrality_constraints(CentralityConstraint(; B = Bm, comp = <=), Xc)
        ge = centrality_constraints(CentralityConstraint(; B = Bm, comp = >=), Xc)
        eq = centrality_constraints(CentralityConstraint(; B = Bm, comp = ==), Xc)
        @test le.ineq.B == [want]
        @test ge.ineq.B == [-want]
        @test le.ineq.A == -ge.ineq.A
        # `==` builds an equality row and leaves the inequality half empty.
        @test isnothing(eq.ineq)
        @test eq.eq.B == [want]
        @test eq.eq.A == le.ineq.A
    end
    # A mixed vector puts each row in its own half, in the order it was given.
    mixed = centrality_constraints([CentralityConstraint(; comp = <=),
                                    CentralityConstraint(; comp = ==, B = MaxValue()),
                                    CentralityConstraint(; comp = >=, B = 0.3)], Xc)
    @test size(mixed.ineq.A) == (2, 8)
    @test size(mixed.eq.A) == (1, 8)
    @test isapprox(mixed.ineq.B, [minimum(cv), -0.3])
    @test isapprox(mixed.eq.B, [maximum(cv)])
end
@testset "A centrality constraint checks no value it is given" begin
    using PortfolioOptimisers, Test, StableRNGs
    #=
    `CentralityConstraint`'s inner constructor validates nothing, and
    `centrality_constraints` adds no check of its own, so a non-finite threshold reaches
    the generated right-hand side unchanged and the model carries it to the solver. Every
    argument is bounded by its type alone. Measured under issue #516.
    =#
    Xc = randn(StableRNG(123456), 250, 8)
    @test isnan(centrality_constraints(CentralityConstraint(; B = NaN), Xc).ineq.B[1])
    @test centrality_constraints(CentralityConstraint(; B = Inf), Xc).ineq.B == [Inf]
    @test_throws TypeError CentralityConstraint(; comp = <)
    @test_throws TypeError CentralityConstraint(; A = NetworkEstimator())
end
@testset "`IntegerPhylogeny` stores the deduplicated row set, not the matrix it is given" begin
    using PortfolioOptimisers, Test, LinearAlgebra
    #=
    The inner constructor replaces `A` with `unique(A + I; dims = 1)`, so `ip.A` is a
    DERIVED quantity: one row per distinct neighbourhood or cluster, which is why the
    stored matrix is usually shorter than it is wide. `length(B)` is checked against that
    row count and not against the number of assets. Measured under issue #516.
    =#
    # Four assets in two clusters of two give two distinct rows out of four.
    Acl = Float64[0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]
    ipc = IntegerPhylogeny(; A = Acl, B = [1, 1])
    @test size(ipc.A) == (2, 4)
    @test ipc.A == Float64[1 1 0 0; 0 0 1 1]
    # A path relates three distinct neighbourhoods, so no row repeats and none is dropped.
    Apa = Float64[0 1 0; 1 0 1; 0 1 0]
    ipp = IntegerPhylogeny(; A = Apa, B = [1, 1, 1])
    @test size(ipp.A) == (3, 3)
    @test ipp.A == Apa + I
    # The vector `B` is checked after the deduplication: four assets need two entries.
    @test_throws DimensionMismatch IntegerPhylogeny(; A = Acl, B = [1, 1, 1, 1])
    # A scalar `B` applies to every row, so it needs no length at all.
    @test IntegerPhylogeny(; A = Acl, B = 1).B == 1
    # The two guards that run before the rewrite.
    @test_throws ArgumentError IntegerPhylogeny(; A = Float64[1 1; 1 1], B = 1)
    @test_throws ArgumentError IntegerPhylogeny(; A = Float64[0 1; 0 0], B = 1)
end
@testset "The integer phylogeny `B` guard fires only for a stated cluster count" begin
    using PortfolioOptimisers, Test
    #=
    `validate_length_integer_phylogeny_constraint_B` has four methods across two names,
    and only the `ClustersEstimator` one does anything. A `NetworkEstimator` reaches the
    fallback and passes at any length, and so does the DEFAULT `OptimalNumberClusters()`,
    whose `max_k` is `nothing` and whose `alg` is a `SecondOrderDifference` rather than an
    integer. Measured under issue #516.
    =#
    V = PortfolioOptimisers.validate_length_integer_phylogeny_constraint_B
    _V = PortfolioOptimisers._validate_length_integer_phylogeny_constraint_B
    @test isnothing(V(ClustersEstimator(; onc = OptimalNumberClusters(; max_k = 3)),
                      [1, 1, 1]))
    @test_throws DomainError V(ClustersEstimator(;
                                                 onc = OptimalNumberClusters(; max_k = 3)),
                               [1, 1, 1, 1])
    # An integer `alg` bounds the length through the private delegate.
    @test isnothing(V(ClustersEstimator(; onc = OptimalNumberClusters(; alg = 4)),
                      collect(1:4)))
    @test_throws DomainError V(ClustersEstimator(; onc = OptimalNumberClusters(; alg = 4)),
                               collect(1:5))
    # Neither bound exists on the default, so a `B` of any length passes.
    dflt = ClustersEstimator()
    @test isnothing(dflt.onc.max_k)
    @test isa(dflt.onc.alg, SecondOrderDifference)
    @test isnothing(V(dflt, collect(1:50)))
    # Every guard is a no-op for a network source, the commonest one.
    @test isnothing(V(NetworkEstimator(), collect(1:99)))
    @test isnothing(_V(nothing, collect(1:99)))
end
@testset "`phylogeny_constraints` passes a result through and keeps a vector in order" begin
    using PortfolioOptimisers, Test, StableRNGs
    #=
    Four methods: one per estimator, an identity for a precomputed result or `nothing`,
    and a broadcast over a vector. The identity returns the SAME object and builds
    nothing, which is what lets a caller hand a fitted constraint back into the pipeline.
    Measured under issue #516.
    =#
    Xp = randn(StableRNG(123456), 250, 8)
    sdp = phylogeny_constraints(SemiDefinitePhylogenyEstimator(), Xp)
    ipr = phylogeny_constraints(IntegerPhylogenyEstimator(), Xp)
    @test isa(sdp, SemiDefinitePhylogeny)
    @test size(sdp.A) == (8, 8)
    @test sdp.p == 0.05
    @test isa(ipr, IntegerPhylogeny)
    @test ipr.B == 1
    pre = SemiDefinitePhylogeny(; A = [0.0 1.0; 1.0 0.0])
    @test phylogeny_constraints(pre, Xp) === pre
    @test isnothing(phylogeny_constraints(nothing, Xp))
    vres = phylogeny_constraints(PortfolioOptimisers.PlCE_PlC[IntegerPhylogenyEstimator(),
                                                              SemiDefinitePhylogenyEstimator(),
                                                              pre], Xp)
    @test isa(vres[1], IntegerPhylogeny)
    @test isa(vres[2], SemiDefinitePhylogeny)
    @test vres[3] === pre
    # The guards of the semidefinite pair.
    @test_throws DomainError SemiDefinitePhylogenyEstimator(; p = -1.0)
    @test_throws DomainError SemiDefinitePhylogeny(; A = [0.0 1.0; 1.0 0.0], p = -0.5)
    @test_throws ArgumentError SemiDefinitePhylogeny(; A = [0.0 1.0; 0.0 0.0])
    @test_throws ArgumentError SemiDefinitePhylogeny(; A = [1.0 1.0; 1.0 1.0])
end
# `DistanceDistance` passes `dims = 2` to `Distances.pairwise` itself, issue #634. The
# implicit `dims` of `Distances.pairwise` is a deprecation, so the call must state the axis.
# The literal sits before the splat of `de.kwargs`, so a caller's own `dims` still wins.
# Defined at top level because a `@testset` body becomes a function.
@testset "DistanceDistance states the pairwise axis (#634)" begin
    using PortfolioOptimisers, Test, StableRNGs, Statistics
    rng = StableRNG(987654321)
    X634 = randn(rng, 60, 5)
    rho634 = cor(X634)
    ce634 = PortfolioOptimisersCovariance()
    metric634 = PortfolioOptimisers.Distances.Euclidean()
    # A second matrix that is square and not symmetric. The two axes then give two
    # different answers, so the axis the call takes is observable.
    B634 = randn(rng, 5, 5)
    de2 = DistanceDistance(; metric = metric634, args = (B634,))
    # A `dims` in the estimator's own `kwargs` must override the literal.
    de1 = DistanceDistance(; metric = metric634, args = (B634,), kwargs = (; dims = 1))
    base = Distance()
    for (Dbase, res2, res1) in
        ((distance(base, rho634), distance(de2, rho634), distance(de1, rho634)),
         (distance(base, ce634, X634), distance(de2, ce634, X634),
          distance(de1, ce634, X634)),
         (cor_and_dist(base, ce634, X634)[2], cor_and_dist(de2, ce634, X634)[2],
          cor_and_dist(de1, ce634, X634)[2]))
        d2 = PortfolioOptimisers.Distances.pairwise(metric634, Dbase, B634; dims = 2)
        d1 = PortfolioOptimisers.Distances.pairwise(metric634, Dbase, B634; dims = 1)
        # The axes disagree, so the two assertions below each pin one axis.
        @test !isapprox(d1, d2)
        @test isapprox(res2, d2)
        @test isapprox(res1, d1)
    end
end
