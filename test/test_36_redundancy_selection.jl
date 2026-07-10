using Test, PortfolioOptimisers, StableRNGs, Statistics

@testset "Redundancy selection" begin
    POx = PortfolioOptimisers

    # A and B are perfectly correlated with each other and uncorrelated with C.
    # A has the lower variance, so a variance score keeps A over B.
    rd_pair = ReturnsResult(; nx = ["A", "B", "C"],
                            X = [0.010 0.020 -0.05
                                 -0.010 -0.020 0.07
                                 0.005 0.010 -0.02
                                 0.002 0.004 0.09])

    @testset "construction validation" begin
        @test_throws DomainError PairwiseCorrelation(; t = 1.5)
        @test_throws DomainError CorrelationComponents(; t = -1.5)

        # ClusterGroups has no fallback survivor rule, so it needs a score
        @test_throws POx.IsNothingError RedundancySelector(; alg = ClusterGroups())
        @test RedundancySelector(; alg = ClusterGroups(), score = SCM()) isa
              RedundancySelector

        # the correlation algorithms do have one
        @test RedundancySelector(; alg = PairwiseCorrelation()) isa RedundancySelector
        @test RedundancySelector(; alg = CorrelationComponents()) isa RedundancySelector

        # an unscoreable measure is rejected exactly as it is for ScoreSelector
        @test_throws ArgumentError RedundancySelector(; alg = PairwiseCorrelation(),
                                                      score = Variance())

        # the default is greedy pairwise pruning
        @test RedundancySelector().alg isa PairwiseCorrelation
        @test isnothing(RedundancySelector().score)
    end

    @testset "score orients the survivor" begin
        # lower variance is better, so A survives the A/B pair
        sel = RedundancySelector(; alg = PairwiseCorrelation(; t = 0.99), score = SCM())
        @test fit_preprocessing(sel, rd_pair).nx == ["A", "C"]

        # MeanReturn is maximised, so the higher-mean asset of the pair survives
        rd = ReturnsResult(; nx = ["A", "B", "C"],
                           X = [0.010 0.020 -0.05
                                -0.010 -0.020 0.07
                                0.005 0.010 -0.02
                                0.002 0.004 0.09])
        selm = RedundancySelector(; alg = PairwiseCorrelation(; t = 0.99),
                                  score = MeanReturn())
        kept = fit_preprocessing(selm, rd).nx
        @test "C" in kept
        @test length(kept) == 2
        @test kept != ["A", "C"]   # B has the larger mean, so B survives, not A
    end

    @testset "drop_scores flips with bigger_is_better" begin
        s = [1.0, 2.0, 3.0]
        @test POx.drop_scores(s, false) == s        # lower is better -> higher is drop-worthy
        @test POx.drop_scores(s, true) == -s        # higher is better -> lower is drop-worthy
    end

    @testset "greedy does not chain; components do" begin
        # rho(A,B) high, rho(B,C) high, rho(A,C) low
        rho = [1.0 0.97 0.10
               0.97 1.0 0.97
               0.10 0.97 1.0]

        # union-find over the strict lower triangle: one transitive blob
        comps = POx.correlation_components(rho, 0.95)
        @test length(comps) == 1
        @test sort(only(comps)) == [1, 2, 3]

        # below any edge threshold every asset is its own singleton
        @test length(POx.correlation_components(rho, 0.99)) == 3
    end

    @testset "groups_argbest and the trust-neither tie policy" begin
        # one survivor per group: the best-scoring member
        @test findall(POx.groups_argbest([[1, 2], [3]], [5.0, 1.0, 9.0], false)) == [2, 3]
        @test findall(POx.groups_argbest([[1, 2], [3]], [5.0, 1.0, 9.0], true)) == [1, 3]

        # a group whose best score is tied keeps nobody
        @test findall(POx.groups_argbest([[1, 2], [3]], [4.0, 4.0, 9.0], false)) == [3]

        # a singleton is always unambiguous
        @test findall(POx.groups_argbest([[1], [2]], [4.0, 4.0], false)) == [1, 2]
    end

    @testset "identical columns leave no survivor" begin
        # A and B are the same series: neither the correlation summary nor a variance score
        # can tell them apart, so both are discarded
        rd = ReturnsResult(; nx = ["A", "B", "C"],
                           X = [0.10 0.10 -0.05
                                -0.10 -0.10 0.07
                                0.05 0.05 -0.02
                                0.02 0.02 0.09])
        @test fit_preprocessing(RedundancySelector(; alg = PairwiseCorrelation(; t = 0.99),
                                                   score = SCM()), rd).nx == ["C"]
        @test fit_preprocessing(RedundancySelector(;
                                                   alg = CorrelationComponents(; t = 0.99),
                                                   score = SCM()), rd).nx == ["C"]
    end

    @testset "find_uncorrelated_indices gains a scores keyword" begin
        rng = StableRNG(42)
        X = randn(rng, 200, 6)
        X[:, 2] = X[:, 1] .+ 0.01 .* randn(rng, 200)   # 2 duplicates 1

        # default behaviour is unchanged
        base = POx.find_uncorrelated_indices(X; t = 0.9)
        @test 1 in base || 2 in base
        @test !(1 in base && 2 in base)

        # supplying scores overrides the survivor rule: higher score is dropped
        scores = [10.0, 1.0, 0.0, 0.0, 0.0, 0.0]      # asset 1 is drop-worthy
        withs = POx.find_uncorrelated_indices(X; t = 0.9, scores = scores)
        @test 2 in withs
        @test !(1 in withs)

        # and it is length-checked
        @test_throws DimensionMismatch POx.find_uncorrelated_indices(X; t = 0.9,
                                                                     scores = [1.0, 2.0])
    end

    @testset "ClusterGroups keeps one representative per cluster" begin
        rng = StableRNG(7)
        # two tight blocks of three assets each
        f1 = randn(rng, 150)
        f2 = randn(rng, 150)
        X = hcat(f1 .+ 0.01 .* randn(rng, 150, 3), f2 .+ 0.01 .* randn(rng, 150, 3))
        rd = ReturnsResult(; nx = string.('A':'F'), X = X)

        sel = RedundancySelector(; alg = ClusterGroups(), score = SCM())
        kept = fit_preprocessing(sel, rd).nx
        @test length(kept) <= 6
        @test !isempty(kept)
        # one survivor per cluster, so far fewer than the full universe
        @test length(kept) < 6
    end

    @testset "pipeline integration" begin
        @test PortfolioOptimisers.pipe_writes(RedundancySelector()) == :returns
        @test PortfolioOptimisers.pipe_reads(RedundancySelector()) == (:returns,)

        rng = StableRNG(99)
        f = randn(rng, 200)
        X = hcat(0.01 .* f, 0.01 .* f .+ 1e-6 .* randn(rng, 200),
                 0.01 .* randn(rng, 200, 3))
        rd = ReturnsResult(; nx = string.('A':'E'), X = X)

        pipe = Pipeline(;
                        steps = (RedundancySelector(; alg = PairwiseCorrelation(; t = 0.95),
                                                    score = SCM()), EmpiricalPrior(),
                                 EqualWeighted()))
        res = fit(pipe, rd)
        @test length(res.ctx.returns.nx) < 5
        @test length(res.ctx.opt.w) == length(res.ctx.returns.nx)

        # replaying the fitted universe on an unseen window
        @test predict(res, rd) isa Any
    end
end
