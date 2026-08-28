using Test, PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra

# an algorithm that implements no redundancy_keep, for the erroring-fallback test
struct UnimplementedRedundancy <: PortfolioOptimisers.AbstractRedundancyAlgorithm end

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

    @testset "requires_score and the erroring fallback" begin
        # the generic method answers `true`, so a new algorithm is asked for a score
        @test POx.requires_score(UnimplementedRedundancy())
        @test POx.requires_score(ClusterGroups())
        @test !POx.requires_score(PairwiseCorrelation())
        @test !POx.requires_score(CorrelationComponents())

        # ClusterGroups needs a score, and the message says so
        err = try
            RedundancySelector(; alg = ClusterGroups())
        catch e
            e
        end
        @test err isa POx.IsNothingError
        msg = sprint(showerror, err)
        @test occursin("ClusterGroups", msg)
        @test occursin("give the RedundancySelector a score", msg)

        # an algorithm implementing no redundancy_keep hits the family's erroring fallback
        rd = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; -0.1 0.3; 0.2 0.1])
        @test_throws ArgumentError POx.redundancy_keep(UnimplementedRedundancy(), rd,
                                                       [1.0, 2.0], false)
        e2 = try
            POx.redundancy_keep(UnimplementedRedundancy(), rd, nothing, false)
        catch e
            e
        end
        @test occursin("does not implement redundancy_keep", e2.msg)
        @test occursin("requires_score", e2.msg)
    end

    @testset "correlation_components partitions the universe" begin
        rho = [1.0 0.97 0.10 0.05
               0.97 1.0 0.12 0.02
               0.10 0.12 1.0 0.01
               0.05 0.02 0.01 1.0]
        comps = POx.correlation_components(rho, 0.95)
        # every index lands in exactly one component
        @test sort(vcat(comps...)) == 1:4
        @test sort(sort.(comps)) == [[1, 2], [3], [4]]
        # an asset with no over-threshold partner is a singleton
        @test count(g -> length(g) == 1, comps) == 2

        # `absolute` joins a strongly NEGATIVE pair, and the raw comparison does not
        rneg = [1.0 -0.97; -0.97 1.0]
        @test length(POx.correlation_components(rneg, 0.95)) == 2
        @test length(POx.correlation_components(abs.(rneg), 0.95)) == 1

        # thresholds outside a correlation's range are rejected; both endpoints are legal
        @test isnothing(POx.assert_correlation_threshold(-1.0))
        @test isnothing(POx.assert_correlation_threshold(1.0))
        @test_throws DomainError POx.assert_correlation_threshold(-1.0001)
        @test_throws DomainError POx.assert_correlation_threshold(1.0001)
    end

    @testset "transitivity: components chain, pairwise pruning does not" begin
        # the docstring's own chain, and the only kind that is realisable: two edges at rho
        # force the third above rho1*rho2 - sqrt((1 - rho1^2)(1 - rho2^2))
        rho = [1.0 0.80 0.32
               0.80 1.0 0.81
               0.32 0.81 1.0]
        @test minimum(eigvals(rho)) > 0      # a correlation matrix, not just a table

        rng = StableRNG(20260828)
        Z = Matrix(qr(randn(rng, 500, 3)).Q)[:, 1:3]
        Z = (Z .- mean(Z; dims = 1)) ./ std(Z; dims = 1)
        X = Z * cholesky(rho).L'
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = X)
        @test isapprox(cor(X), rho; atol = 5e-3)

        # union-find reads the chain as ONE component of three
        comps = POx.correlation_components(cor(X), 0.7)
        @test length(comps) == 1
        @test sort(only(comps)) == [1, 2, 3]

        # so the components algorithm keeps one asset, and greedy pruning keeps more
        cc = POx.select_assets(RedundancySelector(; alg = CorrelationComponents(; t = 0.7)),
                               rd)
        pw = POx.select_assets(RedundancySelector(; alg = PairwiseCorrelation(; t = 0.7)),
                               rd)
        @test count(cc) == 1
        @test count(pw) == 2
        @test count(pw) > count(cc)
        # greedy drops the middle asset and keeps both ends, as PairwiseCorrelation states
        @test findall(pw) == [1, 3]
        @test findall(cc) == [1]
    end

    @testset "the fallback survivor, and a score that overrides it" begin
        rng = StableRNG(11)
        f = randn(rng, 400)
        h = randn(rng, 400)
        # A is tied to both f and h, but scaled small; B is tied to f alone, at full scale
        A = 0.1 .* (f .+ 0.05 .* randn(rng, 400) .+ 0.30 .* h)
        B = f .+ 0.05 .* randn(rng, 400)
        X = hcat(A, B, h)
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = X)

        rho = cor(X)
        summ = [POx.vec_to_real_measure(MeanValue(), x) for x in eachcol(rho)]
        scm = POx.asset_scores(SCM(), X)
        # A and B are one component, and the two rules disagree inside it
        @test sort(sort.(POx.correlation_components(rho, 0.9))) == [[1, 2], [3]]
        @test summ[2] < summ[1]      # B is the least redundant of the pair
        @test scm[1] < scm[2]        # A has the lower variance

        # with no score the least redundant member survives
        @test findall(POx.select_assets(RedundancySelector(;
                                                           alg = CorrelationComponents(;
                                                                                       t = 0.9)),
                                        rd)) == [2, 3]
        # supplying a score overrides that choice
        @test findall(POx.select_assets(RedundancySelector(;
                                                           alg = CorrelationComponents(;
                                                                                       t = 0.9),
                                                           score = SCM()), rd)) == [1, 3]

        # the same nothing-score path through greedy pruning
        keep = POx.select_assets(RedundancySelector(; alg = PairwiseCorrelation(; t = 0.9)),
                                 rd)
        @test count(keep) == 2
        @test 3 in findall(keep)

        # `absolute` reaches the components algorithm too
        Xneg = hcat(f, -f .+ 0.01 .* randn(rng, 400), h)
        rdneg = ReturnsResult(; nx = ["A", "B", "C"], X = Xneg)
        @test count(POx.select_assets(RedundancySelector(;
                                                         alg = CorrelationComponents(;
                                                                                     t = 0.9,
                                                                                     absolute = true)),
                                      rdneg)) == 2
        @test count(POx.select_assets(RedundancySelector(;
                                                         alg = CorrelationComponents(;
                                                                                     t = 0.9,
                                                                                     absolute = false)),
                                      rdneg)) == 3
    end

    @testset "the drop-score sign reaches find_uncorrelated_indices" begin
        rng = StableRNG(5)
        f = randn(rng, 300)
        A = 0.01 .* f .+ 0.0002 .* randn(rng, 300)
        B = 0.01 .* f .+ 0.0002 .* randn(rng, 300) .+ 0.02   # B has the far larger mean
        X = hcat(A, B, 0.01 .* randn(rng, 300))
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = X)

        means = POx.asset_scores(MeanReturn(), X)
        @test PortfolioOptimisers.bigger_is_better(MeanReturn())
        @test means[2] > means[1]
        @test cor(A, B) > 0.95

        # a bigger-is-better measure must keep the HIGH-scoring asset of the pair
        @test findall(POx.select_assets(RedundancySelector(;
                                                           alg = PairwiseCorrelation(;
                                                                                     t = 0.95),
                                                           score = MeanReturn()), rd)) ==
              [2, 3]
        # and a lower-is-better one keeps the low-scoring asset
        @test POx.asset_scores(SCM(), X)[1] < POx.asset_scores(SCM(), X)[2]
        @test findall(POx.select_assets(RedundancySelector(;
                                                           alg = PairwiseCorrelation(;
                                                                                     t = 0.95),
                                                           score = SCM()), rd)) == [1, 3]
    end

    @testset "groups_argbest skips an empty group" begin
        # an empty group contributes nothing and does not stop the walk
        @test findall(POx.groups_argbest([Int[], [1, 2]], [1.0, 2.0], false)) == [1]
        @test findall(POx.groups_argbest([[1, 2], Int[]], [1.0, 2.0], true)) == [2]
        @test !any(POx.groups_argbest([Int[]], [1.0, 2.0], false))
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
