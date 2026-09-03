#=
The incremental fit of the third and fourth co-moments, issue #703.

Four properties are asserted, and they are the four the ticket names: the per-row fit
reproduces the batch number to machine precision, the merge of two states is associative, the
`SemiMoment` arm refuses, and the state's element count is written down so a later change that
widens it is visible.

The last four testsets are the seam-wide rulings of #712, built by #714 and recorded in ADR
0107: every state copies without aliasing, the value form `partial_fit` leaves a kept estimator
alone, each propagation channel does one thing with the state, and a read-out refuses a state
its estimator no longer matches.
=#
using Statistics

@testset "Higher-moment partial fit" begin
    PO = PortfolioOptimisers

    rng = StableRNG(987654321)
    X = randn(rng, 37, 3)
    N = size(X, 2)

    @testset "Exactness against the batch verb" begin
        cskew_b, V_b = coskewness(Coskewness(), X)
        ske = Coskewness()
        for i in axes(X, 1)
            ske = partial_fit!(ske, X[i:i, :])
        end
        cskew_i, V_i = coskewness(ske)
        @test isapprox(cskew_i, cskew_b; atol = 1e-14)
        @test isapprox(V_i, V_b; atol = 1e-14)
        @test ske.cache.n == size(X, 1)
        @test isapprox(ske.cache.mu, vec(mean(X; dims = 1)); atol = 1e-14)

        ckurt_b = cokurtosis(Cokurtosis(), X)
        kte = Cokurtosis()
        for i in axes(X, 1)
            kte = partial_fit!(kte, X[i:i, :])
        end
        @test isapprox(cokurtosis(kte), ckurt_b; atol = 1e-13)

        # One block is the same answer as one row at a time, and `dims = 2` orients the
        # transpose of the same block.
        @test isapprox(first(coskewness(partial_fit!(Coskewness(), X))), cskew_b;
                       atol = 1e-14)
        @test isapprox(cokurtosis(partial_fit!(Cokurtosis(), X)), ckurt_b; atol = 1e-13)
        @test isapprox(first(coskewness(partial_fit!(Coskewness(), transpose(X); dims = 2))),
                       cskew_b; atol = 1e-14)
        @test isapprox(cokurtosis(partial_fit!(Cokurtosis(), transpose(X); dims = 2)),
                       ckurt_b; atol = 1e-13)

        # The single-observation arm of the interface folds one row at a time.
        ske_v = Coskewness()
        kte_v = Cokurtosis()
        for i in axes(X, 1)
            ske_v = partial_fit!(ske_v, X[i, :])
            kte_v = partial_fit!(kte_v, X[i, :])
        end
        @test isapprox(first(coskewness(ske_v)), cskew_b; atol = 1e-14)
        @test isapprox(cokurtosis(kte_v), ckurt_b; atol = 1e-13)

        # The state read-out equals the read-out through the estimator's own field.
        ske2 = partial_fit!(Coskewness(), X)
        @test first(coskewness(ske2, ske2.cache)) == first(coskewness(ske2))
        kte2 = partial_fit!(Cokurtosis(), X)
        @test cokurtosis(kte2, kte2.cache) == cokurtosis(kte2)
    end

    @testset "Associativity of the merge" begin
        A = X[1:9, :]
        B = X[10:22, :]
        C = X[23:37, :]

        sa = partial_fit!(Coskewness(), A).cache
        sb = partial_fit!(Coskewness(), B).cache
        sc = partial_fit!(Coskewness(), C).cache
        lft = PO.merge_states(PO.merge_states(sa, sb), sc)
        rgt = PO.merge_states(sa, PO.merge_states(sb, sc))
        @test lft.n == rgt.n == size(X, 1)
        @test isapprox(lft.mu, rgt.mu; atol = 1e-14)
        @test isapprox(lft.M2, rgt.M2; atol = 1e-13)
        @test isapprox(lft.M3, rgt.M3; atol = 1e-13)
        @test isapprox(first(coskewness(Coskewness(), lft)),
                       first(coskewness(Coskewness(), X)); atol = 1e-14)

        ka = partial_fit!(Cokurtosis(), A).cache
        kb = partial_fit!(Cokurtosis(), B).cache
        kc = partial_fit!(Cokurtosis(), C).cache
        klft = PO.merge_states(PO.merge_states(ka, kb), kc)
        krgt = PO.merge_states(ka, PO.merge_states(kb, kc))
        @test klft.n == krgt.n == size(X, 1)
        @test isapprox(klft.M3, krgt.M3; atol = 1e-13)
        @test isapprox(klft.M4, krgt.M4; atol = 1e-12)
        @test isapprox(cokurtosis(Cokurtosis(), klft), cokurtosis(Cokurtosis(), X);
                       atol = 1e-13)

        # A merge is refused across families and across asset counts.
        @test_throws ArgumentError PO.merge_states(sa, ka)
        sn = partial_fit!(Coskewness(), X[:, 1:2]).cache
        @test_throws DimensionMismatch PO.merge_states(sa, sn)
    end

    @testset "The SemiMoment arm refuses" begin
        for est in (Coskewness(; alg = SemiMoment()), Cokurtosis(; alg = SemiMoment()))
            err = try
                partial_fit!(est, X)
                nothing
            catch e
                e
            end
            @test isa(err, ArgumentError)
            @test occursin("SemiMoment", err.msg)
            @test occursin("clip", err.msg)
        end
    end

    @testset "The configuration refusals" begin
        w = pweights(fill(1 / size(X, 1), size(X, 1)))
        @test_throws ArgumentError partial_fit!(Coskewness(; w = w), X)
        @test_throws ArgumentError partial_fit!(Cokurtosis(; w = w), X)
        @test_throws ArgumentError partial_fit!(Coskewness(;
                                                           me = SimpleExpectedReturns(;
                                                                                      w = w)),
                                                X)
        @test_throws ArgumentError partial_fit!(Cokurtosis(; me = MedianExpectedReturns()),
                                                X)
        # An estimator shown no observation has nothing to read out.
        @test_throws ArgumentError coskewness(Coskewness())
        @test_throws ArgumentError cokurtosis(Cokurtosis())
    end

    @testset "The state's element count" begin
        ske = partial_fit!(Coskewness(), X)
        kte = partial_fit!(Cokurtosis(), X)
        @test isa(ske.cache, PO.AbstractPartialFitState)
        @test isa(kte.cache, PO.AbstractPartialFitState)
        @test length(ske.cache.mu) == N
        @test length(ske.cache.M2) == N^2
        @test length(ske.cache.M3) == N^3
        @test sum(length, (ske.cache.mu, ske.cache.M2, ske.cache.M3)) == N + N^2 + N^3
        @test length(kte.cache.M4) == N^4
        @test sum(length, (kte.cache.mu, kte.cache.M2, kte.cache.M3, kte.cache.M4)) ==
              N + N^2 + N^3 + N^4
        # The state is the one Result an estimator holds, and it is hidden while it is unset.
        @test isnothing(Coskewness().cache)
        @test :cache ∉ PO.pretty_show_fields(Coskewness())
        @test :cache ∈ PO.pretty_show_fields(ske)
    end

    @testset "Every state copies without aliasing" begin
        ske = partial_fit!(Coskewness(), X)
        kte = partial_fit!(Cokurtosis(), X)
        for state in (ske.cache, kte.cache)
            twin = copy(state)
            @test typeof(twin) === typeof(state)
            @test twin !== state
            @test twin.n == state.n
            for name in fieldnames(typeof(state))
                a = getfield(state, name)
                b = getfield(twin, name)
                if isa(a, AbstractArray)
                    @test a == b
                    @test a !== b
                end
            end
        end
    end

    @testset "The value form leaves the kept estimator alone" begin
        # This family overrides `partial_fit`, because its fold builds a fresh state in
        # either verb, so the generic copy of `M4` buys nothing. ADR 0107.
        warm_ske = partial_fit!(Coskewness(), view(X, 1:20, :))
        warm_kte = partial_fit!(Cokurtosis(), view(X, 1:20, :))
        kept_ske = first(coskewness(warm_ske))
        kept_kte = cokurtosis(warm_kte)

        fitted_ske = partial_fit(warm_ske, X[21:end, :])
        fitted_kte = partial_fit(warm_kte, X[21:end, :])
        @test first(coskewness(warm_ske)) == kept_ske
        @test cokurtosis(warm_kte) == kept_kte
        @test warm_ske.cache !== fitted_ske.cache
        @test warm_kte.cache !== fitted_kte.cache
        # The value form answers the whole sample, to the precision the batch verb does.
        @test isapprox(first(coskewness(fitted_ske)), first(coskewness(Coskewness(), X));
                       atol = 1e-14)
        @test isapprox(cokurtosis(fitted_kte), cokurtosis(Cokurtosis(), X); atol = 1e-14)
        # A cold estimator has no state to copy, and the fold seeds one of its own.
        @test isapprox(cokurtosis(partial_fit(Cokurtosis(), X)),
                       cokurtosis(Cokurtosis(), X); atol = 1e-14)
    end

    @testset "Each channel does one thing with the state" begin
        ske = partial_fit!(Coskewness(), X)
        kte = partial_fit!(Cokurtosis(), X)
        # `factory` carries the state, and `obs_weights_view` drops it. ADR 0107.
        @test PO.factory(ske).cache === ske.cache
        @test PO.factory(kte).cache === kte.cache
        @test isnothing(PO.obs_weights_view(ske, 1:10).cache)
        @test isnothing(PO.obs_weights_view(kte, 1:10).cache)
        # The root drop answers for any state, so a family that adds one owes no method.
        @test isnothing(PO.obs_weights_view(ske.cache, 1:10))
        @test isnothing(PO.obs_weights_view(kte.cache, [2, 5]))

        # `port_opt_view` slices to the selected assets, and the slice is exact: the viewed
        # estimator reads what the same estimator reads over those columns alone.
        i = [1, 3]
        vske = PO.port_opt_view(ske, i)
        vkte = PO.port_opt_view(kte, i)
        @test isapprox(first(coskewness(vske)), first(coskewness(Coskewness(), X[:, i]));
                       atol = 1e-14)
        @test isapprox(cokurtosis(vkte), cokurtosis(Cokurtosis(), X[:, i]); atol = 1e-14)
        @test size(vske.cache.M3) == (length(i), length(i)^2)
        @test size(vkte.cache.M4) == (length(i)^2, length(i)^2)

        # The slice copies by index rather than viewing, so a fold on the viewed estimator
        # writes into arrays of its own.
        before = copy(kte.cache.M4)
        partial_fit!(vkte, X[1:1, i])
        @test kte.cache.M4 == before
    end

    @testset "A read-out refuses a state its estimator no longer matches" begin
        # `factory` carries the state and replaces `w`, so the estimator says weighted and
        # holds a state fitted unweighted. Both read-outs refuse. ADR 0107.
        w = pweights(fill(1 / size(X, 1), size(X, 1)))
        ske = PO.factory(partial_fit!(Coskewness(), X), w)
        kte = PO.factory(partial_fit!(Cokurtosis(), X), w)
        @test !isnothing(ske.cache)
        @test !isnothing(kte.cache)
        @test_throws ArgumentError coskewness(ske)
        @test_throws ArgumentError cokurtosis(kte)
    end
end
