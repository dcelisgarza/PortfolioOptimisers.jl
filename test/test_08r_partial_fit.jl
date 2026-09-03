#=
The incremental fit of the four second-order sample estimators, issue #700.

The four properties the seam owes, in the order the ticket states them: exactness against the
batch method, associativity of `merge_states`, the `NaN` a state with too few observations
reads, and the guarantee that `partial_fit!` leaves the caller's estimator untouched. The
refusals follow, one per configuration that the batch answer and the Welford accumulator
disagree on.

The last four testsets are the seam-wide rulings of #712, built by #714 and recorded in ADR
0107: each propagation channel does one thing with the state, the value form `partial_fit`
leaves a kept estimator alone, every state copies without aliasing, and a read-out refuses a
state its estimator no longer matches.
=#
@testset "Partial fit: Welford states for the second-order sample estimators" begin
    using Test, PortfolioOptimisers, Statistics, StableRNGs, StatsBase, LinearAlgebra
    pe = PortfolioOptimisers
    rng = StableRNG(987654321)
    X = randn(rng, 50, 4)
    # Mean 1000 with unit-scale spread is the case Chan, Golub and LeVeque (1983) show the
    # textbook formula loses, and the case prices rather than returns put the library in.
    Xp = X .+ 1000.0
    fold(est, rows) = foldl(partial_fit!, eachrow(rows); init = est)

    @testset "Exactness against the batch method" begin
        me = fold(SimpleExpectedReturns(), X)
        @test isapprox(mean(me), vec(mean(SimpleExpectedReturns(), X; dims = 1));
                       rtol = 1e-14)
        @test mean(me) == mean(SimpleExpectedReturns(), me.cache)

        ve = fold(SimpleVariance(), X)
        @test isapprox(var(ve), vec(var(SimpleVariance(), X; dims = 1)); rtol = 1e-13)
        vef = fold(SimpleVariance(; corrected = false), X)
        @test isapprox(var(vef), vec(var(SimpleVariance(; corrected = false), X; dims = 1));
                       rtol = 1e-13)

        gc = fold(GeneralCovariance(), X)
        @test isapprox(cov(gc), cov(GeneralCovariance(), X); rtol = 1e-13)
        gcu = GeneralCovariance(; ce = StatsBase.SimpleCovariance(; corrected = false))
        @test isapprox(cov(fold(gcu, X)), cov(gcu, X); rtol = 1e-13)

        cv = fold(Covariance(), X)
        @test isapprox(cov(cv), cov(Covariance(), X); rtol = 1e-13)
        # A `Covariance` whose inner estimator is a bare `StatsBase.SimpleCovariance`, so the
        # wrapper walk of `partial_fit_corrected` reads the flag through one level fewer.
        cvb = Covariance(; ce = StatsBase.SimpleCovariance(; corrected = true))
        @test isapprox(cov(fold(cvb, X)), cov(cvb, X); rtol = 1e-13)

        # The badly-scaled case. The asymmetric update is what buys this, and the textbook
        # formula is what loses it.
        @test isapprox(cov(fold(Covariance(), Xp)), cov(Covariance(), Xp); rtol = 1e-9)
        @test maximum(abs, cov(fold(Covariance(), Xp)) - cov(Covariance(), Xp)) < 1e-12
    end

    @testset "The block arm folds the same rows" begin
        # The seam takes two arms per family. A block must answer what the same rows answer
        # one at a time, and `dims = 2` must answer what the transpose answers.
        for (est, batch) in ((SimpleExpectedReturns(), mean), (SimpleVariance(), var),
                             (GeneralCovariance(), cov), (Covariance(), cov))
            @test batch(partial_fit!(est, X)) == batch(fold(est, X))
            @test batch(partial_fit!(est, permutedims(X); dims = 2)) == batch(fold(est, X))
        end
        # Two blocks in sequence answer what one block of both answers.
        two = partial_fit!(partial_fit!(Covariance(), X[1:20, :]), X[21:50, :])
        @test isapprox(cov(two), cov(partial_fit!(Covariance(), X)); rtol = 1e-12)
        @test_throws DomainError partial_fit!(Covariance(), X; dims = 3)
        @test_throws ArgumentError partial_fit!(Covariance(; alg = SemiMoment()), X)
    end

    @testset "Associativity of merge_states" begin
        a = fold(Covariance(), X[1:10, :]).cache
        b = fold(Covariance(), X[11:25, :]).cache
        c = fold(Covariance(), X[26:50, :]).cache
        lhs = pe.merge_states(pe.merge_states(a, b), c)
        rhs = pe.merge_states(a, pe.merge_states(b, c))
        @test lhs.n == rhs.n == 50
        @test isapprox(lhs.mu, rhs.mu; rtol = 1e-12)
        @test isapprox(lhs.M, rhs.M; rtol = 1e-12)
        # Both equal one sequential pass, which is the property that makes a blocked fit and
        # a streamed fit the same estimate.
        seq = fold(Covariance(), X).cache
        @test isapprox(lhs.mu, seq.mu; rtol = 1e-12)
        @test isapprox(lhs.M, seq.M; rtol = 1e-11)

        av = fold(SimpleVariance(), X[1:10, :]).cache
        bv = fold(SimpleVariance(), X[11:50, :]).cache
        mv = pe.merge_states(av, bv)
        seqv = fold(SimpleVariance(), X).cache
        @test mv.n == seqv.n
        @test isapprox(mv.mu, seqv.mu; rtol = 1e-12)
        @test isapprox(mv.M, seqv.M; rtol = 1e-11)

        am = fold(SimpleExpectedReturns(), X[1:10, :]).cache
        bm = fold(SimpleExpectedReturns(), X[11:50, :]).cache
        mm = pe.merge_states(am, bm)
        seqm = fold(SimpleExpectedReturns(), X).cache
        @test mm.n == seqm.n
        @test isapprox(mm.mu, seqm.mu; rtol = 1e-13)
    end

    @testset "Readiness reads NaN" begin
        @test all(isnan,
                  mean(SimpleExpectedReturns(),
                       pe.SimpleExpectedReturnsState(; mu = zeros(4))))
        @test all(isfinite, mean(partial_fit!(SimpleExpectedReturns(), X[1, :])))
        # One observation cannot answer a corrected variance, and answers zero uncorrected.
        @test all(isnan, var(partial_fit!(SimpleVariance(), X[1, :])))
        @test all(iszero, var(partial_fit!(SimpleVariance(; corrected = false), X[1, :])))
        @test all(isnan, cov(partial_fit!(Covariance(), X[1, :])))
        @test all(iszero,
                  cov(partial_fit!(GeneralCovariance(;
                                                     ce = StatsBase.SimpleCovariance(;
                                                                                     corrected = false)),
                                   X[1, :])))
    end

    @testset "The estimator does not change" begin
        base = SimpleVariance()
        one = partial_fit!(base, X[1, :])
        two = partial_fit!(base, X[2, :])
        @test isnothing(base.cache)
        @test one.cache.mu != two.cache.mu
        # Two states built from one estimator share no array, so folding into one cannot
        # move the other.
        @test one.cache.mu !== two.cache.mu
        @test one.cache.M !== two.cache.M

        basec = Covariance()
        onec = partial_fit!(basec, X[1, :])
        twoc = partial_fit!(basec, X[2, :])
        @test isnothing(basec.cache)
        @test onec.cache.mu !== twoc.cache.mu
        @test onec.cache.M !== twoc.cache.M
    end

    @testset "A composite over a fitted inner estimator" begin
        # A transform of a covariance writes no state of its own. It has no one-argument
        # read-out either, because `matrix_processing!` reads `X`, so the composite path
        # this ticket covers is the batch one: a fitted inner estimator must not move it.
        cv = fold(Covariance(), X)
        for outer in (DenoiseCovariance, DetoneCovariance, ProcessedCovariance)
            @test isapprox(cov(outer(; ce = cv), X), cov(outer(; ce = Covariance()), X))
        end
        @test isapprox(cov(PortfolioOptimisersCovariance(; ce = cv), X),
                       cov(PortfolioOptimisersCovariance(; ce = Covariance()), X))
        @test isapprox(cov(CorrelationCovariance(; ce = cv), X),
                       cov(CorrelationCovariance(; ce = Covariance()), X))
    end

    @testset "Refusals" begin
        w = StatsBase.Weights(fill(inv(50), 50))
        # Observation weights describe a fixed sample, so an incremental fit cannot read one.
        @test_throws ArgumentError partial_fit!(SimpleExpectedReturns(; w = w), X[1, :])
        @test_throws ArgumentError partial_fit!(SimpleVariance(; w = w), X[1, :])
        @test_throws ArgumentError partial_fit!(GeneralCovariance(; w = w), X[1, :])
        @test_throws ArgumentError partial_fit!(Covariance(; w = w), X[1, :])
        @test_throws ArgumentError partial_fit!(SimpleVariance(;
                                                               me = SimpleExpectedReturns(;
                                                                                          w = w)),
                                                X[1, :])
        # A centre an incremental fit does not reproduce.
        @test_throws ArgumentError partial_fit!(SimpleVariance(;
                                                               me = MedianExpectedReturns()),
                                                X[1, :])
        # SemiMoment clamps before the covariance, and the clamp reads a whole-sample centre.
        @test_throws ArgumentError partial_fit!(Covariance(; alg = SemiMoment()), X[1, :])
        # Every covariance estimator but `StatsBase.SimpleCovariance`, read through the
        # wrapper walk.
        @test_throws ArgumentError partial_fit!(GeneralCovariance(;
                                                                  ce = Covariance(;
                                                                                  alg = SemiMoment())),
                                                X[1, :])
        @test_throws ArgumentError partial_fit!(Covariance(;
                                                           ce = PortfolioOptimisersCovariance()),
                                                X[1, :])
        # The one-argument read-out reached before the first fold.
        @test_throws ArgumentError mean(SimpleExpectedReturns())
        @test_throws ArgumentError var(SimpleVariance())
        @test_throws ArgumentError cov(Covariance())
        @test_throws ArgumentError cov(GeneralCovariance())
        # An observation whose length is not the number of assets the state describes.
        @test_throws DimensionMismatch partial_fit!(fold(Covariance(), X), X[1, 1:2])
        @test_throws DimensionMismatch partial_fit!(fold(SimpleVariance(), X), X[1, 1:2])
        @test_throws DimensionMismatch partial_fit!(fold(SimpleExpectedReturns(), X),
                                                    X[1, 1:2])
        # A merge of two states of different families, and of two states over different
        # numbers of assets.
        @test_throws ArgumentError pe.merge_states(fold(Covariance(), X).cache,
                                                   fold(SimpleVariance(), X).cache)
        @test_throws DimensionMismatch pe.merge_states(fold(Covariance(), X).cache,
                                                       fold(Covariance(), X[:, 1:2]).cache)
    end

    @testset "State constructors and validation" begin
        @test pe.SimpleExpectedReturnsState(; mu = zeros(3)).n == 0
        @test pe.SimpleVarianceState(; mu = zeros(3)).M == zeros(3)
        @test pe.CovarianceState(; mu = zeros(3)).M == zeros(3, 3)
        @test_throws DomainError pe.SimpleExpectedReturnsState(; n = -1, mu = zeros(3))
        @test_throws PortfolioOptimisers.IsEmptyError pe.SimpleVarianceState(;
                                                                             mu = Float64[])
        @test_throws PortfolioOptimisers.IsNonFiniteError pe.CovarianceState(;
                                                                             mu = [NaN,
                                                                                   0.0])
        @test_throws DimensionMismatch pe.SimpleVarianceState(; mu = zeros(3), M = zeros(2))
        @test_throws DimensionMismatch pe.CovarianceState(; mu = zeros(3), M = zeros(2, 2))
    end

    @testset "A cache is hidden, and each channel does one thing with it" begin
        cv = fold(Covariance(), X)
        ve = fold(SimpleVariance(), X)
        me = fold(SimpleExpectedReturns(), X)
        gc = fold(GeneralCovariance(), X)
        # The `show_fields` overload holds under the global switch the documentation sets,
        # so no rendered docstring gains a `cache` line. ADR 0105.
        pe.set_show_nothing_fields!(true)
        for est in (cv, ve, me, gc)
            @test !occursin("cache", sprint(show, est))
        end
        pe.set_show_nothing_fields!(false)
        for est in (cv, ve, me, gc)
            @test !occursin("cache", sprint(show, est))
        end
        # A per-name `true` entry overrides the overload in the direction of showing.
        pe.set_show_nothing_fields!(:Covariance, true)
        @test occursin("cache", sprint(show, cv))
        pe.set_show_nothing_fields!(:Covariance, nothing)

        # `cache` carries `@fprop @vprop`, so each channel does one thing with it, and
        # ADR 0107 states which. `factory` carries the state, `port_opt_view` slices it to
        # the selected assets, and `obs_weights_view` drops it.
        @test pe.factory(me).cache === me.cache
        @test isnothing(pe.obs_weights_view(ve, 1:10).cache)
        @test isnothing(pe.obs_weights_view(cv, 1:10).cache)
        @test isnothing(pe.obs_weights_view(me, 1:10).cache)
        @test isnothing(pe.obs_weights_view(gc, 1:10).cache)

        # The slice is exact: the viewed estimator reads what the same estimator reads when
        # it is fitted on the selected columns alone.
        i = [1, 3]
        @test isapprox(cov(pe.port_opt_view(cv, i)), cov(Covariance(), X[:, i]);
                       rtol = 1e-13)
        @test isapprox(var(pe.port_opt_view(ve, i)), vec(var(SimpleVariance(), X[:, i]));
                       rtol = 1e-13)
        @test isapprox(mean(pe.port_opt_view(me, i)),
                       vec(mean(SimpleExpectedReturns(), X[:, i])); rtol = 1e-13)
        @test isapprox(cov(pe.port_opt_view(gc, i)), cov(GeneralCovariance(), X[:, i]);
                       rtol = 1e-13)

        # The slice copies by index rather than viewing, so a fold on the viewed estimator
        # writes into arrays of its own and the estimator it was viewed from is untouched.
        vcv = pe.port_opt_view(cv, i)
        @test vcv.cache.mu !== view(cv.cache.mu, i)
        before = copy(cv.cache.M)
        partial_fit!(vcv, X[1, i])
        @test cv.cache.M == before
    end

    @testset "The value form folds a copy" begin
        # `partial_fit` copies the state and folds the copy, so a kept estimator reads what
        # it read before the call. ADR 0107.
        for (est, readout) in
            ((Covariance(), cov), (SimpleVariance(), var), (SimpleExpectedReturns(), mean),
             (GeneralCovariance(), cov))
            warm = fold(est, view(X, 1:30, :))
            kept = readout(warm)
            fitted = partial_fit(warm, view(X, 31:50, :))
            @test readout(warm) == kept
            @test warm.cache !== fitted.cache
            # The value form and the batch verb answer the same sample.
            @test isapprox(readout(fitted), readout(fold(est, X)); rtol = 1e-13)
        end
        # A cold estimator has no state to copy, and the fold seeds one of its own.
        cold = partial_fit(SimpleVariance(), X)
        @test isapprox(var(cold), var(fold(SimpleVariance(), X)); rtol = 1e-13)
    end

    @testset "Every state copies without aliasing" begin
        for est in (fold(Covariance(), X), fold(SimpleVariance(), X),
                    fold(SimpleExpectedReturns(), X), fold(GeneralCovariance(), X))
            state = est.cache
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

    @testset "A read-out refuses a state its estimator no longer matches" begin
        # `factory` carries the state and replaces `w`, so the estimator says weighted and
        # holds a state fitted unweighted. Every read-out refuses. ADR 0107.
        w = StatsBase.Weights(fill(inv(50), 50))
        for (est, readout) in
            ((Covariance(), cov), (SimpleVariance(), var), (SimpleExpectedReturns(), mean),
             (GeneralCovariance(), cov))
            warm = fold(est, X)
            weighted = pe.factory(warm, w)
            @test !isnothing(weighted.cache)
            @test_throws ArgumentError readout(weighted)
        end
    end
end
