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
    using Test, PortfolioOptimisers, Statistics, StableRNGs, StatsBase, LinearAlgebra, Dates
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
        # The value form on an estimator with no fold of its own: the state walk finds
        # nothing to copy, and `partial_fit!` gives the refusal that names the wrapper.
        @test_throws ArgumentError partial_fit(MedianExpectedReturns(), X)
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
        # `@wprop` alone opens the observation channel. A type with no `@wprop` field falls
        # through to the identity, so every state it holds is carried, its children's too.
        # A child viewed on its own still drops its state. #1344.
        poc = fold(PortfolioOptimisersCovariance(), X)
        epr = fold(EmpiricalPrior(), X)
        @test !isnothing(poc.ce.cache) && !isnothing(epr.cache)
        @test pe.obs_weights_view(poc, 1:10) === poc
        @test pe.obs_weights_view(epr, 1:10) === epr
        @test isnothing(pe.obs_weights_view(poc.ce, 1:10).cache)

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

    @testset "The value form folds a copy of every member state" begin
        # A host with no `cache` of its own folds through the states of its members, and the
        # generic method copies the whole tree before the fold, so the kept host reads what
        # it read before the call. Issue #1053.
        rd = ReturnsResult(; nx = ["a", "b", "c", "d"], X = X,
                           ts = Date(2020, 1, 1) .+ Day.(0:49))
        head = ReturnsResult(; nx = rd.nx, X = X[1:30, :], ts = rd.ts[1:30])
        tail = ReturnsResult(; nx = rd.nx, X = X[31:50, :], ts = rd.ts[31:50])
        # A prior that folds `pe`, `ske` and `kte`.
        warm = partial_fit!(HighOrderPriorEstimator(), head)
        kept = prior(warm)
        fitted = partial_fit(warm, tail)
        @test prior(warm).mu == kept.mu
        @test prior(warm).sigma == kept.sigma
        @test size(prior(warm).X, 1) == 30
        @test warm.pe.cache !== fitted.pe.cache
        @test size(prior(fitted).X, 1) == 50
        full = prior(partial_fit!(HighOrderPriorEstimator(), rd))
        @test isapprox(prior(fitted).mu, full.mu; rtol = 1e-13)
        @test isapprox(prior(fitted).kt, full.kt; rtol = 1e-13)
        # Two folds from one warm host do not contaminate each other.
        twin = partial_fit(warm, tail)
        @test twin.pe.cache !== fitted.pe.cache
        @test size(prior(warm).X, 1) == 30
        # A wrapping prior that folds its embedded prior alone, two levels down.
        bl = BlackLittermanPrior(; pe = EmpiricalPrior(),
                                 views = LinearConstraintEstimator(; val = ["a == 0.0001"]),
                                 sets = UniverseSets(; dict = Dict("nx" => rd.nx)))
        warm = partial_fit!(HighOrderPriorEstimator(; pe = bl), head)
        fitted = partial_fit(warm, tail)
        @test size(prior(warm).X, 1) == 30
        @test size(prior(fitted).X, 1) == 50
        @test warm.pe.pe.cache !== fitted.pe.pe.cache
        # A hierarchical optimiser folds `opt.pe` and its own fold context under `opt.cache`.
        warm = partial_fit!(HierarchicalRiskParity(), head)
        fitted = partial_fit(warm, tail)
        @test warm.opt.cache !== fitted.opt.cache
        @test warm.opt.pe.cache !== fitted.opt.pe.cache
        @test size(prior(warm.opt.pe).X, 1) == 30
        @test size(prior(fitted.opt.pe).X, 1) == 50
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

    @testset "The plain arms take the active mask, and a folded frame is repaired on its block (#874)" begin
        # A fold loop hands every member a host folds the Asset Panel's active mask. A plain
        # co-moment carries no per-cell count for it to gate, so it accepts the keyword and
        # ignores it, as the plain first order does; refusing it refused the host.
        amsk = trues(size(X))
        @test cokurtosis(partial_fit!(Cokurtosis(), X; active_mask = amsk)) ==
              cokurtosis(partial_fit!(Cokurtosis(), X))
        @test first(coskewness(partial_fit!(Coskewness(), X; active_mask = amsk))) ==
              first(coskewness(partial_fit!(Coskewness(), X)))
        ske_m, kte_m, ske_v, kte_v = Coskewness(), Cokurtosis(), Coskewness(), Cokurtosis()
        for i in axes(X, 1)
            ske_m = partial_fit!(ske_m, X[i, :]; active_mask = view(amsk, i, :))
            kte_m = partial_fit!(kte_m, X[i, :]; active_mask = view(amsk, i, :))
            ske_v = partial_fit!(ske_v, X[i, :])
            kte_v = partial_fit!(kte_v, X[i, :])
        end
        @test first(coskewness(ske_m)) == first(coskewness(ske_v))
        @test cokurtosis(kte_m) == cokurtosis(kte_v)
        # A plain fold over a changing universe answers `NaN` at every cell naming an
        # asset outside the Coverage Universe. The read-out repairs the finite block and
        # leaves the frame, at both orders, rather than handing the frame to LAPACK: the
        # block is the batch verb over the covered columns alone.
        Xg = copy(X)
        Xg[1:5, 2] .= NaN
        keep = [1, 3]
        ckurt = cokurtosis(partial_fit!(Cokurtosis(), Xg))
        blk = isfinite.(LinearAlgebra.diag(ckurt))
        pairs = vec([i in keep && j in keep for i in 1:N, j in 1:N])
        @test blk == pairs
        @test all(isnan, ckurt[.!blk, :]) && all(isnan, ckurt[:, .!blk])
        @test isapprox(ckurt[blk, blk], cokurtosis(Cokurtosis(), Xg[:, keep]); atol = 1e-13)
        cskew, V = coskewness(partial_fit!(Coskewness(), Xg))
        cskew_b, V_b = coskewness(Coskewness(), Xg[:, keep])
        @test isapprox(V[keep, keep], V_b; atol = 1e-14)
        @test all(isnan, V[2, :]) && all(isnan, V[:, 2])
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
