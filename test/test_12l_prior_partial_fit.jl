#=
The online step of the prior family, issue #968, against the decision of #704 and ADR 0136.

The distinction the file exists to pin is the one the ticket is named for: `EmpiricalPrior`
**folds and carries**, it does not refit. Its moments come off two exact folds and its `X`
comes off a buffer it never reads, so a read-out is `O(N^2)` in the assets and independent of
the observations folded. Test 2 pins that by construction — the read-out's `mu` and `sigma`
are the inner estimators' own folded read-outs, bit for bit — because a regression to
`prior(pe, sample_buffer(pe))` would change no number and pass every parity test in the file.

The two caps are two different knobs and each has its own testset: `max_scenarios` cuts the
scenarios and leaves the moments over every observation, in batch and online alike, and has
no batch equal to assert; `Online`'s `max_history` windows the whole fit and therefore does.
=#
using PortfolioOptimisers, Statistics, LinearAlgebra, StatsBase
# A matrix processing algorithm of a caller's own, which is handed the whole sample. It must
# be declared here: a `struct` is not an expression, so it cannot live inside a `@testset`.
struct AlgProbe <: PortfolioOptimisers.AbstractMatrixProcessingAlgorithm end
# It reads the sample, which is the whole reason the fold refuses it: the scaling it applies
# is a function of the number of observations, which no moment and count could stand in for.
function PortfolioOptimisers.matrix_processing_algorithm!(::AlgProbe, sigma, X; kwargs...)
    sigma .*= size(X, 1)
    return sigma
end
# An inner covariance estimator that folds and answers an **immutable** matrix. Nothing the
# library ships does, because every folded read-out builds a fresh `Matrix`, but the seam is
# open to a caller's own estimator and the composite's read-out writes its processing in
# place — so the copy guard beside it needs a subject.
struct ImmutableCovProbe{T} <: StatsBase.CovarianceEstimator
    cache::T
end
function ImmutableCovProbe(; cache = nothing)
    return ImmutableCovProbe(cache)
end
function PortfolioOptimisers.partial_fit!(ce::ImmutableCovProbe,
                                          x::PortfolioOptimisers.VecNum; kwargs...)
    return ImmutableCovProbe(PortfolioOptimisers.fold_buffer(ce.cache, x))
end
function Statistics.cov(ce::ImmutableCovProbe)
    return LinearAlgebra.Symmetric(Statistics.cov(PortfolioOptimisers.sample_buffer(ce.cache)))
end
function Statistics.cor(ce::ImmutableCovProbe)
    return LinearAlgebra.Symmetric(Statistics.cor(PortfolioOptimisers.sample_buffer(ce.cache)))
end
@testset "Prior partial fit: fold and carry, refit from a buffer" begin
    using Test, PortfolioOptimisers, Statistics, StableRNGs, StatsBase, LinearAlgebra
    pe = PortfolioOptimisers
    rng = StableRNG(987654321)
    X = randn(rng, 80, 6) ./ 100
    F = randn(rng, 80, 3) ./ 100
    fold(est, rows) = foldl(partial_fit!, eachrow(rows); init = est)

    @testset "The identity: a fold and a read-out answer the batch fit" begin
        for est in (EmpiricalPrior(), EmpiricalPrior(; horizon = 21))
            for t in (12, 40, 80)
                o = prior(fold(est, view(X, 1:t, :)))
                b = prior(est, X[1:t, :])
                @test isapprox(o.mu, b.mu; rtol = 1e-12)
                @test isapprox(o.sigma, b.sigma; rtol = 1e-12)
                @test o.X == b.X
            end
        end
        # A block fold reaches the same state as one observation at a time, which is the
        # associativity the buffer and both moment folds already promise separately.
        @test prior(partial_fit!(EmpiricalPrior(), X)).sigma ==
              prior(fold(EmpiricalPrior(), X)).sigma
        # Two blocks, folded in order, are the concatenated block.
        two = partial_fit!(partial_fit!(EmpiricalPrior(), view(X, 1:30, :)),
                           view(X, 31:80, :))
        @test prior(two).sigma == prior(partial_fit!(EmpiricalPrior(), X)).sigma
    end

    @testset "The step is O(N^2): the read-out reads the folds, not the rows" begin
        p = fold(EmpiricalPrior(), X)
        # Bit-exact against the inner estimators' own read-outs. A refit over the buffer
        # would agree to a tolerance and fail here, which is the point.
        @test prior(p).mu == vec(mean(p.me))
        @test prior(p).sigma == cov(p.ce)
        # The rows are carried, and the carried rows are not what the moments came from: the
        # buffer holds 80 observations and the state's own count says the same, so nothing
        # re-read them.
        @test pe.observation_count(p.me) == 80
        @test size(pe.sample_buffer(pe.partial_fit_cache(p)), 1) == 80
        # A second read-out answers the same thing, so the first did not consume the state.
        @test prior(p).sigma == prior(p).sigma
    end

    @testset "The carry is exact, and the Result owns its rows" begin
        p = fold(EmpiricalPrior(), X)
        pr = prior(p)
        @test pr.X == X
        # Materialised, not a view of the live buffer: a further fold must not reach back
        # into a Result already handed out.
        before = copy(pr.X)
        p2 = partial_fit!(p, view(X, 1, :))
        @test pr.X == before
        @test size(prior(p2).X, 1) == 81
    end

    @testset "max_scenarios: the moments keep every observation, `X` keeps the last w" begin
        w = 25
        capped = EmpiricalPrior(; max_scenarios = w)
        plain = prior(EmpiricalPrior(), X)
        b = prior(capped, X)
        @test size(b.X) == (w, 6)
        @test b.X == X[(end - w + 1):end, :]
        @test b.mu == plain.mu
        @test b.sigma == plain.sigma
        # The same knob, the same answer, online.
        o = prior(fold(capped, X))
        @test o.X == X[(end - w + 1):end, :]
        @test isapprox(o.mu, plain.mu; rtol = 1e-12)
        # A cap at or above the sample is a no-op.
        @test prior(EmpiricalPrior(; max_scenarios = 80), X).X == X
        @test prior(EmpiricalPrior(; max_scenarios = 500), X).X == X
        @test_throws DomainError EmpiricalPrior(; max_scenarios = 0)
        @test_throws DomainError EmpiricalPrior(; max_scenarios = -3)
    end

    @testset "Online's max_history windows the whole fit, and has a batch equal" begin
        w = 30
        o = fold(pe.update_online_estimator(pe.Online(EmpiricalPrior(); max_history = w)),
                 X)
        pr = prior(o)
        b = prior(EmpiricalPrior(), X[(end - w + 1):end, :])
        @test pr.X == b.X
        @test isapprox(pr.mu, b.mu; rtol = 1e-12)
        @test isapprox(pr.sigma, b.sigma; rtol = 1e-12)
        # The two caps nest: the moments over the window, the scenarios over its tail.
        n = fold(pe.update_online_estimator(pe.Online(EmpiricalPrior(; max_scenarios = 10);
                                                      max_history = w)), X)
        pn = prior(n)
        @test pn.X == X[(end - 9):end, :]
        @test isapprox(pn.mu, b.mu; rtol = 1e-12)
    end

    @testset "A wrapped prior takes the refit route, an unwrapped one the carry route" begin
        carried = fold(EmpiricalPrior(), X)
        wrapped = fold(pe.update_online_estimator(pe.Online(EmpiricalPrior())), X)
        @test isa(pe.partial_fit_cache(carried), pe.PriorCarryState)
        @test isa(pe.partial_fit_cache(wrapped), pe.SampleBufferState)
        @test isapprox(prior(carried).sigma, prior(wrapped).sigma; rtol = 1e-12)
    end

    @testset "A gapped panel folds to the batch Coverage Universe" begin
        cvg = CoveragePolicy(; min_coverage = 0.2)
        Xg = copy(X)
        Xg[1:30, 3] .= NaN      # lists at observation 31
        Xg[70:end, 5] .= NaN    # delists at observation 70
        Xg[1:75, 6] .= NaN      # never covers the floor
        est = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                             ce = PortfolioOptimisersCovariance(;
                                                                ce = Covariance(;
                                                                                cvg = cvg)),
                             fill_limit = 0.8)
        b = prior(est, Xg)
        o = prior(fold(est, Xg))
        @test isfinite.(b.mu) == isfinite.(o.mu)
        @test isfinite.(diag(b.sigma)) == isfinite.(diag(o.sigma))
        @test isapprox(o.mu[isfinite.(o.mu)], b.mu[isfinite.(b.mu)]; rtol = 1e-12)
        # The investable columns are finite in both, and the non-investable one keeps its gap.
        @test all(isfinite, o.X[:, isfinite.(o.mu)])
        @test !all(isfinite, o.X[:, .!isfinite.(o.mu)])
    end

    @testset "The fill is named once per asset, not once per step" begin
        cvg = CoveragePolicy(; min_coverage = 0.2)
        Xg = copy(X)
        Xg[1:30, 3] .= NaN
        est = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                             ce = PortfolioOptimisersCovariance(;
                                                                ce = Covariance(;
                                                                                cvg = cvg)),
                             fill_limit = 0.05)
        p = fold(est, Xg)
        # The first read-out names asset 3; the second says nothing, because the state
        # remembers. A batch fit, which remembers nothing, names it every time.
        @test_logs (:warn,) prior(p)
        @test_logs prior(p)
        @test 3 in pe.partial_fit_cache(p).named
        @test_logs (:warn,) prior(est, Xg)
        @test_logs (:warn,) prior(est, Xg)
        # `strict` still refuses the first fill, and refuses it whatever the set holds.
        @test_throws ArgumentError prior(p; strict = true)
        # A state copied before the read-out carries a memory of its own.
        q = fold(est, Xg)
        r = pe.partial_fit_cache(q)
        s = copy(r)
        push!(r.named, 3)
        @test isempty(s.named)
    end

    @testset "A refitting prior reaches the batch answer through its buffer" begin
        for est in
            (EntropyPoolingPrior(), OpinionPoolingPrior(; pes = [EntropyPoolingPrior()]))
            o = fold(pe.update_online_estimator(pe.Online(est)), X)
            @test isapprox(prior(o).mu, prior(est, X).mu; rtol = 1e-10)
            @test isapprox(prior(o).sigma, prior(est, X).sigma; rtol = 1e-10)
        end
        # A prior whose batch verb needs a factor matrix seeds the paired buffer instead.
        f = pe.update_online_estimator(pe.Online(FactorPrior()))
        @test isa(pe.partial_fit_cache(f), pe.FactorSampleBufferState)
        for i in axes(X, 1)
            f = partial_fit!(f, view(X, i, :), view(F, i, :))
        end
        b = prior(FactorPrior(), X, F)
        @test isapprox(prior(f).mu, b.mu; rtol = 1e-10)
        @test isapprox(prior(f).sigma, b.sigma; rtol = 1e-10)
        @test prior(f).X == b.X
    end

    @testset "A forwarding host folds through, and owns no buffer" begin
        for est in (HighOrderPriorEstimator(),
                    BlackLittermanPrior(; pe = EmpiricalPrior(),
                                        views = pe.BlackLittermanViews(; P = [1.0 zeros(1, 5)],
                                                                       Q = [0.01])))
            o = prior(fold(est, X))
            b = prior(est, X)
            @test isapprox(o.mu, b.mu; rtol = 1e-10)
            @test isapprox(o.sigma, b.sigma; rtol = 1e-10)
        end
        # No `cache` field: the rows live one level down, in the prior the host wraps.
        @test !hasfield(typeof(HighOrderPriorEstimator()), :cache)
        @test !hasfield(typeof(BlackLittermanPrior(;
                                                   views = pe.BlackLittermanViews(;
                                                                                  P = [1.0 zeros(1,
                                                                                                 5)],
                                                                                  Q = [0.01]))),
                        :cache)
        # A host may still declare the step for the prior it wraps, which then buffers.
        h = HighOrderPriorEstimator(; pe = pe.Online(EntropyPoolingPrior()))
        @test pe.online_fields(h) == (:pe,)
        o = prior(fold(pe.update_online_estimator(h), X))
        b = prior(HighOrderPriorEstimator(; pe = EntropyPoolingPrior()), X)
        @test isapprox(o.mu, b.mu; rtol = 1e-10)
        @test isapprox(o.kt, b.kt; rtol = 1e-10)
    end

    @testset "A mixed host folds what folds and refits the rest from its own rows" begin
        # A `SemiMoment` co-moment cannot fold; the host runs its batch verb over `pr.X`.
        mixed = HighOrderPriorEstimator(; ske = Coskewness(; alg = SemiMoment()))
        @test !pe.supports_partial_fit(mixed.ske)
        @test pe.supports_partial_fit(mixed.kte)
        o = prior(fold(mixed, X))
        b = prior(mixed, X)
        @test isapprox(o.kt, b.kt; rtol = 1e-10)
        @test o.sk == b.sk
        # The same rule one layer down: a `SemiMoment` covariance inside an empirical prior.
        semi = EmpiricalPrior(;
                              ce = PortfolioOptimisersCovariance(;
                                                                 ce = Covariance(;
                                                                                 alg = SemiMoment())))
        @test !pe.supports_partial_fit(semi.ce)
        @test pe.supports_partial_fit(semi.me)
        os = prior(fold(semi, X))
        bs = prior(semi, X)
        @test isapprox(os.sigma, bs.sigma; rtol = 1e-12)
        @test isapprox(os.mu, bs.mu; rtol = 1e-12)
    end

    @testset "The composite covariance folds by composition" begin
        ce = PortfolioOptimisersCovariance()
        @test pe.supports_partial_fit(ce)
        c = fold(ce, X)
        @test isapprox(cov(c), cov(ce, X); rtol = 1e-12)
        @test isapprox(cor(c), cor(ce, X); rtol = 1e-12)
        @test pe.observation_count(c.ce) == 80
        # Denoising reads the effective sample ratio and nothing else, so the shape arm and
        # the matrix arm answer the same matrix. Detoning and the positive-definite repair
        # read `sigma` alone, so they are the matrix methods verbatim.
        dn = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; dn = Denoise()))
        @test isapprox(cov(fold(dn, X)), cov(dn, X); rtol = 1e-12)
        dt = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; dt = Detone()))
        @test isapprox(cov(fold(dt, X)), cov(dt, X); rtol = 1e-12)
        # An inner estimator answering an immutable matrix is copied before the processing
        # writes into it, which is the guard the matrix methods beside it carry.
        im = fold(PortfolioOptimisersCovariance(; ce = ImmutableCovProbe()), X)
        @test ismutable(cov(im))
        @test ismutable(cor(im))
        @test isapprox(cov(im), cov(X); rtol = 1e-10)
        @test isapprox(cor(im), cor(X); rtol = 1e-10)
        # A caller's own algorithm is handed the whole sample, and is refused by name.
        alg = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; alg = AlgProbe()))
        @test !pe.supports_partial_fit(alg)
        @test_throws ArgumentError partial_fit!(alg, view(X, 1, :))
        @test_throws ArgumentError partial_fit!(alg, X)
        # Wrapping it is the route that works, and its read-out is the batch fit.
        w = fold(pe.update_online_estimator(pe.Online(alg)), X)
        @test pe.supports_partial_fit(w)
        @test isapprox(cov(w), cov(alg, X); rtol = 1e-12)
    end

    @testset "The carry state's interface" begin
        a = fold(EmpiricalPrior(), view(X, 1:30, :))
        b = fold(EmpiricalPrior(), view(X, 31:80, :))
        sa, sb = pe.partial_fit_cache(a), pe.partial_fit_cache(b)
        push!(sa.named, 2)
        push!(sb.named, 4)
        m = pe.merge_states(sa, sb)
        @test pe.sample_buffer(m) == X
        @test m.named == Set([2, 4])
        # A copy shares no array and no set.
        c = copy(m)
        push!(m.named, 6)
        @test c.named == Set([2, 4])
        @test c.buf.X !== m.buf.X
        # A slice renumbers the assets, and an asset already named keeps its silence.
        v = pe.port_opt_view(m, [2, 4, 6])
        @test pe.sample_buffer(v) == X[:, [2, 4, 6]]
        @test v.named == Set([1, 2, 3])
        # The observation axis drops it, because no slice of a state exists there.
        @test isnothing(pe.obs_weights_view(m, 1:10))
    end

    @testset "The paired buffer's interface" begin
        s = pe.FactorSampleBufferState()
        for i in 1:20
            s = partial_fit!(s, view(X, i, :), view(F, i, :))
        end
        @test pe.sample_buffer(s.X) == X[1:20, :]
        @test pe.sample_buffer(s.F) == F[1:20, :]
        t = pe.FactorSampleBufferState()
        t = partial_fit!(t, view(X, 21:80, :), view(F, 21:80, :))
        m = pe.merge_states(s, t)
        @test pe.sample_buffer(m.X) == X
        @test pe.sample_buffer(m.F) == F
        c = copy(m)
        @test c.X.X !== m.X.X
        # The selection indexes assets, so the factor half passes through untouched.
        v = pe.port_opt_view(m, [1, 3])
        @test pe.sample_buffer(v.X) == X[:, [1, 3]]
        @test pe.sample_buffer(v.F) == F
        # Halves that fall out of step are refused, and different widths are not.
        @test_throws DimensionMismatch pe.FactorSampleBufferState(; X = s.X, F = t.F)
        @test_throws ArgumentError pe.FactorSampleBufferState(;
                                                              X = pe.SampleBufferState(;
                                                                                       max_history = 5),
                                                              F = pe.SampleBufferState())
        @test_throws ArgumentError pe.assert_factor_sample_buffer(EmpiricalPrior())
        # The block arm through an estimator, beside the observation arm above.
        blk = partial_fit!(pe.update_online_estimator(pe.Online(FactorPrior())), X, F)
        @test pe.sample_buffer(pe.partial_fit_cache(blk).X) == X
        @test pe.sample_buffer(pe.partial_fit_cache(blk).F) == F
        # A mask describes assets, so the pair carries it into the returns half and leaves
        # the factor half the rows alone (#999's contract, applied to the pair).
        amsk = trues(size(X))
        amsk[1:10, 2] .= false
        msk = partial_fit!(pe.update_online_estimator(pe.Online(FactorPrior())), X, F;
                           active_mask = amsk)
        st = pe.partial_fit_cache(msk)
        @test pe.sample_buffer_kwargs(st.X).active_mask == amsk
        @test isempty(pe.sample_buffer_kwargs(st.F))
    end

    @testset "A read-out hands out no accumulator" begin
        # `mean` on a folded estimator copies the Welford accumulator. Before it did not, and
        # a prior that read its `mu` out and carried it into a Result found the Result
        # rewritten at the next observation.
        me = fold(SimpleExpectedReturns(), X)
        m1 = mean(me)
        kept = copy(m1)
        @test m1 !== pe.partial_fit_cache(me).mu
        me = partial_fit!(me, view(X, 1, :))
        @test m1 == kept
        @test mean(me) != kept
    end

    @testset "Rendering: `cache` never, `max_scenarios` only where it is set" begin
        @test pe.show_fields(EmpiricalPrior()) == (:ce, :me, :horizon, :fill_limit)
        @test pe.show_fields(EmpiricalPrior(; max_scenarios = 10)) ==
              (:ce, :me, :horizon, :fill_limit, :max_scenarios)
        @test pe.show_fields(PortfolioOptimisersCovariance()) == (:ce, :mp)
        @test pe.show_fields(FactorPrior()) == (:pe, :mp, :re, :ve, :rsd)
        blv = pe.BlackLittermanViews(; P = [1.0 zeros(1, 5)], Q = [0.01])
        for est in (EmpiricalPrior(), PortfolioOptimisersCovariance(), FactorPrior(),
                    EntropyPoolingPrior(), OpinionPoolingPrior(; pes = [EntropyPoolingPrior()]),
                    MeucciEntropyPoolingPrior(), HighOrderFactorPriorEstimator(),
                    BayesianBlackLittermanPrior(; views = blv),
                    FactorBlackLittermanPrior(; views = blv),
                    AugmentedBlackLittermanPrior(; a_views = blv, f_views = blv))
            @test !(:cache in pe.show_fields(est))
            @test :cache in fieldnames(typeof(est))
        end
    end

    @testset "A wrapped prior folds under a Coverage Policy" begin
        # #999 gave the buffer the per-observation masks; a prior refitting from one
        # therefore answers what a batch fit over the same window under the same policy
        # answers, rather than refusing the mask.
        cvg = CoveragePolicy(; min_coverage = 0.2)
        Xg = copy(X)
        Xg[1:30, 3] .= NaN
        amsk = isfinite.(Xg)
        est = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                             ce = PortfolioOptimisersCovariance(;
                                                                ce = Covariance(;
                                                                                cvg = cvg)),
                             fill_limit = 0.8)
        w = pe.update_online_estimator(pe.Online(est))
        for i in axes(Xg, 1)
            w = partial_fit!(w, view(Xg, i, :); active_mask = view(amsk, i, :))
        end
        o = prior(w)
        b = prior(est, Xg; active_mask = amsk)
        @test isfinite.(o.mu) == isfinite.(b.mu)
        @test isapprox(o.mu[isfinite.(o.mu)], b.mu[isfinite.(b.mu)]; rtol = 1e-12)
    end

    @testset "A read-out before the first fold refuses by name" begin
        @test_throws ArgumentError prior(EmpiricalPrior())
        @test_throws ArgumentError prior(EntropyPoolingPrior())
        @test_throws ArgumentError cov(PortfolioOptimisersCovariance())
    end
end
