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

Issue #1013 (deciding #1009) put the factor rows **inside** the one buffer and made
`needs_factor_returns` answer from the estimator tree, so the factor testsets at the foot of
the file pin the three answers against the batch verb over the same arguments, and the batch
defect the shallow `isa` door hid: a factor leaf under an optional-argument host met a
`MethodError` at the leaf instead of the named refusal.
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
# A caller's own optional-argument prior that **reads** `F` when it is given and leaves
# `needs_factor_returns` at the `_AF` default, `nothing`: the fold must take what it is
# given, exactly as the batch verb does. It embeds nothing, so it defines no recursion.
struct OptionalFactorProbe{T} <: PortfolioOptimisers.AbstractLowOrderPriorEstimator_AF
    cache::T
end
function OptionalFactorProbe(; cache = nothing)
    return OptionalFactorProbe(cache)
end
function PortfolioOptimisers.prior(::OptionalFactorProbe, X::PortfolioOptimisers.MatNum,
                                   F::Union{Nothing, <:PortfolioOptimisers.MatNum} = nothing,
                                   ::Union{Nothing, <:PortfolioOptimisers.AssetPanel} = nothing;
                                   dims::Int = 1, kwargs...)
    # The mean is shifted by the factor mean when `F` is given, so the two fits differ and
    # the test can tell which arity the read-out ran.
    mu = vec(Statistics.mean(X; dims = 1))
    if !isnothing(F)
        mu = mu .+ Statistics.mean(F)
    end
    return LowOrderPrior(; X = X, mu = mu, sigma = Statistics.cov(X))
end
# A caller's own prior that reads a Panel Field and both masks, and refits from a buffer. It
# stands in for the Cross-Sectional Factor Prior, which gains its `cache` in #1468. Its mean
# moves with the field and with the estimation mask, so a read-out that lost either differs
# from the batch fit.
struct PanelFieldProbe{T1, T2} <: PortfolioOptimisers.AbstractLowOrderPriorEstimator_A
    fld::T1
    cache::T2
end
function PanelFieldProbe(; fld::AbstractString = "cap", cache = nothing)
    return PanelFieldProbe(fld, cache)
end
function PortfolioOptimisers.reads_panel_fields(::PanelFieldProbe)
    return true
end
function PortfolioOptimisers.prior(pe::PanelFieldProbe, X::PortfolioOptimisers.MatNum,
                                   ::Union{Nothing, <:PortfolioOptimisers.MatNum} = nothing,
                                   pnl::Union{Nothing, <:PortfolioOptimisers.AssetPanel} = nothing;
                                   dims::Int = 1, kwargs...)
    v = panel_field(pnl, pe.fld).vals
    mu = vec(Statistics.mean(X; dims = 1)) .+ vec(Statistics.mean(v; dims = 1)) .+
         vec(sum(pnl.emsk; dims = 1)) ./ 1000
    return LowOrderPrior(; X = Matrix(X), mu = mu, sigma = Statistics.cov(X) + I)
end
@testset "Prior partial fit: fold and carry, refit from a buffer" begin
    using Test, PortfolioOptimisers, Statistics, StableRNGs, StatsBase, LinearAlgebra
    pe = PortfolioOptimisers
    rng = StableRNG(987654321)
    X = randn(rng, 80, 6) ./ 100
    F = randn(rng, 80, 3) ./ 100
    fold(est, rows) = foldl(partial_fit!, eachrow(rows); init = est)
    fold_pair(est, rows, frows) = foldl((e, (x, f)) -> partial_fit!(e, x, f),
                                        zip(eachrow(rows), eachrow(frows)); init = est)

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
        # A prior whose batch verb needs a factor matrix seeds the same buffer, and the
        # factor rows ride inside it (#1013).
        f = pe.update_online_estimator(pe.Online(FactorPrior()))
        @test isa(pe.partial_fit_cache(f), pe.SampleBufferState)
        for i in axes(X, 1)
            f = partial_fit!(f, view(X, i, :), view(F, i, :))
        end
        b = prior(FactorPrior(), X, F)
        @test isapprox(prior(f).mu, b.mu; rtol = 1e-10)
        @test isapprox(prior(f).sigma, b.sigma; rtol = 1e-10)
        @test prior(f).X == b.X
        @test pe.factor_buffer(pe.partial_fit_cache(f)) == F
        # The five `_F` priors, folded as a block, over the one buffer.
        fv = BlackLittermanViews(; P = [1.0 0 0], Q = [0.01])
        av = BlackLittermanViews(; P = [1.0 zeros(1, 5)], Q = [0.01])
        for est in (FactorPrior(), BayesianBlackLittermanPrior(; views = fv),
                    FactorBlackLittermanPrior(; views = fv),
                    AugmentedBlackLittermanPrior(; a_views = av, f_views = fv),
                    HighOrderFactorPriorEstimator())
            @test pe.needs_factor_returns(est) === true
            o = prior(partial_fit!(pe.update_online_estimator(pe.Online(est)), X, F))
            b = prior(est, X, F)
            @test isapprox(o.mu, b.mu; rtol = 1e-10)
            @test isapprox(o.sigma, b.sigma; rtol = 1e-10)
        end
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
        # A `SemiMoment` co-moment cannot fold; the host runs its batch verb over the rows
        # that the embedded prior folded.
        mixed = HighOrderPriorEstimator(; ske = Coskewness(; alg = SemiMoment()))
        @test !pe.supports_partial_fit(mixed.ske)
        @test pe.supports_partial_fit(mixed.kte)
        o = prior(fold(mixed, X))
        b = prior(mixed, X)
        @test isapprox(o.kt, b.kt; rtol = 1e-10)
        @test isapprox(o.sk, b.sk; rtol = 1e-12)
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

    @testset "The factor rows ride in the one buffer" begin
        s = pe.SampleBufferState()
        for i in 1:20
            s = partial_fit!(s, view(X, i, :), view(F, i, :))
        end
        @test pe.sample_buffer(s) == X[1:20, :]
        @test pe.factor_buffer(s) == F[1:20, :]
        t = partial_fit!(pe.SampleBufferState(), view(X, 21:80, :), view(F, 21:80, :))
        m = pe.merge_states(s, t)
        @test pe.sample_buffer(m) == X
        @test pe.factor_buffer(m) == F
        c = copy(m)
        @test c.F !== m.F && c.F == m.F
        # The selection indexes assets, so the factor rows pass through untouched and are
        # copied rather than shared.
        v = pe.port_opt_view(m, [1, 3])
        @test pe.sample_buffer(v) == X[:, [1, 3]]
        @test pe.factor_buffer(v) == F
        @test v.F !== m.F
        # A cap windows the factor rows with the rest, on the fold and on the merge.
        w = partial_fit!(pe.SampleBufferState(; max_history = 10), X, F)
        @test pe.factor_buffer(w) == F[71:80, :]
        wm = pe.merge_states(partial_fit!(pe.SampleBufferState(; max_history = 10),
                                          view(X, 1:8, :), view(F, 1:8, :)),
                             partial_fit!(pe.SampleBufferState(; max_history = 10),
                                          view(X, 9:16, :), view(F, 9:16, :)))
        @test pe.factor_buffer(wm) == F[7:16, :]
        # A factor block whose rows are not the block's, and one whose width moves.
        @test_throws DimensionMismatch partial_fit!(pe.SampleBufferState(), X,
                                                    view(F, 1:79, :))
        @test_throws DimensionMismatch partial_fit!(s, view(X, 21, :), view(F, 21, 1:2))
        @test_throws DimensionMismatch pe.SampleBufferState(; n = 2, X = zeros(2, 3),
                                                            F = zeros(3, 2))
        # The mixture, refused both ways on the fold and on the merge; an empty buffer
        # records nothing and reseeds.
        @test_throws ArgumentError partial_fit!(s, view(X, 21, :))
        x_only = partial_fit!(pe.SampleBufferState(), view(X, 1:5, :))
        @test_throws ArgumentError partial_fit!(x_only, view(X, 6, :), view(F, 6, :))
        @test_throws ArgumentError pe.merge_states(s, x_only)
        with_width = pe.SampleBufferState(; X = zeros(0, 6), F = zeros(0, 3))
        @test isnothing(partial_fit!(with_width, view(X, 1, :)).F)
        @test pe.factor_buffer(partial_fit!(pe.SampleBufferState(; X = zeros(0, 6)),
                                            view(X, 2, :), view(F, 2, :))) == F[2:2, :]
        # The block arm through an estimator, beside the observation arm above, and a mask
        # rides beside the rows while the factor rows take no mask (#999's contract).
        amsk = trues(size(X))
        amsk[1:10, 2] .= false
        msk = partial_fit!(pe.update_online_estimator(pe.Online(FactorPrior())), X, F;
                           active_mask = amsk)
        st = pe.partial_fit_cache(msk)
        @test pe.sample_buffer(st) == X
        @test pe.factor_buffer(st) == F
        @test pe.sample_buffer_kwargs(st).active_mask == amsk
        # `show` renders the backing beside the rest.
        @test occursin("F", sprint(show, st))
    end

    @testset "The tree answers, and the fold mirrors the batch verb's arity" begin
        ep_f = EntropyPoolingPrior(; pe = FactorPrior())
        @test pe.needs_factor_returns(ep_f) === true
        @test pe.needs_factor_returns(EntropyPoolingPrior()) === false
        @test pe.needs_factor_returns(EmpiricalPrior()) === false
        @test pe.needs_factor_returns(OptionalFactorProbe()) === nothing
        @test pe.needs_factor_returns(pe.Online(ep_f)) === true
        views = BlackLittermanViews(; P = [1.0 zeros(1, 5)], Q = [0.01])
        @test pe.needs_factor_returns(BlackLittermanPrior(; pe = FactorPrior(),
                                                          views = views)) === true
        @test pe.needs_factor_returns(HighOrderPriorEstimator(; pe = ep_f)) === true
        @test pe.needs_factor_returns(MeucciEntropyPoolingPrior(; pe = FactorPrior())) ===
              true
        # Opinion pooling: any `true` wins, all `false` is `false`, else `nothing`.
        @test pe.needs_factor_returns(OpinionPoolingPrior(; pes = [EntropyPoolingPrior()],
                                                          pe2 = ep_f)) === true
        @test pe.needs_factor_returns(OpinionPoolingPrior(; pes = [EntropyPoolingPrior()])) ===
              false
        @test pe.needs_factor_returns(OpinionPoolingPrior(; pes = [EntropyPoolingPrior()],
                                                          pe1 = OptionalFactorProbe())) ===
              nothing
        # `true`: a factor leaf under an optional-argument host records `F`, and the
        # online fit equals the batch fit over the same arguments.
        o = pe.update_online_estimator(pe.Online(ep_f))
        for i in axes(X, 1)
            o = partial_fit!(o, view(X, i, :), view(F, i, :))
        end
        b = prior(ep_f, X, F)
        @test isapprox(prior(o).mu, b.mu; rtol = 1e-8)
        @test isapprox(prior(o).sigma, b.sigma; rtol = 1e-8)
        @test pe.factor_buffer(pe.partial_fit_cache(o)) == F
        # `true` and `x` alone: refused by name before any row is appended.
        o = pe.update_online_estimator(pe.Online(ep_f))
        @test_throws pe.IsNothingError partial_fit!(o, view(X, 1, :))
        @test_throws pe.IsNothingError partial_fit!(o, X)
        @test iszero(pe.partial_fit_cache(o).n)
        # `false`: a tree that never reads `F` drops it, records no factor rows, and equals
        # the fit without it.
        o = pe.update_online_estimator(pe.Online(EntropyPoolingPrior()))
        for i in axes(X, 1)
            o = partial_fit!(o, view(X, i, :), view(F, i, :))
        end
        @test isnothing(pe.partial_fit_cache(o).F)
        @test isapprox(prior(o).mu, prior(EntropyPoolingPrior(), X).mu; rtol = 1e-10)
        # The carry route takes and drops it too, mirroring `prior(EmpiricalPrior(), X, F)`.
        c = fold_pair(EmpiricalPrior(), X, F)
        @test isnothing(pe.partial_fit_cache(c).buf.F)
        @test prior(c).sigma == prior(fold(EmpiricalPrior(), X)).sigma
        # And a forwarding host hands `F` down its tree.
        h = HighOrderPriorEstimator(; pe = pe.Online(ep_f))
        h = partial_fit!(pe.update_online_estimator(h), X, F)
        @test isapprox(prior(h).kt, prior(HighOrderPriorEstimator(; pe = ep_f), X, F).kt;
                       rtol = 1e-8)
        # `nothing`: a caller's `_AF` with no method takes what it is given, both ways,
        # and equals its batch verb over the same arguments.
        with_f = fold_pair(pe.update_online_estimator(pe.Online(OptionalFactorProbe())), X,
                           F)
        without = fold(pe.update_online_estimator(pe.Online(OptionalFactorProbe())), X)
        @test isapprox(prior(with_f).mu, prior(OptionalFactorProbe(), X, F).mu;
                       rtol = 1e-12)
        @test isapprox(prior(without).mu, prior(OptionalFactorProbe(), X).mu; rtol = 1e-12)
        @test !isapprox(prior(with_f).mu, prior(without).mu; rtol = 1e-3)
        # The mixture through a prior, both ways, refused by name.
        @test_throws ArgumentError partial_fit!(with_f, view(X, 1, :))
        @test_throws ArgumentError partial_fit!(without, view(X, 1, :), view(F, 1, :))
    end

    @testset "The batch doors refuse a factor leaf under an optional host by name" begin
        ep_f = EntropyPoolingPrior(; pe = FactorPrior())
        nx = string.("A", 1:6)
        rd = ReturnsResult(; nx = nx, X = X)
        # Before #1013 the shallow `isa` test passed this through to the leaf's
        # `MethodError`.
        @test_throws pe.IsNothingError prior(ep_f, rd)
        @test_throws pe.IsNothingError prior(HighOrderPriorEstimator(; pe = ep_f), rd)
        for uc in (NormalUncertaintySet(; pe = ep_f), DeltaUncertaintySet(; pe = ep_f))
            @test_throws pe.IsNothingError pe.ucs(uc, rd)
            @test_throws pe.IsNothingError pe.mu_ucs(uc, rd)
            @test_throws pe.IsNothingError pe.sigma_ucs(uc, rd)
        end
        # And with `F` present every door fits.
        rdf = ReturnsResult(; nx = nx, X = X, nf = string.("F", 1:3), F = F)
        @test isapprox(prior(ep_f, rdf).mu, prior(ep_f, X, F).mu; rtol = 1e-10)
        @test !isnothing(pe.mu_ucs(NormalUncertaintySet(; pe = ep_f), rdf))
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
        # `mtx_sqrt` holds the square root algorithm since #1410.
        @test pe.show_fields(FactorPrior()) == (:pe, :mp, :re, :ve, :rsd, :mtx_sqrt)
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

    @testset "The docstrings of 12_PriorPartialFit.jl against numbers" begin
        # `dims = 2` reaches the batch fit on both arms, and the carry state holds the rows
        # in the observation-by-row orientation.
        for est in (EmpiricalPrior(), EmpiricalPrior(; horizon = 21))
            o = prior(partial_fit!(est, permutedims(X); dims = 2))
            b = prior(est, X)
            @test isapprox(o.mu, b.mu; rtol = 1e-12)
            @test isapprox(o.sigma, b.sigma; rtol = 1e-12)
            @test o.X == b.X
        end
        @test pe.sample_buffer(partial_fit!(pe.PriorCarryState(), permutedims(X); dims = 2)) ==
              X
        # The identity holds for `ens` under a Scenario Cap, on both arms: it reads the
        # number of observations folded, not the number of rows carried.
        for est in (EmpiricalPrior(; max_scenarios = 25),
                    EmpiricalPrior(; horizon = 21, max_scenarios = 25))
            o = prior(fold(est, X))
            b = prior(est, X)
            @test o.ens == b.ens == 80
            @test o.X == b.X
        end
        @test isnothing(prior(fold(EmpiricalPrior(; max_scenarios = 500), X)).ens)
        # The horizon arm refits a member that does not fold on the log rows it would have
        # folded, and carries the arithmetic rows.
        semi = EmpiricalPrior(; horizon = 21,
                              ce = PortfolioOptimisersCovariance(;
                                                                 ce = Covariance(;
                                                                                 alg = SemiMoment())))
        o = prior(fold(semi, X))
        b = prior(semi, X)
        @test isapprox(o.mu, b.mu; rtol = 1e-12)
        @test isapprox(o.sigma, b.sigma; rtol = 1e-12)
        @test o.X == X
        # The high order read-out refits a co-moment that does not fold over the rows that
        # the embedded prior folded, not over `pr.X`, which a Scenario Cap on the embedded
        # prior cuts. The batch method fits it over every row too (#1328), so the two agree
        # while `pr.X` keeps the last `w` rows.
        w = 25
        h = HighOrderPriorEstimator(; pe = EmpiricalPrior(; max_scenarios = w),
                                    ske = Coskewness(; alg = SemiMoment()))
        o = prior(fold(h, X))
        b = prior(h, X)
        @test isapprox(o.kt, b.kt; rtol = 1e-12)
        @test b.sk == coskewness(Coskewness(; alg = SemiMoment()), X)[1]
        @test isapprox(o.sk, b.sk; rtol = 1e-12)
        @test isapprox(o.V, b.V; rtol = 1e-12)
        @test o.X == b.X == X[(end - w + 1):end, :]
        # The scenario fill writes zeros into the rows of `pr.X` before an asset lists. The
        # refit reads the folded rows, `NaN` entries included, so it answers the batch fit,
        # and a refit over the filled zeros misses it by more than a quarter.
        cvg = CoveragePolicy(; min_coverage = 0.2)
        Xg = copy(X)
        Xg[1:30, 3] .= NaN
        skg = Coskewness(; alg = SemiMoment(), cvg = cvg)
        ktg = Cokurtosis(; cvg = cvg)
        hg = HighOrderPriorEstimator(;
                                     pe = EmpiricalPrior(;
                                                         me = SimpleExpectedReturns(;
                                                                                    cvg = cvg),
                                                         ce = PortfolioOptimisersCovariance(;
                                                                                            ce = Covariance(;
                                                                                                            cvg = cvg)),
                                                         fill_limit = 0.8), kte = ktg,
                                     ske = skg)
        @test !pe.supports_partial_fit(hg.ske)
        @test !pe.supports_partial_fit(hg.kte)
        og = prior(fold(hg, Xg))
        bg = prior(hg, Xg)
        @test iszero(og.X[1:30, 3])
        @test isapprox(og.sk, bg.sk; rtol = 1e-12)
        @test isapprox(og.kt, bg.kt; rtol = 1e-12)
        @test !isapprox(coskewness(skg, og.X)[1], bg.sk; rtol = 1e-1)
        @test !isapprox(cokurtosis(ktg, og.X), bg.kt; rtol = 1e-1)
        # An `Online` window: the buffer holds the last `m` rows, and the refit reads them,
        # which is the batch fit over the same window.
        m = 30
        ho = pe.update_online_estimator(HighOrderPriorEstimator(;
                                                                pe = pe.Online(EmpiricalPrior();
                                                                               max_history = m),
                                                                ske = Coskewness(;
                                                                                 alg = SemiMoment())))
        oo = prior(fold(ho, X))
        @test isapprox(oo.sk,
                       coskewness(Coskewness(; alg = SemiMoment()),
                                  X[(end - m + 1):end, :])[1]; rtol = 1e-12)
        # `combine_factor_answers` is Kleene's strong disjunction, `nothing` the unknown
        # value, over every tuple of one to three answers.
        kleene(a, b) =
            if a === true || b === true
                true
            elseif a === false && b === false
                false
            else
                nothing
            end
        vals = (true, false, nothing)
        for k in 1:3, c in Iterators.product(ntuple(_ -> vals, k)...)
            @test pe.combine_factor_answers(c) === foldl(kleene, c)
        end
        # A member that holds observation weights refuses the fold by name.
        ww = eweights(1:80, inv(40); scale = true)
        @test_throws ArgumentError fold(EmpiricalPrior(;
                                                       me = SimpleExpectedReturns(; w = ww)),
                                        X)
        # The slice of a carry state takes a `Bool` mask and one index.
        s = pe.partial_fit_cache(fold(EmpiricalPrior(), X))
        push!(s.named, 2, 5)
        v = pe.port_opt_view(s, BitVector([0, 1, 0, 0, 1, 1]))
        @test v.named == Set([1, 2])
        @test pe.sample_buffer(v) == X[:, [2, 5, 6]]
        @test pe.port_opt_view(s, 5).named == Set([1])
        # The refusal names both routes that give a factor matrix.
        msg = try
            pe.assert_factor_returns(EntropyPoolingPrior(; pe = FactorPrior()), nothing)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("ReturnsResult.F", msg)
        @test occursin("partial_fit!", msg)
    end

    @testset "A read-out before the first fold refuses by name" begin
        @test_throws ArgumentError prior(EmpiricalPrior())
        @test_throws ArgumentError prior(EntropyPoolingPrior())
        @test_throws ArgumentError cov(PortfolioOptimisersCovariance())
    end

    # Issue #1055: the `Online` docstring named `EmpiricalPrior(; ce = Online(…))`, which the
    # `ce` bound refuses. Only a `pe` slot admits a wrapper, because the prior owns the rows
    # once, at the bottom of its chain; a wrapper two levels down is what the wrapping
    # priors' own `update_online_estimator` method exists to seed.
    @testset "A wrapper lives in a `pe` slot, never in a moment slot" begin
        @test_throws TypeError EmpiricalPrior(;
                                              ce = pe.Online(PortfolioOptimisersCovariance()))
        @test_throws TypeError EmpiricalPrior(; me = pe.Online(SimpleExpectedReturns()))
        @test_throws TypeError HighOrderPriorEstimator(; ske = pe.Online(Coskewness()))
        @test_throws TypeError HighOrderPriorEstimator(; kte = pe.Online(Cokurtosis()))
        views = pe.BlackLittermanViews(; P = [1.0 zeros(1, 5)], Q = [0.01])
        h = HighOrderPriorEstimator(;
                                    pe = BlackLittermanPrior(;
                                                             pe = pe.Online(EmpiricalPrior()),
                                                             views = views))
        # The host's own fields hold no wrapper, so the generic scan finds none: the seeding
        # is the wrapping prior's recursion into `pe.pe`.
        @test pe.online_candidate_fields(h) == ()
        @test pe.online_fields(h.pe) == (:pe,)
        s = pe.update_online_estimator(h)
        @test s.pe.pe.cache isa pe.SampleBufferState
        o = prior(fold(s, X))
        b = prior(HighOrderPriorEstimator(;
                                          pe = BlackLittermanPrior(; pe = EmpiricalPrior(),
                                                                   views = views)), X)
        @test isapprox(o.mu, b.mu; rtol = 1e-10)
        @test isapprox(o.sigma, b.sigma; rtol = 1e-10)
    end
end
# Issue #1467 (ADR 0193): `partial_fit!(pe, rd)` is the fold of every prior, and it mirrors
# `prior(pe, rd)`. The route of the prior decides what the step honours: a refit records the
# estimation mask and the Panel Fields of a prior that reads them, the carry of
# `EmpiricalPrior` refuses what its exact folds cannot take, and a host forwards `rd`. Each
# refusal names its route.
@testset "The fold of every prior over a ReturnsResult, by route" begin
    using Test, PortfolioOptimisers, Statistics, StableRNGs, LinearAlgebra
    po = PortfolioOptimisers
    rng = StableRNG(14670930)
    T, N = 40, 4
    X = randn(rng, T, N) ./ 100
    M = trues(T, N)
    M[1:6, 2] .= false
    X[1:6, 2] .= NaN
    Em = copy(M)
    Em[1:12, 3] .= false
    nx = ["A$i" for i in 1:N]
    cap = NumericPanelField(; name = "cap", vals = randn(rng, T, N))
    masks = AssetPanel(; amsk = M, emsk = Em)
    full = AssetPanel(; pf = [cap], amsk = M, emsk = Em)
    same = AssetPanel(; amsk = M, emsk = copy(M))
    rd(pnl; kwargs...) = ReturnsResult(; nx = nx, X = X, pnl = pnl, kwargs...)
    rows(r, i) = po.port_opt_view(r, i, :)
    steps(est, r) = foldl((e, i) -> partial_fit!(e, rows(r, i:i)), 1:T; init = est)
    online(est; kwargs...) = po.update_online_estimator(Online(est; kwargs...))
    function message(f)
        return try
            f()
            ""
        catch err
            err.msg
        end
    end

    @testset "The refit equals the matrix form and honours the estimation mask" begin
        a = steps(online(EmpiricalPrior()), rd(masks))
        b = partial_fit!(online(EmpiricalPrior()), X; active_mask = M, estimation_mask = Em)
        @test po.sample_buffer_kwargs(a.cache).estimation_mask == Em
        @test isequal(prior(a).mu, prior(b).mu) && isequal(prior(a).sigma, prior(b).sigma)
        @test isequal(prior(a).sigma, prior(EmpiricalPrior(), rd(masks)).sigma)
        # The probe reads the estimation mask, so a refit that dropped it would differ.
        p = steps(online(PanelFieldProbe()), rd(full))
        @test isequal(prior(p).mu, prior(PanelFieldProbe(), rd(full)).mu)
        @test !isequal(prior(p).mu,
                       prior(PanelFieldProbe(),
                             rd(AssetPanel(; pf = [cap], amsk = M, emsk = copy(M)))).mu)
        # No panel and a static panel give no mask to record.
        static = AssetPanel(; pf = [NumericPanelField(; name = "cap", vals = ones(N))])
        @test isnothing(partial_fit!(online(EmpiricalPrior()), rd(static)).cache.A)
        @test isnothing(partial_fit!(online(EmpiricalPrior()), rd(nothing)).cache.A)
    end
    @testset "The Panel Fields reach only a prior that reads them" begin
        m = message(() -> partial_fit!(online(EmpiricalPrior()), rd(full)))
        @test occursin("the refit of `EmpiricalPrior` reads no Panel Field", m)
        @test occursin("1 Panel Field(s)", m)
        p = steps(online(PanelFieldProbe()), rd(full))
        @test isequal(po.panel_field_array(only(p.cache.P)), cap.vals)
        # A block fold is the same state as a fold row by row.
        @test isequal(prior(partial_fit!(online(PanelFieldProbe()), rd(full))).mu,
                      prior(p).mu)
        # A cap windows the Panel Fields with the rows, so the read-out is the batch fit over
        # the last rows.
        w = 15
        pc = steps(online(PanelFieldProbe(); max_history = w), rd(full))
        @test isequal(prior(pc).mu,
                      prior(PanelFieldProbe(), rows(rd(full), (T - w + 1):T)).mu)
    end
    @testset "The carry refuses what its exact folds cannot honour, by name" begin
        m = message(() -> partial_fit!(EmpiricalPrior(), rd(masks)))
        @test occursin("the carry of `EmpiricalPrior` folds its moments exactly", m)
        @test occursin("$(count(Em .!= M)) cell(s)", m)
        m = message(() -> partial_fit!(EmpiricalPrior(), rd(full)))
        @test occursin("the carry of `EmpiricalPrior` reads no Panel Field", m)
        # With equal masks the carry folds the active mask, as its matrix form does.
        a = partial_fit!(EmpiricalPrior(), rd(same))
        b = partial_fit!(EmpiricalPrior(), X; active_mask = M)
        @test isequal(a.cache.buf.A, b.cache.buf.A)
        @test isnothing(partial_fit!(EmpiricalPrior(), rd(nothing)).cache.buf.A)
        static = AssetPanel(; pf = [NumericPanelField(; name = "cap", vals = ones(N))])
        @test isnothing(partial_fit!(EmpiricalPrior(), rd(static)).cache.buf.A)
    end
    @testset "A host forwards the ReturnsResult to the prior it embeds" begin
        Xg = randn(rng, T, N) ./ 100
        rg = ReturnsResult(; nx = nx, X = Xg,
                           pnl = AssetPanel(; pf = [cap], amsk = trues(T, N),
                                            emsk = trues(T, N)))
        h = po.update_online_estimator(HighOrderPriorEstimator(;
                                                               pe = Online(PanelFieldProbe())))
        ho = prior(partial_fit!(h, rg))
        hb = prior(HighOrderPriorEstimator(; pe = PanelFieldProbe()), rg)
        # The read-out reads a view of the buffer and the batch fit reads a matrix, so the
        # means can differ in the last bit.
        @test isapprox(ho.mu, hb.mu; rtol = 1e-12)
        @test isapprox(ho.kt, hb.kt) && isapprox(ho.sk, hb.sk)
        views = po.BlackLittermanViews(; P = [1.0 zeros(1, N - 1)], Q = [0.01])
        bl = po.update_online_estimator(BlackLittermanPrior(;
                                                            pe = Online(PanelFieldProbe()),
                                                            views = views))
        @test isapprox(prior(partial_fit!(bl, rg)).mu,
                       prior(BlackLittermanPrior(; pe = PanelFieldProbe(), views = views),
                             rg).mu)
        # The host of a carry refuses what the carry refuses.
        m = message(() -> partial_fit!(HighOrderPriorEstimator(), rd(masks)))
        @test occursin("the carry of `EmpiricalPrior`", m)
    end
    @testset "The fold refuses what no state records" begin
        @test_throws ArgumentError partial_fit!(online(EmpiricalPrior()),
                                                rd(nothing; iv = fill(0.2, T, N),
                                                   ivpa = 1.0))
        @test_throws IsNothingError partial_fit!(online(EmpiricalPrior()),
                                                 ReturnsResult(; nf = ["F1"],
                                                               F = randn(rng, T, 1)))
    end
    @testset "reads_panel_fields answers from the tree" begin
        @test po.reads_panel_fields(CrossSectionalFactorPrior(; lambda = 1,
                                                              factors = ["market" =>
                                                                             ConstantExposure()]))
        @test !po.reads_panel_fields(EmpiricalPrior())
        @test !po.reads_panel_fields(nothing)
        @test po.reads_panel_fields(Online(PanelFieldProbe()))
        @test po.reads_panel_fields(HighOrderPriorEstimator(; pe = PanelFieldProbe()))
        @test po.reads_panel_fields(EntropyPoolingPrior(; pe = PanelFieldProbe()))
        @test po.reads_panel_fields(MeucciEntropyPoolingPrior(; pe = PanelFieldProbe()))
        @test !po.reads_panel_fields(OpinionPoolingPrior(; pes = [EntropyPoolingPrior()]))
        @test po.reads_panel_fields(OpinionPoolingPrior(; pes = [EntropyPoolingPrior()],
                                                        pe1 = PanelFieldProbe()))
    end
end
