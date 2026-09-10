#=
The sample buffer and the `Online` wrapper that seeds it, issue #967.

The foundation every layer above the moments folds through. `SampleBufferState` is the
partial-fit state a member with no exact incremental fold keeps its observations in, and
`Online` is the configuration that seeds one: it is transient, so it resolves at warm-up
and no wrapper survives into the run. Issue #865 settled the design, and its resolution
comment is the specification these tests read.

Two probe types stand in for the members the layers above will be. They are the shape those
members have: a `cache` field and a keyword constructor, which is all the seam and
`rebuild_estimator` read.

Issue #997 then made the eleven shipped families subjects of their own. Each narrows the
`cache` type parameter of its own `partial_fit!` methods to the state its exact fold reads,
so a wrapped estimator falls through to the generic buffering method and answers every
read-out verb by running the batch verb over the buffer's rows. The last testset drives all
eleven, uncapped and capped.
=#
struct BufferProbe{T} <: PortfolioOptimisers.AbstractEstimator
    cache::T
end
function BufferProbe(; cache = nothing)
    return BufferProbe(cache)
end
struct HostProbe{T1, T2} <: PortfolioOptimisers.AbstractEstimator
    inner::T1
    cache::T2
end
function HostProbe(; inner = nothing, cache = nothing)
    return HostProbe(inner, cache)
end
struct NoCacheProbe{T} <: PortfolioOptimisers.AbstractEstimator
    val::T
end
struct SchedProbe{T1, T2, T3} <: PortfolioOptimisers.AbstractEstimator
    inner::T1
    sched::T2
    cache::T3
end
function SchedProbe(; inner = nothing, sched = nothing, cache = nothing)
    return SchedProbe(inner, sched, cache)
end
@testset "The sample buffer and the Online wrapper" begin
    using Test, PortfolioOptimisers, StableRNGs, Statistics
    po = PortfolioOptimisers
    rng = StableRNG(987654321)
    X = randn(rng, 20, 4)
    # A quote outside an asset's Listing Span is a `NaN` inside a fixed width, and the buffer
    # holds it verbatim. Every comparison below is `isequal`, because `==` is false on it.
    X[3, 2] = NaN
    buf = po.sample_buffer
    fold(state, rows) = foldl(partial_fit!, eachrow(rows); init = state)

    @testset "Append and read back" begin
        # One observation at a time holds exactly the rows a batch call would, in order.
        @test isequal(buf(fold(po.SampleBufferState(), X)), X)
        @test isnan(buf(fold(po.SampleBufferState(), X))[3, 2])
        @test isequal(buf(partial_fit!(po.SampleBufferState(), X)), X)
        # `dims = 2` answers what the transpose answers.
        @test isequal(buf(partial_fit!(po.SampleBufferState(), permutedims(X); dims = 2)),
                      X)
        # A block and then single rows is the warm-up followed by the steps, and it is the
        # same buffer.
        @test isequal(buf(fold(partial_fit!(po.SampleBufferState(), view(X, 1:5, :)),
                               view(X, 6:20, :))), X)
        # An empty block leaves the buffer alone.
        s = partial_fit!(po.SampleBufferState(), X)
        @test isequal(buf(partial_fit!(s, X[1:0, :])), X)
    end

    @testset "Growth" begin
        # The amortised path returns the buffer an exact-fit allocation returns.
        one_at_a_time = fold(po.SampleBufferState(), X)
        all_at_once = partial_fit!(po.SampleBufferState(), X)
        @test isequal(buf(one_at_a_time), buf(all_at_once))
        # An exact fit on the first call: a batch warm-up allocates the rows it is given and
        # no more.
        @test size(all_at_once.X, 1) == size(X, 1)
        # Geometric growth after it. Twenty appends reallocate a handful of times, not twenty.
        caps = Int[]
        s = partial_fit!(po.SampleBufferState(), view(X, 1:1, :))
        for i in 2:size(X, 1)
            s = partial_fit!(s, view(X, i, :))
            push!(caps, size(s.X, 1))
        end
        @test length(unique(caps)) <= 6
        @test s.n == size(X, 1)
        # A capped buffer never grows without bound, however many observations it is shown.
        c = po.SampleBufferState(; max_history = 3)
        for k in 1:200
            c = partial_fit!(c, view(X, mod1(k, 20), :))
        end
        @test c.n == 3
        # A cap of `w` costs at most `2w` rows, whatever the number of appends.
        @test size(c.X, 1) <= 2 * 3
        for k in 201:2000
            c = partial_fit!(c, view(X, mod1(k, 20), :))
        end
        @test size(c.X, 1) <= 2 * 3
        # A large batch append truncates the incoming rows before it copies them, so the
        # buffer never materialises the whole history.
        big = repeat(X, 500)
        @test size(big, 1) == 10_000
        b = partial_fit!(po.SampleBufferState(; max_history = 5), big)
        @test size(b.X, 1) == 5
        @test isequal(buf(b), big[(end - 4):end, :])
    end

    @testset "The width refusal" begin
        s = partial_fit!(po.SampleBufferState(), X)
        @test_throws DimensionMismatch partial_fit!(s, randn(rng, 3))
        @test_throws DimensionMismatch partial_fit!(s, randn(rng, 4, 5))
        # The message names both widths.
        msg = try
            partial_fit!(s, randn(rng, 3))
        catch e
            sprint(showerror, e)
        end
        @test occursin("4 assets", msg)
        @test occursin("3 columns", msg)
        # The seed fixes the width, so the first append of any width is accepted.
        @test size(partial_fit!(po.SampleBufferState(), randn(rng, 2, 7)).X, 2) == 7
    end

    @testset "merge_states is concatenation" begin
        a = partial_fit!(po.SampleBufferState(), view(X, 1:7, :))
        b = partial_fit!(po.SampleBufferState(), view(X, 8:20, :))
        @test isequal(buf(po.merge_states(a, b)), X)
        # Exactly, not to a tolerance: the buffer holds its rows verbatim.
        @test buf(po.merge_states(a, b))[[1:2; 4:20], :] == X[[1:2; 4:20], :]
        # A cap applies to the concatenation, which keeps its last `max_history` rows.
        ca = partial_fit!(po.SampleBufferState(; max_history = 5), view(X, 1:7, :))
        cb = partial_fit!(po.SampleBufferState(; max_history = 5), view(X, 8:20, :))
        @test isequal(buf(po.merge_states(ca, cb)), X[16:20, :])
        # Two buffers of different caps, and two over different universes, are refused.
        @test_throws ArgumentError po.merge_states(a, ca)
        wide = partial_fit!(po.SampleBufferState(), randn(rng, 3, 5))
        @test_throws DimensionMismatch po.merge_states(a, wide)
    end

    @testset "copy protects" begin
        a = partial_fit!(po.SampleBufferState(), view(X, 1:7, :))
        c = copy(a)
        partial_fit!(c, view(X, 8, :))
        @test isequal(buf(a), X[1:7, :])
        @test a.n == 7
        # The value form of the seam folds a copy, so the estimator handed over is untouched.
        warm = po.update_online_estimator(Online(BufferProbe()))
        warm = partial_fit!(warm, view(X, 1:7, :))
        cold = partial_fit(warm, view(X, 8, :))
        @test isequal(po.sample_buffer(warm), X[1:7, :])
        @test isequal(po.sample_buffer(cold), X[1:8, :])
    end

    @testset "port_opt_view slices, obs_weights_view drops" begin
        s = partial_fit!(po.SampleBufferState(), X)
        v = po.port_opt_view(s, [1, 3])
        @test isequal(buf(v), X[:, [1, 3]])
        # The slice copies by index, so a later fold on the view does not write through.
        partial_fit!(v, [1.0, 2.0])
        @test isequal(buf(s), X)
        # No slice of a state exists on the observation axis.
        @test po.obs_weights_view(s, 1:5) === nothing
    end

    @testset "The wrapper resolves away" begin
        o = Online(BufferProbe(); max_history = 5)
        @test o.max_history == 5
        r = po.update_online_estimator(o)
        # No `Online` survives, and the cache holds the state the route requires.
        @test isa(r, BufferProbe)
        @test isa(r.cache, po.SampleBufferState)
        @test r.cache.max_history == 5
        @test isempty(po.online_fields(r))
        # A host resolves the wrappers in its own fields, and rebuilds through its keyword
        # constructor.
        h = HostProbe(; inner = Online(BufferProbe(); max_history = 2))
        @test po.online_fields(h) == (:inner,)
        hr = po.update_online_estimator(h)
        @test isa(hr, HostProbe)
        @test isa(hr.inner, BufferProbe)
        @test hr.inner.cache.max_history == 2
        @test isempty(po.online_fields(hr))
        # A wrapper wrapping a host resolves the host's own wrappers too.
        n = Online(HostProbe(; inner = Online(BufferProbe(); max_history = 3)))
        nr = po.update_online_estimator(n)
        @test isa(nr, HostProbe)
        @test isa(nr.cache, po.SampleBufferState)
        @test isnothing(nr.cache.max_history)
        @test isa(nr.inner, BufferProbe)
        @test nr.inner.cache.max_history == 3
        # A host with no wrapper is returned unchanged, and costs no rebuild.
        plain = HostProbe(; inner = BufferProbe())
        @test po.update_online_estimator(plain) === plain
        @test po.online_candidate_fields(plain) == ()
        @test isnothing(po.update_online_estimator(nothing))
        # Wrappers do not nest, and a cap is a positive count.
        @test_throws ArgumentError Online(o)
        @test_throws DomainError Online(BufferProbe(); max_history = 0)
        # An estimator with nowhere to carry a buffer cannot be wrapped.
        @test_throws ArgumentError Online(NoCacheProbe(1))
    end

    @testset "The generic buffering fold" begin
        r = po.update_online_estimator(Online(BufferProbe()))
        r = partial_fit!(r, X)
        @test isequal(po.sample_buffer(r), X)
        r = partial_fit!(r, view(X, 1, :))
        @test isequal(po.sample_buffer(r), vcat(X, transpose(X[1, :])))
        # An estimator that was never wrapped meets a message naming the wrapper, not a
        # `MethodError`.
        @test_throws ArgumentError partial_fit!(BufferProbe(), X)
        msg = try
            partial_fit!(BufferProbe(), X)
        catch e
            sprint(showerror, e)
        end
        @test occursin("Online", msg)
        @test_throws ArgumentError po.sample_buffer(BufferProbe())
        # A wrapper is not an estimator that folds: it must be resolved first.
        @test_throws ArgumentError po.assert_sample_buffer(Online(BufferProbe()))
        @test_throws ArgumentError partial_fit!(Online(BufferProbe()), X)
    end

    @testset "A wrapper and a schedule compose in one direction" begin
        # Neither wraps the other, because they resolve at different times: a wrapper once at
        # warm-up, a schedule once per fold.
        @test_throws ArgumentError Online(TimeDependent([BufferProbe(), BufferProbe()]))
        @test_throws ArgumentError TimeDependent([Online(BufferProbe()),
                                                  Online(BufferProbe())])
        @test_throws ArgumentError TimeDependent([1, 2]; default = Online(BufferProbe()))
        # A host may hold a wrapper in one field and a schedule in another. Each resolves at
        # its own time, and seeding leaves the schedule untouched.
        h = SchedProbe(; inner = Online(BufferProbe(); max_history = 5),
                       sched = TimeDependent([1, 2]))
        @test po.online_fields(h) == (:inner,)
        @test po.time_dependent_fields(h) == (:sched,)
        hr = po.update_online_estimator(h)
        @test isa(hr.inner.cache, po.SampleBufferState)
        @test hr.inner.cache.max_history == 5
        @test isa(hr.sched, TimeDependent)
        @test isempty(po.online_fields(hr))
        @test po.time_dependent_fields(hr) == (:sched,)
        # An estimator a wrapper wraps may hold schedules of its own, which survive the
        # seeding and resolve per fold afterwards.
        r = po.update_online_estimator(Online(SchedProbe(; sched = TimeDependent([1, 2]));
                                              max_history = 3))
        @test isa(r, SchedProbe)
        @test isa(r.cache, po.SampleBufferState)
        @test r.cache.max_history == 3
        @test po.time_dependent_fields(r) == (:sched,)
        # The two candidate scans are disjoint, so neither resolution reaches the other's
        # wrapper.
        @test po.online_candidate_fields(SchedProbe(; sched = TimeDependent([1, 2]))) == ()
        @test po.time_dependent_candidate_fields(SchedProbe(;
                                                            inner = Online(BufferProbe()))) ==
              ()
        # A *callable* schedule cannot be checked at construction, because its value does not
        # exist until the fold does. A wrapper it returns therefore reaches the fold
        # unresolved, and meets a message naming the warm-up rather than the generic one.
        td = TimeDependent(c -> Online(BufferProbe()))
        @test isa(td, TimeDependent)
        msg = try
            po.assert_sample_buffer(Online(BufferProbe()))
        catch e
            sprint(showerror, e)
        end
        @test occursin("reached a fold unresolved", msg)
        @test occursin("callable", msg)
        # The unwrapped case keeps its own message, which names the wrapper.
        msg = try
            po.assert_sample_buffer(BufferProbe())
        catch e
            sprint(showerror, e)
        end
        @test occursin("Wrap it in `Online`", msg)
    end

    @testset "The cap" begin
        # A capped buffer holds exactly the last `w` rows, fold by fold and in one block.
        for w in (1, 6, 19)
            s = fold(po.SampleBufferState(; max_history = w), X)
            @test s.n == w
            @test isequal(buf(s), X[(end - w + 1):end, :])
            @test isequal(buf(partial_fit!(po.SampleBufferState(; max_history = w), X)),
                          X[(end - w + 1):end, :])
        end
        # A cap wider than the sample keeps every row.
        @test isequal(buf(fold(po.SampleBufferState(; max_history = 100), X)), X)
        # The constructor refuses a state whose valid region leaves its backing matrix, and
        # one that holds more observations than its cap.
        @test_throws DimensionMismatch po.SampleBufferState(; n = 3, X = zeros(2, 2))
        @test_throws DimensionMismatch po.SampleBufferState(; n = 2, off = 1,
                                                            X = zeros(2, 2))
        @test_throws DimensionMismatch po.SampleBufferState(; n = 2, X = zeros(2, 2),
                                                            max_history = 1)
        @test_throws DomainError po.SampleBufferState(; X = zeros(2, 2), max_history = 0)
        @test_throws DomainError po.SampleBufferState(; n = -1, X = zeros(2, 2))
        @test_throws DomainError po.SampleBufferState(; off = -1, X = zeros(2, 2))
    end
    @testset "Online over every family that folds exactly" begin
        # Issue #997. A family that folds exactly narrows the `cache` type parameter of its
        # own `partial_fit!` methods, so a wrapped estimator never meets them: it buffers,
        # and every read-out verb answers by running the batch verb over the buffer's rows.
        # The cap is the window -- there is no special case for an estimator that would
        # otherwise fold.
        rng2 = StableRNG(135792468)
        Y = randn(rng2, 60, 4) ./ 100
        # Wider than the `min_obs` of the exponentially weighted families, which answer
        # `NaN` over a shorter window, and narrower than the sample, so the capped read-out
        # and the uncapped one are different numbers.
        w = 45
        families = ((; name = "SimpleExpectedReturns", est = SimpleExpectedReturns(),
                     readout = e -> Statistics.mean(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.mean(e, Z), exact = true),
                    (; name = "SimpleVariance", est = SimpleVariance(),
                     readout = e -> Statistics.var(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.var(e, Z), exact = true),
                    (; name = "GeneralCovariance", est = GeneralCovariance(),
                     readout = e -> Statistics.cov(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.cov(e, Z), exact = true),
                    (; name = "Covariance", est = Covariance(),
                     readout = e -> Statistics.cov(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.cov(e, Z), exact = true),
                    (; name = "Covariance/SemiMoment",
                     est = Covariance(; alg = SemiMoment()),
                     readout = e -> Statistics.cov(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.cov(e, Z), exact = false),
                    (; name = "Coskewness", est = Coskewness(),
                     readout = e -> first(coskewness(e, po.partial_fit_cache(e))),
                     batch = (e, Z) -> first(coskewness(e, Z)), exact = true),
                    (; name = "Cokurtosis", est = Cokurtosis(),
                     readout = e -> cokurtosis(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> cokurtosis(e, Z), exact = true),
                    (; name = "ExpWeightedExpectedReturns",
                     est = ExpWeightedExpectedReturns(),
                     readout = e -> Statistics.mean(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.mean(e, Z), exact = true),
                    (; name = "ExpWeightedVariance", est = ExpWeightedVariance(),
                     readout = e -> Statistics.var(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.var(e, Z), exact = true),
                    (; name = "ExpWeightedCovariance", est = ExpWeightedCovariance(),
                     readout = e -> Statistics.cov(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.cov(e, Z), exact = true),
                    (; name = "RegimeAdjustedExpWeightedVariance",
                     est = RegimeAdjustedExpWeightedVariance(),
                     readout = e -> Statistics.var(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.var(e, Z), exact = true),
                    (; name = "RegimeAdjustedExpWeightedCovariance",
                     est = RegimeAdjustedExpWeightedCovariance(),
                     readout = e -> Statistics.cov(e, po.partial_fit_cache(e)),
                     batch = (e, Z) -> Statistics.cov(e, Z), exact = true))
        @testset "$(fam.name)" for fam in families
            # Wrapping seeds a buffer, and the fold reaches the generic buffering method
            # rather than the family's own, whether that method folds or refuses.
            wrapped = po.update_online_estimator(Online(fam.est))
            @test isa(wrapped.cache, po.SampleBufferState)
            folded = partial_fit!(wrapped, Y)
            @test isa(folded.cache, po.SampleBufferState)
            @test isequal(po.sample_buffer(folded), Y)
            # An uncapped buffer answers what a batch fit over every observation answers.
            @test isapprox(fam.readout(folded), fam.batch(fam.est, Y))
            # The cap is the window: a capped buffer answers what a batch fit over the last
            # `max_history` observations answers, for a family that folds exactly too.
            capped = partial_fit!(po.update_online_estimator(Online(fam.est;
                                                                    max_history = w)), Y)
            @test isequal(po.sample_buffer(capped), Y[(end - w + 1):end, :])
            @test isapprox(fam.readout(capped), fam.batch(fam.est, Y[(end - w + 1):end, :]))
            # An estimator left unwrapped is untouched: it still folds into its own family
            # state, and its own state is not a buffer.
            if fam.exact
                unwrapped = partial_fit!(fam.est, Y)
                @test !isa(unwrapped.cache, po.SampleBufferState)
                @test isa(unwrapped.cache, po.AbstractPartialFitState)
            else
                @test_throws ArgumentError partial_fit!(fam.est, Y)
            end
        end
        # A buffer records no per-observation activity, so it refuses a Coverage Policy mask
        # rather than folding a block whose mask it would have to discard.
        wrapped = po.update_online_estimator(Online(Covariance()))
        @test_throws ArgumentError partial_fit!(wrapped, Y; active_mask = trues(size(Y)))
        @test_throws ArgumentError partial_fit!(wrapped, view(Y, 1, :);
                                                active_mask = trues(size(Y, 2)))
        @test_throws ArgumentError partial_fit!(po.update_online_estimator(Online(RegimeAdjustedExpWeightedCovariance())),
                                                Y; estimation_mask = trues(size(Y)))
        # The one-argument read-out is bound to every moment algorithm, so a wrapped
        # `SemiMoment` covariance answers it and an unwrapped one that carries nothing meets
        # the named refusal rather than a `MethodError`.
        semi = Covariance(; alg = SemiMoment())
        @test isapprox(Statistics.cov(partial_fit!(po.update_online_estimator(Online(semi)),
                                                   Y)), Statistics.cov(semi, Y))
        @test_throws ArgumentError Statistics.cov(semi)
    end
end
