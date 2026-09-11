"""
$(DocStringExtensions.TYPEDEF)

Carries the observations a prior keeps when it folds its estimate exactly, and the assets whose scenario fill it has already named.

The state of a **fold-and-carry** prior, and what separates that route from a refit. A [`SampleBufferState`](@ref) means refit, everywhere in the library: an estimator carrying one has no recursion of its own, and its read-out is the batch verb over the rows the buffer kept. A prior carrying *this* state has folded its moments exactly, member by member, and keeps the rows for one reason only — [`LowOrderPrior`](@ref) carries `X` for the scenario risk measures, so the observations are **memory, not arithmetic**. Its read-out reads two folded moments and a matrix it already holds, and touches none of the rows. ADR 0136 records the decision.

A prior a wrapper seeds therefore falls to the refit route rather than to this one, which is what makes `Online`'s cap the window of the whole fit.

# The second field, and why a read-out writes to it

`named` is the set of assets a [`scenario_fill`](@ref) has told the caller about. [`strict_diagnostic`](@ref) has no memory of its own, so a batch fit names an asset once and a walk-forward reading out at every step names it at every step: two thousand notices about one listing, and a caller who learns to ignore the channel. The set is that memory, and a read-out reports only the assets outside it.

It is a `Set`, and a read-out writes into it **in place**, because a read-out returns a Prior Result rather than the estimator and there is no other channel by which the memory could survive the call. [`Base.copy`](@ref) copies it, so a state copied before a fold carries a memory of its own, and [`merge_states`](@ref) unions two.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriorCarryState(;
        buf::SampleBufferState = SampleBufferState(),
        named::Set{Int} = Set{Int}()
    ) -> PriorCarryState

Keywords correspond to the struct's fields. The default is the empty seed a first fold builds.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `buf`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `named`: Remapped onto the selected indices, so an asset named before the slice is still named after it.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`SampleBufferState`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`fold_carry`](@ref)
  - [`scenario_fill`](@ref)
  - [`partial_fit!`](@ref)
"""
@concrete struct PriorCarryState <: AbstractPartialFitState
    """
    Buffer of the observations the prior carries, `observations × assets`, held verbatim and `NaN` included.
    """
    buf
    """
    Indices of the assets whose scenario fill has already been named, written in place by a read-out that names one.
    """
    named
end
function PriorCarryState(; buf::SampleBufferState = SampleBufferState(),
                         named::Set{Int} = Set{Int}())::PriorCarryState
    return PriorCarryState(buf, named)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the observations a [`PriorCarryState`](@ref) carries, `observations × assets`.

# Arguments

  - `state`: The carry state to read.

# Returns

  - `X::SubArray`: The observations the state carries, in the order they were folded.

# Related

  - [`PriorCarryState`](@ref)
  - [`sample_buffer`](@ref)
"""
function sample_buffer(state::PriorCarryState)
    return sample_buffer(state.buf)
end
"""
    partial_fit!(state::PriorCarryState, x::VecNum)
    partial_fit!(state::PriorCarryState, X::MatNum; dims::Int = 1)

Folds observations into the buffer a [`PriorCarryState`](@ref) carries.

The buffer's own fold, forwarded, keyword arguments included: a [`CoveragePolicy`](@ref) mask travels into the buffer beside the rows it explains, exactly as it does for a [`SampleBufferState`](@ref) an [`Online`](@ref) seeded. The named-asset set is untouched: a fold reads no moment and names nothing, and only a read-out can find an asset to name.

# Arguments

  - `state`: The carry state to fold into.
  - `x`: One observation, whose entries are the assets.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Returns

  - `state::PriorCarryState`: The state after the last observation.

# Related

  - [`PriorCarryState`](@ref)
  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(state::PriorCarryState, x::VecNum; kwargs...)
    return Accessors.@reset state.buf = partial_fit!(state.buf, x; kwargs...)
end
function partial_fit!(state::PriorCarryState, X::MatNum; dims::Int = 1, kwargs...)
    return Accessors.@reset state.buf = partial_fit!(state.buf, X; dims = dims, kwargs...)
end
"""
    fold_carry(cache::Option{<:PriorCarryState}, x::VecNum)
    fold_carry(cache::Option{<:PriorCarryState}, X::MatNum; dims::Int = 1)

Folds observations into a [`PriorCarryState`](@ref), seeding an empty one where the prior carries none.

The one line a fold-and-carry prior writes for its rows, beside the two that fold its moments. It is [`fold_buffer`](@ref) with the second field carried along, and the seed it builds is uncapped: a cap on the carry is `EmpiricalPrior`'s `max_scenarios`, which is applied at the read-out, and a cap on the *fit* is [`Online`](@ref)'s, which puts the prior on the refit route instead.

# Arguments

  - `cache`: The state the prior carries, or `nothing`.
  - `x`: One observation, whose entries are the assets.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Returns

  - `state::PriorCarryState`: The state after the last observation.

# Related

  - [`PriorCarryState`](@ref)
  - [`fold_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_carry(cache::Option{<:PriorCarryState}, x::VecNum; kwargs...)
    return partial_fit!(isnothing(cache) ? PriorCarryState() : cache, x; kwargs...)
end
function fold_carry(cache::Option{<:PriorCarryState}, X::MatNum; dims::Int = 1, kwargs...)
    return partial_fit!(isnothing(cache) ? PriorCarryState() : cache, X; dims = dims,
                        kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`PriorCarryState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

The buffer's merge, which is concatenation, and the **union** of the two named-asset sets: an asset either half has already told the caller about does not need telling again.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Returns

  - `state::PriorCarryState`: The state the two blocks give when they are folded as one block.

# Related

  - [`PriorCarryState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(a::PriorCarryState, b::PriorCarryState)
    return PriorCarryState(merge_states(a.buf, b.buf), union(a.named, b.named))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`PriorCarryState`](@ref), so the copy shares no array and no set with the original.

The set is copied for the same reason the backing matrix is: a read-out writes into it, so a copy that shared it would report through the original's memory.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::PriorCarryState`: A fresh state, equal to `x`.

# Related

  - [`PriorCarryState`](@ref)
  - [`partial_fit`](@ref)
"""
function Base.copy(x::PriorCarryState)
    return PriorCarryState(copy(x.buf), copy(x.named))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`PriorCarryState`](@ref) to the selected assets.

The buffer is sliced as it always is, and the named-asset set is **remapped** rather than dropped or carried: its entries are asset indices, and the slice renumbers the assets. An asset the state had already named keeps its silence at its new index, and an asset the slice removed is forgotten with it.

# Arguments

  - `x`: The state to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments, forwarded to the buffer.

# Returns

  - `state::PriorCarryState`: The state of the same observations over the selected assets.

# Related

  - [`PriorCarryState`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(x::PriorCarryState, i, args...)
    sel = (1:size(x.buf.X, 2))[i]
    named = Set{Int}(j for (j, k) in enumerate(sel) if k in x.named)
    return PriorCarryState(port_opt_view(x.buf, i, args...), named)
end
"""
    prior(pe::AbstractPriorEstimator; kwargs...)

Reads a prior out of the state its estimator carries, with no data matrix.

The read-out arm of [`prior`](@ref), and the entry point of every prior that has taken the online step. It reads the state and dispatches on its type, which is the whole of the routing:

  - A [`SampleBufferState`](@ref) is a refit. The estimator has no recursion of its own, so its estimate is the batch verb over the observations the buffer kept — every one folded so far when the buffer is uncapped, and the last `max_history` of them when [`Online`](@ref) capped it.
  - A [`FactorSampleBufferState`](@ref) is the same refit over two matrices, for a prior whose batch verb reads a factor matrix beside the returns.
  - A [`PriorCarryState`](@ref) is a fold and a carry, and the families that take it write read-outs of their own.

An estimator that has folded nothing meets [`partial_fit_cache`](@ref)'s refusal, which names the verb that fills the field.

# Arguments

  - `pe`: Prior estimator carrying a state.
  - `kwargs...`: Additional keyword arguments, forwarded to the batch verb.

# Validation

  - `pe.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `pr::AbstractPriorResult`: The prior the estimator's state answers.

# Related

  - [`prior`](@ref)
  - [`SampleBufferState`](@ref)
  - [`FactorSampleBufferState`](@ref)
  - [`PriorCarryState`](@ref)
  - [`Online`](@ref)
  - [`partial_fit_cache`](@ref)
"""
function prior(pe::AbstractPriorEstimator; kwargs...)
    return prior(pe, partial_fit_cache(pe); kwargs...)
end
function prior(pe::AbstractPriorEstimator, state::SampleBufferState; kwargs...)
    return prior(pe, sample_buffer(state); sample_buffer_kwargs(state)..., kwargs...)
end
function prior(pe::AbstractPriorEstimator, state::FactorSampleBufferState; kwargs...)
    return prior(pe, sample_buffer(state.X), sample_buffer(state.F);
                 sample_buffer_kwargs(state.X)..., kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds an observation into a member of a carrying host, where that member folds.

The fold half of the mixed-host rule of ADR 0136. A host that carries the observations folds every member that folds and leaves the rest untouched, because it can run their batch verb over its own rows at the read-out. [`supports_partial_fit`](@ref) is the question, and it is answered from the member's type and its `cache` field, so a host of concrete members resolves the branch at compile time and folds nothing it should not.

A member the host does not hold is `nothing`, and folding it is a no-op.

# Arguments

  - `est`: The member to fold, or `nothing`.
  - `args...`: The observations, forwarded to [`partial_fit!`](@ref).
  - `kwargs...`: Additional keyword arguments, forwarded to [`partial_fit!`](@ref).

# Returns

  - `est`: The member carrying the state after the last observation, or the member unchanged where it does not fold.

# Related

  - [`supports_partial_fit`](@ref)
  - [`partial_fit!`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`HighOrderPriorEstimator`](@ref)
"""
function fold_member(est, args...; kwargs...)
    return supports_partial_fit(est) ? partial_fit!(est, args...; kwargs...) : est
end
function fold_member(::Nothing, args...; kwargs...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads a member's estimate out of its fold, or refits it over the host's own rows.

The read-out half of the mixed-host rule of ADR 0136, and the twin of [`fold_member`](@ref): a member the host folded answers from its state, and a member it did not fold answers from the matrix the host carries. `f` is the member's batch verb — `Statistics.mean`, `Statistics.cov`, [`coskewness`](@ref), [`cokurtosis`](@ref) — and the one-argument form of that same verb is its read-out, which is a convention every family of the seam already follows.

No [`AssetPanel`](@ref) reaches this verb. A panel is fold context rather than sample, and a buffer holds no activity mask, so a member refitted here is refitted over the rows alone; that is [#999](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/999)'s to change.

# Arguments

  - `f`: The member's batch verb.
  - `est`: The member, or `nothing`.
  - `X`: The rows the host carries, as the member was folded on them.
  - `kwargs...`: Additional keyword arguments, forwarded to the **batch** verb alone. A folded member is read with no keyword argument at all, because its state is already the answer and there is no sample left for a keyword to describe.

# Returns

  - The member's estimate.

# Related

  - [`fold_member`](@ref)
  - [`supports_partial_fit`](@ref)
  - [`EmpiricalPrior`](@ref)
"""
function read_member(f::F, est, X::MatNum; kwargs...) where {F}
    return supports_partial_fit(est) ? f(est) : f(est, X; dims = 1, kwargs...)
end
"""
    partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                                    <:Option{<:PriorCarryState}}, X; kwargs...)
    partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                                    <:Option{<:PriorCarryState}}, X; kwargs...)

Folds observations into an [`EmpiricalPrior`](@ref), which folds its moments and carries its rows.

`pe.me` and `pe.ce` fold exactly, so the prior's own step is theirs plus an append: it is quadratic in the number of assets and independent of the number of observations folded, where a refit from a buffer is linear in that number as well. That difference is the whole point of the seam at this layer, and it is why `prior(pe) = prior(pe, sample_buffer(pe))` is the shape to refuse in review — it passes every parity test and throws both exact folds away.

The two arms differ in one line. The no-horizon arm folds the observation as it stands. The **horizon** arm folds `log1p` of it, because the horizon method fits its moments in log space, and carries the observation itself, because the [`LowOrderPrior`](@ref) it returns carries the **arithmetic** returns the caller handed in. Buffering the log rows would silently change the matrix every scenario measure reads.

A member that does not fold — a `SemiMoment` covariance, a composite whose `mp.alg` reads the sample — is left alone by [`fold_member`](@ref) and refitted over the carried rows at the read-out. So a caller writes the estimator they would write in batch, and no member carries a second copy of the sample.

# Algorithm

 1. Fold `pe.me` and `pe.ce` through [`fold_member`](@ref), on the observation under the horizon arm's transform.
 2. Append the arithmetic observation to the carry state through [`fold_carry`](@ref).

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Observations to fold. A matrix holds one observation per row when `dims == 1` and one per column when `dims == 2`; a vector is one observation across the assets.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, forwarded to the two arms.

# Returns

  - `pe`: The estimator, with its arms folded and its `cache` field rebound.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`PriorCarryState`](@ref)
  - [`prior`](@ref)
  - [`fold_member`](@ref)
  - [`fold_carry`](@ref)
"""
function partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                                         <:Option{<:PriorCarryState}}, x::VecNum; kwargs...)
    pe = Accessors.@reset pe.me = fold_member(pe.me, x; kwargs...)
    pe = Accessors.@reset pe.ce = fold_member(pe.ce, x; kwargs...)
    return Accessors.@reset pe.cache = fold_carry(pe.cache, x; kwargs...)
end
function partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                                         <:Option{<:PriorCarryState}}, X::MatNum;
                      dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    pe = Accessors.@reset pe.me = fold_member(pe.me, X; dims = 1, kwargs...)
    pe = Accessors.@reset pe.ce = fold_member(pe.ce, X; dims = 1, kwargs...)
    return Accessors.@reset pe.cache = fold_carry(pe.cache, X; dims = 1, kwargs...)
end
function partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                                         <:Option{<:PriorCarryState}}, x::VecNum; kwargs...)
    xl = log1p.(x)
    pe = Accessors.@reset pe.me = fold_member(pe.me, xl; kwargs...)
    pe = Accessors.@reset pe.ce = fold_member(pe.ce, xl; kwargs...)
    return Accessors.@reset pe.cache = fold_carry(pe.cache, x; kwargs...)
end
function partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                                         <:Option{<:PriorCarryState}}, X::MatNum;
                      dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    Xl = log1p.(X)
    pe = Accessors.@reset pe.me = fold_member(pe.me, Xl; dims = 1, kwargs...)
    pe = Accessors.@reset pe.ce = fold_member(pe.ce, Xl; dims = 1, kwargs...)
    return Accessors.@reset pe.cache = fold_carry(pe.cache, X; dims = 1, kwargs...)
end
"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                             <:Option{<:PriorCarryState}}; strict::Bool = false, kwargs...)
    prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                             <:Option{<:PriorCarryState}}; strict::Bool = false, kwargs...)

Reads an [`EmpiricalPrior`](@ref) out of its fold, with no data matrix.

`mu` comes off `pe.me`, `sigma` comes off `pe.ce`, and `X` is the matrix the carry state already holds; the rows are read, never refitted. The horizon arm then takes the same four steps of the batch method — scale by `pe.horizon`, exponentiate, build the covariance on the ``\\hat{\\mu}_i + 1`` factors, subtract the one — through [`horizon_moments!`](@ref), which is the one place that algebra lives.

`pe.max_scenarios` cuts the rows the Result carries, exactly as it does in batch, and the fill runs over the window that cut leaves. The fill's notice is narrowed by the carry state's named-asset set, so a walk-forward names an asset at the step it lists and not at every step afterwards.

No [`AssetPanel`](@ref) is read. A panel is fold context rather than sample, and the Coverage Universe of this read-out agrees with a batch fit's **by construction**, because [`coverage_mask`](@ref) is a pure function of the rows and the carry state holds exactly the rows a batch fit would have seen.

# Algorithm

 1. Read the carry state, which refuses an estimator that has folded nothing.
 2. Resolve the fill limit against the arms' coverage floor, as the batch method does.
 3. Read `mu` and `sigma` through [`read_member`](@ref), which refits a member that did not fold over the carried rows.
 4. Under the horizon arm, apply [`horizon_moments!`](@ref).
 5. Cut the carried rows to `pe.max_scenarios` with [`scenario_window`](@ref), and materialise them, because the buffer's own storage moves under the next fold.
 6. Fill and return a [`LowOrderPrior`](@ref).

# Arguments

  - `pe`: Empirical prior estimator carrying a [`PriorCarryState`](@ref).
  - `strict`: Whether a zero-filled scenario raises rather than warns.
  - `kwargs...`: Additional keyword arguments, forwarded to the arms.

# Validation

  - `pe.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, mean vector, and covariance matrix.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`PriorCarryState`](@ref)
  - [`partial_fit!`](@ref)
  - [`scenario_window`](@ref)
  - [`scenario_fill`](@ref)
  - [`horizon_moments!`](@ref)
"""
function prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                                  <:Option{<:PriorCarryState}}; strict::Bool = false,
               kwargs...)
    state = partial_fit_cache(pe)
    fill_limit = resolve_fill_limit(pe.fill_limit, coverage_floor(pe))
    X = sample_buffer(state)
    mu = vec(read_member(Statistics.mean, pe.me, X; kwargs...))
    sigma = read_member(Statistics.cov, pe.ce, X; kwargs...)
    # Materialised, not viewed: the buffer's backing matrix is written and reallocated by the
    # next fold, and a Result holding a view of it would change under its holder.
    Xs = Matrix(scenario_window(pe.max_scenarios, X))
    return LowOrderPrior(;
                         X = scenario_fill(Xs, mu, sigma, strict, fill_limit, state.named),
                         mu = mu, sigma = sigma)
end
function prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                                  <:Option{<:PriorCarryState}}; strict::Bool = false,
               kwargs...)
    state = partial_fit_cache(pe)
    fill_limit = resolve_fill_limit(pe.fill_limit, coverage_floor(pe))
    X = sample_buffer(state)
    # The arms were folded on the log-returns, so a member that did not fold is refitted on
    # them too. The carried rows are the arithmetic ones the caller handed in.
    Xl = log1p.(X)
    mu = vec(read_member(Statistics.mean, pe.me, Xl; kwargs...))
    sigma = read_member(Statistics.cov, pe.ce, Xl; kwargs...)
    horizon_moments!(mu, sigma, pe.horizon)
    Xs = Matrix(scenario_window(pe.max_scenarios, X))
    return LowOrderPrior(;
                         X = scenario_fill(Xs, mu, sigma, strict, fill_limit, state.named),
                         mu = mu, sigma = sigma)
end
"""
    partial_fit!(pe::HighOrderPriorEstimator, X; kwargs...)

Folds observations into a [`HighOrderPriorEstimator`](@ref), by forwarding them to its members.

The host **owns no buffer**. It builds `HighOrderPrior(; pr = pr, ...)` around whatever its embedded prior returned, so the observations it needs at read-out are the ones that prior already carries, and a buffer here would be a second copy of the sample. The fold is therefore a forward and nothing else.

`pe.pe` is forwarded unconditionally, because it is the member that carries the rows: an embedded prior that cannot fold refuses here, naming the wrapper that would give it somewhere to buffer, and that refusal is right — the host has no rows of its own to offer it. `pe.kte` and `pe.ske` go through [`fold_member`](@ref) instead, because the host **does** have rows for them by then, and a co-moment that cannot fold is refitted over `pr.X` at the read-out. So `HighOrderPriorEstimator(; ske = Coskewness(; alg = SemiMoment()))` needs no wrapper at the call site and keeps one copy of the sample.

# Arguments

  - `pe`: High order prior estimator.
  - `X`: Observations to fold, a matrix of rows or a single observation.
  - `kwargs...`: Additional keyword arguments, forwarded to the members.

# Returns

  - `pe`: The estimator, with its members folded.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
  - [`fold_member`](@ref)
  - [`supports_partial_fit`](@ref)
"""
function partial_fit!(pe::HighOrderPriorEstimator, X::VecNum_MatNum; kwargs...)
    # `rebuild_estimator`, not `Accessors.@reset`: this host declares forwarded properties,
    # and `@reset` rebuilds a struct by reading every *property*, which on such a type is not
    # the field list at all.
    return rebuild_estimator(pe,
                             (; pe = partial_fit!(pe.pe, X; kwargs...),
                              kte = fold_member(pe.kte, X; kwargs...),
                              ske = fold_member(pe.ske, X; kwargs...)))
end
"""
    prior(pe::HighOrderPriorEstimator; kwargs...)

Reads a [`HighOrderPriorEstimator`](@ref) out of its fold, with no data matrix.

The embedded prior answers first, and the co-moments are read off their own folds where they folded and refitted over `pr.X` where they did not. [`assemble_high_order_prior`](@ref) then does exactly what it does in batch, so the two routes cannot drift.

A co-moment refitted here reads `pr.X`, which is the matrix the embedded prior carries **after** its own `max_scenarios`. So a Scenario Cap on the embedded prior also caps the rows a non-folding co-moment is fitted over, while a folding one is untouched, because it folded every observation as it arrived. A caller who wants the two windows to agree either leaves the cap unset or gives the co-moments a `FullMoment` algorithm.

# Arguments

  - `pe`: High order prior estimator whose members carry states.
  - `kwargs...`: Additional keyword arguments, forwarded to the members.

# Returns

  - `hop::HighOrderPrior`: Result object carrying the low order result and the co-moments.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`assemble_high_order_prior`](@ref)
  - [`read_member`](@ref)
  - [`partial_fit!`](@ref)
"""
function prior(pe::HighOrderPriorEstimator; kwargs...)
    pr = prior(pe.pe; kwargs...)
    kt = read_member(cokurtosis, pe.kte, pr.X; kwargs...)
    sk, V = read_member(coskewness, pe.ske, pr.X; kwargs...)
    return assemble_high_order_prior(pe, pr, kt, sk, V)
end
"""
    partial_fit!(pe::BlackLittermanPrior, X; kwargs...)

Folds observations into a [`BlackLittermanPrior`](@ref), by forwarding them to its embedded prior.

The host owns no buffer, for the reason [`HighOrderPriorEstimator`](@ref) owns none: it returns [`forward_prior`](@ref) of the result its embedded prior answered, so the rows it needs are already carried one level down. The views are configuration and fold nothing.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Observations to fold, a matrix of rows or a single observation.
  - `kwargs...`: Additional keyword arguments, forwarded to the embedded prior.

# Returns

  - `pe`: The estimator, with its embedded prior folded.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`prior`](@ref)
  - [`bl_posterior`](@ref)
"""
function partial_fit!(pe::BlackLittermanPrior, X::VecNum_MatNum; kwargs...)
    return rebuild_estimator(pe, (; pe = partial_fit!(pe.pe, X; kwargs...)))
end
"""
    prior(pe::BlackLittermanPrior; strict::Bool = false, kwargs...)

Reads a [`BlackLittermanPrior`](@ref) out of its fold, with no data matrix.

The embedded prior answers first, and [`bl_posterior`](@ref) is the batch body unchanged: every number it reads — the number of observations behind `tau`, the asset axis, the matrix the processing runs over — comes off the result rather than off a returns matrix, so the fold and the batch fit reach one body.

# Arguments

  - `pe`: Black-Litterman prior estimator whose embedded prior carries a state.
  - `strict`: Whether an unresolved view raises rather than warns.
  - `kwargs...`: Additional keyword arguments, forwarded to the embedded prior and the processing.

# Returns

  - `pr::AbstractPriorResult`: The embedded result with its two moments replaced by the posteriors.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`bl_posterior`](@ref)
  - [`assert_bl_views_axis`](@ref)
  - [`partial_fit!`](@ref)
"""
function prior(pe::BlackLittermanPrior; strict::Bool = false, kwargs...)
    prior_model = prior(pe.pe; strict = strict, kwargs...)
    assert_bl_views_axis(pe, prior_model.X)
    return bl_posterior(pe, prior_model, strict; kwargs...)
end
"""
    online_state_seed(pe::FactorPrior, max_history)
    online_state_seed(pe::BayesianBlackLittermanPrior, max_history)
    online_state_seed(pe::FactorBlackLittermanPrior, max_history)
    online_state_seed(pe::AugmentedBlackLittermanPrior, max_history)
    online_state_seed(pe::HighOrderFactorPriorEstimator, max_history)

Seeds a [`FactorSampleBufferState`](@ref) for a prior whose batch verb needs a factor matrix.

Five priors take `prior(pe, X, F)` with the factor matrix **required**, because they regress the asset returns on the factor returns and there is no fit without both. One buffer cannot keep two sequences, so [`Online`](@ref) seeds the pair for them, and the wrapper's cap reaches both halves because it windows the whole fit.

Every other prior keeps the single-buffer seed. A prior whose factor matrix is *optional* — the entropy pooling family and [`OpinionPoolingPrior`](@ref) — buffers its returns alone, and its read-out runs the batch verb with no factor matrix, which is the configuration those estimators reach by default in batch too.

# Arguments

  - `pe`: The prior the wrapper wraps, read for its type.
  - `max_history`: The wrapper's cap, carried into both halves.

# Returns

  - `state::FactorSampleBufferState`: The empty pair to seed.

# Related

  - [`online_state_seed`](@ref)
  - [`FactorSampleBufferState`](@ref)
  - [`Online`](@ref)
"""
function online_state_seed(::Union{<:FactorPrior, <:BayesianBlackLittermanPrior,
                                   <:FactorBlackLittermanPrior,
                                   <:AugmentedBlackLittermanPrior,
                                   <:HighOrderFactorPriorEstimator},
                           max_history::Option{<:Integer})
    return FactorSampleBufferState(; X = SampleBufferState(; max_history = max_history),
                                   F = SampleBufferState(; max_history = max_history))
end
