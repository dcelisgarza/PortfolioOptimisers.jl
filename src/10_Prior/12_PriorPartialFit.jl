"""
$(DocStringExtensions.TYPEDEF)

Carries the observations that a fold-and-carry prior keeps, and the assets whose scenario fill it has already named.

A prior that carries this state folds its moments exactly, member by member, and keeps the rows for one reason. [`LowOrderPrior`](@ref) carries `X` for the scenario risk measures, so the read-out copies the rows into the result and computes nothing from them. A [`SampleBufferState`](@ref) is different. An estimator that carries one has no recursion of its own, and its read-out runs the batch verb over the rows that the buffer kept.

An [`Online`](@ref) wrapper seeds a [`SampleBufferState`](@ref), so a wrapped prior takes the refit route and not this one. This is why the `max_history` of an `Online` windows the whole fit.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriorCarryState(;
        buf::SampleBufferState = SampleBufferState(),
        named::Set{Int} = Set{Int}()
    ) -> PriorCarryState

Keywords correspond to the struct's fields. The default is the empty state that a first fold builds.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `buf`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `named`: Renumbered onto the selected indices, so an asset named before the slice is still named after it.

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
    Buffer of the observations that the prior carries, `observations × assets`, held as given, `NaN` entries included.
    """
    buf
    """
    Indices of the assets whose scenario fill a read-out has already named. [`strict_diagnostic`](@ref) keeps no record, so without this set a walk-forward that reads out at every step names the same asset at every step. A read-out adds to the set in place, because it returns a Prior Result and not the estimator, and no other return value takes the set to the next step. [`Base.copy`](@ref) copies the set, and [`merge_states`](@ref) takes the union of two.
    """
    named
end
function PriorCarryState(; buf::SampleBufferState = SampleBufferState(),
                         named::Set{Int} = Set{Int}())::PriorCarryState
    return PriorCarryState(buf, named)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the observations that a [`PriorCarryState`](@ref) carries, `observations × assets`.

# Arguments

  - `state`: The carry state to read.

# Returns

  - `X::SubArray`: The observations that the state carries, in the order of the fold.

# Related

  - [`PriorCarryState`](@ref)
  - [`sample_buffer`](@ref)
"""
function sample_buffer(state::PriorCarryState)
    return sample_buffer(state.buf)
end
"""
    returns_buffer(state::SampleBufferState)
    returns_buffer(state::PriorCarryState)
    returns_buffer(state::AbstractPartialFitState)

Reads the buffer of asset returns out of the state that a prior carries.

Two states hold rows at the prior layer, each in a different place. A [`SampleBufferState`](@ref) is the rows, and a [`PriorCarryState`](@ref) keeps them in `buf`. The read-out of an optimiser and the read-out of a forwarding host read the returns through this function and never through the fields of a state, so a new state that holds rows adds one method here. Where the tree of the prior reads factor returns, the buffer holds the factor rows too, and the read-out of an optimiser takes them with [`factor_buffer`](@ref) when the fold context keeps no factor column. This function refuses, by name, a state that holds no rows, such as the exact-fold state of a moment estimator.

# Arguments

  - `state`: The state that the prior carries.

# Validation

  - `state` holds rows. An `ArgumentError` is thrown otherwise.

# Returns

  - `buffer::SampleBufferState`: The rows, the masks folded beside them, and the factor rows where the prior records them.

# Related

  - [`prior_returns_buffer`](@ref)
  - [`SampleBufferState`](@ref)
  - [`PriorCarryState`](@ref)
  - [`factor_buffer`](@ref)
"""
function returns_buffer(state::SampleBufferState)
    return state
end
function returns_buffer(state::PriorCarryState)
    return state.buf
end
function returns_buffer(state::AbstractPartialFitState)
    return throw(ArgumentError("a `$(typeof(state))` carries no rows, so no read-out can rebuild the returns it folded from it. The read-out reads the observations from the prior's own buffer, which a `SampleBufferState` or a `PriorCarryState` holds."))
end
"""
    prior_returns_buffer(pe::AbstractPriorEstimator)
    prior_returns_buffer(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})

Reads the buffer of asset returns out of the prior that owns the folded rows.

One prior in the chain owns the rows. [`HighOrderPriorEstimator`](@ref) and [`BlackLittermanPrior`](@ref) build their result around the prior they embed and own no rows, so this function reads the buffer of the embedded prior, at any depth. Every other prior keeps its rows in its own `cache`. Two read-outs call it. The read-out of an optimiser rebuilds the returns it forwarded, and the read-out of a [`HighOrderPriorEstimator`](@ref) refits a co-moment that does not fold over the rows.

# Arguments

  - `pe`: The prior estimator that received the observations.

# Validation

  - The prior that owns the rows carries a state. [`partial_fit_cache`](@ref) throws an `ArgumentError` otherwise.
  - Everything [`returns_buffer`](@ref) refuses.

# Returns

  - `buffer::SampleBufferState`: The rows that the prior folded, and the masks beside them.

# Related

  - [`returns_buffer`](@ref)
  - [`partial_fit_cache`](@ref)
  - [`returns_result`](@ref)
  - [`sample_buffer`](@ref)
"""
function prior_returns_buffer(pe::AbstractPriorEstimator)
    return returns_buffer(partial_fit_cache(pe))
end
function prior_returns_buffer(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})
    return prior_returns_buffer(pe.pe)
end
"""
    partial_fit!(state::PriorCarryState, x::VecNum; kwargs...)
    partial_fit!(state::PriorCarryState, X::MatNum; dims::Int = 1, kwargs...)

Folds observations into the buffer that a [`PriorCarryState`](@ref) carries.

The method forwards to the fold of the buffer, with its keyword arguments. So a [`CoveragePolicy`](@ref) mask goes into the buffer beside the rows it describes, as it does for a [`SampleBufferState`](@ref) that an [`Online`](@ref) seeded. No factor observation reaches the buffer, because [`EmpiricalPrior`](@ref), the one carrying prior, never reads one and drops it before the carry. The fold leaves the named-asset set as it is, because only a read-out finds an asset to name.

# Arguments

  - `state`: The carry state to fold into.
  - `x`: One observation, with one entry per asset.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, such as `active_mask`, forwarded to the fold of the buffer.

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
    fold_carry(cache::Option{<:PriorCarryState}, x::VecNum; kwargs...)
    fold_carry(cache::Option{<:PriorCarryState}, X::MatNum; dims::Int = 1, kwargs...)

Folds observations into a [`PriorCarryState`](@ref), and seeds an empty state where the prior carries none.

It is [`fold_buffer`](@ref) with the named-asset set kept. The seed has no cap. The cap on the carried rows is the `max_scenarios` of an `EmpiricalPrior`, which the read-out applies. A cap on the fit is the `max_history` of an [`Online`](@ref), which puts the prior on the refit route instead.

# Arguments

  - `cache`: The state that the prior carries, or `nothing`.
  - `x`: One observation, with one entry per asset.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, forwarded to the fold of the buffer.

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

Merges two [`PriorCarryState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

The buffers merge by concatenation. The named-asset sets merge by union, because an asset that either state has already named needs no second notice.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Returns

  - `state::PriorCarryState`: The state that a fold of the two blocks as one block gives.

# Related

  - [`PriorCarryState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(a::PriorCarryState, b::PriorCarryState)
    return PriorCarryState(merge_states(a.buf, b.buf), union(a.named, b.named))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`PriorCarryState`](@ref), so that the copy shares no array and no set with the original.

The method copies the named-asset set for the reason it copies the backing matrix. A read-out writes into the set, so a copy that shared it would record its notices in the set of the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::PriorCarryState`: A new state, equal to `x`.

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

The method slices the buffer as usual. The entries of the named-asset set are asset indices, and the slice renumbers the assets, so the method renumbers the set too. An asset that the state had already named stays named at its new index, and an asset that the slice removes leaves the set.

# Algorithm

 1. Index the asset axis `1:N` with `i`, giving `sel`, the old index of each asset that the slice keeps.
 2. Keep each new index `j` whose old index `sel[j]` is in `x.named`, giving `named`.
 3. Slice the buffer with [`port_opt_view`](@ref), and return the new state.

# Arguments

  - `x`: The state to slice.
  - `i`: Index, indices or `Bool` mask of the assets to keep.
  - `args...`: Additional positional arguments, forwarded to the slice of the buffer.

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

Reads a prior out of the state that its estimator carries, with no data matrix.

This is the read-out method of [`prior`](@ref), and the entry point of every prior that folds. It reads the state and dispatches on its type:

  - A [`SampleBufferState`](@ref) means refit. The estimator has no recursion of its own, so the read-out runs the batch verb over the rows that the buffer kept. These are all the rows folded so far when the buffer has no cap, and the last `max_history` rows when an [`Online`](@ref) set one. The batch verb receives the factor rows that the buffer recorded, through [`factor_buffer`](@ref), or `nothing` where the buffer recorded none. The fold fixed that choice from the estimator tree.
  - A [`PriorCarryState`](@ref) means fold and carry. Each family that takes it has a read-out method of its own.

# Mathematical definition

Under a [`SampleBufferState`](@ref):

```math
\\begin{align}
\\mathcal{P}_T &= \\mathcal{P}(\\mathbf{X}_{T-m+1:T},\\, \\mathbf{F}_{T-m+1:T})\\,, \\\\
m &= \\min(T,\\, M)\\,.
\\end{align}
```

Where:

  - $(math_dict[:P_fold_prior])
  - $(math_dict[:P_batch_prior])
  - ``\\mathbf{X}_{a:b}``: Rows ``a`` to ``b`` of the returns matrix of the folded observations.
  - ``\\mathbf{F}_{a:b}``: Rows ``a`` to ``b`` of the factor returns matrix of the folded factor observations, absent when the fold recorded none.
  - ``m``: Number of rows that the buffer holds.
  - ``M``: Cap on the buffer, the `max_history` of the wrapping [`Online`](@ref), and ``\\infty`` when it is `nothing`.
  - $(math_dict[:T])

# Algorithm

 1. Read the state out of `pe.cache` with [`partial_fit_cache`](@ref).
 2. Dispatch on the type of the state. Under a [`SampleBufferState`](@ref), run the batch verb over the [`sample_buffer`](@ref) and the [`factor_buffer`](@ref) of the state, with the masks that [`sample_buffer_kwargs`](@ref) reads and the caller's keyword arguments.

# Arguments

  - `pe`: Prior estimator carrying a state.
  - `kwargs...`: Additional keyword arguments, forwarded to the batch verb.

# Validation

  - `pe.cache` is not `nothing`. An `ArgumentError` that names the verb which fills the field is thrown otherwise.

# Returns

  - `pr::AbstractPriorResult`: The prior that the state of the estimator gives.

# Related

  - [`prior`](@ref)
  - [`SampleBufferState`](@ref)
  - [`PriorCarryState`](@ref)
  - [`Online`](@ref)
  - [`partial_fit_cache`](@ref)
  - [`factor_buffer`](@ref)
"""
function prior(pe::AbstractPriorEstimator; kwargs...)
    return prior(pe, partial_fit_cache(pe); kwargs...)
end
function prior(pe::AbstractPriorEstimator, state::SampleBufferState; kwargs...)
    return prior(pe, sample_buffer(state), factor_buffer(state);
                 sample_buffer_kwargs(state)..., kwargs...)
end
"""
    needs_factor_returns(pe::AbstractHiLoOrderPriorEstimator_F) -> true
    needs_factor_returns(pe::AbstractLowOrderPriorEstimator_A) -> false
    needs_factor_returns(pe::AbstractLowOrderPriorEstimator_AF) -> nothing
    needs_factor_returns(pe::Online)
    needs_factor_returns(pe::HighOrderPriorEstimator)
    needs_factor_returns(pe::BlackLittermanPrior)
    needs_factor_returns(pe::EntropyPoolingPrior)
    needs_factor_returns(pe::MeucciEntropyPoolingPrior)
    needs_factor_returns(pe::OpinionPoolingPrior)

Answers whether the fit of a prior reads a factor matrix, from its estimator tree.

The answer depends on the tree that an estimator embeds, not on the type of the host. A prior whose factor argument is optional never reads the argument, and passes it to the prior it embeds. Only a member that requires the argument reads it. So the answer has three values, and the abstract type of the member gives each one:

  - `true` for a member that requires factor returns, [`AbstractHiLoOrderPriorEstimator_F`](@ref). Its batch verb declares `F::MatNum` with no default, and refuses a fit without one.
  - `false` for a member that never reads them, [`AbstractLowOrderPriorEstimator_A`](@ref). Its batch verb declares the argument and ignores it, so a fold drops it as the batch verb does.
  - `nothing` for a member whose factor argument is optional, [`AbstractLowOrderPriorEstimator_AF`](@ref). The type does not say, so the fold takes what it receives, as the batch verb does.

Each optional-argument host of the library recurses into the prior it embeds and answers the value of the leaf. So `EntropyPoolingPrior(; pe = FactorPrior())` answers `true`, and `EntropyPoolingPrior()` answers `false`. [`OpinionPoolingPrior`](@ref) holds several priors and combines their answers with [`combine_factor_answers`](@ref). An [`Online`](@ref) answers for the estimator it wraps. A caller's own optional-argument subtype that embeds a prior must define the recursion. A subtype that reads `F` itself can keep the default, which takes what it receives.

The doors that check for a missing factor matrix read this predicate. These are the [`ReturnsResult`](@ref) method of the prior, the step of the online optimiser, and the three uncertainty-set doors. The predicate finds a factor leaf under an optional-argument host, which an `isa` test on the host cannot find. It also decides what the refit route of [`partial_fit!`](@ref) does with the factor observation it receives.

# Arguments

  - `pe`: The prior estimator, or the wrapper around one.

# Returns

  - `needs::Union{Bool, Nothing}`: `true`, `false` or `nothing`, as the list above states.

# Related

  - [`AbstractHiLoOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`combine_factor_answers`](@ref)
  - [`prior`](@ref)
  - [`partial_fit!`](@ref)
  - [`SampleBufferState`](@ref)
"""
function needs_factor_returns(::AbstractHiLoOrderPriorEstimator_F)
    return true
end
function needs_factor_returns(::AbstractLowOrderPriorEstimator_A)
    return false
end
function needs_factor_returns(::AbstractLowOrderPriorEstimator_AF)
    return nothing
end
function needs_factor_returns(pe::Online)
    return needs_factor_returns(pe.est)
end
function needs_factor_returns(pe::HighOrderPriorEstimator)
    # The factor matrix is handed to the embedded prior and never read here, so the tree
    # answers (see [`needs_factor_returns`](@ref)).
    return needs_factor_returns(pe.pe)
end
function needs_factor_returns(pe::BlackLittermanPrior)
    # The factor matrix is handed to the embedded prior and never read here, so the tree
    # answers (see [`needs_factor_returns`](@ref)).
    return needs_factor_returns(pe.pe)
end
function needs_factor_returns(pe::MeucciEntropyPoolingPrior)
    # The factor matrix is handed to the embedded prior through `ep_prior` and never read
    # here, so the tree answers (see [`needs_factor_returns`](@ref)).
    return needs_factor_returns(pe.pe)
end
function needs_factor_returns(pe::EntropyPoolingPrior)
    # The factor matrix is handed to the embedded prior through `ep_prior` and never read
    # here, so the tree answers (see [`needs_factor_returns`](@ref)).
    return needs_factor_returns(pe.pe)
end
function needs_factor_returns(pe::OpinionPoolingPrior)
    # The factor matrix is handed to `pe.pe1`, to every pooled `pe.pes` and to `pe.pe2`, and
    # never read here, so the members answer together: `true` when any requires it, `false`
    # when none reads it, and `nothing` otherwise (see [`needs_factor_returns`](@ref)).
    answers = (needs_factor_returns(pe.pe2), (needs_factor_returns(p) for p in pe.pes)...)
    if !isnothing(pe.pe1)
        answers = (answers..., needs_factor_returns(pe.pe1))
    end
    return combine_factor_answers(answers)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Combines the answers of several embedded priors into one, for a host that holds more than one.

The rule is the strong three-valued disjunction of Kleene, with `nothing` as the unknown value. A fit that reaches a member which requires factor returns is refused without them, so one `true` decides the answer. A fit whose members all drop factor returns drops them too. In every other case at least one member takes what it receives and no member requires it, so the answer is `nothing`.

# Mathematical definition

```math
\\begin{align}
c(a_1, \\ldots, a_K) &= \\begin{cases}
\\top & \\text{if } a_k = \\top \\text{ for some } k\\,, \\\\
\\bot & \\text{if } a_k = \\bot \\text{ for every } k\\,, \\\\
\\mathrm{U} & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``c``: Combined answer.
  - ``a_k``: Answer of member ``k``, which is ``\\top`` for `true`, ``\\bot`` for `false` and ``\\mathrm{U}`` for `nothing`.
  - ``K``: Number of members.

The combination is commutative and associative, so neither the order nor the grouping of the members changes the answer.

# Arguments

  - `answers`: The answers of the members, each `true`, `false` or `nothing`.

# Returns

  - `needs::Union{Bool, Nothing}`: The combined answer.

# Related

  - [`needs_factor_returns`](@ref)
  - [`OpinionPoolingPrior`](@ref)
"""
function combine_factor_answers(answers)
    return if any(a -> a === true, answers)
        true
    elseif all(a -> a === false, answers)
        false
    else
        nothing
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a missing factor matrix at a door whose prior tree requires one.

Five doors share this refusal: the [`ReturnsResult`](@ref) method of the prior, the step of the online optimiser, and the three uncertainty-set doors. The refit route of [`partial_fit!`](@ref) reaches it too, through [`fold_factor_argument`](@ref). One method holds the message and the test, so the two cannot differ between callers. The test is [`needs_factor_returns`](@ref) answering `true`. That predicate walks the estimator tree, so this method refuses a factor leaf under an optional-argument host by name, before the leaf meets a `MethodError` one call later.

A `pe` of `nothing` is an uncertainty set with no prior of its own. It reads no factor matrix, so the method checks nothing. The returns-data method refuses such a set later, by name, through [`ucs_prior`](@ref).

# Arguments

  - `pe`: The prior estimator that the door passes the matrix to, or `nothing`.
  - `F`: The factor matrix that the carrier holds, the factor observation that a fold receives, or `nothing`.

# Validation

  - `!isnothing(F)`, when `needs_factor_returns(pe) === true`. An `IsNothingError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`needs_factor_returns`](@ref)
  - [`prior`](@ref)
  - [`ReturnsResult`](@ref)
  - [`ucs_prior`](@ref)
"""
function assert_factor_returns(pe::Union{<:AbstractPriorEstimator, <:Online},
                               F::Option{<:VecNum_MatNum})::Nothing
    if needs_factor_returns(pe) === true
        @argcheck(!isnothing(F),
                  IsNothingError("the estimator tree of this prior holds a factor prior, which needs factor returns, but `F` is `nothing`. Set `ReturnsResult.F`, for example with `prices_to_returns` on factor prices, or pass `F` to `partial_fit!`."))
    end
    return nothing
end
function assert_factor_returns(::Nothing, ::Option{<:VecNum_MatNum})::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds observations, and the factor observations beside them, into the sample buffer that a prior carries.

This is the refit route of the prior family. Every prior that carries a [`SampleBufferState`](@ref) reaches this method, which has the arity of the prior's own batch verb, `prior(pe, X, F)`. The estimator tree decides what the fold does with `F`, through [`needs_factor_returns`](@ref), and the fold does what the batch verb does with the same argument:

  - `true`. The tree holds a factor leaf. The fold refuses a call without `F` by name, with the refusal of the doors, before it appends a row. A call with `F` records it.
  - `false`. The tree never reads `F`, so the fold drops it, as `prior(EmpiricalPrior(), X, F)` drops it, and the buffer records the rows alone.
  - `nothing`. The tree does not say, so the buffer records `F` when the call gives it and not otherwise, as the batch verb of an optional-argument prior does.

The buffer fixes what it records at its first append and refuses a mixture. So a run that gives `F` at one step and not at the next is refused at that step, by name.

# Algorithm

 1. Answer [`needs_factor_returns`](@ref) for the tree, and apply the answer to `F` with [`fold_factor_argument`](@ref), giving the `F` that the buffer records.
 2. Read the buffer out of `pe.cache` with [`assert_sample_buffer`](@ref), giving `state`.
 3. Fold a matrix, its factor block and its masks into `state` through the block method of [`partial_fit!`](@ref), or a vector, its factor observation and its masks through the single-observation method.
 4. Rebuild the prior with its `cache` set to the new `state`, and return it.

# Arguments

  - `pe`: The prior whose buffer the method folds forward.
  - `X`: Observations to fold. A matrix holds one observation per row when `dims == 1`, and one per column when `dims == 2`. A vector is one observation across the assets, and `dims` then has no effect.
  - `F`: The factor observations, in the orientation of `X`, or `nothing`.
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the block, of the shape of `X`, or with one entry per asset when `X` is one observation, or `nothing`.
  - `estimation_mask`: The estimation mask, on the same terms as `active_mask`.

# Validation

  - `F` is not `nothing` when `needs_factor_returns(pe) === true`. An `IsNothingError` is thrown otherwise.
  - `pe` carries a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.
  - Every condition that the fold of the buffer checks.

# Returns

  - `pe`: The prior, with its `cache` field set to the buffer after the last observation.

# Related

  - [`needs_factor_returns`](@ref)
  - [`fold_factor_argument`](@ref)
  - [`SampleBufferState`](@ref)
  - [`factor_buffer`](@ref)
  - [`Online`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(pe::AbstractPriorEstimator, X::VecNum_MatNum,
                      F::Option{<:VecNum_MatNum} = nothing; dims::Int = 1,
                      active_mask = nothing, estimation_mask = nothing)
    F = fold_factor_argument(needs_factor_returns(pe), pe, F)
    state = assert_sample_buffer(pe)
    state = if isa(X, MatNum)
        partial_fit!(state, X, F; dims = dims, active_mask = active_mask,
                     estimation_mask = estimation_mask)
    else
        partial_fit!(state, X, F; active_mask = active_mask,
                     estimation_mask = estimation_mask)
    end
    # `rebuild_estimator`, not `Accessors.@reset`, for the reason the generic buffering arm
    # gives: every prior that buffers declares forwarded properties, which `@reset` refuses.
    return rebuild_estimator(pe, (; cache = state))
end
"""
    fold_factor_argument(needs::Bool, pe::AbstractPriorEstimator, F::Option{<:VecNum_MatNum})
    fold_factor_argument(::Nothing, ::AbstractPriorEstimator, F::Option{<:VecNum_MatNum})

Applies the answer of the estimator tree to the factor argument of a fold.

The method reads the three answers of [`needs_factor_returns`](@ref) as the refit route needs them. `true` refuses a missing `F` by name and passes a present one through. `false` drops `F`. `nothing` passes `F` through as it is. The method dispatches on the answer, so a tree whose answer follows from its type, which is true of every tree in the library, costs the fold no branch at run time.

# Arguments

  - `needs`: The answer of [`needs_factor_returns`](@ref).
  - `pe`: The prior, for the message of the refusal.
  - `F`: The factor argument of the fold, or `nothing`.

# Validation

  - `F` is not `nothing` when `needs` is `true`. An `IsNothingError` is thrown otherwise.

# Returns

  - `F`: The factor argument that the buffer records, or `nothing`.

# Related

  - [`needs_factor_returns`](@ref)
  - [`assert_factor_returns`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_factor_argument(needs::Bool, pe::AbstractPriorEstimator,
                              F::Option{<:VecNum_MatNum})
    if needs
        assert_factor_returns(pe, F)
        return F
    end
    return nothing
end
function fold_factor_argument(::Nothing, ::AbstractPriorEstimator,
                              F::Option{<:VecNum_MatNum})
    return F
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds an observation into a member of a carrying host, where that member folds.

This is the fold half of the rule for a mixed host. A host that carries the observations folds every member that folds, and leaves the other members as they are, because it can run their batch verb over its own rows at the read-out. [`supports_partial_fit`](@ref) answers the question from the type of the member and its `cache` field. So for a host of concrete members the compiler resolves the branch, and the host folds no member that must not fold.

A member that the host does not hold is `nothing`, and a fold of it returns `nothing`.

# Arguments

  - `est`: The member to fold, or `nothing`.
  - `args...`: The observations, forwarded to [`partial_fit!`](@ref).
  - `kwargs...`: Additional keyword arguments, forwarded to [`partial_fit!`](@ref).

# Returns

  - `est`: The member with the state after the last observation, or the member unchanged where it does not fold.

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

Reads the estimate of a member out of its fold, or refits the member over the rows of the host.

This is the read-out half of the rule for a mixed host, and [`fold_member`](@ref) is the fold half. A member that the host folded answers from its state. A member that the host did not fold answers from the matrix that the host passes. `f` is the batch verb of the member, for example `Statistics.mean`, `Statistics.cov`, [`coskewness`](@ref) or [`cokurtosis`](@ref). The one-argument method of the same verb is its read-out, and every family that folds follows this convention.

No [`AssetPanel`](@ref) reaches this verb. A panel describes the fold and is not a sample, and a buffer holds no activity mask, so the verb refits a member over the rows alone.

# Arguments

  - `f`: The batch verb of the member.
  - `est`: The member, or `nothing`.
  - `X`: The rows that the host carries, which are the rows the member was folded on.
  - `kwargs...`: Additional keyword arguments, forwarded to the batch verb alone. A folded member is read with no keyword argument, because its state is already the estimate.

# Returns

  - `val`: The estimate of the member, in the shape that its verb returns.

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
                                    <:Option{<:PriorCarryState}}, X, F = nothing; kwargs...)
    partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                                    <:Option{<:PriorCarryState}}, X, F = nothing; kwargs...)

Folds observations into an [`EmpiricalPrior`](@ref), which folds its moments and carries its rows.

`pe.me` and `pe.ce` fold exactly, so a step of the prior is their two steps and an append to the carried rows. The cost of a step grows with the square of the number of assets, and does not grow with the number of observations already folded. The cost of a refit from a buffer grows with that number too. This difference is the reason the carry route exists. A read-out written as `prior(pe, sample_buffer(pe))` passes every parity test and discards both exact folds, so a review must refuse that form.

The two methods differ in one line. The method with no horizon folds the observation as it is. The horizon method folds `log1p` of the observation, because the batch horizon method fits its moments on log returns. It carries the observation itself, because the [`LowOrderPrior`](@ref) it returns carries the arithmetic returns that the caller passed. A buffer of log rows would change the matrix that every scenario risk measure reads.

A member that does not fold, such as a `SemiMoment` covariance or a composite whose `mp.alg` reads the sample, stays as it is under [`fold_member`](@ref), and the read-out refits it over the carried rows. So a caller writes the estimator that they would write in batch, and no member keeps a second copy of the sample.

The fold takes a factor observation and drops it, because `prior(EmpiricalPrior(), X, F)` declares `F` and never reads it. The fold has the arity of the batch verb, so a host that passes `F` down its tree meets no `MethodError` here. [`needs_factor_returns`](@ref) answers `false` for this estimator, so no estimator above it keeps a factor row for it.

# Algorithm

 1. Orient a matrix `X` to `observations × assets` with `dims_oriented`, transposing it when `dims == 2`.
 2. Under the horizon method, take `log1p` of the observations, giving `xl` or `Xl`.
 3. Fold `pe.me` and `pe.ce` through [`fold_member`](@ref), on the observations of step 2 under the horizon method and on the observations as they are otherwise.
 4. Append the arithmetic observations to the carry state through [`fold_carry`](@ref), and return the estimator with its members and its `cache` field rebound.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Observations to fold. A matrix holds one observation per row when `dims == 1`, and one per column when `dims == 2`. A vector is one observation across the assets.
  - `F`: The factor observations beside `X`, or `nothing`. The fold drops them, as the batch verb does.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, forwarded to the two members and to the carry state.

# Validation

  - $(val_dict[:dims])
  - Every condition that the fold of `pe.me` or `pe.ce` checks. A member that holds observation weights refuses the fold by name, because a weight vector reweights every past observation when a new one arrives.

# Returns

  - `pe`: The estimator, with its members folded and its `cache` field rebound.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`PriorCarryState`](@ref)
  - [`prior`](@ref)
  - [`fold_member`](@ref)
  - [`fold_carry`](@ref)
"""
function partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                                         <:Option{<:PriorCarryState}}, x::VecNum,
                      ::Option{<:VecNum_MatNum} = nothing; kwargs...)
    pe = Accessors.@reset pe.me = fold_member(pe.me, x; kwargs...)
    pe = Accessors.@reset pe.ce = fold_member(pe.ce, x; kwargs...)
    return Accessors.@reset pe.cache = fold_carry(pe.cache, x; kwargs...)
end
function partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                                         <:Option{<:PriorCarryState}}, X::MatNum,
                      ::Option{<:VecNum_MatNum} = nothing; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    pe = Accessors.@reset pe.me = fold_member(pe.me, X; dims = 1, kwargs...)
    pe = Accessors.@reset pe.ce = fold_member(pe.ce, X; dims = 1, kwargs...)
    return Accessors.@reset pe.cache = fold_carry(pe.cache, X; dims = 1, kwargs...)
end
function partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                                         <:Option{<:PriorCarryState}}, x::VecNum,
                      ::Option{<:VecNum_MatNum} = nothing; kwargs...)
    xl = log1p.(x)
    pe = Accessors.@reset pe.me = fold_member(pe.me, xl; kwargs...)
    pe = Accessors.@reset pe.ce = fold_member(pe.ce, xl; kwargs...)
    return Accessors.@reset pe.cache = fold_carry(pe.cache, x; kwargs...)
end
function partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                                         <:Option{<:PriorCarryState}}, X::MatNum,
                      ::Option{<:VecNum_MatNum} = nothing; dims::Int = 1, kwargs...)
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

`mu` comes from `pe.me`, `sigma` comes from `pe.ce`, and `X` is the matrix that the carry state already holds. The read-out copies the rows, and does not refit a member that folded. The horizon method then applies the algebra of the batch method through [`horizon_moments!`](@ref), which holds that algebra once.

`pe.max_scenarios` cuts the rows that the result carries, as it does in batch, and the fill runs over the rows that remain. When the cap cuts, `ens` holds the number of observations folded, through [`scenario_ens`](@ref), so a consumer that prices a sample size reads ``T`` and not the number of rows carried. The named-asset set of the carry state limits the notice of the fill, so a walk-forward names an asset at the first step whose fill reaches it, and at no later step.

No [`AssetPanel`](@ref) is read. A panel describes the fold and is not a sample. The Coverage Universe of this read-out equals the Coverage Universe of a batch fit, because [`coverage_mask`](@ref) is a function of the rows alone, and the carry state holds the rows that a batch fit reads.

# Mathematical definition

```math
\\begin{align}
\\mathcal{P}_T &= \\mathcal{P}(\\mathbf{X})\\,, \\\\
\\mathbf{X} &= \\begin{bmatrix} \\boldsymbol{x}_1 & \\cdots & \\boldsymbol{x}_T \\end{bmatrix}^\\intercal\\,.
\\end{align}
```

Where:

  - $(math_dict[:P_fold_prior])
  - $(math_dict[:P_batch_prior])
  - $(math_dict[:X_returns])
  - $(math_dict[:x_t_obs])
  - $(math_dict[:T])

The equality holds for `mu`, `sigma`, `X` and `ens`, with and without a horizon, and under a Scenario Cap. In floating point, the moments of a member that folds differ from the batch moments by rounding, and every other entry is equal.

# Algorithm

 1. Read the carry state with [`partial_fit_cache`](@ref), giving `state`.
 2. Resolve the fill limit against the coverage floor of the members with [`resolve_fill_limit`](@ref), as the batch method does, giving `fill_limit`.
 3. Read the carried rows with [`sample_buffer`](@ref), giving `X`. Under the horizon method, take `log1p` of the rows, giving `Xl`.
 4. Read `mu` and `sigma` through [`read_member`](@ref), which refits a member that did not fold over `X`, or over `Xl` under the horizon method.
 5. Under the horizon method, convert `mu` and `sigma` with [`horizon_moments!`](@ref).
 6. Cut the carried rows to `pe.max_scenarios` with [`scenario_window`](@ref), and copy them into a new matrix `Xs`, because the next fold writes into the storage of the buffer.
 7. Fill `Xs` with [`scenario_fill`](@ref), and return a [`LowOrderPrior`](@ref) whose `ens` [`scenario_ens`](@ref) reads from the rows before the cut.

# Arguments

  - `pe`: Empirical prior estimator carrying a [`PriorCarryState`](@ref).
  - `strict`: Whether a zero-filled scenario raises rather than warns.
  - `kwargs...`: Additional keyword arguments, forwarded to the batch verb of a member that did not fold.

# Validation

  - `pe.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.
  - Under `strict`, no investable column needs a fill. An `ArgumentError` is thrown otherwise.

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
                         mu = mu, sigma = sigma, ens = scenario_ens(pe.max_scenarios, X))
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
                         mu = mu, sigma = sigma, ens = scenario_ens(pe.max_scenarios, X))
end
"""
    partial_fit!(pe::HighOrderPriorEstimator, X, F = nothing; kwargs...)

Folds observations into a [`HighOrderPriorEstimator`](@ref) by forwarding them to its members.

The host keeps no buffer. It builds `HighOrderPrior(; pr = pr, ...)` around the result of its embedded prior, so the rows it needs at the read-out are the rows that prior carries. A buffer here would be a second copy of the sample.

The host forwards the fold to `pe.pe` in every case, because that member carries the rows. An embedded prior that cannot fold refuses here, and names the wrapper that gives it a buffer. The refusal is correct, because the host has no rows of its own to give it. The factor observations go with the rows, as the batch verb passes `F` to the embedded prior, and the tree of the embedded prior decides what happens to them. `pe.kte` and `pe.ske` go through [`fold_member`](@ref) instead, because the host has rows for them at the read-out, and the read-out refits a co-moment that cannot fold. So `HighOrderPriorEstimator(; ske = Coskewness(; alg = SemiMoment()))` needs no wrapper and keeps one copy of the sample.

# Algorithm

 1. Fold `pe.pe` with the observations and `F` through [`partial_fit!`](@ref).
 2. Fold `pe.kte` and `pe.ske` with the observations through [`fold_member`](@ref).
 3. Rebuild the estimator with the three folded members with `rebuild_estimator`, and return it.

# Arguments

  - `pe`: High order prior estimator.
  - `X`: Observations to fold, a matrix of rows or one observation.
  - `F`: The factor observations beside `X`, or `nothing`. The method forwards them to the embedded prior.
  - `kwargs...`: Additional keyword arguments, forwarded to the members.

# Validation

  - `pe.pe` folds. The embedded prior refuses the fold by name otherwise.

# Returns

  - `pe`: The estimator, with its members folded.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
  - [`fold_member`](@ref)
  - [`supports_partial_fit`](@ref)
"""
function partial_fit!(pe::HighOrderPriorEstimator, X::VecNum_MatNum,
                      F::Option{<:VecNum_MatNum} = nothing; kwargs...)
    # `rebuild_estimator`, not `Accessors.@reset`: this host declares forwarded properties,
    # and `@reset` rebuilds a struct by reading every *property*, which on such a type is not
    # the field list at all.
    return rebuild_estimator(pe,
                             (; pe = partial_fit!(pe.pe, X, F; kwargs...),
                              kte = fold_member(pe.kte, X; kwargs...),
                              ske = fold_member(pe.ske, X; kwargs...)))
end
"""
    prior(pe::HighOrderPriorEstimator; kwargs...)

Reads a [`HighOrderPriorEstimator`](@ref) out of its fold, with no data matrix.

The embedded prior answers first. A co-moment that folded answers from its state. The read-out refits a co-moment that did not fold over the rows that the embedded prior folded, which [`prior_returns_buffer`](@ref) reads. [`assemble_high_order_prior`](@ref) then builds the result as the batch method does.

The refit reads the folded rows and not `pr.X`, because `pr.X` holds the scenarios of the embedded prior's result, and these are not always the folded rows. A Scenario Cap on the embedded prior keeps only the last rows, and the scenario fill writes zeros into the early rows of an asset that lists late. The batch method fits both co-moments over every row that the caller passes, `NaN` entries included, so a refit over `pr.X` differs from it in both cases. Under an [`Online`](@ref) window the buffer holds the last `max_history` rows, which is the window that the batch equal of that route fits over. The host keeps no copy of the rows, because the embedded prior already holds them.

# Mathematical definition

```math
\\begin{align}
\\mathcal{P}_T &= \\mathcal{P}(\\mathbf{X})\\,.
\\end{align}
```

Where:

  - $(math_dict[:P_fold_prior])
  - $(math_dict[:P_batch_prior])
  - $(math_dict[:X_returns])

The equality holds for the embedded result and for every co-moment, the ones that fold and the ones that the read-out refits. In floating point, each co-moment differs from the batch co-moment by rounding.

# Algorithm

 1. Read the embedded result `pr` out of `pe.pe` with [`prior`](@ref).
 2. Read the rows that the embedded prior folded with [`prior_returns_buffer`](@ref) and [`sample_buffer`](@ref), giving `X`.
 3. Read `kt` out of `pe.kte` through [`read_member`](@ref), which refits it over `X` where it did not fold.
 4. Read `sk` and `V` out of `pe.ske` through [`read_member`](@ref), on the same terms.
 5. Build the result with [`assemble_high_order_prior`](@ref), and return it.

# Arguments

  - `pe`: High order prior estimator whose members carry states.
  - `kwargs...`: Additional keyword arguments, forwarded to the members.

# Validation

  - Every condition that the read-out of `pe.pe` checks.
  - Everything [`prior_returns_buffer`](@ref) refuses.

# Returns

  - `hop::HighOrderPrior`: Result object carrying the low order result and the co-moments.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`assemble_high_order_prior`](@ref)
  - [`read_member`](@ref)
  - [`prior_returns_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
function prior(pe::HighOrderPriorEstimator; kwargs...)
    pr = prior(pe.pe; kwargs...)
    X = sample_buffer(prior_returns_buffer(pe.pe))
    kt = read_member(cokurtosis, pe.kte, X; kwargs...)
    sk, V = read_member(coskewness, pe.ske, X; kwargs...)
    return assemble_high_order_prior(pe, pr, kt, sk, V)
end
"""
    partial_fit!(pe::BlackLittermanPrior, X, F = nothing; kwargs...)

Folds observations into a [`BlackLittermanPrior`](@ref) by forwarding them to its embedded prior.

The host keeps no buffer, for the reason that [`HighOrderPriorEstimator`](@ref) keeps none. It returns [`forward_prior`](@ref) of the result of its embedded prior, so the rows it needs are one level down. The factor observations go down with the rows, as the batch verb passes `F` down. The views are configuration, and they fold nothing.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Observations to fold, a matrix of rows or one observation.
  - `F`: The factor observations beside `X`, or `nothing`. The method forwards them to the embedded prior.
  - `kwargs...`: Additional keyword arguments, forwarded to the embedded prior.

# Validation

  - `pe.pe` folds. The embedded prior refuses the fold by name otherwise.

# Returns

  - `pe`: The estimator, with its embedded prior folded.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`prior`](@ref)
  - [`bl_posterior`](@ref)
"""
function partial_fit!(pe::BlackLittermanPrior, X::VecNum_MatNum,
                      F::Option{<:VecNum_MatNum} = nothing; kwargs...)
    return rebuild_estimator(pe, (; pe = partial_fit!(pe.pe, X, F; kwargs...)))
end
"""
    prior(pe::BlackLittermanPrior; strict::Bool = false, kwargs...)

Reads a [`BlackLittermanPrior`](@ref) out of its fold, with no data matrix.

The embedded prior answers first, and [`bl_posterior`](@ref) is the body of the batch method, unchanged. That body reads every number from the result and none from a returns matrix. These numbers are the number of observations behind `tau`, the asset axis, and the matrix that the processing runs over. So the fold and the batch fit run one body.

# Mathematical definition

```math
\\begin{align}
\\mathcal{P}_T &= \\mathcal{P}(\\mathbf{X})\\,.
\\end{align}
```

Where:

  - $(math_dict[:P_fold_prior])
  - $(math_dict[:P_batch_prior])
  - $(math_dict[:X_returns])

# Algorithm

 1. Read the embedded result `prior_model` out of `pe.pe` with [`prior`](@ref), passing `strict`.
 2. Check the asset axis of the views against `prior_model.X` with [`assert_bl_views_axis`](@ref).
 3. Compute the posterior with [`bl_posterior`](@ref), and return it.

# Arguments

  - `pe`: Black-Litterman prior estimator whose embedded prior carries a state.
  - `strict`: Whether an unresolved view raises rather than warns.
  - `kwargs...`: Additional keyword arguments, forwarded to the embedded prior and the processing.

# Validation

  - Every condition that the read-out of `pe.pe` checks.
  - The views of a [`LinearConstraintEstimator`](@ref) name one entry per asset of `prior_model.X`. A `DimensionMismatch` is thrown otherwise.

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
    update_online_estimator(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})

Resolves the [`Online`](@ref) declarations under the prior that a wrapping prior embeds, at warm-up.

The generic method scans the fields of the host alone, and a wrapping prior holds its embedded prior in its own `pe` field, so the wrapping prior defines the recursion itself. `HighOrderPriorEstimator(; pe = BlackLittermanPrior(; pe = Online(EmpiricalPrior()), ...))` resolves through this method. The generic scan of the host's own fields finds no wrapper there, because `pe` holds a plain prior, so the generic method would leave the wrapper two levels down unseeded. A wrapper in the host's own `pe`, as in `HighOrderPriorEstimator(; pe = Online(EmpiricalPrior()))`, is seeded the same way, because the recursion meets it at the first level. The generic method still scans the co-moments of a [`HighOrderPriorEstimator`](@ref).

# Algorithm

 1. Resolve every field that [`online_fields`](@ref) names with [`update_online_estimator`](@ref), giving `repl`.
 2. Resolve `pe.pe` with [`update_online_estimator`](@ref), which recurses through a wrapping prior.
 3. Rebuild the prior from `repl` and the resolved `pe.pe`, the second taking precedence for `pe`, and return it.

# Arguments

  - `pe`: The wrapping prior.

# Returns

  - `pe`: The prior, with every wrapper under its embedded prior resolved.

# Related

  - [`update_online_estimator`](@ref)
  - [`Online`](@ref)
"""
function update_online_estimator(pe::Union{<:HighOrderPriorEstimator,
                                           <:BlackLittermanPrior})
    fns = online_fields(pe)
    repl = NamedTuple{fns}(map(f -> update_online_estimator(getfield(pe, f)), fns))
    return rebuild_estimator(pe, merge(repl, (; pe = update_online_estimator(pe.pe))))
end
