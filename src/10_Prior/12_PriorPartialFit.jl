"""
$(DocStringExtensions.TYPEDEF)

Carries the observations that a fold-and-carry prior keeps, and the assets whose scenario fill it has already named.

A prior that carries this state folds its moments exactly, member by member, and keeps the rows for one reason. [`LowOrderPrior`](@ref) carries `X` for the scenario risk measures, so `prior(pe)` with no data copies the rows into the result and computes nothing from them. A [`SampleBufferState`](@ref) is different. An estimator that carries one has no recursion of its own, and its call with no data runs the batch verb over the rows that the buffer kept.

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
    Indices of the assets whose scenario fill a call of `prior(pe)` with no data has already named. [`strict_diagnostic`](@ref) keeps no record, so without this set a walk-forward that reads out at every step names the same asset at every step. That call adds to the set in place, because it returns a Prior Result and not the estimator, and no other return value takes the set to the next step. [`Base.copy`](@ref) copies the set, and [`merge_states`](@ref) takes the union of two.
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
    returns_buffer(state::CrossSectionalCarryState)
    returns_buffer(state::AbstractPartialFitState)

Reads the buffer of asset returns out of the state that a prior carries.

Three states hold rows at the prior layer, each in a different place. A [`SampleBufferState`](@ref) is the rows. A [`PriorCarryState`](@ref) and a [`CrossSectionalCarryState`](@ref) keep them in `buf`. `optimise(opt)` with no data, and `prior(pe)` with no data on a prior that forwards its fold to an embedded prior, read the returns through this function. Neither reads the fields of a state, so a new state that holds rows adds one method here. Where the tree of the prior reads factor returns, the buffer holds the factor rows too, and `optimise(opt)` with no data takes them with [`factor_buffer`](@ref) when the fold context keeps no factor column. This function throws an `ArgumentError` that names the type of a state that holds no rows, such as the exact-fold state of a moment estimator.

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
    return throw(ArgumentError("a `$(typeof(state))` carries no rows, so a call with no data cannot rebuild the returns that it folded. That call reads the observations from the prior's own buffer, which a `SampleBufferState` or a `PriorCarryState` holds."))
end
"""
    prior_returns_buffer(pe::AbstractPriorEstimator)
    prior_returns_buffer(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})

Reads the buffer of asset returns out of the prior that owns the folded rows.

One prior in the chain owns the rows. [`HighOrderPriorEstimator`](@ref) and [`BlackLittermanPrior`](@ref) build their result around the prior they embed and own no rows, so this function reads the buffer of the embedded prior, at any depth. Every other prior keeps its rows in its own `cache`. Two calls with no data use it. `optimise(opt)` with no data rebuilds the returns it forwarded, and `prior(pe)` with no data on a [`HighOrderPriorEstimator`](@ref) refits a co-moment that does not fold over the rows.

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

The method forwards to the fold of the buffer, with its keyword arguments. So a [`CoveragePolicy`](@ref) mask goes into the buffer beside the rows it describes, as it does for a [`SampleBufferState`](@ref) that an [`Online`](@ref) seeded. No factor observation reaches the buffer, because [`EmpiricalPrior`](@ref), the one carrying prior, never reads one and drops it before the carry. The fold leaves the named-asset set as it is, because only `prior(pe)` with no data finds an asset to name.

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

It is [`fold_buffer`](@ref) with the named-asset set kept. The seed has no cap. The cap on the carried rows is the `max_scenarios` of an `EmpiricalPrior`, which `prior(pe)` with no data applies. A cap on the fit is the `max_history` of an [`Online`](@ref), which puts the prior on the refit route instead.

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

The method copies the named-asset set for the reason it copies the backing matrix. `prior(pe)` with no data writes into the set, so a copy that shared it would record its notices in the set of the original.

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

This is the method of [`prior`](@ref) with no data, and the entry point of every prior that folds. It reads the state and dispatches on its type:

  - A [`SampleBufferState`](@ref) means refit. The estimator has no recursion of its own, so the method runs the batch verb over the rows that the buffer kept. These are all the rows folded so far when the buffer has no cap, and the last `max_history` rows when an [`Online`](@ref) set one. The batch verb receives the factor rows that the buffer recorded, through [`factor_buffer`](@ref), or `nothing` where the buffer recorded none. The fold fixed that choice from the estimator tree.
  - A [`PriorCarryState`](@ref) means fold and carry. Each family that takes it has its own method of `prior` with no data.

# Mathematical definition

Under a [`SampleBufferState`](@ref):

```math
\\begin{align}
\\mathcal{P}_T &= \\mathcal{P}(\\mathbf{X}_{T-m+1:T},\\, \\mathbf{F}_{T-m+1:T},\\, \\mathcal{A}_{T-m+1:T})\\,, \\\\
m &= \\min(T,\\, M)\\,.
\\end{align}
```

Where:

  - $(math_dict[:P_fold_prior])
  - $(math_dict[:P_batch_prior])
  - ``\\mathbf{X}_{a:b}``: Rows ``a`` to ``b`` of the returns matrix of the folded observations.
  - ``\\mathbf{F}_{a:b}``: Rows ``a`` to ``b`` of the factor returns matrix of the folded factor observations, absent when the fold recorded none.
  - ``\\mathcal{A}_{a:b}``: The Asset Panel of rows ``a`` to ``b``, with the folded Panel Fields and both folded masks, absent when the fold recorded no Panel Field.
  - ``m``: Number of rows that the buffer holds.
  - ``M``: Cap on the buffer, the `max_history` of the wrapping [`Online`](@ref), and ``\\infty`` when it is `nothing`.
  - $(math_dict[:T])

# Algorithm

 1. Read the state out of `pe.cache` with [`partial_fit_cache`](@ref).
 2. Dispatch on the type of the state. Under a [`SampleBufferState`](@ref), run the batch verb over the [`sample_buffer`](@ref) and the [`factor_buffer`](@ref) of the state with [`buffer_prior`](@ref), which gives the masks and the Panel Fields that the buffer records, and the caller's keyword arguments.

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
  - [`sample_buffer_panel`](@ref)
"""
function prior(pe::AbstractPriorEstimator; kwargs...)
    return prior(pe, partial_fit_cache(pe); kwargs...)
end
function prior(pe::AbstractPriorEstimator, state::SampleBufferState; kwargs...)
    return buffer_prior(pe, state, sample_buffer_panel(state); kwargs...)
end
"""
    buffer_prior(pe::AbstractPriorEstimator, state::SampleBufferState, pnl::Nothing; kwargs...)
    buffer_prior(pe::AbstractPriorEstimator, state::SampleBufferState, pnl::AssetPanel; kwargs...)

Runs the batch verb of a prior over the rows of its sample buffer, with the masks as keywords or inside the rebuilt Asset Panel.

This is the call with no data of the refit route. A buffer that records no Panel Field gives its masks as the keywords that [`sample_buffer_kwargs`](@ref) reads, as a matrix call of the batch verb takes them. A buffer that records Panel Fields gives the panel that [`sample_buffer_panel`](@ref) rebuilds as the third positional argument, as `prior(pe, rd)` gives `rd.pnl`. That panel holds both masks, so the call gives no mask keyword.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: call `prior(pe, X, F; masks..., kwargs...)` over the rows of the buffer.
 2. `pnl` is an Asset Panel: call `prior(pe, X, F, pnl; kwargs...)` over the rows of the buffer.

# Arguments

  - `pe`: Prior estimator whose `cache` holds `state`.
  - `state`: The buffer to read.
  - `pnl`: The panel of the rows of the buffer, or `nothing`.
  - `kwargs...`: Additional keyword arguments, forwarded to the batch verb.

# Returns

  - `pr::AbstractPriorResult`: The batch fit over the rows of the buffer.

# Related

  - [`prior`](@ref)
  - [`sample_buffer_panel`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`SampleBufferState`](@ref)
"""
function buffer_prior(pe::AbstractPriorEstimator, state::SampleBufferState, ::Nothing;
                      kwargs...)
    return prior(pe, sample_buffer(state), factor_buffer(state);
                 sample_buffer_kwargs(state)..., kwargs...)
end
function buffer_prior(pe::AbstractPriorEstimator, state::SampleBufferState, pnl::AssetPanel;
                      kwargs...)
    return prior(pe, sample_buffer(state), factor_buffer(state), pnl; kwargs...)
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

The answer depends on the tree that an estimator embeds, not on the type of the outer estimator. A prior whose factor argument is optional never reads the argument, and passes it to the prior it embeds. Only a member that requires the argument reads it. So the answer has three values, and the abstract type of the member gives each one:

  - `true` for a member that requires factor returns, [`AbstractHiLoOrderPriorEstimator_F`](@ref). Its batch verb declares `F::MatNum` with no default, and refuses a fit without one.
  - `false` for a member that never reads them, [`AbstractLowOrderPriorEstimator_A`](@ref). Its batch verb declares the argument and ignores it, so a fold drops it as the batch verb does.
  - `nothing` for a member whose factor argument is optional, [`AbstractLowOrderPriorEstimator_AF`](@ref). The type does not say, so the fold takes what it receives, as the batch verb does.

Each outer estimator of the library whose factor argument is optional recurses into the prior it embeds and answers the value of the leaf. So `EntropyPoolingPrior(; pe = FactorPrior())` answers `true`, and `EntropyPoolingPrior()` answers `false`. [`OpinionPoolingPrior`](@ref) holds several priors and combines their answers with [`combine_factor_answers`](@ref). An [`Online`](@ref) answers for the estimator it wraps. A caller's own optional-argument subtype that embeds a prior must define the recursion. A subtype that reads `F` itself can keep the default, which takes what it receives.

The checks at entry that look for a missing factor matrix read this predicate. These are the [`ReturnsResult`](@ref) method of the prior, the step of the online optimiser, and the `ReturnsResult` methods of [`ucs`](@ref), [`mu_ucs`](@ref) and [`sigma_ucs`](@ref). The predicate finds a factor leaf under an outer estimator whose factor argument is optional, which an `isa` test on the outer estimator cannot find. It also decides what the refit route of [`partial_fit!`](@ref) does with the factor observation it receives.

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

Combines the answers of several embedded priors into one, for an outer estimator that holds more than one.

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
    reads_panel_fields(pe::AbstractPriorEstimator) -> false
    reads_panel_fields(pe::CrossSectionalFactorPrior) -> true
    reads_panel_fields(pe::Nothing) -> false
    reads_panel_fields(pe::Online)
    reads_panel_fields(pe::Union{HighOrderPriorEstimator, BlackLittermanPrior,
                                 MeucciEntropyPoolingPrior, EntropyPoolingPrior})
    reads_panel_fields(pe::OpinionPoolingPrior)

Answers whether the fit of a prior reads the Panel Fields of its Asset Panel, from its estimator tree.

Every prior receives the Asset Panel, but most read its masks alone. A [`CrossSectionalFactorPrior`](@ref) computes its factors from the Panel Fields at each row, so for it a time-varying Panel Field is sample. The online step reads this predicate. It gives the Panel Fields to the buffer of a prior whose tree reads them, and it refuses them for any other prior. The buffer then records them, and the call with no data rebuilds the panel.

Each outer estimator of the library that embeds a prior passes the panel to it, and never reads a Panel Field itself. So each one recurses into the prior it embeds, as [`needs_factor_returns`](@ref) does. [`OpinionPoolingPrior`](@ref) answers `true` when any prior it holds reads them. An [`Online`](@ref) answers for the estimator it wraps. The answer is a `Bool`, because no prior reads the Panel Fields as an option. A caller's own prior that reads them must define a method that answers `true`.

# Arguments

  - `pe`: The prior estimator, the wrapper around one, or `nothing`.

# Returns

  - `reads::Bool`: `true` when the tree holds a prior that reads the Panel Fields.

# Related

  - [`needs_factor_returns`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function reads_panel_fields(::AbstractPriorEstimator)
    return false
end
function reads_panel_fields(::CrossSectionalFactorPrior)
    return true
end
function reads_panel_fields(::Nothing)
    return false
end
function reads_panel_fields(pe::Online)
    return reads_panel_fields(pe.est)
end
function reads_panel_fields(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior,
                                      <:MeucciEntropyPoolingPrior, <:EntropyPoolingPrior})
    return reads_panel_fields(pe.pe)
end
function reads_panel_fields(pe::OpinionPoolingPrior)
    return any(reads_panel_fields, (pe.pe1, pe.pe2, pe.pes...))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a missing factor matrix at a check at entry whose prior tree requires one.

Five checks at entry share this refusal: the [`ReturnsResult`](@ref) method of the prior, the step of the online optimiser, and the `ReturnsResult` methods of [`ucs`](@ref), [`mu_ucs`](@ref) and [`sigma_ucs`](@ref). The refit route of [`partial_fit!`](@ref) reaches it too, through [`fold_factor_argument`](@ref). One method holds the message and the test, so the two cannot differ between callers. The test is [`needs_factor_returns`](@ref) answering `true`. That predicate walks the estimator tree. So this method refuses a factor leaf under an outer estimator whose factor argument is optional with an `IsNothingError` that names `F`, before the leaf meets a `MethodError` one call later.

A `pe` of `nothing` is an uncertainty set with no prior of its own. It reads no factor matrix, so the method checks nothing. The returns-data method refuses such a set later, with an error that names it, through [`ucs_prior`](@ref).

# Arguments

  - `pe`: The prior estimator that the check at entry passes the matrix to, or `nothing`.
  - `F`: The factor matrix that the `ReturnsResult` holds, the factor observation that a fold receives, or `nothing`.

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

  - `true`. The tree holds a factor leaf. The fold refuses a call without `F` with the `IsNothingError` of [`assert_factor_returns`](@ref), which names `F`, before it appends a row. A call with `F` records it.
  - `false`. The tree never reads `F`, so the fold drops it, as `prior(EmpiricalPrior(), X, F)` drops it, and the buffer records the rows alone.
  - `nothing`. The tree does not say, so the buffer records `F` when the call gives it and not otherwise, as the batch verb of an optional-argument prior does.

The buffer fixes what it records at its first append and refuses a mixture. So a run that gives `F` at one step and not at the next is refused at that step with an error.

A prior that reads Panel Fields, as [`reads_panel_fields`](@ref) answers, refuses this form, because a matrix carries no Asset Panel. It folds a [`ReturnsResult`](@ref) instead.

# Algorithm

 1. Refuse a prior that reads Panel Fields. Answer [`needs_factor_returns`](@ref) for the tree, and apply the answer to `F` with [`fold_factor_argument`](@ref), giving the `F` that the buffer records.
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

  - `reads_panel_fields(pe)` is `false`. An `ArgumentError` is thrown otherwise.
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
    @argcheck(!reads_panel_fields(pe),
              ArgumentError("`$(typeof(pe).name.name)` reads its Panel Fields off an Asset Panel, and the matrix form of `partial_fit!` carries none. Fold a `ReturnsResult` whose `pnl` is a time-varying Asset Panel with `partial_fit!(pe, rd)`."))
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
    partial_fit!(pe::AbstractPriorEstimator, rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into a prior. This is the fold of every prior.

It mirrors `prior(pe, rd)`. It reads the returns, the factor returns and both masks of `rd`, and the Panel Fields of `rd.pnl` when the tree of the prior reads them. The online step of an optimiser folds its prior through this method. The matrix form, `partial_fit!(pe, X, F; active_mask, estimation_mask)`, stays for a prior that reads no Panel Field.

The type of the prior selects the route, and each route states what it can honour. This method is the refit route, which every prior that carries a [`SampleBufferState`](@ref) takes. The buffer records what the batch fit reads, so the route honours all of it:

  - The estimation mask, beside the active mask. No panel, or a static panel, gives no mask.
  - The Panel Fields, when [`reads_panel_fields`](@ref) answers `true`. The buffer records them, and the call with no data gives the batch verb the panel that [`sample_buffer_panel`](@ref) rebuilds. For any other prior a Panel Field is refused by name, because the buffer has no slot that the prior reads.

The carry of [`EmpiricalPrior`](@ref) has a method of its own, because its exact folds take no estimation mask. [`HighOrderPriorEstimator`](@ref) and [`BlackLittermanPrior`](@ref) forward `rd` to the prior they embed. [`CrossSectionalFactorPrior`](@ref) has a method of its own, which refuses a prior with no buffer and applies its Choice Rule after the fold.

# Algorithm

 1. Fold `rd` into the buffer with [`refit_prior_step`](@ref).

# Arguments

  - `pe`: The prior whose buffer the method folds forward.
  - $(arg_dict[:rd])

# Validation

  - The rules of [`refit_prior_fold`](@ref).

# Returns

  - `pe`: The prior, with its `cache` field set to the buffer after the last observation.

# Related

  - [`prior`](@ref)
  - [`reads_panel_fields`](@ref)
  - [`refit_prior_fold`](@ref)
  - [`SampleBufferState`](@ref)
  - [`sample_buffer_panel`](@ref)
  - [`Online`](@ref)
"""
function partial_fit!(pe::AbstractPriorEstimator, rd::ReturnsResult)
    return refit_prior_step(pe, rd)
end
"""
    refit_prior_step(pe::AbstractPriorEstimator, rd::ReturnsResult)
    refit_prior_step(pe::CrossSectionalFactorPrior, rd::ReturnsResult)

Takes the step of the refit route of a prior: the fold of [`refit_prior_fold`](@ref), and the work a prior adds around it.

A prior that adds no work folds `rd` with [`refit_prior_fold`](@ref). A [`CrossSectionalFactorPrior`](@ref) refuses an observed factor first, with [`assert_cross_sectional_online_factors`](@ref), because the buffer records no Exogenous Series. It applies its Choice Rule after the fold, with [`cross_sectional_pin_choice`](@ref). The step is a function of its own, and not a method of [`partial_fit!`](@ref), so every method of that verb that reads a family state narrows the `cache` of its estimator, and a buffer reaches this route alone.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any prior: fold `rd` with [`refit_prior_fold`](@ref).
 2. A [`CrossSectionalFactorPrior`](@ref): refuse an observed factor, fold `rd` with [`refit_prior_fold`](@ref), and pin the dropped members under a [`PinnedChoice`](@ref).

# Arguments

  - `pe`: The prior whose buffer the step folds forward.
  - $(arg_dict[:rd])

# Validation

  - The rules of [`refit_prior_fold`](@ref), and for a Cross-Sectional Factor Prior those of [`assert_cross_sectional_online_factors`](@ref) and [`cross_sectional_pinned_families`](@ref).

# Returns

  - `pe`: The prior, with its `cache` field set to the buffer after the last observation.

# Related

  - [`partial_fit!`](@ref)
  - [`refit_prior_fold`](@ref)
  - [`cross_sectional_pin_choice`](@ref)
"""
function refit_prior_step(pe::AbstractPriorEstimator, rd::ReturnsResult)
    return refit_prior_fold(pe, rd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observations of a [`ReturnsResult`](@ref) into the sample buffer of a prior. This is the refit route of [`partial_fit!`](@ref).

The generic method of [`partial_fit!`](@ref) over a `ReturnsResult` is this function. A prior with a step of its own, such as [`CrossSectionalFactorPrior`](@ref), calls it for the fold and adds its own work around it.

# Algorithm

 1. Refuse `rd` without returns, and `rd` with an implied-volatility surface, with [`assert_prior_fold_returns`](@ref).
 2. Apply the answer of [`needs_factor_returns`](@ref) to `rd.F` with [`fold_factor_argument`](@ref).
 3. Read the buffer out of `pe.cache` with [`assert_sample_buffer`](@ref).
 4. Read the masks and the Panel Fields of `rd.pnl` with [`refit_step_kwargs`](@ref). Fold `rd.X`, the factor returns and those keywords into the buffer through the block method of [`partial_fit!`](@ref).
 5. Rebuild the prior with its `cache` set to the new buffer, and return it.

# Arguments

  - `pe`: The prior whose buffer the function folds forward.
  - $(arg_dict[:rd])

# Validation

  - The rules of [`assert_prior_fold_returns`](@ref).
  - `rd.F` is not `nothing` when `needs_factor_returns(pe) === true`. An `IsNothingError` is thrown otherwise.
  - `pe` carries a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.
  - The rules of [`refit_step_kwargs`](@ref).
  - Every condition that the fold of the buffer checks.

# Returns

  - `pe`: The prior, with its `cache` field set to the buffer after the last observation.

# Related

  - [`partial_fit!`](@ref)
  - [`refit_step_kwargs`](@ref)
  - [`SampleBufferState`](@ref)
"""
function refit_prior_fold(pe::AbstractPriorEstimator, rd::ReturnsResult)
    assert_prior_fold_returns(rd)
    F = fold_factor_argument(needs_factor_returns(pe), pe, rd.F)
    state = partial_fit!(assert_sample_buffer(pe), rd.X, F;
                         refit_step_kwargs(pe, rd.pnl)...)
    return rebuild_estimator(pe, (; cache = state))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses returns data that the fold of a prior cannot take.

Each form of [`partial_fit!`](@ref) of a prior over a [`ReturnsResult`](@ref) calls it first, and so does [`step_active_mask`](@ref). A fold records the returns, the factor returns, the masks and the Panel Fields. An implied-volatility surface has no slot in any state. A fold that took it would fold a covariance that reads the surface without it, and it would read out an answer that a batch fit does not give.

# Arguments

  - $(arg_dict[:rd])

# Validation

  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - `rd.iv` is `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`partial_fit!`](@ref)
  - [`step_active_mask`](@ref)
"""
function assert_prior_fold_returns(rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    @argcheck(isnothing(rd.iv),
              ArgumentError("the online step takes no implied-volatility surface: a prior's `partial_fit!` folds the returns and the factors alone, so a step carrying `iv` would fold a covariance that reads the surface without it and read out an answer a batch fit would not give. Drop `iv` from `rd`, or fit in batch."))
    return nothing
end
"""
    refit_step_kwargs(pe::AbstractPriorEstimator, pnl::Nothing) -> NamedTuple
    refit_step_kwargs(pe::AbstractPriorEstimator, pnl::AssetPanel{<:Any, Nothing, Nothing}) -> NamedTuple
    refit_step_kwargs(pe::AbstractPriorEstimator, pnl::AssetPanel) -> NamedTuple

Reads the keywords that the refit route of a prior folds from an Asset Panel: both masks, and the Panel Fields when the prior reads them.

The refit honours everything that the batch fit reads from the panel, because its buffer records it. No panel and a static panel give no keyword. A static panel has no observation axis, so it gives no row to record.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing` or static: return an empty `NamedTuple`.
 2. `pnl` is time-varying: take `active_mask = pnl.amsk` and `estimation_mask = pnl.emsk`. Add `panel_fields = pnl.pf` when [`step_panel_fields`](@ref) answers `true`.

# Arguments

  - `pe`: The prior.
  - `pnl`: The Asset Panel of the step, or `nothing`.

# Validation

  - The rules of [`step_panel_fields`](@ref).

# Returns

  - `kwargs::NamedTuple`: The keywords of the fold of the buffer.

# Related

  - [`partial_fit!`](@ref)
  - [`step_panel_fields`](@ref)
  - [`reads_panel_fields`](@ref)
"""
function refit_step_kwargs(::AbstractPriorEstimator, ::Nothing)
    return (;)
end
function refit_step_kwargs(::AbstractPriorEstimator, ::AssetPanel{<:Any, Nothing, Nothing})
    return (;)
end
function refit_step_kwargs(pe::AbstractPriorEstimator, pnl::AssetPanel)
    kw = (; active_mask = pnl.amsk, estimation_mask = pnl.emsk)
    if step_panel_fields(pe, pnl, "the refit of `$(nameof(typeof(pe)))`")
        return (; kw..., panel_fields = pnl.pf)
    end
    return kw
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Answers whether a step gives Panel Fields to the fold of a prior, and refuses Panel Fields that the prior does not read.

A buffer records the Panel Fields only for a prior whose tree reads them, as [`reads_panel_fields`](@ref) answers. For any other prior the buffer has no slot that the prior reads. So the step cannot keep the Panel Fields, and `optimise(opt)` with no returns cannot rebuild them. The refusal names the route of the prior.

# Arguments

  - `pe`: The prior.
  - `pnl`: The time-varying Asset Panel of the step.
  - `route`: The name of the route, for the message.

# Validation

  - `pnl.pf` is empty, or `reads_panel_fields(pe)` is `true`. An `ArgumentError` is thrown otherwise.

# Returns

  - `gives::Bool`: `true` when `pnl` holds Panel Fields and the prior reads them.

# Related

  - [`reads_panel_fields`](@ref)
  - [`refit_step_kwargs`](@ref)
  - [`partial_fit!`](@ref)
"""
function step_panel_fields(pe::AbstractPriorEstimator, pnl::AssetPanel,
                           route::AbstractString)::Bool
    if isempty(pnl.pf)
        return false
    end
    @argcheck(reads_panel_fields(pe),
              ArgumentError("$route reads no Panel Field, and this Asset Panel holds $(length(pnl.pf)) Panel Field(s). A prior's buffer records the Panel Fields only when its estimator tree reads them, so the step has nowhere to keep these, and `optimise(opt)` with no returns could not rebuild the panel. Drop the fields from the panel handed to the step, or fit in batch."))
    return true
end
"""
    fold_factor_argument(needs::Bool, pe::AbstractPriorEstimator, F::Option{<:VecNum_MatNum})
    fold_factor_argument(::Nothing, ::AbstractPriorEstimator, F::Option{<:VecNum_MatNum})

Applies the answer of the estimator tree to the factor argument of a fold.

The method reads the three answers of [`needs_factor_returns`](@ref) as the refit route needs them. `true` refuses a missing `F` with an `IsNothingError` that names it, and passes a present one through. `false` drops `F`. `nothing` passes `F` through as it is. The method dispatches on the answer, so a tree whose answer follows from its type, which is true of every tree in the library, costs the fold no branch at run time.

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

Folds an observation into a member of an outer estimator that carries the observations, where that member folds.

This is the fold half of the rule for an outer estimator with mixed members. An outer estimator that carries the observations folds every member that folds, and leaves the other members as they are, because it can run their batch verb over its own rows in the call with no data. [`supports_partial_fit`](@ref) answers the question from the type of the member and its `cache` field. So for an outer estimator of concrete members the compiler resolves the branch, and the outer estimator folds no member that must not fold.

A member that the outer estimator does not hold is `nothing`, and a fold of it returns `nothing`.

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

Reads the estimate of a member out of its fold, or refits the member over the rows of the outer estimator.

This half of the rule for an outer estimator with mixed members reads the estimate, and [`fold_member`](@ref) is the fold half. A member that the outer estimator folded answers from its state. A member that the outer estimator did not fold answers from the matrix that the outer estimator passes. `f` is the batch verb of the member, for example `Statistics.mean`, `Statistics.cov`, [`coskewness`](@ref) or [`cokurtosis`](@ref). The one-argument method of the same verb reads the estimate from the state, and every family that folds follows this convention.

No [`AssetPanel`](@ref) reaches this verb. A panel describes the fold and is not a sample, and a buffer holds no activity mask, so the verb refits a member over the rows alone.

# Arguments

  - `f`: The batch verb of the member.
  - `est`: The member, or `nothing`.
  - `X`: The rows that the outer estimator carries, which are the rows the member was folded on.
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

`pe.me` and `pe.ce` fold exactly, so a step of the prior is their two steps and an append to the carried rows. The cost of a step grows with the square of the number of assets, and does not grow with the number of observations already folded. The cost of a refit from a buffer grows with that number too. This difference is the reason the carry route exists. A call with no data that is written as `prior(pe, sample_buffer(pe))` passes every parity test and discards both exact folds, so a review must refuse that form.

The two methods differ in one line. The method with no horizon folds the observation as it is. The horizon method folds `log1p` of the observation, because the batch horizon method fits its moments on log returns. It carries the observation itself, because the [`LowOrderPrior`](@ref) it returns carries the arithmetic returns that the caller passed. A buffer of log rows would change the matrix that every scenario risk measure reads.

A member that does not fold, such as a `SemiMoment` covariance or a composite whose `mp.alg` reads the sample, stays as it is under [`fold_member`](@ref), and `prior(pe)` with no data refits it over the carried rows. So a caller writes the estimator that they would write in batch, and no member keeps a second copy of the sample.

The fold takes a factor observation and drops it, because `prior(EmpiricalPrior(), X, F)` declares `F` and never reads it. The fold has the arity of the batch verb, so an outer estimator that passes `F` down its tree meets no `MethodError` here. [`needs_factor_returns`](@ref) answers `false` for this estimator, so no estimator above it keeps a factor row for it.

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
  - Every condition that the fold of `pe.me` or `pe.ce` checks. A member that holds observation weights refuses the fold with an `ArgumentError` that names the member, because a weight vector reweights every past observation when a new one arrives.

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
    partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, <:Any, <:Any, <:Any,
                                    <:Option{<:PriorCarryState}}, rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into an [`EmpiricalPrior`](@ref) on its carry route.

This is the carry form of the fold of every prior, [`partial_fit!`](@ref) over a `ReturnsResult`. The carry folds `pe.me` and `pe.ce` exactly, and the exact folds of the moment layer take an active mask and no estimation mask. So the route cannot honour an estimation mask that differs from the active mask, and it refuses one by name. The refit route honours it: wrap the prior in [`Online`](@ref). The prior reads no Panel Field, so a Panel Field is refused as the refit route refuses it for such a prior.

# Algorithm

 1. Refuse `rd` without returns, and `rd` with an implied-volatility surface, with [`assert_prior_fold_returns`](@ref).
 2. Refuse the Panel Fields and an estimation mask that differs from the active mask with [`assert_carry_step_panel`](@ref).
 3. Fold `rd.X` with the active mask that [`step_active_kwargs`](@ref) reads through the matrix form of the carry.

# Arguments

  - `pe`: Empirical prior estimator on its carry route.
  - $(arg_dict[:rd])

# Validation

  - The rules of [`assert_prior_fold_returns`](@ref) and of [`assert_carry_step_panel`](@ref).
  - Every condition that the matrix form of the carry checks.

# Returns

  - `pe`: The estimator, with its members folded and its `cache` field rebound.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`PriorCarryState`](@ref)
  - [`assert_carry_step_panel`](@ref)
  - [`step_active_kwargs`](@ref)
"""
function partial_fit!(pe::EmpiricalPrior{<:Any, <:Any, <:Any, <:Any, <:Any,
                                         <:Option{<:PriorCarryState}}, rd::ReturnsResult)
    assert_prior_fold_returns(rd)
    assert_carry_step_panel(pe, rd.pnl)
    return partial_fit!(pe, rd.X; step_active_kwargs(rd.pnl)...)
end
"""
    assert_carry_step_panel(pe::EmpiricalPrior, pnl::Nothing) -> nothing
    assert_carry_step_panel(pe::EmpiricalPrior, pnl::AssetPanel{<:Any, Nothing, Nothing}) -> nothing
    assert_carry_step_panel(pe::EmpiricalPrior, pnl::AssetPanel) -> nothing

Refuses the parts of a time-varying Asset Panel that the carry of an [`EmpiricalPrior`](@ref) cannot honour.

The carry folds its moments exactly, and those folds take the active mask alone. So the route refuses an estimation mask that differs from the active mask, and it names the route and the refit that honours it. The prior reads no Panel Field, so [`step_panel_fields`](@ref) refuses a panel that holds one.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing` or static: return.
 2. `pnl` is time-varying: refuse its Panel Fields with [`step_panel_fields`](@ref). Refuse an `emsk` that differs from `amsk`, and count the cells that differ.

# Arguments

  - `pe`: Empirical prior estimator on its carry route.
  - `pnl`: The Asset Panel of the step, or `nothing`.

# Validation

  - The rules of [`step_panel_fields`](@ref).
  - `pnl.emsk == pnl.amsk`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`partial_fit!`](@ref)
  - [`step_panel_fields`](@ref)
  - [`Online`](@ref)
"""
function assert_carry_step_panel(::EmpiricalPrior, ::Nothing)
    return nothing
end
function assert_carry_step_panel(::EmpiricalPrior, ::AssetPanel{<:Any, Nothing, Nothing})
    return nothing
end
function assert_carry_step_panel(pe::EmpiricalPrior, pnl::AssetPanel)
    route = "the carry of `EmpiricalPrior`"
    step_panel_fields(pe, pnl, route)
    @argcheck(pnl.emsk == pnl.amsk,
              ArgumentError("$route folds its moments exactly, and the exact folds of the moment layer take an active mask and no estimation mask, so this route cannot honour an estimation universe narrower than the active one. This panel's `emsk` differs from its `amsk` at $(count(pnl.emsk .!= pnl.amsk)) cell(s). Wrap the prior in `Online` to refit it, which honours the estimation mask, pass the active mask as both, or fit in batch."))
    return nothing
end
"""
    step_active_kwargs(pnl::Nothing) -> NamedTuple
    step_active_kwargs(pnl::AssetPanel{<:Any, Nothing, Nothing}) -> NamedTuple
    step_active_kwargs(pnl::AssetPanel) -> NamedTuple

Reads the active mask of the Asset Panel of a step, as the keyword that an exact fold takes.

The carry of an [`EmpiricalPrior`](@ref) and the co-moment members of a [`HighOrderPriorEstimator`](@ref) fold exactly, and their folds take the active mask alone. No panel and a static panel give no keyword.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing` or static: return an empty `NamedTuple`.
 2. `pnl` is time-varying: return `(; active_mask = pnl.amsk)`.

# Arguments

  - `pnl`: The Asset Panel of the step, or `nothing`.

# Returns

  - `kwargs::NamedTuple`: The active mask keyword, or no keyword.

# Related

  - [`partial_fit!`](@ref)
  - [`refit_step_kwargs`](@ref)
"""
function step_active_kwargs(::Nothing)
    return (;)
end
function step_active_kwargs(::AssetPanel{<:Any, Nothing, Nothing})
    return (;)
end
function step_active_kwargs(pnl::AssetPanel)
    return (; active_mask = pnl.amsk)
end
"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                             <:Option{<:PriorCarryState}}; strict::Bool = false, kwargs...)
    prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                             <:Option{<:PriorCarryState}}; strict::Bool = false, kwargs...)

Reads an [`EmpiricalPrior`](@ref) out of its fold, with no data matrix.

`mu` comes from `pe.me`, `sigma` comes from `pe.ce`, and `X` is the matrix that the carry state already holds. The method copies the rows, and does not refit a member that folded. The horizon method then applies the algebra of the batch method through [`horizon_moments!`](@ref), which holds that algebra once.

`pe.max_scenarios` cuts the rows that the result carries, as it does in batch, and the fill runs over the rows that remain. When the cap cuts, `ens` holds the number of observations folded, through [`scenario_ens`](@ref), so a consumer that prices a sample size reads ``T`` and not the number of rows carried. The named-asset set of the carry state limits the notice of the fill, so a walk-forward names an asset at the first step whose fill reaches it, and at no later step.

No [`AssetPanel`](@ref) is read. A panel describes the fold and is not a sample. The Coverage Universe of this method equals the Coverage Universe of a batch fit, because [`coverage_mask`](@ref) is a function of the rows alone, and the carry state holds the rows that a batch fit reads.

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

The estimator keeps no buffer. It builds `HighOrderPrior(; pr = pr, ...)` around the result of its embedded prior, so the rows it needs in `prior(pe)` with no data are the rows that prior carries. A buffer here would be a second copy of the sample.

The estimator forwards the fold to `pe.pe` in every case, because that member carries the rows. An embedded prior that cannot fold refuses here, and names the wrapper that gives it a buffer. The refusal is correct, because the estimator has no rows of its own to give it. The factor observations go with the rows, as the batch verb passes `F` to the embedded prior, and the tree of the embedded prior decides what happens to them. `pe.kte` and `pe.ske` go through [`fold_member`](@ref) instead, because the estimator has rows for them in `prior(pe)` with no data, and that call refits a co-moment that cannot fold. So `HighOrderPriorEstimator(; ske = Coskewness(; alg = SemiMoment()))` needs no wrapper and keeps one copy of the sample.

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

  - `pe.pe` folds. The embedded prior refuses the fold otherwise, with an error that names the wrapper that gives it a buffer.

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
    # `rebuild_estimator`, not `Accessors.@reset`: this estimator declares forwarded
    # properties, and `@reset` rebuilds a struct by reading every *property*, which on such
    # a type is not the field list at all.
    return rebuild_estimator(pe,
                             (; pe = partial_fit!(pe.pe, X, F; kwargs...),
                              kte = fold_member(pe.kte, X; kwargs...),
                              ske = fold_member(pe.ske, X; kwargs...)))
end
"""
    prior(pe::HighOrderPriorEstimator; kwargs...)

Reads a [`HighOrderPriorEstimator`](@ref) out of its fold, with no data matrix.

The embedded prior answers first. A co-moment that folded answers from its state. The method refits a co-moment that did not fold over the rows that the embedded prior folded, which [`prior_returns_buffer`](@ref) reads. [`assemble_high_order_prior`](@ref) then builds the result as the batch method does.

The refit reads the folded rows and not `pr.X`, because `pr.X` holds the scenarios of the embedded prior's result, and these are not always the folded rows. A Scenario Cap on the embedded prior keeps only the last rows, and the scenario fill writes zeros into the early rows of an asset that lists late. The batch method fits both co-moments over every row that the caller passes, `NaN` entries included, so a refit over `pr.X` differs from it in both cases. Under an [`Online`](@ref) window the buffer holds the last `max_history` rows, which is the window that the batch equal of that route fits over. The estimator keeps no copy of the rows, because the embedded prior already holds them.

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

The equality holds for the embedded result and for every co-moment, the ones that fold and the ones that the method refits. In floating point, each co-moment differs from the batch co-moment by rounding.

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

  - Every condition that `prior(pe.pe)` with no data checks.
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

The estimator keeps no buffer, for the reason that [`HighOrderPriorEstimator`](@ref) keeps none. It returns [`forward_prior`](@ref) of the result of its embedded prior, so the rows it needs are one level down. The factor observations go down with the rows, as the batch verb passes `F` down. The views are configuration, and they fold nothing.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Observations to fold, a matrix of rows or one observation.
  - `F`: The factor observations beside `X`, or `nothing`. The method forwards them to the embedded prior.
  - `kwargs...`: Additional keyword arguments, forwarded to the embedded prior.

# Validation

  - `pe.pe` folds. The embedded prior refuses the fold otherwise, with an error that names the wrapper that gives it a buffer.

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
    partial_fit!(pe::HighOrderPriorEstimator, rd::ReturnsResult)
    partial_fit!(pe::BlackLittermanPrior, rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into a prior that embeds another prior, by forwarding `rd` to it.

This is the forwarding form of the fold of every prior, [`partial_fit!`](@ref) over a `ReturnsResult`. The outer prior keeps no rows. Its embedded prior keeps them, so the route of the embedded prior decides what the step honours: the estimation mask and the Panel Fields reach it unchanged, and it refuses what its route cannot honour. A [`HighOrderPriorEstimator`](@ref) also folds `pe.kte` and `pe.ske` through [`fold_member`](@ref). Those folds are exact, and they take the active mask alone, which [`step_active_kwargs`](@ref) reads.

# Algorithm

 1. Fold `pe.pe` with `rd` through [`partial_fit!`](@ref).
 2. For a `HighOrderPriorEstimator`, fold `pe.kte` and `pe.ske` with `rd.X` and the active mask of `rd.pnl` through [`fold_member`](@ref).
 3. Rebuild the estimator with its folded members with `rebuild_estimator`, and return it.

# Arguments

  - `pe`: The prior that embeds another prior.
  - $(arg_dict[:rd])

# Validation

  - Every condition that the fold of `pe.pe` over `rd` checks.

# Returns

  - `pe`: The estimator, with its members folded.

# Related

  - [`HighOrderPriorEstimator`](@ref)
  - [`BlackLittermanPrior`](@ref)
  - [`fold_member`](@ref)
  - [`step_active_kwargs`](@ref)
"""
function partial_fit!(pe::HighOrderPriorEstimator, rd::ReturnsResult)
    kw = step_active_kwargs(rd.pnl)
    return rebuild_estimator(pe,
                             (; pe = partial_fit!(pe.pe, rd),
                              kte = fold_member(pe.kte, rd.X; kw...),
                              ske = fold_member(pe.ske, rd.X; kw...)))
end
function partial_fit!(pe::BlackLittermanPrior, rd::ReturnsResult)
    return rebuild_estimator(pe, (; pe = partial_fit!(pe.pe, rd)))
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

  - Every condition that `prior(pe.pe)` with no data checks.
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

The generic method scans the fields of the wrapping prior alone, and a wrapping prior holds its embedded prior in its own `pe` field, so the wrapping prior defines the recursion itself. `HighOrderPriorEstimator(; pe = BlackLittermanPrior(; pe = Online(EmpiricalPrior()), ...))` resolves through this method. The generic scan of the own fields of the wrapping prior finds no wrapper there, because `pe` holds a plain prior, so the generic method would leave the wrapper two levels down unseeded. A wrapper in the own `pe` of the wrapping prior, as in `HighOrderPriorEstimator(; pe = Online(EmpiricalPrior()))`, is seeded the same way, because the recursion meets it at the first level. The generic method still scans the co-moments of a [`HighOrderPriorEstimator`](@ref).

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
