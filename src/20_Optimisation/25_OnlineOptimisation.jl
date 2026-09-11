"""
    returns_buffer(state::SampleBufferState)
    returns_buffer(state::PriorCarryState)
    returns_buffer(state::FactorSampleBufferState)
    returns_buffer(state::AbstractPartialFitState)

Reads the buffer of asset returns out of the state a prior carries.

Three states carry rows at the prior layer, and each keeps them in a different place: a [`SampleBufferState`](@ref) is the rows, a [`PriorCarryState`](@ref) holds them in `buf`, and a [`FactorSampleBufferState`](@ref) holds them in its returns half. The optimiser's read-out reads the returns through this verb rather than through the state's fields, so a fourth state with rows of its own adds one method here and nothing else. A state that carries no rows — an exact-fold state of the moment layer, which a prior never holds — is refused by name.

# Arguments

  - `state`: The state the prior carries.

# Validation

  - `state` carries rows. An `ArgumentError` is thrown otherwise.

# Returns

  - `buffer::SampleBufferState`: The rows, and the masks folded beside them.

# Related

  - [`prior_returns_buffer`](@ref)
  - [`SampleBufferState`](@ref)
  - [`PriorCarryState`](@ref)
  - [`FactorSampleBufferState`](@ref)
"""
function returns_buffer(state::SampleBufferState)
    return state
end
function returns_buffer(state::PriorCarryState)
    return state.buf
end
function returns_buffer(state::FactorSampleBufferState)
    return state.X
end
function returns_buffer(state::AbstractPartialFitState)
    return throw(ArgumentError("a `$(typeof(state))` carries no rows, so an optimiser cannot rebuild the returns it folded from it. The read-out reads the observations from the prior's own buffer, which a `SampleBufferState`, a `PriorCarryState` or a `FactorSampleBufferState` holds."))
end
"""
    prior_returns_buffer(pe::AbstractPriorEstimator)
    prior_returns_buffer(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})

Reads the buffer of asset returns out of the prior an optimiser forwarded its observations to.

The rows are owned once, at the bottom of the chain, and the two priors that wrap another prior own none of their own — [`HighOrderPriorEstimator`](@ref) and [`BlackLittermanPrior`](@ref) build their result around the embedded prior's, so this walks down to it. Every other prior carries its rows in its own `cache`, and [`partial_fit_cache`](@ref) refuses one that has folded nothing.

# Arguments

  - `pe`: The prior estimator the optimiser forwards to.

# Validation

  - The prior that owns the rows carries a state. An `ArgumentError` is thrown otherwise.

# Returns

  - `buffer::SampleBufferState`: The rows the optimiser has folded, and the masks beside them.

# Related

  - [`returns_buffer`](@ref)
  - [`partial_fit_cache`](@ref)
  - [`returns_result`](@ref)
"""
function prior_returns_buffer(pe::AbstractPriorEstimator)
    return returns_buffer(partial_fit_cache(pe))
end
function prior_returns_buffer(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})
    return prior_returns_buffer(pe.pe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the active mask an observation's Asset Panel carries into the prior's step, and refuses what the step cannot carry.

Three shapes reach here. No panel, or a static one, contributes no mask: the panel is context, pinned by the fold context. A time-varying panel of masks alone — the ingestion layer's shape — hands its active mask over, and the mask rides into the prior's buffer beside the rows. A time-varying panel that carries **Panel Fields** is refused: a time-varying field's rows are sample, and the buffers of the seam hold numbers alone, so the step has nowhere to keep a categorical or a tensor field and cannot rebuild the panel a read-out needs.

The estimation mask does not travel the step. The exact folds of the moment layer take an active mask and no estimation mask, and only the two regime-adjusted families read one, so a step whose estimation universe is narrower than its active universe would fold a plain member with the wrong universe in silence. It is refused instead, by name, where the two differ.

# Arguments

  - `rd`: The carrier of the observations being folded.

# Validation

  - `rd.iv` is `nothing`. An `ArgumentError` is thrown otherwise: the prior's step takes no implied-volatility surface.
  - A time-varying `rd.pnl` carries no Panel Field. An `ArgumentError` is thrown otherwise.
  - A time-varying `rd.pnl` has its estimation mask equal to its active mask. An `ArgumentError` is thrown otherwise.

# Returns

  - `amsk::Option{<:AbstractMatrix{<:Bool}}`: The active mask of the observations, or `nothing`.

# Related

  - [`partial_fit!`](@ref)
  - [`ReturnsBufferState`](@ref)
  - [`AssetPanel`](@ref)
"""
function step_active_mask(rd::ReturnsResult)
    @argcheck(isnothing(rd.iv),
              ArgumentError("the online step takes no implied-volatility surface: a prior's `partial_fit!` folds the returns and the factors alone, so a step carrying `iv` would fold a covariance that reads the surface without it and read out an answer a batch fit would not give. Drop `iv` from the carrier, or fit in batch."))
    pnl = rd.pnl
    if isnothing(pnl) || panel_is_static(pnl)
        return nothing
    end
    @argcheck(isempty(pnl.pf),
              ArgumentError("the online step carries a time-varying Asset Panel through its masks alone, and this one holds $(length(pnl.pf)) Panel Field(s): a time-varying field's rows are sample, and the seam's buffers hold numbers, so the step has nowhere to keep them and a read-out could not rebuild the panel. Drop the fields from the panel handed to the step, or fit in batch."))
    @argcheck(pnl.emsk == pnl.amsk,
              ArgumentError("the estimation mask does not travel the online step: the exact folds of the moment layer take an active mask and no estimation mask, so an estimation universe narrower than the active one cannot be honoured online. This panel's `emsk` differs from its `amsk` at $(count(pnl.emsk .!= pnl.amsk)) cell(s). Pass the active mask as both, or fit in batch."))
    return pnl.amsk
end
"""
    fold_prior(pe::AbstractPriorEstimator, rd::ReturnsResult)
    fold_prior(pe::AbstractPriorResult, rd::ReturnsResult)
    fold_prior(pe::TimeDependent, rd::ReturnsResult)

Forwards the observations of a carrier to the prior, in the prior's own arity.

The one forward the optimiser's step makes, and the rule for every optimiser: **an optimiser forwards the observation to its prior and to nothing else**, and everything it holds beside the prior takes its ordinary batch fit at read-out, from the reconstituted fold context. The carrier is unpacked on the way down exactly as [`prior`](@ref) unpacks it in batch — a prior that requires factor returns receives `X` and `F`, every other receives `X` — and the active mask of a time-varying panel rides as the keyword the prior's step takes.

Two refusals. A prior that is already a [`AbstractPriorResult`](@ref) has no state to fold into: it is batch configuration, and an optimiser holding one runs `optimise(opt, rd)`. A [`TimeDependent`](@ref) on the prior is refused because a schedule swaps the estimator that carries the state, and a member that never saw the folded rows cannot be handed them; a fold loop resolves the schedule before it steps (see [#870](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/870)).

# Arguments

  - `pe`: The prior the optimiser holds.
  - `rd`: The carrier of the observations to fold, `observations × assets`.

# Validation

  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - `rd.F` is not `nothing` when `pe` requires factor returns. An `IsNothingError` is thrown otherwise.
  - Everything [`step_active_mask`](@ref) refuses.

# Returns

  - `pe`: The prior, with the observations folded into its state.

# Related

  - [`partial_fit!`](@ref)
  - [`step_active_mask`](@ref)
  - [`prior`](@ref)
"""
function fold_prior(pe::AbstractPriorEstimator, rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    amsk = step_active_mask(rd)
    kw = isnothing(amsk) ? (;) : (; active_mask = amsk)
    if isa(pe, AbstractHiLoOrderPriorEstimator_F)
        @argcheck(!isnothing(rd.F),
                  IsNothingError("this is a factor prior; it needs factor returns. ReturnsResult.F is nothing — populate F (e.g. via prices_to_returns on factor prices)."))
        return partial_fit!(pe, rd.X, rd.F; kw...)
    else
        return partial_fit!(pe, rd.X; kw...)
    end
end
function fold_prior(pe::AbstractPriorResult, ::ReturnsResult)
    return throw(ArgumentError("`pe` holds a fitted `$(typeof(pe))`, which has no state to fold an observation into: a prior result is batch configuration. Hand the optimiser the prior estimator to take the online step, or run `optimise(opt, rd)`."))
end
function fold_prior(pe::TimeDependent, ::ReturnsResult)
    return throw(ArgumentError("`pe` holds a `TimeDependent` schedule of priors, and the online step cannot fold into a schedule: it swaps the estimator that carries the state, and a member that never saw the folded rows cannot be handed them. Resolve the schedule to one prior before stepping, which is what a fold loop does before it steps (#870)."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observations of a carrier into the fold context a host keeps, seeding it on the first step.

# Arguments

  - `cache`: The context the host carries, or `nothing` before the first step.
  - `rd`: The carrier of the observations.
  - `max_history`: The cap the context takes when it is seeded, which is the cap of the buffer that holds the returns so the two stay in step.
  - `own_returns`: Whether the context keeps the returns itself, which is `true` for a head that holds no prior.

# Returns

  - `state::ReturnsBufferState`: The context after the observations.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_context(cache::Option{<:ReturnsBufferState}, rd::ReturnsResult,
                      max_history::Option{<:Integer}, own_returns::Bool)
    state = isnothing(cache) ? ReturnsBufferState(; max_history = max_history) : cache
    return partial_fit!(state, rd; own_returns = own_returns)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The step of a host that holds a prior and a fold context: forward to the prior, then record the context.

The prior is folded first, because the context takes its cap from the buffer the prior seeds — a wrapper's `max_history` windows the prior's fit, and the context must drop the same rows at the same step so the read-out's carrier lines up with the prior's rows.

# Arguments

  - `host`: An estimator holding `pe` and `cache`.
  - `rd`: The carrier of the observations, `observations × assets`.

# Returns

  - `host`: The host with its prior folded and its context recorded.

# Related

  - [`fold_prior`](@ref)
  - [`fold_context`](@ref)
  - [`prior_returns_buffer`](@ref)
"""
function fold_returns(host, rd::ReturnsResult)
    pe = fold_prior(host.pe, rd)
    rows = prior_returns_buffer(pe)
    cache = fold_context(host.cache, rd, rows.max_history, false)
    return rebuild_estimator(host, (; pe = pe, cache = cache))
end
"""
    partial_fit!(opt::JuMPOptimisationEstimator, rd::ReturnsResult)
    partial_fit!(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity}, rd::ReturnsResult)
    partial_fit!(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling}, rd::ReturnsResult)
    partial_fit!(opt::Union{<:EqualWeighted, <:RandomWeighted}, rd::ReturnsResult)
    partial_fit!(opt::FiniteAllocationOptimisationEstimator, rd::ReturnsResult)
    partial_fit!(td::TD_OptE_Opt, rd::ReturnsResult)

Folds observations into an optimiser, without solving.

The optimiser's online step, decided by [#867](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/867): two verbs and one forward. This verb folds and returns the estimator; `optimise(opt)` with no returns reads the state out and solves once. The step forwards the observations to the **prior alone**, through [`fold_prior`](@ref), and records the rest of the carrier in a [`ReturnsBufferState`](@ref) so the read-out can rebuild the carrier the batch path reads. Everything above the prior — the clustering estimator, the constraint estimators, every uncertainty set, a meta-optimiser's inner optimisers — is untouched by the step and fitted by the read-out exactly as batch fits it.

The arity mirrors the batch verb: `optimise(opt, rd)` takes a carrier, so the step takes one, holding one observation or a block of them, and unpacks it on the way down to the prior's own arity. A JuMP head forwards to the [`JuMPOptimiser`](@ref) it holds and a hierarchical head to its [`HierarchicalOptimiser`](@ref); the bundle is the host of the prior and the context. The three meta-optimisers and [`InverseVolatility`](@ref) hold their prior directly and are their own host. [`EqualWeighted`](@ref) and [`RandomWeighted`](@ref) hold no prior but read the observations — each derives the Coverage Universe of its own window — so they are the bottom of their chain and their context keeps the rows itself.

Two families are refused by name. A finite allocation converts a weight vector and prices into share counts and reads no returns window, so it has no online step and needs none: `optimise(da, w, p)` is its whole verb. A [`TimeDependent`](@ref) schedule of optimisers is resolved by a fold loop before the loop steps, and a bare step has no fold to resolve it with.

The identity the step keeps is the seam's: after `t` observations, `optimise(opt)` equals `optimise(opt, rd[1:t])` — exactly for the carrier the read-out rebuilds, and to the moment layer's own tolerance for the weights.

# Arguments

  - `opt`: The optimiser to fold into.
  - `rd`: The carrier holding the observations to fold, `observations × assets`. A one-row carrier is one step of a walk-forward; a block is a warm-up.

# Validation

  - Everything [`fold_prior`](@ref) and [`ReturnsBufferState`](@ref)'s step refuse.

# Returns

  - `opt`: The optimiser, with its prior folded and its context recorded.

# Related

  - [`optimise`](@ref)
  - [`fold_prior`](@ref)
  - [`ReturnsBufferState`](@ref)
  - [`returns_result`](@ref)
  - [`update_online_estimator`](@ref)
"""
function partial_fit!(opt::JuMPOptimisationEstimator, rd::ReturnsResult)
    return rebuild_estimator(opt, (; opt = partial_fit!(opt.opt, rd)))
end
function partial_fit!(opt::Union{<:HierarchicalRiskParity,
                                 <:HierarchicalEqualRiskContribution,
                                 <:SchurComplementHierarchicalRiskParity},
                      rd::ReturnsResult)
    return rebuild_estimator(opt, (; opt = partial_fit!(opt.opt, rd)))
end
function partial_fit!(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                 <:InverseVolatility, <:NestedClustered, <:Stacking,
                                 <:SubsetResampling}, rd::ReturnsResult)
    return fold_returns(opt, rd)
end
function partial_fit!(opt::Union{<:EqualWeighted, <:RandomWeighted}, rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    max_history = isnothing(opt.cache) ? nothing : opt.cache.max_history
    return rebuild_estimator(opt,
                             (; cache = fold_context(opt.cache, rd, max_history, true)))
end
function partial_fit!(opt::FiniteAllocationOptimisationEstimator, ::ReturnsResult)
    return throw(ArgumentError("a `$(typeof(opt))` has no online step: a finite allocation converts a weight vector and prices into share counts and reads no returns window, so there is nothing to fold. Take the step on the optimiser that produces the weights, and allocate its read-out with `optimise(da, w, p)`."))
end
function partial_fit!(::TD_OptE_Opt, ::ReturnsResult)
    return throw(ArgumentError("a `TimeDependent` schedule of optimisers has no online step of its own: a fold loop resolves the schedule to the fold's optimiser before it steps, and a bare step has no fold to resolve it with. Resolve it with `reset_time_dependent_estimator`, or step inside a fold loop."))
end
"""
    online_state_seed(opt::Union{<:EqualWeighted, <:RandomWeighted}, max_history)
    online_state_seed(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling}, max_history)

Seeds the fold context an [`Online`](@ref) declares on a prior-less head, and refuses the wrapper on a host that holds a prior.

`Online(EqualWeighted(); max_history = w)` windows the rows the head keeps, so its seed is a [`ReturnsBufferState`](@ref) carrying the cap. A host that holds a prior takes its window from the prior — `Online` wraps the prior, and the context follows the prior's buffer — so wrapping the host itself would declare a second cap over the same rows, and it is refused by name.

# Arguments

  - `opt`: The estimator the wrapper wraps.
  - `max_history`: The wrapper's cap.

# Returns

  - `state::ReturnsBufferState`: The empty context to seed.

# Related

  - [`Online`](@ref)
  - [`ReturnsBufferState`](@ref)
  - [`update_online_estimator`](@ref)
"""
function online_state_seed(::Union{<:EqualWeighted, <:RandomWeighted},
                           max_history::Option{<:Integer})
    return ReturnsBufferState(; max_history = max_history)
end
function online_state_seed(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                      <:InverseVolatility, <:NestedClustered, <:Stacking,
                                      <:SubsetResampling}, ::Option{<:Integer})
    return throw(ArgumentError("`Online` wraps the prior, not the `$(typeof(opt).name.name)` that holds it: the optimiser's window is its prior's, so declare the step as `pe = Online(prior; max_history = …)` and the fold context follows the prior's buffer."))
end
"""
    update_online_member(pe::AbstractPriorEstimator)
    update_online_member(pe::Online)
    update_online_member(pe)

Resolves the [`Online`](@ref) declarations under a host's `pe` slot, and passes anything else through.

The slot admits a prior estimator, a prior result and a schedule. A wrapper is resolved; a prior estimator is scanned for wrappers in its own fields; a result and a schedule hold no wrapper to resolve and are returned as they are — the step refuses them by name later, where the reason is stated.

# Related

  - [`update_online_estimator`](@ref)
  - [`Online`](@ref)
"""
function update_online_member(pe::Union{<:AbstractPriorEstimator, <:Online})
    return update_online_estimator(pe)
end
function update_online_member(pe)
    return pe
end
"""
    update_online_estimator(opt::JuMPOptimisationEstimator)
    update_online_estimator(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})
    update_online_estimator(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})

Resolves the [`Online`](@ref) declarations an optimiser carries in its prior, at warm-up.

An optimiser hands its prior across a boundary of its own, so it writes the recursion the generic method does not: the JuMP and hierarchical heads recurse into the bundle they hold, and the bundle, the meta-optimisers and [`InverseVolatility`](@ref) recurse into `pe`. The forward reaches the prior alone, as the step does — a wrapper inside an inner optimiser of a meta-optimiser, or inside a fallback, is a batch configuration of that optimiser and is not resolved here.

# Arguments

  - `opt`: The optimiser.

# Returns

  - `opt`: The optimiser, with every wrapper under its prior resolved to an estimator carrying a seeded buffer.

# Related

  - [`Online`](@ref)
  - [`update_online_member`](@ref)
  - [`partial_fit!`](@ref)
"""
function update_online_estimator(opt::JuMPOptimisationEstimator)
    return rebuild_estimator(opt, (; opt = update_online_estimator(opt.opt)))
end
function update_online_estimator(opt::Union{<:HierarchicalRiskParity,
                                            <:HierarchicalEqualRiskContribution,
                                            <:SchurComplementHierarchicalRiskParity})
    return rebuild_estimator(opt, (; opt = update_online_estimator(opt.opt)))
end
function update_online_estimator(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                            <:InverseVolatility, <:NestedClustered,
                                            <:Stacking, <:SubsetResampling})
    return rebuild_estimator(opt, (; pe = update_online_member(opt.pe)))
end
"""
    returns_result(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
    returns_result(host::Union{<:EqualWeighted, <:RandomWeighted})
    returns_result(opt::JuMPOptimisationEstimator)
    returns_result(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})

Rebuilds the [`ReturnsResult`](@ref) of the observations an optimiser has folded.

The reconstitution verb of the read-out, on the optimiser: the rows come from the prior's buffer — or from the head's own, where it holds no prior — and every other column and the pinned context come from the [`ReturnsBufferState`](@ref) the host keeps. The result is a fresh carrier, equal field by field to the one a batch fit over the same observations would have read.

# Arguments

  - `host`: The optimiser, or the bundle it holds.

# Validation

  - The host carries a fold context. An `ArgumentError` is thrown otherwise.

# Returns

  - `rd::ReturnsResult`: The carrier of the observations folded so far.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`prior_returns_buffer`](@ref)
  - [`partial_fit!`](@ref)
  - [`optimise`](@ref)
"""
function returns_result(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                    <:InverseVolatility, <:NestedClustered, <:Stacking,
                                    <:SubsetResampling})
    return returns_result(partial_fit_cache(host), prior_returns_buffer(host.pe))
end
function returns_result(host::Union{<:EqualWeighted, <:RandomWeighted})
    state = partial_fit_cache(host)
    return returns_result(state, state.X)
end
function returns_result(opt::JuMPOptimisationEstimator)
    return returns_result(opt.opt)
end
function returns_result(opt::Union{<:HierarchicalRiskParity,
                                   <:HierarchicalEqualRiskContribution,
                                   <:SchurComplementHierarchicalRiskParity})
    return returns_result(opt.opt)
end
"""
    online_readout(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
    online_readout(host::Union{<:EqualWeighted, <:RandomWeighted})
    online_readout(opt::JuMPOptimisationEstimator)
    online_readout(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})
    online_readout(opt::FiniteAllocationOptimisationEstimator)
    online_readout(opt::OptimisationEstimator)

Turns a folded optimiser into the batch call that reads it out: an estimator and a carrier.

The read-out **reconstitutes the fold context and calls the ordinary batch path**, so nothing above the prior needs a method of its own. It rebuilds the carrier with [`returns_result`](@ref), swaps the folded prior for its read-out — a prior result, which the batch path does not refit, so the prior's fold is what the solve reads and the step stays quadratic in the assets — and drops the context, because the estimator it hands back is a batch configuration over that carrier. The state is never written: the read-out is a pure function of it, callable any number of times for the same answer, which is what lets the fallback chain walk unchanged.

The path is chosen by dispatch on the host's `cache`. A host carrying `nothing` has taken no step, and [`readout_without_state`](@ref) decides what that means: an optimiser whose prior is already a result is handed back with an empty carrier, which is the batch entry it has always had, and any other is refused by name. A finite allocation is refused by name, as its step is.

# Arguments

  - `opt`: The optimiser to read out.

# Returns

  - `(opt, rd)::Tuple`: The batch estimator and the carrier to run it over.

# Related

  - [`optimise`](@ref)
  - [`returns_result`](@ref)
  - [`partial_fit!`](@ref)
  - [`prior`](@ref)
"""
function online_readout(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                    <:InverseVolatility, <:NestedClustered, <:Stacking,
                                    <:SubsetResampling})
    if isnothing(host.cache)
        return readout_without_state(host, host.pe)
    end
    rd = returns_result(host)
    return rebuild_estimator(host, (; pe = prior(host.pe), cache = nothing)), rd
end
function online_readout(host::Union{<:EqualWeighted, <:RandomWeighted})
    if isnothing(host.cache)
        return readout_without_state(host, nothing)
    end
    rd = returns_result(host)
    return rebuild_estimator(host, (; cache = nothing)), rd
end
function online_readout(opt::JuMPOptimisationEstimator)
    host, rd = online_readout(opt.opt)
    return rebuild_estimator(opt, (; opt = host)), rd
end
function online_readout(opt::Union{<:HierarchicalRiskParity,
                                   <:HierarchicalEqualRiskContribution,
                                   <:SchurComplementHierarchicalRiskParity})
    host, rd = online_readout(opt.opt)
    return rebuild_estimator(opt, (; opt = host)), rd
end
"""
    readout_without_state(host, pe::AbstractPriorResult)
    readout_without_state(host, pe)

The fold-less entry of a host that has taken no step.

A host whose prior is already a result needs no returns to solve, and is handed back with an empty carrier: that is the batch entry `optimise(opt)` has always had. Any other host has nothing to answer from — no state and no prior result — and is refused by name, pointing at the step that fills the state and at the batch verb that takes the returns.

# Related

  - [`online_readout`](@ref)
  - [`partial_fit!`](@ref)
"""
function readout_without_state(host, ::AbstractPriorResult)
    return host, ReturnsResult()
end
function readout_without_state(host, ::Any)
    return throw(ArgumentError("`optimise(opt)` with no returns reads the state the online step wrote, and this `$(typeof(host).name.name)` has taken no step: its `cache` is `nothing`. Fold observations with `partial_fit!(opt, rd)` first, or pass the returns to `optimise(opt, rd)`."))
end
function online_readout(opt::FiniteAllocationOptimisationEstimator)
    return throw(ArgumentError("a `$(typeof(opt))` has no online read-out, because it has no online step: a finite allocation reads no returns window. Allocate a weight vector with `optimise(da, w, p)`."))
end
function online_readout(opt::OptimisationEstimator)
    return opt, ReturnsResult()
end
"""
    optimise(opt::OptimisationEstimator; kwargs...)

Reads a folded optimiser out: rebuilds the carrier from the state, and solves once over it.

The read-out verb of the online step, and the fold-less entry of every optimiser. With no returns to fit, an optimiser answers from what it holds: a prior that is already a result, which is the batch configuration this entry has always served, or the state its [`partial_fit!`](@ref) steps wrote, which [`online_readout`](@ref) turns into a batch estimator and the carrier of the observations folded so far. The ordinary `optimise(opt, rd)` then runs — the constraint estimators, the clustering, every uncertainty set and every inner optimiser fitted from that carrier exactly as batch fits them — and the fallback chain walks unchanged, because the read-out is pure and a failed solve leaves the state where the last fold put it.

```julia
mr = MeanRisk(; opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv))
mr = update_online_estimator(mr)          # warm-up: seeds any Online buffer
for t in 1:warmup
    mr = partial_fit!(mr, port_opt_view(rd, t:t, :))   # folds; solves nothing
end
res = optimise(mr)                        # reads out; solves once
```

# Arguments

  - `opt`: The optimiser to read out.
  - `kwargs...`: Keyword arguments of the batch verb — `dims`, `str_names`, `save` — forwarded to it.

# Returns

  - `res::OptimisationResult`: The result the batch verb gives over the observations folded so far.

# Related

  - [`partial_fit!`](@ref)
  - [`online_readout`](@ref)
  - [`returns_result`](@ref)
  - [`update_online_estimator`](@ref)
"""
function optimise(opt::OptimisationEstimator; kwargs...)
    opt, rd = online_readout(opt)
    return optimise(opt, rd; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a host of the online step except its `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the optimiser at every site that renders one. Set `set_show_nothing_fields!` for the type to render it. ADR 0105 records the decision.

# Arguments

  - `opt`: The host.

# Returns

  - `fields::Tuple`: Every field name but `cache`.

# Related

  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                <:InverseVolatility, <:EqualWeighted, <:RandomWeighted,
                                <:NestedClustered, <:Stacking, <:SubsetResampling})
    return filter(!=(:cache), fieldnames(typeof(opt)))
end
