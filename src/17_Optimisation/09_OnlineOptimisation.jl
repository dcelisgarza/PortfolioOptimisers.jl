"""
    returns_buffer(state::SampleBufferState)
    returns_buffer(state::PriorCarryState)
    returns_buffer(state::AbstractPartialFitState)

Reads the buffer of asset returns out of the state that a prior carries.

Two states hold rows at the prior layer, each in a different place. A [`SampleBufferState`](@ref) is the rows, and a [`PriorCarryState`](@ref) keeps them in `buf`. The read-out of an optimiser reads the returns through this function and never through the fields of a state, so a new state that holds rows adds one method here. Where the tree of the prior reads factor returns, the buffer holds the factor rows too, and the read-out takes them with [`factor_buffer`](@ref) when the fold context keeps no factor column. This function refuses, by name, a state that holds no rows, such as the exact-fold state of a moment estimator.

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
    return throw(ArgumentError("a `$(typeof(state))` carries no rows, so an optimiser cannot rebuild the returns it folded from it. The read-out reads the observations from the prior's own buffer, which a `SampleBufferState` or a `PriorCarryState` holds."))
end
"""
    prior_returns_buffer(pe::AbstractPriorEstimator)
    prior_returns_buffer(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior})

Reads the buffer of asset returns out of the prior that an optimiser forwarded its observations to.

One prior in the chain owns the rows. [`HighOrderPriorEstimator`](@ref) and [`BlackLittermanPrior`](@ref) build their result around the prior they embed and own no rows, so this function reads the buffer of the embedded prior, at any depth. Every other prior keeps its rows in its own `cache`.

# Arguments

  - `pe`: The prior estimator that the optimiser forwards to.

# Validation

  - The prior that owns the rows carries a state. [`partial_fit_cache`](@ref) throws an `ArgumentError` otherwise.
  - Everything [`returns_buffer`](@ref) refuses.

# Returns

  - `buffer::SampleBufferState`: The rows that the optimiser has folded, and the masks beside them.

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

Reads the active mask of the Asset Panel that a step carries to the prior, and refuses what the step cannot carry.

A carrier with no panel, or with a static one, gives no mask, because the fold context pins a static panel. A time-varying panel that holds masks alone, which is the shape the ingestion layer writes, gives its active mask, and the prior folds the mask into its buffer beside the rows. This function refuses a time-varying panel that holds Panel Fields. The rows of a time-varying field are sample, and the buffers of the step hold numbers alone, so the step has no place to keep a categorical or a tensor field, and the read-out could not rebuild the panel.

The estimation mask does not travel the step. The exact folds of the moment layer take an active mask and no estimation mask, so a step whose estimation universe is narrower than its active universe would fit the wrong universe without a word. The step refuses such a panel by name.

# Algorithm

 1. Refuse a carrier that holds an implied-volatility surface.
 2. Return `nothing` when the carrier holds no panel, or when its panel is static.
 3. Refuse a time-varying panel that holds a Panel Field.
 4. Refuse a time-varying panel whose estimation mask differs from its active mask.
 5. Return the active mask `pnl.amsk`.

# Arguments

  - `rd`: The carrier of the observations to fold.

# Validation

  - `rd.iv` is `nothing`. An `ArgumentError` is thrown otherwise, because the step of a prior takes no implied-volatility surface.
  - A time-varying `rd.pnl` holds no Panel Field. An `ArgumentError` is thrown otherwise.
  - A time-varying `rd.pnl` has an estimation mask equal to its active mask. An `ArgumentError` is thrown otherwise.

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

Forwards the observations of a carrier to the prior, in the arity of the prior's own step.

This is the one forward that the step of an optimiser makes. An optimiser forwards each observation to its prior and to nothing else. The read-out fits every other member in batch, from the fold context it rebuilds. The carrier is unpacked as [`prior`](@ref) unpacks it in batch: `rd.X` and `rd.F` pass unchanged, and the active mask of a time-varying panel passes as the `active_mask` keyword. The tree of the prior decides what to do with `F`, as its batch function does.

A prior that is already an [`AbstractPriorResult`](@ref) has no state to fold into. It is batch configuration, and an optimiser that holds one runs `optimise(opt, rd)`. This function also refuses a [`TimeDependent`](@ref) schedule on the prior. A schedule swaps the estimator that carries the state, and a member that never saw the folded rows cannot take them over.

# Algorithm

 1. Refuse a carrier with no `X`.
 2. Refuse a carrier with no `F` when the tree of `pe` requires one, through [`needs_factor_returns`](@ref).
 3. Read the active mask `amsk` with [`step_active_mask`](@ref).
 4. Fold `rd.X` and `rd.F` into `pe` with [`partial_fit!`](@ref), passing `amsk` as `active_mask` when it is not `nothing`.

# Arguments

  - `pe`: The prior that the optimiser holds.
  - `rd`: The carrier of the observations to fold, `observations × assets`.

# Validation

  - `pe` is an [`AbstractPriorEstimator`](@ref). An `ArgumentError` is thrown for a prior result or a schedule.
  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - `rd.F` is not `nothing` when `needs_factor_returns(pe) === true`. An `IsNothingError` is thrown otherwise.
  - Everything [`step_active_mask`](@ref) refuses.

# Returns

  - `pe`: The prior, with the observations folded into its state.

# Related

  - [`partial_fit!`](@ref)
  - [`step_active_mask`](@ref)
  - [`needs_factor_returns`](@ref)
  - [`prior`](@ref)
  - [`assert_online_entry`](@ref): refuses a schedule on the prior before the warm-up of the fold loop.
"""
function fold_prior(pe::AbstractPriorEstimator, rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    assert_factor_returns(pe, rd.F)
    amsk = step_active_mask(rd)
    kw = isnothing(amsk) ? (;) : (; active_mask = amsk)
    return partial_fit!(pe, rd.X, rd.F; kw...)
end
function fold_prior(pe::AbstractPriorResult, ::ReturnsResult)
    return throw(ArgumentError("`pe` holds a fitted `$(typeof(pe))`, which has no state to fold an observation into: a prior result is batch configuration. Hand the optimiser the prior estimator to take the online step, or run `optimise(opt, rd)`."))
end
function fold_prior(pe::TimeDependent, ::ReturnsResult)
    return throw(ArgumentError("`pe` holds a `TimeDependent` schedule of priors, and the online step cannot fold into a schedule: it swaps the estimator that carries the state, and a member that never saw the folded rows cannot be handed them. No loop resolves a schedule before stepping — a schedule reaches stateless fields only. Hold one prior in `pe`, and schedule a field that carries no state."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observations of a carrier into the fold context of a host, and seeds the context on the first step.

# Algorithm

 1. Take `cache` as the context `state`. When `cache` is `nothing`, seed an empty [`ReturnsBufferState`](@ref) with the cap `max_history` instead.
 2. Fold `rd` into `state` with [`partial_fit!`](@ref), under `own_returns` and `own_factors`.

# Arguments

  - `cache`: The context that the host carries, or `nothing` before the first step.
  - `rd`: The carrier of the observations.
  - `max_history`: The cap of a new context. It is the cap of the buffer that holds the returns, so the context and that buffer drop the same rows.
  - `own_returns`: Whether the context keeps the returns itself. It is `true` for a head that holds no prior.
  - `own_factors`: Whether the context keeps the factor column itself. It is `true` for a head that holds no prior, and for a host whose prior never reads factor returns.

# Validation

  - Everything the step of [`ReturnsBufferState`](@ref) refuses.

# Returns

  - `state::ReturnsBufferState`: The context after the observations.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_context(cache::Option{<:ReturnsBufferState}, rd::ReturnsResult,
                      max_history::Option{<:Integer}, own_returns::Bool, own_factors::Bool)
    state = isnothing(cache) ? ReturnsBufferState(; max_history = max_history) : cache
    return partial_fit!(state, rd; own_returns = own_returns, own_factors = own_factors)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Takes the step of a host that holds a prior and a fold context. It forwards the observations to the prior, then records the carrier in the context.

The prior folds first, because the context takes its cap from the buffer of the prior. The `max_history` of an [`Online`](@ref) wrapper caps the rows of the prior, and the context must drop the same rows at the same step, so the carrier that the read-out rebuilds matches the rows of the prior.

One place holds `F`. The context keeps the factor column only when the tree of the prior never reads it, that is, when [`needs_factor_returns`](@ref) answers `false`. Otherwise the buffer of the prior records it, and the read-out reads it back through [`prior_returns_buffer`](@ref), as it reads the returns.

# Algorithm

 1. Fold `rd` into `host.pe` with [`fold_prior`](@ref), giving the folded prior `pe`.
 2. Read the buffer `rows` of `pe` with [`prior_returns_buffer`](@ref).
 3. Set `own_factors` to `true` when `needs_factor_returns(pe) === false`.
 4. Fold `rd` into `host.cache` with [`fold_context`](@ref), under the cap `rows.max_history`, giving `cache`. The context never keeps the returns.
 5. Rebuild `host` with `pe` and `cache`.

# Arguments

  - `host`: An estimator that holds `pe` and `cache`.
  - `rd`: The carrier of the observations, `observations × assets`.

# Validation

  - Everything [`fold_prior`](@ref) and [`fold_context`](@ref) refuse.

# Returns

  - `host`: The host, with its prior folded and its context recorded.

# Related

  - [`fold_prior`](@ref)
  - [`fold_context`](@ref)
  - [`prior_returns_buffer`](@ref)
  - [`needs_factor_returns`](@ref)
"""
function fold_returns(host, rd::ReturnsResult)
    pe = fold_prior(host.pe, rd)
    rows = prior_returns_buffer(pe)
    own_factors = needs_factor_returns(pe) === false
    cache = fold_context(host.cache, rd, rows.max_history, false, own_factors)
    return rebuild_estimator(host, (; pe = pe, cache = cache))
end
"""
    partial_fit!(opt::JuMPOptimisationEstimator, rd::ReturnsResult)
    partial_fit!(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity}, rd::ReturnsResult)
    partial_fit!(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling}, rd::ReturnsResult)
    partial_fit!(opt::Union{<:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio}, rd::ReturnsResult)
    partial_fit!(opt::PreviousWeights, rd::ReturnsResult)
    partial_fit!(opt::FiniteAllocationOptimisationEstimator, rd::ReturnsResult)
    partial_fit!(td::TD_OptE_Opt, rd::ReturnsResult)

Folds observations into an optimiser, and solves nothing.

The online step of an optimiser has two functions. `partial_fit!(opt, rd)` folds the observations and returns the estimator, and `optimise(opt)` with no returns reads the state out and solves once. The step forwards the observations to the prior alone, through [`fold_prior`](@ref), and records the rest of the carrier in a [`ReturnsBufferState`](@ref), so the read-out can rebuild the carrier that the batch path reads. The step does not change the members above the prior: the clustering estimator, the constraint estimators, the uncertainty sets, and the inner optimisers of a meta-optimiser. The read-out fits each of them as a batch run does.

The step takes a carrier, as `optimise(opt, rd)` does. The carrier holds one observation or a block of them, and the step unpacks it into the arity of the prior's step. A JuMP head forwards to the [`JuMPOptimiser`](@ref) it holds, and a hierarchical head to its [`HierarchicalOptimiser`](@ref). The optimiser that holds the prior and the context is the host. The three meta-optimisers and [`InverseVolatility`](@ref) hold their prior directly, so each is its own host. [`EqualWeighted`](@ref), [`RandomWeighted`](@ref) and [`BestConstantRebalancedPortfolio`](@ref) hold no prior but read the observations, so their context keeps the rows itself. [`PreviousWeights`](@ref) reads no observation, so its step returns it unchanged and it carries no context.

After `t` observations, `optimise(opt)` equals `optimise(opt, rd[1:t])`. The carrier that the read-out rebuilds equals the batch carrier field by field. The weights of a family that solves no programme agree to rounding, and the weights of a JuMP family agree to the tolerance of its solver.

# Algorithm

 1. A JuMP head or a hierarchical head steps the bundle it holds in `opt`, and rebuilds itself around the result.
 2. A host that holds a prior steps through [`fold_returns`](@ref).
 3. A head that holds no prior checks that `rd.X` is present. It then folds `rd` into its own context with [`fold_context`](@ref), and the context keeps the returns and the factor column.
 4. [`PreviousWeights`](@ref) returns `opt` unchanged.

# Arguments

  - `opt`: The optimiser to fold into.
  - `rd`: The carrier of the observations to fold, `observations × assets`. A carrier of one row is one step of a walk-forward, and a block is a warm-up.

# Validation

  - `opt` is not a finite allocation. An `ArgumentError` is thrown otherwise, because a finite allocation converts a weight vector and prices into share counts and reads no returns window.
  - `opt` is not a [`TimeDependent`](@ref) schedule of optimisers. An `ArgumentError` is thrown otherwise, because a schedule swaps the optimiser that carries the state.
  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - Everything [`fold_prior`](@ref) and the step of [`ReturnsBufferState`](@ref) refuse.

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
function partial_fit!(opt::Union{<:EqualWeighted, <:RandomWeighted,
                                 <:BestConstantRebalancedPortfolio}, rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    max_history = isnothing(opt.cache) ? nothing : opt.cache.max_history
    return rebuild_estimator(opt,
                             (;
                              cache = fold_context(opt.cache, rd, max_history, true, true)))
end
function partial_fit!(opt::PreviousWeights, ::ReturnsResult)
    return opt
end
function partial_fit!(opt::FiniteAllocationOptimisationEstimator, ::ReturnsResult)
    return throw(ArgumentError("a `$(typeof(opt))` has no online step: a finite allocation converts a weight vector and prices into share counts and reads no returns window, so there is nothing to fold. Take the step on the optimiser that produces the weights, and allocate its read-out with `optimise(da, w, p)`."))
end
function partial_fit!(::TD_OptE_Opt, ::ReturnsResult)
    return throw(ArgumentError("a `TimeDependent` schedule of optimisers has no online step: a schedule swaps the optimiser that carries the state, and no loop resolves a schedule before stepping — a schedule reaches stateless fields only. Step one optimiser, and schedule a field that carries no state."))
end
"""
    online_state_seed(opt::Union{<:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio}, max_history)
    online_state_seed(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling}, max_history)

Seeds the fold context that an [`Online`](@ref) wrapper declares on a head with no prior, and refuses the wrapper on a host that holds a prior.

`Online(EqualWeighted(); max_history = w)` caps the rows that the head keeps, so its seed is an empty [`ReturnsBufferState`](@ref) with the cap `w`. A host that holds a prior takes its cap from the prior. `Online` wraps the prior, and the context follows the buffer of the prior, so a wrapper on the host would declare a second cap over the same rows.

# Arguments

  - `opt`: The estimator that the wrapper wraps.
  - `max_history`: The cap of the wrapper.

# Validation

  - `opt` holds no prior. An `ArgumentError` is thrown otherwise, and it tells the caller to wrap the prior.

# Returns

  - `state::ReturnsBufferState`: The empty context to seed.

# Related

  - [`Online`](@ref)
  - [`ReturnsBufferState`](@ref)
  - [`update_online_estimator`](@ref)
"""
function online_state_seed(::Union{<:EqualWeighted, <:RandomWeighted,
                                   <:BestConstantRebalancedPortfolio},
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

Resolves the [`Online`](@ref) wrappers in the `pe` slot of a host, and returns any other value unchanged.

The slot takes a prior estimator, a prior result or a schedule. This function resolves a wrapper, and it searches a prior estimator for wrappers in its own fields. A prior result and a schedule hold no wrapper to resolve, so they pass unchanged, and the step refuses them later with the reason.

# Related

  - [`update_online_estimator`](@ref)
  - [`Online`](@ref)
  - [`fold_prior`](@ref): refuses a prior result and a schedule at the step.
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

Resolves the [`Online`](@ref) wrappers in the prior of an optimiser, at the warm-up.

The JuMP and hierarchical heads resolve the bundle they hold, and the bundle, the meta-optimisers and [`InverseVolatility`](@ref) resolve their `pe`. The step forwards observations to that prior alone, so this function resolves no other wrapper. A wrapper in an inner optimiser of a meta-optimiser, in a fallback, or in any other field stays unresolved, and the entry of the online arm of the fold loop refuses it by name.

# Arguments

  - `opt`: The optimiser.

# Validation

  - Everything [`online_state_seed`](@ref) refuses.

# Returns

  - `opt`: The optimiser, with each wrapper in its prior resolved to an estimator that carries a seeded buffer.

# Related

  - [`Online`](@ref)
  - [`update_online_member`](@ref)
  - [`online_unreached_path`](@ref)
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
    online_unreached_path(opt::Union{<:JuMPOptimisationEstimator, <:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity, <:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
    online_unreached_path(opt)

Names the first [`Online`](@ref) wrapper in an optimiser that the warm-up leaves unresolved, or answers `nothing`.

The warm-up of an optimiser resolves the wrappers in its prior alone, through [`update_online_estimator`](@ref). A wrapper in any other field, such as an inner optimiser of a meta-optimiser or a fallback, never gets a seeded buffer, and the batch fit of the read-out refuses it. This function runs the warm-up and names the wrapper that is left, so the entry of the online arm can refuse it before the first fold. Any other estimator answers `nothing`.

# Algorithm

 1. Resolve the wrappers of `opt` with [`update_online_estimator`](@ref).
 2. Return the dotted path of the first wrapper in the result, from [`online_wrapper_path`](@ref).

# Arguments

  - `opt`: The optimiser that enters the online arm.

# Validation

  - Everything [`update_online_estimator`](@ref) refuses.

# Returns

  - `path::Option{<:String}`: The dotted path of the unresolved wrapper, or `nothing`.

# Related

  - [`assert_online_entry`](@ref)
  - [`update_online_estimator`](@ref)
  - [`online_wrapper_path`](@ref)
  - [`assert_batch_entry`](@ref): the refusal that the read-out would reach later.
"""
function online_unreached_path(opt::Union{<:JuMPOptimisationEstimator,
                                          <:HierarchicalRiskParity,
                                          <:HierarchicalEqualRiskContribution,
                                          <:SchurComplementHierarchicalRiskParity,
                                          <:JuMPOptimiser, <:HierarchicalOptimiser,
                                          <:InverseVolatility, <:NestedClustered,
                                          <:Stacking, <:SubsetResampling})
    return online_wrapper_path(update_online_estimator(opt))
end
function online_unreached_path(::Any)
    return nothing
end
"""
    assert_stateless_schedule(opt::JuMPOptimisationEstimator)
    assert_stateless_schedule(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})
    assert_stateless_schedule(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
    assert_stateless_schedule(opt)

Refuses a [`TimeDependent`](@ref) schedule on a field that carries a state, by name.

A schedule replaces the value of its field at every fold, and the loop threads the state through that value. The replacement drops the state, so the value that a fold receives never saw the rows folded before it. The one such field that a schedule can reach is the `pe` of a host, whose type bound admits a schedule. The `opt` field of a JuMP head or of a hierarchical head holds the bundle, and its bound refuses a schedule at construction, so the heads only recurse. This function walks the route of [`update_online_estimator`](@ref). A schedule on any other field passes, the inner optimisers of a meta-optimiser included. The loop resolves such a schedule on the copy it fits at each fold, and the state stays on the estimator it threads. No schedule carries a state across a swap.

# Arguments

  - `opt`: The estimator that enters the online arm of the fold loop.

# Validation

  - Everything [`assert_stateless_prior`](@ref) refuses.

# Related

  - [`assert_online_entry`](@ref)
  - [`update_online_estimator`](@ref)
  - [`TimeDependent`](@ref)
"""
function assert_stateless_schedule(opt::JuMPOptimisationEstimator)
    return assert_stateless_schedule(opt.opt)
end
function assert_stateless_schedule(opt::Union{<:HierarchicalRiskParity,
                                              <:HierarchicalEqualRiskContribution,
                                              <:SchurComplementHierarchicalRiskParity})
    return assert_stateless_schedule(opt.opt)
end
function assert_stateless_schedule(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                              <:InverseVolatility, <:NestedClustered,
                                              <:Stacking, <:SubsetResampling})
    return assert_stateless_prior(opt.pe, opt)
end
function assert_stateless_schedule(::Any)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses the prior of a host when it is a [`TimeDependent`](@ref) schedule, and passes any other value.

# Arguments

  - `pe`: The prior of the host.
  - `host`: The host, which the message names.

# Validation

  - `pe` is not a [`TimeDependent`](@ref). An `ArgumentError` is thrown otherwise.

# Related

  - [`assert_stateless_schedule`](@ref)
"""
function assert_stateless_prior(::TimeDependent, host)
    return throw(ArgumentError("`$(typeof(host).name.name).pe` holds a `TimeDependent` schedule, and the online arm of the fold loop cannot step it: `pe` carries the partial-fit state the loop threads from fold to fold, and a schedule replaces that value every fold, so the prior a fold is handed never saw the rows folded before it. A schedule reaches stateless fields only. Hold one prior in `pe`, and schedule a field that carries no state, or refit every fold with `ff = nothing`."))
end
function assert_stateless_prior(::Any, ::Any)
    return nothing
end
function online_entry_state(::TimeDependent)
    # A schedule's entries are batch configuration, resolved per fold; the loop threads no
    # state through them, so a state one of them carries is not a state at entry.
    return nothing
end
function online_wrapper_path(::TimeDependent)
    # A wrapper among a schedule's entries, or as its default, is refused at construction.
    return nothing
end
"""
    assert_online_entry(est::TimeDependent)
    assert_online_entry(est)

Refuses, at the entry of the online arm of the fold loop, an estimator that is not the configuration alone.

The loop starts cold. It reads its argument as configuration, as the batch loop does, where `prior(pe, X)` ignores any state that `pe` carries. The warm-up folds the first training window into an estimator that has folded nothing, so a state at entry would count its rows twice. [`Resume`](@ref) is the entry for an estimator that carries a state. Every refusal runs before the first solve.

# Algorithm

 1. Refuse a [`TimeDependent`](@ref) schedule of optimisers at the root.
 2. Refuse a schedule on the `pe` of a host, through [`assert_stateless_schedule`](@ref).
 3. Find the first state in the tree of `est` with [`online_entry_state`](@ref), and refuse `est` when there is one.
 4. Find the first wrapper that the warm-up leaves unresolved with [`online_unreached_path`](@ref), and refuse `est` when there is one.

# Arguments

  - `est`: The estimator that the loop takes.

# Validation

  - `est` is not a `TimeDependent`. An `ArgumentError` is thrown otherwise, because the loop threads one estimator and a schedule is a different one at each fold.
  - No `pe` on the route of the step holds a `TimeDependent`. An `ArgumentError` is thrown otherwise.
  - No `cache` in the tree of `est` holds a state. An `ArgumentError` that names the field is thrown otherwise.
  - Every [`Online`](@ref) wrapper in `est` sits where the warm-up resolves it. An `ArgumentError` that names the field is thrown otherwise.

# Related

  - [`online_folds`](@ref)
  - [`assert_stateless_schedule`](@ref)
  - [`online_entry_state`](@ref)
  - [`online_unreached_path`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
  - [`Resume`](@ref)
"""
function assert_online_entry(::TimeDependent)
    return throw(ArgumentError("the online arm of the fold loop takes one estimator and threads it from fold to fold, and a `TimeDependent` schedule of optimisers is a different one per fold, so it has no state to thread. A schedule reaches stateless fields only. Step one optimiser, and schedule a field that carries no state, or refit every fold with `ff = nothing`."))
end
"""
    assert_online_fee_source(est, pws)

Refuses, at the entry of the online arm of the fold loop, an estimator whose fee needs a Previous-Weights Source that the scheme does not give.

This method passes every estimator. The one family that refuses, [`OnlinePortfolioSelection`](@ref), adds its own method.

# Related

  - [`online_folds`](@ref)
  - [`assert_online_entry`](@ref)
"""
function assert_online_fee_source(::Any, ::Any)::Nothing
    return nothing
end
function assert_online_entry(est)
    assert_stateless_schedule(est)
    path = online_entry_state(est)
    @argcheck(isnothing(path),
              ArgumentError("`$(typeof(est).name.name)` enters the online arm of the fold loop carrying a partial-fit state at `$(path)`, and the loop starts cold: its argument is the configuration alone, and the warm-up folds the first training window into an estimator that has folded nothing. Hand the loop the estimator with every `cache` at `nothing`, or read the stepped estimator out by hand with `fit_and_predict(opt, rd; test_idx)`."))
    wpath = online_unreached_path(est)
    @argcheck(isnothing(wpath),
              ArgumentError("`$(typeof(est).name.name)` enters the online arm of the fold loop holding an `Online` at `$(wpath)`, where the warm-up does not resolve it. The online step forwards each observation to the prior of the optimiser alone, so a wrapper in any other field never gets a buffer, and the batch fit of the read-out would refuse it. Move the wrapper to the prior of the optimiser, or set `$(wpath)` to the estimator it wraps, and the read-out refits it from the folded rows."))
    return nothing
end
"""
    returns_result(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
    returns_result(host::Union{<:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio})
    returns_result(opt::JuMPOptimisationEstimator)
    returns_result(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})

Rebuilds the [`ReturnsResult`](@ref) of the observations that an optimiser has folded.

This is the rebuild step of the read-out. The rows come from the buffer of the prior, or from the context of a head that holds no prior. Every other column and the pinned context come from the [`ReturnsBufferState`](@ref) of the host. The factor column comes from the context when the tree of the prior never reads it, and from the buffer of the prior otherwise. The result is a new carrier, equal field by field to the carrier that a batch fit over the same observations reads.

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
function returns_result(host::Union{<:EqualWeighted, <:RandomWeighted,
                                    <:BestConstantRebalancedPortfolio})
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
    held_timestamps(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling, <:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio})
    held_timestamps(opt::JuMPOptimisationEstimator)
    held_timestamps(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})
    held_timestamps(::PreviousWeights)

Reads the timestamps that the fold context of an optimiser holds, or answers `nothing` when it holds none.

The alignment check of [`Resume`](@ref) reads this accessor. The [`ReturnsBufferState`](@ref) keeps the timestamps of the folded observations and drops the oldest under a cap, so under `Online(pe; max_history = w)` it holds the last `w`. A head forwards to the bundle it holds, and a host reads its own context, as in [`returns_result`](@ref). A host that has taken no step answers `nothing`. [`PreviousWeights`](@ref) reads no observation and keeps no context, so it answers `nothing`, and a resume refuses it.

# Arguments

  - `opt`: The stepped optimiser.

# Returns

  - `ts::Option{<:AbstractVector}`: The held timestamps, in order, or `nothing`.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`returns_result`](@ref)
  - [`resume_fold_count`](@ref)
  - [`Resume`](@ref)
  - [`Pipeline`](@ref): its method reads the state that the row owner keeps.
"""
function held_timestamps(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                     <:InverseVolatility, <:NestedClustered, <:Stacking,
                                     <:SubsetResampling, <:EqualWeighted, <:RandomWeighted,
                                     <:BestConstantRebalancedPortfolio})
    return context_timestamps(host.cache)
end
function held_timestamps(opt::JuMPOptimisationEstimator)
    return held_timestamps(opt.opt)
end
function held_timestamps(opt::Union{<:HierarchicalRiskParity,
                                    <:HierarchicalEqualRiskContribution,
                                    <:SchurComplementHierarchicalRiskParity})
    return held_timestamps(opt.opt)
end
function held_timestamps(::PreviousWeights)
    return nothing
end
"""
    online_readout(host::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
    online_readout(host::Union{<:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio})
    online_readout(opt::JuMPOptimisationEstimator)
    online_readout(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})
    online_readout(opt::FiniteAllocationOptimisationEstimator)
    online_readout(opt::OptimisationEstimator)

Turns a folded optimiser into the batch call that reads it out, an estimator and a carrier.

The read-out rebuilds the carrier and calls the ordinary batch path, so no member above the prior needs a method of its own. The prior becomes its read-out, a prior result, which the batch path does not refit, so the solve reads the folded prior. The read-out drops the context, because the estimator it returns is a batch configuration over the rebuilt carrier. It never writes the state, so two read-outs of one estimator give the same answer, and a fallback reads the same carrier as the optimiser it replaces.

The `cache` of the host selects the branch. A host whose `cache` is `nothing` has taken no step, and [`readout_without_state`](@ref) answers for it. This function returns any other optimiser, such as [`PreviousWeights`](@ref), with an empty carrier, and its batch function decides whether it answers without returns.

# Algorithm

 1. A JuMP head or a hierarchical head reads out the bundle it holds in `opt`, and rebuilds itself around the bundle.
 2. A host with no state goes to [`readout_without_state`](@ref).
 3. Rebuild the carrier `rd` with [`returns_result`](@ref).
 4. Rebuild the host with `cache = nothing`. A host that holds a prior also replaces `pe` with `prior(pe)`, the result of the folded prior.
 5. Return the host and `rd`.

# Arguments

  - `opt`: The optimiser to read out.

# Validation

  - `opt` is not a finite allocation. An `ArgumentError` is thrown otherwise, because a finite allocation has no step.
  - Everything [`readout_without_state`](@ref) and [`returns_result`](@ref) refuse.

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
function online_readout(host::Union{<:EqualWeighted, <:RandomWeighted,
                                    <:BestConstantRebalancedPortfolio})
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

Answers for a host that has taken no step.

A host whose prior is already a result needs no returns to solve. This function returns it with an empty carrier, which is the batch entry that `optimise(opt)` has always had. Any other host has no state and no prior result to answer from, and this function refuses it by name. The message names the step that fills the state and the batch function that takes the returns.

# Validation

  - `pe` is an [`AbstractPriorResult`](@ref). An `ArgumentError` is thrown otherwise.

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

Reads out a folded optimiser. It rebuilds the carrier from the state and solves once over it.

This is the read-out of the online step, and the entry of every optimiser that takes no returns. The optimiser answers from what it holds. That is a prior that is already a result, which this entry has always served, or the state that its [`partial_fit!`](@ref) steps wrote. The ordinary `optimise(opt, rd)` then runs over the rebuilt carrier, and it fits the constraint estimators, the clustering, the uncertainty sets and the inner optimisers as a batch run does. A failed solve changes no state, so the fallback chain runs as in batch.

```julia
mr = MeanRisk(; opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv))
for t in 1:warmup
    mr = partial_fit!(mr, port_opt_view(rd, t:t, :))   # folds; solves nothing
end
res = optimise(mr)                                     # reads out; solves once
```

# Algorithm

 1. Turn `opt` into a batch estimator and a carrier `rd` with [`online_readout`](@ref).
 2. Run `optimise(opt, rd; kwargs...)` on them.

# Arguments

  - `opt`: The optimiser to read out.
  - `kwargs...`: The keyword arguments of the batch function, `dims`, `str_names` and `save`, passed on unchanged.

# Validation

  - Everything [`online_readout`](@ref) refuses.

# Returns

  - `res::OptimisationResult`: The result that the batch function gives over the observations folded so far.

# Related

  - [`partial_fit!`](@ref)
  - [`online_readout`](@ref)
  - [`returns_result`](@ref)
  - [`update_online_estimator`](@ref): resolves an [`Online`](@ref) prior before the first step.
"""
function optimise(opt::OptimisationEstimator; kwargs...)
    opt, rd = online_readout(opt)
    return optimise(opt, rd; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a host of the online step except its `cache`.

The `cache` holds the state of an incremental fit, not the configuration that a reader looks up, and it would print under every optimiser that holds it. `set_show_nothing_fields!(:Name, true)` renders it for the type `Name`.

# Arguments

  - `opt`: The host.

# Returns

  - `fields::Tuple`: Every field name except `cache`.

# Related

  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                <:InverseVolatility, <:EqualWeighted, <:RandomWeighted,
                                <:BestConstantRebalancedPortfolio, <:NestedClustered,
                                <:Stacking, <:SubsetResampling})
    return filter(!=(:cache), fieldnames(typeof(opt)))
end
