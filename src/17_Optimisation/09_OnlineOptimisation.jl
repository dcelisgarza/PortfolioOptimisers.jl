"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the active mask of the Asset Panel that a step carries to an exact fold that is not a prior, and refuses what that fold cannot carry.

The step of a covariance estimator in [`covariance_forecast_evaluation`](@ref) and the step of an online portfolio selection read the mask through this function. A prior does not: it folds through [`partial_fit!`](@ref) over the `ReturnsResult`, whose route decides what the step honours. A [`ReturnsResult`](@ref) with no panel, or with a static one, gives no mask, because the fold context pins a static panel. A time-varying panel that holds masks alone, which is the shape the ingestion layer writes, gives its active mask, and the fold records the mask beside the rows. This function refuses a time-varying panel that holds Panel Fields, because these folds read none and hold numbers alone.

The estimation mask does not travel these folds. The exact folds of the moment layer take an active mask and no estimation mask, so a step whose estimation universe is narrower than its active universe would fit the wrong universe without a word. The function refuses such a panel with an `ArgumentError`.

# Algorithm

 1. Refuse `rd` without returns, and `rd` with an implied-volatility surface, with [`assert_prior_fold_returns`](@ref).
 2. Return `nothing` when `rd` holds no panel, or when its panel is static.
 3. Refuse a time-varying panel that holds a Panel Field.
 4. Refuse a time-varying panel whose estimation mask differs from its active mask.
 5. Return the active mask `pnl.amsk`.

# Arguments

  - `rd`: The returns data to fold.

# Validation

  - The rules of [`assert_prior_fold_returns`](@ref).
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
    assert_prior_fold_returns(rd)
    pnl = rd.pnl
    if isnothing(pnl) || panel_is_static(pnl)
        return nothing
    end
    @argcheck(isempty(pnl.pf),
              ArgumentError("the online step carries a time-varying Asset Panel through its masks alone, and this one holds $(length(pnl.pf)) Panel Field(s): this fold reads no Panel Field and records none, so the step has nowhere to keep them and a call with no returns could not rebuild the panel. Drop the fields from the panel handed to the step, or fit in batch."))
    @argcheck(pnl.emsk == pnl.amsk,
              ArgumentError("the estimation mask does not travel the online step: the exact folds of the moment layer take an active mask and no estimation mask, so an estimation universe narrower than the active one cannot be honoured online. This panel's `emsk` differs from its `amsk` at $(count(pnl.emsk .!= pnl.amsk)) cell(s). Pass the active mask as both, or fit in batch."))
    return pnl.amsk
end
"""
    fold_prior(pe::AbstractPriorEstimator, rd::ReturnsResult)
    fold_prior(pe::AbstractPriorResult, rd::ReturnsResult)
    fold_prior(pe::TimeDependent, rd::ReturnsResult)

Forwards the observations of `rd` to the prior, in the arity of the prior's own step.

This is the one forward that the step of an optimiser makes. An optimiser forwards each observation to its prior and to nothing else. `optimise(opt)` with no returns fits every other member in batch, from the fold context it rebuilds. The step folds `rd` through [`partial_fit!`](@ref) over the `ReturnsResult`, which reads it as [`prior`](@ref) reads it in batch. The route of the prior decides what the step honours. A refit honours the estimation mask and records the Panel Fields of a prior that reads them. The carry of [`EmpiricalPrior`](@ref) refuses an estimation mask that differs from the active mask, and each route refuses a Panel Field that its prior does not read. The tree of the prior decides what to do with `F`, as its batch function does.

A prior that is already an [`AbstractPriorResult`](@ref) has no state to fold into. It is batch configuration, and an optimiser that holds one runs `optimise(opt, rd)`. This function also refuses a [`TimeDependent`](@ref) schedule on the prior. A schedule swaps the estimator that carries the state, and a member that never saw the folded rows cannot take them over.

# Algorithm

 1. Fold `rd` into `pe` with [`partial_fit!`](@ref) over the `ReturnsResult`, which refuses by route what the prior cannot honour.

# Arguments

  - `pe`: The prior that the optimiser holds.
  - `rd`: The returns data to fold, `observations × assets`.

# Validation

  - `pe` is an [`AbstractPriorEstimator`](@ref). An `ArgumentError` is thrown for a prior result or a schedule.
  - Everything [`partial_fit!`](@ref) of the prior over `rd` refuses.

# Returns

  - `pe`: The prior, with the observations folded into its state.

# Related

  - [`partial_fit!`](@ref)
  - [`reads_panel_fields`](@ref)
  - [`needs_factor_returns`](@ref)
  - [`prior`](@ref)
  - [`assert_online_entry`](@ref): refuses a schedule on the prior before the warm-up of the fold loop.
"""
function fold_prior(pe::AbstractPriorEstimator, rd::ReturnsResult)
    return partial_fit!(pe, rd)
end
function fold_prior(pe::AbstractPriorResult, ::ReturnsResult)
    return throw(ArgumentError("`pe` holds a fitted `$(typeof(pe))`, which has no state to fold an observation into: a prior result is batch configuration. Hand the optimiser the prior estimator to take the online step, or run `optimise(opt, rd)`."))
end
function fold_prior(pe::TimeDependent, ::ReturnsResult)
    return throw(ArgumentError("`pe` holds a `TimeDependent` schedule of priors, and the online step cannot fold into a schedule: it swaps the estimator that carries the state, and a member that never saw the folded rows cannot be handed them. No loop resolves a schedule before stepping — a schedule reaches stateless fields only. Hold one prior in `pe`, and schedule a field that carries no state."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observations of `rd` into the fold context of an optimiser, and seeds the context on the first step.

# Algorithm

 1. Take `cache` as the context `state`. When `cache` is `nothing`, seed an empty [`ReturnsBufferState`](@ref) with the cap `max_history` instead.
 2. Fold `rd` into `state` with [`partial_fit!`](@ref), under `own_returns` and `own_factors`.

# Arguments

  - `cache`: The context that the optimiser holds, or `nothing` before the first step.
  - `rd`: The returns data to fold.
  - `max_history`: The cap of a new context. It is the cap of the buffer that holds the returns, so the context and that buffer drop the same rows.
  - `own_returns`: Whether the context keeps the returns itself. It is `true` for a head that holds no prior.
  - `own_factors`: Whether the context keeps the factor column itself. It is `true` for a head that holds no prior, and for an optimiser whose prior never reads factor returns.

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

Takes the step of an optimiser that holds a prior and a fold context. It forwards the observations to the prior, then records `rd` in the context.

The prior folds first, because the context takes its cap from the buffer of the prior. The `max_history` of an [`Online`](@ref) wrapper caps the rows of the prior, and the context must drop the same rows at the same step, so the [`ReturnsResult`](@ref) that `optimise(opt)` with no returns rebuilds matches the rows of the prior.

One place holds `F`. The context keeps the factor column only when the tree of the prior never reads it, that is, when [`needs_factor_returns`](@ref) answers `false`. Otherwise the buffer of the prior records it, and `optimise(opt)` with no returns reads it back through [`prior_returns_buffer`](@ref), as it reads the returns.

# Algorithm

 1. Fold `rd` into `opt.pe` with [`fold_prior`](@ref), giving the folded prior `pe`.
 2. Read the buffer `rows` of `pe` with [`prior_returns_buffer`](@ref).
 3. Set `own_factors` to `true` when `needs_factor_returns(pe) === false`.
 4. Fold `rd` into `opt.cache` with [`fold_context`](@ref), under the cap `rows.max_history`, giving `cache`. The context never keeps the returns.
 5. Rebuild `opt` with `pe` and `cache`.

# Arguments

  - `opt`: An estimator that holds `pe` and `cache`.
  - `rd`: The returns data to fold, `observations × assets`.

# Validation

  - Everything [`fold_prior`](@ref) and [`fold_context`](@ref) refuse.

# Returns

  - `opt`: The optimiser, with its prior folded and its context recorded.

# Related

  - [`fold_prior`](@ref)
  - [`fold_context`](@ref)
  - [`prior_returns_buffer`](@ref)
  - [`needs_factor_returns`](@ref)
"""
function fold_returns(opt, rd::ReturnsResult)
    pe = fold_prior(opt.pe, rd)
    rows = prior_returns_buffer(pe)
    own_factors = needs_factor_returns(pe) === false
    cache = fold_context(opt.cache, rd, rows.max_history, false, own_factors)
    return rebuild_estimator(opt, (; pe = pe, cache = cache))
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

The online step of an optimiser has two functions. `partial_fit!(opt, rd)` folds the observations and returns the estimator, and `optimise(opt)` with no returns reads the state out and solves once. The step forwards the observations to the prior alone, through [`fold_prior`](@ref), and records the rest of `rd` in a [`ReturnsBufferState`](@ref), so `optimise(opt)` with no returns can rebuild the [`ReturnsResult`](@ref) that the batch path reads. The step does not change the members above the prior: the clustering estimator, the constraint estimators, the uncertainty sets, and the inner optimisers of a meta-optimiser. `optimise(opt)` with no returns fits each of them as a batch run does.

The step takes a `ReturnsResult`, as `optimise(opt, rd)` does. `rd` holds one observation or a block of them, and the step unpacks it into the arity of the prior's step. A JuMP head forwards to the [`JuMPOptimiser`](@ref) it holds, and a hierarchical head to its [`HierarchicalOptimiser`](@ref). The optimiser that holds the prior and the context takes the step. The three meta-optimisers and [`InverseVolatility`](@ref) hold their prior directly, so each takes its own step. [`EqualWeighted`](@ref), [`RandomWeighted`](@ref) and [`BestConstantRebalancedPortfolio`](@ref) hold no prior but read the observations, so their context keeps the rows itself. [`PreviousWeights`](@ref) reads no observation, so its step returns it unchanged and it carries no context.

After `t` observations, `optimise(opt)` equals `optimise(opt, rd[1:t])`. The `ReturnsResult` that `optimise(opt)` with no returns rebuilds equals `rd[1:t]` field by field. The weights of a family that solves no programme agree to rounding, and the weights of a JuMP family agree to the tolerance of its solver.

# Algorithm

 1. A JuMP head or a hierarchical head steps the bundle it holds in `opt`, and rebuilds itself around the result.
 2. An optimiser that holds a prior steps through [`fold_returns`](@ref).
 3. A head that holds no prior checks that `rd.X` is present. It then folds `rd` into its own context with [`fold_context`](@ref), and the context keeps the returns and the factor column.
 4. [`PreviousWeights`](@ref) returns `opt` unchanged.

# Arguments

  - `opt`: The optimiser to fold into.
  - `rd`: The returns data to fold, `observations × assets`. One row is one step of a walk-forward, and a block is a warm-up.

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
    return throw(ArgumentError("a `$(typeof(opt))` has no online step: a finite allocation converts a weight vector and prices into share counts and reads no returns window, so there is nothing to fold. Take the step on the optimiser that produces the weights, and allocate its weights with `optimise(da, w, p)`."))
end
function partial_fit!(::TD_OptE_Opt, ::ReturnsResult)
    return throw(ArgumentError("a `TimeDependent` schedule of optimisers has no online step: a schedule swaps the optimiser that carries the state, and no loop resolves a schedule before stepping — a schedule reaches stateless fields only. Step one optimiser, and schedule a field that carries no state."))
end
"""
    online_state_seed(opt::Union{<:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio}, max_history)
    online_state_seed(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling}, max_history)

Seeds the fold context that an [`Online`](@ref) wrapper declares on a head with no prior, and refuses the wrapper on an optimiser that holds a prior.

`Online(EqualWeighted(); max_history = w)` caps the rows that the head keeps, so its seed is an empty [`ReturnsBufferState`](@ref) with the cap `w`. An optimiser that holds a prior takes its cap from the prior. `Online` wraps the prior, and the context follows the buffer of the prior, so a wrapper on the optimiser would declare a second cap over the same rows.

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

Resolves the [`Online`](@ref) wrappers in the `pe` slot of an optimiser, and returns any other value unchanged.

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

The JuMP and hierarchical heads resolve the bundle they hold, and the bundle, the meta-optimisers and [`InverseVolatility`](@ref) resolve their `pe`. The step forwards observations to that prior alone, so this function resolves no other wrapper. A wrapper in an inner optimiser of a meta-optimiser, in a fallback, or in any other field stays unresolved, and the entry of the online arm of the fold loop refuses it with an error that names the field.

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

The warm-up of an optimiser resolves the wrappers in its prior alone, through [`update_online_estimator`](@ref). A wrapper in any other field, such as an inner optimiser of a meta-optimiser or a fallback, never gets a seeded buffer, and the batch fit of `optimise(opt)` with no returns refuses it. This function runs the warm-up and names the wrapper that is left, so the entry of the online arm can refuse it before the first fold. Any other estimator answers `nothing`.

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
  - [`assert_batch_entry`](@ref): the refusal that `optimise(opt)` with no returns would reach later.
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

Refuses a [`TimeDependent`](@ref) schedule on a field that carries a state, with an error that names the field.

A schedule replaces the value of its field at every fold, and the loop threads the state through that value. The replacement drops the state, so the value that a fold receives never saw the rows folded before it. The one such field that a schedule can reach is the `pe` of an optimiser, whose type bound admits a schedule. The `opt` field of a JuMP head or of a hierarchical head holds the bundle, and its bound refuses a schedule at construction, so the heads only recurse. This function walks the route of [`update_online_estimator`](@ref). A schedule on any other field passes, the inner optimisers of a meta-optimiser included. The loop resolves such a schedule on the copy it fits at each fold, and the state stays on the estimator it threads. No schedule carries a state across a swap.

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

Refuses the prior of an optimiser when it is a [`TimeDependent`](@ref) schedule, and passes any other value.

# Arguments

  - `pe`: The prior of `opt`.
  - `opt`: The optimiser that holds `pe`, which the message names.

# Validation

  - `pe` is not a [`TimeDependent`](@ref). An `ArgumentError` is thrown otherwise.

# Related

  - [`assert_stateless_schedule`](@ref)
"""
function assert_stateless_prior(::TimeDependent, opt)
    return throw(ArgumentError("`$(typeof(opt).name.name).pe` holds a `TimeDependent` schedule, and the online arm of the fold loop cannot step it: `pe` carries the partial-fit state the loop threads from fold to fold, and a schedule replaces that value every fold, so the prior a fold is handed never saw the rows folded before it. A schedule reaches stateless fields only. Hold one prior in `pe`, and schedule a field that carries no state, or refit every fold with `ff = nothing`."))
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
 2. Refuse a schedule on the `pe` of an optimiser, through [`assert_stateless_schedule`](@ref).
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
              ArgumentError("`$(typeof(est).name.name)` enters the online arm of the fold loop holding an `Online` at `$(wpath)`, where the warm-up does not resolve it. The online step forwards each observation to the prior of the optimiser alone, so a wrapper in any other field never gets a buffer, and the batch fit of `optimise(opt)` with no returns would refuse it. Move the wrapper to the prior of the optimiser, or set `$(wpath)` to the estimator it wraps, and `optimise(opt)` with no returns refits it from the folded rows."))
    return nothing
end
"""
    returns_result(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
    returns_result(opt::Union{<:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio})
    returns_result(opt::JuMPOptimisationEstimator)
    returns_result(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})

Rebuilds the [`ReturnsResult`](@ref) of the observations that an optimiser has folded.

This is the step of `optimise(opt)` with no returns that rebuilds the returns data. The rows come from the buffer of the prior, or from the context of a head that holds no prior. Every other column and the pinned context come from the [`ReturnsBufferState`](@ref) of the optimiser. The factor column comes from the context when the tree of the prior never reads it, and from the buffer of the prior otherwise. The result is a new `ReturnsResult`, equal field by field to the one that a batch fit over the same observations reads.

# Arguments

  - `opt`: The optimiser, or the bundle it holds.

# Validation

  - `opt` carries a fold context. An `ArgumentError` is thrown otherwise.

# Returns

  - `rd::ReturnsResult`: The observations folded so far.

# Related

  - [`ReturnsBufferState`](@ref)
  - [`prior_returns_buffer`](@ref)
  - [`partial_fit!`](@ref)
  - [`optimise`](@ref)
"""
function returns_result(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                   <:InverseVolatility, <:NestedClustered, <:Stacking,
                                   <:SubsetResampling})
    return returns_result(partial_fit_cache(opt), prior_returns_buffer(opt.pe))
end
function returns_result(opt::Union{<:EqualWeighted, <:RandomWeighted,
                                   <:BestConstantRebalancedPortfolio})
    state = partial_fit_cache(opt)
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
    held_timestamps(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling, <:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio})
    held_timestamps(opt::JuMPOptimisationEstimator)
    held_timestamps(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})
    held_timestamps(::PreviousWeights)

Reads the timestamps that the fold context of an optimiser holds, or answers `nothing` when it holds none.

The alignment check of [`Resume`](@ref) reads this accessor. The [`ReturnsBufferState`](@ref) keeps the timestamps of the folded observations and drops the oldest under a cap, so under `Online(pe; max_history = w)` it holds the last `w`. A head forwards to the bundle it holds, and an optimiser that holds the context reads it, as in [`returns_result`](@ref). An optimiser that has taken no step answers `nothing`. [`PreviousWeights`](@ref) reads no observation and keeps no context, so it answers `nothing`, and a resume refuses it.

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
function held_timestamps(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                    <:InverseVolatility, <:NestedClustered, <:Stacking,
                                    <:SubsetResampling, <:EqualWeighted, <:RandomWeighted,
                                    <:BestConstantRebalancedPortfolio})
    return context_timestamps(opt.cache)
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
    batch_from_state(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser, <:InverseVolatility, <:NestedClustered, <:Stacking, <:SubsetResampling})
    batch_from_state(opt::Union{<:EqualWeighted, <:RandomWeighted, <:BestConstantRebalancedPortfolio})
    batch_from_state(opt::JuMPOptimisationEstimator)
    batch_from_state(opt::Union{<:HierarchicalRiskParity, <:HierarchicalEqualRiskContribution, <:SchurComplementHierarchicalRiskParity})
    batch_from_state(opt::FiniteAllocationOptimisationEstimator)
    batch_from_state(opt::OptimisationEstimator)

Turns a folded optimiser into the batch call that reads it out, an estimator and a [`ReturnsResult`](@ref).

`optimise(opt)` with no returns rebuilds the `ReturnsResult` and calls the ordinary batch path, so no member above the prior needs a method of its own. The prior becomes `prior(pe)`, a prior result, which the batch path does not refit, so the solve reads the folded prior. This function drops the context, because the estimator it returns is a batch configuration over the rebuilt `ReturnsResult`. It never writes the state, so two calls on one estimator give the same answer, and a fallback reads the same `ReturnsResult` as the optimiser it replaces.

The `cache` of the optimiser selects the branch. An optimiser whose `cache` is `nothing` has taken no step, and [`batch_without_state`](@ref) answers for it. This function returns any other optimiser, such as [`PreviousWeights`](@ref), with an empty `ReturnsResult`, and its batch function decides whether it answers without returns.

# Algorithm

 1. A JuMP head or a hierarchical head reads out the bundle it holds in `opt`, and rebuilds itself around the bundle.
 2. An optimiser with no state goes to [`batch_without_state`](@ref).
 3. Rebuild `rd` with [`returns_result`](@ref).
 4. Rebuild the optimiser with `cache = nothing`. An optimiser that holds a prior also replaces `pe` with `prior(pe)`, the result of the folded prior.
 5. Return the optimiser and `rd`.

# Arguments

  - `opt`: The optimiser to read out.

# Validation

  - `opt` is not a finite allocation. An `ArgumentError` is thrown otherwise, because a finite allocation has no step.
  - Everything [`batch_without_state`](@ref) and [`returns_result`](@ref) refuse.

# Returns

  - `(opt, rd)::Tuple`: The batch estimator and the `ReturnsResult` to run it over.

# Related

  - [`optimise`](@ref)
  - [`returns_result`](@ref)
  - [`partial_fit!`](@ref)
  - [`prior`](@ref)
"""
function batch_from_state(opt::Union{<:JuMPOptimiser, <:HierarchicalOptimiser,
                                     <:InverseVolatility, <:NestedClustered, <:Stacking,
                                     <:SubsetResampling})
    if isnothing(opt.cache)
        return batch_without_state(opt, opt.pe)
    end
    rd = returns_result(opt)
    return rebuild_estimator(opt, (; pe = prior(opt.pe), cache = nothing)), rd
end
function batch_from_state(opt::Union{<:EqualWeighted, <:RandomWeighted,
                                     <:BestConstantRebalancedPortfolio})
    if isnothing(opt.cache)
        return batch_without_state(opt, nothing)
    end
    rd = returns_result(opt)
    return rebuild_estimator(opt, (; cache = nothing)), rd
end
function batch_from_state(opt::JuMPOptimisationEstimator)
    inner, rd = batch_from_state(opt.opt)
    return rebuild_estimator(opt, (; opt = inner)), rd
end
function batch_from_state(opt::Union{<:HierarchicalRiskParity,
                                     <:HierarchicalEqualRiskContribution,
                                     <:SchurComplementHierarchicalRiskParity})
    inner, rd = batch_from_state(opt.opt)
    return rebuild_estimator(opt, (; opt = inner)), rd
end
"""
    batch_without_state(opt, pe::AbstractPriorResult)
    batch_without_state(opt, pe)

Answers for an optimiser that has taken no step.

An optimiser whose prior is already a result needs no returns to solve. This function returns it with an empty [`ReturnsResult`](@ref), which is the batch entry that `optimise(opt)` has always had. Any other optimiser has no state and no prior result to answer from, and this function refuses it with an `ArgumentError`. The message names the step that fills the state and the batch function that takes the returns.

# Validation

  - `pe` is an [`AbstractPriorResult`](@ref). An `ArgumentError` is thrown otherwise.

# Related

  - [`batch_from_state`](@ref)
  - [`partial_fit!`](@ref)
"""
function batch_without_state(opt, ::AbstractPriorResult)
    return opt, ReturnsResult()
end
function batch_without_state(opt, ::Any)
    return throw(ArgumentError("`optimise(opt)` with no returns reads the state the online step wrote, and this `$(typeof(opt).name.name)` has taken no step: its `cache` is `nothing`. Fold observations with `partial_fit!(opt, rd)` first, or pass the returns to `optimise(opt, rd)`."))
end
function batch_from_state(opt::FiniteAllocationOptimisationEstimator)
    return throw(ArgumentError("a `$(typeof(opt))` has no online step, so `optimise(opt)` with no returns has no state to read: a finite allocation reads no returns window. Allocate a weight vector with `optimise(da, w, p)`."))
end
function batch_from_state(opt::OptimisationEstimator)
    return opt, ReturnsResult()
end
"""
    optimise(opt::OptimisationEstimator; kwargs...)

Reads out a folded optimiser. It rebuilds the [`ReturnsResult`](@ref) from the state and solves once over it.

This method reads the state of the online step, and it is also the entry of every optimiser that takes no returns. The optimiser answers from what it holds. That is a prior that is already a result, which this entry has always served, or the state that its [`partial_fit!`](@ref) steps wrote. The ordinary `optimise(opt, rd)` then runs over the rebuilt `ReturnsResult`, and it fits the constraint estimators, the clustering, the uncertainty sets and the inner optimisers as a batch run does. A failed solve changes no state, so the fallback chain runs as in batch.

```julia
mr = MeanRisk(; opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv))
for t in 1:warmup
    mr = partial_fit!(mr, port_opt_view(rd, t:t, :))   # folds; solves nothing
end
res = optimise(mr)                                     # reads out; solves once
```

# Algorithm

 1. Turn `opt` into a batch estimator and its `rd` with [`batch_from_state`](@ref).
 2. Run `optimise(opt, rd; kwargs...)` on them.

# Arguments

  - `opt`: The optimiser to read out.
  - `kwargs...`: The keyword arguments of the batch function, `dims`, `str_names` and `save`, passed on unchanged.

# Validation

  - Everything [`batch_from_state`](@ref) refuses.

# Returns

  - `res::OptimisationResult`: The result that the batch function gives over the observations folded so far.

# Related

  - [`partial_fit!`](@ref)
  - [`batch_from_state`](@ref)
  - [`returns_result`](@ref)
  - [`update_online_estimator`](@ref): resolves an [`Online`](@ref) prior before the first step.
"""
function optimise(opt::OptimisationEstimator; kwargs...)
    opt, rd = batch_from_state(opt)
    return optimise(opt, rd; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an optimiser that takes the online step, except its `cache`.

The `cache` holds the state of an incremental fit, not the configuration that a reader looks up, and it would print under every optimiser that holds it. `set_show_nothing_fields!(:Name, true)` renders it for the type `Name`.

# Arguments

  - `opt`: The optimiser.

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
