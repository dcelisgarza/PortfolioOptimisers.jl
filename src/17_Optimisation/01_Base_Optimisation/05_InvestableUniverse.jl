"""
    port_opt_view(opt, i, args...)

Return a view or subset of an optimisation estimator for a given cluster index `i`.

Default fallback returns the estimator unchanged. Overridden for composite estimators (e.g. [`JuMPOptimiser`](@ref), [`HierarchicalRiskParity`](@ref)) to slice all sub-estimators for the `i`-th cluster.

# Arguments

  - `opt`: Optimisation estimator or result.
  - `i`: Cluster or asset index.
  - `args...`: Additional arguments (e.g. asset returns matrix).

# Returns

  - Sliced or unchanged optimisation estimator.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`NestedClustered`](@ref)
"""
function port_opt_view(opt::AbstractOptimisationEstimator, ::Any, args...)
    return opt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

A precomputed optimisation result cannot be restricted to an asset subset.

Its weights were solved over the full universe and a sub-portfolio of them has no defined meaning, so an asset-subset view of a result throws. In particular, a [`TimeDependent`](@ref) schedule holding result entries is incompatible with asset-subsampling cross-validation ([`MultipleRandomised`](@ref)), whose fold loops view the optimiser to each fold's asset subset before the swap. The trivial all-assets view (`Colon`) passes the result through unchanged.

# Related

  - [`port_opt_view`](@ref)
  - [`TimeDependent`](@ref)
  - [`MultipleRandomised`](@ref)
"""
function port_opt_view(res::NonFiniteAllocationOptimisationResult, ::Colon, args...)
    return res
end
function port_opt_view(::NonFiniteAllocationOptimisationResult, ::Any, args...)
    return throw(ArgumentError("a precomputed optimisation result cannot be viewed to an asset subset: its weights were solved over the full universe and a sub-portfolio of them has no defined meaning. A TimeDependent schedule holding precomputed results is therefore incompatible with asset-subsampling cross-validation (e.g. MultipleRandomised); use estimator entries there instead."))
end
"""
    non_investable_universe(opt::AbstractOptimisationEstimator, ni::VecStr)
    non_investable_universe(opt, ni::VecStr)

Mint the Non-Investable Axis on the [`UniverseSets`](@ref) an optimisation estimator carries.

A door calls this **after** it has viewed the estimator, and it is the only caller. Minting before the view would not survive it: [`port_opt_view`](@ref)`(::UniverseSets, i)` drops the axis, precisely so that a cluster of a nested optimisation cannot inherit its parent's departures and charge every one of them again. So the door reads the departed names off the *unreduced* returns data, takes the view, and declares the axis on what comes back.

The generic method returns `opt` untouched, and it is the right answer for every estimator that carries no sets: there is no axis to declare one on. A head that owns a `sets` field writes one method beside its own [`port_opt_view`](@ref), and a head that reaches one through a nested optimiser forwards to that optimiser's method. Both are one line, and they are written per type rather than derived by reflection for the reason [`port_opt_view`](@ref) is: a field's meaning is the type's to state.

# Arguments

  - `opt`: The optimisation estimator, already reduced to the Investable Mask.
  - `ni`: The names the mask left out, from [`non_investable_names`](@ref).

# Returns

  - `opt`: The estimator, carrying the Non-Investable Axis where it carries a [`UniverseSets`](@ref).

# Related

  - [`non_investable_sets`](@ref)
  - [`non_investable_names`](@ref)
  - [`investable_reduction`](@ref)
  - [`coverage_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function non_investable_universe(opt, ::VecStr)
    return opt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Derive the Investable Mask of a fitted prior, and reduce the prior, the optimiser and the returns data to the assets it keeps.

A Prior Estimator fits on the coverage universe and hands back a result on the full asset universe, in which an asset it could not estimate carries `NaN` in `mu` and on the diagonal of `sigma`. Nothing downstream of the fit can solve over such an asset, so the reduction happens once, at the optimiser's entry, and every constraint the caller stated over the full universe is sliced by the same index.

Every optimisation family reduces through this one verb, which is why it is bound to [`AbstractOptimisationEstimator`](@ref) and lives here rather than beside the JuMP prelude: the hierarchical, naive and meta files load before that prelude and reach it without a back reference. A meta-optimiser composes two masks, its own at its entry and each inner head's inside its own solve, and both expand.

Three methods, and the branch is dispatch rather than a condition. The first derives the mask; the `nothing` method is the all-investable path and returns its arguments untouched; the `BitVector` method takes the three views. A universe with nothing to exclude therefore costs one pass over two vectors and no allocation.

The optimiser is viewed at `pr.X`, the *unreduced* returns matrix, because [`port_opt_view`](@ref) slices a tracking estimator against it by the same asset index. The prior is reduced after, so the matrix the view reads is still the full one.

The mask rides as a `BitVector` because the expansion needs the length of the full universe and nothing else carries it once the prior is reduced. The three views take `findall(imsk)` instead, which is the integer index every other caller of [`port_opt_view`](@ref) passes.

# Algorithm

 1. Derive the Investable Mask from the fitted prior with [`investable_mask`](@ref).
 2. Return the mask, the prior, the optimiser and the returns data unchanged when the mask is `nothing`.
 3. Otherwise read the departed names off the *unreduced* `rd.nx` with [`non_investable_names`](@ref), and announce them once with [`announce_non_investable`](@ref).
 4. Take a [`port_opt_view`](@ref) of each of the three at `findall(imsk)`.
 5. Declare the Non-Investable Axis on the viewed optimiser with [`non_investable_universe`](@ref), and return it beside the mask and the other two views. The axis is declared after the view because the view drops one, so a name-keyed constraint stated for a departed asset resolves here and nowhere deeper.

# Arguments

  - $(arg_dict[:pr])
  - `opt::AbstractOptimisationEstimator`: The optimisation estimator, holding every constraint estimator the caller stated over the full universe.
  - $(arg_dict[:rd])

# Returns

  - `(imsk, pr, opt, rd)`: The Investable Mask and the three reduced to it, or `nothing` and the three unchanged.

# Related

  - [`investable_mask`](@ref)
  - [`expand_investable_weights`](@ref)
  - [`port_opt_view`](@ref)
  - [`non_investable_names`](@ref)
  - [`non_investable_universe`](@ref)
  - [`announce_non_investable`](@ref)
"""
function investable_reduction(pr::AbstractPriorResult, opt::AbstractOptimisationEstimator,
                              rd::ReturnsResult)
    return investable_reduction(investable_mask(pr), pr, opt, rd)
end
function investable_reduction(::Nothing, pr::AbstractPriorResult,
                              opt::AbstractOptimisationEstimator, rd::ReturnsResult)
    return nothing, pr, opt, rd
end
function investable_reduction(imsk::BitVector, pr::AbstractPriorResult,
                              opt::AbstractOptimisationEstimator, rd::ReturnsResult)
    idx = findall(imsk)
    # Read the departed names before the view, declare them after it: the view drops the
    # Non-Investable Axis, so this door is the one place a name-keyed constraint stated
    # for an asset that left can still be resolved.
    ni = non_investable_names(rd.nx, imsk)
    announce_non_investable(ni)
    return imsk, port_opt_view(pr, idx),
           non_investable_universe(port_opt_view(opt, idx, pr.X), ni),
           port_opt_view(rd, idx)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Derive the Coverage Universe of a window, and reduce a prior-free optimiser and its returns data to the assets it keeps.

This is the twin of [`investable_reduction`](@ref) for a head that fits no prior. No Prior Result exists to derive an Investable Mask from, so the head derives the Coverage Universe of its own window instead, through the verb the priors use: an asset is kept when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window. A stale finite price during an inactive spell is therefore outside the mask, and the asset weights nothing.

The two masks are the same object downstream. The head carries the Coverage Universe as the `imsk` of its result, and the result's keyword constructor expands the weights through [`expand_investable_weights`](@ref), as every other family's does, so every optimisation result of the library carries a mask and a reader has one idiom.

Three methods, and the branch is dispatch rather than a condition, as it is in [`investable_reduction`](@ref). The first derives the mask; the `nothing` method is the all-covered path and returns its arguments untouched; the `BitVector` method takes the two views. The optimiser is viewed at `rd.X`, the *unreduced* returns matrix, because [`port_opt_view`](@ref) slices an estimator against it by the same asset index.

An all-dead window throws an `IsEmptyError` where the mask is derived, so the refusal is [`coverage_mask`](@ref)'s and every prior-free head has it for free.

# Algorithm

 1. Derive the Coverage Universe of `rd.X` and `rd.pnl` with [`coverage_mask`](@ref).
 2. Return the mask, the optimiser and the returns data unchanged when the mask is `nothing`.
 3. Otherwise read the departed names with [`non_investable_names`](@ref), announce them once with [`announce_non_investable`](@ref), take a [`port_opt_view`](@ref) of each of the two at `findall(cmsk)`, and declare the Non-Investable Axis on the viewed optimiser with [`non_investable_universe`](@ref).

# Arguments

  - `opt::AbstractOptimisationEstimator`: The optimisation estimator, holding every constraint estimator the caller stated over the full universe.
  - $(arg_dict[:rd])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])
  - At least one asset must be in the Coverage Universe.

# Returns

  - `(cmsk, opt, rd)`: The Coverage Universe and the two reduced to it, or `nothing` and the two unchanged.

# Related

  - [`coverage_mask`](@ref)
  - [`investable_reduction`](@ref)
  - [`expand_investable_weights`](@ref)
  - [`port_opt_view`](@ref)
"""
function coverage_reduction(opt::AbstractOptimisationEstimator, rd::ReturnsResult;
                            dims::Int = 1)
    return coverage_reduction(coverage_mask(rd.X, rd.pnl; dims = dims), opt, rd)
end
function coverage_reduction(::Nothing, opt::AbstractOptimisationEstimator,
                            rd::ReturnsResult)
    return nothing, opt, rd
end
function coverage_reduction(cmsk::BitVector, opt::AbstractOptimisationEstimator,
                            rd::ReturnsResult)
    idx = findall(cmsk)
    # A door is a door: the Coverage Universe and the Investable Mask are the same object
    # downstream, so a prior-free head declares the Non-Investable Axis exactly as a
    # prior-fitting one does, and a departure behaves the same in every family.
    ni = non_investable_names(rd.nx, cmsk)
    announce_non_investable(ni)
    return cmsk, non_investable_universe(port_opt_view(opt, idx, rd.X), ni),
           port_opt_view(rd, idx)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Expand a solved weight vector from the investable subset back onto the full asset universe.

The optimiser solves over the assets the Investable Mask keeps, so its weight vector is shorter than the universe the caller stated. This puts each solved weight back at its own asset and writes a zero everywhere else, which is what a non-investable asset holds: the optimiser could not trade it.

A failed solve carries `NaN` at every solved position. The expansion keeps that distinction — `NaN` where the optimiser tried and failed, zero where it never could — rather than flattening both to zero.

The `nothing` mask returns the weights unchanged, so nothing is copied when every asset is investable, and a `nothing` weight vector stays `nothing`, which is what a naive head records when its finaliser gave up. The vector-of-vectors method serves the efficient-frontier route, where one weight vector is recorded per sweep point. [`JuMPOptimisationSolution`](@ref) carries its own methods beside the JuMP prelude, and they delegate to the plain weight vector here, so one length check and one message serve every family.

The **keyword** constructor of each optimisation result is the caller, and every family builds its result through it. The positional constructor never expands, because every return-code rebuild goes through it and a second pass would expand twice.

# Arguments

  - $(arg_dict[:imsk])
  - `w`: The weights the optimisation solved over the investable universe: one weight vector, or a vector of them on the efficient-frontier route.

# Validation

  - The solved weight vector must hold one weight per investable asset.

# Returns

  - `w`: The weights, or the vector of them, on the full asset universe.

# Related

  - [`investable_mask`](@ref)
  - [`investable_reduction`](@ref)
  - [`set_retcode`](@ref)
"""
function expand_investable_weights(::Nothing, w::Option{<:VecNum_VecVecNum})
    return w
end
function expand_investable_weights(::BitVector, ::Nothing)
    return nothing
end
function expand_investable_weights(imsk::BitVector, w::VecNum)
    @argcheck(count(imsk) == length(w),
              DimensionMismatch("the investable mask keeps $(count(imsk)) of $(length(imsk)) assets, but the solution holds $(length(w)) weights; the mask and the weights must come from the same optimisation"))
    wf = zeros(eltype(w), length(imsk))
    wf[imsk] = w
    return wf
end
function expand_investable_weights(imsk::BitVector, w::VecVecNum)
    return [expand_investable_weights(imsk, wi) for wi in w]
end
"""
    optimise(opt::OptimisationEstimator, args...; kwargs...) -> OptimisationResult
    optimise(opt::OptimisationResult, args...; kwargs...) -> OptimisationResult

Run portfolio optimisation using the given estimator `opt` and return an [`OptimisationResult`](@ref).

If `opt` returns an [`OptimisationFailure`](@ref), the fallback estimator is tried automatically until either a successful result is obtained or all fallbacks are exhausted.

Passing an [`OptimisationResult`](@ref) directly returns it unchanged (pass-through method).

# Arguments

  - `opt`: Optimisation estimator (e.g. a [`JuMPOptimisationEstimator`](@ref) subtype).
  - $(arg_dict[:ignargs])
  - $(arg_dict[:ignkwargs])

# Returns

  - [`OptimisationResult`](@ref): The optimisation result.

# Related

  - [`OptimisationEstimator`](@ref)
  - [`OptimisationResult`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function optimise(opt::OptimisationResult, args...; kwargs...)
    return opt
end
"""
    _optimise(opt, rd, args...; dims, str_names, save, kwargs...)

Internal dispatch function for portfolio optimisation.

Called by [`optimise`](@ref) to perform the actual optimisation. Each optimisation estimator type implements its own overload. Returns the estimator-specific result type.

# Arguments

  - `opt`: Optimisation estimator (e.g. [`MeanRisk`](@ref), [`RiskBudgeting`](@ref), etc.).
  - `rd::ReturnsResult`: Returns data.
  - `dims::Int`: Observation dimension.
  - `str_names::Bool`: Whether to use string names in the JuMP model.
  - `save::Bool`: Whether to save the JuMP model in the result.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Estimator-specific optimisation result.

# Related

  - [`optimise`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function _optimise end
# A precomputed result is a fallback that the `fb` aliases admit, so the fallback loop of
# `optimise` answers it as it stands.
function _optimise(res::OptimisationResult, args...; kwargs...)
    return res
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

High level optimisation function that wraps around estimator-specific optimisation functions. This takes care of fallback methods if the primary optimisation fails. It returns the first successful optimisation result, or the last failure when every fallback fails, and stores the `(estimator, result)` pair of every failed attempt in the `fb` field of that result, in the order they ran (see [`FbChain`](@ref)). When no fallback was needed, `fb` is `nothing`.

This is a fold-less entry point, so time-dependent schedules are inert here: the estimator is reset to its fold-less values (see [`reset_time_dependent_estimator`](@ref)) before the solve — in particular a scheduled fallback resets to its `default`, or to `nothing` (no fallback) when it has none, *before* the fallback chain is walked. Inside a fold loop this reset is a no-op, because the loop resolves every schedule before optimising.

It is a batch fit, so an [`Online`](@ref) anywhere in the estimator's tree is refused by name through [`assert_batch_entry`](@ref) before any solve: the wrapper resolves only at the warm-up of the fold loop's online arm, and a plain `optimise` runs none. The read-out of a stepped estimator, `optimise(opt)`, never meets this refusal, because the warm-up that seeded its buffer replaced the wrapper.

# Arguments

  - `opt::OptimisationEstimator`: The optimisation estimator to use.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])

# Validation

  - No field in the tree of `opt` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise.
"""
function optimise(opt::OptimisationEstimator, args...; kwargs...)
    assert_batch_entry(opt, "`optimise`")
    fb = Tuple{OptimisationEstimator, OptimisationResult}[]
    current_opt = reset_time_dependent_estimator(opt)
    res = nothing
    while true
        res = _optimise(current_opt, args...; kwargs...)
        if isa(res.retcode, OptimisationSuccess) || isnothing(current_opt.fb)
            break
        else
            push!(fb, (current_opt, res))
            current_opt = current_opt.fb
            @warn("Using fallback method. Please ignore previous optimisation failure warnings.")
        end
    end
    return isempty(fb) ? res : factory(res, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Optimise with a [`TimeDependent`](@ref) schedule standing in for the optimiser, outside any fold loop.

There are no folds to index, so the schedule resolves to its `default` and that optimiser runs (see [`reset_time_dependent_estimator`](@ref)); a schedule with no `default` throws a [`TimeDependentDefaultError`](@ref). Inside a fold loop this method is never reached — the loop resolves entry `i` first.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
  - [`cross_val_predict`](@ref)
"""
function optimise(td::TD_OptE_Opt, args...; kwargs...)
    return optimise(reset_time_dependent_estimator(td), args...; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `res` is a valid internal optimisation result.

Default no-op. Overridden for result types that must satisfy internal constraints before use.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function assert_internal_optimiser(::NonFiniteAllocationOptimisationResult)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `res` is a valid external optimisation result.

Default no-op. Overridden for result types that must satisfy external interface constraints.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function assert_external_optimiser(::NonFiniteAllocationOptimisationResult)::Nothing
    return nothing
end

export optimise
public _optimise
