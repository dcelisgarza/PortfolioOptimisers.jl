"""
    port_opt_view(opt::AbstractOptimisationEstimator, i, args...) -> AbstractOptimisationEstimator

Return an optimisation estimator unchanged under an asset-subset view.

This is the fallback for an optimisation estimator with no view method of its own. An estimator declared through `@propagatable` gets a generated method that views each asset-indexed field it carries, and a head such as [`JuMPOptimiser`](@ref) writes its own. This method serves the rest, so an asset-indexed field of such an estimator keeps the width of the full universe.

# Arguments

  - `opt`: The optimisation estimator.
  - `i`: The asset index of the subset: an asset subset of a fold, a cluster of a nested optimisation, or the Investable Mask.
  - `args...`: Ignored. A caller that views a tracking estimator passes the unreduced returns matrix here.

# Returns

  - `opt`: The estimator, unchanged.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`NestedClustered`](@ref)
  - [`investable_reduction`](@ref)
"""
function port_opt_view(opt::AbstractOptimisationEstimator, ::Any, args...)
    return opt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

A precomputed optimisation result cannot be restricted to an asset subset.

Its weights were solved over the full universe, and a sub-portfolio of them has no defined meaning, so an asset-subset view of a result throws. A [`TimeDependent`](@ref) schedule that holds a result is therefore refused by asset-subsampling cross-validation ([`MultipleRandomised`](@ref)), whose fold loop views the optimiser to the asset subset of each fold before the swap. The all-assets view (`Colon`) passes the result through unchanged.

# Validation

  - `i` must be `Colon()`. Any other index throws an `ArgumentError`.

# Returns

  - `res`: The result, unchanged.

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
# A precomputed fallback answers on the universe it was solved on, so the view of its
# holder keeps it. See `view_child`.
function view_child(res::NonFiniteAllocationOptimisationResult, ::Any, args...)
    return res
end
"""
    non_investable_universe(opt, ni::VecStr)

Declare the Non-Investable Axis on the [`UniverseSets`](@ref) that an optimisation estimator carries.

A door calls this **after** it has viewed the estimator, and no other code calls it. An axis declared before the view would not survive it: [`port_opt_view`](@ref)`(::UniverseSets, i)` drops the axis, so that a cluster of a nested optimisation does not inherit the departures of its parent and charge each of them again. So the door reads the departed names off the *unreduced* returns data, takes the view, and declares the axis on the viewed estimator.

The generic method returns `opt` unchanged. That is correct for every estimator that carries no sets, because it has no axis to declare. A head that owns a `sets` field writes one method beside its own [`port_opt_view`](@ref), and a head that reaches one through a nested optimiser forwards to the method of that optimiser. Each method is one line, and it is written per type rather than derived by reflection, for the same reason as [`port_opt_view`](@ref): the type states what its field means.

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

A Prior Estimator fits on the Coverage Universe and returns a result on the full asset universe, in which an asset that it could not estimate carries `NaN` in `mu` and on the diagonal of `sigma`. No code after the fit can solve over such an asset. So the reduction happens once, at the entry of the optimiser, and the same index slices every constraint that the caller stated over the full universe.

Every optimisation family reduces through this one verb. That is why it is bound to [`AbstractOptimisationEstimator`](@ref) and lives here rather than beside the JuMP prelude: the hierarchical, naive and meta files load before that prelude and reach this verb without a back reference. A meta-optimiser composes two masks, its own at its entry and the mask of each inner head inside the solve of that head, and both expand.

The branch is dispatch rather than a condition, over three methods. The first derives the mask. The `nothing` method is the all-investable path and returns its arguments unchanged. The `BitVector` method takes the three views. A universe with nothing to exclude therefore costs the derivation of the mask and nothing more: no view, no copy of the data and no expansion.

The optimiser is viewed with `pr.X`, the *unreduced* returns matrix, because [`port_opt_view`](@ref) slices a tracking estimator against it by the same asset index. The prior is reduced after the optimiser, so the view reads the full matrix.

The mask stays a `BitVector` because the expansion needs the length of the full universe, and nothing else carries that length once the prior is reduced. The three views take `findall(imsk)` instead, which is the integer index that every other caller of [`port_opt_view`](@ref) passes.

# Algorithm

 1. Derive the Investable Mask from the fitted prior with [`investable_mask`](@ref).
 2. Return the mask, the prior, the optimiser and the returns data unchanged when the mask is `nothing`.
 3. Otherwise read the departed names `ni` off the *unreduced* `rd.nx` with [`non_investable_names`](@ref), and announce them once with [`announce_non_investable`](@ref).
 4. Take a [`port_opt_view`](@ref) of each of the three at `findall(imsk)`.
 5. Declare the Non-Investable Axis on the viewed optimiser with [`non_investable_universe`](@ref), and return it beside the mask and the other two views. The axis is declared after the view because the view drops it, so a name-keyed constraint stated for a departed asset resolves here and nowhere deeper.

# Arguments

  - $(arg_dict[:pr])
  - `opt::AbstractOptimisationEstimator`: The optimisation estimator, holding every constraint estimator the caller stated over the full universe.
  - $(arg_dict[:rd])

# Validation

  - At least one asset must be investable. [`investable_mask`](@ref) throws an `IsEmptyError` otherwise.

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

This is the twin of [`investable_reduction`](@ref) for a head that fits no prior. No Prior Result exists to derive an Investable Mask from, so the head derives the Coverage Universe of its own window through the verb that the priors use. An asset is kept when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window. A stale finite price during an inactive spell is therefore outside the mask, and the asset gets a zero weight.

Downstream, the two masks are one object. The head carries the Coverage Universe as the `imsk` of its result, and the keyword constructor of the result expands the weights through [`expand_investable_weights`](@ref), as it does in every other family. So every optimisation result of the library carries a mask, and a reader meets one idiom.

The branch is dispatch rather than a condition, over three methods, as in [`investable_reduction`](@ref). The first derives the mask. The `nothing` method is the all-covered path and returns its arguments unchanged. The `BitVector` method takes the two views. The optimiser is viewed with `rd.X`, the *unreduced* returns matrix, because [`port_opt_view`](@ref) slices an estimator against it by the same asset index.

# Algorithm

 1. Derive the Coverage Universe `cmsk` of `rd.X` and `rd.pnl` with [`coverage_mask`](@ref).
 2. Return the mask, the optimiser and the returns data unchanged when the mask is `nothing`.
 3. Otherwise read the departed names `ni` off the *unreduced* `rd.nx` with [`non_investable_names`](@ref), and announce them once with [`announce_non_investable`](@ref).
 4. Take a [`port_opt_view`](@ref) of the optimiser and of the returns data at `findall(cmsk)`.
 5. Declare the Non-Investable Axis on the viewed optimiser with [`non_investable_universe`](@ref), and return it beside the mask and the viewed returns data.

# Arguments

  - `opt::AbstractOptimisationEstimator`: The optimisation estimator, holding every constraint estimator the caller stated over the full universe.
  - $(arg_dict[:rd])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])
  - At least one asset must be in the Coverage Universe. [`coverage_mask`](@ref) throws an `IsEmptyError` otherwise, so every prior-free head refuses an all-dead window in one place.

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

The optimiser solves over the assets that the Investable Mask keeps, so its weight vector is shorter than the universe that the caller stated. This puts each solved weight back at its own asset and writes a zero at every other asset. A zero is what a non-investable asset holds, because the optimiser could not trade it.

A failed solve carries `NaN` at every solved position. The expansion keeps two cases apart: `NaN` where the optimiser tried and failed, and zero where it could not try. It does not flatten both to zero.

The `nothing` mask returns the weights unchanged, so nothing is copied when every asset is investable. A `nothing` weight vector stays `nothing`, which is what a naive head records when its finaliser gave up. The vector-of-vectors method serves the efficient-frontier route, which records one weight vector per sweep point. [`JuMPOptimisationSolution`](@ref) carries its own methods beside the JuMP prelude, and they delegate to the plain weight vector here, so one length check and one message serve every family.

The **keyword** constructor of each optimisation result is the caller, and every family builds its result through it. The positional constructor never expands, because every rebuild of a return code goes through it, and a second pass would expand twice.

# Mathematical definition

```math
\\begin{align}
\\tilde{w}_{i} &= \\begin{cases}
    w_{k(i)} & \\text{if } m_{i} = 1\\,, \\\\
    0 & \\text{if } m_{i} = 0\\,,
\\end{cases} \\quad i = 1, \\ldots, N\\,, \\\\
k(i) &= \\sum_{j=1}^{i} m_{j}\\,.
\\end{align}
```

Where:

  - ``\\tilde{w}_{i}``: Weight of asset ``i`` on the full asset universe.
  - ``w_{k}``: Solved weight at position ``k`` of the investable assets, in their order in the full universe.
  - ``m_{i}``: Entry ``i`` of the Investable Mask, ``1`` when asset ``i`` is investable.
  - ``k(i)``: Position of asset ``i`` among the investable assets.
  - $(math_dict[:N])

The expansion keeps the sum of the weights, ``\\sum_{i=1}^{N} \\tilde{w}_{i} = \\sum_{k} w_{k}``, so a budget stated over the investable assets holds over the full universe.

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

Run a portfolio optimisation with the estimator `opt`, and return an [`OptimisationResult`](@ref).

When the solve of `opt` returns an [`OptimisationFailure`](@ref), the fallback in `opt.fb` runs next, and the chain continues until one attempt succeeds or no fallback is left. A fallback can also be a precomputed result, which is then the answer.

A result passed as `opt` is returned unchanged.

# Arguments

  - `opt`: Optimisation estimator (for example a [`JuMPOptimisationEstimator`](@ref) subtype), or a result to pass through.
  - `args`: Positional arguments forwarded to the solve of the estimator, usually the [`ReturnsResult`](@ref). A result ignores them.
  - `kwargs`: Keyword arguments forwarded to the solve of the estimator. A result ignores them.

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
    _optimise(opt, args...; kwargs...)

Solve one optimisation estimator once, with no fallback.

[`optimise`](@ref) calls it for each attempt of the fallback chain, and it is the method that a new optimisation estimator writes. Each estimator type writes its own method, and the method returns the result type of that estimator. A precomputed result in the chain answers itself, because it has nothing to solve.

# Arguments

  - `opt`: Optimisation estimator (for example [`MeanRisk`](@ref) or [`RiskBudgeting`](@ref)), or a precomputed result.
  - `args`: The data that the estimator solves over: a [`ReturnsResult`](@ref) for a continuous optimiser, or a [`FiniteAllocationInput`](@ref) for a finite allocator.
  - `kwargs`: The keyword arguments that the estimator reads. A continuous optimiser reads `dims`, the observation dimension, and a JuMP head also reads `str_names` and `save`, which name the model variables and keep the model on the result. Every method ignores the keywords it does not read.

# Returns

  - `res::OptimisationResult`: The result of the estimator. Its `retcode` decides whether [`optimise`](@ref) runs the fallback.

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

Solve an optimisation estimator, and walk its fallback chain when a solve fails.

The answer is the first attempt that succeeds, or the last failure when every attempt fails. The `fb` field of that answer records the `(estimator, result)` pair of each failed attempt, in the order they ran (see [`FbChain`](@ref)). When no fallback was needed, `fb` is `nothing`. A fallback that is a precomputed result ends the chain: it has nothing to solve, and it is the answer whatever its return code.

This is a fold-less entry point, so a time-dependent schedule has no fold to select an entry from. The estimator is reset to its fold-less values before the solve (see [`reset_time_dependent_estimator`](@ref)). A scheduled fallback resets to its `default`, or to `nothing` when it has none, *before* the chain is walked. Inside a fold loop the reset changes nothing, because the loop resolves every schedule before it optimises.

It is a batch fit, so [`assert_batch_entry`](@ref) refuses an [`Online`](@ref) anywhere in the tree of the estimator before any solve. The wrapper resolves only at the warm-up of the online arm of the fold loop, and a plain `optimise` runs no warm-up. The read-out of a stepped estimator, `optimise(opt)`, never meets this refusal, because the warm-up that seeded its buffer replaced the wrapper.

# Algorithm

 1. Refuse an [`Online`](@ref) in the tree of `opt` with [`assert_batch_entry`](@ref).
 2. Reset every time-dependent schedule of `opt` to its fold-less value with [`reset_time_dependent_estimator`](@ref), giving `current_opt`.
 3. Solve `current_opt` with [`_optimise`](@ref), giving `res`. A precomputed result gives itself.
 4. Stop when `res.retcode` is an [`OptimisationSuccess`](@ref), when `current_opt` is a precomputed result, or when `current_opt.fb` is `nothing`.
 5. Otherwise record `(current_opt, res)` in the chain `fb`, take `current_opt.fb` as the new `current_opt`, warn that the fallback runs, and return to step 3.
 6. Return `res` when `fb` is empty. Otherwise return `res` rebuilt with `fb` through [`factory`](@ref).

# Arguments

  - `opt::OptimisationEstimator`: The optimisation estimator to use.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])

# Validation

  - No field in the tree of `opt` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise.

# Returns

  - `res::OptimisationResult`: The answer of the chain, carrying the failed attempts in `fb`.

# Related

  - [`_optimise`](@ref)
  - [`FbChain`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
  - [`assert_batch_entry`](@ref)
"""
function optimise(opt::OptimisationEstimator, args...; kwargs...)
    assert_batch_entry(opt, "`optimise`")
    fb = Tuple{OptimisationEstimator, OptimisationResult}[]
    current_opt = reset_time_dependent_estimator(opt)
    res = _optimise(current_opt, args...; kwargs...)
    # A precomputed result answers itself and ends the chain: its own `fb` records how it
    # was answered, and is not a fallback to walk on to.
    while !isa(res.retcode, OptimisationSuccess) &&
          isa(current_opt, OptimisationEstimator) &&
          !isnothing(current_opt.fb)
        push!(fb, (current_opt, res))
        current_opt = current_opt.fb
        @warn("Using fallback method. Please ignore previous optimisation failure warnings.")
        res = _optimise(current_opt, args...; kwargs...)
    end
    return isempty(fb) ? res : factory(res, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Optimise with a [`TimeDependent`](@ref) schedule standing in for the optimiser, outside any fold loop.

There are no folds to index, so the schedule resolves to its `default`, and that optimiser runs (see [`reset_time_dependent_estimator`](@ref)). A fold loop never reaches this method, because the loop resolves the entry of each fold first.

# Validation

  - The schedule must carry a `default`. A [`TimeDependentDefaultError`](@ref) is thrown otherwise.

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

Accept a precomputed optimisation result where an optimiser must run on assets that it did not choose.

A meta-optimiser calls `assert_internal_optimiser` on an inner optimiser, and a cross-validation entry calls it on the optimiser of each fold. The methods for estimators refuse a precomputed input that a subset of the assets cannot reuse. A result holds no estimator to refit, so it passes. An asset-subset view of it still throws, in [`port_opt_view`](@ref).

# Returns

  - `nothing`.

# Related

  - [`assert_external_optimiser`](@ref)
  - [`port_opt_view`](@ref)
  - [`NestedClustered`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function assert_internal_optimiser(::NonFiniteAllocationOptimisationResult)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accept a precomputed optimisation result where an optimiser must run on a universe that it does not know in advance.

A meta-optimiser calls `assert_external_optimiser` on its outer optimiser, which solves over synthetic assets, and a cross-validation entry calls it on the optimiser of each fold. The methods for estimators refuse a precomputed prior or constraint that such a universe cannot reuse. A result holds no estimator to refit, so it passes.

# Returns

  - `nothing`.

# Related

  - [`assert_internal_optimiser`](@ref)
  - [`NestedClustered`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function assert_external_optimiser(::NonFiniteAllocationOptimisationResult)::Nothing
    return nothing
end

export optimise
public _optimise
