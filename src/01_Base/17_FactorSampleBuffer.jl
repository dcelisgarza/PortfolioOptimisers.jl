"""
$(DocStringExtensions.TYPEDEF)

Carries the two matrices an estimator keeps when its batch verb reads a returns matrix and a factor matrix.

The paired buffer of the [`partial_fit!`](@ref) seam. A [`SampleBufferState`](@ref) holds one matrix, and `partial_fit!(est, x)` folds one observation, so a family whose batch verb is `prior(pe, X, F)` — a factor prior, which regresses the asset returns on the factor returns — has two sequences to keep and no single buffer that keeps them. This is the pair, and every operation of the seam is the operation of its two halves.

Each half carries whatever a [`SampleBufferState`](@ref) carries, per-observation masks included; a mask describes assets, so it reaches the returns half alone.

The halves are **independent buffers over independent widths**. A returns matrix describes assets and a factor matrix describes factors, and neither count constrains the other, so the two widths are fixed and checked separately. What they share is the observation axis: one call to [`partial_fit!`](@ref) appends one row to each, so the two halves carry the same number of observations and the `t`-th row of one is contemporaneous with the `t`-th row of the other.

[`Online`](@ref)'s cap reaches both halves, because it windows the whole fit.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorSampleBufferState(;
        X::SampleBufferState = SampleBufferState(),
        F::SampleBufferState = SampleBufferState()
    ) -> FactorSampleBufferState

Keywords correspond to the struct's fields. The default is the empty seed [`Online`](@ref) builds for a family whose [`online_state_seed`](@ref) asks for a pair: both halves empty, and the first append to each fixing that half's width and element type.

## Validation

  - Both halves carry the same cap. An `ArgumentError` is thrown otherwise.
  - Both halves hold the same number of observations. A `DimensionMismatch` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its halves are treated differently:

  - `X`: Sliced to the selected assets via [`port_opt_view`](@ref).
  - `F`: Copied unchanged, because the selection indexes assets and this half describes factors.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`online_state_seed`](@ref)
  - [`partial_fit!`](@ref)
"""
@concrete struct FactorSampleBufferState <: AbstractPartialFitState
    """
    Buffer of the asset returns, `observations × assets`.
    """
    X
    """
    Buffer of the factor returns, `observations × factors`.
    """
    F
end
function FactorSampleBufferState(; X::SampleBufferState = SampleBufferState(),
                                 F::SampleBufferState = SampleBufferState())::FactorSampleBufferState
    assert_factor_sample_buffer_state(X, F)
    return FactorSampleBufferState(X, F)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a [`FactorSampleBufferState`](@ref) whose halves do not describe one sequence of observations.

The two halves are appended to together, so they must agree on the two things a joint append fixes: the cap, which windows both, and the number of observations, which is what makes the `t`-th row of one contemporaneous with the `t`-th row of the other. The two **widths** are deliberately not compared: an asset count and a factor count are unrelated.

# Arguments

  - `X`: Buffer of the asset returns.
  - `F`: Buffer of the factor returns.

# Validation

  - `X.max_history == F.max_history`. An `ArgumentError` is thrown otherwise.
  - `X.n == F.n`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`FactorSampleBufferState`](@ref)
  - [`SampleBufferState`](@ref)
"""
function assert_factor_sample_buffer_state(X::SampleBufferState, F::SampleBufferState)
    @argcheck(X.max_history == F.max_history,
              ArgumentError("the two halves of a factor sample buffer are windowed together, so they carry one cap, but the returns half has a `max_history` of $(X.max_history) and the factor half has $(F.max_history)."))
    @argcheck(X.n == F.n,
              DimensionMismatch("the two halves of a factor sample buffer are appended to together, so they hold one number of observations, but the returns half holds $(X.n) and the factor half holds $(F.n)."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the paired sample buffer an estimator carries, and refuses an estimator that carries none.

The two-matrix twin of [`assert_sample_buffer`](@ref). A family whose batch verb reads a returns matrix and a factor matrix reads its state through this function, so an estimator wrapped for the wrong shape — or never wrapped at all — meets a message naming the pair rather than a `MethodError`.

# Arguments

  - `est`: Estimator whose `cache` field carries the pair.

# Validation

  - `est` has a `cache` field holding a [`FactorSampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `state::FactorSampleBufferState`: The pair the estimator carries.

# Related

  - [`FactorSampleBufferState`](@ref)
  - [`assert_sample_buffer`](@ref)
  - [`Online`](@ref)
"""
function assert_factor_sample_buffer(est::Union{<:AbstractEstimator,
                                                <:StatsBase.CovarianceEstimator})
    cache = hasfield(typeof(est), :cache) ? getfield(est, :cache) : nothing
    @argcheck(isa(cache, FactorSampleBufferState),
              ArgumentError("`$(typeof(est))` carries no factor sample buffer, so it has nowhere to keep the two matrices its refit reads. Wrap it in `Online` and resolve the wrapper with `update_online_estimator` before folding, which seeds the pair through `online_state_seed`."))
    return cache::FactorSampleBufferState
end
"""
    partial_fit!(state::FactorSampleBufferState, x::VecNum, f::VecNum)
    partial_fit!(state::FactorSampleBufferState, X::MatNum, F::MatNum; dims::Int = 1)

Folds a returns observation and its contemporaneous factor observation into a [`FactorSampleBufferState`](@ref).

Each half is folded by [`partial_fit!`](@ref) on [`SampleBufferState`](@ref), so the growth rule, the cap, the width check and the per-observation masks are the single buffer's, unchanged. The pair is then checked, which is what refuses a block of returns and a block of factors of different lengths — the one error a caller can make here that neither half can see on its own.

A [`CoveragePolicy`](@ref) mask describes **assets**, so it travels into the returns half alone. A factor is not an asset and no verb of the library masks one, so the factor half takes the rows and nothing else.

# Arguments

  - `state`: The pair to fold into.
  - `x`: One returns observation, whose entries are the assets.
  - `f`: One factor observation, whose entries are the factors.
  - $(arg_dict[:X])
  - `F`: Factor returns to fold, oriented as `X` is.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, forwarded to the **returns** half alone.

# Validation

  - $(val_dict[:dims])
  - `X` and `F` carry the same number of observations. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::FactorSampleBufferState`: The pair after the last observation.

# Related

  - [`FactorSampleBufferState`](@ref)
  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(state::FactorSampleBufferState, x::VecNum, f::VecNum; kwargs...)
    state = Accessors.@reset state.X = partial_fit!(state.X, x; kwargs...)
    state = Accessors.@reset state.F = partial_fit!(state.F, f)
    assert_factor_sample_buffer_state(state.X, state.F)
    return state
end
function partial_fit!(state::FactorSampleBufferState, X::MatNum, F::MatNum; dims::Int = 1,
                      kwargs...)
    state = Accessors.@reset state.X = partial_fit!(state.X, X; dims = dims, kwargs...)
    state = Accessors.@reset state.F = partial_fit!(state.F, F; dims = dims)
    assert_factor_sample_buffer_state(state.X, state.F)
    return state
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`FactorSampleBufferState`](@ref) fitted on disjoint blocks into the pair of the concatenated block.

The merge of each half, which is concatenation, so the pair merges exactly for the reason the single buffer does. The result is checked, which refuses a pair whose halves would fall out of step.

# Arguments

  - `a`: The pair of the first block of observations.
  - `b`: The pair of the second block of observations.

# Returns

  - `state::FactorSampleBufferState`: The pair the two blocks give when they are folded as one block.

# Related

  - [`FactorSampleBufferState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(a::FactorSampleBufferState, b::FactorSampleBufferState)
    return FactorSampleBufferState(; X = merge_states(a.X, b.X), F = merge_states(a.F, b.F))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`FactorSampleBufferState`](@ref), so the copy shares no array with the original.

# Arguments

  - `x`: The pair to copy.

# Returns

  - `state::FactorSampleBufferState`: A fresh pair, equal to `x`, whose halves share no array with it.

# Related

  - [`FactorSampleBufferState`](@ref)
  - [`partial_fit`](@ref)
"""
function Base.copy(x::FactorSampleBufferState)
    return FactorSampleBufferState(copy(x.X), copy(x.F))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`FactorSampleBufferState`](@ref) to the selected assets.

The selection indexes **assets**, so the returns half is sliced and the factor half is not: a factor is not an asset, and a fit over a subset of the universe reads the same factors. The factor half is copied rather than shared, for the reason the returns half is copied — a later fold on the viewed estimator must not write through into the estimator the view was taken from.

# Arguments

  - `x`: The pair to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments, forwarded to the returns half.

# Returns

  - `state::FactorSampleBufferState`: The pair of the same observations over the selected assets.

# Related

  - [`FactorSampleBufferState`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(x::FactorSampleBufferState, i, args...)
    return FactorSampleBufferState(port_opt_view(x.X, i, args...), copy(x.F))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds a returns block and its factor block into the paired buffer an estimator carries.

The two-matrix arm of [`partial_fit!`](@ref), and the method every estimator carrying a [`FactorSampleBufferState`](@ref) reaches. It is the one-matrix buffering arm with a second matrix: it reads the pair out of the `cache` field, folds both halves, and rebinds the field. A [`CoveragePolicy`](@ref) mask is carried, as the one-matrix arm carries it, and it reaches the returns half alone.

# Algorithm

 1. Read the pair out of the `cache` field with [`assert_factor_sample_buffer`](@ref).
 2. Fold two matrices through the block arm and two vectors through the single-observation arm.
 3. Rebuild `est` with its `cache` rebound, and return it.

# Arguments

  - `est`: Estimator whose buffer is folded forward.
  - `X`: Returns to fold, one observation per row when `dims == 1` and one per column when `dims == 2`, or a vector holding one observation.
  - `F`: Factor returns to fold, oriented as `X` is.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, forwarded to the returns half of the pair.

# Validation

  - `est` carries a [`FactorSampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.
  - `X` and `F` are both matrices or both vectors. A `MethodError` is thrown otherwise.

# Returns

  - `est`: The estimator, with its `cache` field rebound to the pair after the last observation.

# Related

  - [`FactorSampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`Online`](@ref)
"""
function partial_fit!(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator},
                      X::VecNum_MatNum, F::VecNum_MatNum; dims::Int = 1, kwargs...)
    state = assert_factor_sample_buffer(est)
    state = if isa(X, MatNum) && isa(F, MatNum)
        partial_fit!(state, X, F; dims = dims, kwargs...)
    else
        partial_fit!(state, X::VecNum, F::VecNum; kwargs...)
    end
    # `rebuild_estimator` for the reason the one-matrix arm uses it: every prior that
    # buffers declares forwarded properties, which `Accessors.@reset` refuses.
    return rebuild_estimator(est, (; cache = state))
end
