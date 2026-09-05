"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Exposure Estimator types.

An Exposure Estimator produces one Factor Exposure: an asset's loading on one factor at every observation, built from Descriptors, from a one-hot Panel Field, or from nothing at all. The estimator is configuration, so it names the Descriptors and the Panel Fields it reads and holds no data.

All concrete types producing a Factor Exposure should be subtypes of `AbstractExposureEstimator`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractExposureEstimator` and implement the following methods:

## `factor_exposure`

  - [`factor_exposure(xe::AbstractExposureEstimator, rd::ReturnsResult)`](@ref): Computes the Factor Exposure of a carrier.

### Arguments

  - `xe`: The concrete subtype instance.
  - `rd`: The returns result that carries the Asset Panel.

### Returns

  - `L::Array{<:Real}`: The Factor Exposure, `observations × assets` for a member producing one factor, `observations × assets × factors` for a member expanding one Panel Field into many, `NaN` wherever the active mask is `false`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`factor_exposure`](@ref)
  - [`CompositeExposure`](@ref)
  - [`DerivedExposure`](@ref)
  - [`OneHotExposure`](@ref)
  - [`ConstantExposure`](@ref)
  - [`AbstractDescriptorEstimator`](@ref)
  - [`AssetPanel`](@ref)
"""
abstract type AbstractExposureEstimator <: AbstractEstimator end
"""
    factor_exposure(xe::AbstractExposureEstimator, rd::ReturnsResult) -> Array{<:Real}

Compute the Factor Exposure of a carrier.

This is the verb every Exposure Estimator answers. A member that produces one factor returns an `observations × assets` matrix, and a member that expands one Panel Field into many returns an `observations × assets × factors` array. Every member follows two conventions: the value at an observation uses information up to and including that observation, and every cell where the active mask of the Asset Panel is `false` is `NaN`.

[`DerivedExposure`](@ref) reads the exposure of another factor, so it answers a three-argument method and refuses this one. The caller that holds the factor list computes the factors in dependency order and passes the source exposure.

# Arguments

  - `xe`: Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Returns

  - `L::Array{<:Real}`: The Factor Exposure.

# Related

  - [`AbstractExposureEstimator`](@ref)
  - [`CompositeExposure`](@ref)
  - [`DerivedExposure`](@ref)
  - [`OneHotExposure`](@ref)
  - [`ConstantExposure`](@ref)
  - [`descriptor`](@ref)
  - [`ReturnsResult`](@ref)
"""
function factor_exposure end
"""
    assert_exposure_family(family::AbstractString) -> nothing

Check that a Factor Family label is not the empty string.

Every Exposure Estimator carries the label of the Factor Family its factor belongs to, and the label is a key: the family basis, the neutralisation and the attribution all group the factors by it. An empty label names no family, so it is refused in the constructor.

# Arguments

  - `family`: The Factor Family label.

# Validation

  - `!isempty(family)`. Raises an [`IsEmptyError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`AbstractExposureEstimator`](@ref)
  - [`CompositeExposure`](@ref)
  - [`ConstantExposure`](@ref)
"""
function assert_exposure_family(family::AbstractString)::Nothing
    @argcheck(!isempty(family),
              IsEmptyError("family labels the Factor Family the factor belongs to, so it cannot be the empty string"))
    return nothing
end
"""
    exposure_benchmark_weights(rd::ReturnsResult, name::AbstractString) -> Matrix{<:Real}

Read the benchmark weights a cross-sectional transform of an exposure is weighted by.

A benchmark weight is a selector first and a weight second, so a cell that carries no weight is out of the estimation set of its observation rather than in it with an unknown weight. This reads the named Panel Field through [`panel_field_values`](@ref) and writes a zero into every cell the Asset Panel does not observe and every cell it does not activate, which is the one shape [`cross_sectional_transform`](@ref) accepts.

# Algorithm

 1. Read the named numeric Panel Field.
 2. Write a zero into every cell that is not finite, and into every cell where the active mask is `false`.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `name`: Name of the numeric Panel Field holding the benchmark weights.

# Validation

  - The rules of [`panel_field_values`](@ref).

# Returns

  - `W::Matrix{<:Real}`: The benchmark weights, `observations × assets`, zero where the cell is unobserved or inactive.

# Related

  - [`CompositeExposure`](@ref)
  - [`DerivedExposure`](@ref)
  - [`panel_field_values`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
function exposure_benchmark_weights(rd::ReturnsResult, name::AbstractString)::Matrix{<:Real}
    W = panel_field_values(rd, name)
    exposure_weight_fill!(W, rd.pnl)
    return W
end
"""
    exposure_weight_fill!(W::AbstractMatrix{<:Real}, pnl::AssetPanel) -> nothing

Write a zero into every benchmark weight the Asset Panel does not observe or does not activate, in place.

# Arguments

  - `W`: The benchmark weights, `observations × assets`, changed in place.
  - `pnl`: The Asset Panel whose active mask is read.

# Validation

  - `size(W) == size(pnl.amsk)`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`. `W` carries the filled weights.

# Related

  - [`exposure_benchmark_weights`](@ref)
  - [`AssetPanel`](@ref)
"""
function exposure_weight_fill!(W::AbstractMatrix{<:Real}, pnl::AssetPanel)::Nothing
    amsk = pnl.amsk
    @argcheck(size(W) == size(amsk),
              DimensionMismatch("the benchmark weights are observations × assets, so they must match the active mask of the Asset Panel, got size(W) = $(size(W)) and size(pnl.amsk) = $(size(amsk))"))
    Tf = eltype(W)
    for k in CartesianIndices(W)
        if !isfinite(W[k]) || !amsk[k]
            W[k] = zero(Tf)
        end
    end
    return nothing
end
"""
    exposure_group_labels(rd::ReturnsResult, group::Nothing) -> nothing
    exposure_group_labels(rd::ReturnsResult, group::AbstractString) -> Matrix{Int}

Return the group labels the cross-sectional transforms of an exposure partition each observation by.

A member that names no grouping field transforms each observation as one cross-section, and a member that names one transforms each group of the observation on its own.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `group`: Name of the categorical Panel Field to group by, or `nothing`.

# Validation

  - The rules of [`cross_sectional_groups`](@ref).

# Returns

  - `groups::Option{<:Matrix{Int}}`: The group labels, `observations × assets`, or `nothing`.

# Related

  - [`CompositeExposure`](@ref)
  - [`DerivedExposure`](@ref)
  - [`cross_sectional_groups`](@ref)
  - [`CS_MISSING_GROUP`](@ref)
"""
function exposure_group_labels(::ReturnsResult, ::Nothing)::Nothing
    return nothing
end
function exposure_group_labels(rd::ReturnsResult, group::AbstractString)::Matrix{Int}
    return cross_sectional_groups(rd.pnl, group)
end
"""
    exposure_transform(ct::Nothing, X::MatNum, w::Option{<:MatNum},
                       groups::Option{<:AbstractMatrix{<:Integer}}) -> MatNum
    exposure_transform(ct::AbstractCrossSectionalTransform, X::MatNum, w::Option{<:MatNum},
                       groups::Option{<:AbstractMatrix{<:Integer}}) -> Matrix{<:Real}

Apply one optional cross-sectional transform slot of an Exposure Estimator.

An exposure holds its outlier slot and its scoring slot as `Option{<:AbstractCrossSectionalTransform}`, where `nothing` says that the step is skipped. This is the one place that reading is written, so a member states its two slots and never its two branches.

# Arguments

  - `ct`: The cross-sectional transform, or `nothing`.
  - `X`: Data matrix `observations × assets`.
  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.
  - `groups`: Group label matrix `observations × assets`, or `nothing`.

# Returns

  - `Y::MatNum`: `X` itself when `ct` is `nothing`, the transformed matrix otherwise.

# Related

  - [`CompositeExposure`](@ref)
  - [`DerivedExposure`](@ref)
  - [`cross_sectional_transform`](@ref)
  - [`AbstractCrossSectionalTransform`](@ref)
"""
function exposure_transform(::Nothing, X::MatNum, ::Option{<:MatNum},
                            ::Option{<:AbstractMatrix{<:Integer}})::MatNum
    return X
end
function exposure_transform(ct::AbstractCrossSectionalTransform, X::MatNum,
                            w::Option{<:MatNum},
                            groups::Option{<:AbstractMatrix{<:Integer}})::MatNum
    return cross_sectional_transform(ct, X; w = w, groups = groups)
end
"""
    exposure_active_fill!(L::AbstractMatrix{<:Real}, pnl::AssetPanel) -> nothing
    exposure_active_fill!(L::AbstractArray{<:Real, 3}, pnl::AssetPanel) -> nothing

Write `NaN` into every cell of a Factor Exposure where the active mask of the Asset Panel is `false`, in place.

An asset that is not listed at an observation has no Factor Exposure there, whatever its Panel Fields hold. A member that builds its exposure out of Descriptors inherits the convention from [`descriptor_active_fill!`](@ref), and a member that builds one from a Panel Field or from nothing at all calls this.

# Arguments

  - `L`: The Factor Exposure, `observations × assets` or `observations × assets × factors`, changed in place.
  - `pnl`: The Asset Panel whose active mask is read.

# Validation

  - The first two axes of `L` match the active mask. Raises a `DimensionMismatch`.

# Returns

  - `nothing`. `L` carries the filled Factor Exposure.

# Related

  - [`factor_exposure`](@ref)
  - [`OneHotExposure`](@ref)
  - [`ConstantExposure`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function exposure_active_fill!(L::AbstractMatrix{<:Real}, pnl::AssetPanel)::Nothing
    return descriptor_active_fill!(L, pnl)
end
function exposure_active_fill!(L::AbstractArray{<:Real, 3}, pnl::AssetPanel)::Nothing
    amsk = pnl.amsk
    @argcheck(size(L)[1:2] == size(amsk),
              DimensionMismatch("a one-hot Factor Exposure is observations × assets × factors, so its first two axes must match the active mask of the Asset Panel, got size(L) = $(size(L)) and size(pnl.amsk) = $(size(amsk))"))
    Tf = eltype(L)
    for k in CartesianIndices(L)
        if !amsk[k[1], k[2]]
            L[k] = Tf(NaN)
        end
    end
    return nothing
end

export factor_exposure
