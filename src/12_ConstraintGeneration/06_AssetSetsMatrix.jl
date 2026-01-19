"""
    struct AssetSetsMatrixEstimator{T1} <: AbstractConstraintEstimator
        val::T1
    end

Estimator for constructing asset set membership matrices from asset groupings.

`AssetSetsMatrixEstimator` is a container type for specifying the key or group name used to generate a binary asset-group membership matrix from an [`AssetSets`](@ref) object. This is used in constraint generation and portfolio construction workflows that require mapping assets to groups or categories.

# Fields

  - `val`: The key or group name to extract from the asset sets.

# Constructor

    AssetSetsMatrixEstimator(; val::AbstractString)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(val)`.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx",
                        dict = Dict("nx" => ["A", "B", "C"],
                                    "nx_sector" => ["Tech", "Tech", "Finance"]));

julia> est = AssetSetsMatrixEstimator(; val = "nx_sector")
AssetSetsMatrixEstimator
  val ┴ String: "nx_sector"

julia> asset_sets_matrix(est, sets)
2×3 transpose(::BitMatrix) with eltype Bool:
 1  1  0
 0  0  1
```

# Related

  - [`AssetSets`](@ref)
  - [`asset_sets_matrix`]-(@ref)
  - [`AbstractConstraintEstimator`](@ref)
"""
struct AssetSetsMatrixEstimator{T1} <: AbstractConstraintEstimator
    val::T1
    function AssetSetsMatrixEstimator(val::AbstractString)
        @argcheck(!isempty(val))
        return new{typeof(val)}(val)
    end
end
function AssetSetsMatrixEstimator(; val::AbstractString)
    return AssetSetsMatrixEstimator(val)
end
const MatNum_ASetMatE = Union{<:AssetSetsMatrixEstimator, <:MatNum}
const VecMatNum_ASetMatE = AbstractVector{<:MatNum_ASetMatE}
const MatNum_ASetMatE_VecMatNum_ASetMatE = Union{<:MatNum_ASetMatE, <:VecMatNum_ASetMatE}
"""
    asset_sets_matrix(smtx::AbstractString, sets::AssetSets)

Construct a binary asset-group membership matrix from asset set groupings.

`asset_sets_matrix` generates a binary (0/1) matrix indicating asset membership in groups or categories, based on the key or group name `smtx` in the provided [`AssetSets`](@ref). Each row corresponds to a unique group value, and each column to an asset in the universe. This is used in constraint generation and portfolio construction workflows that require mapping assets to groups or categories.

# Arguments

  - `smtx`: The key or group name to extract from the asset sets.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.

# Returns

  - `A::BitMatrix`: A binary matrix of size (number of groups) × (number of assets), where `A[i, j] == 1` if asset `j` belongs to group `i`.

# Details

  - The function checks that `smtx` exists in `sets.dict` and that its length matches the asset universe.
  - Each unique value in `sets.dict[smtx]` defines a group.
  - The output matrix is transposed so that rows correspond to groups and columns to assets.

# Validation

  - `haskey(sets.dict, smtx)`.
  - Throws an `AssertionError` if the length of `sets.dict[smtx]` does not match the asset universe.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx",
                        dict = Dict("nx" => ["A", "B", "C"],
                                    "nx_sector" => ["Tech", "Tech", "Finance"]));

julia> asset_sets_matrix("nx_sector", sets)
2×3 transpose(::BitMatrix) with eltype Bool:
 1  1  0
 0  0  1
```

# Related

  - [`AssetSets`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
  - [`asset_sets_matrix_view`](@ref)
"""
function asset_sets_matrix(smtx::AbstractString, sets::AssetSets)
    @argcheck(haskey(sets.dict, smtx), KeyError("key $smtx not found in `sets.dict`"))
    all_sets = sets.dict[smtx]
    @argcheck(length(sets.dict[sets.key]) == length(all_sets),
              AssertionError("The following conditions must be met:\n`sets.dict` must contain key $smtx => haskey(sets.dict, smtx) = $(haskey(sets.dict, smtx))\nlengths of sets.dict[sets.key] and `all_sets` must be equal:\nlength(sets.dict[sets.key]) => length(sets.dict[$(sets.key)]) => $(length(sets.dict[sets.key]))\nlength(all_sets) => $(length(all_sets))"))
    unique_sets = unique(all_sets)
    A = BitMatrix(undef, length(all_sets), length(unique_sets))
    for (i, val) in pairs(unique_sets)
        A[:, i] = all_sets .== val
    end
    return transpose(A)
end
"""
    asset_sets_matrix(smtx::Option{<:MatNum}, args...)

No-op fallback for asset set membership matrix construction.

This method returns the input matrix `smtx` unchanged. It is used as a fallback when the asset set membership matrix is already provided as an `MatNum` or is `nothing`, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `smtx`: An existing asset set membership matrix (`MatNum`) or `nothing`.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `smtx::Option{<:MatNum}`: The input matrix or `nothing`, unchanged.

# Related

  - [`AssetSets`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
  - [`asset_sets_matrix`]-(@ref)
"""
function asset_sets_matrix(smtx::Option{<:MatNum}, args...)
    return smtx
end
"""
    asset_sets_matrix(smtx::AssetSetsMatrixEstimator, sets::AssetSets)

This method is a wrapper calling:

    asset_sets_matrix(smtx.val, sets)

It is used for type stability and to provide a uniform interface for processing constraint estimators, as well as simplifying the use of multiple estimators simulatneously.

# Related

  - [`asset_sets_matrix`]-(@ref)
"""
function asset_sets_matrix(smtx::AssetSetsMatrixEstimator, sets::AssetSets)
    return asset_sets_matrix(smtx.val, sets)
end
"""
    asset_sets_matrix(smtx::VecMatNum_ASetMatE,
                      sets::AssetSets)

Broadcasts [`asset_sets_matrix`]-(@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simulatneously.
"""
function asset_sets_matrix(smtx::VecMatNum_ASetMatE, sets::AssetSets)
    return [asset_sets_matrix(smtxi, sets) for smtxi in smtx]
end
function asset_sets_matrix_view(smtx::MatNum, i; kwargs...)
    return view(smtx, :, i)
end
function asset_sets_matrix_view(smtx::Option{<:AssetSetsMatrixEstimator}, ::Any; kwargs...)
    return smtx
end
function asset_sets_matrix_view(smtx::VecMatNum_ASetMatE, i; kwargs...)
    val = [asset_sets_matrix_view(smtxi, i; kwargs...) for smtxi in smtx]
    if isabstracttype(eltype(val))
        val = concrete_typed_array(val)
    end
    return val
end

export AssetSetsMatrixEstimator, asset_sets_matrix
