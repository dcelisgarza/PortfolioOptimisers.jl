"""
$(DocStringExtensions.TYPEDEF)

Estimator for constructing asset set membership matrices from asset groupings.

`AssetSetsMatrixEstimator` is a container type for specifying the key or group name used to generate a binary asset-group membership matrix from an [`AssetSets`](@ref) object. This is used in constraint generation and portfolio construction workflows that require mapping assets to groups or categories.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AssetSetsMatrixEstimator(;
        val::AbstractString
    ) -> AssetSetsMatrixEstimator

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(val)`.

# Examples

```jldoctest
julia> sets = AssetSets(; key = \"nx\",
                        dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"],
                                    \"nx_sector\" => [\"Tech\", \"Tech\", \"Finance\"]));

julia> est = AssetSetsMatrixEstimator(; val = \"nx_sector\")
AssetSetsMatrixEstimator
  val ┴ String: "nx_sector"

julia> asset_sets_matrix(est, sets)
2×3 transpose(::BitMatrix) with eltype Bool:
 1  1  0
 0  0  1
```

# Related

  - [`AssetSets`](@ref)
  - [`asset_sets_matrix`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
"""
@concrete struct AssetSetsMatrixEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:asets_val])
    """
    val
    function AssetSetsMatrixEstimator(val::AbstractString)::AssetSetsMatrixEstimator
        @argcheck(!isempty(val), IsEmptyError("val cannot be empty"))
        return new{typeof(val)}(val)
    end
end
function AssetSetsMatrixEstimator(; val::AbstractString)::AssetSetsMatrixEstimator
    return AssetSetsMatrixEstimator(val)
end
"""
    const MatNum_ASetMatE = Union{<:AssetSetsMatrixEstimator, <:MatNum}

Alias for an asset sets matrix estimator or a numeric matrix.

Matches either an [`AssetSetsMatrixEstimator`](@ref) or a plain numeric matrix. Used internally in constraint generation that accepts a pre-computed membership matrix or an estimator.

# Related

  - [`AssetSetsMatrixEstimator`](@ref)
  - [`MatNum`](@ref)
  - [`asset_sets_matrix`](@ref)
"""
const MatNum_ASetMatE = Union{<:AssetSetsMatrixEstimator, <:MatNum}
"""
    const VecMatNum_ASetMatE = AbstractVector{<:MatNum_ASetMatE}

Alias for a vector of asset sets matrix estimators or numeric matrices.

Represents a collection of [`MatNum_ASetMatE`](@ref) elements, enabling batch processing.

# Related

  - [`MatNum_ASetMatE`](@ref)
  - [`MatNum_ASetMatE_VecMatNum_ASetMatE`](@ref)
"""
const VecMatNum_ASetMatE = AbstractVector{<:MatNum_ASetMatE}
"""
    const MatNum_ASetMatE_VecMatNum_ASetMatE = Union{<:MatNum_ASetMatE, <:VecMatNum_ASetMatE}

Alias for a single or vector of asset sets matrix estimators or numeric matrices.

Matches either a single [`MatNum_ASetMatE`](@ref) or a vector of them. Used for dispatch in asset set matrix operations that accept one or many estimators or matrices.

# Related

  - [`MatNum_ASetMatE`](@ref)
  - [`VecMatNum_ASetMatE`](@ref)
"""
const MatNum_ASetMatE_VecMatNum_ASetMatE = Union{<:MatNum_ASetMatE, <:VecMatNum_ASetMatE}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

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
julia> sets = AssetSets(; key = \"nx\",
                        dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"],
                                    \"nx_sector\" => [\"Tech\", \"Tech\", \"Finance\"]));

julia> asset_sets_matrix(\"nx_sector\", sets)
2×3 transpose(::BitMatrix) with eltype Bool:
 1  1  0
 0  0  1
```

# Related

  - [`AssetSets`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
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
  - [`asset_sets_matrix`](@ref)
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

  - [`asset_sets_matrix`](@ref)
"""
function asset_sets_matrix(smtx::AssetSetsMatrixEstimator, sets::AssetSets)
    return asset_sets_matrix(smtx.val, sets)
end
"""
    asset_sets_matrix(smtx::VecMatNum_ASetMatE,
                      sets::AssetSets)

Broadcasts [`asset_sets_matrix`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simulatneously.
"""
function asset_sets_matrix(smtx::VecMatNum_ASetMatE, sets::AssetSets)
    return [asset_sets_matrix(smtxi, sets) for smtxi in smtx]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a list of taxonomy keys can produce a graded feature matrix.

Shared by [`asset_sets_features`](@ref) and [`AssetSetsFeatures`](@ref), so the estimator rejects a bad key list at construction and the bare entry point rejects it at the call, from one encoding of the rule.

# Two keys is a floor, not a style preference

Every `Distances.jl` semimetric is invariant to permuting coordinates, so a **one-hot** feature matrix — which is what a single partition gives — can only distinguish "same group" from "different group": the distance takes at most two values for *every* metric, and clustering it recovers the partition it was built from. Grading relatedness needs at least two partitions to agree or disagree about.

Duplicate keys are refused rather than deduplicated: repeating a key doubles its block, silently reweighting that partition against the others.

# Arguments

  - `vals`: Group name keys to stack into the feature axis.

# Returns

  - `nothing`.

# Validation

  - `length(vals) >= 2`.
  - `allunique(vals)`.

# Related

  - [`asset_sets_features`](@ref)
  - [`AssetSetsFeatures`](@ref)
"""
function assert_feature_keys(vals::AbstractVector{<:AbstractString})::Nothing
    @argcheck(length(vals) >= 2,
              ArgumentError("`vals` must name at least two keys, because a single partition is one-hot and its distance matrix is two-valued for every metric. Got\nlength(vals) => $(length(vals))\nvals => $vals"))
    @argcheck(allunique(vals),
              ArgumentError("`vals` must not repeat a key, because a repeated key doubles its block and silently reweights that partition. Got\nvals => $vals"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Stack taxonomy memberships into an `assets × features` feature matrix.

`asset_sets_features` concatenates one [`asset_sets_matrix`](@ref) block per key in `vals`, transposed to assets-major, into the exogenous feature matrix [`FeatureDistance`](@ref) consumes. Feature `k` reads "belongs to group `k`", so two assets are close when they are classified together across many taxonomies.

This is the **exogenous** feature source: a sector, industry or country classification is structure that return correlations do not contain, which is exactly what a feature distance exists to bring in. Every other producer in the library derives `Z` from the returns.

# Arguments

  - `vals`: Group name keys in `sets.dict`, at least two (see [`assert_feature_keys`](@ref)).
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.

# Returns

  - `Z::Matrix{Float64}`: An `assets × features` matrix, `Z[i, k] == 1` when asset `i` belongs to group `k`. The feature count is the total number of distinct group values across every key.

# Nested versus crossed taxonomies

The reading depends on how the keys relate, and both are useful:

  - **Nested** (sector ⊃ industry ⊃ sub-industry): two assets share a level only if they share every coarser one, so the cosine similarity `shared / L` *counts the classification levels they agree on* — a depth-graded relatedness.
  - **Crossed** (sector + country): the keys are independent, so the same count reads as how many independent attributes two assets happen to share.

# The row norms are equal, exactly

`asset_sets_matrix` builds its groups from `unique(all_sets)`, so **every key is a partition**: each asset lands in exactly one group per key, giving every row exactly `L = length(vals)` ones. All rows therefore have norm `sqrt(L)` and

```
cos(i, j) = shared(i, j) / L ∈ [0, 1]
```

exactly, with no standardisation needed — the only producer with that property.

# `Float64`, not `BitMatrix`

The result is dense `Float64` rather than the `BitMatrix` [`asset_sets_matrix`](@ref) returns, so [`AngularDist`](@ref) keeps its BLAS `gemm` path.

# Views

An asset view of an `AssetSets` slices the groups prefixed by `sets.key` and leaves the rest alone, so a key named for a view to reach must carry that prefix — `\"nx_sector\"`, not `\"sector\"`. An unprefixed key does not fail silently: [`asset_sets_matrix`](@ref)'s length check throws on the next call, because the sliced universe no longer matches the unsliced group.

# Validation

  - `length(vals) >= 2` and `allunique(vals)` (see [`assert_feature_keys`](@ref)).
  - Each key exists in `sets.dict` and has the length of the asset universe (enforced by [`asset_sets_matrix`](@ref)).

# Examples

```jldoctest
julia> sets = AssetSets(; key = \"nx\",
                        dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"],
                                    \"nx_sector\" => [\"Tech\", \"Tech\", \"Finance\"],
                                    \"nx_country\" => [\"US\", \"UK\", \"UK\"]));

julia> Z = asset_sets_features([\"nx_sector\", \"nx_country\"], sets)
3×4 Matrix{Float64}:
 1.0  0.0  1.0  0.0
 1.0  0.0  0.0  1.0
 0.0  1.0  0.0  1.0
```

# Feeding the user-supplied carrier

[`ReturnsResult`](@ref) requires `nz` whenever `Z` is set. Take it from [`asset_sets_feature_names`](@ref) rather than rebuilding the column order by hand:

```julia
ReturnsResult(; nx = nx, X = X, nz = asset_sets_feature_names(vals, sets),
              Z = asset_sets_features(vals, sets))
```

# Related

  - [`AssetSets`](@ref)
  - [`asset_sets_matrix`](@ref)
  - [`asset_sets_feature_names`](@ref)
  - [`assert_feature_keys`](@ref)
  - [`AssetSetsFeatures`](@ref)
  - [`FeatureDistance`](@ref)
"""
function asset_sets_features(vals::AbstractVector{<:AbstractString},
                             sets::AssetSets)::Matrix{Float64}
    assert_feature_keys(vals)
    return Float64.(reduce(hcat, (transpose(asset_sets_matrix(v, sets)) for v in vals)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Name the columns [`asset_sets_features`](@ref) produces, in their own order.

[`ReturnsResult`](@ref) requires `nz` whenever `Z` is set, so the user-supplied carrier needs these names — and reproducing the column order by hand is exactly the kind of restatement that drifts. The pair is built from one traversal of `vals`, so the two can only agree.

# Names are qualified by their key

Each name is `\"<key>=<group>\"` rather than the bare group value, because a **nested** taxonomy reuses its values across levels: an integrated-oil producer is in industry `IntegratedOil` and sub-industry `IntegratedOil`, so bare values would collide and [`ReturnsResult`](@ref)'s uniqueness check would reject them.

# Arguments

  - `vals`: The same group name keys, in the same order, passed to [`asset_sets_features`](@ref).
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.

# Returns

  - `nz::Vector{String}`: One name per feature column, `length(nz) == size(Z, 2)`.

# Examples

```jldoctest
julia> sets = AssetSets(; key = \"nx\",
                        dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"],
                                    \"nx_sector\" => [\"Tech\", \"Tech\", \"Finance\"],
                                    \"nx_country\" => [\"US\", \"UK\", \"UK\"]));

julia> asset_sets_feature_names([\"nx_sector\", \"nx_country\"], sets)
4-element Vector{String}:
 "nx_sector=Tech"
 "nx_sector=Finance"
 "nx_country=US"
 "nx_country=UK"
```

# Related

  - [`asset_sets_features`](@ref)
  - [`AssetSets`](@ref)
  - [`ReturnsResult`](@ref)
"""
function asset_sets_feature_names(vals::AbstractVector{<:AbstractString},
                                  sets::AssetSets)::Vector{String}
    assert_feature_keys(vals)
    return [string(v, "=", g) for v in vals for g in unique(sets.dict[v])]
end
"""
    port_opt_view(smtx, i; kwargs...)

Get a column view or subset of an asset sets membership matrix for asset index `i`.

Returns a column view for matrix inputs, the estimator unchanged for estimator inputs, or processes vectors element-wise.

# Arguments

  - `smtx`: Asset sets matrix, estimator, or vector thereof.
  - `i`: Asset index or range to slice.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Column view of the matrix, or the estimator unchanged.

# Related

  - [`asset_sets_matrix`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
"""
function port_opt_view(smtx::MatNum, i, args...; kwargs...)
    return view(smtx, :, i)
end
function port_opt_view(smtx::VecMatNum_ASetMatE, i, args...; kwargs...)
    val = [port_opt_view(smtxi, i; kwargs...) for smtxi in smtx]
    if isabstracttype(eltype(val))
        val = concrete_typed_array(val)
    end
    return val
end

export AssetSetsMatrixEstimator, asset_sets_matrix, asset_sets_features,
       asset_sets_feature_names
