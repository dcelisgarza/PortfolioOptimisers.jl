"""
$(DocStringExtensions.TYPEDEF)

Estimator for constructing asset set membership matrices from asset groupings.

`AssetSetsMatrixEstimator` is a container type for specifying the key or group name used to generate a binary asset-group membership matrix from a [`UniverseSets`](@ref) object. This is used in constraint generation and portfolio construction workflows that require mapping assets to groups or categories.

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
julia> sets = UniverseSets(; xkey = \"nx\",
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

  - [`UniverseSets`](@ref)
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

Read the taxonomy column `sets.dict[key]`, checking that the key exists.

The sibling of [`factor_universe`](@ref) and [`feature_universe`](@ref) for a **group name key**, and written for the same reason: one shared helper whose message names the key and says what to do about it, so every consumer of a caller-supplied taxonomy key fails the same way. A bare `sets.dict[key]` raises a `KeyError` carrying the key alone, which says nothing about which of the two producers asked for it, and offers no help with a typo.

`need` names the consumer. The suggestion comes from [`suggest_declared_key`](@ref), the looser configuration shared by every declaration-key suggestion: the candidates here are `sets.dict` keys the caller authored, not asset names, so the info-leak boundary of ADR 0026 does not apply.

The exception stays a `KeyError`, as [`factor_universe`](@ref) and [`feature_universe`](@ref) do, because a missing key is what it is — only the message improves.

# Arguments

  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `key`: The group name key to read.
  - `need`: Names the consumer in the error message.

# Returns

  - `col`: The value of `sets.dict[key]`, one group per asset.

# Related

  - [`asset_sets_matrix`](@ref)
  - [`asset_sets_feature_names`](@ref)
  - [`factor_universe`](@ref)
  - [`feature_universe`](@ref)
  - [`suggest_declared_key`](@ref)
"""
function taxonomy_column(sets::UniverseSets, key::AbstractString, need::AbstractString)
    @argcheck(haskey(sets.dict, key),
              KeyError("$key (a group name key), required by $need. `sets.dict` holds no such key: correct the spelling$(suggest_declared_key(key, keys(sets.dict))), or add `$key => <one group per asset>` to `sets.dict`."))
    return sets.dict[key]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Construct a binary asset-group membership matrix from asset set groupings.

`asset_sets_matrix` generates a binary (0/1) matrix indicating asset membership in groups or categories, based on the key or group name `smtx` in the provided [`UniverseSets`](@ref). Each row corresponds to a unique group value, and each column to an asset in the universe. This is used in constraint generation and portfolio construction workflows that require mapping assets to groups or categories.

# Arguments

  - `smtx`: The key or group name to extract from the asset sets.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.

# Returns

  - `A::BitMatrix`: A binary matrix of size (number of groups) × (number of assets), where `A[i, j] == 1` if asset `j` belongs to group `i`.

# Details

  - The function checks that `smtx` exists in `sets.dict` and that its length matches the asset universe.
  - Each unique value in `sets.dict[smtx]` defines a group.
  - The output matrix is transposed so that rows correspond to groups and columns to assets.

# Validation

  - `haskey(sets.dict, smtx)`, via [`taxonomy_column`](@ref).
  - Throws an `AssertionError` if the length of `sets.dict[smtx]` does not match the asset universe.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\",
                           dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"],
                                       \"nx_sector\" => [\"Tech\", \"Tech\", \"Finance\"]));

julia> asset_sets_matrix(\"nx_sector\", sets)
2×3 transpose(::BitMatrix) with eltype Bool:
 1  1  0
 0  0  1
```

# Related

  - [`UniverseSets`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
"""
function asset_sets_matrix(smtx::AbstractString, sets::UniverseSets)
    all_sets = taxonomy_column(sets, smtx, "asset_sets_matrix")
    @argcheck(length(sets.dict[sets.xkey]) == length(all_sets),
              AssertionError("lengths of sets.dict[sets.xkey] and `all_sets` must be equal:\nlength(sets.dict[sets.xkey]) => length(sets.dict[$(sets.xkey)]) => $(length(sets.dict[sets.xkey]))\nlength(all_sets) => length(sets.dict[$smtx]) => $(length(all_sets))"))
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

  - [`UniverseSets`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
  - [`asset_sets_matrix`](@ref)
"""
function asset_sets_matrix(smtx::Option{<:MatNum}, args...)
    return smtx
end
"""
    asset_sets_matrix(smtx::AssetSetsMatrixEstimator, sets::UniverseSets)

This method is a wrapper calling:

    asset_sets_matrix(smtx.val, sets)

It is used for type stability and to provide a uniform interface for processing constraint estimators, as well as simplifying the use of multiple estimators simulatneously.

# Related

  - [`asset_sets_matrix`](@ref)
"""
function asset_sets_matrix(smtx::AssetSetsMatrixEstimator, sets::UniverseSets)
    return asset_sets_matrix(smtx.val, sets)
end
"""
    asset_sets_matrix(smtx::VecMatNum_ASetMatE,
                      sets::UniverseSets)

Broadcasts [`asset_sets_matrix`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simulatneously.
"""
function asset_sets_matrix(smtx::VecMatNum_ASetMatE, sets::UniverseSets)
    return [asset_sets_matrix(smtxi, sets) for smtxi in smtx]
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for markers that reinterpret a value in a graded feature program.

A bare `Number` in [`asset_sets_features`](@ref)' pair grammar **sets** a cell, absolutely. A marker says the number means something else, resolved against the cell's **natural value** by [`resolve_feature_value`](@ref). [`Scale`](@ref) is the sole member; the family is open so a second reading can be added without touching the resolver.

# Why the marker, and not nesting depth

The alternative was to let *position* decide — a top-level number scales, a nested one sets. That is invisible on a one-hot taxonomy, where the natural value is always `1.0` and the two readings coincide, and it diverges silently the moment a taxonomy carries numbers, which is exactly what the graded grammar adds. A marker makes the reading something the caller writes down.

# Related

  - [`Scale`](@ref)
  - [`resolve_feature_value`](@ref)
  - [`asset_sets_features`](@ref)
"""
abstract type AbstractFeatureValue <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Multiply a cell's **natural value** by `val`, rather than setting the cell to `val`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Scale(; val::Number = 1.0) -> Scale

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(val)`.

# The natural value, and the two properties it forces

`Scale` scales the **key's own datum**, never the value already accumulated in the cell. At the top level of a program there is no accumulated value — the matrix starts at zero, and scaling zero is useless — so the only coherent referent is the underlying datum:

  - a **numeric** taxonomy key: the asset's own number (`nx_esg` at `0.30`, `0.80`, `0.50`);
  - a **one-hot** key, an asset node or a group member: `1.0` when the row belongs, `0.0` when it does not.

Two consequences follow, and neither is a defect:

 1. The program stays a pure overwrite. Every write is `resolve_feature_value(v, natural)` and never a read-modify-write, so **last-wins** ordering survives the marker.
 2. **Scaling a cross edge gives zero.** `\"C\" => [\"nx_country\" => \"US\" => Scale(0.3)]` scales C's *US membership*, and C is UK — so the cell is `0.0`, not `0.3`. Use a bare number to set a cross edge.

# Examples

```jldoctest
julia> resolve_feature_value(Scale(1.3), 0.5)
0.65

julia> resolve_feature_value(0.7, 0.5)
0.7
```

# Related

  - [`AbstractFeatureValue`](@ref)
  - [`resolve_feature_value`](@ref)
  - [`asset_sets_features`](@ref)
"""
@concrete struct Scale <: AbstractFeatureValue
    """
    `val`: Multiplier applied to the cell's natural value.
    """
    val
    function Scale(val::Number)::Scale
        @argcheck(isfinite(val), DomainError(val, "`val` must be finite"))
        return new{typeof(val)}(val)
    end
end
function Scale(; val::Number = 1.0)::Scale
    return Scale(val)
end
"""
    resolve_feature_value(v::Number, natural::Number) -> Number
    resolve_feature_value(v::Scale, natural::Number) -> Number

Resolve a graded-feature-program value `v` against the cell's `natural` value.

A bare number is absolute and ignores `natural`; a [`Scale`](@ref) multiplies it. Extending [`AbstractFeatureValue`](@ref) means adding one method here and nothing else.

# Related

  - [`AbstractFeatureValue`](@ref)
  - [`Scale`](@ref)
  - [`asset_sets_features`](@ref)
"""
function resolve_feature_value(v::Number, ::Number)
    return v
end
function resolve_feature_value(v::Scale, natural::Number)
    return v.val * natural
end
"""
    const Num_AFeatVal = Union{<:Number, <:AbstractFeatureValue}

Alias for a resolved value in a graded feature program: a bare number, or a marker that reinterprets one.

This is the grammar's `value` production, and it is what distinguishes a *value* from a nested target list when a program entry is parsed.

# Related

  - [`AbstractFeatureValue`](@ref)
  - [`Scale`](@ref)
  - [`asset_sets_features`](@ref)
"""
const Num_AFeatVal = Union{<:Number, <:AbstractFeatureValue}
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
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `strict`: Accepted for a uniform interface with the graded method and **ignored**. `strict` governs name resolution, and on this path every name is a `sets.dict` key whose absence is an unconditional `KeyError` from [`taxonomy_column`](@ref) — there is no soft failure for it to govern.

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

An asset view of a `UniverseSets` slices the groups prefixed by `sets.xkey` and leaves the rest alone, so a key named for a view to reach must carry that prefix — `\"nx_sector\"`, not `\"sector\"`. An unprefixed key does not fail silently: [`asset_sets_matrix`](@ref)'s length check throws on the next call, because the sliced universe no longer matches the unsliced group.

# Validation

  - `length(vals) >= 2` and `allunique(vals)` (see [`assert_feature_keys`](@ref)).
  - Each key exists in `sets.dict` and has the length of the asset universe (enforced by [`asset_sets_matrix`](@ref)).

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\",
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

  - [`UniverseSets`](@ref)
  - [`asset_sets_matrix`](@ref)
  - [`asset_sets_feature_names`](@ref)
  - [`assert_feature_keys`](@ref)
  - [`AssetSetsFeatures`](@ref)
  - [`FeatureDistance`](@ref)
"""
function asset_sets_features(vals::AbstractVector{<:AbstractString}, sets::UniverseSets;
                             strict::Bool = false)::Matrix{Float64}
    assert_feature_keys(vals)
    return Float64.(reduce(hcat, (transpose(asset_sets_matrix(v, sets)) for v in vals)))
end
"""
    feature_program_candidates(sets::UniverseSets, nz) -> Vector{String}

Build the `did_you_mean` pool for a graded feature program: asset names, every `sets.dict` key, every distinct value of every taxonomy key, and every declared feature node.

A graded program's names live in four namespaces at once, so a pool narrower than their union would answer "did you mean" with silence on the commonest typo.

# Related

  - [`asset_sets_features`](@ref)
  - [`did_you_mean`](@ref)
"""
function feature_program_candidates(sets::UniverseSets, nz)::Vector{String}
    dict = sets.dict
    xkey = sets.xkey
    pool = String[]
    append!(pool, string.(dict[xkey]))
    append!(pool, string.(keys(dict)))
    for (k, v) in dict
        if k != xkey && startswith(k, xkey)
            append!(pool, string.(unique(v)))
        end
    end
    append!(pool, string.(nz))
    return unique!(pool)
end
"""
    is_feature_taxonomy_key(k, sets::UniverseSets) -> Bool

Whether a name in a graded feature program is a **taxonomy key**: a `sets.xkey`-prefixed key of `sets.dict`.

[`UniverseSets`](@ref) guarantees every `sets.xkey`-prefixed dict key is asset-parallel, so the prefix rule alone decides the question — no new convention is needed, and row-selector precedence reduces to prefix, then asset, then group, exactly as [`estimator_to_val`](@ref) resolves.

# Related

  - [`asset_sets_features`](@ref)
  - [`is_feature_factor_key`](@ref)
  - [`UniverseSets`](@ref)
"""
function is_feature_taxonomy_key(k, sets::UniverseSets)::Bool
    return isa(k, AbstractString) && startswith(k, sets.xkey) && haskey(sets.dict, k)
end
"""
    is_feature_factor_key(k, sets::UniverseSets) -> Bool

Whether a name in a graded feature program declares the **factor axis**, by carrying the `sets.fkey` or `sets.ufkey` prefix.

Such a name is factor-length, so it can index neither the rows (assets) nor the declared nodes. It is refused **by name** rather than falling through to the plain-group branch, where it would fail later on a length mismatch that names neither the axis nor the cause. [`feature_rows`](@ref) puts the test between the asset branch and the group branch, and [`feature_factor_key_msg`](@ref) writes the diagnostic.

# Related

  - [`asset_sets_features`](@ref)
  - [`is_feature_taxonomy_key`](@ref)
  - [`feature_factor_key_msg`](@ref)
  - [`feature_rows`](@ref)
  - [`UniverseSets`](@ref)
"""
function is_feature_factor_key(k, sets::UniverseSets)::Bool
    return isa(k, AbstractString) && (startswith(k, sets.fkey) || startswith(k, sets.ufkey))
end
"""
    feature_grammar_msg(term) -> String

Build the error text for a malformed **term** of a graded feature program — an entry or a target — and print the grammar itself.

A malformed term is a syntax error, so the fastest fix is seeing the production it missed. Unlike the name diagnostics, this one never routes through [`strict_diagnostic`](@ref): its callers throw an `ArgumentError` whatever `strict` says, because there is no reading of the term to fall back to.

# Related

  - [`asset_sets_features`](@ref)
  - [`feature_entry!`](@ref)
  - [`feature_target!`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function feature_grammar_msg(term)
    return "`$(term)` is not a well-formed graded feature program term. The grammar is\n" *
           "  entry  := rowsel => targets              # row scope, then explicit columns\n" *
           "          | taxkey [=> group] => value     # diagonal: those rows, their own membership\n" *
           "  rowsel := asset | group | taxkey [=> group]\n" *
           "  target := taxkey [=> group] => value | asset => value | group => value\n" *
           "  value  := Number | <:AbstractFeatureValue\n" *
           "`targets` is one target or a vector of them, and every target names its column in full: there is no ambient scope."
end
"""
    feature_factor_key_msg(k, sets::UniverseSets) -> String

Build the warning/error text for a name `k` that a graded feature program wrote in a row-selector or column-target position, but that declares the **factor axis** (see [`is_feature_factor_key`](@ref)).

The message names the two prefixes and the reason, and names sizes rather than universes — the same info-leak-safe discipline as [`unknown_variable_msg`](@ref). It carries **no** [`did_you_mean`](@ref) suggestion: the name resolved perfectly well, on the wrong axis, so there is no typo to propose.

# Related

  - [`asset_sets_features`](@ref)
  - [`is_feature_factor_key`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`unknown_variable_msg`](@ref)
"""
function feature_factor_key_msg(k, sets::UniverseSets)
    return "`$(k)` names the factor axis (prefix `$(sets.fkey)`/`$(sets.ufkey)`), which is neither a row selector nor a column target: a graded feature program indexes assets by row and declared feature nodes by column, and a factor-length list is neither; term dropped"
end
"""
    feature_missing_group_value_msg(key, group, col) -> String

Build the warning/error text for a group value `group` of the taxonomy key `key` that matches no asset, where `col` is that key's column of `sets.dict`.

The message names the *count* of distinct values under the key, never the values themselves — the info-leak-safe discipline of [`unknown_variable_msg`](@ref) — and appends a [`did_you_mean`](@ref) suggestion drawn from those values. It drops `group` from its own candidate pool first, because the pool of a graded program is deliberately wide enough to contain a name that is nonetheless invalid in the position it was written.

# Related

  - [`asset_sets_features`](@ref)
  - [`feature_diagonal!`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`did_you_mean`](@ref)
"""
function feature_missing_group_value_msg(key, group, col)
    vals = string.(unique(col))
    return "group value `$(group)` of taxonomy key `$(key)` matches no asset ($(length(vals)) distinct values under that key); term dropped" *
           did_you_mean(string(group), filter(!=(string(group)), vals))
end
"""
    feature_unknown_name_msg(name, nx, key, pool; axis::AbstractString = "asset") -> String

Build the warning/error text for a `name` of a graded feature program that resolves in no namespace of the axis it was written on.

The function wraps [`unknown_variable_msg`](@ref), so it inherits that message's shape and names the size of `nx` rather than its members. `axis` is `"asset"` for a row selector and `"feature"` for a column target, and `key` is `sets.xkey` or `sets.zkey` to match. `pool` is the wide candidate pool of [`feature_program_candidates`](@ref); the function drops `name` from it before suggesting, because that pool is deliberately wide enough — taxonomy *values* and declared nodes included — to contain a name that is nonetheless invalid in the position it was written.

# Related

  - [`asset_sets_features`](@ref)
  - [`feature_program_candidates`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`unknown_variable_msg`](@ref)
"""
function feature_unknown_name_msg(name, nx, key, pool; axis::AbstractString = "asset")
    return unknown_variable_msg(name, nx, key; candidates = filter(!=(string(name)), pool),
                                axis = axis)
end
"""
    feature_numeric_column(col) -> Bool

Whether a taxonomy key's data are numbers, which is what decides the **diagonal** form's column and natural value.

A *categorical* key writes each asset into the node its own group value names, at a natural value of `1.0`. A *numeric* key has one node — the key with `sets.xkey * "_"` stripped — and the asset's own number is the natural value. The two are the same production in the grammar and differ only here.

# Related

  - [`asset_sets_features`](@ref)
  - [`Scale`](@ref)
"""
function feature_numeric_column(col)::Bool
    return !isempty(col) && all(x -> isa(x, Number), col)
end
"""
    feature_write!(Z::Matrix{Float64}, rows, node, natural, v, sets::UniverseSets, nz, zidx,
                   pool, strict::Bool) -> Nothing

Write one column of a graded feature program: resolve `node` against the declared axis, then set `Z[i, col] = resolve_feature_value(v, natural(i))` for every `i` in `rows`.

Every write in the program funnels through here, which is what makes **last-wins** a property of the traversal rather than a rule each production has to honour: the assignment is a pure overwrite, never a read-modify-write, so a later entry simply replaces an earlier one.

The column is resolved **before** `natural` is ever called, so an unknown node costs one diagnostic and no work — and `natural` never has to be defined for a node that does not exist.

# Related

  - [`asset_sets_features`](@ref)
  - [`resolve_feature_value`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function feature_write!(Z::Matrix{Float64}, rows, node, natural, v, sets::UniverseSets, nz,
                        zidx, pool, strict::Bool)::Nothing
    col = get(zidx, string(node), nothing)
    if isnothing(col)
        return strict_diagnostic(feature_unknown_name_msg(node, nz, sets.zkey, pool;
                                                          axis = "feature"), strict)
    end
    for i in rows
        Z[i, col] = resolve_feature_value(v, natural(i))
    end
    return nothing
end
"""
    feature_diagonal!(Z::Matrix{Float64}, key, group, v, sets::UniverseSets, nx, nz, zidx,
                      pool, strict::Bool) -> Nothing

Apply the grammar's **diagonal** production, `taxkey [=> group] => value`: write each selected asset's own membership, rather than a column named from outside.

This is the production that makes a bare number on the right of a taxonomy key unambiguous, and it is why the uniform "targets are always fully qualified" rule costs nothing — `\"nx_country\" => \"UK\" => 0.5` says what the two-level nested form used to say, one bracket shorter.

# Related

  - [`asset_sets_features`](@ref)
  - [`feature_write!`](@ref)
  - [`feature_numeric_column`](@ref)
"""
function feature_diagonal!(Z::Matrix{Float64}, key, group, v, sets::UniverseSets, nx, nz,
                           zidx, pool, strict::Bool)::Nothing
    col = sets.dict[key]
    if feature_numeric_column(col)
        rows = if isnothing(group)
            eachindex(nx)
        else
            findall(x -> isequal(x, group), col)
        end
        if isempty(rows)
            return strict_diagnostic(feature_missing_group_value_msg(key, group, col),
                                     strict)
        end
        node = chopprefix(key, sets.xkey * "_")
        return feature_write!(Z, rows, node, i -> float(col[i]), v, sets, nz, zidx, pool,
                              strict)
    end
    if !isnothing(group)
        rows = findall(x -> isequal(x, group), col)
        if isempty(rows)
            return strict_diagnostic(feature_missing_group_value_msg(key, group, col),
                                     strict)
        end
        return feature_write!(Z, rows, group, i -> 1.0, v, sets, nz, zidx, pool, strict)
    end
    for g in unique(col)
        rows = findall(x -> isequal(x, g), col)
        feature_write!(Z, rows, g, i -> 1.0, v, sets, nz, zidx, pool, strict)
    end
    return nothing
end
"""
    feature_rows(sel, sets::UniverseSets, nx, pool, strict::Bool) -> Option{Vector{Int}}

Resolve a **non-taxonomy** row selector — an asset or an asset group — to row indices, or `nothing` when the name does not resolve.

Taxonomy keys are handled by the caller, because the `sets.xkey` prefix rule settles them before any lookup. What is left is exactly [`estimator_to_val`](@ref)'s precedence, asset first and then group, resolved by the shared [`resolve_axis_name`](@ref), with the factor axis refused between them so a factor-length list is diagnosed by name rather than by an eventual length mismatch. The factor test therefore runs before the resolution, not after: a factor key that is *also* a `sets.dict` key would otherwise expand to factor names and be reported as a group of missing assets.

# Related

  - [`asset_sets_features`](@ref)
  - [`estimator_to_val`](@ref)
  - [`resolve_axis_name`](@ref)
  - [`axis_name_indices`](@ref)
  - [`missing_group_assets_msg`](@ref)
"""
function feature_rows(sel, sets::UniverseSets, nx, pool, strict::Bool)
    if !(sel in nx) && is_feature_factor_key(sel, sets)
        strict_diagnostic(feature_factor_key_msg(sel, sets), strict)
        return nothing
    end
    members = resolve_axis_name(sel, nx, sets.dict)
    if isnothing(members)
        strict_diagnostic(feature_unknown_name_msg(sel, nx, sets.xkey, pool), strict)
        return nothing
    end
    return axis_name_indices(members, nx,
                             m -> strict_diagnostic(missing_group_assets_msg(sel, m, nx,
                                                                             sets.xkey),
                                                    strict))
end
"""
    feature_target!(Z::Matrix{Float64}, rows, target::Union{<:Pair, <:AbstractVector{<:Pair}}, sets::UniverseSets, nx, nz, zidx, pool,
                    strict::Bool) -> Nothing

Apply the grammar's `target` production inside an already-resolved row scope, singly or over a vector.

Every target names its column in full, so this reads left to right with no ambient state: a taxonomy key with a group value names that value's node, a numeric taxonomy key names its own node, an asset names its own node, and a group expands to one node per member. The natural value is the row's membership of the node the target named — which is why scaling a cross edge gives zero.

# Related

  - [`asset_sets_features`](@ref)
  - [`feature_write!`](@ref)
  - [`Scale`](@ref)
"""
function feature_target!(Z::Matrix{Float64}, rows, target, sets::UniverseSets, nx, nz, zidx,
                         pool, strict::Bool)::Nothing
    throw(ArgumentError(feature_grammar_msg(target)))
    return nothing
end
function feature_target!(Z::Matrix{Float64}, rows, target::Pair, sets::UniverseSets, nx, nz,
                         zidx, pool, strict::Bool)::Nothing
    k, rest = target
    if is_feature_taxonomy_key(k, sets)
        col = sets.dict[k]
        if isa(rest, Pair)
            g, v = rest
            if !isa(v, Num_AFeatVal)
                throw(ArgumentError(feature_grammar_msg(target)))
            end
            return feature_write!(Z, rows, g, i -> ifelse(isequal(col[i], g), 1.0, 0.0), v,
                                  sets, nz, zidx, pool, strict)
        end
        if !isa(rest, Num_AFeatVal)
            throw(ArgumentError(feature_grammar_msg(target)))
        end
        @argcheck(feature_numeric_column(col),
                  ArgumentError("`$(k)` is a categorical taxonomy key, so a target naming it must also name a group value: `\"$(k)\" => <group> => <value>`. The bare form `key => value` names the key's *own* node, which only exists for a numeric key. Got\ntarget => $(target)"))
        node = chopprefix(k, sets.xkey * "_")
        return feature_write!(Z, rows, node, i -> float(col[i]), rest, sets, nz, zidx, pool,
                              strict)
    end
    # Before the grammar check, because a factor key wearing a taxonomy key's two-level
    # shape would otherwise be reported as a syntax error and send the caller looking for a
    # missing bracket instead of a wrong axis.
    if is_feature_factor_key(k, sets)
        return strict_diagnostic(feature_factor_key_msg(k, sets), strict)
    end
    if !isa(rest, Num_AFeatVal)
        throw(ArgumentError(feature_grammar_msg(target)))
    end
    # An asset names its own node, a group names one node per member, and `resolve_axis_name`
    # answers both: an asset resolves to itself, so the two branches are one loop.
    members = resolve_axis_name(k, nx, sets.dict)
    if isnothing(members)
        return strict_diagnostic(feature_unknown_name_msg(k, nx, sets.xkey, pool), strict)
    end
    for m in members
        feature_write!(Z, rows, m, i -> ifelse(isequal(nx[i], m), 1.0, 0.0), rest, sets, nz,
                       zidx, pool, strict)
    end
    return nothing
end
function feature_target!(Z::Matrix{Float64}, rows, targets::AbstractVector,
                         sets::UniverseSets, nx, nz, zidx, pool, strict::Bool)::Nothing
    for t in targets
        feature_target!(Z, rows, t, sets, nx, nz, zidx, pool, strict)
    end
    return nothing
end
"""
    feature_entry!(Z::Matrix{Float64}, entry::Pair, sets::UniverseSets, nx, nz, zidx, pool,
                   strict::Bool) -> Nothing

Apply one entry of a graded feature program.

# How the two productions are told apart

`entry := rowsel => targets | taxkey [=> group] => value`, and Julia's `=>` is right-associative, so both arrive as a `Pair` whose right side may nest. Three tests separate them, in this order:

 1. The left side is a taxonomy key and the right side bottoms out in a **value** — `\"nx_sector\" => 2.0`, `\"nx_country\" => \"UK\" => 0.5`. That is the diagonal, by decision: a bare value on the right of a taxonomy key always means "these rows, their own membership".
 2. The left side is a taxonomy key, the right side is `g => tail` with `tail` a target list, and `g` is *itself* a taxonomy key. Then `g` starts a target, so the left side is a bare row selector over the whole universe. This is the one genuine ambiguity in the grammar and it is resolved by the same prefix rule everything else uses.
 3. Otherwise the left side is a row selector — restricted by `g` when the left side is a taxonomy key, an asset or a group when it is not.

# Related

  - [`asset_sets_features`](@ref)
  - [`feature_diagonal!`](@ref)
  - [`feature_target!`](@ref)
  - [`feature_rows`](@ref)
"""
function feature_entry!(Z::Matrix{Float64}, entry::Pair, sets::UniverseSets, nx, nz, zidx,
                        pool, strict::Bool)::Nothing
    lhs, rhs = entry
    if is_feature_taxonomy_key(lhs, sets)
        if isa(rhs, Num_AFeatVal)
            return feature_diagonal!(Z, lhs, nothing, rhs, sets, nx, nz, zidx, pool, strict)
        end
        if isa(rhs, Pair)
            g, tail = rhs
            if isa(tail, Num_AFeatVal)
                return feature_diagonal!(Z, lhs, g, tail, sets, nx, nz, zidx, pool, strict)
            end
            if is_feature_taxonomy_key(g, sets)
                return feature_target!(Z, eachindex(nx), rhs, sets, nx, nz, zidx, pool,
                                       strict)
            end
            col = sets.dict[lhs]
            rows = findall(x -> isequal(x, g), col)
            if isempty(rows)
                return strict_diagnostic(feature_missing_group_value_msg(lhs, g, col),
                                         strict)
            end
            return feature_target!(Z, rows, tail, sets, nx, nz, zidx, pool, strict)
        end
        return feature_target!(Z, eachindex(nx), rhs, sets, nx, nz, zidx, pool, strict)
    end
    rows = feature_rows(lhs, sets, nx, pool, strict)
    if isnothing(rows) || isempty(rows)
        return nothing
    end
    return feature_target!(Z, rows, rhs, sets, nx, nz, zidx, pool, strict)
end
"""
    asset_sets_features(vals::AbstractVector{<:Pair}, sets::UniverseSets;
                        strict::Bool = false) -> Matrix{Float64}

Resolve an ordered **edge-authoring program** into an `assets × features` matrix over the declared feature axis `sets.dict[sets.zkey]`.

This is the *graded* contract. The group-name-key method above is the degenerate case of it — a partition stack with every written cell at `1.0` — and the two are separated by dispatch on `vals`' element type, so today's callers, today's matrix and today's `\"<key>=<group>\"` names are untouched.

# The grammar

```
entry  := rowsel => targets                    # row scope, then explicit columns
        | taxkey [=> group] => value           # diagonal: those rows, their own membership
rowsel := asset | group | taxkey [=> group]
target := taxkey [=> group] => value | asset => value | group => value
value  := Number                               # sets, absolutely
        | <:AbstractFeatureValue               # Scale(x): x × the key's natural value
```

`targets` is one target or a vector of them. Entries are applied **in order** and each write is a pure overwrite, so **last wins** — repeating a key is the point, not a mistake, which is why `allunique` does not carry into this path.

Every target names its column **in full**. There is no ambient scope and no fallback chain: `\"UK\"` inside a `nx_country` entry would otherwise be resolved as a country by proximity rather than by what the caller wrote, and `UK` is also a real ticker.

# Two things the declared axis buys

 1. **Column order.** A `Dict` has none, so without a declared list the feature axis would be whatever order the taxonomy happened to iterate in.
 2. **Fold invariance.** `size(Z, 2)` does not change under an asset view, because [`port_opt_view(::UniverseSets, i, args...)`](@ref) passes `zkey` through — the axis is *authored*, not summarised. This is the exact opposite of the group-name-key path, and both are documented because both are true. The cost is accepted: an asset node whose asset a view dropped survives as an **all-zero column**, benign for every blessed metric except `Distances.CorrDist`, which centres each row.

# Names are bare, and what that forbids

Nodes are named plainly — `\"Tech\"`, `\"US\"`, `\"esg\"` (a numeric key with `sets.xkey * \"_\"` stripped), `\"A\"` for an asset node — because the caller authored the axis and qualifying it would make them write the prefix twice.

The accepted cost: **a nested taxonomy with a repeated value is inexpressible in graded mode.** With `nx_industry` and `nx_subindustry` both containing `IntegratedOil`, both land on the one bare node and the later entry overwrites the earlier — harmless under one-hot, where both wrote `1.0`, silently lossy under grading. The group-name-key path qualifies its names and stays the tool for that case.

# What is refused, and what is not

`strict` governs **names only** — an unknown node, asset or group warns with a [`did_you_mean`](@ref) suggestion, and throws under `strict`. Nothing structural is refused:

  - **An all-zero row is legal**, and is the one genuine silent-wrongness case grading opens: an asset no entry touches has a zero row, and [`FeatureDistance`](@ref)'s zero-norm convention declares zero rows mutually *identical*, so forgotten assets cluster together at distance `0`.
  - **A one-column matrix is legal**: [`assert_feature_keys`](@ref)' two-key floor is a property of stacking partitions and does not carry here.

Only non-emptiness is unconditional. A **malformed entry** — one that matches no production — throws regardless of `strict`, because there is no reading of it to fall back to.

# Arguments

  - `vals`: The ordered program.
  - `sets`: A [`UniverseSets`](@ref) whose `dict` declares the feature axis under `sets.zkey`.
  - `strict`: Whether an unresolvable name throws instead of warning.

# Returns

  - `Z::Matrix{Float64}`: An `assets × length(sets.dict[sets.zkey])` matrix, zero-initialised.

# Validation

  - `!isempty(vals)`.
  - `haskey(sets.dict, sets.zkey)` (see [`feature_universe`](@ref)).

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\", zkey = \"nz\",
                           dict = Dict{String, Any}(\"nx\" => [\"A\", \"B\", \"C\"],
                                                    \"nz\" => [\"Tech\", \"Finance\", \"esg\"],
                                                    \"nx_sector\" => [\"Tech\", \"Tech\", \"Finance\"],
                                                    \"nx_esg\" => [0.30, 0.80, 0.50]));

julia> asset_sets_features([\"nx_sector\" => 2.0, \"nx_esg\" => Scale(1.3),
                            \"B\" => [\"nx_sector\" => \"Finance\" => 0.2]], sets)
3×3 Matrix{Float64}:
 2.0  0.0  0.39
 2.0  0.2  1.04
 0.0  2.0  0.65
```

# Related

  - [`UniverseSets`](@ref)
  - [`Scale`](@ref)
  - [`resolve_feature_value`](@ref)
  - [`asset_sets_feature_names`](@ref)
  - [`feature_universe`](@ref)
  - [`AssetSetsFeatures`](@ref)
  - [`FeatureDistance`](@ref)
"""
function asset_sets_features(vals::AbstractVector{<:Pair}, sets::UniverseSets;
                             strict::Bool = false)::Matrix{Float64}
    @argcheck(!isempty(vals),
              IsEmptyError("`vals` cannot be empty: a program with no entries can only produce the all-zero matrix, which declares every asset identical"))
    nx = sets.dict[sets.xkey]
    nz = feature_universe(sets, "a graded `asset_sets_features` program")
    Z = zeros(Float64, length(nx), length(nz))
    zidx = Dict(string(n) => k for (k, n) in pairs(nz))
    pool = feature_program_candidates(sets, nz)
    for entry in vals
        feature_entry!(Z, entry, sets, nx, nz, zidx, pool, strict)
    end
    return Z
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Name the columns [`asset_sets_features`](@ref) produces, in their own order.

[`ReturnsResult`](@ref) requires `nz` whenever `Z` is set, so the user-supplied carrier needs these names — and reproducing the column order by hand is exactly the kind of restatement that drifts. The pair is built from one traversal of `vals`, so the two can only agree.

# Names are qualified by their key

Each name is `\"<key>=<group>\"` rather than the bare group value, because a **nested** taxonomy reuses its values across levels: an integrated-oil producer is in industry `IntegratedOil` and sub-industry `IntegratedOil`, so bare values would collide and [`ReturnsResult`](@ref)'s uniqueness check would reject them.

# Arguments

  - `vals`: The same group name keys, in the same order, passed to [`asset_sets_features`](@ref).
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.

# Returns

  - `nz::Vector{String}`: One name per feature column, `length(nz) == size(Z, 2)`.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\",
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
  - [`UniverseSets`](@ref)
  - [`ReturnsResult`](@ref)
"""
function asset_sets_feature_names(vals::AbstractVector{<:AbstractString},
                                  sets::UniverseSets)::Vector{String}
    assert_feature_keys(vals)
    return [string(v, "=", g) for v in vals
            for g in unique(taxonomy_column(sets, v, "asset_sets_feature_names"))]
end
"""
    asset_sets_feature_names(vals::AbstractVector{<:Pair}, sets::UniverseSets) -> Vector{String}

Name the columns a graded [`asset_sets_features`](@ref) program produces: the **declared** axis itself, `sets.dict[sets.zkey]`.

The pairing with the matrix is trivial here, and deliberately so — the caller authored the axis, so the recipe below keeps the shape it has on the group-name-key path while the program itself carries no naming rule at all.

# Names are bare, not qualified

The exact opposite of the group-name-key method above, and for a stated reason. That path *derives* its axis by stacking partitions, so it must qualify (`\"nx_industry=IntegratedOil\"`) or a nested taxonomy would collide with itself. This path's axis is **authored**, so the names are whatever the caller wrote — `\"Tech\"`, `\"US\"`, `\"esg\"`, `\"A\"` — and qualifying them would mean writing the prefix twice, once in `dict[zkey]` and again in every target.

The cost of bareness is a graded-mode limitation, documented on [`asset_sets_features`](@ref): a nested taxonomy with a repeated value cannot be expressed, because both levels land on the one node.

# Arguments

  - `vals`: The program. Read only for dispatch — the axis does not depend on it.
  - `sets`: A [`UniverseSets`](@ref) whose `dict` declares the feature axis under `sets.zkey`.

# Returns

  - `nz::Vector{String}`: The declared feature axis, `length(nz) == size(Z, 2)`.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\", zkey = \"nz\",
                           dict = Dict{String, Any}(\"nx\" => [\"A\", \"B\", \"C\"],
                                                    \"nz\" => [\"Tech\", \"Finance\", \"esg\"],
                                                    \"nx_sector\" => [\"Tech\", \"Tech\", \"Finance\"]));

julia> asset_sets_feature_names([\"nx_sector\" => 2.0], sets)
3-element Vector{String}:
 "Tech"
 "Finance"
 "esg"
```

# Related

  - [`asset_sets_features`](@ref)
  - [`UniverseSets`](@ref)
  - [`feature_universe`](@ref)
  - [`ReturnsResult`](@ref)
"""
function asset_sets_feature_names(vals::AbstractVector{<:Pair},
                                  sets::UniverseSets)::Vector{String}
    return string.(feature_universe(sets, "a graded `asset_sets_feature_names` call"))
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
    return concrete_typed_array_if_abstract([port_opt_view(smtxi, i, args...; kwargs...)
                                             for smtxi in smtx])
end
# A vector of estimators alone matches both the signature above and the generic vector method
# in `02_Tools.jl`, and neither is more specific: `MatNum` is outside the generic's element
# union, `Nothing` is outside this one's. This method is that intersection, so the two never
# tie. Its body is the one above, because a membership matrix and its estimator must produce
# the same element type.
function port_opt_view(smtx::AbstractVector{<:AssetSetsMatrixEstimator}, i, args...;
                       kwargs...)
    return concrete_typed_array_if_abstract([port_opt_view(smtxi, i, args...; kwargs...)
                                             for smtxi in smtx])
end

export AssetSetsMatrixEstimator, asset_sets_matrix, asset_sets_features,
       asset_sets_feature_names, Scale, resolve_feature_value
