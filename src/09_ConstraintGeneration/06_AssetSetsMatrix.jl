"""
$(DocStringExtensions.TYPEDEF)

Names the group name key a binary asset-group membership matrix is built from.

The key is read out of a [`UniverseSets`](@ref) by [`asset_sets_matrix`](@ref), which returns one row per distinct group value and one column per asset. A row of that matrix is the set indicator a group weight constraint sums the weights over.

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

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equations 9.2-9.4.
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

The sibling of [`factor_universe`](@ref) for a **group name key**, and written for the same reason: one shared helper whose message names the key and says what to do about it, so every consumer of a caller-supplied taxonomy key fails the same way. A bare `sets.dict[key]` raises a `KeyError` carrying the key alone, which says nothing about which of the two producers asked for it, and offers no help with a typo.

`need` names the consumer. The suggestion comes from [`suggest_declared_key`](@ref), the looser configuration shared by every declaration-key suggestion: the candidates here are `sets.dict` keys the caller authored, not asset names, so the info-leak boundary of ADR 0026 does not apply.

# Arguments

  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `key`: The group name key to read.
  - `need`: Names the consumer in the error message.

# Validation

  - `haskey(sets.dict, key)`. Raises a `KeyError` naming `key`, naming `need` as the consumer that asked for it, and carrying a [`suggest_declared_key`](@ref) suggestion. The exception stays a `KeyError`, as [`factor_universe`](@ref) does, because a missing key is what it is — only the message improves.

# Returns

  - `col`: The value of `sets.dict[key]`, one group per asset.

# Related

  - [`asset_sets_matrix`](@ref)
  - [`panel_input`](@ref)
  - [`factor_universe`](@ref)
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

# Algorithm

 1. Read the taxonomy column `all_sets` from `sets.dict[smtx]`, through [`taxonomy_column`](@ref).
 2. Check that `all_sets` has the length of the asset universe.
 3. Take `unique_sets = unique(all_sets)`, the distinct group values in order of first appearance. Each one defines a group, and the order of `unique_sets` is the order of the groups in the result.
 4. For each group `val` of `unique_sets`, write the indicator `all_sets .== val` into column `i` of a `BitMatrix` `A`, giving an `assets × groups` matrix.
 5. Return `transpose(A)`, which turns the groups into the rows and the assets into the columns.

# Arguments

  - `smtx`: The key or group name to extract from the asset sets.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.

# Validation

  - `haskey(sets.dict, smtx)`, via [`taxonomy_column`](@ref).
  - Throws an `AssertionError` if the length of `sets.dict[smtx]` does not match the asset universe.

# Returns

  - `A`: The `transpose` of a `BitMatrix`, of size (number of groups) × (number of assets), where `A[i, j] == 1` if asset `j` belongs to group `i`. The orientation is **`groups × assets`**. The row order is `unique(sets.dict[smtx])`, the distinct group values in order of first appearance.

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

# Arguments

  - `smtx`: An [`AssetSetsMatrixEstimator`](@ref) naming the group name key to read.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.

# Validation

  - Delegated to the group name key method, which validates `smtx.val` (see [`asset_sets_matrix`](@ref)).

# Returns

  - `A`: The `transpose` of a `BitMatrix`, of size (number of groups) × (number of assets), exactly as the group name key method returns it.

# Related

  - [`asset_sets_matrix`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
  - [`UniverseSets`](@ref)
"""
function asset_sets_matrix(smtx::AssetSetsMatrixEstimator, sets::UniverseSets)
    return asset_sets_matrix(smtx.val, sets)
end
"""
    asset_sets_matrix(smtx::VecMatNum_ASetMatE,
                      sets::UniverseSets)

Broadcasts [`asset_sets_matrix`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simulatneously.

# Arguments

  - `smtx`: A vector whose entries are each an [`AssetSetsMatrixEstimator`](@ref) or an already-built numeric matrix.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.

# Validation

  - Delegated per entry to the method that entry dispatches to.

# Returns

  - `A`: One result per entry of `smtx`, in the order of `smtx`. Each entry keeps whatever its own method returns, so a `groups × assets` transpose and a passed-through matrix can sit side by side.

# Related

  - [`asset_sets_matrix`](@ref)
  - [`VecMatNum_ASetMatE`](@ref)
  - [`UniverseSets`](@ref)
"""
function asset_sets_matrix(smtx::VecMatNum_ASetMatE, sets::UniverseSets)
    return [asset_sets_matrix(smtxi, sets) for smtxi in smtx]
end
"""
    panel_input_name(sets::UniverseSets, key::AbstractString) -> String

Name the Panel Field a [`UniverseSets`](@ref) key becomes: the key with the asset prefix stripped.

Every Panel Field of an [`AssetPanel`](@ref) is asset-parallel by construction, so the `xkey` prefix that marks a key as asset-length says nothing there and comes off, with its separating underscore: `"nx_sector"` becomes `"sector"`. A key that carries no prefix keeps its name.

# Algorithm

 1. Return the key unchanged when it does not start with `sets.xkey` followed by an underscore.
 2. Otherwise return the key with that prefix and the underscore removed.

# Arguments

  - `sets`: The universe sets, read for `xkey`.
  - `key`: The key to name.

# Returns

  - `name::String`: The Panel Field name.

# Related

  - [`panel_input`](@ref)
  - [`UniverseSets`](@ref)
  - [`AssetPanel`](@ref)
"""
function panel_input_name(sets::UniverseSets, key::AbstractString)
    pre = string(sets.xkey, "_")
    return if startswith(key, pre)
        String(chop(key; head = length(pre), tail = 0))
    else
        String(key)
    end
end
"""
    panel_input(sets::UniverseSets, key::AbstractString; name = nothing, levels = nothing,
                alg = NoPanelFill())
    panel_input(sets::UniverseSets, key::Pair; name = nothing, levels = nothing,
                alg = NoPanelFill())
    panel_input(sets::UniverseSets, keys::AbstractVector)

Turn a [`UniverseSets`](@ref) key into the raw Panel Field input [`asset_panel`](@ref) reads.

A key is one vector over the asset axis, which is the static raw form of one Panel Field. This is the **one bridge** between a taxonomy and the panel: a string-valued key becomes a [`CategoricalPanelInput`](@ref) and a number-valued key a [`NumericPanelInput`](@ref), by dispatch on the key's element type. A key of mixed element type is refused, because neither reading is right and guessing one is worse than saying so.

An entry `key => InputType` **forces** the type, in both the scalar and the vector form: `"nx_rating" => CategoricalPanelInput` reads a numeric rating as a label rather than as a number. There is no `kind` keyword; the pair is the whole mechanism.

The field is named by [`panel_input_name`](@ref). A nested taxonomy is several keys, so several fields, and the vector form maps the same rule over them.

The result is **static**: a taxonomy has no observation axis. It joins a time-varying panel by the lazy lift [`asset_panel`](@ref) applies, so a taxonomy sits beside a fundamentals table in one panel.

# The vector form takes no per-field keyword

`name`, `levels` and `alg` describe **one** field, so the vector form does not take them: a key that needs one of them goes through the scalar form, and the two results go into the same `asset_panel` call. This is why the vector form is a convenience rather than the primary route.

# Algorithm

The scalar form takes four steps:

 1. Read the key's values off `sets.dict`, which raises a `KeyError` when the key is absent.
 2. Choose the input type: the forced one from a `Pair`, and otherwise [`CategoricalPanelInput`](@ref) for a string element type and [`NumericPanelInput`](@ref) for a number element type. Refuse anything else.
 3. Name the field with [`panel_input_name`](@ref), unless `name` overrides it.
 4. Build the input.

The vector form takes one step:

 1. Map the scalar form over the entries.

# Arguments

  - `sets`: The universe sets holding the key.
  - `key`: The key, or `key => InputType` to force the type.
  - `keys`: A vector of either form.
  - `name`: Panel Field name, or `nothing` to derive it from the key.
  - `levels`: Category levels, or `nothing` to let the builder sort them. Categorical only.
  - `alg`: Fill policy for the raw values.

# Validation

  - `haskey(sets.dict, key)`. Raises a `KeyError`.
  - The key's element type is a string or a number throughout. Raises an `ArgumentError`.

# Returns

  - From the scalar form, one `AbstractPanelFieldInput`. From the vector form, one per entry.

# Examples

```jldoctest
julia> sets = UniverseSets(;
                           dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"],
                                       \"nx_sector\" => [\"Fin\", \"Tech\", \"Fin\"]));

julia> inp = panel_input(sets, \"nx_sector\");

julia> inp.name
\"sector\"

julia> pnl = asset_panel([inp]);

julia> panel_feature_matrix(pnl)[1]
2-element Vector{String}:
 \"sector=Fin\"
 \"sector=Tech\"
```

# Related

  - [`asset_panel`](@ref)
  - [`panel_input_name`](@ref)
  - [`CategoricalPanelInput`](@ref)
  - [`NumericPanelInput`](@ref)
  - [`UniverseSets`](@ref)
  - [`AssetPanel`](@ref)
"""
function panel_input(sets::UniverseSets, key::AbstractString;
                     name::Option{<:AbstractString} = nothing,
                     levels::Option{<:VecStr} = nothing,
                     alg::AbstractPanelFillAlgorithm = NoPanelFill())
    vals = taxonomy_column(sets, key, "a `panel_input` bridge")
    return panel_input_build(panel_input_kind(vals, key), sets, key, vals, name, levels,
                             alg)
end
function panel_input(sets::UniverseSets, key::Pair;
                     name::Option{<:AbstractString} = nothing,
                     levels::Option{<:VecStr} = nothing,
                     alg::AbstractPanelFillAlgorithm = NoPanelFill())
    k = first(key)
    vals = taxonomy_column(sets, k, "a `panel_input` bridge")
    return panel_input_build(last(key), sets, k, vals, name, levels, alg)
end
function panel_input(sets::UniverseSets, keys::AbstractVector)
    @argcheck(!isempty(keys),
              IsEmptyError("`panel_input` needs at least one key: an empty vector builds no Panel Field"))
    return [panel_input(sets, k) for k in keys]
end
"""
    panel_input_kind(vals::AbstractVector, key::AbstractString) -> Type

Choose the raw Panel Field input type a [`UniverseSets`](@ref) key's values ask for.

The element type is the whole rule: a string-valued key is a classification and a number-valued key is a quantity. A mixed element type is refused, because a taxonomy that carries both in one key describes two things.

# Algorithm

 1. Return [`CategoricalPanelInput`](@ref) when every value is a string.
 2. Return [`NumericPanelInput`](@ref) when every value is a number.
 3. Otherwise raise, naming the key and the two forms that resolve it.

# Arguments

  - `vals`: The key's values.
  - `key`: The key, named in the message.

# Validation

  - Every value is a string, or every value is a number. Raises an `ArgumentError`.

# Returns

  - The input type.

# Related

  - [`panel_input`](@ref)
  - [`CategoricalPanelInput`](@ref)
  - [`NumericPanelInput`](@ref)
"""
function panel_input_kind(vals::AbstractVector, key::AbstractString)
    if all(v -> isa(v, AbstractString), vals)
        return CategoricalPanelInput
    end
    if all(v -> isa(v, Number), vals)
        return NumericPanelInput
    end
    return throw(ArgumentError("the UniverseSets key `$key` mixes element types, so `panel_input` cannot tell whether it is a classification or a quantity. Give the key one element type, or force the reading with `panel_input(sets, \"$key\" => CategoricalPanelInput)` or `=> NumericPanelInput`. Got\neltype(sets.dict[\"$key\"]) => $(eltype(vals))"))
end
"""
    panel_input_build(::Type{CategoricalPanelInput}, sets, key, vals, name, levels, alg)
    panel_input_build(::Type{NumericPanelInput}, sets, key, vals, name, levels, alg)

Build the raw Panel Field input [`panel_input`](@ref) resolved to.

# Algorithm

The method that Julia selects is the algorithm. Each builds its own input type from the key's values, named by [`panel_input_name`](@ref) unless `name` overrides it. `levels` reaches the categorical form alone; the numeric form has no levels to declare and refuses one rather than ignoring it.

# Arguments

  - The first positional: the input type to build.
  - `sets`: The universe sets, read for `xkey`.
  - `key`: The key the values came from.
  - `vals`: The key's values.
  - `name`: Panel Field name, or `nothing` to derive it.
  - `levels`: Category levels, or `nothing`.
  - `alg`: Fill policy.

# Validation

  - `levels` is `nothing` on the numeric form. Raises an `ArgumentError`.

# Returns

  - One `AbstractPanelFieldInput`.

# Related

  - [`panel_input`](@ref)
  - [`panel_input_kind`](@ref)
  - [`panel_input_name`](@ref)
"""
function panel_input_build(::Type{CategoricalPanelInput}, sets::UniverseSets,
                           key::AbstractString, vals::AbstractVector,
                           name::Option{<:AbstractString}, levels::Option{<:VecStr},
                           alg::AbstractPanelFillAlgorithm)
    return CategoricalPanelInput(; name = if isnothing(name)
                                     panel_input_name(sets, key)
                                 else
                                     String(name)
                                 end, vals = string.(vals), levels = levels, alg = alg)
end
function panel_input_build(::Type{NumericPanelInput}, sets::UniverseSets,
                           key::AbstractString, vals::AbstractVector,
                           name::Option{<:AbstractString}, levels::Option{<:VecStr},
                           alg::AbstractPanelFillAlgorithm)
    @argcheck(isnothing(levels),
              ArgumentError("a numeric Panel Field declares no levels, so `levels` has no meaning for the UniverseSets key `$key`. Drop it, or force the categorical reading with `\"$key\" => CategoricalPanelInput`."))
    return NumericPanelInput(; name = if isnothing(name)
                                 panel_input_name(sets, key)
                             else
                                 String(name)
                             end, vals = vals, alg = alg)
end
"""
    port_opt_view(smtx::MatNum, i, args...; kwargs...)
    port_opt_view(smtx::VecMatNum_ASetMatE, i, args...; kwargs...)
    port_opt_view(smtx::AbstractVector{<:AssetSetsMatrixEstimator}, i, args...; kwargs...)

Take an asset view of an asset-group membership matrix, or of a vector of matrices and estimators.

The matrix method slices **columns**, so it expects the **`groups × assets`** matrix [`asset_sets_matrix`](@ref) returns, whose column axis is the assets. An estimator names a key rather than holding data, and the key is resolved after the view, so an estimator entry is carried through unchanged.

# Algorithm

 1. For a matrix, return `view(smtx, :, i)`, the columns of the selected assets.
 2. For a vector, take the view of each entry through the method that entry dispatches to, then narrow the element type with [`concrete_typed_array_if_abstract`](@ref).

# Arguments

  - `smtx`: A `groups × assets` membership matrix, or a vector whose entries are each such a matrix or an [`AssetSetsMatrixEstimator`](@ref).
  - `i`: The asset index or range to slice.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - For a matrix, a column view of size `groups × length(i)`. For a vector, one result per entry, in the order of `smtx`, in a concretely typed array.

# Related

  - [`asset_sets_matrix`](@ref)
  - [`AssetSetsMatrixEstimator`](@ref)
  - [`VecMatNum_ASetMatE`](@ref)
  - [`concrete_typed_array_if_abstract`](@ref)
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

export AssetSetsMatrixEstimator, asset_sets_matrix, panel_input
