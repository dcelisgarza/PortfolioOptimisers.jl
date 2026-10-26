"""
$(DocStringExtensions.TYPEDEF)

Names the Universe Sets key whose groups become the rows of a binary asset-group membership matrix.

[`asset_sets_matrix`](@ref) reads the key from a [`UniverseSets`](@ref). It returns one row for each distinct group value and one column for each asset. A row is the indicator of one group, and a sub-group constraint sums over the assets that the row marks.

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

  - $(ref_dict[:chang2000]) Section 2.6, Equation 15.
  - $(ref_dict[:cajas2025]) Section 9.1, Equations 9.2 to 9.4. It adds the row of indicator coefficients and the equality over a group.
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
    const MatNum_ASetMatE = Union{AssetSetsMatrixEstimator, MatNum}

Alias for an asset sets matrix estimator or a numeric matrix.

Matches an [`AssetSetsMatrixEstimator`](@ref) or a precomputed numeric membership matrix. A JuMP optimiser checks `smtx` and `sgmtx` against it when the sub-group cardinality or threshold that reads the matrix is a scalar.

# Related

  - [`AssetSetsMatrixEstimator`](@ref)
  - [`MatNum`](@ref)
  - [`asset_sets_matrix`](@ref)
"""
const MatNum_ASetMatE = Union{AssetSetsMatrixEstimator, MatNum}
"""
    const VecMatNum_ASetMatE = AbstractVector{<:MatNum_ASetMatE}

Alias for a vector of asset sets matrix estimators or numeric matrices.

Each entry is a [`MatNum_ASetMatE`](@ref). [`asset_sets_matrix`](@ref) and [`port_opt_view`](@ref) return one result for each entry, in the order of the vector.

# Related

  - [`MatNum_ASetMatE`](@ref)
  - [`MatNum_ASetMatE_VecMatNum_ASetMatE`](@ref)
"""
const VecMatNum_ASetMatE = AbstractVector{<:MatNum_ASetMatE}
"""
    const MatNum_ASetMatE_VecMatNum_ASetMatE = Union{MatNum_ASetMatE, VecMatNum_ASetMatE}

Alias for one asset sets matrix estimator or numeric matrix, or a vector of them.

Matches a [`MatNum_ASetMatE`](@ref) or a [`VecMatNum_ASetMatE`](@ref). The `smtx` and `sgmtx` fields of [`JuMPOptimiser`](@ref) and [`ProgrammeAllocationSet`](@ref) take it.

# Related

  - [`MatNum_ASetMatE`](@ref)
  - [`VecMatNum_ASetMatE`](@ref)
"""
const MatNum_ASetMatE_VecMatNum_ASetMatE = Union{MatNum_ASetMatE, VecMatNum_ASetMatE}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the taxonomy column `sets.dict[key]`, and check that the key exists.

It does for a group name key what [`factor_universe`](@ref) does for the factor universe key. [`asset_sets_matrix`](@ref) and [`panel_input`](@ref) read a Universe Sets key through it, so a missing key gives the same message in both. A bare `sets.dict[key]` raises a `KeyError` that holds the key alone. That error does not tell which function asked for the key, and it suggests no correction for a typo.

`need` names the caller. [`suggest_declared_key`](@ref) gives the suggestion, with the configuration that admits more matches than the global default. The global default is strict so that a message does not show a real asset name. The candidates here are `sets.dict` keys that the caller wrote, so that risk does not apply.

# Arguments

  - `sets`: A [`UniverseSets`](@ref) that holds the asset universe and the groups.
  - `key`: The group name key to read.
  - `need`: The name of the caller, which the error message gives.

# Validation

  - `haskey(sets.dict, key)`. Raises a `KeyError`. The message names `key` and `need`, and ends with a [`suggest_declared_key`](@ref) suggestion. The exception stays a `KeyError`, as in [`factor_universe`](@ref), because the key is missing. The message is the only change.

# Returns

  - `col`: The value of `sets.dict[key]`, with one group for each asset.

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

Build a binary asset-group membership matrix from a Universe Sets key.

The key `smtx` names a column of `sets.dict` that holds one group value for each asset. The matrix has one row for each distinct group value and one column for each asset of the universe. The entry `[i, j]` is one when asset `j` is in group `i`.

# Algorithm

 1. Read the taxonomy column `all_sets` from `sets.dict[smtx]`, through [`taxonomy_column`](@ref).
 2. Check that `all_sets` has the length of the asset universe.
 3. Take `unique_sets = unique(all_sets)`, the distinct group values in order of first appearance. Each value defines one group. The order of `unique_sets` is the order of the groups in the result.
 4. For each group `val` of `unique_sets`, write the indicator `all_sets .== val` into column `i` of a `BitMatrix` `A`. `A` is an `assets × groups` matrix.
 5. Return `transpose(A)`. Its rows are the groups and its columns are the assets.

# Arguments

  - `smtx`: The group name key to read from `sets.dict`.
  - `sets`: A [`UniverseSets`](@ref) that holds the asset universe and the groups.

# Validation

  - `haskey(sets.dict, smtx)`, through [`taxonomy_column`](@ref).
  - `length(sets.dict[smtx]) == length(sets.dict[sets.xkey])`. Raises an `AssertionError`.

# Returns

  - `A`: The `transpose` of a `BitMatrix`, of size (number of groups) × (number of assets), where `A[i, j] == 1` if asset `j` is in group `i`. The orientation is `groups × assets`. The row order is `unique(sets.dict[smtx])`, the distinct group values in order of first appearance.

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

Return a precomputed membership matrix, or `nothing`, unchanged.

A caller can give the membership matrix itself in place of an [`AssetSetsMatrixEstimator`](@ref). With this method, one call to `asset_sets_matrix` takes an estimator, a matrix, or `nothing` when no matrix is set.

# Arguments

  - `smtx`: A precomputed membership matrix, or `nothing`.
  - `args...`: More positional arguments, which the method ignores.

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

Build the membership matrix from the group name key that `smtx` holds. The method calls:

    asset_sets_matrix(smtx.val, sets)

# Arguments

  - `smtx`: An [`AssetSetsMatrixEstimator`](@ref) that holds the group name key to read.
  - `sets`: A [`UniverseSets`](@ref) that holds the asset universe and the groups.

# Validation

  - The group name key method of [`asset_sets_matrix`](@ref) checks `smtx.val`.

# Returns

  - `A`: The `transpose` of a `BitMatrix`, of size (number of groups) × (number of assets), as the group name key method returns it.

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

Apply [`asset_sets_matrix`](@ref) to each entry of the vector.

Each entry dispatches to its own method, so one vector can hold estimators and precomputed matrices together.

# Arguments

  - `smtx`: A vector whose entries are each an [`AssetSetsMatrixEstimator`](@ref) or a precomputed numeric matrix.
  - `sets`: A [`UniverseSets`](@ref) that holds the asset universe and the groups.

# Validation

  - The method of each entry checks that entry.

# Returns

  - `A`: One result for each entry of `smtx`, in the order of `smtx`. Each result is what the method of its entry returns. A `groups × assets` transpose and an unchanged precomputed matrix can be in the same vector.

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

Name the Panel Field that a [`UniverseSets`](@ref) key becomes, which is the key without its asset prefix.

Every Panel Field of an [`AssetPanel`](@ref) is indexed by asset. The `xkey` prefix marks a Universe Sets key as asset-length, so it gives no information on a Panel Field. The function removes the prefix and the underscore after it, so `"nx_sector"` becomes `"sector"`. A key with no prefix keeps its name.

# Algorithm

 1. Return the key unchanged when it does not start with `sets.xkey` followed by an underscore.
 2. Otherwise return the key with that prefix and the underscore removed.

# Arguments

  - `sets`: The [`UniverseSets`](@ref), which gives `xkey`.
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

Convert a [`UniverseSets`](@ref) key into the raw Panel Field input that [`asset_panel`](@ref) reads.

A key is one vector over the asset axis, which is the static raw form of one Panel Field. A Universe Sets key enters an Asset Panel through this function. The values of the key select the input type. A key of strings becomes a [`CategoricalPanelInput`](@ref), and a key of numbers becomes a [`NumericPanelInput`](@ref). The function refuses a key that mixes the two, because it cannot know which reading the caller wants.

An entry `key => InputType` forces the input type, in the scalar form and in the vector form. For example, `"nx_rating" => CategoricalPanelInput` reads a numeric rating as a label, not as a number. No keyword sets the type, and the pair is the only way to force it.

[`panel_input_name`](@ref) names the field. A nested taxonomy has many keys, so it gives many fields. The vector form applies the same rule to each key.

The result is static, because a taxonomy has no observation axis. When [`asset_panel`](@ref) receives it with a time-varying input, it lifts the taxonomy to the observation axis as a field that is constant in time. One panel can thus hold a taxonomy and a table of fundamentals.

# Algorithm

The scalar form takes four steps:

 1. Read the values of the key from `sets.dict`, through [`taxonomy_column`](@ref).
 2. Choose the input type. A `Pair` gives the forced type. Otherwise [`panel_input_kind`](@ref) chooses it from the values.
 3. Name the field with [`panel_input_name`](@ref), unless `name` overrides it.
 4. Build the input with [`panel_input_build`](@ref).

The vector form takes one step:

 1. Apply the scalar form to each entry.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) that holds the key.
  - `key`: The key, or `key => InputType` to force the type.
  - `keys`: A vector of either form. The vector form takes no `name`, `levels` or `alg`, because each of them describes one field. Give a key that needs one of them to the scalar form, and pass both results to the same [`asset_panel`](@ref) call.
  - `name`: The Panel Field name, or `nothing` to derive it from the key.
  - `levels`: The category levels, or `nothing` to let the builder sort them. Only the categorical form takes levels.
  - `alg`: The fill policy for the raw values.

# Validation

  - `haskey(sets.dict, key)`. Raises a `KeyError`.
  - Every value of the key is a string, or every value is a number, unless a `Pair` forces the type. Raises an `ArgumentError`.
  - `levels` is `nothing` when the input type is numeric. Raises an `ArgumentError`.
  - `!isempty(keys)` on the vector form. Raises an `IsEmptyError`.

# Returns

  - The scalar form returns one `AbstractPanelFieldInput`. The vector form returns one input for each entry, in the order of `keys`.

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

Choose the raw Panel Field input type from the values of a [`UniverseSets`](@ref) key.

A key of strings is a classification, and a key of numbers is a quantity. The function refuses a key that holds both, because such a key describes two things.

# Algorithm

 1. Return [`CategoricalPanelInput`](@ref) when every value is a string.
 2. Return [`NumericPanelInput`](@ref) when every value is a number.
 3. Otherwise raise an `ArgumentError`. The message names the key and the two `Pair` forms that force a type.

# Arguments

  - `vals`: The values of the key.
  - `key`: The key, which the error message names.

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

Build the raw Panel Field input of the type that [`panel_input`](@ref) chose.

# Algorithm

The method that Julia selects is the algorithm. Each method builds its own input type from the values of the key. [`panel_input_name`](@ref) names the input, unless `name` overrides it. Only the categorical method passes `levels` on. A numeric input has no levels, so the numeric method refuses `levels` and does not ignore it.

# Arguments

  - The first positional argument: The input type to build.
  - `sets`: The [`UniverseSets`](@ref), which gives `xkey`.
  - `key`: The key that holds the values.
  - `vals`: The values of the key.
  - `name`: The Panel Field name, or `nothing` to derive it.
  - `levels`: The category levels, or `nothing`.
  - `alg`: The fill policy.

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

The matrix method slices columns. It expects the `groups × assets` matrix that [`asset_sets_matrix`](@ref) returns, whose columns are the assets. An estimator holds a key and no data, and [`asset_sets_matrix`](@ref) reads the key after the view. The view of an estimator entry is thus the entry, unchanged.

# Algorithm

 1. For a matrix, return `view(smtx, :, i)`, the columns of the selected assets.
 2. For a vector, take the view of each entry through the method of that entry. Then narrow the element type with [`concrete_typed_array_if_abstract`](@ref).

# Arguments

  - `smtx`: A `groups × assets` membership matrix, or a vector whose entries are each such a matrix or an [`AssetSetsMatrixEstimator`](@ref).
  - `i`: The asset index or range to slice.
  - `args...`: More positional arguments, which the method ignores.
  - `kwargs...`: More keyword arguments, which the method ignores.

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
# in `02_Tools/02_ArrayViews.jl`, and neither is more specific: `MatNum` is outside the generic's element
# union, `Nothing` is outside this one's. This method is that intersection, so the two never
# tie. Its body is the one above, because a membership matrix and its estimator must produce
# the same element type.
function port_opt_view(smtx::AbstractVector{<:AssetSetsMatrixEstimator}, i, args...;
                       kwargs...)
    return concrete_typed_array_if_abstract([port_opt_view(smtxi, i, args...; kwargs...)
                                             for smtxi in smtx])
end

export AssetSetsMatrixEstimator, asset_sets_matrix, panel_input
