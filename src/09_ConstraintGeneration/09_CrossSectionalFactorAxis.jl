"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the raw factor names and the Factor Family label of one Factor Exposure.

A [`OneHotExposure`](@ref) expands to one `"<field>=<level>"` name per level of the Panel Field it reads, so the name that the caller pairs with it names the block and not a column. A [`CurrencyExposure`](@ref) expands the same way, less its base currency. Any other member of [`AbstractObservedExposureEstimator`](@ref) takes the name the caller pairs with it when it gives one factor, as an [`ObservedExposure`](@ref) does, and names each factor by its series of the Exogenous Series when it gives several, under its own family label. Every other Exposure Estimator produces one factor, and that factor takes the name of the caller.

# Arguments

  - `nm::AbstractString`: The name that the caller pairs with the estimator. The one-hot and the currency methods ignore it.
  - `xe`: An Exposure Estimator.
  - `rd`: Returns data that carries the Asset Panel. The one-hot and the currency methods read the levels of their Panel Field, and every other method ignores it.

# Returns

  - `nf::Vector{String}`: The raw factor names the estimator produces, in column order.
  - `fam::Vector{String}`: The estimator's Factor Family label, repeated once per name.

# Related

  - [`cross_sectional_factor_axis`](@ref)
  - [`OneHotExposure`](@ref)
  - [`CurrencyExposure`](@ref)
  - [`ObservedExposure`](@ref)
  - [`one_hot_exposure_names`](@ref)
"""
function exposure_axis_names(nm::AbstractString, xe::AbstractExposureEstimator,
                             ::ReturnsResult)
    return [String(nm)], [String(xe.family)]
end
function exposure_axis_names(::AbstractString, xe::OneHotExposure, rd::ReturnsResult)
    nf = one_hot_exposure_names(xe, rd)
    return nf, fill(String(xe.family), length(nf))
end
function exposure_axis_names(::AbstractString, xe::CurrencyExposure, rd::ReturnsResult)
    nf = panel_field_labels(one_hot_field(rd, xe.field))[currency_level_columns(xe, rd)]
    return nf, fill(String(xe.family), length(nf))
end
"""
$(DocStringExtensions.TYPEDEF)

The Cross-Sectional Factor Axis that a set of Factor Exposures produces, before any fit.

[`cross_sectional_factor_axis`](@ref) returns it. It holds the name of each factor and the Factor Family label of each name, in the column order of the exposures. The fitted [`CrossSectionalFactorModel`](@ref) stores the same two lists in its `nf` and `fam`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalFactorAxisResult(nf::VecStr, fam::VecStr) -> CrossSectionalFactorAxisResult
    CrossSectionalFactorAxisResult(; nf::VecStr, fam::VecStr) -> CrossSectionalFactorAxisResult

Keywords correspond to the struct's fields. The keyword form calls the positional form, so the two forms run the same checks.

## Validation

  - `length(nf) == length(fam)`. Raises a `DimensionMismatch` otherwise.
  - `nf` repeats no name. Raises an `ArgumentError` otherwise.

# Examples

```jldoctest
julia> ax = CrossSectionalFactorAxisResult(; nf = [\"market\", \"value\", \"size\"],
                                           fam = [\"market\", \"style\", \"style\"]);

julia> ax.fam
3-element Vector{String}:
 "market"
 "style"
 "style"
```

# Related

  - [`cross_sectional_factor_axis`](@ref)
  - [`cross_sectional_factor_sets`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
@concrete struct CrossSectionalFactorAxisResult <: AbstractResult
    """
    The raw name of each factor, in the column order of the exposures. A one-hot member gives one name for each level of its Panel Field.
    """
    nf
    """
    The Factor Family label of each name in `nf`, in the same order.
    """
    fam
    function CrossSectionalFactorAxisResult(nf::VecStr,
                                            fam::VecStr)::CrossSectionalFactorAxisResult
        @argcheck(length(nf) == length(fam),
                  DimensionMismatch("fam ($(length(fam))) must have one label for each factor name in nf ($(length(nf)))"))
        @argcheck(allunique(nf),
                  ArgumentError("the cross-sectional factor axis repeats a name. Got $nf"))
        return new{typeof(nf), typeof(fam)}(nf, fam)
    end
end
function CrossSectionalFactorAxisResult(; nf::VecStr,
                                        fam::VecStr)::CrossSectionalFactorAxisResult
    return CrossSectionalFactorAxisResult(nf, fam)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the cross-sectional factor axis that a set of Factor Exposures produces, before any fit.

The axis holds one name for each member that is not one-hot, and one name for each level of a one-hot member. A one-hot member reads the levels that its Panel Field declares, and not the levels that the observations hold, so the axis is the same in every fold. A caller who writes a [`FactorSpace`](@ref) mandate reads the axis before a prior exists, and the fitted [`CrossSectionalFactorModel`](@ref) stores the same answer in its `nf` and `fam`.

# Algorithm

 1. Take the Pairs in the order the caller wrote them, which is the column order of the exposures.
 2. Expand each Pair through [`exposure_axis_names`](@ref), so a one-hot member contributes one name per level.
 3. Build the [`CrossSectionalFactorAxisResult`](@ref) of the names and the labels. Its constructor refuses a repeated factor name.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.
  - `rd`: Returns data that carries the Asset Panel, which declares the levels of each one-hot member.

# Validation

  - `factors` is not empty.
  - No factor name appears twice.

# Returns

  - `ax::CrossSectionalFactorAxisResult`: The raw factor names in column order, in `ax.nf`, and the Factor Family label of each name, in `ax.fam`.

# Related

  - [`CrossSectionalFactorAxisResult`](@ref)
  - [`cross_sectional_factor_sets`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`exposure_axis_names`](@ref)
"""
function cross_sectional_factor_axis(factors::AbstractVector{<:Pair},
                                     rd::ReturnsResult)::CrossSectionalFactorAxisResult
    @argcheck(!isempty(factors), IsEmptyError("factors cannot be empty"))
    nf = String[]
    fam = String[]
    for pr in factors
        n, f = exposure_axis_names(String(first(pr)), last(pr), rd)
        append!(nf, n)
        append!(fam, f)
    end
    return CrossSectionalFactorAxisResult(nf, fam)
end
"""
    cross_sectional_factor_sets(factors::AbstractVector{<:Pair}, rd::ReturnsResult,
                                sets::Option{<:UniverseSets} = nothing) -> UniverseSets

Declare the cross-sectional factor axis and its Factor Family groups on a [`UniverseSets`](@ref).

# Algorithm

 1. Read the axis with [`cross_sectional_factor_axis`](@ref).
 2. Take the universe to widen. When `sets` is `nothing`, build a new one over `rd.nx` with the default key prefixes.
 3. Copy the dictionary of that universe with [`cross_sectional_sets_dict`](@ref), and write the names under its `cfkey`, so [`factor_axis_key`](@ref) finds them from a [`CrossSectionalFactorModel`](@ref).
 4. Write one plain group per Factor Family label, which holds the names of the members of that family. A plain group carries no key prefix, so a constraint generator reads it as a group of names and not as a partition of an axis.
 5. Build the widened [`UniverseSets`](@ref) with the key prefixes of step 2, through [`rebuild_struct`](@ref). The keyword constructor runs every one of its guards again.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.
  - `rd`: Returns data that carries the Asset Panel, which declares the levels of each one-hot member. It also carries the asset names that a new sets declares.
  - `sets`: A declared universe to widen. When it is `nothing`, the function builds a new one over `rd.nx` with the default key prefixes.

# Validation

  - The rules of [`cross_sectional_factor_axis`](@ref).
  - A new sets needs `rd.nx`, because the asset axis is the one mandatory axis of a [`UniverseSets`](@ref).
  - A Factor Family label that equals a factor name is refused, unless the family holds that one factor and nothing else. Otherwise one name would name two different lists.
  - A Factor Family label that starts with `xkey`, `uxkey`, `tfkey`, `utfkey`, `cfkey` or `ucfkey` of the universe, or equals its `nikey`, is refused. [`UniverseSets`](@ref) reads such a key as an axis, as a partition of an axis, or as the Non-Investable Axis, and not as a group of names. A label that only starts with `nikey`, such as `"nikkei"` against the default `"ni"`, is a plain group.
  - A widened universe that already declares the cross-sectional axis, or a group under a Factor Family label, must declare the same list. [`cross_sectional_sets_write!`](@ref) refuses a different list and does not replace it.
  - The rules of [`UniverseSets`](@ref).

# Returns

  - `sets::UniverseSets`: The declared universe, which carries the cross-sectional factor axis and one group per Factor Family.

# Related

  - [`cross_sectional_factor_axis`](@ref)
  - [`cross_sectional_sets_write!`](@ref)
  - [`UniverseSets`](@ref)
  - [`factor_axis_key`](@ref)
  - [`FactorSpace`](@ref)
"""
function cross_sectional_factor_sets(factors::AbstractVector{<:Pair}, rd::ReturnsResult,
                                     sets::Option{<:UniverseSets} = nothing)::UniverseSets
    (; nf, fam) = cross_sectional_factor_axis(factors, rd)
    us = if isnothing(sets)
        UniverseSets(; dict = cross_sectional_sets_dict(rd, nothing))
    else
        sets
    end
    pre = (us.xkey, us.uxkey, us.tfkey, us.utfkey, us.cfkey, us.ucfkey)
    dict = cross_sectional_sets_dict(rd, us)
    cross_sectional_sets_write!(dict, us.cfkey, nf)
    for nm in unique(fam)
        @argcheck(nm != us.nikey && !any(p -> startswith(nm, p), pre),
                  ArgumentError("the Factor Family $nm starts with a key prefix of the declared universe $pre, or equals its nikey $(us.nikey), so UniverseSets would read its group as an axis and not as a group of names. Rename the Factor Family"))
        mem = [nf[i] for i in eachindex(nf) if fam[i] == nm]
        @argcheck(nm ∉ nf || mem == [nm],
                  ArgumentError("the Factor Family $nm has the name of a factor, and it holds $mem. A label may name a factor only when the family is that one factor"))
        cross_sectional_sets_write!(dict, nm, mem)
    end
    return rebuild_struct(us, (; dict = dict))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Write one entry of a cross-sectional factor axis into a universe dictionary, and refuse to replace a different one.

A widened universe carries every entry that the caller declared. A Factor Family label is a plain group name, so a group of the caller can have the same name. A silent replacement of that group would change the meaning of every constraint written against it. So the function refuses a key that holds a different list, and the error names the key. A key that holds the same list keeps its value.

# Arguments

  - `dict`: The dictionary that the caller builds.
  - `key`: The key to write.
  - `val`: The list to write under it.

# Validation

  - `dict` holds no entry under `key`, or holds `val` there.

# Returns

  - `nothing`.

# Examples

```jldoctest
julia> d = Dict(\"style\" => [\"a\"]);

julia> PortfolioOptimisers.cross_sectional_sets_write!(d, \"style\", [\"a\"]);

julia> PortfolioOptimisers.cross_sectional_sets_write!(d, \"industry\", [\"b\", \"c\"]);

julia> d[\"industry\"]
2-element Vector{String}:
 "b"
 "c"
```

# Related

  - [`cross_sectional_factor_sets`](@ref)
  - [`cross_sectional_sets_dict`](@ref)
"""
function cross_sectional_sets_write!(dict::AbstractDict, key::AbstractString,
                                     val::AbstractVector)::Nothing
    @argcheck(!haskey(dict, key) || dict[key] == val,
              ArgumentError("the declared universe already holds $(dict[key]) under $key, and the cross-sectional factor axis would write $val there. Rename the Factor Family, or drop the entry from the universe"))
    dict[key] = val
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the dictionary that [`cross_sectional_factor_sets`](@ref) writes a cross-sectional factor axis into.

The method for a declared universe copies its dictionary, so every axis that the universe declares stays in the result. The copy is shallow, and the caller only adds keys to it, so the declared universe does not change. The method for `nothing` declares the asset axis alone, under the default `xkey`, `"nx"`.

# Arguments

  - `rd`: Returns data carrying the asset names.
  - `sets`: The declared universe to widen, or `nothing`.

# Validation

  - `rd.nx` is set when `sets` is `nothing`.

# Returns

  - `dict::Dict{String}`: The dictionary to write the axis into. Its value type is the `typejoin` of the type of the values that it copies and `Vector{String}`, the type of the lists that [`cross_sectional_factor_sets`](@ref) writes into it.

# Related

  - [`cross_sectional_factor_sets`](@ref)
  - [`UniverseSets`](@ref)
"""
function cross_sectional_sets_dict(rd::ReturnsResult, ::Nothing)::Dict{String}
    @argcheck(!isnothing(rd.nx),
              ArgumentError("a new UniverseSets needs the asset names, and rd.nx is unset. Pass a declared universe to widen, or set nx on the returns data"))
    return Dict{String, typejoin(typeof(rd.nx), Vector{String})}("nx" => rd.nx)
end
function cross_sectional_sets_dict(::ReturnsResult, sets::UniverseSets)::Dict{String}
    # `typejoin`, not `promote_type`: `promote_type(Vector{Symbol}, Vector{String})` is
    # `Vector{Any}`, and the copy would convert each list of names to it (#1979).
    return Dict{String, typejoin(valtype(sets.dict), Vector{String})}(sets.dict)
end

export CrossSectionalFactorAxisResult, cross_sectional_factor_axis,
       cross_sectional_factor_sets
