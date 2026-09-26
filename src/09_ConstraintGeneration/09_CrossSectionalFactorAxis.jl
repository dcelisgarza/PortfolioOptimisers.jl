"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the raw factor names and the Factor Family label of one Factor Exposure.

A [`OneHotExposure`](@ref) expands to one `"<field>=<level>"` name per level of the Panel Field it reads, so the name that the caller pairs with it names the block and not a column. Every other Exposure Estimator produces one factor, and that factor takes the name of the caller.

# Arguments

  - `nm::AbstractString`: The name that the caller pairs with the estimator. The one-hot method ignores it.
  - `xe`: An Exposure Estimator.
  - `rd`: Returns data that carries the Asset Panel. The one-hot method reads the levels of its Panel Field, and every other method ignores it.

# Returns

  - `nf::Vector{String}`: The raw factor names the estimator produces, in column order.
  - `fam::Vector{String}`: The estimator's Factor Family label, repeated once per name.

# Related

  - [`cross_sectional_factor_axis`](@ref)
  - [`OneHotExposure`](@ref)
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the cross-sectional factor axis that a set of Factor Exposures produces, before any fit.

The axis holds one name for each member that is not one-hot, and one name for each level of a one-hot member. A one-hot member reads the levels that its Panel Field declares, and not the levels that the observations hold, so the axis is the same in every fold. A caller who writes a [`FactorSpace`](@ref) mandate reads the axis before a prior exists, and the fitted [`CrossSectionalFactorModel`](@ref) stores the same answer in its `nf` and `fam`.

# Algorithm

 1. Take the Pairs in the order the caller wrote them, which is the column order of the exposures.
 2. Expand each Pair through [`exposure_axis_names`](@ref), so a one-hot member contributes one name per level.
 3. Refuse a repeated factor name.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.
  - `rd`: Returns data that carries the Asset Panel, which declares the levels of each one-hot member.

# Validation

  - `factors` is not empty.
  - No factor name appears twice.

# Returns

  - `nf::Vector{String}`: The raw factor names, in column order.
  - `fam::Vector{String}`: The Factor Family label of each name.

# Related

  - [`cross_sectional_factor_sets`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`exposure_axis_names`](@ref)
"""
function cross_sectional_factor_axis(factors::AbstractVector{<:Pair}, rd::ReturnsResult)
    @argcheck(!isempty(factors), IsEmptyError("factors cannot be empty"))
    nf = String[]
    fam = String[]
    for pr in factors
        n, f = exposure_axis_names(String(first(pr)), last(pr), rd)
        append!(nf, n)
        append!(fam, f)
    end
    @argcheck(allunique(nf),
              ArgumentError("the cross-sectional factor axis repeats a name. Got $nf"))
    return (; nf = nf, fam = fam)
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
 5. Build the widened [`UniverseSets`](@ref) with the key prefixes of step 2. The constructor runs every one of its guards again.

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
    return UniverseSets(; xkey = us.xkey, uxkey = us.uxkey, tfkey = us.tfkey,
                        utfkey = us.utfkey, cfkey = us.cfkey, ucfkey = us.ucfkey,
                        nikey = us.nikey, dict = dict)
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
julia> d = Dict{String, Any}(\"style\" => [\"a\"]);

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

  - `dict::Dict{String, Any}`: The dictionary to write the axis into.

# Related

  - [`cross_sectional_factor_sets`](@ref)
  - [`UniverseSets`](@ref)
"""
function cross_sectional_sets_dict(rd::ReturnsResult, ::Nothing)::Dict{String, Any}
    @argcheck(!isnothing(rd.nx),
              ArgumentError("a new UniverseSets needs the asset names, and rd.nx is unset. Pass a declared universe to widen, or set nx on the returns data"))
    return Dict{String, Any}("nx" => rd.nx)
end
function cross_sectional_sets_dict(::ReturnsResult, sets::UniverseSets)::Dict{String, Any}
    dict = Dict{String, Any}()
    for (k, v) in sets.dict
        dict[k] = v
    end
    return dict
end

export cross_sectional_factor_axis, cross_sectional_factor_sets
