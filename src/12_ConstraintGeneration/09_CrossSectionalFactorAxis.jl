"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the raw factor names and the Factor Family label of one Factor Exposure.

A [`OneHotExposure`](@ref) expands to one `"<field>=<level>"` name per level of the Panel Field it reads, so the name the caller paired it with names the block rather than a column. Every other Exposure Estimator produces one factor, which takes the caller's name.

# Arguments

  - `nm::AbstractString`: Name the caller paired the estimator with.
  - `xe`: An Exposure Estimator.
  - `rd`: Returns data carrying the Asset Panel the levels are read from.

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

Return the cross-sectional factor axis a set of Factor Exposures produces, before any fit.

The axis is fixed by the Asset Panel's field index, so it is the same in every fold. A caller who writes a [`FactorSpace`](@ref) mandate reads it before a prior exists, and the fitted [`CrossSectionalFactorModel`](@ref) stores the same answer in its `nf` and `fam`.

# Algorithm

 1. Take the Pairs in the order the caller wrote them, which is the column order of the exposures.
 2. Expand each Pair through [`exposure_axis_names`](@ref), so a one-hot member contributes one name per level.
 3. Refuse a repeated factor name.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.
  - `rd`: Returns data carrying the Asset Panel the one-hot levels are read from.

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
 2. Write the names under `sets.cfkey`, so [`factor_axis_key`](@ref) finds them from a [`CrossSectionalFactorModel`](@ref).
 3. Write one plain group per Factor Family label, holding that family's member names. A plain group carries no axis prefix, so a constraint generator reads it as a group of names rather than as a partition of an axis.
 4. Build the widened [`UniverseSets`](@ref), which re-runs every guard of the constructor.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.
  - `rd`: Returns data carrying the Asset Panel the one-hot levels are read from, and the asset names a new sets declares.
  - `sets`: A declared universe to widen. When it is `nothing`, a new one is built over `rd.nx` with the default key prefixes.

# Validation

  - The rules of [`cross_sectional_factor_axis`](@ref).
  - A new sets needs `rd.nx`, because the asset axis is the one mandatory axis of a [`UniverseSets`](@ref).
  - A Factor Family label equal to a factor name is refused, unless the family holds that one factor and nothing else. Two different lists would otherwise answer to one name.
  - The rules of [`UniverseSets`](@ref).

# Returns

  - `sets::UniverseSets`: The declared universe, carrying the cross-sectional factor axis and one group per Factor Family.

# Related

  - [`cross_sectional_factor_axis`](@ref)
  - [`UniverseSets`](@ref)
  - [`factor_axis_key`](@ref)
  - [`FactorSpace`](@ref)
"""
function cross_sectional_factor_sets(factors::AbstractVector{<:Pair}, rd::ReturnsResult,
                                     sets::Option{<:UniverseSets} = nothing)::UniverseSets
    res = cross_sectional_factor_axis(factors, rd)
    nf = res.nf
    fam = res.fam
    dict = cross_sectional_sets_dict(rd, sets)
    cfkey = isnothing(sets) ? "ncf" : sets.cfkey
    dict[cfkey] = nf
    for nm in unique(fam)
        mem = [nf[i] for i in eachindex(nf) if fam[i] == nm]
        @argcheck(nm ∉ nf || mem == [nm],
                  ArgumentError("the Factor Family $nm has the name of a factor, and it holds $mem. A label may name a factor only when the family is that one factor"))
        dict[nm] = mem
    end
    return if isnothing(sets)
        UniverseSets(; dict = dict)
    else
        UniverseSets(; xkey = sets.xkey, uxkey = sets.uxkey, tfkey = sets.tfkey,
                     utfkey = sets.utfkey, cfkey = sets.cfkey, ucfkey = sets.ucfkey,
                     zkey = sets.zkey, dict = dict)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the dictionary a cross-sectional factor axis is written into.

Widening copies the declared universe's own dictionary, so every axis it already carries survives. Building one from nothing declares the asset axis alone, under the default `xkey`.

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
