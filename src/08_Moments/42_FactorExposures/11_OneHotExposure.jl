"""
$(DocStringExtensions.TYPEDEF)

A Factor Exposure that expands one categorical Panel Field into one factor per level.

An industry classification and a country of listing are the standard cases: the asset carries a one on the level it belongs to and a zero on every other level, so the cross-sectional regression estimates one factor return per level. The member is the only one whose Factor Exposure carries a third axis.

The factors are named `\"<field>=<level>\"`, in the level order the Panel Field declares, which is the expansion form a taxonomy key already uses. [`one_hot_exposure_names`](@ref) returns them.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OneHotExposure(; field::AbstractString, family::AbstractString) -> OneHotExposure

Keywords correspond to the struct's fields. `family` takes no default, because a one-hot expansion is a family of its own and only the caller knows whether it is an industry, a country or a sector.

## Validation

  - `!isempty(field)` and `!isempty(family)`.

# Examples

```jldoctest
julia> OneHotExposure(; field = \"industry\", family = \"industry\")
OneHotExposure
   field ┼ String: \"industry\"
  family ┴ String: \"industry\"
```

# Related

  - [`AbstractExposureEstimator`](@ref)
  - [`factor_exposure`](@ref)
  - [`one_hot_exposure_names`](@ref)
  - [`ConstantExposure`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`panel_field_labels`](@ref)
"""
@concrete struct OneHotExposure <: AbstractExposureEstimator
    """
    Name of the categorical Panel Field to expand.
    """
    field
    """
    Label of the Factor Family the levels belong to.
    """
    family
    function OneHotExposure(field::AbstractString, family::AbstractString)
        assert_panel_terms(field, :field)
        assert_exposure_family(family)
        return new{typeof(field), typeof(family)}(field, family)
    end
end
function OneHotExposure(; field::AbstractString, family::AbstractString)::OneHotExposure
    return OneHotExposure(field, family)
end
"""
    one_hot_field(rd::ReturnsResult, name::AbstractString) -> PanelField

Look the categorical Panel Field a [`OneHotExposure`](@ref) expands up on a carrier's Asset Panel.

The lookup and its two refusals are written once, because the verb and the factor names both need the field.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `name`: The Panel Field's name.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - `name` names a Panel Field, and its kind is [`CategoricalPanelField`](@ref). Raises a `KeyError` or an `ArgumentError`.

# Returns

  - `f::PanelField`: The Panel Field.

# Related

  - [`OneHotExposure`](@ref)
  - [`one_hot_exposure_names`](@ref)
  - [`panel_field`](@ref)
  - [`CategoricalPanelField`](@ref)
"""
function one_hot_field(rd::ReturnsResult, name::AbstractString)::CategoricalPanelField
    pnl = rd.pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a one-hot Factor Exposure reads its Panel Field off an Asset Panel, and rd.pnl is nothing. Build the carrier with the `pnl` that asset_panel returns."))
    f = panel_field(pnl, name)
    @argcheck(isa(f, CategoricalPanelField),
              ArgumentError("a one-hot Factor Exposure expands one level per factor, so the Panel Field \"$name\" must be a CategoricalPanelField, got a $(nameof(typeof(f)))"))
    @argcheck(ndims(f.codes) == 2,
              DimensionMismatch("a one-hot Factor Exposure is read per observation and asset, so the Panel Field \"$name\" must be time-varying; this Asset Panel is static"))
    return f
end
"""
    one_hot_exposure_names(xe::OneHotExposure, rd::ReturnsResult) -> Vector{String}

Return the factor names of a one-hot Factor Exposure, one per level, in column order.

The names are the column labels the Panel Field contributes to the feature axis, `\"<field>=<level>\"`, so the factor axis of the fit and the feature axis of the carrier spell one level the same way.

# Arguments

  - `xe`: One-hot Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`one_hot_field`](@ref).

# Returns

  - `names::Vector{String}`: The factor names.

# Examples

```jldoctest
julia> pnl = asset_panel([CategoricalPanelInput(; name = \"sector\",
                                                vals = [\"tech\" \"banks\"; \"tech\" \"tech\"])];
                         amsk = trues(2, 2), emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> PortfolioOptimisers.one_hot_exposure_names(OneHotExposure(; field = \"sector\",
                                                                 family = \"sector\"), rd)
2-element Vector{String}:
 \"sector=banks\"
 \"sector=tech\"
```

# Related

  - [`OneHotExposure`](@ref)
  - [`one_hot_field`](@ref)
  - [`panel_field_labels`](@ref)
"""
function one_hot_exposure_names(xe::OneHotExposure, rd::ReturnsResult)::Vector{String}
    f = one_hot_field(rd, xe.field)
    return panel_field_labels(f)
end
"""
    one_hot_level_fill!(B::AbstractArray{<:Real, 3}) -> nothing

Write `NaN` across every level of a one-hot Factor Exposure where the asset sets no level, in place.

An asset that belongs to no level of the classification has no exposure to any of its factors, and a row of zeros would say that it belongs to none of them with certainty. A level counts as set when its entry is finite and not zero.

# Arguments

  - `B`: The one-hot block, `observations × assets × levels`, changed in place.

# Returns

  - `nothing`. `B` carries the filled block.

# Related

  - [`OneHotExposure`](@ref)
  - [`one_hot_observed_fill!`](@ref)
  - [`factor_exposure`](@ref)
"""
function one_hot_level_fill!(B::AbstractArray{<:Real, 3})::Nothing
    Tf = eltype(B)
    for t in axes(B, 1), i in axes(B, 2)
        set = false
        for l in axes(B, 3)
            v = B[t, i, l]
            if isfinite(v) && !iszero(v)
                set = true
                break
            end
        end
        if !set
            for l in axes(B, 3)
                B[t, i, l] = Tf(NaN)
            end
        end
    end
    return nothing
end
"""
    one_hot_observed_fill!(B::AbstractArray{<:Real, 3}, Z::Arr3Num, ocols::Nothing) -> nothing
    one_hot_observed_fill!(B::AbstractArray{<:Real, 3}, Z::Arr3Num, ocols::VecInt) -> nothing

Write `NaN` across every level of a one-hot Factor Exposure where the Panel Field was not observed, in place.

A blank never reaches a carrier: the builder resolves it to a fill value and records the resolution in an observed-mask column. The read undoes that resolution, so a level the fill wrote does not become a classification the asset never carried.

# Arguments

  - `B`: The one-hot block, `observations × assets × levels`, changed in place.
  - `Z`: Time-varying feature matrix `observations × assets × features` the observed-mask column lives in.
  - `ocols`: Columns of `Z` holding the Panel Field's observed mask, or `nothing` when the Panel Field cannot blank.

# Returns

  - `nothing`. `B` carries the filled block.

# Related

  - [`OneHotExposure`](@ref)
  - [`one_hot_level_fill!`](@ref)
  - [`factor_exposure`](@ref)
"""
function one_hot_observed_fill!(::AbstractArray{<:Real, 3}, ::Nothing)::Nothing
    return nothing
end
function one_hot_observed_fill!(B::AbstractArray{<:Real, 3},
                                omsk::AbstractMatrix{Bool})::Nothing
    Tf = eltype(B)
    for t in axes(B, 1), i in axes(B, 2)
        if !omsk[t, i]
            for l in axes(B, 3)
                B[t, i, l] = Tf(NaN)
            end
        end
    end
    return nothing
end
"""
    factor_exposure(xe::OneHotExposure, rd::ReturnsResult) -> Array{<:Real, 3}

Compute the Factor Exposure of a categorical Panel Field, one factor per level.

# Algorithm

 1. Look the Panel Field up, and expand its codes into a one-hot block.
 2. Write `NaN` across the levels of every cell the Panel Field does not observe.
 3. Write `NaN` across the levels of every cell that sets no level.
 4. Write `NaN` across the levels of every cell the active mask does not activate.

# Arguments

  - `xe`: One-hot Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`one_hot_field`](@ref).

# Returns

  - `L::Array{<:Real, 3}`: The Factor Exposure, `observations × assets × levels`, in the level order the Panel Field declares.

# Examples

```jldoctest
julia> pnl = asset_panel([CategoricalPanelInput(; name = \"sector\",
                                                vals = [\"tech\" \"banks\"; \"tech\" \"tech\"])];
                         amsk = [true true; true false], emsk = [true true; true false]);

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> factor_exposure(OneHotExposure(; field = \"sector\", family = \"sector\"), rd)
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 0.0    1.0
 0.0  NaN

[:, :, 2] =
 1.0    0.0
 1.0  NaN
```

# Related

  - [`OneHotExposure`](@ref)
  - [`AbstractExposureEstimator`](@ref)
  - [`one_hot_exposure_names`](@ref)
  - [`one_hot_level_fill!`](@ref)
  - [`exposure_active_fill!`](@ref)
"""
function factor_exposure(xe::OneHotExposure, rd::ReturnsResult)::Array{<:Real, 3}
    f = one_hot_field(rd, xe.field)
    codes = f.codes
    Tf = Float64
    B = zeros(Tf, size(codes, 1), size(codes, 2), length(f.levels))
    for i in CartesianIndices(codes)
        B[i, codes[i]] = one(Tf)
    end
    one_hot_observed_fill!(B, f.omsk)
    one_hot_level_fill!(B)
    exposure_active_fill!(B, rd.pnl)
    return B
end

export OneHotExposure
