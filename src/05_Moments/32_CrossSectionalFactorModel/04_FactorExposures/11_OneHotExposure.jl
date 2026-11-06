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

Look the categorical Panel Field a [`OneHotExposure`](@ref) expands up on the Asset Panel of a [`ReturnsResult`](@ref).

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
              IsNothingError("a one-hot Factor Exposure reads its Panel Field off an Asset Panel, and rd.pnl is nothing. Build the ReturnsResult with the `pnl` that asset_panel returns."))
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

The names are the column labels the Panel Field contributes to the feature axis, `\"<field>=<level>\"`, so the factor axis of the fit and the feature axis of the `ReturnsResult` spell one level the same way.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Write a one-hot Factor Exposure into `B` in one pass, in place.

A live cell takes one at the level of its code and zero at every other level. A cell that is not live takes `NaN` across every level. The codes of a categorical Panel Field lie in its levels, so every live cell sets exactly one level. A blank never reaches a `ReturnsResult`: the builder resolves it to a fill value and records the resolution in an observed mask. So the caller marks a cell that the Panel Field does not observe as not live, and a level that a fill wrote does not become a classification the asset never carried. The observation axis is the slowest axis of `B`, so it is the outer loop, and the pass writes `B` in its memory order.

# Arguments

  - `B`: The one-hot block, `assets × levels × observations`, written in place. Its element type must hold `NaN`.
  - `codes`: The codes of the Panel Field, `observations × assets`, each in `1:size(B, 2)`.
  - `live`: Mask `observations × assets`, `true` where the Panel Field observes the cell and the Asset Panel activates it.

# Validation

  - `eltype(B)` holds `NaN`, through [`nan_fill_value`](@ref), whether or not a cell needs it. Raises an `ArgumentError`.

# Returns

  - `nothing`. `B` carries the block.

# Related

  - [`OneHotExposure`](@ref)
  - [`factor_exposure`](@ref)
  - [`exposure_active_fill!`](@ref)
"""
function one_hot_fill!(B::AbstractArray{<:Number, 3}, codes::AbstractMatrix{<:Integer},
                       live::AbstractMatrix{Bool})::Nothing
    nan = nan_fill_value(B)
    on, off = one(eltype(B)), zero(eltype(B))
    for t in axes(B, 3), l in axes(B, 2), i in axes(B, 1)
        B[i, l, t] = live[t, i] ? (codes[t, i] == l ? on : off) : nan
    end
    return nothing
end
"""
    factor_exposure(xe::OneHotExposure, rd::ReturnsResult) -> Array{<:Real, 3}

Compute the Factor Exposure of a categorical Panel Field, one factor per level.

# Algorithm

 1. Look the Panel Field up.
 2. Mark a cell live when the Panel Field observes it and the active mask of the Asset Panel activates it.
 3. Expand the codes into a one-hot block through [`one_hot_fill!`](@ref), with `NaN` across the levels of every cell that is not live.

# Arguments

  - `xe`: One-hot Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`one_hot_field`](@ref).
  - The active mask of the Asset Panel has the size of the codes. Raises a `DimensionMismatch`.
  - The element type of the returns holds `NaN`, through [`nan_fill_value`](@ref). Raises an `ArgumentError`.

# Returns

  - `L::Array{<:Real, 3}`: The Factor Exposure, `assets × levels × observations`, in the level order the Panel Field declares.

# Examples

```jldoctest
julia> pnl = asset_panel([CategoricalPanelInput(; name = \"sector\",
                                                vals = [\"tech\" \"banks\"; \"tech\" \"tech\"])];
                         amsk = [true true; true false], emsk = [true true; true false]);

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> factor_exposure(OneHotExposure(; field = \"sector\", family = \"sector\"), rd)
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 0.0  1.0
 1.0  0.0

[:, :, 2] =
   0.0    1.0
 NaN    NaN
```

# Related

  - [`OneHotExposure`](@ref)
  - [`AbstractExposureEstimator`](@ref)
  - [`one_hot_exposure_names`](@ref)
  - [`one_hot_fill!`](@ref)
  - [`exposure_active_fill!`](@ref)
"""
function factor_exposure(xe::OneHotExposure, rd::ReturnsResult)::Array{<:Real, 3}
    (; codes, live, nl) = one_hot_inputs(xe, rd)
    Tf = real(eltype(rd.X))
    B = Array{Tf}(undef, size(codes, 2), nl, size(codes, 1))
    one_hot_fill!(B, codes, live)
    return B
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return what [`one_hot_fill!`](@ref) reads to expand a [`OneHotExposure`](@ref): the codes of its Panel Field, the cells that are live, and the number of levels.

[`factor_exposure`](@ref) fills a block of its own from them, and the exposure history of a [`CrossSectionalFactorPrior`](@ref) fills the columns of the member in the history, with no block between. Both read the same checks.

# Arguments

  - `xe`: One-hot Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`one_hot_field`](@ref).
  - The codes match the active mask of the Asset Panel. Raises a `DimensionMismatch`.

# Returns

  - `codes::AbstractMatrix{<:Integer}`: The codes of the Panel Field, `observations × assets`.
  - `live::AbstractMatrix{Bool}`: `true` where the Panel Field observes the cell and the active mask activates it.
  - `nl::Int`: The number of levels, one factor each.

# Related

  - [`factor_exposure`](@ref)
  - [`one_hot_fill!`](@ref)
  - [`one_hot_field`](@ref)
"""
function one_hot_inputs(xe::OneHotExposure, rd::ReturnsResult)
    f = one_hot_field(rd, xe.field)
    codes = f.codes
    amsk = rd.pnl.amsk
    @argcheck(size(amsk) == size(codes),
              DimensionMismatch("the codes of a one-hot Factor Exposure must match the active mask of the Asset Panel, got size(codes) = $(size(codes)) and size(pnl.amsk) = $(size(amsk))"))
    live = isnothing(f.omsk) ? amsk : f.omsk .& amsk
    return (; codes = codes, live = live, nl = length(f.levels))
end
function lookback(::OneHotExposure)::Integer
    return 1
end

export OneHotExposure
