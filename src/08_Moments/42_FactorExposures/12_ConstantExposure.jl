"""
$(DocStringExtensions.TYPEDEF)

A Factor Exposure equal to one for every asset at every observation.

The member is the intercept of the Cross-Sectional Regression, and its factor return is the common return the styles and the classifications are measured against. It reads no Panel Field: it takes its shape from the active mask of the Asset Panel alone.

# Mathematical definition

```math
\\begin{align}
x_{t,j} &= 1\\,.
\\end{align}
```

Where:

  - ``x_{t,j}``: Factor Exposure of asset ``j`` at observation ``t``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConstantExposure(; family::AbstractString = \"market\") -> ConstantExposure

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(family)`.

# Examples

```jldoctest
julia> ConstantExposure()
ConstantExposure
  family ┴ String: \"market\"
```

# Related

  - [`AbstractExposureEstimator`](@ref)
  - [`factor_exposure`](@ref)
  - [`OneHotExposure`](@ref)
  - [`CompositeExposure`](@ref)
"""
@concrete struct ConstantExposure <: AbstractExposureEstimator
    """
    Label of the Factor Family the factor belongs to.
    """
    family
    function ConstantExposure(family::AbstractString)
        assert_exposure_family(family)
        return new{typeof(family)}(family)
    end
end
function ConstantExposure(; family::AbstractString = "market")::ConstantExposure
    return ConstantExposure(family)
end
"""
    factor_exposure(xe::ConstantExposure, rd::ReturnsResult) -> Matrix{<:Real}

Compute the constant Factor Exposure of a carrier.

# Algorithm

 1. Fill an `observations × assets` matrix with ones.
 2. Write `NaN` into every cell the active mask does not activate.

# Arguments

  - `xe`: Constant Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).

# Returns

  - `L::Matrix{<:Real}`: The Factor Exposure, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"a\", vals = [1.0 2.0; 3.0 4.0])];
                         amsk = [true true; true false], emsk = [true true; true false]);

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> factor_exposure(ConstantExposure(), rd)
2×2 Matrix{Float64}:
 1.0    1.0
 1.0  NaN
```

# Related

  - [`ConstantExposure`](@ref)
  - [`AbstractExposureEstimator`](@ref)
  - [`exposure_active_fill!`](@ref)
"""
function factor_exposure(::ConstantExposure, rd::ReturnsResult)::Matrix{<:Real}
    pnl = rd.pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a constant Factor Exposure takes its shape from the active mask of an Asset Panel, and rd.pnl is nothing. Build the carrier with the `pnl` that asset_panel returns."))
    L = ones(float(eltype(rd.X)), size(pnl.amsk))
    exposure_active_fill!(L, pnl)
    return L
end

export ConstantExposure
