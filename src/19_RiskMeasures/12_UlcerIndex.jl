"""
$(DocStringExtensions.TYPEDEF)

Represents the Ulcer Index risk measure.

`UlcerIndex` measures the depth and duration of drawdowns in portfolio returns. It is defined as the root-mean-square of the absolute drawdown series, capturing both the magnitude and persistence of losses.

# Mathematical Definition

Define the absolute drawdown series:

```math
c_t = \\sum_{s=1}^{t} x_s\\,, \\qquad d_t = c_t - \\max_{0 \\leq s \\leq t} c_s \\leq 0\\,.
```

The Ulcer Index is:

```math
\\mathrm{UI}(\\boldsymbol{x}) = \\sqrt{\\frac{1}{T} \\sum_{t=1}^{T} d_t^2} = \\frac{\\lVert \\boldsymbol{d} \\rVert_2}{\\sqrt{T}}\\,.
```

# Fields

  - `settings`: Risk measure configuration.

# Constructors

    UlcerIndex(;
        settings::RiskMeasureSettings = RiskMeasureSettings()
    ) -> UlcerIndex

Keywords correspond to the struct's fields.

# Functor

    (r::UlcerIndex)(x::VecNum)

Computes the Ulcer Index of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> UlcerIndex()
UlcerIndex
  settings ┴ RiskMeasureSettings
               scale ┼ Float64: 1.0
                  ub ┼ nothing
                 rke ┴ Bool: true
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`AverageDrawdown`](@ref)
  - [`MaximumDrawdown`](@ref)
  - [`RelativeUlcerIndex`](@ref)
"""
@concrete struct UlcerIndex <: RiskMeasure
    settings
    function UlcerIndex(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function UlcerIndex(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return UlcerIndex(settings)
end
function (::UlcerIndex)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return LinearAlgebra.norm(dd, 2) / sqrt(length(x))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Ulcer Index risk measure for hierarchical optimisation.

`RelativeUlcerIndex` applies the Ulcer Index framework to the relative (compounded) drawdown series.

# Mathematical Definition

Define the relative drawdown series:

```math
C_t = \\prod_{s=1}^{t} (1 + x_s)\\,, \\qquad rd_t = \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1 \\leq 0\\,.
```

The Relative Ulcer Index is:

```math
\\mathrm{RUI}(\\boldsymbol{x}) = \\frac{\\lVert \\boldsymbol{rd} \\rVert_2}{\\sqrt{T}}\\,.
```

# Fields

  - `settings`: Hierarchical risk measure configuration.

# Constructors

    RelativeUlcerIndex(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings()
    ) -> RelativeUlcerIndex

Keywords correspond to the struct's fields.

# Functor

    (r::RelativeUlcerIndex)(x::VecNum)

Computes the Relative Ulcer Index of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeUlcerIndex()
RelativeUlcerIndex
  settings ┴ HierarchicalRiskMeasureSettings
               scale ┴ Float64: 1.0
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`UlcerIndex`](@ref)
  - [`RelativeAverageDrawdown`](@ref)
"""
@concrete struct RelativeUlcerIndex <: HierarchicalRiskMeasure
    settings
    function RelativeUlcerIndex(settings::HierarchicalRiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function RelativeUlcerIndex(;
                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return RelativeUlcerIndex(settings)
end
function (::RelativeUlcerIndex)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return LinearAlgebra.norm(dd, 2) / sqrt(length(x))
end

export UlcerIndex, RelativeUlcerIndex
