"""
$(DocStringExtensions.TYPEDEF)

Represents the Range risk measure.

`Range` computes the difference between the maximum and minimum portfolio returns, measuring the full spread of the return distribution. It is a simple measure of the total variability across all scenarios.

# Mathematical Definition

```math
\\mathrm{Range}(\\boldsymbol{x}) = \\max_{1 \\leq t \\leq T} x_t - \\min_{1 \\leq t \\leq T} x_t\\,.
```

# Fields

  - `settings`: Risk measure configuration.

# Constructors

    Range(;
        settings::RiskMeasureSettings = RiskMeasureSettings()
    ) -> Range

Keywords correspond to the struct's fields.

# Functor

    (r::Range)(x::VecNum)

Computes the Range of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> Range()
Range
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`WorstRealisation`](@ref)
  - [`ValueatRiskRange`](@ref)
"""
@concrete struct Range <: RiskMeasure
    settings
    function Range(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function Range(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return Range(settings)
end
function (::Range)(x::VecNum)
    lb, ub = extrema(x)
    return ub - lb
end

export Range
