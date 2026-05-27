"""
$(DocStringExtensions.TYPEDEF)

Represents the Maximum Drawdown risk measure.

`MaximumDrawdown` computes the largest peak-to-trough decline in the cumulative portfolio returns. It captures the worst-case loss from a previous high.

# Mathematical definition

Define the absolute drawdown series:

```math
\\begin{align}
c_t &= \\sum_{s=1}^{t} x_s\\,, \\\\
d_t &= c_t - \\max_{0 \\leq s \\leq t} c_s \\leq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:ct])
  - $(math_dict[:dtdd])

The Maximum Drawdown is the most negative value in the drawdown series:

```math
\\begin{align}
\\mathrm{MDD}(\\boldsymbol{x}) &= -\\min_{1 \\leq t \\leq T} d_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MDD}(\\boldsymbol{x})``: Maximum drawdown.
  - $(math_dict[:T])
  - $(math_dict[:dtdd])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumDrawdown(;
        settings::RiskMeasureSettings = RiskMeasureSettings()
    ) -> MaximumDrawdown

Keywords correspond to the struct's fields.

# Functor

    (r::MaximumDrawdown)(x::VecNum)

Computes the Maximum Drawdown of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> MaximumDrawdown()
MaximumDrawdown
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`AverageDrawdown`](@ref)
  - [`DrawdownatRisk`](@ref)
  - [`RelativeMaximumDrawdown`](@ref)
"""
@concrete struct MaximumDrawdown <: RiskMeasure
    "$(field_dict[:settings_rm])"
    settings
    function MaximumDrawdown(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function MaximumDrawdown(;
                         settings::RiskMeasureSettings = RiskMeasureSettings())::MaximumDrawdown
    return MaximumDrawdown(settings)
end
function (::MaximumDrawdown)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return -minimum(dd)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Maximum Drawdown risk measure for hierarchical optimisation.

`RelativeMaximumDrawdown` computes the maximum of the relative (compounded) drawdown series.

# Mathematical definition

Define the relative drawdown series:

```math
\\begin{align}
C_t &= \\prod_{s=1}^{t} (1 + x_s)\\,, \\\\
rd_t &= \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1 \\leq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:Ct])
  - $(math_dict[:rdt])

The Relative Maximum Drawdown is:

```math
\\begin{align}
\\mathrm{RMDD}(\\boldsymbol{x}) &= -\\min_{1 \\leq t \\leq T} rd_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RMDD}(\\boldsymbol{x})``: Relative maximum drawdown.
  - $(math_dict[:T])
  - $(math_dict[:rdt])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeMaximumDrawdown(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings()
    ) -> RelativeMaximumDrawdown

Keywords correspond to the struct's fields.

# Functor

    (r::RelativeMaximumDrawdown)(x::VecNum)

Computes the Relative Maximum Drawdown of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeMaximumDrawdown()
RelativeMaximumDrawdown
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`MaximumDrawdown`](@ref)
  - [`RelativeAverageDrawdown`](@ref)
"""
@concrete struct RelativeMaximumDrawdown <: HierarchicalRiskMeasure
    "$(field_dict[:settings_rm])"
    settings
    function RelativeMaximumDrawdown(settings::HierarchicalRiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function RelativeMaximumDrawdown(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())::RelativeMaximumDrawdown
    return RelativeMaximumDrawdown(settings)
end
function (::RelativeMaximumDrawdown)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return -minimum(dd)
end

export MaximumDrawdown, RelativeMaximumDrawdown
