"""
$(DocStringExtensions.TYPEDEF)

Measures the Maximum Drawdown, the deepest fall of the uncompounded cumulative return below its running peak.

The drawdown series is the one of [`absolute_drawdown_vec`](@ref): the cumulative return is a sum of the returns, and its running peak starts at ``c_0 = 0``. A first return that is negative is thus a drawdown. [`RelativeMaximumDrawdown`](@ref) takes the deepest point of the compounded drawdown series.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MDD}(\\boldsymbol{x}) &= -\\min_{1 \\leq t \\leq T} d_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MDD}(\\boldsymbol{x})``: Maximum drawdown.
  - $(math_dict[:xret])
  - $(math_dict[:dtdd]) [`absolute_drawdown_vec`](@ref) states the series.
  - $(math_dict[:T])

A path that never falls below its running peak has a Maximum Drawdown of zero. The measure is the limit of [`DrawdownatRisk`](@ref) and of [`ConditionalDrawdownatRisk`](@ref) as `alpha` falls to zero, and it is the [`WorstRealisation`](@ref) of the drawdown series. The measure takes no observation weights, because a positive weight does not change the deepest point of a series.

# Algorithm

 1. Compute the drawdown series `dd` with [`absolute_drawdown_vec`](@ref).
 2. Return the negated minimum of `dd`.

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

On the path of [`AverageDrawdown`](@ref), the drawdown series is `[0, -0.25, -0.1875, -0.3125, 0]`.

```jldoctest
julia> x = [0.125, -0.25, 0.0625, -0.125, 0.375];

julia> MaximumDrawdown()(x)
0.3125
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`AverageDrawdown`](@ref)
  - [`DrawdownatRisk`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`WorstRealisation`](@ref)
  - [`RelativeMaximumDrawdown`](@ref)

# References

  - $(ref_dict[:cdar]) Section 3.1, Definition 3.2, Equation 3.3.
"""
@concrete struct MaximumDrawdown <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
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

Measures the Relative Maximum Drawdown, the deepest fall of the compounded wealth below its running peak, as a fraction of that peak.

The drawdown series is the one of [`relative_drawdown_vec`](@ref): the wealth compounds the returns, and its running peak starts at ``C_0 = 1``. [`MaximumDrawdown`](@ref) takes the deepest point of the uncompounded series. The compounded wealth is a product of linear functions of the weights, so the library states no JuMP formulation of the measure, and it is a [`HierarchicalRiskMeasure`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathrm{RMDD}(\\boldsymbol{x}) &= -\\min_{1 \\leq t \\leq T} rd_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RMDD}(\\boldsymbol{x})``: Relative maximum drawdown.
  - $(math_dict[:xret])
  - $(math_dict[:rdt]) [`relative_drawdown_vec`](@ref) states the series.
  - $(math_dict[:T])

The measure is the limit of [`RelativeConditionalDrawdownatRisk`](@ref) as `alpha` falls to zero. It is never above one when every return is above ``-1``, because the wealth then stays positive.

Section 3 of [cdar](@cite) defines the drawdown of the uncompounded cumulative return only. It names the relative drawdown and states that it lacks the properties of the absolute drawdown, but it defines no measure on it. The deepest point of the relative drawdown series is the library's extension of that work.

# Algorithm

 1. Compute the relative drawdown series `dd` with [`relative_drawdown_vec`](@ref).
 2. Return the negated minimum of `dd`.

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

On the path of [`AverageDrawdown`](@ref), the deepest point of the compounded wealth is 30.27 % below its peak, which is less than the 0.3125 of the uncompounded series.

```jldoctest
julia> x = [0.125, -0.25, 0.0625, -0.125, 0.375];

julia> RelativeMaximumDrawdown()(x)
0.302734375
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`MaximumDrawdown`](@ref)
  - [`RelativeAverageDrawdown`](@ref)
  - [`RelativeConditionalDrawdownatRisk`](@ref)

# References

  - $(ref_dict[:cdar]) Section 3, Equation 3.2.
"""
@concrete struct RelativeMaximumDrawdown <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
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

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::MaximumDrawdown) = NetReturnsInput()
risk_input_kind(::RelativeMaximumDrawdown) = NetReturnsInput()

export MaximumDrawdown, RelativeMaximumDrawdown
