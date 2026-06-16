"""
$(DocStringExtensions.TYPEDEF)

Represents the Average Drawdown risk measure.

`AverageDrawdown` computes the mean of the absolute drawdown series of the portfolio returns. It provides a measure of the average magnitude of drawdowns over the sample period.

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

The Average Drawdown is:

```math
\\begin{align}
\\mathrm{ADD}(\\boldsymbol{x}) &= -\\frac{1}{T} \\sum_{t=1}^{T} d_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ADD}(\\boldsymbol{x})``: Average drawdown.
  - $(math_dict[:T])
  - $(math_dict[:dtdd])

For observation-weighted samples, the weighted mean is used instead.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AverageDrawdown(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing
    ) -> AverageDrawdown

Keywords correspond to the struct's fields.

## Validation

  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::AverageDrawdown)(x::VecNum)

Computes the Average Drawdown of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> AverageDrawdown()
AverageDrawdown
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`MaximumDrawdown`](@ref)
  - [`UlcerIndex`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`RelativeAverageDrawdown`](@ref)
"""
@propagatable @concrete struct AverageDrawdown <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w_rm])
    """
    @pprop w
    function AverageDrawdown(settings::RiskMeasureSettings, w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function AverageDrawdown(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Option{<:ObsWeights} = nothing)::AverageDrawdown
    return AverageDrawdown(settings, w)
end
function (r::AverageDrawdown)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return -(isnothing(r.w) ? Statistics.mean(dd) : Statistics.mean(dd, r.w))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Average Drawdown risk measure for hierarchical optimisation.

`RelativeAverageDrawdown` computes the mean of the relative (compounded) drawdown series of the portfolio returns.

# Mathematical definition

Define the compounded wealth process and relative drawdown series:

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

The Relative Average Drawdown is:

```math
\\begin{align}
\\mathrm{RADD}(\\boldsymbol{x}) &= -\\frac{1}{T} \\sum_{t=1}^{T} rd_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RADD}(\\boldsymbol{x})``: Relative average drawdown.
  - $(math_dict[:T])
  - $(math_dict[:rdt])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeAverageDrawdown(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeAverageDrawdown

Keywords correspond to the struct's fields.

## Validation

  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativeAverageDrawdown)(x::VecNum)

Computes the Relative Average Drawdown of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeAverageDrawdown()
RelativeAverageDrawdown
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
         w ┴ nothing
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`AverageDrawdown`](@ref)
  - [`RelativeMaximumDrawdown`](@ref)
"""
@propagatable @concrete struct RelativeAverageDrawdown <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w_rm])
    """
    @pprop w
    function RelativeAverageDrawdown(settings::HierarchicalRiskMeasureSettings,
                                     w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function RelativeAverageDrawdown(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                 w::Option{<:ObsWeights} = nothing)::RelativeAverageDrawdown
    return RelativeAverageDrawdown(settings, w)
end
function (r::RelativeAverageDrawdown)(x::VecNum)
    dd = relative_drawdown_vec(x)
    w = get_observation_weights(r.w, x)
    return -(isnothing(r.w) ? Statistics.mean(dd) : Statistics.mean(dd, w))
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::AverageDrawdown) = NetReturnsInput()
risk_input_kind(::RelativeAverageDrawdown) = NetReturnsInput()

export AverageDrawdown, RelativeAverageDrawdown
