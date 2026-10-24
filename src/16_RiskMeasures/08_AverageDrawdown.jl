"""
$(DocStringExtensions.TYPEDEF)

Measures the Average Drawdown, the mean depth of the uncompounded drawdown over the sample.

The drawdown series is the one of [`absolute_drawdown_vec`](@ref): the cumulative return is a sum of the returns, and its running peak starts at ``c_0 = 0``. [`RelativeAverageDrawdown`](@ref) takes the same mean of the compounded drawdown series.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ADD}(\\boldsymbol{x}) &= -\\frac{1}{T} \\sum_{t=1}^{T} d_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ADD}(\\boldsymbol{x})``: Average drawdown.
  - $(math_dict[:xret])
  - $(math_dict[:dtdd]) [`absolute_drawdown_vec`](@ref) states the series.
  - $(math_dict[:T])

With observation weights, the measure takes the weighted mean:

```math
\\begin{align}
\\mathrm{ADD}(\\boldsymbol{x}) &= -\\frac{1}{W_{T}} \\sum_{t=1}^{T} w_{t} d_t\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_obs])
  - $(math_dict[:W_T_total])

The drawdown series reads the full return path, so an observation with zero weight still moves every later drawdown. A path that never falls below its running peak has an Average Drawdown of zero. The measure is the limit of [`ConditionalDrawdownatRisk`](@ref) as `alpha` rises to one, because the tail then holds the whole sample.

[cdar](@cite) defines the measure without weights. The weighted mean is the library's extension of that work.

# Algorithm

 1. Compute the drawdown series with [`absolute_drawdown_vec`](@ref).
 2. Resolve the observation weights with [`checked_observation_weights`](@ref).
 3. Return [`average_drawdown`](@ref) of the series and the weights.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AverageDrawdown(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing
    ) -> AverageDrawdown

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow_rm])

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

The drawdown series of the path below is `[0, -0.25, -0.1875, -0.3125, 0]`. The last return lifts the cumulative return to a new peak. A weight of four on that period moves the mean towards zero.

```jldoctest
julia> x = [0.125, -0.25, 0.0625, -0.125, 0.375];

julia> AverageDrawdown()(x)
0.15

julia> AverageDrawdown(; w = pweights([1.0, 1.0, 1.0, 1.0, 4.0]))(x)
0.09375
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`MaximumDrawdown`](@ref)
  - [`UlcerIndex`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`RelativeAverageDrawdown`](@ref)
  - [`average_drawdown`](@ref)

# References

  - $(ref_dict[:cdar]) Section 3.1, Definition 3.2, Equation 3.4.
"""
@propagatable @concrete struct AverageDrawdown <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:oow])
    """
    @pprop w
    function AverageDrawdown(settings::RiskMeasureSettings, w::Option{<:ObsWeights})
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function AverageDrawdown(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Option{<:ObsWeights} = nothing)::AverageDrawdown
    return AverageDrawdown(settings, w)
end
"""
    average_drawdown(dd::VecNum, ::Nothing) -> Number
    average_drawdown(dd::VecNum, w::VecNum) -> Number

Aggregate a drawdown series into its mean drawdown, returned as a nonnegative loss.

[`AverageDrawdown`](@ref) and [`RelativeAverageDrawdown`](@ref) both call this kernel. They differ only in the series that they give it: [`absolute_drawdown_vec`](@ref) for the first, and [`relative_drawdown_vec`](@ref) for the second. [`ulcer_index`](@ref) is the kernel of the root-mean-square drawdown.

The type of the second argument selects the method. Callers resolve the observation weights with [`checked_observation_weights`](@ref) first, so the kernel sees a weight vector or `nothing`, and never a [`DynamicAbstractWeights`](@ref). The weighted method takes any weight vector, a plain one too.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ADD}(\\boldsymbol{\\delta}) &= -\\frac{1}{T} \\sum_{t=1}^{T} \\delta_t\\,, \\\\
\\mathrm{ADD}_{w}(\\boldsymbol{\\delta}) &= -\\frac{1}{W_{T}} \\sum_{t=1}^{T} w_{t} \\delta_t\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\delta}``: Drawdown series `dd` ``T \\times 1``, absolute or relative, every entry ``\\leq 0``.
  - ``\\mathrm{ADD}(\\boldsymbol{\\delta})``, ``\\mathrm{ADD}_{w}(\\boldsymbol{\\delta})``: The mean drawdown of the methods without and with weights.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:W_T_total])
  - $(math_dict[:T])

# Arguments

  - `dd::VecNum`: Drawdown series, every entry ``\\leq 0``. The function does not write it.
  - `w`: Resolved observation weights, or `nothing` for the mean without weights.

# Returns

  - `Number`: Mean drawdown, returned as a nonnegative loss.

# Related

  - [`AverageDrawdown`](@ref)
  - [`RelativeAverageDrawdown`](@ref)
  - [`ulcer_index`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`checked_observation_weights`](@ref)
"""
function average_drawdown(dd::VecNum, ::Nothing)
    return -Statistics.mean(dd)
end
function average_drawdown(dd::VecNum, w::VecNum)
    return -LinearAlgebra.dot(w, dd) / sum(w)
end
function (r::AverageDrawdown)(x::VecNum)
    return average_drawdown(absolute_drawdown_vec(x), checked_observation_weights(r.w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Relative Average Drawdown, the mean depth of the compounded drawdown over the sample.

The drawdown series is the one of [`relative_drawdown_vec`](@ref): the wealth compounds the returns, and its running peak starts at ``C_0 = 1``. Each drawdown is thus a fraction of the peak wealth. [`AverageDrawdown`](@ref) takes the same mean of the uncompounded series. The compounded wealth is a product of linear functions of the weights, so the library states no JuMP formulation of the measure, and it is a [`HierarchicalRiskMeasure`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathrm{RADD}(\\boldsymbol{x}) &= -\\frac{1}{T} \\sum_{t=1}^{T} rd_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RADD}(\\boldsymbol{x})``: Relative average drawdown.
  - $(math_dict[:xret])
  - $(math_dict[:rdt]) [`relative_drawdown_vec`](@ref) states the series.
  - $(math_dict[:T])

With observation weights, the measure takes the weighted mean, as [`AverageDrawdown`](@ref) states, with ``rd_t`` in place of ``d_t``.

Section 3 of [cdar](@cite) defines the drawdown of the uncompounded cumulative return only. It names the relative drawdown and states that it lacks the properties of the absolute drawdown, but it defines no measure on it. The mean of the relative drawdown series is the library's extension of that work.

# Algorithm

 1. Compute the relative drawdown series with [`relative_drawdown_vec`](@ref).
 2. Resolve the observation weights with [`checked_observation_weights`](@ref).
 3. Return [`average_drawdown`](@ref) of the series and the weights.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeAverageDrawdown(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeAverageDrawdown

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow_rm])

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

On the path of [`AverageDrawdown`](@ref), the relative drawdown series is `[0, -0.25, -0.203125, -0.302734375, -0.041259765625]`. The compounded wealth does not reach a new peak at the last return, so the mean is larger than the 0.15 of the uncompounded series.

```jldoctest
julia> x = [0.125, -0.25, 0.0625, -0.125, 0.375];

julia> RelativeAverageDrawdown()(x)
0.159423828125
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`AverageDrawdown`](@ref)
  - [`RelativeMaximumDrawdown`](@ref)
  - [`RelativeConditionalDrawdownatRisk`](@ref)
  - [`average_drawdown`](@ref)

# References

  - $(ref_dict[:cdar]) Section 3, Equation 3.2.
"""
@propagatable @concrete struct RelativeAverageDrawdown <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:oow])
    """
    @pprop w
    function RelativeAverageDrawdown(settings::HierarchicalRiskMeasureSettings,
                                     w::Option{<:ObsWeights})
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function RelativeAverageDrawdown(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                 w::Option{<:ObsWeights} = nothing)::RelativeAverageDrawdown
    return RelativeAverageDrawdown(settings, w)
end
function (r::RelativeAverageDrawdown)(x::VecNum)
    return average_drawdown(relative_drawdown_vec(x), checked_observation_weights(r.w, x))
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::AverageDrawdown) = NetReturnsInput()
risk_input_kind(::RelativeAverageDrawdown) = NetReturnsInput()

export AverageDrawdown, RelativeAverageDrawdown
