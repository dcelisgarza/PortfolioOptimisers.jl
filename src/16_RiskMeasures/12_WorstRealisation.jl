"""
$(DocStringExtensions.TYPEDEF)

Measures the Worst Realisation, the largest loss of the sample, which is the negated minimum return.

It is the limit of [`ValueatRisk`](@ref) and of [`ConditionalValueatRisk`](@ref) as `alpha` falls to zero, and the [`OrderedWeightsArray`](@ref) with the weights of [`owa_wr`](@ref) gives the same value. When every return is positive, the measure is negative: the worst outcome is then the smallest gain. [`MaximumDrawdown`](@ref) is the Worst Realisation of the drawdown series.

# Mathematical definition

```math
\\begin{align}
\\mathrm{WR}(\\boldsymbol{x}) &= -\\min_{1 \\leq t \\leq T} x_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{WR}(\\boldsymbol{x})``: Worst realisation of portfolio returns.
  - $(math_dict[:xret])
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WorstRealisation(;
        settings::RiskMeasureSettings = RiskMeasureSettings()
    ) -> WorstRealisation

Keywords correspond to the struct's fields.

# Functor

    (r::WorstRealisation)(x::VecNum)

Computes the Worst Realisation of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> WorstRealisation()
WorstRealisation
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
```

On the path of [`AverageDrawdown`](@ref), the worst return is ``-0.25``.

```jldoctest
julia> WorstRealisation()([0.125, -0.25, 0.0625, -0.125, 0.375])
0.25

julia> WorstRealisation()([0.1, 0.2])
-0.1
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Range`](@ref)
  - [`ValueatRisk`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`MaximumDrawdown`](@ref)
  - [`owa_wr`](@ref)

# References

  - $(ref_dict[:minimax])
  - $(ref_dict[:cajas2025]) Section 7.2.2.8, Equation 7.71. $(ref_dict[:no_original_source])
"""
@concrete struct WorstRealisation <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    function WorstRealisation(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function WorstRealisation(;
                          settings::RiskMeasureSettings = RiskMeasureSettings())::WorstRealisation
    return WorstRealisation(settings)
end
function (::WorstRealisation)(x::VecNum)
    return -minimum(x)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::WorstRealisation) = NetReturnsInput()

export WorstRealisation
