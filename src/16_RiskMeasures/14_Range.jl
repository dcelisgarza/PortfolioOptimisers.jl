"""
$(DocStringExtensions.TYPEDEF)

Measures the Range, the largest return of the sample minus the smallest one.

It is the [`WorstRealisation`](@ref) plus the largest return. It is the limit of [`ValueatRiskRange`](@ref) and of [`ConditionalValueatRiskRange`](@ref) as `alpha` and `beta` fall to zero, and the [`OrderedWeightsArray`](@ref) with the weights of [`owa_rg`](@ref) gives the same value. A sample of one observation, or of equal returns, has a Range of zero.

# Mathematical definition

```math
\\begin{align}
\\mathrm{Range}(\\boldsymbol{x}) &= \\max_{1 \\leq t \\leq T} x_t - \\min_{1 \\leq t \\leq T} x_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{Range}(\\boldsymbol{x})``: Full return spread of the portfolio.
  - $(math_dict[:xret])
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

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

On the path of [`AverageDrawdown`](@ref), the largest return is 0.375 and the smallest is ``-0.25``.

```jldoctest
julia> Range()([0.125, -0.25, 0.0625, -0.125, 0.375])
0.625

julia> Range()([-0.1])
0.0
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`WorstRealisation`](@ref)
  - [`ValueatRiskRange`](@ref)
  - [`ConditionalValueatRiskRange`](@ref)
  - [`owa_rg`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.3.1, Equation 7.73. $(ref_dict[:no_original_source])
"""
@concrete struct Range <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    function Range(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function Range(; settings::RiskMeasureSettings = RiskMeasureSettings())::Range
    return Range(settings)
end
function (::Range)(x::VecNum)
    lb, ub = extrema(x)
    return ub - lb
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::Range) = NetReturnsInput()

export Range
