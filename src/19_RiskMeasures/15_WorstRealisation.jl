"""
$(DocStringExtensions.TYPEDEF)

Represents the Worst Realisation risk measure.

`WorstRealisation` returns the maximum loss (i.e., the negative minimum return) over all observed scenarios. It is the most conservative risk measure, capturing the single worst outcome in the sample.

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

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Range`](@ref)
  - [`ValueatRisk`](@ref)
"""
@concrete struct WorstRealisation <: RiskMeasure
    "$(field_dict[:settings_rm])"
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

export WorstRealisation
