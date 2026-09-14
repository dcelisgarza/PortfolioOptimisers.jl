"""
$(DocStringExtensions.TYPEDEF)

Represents the Equal Risk Measure for hierarchical portfolio optimisation.

`EqualRisk` reports the same risk for every portfolio: the reciprocal of the length of the weight vector it is given. A hierarchical optimiser weights a cluster by the reciprocal of its risk, so a constant risk makes every split an even one.

# Mathematical definition

For a portfolio of ``N`` assets with weights ``\\boldsymbol{w} \\in \\mathbb{R}^N``:

```math
\\begin{align}
\\mathrm{EqR}(\\boldsymbol{w}) &= \\frac{1}{N}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{EqR}(\\boldsymbol{w})``: Equal risk of the portfolio.
  - $(math_dict[:w_port])
  - $(math_dict[:N])

``N`` is the length of the weight vector the functor receives, not the size of a cluster. A hierarchical optimiser passes the full-length weight vector with zeros outside the cluster, so every cluster reports the same ``1/N`` and every bisection splits the weight evenly.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EqualRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings()
    ) -> EqualRisk

Keywords correspond to the struct's fields.

# Functor

    (r::EqualRisk)(w::VecNum)

Returns the reciprocal of the length of the weight vector `w`.

## Arguments

  - `w::VecNum`: Portfolio weights vector.

# Examples

```jldoctest
julia> EqualRisk()
EqualRisk
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
```

# Related

  - [`number_effective_assets`](@ref)
  - [`set_weight_norm_2_constraints!`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`RiskRatio`](@ref)
"""
@concrete struct EqualRisk <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    function EqualRisk(settings::HierarchicalRiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function EqualRisk(;
                   settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())::EqualRisk
    return EqualRisk(settings)
end
function (::EqualRisk)(w::VecNum)
    return inv(length(w))
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::EqualRisk) = WeightsInput()

export EqualRisk
