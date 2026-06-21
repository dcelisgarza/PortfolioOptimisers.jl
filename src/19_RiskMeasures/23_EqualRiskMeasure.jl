"""
$(DocStringExtensions.TYPEDEF)

Represents the Equal Risk Measure for hierarchical portfolio optimisation.

`EqualRisk` assigns an equal risk contribution to each asset by returning the reciprocal of the number of assets. It is used in equal-risk-contribution (ERC) strategies.

# Mathematical definition

For a portfolio of ``N`` assets with weights ``\\boldsymbol{w} \\in \\mathbb{R}^N``:

```math
\\begin{align}
\\mathrm{ERC}(\\boldsymbol{w}) &= \\frac{1}{N}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ERC}(\\boldsymbol{w})``: Equal risk contribution per asset.
  - $(math_dict[:w_port])
  - $(math_dict[:N])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EqualRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings()
    ) -> EqualRisk

Keywords correspond to the struct's fields.

# Functor

    (r::EqualRisk)(w::VecNum)

Returns the equal risk contribution for a weight vector `w`.

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
  - [`set_number_effective_assets!`](@ref)
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
