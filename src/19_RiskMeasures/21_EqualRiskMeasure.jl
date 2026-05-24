"""
$(DocStringExtensions.TYPEDEF)

Represents the Equal Risk Measure for hierarchical portfolio optimisation.

`EqualRiskMeasure` assigns an equal risk contribution to each asset by returning the reciprocal of the number of assets. It is used in equal-risk-contribution (ERC) strategies.

# Mathematical definition

For a portfolio of ``N`` assets with weights ``\\boldsymbol{w} \\in \\mathbb{R}^N``:

```math
\\mathrm{ERC}(\\boldsymbol{w}) = \\frac{1}{N}\\,.
```

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EqualRiskMeasure(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings()
    ) -> EqualRiskMeasure

Keywords correspond to the struct's fields.

# Functor

    (r::EqualRiskMeasure)(w::VecNum)

Returns the equal risk contribution for a weight vector `w`.

## Arguments

  - `w::VecNum`: Portfolio weights vector.

# Examples

```jldoctest
julia> EqualRiskMeasure()
EqualRiskMeasure
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
```

# Related

  - [`number_effective_assets`](@ref)
  - [`set_number_effective_assets!`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`RiskRatioRiskMeasure`](@ref)
"""
@concrete struct EqualRiskMeasure <: HierarchicalRiskMeasure
    "$(field_dict[:settings_rm])"
    settings
    function EqualRiskMeasure(settings::HierarchicalRiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function EqualRiskMeasure(;
                          settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())::EqualRiskMeasure
    return EqualRiskMeasure(settings)
end
function (::EqualRiskMeasure)(w::VecNum)
    return inv(length(w))
end

export EqualRiskMeasure
