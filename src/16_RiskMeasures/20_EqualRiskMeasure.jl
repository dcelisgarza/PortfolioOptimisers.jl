"""
$(DocStringExtensions.TYPEDEF)

Measures the same risk for every portfolio, so that a hierarchical optimiser splits every cluster evenly.

The risk is the reciprocal of the length of the weights vector. A hierarchical optimiser weights a cluster by the reciprocal of its risk, so a constant risk makes every split an even one. [`HierarchicalRiskParity`](@ref) gives each half of each bisection one half of the weight of the parent, so its weights are ``1/N`` only when ``N`` is a power of two. [`HierarchicalEqualRiskContribution`](@ref) splits each cluster evenly and gives the assets of a cluster equal weights. The risk does not change when the weights change, so every contribution of [`risk_contribution`](@ref) is zero.

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

Returns the reciprocal of the length of the weights vector `w`, in the element type of `w`. An integer element type gives a floating point number.

## Arguments

  - `w::VecNum`: Portfolio weights vector.

# Examples

```jldoctest
julia> EqualRisk()
EqualRisk
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0

julia> EqualRisk()([0.5, 0.3, 0.2, 0.0])
0.25

julia> EqualRisk()(Float32[0.6, 0.4])
0.5f0
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`number_effective_assets`](@ref): It equals ``N`` at the equal weights ``1/N``.
  - [`ScaleFreeRiskMeasures`](@ref)
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
    return one(eltype(w)) / length(w)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::EqualRisk) = WeightsInput()

export EqualRisk
