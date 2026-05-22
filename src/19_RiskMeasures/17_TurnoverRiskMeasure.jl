"""
$(DocStringExtensions.TYPEDEF)

Represents the Turnover risk measure.

`TurnoverRiskMeasure` penalises portfolio rebalancing by measuring the ``L^1`` distance between the new portfolio weights and a reference weight vector. It controls trading costs and limits portfolio drift.

# Mathematical Definition

Let ``\\boldsymbol{w}`` be the new portfolio weights and ``\\boldsymbol{w}_0`` the reference (previous or target) weights:

```math
\\mathrm{Turnover}(\\boldsymbol{w}) = \\lVert \\boldsymbol{w}_0 - \\boldsymbol{w} \\rVert_1 = \\sum_{i=1}^{N} |w_{0,i} - w_i|\\,.
```

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TurnoverRiskMeasure(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::VecNum,
        fixed::Bool = false
    ) -> TurnoverRiskMeasure

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`.

# Functor

    (r::TurnoverRiskMeasure)(w::VecNum)

Computes the Turnover risk of a portfolio weight vector `w`.

## Arguments

  - `w::VecNum`: New portfolio weights vector.

# Examples

```jldoctest
julia> TurnoverRiskMeasure(; w = [0.5, 0.5])
TurnoverRiskMeasure
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ Vector{Float64}: [0.5, 0.5]
     fixed ┴ Bool: false
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`TrackingRiskMeasure`](@ref)
"""
@concrete struct TurnoverRiskMeasure <: RiskMeasure
    "$(field_dict[:settings_rm])"
    settings
    "$(field_dict[:w_ref])"
    w
    "$(field_dict[:fixed])"
    fixed
    function TurnoverRiskMeasure(settings::RiskMeasureSettings, w::VecNum, fixed::Bool)
        @argcheck(!isempty(w))
        return new{typeof(settings), typeof(w), typeof(fixed)}(settings, w, fixed)
    end
end
function TurnoverRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             w::VecNum, fixed::Bool = false)::TurnoverRiskMeasure
    return TurnoverRiskMeasure(settings, w, fixed)
end
function (r::TurnoverRiskMeasure)(w::VecNum)
    return LinearAlgebra.norm(r.w - w, 1)
end
function risk_measure_view(r::TurnoverRiskMeasure, i, args...)
    w = view(r.w, i)
    return TurnoverRiskMeasure(; settings = r.settings, w = w, fixed = r.fixed)
end
function needs_previous_weights(r::TurnoverRiskMeasure)
    return !r.fixed
end
function factory(r::TurnoverRiskMeasure, w::VecNum)::TurnoverRiskMeasure
    return if r.fixed
        r
    else
        TurnoverRiskMeasure(; settings = r.settings, w = w, fixed = r.fixed)
    end
end
function factory(r::TurnoverRiskMeasure, ::Any, ::Any, ::Any, w::Option{<:VecNum} = nothing,
                 args...; kwargs...)::TurnoverRiskMeasure
    return factory(r, w)
end

export TurnoverRiskMeasure
