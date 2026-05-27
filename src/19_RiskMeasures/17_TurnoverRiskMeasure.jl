"""
$(DocStringExtensions.TYPEDEF)

Represents the Turnover risk measure.

`TurnoverRiskMeasure` penalises portfolio rebalancing by measuring the ``L^1`` distance between the new portfolio weights and a reference weight vector. It controls trading costs and limits portfolio drift.

# Mathematical definition

Let ``\\boldsymbol{w}`` be the new portfolio weights and ``\\boldsymbol{w}_0`` the reference (previous or target) weights:

```math
\\begin{align}
\\mathrm{Turnover}(\\boldsymbol{w}) &= \\lVert \\boldsymbol{w}_0 - \\boldsymbol{w} \\rVert_1 = \\sum_{i=1}^{N} |w_{0,i} - w_i|\\,.
\\end{align}
```

Where:

  - ``\\mathrm{Turnover}(\\boldsymbol{w})``: Portfolio turnover.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{w}_0``: Reference (previous or target) portfolio weights vector ``N \\times 1``.
  - $(math_dict[:N])

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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`TurnoverRiskMeasure`](@ref) `r` sliced to the asset indices `i`.

Used for cluster-based optimisation where only a subset of assets is considered.

# Related

  - [`TurnoverRiskMeasure`](@ref)
  - [`risk_measure_view`](@ref)
"""
function risk_measure_view(r::TurnoverRiskMeasure, i, args...)
    w = view(r.w, i)
    return TurnoverRiskMeasure(; settings = r.settings, w = w, fixed = r.fixed)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`TurnoverRiskMeasure`](@ref) `r` requires previous portfolio weights.

Returns `true` if `r.fixed` is `false` (i.e., the reference weights are updated each period).

# Related

  - [`TurnoverRiskMeasure`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(r::TurnoverRiskMeasure)
    return !r.fixed
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`TurnoverRiskMeasure`](@ref) updating the reference weights to `w`.

If `r.fixed` is `true`, returns `r` unchanged. Otherwise, constructs a new instance with `w` as the reference weight vector.

# Related

  - [`TurnoverRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::TurnoverRiskMeasure, w::VecNum)::TurnoverRiskMeasure
    return if r.fixed
        r
    else
        TurnoverRiskMeasure(; settings = r.settings, w = w, fixed = r.fixed)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`TurnoverRiskMeasure`](@ref) from a full optimisation context, forwarding the optional weight argument `w` to `factory(r, w)`.

Ignores prior result, solver, and uncertainty set arguments — only `w` is used.

# Related

  - [`TurnoverRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::TurnoverRiskMeasure, ::Any, ::Any, ::Any, w::Option{<:VecNum} = nothing,
                 args...; kwargs...)::TurnoverRiskMeasure
    return factory(r, w)
end

export TurnoverRiskMeasure
