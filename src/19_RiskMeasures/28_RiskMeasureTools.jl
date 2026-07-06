function no_bounds_risk_measure(r::RiskMeasure, ::Any = nothing)
    settings = r.settings
    return Accessors.@set r.settings = RiskMeasureSettings(; rke = settings.rke,
                                                           scale = settings.scale)
end
function no_bounds_no_risk_expr_risk_measure(r::RiskMeasure, ::Any = nothing)
    return Accessors.@set r.settings = RiskMeasureSettings(; rke = false, scale = 1)
end
function no_risk_expr_risk_measure(r::RiskMeasure)
    settings = r.settings
    return Accessors.@set r.settings = RiskMeasureSettings(; rke = false, ub = settings.ub,
                                                           scale = settings.scale)
end
function bounds_risk_measure(r::RiskMeasure, ub::Number)
    settings = r.settings
    return Accessors.@set r.settings = RiskMeasureSettings(; ub = ub, rke = settings.rke,
                                                           scale = settings.scale)
end
function no_bounds_risk_measure(r::HierarchicalRiskMeasure, ::Any = nothing)
    return r
end
function no_bounds_no_risk_expr_risk_measure(r::HierarchicalRiskMeasure, ::Any = nothing)
    return r
end
function no_risk_expr_risk_measure(r::HierarchicalRiskMeasure)
    return r
end
function bounds_risk_measure(r::HierarchicalRiskMeasure, ::Any = nothing)
    return r
end
"""
    no_bounds_risk_measure(r, flag = nothing)

Return a copy of risk measure `r` with its upper-bound constraint removed while
preserving all other settings. For hierarchical risk measures, returns the input
unchanged. For vectors of risk measures, applies element-wise.

# Arguments

  - `r`: Risk measure or vector of risk measures.
  - `flag`: Ignored; kept for dispatch compatibility.

# Returns

  - Risk measure without upper bounds.

# Related

  - [`bounds_risk_measure`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
"""
function no_bounds_risk_measure(rs::VecBaseRM, flag::Any = nothing)
    return no_bounds_risk_measure.(rs, flag)
end
"""
    no_bounds_no_risk_expr_risk_measure(r, flag = nothing)

Return a copy of risk measure `r` with both its upper-bound constraint and its risk
expression flag disabled. For hierarchical risk measures, returns the input unchanged.
For vectors of risk measures, applies element-wise.

# Arguments

  - `r`: Risk measure or vector of risk measures.
  - `flag`: Ignored; kept for dispatch compatibility.

# Returns

  - Risk measure without bounds or risk expression.

# Related

  - [`no_bounds_risk_measure`](@ref)
  - [`no_risk_expr_risk_measure`](@ref)
"""
function no_bounds_no_risk_expr_risk_measure(r::VecBaseRM, flag::Any = nothing)
    return no_bounds_no_risk_expr_risk_measure.(r, flag)
end
"""
    no_risk_expr_risk_measure(r)

Return a copy of risk measure `r` with its risk expression flag disabled while
preserving its upper-bound constraint. For hierarchical risk measures, returns the
input unchanged. For vectors of risk measures, applies element-wise.

# Arguments

  - `r`: Risk measure or vector of risk measures.

# Returns

  - Risk measure without risk expression flag.

# Related

  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
  - [`no_bounds_risk_measure`](@ref)
"""
function no_risk_expr_risk_measure(r::VecBaseRM)
    return no_risk_expr_risk_measure.(r)
end
"""
    bounds_risk_measure(r, ubs)

Return a copy of risk measure `r` (or each element of vector `r`) with its upper-bound
set to `ubs` (or the corresponding element of `ubs`). Hierarchical risk measures are
returned unchanged.

# Arguments

  - `r`: Risk measure or vector of risk measures.
  - `ubs`: Upper-bound value or vector of upper-bound values.

# Returns

  - Risk measure or vector of risk measures with updated upper bounds.

# Related

  - [`no_bounds_risk_measure`](@ref)
"""
function bounds_risk_measure(r::VecBaseRM, ubs::VecNum)
    return bounds_risk_measure.(r, ubs)
end

export no_bounds_risk_measure, no_bounds_no_risk_expr_risk_measure,
       no_risk_expr_risk_measure, bounds_risk_measure
