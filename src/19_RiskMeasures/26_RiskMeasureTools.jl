for r in setdiff(traverse_concrete_subtypes(RiskMeasure), (UncertaintySetVariance,))
    eval(quote
             function no_bounds_risk_measure(r::$(r), ::Any = nothing)
                 pnames = Tuple(setdiff(propertynames(r), (:settings,)))
                 settings = r.settings
                 settings = RiskMeasureSettings(; rke = settings.rke,
                                                scale = settings.scale)
                 return if isempty(pnames)
                     $(r)(; settings = settings)
                 else
                     $(r)(; settings = settings,
                          NamedTuple{pnames}(getproperty.(r, pnames))...)
                 end
             end
         end)
end
for r in setdiff(traverse_concrete_subtypes(RiskMeasure), (UncertaintySetVariance,))
    eval(quote
             function no_bounds_no_risk_expr_risk_measure(r::$(r), ::Any = nothing)
                 pnames = Tuple(setdiff(propertynames(r), (:settings,)))
                 settings = r.settings
                 settings = RiskMeasureSettings(; rke = false, scale = 1)
                 return if isempty(pnames)
                     $(r)(; settings = settings)
                 else
                     $(r)(; settings = settings,
                          NamedTuple{pnames}(getproperty.(r, pnames))...)
                 end
             end
             function no_risk_expr_risk_measure(r::$(r))
                 pnames = Tuple(setdiff(propertynames(r), (:settings,)))
                 settings = r.settings
                 settings = RiskMeasureSettings(; rke = false, ub = settings.ub,
                                                scale = settings.scale)
                 return if isempty(pnames)
                     $(r)(; settings = settings)
                 else
                     $(r)(; settings = settings,
                          NamedTuple{pnames}(getproperty.(r, pnames))...)
                 end
             end
         end)
end
for r in traverse_concrete_subtypes(RiskMeasure)
    eval(quote
             function bounds_risk_measure(r::$(r), ub::Number)
                 pnames = Tuple(setdiff(propertynames(r), (:settings,)))
                 settings = r.settings
                 settings = RiskMeasureSettings(; ub = ub, rke = settings.rke,
                                                scale = settings.scale)
                 return if isempty(pnames)
                     $(r)(; settings = settings)
                 else
                     $(r)(; settings = settings,
                          NamedTuple{pnames}(getproperty.(r, pnames))...)
                 end
             end
         end)
end
for r in traverse_concrete_subtypes(HierarchicalRiskMeasure)
    eval(quote
             function no_bounds_risk_measure(r::$(r), ::Any = nothing)
                 return r
             end
             function no_bounds_no_risk_expr_risk_measure(r::$(r), ::Any = nothing)
                 return r
             end
             function no_risk_expr_risk_measure(r::$(r))
                 return r
             end
             function bounds_risk_measure(r::$(r), ::Any = nothing)
                 return r
             end
         end)
end
"""
    no_bounds_risk_measure(r, flag = nothing)

Return a copy of risk measure `r` with its upper-bound constraint removed while
preserving all other settings. Hierarchical risk measures are returned unchanged.
For vectors of risk measures, applies element-wise.

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
expression flag disabled. Hierarchical risk measures are returned unchanged. For vectors
of risk measures, applies element-wise.

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
preserving its upper-bound constraint. Hierarchical risk measures are returned unchanged.
For vectors of risk measures, applies element-wise.

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
