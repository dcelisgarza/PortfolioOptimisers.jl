for r ∈ setdiff(traverse_subtypes(RiskMeasure), (UncertaintySetVariance,))
    eval(quote
             function no_bounds_risk_measure(r::$(r), args...)
                 pnames = setdiff(propertynames(r), (:settings,))
                 settings = r.settings
                 settings = RiskMeasureSettings(; rke = settings.rke,
                                                scale = settings.scale)
                 return if isempty(pnames)
                     $(r)(settings)
                 else
                     $(r)(settings, getproperty.(Ref(r), pnames)...)
                 end
             end
         end)
end
function no_bounds_risk_measure(r::AbstractVector{<:RiskMeasure}, args...)
    return no_bounds_risk_measure.(r, Ref(args)...)
end
function no_bounds_first_risk_measure(r::AbstractVector{<:RiskMeasure}, args...)
    return no_bounds_risk_measure(r[1], args...)
end
function no_bounds_first_risk_measure(r::RiskMeasure, args...)
    return no_bounds_risk_measure(r, args...)
end
for r ∈ traverse_subtypes(RiskMeasure)
    eval(quote
             function no_bounds_no_risk_expr_risk_measure(r::$(r))
                 pnames = setdiff(propertynames(r), (:settings,))
                 settings = r.settings
                 settings = RiskMeasureSettings(; rke = false, scale = settings.scale)
                 return if isempty(pnames)
                     $(r)(settings)
                 else
                     $(r)(settings, getproperty.(Ref(r), pnames)...)
                 end
             end
             function bounds_risk_measure_expression(r::$(r), ub::Real)
                 pnames = setdiff(propertynames(r), (:settings,))
                 settings = r.settings
                 settings = RiskMeasureSettings(; ub = ub, rke = true,
                                                scale = settings.scale)
                 return if isempty(pnames)
                     $(r)(settings)
                 else
                     $(r)(settings, getproperty.(Ref(r), pnames)...)
                 end
             end
         end)
end
function bounds_first_risk_measure(r::AbstractVector{<:RiskMeasure}, ub::Real)
    r[1] = bounds_risk_measure_expression(r[1], ub)
    return r
end
function bounds_first_risk_measure(r::RiskMeasure, ub::Real)
    return bounds_risk_measure_expression(r, ub)
end
for r ∈ traverse_subtypes(HierarchicalRiskMeasure)
    eval(quote
             function no_bounds_no_risk_expr_risk_measure(r::$(r))
                 return r
             end
         end)
end
function no_bounds_no_risk_expr_risk_measure(r::AbstractVector{<:AbstractBaseRiskMeasure})
    return no_bounds_no_risk_expr_risk_measure.(r)
end