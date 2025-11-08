for r in setdiff(traverse_concrete_subtypes(RiskMeasure), (UncertaintySetVariance,))
    eval(quote
             function no_bounds_risk_measure(r::$(r), args...)
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
function no_bounds_risk_measure(rs::VecRM, args...)
    return [no_bounds_risk_measure(r, args...) for r in rs]
end
for r in traverse_concrete_subtypes(RiskMeasure)
    eval(quote
             function no_bounds_no_risk_expr_risk_measure(r::$(r))
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
             function no_bounds_no_risk_expr_risk_measure(r::$(r))
                 return r
             end
         end)
end
function no_bounds_no_risk_expr_risk_measure(r::AbstractVector{<:AbstractBaseRiskMeasure})
    return no_bounds_no_risk_expr_risk_measure.(r)
end

export no_bounds_risk_measure, bounds_risk_measure
