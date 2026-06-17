"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add `GenericValueatRiskRange` constraints to `model` by delegating to the loss- and
gain-side sub-constraints and summing the resulting expressions.

Calls [`set_risk_constraints!`](@ref) twice — once for `r.loss` with `loss = true` (applied
to the net portfolio returns) and once for `r.gain` with `loss = false` (applied to the
negated net portfolio returns). The two expressions are summed into the composite range
expression, which is then registered with [`set_risk_bounds_and_expression!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::GenericValueatRiskRange`: The generic Value-at-Risk range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `genvar_range_risk`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`GenericValueatRiskRange`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::GenericValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:genvar_range_, i)
    loss = set_risk_constraints!(model, Symbol(:loss_, i), r.loss, opt, pr, args...;
                                 loss = true, prefix = prefix, kwargs...)
    gain = set_risk_constraints!(model, Symbol(:gain_, i), r.gain, opt, pr, args...;
                                 loss = false, prefix = prefix, kwargs...)
    genvar_range_risk = model[key] = JuMP.@expression(model, loss + gain)
    set_risk_bounds_and_expression!(model, opt, genvar_range_risk, r.settings, key)
    return genvar_range_risk
end
