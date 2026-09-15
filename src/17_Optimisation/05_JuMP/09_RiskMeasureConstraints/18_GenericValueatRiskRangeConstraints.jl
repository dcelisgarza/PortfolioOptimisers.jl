"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add `GenericValueatRiskRange` constraints to `model` by delegating to the loss- and
gain-side sub-constraints and summing the resulting expressions.

Calls [`set_range_risk_constraints!`](@ref), which reads the two tails from
[`range_tails`](@ref) — here `r.loss` and `r.gain` as given — and builds each through
[`set_risk_constraints!`](@ref). This is the same path every other range measure takes.

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
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::GenericValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :genvar_range_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
end
