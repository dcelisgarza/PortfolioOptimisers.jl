function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...) end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...) end
function set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...) end