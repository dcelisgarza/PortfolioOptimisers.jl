# https://github.com/oxfordcontrol/Clarabel.jl/blob/4915b83e0d900d978681d5e8f3a3a5b8e18086f0/warmstart_test/portfolioOpt/higherorderRiskMeansure.jl#L23
struct PowerValueatRisk <: RiskMeasure end
struct PowerValueatRiskRange <: RiskMeasure end
struct PowerDrawdownatRisk <: RiskMeasure end
struct RelativePowerDrawdownatRisk <: HierarchicalRiskMeasure end
