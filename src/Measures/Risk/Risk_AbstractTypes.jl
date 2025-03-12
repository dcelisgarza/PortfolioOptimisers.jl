abstract type AbstractRiskMeasure end
abstract type OptimisationRiskMeasure <: AbstractRiskMeasure end
abstract type NoOptimisationRiskMeasure <: AbstractRiskMeasure end
abstract type RiskMeasure <: OptimisationRiskMeasure end
abstract type HCRiskMeasure <: OptimisationRiskMeasure end
