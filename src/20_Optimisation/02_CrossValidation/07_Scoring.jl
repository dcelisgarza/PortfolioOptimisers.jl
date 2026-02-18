abstract type AbstractCrossValidationScorer <: AbstractEstimator end
const CrossValScorer = Union{<:AbstractCrossValidationScorer, <:Function}
