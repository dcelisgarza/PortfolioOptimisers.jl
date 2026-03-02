abstract type AbstractCrossValidationScorer <: AbstractEstimator end
abstract type PredictionScorer <: AbstractCrossValidationScorer end
abstract type PopulationScorer <: AbstractCrossValidationScorer end
const PredictionCrossValScorer = Union{<:PredictionScorer, <:Function}
const PopulationCrossValScorer = Union{<:PopulationScorer, <:Function}
struct NearestQuantilePrediction{T1, T2, T3} <: PredictionScorer
    r::T1
    q::T2
    kwargs::T3
    function NearestQuantilePrediction(r::AbstractBaseRiskMeasure, q::Real,
                                       kwargs::NamedTuple)
        return new{typeof(r), typeof(q), typeof(kwargs)}(r, q, kwargs)
    end
end
function NearestQuantilePrediction(; r::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
                                   q::Real = 0.5, kwargs::NamedTuple = (;))
    return NearestQuantilePrediction(r, q, kwargs)
end
function (s::NearestQuantilePrediction)(ppred::PopulationPredictionResult)
    return quantile_by_measure(ppred, s.r, s.q; s.kwargs...)
end
#! Start: Use these for scoring grid/random search cv
function _map_to_population_measures(::VecNum, rks::VecNum, f)
    #! Vector of predictions which are not pareto fronts. Get the mean of them all, because they are all the same point.
    return f(rks)
end
function _map_to_population_measures(::VecVecNum, rks::VecNum, f)
    #! This is a single pareto front, return the risk as each entry is a point in pareto front.
    return rks
end
function _map_to_population_measures(::VecVecNum, rks::VecVecNum, f)
    #! This is a vector of predictions each of which is a pareto front, so we need to get the mean of each point in the frontier across all predictions.
    return dropdims(f(reduce(hcat, rks); dims = 2); dims = 2)
end
function map_to_population_measures(r::AbstractBaseRiskMeasure,
                                    ppred::PopulationPredictionResult, f = mean)
    rks = expected_risk(r, ppred)
    return _map_to_population_measures(ppred.pred[1].rd[1].X, rks, f)
end
#! End: Use these for scoring grid/random search cv

export NearestQuantilePrediction
