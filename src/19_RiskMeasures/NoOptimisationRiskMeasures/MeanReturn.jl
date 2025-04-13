struct MeanReturn{T1 <: Union{Nothing, <:AbstractWeights}} <: NoOptimisationRiskMeasure
    w::T1
end
function MeanReturn(; w::AbstractWeights = nothing)
    return MeanReturn{typeof(w)}(w)
end
function (r::MeanReturn)(x::AbstractVector)
    return isnothing(r.w) ? mean(x) : mean(x, r.w)
end
function risk_measure_factory(r::MeanReturn, args...)
    return r(; w = r.w)
end
function risk_measure_factory(r::MeanReturn, prior::EntropyPoolingResult, args...)
    w = risk_measure_nothing_vec_factory(r.w, prior.w)
    return r(; w = w)
end

export MeanReturn
