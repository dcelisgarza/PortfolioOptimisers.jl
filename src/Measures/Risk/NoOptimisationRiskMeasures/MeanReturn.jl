struct MeanReturn{T1 <: Union{Nothing, <:AbstractWeights}} <: NoOptimisationRiskMeasure
    w::T1
end
function MeanReturn(; w::AbstractWeights = nothing)
    return MeanReturn{typeof(w)}(w)
end
function (r::MeanReturn)(x::AbstractVector)
    return isnothing(r.w) ? mean(x) : mean(x, r.w)
end

export MeanReturn
