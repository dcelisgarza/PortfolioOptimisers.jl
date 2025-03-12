struct PortfolioMean{T1 <: Union{Nothing, <:AbstractWeights}} <: NoOptimisationRiskMeasure
    w::T1
end
function PortfolioMean(; w::AbstractWeights = nothing)
    return PortfolioMean{typeof(w)}(w)
end
function (r::PortfolioMean)(x::AbstractVector)
    return isnothing(r.w) ? mean(x) : mean(x, r.w)
end

export PortfolioMean
