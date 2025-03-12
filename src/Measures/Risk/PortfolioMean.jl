struct PortfolioMean{T1 <: Union{Nothing, <:AbstractWeights}} <: NoOptimisationRiskMeasure
    w::T1
end
function PortfolioMean(; w::AbstractWeights = nothing)
    return PortfolioMean{typeof(w)}(w)
end

export PortfolioMean
