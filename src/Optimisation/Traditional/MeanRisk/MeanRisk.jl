struct MeanRisk{T1 <: Union{<:RiskMeasure}} <: TraditionalOptimisationType
    r::T1
end

export MeanRisk
