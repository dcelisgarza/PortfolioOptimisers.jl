struct SD{T1 <: RiskMeasureSettings, T2 <: Union{Nothing, <:AbstractMatrix}} <:
       SigmaRiskMeasure
    settings::T1
    sigma::T2
end
function SD(; settings::RiskMeasureSettings = RiskMeasureSettings(),
            sigma::Union{Nothing, <:AbstractMatrix} = nothing)
    if !isnothing(sigma) && !isempty(sigma)
        @smart_assert(size(sigma, 1) == size(sigma, 2))
    end
    return SD{typeof(settings), typeof(sigma)}(settings, sigma)
end
function (sd::SD)(w::AbstractVector)
    return sqrt(dot(w, sd.sigma, w))
end

export SD
