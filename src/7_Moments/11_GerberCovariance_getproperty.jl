function getproperty(ce::Union{<:Gerber0Covariance, <:Gerber1Covariance,
                               <:Gerber2Covariance}, sym::Symbol)
    if sym ∈ (:ve, :pdm, :threshold)
        return getfield(ce.ce, sym)
    else
        return getfield(ce, sym)
    end
end
function getproperty(ce::Union{<:Gerber0NormalisedCovariance, <:Gerber1NormalisedCovariance,
                               <:Gerber2NormalisedCovariance}, sym::Symbol)
    if sym ∈ (:me, :ve, :pdm, :threshold)
        return getfield(ce.ce, sym)
    else
        return getfield(ce, sym)
    end
end
