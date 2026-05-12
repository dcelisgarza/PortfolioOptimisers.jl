abstract type RegimeAdjustedTarget <: AbstractAlgorithm end
function min_active_assets(::RegimeAdjustedTarget)
    return 1
end
struct MahalanobisTarget <: RegimeAdjustedTarget end
function min_active_assets(::MahalanobisTarget)
    return 2
end
struct DiagonalTarget <: RegimeAdjustedTarget end
@concrete struct PortfolioTarget <: RegimeAdjustedTarget
    w
    function PortfolioTarget(w::Option{<:EstValType})
        if isa(w, AbstractVector)
            @argcheck(!isempty(w))
        end
    end
end

@concrete struct RegimeAdjustedExpWeightedCovariance <: AbstractCovarianceEstimator
    decay
    cor_decay
    hac_lags
    regime_method
    regime_decay
    regime_target
end
