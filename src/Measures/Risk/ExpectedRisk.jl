function expected_risk(r::Union{WorstRealisation, ValueatRisk, ValueatRiskRange,
                                ConditionalValueatRisk,
                                DistributionallyRobustConditionalValueatRisk,
                                EntropicValueatRisk, EntropicValueatRiskRange,
                                RelativisticValueatRisk, RelativisticValueatRiskRange,
                                DrawdownatRisk, MaximumDrawdown, AverageDrawdown,
                                ConditionalDrawdownatRisk, UlcerIndex,
                                EntropicDrawdownatRisk, RelativisticDrawdownatRisk,
                                RelativeDrawdownatRisk, RelativeMaximumDrawdown,
                                RelativeAverageDrawdown, RelativeConditionalDrawdownatRisk,
                                RelativeUlcerIndex, RelativeEntropicDrawdownatRisk,
                                RelativeRelativisticDrawdownatRisk, GiniMeanDifference,
                                Range, ConditionalValueatRiskRange, TailGini, TailGiniRange,
                                OrderedWeightsArray, FourthCentralMoment, Skewness,
                                SemiSkewness, Kurtosis, SemiKurtosis}, w::AbstractVector,
                       X::AbstractMatrix; fees::Fees = Fees(), kwargs...)
    return r(calc_net_returns(w, X, fees))
end
function expected_risk(r::Union{MeanAbsoluteDeviation, SemiStandardDeviation,
                                FirstLowerPartialMoment, ThirdLowerPartialMoment,
                                FourthLowerPartialMoment, TrackingRiskMeasure,
                                SquareRootKurtosis, SquareRootSemiKurtosis, SemiVariance,
                                BrownianDistanceVariance}, w::AbstractVector,
                       X::AbstractMatrix; fees::Fees = Fees(), kwargs...)
    return r(w, X, fees)
end
function expected_risk(r::Union{StandardDeviation, NegativeSkewness, NegativeSemiSkewness,
                                TurnoverRiskMeasure, Variance, UncertaintySetVariance,
                                NegativeQuadraticSkewness, NegativeQuadraticSemiSkewness},
                       w::AbstractVector, args...; kwargs...)
    return r(w)
end
function _expected_risk(::SumScalariser, rs::AbstractVector{<:RiskMeasure},
                        w::AbstractVector, X::AbstractMatrix, fees::Fees = Fees())
    rk = zero(eltype(X))
    for r ∈ rs
        rk += r.settings.scale * expected_risk(r, w, X; fees = fees)
    end
    return rk
end
function _expected_risk(::MaxScalariser, rs::AbstractVector{<:RiskMeasure},
                        w::AbstractVector, X::AbstractMatrix, fees::Fees = Fees())
    rk = zero(eltype(X))
    for r ∈ rs
        ri = r.settings.scale * expected_risk(r, w, X; fees = fees)
        if ri > rk
            rk = ri
        end
    end
    return rk
end
function _expected_risk(sc::LogSumExpScalariser, rs::AbstractVector{<:RiskMeasure},
                        w::AbstractVector, X::AbstractMatrix, fees::Fees = Fees())
    rk = zero(eltype(X))
    for r ∈ rs
        rk += r.settings.scale * sc.gamma * expected_risk(r, w, X; fees = fees)
    end
    return log(exp(rk)) / sc.gamma
end
function expected_risk(rs::AbstractVector{<:RiskMeasure}, w::AbstractVector,
                       X::AbstractMatrix; fees::Fees = Fees(),
                       scalariser::Scalariser = SumScalariser())
    return _expected_risk(scalariser, rs, w, X, fees)
end
