abstract type AbstractGerberCovariance <: AbstractCovarianceEstimator end
abstract type AbstractBaseGerberCovariance <: AbstractGerberCovariance end
abstract type AbstractBaseNormalisedGerberCovariance <: AbstractBaseGerberCovariance end
struct BaseGerberCovariance{T1 <: StatsBase.CovarianceEstimator,
                            T2 <: PosDefMatrixEstimator, T3 <: Real} <:
       AbstractBaseGerberCovariance
    ve::T1
    pdm::T2
    threshold::T3
end
function BaseGerberCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                              pdm::Union{Nothing, <:PosDefMatrixEstimator} = NearestPosDef(),
                              threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return BaseGerberCovariance{typeof(ve), typeof(pdm), typeof(threshold)}(ve, pdm,
                                                                            threshold)
end
function w_moment_factory(ce::BaseGerberCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return BaseGerberCovariance(; ve = w_moment_factory(ce.ve, w), pdm = ce.pdm,
                                threshold = ce.threshold)
end
struct BaseNormalisedCovariance{T1 <: AbstractExpectedReturnsEstimator,
                                T2 <: StatsBase.CovarianceEstimator,
                                T3 <: PosDefMatrixEstimator, T4 <: Real} <:
       AbstractBaseGerberCovariance
    me::T1
    ve::T2
    pdm::T3
    threshold::T4
end
function BaseNormalisedCovariance(;
                                  me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                  ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                                  pdm::Union{Nothing, <:PosDefMatrixEstimator} = NearestPosDef(),
                                  threshold::Real = 0.5)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return BaseNormalisedCovariance{typeof(me), typeof(ve), typeof(pdm), typeof(threshold)}(me,
                                                                                            ve,
                                                                                            pdm,
                                                                                            threshold)
end
function w_moment_factory(ce::BaseNormalisedCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return BaseNormalisedCovariance(; me = w_moment_factory(ce.me, w),
                                    ve = w_moment_factory(ce.ve, w), pdm = ce.pdm,
                                    threshold = ce.threshold)
end

export BaseGerberCovariance, BaseNormalisedCovariance
