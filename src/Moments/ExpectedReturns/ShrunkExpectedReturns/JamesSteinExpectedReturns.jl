struct JamesSteinExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                 T2 <: ShrunkExpectedReturnsTarget,
                                 T3 <: Union{Nothing, <:AbstractWeights}} <:
       ShrunkExpectedReturnsEstimator
    ce::T1
    target::T2
    w::T3
end
function JamesSteinExpectedReturns(; ce::StatsBase.CovarianceEstimator,
                                   target::ShrunkExpectedReturnsTarget = SERT_GrandMean(),
                                   w::Union{Nothing, <:AbstractWeights} = nothing)
    return JamesSteinExpectedReturns{typeof(ce), typeof(target), typeof(w)}(ce, target, w)
end

export JamesSteinExpectedReturns
