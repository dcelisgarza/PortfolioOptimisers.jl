struct EmpiricalPartialFactorPriorResult{T1 <: EmpiricalPriorResult,
                                         T2 <: PartialFactorPriorResult} <:
       AbstractPriorResult_APF
    pr::T1
    fpr::T2
end
function EmpiricalPartialFactorPriorResult(; pr::EmpiricalPriorResult,
                                           fpr::PartialFactorPriorResult)
    return EmpiricalPartialFactorPriorResult{typeof(pr), typeof(fpr)}(pr, fpr)
end
function Base.getproperty(obj::EmpiricalPartialFactorPriorResult, sym::Symbol)
    return if sym == :X
        obj.pr.X
    elseif sym == :mu
        obj.pr.mu
    elseif sym == :sigma
        obj.pr.sigma
    elseif sym == :f_mu
        obj.fpr.mu
    elseif sym == :f_sigma
        obj.fpr.sigma
    elseif sym == :loadings
        obj.fpr.loadings
    else
        getfield(obj, sym)
    end
end
function prior_view(pr::EmpiricalPartialFactorPriorResult, i::AbstractVector)
    return EmpiricalPartialFactorPriorResult(; pr = prior_view(pr.pr, i),
                                             fpr = prior_view(pr.fpr, i))
end

export EmpiricalPartialFactorPriorResult
