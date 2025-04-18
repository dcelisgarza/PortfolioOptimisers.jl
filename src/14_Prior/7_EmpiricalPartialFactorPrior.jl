struct EmpiricalPartialFactorPriorResult{T1 <: EmpiricalPriorResult,
                                         T2 <: PartialFactorPriorResult} <:
       AbstractPriorResult_APF
    pr::T1
    fm::T2
end
function EmpiricalPartialFactorPriorResult(; pr::EmpiricalPriorResult,
                                           fm::PartialFactorPriorResult)
    return EmpiricalPartialFactorPriorResult{typeof(pr), typeof(fm)}(pr, fm)
end
function Base.getproperty(obj::EmpiricalPartialFactorPriorResult, sym::Symbol)
    return if sym == :X
        obj.pr.X
    elseif sym == :mu
        obj.pr.mu
    elseif sym == :sigma
        obj.pr.sigma
    elseif sym == :f_mu
        obj.fm.mu
    elseif sym == :f_sigma
        obj.fm.sigma
    elseif sym == :loadings
        obj.fm.loadings
    else
        getfield(obj, sym)
    end
end
function prior_view(pr::EmpiricalPartialFactorPriorResult, i::AbstractVector)
    return EmpiricalPartialFactorPriorResult(; pr = prior_view(pr.pr, i),
                                             fm = prior_view(pr.fm, i))
end

export EmpiricalPartialFactorPriorResult