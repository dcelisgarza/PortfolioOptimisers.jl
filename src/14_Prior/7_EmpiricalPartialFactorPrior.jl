struct EmpiricalPartialFactorPriorResult{T1 <: EmpiricalPriorResult,
                                         T2 <: PartialFactorPriorResult} <:
       AbstractPriorResult_APF
    pm::T1
    fm::T2
end
function EmpiricalPartialFactorPriorResult(; pm::EmpiricalPriorResult,
                                           fm::PartialFactorPriorResult)
    return EmpiricalPartialFactorPriorResult{typeof(pm), typeof(fm)}(pm, fm)
end
function Base.getproperty(obj::EmpiricalPartialFactorPriorResult, sym::Symbol)
    return if sym == :X
        obj.pm.X
    elseif sym == :mu
        obj.pm.mu
    elseif sym == :sigma
        obj.pm.sigma
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
function prior_view(pm::EmpiricalPartialFactorPriorResult, i::AbstractVector)
    return EmpiricalPartialFactorPriorResult(; pm = prior_view(pm.pm, i),
                                             fm = prior_view(pm.fm, i))
end

export EmpiricalPartialFactorPriorResult