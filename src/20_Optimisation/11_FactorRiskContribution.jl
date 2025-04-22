struct FactorRiskContribution{T1 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                              T2 <: ObjectiveFunction, T3 <: JuMPOptimiser} <:
       JuMPOptimisationEstimator
    r::T1
    obj::T2
    opt::T3
end
function FactorRiskContribution(;
                                r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                                obj::ObjectiveFunction = MinimumRisk(),
                                opt::JuMPOptimiser = JuMPOptimiser())
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    return FactorRiskContribution{typeof(r), typeof(obj), typeof(opt)}(r, obj, opt)
end

export FactorRiskContribution
