struct MeanRisk{T1 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                T2 <: ObjectiveFunction, T3 <: PortfolioReturnType, T4 <: Scalariser,
                T5 <: CustomConstraint, T6 <: CustomObjective} <:
       TraditionalOptimisationType
    risk::T1
    obj::T2
    ret::T3
    sc::T4
    cc::T5
    co::T6
end
function MeanRisk(; risk::Union{RiskMeasure, AbstractVector{<:RiskMeasure}} = Variance(),
                  obj::ObjectiveFunction = MinimumRisk(),
                  ret::PortfolioReturnType = ArithmeticReturn(),
                  sc::Scalariser = SumScalariser(),
                  cc::CustomConstraint = NoCustomConstraint(),
                  co::CustomObjective = NoCustomObjective())
    return MeanRisk{typeof(risk), typeof(obj), typeof(ret), typeof(sc), typeof(cc),
                    typeof(co)}(risk, obj, ret, sc, cc, co)
end

export MeanRisk
