struct MeanRisk{T1 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                T2 <: ObjectiveFunction, T3 <: PortfolioReturnType, T4 <: Scalariser,
                T5 <: CustomConstraint, T6 <: CustomObjective, T7 <: AbstractVector} <:
       TraditionalOptimisationType
    risk::T1
    obj::T2
    ret::T3
    sc::T4
    cc::T5
    co::T6
    wi::T7
end
function MeanRisk(; risk::Union{RiskMeasure, AbstractVector{<:RiskMeasure}} = Variance(),
                  obj::ObjectiveFunction = MinimumRisk(),
                  ret::PortfolioReturnType = ArithmeticReturn(),
                  sc::Scalariser = SumScalariser(),
                  cc::CustomConstraint = NoCustomConstraint(),
                  co::CustomObjective = NoCustomObjective(),
                  wi::AbstractVector = Vector{Float64}())
    return MeanRisk{typeof(risk), typeof(obj), typeof(ret), typeof(sc), typeof(cc),
                    typeof(co), typeof(wi)}(risk, obj, ret, sc, cc, co, wi)
end
function optimise!(X::AbstractMatrix, opt::MeanRisk; os::Real = 1.0, cs::Real = 1.0,
                   str_names::Bool = false)
    return nothing
end

export MeanRisk
