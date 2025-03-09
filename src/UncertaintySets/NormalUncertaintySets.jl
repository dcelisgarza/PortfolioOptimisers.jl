struct NormalUncertaintySetEstimator{T1 <: AbstractPriorEstimator,
                                     T2 <: UncertaintySetClass, T4 <: Integer, T6 <: Real,
                                     T7 <: Union{Nothing, <:Integer}} <:
       UncertaintySetEstimator
    pe::T1
    class::T2
    n_sim::T4
    q::T6
    seed::T7
end
function NormalUncertaintySetEstimator(;
                                       pe::AbstractPriorEstimator = EmpiricalPriorEstimator(),
                                       class::UncertaintySetClass = BoxUncertaintySetClass(),
                                       n_sim::Integer = 3_000, q::Real = 0.05,
                                       seed::Union{<:Integer, Nothing} = nothing)
    @smart_assert(n_sim > zero(n_sim))
    @smart_assert(zero(q) < q < one(q))
    return NormalUncertaintySetEstimator{typeof(pe), typeof(class), typeof(n_sim),
                                         typeof(q), typeof(seed)}(pe, class, n_sim, q, seed)
end

export NormalUncertaintySetEstimator