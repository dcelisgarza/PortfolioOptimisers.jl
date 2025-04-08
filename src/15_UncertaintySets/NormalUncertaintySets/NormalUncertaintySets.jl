struct NormalUncertaintySetEstimator{T1 <: AbstractPriorEstimator,
                                     T2 <: AbstractUncertaintySetAlgorithm, T3 <: Integer,
                                     T4 <: Real, T5 <: AbstractRNG,
                                     T6 <: Union{Nothing, <:Integer}} <:
       AbstractUncertaintySetEstimator
    pe::T1
    class::T2
    n_sim::T3
    q::T4
    rng::T5
    seed::T6
end
function NormalUncertaintySetEstimator(;
                                       pe::AbstractPriorEstimator = EmpiricalPriorEstimator(),
                                       class::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                                       n_sim::Integer = 3_000, q::Real = 0.05,
                                       rng::AbstractRNG = Random.default_rng(),
                                       seed::Union{<:Integer, Nothing} = nothing)
    @smart_assert(n_sim > zero(n_sim))
    @smart_assert(zero(q) < q < one(q))
    return NormalUncertaintySetEstimator{typeof(pe), typeof(class), typeof(n_sim),
                                         typeof(q), typeof(rng), typeof(seed)}(pe, class,
                                                                               n_sim, q,
                                                                               rng, seed)
end
function commutation_matrix(x::AbstractMatrix)
    m, n = size(x)
    mn = m * n
    row = 1:mn
    col = vec(transpose(reshape(row, m, n)))
    data = range(; start = 1, stop = 1, length = mn)
    return sparse(row, col, data, mn, mn)
end

export NormalUncertaintySetEstimator
